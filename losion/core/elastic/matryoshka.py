"""
Matryoshka Elastic Inference — Nested Transformer for elastic deployment.

Diadaptasi dari Gemma 3n / MatFormer (Google DeepMind, 2025):
Matryoshka Nested Transformer memungkinkan satu set weight
menghasilkan banyak submodel yang valid, dari kecil hingga besar.

Konsep kunci dari MatFormer:
1. **Nested Submodels**: FFN layers di-train dengan nested structure
2. **Elastic Inference**: Saat inference, bisa memilih submodel size sesuai kebutuhan
3. **Mix'n'Match**: Bisa mengkombinasikan submodels berbeda di setiap layer
4. **Zero Additional Cost**: Tidak ada biaya training tambahan — semua submodels
   di-train sekaligus dalam satu weight set

Bagaimana MatFormer bekerja:
- FFN weight matrix: W ∈ R^{d_in × d_out}
- Submodel dengan faktor f: W_f = W[:, :f*d_out]
- Setiap submatrix W_f adalah model yang valid
- Training: Matryoshka loss memastikan semua submodels perform well

Adaptasi untuk Losion:
1. **Elastic FFN**: Setiap FFN layer mendukung nested extraction
2. **Quality-Speed Tradeoff**: Submodel kecil = cepat, submodel besar = akurat
3. **Adaptive Depth**: Router bisa menentukan submodel size per token
4. **Deployment Flexibility**: Satu checkpoint → banyak deployment configurations

Contoh:
- Full model (48B): Semua parameters aktif
- Medium model (~30B): 60% parameters aktif
- Small model (~15B): 30% parameters aktif
- Tiny model (~7B): 15% parameters aktif

Referensi:
- Google DeepMind, "Gemma 3n model overview" (2025)
- Devvrit et al., "MatFormer: Nested Transformer for Elastic Inference" (2023)

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MatryoshkaConfig:
    """Konfigurasi untuk Matryoshka Elastic Inference.

    Attributes:
        d_model: Dimensi model.
        d_ff: Dimensi feed-forward (full size).
        granularity_factors: List faktor granularitas [0.25, 0.5, 0.75, 1.0]
            Misalnya: [0.25, 0.5, 0.75, 1.0] berarti 4 submodel sizes
        matryoshka_loss_weight: Bobot Matryoshka loss dalam training.
            Loss total = main_loss + weight * sum(submodel_losses)
        use_adaptive: Gunakan adaptive submodel selection per token.
    """

    d_model: int = 2048
    d_ff: int = 8192
    granularity_factors: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    matryoshka_loss_weight: float = 0.1
    use_adaptive: bool = True

    def __post_init__(self) -> None:
        if not self.granularity_factors:
            raise ValueError("granularity_factors tidak boleh kosong")
        if not all(0 < f <= 1.0 for f in self.granularity_factors):
            raise ValueError("Semua granularity_factors harus di (0, 1.0]")
        if self.matryoshka_loss_weight < 0:
            raise ValueError("matryoshka_loss_weight tidak boleh negatif")


class MatryoshkaLayer(nn.Module):
    """Matryoshka FFN Layer — nested feed-forward dengan elastic inference.

    Mengimplementasikan MatFormer-style nested FFN:
    - Training: Semua submodels di-train sekaligus
    - Inference: Pilih submodel size sesuai kebutuhan

    FFN menggunakan SwiGLU:
    output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Nested structure:
    - gate_proj: [d_model, d_ff] → sub: [d_model, f*d_ff]
    - up_proj:   [d_model, d_ff] → sub: [d_model, f*d_ff]
    - down_proj: [d_ff, d_model] → sub: [f*d_ff, d_model]

    Args:
        config: Konfigurasi Matryoshka.
    """

    def __init__(self, config: MatryoshkaConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.granularity_factors = sorted(config.granularity_factors)

        # === Full-size FFN weights ===
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

        # === Adaptive selector (opsional) ===
        if config.use_adaptive:
            self.size_selector = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 8, bias=False),
                nn.SiLU(),
                nn.Linear(config.d_model // 8, 1, bias=False),
                nn.Sigmoid(),  # [0, 1] → map ke granularity factor
            )

    def forward(
        self,
        x: torch.Tensor,
        granularity_factor: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass Matryoshka FFN.

        Args:
            x: Input tensor [batch, seq, d_model]
            granularity_factor: Faktor granularitas (0.0-1.0).
                None = full model (1.0).

        Returns:
            Tuple (output, info):
            - output: [batch, seq, d_model]
            - info: Dictionary statistik
        """
        # Default: full model
        factor = granularity_factor or 1.0
        factor = min(max(factor, min(self.granularity_factors)), 1.0)

        # === Adaptive factor selection ===
        if granularity_factor is None and self.config.use_adaptive:
            # Model memilih faktor sendiri berdasarkan input
            score = self.size_selector(x.mean(dim=1, keepdim=False))  # [batch, 1]
            # Map score ke faktor terdekat
            factor = self._score_to_factor(score.mean().item())

        # Hitung dimensi aktif
        d_ff_active = max(1, int(self.d_ff * factor))

        if factor >= 1.0:
            # Full model — standard forward
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            output = self.down_proj(gate * up)
        else:
            # Submodel — gunakan slice dari weight matrix
            gate_weight = self.gate_proj.weight[:d_ff_active, :]
            up_weight = self.up_proj.weight[:d_ff_active, :]
            down_weight = self.down_proj.weight[:, :d_ff_active]

            gate = F.silu(F.linear(x, gate_weight))
            up = F.linear(x, up_weight)
            output = F.linear(gate * up, down_weight)

        info = {
            "granularity_factor": factor,
            "d_ff_active": d_ff_active,
            "d_ff_total": self.d_ff,
            "parameter_fraction": factor,  # Approximate
        }

        return output, info

    def _score_to_factor(self, score: float) -> float:
        """Konversi score [0, 1] ke granularity factor terdekat.

        Args:
            score: Value dari size_selector [0, 1]

        Returns:
            Granularity factor terdekat dari config
        """
        # Map: score rendah → faktor kecil, score tinggi → faktor besar
        # Cari faktor terdekat
        min_dist = float("inf")
        best_factor = self.granularity_factors[-1]  # Default: full

        for f in self.granularity_factors:
            dist = abs(score - f)
            if dist < min_dist:
                min_dist = dist
                best_factor = f

        return best_factor

    def extract_submodel(
        self,
        granularity_factor: float,
    ) -> Dict[str, nn.Parameter]:
        """Ekstrak submodel pada granularity factor tertentu.

        Menghasilkan weight subset untuk deployment yang lebih kecil.
        Berguna untuk:
        - Edge deployment: faktor kecil (0.25)
        - Server deployment: faktor besar (1.0)
        - Mobile deployment: faktor sangat kecil (0.125)

        Args:
            granularity_factor: Faktor granularitas target.

        Returns:
            Dictionary berisi weight submodel.
        """
        d_ff_sub = max(1, int(self.d_ff * granularity_factor))

        submodel = {
            "gate_proj.weight": self.gate_proj.weight[:d_ff_sub, :].clone(),
            "up_proj.weight": self.up_proj.weight[:d_ff_sub, :].clone(),
            "down_proj.weight": self.down_proj.weight[:, :d_ff_sub].clone(),
        }

        return submodel

    def compute_matryoshka_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Any = None,
    ) -> torch.Tensor:
        """Hitung Matryoshka loss — memastikan semua submodels perform well.

        Loss = main_loss + weight * sum(submodel_losses)

        Setiap submodel di-evaluate pada input yang sama, dan loss
        dihitung untuk setiap ukuran. Ini memastikan bahwa submodel
        kecil tetap berguna.

        Args:
            x: Input tensor [batch, seq, d_model]
            target: Target output (dari full model)
            loss_fn: Loss function (default: MSELoss)

        Returns:
            Matryoshka loss scalar
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        total_loss = torch.tensor(0.0, device=x.device)

        for factor in self.granularity_factors:
            output, _ = self.forward(x, granularity_factor=factor)
            sub_loss = loss_fn(output, target)
            total_loss = total_loss + sub_loss

        # Normalize
        total_loss = total_loss / len(self.granularity_factors)

        return total_loss * self.config.matryoshka_loss_weight


class ElasticExtractor:
    """Utility untuk mengekstrak model pada berbagai ukuran.

    Memungkinkan konversi satu checkpoint menjadi banyak model
    dengan ukuran berbeda. Berguna untuk deployment.

    Contoh penggunaan:
        extractor = ElasticExtractor(model)
        tiny_model = extractor.extract(0.25)   # 25% parameters
        small_model = extractor.extract(0.5)   # 50% parameters
        medium_model = extractor.extract(0.75)  # 75% parameters
        full_model = extractor.extract(1.0)     # 100% parameters

    Setiap extracted model adalah model yang valid dan bisa
    di-deploy secara independen.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def extract(
        self,
        granularity_factor: float,
    ) -> nn.Module:
        """Ekstrak submodel pada granularity factor tertentu.

        Args:
            granularity_factor: Faktor granularitas (0.0-1.0).

        Returns:
            Submodel dengan parameter lebih sedikit.
        """
        # Deep copy model
        import copy
        submodel = copy.deepcopy(self.model)

        # Iterasi semua MatryoshkaLayer dan ekstrak
        for name, module in submodel.named_modules():
            if isinstance(module, MatryoshkaLayer):
                d_ff_sub = max(1, int(module.d_ff * granularity_factor))

                # Replace weight matrices dengan subset
                with torch.no_grad():
                    module.gate_proj.weight.data = module.gate_proj.weight.data[:d_ff_sub, :].clone()
                    module.up_proj.weight.data = module.up_proj.weight.data[:d_ff_sub, :].clone()
                    module.down_proj.weight.data = module.down_proj.weight.data[:, :d_ff_sub].clone()

                # Update dimensions
                module.d_ff = d_ff_sub
                module.gate_proj.out_features = d_ff_sub
                module.up_proj.out_features = d_ff_sub
                module.down_proj.in_features = d_ff_sub

        return submodel

    def get_available_sizes(self) -> List[Dict[str, Any]]:
        """Dapatkan daftar ukuran model yang tersedia.

        Returns:
            List dictionaries berisi info ukuran
        """
        sizes = []

        # Cari granularity factors dari model
        factors = set()
        for name, module in self.model.named_modules():
            if isinstance(module, MatryoshkaLayer):
                factors.update(module.granularity_factors)

        factors = sorted(factors)

        total_params = sum(p.numel() for p in self.model.parameters())

        for factor in factors:
            estimated_params = int(total_params * factor)
            sizes.append({
                "granularity_factor": factor,
                "estimated_parameters": estimated_params,
                "parameter_label": f"~{estimated_params / 1e9:.1f}B" if estimated_params >= 1e9
                                  else f"~{estimated_params / 1e6:.0f}M",
            })

        return sizes

    def mix_and_match(
        self,
        layer_factors: Dict[int, float],
    ) -> nn.Module:
        """Mix'n'Match: ukuran berbeda per layer.

        Dari Gemma 3n: bisa menggunakan submodel berbeda
        di setiap layer. Misalnya, layer awal kecil (0.25),
        layer tengah sedang (0.5), layer akhir besar (1.0).

        Args:
            layer_factors: Dictionary {layer_idx: granularity_factor}

        Returns:
            Mixed-size model
        """
        import copy
        submodel = copy.deepcopy(self.model)

        layer_idx = 0
        for name, module in submodel.named_modules():
            if isinstance(module, MatryoshkaLayer):
                factor = layer_factors.get(layer_idx, 1.0)
                d_ff_sub = max(1, int(module.d_ff * factor))

                with torch.no_grad():
                    module.gate_proj.weight.data = module.gate_proj.weight.data[:d_ff_sub, :].clone()
                    module.up_proj.weight.data = module.up_proj.weight.data[:d_ff_sub, :].clone()
                    module.down_proj.weight.data = module.down_proj.weight.data[:, :d_ff_sub].clone()

                module.d_ff = d_ff_sub
                module.gate_proj.out_features = d_ff_sub
                module.up_proj.out_features = d_ff_sub
                module.down_proj.in_features = d_ff_sub

                layer_idx += 1

        return submodel
