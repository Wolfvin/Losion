"""
Attention-preferred LoRA untuk Losion Framework.

LoRA adapters yang secara preferensial menargetkan attention layers
(yang lebih diuntungkan dari fine-tuning) dibanding FFN/MoE layers.

Mengapa attention-preferred?
- Research menunjukkan bahwa fine-tuning attention layers memberikan
  dampak lebih besar pada kualitas dibanding fine-tuning FFN layers
- Attention projections (Q, K, V, O) mengontrol bagaimana model
  mengakses informasi konteks — perubahan kecil berdampak besar
- FFN/MoE layers lebih bersifat "knowledge storage" — memerlukan
  rank lebih tinggi untuk mengubah pengetahuan, tapi kurang efisien
- Asymmetric rank: rank tinggi untuk attention, rank rendah untuk FFN
  menghasilkan tradeoff parameter-quality yang lebih baik

Key features:
1. Asymmetric LoRA rank: rank tinggi untuk attention, rendah untuk FFN
2. Automatic rank allocation berdasarkan layer sensitivity
3. Compatible dengan existing LoRA implementations
4. Merge weights untuk inference tanpa overhead
5. Selective layer adaptation

Arsitektur LoRA standar:
    output = base_output + (alpha / r) * up_proj(down_proj(x))
    dimana:
    - down_proj: (in_features, r) — proyeksi ke rank rendah
    - up_proj: (r, out_features) — proyeksi kembali ke dimensi penuh
    - alpha: scaling factor untuk mengontrol magnitude adaptasi
    - r: rank LoRA

Perbedaan dengan LoRA standar:
- Rank berbeda per tipe layer (attention vs FFN vs SSM)
- Automatic rank allocation berdasarkan sensitivitas
- Target module selection yang lebih cerdas

Komponen:
1. AttnLoRAConfig — Konfigurasi attention-preferred LoRA
2. AttnLoRALayer — Single LoRA adapter layer
3. AttnLoRAModel — Applier untuk model Losion

Referensi:
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models" (2024)
- Zach et al., "Contrastive Preference Optimization" — LoRA rank analysis

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

@dataclass
class AttnLoRAConfig:
    """Konfigurasi untuk attention-preferred LoRA.

    Attributes:
        attn_rank: LoRA rank untuk attention layers (Q, K, V, O projections).
            Biasanya nilai yang lebih tinggi karena attention lebih sensitif
            terhadap fine-tuning.
        ffn_rank: LoRA rank untuk FFN/MoE layers. Biasanya lebih rendah
            karena FFN kurang diuntungkan dari fine-tuning.
        ssm_rank: LoRA rank untuk SSM layers. SSM layers memerlukan
            adaptasi yang berbeda karena struktur sequential-nya.
        alpha: LoRA scaling factor. Mengontrol magnitude adaptasi:
            output = base_output + (alpha / r) * lora_output
            Alpha yang lebih besar → adaptasi lebih kuat.
        target_modules: Tipe module yang akan di-adapt. Pilihan:
            "attention", "ffn", "ssm", "all". Default: "attention".
        dropout: Dropout rate untuk LoRA layers.
        merge_at_inference: Jika True, merge LoRA weights ke base weights
            sebelum inference (menghilangkan overhead LoRA).
        init_scale: Scale untuk inisialisasi LoRA weights.
            0.0 → zero init (adaptasi dimulai dari nol, stabil).
        rank_allocation: Mode alokasi rank — "fixed" atau "auto".
            "fixed": Gunakan rank sesuai config.
            "auto": Sesuaikan rank berdasarkan layer sensitivity.
        sensitivity_samples: Jumlah samples untuk sensitivity estimation
            (hanya digunakan jika rank_allocation="auto").
    """

    attn_rank: int = 16
    ffn_rank: int = 4
    ssm_rank: int = 8
    alpha: float = 32.0
    target_modules: str = "all"
    dropout: float = 0.0
    merge_at_inference: bool = True
    init_scale: float = 0.0
    rank_allocation: str = "fixed"
    sensitivity_samples: int = 100

    def __post_init__(self) -> None:
        if self.attn_rank < 1:
            raise ValueError(f"attn_rank harus >= 1, mendapat {self.attn_rank}")
        if self.ffn_rank < 1:
            raise ValueError(f"ffn_rank harus >= 1, mendapat {self.ffn_rank}")
        if self.ssm_rank < 1:
            raise ValueError(f"ssm_rank harus >= 1, mendapat {self.ssm_rank}")
        if self.target_modules not in ("attention", "ffn", "ssm", "all"):
            raise ValueError(
                f"target_modules harus 'attention', 'ffn', 'ssm', atau 'all', "
                f"mendapat '{self.target_modules}'"
            )
        if self.rank_allocation not in ("fixed", "auto"):
            raise ValueError(
                f"rank_allocation harus 'fixed' atau 'auto', "
                f"mendapat '{self.rank_allocation}'"
            )


# ---------------------------------------------------------------------------
# AttnLoRALayer — Single LoRA adapter
# ---------------------------------------------------------------------------

class AttnLoRALayer(nn.Module):
    """
    Single LoRA adapter layer.

    Mengimplementasikan standard LoRA dengan down projection dan up projection:
        lora_output = up_proj(dropout(down_proj(x)))
        output = base_output + (alpha / r) * lora_output

    Mendukung:
    - Dropout untuk regularisasi
    - Scaling factor alpha/r untuk mengontrol magnitude
    - Zero initialization untuk stabil training
    - Merge ke base weights untuk inference tanpa overhead

    Args:
        in_features: Dimensi input.
        out_features: Dimensi output.
        rank: LoRA rank (bisa berbeda per tipe layer).
        alpha: Scaling factor.
        dropout: Dropout rate.
        init_scale: Scale untuk inisialisasi.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # ---- LoRA projections ----
        # down_proj: proyeksi ke rank rendah (bottleneck)
        self.down_proj = nn.Linear(in_features, rank, bias=False)
        # up_proj: proyeksi kembali ke dimensi penuh
        self.up_proj = nn.Linear(rank, out_features, bias=False)

        # ---- Dropout ----
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # ---- Initialization ----
        # down_proj: random init (Kaiming/He)
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        # up_proj: zero init (adaptasi dimulai dari nol)
        if init_scale == 0.0:
            nn.init.zeros_(self.up_proj.weight)
        else:
            nn.init.normal_(self.up_proj.weight, std=init_scale)

        # ---- Merged flag ----
        self._merged = False

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass LoRA.

        Jika belum merged:
            output = base_output + scaling * up_proj(dropout(down_proj(x)))
        Jika sudah merged:
            output = base_output (LoRA sudah di-merge ke base weights)

        Args:
            x: Input tensor (sebelum base linear layer).
            base_output: Output dari base linear layer.

        Returns:
            Adapted output tensor.
        """
        if self._merged:
            return base_output

        # LoRA path
        lora_input = self.lora_dropout(x)
        lora_output = self.up_proj(self.down_proj(lora_input))

        # Combine dengan scaling
        return base_output + self.scaling * lora_output

    def merge(self, base_weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA weights ke base weights.

        Setelah merge, forward pass tidak lagi memiliki overhead LoRA.
        Formula: W_merged = W_base + (alpha / r) * up_proj @ down_proj

        Args:
            base_weight: Base weight tensor, (out_features, in_features).

        Returns:
            Merged weight tensor, (out_features, in_features).
        """
        # W_lora = up_proj @ down_proj → (out_features, rank) @ (rank, in_features)
        lora_weight = self.up_proj.weight.data @ self.down_proj.weight.data  # (out, in)
        merged_weight = base_weight + self.scaling * lora_weight
        self._merged = True
        return merged_weight

    def unmerge(self, base_weight: torch.Tensor) -> torch.Tensor:
        """
        Unmerge LoRA weights dari base weights.

        Kembalikan ke mode terpisah (LoRA overhead aktif).

        Args:
            base_weight: Merged weight tensor, (out_features, in_features).

        Returns:
            Original base weight tensor, (out_features, in_features).
        """
        lora_weight = self.up_proj.weight.data @ self.down_proj.weight.data
        original_weight = base_weight - self.scaling * lora_weight
        self._merged = False
        return original_weight

    def lora_weight_norm(self) -> torch.Tensor:
        """Hitung norm dari LoRA delta weight."""
        lora_weight = self.up_proj.weight.data @ self.down_proj.weight.data
        return lora_weight.norm()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"merged={self._merged}"
        )


# ---------------------------------------------------------------------------
# Module type classification helpers
# ---------------------------------------------------------------------------

# Nama module yang mengindikasikan attention projections
ATTENTION_PATTERNS = {"q_proj", "k_proj", "v_proj", "o_proj", "query", "key", "value", "out_proj"}

# Nama module yang mengindikasikan FFN/MoE projections
FFN_PATTERNS = {"gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3"}

# Nama module yang mengindikasikan SSM projections
SSM_PATTERNS = {"in_proj", "x_proj", "dt_proj", "ssd", "wkv", "delta"}


def classify_module(name: str) -> str:
    """
    Klasifikasikan module berdasarkan nama.

    Args:
        name: Nama module (dotted path).

    Returns:
        Tipe module: "attention", "ffn", "ssm", atau "other".
    """
    # Ambil nama terakhir dari dotted path
    parts = name.lower().split(".")
    last_part = parts[-1]

    for pattern in ATTENTION_PATTERNS:
        if pattern in last_part or pattern in name.lower():
            return "attention"

    for pattern in FFN_PATTERNS:
        if pattern in last_part or pattern in name.lower():
            return "ffn"

    for pattern in SSM_PATTERNS:
        if pattern in last_part or pattern in name.lower():
            return "ssm"

    return "other"


def get_rank_for_type(module_type: str, config: AttnLoRAConfig) -> int:
    """
    Dapatkan LoRA rank berdasarkan tipe module.

    Args:
        module_type: Tipe module ("attention", "ffn", "ssm", "other").
        config: Konfigurasi LoRA.

    Returns:
        LoRA rank untuk tipe tersebut.
    """
    if module_type == "attention":
        return config.attn_rank
    elif module_type == "ffn":
        return config.ffn_rank
    elif module_type == "ssm":
        return config.ssm_rank
    else:
        return min(config.ffn_rank, config.ssm_rank)


# ---------------------------------------------------------------------------
# AttnLoRAModel — Apply LoRA to a Losion model
# ---------------------------------------------------------------------------

class AttnLoRAModel:
    """
    Applies attention-preferred LoRA to a Losion model.

    Menambahkan LoRA adapters ke model dengan rank yang berbeda
    berdasarkan tipe layer:
    - Attention layers (Q, K, V, O): rank tinggi
    - FFN/MoE layers: rank rendah
    - SSM layers: rank sedang

    Alur penggunaan:
    1. apply(): Tambahkan LoRA adapters ke model
    2. Train model dengan LoRA (freeze base weights, train adapters)
    3. merge(): Merge LoRA weights ke base weights untuk inference
    4. Atau: unmerge() untuk kembali ke mode training

    Selective adaptation:
    - target_modules="attention": Hanya adapt attention layers
    - target_modules="ffn": Hanya adapt FFN/MoE layers
    - target_modules="ssm": Hanya adapt SSM layers
    - target_modules="all": Adapt semua layer

    Args:
        model: Model yang akan di-adapt.
        config: Konfigurasi attention-preferred LoRA.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AttnLoRAConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or AttnLoRAConfig()

        # ---- Daftar LoRA adapters yang ditambahkan ----
        self.lora_layers: Dict[str, AttnLoRALayer] = {}

        # ---- Original weights (sebelum merge) ----
        self._original_weights: Dict[str, torch.Tensor] = {}

    def apply(self) -> nn.Module:
        """
        Tambahkan LoRA adapters ke model.

        Mengiterasi semua nn.Linear modules dan menambahkan AttnLoRALayer
        sesuai dengan konfigurasi. Rank ditentukan berdasarkan tipe layer.

        Returns:
            Model dengan LoRA adapters.
        """
        self.lora_layers.clear()
        self._original_weights.clear()

        target = self.config.target_modules

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Klasifikasikan module
            module_type = classify_module(name)

            # Cek apakah module ini ditarget
            if target != "all" and module_type != target:
                # Juga cek "other" modules — skip jika tidak cocok
                if module_type == "other":
                    continue
                if module_type != target:
                    continue

            # Tentukan rank
            rank = get_rank_for_type(module_type, self.config)

            # Jika rank_allocation = "auto", sesuaikan berdasarkan dimensi
            if self.config.rank_allocation == "auto":
                rank = self._auto_rank(module, rank)

            # Buat LoRA adapter
            lora_layer = AttnLoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
                init_scale=self.config.init_scale,
            )

            # Pindahkan ke device yang sama
            lora_layer = lora_layer.to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

            # Simpan adapter
            self.lora_layers[name] = lora_layer

        # ---- Freeze base model parameters ----
        for param in self.model.parameters():
            param.requires_grad = False

        # ---- Register LoRA layers sebagai submodules ----
        # Kita perlu menambahkan lora_layers ke model agar mereka
        # termasuk dalam state_dict dan parameter_groups
        for name, lora in self.lora_layers.items():
            # Buat nama attribute yang valid
            attr_name = f"_lora_{name.replace('.', '_')}"
            setattr(self.model, attr_name, lora)

        # ---- Unfreeze LoRA parameters ----
        for name, lora in self.lora_layers.items():
            for param in lora.parameters():
                param.requires_grad = True

        return self.model

    def _auto_rank(self, module: nn.Module, base_rank: int) -> int:
        """
        Sesuaikan rank berdasarkan ukuran module.

        Module yang lebih besar mendapat rank yang lebih tinggi
        (tapi tetap dibatasi agar efisien).

        Args:
            module: nn.Linear module.
            base_rank: Base rank dari konfigurasi.

        Returns:
            Adjusted rank.
        """
        # Heuristic: rank proporsional ke min(in_features, out_features)
        min_dim = min(module.in_features, module.out_features)
        # Max rank: 25% dari min dimension, atau base_rank * 2
        max_rank = min(min_dim // 4, base_rank * 2)
        # Min rank: base_rank // 2
        min_rank = max(1, base_rank // 2)

        # Sesuaikan berdasarkan dimensi
        if min_dim < 256:
            return min_rank
        elif min_dim < 1024:
            return base_rank
        else:
            return min(max_rank, base_rank * 2)

    def merge(self) -> nn.Module:
        """
        Merge LoRA weights ke base weights untuk inference.

        Setelah merge, forward pass tidak memiliki overhead LoRA.
        LoRA adapters dihapus dari model.

        Returns:
            Model dengan merged weights.
        """
        for name, lora in self.lora_layers.items():
            # Temukan base module
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            base_module = getattr(parent, parts[-1])

            if isinstance(base_module, nn.Linear):
                # Merge weights
                merged_weight = lora.merge(base_module.weight.data)
                base_module.weight.data.copy_(merged_weight)

        return self.model

    def unmerge(self) -> nn.Module:
        """
        Unmerge LoRA weights, kembali ke mode training.

        Returns:
            Model dengan separated LoRA weights.
        """
        for name, lora in self.lora_layers.items():
            if not lora._merged:
                continue

            # Temukan base module
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            base_module = getattr(parent, parts[-1])

            if isinstance(base_module, nn.Linear):
                # Unmerge weights
                original_weight = lora.unmerge(base_module.weight.data)
                base_module.weight.data.copy_(original_weight)

        return self.model

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """
        Dapatkan semua LoRA parameters (untuk optimizer).

        Returns:
            List LoRA parameters.
        """
        params = []
        for lora in self.lora_layers.values():
            params.extend(lora.parameters())
        return params

    def lora_param_count(self) -> Dict[str, int]:
        """
        Hitung jumlah LoRA parameters per tipe layer.

        Returns:
            Dictionary berisi parameter count per tipe.
        """
        counts: Dict[str, int] = {"attention": 0, "ffn": 0, "ssm": 0, "other": 0, "total": 0}
        for name, lora in self.lora_layers.items():
            module_type = classify_module(name)
            n_params = sum(p.numel() for p in lora.parameters())
            counts[module_type] = counts.get(module_type, 0) + n_params
            counts["total"] += n_params
        return counts

    def base_param_count(self) -> int:
        """Hitung jumlah base parameters (frozen)."""
        return sum(
            p.numel() for p in self.model.parameters()
            if not p.requires_grad
        )

    def trainable_ratio(self) -> float:
        """
        Hitung rasio parameter yang trainable (LoRA) vs total.

        Returns:
            Rasio trainable parameters.
        """
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return trainable / max(total, 1)

    def get_rank_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Dapatkan ringkasan rank per tipe layer.

        Returns:
            Dictionary berisi rank info per tipe layer.
        """
        summary: Dict[str, Dict[str, int]] = {}
        for name, lora in self.lora_layers.items():
            module_type = classify_module(name)
            if module_type not in summary:
                summary[module_type] = {"count": 0, "total_rank": 0, "avg_rank": 0}
            summary[module_type]["count"] += 1
            summary[module_type]["total_rank"] += lora.rank
            summary[module_type]["avg_rank"] = (
                summary[module_type]["total_rank"] // summary[module_type]["count"]
            )
        return summary
