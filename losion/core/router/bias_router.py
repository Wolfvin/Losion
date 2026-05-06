"""
BiasRouter — Bias-Based Router untuk aux-loss-free load balancing.

Diadaptasi dari DeepSeek-V3: menggunakan dynamic bias yang disesuaikan
selama training, bukan auxiliary loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PathwayRoutingInfo:
    """Informasi routing untuk Tri-Jalur."""

    routing_weights: torch.Tensor  # [batch, seq, num_pathways]
    active_pathways: torch.Tensor  # [batch, seq, top_k] — indeks jalur aktif
    pathway_loads: torch.Tensor  # [num_pathways] — jumlah token per jalur
    bias_values: torch.Tensor  # [num_pathways] — nilai bias saat ini


class BiasRouter(nn.Module):
    """
    Bias-Based Router — aux-loss-free load balancing (DeepSeek-V3).

    Alih-alih auxiliary loss, menggunakan dynamic bias yang disesuaikan
    selama training. Bias menggeser probabilitas routing untuk menyeimbangkan
    distribusi, tanpa mengganggu kualitas representasi.

    Routing untuk Tri-Jalur:
    - 3 output: [jalur_1_prob, jalur_2_prob, jalur_3_prob]
    - Soft routing (bukan hard): token bisa diproses oleh 2-3 jalur
    - Bias di-update setiap N steps berdasarkan load imbalance

    Args:
        d_model: Model dimension
        num_pathways: Number of pathways (default 3)
        top_k_pathways: Number of active pathways per token (default 2)
        bias_lr: Learning rate for bias update (default 0.01)
    """

    def __init__(
        self,
        d_model: int,
        num_pathways: int = 3,
        top_k_pathways: int = 2,
        bias_lr: float = 0.01,
        bias_update_interval: int = 1000,
        warmup_bias_update_interval: int = 100,
        warmup_steps: int = 2000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_pathways = num_pathways
        self.top_k_pathways = min(top_k_pathways, num_pathways)
        self.bias_lr = bias_lr
        self.bias_update_interval = bias_update_interval
        self.warmup_bias_update_interval = warmup_bias_update_interval
        self.warmup_steps = warmup_steps

        # Gating network: proyeksi linear ke dimensi jalur
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_pathways, bias=False),
        )

        # Learnable bias untuk load balancing
        # Tidak di-include dalam gradient — hanya di-update manual
        self.register_buffer("bias", torch.zeros(num_pathways))

        # Temperature untuk sharpening/softening distribusi
        self.register_buffer(
            "temperature", torch.tensor(1.0)
        )

        # Running statistics untuk bias update
        self.register_buffer(
            "running_load", torch.zeros(num_pathways)
        )
        self.register_buffer(
            "update_count", torch.tensor(0, dtype=torch.long)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, PathwayRoutingInfo]:
        """
        Hitung routing weights untuk Tri-Jalur.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tuple:
                - routing_weights: [batch, seq_len, num_pathways] — bobot routing
                - routing_info: PathwayRoutingInfo dengan detail routing
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input harus 3D [batch, seq, d_model], mendapat {x.dim()}D"
            )

        batch_size, seq_len, _ = x.shape

        # Hitung logits melalui gating network + bias
        logits = self.gate_proj(x) + self.bias  # [batch, seq, num_pathways]

        # Temperature scaling
        logits = logits / self.temperature.clamp(min=0.1)

        # Soft routing: hitung full softmax untuk semua jalur
        routing_weights = F.softmax(logits, dim=-1)  # [batch, seq, num_pathways]

        # Top-K pathway selection
        top_k_weights, active_pathways = torch.topk(
            routing_weights, self.top_k_pathways, dim=-1
        )

        # Renormalize top-K weights
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Hitung load per pathway
        pathway_loads = torch.zeros(
            self.num_pathways, dtype=torch.long, device=x.device
        )
        for k in range(self.top_k_pathways):
            for p in range(self.num_pathways):
                pathway_loads[p] += (active_pathways[:, :, k] == p).sum()

        # Update running statistics
        with torch.no_grad():
            token_count = batch_size * seq_len
            instant_load = pathway_loads.float() / (token_count + 1e-8)
            # Exponential moving average
            momentum = 0.9
            self.running_load.mul_(momentum).add_(
                instant_load, alpha=1 - momentum
            )
            self.update_count.add_(1)

        routing_info = PathwayRoutingInfo(
            routing_weights=routing_weights,
            active_pathways=active_pathways,
            pathway_loads=pathway_loads,
            bias_values=self.bias.clone(),
        )

        return routing_weights, routing_info

    def get_effective_update_interval(self, global_step: int) -> int:
        """Get the effective bias update interval for the current training step.

        v2.5.0: During warmup (early training), bias updates are more frequent
        to prevent routing collapse. DeepSeek-V3 uses frequent updates during
        the initial phase; a fixed interval of 1000 steps is too infrequent
        when the model hasn't converged yet.

        The interval linearly interpolates from warmup_bias_update_interval
        to bias_update_interval over warmup_steps.

        Args:
            global_step: Current training step.

        Returns:
            Effective update interval for this step.
        """
        if global_step >= self.warmup_steps:
            return self.bias_update_interval
        # Linear interpolation from warmup to steady-state interval
        progress = global_step / max(self.warmup_steps, 1)
        interval = int(
            self.warmup_bias_update_interval
            + progress * (self.bias_update_interval - self.warmup_bias_update_interval)
        )
        return max(interval, 1)

    def should_update_bias(self, global_step: int) -> bool:
        """Check if bias should be updated at this training step.

        Uses warmup-aware interval (more frequent during early training).

        Args:
            global_step: Current training step.

        Returns:
            True if bias should be updated this step.
        """
        interval = self.get_effective_update_interval(global_step)
        return global_step > 0 and global_step % interval == 0

    def update_bias(
        self,
        pathway_loads: Optional[torch.Tensor] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """
        Update bias berdasarkan load imbalance.

        Mekanisme DeepSeek-V3:
        - Pathway yang kelebihan beban → bias negatif (dikurangi)
        - Pathway yang kekurangan beban → bias positif (ditambah)
        - Update bersifat soft, tidak drastis

        Dapat dipanggil dengan custom loads atau menggunakan
        running statistics yang terkumpul.

        Args:
            pathway_loads: Custom load counts [num_pathways], opsional
            total_tokens: Total token count, opsional
        """
        with torch.no_grad():
            if pathway_loads is not None and total_tokens is not None:
                # Gunakan custom loads
                if total_tokens == 0:
                    return
                ideal_load = total_tokens / self.num_pathways
                relative_load = pathway_loads.float() / (ideal_load + 1e-8)
            else:
                # Gunakan running statistics
                if self.update_count < 1:
                    return
                # Ideal: distribusi merata = 1/num_pathways per pathway
                ideal = 1.0 / self.num_pathways
                relative_load = self.running_load / (ideal + 1e-8)

            # Hitung deviasi
            deviation = relative_load - 1.0

            # Update bias: arah berlawanan dengan deviasi
            bias_update = -self.bias_lr * deviation

            # Clamp update agar tidak terlalu besar
            bias_update = bias_update.clamp(-0.1, 0.1)

            self.bias.add_(bias_update)

            # Clamp bias ke range wajar
            self.bias.clamp_(-2.0, 2.0)

    def set_temperature(self, temperature: float) -> None:
        """
        Set temperature untuk routing distribution.

        Temperature rendah → distribusi lebih tajam (lebih deterministic)
        Temperature tinggi → distribusi lebih merata (lebih explorative)

        Args:
            temperature: Nilai temperature (> 0)
        """
        if temperature <= 0:
            raise ValueError(f"Temperature harus > 0, mendapat {temperature}")
        self.temperature.fill_(temperature)

    def get_pathway_assignment(
        self, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Konversi soft routing ke hard assignment.

        Berguna untuk inference di mana kita ingin routing deterministic.

        Args:
            routing_weights: [batch, seq, num_pathways]

        Returns:
            Hard assignment: [batch, seq] — indeks jalur terpilih
        """
        return routing_weights.argmax(dim=-1)

    def get_load_balance_metric(
        self, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Hitung metric load balance dari distribusi routing.

        Metric ini BUKAN loss — hanya untuk monitoring.
        Nilai 0 = perfect balance, nilai tinggi = imbalance.

        Args:
            routing_weights: [batch, seq, num_pathways]

        Returns:
            Scalar metric
        """
        with torch.no_grad():
            # Rata-rata weight per pathway
            avg_weights = routing_weights.mean(dim=(0, 1))  # [num_pathways]
            ideal = 1.0 / self.num_pathways
            metric = (avg_weights - ideal).pow(2).sum()
            return metric
