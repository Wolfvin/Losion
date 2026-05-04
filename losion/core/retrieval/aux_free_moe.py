"""
Auxiliary-Loss-Free MoE dengan Multi-Token Prediction (MTP) — DeepSeek-V3 style.

Mengeliminasi auxiliary loss yang merusak kualitas dalam MoE, menggantikannya
dengan bias-based balancing yang diadaptasi dari DeepSeek-V3. Ditambah
Multi-Token Prediction (MTP) heads yang memberikan sinyal training komplementer
untuk meningkatkan expert specialization tanpa auxiliary loss.

Arsitektur:
1. AuxFreeMoERouter — DeepSeek-V3 style aux-loss-free router
   Menggunakan dynamic bias (bukan gradient-based) untuk load balancing.
   Bias di-update via running statistics (EMA), bukan melalui backpropagation.
   Tidak ada auxiliary loss yang dikembalikan — hanya monitoring metrics.

   Mekanisme:
   - Standard routing: logits = gate_proj(x) + bias
   - Bias update: bias -= lr * (load_deviation_from_ideal)
   - Expert yang kelebihan beban → bias negatif (dikurangi)
   - Expert yang kekurangan beban → bias positif (ditambah)
   - Update bersifat soft, non-gradient, tidak mengganggu representasi

2. MTPMoEHead — Multi-Token Prediction head untuk MoE training
   Lightweight prediction head yang memprediksi token masa depan (n token ahead).
   Berbagi infrastruktur expert dengan MoE utama → tanpa overhead parameter.
   Training loss: multi-token cross-entropy dengan geometric decay weights.

   Mekanisme:
   - Input hidden state → n_future prediction heads
   - Setiap head memprediksi token t+1, t+2, ..., t+n
   - Weight: λ_k = λ^(k-1) dimana λ ∈ (0, 1) (geometric decay)
   - Loss = Σ_k λ_k * CE(pred_k, target_k)
   - Memberikan sinyal training yang kaya untuk expert specialization

3. AuxFreeMoE — Complete aux-loss-free MoE dengan MTP training
   Menggabungkan AuxFreeMoERouter + MTPMoEHead + standard experts.
   Kompatibel dengan interface MoE yang ada (ExpertChoiceMoE, dll).

Keuntungan:
- Tidak ada quality degradation dari auxiliary loss
- Load balancing tetap tercapai via bias adjustment
- MTP memberikan sinyal training komplementer untuk expert specialization
- Parameter overhead minimal: hanya n_future prediction heads kecil
- Compatible dengan shared expert (DeepSeek-V3 style)

Referensi:
- DeepSeek-AI, "DeepSeek-V3" (2024) — Aux-loss-free load balancing
- Gloeckle et al., "Better & Faster Large Language Models via Multi-token
  Prediction" (2024) — MTP for improved training
- Losion BiasRouter — Existing bias-based routing implementation

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AuxFreeRoutingInfo:
    """Informasi routing untuk aux-loss-free MoE.

    Attributes:
        routing_weights: [batch, seq, num_experts] — bobot routing (softmax)
        top_k_indices: [batch, seq, top_k] — indeks expert terpilih
        top_k_weights: [batch, seq, top_k] — bobot expert terpilih
        expert_loads: [num_experts] — jumlah token per expert
        bias_values: [num_experts] — nilai bias saat ini
        load_balance_metric: Metric load balance (monitoring only, BUKAN loss)
    """

    routing_weights: torch.Tensor
    top_k_indices: torch.Tensor
    top_k_weights: torch.Tensor
    expert_loads: torch.Tensor
    bias_values: torch.Tensor
    load_balance_metric: float


# ============================================================================
# AuxFreeMoERouter — DeepSeek-V3 Style Aux-Loss-Free Router
# ============================================================================

class AuxFreeMoERouter(nn.Module):
    """
    DeepSeek-V3 style aux-loss-free router.

    Alih-alih menggunakan auxiliary loss untuk load balancing (yang merusak
    kualitas representasi), router ini menggunakan dynamic bias yang
    disesuaikan selama training berdasarkan running statistics.

    Perbedaan dengan auxiliary loss approach:
    - Auxiliary loss: L_total = L_main + α * L_aux → menggeser gradient,
      mengorbankan kualitas representasi untuk load balancing
    - Bias-based: bias di-update secara non-gradient → tidak mengganggu
      representasi, kualitas terjaga

    Integrasi dengan Losion BiasRouter:
    - Konsep serupa dengan BiasRouter di losion/core/router/bias_router.py
    - Perbedaan: BiasRouter untuk Tri-Jalur routing (3 pathways),
      AuxFreeMoERouter untuk MoE expert routing (N experts)
    - AuxFreeMoERouter menggunakan top-K expert selection + renormalization
    - EMA-based bias update yang lebih stabil

    Args:
        d_model: Dimensi model.
        num_experts: Jumlah total experts.
        top_k: Jumlah expert aktif per token (default 2).
        bias_update_rate: Learning rate untuk bias update (default 0.01).
        bias_momentum: Momentum untuk EMA statistics (default 0.9).
        bias_clamp: Clamp range untuk bias values (default [-2.0, 2.0]).
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        bias_update_rate: float = 0.01,
        bias_momentum: float = 0.9,
        bias_clamp: float = 2.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.bias_update_rate = bias_update_rate
        self.bias_momentum = bias_momentum
        self.bias_clamp = bias_clamp

        # Gating network: proyeksi ke dimensi expert
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_experts, bias=False),
        )

        # Dynamic bias untuk load balancing (non-gradient)
        self.register_buffer("bias", torch.zeros(num_experts))

        # Running statistics untuk bias update (EMA)
        self.register_buffer("running_load", torch.zeros(num_experts))
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))

        # Temperature untuk routing distribution
        self.register_buffer("temperature", torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, AuxFreeRoutingInfo]:
        """
        Hitung routing untuk aux-loss-free MoE.

        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Tuple (top_k_weights, top_k_indices, routing_info):
            - top_k_weights: [batch, seq_len, top_k]
            - top_k_indices: [batch, seq_len, top_k]
            - routing_info: Detail routing untuk monitoring
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input harus 3D [batch, seq, d_model], mendapat {x.dim()}D"
            )

        batch_size, seq_len, _ = x.shape

        # Hitung logits melalui gating network + bias
        logits = self.gate_proj(x) + self.bias  # [batch, seq, num_experts]

        # Temperature scaling
        logits = logits / self.temperature.clamp(min=0.1)

        # Softmax untuk routing weights
        routing_weights = F.softmax(logits, dim=-1)  # [batch, seq, num_experts]

        # Top-K expert selection
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        # Renormalize top-K weights
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Hitung load per expert — vectorized via bincount (O(batch*seq*top_k) vs O(top_k*num_experts*batch*seq))
        expert_loads = torch.zeros(
            self.num_experts, dtype=torch.long, device=x.device
        )
        # Flatten top_k_indices dan hitung dengan bincount
        flat_indices = top_k_indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.num_experts)
        expert_loads = counts[:self.num_experts]

        # Update running statistics
        with torch.no_grad():
            token_count = batch_size * seq_len
            instant_load = expert_loads.float() / (token_count + 1e-8)
            self.running_load.mul_(self.bias_momentum).add_(
                instant_load, alpha=1 - self.bias_momentum
            )
            self.update_count.add_(1)

        # Load balance metric (monitoring only, BUKAN loss)
        ideal = 1.0 / self.num_experts
        avg_weights = routing_weights.mean(dim=(0, 1))
        load_balance_metric = (avg_weights - ideal).pow(2).sum().item()

        routing_info = AuxFreeRoutingInfo(
            routing_weights=routing_weights,
            top_k_indices=top_k_indices,
            top_k_weights=top_k_weights,
            expert_loads=expert_loads,
            bias_values=self.bias.clone(),
            load_balance_metric=load_balance_metric,
        )

        return top_k_weights, top_k_indices, routing_info

    def update_bias(self) -> None:
        """
        Update bias berdasarkan load imbalance (DeepSeek-V3 mechanism).

        Dipanggil secara periodik selama training (misalnya setiap N steps).
        Tidak menggunakan gradient — pure running statistics based.

        Mekanisme:
        - Expert yang kelebihan beban → bias negatif (dikurangi)
        - Expert yang kekurangan beban → bias positif (ditambah)
        - Update bersifat soft, tidak drastis
        """
        with torch.no_grad():
            if self.update_count < 1:
                return

            # Ideal load distribution: 1/num_experts per expert
            ideal = 1.0 / self.num_experts
            relative_load = self.running_load / (ideal + 1e-8)

            # Deviasi dari ideal
            deviation = relative_load - 1.0

            # Update bias: arah berlawanan dengan deviasi
            bias_update = -self.bias_update_rate * deviation

            # Clamp update agar tidak terlalu besar
            bias_update = bias_update.clamp(-0.1, 0.1)

            self.bias.add_(bias_update)

            # Clamp bias ke range wajar
            self.bias.clamp_(-self.bias_clamp, self.bias_clamp)


# ============================================================================
# MTPMoEHead — Multi-Token Prediction Head untuk MoE Training
# ============================================================================

class MTPMoEHead(nn.Module):
    """
    Multi-Token Prediction head untuk MoE training.

    Memperkenalkan prediction heads tambahan yang memprediksi token masa depan
    (t+1, t+2, ..., t+n_future). Head ini berbagi infrastruktur expert dengan
    MoE utama, sehingga overhead parameter minimal.

    Mekanisme:
    1. Input hidden state → n_future prediction heads
    2. Setiap head: linear projection → logits (shared vocab)
    3. Training loss: Σ_k λ_k * CE(pred_k, target_k)
       dimana λ_k = λ^(k-1) (geometric decay)
    4. Memberikan sinyal training yang kaya untuk expert specialization

    Mengapa MTP membantu expert specialization:
    - Memprediksi token jauh membutuhkan representasi yang lebih "global"
    - Ini memaksa experts untuk belajar representasi yang lebih informatif
    - Sinyal gradient dari MTP heads membantu experts terspecialisasi lebih baik
    - Tanpa MTP, experts hanya dioptimasi untuk next-token → representasi dangkal

    Args:
        d_model: Dimensi model.
        vocab_size: Ukuran vocabulary.
        n_future: Jumlah token masa depan yang diprediksi (default 4).
        decay_lambda: Decay factor untuk geometric weights (default 0.5).
            Weight untuk token ke-k: λ^(k-1)
            λ=0.5: [1.0, 0.5, 0.25, 0.125] untuk n_future=4
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_future: int = 4,
        decay_lambda: float = 0.5,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_future = n_future
        self.decay_lambda = decay_lambda

        # Prediction heads: satu per future token
        # Setiap head: d_model → vocab_size (lightweight)
        self.pred_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_future)
        ])

        # Norm untuk stabilisasi input ke setiap head
        self.pred_norms = nn.ModuleList([
            nn.RMSNorm(d_model, eps=1e-5)
            for _ in range(n_future)
        ])

        # Pre-compute geometric decay weights
        weights = [decay_lambda ** k for k in range(n_future)]
        self.register_buffer(
            "decay_weights",
            torch.tensor(weights, dtype=torch.float32),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Hitung MTP predictions dan loss.

        Args:
            hidden_states: Hidden states dari MoE, (batch, seq_len, d_model).
            targets: Target token IDs, (batch, seq_len). Jika None, tidak
                menghitung loss (inference mode).

        Returns:
            Tuple (mtp_loss, predictions):
            - mtp_loss: Scalar loss (atau None jika targets=None). BUKAN
                auxiliary loss — ini adalah training signal yang komplementer.
            - predictions: Dict berisi logits per future head.
        """
        batch_size, seq_len, _ = hidden_states.shape
        predictions = {}
        total_loss = None

        for k in range(self.n_future):
            # Norm input
            normed = self.pred_norms[k](hidden_states)

            # Predict token t+k+1 dari hidden state t
            # Shift: hidden[t] memprediksi target[t+k+1]
            logits = self.pred_heads[k](normed)  # (batch, seq_len, vocab_size)
            predictions[f"logits_k{k+1}"] = logits

            if targets is not None:
                # Target shift: untuk memprediksi t+k+1, target dimulai dari k+1
                # Hidden[t] → target[t+k+1]
                if k + 1 < seq_len:
                    # Align: prediksi dari hidden[0:seq_len-k-1] → target[k+1:seq_len]
                    pred_logits = logits[:, :seq_len - k - 1, :].contiguous()
                    target_ids = targets[:, k + 1:].contiguous()

                    # Cross-entropy loss
                    loss = F.cross_entropy(
                        pred_logits.view(-1, self.vocab_size),
                        target_ids.view(-1),
                        ignore_index=-100,  # Ignore padding
                    )

                    # Weight dengan geometric decay
                    weighted_loss = self.decay_weights[k] * loss

                    if total_loss is None:
                        total_loss = weighted_loss
                    else:
                        total_loss = total_loss + weighted_loss

        return total_loss, predictions


# ============================================================================
# AuxFreeMoE — Complete Aux-Loss-Free MoE dengan MTP Training
# ============================================================================

class AuxFreeMoE(nn.Module):
    """
    Complete aux-loss-free MoE dengan Multi-Token Prediction training.

    Menggabungkan:
    1. AuxFreeMoERouter — Bias-based routing tanpa auxiliary loss
    2. MTPMoEHead — Multi-token prediction untuk expert specialization
    3. Standard expert FFNs — SwiGLU experts + optional shared expert

    Kompatibel dengan interface MoE yang ada (ExpertChoiceMoE, dll):
    - Input: (batch, seq_len, d_model)
    - Output: (output, routing_info)
    - auxiliary_losses: Dict kosong atau monitoring-only metrics

    Perbedaan dengan ExpertChoiceMoE:
    - ExpertChoiceMoE: expert memilih token → guaranteed balance, tapi
      bisa ada dropped tokens dan tidak ada training signal tambahan
    - AuxFreeMoE: token memilih expert + bias balancing → lebih fleksibel,
      MTP memberikan sinyal training komplementer

    Perbedaan dengan standard MoE + aux loss:
    - Standard MoE: L_total = L_main + α * L_aux → quality degradation
    - AuxFreeMoE: L_total = L_main + β * L_mtp → no quality degradation,
      MTP justru meningkatkan kualitas representasi

    Args:
        d_model: Dimensi model.
        d_ff: Dimensi feed-forward per expert.
        num_experts: Jumlah total experts.
        top_k: Jumlah expert aktif per token (default 2).
        vocab_size: Ukuran vocabulary (untuk MTP heads).
        use_shared_expert: Gunakan shared expert (selalu aktif).
        use_mtp: Gunakan MTP heads (default True).
        n_future: Jumlah future tokens untuk MTP (default 4).
        mtp_loss_weight: Bobot MTP loss dalam total loss (default 0.1).
        bias_update_rate: Learning rate untuk router bias update.
        capacity_factor: Capacity factor (untuk compatibility, tidak digunakan langsung).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 64,
        top_k: int = 2,
        vocab_size: int = 32000,
        use_shared_expert: bool = True,
        use_mtp: bool = True,
        n_future: int = 4,
        mtp_loss_weight: float = 0.1,
        bias_update_rate: float = 0.01,
        capacity_factor: float = 1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.vocab_size = vocab_size
        self.use_shared_expert = use_shared_expert
        self.use_mtp = use_mtp
        self.n_future = n_future
        self.mtp_loss_weight = mtp_loss_weight

        # === Expert FFNs ===
        self.experts = nn.ModuleList([
            self._make_expert(d_model, d_ff) for _ in range(num_experts)
        ])

        # === Shared Expert ===
        if use_shared_expert:
            self.shared_expert = self._make_expert(d_model, d_ff)
            self.shared_expert_scale = nn.Parameter(torch.ones(1))

        # === Aux-Free Router ===
        self.router = AuxFreeMoERouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            bias_update_rate=bias_update_rate,
        )

        # === MTP Head ===
        if use_mtp:
            self.mtp_head = MTPMoEHead(
                d_model=d_model,
                vocab_size=vocab_size,
                n_future=n_future,
            )

    def _make_expert(self, d_model: int, d_ff: int) -> nn.ModuleDict:
        """Buat satu expert FFN dengan SwiGLU."""
        return nn.ModuleDict({
            "gate_proj": nn.Linear(d_model, d_ff, bias=False),
            "up_proj": nn.Linear(d_model, d_ff, bias=False),
            "down_proj": nn.Linear(d_ff, d_model, bias=False),
        })

    def _expert_forward(
        self, expert: nn.ModuleDict, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass satu expert (SwiGLU)."""
        gate = F.silu(expert["gate_proj"](x))
        up = expert["up_proj"](x)
        return expert["down_proj"](gate * up)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, AuxFreeRoutingInfo, Dict[str, torch.Tensor]]:
        """
        Forward pass Aux-Free MoE.

        Args:
            x: Input tensor [batch, seq_len, d_model].
            targets: Target token IDs untuk MTP loss, (batch, seq_len).
                Opsional — jika None, MTP loss tidak dihitung.

        Returns:
            Tuple (output, routing_info, auxiliary_losses):
            - output: (batch, seq_len, d_model)
            - routing_info: AuxFreeRoutingInfo dengan detail routing
            - auxiliary_losses: Dict kosong atau monitoring metrics.
                BUKAN auxiliary loss — tidak ada quality-degrading loss.
                Jika MTP aktif: {"mtp_loss": tensor} (training signal)
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input harus 3D [batch, seq, d_model], mendapat {x.dim()}D"
            )

        batch_size, seq_len, _ = x.shape
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)  # [total_tokens, d_model]
        total_tokens = x_flat.shape[0]

        # === Shared Expert ===
        shared_output = torch.zeros_like(x_flat)
        if self.use_shared_expert:
            shared_output = (
                self._expert_forward(self.shared_expert, x_flat)
                * self.shared_expert_scale
            )

        # === Routing ===
        top_k_weights, top_k_indices, routing_info = self.router(x)
        # top_k_weights: [batch, seq_len, top_k]
        # top_k_indices: [batch, seq_len, top_k]

        # Reshape untuk processing
        top_k_weights_flat = top_k_weights.reshape(-1, self.top_k)  # [total_tokens, top_k]
        top_k_indices_flat = top_k_indices.reshape(-1, self.top_k)  # [total_tokens, top_k]

        # === Compute Routed Expert Output ===
        routed_output = torch.zeros_like(x_flat)

        # Untuk setiap token, proses top_k experts
        for k in range(self.top_k):
            # Expert indices untuk posisi k
            expert_indices = top_k_indices_flat[:, k]  # [total_tokens]
            weights = top_k_weights_flat[:, k]  # [total_tokens]

            # Group tokens by expert untuk efisiensi
            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if not mask.any():
                    continue

                # Gather tokens untuk expert e
                expert_input = x_flat[mask]  # [n_tokens, d_model]

                # Process through expert
                expert_output = self._expert_forward(
                    self.experts[e], expert_input
                )  # [n_tokens, d_model]

                # Weight by routing weights dan accumulate
                expert_weights = weights[mask].unsqueeze(-1)
                routed_output[mask] += expert_weights * expert_output

        # === Combine ===
        output = (shared_output + routed_output).reshape(orig_shape)

        # === MTP Loss (training signal, BUKAN auxiliary loss) ===
        auxiliary_losses: Dict[str, torch.Tensor] = {}

        if self.use_mtp and self.training:
            mtp_loss, mtp_predictions = self.mtp_head(output, targets)
            if mtp_loss is not None:
                auxiliary_losses["mtp_loss"] = mtp_loss * self.mtp_loss_weight
            auxiliary_losses["mtp_loss_raw"] = mtp_loss if mtp_loss is not None else torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Monitoring metrics (bukan loss)
        auxiliary_losses["load_balance_metric"] = torch.tensor(
            routing_info.load_balance_metric
        )
        auxiliary_losses["bias_norm"] = self.router.bias.norm()

        return output, routing_info, auxiliary_losses

    def update_router_bias(self) -> None:
        """
        Update router bias berdasarkan running statistics.

        Dipanggil secara periodik selama training (misalnya setiap 100 steps).
        Ini adalah mekanisme DeepSeek-V3 untuk load balancing tanpa auxiliary loss.
        """
        self.router.update_bias()

    def get_expert_specialization(
        self, x: torch.Tensor, top_k: int = 5
    ) -> Dict[int, List[int]]:
        """
        Analisis spesialisasi expert.

        Mengembalikan token mana yang paling sering dipilih
        oleh setiap expert. Berguna untuk memahami apakah
        expert benar-benar terspecialisasi.

        Args:
            x: Input tensor [batch, seq, d_model]
            top_k: Jumlah top tokens yang ditampilkan per expert

        Returns:
            Dictionary {expert_idx: [top_token_indices]}
        """
        with torch.no_grad():
            _, top_k_indices, _ = self.router(x)

            specialization = {}
            for e in range(self.num_experts):
                token_indices = []
                for k in range(self.top_k):
                    mask = (top_k_indices[:, :, k] == e)
                    indices = mask.nonzero(as_tuple=True)
                    for b, s in zip(*indices):
                        token_indices.append(b.item() * x.shape[1] + s.item())

                # Ambil top_k tokens yang paling sering muncul
                from collections import Counter
                counts = Counter(token_indices)
                top_tokens = [idx for idx, _ in counts.most_common(top_k)]
                specialization[e] = top_tokens

            return specialization

    def get_load_balance_report(self) -> Dict[str, float]:
        """
        Dapatkan laporan load balance saat ini.

        Returns:
            Dictionary dengan metrics load balance.
        """
        with torch.no_grad():
            ideal = 1.0 / self.num_experts
            max_deviation = (self.router.running_load - ideal).abs().max().item()
            mean_deviation = (self.router.running_load - ideal).abs().mean().item()
            bias_range = (self.router.bias.max() - self.router.bias.min()).item()

            return {
                "max_load_deviation": max_deviation,
                "mean_load_deviation": mean_deviation,
                "bias_range": bias_range,
                "bias_max": self.router.bias.max().item(),
                "bias_min": self.router.bias.min().item(),
                "update_count": self.router.update_count.item(),
                "ideal_load": ideal,
            }
