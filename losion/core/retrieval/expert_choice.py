"""
Expert Choice Routing — Optimal load balancing for MoE (Google Research, 2022).

Diadaptasi dari "Mixture-of-Experts with Expert Choice Routing"
(Zhou et al., Google Research, 2022): alih-alih token memilih expert
(token-choice routing), expert memilih token (expert-choice routing).

Perbedaan utama dengan token-choice routing:
1. Token-Choice: Setiap token memilih top-K experts → bisa terjadi load imbalance
2. Expert-Choice: Setiap expert memilih top-K tokens → load SEIMBANG secara otomatis

Keunggulan Expert Choice:
- **Guaranteed load balancing**: Setiap expert memproses jumlah token yang sama
- **No auxiliary loss**: Tidak perlu aux loss untuk load balancing
- **Better expert specialization**: Expert memilih token yang paling cocok
- **Simpler training**: Tidak perlu tuning aux loss weight
- **Flexible capacity**: Kapasitas per expert bisa dikonfigurasi

Mekanisme:
1. Hitung affinity score: S[i,j] = similarity(token_i, expert_j)
2. Untuk setiap expert j: pilih top-K tokens dengan score tertinggi
3. Setiap expert memproses K token terpilihnya
4. Output: weighted combination berdasarkan affinity scores

Limitasi yang diatasi:
- Token yang tidak dipilih oleh expert mana pun → "dropped tokens"
- Solusi: capacity factor > 1.0 atau shared expert sebagai fallback

Referensi:
- Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (2022)
- Google Flaxformer implementation: flaxformer/architectures/moe/routing.py

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExpertChoiceRoutingInfo:
    """Informasi routing untuk Expert Choice routing.

    Attributes:
        expert_indices: [num_experts, top_k] — token indices per expert
        expert_weights: [num_experts, top_k] — affinity scores per expert
        token_to_expert: [batch, seq, num_experts] — soft assignment
        dropped_tokens: Jumlah token yang tidak dipilih expert mana pun
        load_balance_score: Metric load balance (1.0 = perfect)
    """

    expert_indices: torch.Tensor  # [num_experts, top_k]
    expert_weights: torch.Tensor  # [num_experts, top_k]
    token_to_expert: torch.Tensor  # [batch, seq, num_experts]
    dropped_tokens: int
    load_balance_score: float


class ExpertChoiceRouter(nn.Module):
    """Expert Choice Router — experts choose tokens, not vice versa.

    Diadaptasi dari Google Research (2022): setiap expert memilih
    top-K tokens berdasarkan affinity score. Ini menjamin load
    balancing secara otomatis tanpa auxiliary loss.

    Alur:
    1. Hitung affinity: logits = gate_proj(token)  # [total_tokens, num_experts]
    2. Transpose: expert_view = logits.T  # [num_experts, total_tokens]
    3. Top-K per expert: setiap expert memilih top capacity tokens
    4. Normalisasi: softmax dalam expert selection
    5. Dispatch: kirim token terpilih ke expert masing-masing

    Args:
        d_model: Dimensi model.
        num_experts: Jumlah total experts.
        capacity_factor: Faktor kapasitas per expert.
            capacity = total_tokens * capacity_factor / num_experts
            Nilai 1.0 = setiap expert memproses rata-rata jumlah token
            Nilai > 1.0 = setiap expert memproses lebih banyak (overlap)
            Nilai < 1.0 = setiap expert memproses lebih sedikit (dropout)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        capacity_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # Gating network
        self.gate_proj = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, ExpertChoiceRoutingInfo]:
        """Hitung Expert Choice routing.

        Args:
            x: Input tensor [total_tokens, d_model] (sudah di-flatten)

        Returns:
            Tuple (gate_values, expert_indices, routing_info):
            - gate_values: [num_experts, capacity] — affinity scores
            - expert_indices: [num_experts, capacity] — token indices per expert
            - routing_info: Detail routing untuk monitoring
        """
        total_tokens = x.shape[0]

        # Hitung kapasitas per expert
        capacity = int(total_tokens * self.capacity_factor / self.num_experts)
        capacity = max(capacity, 1)

        # Hitung affinity scores
        logits = self.gate_proj(x)  # [total_tokens, num_experts]

        # Expert Choice: setiap expert memilih top-K tokens
        # Transpose untuk melihat dari perspektif expert
        expert_logits = logits.T  # [num_experts, total_tokens]

        # Top-K per expert
        top_k_scores, top_k_indices = torch.topk(
            expert_logits, capacity, dim=-1
        )  # [num_experts, capacity]

        # Normalisasi scores per expert
        gate_values = F.softmax(top_k_scores, dim=-1)  # [num_experts, capacity]

        # Hitung dropped tokens (token yang tidak dipilih expert mana pun)
        selected_mask = torch.zeros(total_tokens, dtype=torch.bool, device=x.device)
        for e in range(self.num_experts):
            selected_mask.scatter_(0, top_k_indices[e], True)
        dropped_tokens = total_tokens - selected_mask.sum().item()

        # Hitung soft token-to-expert assignment (untuk monitoring)
        token_to_expert = F.softmax(logits, dim=-1)  # [total_tokens, num_experts]

        # Load balance score: ideal = 1.0 (setiap expert = capacity)
        # Karena expert choice, ini secara alami = 1.0
        load_balance_score = 1.0  # Guaranteed by expert choice

        routing_info = ExpertChoiceRoutingInfo(
            expert_indices=top_k_indices,
            expert_weights=gate_values,
            token_to_expert=token_to_expert,
            dropped_tokens=dropped_tokens,
            load_balance_score=load_balance_score,
        )

        return gate_values, top_k_indices, routing_info


class ExpertChoiceMoE(nn.Module):
    """Mixture of Experts dengan Expert Choice Routing.

    Menggabungkan Expert Choice routing dengan MoE layer,
    menggantikan token-choice routing tradisional.

    Keunggulan dibanding BiasRouter (DeepSeek-V3 style):
    - Load balancing dijamin secara otomatis
    - Tidak perlu manual bias update
    - Expert specialization lebih baik (expert memilih yang cocok)
    - Implementasi lebih sederhana

    Kompatibel dengan shared expert (DeepSeek-V3 style):
    - Shared expert selalu aktif untuk semua token
    - Routed experts memilih token via expert choice
    - Output = shared + routed

    Args:
        d_model: Dimensi model.
        d_ff: Dimensi feed-forward per expert.
        num_experts: Jumlah total experts.
        capacity_factor: Faktor kapasitas per expert.
        use_shared_expert: Gunakan shared expert (selalu aktif).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 64,
        capacity_factor: float = 1.0,
        use_shared_expert: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.use_shared_expert = use_shared_expert

        # === Expert FFNs ===
        self.experts = nn.ModuleList([
            self._make_expert(d_model, d_ff) for _ in range(num_experts)
        ])

        # === Shared Expert ===
        if use_shared_expert:
            self.shared_expert = self._make_expert(d_model, d_ff)
            self.shared_expert_scale = nn.Parameter(torch.ones(1))

        # === Expert Choice Router ===
        self.router = ExpertChoiceRouter(d_model, num_experts, capacity_factor)

        # === Fallback for Dropped Tokens ===
        # Token yang tidak dipilih expert mana pun dilewatkan melalui
        # lightweight linear layer
        self.fallback = nn.Linear(d_model, d_model, bias=False)

    def _make_expert(self, d_model: int, d_ff: int) -> nn.Module:
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
    ) -> Tuple[torch.Tensor, ExpertChoiceRoutingInfo]:
        """Forward pass Expert Choice MoE.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tuple (output, routing_info)
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

        # === Expert Choice Routing ===
        gate_values, expert_indices, routing_info = self.router(x_flat)
        # gate_values: [num_experts, capacity]
        # expert_indices: [num_experts, capacity]

        capacity = expert_indices.shape[1]

        # === Compute Routed Expert Output ===
        routed_output = torch.zeros_like(x_flat)

        # Track token counts for renormalization (prevent double-counting
        # when same token is selected by multiple experts)
        token_counts = torch.zeros(total_tokens, 1, device=x.device, dtype=x_flat.dtype)

        # Track which tokens were processed
        processed_tokens = torch.zeros(
            total_tokens, dtype=torch.bool, device=x.device
        )

        for e in range(self.num_experts):
            # Token indices for this expert
            token_indices = expert_indices[e]  # [capacity]
            gate_weights = gate_values[e]  # [capacity]

            # Gather input tokens
            expert_input = x_flat[token_indices]  # [capacity, d_model]

            # Process through expert
            expert_output = self._expert_forward(
                self.experts[e], expert_input
            )  # [capacity, d_model]

            # Weight by gate values and accumulate
            weighted_output = gate_weights.unsqueeze(-1) * expert_output

            # Scatter back to output
            for c in range(capacity):
                idx = token_indices[c].item()
                routed_output[idx] += weighted_output[c]
                token_counts[idx] += 1
                processed_tokens[idx] = True

        # Renormalize by token count to prevent double-counting
        routed_output = routed_output / token_counts.clamp(min=1)

        # === Handle Dropped Tokens ===
        dropped_mask = ~processed_tokens
        if dropped_mask.any():
            # Gunakan fallback layer untuk dropped tokens
            dropped_input = x_flat[dropped_mask]
            fallback_output = self.fallback(dropped_input)
            routed_output[dropped_mask] += fallback_output

        # === Combine ===
        output = (shared_output + routed_output).reshape(orig_shape)

        return output, routing_info

    def get_expert_specialization(
        self, x: torch.Tensor, top_k: int = 5
    ) -> Dict[int, List[int]]:
        """Analisis spesialisasi expert.

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
            x_flat = x.reshape(-1, self.d_model)
            _, expert_indices, _ = self.router(x_flat)

            specialization = {}
            for e in range(self.num_experts):
                token_indices = expert_indices[e].tolist()
                # Ambil top_k tokens yang paling sering muncul
                from collections import Counter
                counts = Counter(token_indices)
                top_tokens = [idx for idx, _ in counts.most_common(top_k)]
                specialization[e] = top_tokens

            return specialization
