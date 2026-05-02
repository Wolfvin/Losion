"""
Routing Mamba (RoM) — Sparse Mixture of SSM Projection Experts.

Implementation of Routing Mamba from Microsoft Research (NeurIPS 2025).

Scale SSM parameters using sparse mixtures of linear projection experts,
combining MoE routing efficiency with SSM sequence modeling. Instead of a
single set of B, C, dt projections (as in Mamba-2), Routing Mamba maintains
multiple expert projection sets and routes each token to a sparse subset via
learned top-K gating. A_log and D remain shared across all experts.

Load balancing: DeepSeek-V3 aux-loss-free approach with dynamic bias updated
via EMA statistics rather than gradient-based auxiliary losses.

References:
    - Routing Mamba (RoM): Microsoft Research, NeurIPS 2025
      Paper: https://neurips.cc/virtual/2025/poster/116256
    - Gu & Dao, "Mamba-2: A Generalized State Space Model with Structured
      State Space Duality" (2024), arXiv:2405.21060
    - DeepSeek-AI, "DeepSeek-V3" (2024) — Aux-loss-free load balancing

Compatible with Losion's SSM pathway interface (same forward signature as
Mamba2SSD, with an additional aux_loss return value).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.core.ssm.mamba2 import ssd_chunk_scan


@dataclass
class RoutingMambaConfig:
    """Configuration for Routing Mamba (RoM) layer.

    Attributes:
        num_experts: Total number of SSM projection experts.
        num_active_experts: Number of experts activated per token (top-K).
        d_state: SSM state dimension (N).
        d_conv: Local causal convolution kernel width.
        expand: Inner dimension expansion factor (d_inner = expand * d_model).
        d_model: Model input/output dimension.
        use_shared_expert: Whether to include a shared (non-routed) expert.
        bias_lr: Learning rate for DeepSeek-V3 style dynamic bias routing.
    """

    num_experts: int = 4
    num_active_experts: int = 2
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    d_model: int = 768
    use_shared_expert: bool = True
    bias_lr: float = 0.01


class SSMExpertRouter(nn.Module):
    """Routes tokens to SSM projection experts with aux-loss-free balancing.

    Uses DeepSeek-V3 style dynamic bias: per-expert bias values are updated
    via EMA tracking of expert load statistics. Overloaded experts receive
    negative bias (discouraged), underloaded get positive bias (encouraged).

    Args:
        d_inner: Inner dimension of the SSM layer (router input dim).
        num_experts: Total number of projection experts.
        num_active_experts: Number of active experts per token (top-K).
        bias_lr: Learning rate for periodic bias updates.
    """

    def __init__(
        self,
        d_inner: int,
        num_experts: int,
        num_active_experts: int,
        bias_lr: float = 0.01,
    ) -> None:
        super().__init__()

        self.d_inner = d_inner
        self.num_experts = num_experts
        self.num_active_experts = min(num_active_experts, num_experts)
        self.bias_lr = bias_lr

        self.gate_proj = nn.Linear(d_inner, num_experts, bias=False)
        self.register_buffer("bias", torch.zeros(num_experts))
        self.register_buffer("running_load", torch.zeros(num_experts))
        self.register_buffer("update_count", torch.tensor(0, dtype=torch.long))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to SSM projection experts.

        Args:
            x: Input hidden state, shape ``(batch, seq_len, d_inner)``.

        Returns:
            Tuple ``(expert_weights, expert_indices, aux_loss)``:

            - expert_weights — ``(batch, seq_len, K)`` softmax-normalised
              routing weights for the selected experts.
            - expert_indices — ``(batch, seq_len, K)`` integer indices of
              the top-K selected experts.
            - aux_loss — scalar, load-balance monitoring metric (near zero
              when balanced). Not used for gradient updates; actual
              balancing is done via :meth:`update_bias`.
        """
        batch, seq_len, _ = x.shape

        logits = self.gate_proj(x) + self.bias  # (batch, seq, E)

        # Top-K expert selection
        top_k_weights, top_k_indices = torch.topk(
            logits, self.num_active_experts, dim=-1
        )
        expert_weights = F.softmax(top_k_weights, dim=-1)

        # EMA load tracking (non-gradient)
        with torch.no_grad():
            expert_loads = torch.zeros(
                self.num_experts, dtype=torch.float, device=x.device
            )
            for k in range(self.num_active_experts):
                for e in range(self.num_experts):
                    expert_loads[e] += (top_k_indices[:, :, k] == e).float().sum()

            total_tokens = batch * seq_len
            instant_load = expert_loads / (total_tokens + 1e-8)
            self.running_load.mul_(0.9).add_(instant_load, alpha=0.1)
            self.update_count.add_(1)

        # Monitoring metric (NOT a training loss)
        ideal = 1.0 / self.num_experts
        avg_routing = F.softmax(logits, dim=-1).mean(dim=(0, 1))
        aux_loss = (avg_routing - ideal).pow(2).sum()

        return expert_weights, top_k_indices, aux_loss

    def update_bias(self) -> None:
        """Update routing bias from EMA load statistics (DeepSeek-V3).

        Overloaded experts get negative bias, underloaded get positive.
        Non-gradient — does not affect representation quality.
        Call periodically during training (e.g. every N steps).
        """
        with torch.no_grad():
            if self.update_count < 1:
                return
            ideal = 1.0 / self.num_experts
            relative_load = self.running_load / (ideal + 1e-8)
            deviation = relative_load - 1.0
            bias_update = -self.bias_lr * deviation
            self.bias.add_(bias_update.clamp(-0.1, 0.1))
            self.bias.clamp_(-2.0, 2.0)


class RoutingMamba(nn.Module):
    """Routing Mamba (RoM) — MoE routing over SSM projections.

    Drop-in replacement for :class:`Mamba2SSD` that scales SSM parameters
    using sparse mixtures of linear projection experts. Each expert provides
    its own B, C, dt projections while sharing the A_log matrix and D skip
    connection. Top-K experts are selected per token; their projections are
    combined via weighted sum. An optional shared expert (always active)
    additively contributes its own B, C, dt.

    Forward signatures (compatible with Losion SSM pathway):

    * ``forward(x, state=None) -> (output, updated_state, aux_loss)``
    * ``forward_inference(x, state=None) -> (output, updated_state)``

    Args:
        config: :class:`RoutingMambaConfig` with hyperparameters.
        d_model: Convenience override for ``config.d_model``.
        chunk_size: Chunk size for SSD parallel scan during training.
        dt_min: Lower bound for dt initialisation.
        dt_max: Upper bound for dt initialisation.
    """

    def __init__(
        self,
        config: Optional[RoutingMambaConfig] = None,
        d_model: Optional[int] = None,
        chunk_size: int = 256,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()

        if config is None:
            config = RoutingMambaConfig()
        if d_model is not None:
            config.d_model = d_model

        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.num_experts = config.num_experts
        self.num_active_experts = config.num_active_experts
        self.use_shared_expert = config.use_shared_expert
        self.chunk_size = chunk_size
        self.d_inner = int(config.expand * config.d_model)

        # Input projection (same as Mamba-2): d_model -> d_inner * 2
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Causal conv1d (same as Mamba-2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # Shared A_log (same as Mamba-2): (d_inner, d_state), negative values
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))

        # Shared D skip connection: (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Expert-specific B, C, dt projection layers
        self.B_projs = nn.ModuleList([
            nn.Linear(self.d_inner, self.d_state, bias=False)
            for _ in range(self.num_experts)
        ])
        self.C_projs = nn.ModuleList([
            nn.Linear(self.d_inner, self.d_state, bias=False)
            for _ in range(self.num_experts)
        ])
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.d_inner, self.d_inner, bias=True)
            for _ in range(self.num_experts)
        ])

        # Mamba-2 style log-uniform dt bias initialisation
        for e in range(self.num_experts):
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            self.dt_projs[e].bias.data.copy_(
                torch.log(torch.exp(dt_init) - 1)
            )

        # Shared expert (always active, non-routed)
        if self.use_shared_expert:
            self.shared_B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
            self.shared_C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
            self.shared_dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
            self.shared_expert_gate = nn.Parameter(torch.ones(1))

            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            self.shared_dt_proj.bias.data.copy_(
                torch.log(torch.exp(dt_init) - 1)
            )

        # SSM Expert Router
        self.router = SSMExpertRouter(
            d_inner=self.d_inner,
            num_experts=self.num_experts,
            num_active_experts=self.num_active_experts,
            bias_lr=config.bias_lr,
        )

        # Output projection and norm (same as Mamba-2)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.norm = nn.RMSNorm(self.d_inner, eps=1e-5)

    def _compute_expert_params(
        self,
        x_conv: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute effective B, C, dt via soft mixture of expert projections.

        Gathers the B, C, dt from the top-K selected experts and combines
        them using routing weights to produce effective per-token parameters.

        Args:
            x_conv: Convolved input, ``(B, L, d_inner)``.
            expert_weights: Routing weights, ``(B, L, K)``.
            expert_indices: Expert indices, ``(B, L, K)``.

        Returns:
            ``(B_eff, C_eff, dt_eff)`` — ``(B, L, d_state)`` for B/C
            and ``(B, L, d_inner)`` for dt.
        """
        # Compute all expert projections (batched)
        B_all = torch.stack(
            [self.B_projs[e](x_conv) for e in range(self.num_experts)], dim=2
        )  # (B, L, E, d_state)
        C_all = torch.stack(
            [self.C_projs[e](x_conv) for e in range(self.num_experts)], dim=2
        )
        dt_all = torch.stack(
            [self.dt_projs[e](x_conv) for e in range(self.num_experts)], dim=2
        )  # (B, L, E, d_inner)

        # Gather top-K expert projections
        idx_B = expert_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_state)
        idx_dt = expert_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_inner)

        B_selected = torch.gather(B_all, 2, idx_B)    # (B, L, K, d_state)
        C_selected = torch.gather(C_all, 2, idx_B)     # same indices
        dt_selected = torch.gather(dt_all, 2, idx_dt)  # (B, L, K, d_inner)

        # Weighted combination over K selected experts
        w = expert_weights.unsqueeze(-1)  # (B, L, K, 1)
        B_eff = (B_selected * w).sum(dim=2)    # (B, L, d_state)
        C_eff = (C_selected * w).sum(dim=2)
        dt_eff = (dt_selected * w).sum(dim=2)  # (B, L, d_inner)

        return B_eff, C_eff, dt_eff

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training with chunk-based parallel scan.

        Args:
            x: Input tensor, ``(batch, seq_len, d_model)``.
            state: Optional SSM state, ``(batch, d_inner, d_state)``.

        Returns:
            ``(output, updated_state, aux_loss)``:
            - output — ``(batch, seq_len, d_model)``
            - updated_state — ``(batch, d_inner, d_state)``
            - aux_loss — scalar, load-balance monitoring metric
        """
        batch, seq_len, _ = x.shape

        if seq_len == 0:
            dummy_out = torch.zeros(
                batch, 0, self.d_model, dtype=x.dtype, device=x.device
            )
            dummy_state = (
                state if state is not None
                else torch.zeros(
                    batch, self.d_inner, self.d_state,
                    dtype=x.dtype, device=x.device,
                )
            )
            return dummy_out, dummy_state, torch.tensor(0.0, device=x.device)

        # Step 1: Input projection + gating split (same as Mamba-2)
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # Step 2: Causal conv1d (same as Mamba-2)
        x_conv = x_proj.transpose(1, 2)          # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)           # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Step 3: Route tokens to experts
        expert_weights, expert_indices, aux_loss = self.router(x_conv)

        # Step 4: Effective B, C, dt via expert mixture
        B, C, dt = self._compute_expert_params(
            x_conv, expert_weights, expert_indices
        )

        # Step 5: Add shared expert contributions (if enabled)
        if self.use_shared_expert:
            gate_val = torch.sigmoid(self.shared_expert_gate)
            B = B + gate_val * self.shared_B_proj(x_conv)
            C = C + gate_val * self.shared_C_proj(x_conv)
            dt = dt + gate_val * self.shared_dt_proj(x_conv)

        # Step 6: Discretise dt via softplus
        dt = F.softplus(dt + 1e-4)  # (B, L, d_inner)

        # Step 7: Shared A matrix (negative for stability)
        A = -torch.exp(self.A_log.float()).to(dtype=x_conv.dtype)
        dt_avg = dt.mean(dim=-1)  # (B, L)
        A_avg = A.mean(dim=0)     # (d_state,)

        # Step 8: SSD chunk scan
        y, final_state = ssd_chunk_scan(
            x_seq=x_conv,
            A=A_avg.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1),
            B=B, C=C, dt=dt_avg,
            chunk_size=self.chunk_size,
            initial_state=state,
        )

        # Step 9: Skip connection D (shared across experts)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # Step 10: Gating + norm + output projection
        y = y * F.silu(z)
        y = self.norm(y)
        output = self.out_proj(y)

        return output, final_state, aux_loss

    def forward_inference(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for autoregressive inference — O(1) per token.

        Args:
            x: Input tensor, ``(batch, 1, d_model)``.
            state: SSM state, ``(batch, d_inner, d_state)``. If *None*,
                a zero state is initialised.

        Returns:
            ``(output, updated_state)``.
        """
        batch = x.shape[0]

        if state is None:
            state = torch.zeros(
                batch, self.d_inner, self.d_state,
                dtype=x.dtype, device=x.device,
            )

        # Input projection + gating split
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # Simplified conv for single-token (same as Mamba2SSD — in
        # production a conv state cache would be maintained)
        x_conv = F.silu(x_proj)

        # Route single token
        expert_weights, expert_indices, _ = self.router(x_conv)

        # Effective B, C, dt via expert mixture
        B, C, dt = self._compute_expert_params(
            x_conv, expert_weights, expert_indices
        )

        # Shared expert
        if self.use_shared_expert:
            gate_val = torch.sigmoid(self.shared_expert_gate)
            B = B + gate_val * self.shared_B_proj(x_conv)
            C = C + gate_val * self.shared_C_proj(x_conv)
            dt = dt + gate_val * self.shared_dt_proj(x_conv)

        dt = F.softplus(dt + 1e-4)  # (B, 1, d_inner)
        A = -torch.exp(self.A_log.float()).to(dtype=x_conv.dtype)

        # Sequential SSM update: O(1) per token
        dt_sq = dt.squeeze(1)                                    # (B, d_inner)
        dA = torch.exp(dt_sq.unsqueeze(-1) * A.unsqueeze(0))    # (B, d_inner, d_state)
        dB = dt_sq.unsqueeze(-1) * B.squeeze(1).unsqueeze(1)    # (B, d_inner, d_state)
        dBx = x_conv.squeeze(1).unsqueeze(-1) * dB              # (B, d_inner, d_state)
        new_state = dA * state + dBx

        # Output: y = C^T h + D * x
        y = torch.sum(C.squeeze(1).unsqueeze(1) * new_state, dim=-1)
        y = y + x_conv.squeeze(1) * self.D.unsqueeze(0)
        y = y * F.silu(z.squeeze(1))
        y = self.norm(y.unsqueeze(1)).squeeze(1)
        output = self.out_proj(y)

        return output.unsqueeze(1), new_state
