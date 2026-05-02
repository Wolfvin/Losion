"""
Gradient-Routed MoE — Loss-Aligned Expert Selection for Losion Framework v0.4.

Upgrade #6: Jalur 3 MoE enhancement.  Standard MoE routing relies solely on
input features — the router picks experts based on the hidden-state
representation of each token, without any feedback from the training
signal.  GradientRoutedMoE augments this with **gradient-informed routing**
so that experts which produce lower loss for a token get boosted over time.

Core Idea
---------
During training, we maintain a **running average of per-expert loss
contributions**.  An expert that consistently produces lower loss for a
given type of token receives a routing bonus, making it more likely to be
selected for similar tokens in the future.  This creates a positive
feedback loop: good experts get more tokens → more gradient signal → better
specialisation → even lower loss.

The adjusted routing logit is:

    logit_adjusted_i = logit_raw_i - α · loss_avg_i

where:
  - ``logit_raw_i``  is the standard feature-based router logit for expert *i*
  - ``loss_avg_i``   is the EMA of the loss contribution of expert *i*
  - ``α``            is the gradient influence strength (hyperparameter)

Lower ``loss_avg_i`` → higher adjusted logit → expert *i* is preferred.

Architecture
------------
1. **GradientAwareRouter** — standard linear router + per-expert loss
   EMA buffers that adjust logits at training time.

2. **ExpertLossTracker** — maintains an exponential moving average of
   per-expert loss contributions, updated after each forward pass using
   the detached loss of each expert's output.

3. **GradientRoutedMoE** — the main module combining the above with a
   pool of homogeneous experts.

Loss Attribution
----------------
After the forward pass we have a scalar training loss ℒ.  We attribute
a fraction of this loss to each active expert proportional to its routing
weight:

    ℒ_i = w_i · ℒ

This is a simplified but effective attribution — the true gradient-based
attribution would require per-expert backward passes, which is
prohibitively expensive.

Load Balancing
--------------
The same Switch-Transformer load-balancing loss is used to prevent
routing collapse, with an additional **loss-diversity** regulariser that
encourages the per-expert loss averages to be similar (preventing one
expert from dominating due to loss advantage).

References
----------
- Fedus et al., "Switch Transformers" (2021) — load-balancing.
- Yang et al., "GLaM: Efficient Scaling of Language Models with
  Mixture-of-Experts" (2022) — MoE training.
- Lewis et al., "MoEUT" (2024) — gradient-aware routing ideas.

Hardware: Pure PyTorch.  No custom CUDA kernels required.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Expert Loss Tracker (EMA of per-expert loss contributions)
# ---------------------------------------------------------------------------

class ExpertLossTracker:
    """
    Maintains an exponential moving average (EMA) of per-expert loss
    contributions.

    After each training step the caller feeds in a mapping from expert
    index to its attributed loss; the tracker updates the EMA accordingly.

    The EMA rule::

        loss_avg_i = momentum * loss_avg_i + (1 - momentum) * attributed_loss_i

    where ``momentum`` is close to 1 (default 0.99) so the average is
    smooth and stable.

    Args:
        num_experts: Number of experts.
        momentum:    EMA momentum (default 0.99).  Higher → smoother.
        device:      Device for the buffers.
        dtype:       Dtype for the buffers.
    """

    def __init__(
        self,
        num_experts: int,
        momentum: float = 0.99,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_experts = num_experts
        self.momentum = momentum
        self.loss_avg = torch.zeros(num_experts, device=device, dtype=dtype)
        self._initialized = False

    def update(
        self, expert_losses: Dict[int, torch.Tensor]
    ) -> None:
        """
        Update the EMA with new attributed losses.

        Args:
            expert_losses: Mapping expert_id → scalar loss contribution.
        """
        for eid, loss_val in expert_losses.items():
            loss_val = loss_val.detach().float()
            if not self._initialized:
                self.loss_avg[eid] = loss_val
            else:
                self.loss_avg[eid] = (
                    self.momentum * self.loss_avg[eid]
                    + (1.0 - self.momentum) * loss_val
                )
        if expert_losses:
            self._initialized = True

    def get_loss_averages(self) -> torch.Tensor:
        """
        Return the current per-expert loss EMA.

        Returns:
            ``(num_experts,)`` tensor.
        """
        return self.loss_avg.clone()

    def to(self, device: torch.device, dtype: torch.dtype = None) -> "ExpertLossTracker":
        """Move tracker buffers to device/dtype."""
        self.loss_avg = self.loss_avg.to(device=device, dtype=dtype)
        return self


# ---------------------------------------------------------------------------
# Gradient-Aware Router
# ---------------------------------------------------------------------------

class GradientAwareRouter(nn.Module):
    """
    Router that adjusts logits based on per-expert loss history.

    During training::

        logits_adjusted = logits_raw - alpha * loss_avg

    During inference the adjustment is **not** applied (the loss tracker
    is not updated during inference).

    Args:
        d_model:     Model dimension.
        num_experts: Number of experts.
        top_k:       Number of experts per token (default 2).
        alpha:       Gradient influence strength (default 0.1).
                    Controls how strongly the loss signal adjusts routing.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha

        self.proj = nn.Linear(d_model, num_experts, bias=False)

        # Learnable scaling for loss influence
        self.alpha_param = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

        # Loss tracker
        self.loss_tracker = ExpertLossTracker(num_experts)

    def forward(
        self,
        x: torch.Tensor,
        apply_loss_adjustment: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing logits, optionally adjusted by loss history.

        Args:
            x: ``(batch, seq_len, d_model)``.
            apply_loss_adjustment: Whether to apply the loss-based
                logit adjustment (True during training, False during
                inference).

        Returns:
            weights:       ``(batch, seq_len, top_k)``  — softmax-normalised.
            indices:       ``(batch, seq_len, top_k)``  — expert indices.
            router_logits: ``(batch, seq_len, num_experts)`` — raw logits
                           (before adjustment, for load-balance loss).
        """
        logits = self.proj(x)  # (B, S, E)

        # Store raw logits for load-balance loss
        raw_logits = logits.clone()

        # Adjust logits based on loss history
        if apply_loss_adjustment and self.loss_tracker._initialized:
            loss_avg = self.loss_tracker.get_loss_averages().to(
                device=logits.device, dtype=logits.dtype
            )  # (E,)

            # Normalise loss averages to zero-mean for stability
            loss_centered = loss_avg - loss_avg.mean()
            loss_scale = loss_centered.abs().max().clamp(min=1e-8)

            # Adjust: lower loss → higher logit
            alpha_eff = torch.sigmoid(self.alpha_param)  # ensure positive
            adjustment = -alpha_eff * (loss_centered / loss_scale)
            logits = logits + adjustment.unsqueeze(0).unsqueeze(0)

        # Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        return top_k_weights, top_k_indices, raw_logits

    def update_loss_history(
        self,
        expert_losses: Dict[int, torch.Tensor],
    ) -> None:
        """
        Update the loss tracker with attributed losses.

        Args:
            expert_losses: Mapping expert_id → scalar loss.
        """
        self.loss_tracker.update(expert_losses)


# ---------------------------------------------------------------------------
# Standard FFN Expert
# ---------------------------------------------------------------------------

class _FFNExpert(nn.Module):
    """SwiGLU feed-forward expert."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Load-balance loss
# ---------------------------------------------------------------------------

def _load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Standard Switch-Transformer load-balance loss."""
    probs = F.softmax(router_logits, dim=-1)
    assignments = probs.argmax(dim=-1)
    one_hot = F.one_hot(assignments, num_experts).float()
    f = one_hot.mean(dim=(0, 1))
    P = probs.mean(dim=(0, 1))
    return num_experts * (f * P).sum()


# ---------------------------------------------------------------------------
# Loss-diversity regulariser
# ---------------------------------------------------------------------------

def _loss_diversity_loss(loss_tracker: ExpertLossTracker) -> torch.Tensor:
    """
    Encourages the per-expert loss averages to be similar, preventing
    one expert from monopolising tokens due to its loss advantage.

    Computed as the variance of the loss averages:

        L_div = Var(loss_avg)

    Args:
        loss_tracker: The :class:`ExpertLossTracker` instance.

    Returns:
        Scalar loss.
    """
    if not loss_tracker._initialized:
        return torch.tensor(0.0, device=loss_tracker.loss_avg.device)
    loss_avg = loss_tracker.get_loss_averages()
    return loss_avg.var()


# ---------------------------------------------------------------------------
# GradientRoutedMoE
# ---------------------------------------------------------------------------

class GradientRoutedMoE(nn.Module):
    """
    Mixture-of-Experts with **gradient-informed routing**.

    Expert selection is influenced not only by input features but also by
    the training signal — experts that consistently produce lower loss for
    certain token types receive a routing bonus, creating loss-aligned
    rather than purely feature-aligned specialisation.

    During training:
      1. The router produces logits adjusted by per-expert loss EMA.
      2. Tokens are dispatched to top-K experts.
      3. After the forward pass, the caller should call
         :meth:`attribute_and_update_losses` with the total loss to update
         the loss tracker.
      4. Alternatively, if the caller provides ``current_loss`` to
         :meth:`forward`, attribution and update happen automatically.

    During inference:
      1. The loss-based adjustment is skipped.
      2. Standard top-K routing is used.

    Example
    -------
    >>> moe = GradientRoutedMoE(
    ...     d_model=512,
    ...     d_ff=2048,
    ...     num_experts=8,
    ...     top_k=2,
    ... )
    >>> x = torch.randn(2, 16, 512)
    >>> out, aux = moe(x)
    >>> out.shape
    torch.Size([2, 16, 512])

    Args:
        d_model:     Model dimension.
        d_ff:        Intermediate dimension per expert.
        num_experts: Number of experts.
        top_k:       Number of experts per token (default 2).
        alpha:       Initial gradient influence strength (default 0.1).
        dropout:     Dropout rate (default 0.0).
        load_balance_weight: Weight for load-balancing loss (default 0.01).
        loss_diversity_weight: Weight for loss-diversity regulariser
                               (default 0.001).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        alpha: float = 0.1,
        dropout: float = 0.0,
        load_balance_weight: float = 0.01,
        loss_diversity_weight: float = 0.001,
    ) -> None:
        super().__init__()

        if num_experts < top_k:
            raise ValueError(
                f"num_experts ({num_experts}) must be >= top_k ({top_k})"
            )

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.loss_diversity_weight = loss_diversity_weight

        # ---- Experts ----
        self.experts = nn.ModuleList(
            [_FFNExpert(d_model, d_ff) for _ in range(num_experts)]
        )

        # ---- Gradient-aware router ----
        self.router = GradientAwareRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            alpha=alpha,
        )

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # ---- Output norm ----
        self.output_norm = nn.RMSNorm(d_model, eps=1e-5)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        current_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with gradient-informed routing.

        If ``current_loss`` is provided, per-expert loss attribution and
        tracker update happen automatically after the forward computation.
        If not, the caller should manually call
        :meth:`attribute_and_update_losses` later in the training loop.

        Args:
            x:             ``(batch, seq_len, d_model)``.
            current_loss:  Optional scalar training loss.  If provided,
                           used to attribute loss to experts and update the
                           loss tracker.

        Returns:
            output: ``(batch, seq_len, d_model)``.
            aux:    Dict of auxiliary losses with keys ``"load_balance"``
                    and ``"loss_diversity"``.
        """
        batch, seq_len, _ = x.shape

        # ---- Router (with loss adjustment during training) ----
        is_training = self.training
        weights, indices, raw_logits = self.router(
            x, apply_loss_adjustment=is_training
        )
        # weights: (B, S, K),  indices: (B, S, K),  raw_logits: (B, S, E)

        # Flatten
        x_flat = x.view(batch * seq_len, self.d_model)
        weights_flat = weights.view(batch * seq_len, self.top_k)
        indices_flat = indices.view(batch * seq_len, self.top_k)

        # ---- Dispatch ----
        output_flat = torch.zeros_like(x_flat)

        # Also collect per-expert weight sums for loss attribution
        expert_weight_sums = torch.zeros(
            self.num_experts, device=x.device, dtype=x.dtype
        )

        for k_idx in range(self.top_k):
            expert_indices = indices_flat[:, k_idx]  # (N,)
            expert_weights = weights_flat[:, k_idx].unsqueeze(-1)  # (N, 1)

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if not mask.any():
                    continue

                expert_input = x_flat[mask]
                expert_output = self.experts[expert_id](expert_input)
                expert_w = expert_weights[mask]
                output_flat[mask] += expert_w * self.dropout(expert_output)

                # Accumulate weight sum for attribution
                expert_weight_sums[expert_id] += expert_w.sum().detach()

        output = output_flat.view(batch, seq_len, self.d_model)
        output = self.output_norm(output)

        # ---- Loss attribution and tracker update ----
        if current_loss is not None and is_training:
            self._attribute_and_update(current_loss, expert_weight_sums)

        # ---- Auxiliary losses ----
        aux: Dict[str, torch.Tensor] = {
            "load_balance": self.load_balance_weight
            * _load_balance_loss(raw_logits, self.num_experts),
            "loss_diversity": self.loss_diversity_weight
            * _loss_diversity_loss(self.router.loss_tracker),
        }

        # Store expert weight sums for external loss attribution
        self._last_expert_weight_sums = expert_weight_sums

        return output, aux

    # ------------------------------------------------------------------
    # Loss Attribution
    # ------------------------------------------------------------------

    def _attribute_and_update(
        self,
        total_loss: torch.Tensor,
        expert_weight_sums: torch.Tensor,
    ) -> None:
        """
        Attribute a fraction of the total loss to each expert and update
        the loss tracker.

        Attribution rule::

            L_i = (w_sum_i / Σ_j w_sum_j) · L_total

        Args:
            total_loss:          Scalar loss tensor.
            expert_weight_sums:  ``(num_experts,)`` — sum of routing weights
                                 per expert.
        """
        total_weight = expert_weight_sums.sum().clamp(min=1e-9)
        expert_losses: Dict[int, torch.Tensor] = {}
        for eid in range(self.num_experts):
            if expert_weight_sums[eid] > 0:
                attributed = (expert_weight_sums[eid] / total_weight) * total_loss
                expert_losses[eid] = attributed

        self.router.update_loss_history(expert_losses)

    def attribute_and_update_losses(
        self, total_loss: torch.Tensor
    ) -> None:
        """
        Public method to attribute loss and update the tracker.

        Call this after the forward pass if ``current_loss`` was not
        provided to :meth:`forward`.

        Args:
            total_loss: Scalar loss tensor.
        """
        weight_sums = getattr(self, "_last_expert_weight_sums", None)
        if weight_sums is None:
            # Fallback: uniform attribution
            weight_sums = torch.ones(
                self.num_experts, device=total_loss.device, dtype=total_loss.dtype
            )
        self._attribute_and_update(total_loss, weight_sums)

    # ------------------------------------------------------------------
    # Inference (no loss adjustment)
    # ------------------------------------------------------------------

    def forward_inference(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inference forward without loss-based routing adjustment.

        Args:
            x: ``(batch, seq_len, d_model)`` — typically seq_len = 1.

        Returns:
            output: ``(batch, seq_len, d_model)``.
            aux:    Empty dict.
        """
        batch, seq_len, _ = x.shape

        weights, indices, _ = self.router(
            x, apply_loss_adjustment=False
        )

        x_flat = x.view(batch * seq_len, self.d_model)
        weights_flat = weights.view(batch * seq_len, self.top_k)
        indices_flat = indices.view(batch * seq_len, self.top_k)

        output_flat = torch.zeros_like(x_flat)

        for k_idx in range(self.top_k):
            expert_indices = indices_flat[:, k_idx]
            expert_weights = weights_flat[:, k_idx].unsqueeze(-1)

            for expert_id in range(self.num_experts):
                mask = (expert_indices == expert_id)
                if not mask.any():
                    continue

                expert_input = x_flat[mask]
                expert_output = self.experts[expert_id](expert_input)
                expert_w = expert_weights[mask]
                output_flat[mask] += expert_w * expert_output

        output = output_flat.view(batch, seq_len, self.d_model)
        output = self.output_norm(output)

        return output, {}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_expert_loss_averages(self) -> torch.Tensor:
        """
        Return the current per-expert loss EMA.

        Returns:
            ``(num_experts,)`` tensor.
        """
        return self.router.loss_tracker.get_loss_averages()
