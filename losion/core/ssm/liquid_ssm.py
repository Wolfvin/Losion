"""
Liquid SSM Variant — Adaptive Compute Depth for Losion Framework v0.4.

Extends the v0.3 SSMTerpaduLayer with liquid (adaptive) time constants
and per-token compute depth selection. Tokens that are "easy" can
early-exit after a single sub-layer pass, while complex tokens receive
the full multi-layer treatment.

Key components:
1. ComplexityGate — learned module estimating per-token, per-head complexity
2. LiquidSSD — Mamba-2 SSD with liquid (input-adaptive) time constants
3. LiquidSSMTerpaduLayer — enhanced SSMTerpaduLayer with adaptive depth

Depth levels:
    1 (fast)     — single SSD pass only
    2 (standard) — SSD + one additional sub-layer (WKV or Delta)
    3 (deep)     — full interleaving through all sub-layers

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
No custom kernels required.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Depth level constants
# ---------------------------------------------------------------------------

DEPTH_FAST = 1          # Single SSD pass — minimal compute
DEPTH_STANDARD = 2      # SSD + one additional sub-layer
DEPTH_DEEP = 3          # Full interleaving through all sub-layers
NUM_DEPTH_LEVELS = 3


# ---------------------------------------------------------------------------
# ComplexityGate
# ---------------------------------------------------------------------------

class ComplexityGate(nn.Module):
    """
    Learned module that estimates per-token complexity and maps it to a
    discrete depth level (1 = fast, 2 = standard, 3 = deep).

    Architecture
    ------------
    A lightweight MLP with a bottleneck that projects the input into a
    compact "complexity embedding" before predicting depth logits.

        input (d_model)
          → Linear(d_model, bottleneck_dim)
          → SiLU
          → Linear(bottleneck_dim, bottleneck_dim)
          → SiLU
          → Linear(bottleneck_dim, 3)   # depth logits

    The gate also produces a continuous **complexity scalar** in [0, 1]
    (via sigmoid on an auxiliary head) that is used by LiquidSSD to
    modulate time constants.

    Args:
        d_model:       Model dimension.
        bottleneck_dim: Width of the bottleneck MLP (default 64).
        num_heads:     Number of SSM heads — used to produce per-head
                       complexity scores (default 8).
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads

        # --- Depth-classification head ---
        self.depth_mlp = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, NUM_DEPTH_LEVELS, bias=True),
        )

        # --- Continuous complexity head (for liquid time-constant modulation) ---
        self.complexity_head = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, num_heads, bias=True),
        )

        # Initialize final layer to bias toward "standard" depth at startup
        with torch.no_grad():
            self.depth_mlp[-1].bias.zero_()
            self.depth_mlp[-1].bias[1].fill_(1.0)   # slight bias to level 2

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate token complexity.

        Args:
            x: Input tensor, shape ``(batch, seq_len, d_model)``.

        Returns:
            depth_logits:   ``(batch, seq_len, 3)`` — unnormalised scores
                            for depth levels [fast, standard, deep].
            depth_probs:    ``(batch, seq_len, 3)`` — softmax probabilities.
            complexity:     ``(batch, seq_len, num_heads)`` — continuous
                            complexity scalar per head, in ``(0, 1)``.
        """
        depth_logits = self.depth_mlp(x)                           # (B, S, 3)
        depth_probs = F.softmax(depth_logits, dim=-1)              # (B, S, 3)
        complexity = torch.sigmoid(self.complexity_head(x))        # (B, S, H)
        return depth_logits, depth_probs, complexity


# ---------------------------------------------------------------------------
# LiquidSSD — Mamba-2 SSD with Liquid (Adaptive) Time Constants
# ---------------------------------------------------------------------------

class LiquidSSD(nn.Module):
    """
    Mamba-2 SSD with **liquid** (input-adaptive) time constants.

    In standard Mamba-2 the step-size ``dt`` and decay ``A`` are
    data-dependent but *not* explicitly conditioned on a running
    estimate of input complexity.  LiquidSSD adds a **complexity
    modulation** that:

    1. Receives a per-token, per-head complexity scalar from
       :class:`ComplexityGate`.
    2. Uses it to rescale the effective ``dt`` so that *complex* tokens
       get larger time-steps (slower state decay → more memory), while
       *simple* tokens get smaller time-steps (faster decay → less
       memory, more local).
    3. Also modulates the ``A`` matrix via a learned mixing coefficient.

    The liquid time-constant rule::

        dt_eff  = dt_base * (1 + complexity_scale * (2 * complexity - 1))
        A_eff   = A_base  * (1 - mix_coeff * complexity)

    where ``complexity_scale`` and ``mix_coeff`` are learnable parameters
    initialised conservatively so the layer starts close to plain Mamba-2.

    Args:
        d_model:      Model dimension.
        d_state:      SSM state dimension (default 128).
        d_conv:       Local convolution width (default 4).
        expand:       Expansion factor (default 2).
        chunk_size:   SSD chunk size (default 256).
        num_heads:    Number of heads for complexity modulation (default 8).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 256,
        num_heads: int = 8,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        use_bias: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.d_inner = int(expand * d_model)
        self.dt_init_floor = dt_init_floor

        # ---- Input projection (same as Mamba2SSD) ----
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=use_bias)

        # ---- Local causal convolution ----
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # ---- SSM parameter projections ----
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # ---- dt bias (per-channel, same init as Mamba2SSD) ----
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_softplus = torch.log(torch.exp(dt_init) - 1)
        self.dt_bias = nn.Parameter(inv_softplus)

        # ---- A in log-domain ----
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))

        # ---- D skip connection ----
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ---- Liquid modulation parameters ----
        # complexity_scale: how strongly complexity rescales dt
        # Initialised near zero so the layer starts as plain Mamba-2
        self.complexity_scale = nn.Parameter(torch.zeros(num_heads))
        # mix_coeff: how strongly complexity modulates A
        self.mix_coeff = nn.Parameter(torch.zeros(num_heads))

        # ---- Complexity-to-d_inner projection ----
        # We receive per-head complexity (num_heads) and must produce
        # a per-d_inner scaling factor.  Simple learned linear map.
        self.complexity_to_inner = nn.Linear(num_heads, self.d_inner, bias=False)

        # ---- Output ----
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)
        self.norm = nn.RMSNorm(self.d_inner, eps=1e-5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_dt(self) -> torch.Tensor:
        """Return dt bias after softplus."""
        return F.softplus(self.dt_bias + 1e-4)

    def _apply_liquid_modulation(
        self,
        dt: torch.Tensor,
        complexity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Modulate dt with per-head complexity.

        The modulation rule::

            modulation = 1 + scale * (2 * c_projected - 1)

        where ``c_projected`` is complexity projected to ``d_inner``
        dimensions via a learned linear map.  This means:

        * complexity ≈ 0.5 → modulation ≈ 1 (no change)
        * complexity ≈ 1.0 → modulation > 1 (larger dt, more memory)
        * complexity ≈ 0.0 → modulation < 1 (smaller dt, less memory)

        Args:
            dt:          ``(batch, seq_len, d_inner)``
            complexity:  ``(batch, seq_len, num_heads)``

        Returns:
            dt_eff: ``(batch, seq_len, d_inner)``
        """
        # Project complexity from num_heads → d_inner
        # complexity: (B, S, H) → (B, S, d_inner)
        c_inner = self.complexity_to_inner(complexity)  # (B, S, d_inner)

        # Per-d_inner scale (learnable, initialised near 0)
        # complexity_scale has shape (num_heads,); expand to (d_inner,)
        # so each head group has its own scale instead of a single scalar.
        scale = torch.sigmoid(self.complexity_scale)  # (num_heads,)
        channels_per_head = self.d_inner // self.complexity_scale.shape[0]
        scale_expanded = scale.repeat_interleave(channels_per_head)  # (d_inner,)
        # Pad if d_inner is not evenly divisible
        if scale_expanded.shape[0] < self.d_inner:
            pad = self.d_inner - scale_expanded.shape[0]
            scale_expanded = F.pad(scale_expanded, (0, pad), value=0.0)
        # Reshape for broadcasting: (1, 1, d_inner)
        scale_expanded = scale_expanded.unsqueeze(0).unsqueeze(0)

        # Modulation: (1 + scale * (2 * c_projected - 1))
        # c_inner ∈ (0, 1)-ish after sigmoid in the gate,
        # but complexity_to_inner is a linear layer so values
        # are unbounded.  We apply sigmoid to normalise first.
        c_normalised = torch.sigmoid(c_inner)  # (B, S, d_inner) ∈ (0, 1)

        modulation = 1.0 + scale_expanded * (2.0 * c_normalised - 1.0)

        dt_eff = dt * modulation
        # Clamp to avoid numerical issues
        dt_eff = dt_eff.clamp(min=1e-6, max=1.0)
        return dt_eff

    # ------------------------------------------------------------------
    # Sequential SSM scan (same algorithm as Mamba2SSD, but with liquid dt)
    # ------------------------------------------------------------------

    @staticmethod
    def _ssd_scan(
        x_seq: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential SSM scan with per-token liquid dt.

        Args:
            x_seq:          ``(batch, seq_len, d_inner)``
            A:              ``(batch, seq_len, d_state)`` — negative
            B:              ``(batch, seq_len, d_state)``
            C:              ``(batch, seq_len, d_state)``
            dt:             ``(batch, seq_len)`` — liquid step sizes
            initial_state:  ``(batch, d_inner, d_state)`` or None

        Returns:
            (output, final_state)
        """
        batch, seq_len, d_inner = x_seq.shape

        if initial_state is None:
            h = torch.zeros(
                batch, d_inner, A.shape[-1],
                dtype=x_seq.dtype, device=x_seq.device,
            )
        else:
            h = initial_state.clone()

        dA = torch.exp(dt.unsqueeze(-1) * A)   # (B, S, d_state)
        dB = dt.unsqueeze(-1) * B               # (B, S, d_state)

        outputs = []
        for t in range(seq_len):
            h = h * dA[:, t, :].unsqueeze(1)
            h = h + x_seq[:, t, :].unsqueeze(-1) * dB[:, t, :].unsqueeze(1)
            y_t = torch.sum(h * C[:, t, :].unsqueeze(1), dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, h

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input: torch.Tensor,
        complexity: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional liquid time-constant modulation.

        Args:
            input:           ``(batch, seq_len, d_model)``
            complexity:      ``(batch, seq_len, num_heads)`` from
                             :class:`ComplexityGate`.  If *None*, the
                             layer behaves exactly like a plain Mamba-2 SSD.
            initial_state:   ``(batch, d_inner, d_state)`` or None.

        Returns:
            (output, final_state)
        """
        batch, seq_len, _ = input.shape

        if seq_len == 0:
            dummy_out = torch.zeros(
                batch, 0, self.d_model, dtype=input.dtype, device=input.device
            )
            dummy_state = (
                initial_state
                if initial_state is not None
                else torch.zeros(
                    batch, self.d_inner, self.d_state,
                    dtype=input.dtype, device=input.device,
                )
            )
            return dummy_out, dummy_state

        # ---- Step 1: Input projection ----
        xz = self.in_proj(input)
        x, z = xz.chunk(2, dim=-1)

        # ---- Step 2: Local causal convolution ----
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # ---- Step 3: SSM parameters ----
        ssm_params = self.x_proj(x_conv)
        B = ssm_params[..., : self.d_state]
        C = ssm_params[..., self.d_state : self.d_state * 2]

        dt_full = F.softplus(
            self.dt_proj(x_conv) + self.dt_bias.unsqueeze(0).unsqueeze(0) + self.dt_init_floor
        )  # (B, S, d_inner)

        # ---- Step 3b: Liquid modulation ----
        if complexity is not None:
            dt_full = self._apply_liquid_modulation(dt_full, complexity)

        # ---- Step 4: A in discrete domain ----
        A = -torch.exp(self.A_log.float()).to(dtype=x_conv.dtype)  # (d_inner, d_state)

        # If complexity is given, also modulate A
        if complexity is not None:
            mix = torch.sigmoid(self.mix_coeff)  # (H,) positive
            # Reduce complexity to a scalar per token
            c_scalar = complexity.mean(dim=-1)  # (B, S)
            # A modulation: A_eff = A * (1 - mix * complexity)
            # Per-channel modulation: mix is (H,) → scale per channel
            # Expand mix to (d_inner,): each head covers channels_per_head channels
            channels_per_head = self.d_inner // self.num_heads
            mix_expanded = mix.repeat_interleave(channels_per_head)
            if mix_expanded.shape[0] < self.d_inner:
                pad = self.d_inner - mix_expanded.shape[0]
                mix_expanded = F.pad(mix_expanded, (0, pad), value=0.0)
            # A_eff per channel: (d_inner, d_state)
            a_mod = 1.0 - (mix_expanded.unsqueeze(-1) * c_scalar.mean(dim=-1).mean(dim=-1)).unsqueeze(-1)
            A_eff = A * a_mod
        else:
            A_eff = A

        dt_eff = dt_full

        # ---- Step 5: SSD scan with per-channel dt and A ----
        from losion.core.ssm.mamba2 import ssd_chunk_scan as _ssd_chunk_scan
        y, final_state = _ssd_chunk_scan(
            x_seq=x_conv,
            A=A_eff,  # (d_inner, d_state) — per-channel
            B=B,
            C=C,
            dt=dt_eff,  # (B, S, d_inner) — per-channel
            initial_state=initial_state,
        )

        # ---- Step 6: Skip connection D ----
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # ---- Step 7: Gating + output ----
        y = y * F.silu(z)
        y = self.norm(y)
        output = self.out_proj(y)

        return output, final_state

    def forward_inference(
        self,
        input: torch.Tensor,
        state: torch.Tensor,
        complexity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-token inference (O(1) per token) with optional liquid
        modulation.

        Args:
            input:       ``(batch, 1, d_model)``
            state:       ``(batch, d_inner, d_state)``
            complexity:  ``(batch, 1, num_heads)`` or None.

        Returns:
            (output, new_state)
        """
        batch = input.shape[0]

        xz = self.in_proj(input)
        x, z = xz.chunk(2, dim=-1)

        # Skip conv1d for single-token (assumes cached)
        x_conv = F.silu(x)

        ssm_params = self.x_proj(x_conv)
        B = ssm_params[..., : self.d_state]
        C = ssm_params[..., self.d_state:self.d_state * 2]

        dt_full = F.softplus(
            self.dt_proj(x_conv) + self.dt_bias.unsqueeze(0).unsqueeze(0) + self.dt_init_floor
        )

        if complexity is not None:
            dt_full = self._apply_liquid_modulation(dt_full, complexity)

        A = -torch.exp(self.A_log.float()).to(dtype=x_conv.dtype)

        # A modulation for inference — apply before exponentiation (consistent with training)
        if complexity is not None:
            mix = torch.sigmoid(self.mix_coeff)
            c_scalar = complexity.mean(dim=-1)  # (B, 1)
            a_mod = 1.0 - (mix.mean() * c_scalar)  # (B, 1)
            A_mod = A * a_mod.unsqueeze(-1)  # a_mod applied to A, not to exp(dA)
        else:
            A_mod = A

        dt_squeezed = dt_full.squeeze(1)
        dA = torch.exp(dt_squeezed.unsqueeze(-1) * A_mod.unsqueeze(0))

        dB = dt_squeezed.unsqueeze(-1) * B.squeeze(1).unsqueeze(1)
        dBx = x_conv.squeeze(1).unsqueeze(-1) * dB

        new_state = dA * state + dBx

        y = torch.sum(
            C.squeeze(1).unsqueeze(1) * new_state, dim=-1
        )
        y = y + x_conv.squeeze(1) * self.D.unsqueeze(0)
        y = y * F.silu(z.squeeze(1))
        y = self.norm(y.unsqueeze(1)).squeeze(1)
        output = self.out_proj(y)

        return output.unsqueeze(1), new_state


# ---------------------------------------------------------------------------
# LiquidSSMTerpaduLayer
# ---------------------------------------------------------------------------

class LiquidSSMTerpaduLayer(nn.Module):
    """
    Enhanced :class:`SSMTerpaduLayer` with **liquid / adaptive compute
    depth**.

    On each forward pass a :class:`ComplexityGate` estimates how much
    SSM processing every token needs and assigns it to one of three
    depth levels:

    * **Depth 1 (fast)** — Only a single SSD pass.  Best for easy /
      highly-predictable tokens.
    * **Depth 2 (standard)** — SSD pass + one additional sub-layer
      (either WKV or Delta, chosen by the interleaving scheduler).
    * **Depth 3 (deep)** — Full interleaving through all sub-layers
      per the scheduler's pattern.

    During **training** all sub-layer outputs are computed (for gradient
    flow) but are soft-weighted according to the depth probabilities.
    During **inference** (``forward_inference``), a hard early-exit is
    used so that easy tokens genuinely skip computation.

    The layer is **backward-compatible** with :class:`SSMTerpaduLayer`:
    passing ``use_liquid=False`` or ``fixed_depth=2`` disables the
    adaptive mechanism and the layer behaves identically to its v0.3
    counterpart.

    Args:
        d_model:              Model dimension.
        d_state:              SSM state dimension.
        d_conv:               Local convolution width.
        expand:               Expansion factor.
        chunk_size:           SSD chunk size.
        n_heads:              Number of heads (WKV / Delta / complexity).
        d_head:               Dimension per head.
        interleaving_ratios:  Tuple ``(ssd, wkv, delta)`` — default ``(4, 1, 1)``.
        dropout:              Dropout rate.
        use_liquid:           Whether to enable liquid adaptive depth (default True).
        fixed_depth:          If > 0, forces all tokens to this depth level
                              and disables the ComplexityGate.
        complexity_bottleneck: Width of the ComplexityGate MLP.
        depth_entropy_weight:  Weight for the auxiliary depth-entropy
                               regularisation loss (encourages the gate to
                               make decisive choices).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 256,
        n_heads: int = 8,
        d_head: int = 64,
        interleaving_ratios: Tuple[int, int, int] = (4, 1, 1),
        dropout: float = 0.0,
        use_liquid: bool = True,
        fixed_depth: int = 0,
        complexity_bottleneck: int = 64,
        depth_entropy_weight: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.chunk_size = chunk_size
        self.n_heads = n_heads
        self.d_head = d_head
        self.interleaving_ratios = interleaving_ratios
        self.dropout_rate = dropout
        self.use_liquid = use_liquid and fixed_depth == 0
        self.fixed_depth = fixed_depth
        self.depth_entropy_weight = depth_entropy_weight

        # ---- Sub-layers (re-use existing modules via lazy import) ----
        # We import here to keep the file self-contained while allowing
        # the package __init__.py to handle circular-import-free ordering.
        from .mamba2 import Mamba2SSD
        from .rwkv7 import RWKV7WKV
        from .delta_net import GatedDeltaNet

        # Use LiquidSSD when liquid mode is on; otherwise plain Mamba2SSD
        if self.use_liquid:
            self.ssd = LiquidSSD(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                chunk_size=chunk_size,
                num_heads=n_heads,
            )
        else:
            self.ssd = Mamba2SSD(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                chunk_size=chunk_size,
            )

        self.wkv = RWKV7WKV(
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
        )

        self.delta = GatedDeltaNet(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            chunk_size=chunk_size,
        )

        # ---- Interleaving scheduler ----
        from .ssm_layer import InterleavingScheduler
        self.scheduler = InterleavingScheduler(interleaving_ratios)

        # ---- Complexity gate (only when liquid mode is active) ----
        if self.use_liquid:
            self.complexity_gate = ComplexityGate(
                d_model=d_model,
                bottleneck_dim=complexity_bottleneck,
                num_heads=n_heads,
            )
        else:
            self.complexity_gate = None  # type: ignore[assignment]

        # ---- LayerNorms per depth level ----
        # Depth 1: 1 norm  (SSD only)
        # Depth 2: 2 norms (SSD + 1 more)
        # Depth 3: total_blocks norms (full schedule)
        total_blocks = self.scheduler.get_total_blocks()
        self.layer_norms = nn.ModuleList(
            [nn.RMSNorm(d_model, eps=1e-5) for _ in range(total_blocks)]
        )

        # ---- Depth-level output projections ----
        # Each depth level produces a d_model output; we blend them
        # with the depth probabilities during training.
        self.depth_projections = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(NUM_DEPTH_LEVELS)
        ])

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---- Routing gate (same as SSMTerpaduLayer) ----
        self.routing_gate = nn.Linear(d_model, 3, bias=False)

        # ---- Output norm ----
        self.output_norm = nn.RMSNorm(d_model, eps=1e-5)

    # ------------------------------------------------------------------
    # State container (re-export for convenience)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_ssm_state_cls():
        """Lazy import of SSMState to avoid circular imports."""
        from .ssm_layer import SSMState
        return SSMState

    # ------------------------------------------------------------------
    # SSD dispatch helper (handles LiquidSSD vs Mamba2SSD)
    # ------------------------------------------------------------------

    def _call_ssd(
        self,
        x: torch.Tensor,
        complexity: Optional[torch.Tensor],
        initial_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call SSD sub-layer, passing *complexity* only for LiquidSSD."""
        if isinstance(self.ssd, LiquidSSD) and complexity is not None:
            return self.ssd(x, complexity=complexity, initial_state=initial_state)
        return self.ssd(x, initial_state=initial_state)

    def _call_ssd_inference(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        complexity: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call SSD inference, passing *complexity* only for LiquidSSD."""
        if isinstance(self.ssd, LiquidSSD) and complexity is not None:
            return self.ssd.forward_inference(x, state, complexity=complexity)
        return self.ssd.forward_inference(x, state)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _compute_depth_outputs(
        self,
        hidden: torch.Tensor,
        ssd_state: Optional[torch.Tensor],
        wkv_state: Optional[tuple],
        delta_state: Optional[torch.Tensor],
        complexity: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[tuple], Optional[torch.Tensor]]:
        """
        Run sub-layers and collect outputs at each depth level.

        Returns:
            depth_outputs: list of 3 tensors, one per depth level,
                           each ``(batch, seq_len, d_model)``.
            ssd_state, wkv_state, delta_state: updated sub-layer states.
        """
        batch, seq_len, _ = hidden.shape
        device, dtype = hidden.device, hidden.dtype

        # --- Depth 1: SSD only ---
        h1 = self.layer_norms[0](hidden)
        ssd_out, ssd_state = self._call_ssd(h1, complexity, ssd_state)
        depth1_out = hidden + self.dropout(ssd_out)

        # --- Depth 2: SSD + one more (pick the first non-SSD block from schedule) ---
        # Find the first non-SSD block type
        total_blocks = self.scheduler.get_total_blocks()
        second_block_type = None
        for bi in range(total_blocks):
            bt = self.scheduler.get_block_type(bi)
            if bt != "ssd":
                second_block_type = bt
                break
        # Fallback: if schedule is all SSD, just use WKV
        if second_block_type is None:
            second_block_type = "wkv"

        # Use the second layer norm
        h2 = self.layer_norms[1](depth1_out) if len(self.layer_norms) > 1 else depth1_out
        if second_block_type == "wkv":
            extra_out, wkv_state = self.wkv(h2, wkv_state)
        else:
            extra_out, delta_state = self.delta(h2, delta_state)

        depth2_out = depth1_out + self.dropout(extra_out)

        # --- Depth 3: Full interleaving ---
        h3 = depth2_out
        block_idx_offset = 2  # we already used norms 0 and 1
        for bi in range(total_blocks):
            # Skip the first two blocks we already processed
            if bi < 2:
                continue
            residual = h3
            norm_idx = min(bi, len(self.layer_norms) - 1)
            h3_normed = self.layer_norms[norm_idx](h3)

            block_type = self.scheduler.get_block_type(bi)
            if block_type == "ssd":
                block_out, ssd_state = self._call_ssd(
                    h3_normed, complexity, ssd_state
                )
            elif block_type == "wkv":
                block_out, wkv_state = self.wkv(h3_normed, wkv_state)
            elif block_type == "delta":
                block_out, delta_state = self.delta(h3_normed, delta_state)
            else:
                block_out, ssd_state = self._call_ssd(
                    h3_normed, complexity, ssd_state
                )

            h3 = residual + self.dropout(block_out)

        depth3_out = h3

        return [depth1_out, depth2_out, depth3_out], ssd_state, wkv_state, delta_state

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input: torch.Tensor,
        ssm_state: Optional["SSMState"] = None,   # noqa: F821
        routing_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, "SSMState", Dict[str, torch.Tensor]]:  # noqa: F821
        """
        Forward pass with adaptive compute depth.

        During training the outputs of all three depth levels are
        blended using the soft depth probabilities (for gradient flow).
        An auxiliary *depth entropy loss* is returned to encourage the
        gate to make decisive choices.

        Args:
            input:            ``(batch, seq_len, d_model)``
            ssm_state:        Previous :class:`SSMState` (optional).
            routing_weights:  Optional routing weights ``(batch, 3)`` or
                              ``(batch, seq_len, 3)``.

        Returns:
            output:          ``(batch, seq_len, d_model)``
            new_ssm_state:   Updated :class:`SSMState`.
            aux_losses:      Dict of auxiliary losses (``depth_entropy``,
                              ``complexity_l1``) that the caller should
                              add to the main loss.
        """
        SSMState = self._make_ssm_state_cls()
        batch, seq_len, _ = input.shape
        aux_losses: Dict[str, torch.Tensor] = {}

        if seq_len == 0:
            dummy_out = torch.zeros(
                batch, 0, self.d_model, dtype=input.dtype, device=input.device
            )
            return dummy_out, SSMState(), aux_losses

        # ---- Unpack state ----
        if ssm_state is None:
            ssd_state = None
            wkv_state = None
            delta_state = None
        else:
            ssd_state = ssm_state.ssd_state
            wkv_state = ssm_state.wkv_state
            delta_state = ssm_state.delta_state

        # ---- Complexity estimation ----
        if self.use_liquid and self.complexity_gate is not None:
            depth_logits, depth_probs, complexity = self.complexity_gate(input)
            # depth_probs: (B, S, 3)
            # complexity:  (B, S, H)

            # Auxiliary: depth entropy loss (encourages decisive choice)
            entropy = -(depth_probs * (depth_probs + 1e-8).log()).sum(dim=-1)  # (B, S)
            max_entropy = math.log(NUM_DEPTH_LEVELS)
            normalized_entropy = entropy / max_entropy
            aux_losses["depth_entropy"] = self.depth_entropy_weight * normalized_entropy.mean()

            # Auxiliary: complexity L1 (mild sparsity on complexity scores)
            aux_losses["complexity_l1"] = 0.001 * complexity.mean()
        else:
            depth_probs = None
            complexity = None

        # ---- Compute all depth-level outputs ----
        depth_outputs, ssd_state, wkv_state, delta_state = self._compute_depth_outputs(
            hidden=input,
            ssd_state=ssd_state,
            wkv_state=wkv_state,
            delta_state=delta_state,
            complexity=complexity,
        )

        # ---- Blend depth outputs ----
        if depth_probs is not None:
            # Soft blending: project each depth output, then weight-sum
            projected = []
            for lvl in range(NUM_DEPTH_LEVELS):
                projected.append(self.depth_projections[lvl](depth_outputs[lvl]))
            stacked = torch.stack(projected, dim=2)  # (B, S, 3, d_model)

            w = depth_probs.unsqueeze(-1)  # (B, S, 3, 1)
            blended = (stacked * w).sum(dim=2)  # (B, S, d_model)
            output = self.output_norm(blended)
        else:
            # No liquid mode — use the fixed depth or full depth
            if self.fixed_depth > 0 and self.fixed_depth <= NUM_DEPTH_LEVELS:
                chosen = depth_outputs[self.fixed_depth - 1]
            elif routing_weights is not None:
                # Dynamic routing: blend all three (same logic as SSMTerpaduLayer)
                projected = []
                for lvl in range(NUM_DEPTH_LEVELS):
                    projected.append(self.depth_projections[lvl](depth_outputs[lvl]))
                stacked = torch.stack(projected, dim=2)
                if routing_weights.dim() == 2:
                    routing_weights = routing_weights.unsqueeze(1)
                rw = F.softmax(routing_weights, dim=-1).unsqueeze(-1)
                if rw.shape[1] == 1 and stacked.shape[1] > 1:
                    rw = rw.expand(-1, stacked.shape[1], -1, -1)
                blended = (stacked * rw).sum(dim=2)
                chosen = blended
            else:
                # Default: full depth (depth 3 = full interleaving)
                chosen = depth_outputs[2]

            output = self.output_norm(chosen)

        # ---- Pack new state ----
        new_state = SSMState(
            ssd_state=ssd_state,
            wkv_state=wkv_state,
            delta_state=delta_state,
        )

        return output, new_state, aux_losses

    # ------------------------------------------------------------------
    # Inference (hard early exit)
    # ------------------------------------------------------------------

    def forward_inference(
        self,
        input: torch.Tensor,
        ssm_state: Optional["SSMState"] = None,  # noqa: F821
        routing_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, "SSMState"]:  # noqa: F821
        """
        Single-token inference with **hard early exit**.

        Tokens assessed as "easy" (depth 1) genuinely skip the
        additional sub-layer computation.

        Args:
            input:            ``(batch, 1, d_model)``
            ssm_state:        Previous :class:`SSMState`.
            routing_weights:  Optional ``(batch, 3)``.

        Returns:
            (output, new_ssm_state)
        """
        SSMState = self._make_ssm_state_cls()
        batch = input.shape[0]

        if ssm_state is None:
            ssd_state = None
            wkv_state = None
            delta_state = None
        else:
            ssd_state = ssm_state.ssd_state
            wkv_state = ssm_state.wkv_state
            delta_state = ssm_state.delta_state

        # ---- Complexity estimation ----
        if self.use_liquid and self.complexity_gate is not None:
            _, depth_probs, complexity = self.complexity_gate(input)
            # Hard depth selection
            depth_idx = depth_probs.argmax(dim=-1)  # (B,) or (B, 1)
            if depth_idx.dim() > 1:
                depth_idx = depth_idx.squeeze(-1)
            # depth_idx values: 0 → fast, 1 → standard, 2 → deep
            selected_depth = depth_idx + 1  # 1, 2, or 3
        else:
            complexity = None
            selected_depth = None

        # ---- Default: full interleaving (same as SSMTerpaduLayer) ----
        if not self.use_liquid or selected_depth is None:
            if self.fixed_depth > 0:
                selected_depth = torch.full(
                    (batch,), self.fixed_depth, dtype=torch.long, device=input.device
                )
            else:
                selected_depth = torch.full(
                    (batch,), DEPTH_DEEP, dtype=torch.long, device=input.device
                )

        # ---- Process per depth level ----
        # For simplicity in inference, we process in stages and early-exit
        hidden = input

        # --- Stage 1: SSD (all tokens) ---
        residual = hidden
        h_normed = self.layer_norms[0](hidden)
        ssd_out, ssd_state = self._call_ssd_inference(
            h_normed, ssd_state, complexity=complexity
        )
        hidden = residual + ssd_out

        # Check which tokens can early-exit at depth 1
        # (For batch efficiency, we still process all but weight the outputs)
        fast_mask = (selected_depth == DEPTH_FAST)  # (B,)

        # --- Stage 2: additional sub-layer for depth >= 2 ---
        if not fast_mask.all():
            # Find the second block type from schedule
            second_block_type = None
            total_blocks = self.scheduler.get_total_blocks()
            for bi in range(total_blocks):
                bt = self.scheduler.get_block_type(bi)
                if bt != "ssd":
                    second_block_type = bt
                    break
            if second_block_type is None:
                second_block_type = "wkv"

            residual2 = hidden
            h2_normed = self.layer_norms[1](hidden) if len(self.layer_norms) > 1 else hidden

            if second_block_type == "wkv":
                extra_out, wkv_state = self.wkv.forward_inference(h2_normed, wkv_state)
            else:
                extra_out, delta_state = self.delta.forward_inference(h2_normed, delta_state)

            depth2_hidden = residual2 + extra_out

            # Merge: fast tokens keep depth1 output, others use depth2
            mask2 = (selected_depth >= DEPTH_STANDARD).unsqueeze(-1).unsqueeze(-1).float()
            # (B, 1, 1) for broadcasting with (B, 1, d_model)
            hidden = hidden * (1 - mask2) + depth2_hidden * mask2
        else:
            depth2_hidden = hidden

        # --- Stage 3: remaining blocks for depth == 3 ---
        deep_mask = (selected_depth == DEPTH_DEEP)
        if deep_mask.any():
            total_blocks = self.scheduler.get_total_blocks()
            h3 = hidden
            for bi in range(2, total_blocks):
                residual3 = h3
                norm_idx = min(bi, len(self.layer_norms) - 1)
                h3_normed = self.layer_norms[norm_idx](h3)

                block_type = self.scheduler.get_block_type(bi)
                if block_type == "ssd":
                    block_out, ssd_state = self._call_ssd_inference(
                        h3_normed, ssd_state, complexity=complexity
                    )
                elif block_type == "wkv":
                    block_out, wkv_state = self.wkv.forward_inference(h3_normed, wkv_state)
                elif block_type == "delta":
                    block_out, delta_state = self.delta.forward_inference(h3_normed, delta_state)
                else:
                    block_out, ssd_state = self._call_ssd_inference(
                        h3_normed, ssd_state, complexity=complexity
                    )

                h3 = residual3 + block_out

            depth3_hidden = h3
            # Merge deep tokens
            mask3 = deep_mask.unsqueeze(-1).unsqueeze(-1).float()
            hidden = hidden * (1 - mask3) + depth3_hidden * mask3

        # ---- Output projection & norm ----
        # Apply the appropriate depth projection
        # For inference we pick based on selected depth per sample
        if self.use_liquid:
            # Gather the right projection for each sample
            # projections: list of 3 Linear(d_model, d_model)
            proj_outputs = []
            for lvl in range(NUM_DEPTH_LEVELS):
                proj_outputs.append(self.depth_projections[lvl](hidden))

            stacked_proj = torch.stack(proj_outputs, dim=1)  # (B, 3, 1, d_model)
            # Index by selected_depth
            idx = (torch.arange(batch, device=input.device), selected_depth - 1)
            gathered = stacked_proj[idx[0], idx[1]]  # (B, 1, d_model)
            output = self.output_norm(gathered)
        else:
            # Use fixed depth projection or full schedule output
            if self.fixed_depth > 0:
                output = self.depth_projections[self.fixed_depth - 1](hidden)
            else:
                output = hidden
            output = self.output_norm(output)

        new_state = SSMState(
            ssd_state=ssd_state,
            wkv_state=wkv_state,
            delta_state=delta_state,
        )

        return output, new_state

    # ------------------------------------------------------------------
    # Routing logits (backward-compatible API)
    # ------------------------------------------------------------------

    def get_routing_logits(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute routing logits from input (same API as SSMTerpaduLayer).

        Args:
            input: ``(batch, seq_len, d_model)``

        Returns:
            ``(batch, seq_len, 3)`` — [ssd, wkv, delta] logits.
        """
        return self.routing_gate(input)

    # ------------------------------------------------------------------
    # State initialisation (backward-compatible API)
    # ------------------------------------------------------------------

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> "SSMState":  # noqa: F821
        """
        Initialise an empty :class:`SSMState`.

        Args:
            batch_size: Batch size.
            device:     Tensor device.
            dtype:      Tensor dtype.

        Returns:
            Freshly initialised :class:`SSMState`.
        """
        SSMState = self._make_ssm_state_cls()
        d_inner = int(self.expand * self.d_model)

        ssd_state = torch.zeros(
            batch_size, d_inner, self.d_state,
            dtype=dtype, device=device,
        )

        # RWKV7WKV expects flat state of shape (batch, n_heads * d_head)
        wkv_flat_dim = self.n_heads * self.d_head
        wkv_state = (
            torch.zeros(batch_size, wkv_flat_dim, dtype=dtype, device=device),
            torch.zeros(batch_size, wkv_flat_dim, dtype=dtype, device=device),
        )

        delta_state = torch.zeros(
            batch_size, self.n_heads, self.d_head, self.d_head,
            dtype=dtype, device=device,
        )

        return SSMState(
            ssd_state=ssd_state,
            wkv_state=wkv_state,
            delta_state=delta_state,
        )
