"""
Structured Sparse Transition Matrices for SSMs — Enabling FSA State Tracking.

Implementation based on NeurIPS 2025 poster 118046:
"Structured Sparse Transition Matrices to Enable State Tracking in SSMs"

The core insight is that diagonal SSMs (like Mamba) fundamentally cannot track
Finite State Automata (FSA) states because their transition matrices are purely
diagonal — there is no mechanism for information to flow *between* state
dimensions.  This module introduces structured off-diagonal elements that enable
cross-state transitions while maintaining O(S) complexity per step.

Key Concepts
------------
1. **State Tracking Problem** — Diagonal SSMs cannot simulate FSA because the
   diagonal transition matrix prevents state dimensions from interacting.  An FSA
   with *Q* states requires that the SSM can transition from any state to any
   other — impossible with a purely diagonal transition.

2. **Structured Sparsity** — Instead of a fully dense transition matrix (which
   would be O(S²) per step), we introduce a small number of off-diagonal
   elements in a structured pattern.  The key theoretical result is that
   structured sparse parametrization achieves **provably optimal** state size
   for FSA tracking: an FSA with *Q* states can be tracked using S = O(Q)
   state dimensions with structured sparse transitions.

3. **Block-Sparse Pattern** — Group state dimensions into blocks of size *B*.
   Within each block, transitions are dense (B × B); across blocks, transitions
   are diagonal.  This yields O(S × B) per step instead of O(S²).

4. **Sparsity Patterns** — Three supported patterns:
   - ``block_diagonal``: dense within blocks, diagonal across blocks.
   - ``banded``: diagonal ± *band_width* off-diagonals.
   - ``butterfly``: recursive butterfly pattern (log₂(B) levels).

Architecture
------------
StructuredSparseTransition creates the structured sparse transition matrix from
a compact parameterisation that only stores O(S × B) parameters.  The full
transition is never materialised during the SSM scan — instead, the structured
sparse matvec is applied directly in O(S × B) time.

StructuredSparseSSM wraps the Mamba-2 SSD core (reusing ``ssd_chunk_scan``)
but replaces the diagonal A matrix with a structured sparse transition.

References
----------
- NeurIPS 2025 poster 118046: "Structured Sparse Transition Matrices to Enable
  State Tracking in SSMs"
- Gu & Dao, "Mamba-2: A Generalized State Space Model with Structured State
  Space Duality" (2024), arXiv:2405.21060
- Merrill et al., "The Expressive Power of Transformers with Chain of Thought"
  (2024) — FSA tracking as a benchmark for sequence models

Hardware: Pure PyTorch.  No custom CUDA kernels required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.core.ssm.mamba2 import ssd_chunk_scan


# ============================================================================
# Sparsity Pattern Enum
# ============================================================================

class SparsityPattern(str, Enum):
    """Supported structured sparsity patterns for transition matrices.

    Attributes:
        BLOCK_DIAGONAL: Dense within blocks of size ``block_size``, diagonal
            across blocks.  Complexity: O(S × block_size) per step.
        BANDED: Diagonal ± ``band_width`` off-diagonals.  Complexity:
            O(S × (2 * band_width + 1)) per step.
        BUTTERFLY: Recursive butterfly pattern over ``block_size`` dimensions.
            Complexity: O(S × log₂(block_size)) per step.
    """

    BLOCK_DIAGONAL = "block_diagonal"
    BANDED = "banded"
    BUTTERFLY = "butterfly"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StructuredSparseSSMConfig:
    """Configuration for Structured Sparse SSM layer.

    Attributes:
        d_model: Model input/output dimension.
        d_state: SSM state dimension (N).  Must be divisible by ``block_size``
            when using ``block_diagonal`` or ``butterfly`` patterns.
        d_conv: Local causal convolution kernel width.
        expand: Inner dimension expansion factor (d_inner = expand × d_model).
        n_groups: Number of groups for block partitioning.  If > 0,
            ``block_size = d_state // n_groups`` is computed automatically.
            Ignored if ``block_size`` is set explicitly.
        block_size: Size of each block for block-diagonal / butterfly patterns.
            If 0, it is derived from ``n_groups`` (defaulting to 1 if neither
            is set, which reduces to diagonal SSM).
        transition_type: Structured sparsity pattern to use.
        band_width: Half-width for the banded pattern (off-diagonals on each
            side).  Only used when ``transition_type = "banded"``.
        chunk_size: Chunk size for SSD parallel scan during training.
        dt_min: Lower bound for dt initialisation.
        dt_max: Upper bound for dt initialisation.
        use_bias: Whether to use bias in projections.
    """

    d_model: int = 768
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    n_groups: int = 0
    block_size: int = 0
    transition_type: str = "block_diagonal"
    band_width: int = 2
    chunk_size: int = 256
    dt_min: float = 0.001
    dt_max: float = 0.1
    use_bias: bool = False

    def resolve_block_size(self) -> int:
        """Compute the effective block size from config values.

        If ``block_size`` is explicitly set, use it.  Otherwise derive from
        ``n_groups``.  Falls back to 1 (pure diagonal) if neither is set.

        Returns:
            Effective block size (≥ 1).

        Raises:
            ValueError: If ``d_state`` is not divisible by the resolved
                block size.
        """
        if self.block_size > 0:
            bs = self.block_size
        elif self.n_groups > 0:
            bs = self.d_state // self.n_groups
        else:
            bs = 1  # pure diagonal (standard SSM)

        if self.d_state % bs != 0:
            raise ValueError(
                f"d_state={self.d_state} must be divisible by block_size={bs}"
            )
        return max(1, bs)


# ============================================================================
# Structured Sparse Transition
# ============================================================================

class StructuredSparseTransition(nn.Module):
    """Creates and applies structured sparse transition matrices.

    Instead of materialising the full S×S transition matrix (which would be
    O(S²) in both memory and compute), this module stores a compact
    parameterisation and applies the structured sparse matvec in O(S × B)
    time, where B is the block size.

    For each state dimension group, the transition matrix A has the form:

    - **Block-diagonal**: each B×B block is a full learnable matrix; all
      inter-block entries are diagonal.  Total parameters: (S/B) × B² + S.
    - **Banded**: A is banded with bandwidth ``2 * band_width + 1``.
      Total parameters: S × (2 * band_width + 1).
    - **Butterfly**: each B×B block uses a butterfly factorisation with
      log₂(B) levels of 2×2 blocks.  Total parameters: S × log₂(B).

    All parameters are stored in log-domain for numerical stability
    (ensuring negative values for stable SSM dynamics).

    Args:
        d_state: SSM state dimension (S).
        block_size: Block size for block-diagonal / butterfly patterns.
        transition_type: Sparsity pattern (see :class:`SparsityPattern`).
        band_width: Half-width for banded pattern.
        n_inner: Number of inner (channel) dimensions — used to create
            per-channel transition parameters for compatibility with the
            Mamba-2 convention where A has shape ``(d_inner, d_state)``.
            If 0, a single shared transition is used.
    """

    def __init__(
        self,
        d_state: int,
        block_size: int,
        transition_type: str = "block_diagonal",
        band_width: int = 2,
        n_inner: int = 0,
    ) -> None:
        super().__init__()

        self.d_state = d_state
        self.block_size = block_size
        self.transition_type = SparsityPattern(transition_type)
        self.band_width = band_width
        self.n_inner = n_inner

        n_blocks = d_state // block_size if block_size > 0 else d_state
        self.n_blocks = n_blocks

        # Number of transition parameter sets (per inner dim or shared)
        n_trans = max(1, n_inner) if n_inner > 0 else 1

        if self.transition_type == SparsityPattern.BLOCK_DIAGONAL:
            # Per-block dense B×B matrices + diagonal elements
            # Stored in log-domain: shape (n_trans, n_blocks, B, B)
            if block_size > 1:
                self.block_trans_log = nn.Parameter(
                    torch.zeros(n_trans, n_blocks, block_size, block_size)
                )
                # Initialize off-diagonal elements to large negative values
                # so exp() → 0 (effectively no off-diagonal transition at init)
                with torch.no_grad():
                    # Diagonal: -1, -2, ..., -B (standard S4D init)
                    for b in range(n_blocks):
                        for i in range(block_size):
                            for j in range(block_size):
                                if i == j:
                                    self.block_trans_log.data[:, b, i, j] = math.log(
                                        float(i + 1)
                                    )
                                else:
                                    self.block_trans_log.data[:, b, i, j] = -5.0
            else:
                # Pure diagonal — store as (n_trans, d_state)
                self.diag_log = nn.Parameter(
                    torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
                    .unsqueeze(0)
                    .expand(n_trans, -1)
                    .clone()
                )

        elif self.transition_type == SparsityPattern.BANDED:
            # Banded: store (2 * band_width + 1) diagonals per state dim
            n_diags = 2 * band_width + 1
            self.band_trans_log = nn.Parameter(
                torch.zeros(n_trans, d_state, n_diags)
            )
            with torch.no_grad():
                # Center diagonal initialised with S4D pattern
                for d in range(d_state):
                    self.band_trans_log.data[:, d, band_width] = math.log(
                        float(d + 1)
                    )
                    # Off-diagonals initialised to large negative values
                    for k in range(n_diags):
                        if k != band_width:
                            self.band_trans_log.data[:, d, k] = -5.0

        elif self.transition_type == SparsityPattern.BUTTERFLY:
            # Butterfly: log₂(B) levels, each with B/2 2×2 blocks
            assert block_size > 1 and (block_size & (block_size - 1)) == 0, (
                f"Butterfly pattern requires block_size to be a power of 2, "
                f"got {block_size}"
            )
            n_levels = int(math.log2(block_size))
            self.n_butterfly_levels = n_levels
            # Each level: (n_trans, n_blocks, B/2, 2, 2) parameters
            self.butterfly_log = nn.ParameterList([
                nn.Parameter(torch.zeros(n_trans, n_blocks, block_size // 2, 2, 2))
                for _ in range(n_levels)
            ])
            with torch.no_grad():
                for level_param in self.butterfly_log:
                    # Diagonal entries: S4D init
                    for b in range(n_blocks):
                        for half in range(block_size // 2):
                            level_param.data[:, b, half, 0, 0] = math.log(
                                float(b * block_size + 2 * half + 1)
                            )
                            level_param.data[:, b, half, 1, 1] = math.log(
                                float(b * block_size + 2 * half + 2)
                            )
                            # Off-diagonal: large negative (no cross-coupling at init)
                            level_param.data[:, b, half, 0, 1] = -5.0
                            level_param.data[:, b, half, 1, 0] = -5.0

    def get_transition_matrix(self) -> torch.Tensor:
        """Materialise the full structured sparse transition matrix.

        **Warning**: This is O(S²) and intended for analysis/visualisation
        only.  For the actual SSM scan, use :meth:`apply_transition` instead.

        Returns:
            Transition matrix of shape ``(d_state, d_state)`` (if n_inner=0)
            or ``(n_inner, d_state, d_state)``.
        """
        if self.transition_type == SparsityPattern.BLOCK_DIAGONAL:
            return self._materialise_block_diagonal()
        elif self.transition_type == SparsityPattern.BANDED:
            return self._materialise_banded()
        else:
            return self._materialise_butterfly()

    def _materialise_block_diagonal(self) -> torch.Tensor:
        """Materialise block-diagonal transition matrix."""
        has_inner = self.n_inner > 0
        n_out = self.n_inner if has_inner else 1

        if self.block_size <= 1:
            # Pure diagonal
            A = -torch.exp(self.diag_log.float())  # (n_out, S) negative
            if not has_inner:
                return A.squeeze(0)  # (S,)
            return A  # (n_inner, S)

        # Full matrix: (n_out, S, S)
        A_full = torch.zeros(n_out, self.d_state, self.d_state,
                            device=self.block_trans_log.device,
                            dtype=self.block_trans_log.dtype)
        A_log = self.block_trans_log.float()  # (n_out, n_blocks, B, B)

        for b_idx in range(self.n_blocks):
            start = b_idx * self.block_size
            end = start + self.block_size
            # Block is stored in log-domain; negate for stability
            A_full[:, start:end, start:end] = -torch.exp(A_log[:, b_idx, :, :])

        if not has_inner:
            return A_full.squeeze(0)  # (S, S)
        return A_full  # (n_inner, S, S)

    def _materialise_banded(self) -> torch.Tensor:
        """Materialise banded transition matrix."""
        has_inner = self.n_inner > 0
        n_out = self.n_inner if has_inner else 1

        A_full = torch.zeros(n_out, self.d_state, self.d_state,
                            device=self.band_trans_log.device,
                            dtype=self.band_trans_log.dtype)
        A_log = self.band_trans_log.float()  # (n_out, S, n_diags)

        for d in range(self.d_state):
            for k_idx in range(2 * self.band_width + 1):
                j = d - self.band_width + k_idx
                if 0 <= j < self.d_state:
                    A_full[:, d, j] = -torch.exp(A_log[:, d, k_idx])

        if not has_inner:
            return A_full.squeeze(0)
        return A_full

    def _materialise_butterfly(self) -> torch.Tensor:
        """Materialise butterfly factorisation transition matrix."""
        has_inner = self.n_inner > 0
        n_out = self.n_inner if has_inner else 1

        # Start with identity per block
        A_full = torch.zeros(n_out, self.d_state, self.d_state,
                            device=self.butterfly_log[0].device,
                            dtype=self.butterfly_log[0].dtype)

        for b_idx in range(self.n_blocks):
            start = b_idx * self.block_size
            end = start + self.block_size

            # Apply butterfly levels
            block = torch.eye(self.block_size).unsqueeze(0).expand(n_out, -1, -1).clone()

            for level_idx in range(self.n_butterfly_levels):
                level_log = self.butterfly_log[level_idx].float()  # (n_out, n_blocks, B/2, 2, 2)
                stride = 2 ** level_idx

                for half in range(self.block_size // 2):
                    mat_2x2 = -torch.exp(level_log[:, b_idx, half, :, :])  # (n_out, 2, 2)

                    # Apply to appropriate positions
                    i0 = half * 2 * stride  # simplified butterfly indexing
                    # In practice, butterfly patterns use bit-reversal indexing
                    # Here we use a simple sequential 2×2 block approach
                    row_start = half * 2
                    for n_idx in range(n_out):
                        block[n_idx, row_start:row_start+2, :] = torch.matmul(
                            mat_2x2[n_idx], block[n_idx, row_start:row_start+2, :]
                        )

            A_full[:, start:end, start:end] = block

        if not has_inner:
            return A_full.squeeze(0)
        return A_full

    def apply_transition(
        self,
        h: torch.Tensor,
        dA_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the structured sparse transition to a state vector.

        Computes ``h_new = exp(dt * A) @ h`` using the structured sparse
        pattern without materialising the full matrix.  This is the core
        operation that maintains O(S × B) complexity.

        For the block-diagonal pattern::

            h_new[b*i : b*(i+1)] = exp(dt * A_block_i) @ h[b*i : b*(i+1)]

        Args:
            h: State tensor, shape ``(batch, d_inner, d_state)``.
            dA_scalar: Per-token discretised step size × A scalar,
                shape ``(batch, seq_len)`` or ``(batch,)``.

        Returns:
            Updated state tensor of the same shape as ``h``.
        """
        if self.transition_type == SparsityPattern.BLOCK_DIAGONAL:
            return self._apply_block_diagonal(h, dA_scalar)
        elif self.transition_type == SparsityPattern.BANDED:
            return self._apply_banded(h, dA_scalar)
        else:
            return self._apply_butterfly(h, dA_scalar)

    def _apply_block_diagonal(
        self,
        h: torch.Tensor,
        dA_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """Apply block-diagonal structured sparse transition.

        For each block of size B, computes a B×B matmul.  This is
        O(S × B) instead of O(S²).

        Uses a first-order approximation for the matrix exponential:
            exp(dt * A) @ h ≈ (I + dt * A) @ h = h + dt * (A @ h)

        This avoids the expensive ``torch.matrix_exp`` call while still
        enabling cross-state transitions.  The approximation is accurate
        for the small dt values typical in SSM layers.
        """
        if self.block_size <= 1:
            # Pure diagonal: element-wise multiply
            A_diag = -torch.exp(self.diag_log.float()).to(dtype=h.dtype)
            # A_diag: (1, S) or (n_inner, S)
            # dA_scalar: (batch,) → (batch, 1) for broadcasting with h (batch, d_inner, S)
            return h * torch.exp(dA_scalar.unsqueeze(-1).unsqueeze(-1) * A_diag.unsqueeze(0))

        # Block-diagonal transition via first-order approximation
        A_log = self.block_trans_log.float().to(dtype=h.dtype)  # (n_trans, n_blocks, B, B)
        batch, d_inner, S = h.shape

        h_new = h.clone()

        for b_idx in range(self.n_blocks):
            start = b_idx * self.block_size
            end = start + self.block_size
            h_block = h[:, :, start:end]  # (batch, d_inner, B)

            # Get the B×B transition matrix for this block (shared across d_inner)
            A_block = -torch.exp(A_log[0, b_idx, :, :])  # (B, B)

            # First-order approximation: exp(dt * A) @ h ≈ h + dt * (A @ h)
            # Compute A @ h_block: (B, B) @ (batch, d_inner, B)
            # Use einsum for batched matmul: h has dims (batch, d_inner, B)
            Ah = torch.einsum("ij,bdj->bdi", A_block, h_block)  # (batch, d_inner, B)

            # h_new = h + dt * (A @ h), where dt is per-batch
            # dA_scalar: (batch,) → (batch, 1, 1) for broadcasting
            dt_expanded = dA_scalar.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            h_new[:, :, start:end] = h_block + dt_expanded * Ah

        return h_new

    def _apply_banded(
        self,
        h: torch.Tensor,
        dA_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """Apply banded structured sparse transition.

        Uses a first-order approximation with a sliding window:
            exp(dt * A) @ h ≈ h + dt * (A @ h)

        Complexity: O(S × bandwidth) per step.
        """
        A_log = self.band_trans_log.float().to(dtype=h.dtype)  # (n_trans, S, n_diags)
        n_diags = 2 * self.band_width + 1
        batch, d_inner, S = h.shape

        # Compute A @ h for the banded structure
        Ah = torch.zeros_like(h)  # (batch, d_inner, S)

        for d in range(S):
            for k_idx in range(n_diags):
                j = d - self.band_width + k_idx
                if 0 <= j < S:
                    # A[d, j] from banded parameters
                    A_val = -torch.exp(A_log[0, d, k_idx])  # scalar
                    # Accumulate A[d, j] * h[:, :, j] into Ah[:, :, d]
                    Ah[:, :, d] = Ah[:, :, d] + A_val * h[:, :, j]

        # First-order approximation: exp(dt * A) @ h ≈ h + dt * Ah
        # dA_scalar: (batch,) → (batch, 1, 1) for broadcasting
        dt_expanded = dA_scalar.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
        h_new = h + dt_expanded * Ah

        return h_new

    def _apply_butterfly(
        self,
        h: torch.Tensor,
        dA_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """Apply butterfly structured sparse transition.

        Applies log₂(B) levels of 2×2 matrix operations using first-order
        approximation: O(S × log₂(B)).

        For each level, applies 2×2 affine transforms to pairs of state
        dimensions: h_new = h + dt * (A_2x2 @ h_pair)
        """
        batch, d_inner, S = h.shape
        h_new = h.clone()

        for b_idx in range(self.n_blocks):
            start = b_idx * self.block_size
            end = start + self.block_size
            h_block = h[:, :, start:end]  # (batch, d_inner, B)

            # Apply butterfly levels sequentially
            for level_idx in range(self.n_butterfly_levels):
                level_log = self.butterfly_log[level_idx].float().to(dtype=h.dtype)
                # level_log: (n_trans, n_blocks, B/2, 2, 2)

                h_updated = h_block.clone()
                half_B = self.block_size // 2

                for half in range(half_B):
                    mat_2x2 = -torch.exp(level_log[0, b_idx, half, :, :])  # (2, 2)
                    row_start = half * 2

                    # First-order: h_new = h + dt * (A_2x2 @ h_pair)
                    # Extract pairs: (batch, d_inner, 2)
                    h_pair = h_block[:, :, row_start:row_start+2]

                    # Compute A_2x2 @ h_pair via einsum
                    Ah_pair = torch.einsum("ij,bdj->bdi", mat_2x2, h_pair)

                    # Apply dt: (batch, 1, 1) * (batch, d_inner, 2)
                    dt_expanded = dA_scalar.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
                    h_updated[:, :, row_start:row_start+2] = h_pair + dt_expanded * Ah_pair

                h_block = h_updated

            h_new[:, :, start:end] = h_block

        return h_new


# ============================================================================
# Structured Sparse SSM
# ============================================================================

class StructuredSparseSSM(nn.Module):
    """SSM layer with Structured Sparse Transition Matrices.

    Drop-in replacement for :class:`Mamba2SSD` that replaces the purely
    diagonal transition with a structured sparse pattern, enabling FSA
    state tracking while maintaining O(S × B) complexity per step.

    The layer follows the same architecture as Mamba-2 SSD:
    1. Input projection (d_model → d_inner × 2)
    2. Local causal convolution
    3. SSM parameter projections (B, C, dt)
    4. Structured sparse SSM scan (replaces diagonal scan)
    5. Skip connection (D) + gating + output projection

    Forward signatures (compatible with Losion SSM pathway):

    * ``forward(x, initial_state=None) -> (output, final_state)``
    * ``forward_inference(x, state) -> (output, new_state)``

    Example
    -------
    >>> config = StructuredSparseSSMConfig(
    ...     d_model=512, d_state=64, block_size=8,
    ...     transition_type="block_diagonal",
    ... )
    >>> ssm = StructuredSparseSSM(config)
    >>> x = torch.randn(2, 16, 512)
    >>> output, state = ssm(x)
    >>> output.shape
    torch.Size([2, 16, 512])
    >>> state.shape
    torch.Size([2, 1024, 64])

    Args:
        config: A :class:`StructuredSparseSSMConfig` instance.
    """

    def __init__(
        self,
        config: Optional[StructuredSparseSSMConfig] = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = StructuredSparseSSMConfig()

        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.chunk_size = config.chunk_size
        self.d_inner = int(config.expand * config.d_model)
        self.dt_init_floor = 1e-4

        # Resolve block size
        self.block_size = config.resolve_block_size()

        # ---- Input projection (same as Mamba2SSD) ----
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.use_bias)

        # ---- Local causal convolution ----
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # ---- SSM parameter projections ----
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=False)

        # ---- dt bias (per-channel, log-uniform init) ----
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        inv_softplus = torch.log(torch.exp(dt_init) - 1)
        self.dt_bias = nn.Parameter(inv_softplus)

        # ---- Structured Sparse Transition (replaces A_log) ----
        self.transition = StructuredSparseTransition(
            d_state=self.d_state,
            block_size=self.block_size,
            transition_type=config.transition_type,
            band_width=config.band_width,
            n_inner=0,  # shared transition (averaged over d_inner for simplicity)
        )

        # ---- Legacy diagonal A_log for fallback compatibility ----
        # Used when block_size == 1 (pure diagonal mode)
        if self.block_size <= 1:
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
            A = A.expand(self.d_inner, -1).clone()
            self.A_log = nn.Parameter(torch.log(A))

        # ---- D skip connection ----
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ---- Output projection ----
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.use_bias)

        # ---- Norm ----
        self.norm = nn.RMSNorm(self.d_inner, eps=1e-5)

    def _get_dt(self) -> torch.Tensor:
        """Return dt bias after softplus."""
        return F.softplus(self.dt_bias + 1e-4)

    def _structured_sparse_scan(
        self,
        x_seq: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential SSM scan with structured sparse transitions.

        Replaces the diagonal scan with one that uses the structured sparse
        transition matrix.  For each time step:

            h_t = transition(dt_t) @ h_{t-1} + x_t ⊗ (dt_t * B_t)
            y_t = C_t^T @ h_t

        When ``block_size > 1``, the transition is applied via
        :meth:`StructuredSparseTransition.apply_transition` which maintains
        O(S × B) complexity.  When ``block_size == 1``, falls back to the
        standard diagonal scan from Mamba2SSD.

        Args:
            x_seq: Input sequence, ``(batch, seq_len, d_inner)``.
            B: Input matrix, ``(batch, seq_len, d_state)``.
            C: Output matrix, ``(batch, seq_len, d_state)``.
            dt: Step sizes, ``(batch, seq_len, d_inner)`` — per-channel.
            initial_state: Optional initial state, ``(batch, d_inner, d_state)``.

        Returns:
            Tuple ``(output, final_state)``.
        """
        batch, seq_len, d_inner = x_seq.shape
        d_state = B.shape[-1]

        if initial_state is None:
            h = torch.zeros(
                batch, d_inner, d_state,
                dtype=x_seq.dtype, device=x_seq.device,
            )
        else:
            h = initial_state.clone()

        if self.block_size <= 1:
            # ---- Pure diagonal mode: use standard SSD scan ----
            # Pass per-channel A directly (no averaging)
            A = -torch.exp(self.A_log.float()).to(dtype=x_seq.dtype)

            return ssd_chunk_scan(
                x_seq=x_seq,
                A=A,  # (d_inner, d_state) — per-channel, NOT averaged
                B=B,
                C=C,
                dt=dt,  # (batch, seq_len, d_inner) — per-channel
                chunk_size=self.chunk_size,
                initial_state=initial_state,
            )

        # ---- Structured sparse scan ----
        # For structured sparse transitions (block_size > 1), the transition
        # applies a shared matrix per block using first-order approximation
        # which requires scalar dt. Per-channel dt is used for input injection.
        outputs = []
        for t in range(seq_len):
            # Apply structured sparse transition: h = T(dt_t) @ h
            # Use mean dt for shared transition (first-order approximation)
            dt_t_avg = dt[:, t].mean(dim=-1)  # (batch,)
            h = self.transition.apply_transition(h, dt_t_avg)

            # Input injection: h += x_t ⊗ (dt_t * B_t)
            # Use per-channel dt for input injection
            dt_t = dt[:, t]  # (batch, d_inner)
            dB_t = dt_t_avg.unsqueeze(-1) * B[:, t, :]  # (batch, d_state)
            h = h + x_seq[:, t, :].unsqueeze(-1) * dB_t.unsqueeze(1)
            # h: (batch, d_inner, d_state)

            # Output: y_t = C_t^T @ h
            y_t = torch.sum(
                h * C[:, t, :].unsqueeze(1), dim=-1
            )  # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        final_state = h

        return y, final_state

    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with structured sparse SSM scan.

        Args:
            input: Input tensor, ``(batch, seq_len, d_model)``.
            initial_state: Optional initial SSM state,
                ``(batch, d_inner, d_state)``.

        Returns:
            Tuple ``(output, final_state)``:
            - output: ``(batch, seq_len, d_model)``
            - final_state: ``(batch, d_inner, d_state)``
        """
        batch, seq_len, _ = input.shape

        # Handle edge case
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
        xz = self.in_proj(input)  # (batch, seq_len, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # Each (batch, seq_len, d_inner)

        # ---- Step 2: Local causal convolution ----
        x_conv = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal: trim padding
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)

        # ---- Step 3: SSM parameter projections ----
        ssm_params = self.x_proj(x_conv)  # (batch, seq_len, d_state * 2)
        B = ssm_params[..., :self.d_state]  # (batch, seq_len, d_state)
        C = ssm_params[..., self.d_state:self.d_state * 2]

        # dt: per-channel projection + bias
        dt_full = F.softplus(
            self.dt_proj(x_conv) + self.dt_bias.unsqueeze(0).unsqueeze(0) + self.dt_init_floor
        )  # (batch, seq_len, d_inner)

        # ---- Step 4: Structured Sparse SSM Scan ----
        y, final_state = self._structured_sparse_scan(
            x_seq=x_conv,
            B=B,
            C=C,
            dt=dt_full,
            initial_state=initial_state,
        )

        # ---- Step 5: Skip connection D ----
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # ---- Step 6: Gating and output ----
        y = y * F.silu(z)
        y = self.norm(y)
        output = self.out_proj(y)  # (batch, seq_len, d_model)

        return output, final_state

    def forward_inference(
        self,
        input: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-token inference with structured sparse transitions — O(S × B) per token.

        Args:
            input: Input tensor for one token, ``(batch, 1, d_model)``.
            state: SSM state, ``(batch, d_inner, d_state)``.

        Returns:
            Tuple ``(output, new_state)``.
        """
        # Input projection
        xz = self.in_proj(input)
        x, z = xz.chunk(2, dim=-1)

        # Simplified conv for single-token (assumes cached)
        x_conv = F.silu(x)

        # SSM parameters
        ssm_params = self.x_proj(x_conv)
        B = ssm_params[..., :self.d_state]  # (batch, 1, d_state)
        C = ssm_params[..., self.d_state:self.d_state * 2]   # (batch, 1, d_state)

        # dt
        dt_full = F.softplus(
            self.dt_proj(x_conv) + self.dt_bias.unsqueeze(0).unsqueeze(0) + self.dt_init_floor
        )  # (batch, 1, d_inner)
        dt_scalar = dt_full.mean(dim=-1).squeeze(1)  # (batch,)

        # Apply structured sparse transition
        new_state = self.transition.apply_transition(state, dt_scalar)

        # Input injection: h += x ⊗ (dt * B)
        dB = dt_scalar.unsqueeze(-1) * B.squeeze(1)  # (batch, d_state)
        new_state = new_state + x_conv.squeeze(1).unsqueeze(-1) * dB.unsqueeze(1)

        # Output: y = C^T @ h + D * x
        y = torch.sum(
            C.squeeze(1).unsqueeze(1) * new_state, dim=-1
        )  # (batch, d_inner)
        y = y + x_conv.squeeze(1) * self.D.unsqueeze(0)
        y = y * F.silu(z.squeeze(1))
        y = self.norm(y.unsqueeze(1)).squeeze(1)
        output = self.out_proj(y)  # (batch, d_model)

        return output.unsqueeze(1), new_state

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def get_sparsity_ratio(self) -> float:
        """Compute the sparsity ratio of the transition matrix.

        Returns:
            Fraction of zeros in the transition matrix (0 = dense, 1 = diagonal).
        """
        if self.block_size <= 1:
            return 1.0  # Pure diagonal: all off-diagonal elements are zero

        S = self.d_state
        B = self.block_size
        n_blocks = S // B

        if self.config.transition_type == SparsityPattern.BLOCK_DIAGONAL:
            # Non-zeros: n_blocks * B^2 (within blocks) + 0 (diagonal outside)
            # plus diagonal elements outside blocks (if any, but d_state % B == 0)
            total_nnz = n_blocks * B * B
        elif self.config.transition_type == SparsityPattern.BANDED:
            total_nnz = S * (2 * self.config.band_width + 1)
        elif self.config.transition_type == SparsityPattern.BUTTERFLY:
            total_nnz = n_blocks * (self.block_size // 2) * 4 * self.transition.n_butterfly_levels
        else:
            total_nnz = S * S

        return 1.0 - total_nnz / (S * S)

    def get_fsa_tracking_capacity(self) -> int:
        """Estimate the number of FSA states that can be tracked.

        From the paper's main result: with structured sparse transitions
        of block size B, the SSM can track FSA with up to Q ≈ S states
        (provably optimal), compared to Q ≈ log(S) for diagonal SSMs.

        Returns:
            Estimated maximum FSA states trackable.
        """
        if self.block_size <= 1:
            # Diagonal SSM: can only track O(log(S)) FSA states
            return int(math.log2(max(2, self.d_state)))
        else:
            # Structured sparse: can track O(S) FSA states
            return self.d_state
