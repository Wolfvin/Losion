"""
Asymmetric MoE Placement — Selective MoE Layers for Losion Framework v0.4.

Upgrade #11: Not every layer needs MoE.  In a standard Transformer every
FFN layer is replaced by MoE, which leads to a large parameter count even
when some layers would benefit just as much from a dense FFN.

AsymmetricMoEPlacement allows a **configurable pattern** that specifies
which layers use MoE and which use a dense FFN:

    pattern = "dense-moe-moe-dense-moe-moe"

This means:
  - Layer 0: dense FFN   (fewer parameters, lower latency)
  - Layer 1: MoE         (more parameters, expert routing)
  - Layer 2: MoE
  - Layer 3: dense FFN
  - Layer 4: MoE
  - Layer 5: MoE

Benefits:
  * **Fewer total parameters** — dense FFN is much smaller than MoE.
  * **Targeted capacity** — MoE is placed where it matters most
    (typically deeper layers for representation richness, or middle
    layers for contextual diversity).
  * **Faster inference** — dense layers avoid expert-dispatch overhead.

Common Patterns
---------------
- ``"late-moe"``:     All dense until the last N layers.  Early layers
                       capture generic patterns; MoE adds specialisation
                       at the top.
- ``"early-moe"``:    MoE in the first N layers; dense later.  Rare but
                       useful for input-dependent early routing.
- ``"alternating"``:  Dense-MoE-Dense-MoE-...  Balances capacity.
- ``"custom"``:       User-specified string of "dense"/"moe" tokens.

Architecture
------------
1. **AsymmetricPlacementConfig** — parses and validates a placement
   pattern string, providing convenient accessors.

2. **DenseFFN** — a standard SwiGLU feed-forward block used in place of
   MoE for "dense" layers.

3. **AsymmetricMoEPlacement** — a ``nn.Module`` that constructs a stack
   of layers following the pattern.  Each "moe" layer uses a provided
   MoE module; each "dense" layer uses :class:`DenseFFN`.

References
----------
- Fedus et al., "Switch Transformers" (2021) — MoE placement analysis.
- Du et al., "DeepSeekMoE" (2024) — fine-grained expert segmentation.
- Lieber et al., "Jamba" (2024) — alternating SSM/attention/MoE layers.

Hardware: Pure PyTorch.  No custom CUDA kernels required.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Placement Config
# ---------------------------------------------------------------------------

class AsymmetricPlacementConfig:
    """
    Configuration for asymmetric MoE placement.

    Parses a pattern string like ``"dense-moe-moe-dense-moe-moe"`` into
    a structured representation and provides convenient accessors.

    The pattern string must consist of tokens ``"dense"`` and ``"moe"``
    separated by hyphens or spaces.  Shorthand: ``"d"`` = dense, ``"m"`` = moe.

    Predefined shortcuts:
        - ``"late-moe-{N}"``:  First N//2 layers dense, last N//2 MoE.
        - ``"early-moe-{N}"``: First N//2 layers MoE, last N//2 dense.
        - ``"alternating-{N}"``: Dense-MoE-Dense-MoE-... for N layers.

    Args:
        pattern: Placement pattern string or list of booleans
                 (True = MoE, False = dense).
        num_layers: Total number of layers (required for predefined
                    shortcuts or when pattern is a list).

    Example
    -------
    >>> cfg = AsymmetricPlacementConfig("dense-moe-moe-dense-moe-moe")
    >>> cfg.num_layers
    6
    >>> cfg.is_moe_layer(0)
    False
    >>> cfg.is_moe_layer(1)
    True
    >>> cfg.moe_layer_indices
    [1, 2, 4, 5]
    >>> cfg.dense_layer_indices
    [0, 3]
    """

    # Token mapping
    _TOKEN_MAP = {
        "dense": False,
        "d": False,
        "moe": True,
        "m": True,
    }

    def __init__(
        self,
        pattern: Union[str, List[bool]],
        num_layers: Optional[int] = None,
    ) -> None:
        if isinstance(pattern, list):
            # Direct boolean list
            self._placement = list(pattern)
        else:
            self._placement = self._parse_pattern(pattern, num_layers)

        self._validate()

    def _parse_pattern(
        self, pattern: str, num_layers: Optional[int]
    ) -> List[bool]:
        """
        Parse a pattern string into a list of booleans.

        Args:
            pattern:    Pattern string.
            num_layers: Required for predefined shortcuts.

        Returns:
            List of booleans (True = MoE, False = dense).
        """
        pattern = pattern.strip().lower()

        # Check for predefined shortcuts
        if pattern.startswith("late-moe"):
            n = self._extract_num(pattern, num_layers)
            half = n // 2
            return [False] * half + [True] * (n - half)

        if pattern.startswith("early-moe"):
            n = self._extract_num(pattern, num_layers)
            half = n // 2
            return [True] * half + [False] * (n - half)

        if pattern.startswith("alternating"):
            n = self._extract_num(pattern, num_layers)
            return [(i % 2 == 1) for i in range(n)]

        if pattern == "all-moe":
            n = num_layers or 12
            return [True] * n

        if pattern == "all-dense":
            n = num_layers or 12
            return [False] * n

        # Explicit pattern: "dense-moe-moe-dense-..."
        tokens = re.split(r"[-\s,]+", pattern)
        placement = []
        for tok in tokens:
            if not tok:
                continue
            if tok not in self._TOKEN_MAP:
                raise ValueError(
                    f"Unknown pattern token '{tok}'. "
                    f"Expected one of: {list(self._TOKEN_MAP.keys())}"
                )
            placement.append(self._TOKEN_MAP[tok])

        if not placement:
            raise ValueError(f"Empty pattern: '{pattern}'")

        return placement

    @staticmethod
    def _extract_num(pattern: str, default: Optional[int]) -> int:
        """Extract the numeric suffix from a shortcut pattern."""
        match = re.search(r"(\d+)", pattern)
        if match:
            return int(match.group(1))
        if default is not None:
            return default
        raise ValueError(
            f"Pattern '{pattern}' requires a numeric suffix or num_layers."
        )

    def _validate(self) -> None:
        """Validate the placement list."""
        if not self._placement:
            raise ValueError("Placement list must not be empty.")
        if not all(isinstance(v, bool) for v in self._placement):
            raise ValueError("All placement entries must be booleans.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        """Total number of layers."""
        return len(self._placement)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Return True if the layer at ``layer_idx`` is an MoE layer."""
        return self._placement[layer_idx]

    def is_dense_layer(self, layer_idx: int) -> bool:
        """Return True if the layer at ``layer_idx`` is a dense FFN layer."""
        return not self._placement[layer_idx]

    @property
    def moe_layer_indices(self) -> List[int]:
        """Indices of MoE layers."""
        return [i for i, v in enumerate(self._placement) if v]

    @property
    def dense_layer_indices(self) -> List[int]:
        """Indices of dense FFN layers."""
        return [i for i, v in enumerate(self._placement) if not v]

    @property
    def moe_fraction(self) -> float:
        """Fraction of layers that are MoE."""
        return sum(self._placement) / len(self._placement)

    @property
    def placement(self) -> List[bool]:
        """Full placement list (True = MoE, False = dense)."""
        return list(self._placement)

    def __repr__(self) -> str:
        tokens = ["moe" if v else "dense" for v in self._placement]
        return f"AsymmetricPlacementConfig('{ '-'.join(tokens) }')"


# ---------------------------------------------------------------------------
# Dense FFN (used in place of MoE)
# ---------------------------------------------------------------------------

class DenseFFN(nn.Module):
    """
    Standard SwiGLU feed-forward block, used in "dense" layers.

    Much smaller than an MoE layer with the same ``d_model`` because
    there is only one expert (no routing, no multiple projections).

    Args:
        d_model: Model dimension.
        d_ff:    Intermediate dimension (default 4 * d_model).
        dropout: Dropout rate (default 0.0).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or (4 * d_model)

        self.up_proj = nn.Linear(d_model, self.d_ff, bias=False)
        self.gate_proj = nn.Linear(d_model, self.d_ff, bias=False)
        self.down_proj = nn.Linear(self.d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.RMSNorm(d_model, eps=1e-5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            output: ``(batch, seq_len, d_model)``.
            aux:    Empty dict (no auxiliary losses for dense layers).
        """
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        out = self.dropout(out)
        out = self.norm(out)
        return out, {}


# ---------------------------------------------------------------------------
# AsymmetricMoEPlacement
# ---------------------------------------------------------------------------

class AsymmetricMoEPlacement(nn.Module):
    """
    Stack of layers following an asymmetric MoE/dense placement pattern.

    Each layer is either:
      - A **dense FFN** (:class:`DenseFFN`) — small, fast, no routing.
      - An **MoE layer** (user-provided module) — large, expert-routed.

    The placement is specified by a :class:`AsymmetricPlacementConfig`
    (or a pattern string that is converted into one).

    This module handles the construction and forwarding through the
    heterogenous stack, returning outputs and aggregated auxiliary losses.

    Example
    -------
    >>> # Assume we have a factory that creates MoE layers
    >>> def make_moe():
    ...     return HeterogeneousMoE(d_model=256, expert_ff_dims=[512, 1024, 512, 1024])
    ...
    >>> placement = AsymmetricMoEPlacement(
    ...     d_model=256,
    ...     d_ff=1024,
    ...     pattern="dense-moe-moe-dense-moe-moe",
    ...     moe_factory=make_moe,
    ... )
    >>> x = torch.randn(2, 16, 256)
    >>> out, aux = placement(x)
    >>> out.shape
    torch.Size([2, 16, 256])

    Args:
        d_model:     Model dimension.
        d_ff:        Intermediate dimension for dense FFN layers.
        pattern:     Placement pattern string or list of booleans, passed
                     to :class:`AsymmetricPlacementConfig`.
        moe_factory: Callable that returns a new MoE ``nn.Module`` each
                     time it is called.  Must accept no arguments and
                     return a module with a ``forward(x)`` method that
                     returns ``(output, aux_dict)``.
        dense_dropout: Dropout for dense FFN layers (default 0.0).
        num_layers:  Total number of layers (needed for predefined
                     shortcut patterns).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        pattern: Union[str, List[bool]],
        moe_factory: Any,  # Callable[[], nn.Module]
        dense_dropout: float = 0.0,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        # Parse placement
        self.config = AsymmetricPlacementConfig(pattern, num_layers)

        # ---- Build layers ----
        self.layers = nn.ModuleList()
        for layer_idx in range(self.config.num_layers):
            if self.config.is_moe_layer(layer_idx):
                self.layers.append(moe_factory())
            else:
                self.layers.append(
                    DenseFFN(d_model=d_model, d_ff=d_ff, dropout=dense_dropout)
                )

        # ---- Per-layer norms (before each layer) ----
        self.layer_norms = nn.ModuleList(
            [nn.RMSNorm(d_model, eps=1e-5) for _ in range(self.config.num_layers)]
        )

        # ---- Final output norm ----
        self.output_norm = nn.RMSNorm(d_model, eps=1e-5)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the asymmetric layer stack.

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            output: ``(batch, seq_len, d_model)``.
            aux:    Dict of aggregated auxiliary losses from MoE layers.
        """
        batch, seq_len, _ = x.shape
        aux: Dict[str, torch.Tensor] = {}

        hidden = x
        for layer_idx in range(self.config.num_layers):
            # Pre-norm
            residual = hidden
            hidden_normed = self.layer_norms[layer_idx](hidden)

            # Forward through the layer
            layer_out, layer_aux = self.layers[layer_idx](hidden_normed)

            # Residual connection
            hidden = residual + layer_out

            # Aggregate auxiliary losses
            for key, val in layer_aux.items():
                if key in aux:
                    aux[key] = aux[key] + val
                else:
                    aux[key] = val

        output = self.output_norm(hidden)
        return output, aux

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a specific layer uses MoE."""
        return self.config.is_moe_layer(layer_idx)

    @property
    def moe_layer_indices(self) -> List[int]:
        """Indices of MoE layers."""
        return self.config.moe_layer_indices

    @property
    def dense_layer_indices(self) -> List[int]:
        """Indices of dense FFN layers."""
        return self.config.dense_layer_indices

    @property
    def moe_fraction(self) -> float:
        """Fraction of layers that are MoE."""
        return self.config.moe_fraction

    def parameter_count_by_type(self) -> Dict[str, int]:
        """
        Return parameter counts broken down by layer type.

        Returns:
            Dict with keys ``"moe"``, ``"dense"``, and ``"total"``.
        """
        moe_params = 0
        dense_params = 0
        for i in range(self.config.num_layers):
            layer_params = sum(p.numel() for p in self.layers[i].parameters())
            if self.config.is_moe_layer(i):
                moe_params += layer_params
            else:
                dense_params += layer_params
        # Also count the norms
        norm_params = sum(p.numel() for p in self.layer_norms.parameters())
        norm_params += sum(p.numel() for p in self.output_norm.parameters())
        return {
            "moe": moe_params,
            "dense": dense_params,
            "norms": norm_params,
            "total": moe_params + dense_params + norm_params,
        }
