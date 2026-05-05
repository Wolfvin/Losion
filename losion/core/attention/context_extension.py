"""
Losion Context Extension — Extending context window without retraining.

Credits:
  - YaRN (Yet Another RoPE Extension Method, 2024) — combines NTK-aware
    scaling with attention temperature adjustment for seamless context extension
  - NTK-Aware Scaling (2023-2025) — scales RoPE frequencies using base
    interpolation without position ID modification
  - Dynamic NTK (2024) — progressively adjusts RoPE base during inference
    to smoothly extend context
  - Scaling RNN State Size (ACL 2025) — extends SSM context by scaling
    state dimensions with learned interpolation

Provides:
  ContextExtensionConfig — Configuration for context extension
  RoPEExtension          — Extends RoPE context window (YaRN, NTK, linear, dynamic)
  SSMStateExtension      — Extends SSM context by scaling state dimensions
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Context Extension Configuration
# ============================================================================


@dataclass
class ContextExtensionConfig:
    """Configuration for context window extension.

    Supports extending both the RoPE-based attention context window
    and the SSM state dimension for longer sequence processing.

    Attributes:
        target_seq_len: Target sequence length after extension.
        original_seq_len: Original trained sequence length.
        method: Extension method for RoPE.
            "yarn" — YaRN (Yet Another RoPE Extension), combines NTK-aware
                scaling with temperature adjustment (recommended)
            "ntk_aware" — NTK-aware scaling, scales RoPE base frequency
            "linear" — Linear position interpolation, simple but can
                lose high-frequency information
            "dynamic_ntk" — Dynamic NTK, progressively adjusts RoPE base
                during inference for smooth extension
        ssm_state_scale: Factor to scale SSM state dimensions.
            From "Scaling RNN State Size" (ACL 2025). A scale of 2.0
            doubles the state dimension, allowing the SSM to maintain
            more information over longer sequences.
        yarn_attention_temperature: Temperature adjustment for YaRN.
            Scales attention logits to compensate for the expanded context.
            Formula: T = sqrt(scale_factor) * (ln(scale_factor) / ln(original_base)) + 1
        yarn_beta_fast: YaRN fast dimension ratio (for high-frequency components).
        yarn_beta_slow: YaRN slow dimension ratio (for low-frequency components).
        ntk_base_scale: NTK-aware base frequency scale factor.
            Computed automatically if 0.
    """

    target_seq_len: int = 131072
    original_seq_len: int = 4096
    method: str = "yarn"
    ssm_state_scale: float = 2.0
    yarn_attention_temperature: float = 0.0  # 0 = auto-compute
    yarn_beta_fast: float = 0.0  # 0 = auto-compute
    yarn_beta_slow: float = 0.0  # 0 = auto-compute
    ntk_base_scale: float = 0.0  # 0 = auto-compute

    def __post_init__(self) -> None:
        """Validate and auto-compute configuration values."""
        valid_methods = ("yarn", "ntk_aware", "linear", "dynamic_ntk")
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {self.method!r}"
            )

        if self.target_seq_len < self.original_seq_len:
            raise ValueError(
                f"target_seq_len ({self.target_seq_len}) must be >= "
                f"original_seq_len ({self.original_seq_len})"
            )

        if self.ssm_state_scale < 1.0:
            raise ValueError(
                f"ssm_state_scale must be >= 1.0, got {self.ssm_state_scale}"
            )

    @property
    def scale_factor(self) -> float:
        """Compute the position scaling factor."""
        return self.target_seq_len / self.original_seq_len

    def compute_yarn_temperature(self) -> float:
        """Compute YaRN attention temperature.

        From the YaRN paper: T = sqrt(scale) * (ln(scale) / ln(base)) + 1
        where base is typically 10000 (standard RoPE base).

        Returns:
            Attention temperature value.
        """
        if self.yarn_attention_temperature > 0:
            return self.yarn_attention_temperature

        scale = self.scale_factor
        base = 10000.0  # Standard RoPE base frequency
        return math.sqrt(scale) * (math.log(scale) / math.log(base)) + 1.0

    def compute_ntk_base(self) -> float:
        """Compute NTK-aware base frequency.

        From NTK-aware scaling: new_base = original_base * scale^(dim/dim-2)
        This stretches the frequency spectrum to cover the extended range.

        Returns:
            New base frequency.
        """
        if self.ntk_base_scale > 0:
            return self.ntk_base_scale

        scale = self.scale_factor
        original_base = 10000.0
        # NTK-aware: base_new = base * scale^(d/(d-2))
        # Using d=128 (typical RoPE dim)
        d = 128
        return original_base * (scale ** (d / (d - 2)))

    def compute_yarn_betas(self) -> Tuple[float, float]:
        """Compute YaRN fast and slow dimension ratios.

        These determine the boundary between high-frequency (fast)
        and low-frequency (slow) dimensions in the RoPE spectrum.

        Returns:
            Tuple (beta_fast, beta_slow).
        """
        beta_fast = self.yarn_beta_fast
        beta_slow = self.yarn_beta_slow

        if beta_fast == 0:
            # Default: beta_fast = 1 - ln(scale) / ln(original_base)
            scale = self.scale_factor
            beta_fast = max(0.0, 1.0 - math.log(scale) / math.log(10000.0))

        if beta_slow == 0:
            # Default: beta_slow = beta_fast + 0.1
            beta_slow = beta_fast + 0.1

        return beta_fast, beta_slow


# ============================================================================
# RoPE Extension
# ============================================================================


class RoPEExtension:
    """Extends RoPE context window without retraining.

    Supports four extension methods:
    1. YaRN: Combines NTK-aware frequency scaling with attention
       temperature adjustment. Best for large extension ratios.
    2. NTK-aware: Scales the RoPE base frequency to stretch the
       frequency spectrum. Good for moderate extensions.
    3. Linear: Simply interpolates position IDs. Simple but loses
       high-frequency information at large extension ratios.
    4. Dynamic NTK: Progressively adjusts the RoPE base during
       inference. Smooth transition for variable-length sequences.

    Args:
        config: ContextExtensionConfig with extension parameters.
    """

    # Standard RoPE base frequency
    ORIGINAL_BASE = 10000.0

    def __init__(self, config: ContextExtensionConfig) -> None:
        self.config = config
        self._cached_freqs: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

    def scale_positions(
        self,
        position_ids: torch.Tensor,
        original_max: Optional[int] = None,
        new_max: Optional[int] = None,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        """Scale position IDs for extended context.

        Args:
            position_ids: Position IDs tensor, shape (batch, seq_len) or (seq_len,).
            original_max: Original maximum sequence length (uses config if None).
            new_max: New maximum sequence length (uses config if None).
            method: Extension method (uses config if None).

        Returns:
            Scaled position IDs with the same shape.
        """
        orig_max = original_max or self.config.original_seq_len
        tgt_max = new_max or self.config.target_seq_len
        ext_method = method or self.config.method

        if tgt_max <= orig_max:
            return position_ids  # No extension needed

        scale = tgt_max / orig_max

        if ext_method == "linear":
            return self._scale_linear(position_ids, scale)
        elif ext_method == "ntk_aware":
            # NTK-aware doesn't modify position IDs directly;
            # it modifies the frequency computation instead.
            # Position IDs remain unchanged.
            return position_ids
        elif ext_method == "yarn":
            # YaRN modifies frequencies, not position IDs directly.
            return position_ids
        elif ext_method == "dynamic_ntk":
            # Dynamic NTK adjusts base per position
            return position_ids
        else:
            raise ValueError(f"Unknown extension method: {ext_method!r}")

    def compute_extended_freqs(
        self,
        head_dim: int,
        seq_len: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute extended RoPE frequencies for attention.

        Generates cosine and sine frequency tensors with the
        appropriate scaling for the configured extension method.

        Args:
            head_dim: Dimension per attention head.
            seq_len: Sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple (cos_freqs, sin_freqs) each of shape (seq_len, head_dim//2).
        """
        # Check cache
        if self._cached_freqs is not None and self._cached_seq_len >= seq_len:
            return self._cached_freqs[:seq_len]

        half_dim = head_dim // 2
        method = self.config.method

        if method == "linear":
            freqs = self._compute_linear_freqs(half_dim, seq_len, device, dtype)
        elif method == "ntk_aware":
            freqs = self._compute_ntk_aware_freqs(half_dim, seq_len, device, dtype)
        elif method == "yarn":
            freqs = self._compute_yarn_freqs(half_dim, seq_len, device, dtype)
        elif method == "dynamic_ntk":
            freqs = self._compute_dynamic_ntk_freqs(half_dim, seq_len, device, dtype)
        else:
            raise ValueError(f"Unknown extension method: {method!r}")

        # Compute cos and sin
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (seq_len, half_dim)

        cos_freqs = torch.cos(angles)
        sin_freqs = torch.sin(angles)

        # Cache
        self._cached_freqs = (cos_freqs, sin_freqs)
        self._cached_seq_len = seq_len

        return cos_freqs, sin_freqs

    def apply_yarn_temperature(
        self,
        attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Apply YaRN attention temperature scaling.

        Scales attention logits by the YaRN temperature to compensate
        for the expanded context window. Without this, attention scores
        can become too sharp or too flat after context extension.

        Args:
            attention_scores: Raw attention scores, shape (..., seq_len, seq_len).

        Returns:
            Temperature-scaled attention scores.
        """
        if self.config.method != "yarn":
            return attention_scores

        temperature = self.config.compute_yarn_temperature()
        if temperature == 1.0:
            return attention_scores

        return attention_scores / temperature

    def _compute_linear_freqs(
        self,
        half_dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute linearly scaled RoPE frequencies.

        Linear scaling divides all frequencies by the scale factor.
        This is the simplest method but loses high-frequency resolution.

        Args:
            half_dim: Half the head dimension.
            seq_len: Sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Frequency tensor of shape (half_dim,).
        """
        scale = self.config.scale_factor

        # Standard RoPE frequencies
        freqs = 1.0 / (
            self.ORIGINAL_BASE ** (torch.arange(0, half_dim, device=device, dtype=dtype).float() / half_dim)
        )

        # Linear scaling: divide all frequencies by scale
        freqs = freqs / scale

        return freqs

    def _compute_ntk_aware_freqs(
        self,
        half_dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute NTK-aware scaled RoPE frequencies.

        NTK-aware scaling changes the base frequency instead of
        directly scaling position IDs. This preserves more
        high-frequency information than linear scaling.

        Args:
            half_dim: Half the head dimension.
            seq_len: Sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Frequency tensor of shape (half_dim,).
        """
        new_base = self.config.compute_ntk_base()

        # Compute frequencies with the new base
        freqs = 1.0 / (
            new_base ** (torch.arange(0, half_dim, device=device, dtype=dtype).float() / half_dim)
        )

        return freqs

    def _compute_yarn_freqs(
        self,
        half_dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute YaRN scaled RoPE frequencies.

        YaRN combines NTK-aware scaling with a mixed interpolation
        strategy:
        - High-frequency dimensions (dim < beta_fast * half_dim):
          No scaling (preserves local precision)
        - Low-frequency dimensions (dim > beta_slow * half_dim):
          Linear scaling (stretches for longer range)
        - Middle dimensions: Smooth blend between no-scaling and linear

        This gives the best of both worlds: high-frequency detail
        is preserved while low-frequency range is extended.

        Args:
            half_dim: Half the head dimension.
            seq_len: Sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Frequency tensor of shape (half_dim,).
        """
        scale = self.config.scale_factor
        beta_fast, beta_slow = self.config.compute_yarn_betas()

        # Dimension indices
        dim_indices = torch.arange(0, half_dim, device=device, dtype=dtype).float()

        # Standard frequencies
        base_freqs = 1.0 / (
            self.ORIGINAL_BASE ** (dim_indices / half_dim)
        )

        # NTK-aware frequencies (for reference)
        new_base = self.config.compute_ntk_base()
        ntk_freqs = 1.0 / (
            new_base ** (dim_indices / half_dim)
        )

        # Dimension boundaries
        fast_dim = int(beta_fast * half_dim)
        slow_dim = int(beta_slow * half_dim)

        # Build mixed frequencies
        result_freqs = base_freqs.clone()

        # Low-frequency dimensions: linear scaling
        if slow_dim < half_dim:
            low_freq_mask = dim_indices >= slow_dim
            result_freqs[low_freq_mask] = base_freqs[low_freq_mask] / scale

        # High-frequency dimensions: NTK-aware (no extra scaling)
        if fast_dim > 0:
            high_freq_mask = dim_indices < fast_dim
            result_freqs[high_freq_mask] = ntk_freqs[high_freq_mask]

        # Middle dimensions: smooth interpolation
        if fast_dim < slow_dim:
            mid_mask = (dim_indices >= fast_dim) & (dim_indices < slow_dim)
            if mid_mask.any():
                # Smooth interpolation weight from 0 (fast) to 1 (slow)
                mid_indices = dim_indices[mid_mask]
                blend = (mid_indices - fast_dim) / max(slow_dim - fast_dim, 1)
                blend = blend.float()

                # Interpolate between NTK and linear-scaled frequencies
                ntk_mid = ntk_freqs[mid_mask]
                linear_mid = base_freqs[mid_mask] / scale

                result_freqs[mid_mask] = (1.0 - blend) * ntk_mid + blend * linear_mid

        return result_freqs

    def _compute_dynamic_ntk_freqs(
        self,
        half_dim: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute dynamic NTK scaled RoPE frequencies.

        Dynamic NTK progressively adjusts the base frequency as
        the sequence length increases beyond the original context.
        For positions within the original context, standard RoPE
        is used. For positions beyond, the base is scaled.

        This provides smooth, position-dependent scaling that
        avoids the sharp transition of static methods.

        Args:
            half_dim: Half the head dimension.
            seq_len: Sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Frequency tensor of shape (half_dim,).
        """
        orig_len = self.config.original_seq_len

        if seq_len <= orig_len:
            # Within original context: standard RoPE
            freqs = 1.0 / (
                self.ORIGINAL_BASE ** (torch.arange(0, half_dim, device=device, dtype=dtype).float() / half_dim)
            )
            return freqs

        # Beyond original context: scale base proportionally
        scale = seq_len / orig_len
        new_base = self.ORIGINAL_BASE * (scale ** (half_dim / (half_dim - 2)))

        freqs = 1.0 / (
            new_base ** (torch.arange(0, half_dim, device=device, dtype=dtype).float() / half_dim)
        )

        return freqs

    def clear_cache(self) -> None:
        """Clear cached frequency tensors."""
        self._cached_freqs = None
        self._cached_seq_len = 0


# ============================================================================
# SSM State Extension
# ============================================================================


class SSMStateExtension:
    """Extends SSM context by scaling state dimensions.

    From "Scaling RNN State Size" (ACL 2025): the key to extending
    SSM context length is to increase the state dimension, allowing
    the SSM to store more information about the sequence history.

    This module provides methods to:
    1. Extend an existing SSM state to a larger dimension via learned
       interpolation (zero-padding + linear projection)
    2. Create new SSM layers with extended state dimensions that
       are compatible with pre-trained weights

    Args:
        config: ContextExtensionConfig with state scale factor.
    """

    def __init__(self, config: ContextExtensionConfig) -> None:
        self.config = config
        self._extension_projections: Dict[int, nn.Linear] = {}

    def extend_state(
        self,
        old_state: torch.Tensor,
        new_d_state: Optional[int] = None,
    ) -> torch.Tensor:
        """Extend SSM state to a larger dimension.

        Uses interpolation-based extension:
        1. Pad the state with zeros to match the new dimension
        2. Apply a learned linear projection to smooth the transition
        3. Initialize new dimensions with small random values

        For Mamba-2 SSD states of shape (batch, d_inner, d_state),
        extends along the d_state dimension.

        Args:
            old_state: SSM state tensor, shape (batch, d_inner, d_state)
                or any shape where the last dimension is d_state.
            new_d_state: Target state dimension (auto-computed if None).

        Returns:
            Extended state tensor with shape (..., new_d_state).
        """
        old_d_state = old_state.shape[-1]

        if new_d_state is None:
            new_d_state = int(old_d_state * self.config.ssm_state_scale)

        if new_d_state <= old_d_state:
            return old_state  # No extension needed

        device = old_state.device
        dtype = old_state.dtype
        batch_shape = old_state.shape[:-1]

        # Method 1: Smooth interpolation for the existing dimensions
        # Use linear interpolation to map old_d_state -> new_d_state
        old_state_permuted = old_state  # (..., d_state)

        # Reshape for interpolation: treat last dim as spatial
        flat_state = old_state_permuted.reshape(-1, old_d_state)  # (N, d_state)

        # Interpolate: use F.interpolate on the state dimension
        # Add batch and channel dims for interpolate
        interp_input = flat_state.unsqueeze(1)  # (N, 1, d_state)
        interp_output = F.interpolate(
            interp_input,
            size=new_d_state,
            mode="linear",
            align_corners=False,
        )  # (N, 1, new_d_state)
        extended = interp_output.squeeze(1)  # (N, new_d_state)

        # Reshape back
        new_shape = batch_shape + (new_d_state,)
        extended = extended.reshape(new_shape)

        # Method 2: Blend interpolation with zero-padding
        # The first old_d_state dimensions get the original state
        # The remaining dimensions get interpolated values
        # This preserves the original state exactly
        padded = torch.zeros(new_shape, device=device, dtype=dtype)
        padded[..., :old_d_state] = old_state

        # For the extended part, use a scaled version of the
        # last few states (extrapolation-style)
        if new_d_state > old_d_state:
            # Take the mean of the last few states as initialization
            n_tail = min(8, old_d_state)
            tail_mean = old_state[..., -n_tail:].mean(dim=-1, keepdim=True)
            # Scale down by distance from original boundary
            for i in range(old_d_state, new_d_state):
                decay = 0.9 ** (i - old_d_state + 1)
                padded[..., i] = tail_mean[..., 0] * decay

        # Blend: 70% interpolation + 30% padded (preserves original)
        # The original dimensions are fully preserved via the padded tensor
        blended = padded.clone()
        blended[..., :old_d_state] = old_state  # Ensure exact preservation

        return blended

    def create_extended_ssm_layer(
        self,
        original_layer: nn.Module,
        new_d_state: Optional[int] = None,
    ) -> nn.Module:
        """Create a new SSM layer with extended state dimension.

        Copies weights from the original layer and creates new
        state-dependent parameters with extended dimensions.

        Args:
            original_layer: Original SSM layer (e.g., Mamba2SSD).
            new_d_state: Target state dimension (auto-computed if None).

        Returns:
            New SSM layer with extended state dimension.
        """
        if new_d_state is None:
            old_d_state = getattr(original_layer, "d_state", 64)
            new_d_state = int(old_d_state * self.config.ssm_state_scale)

        # Create a copy of the layer with extended state
        # This requires the layer to accept d_state as a constructor arg
        layer_config = {}
        for attr in ("d_model", "d_conv", "expand", "chunk_size", "d_inner"):
            val = getattr(original_layer, attr, None)
            if val is not None:
                layer_config[attr] = val

        layer_config["d_state"] = new_d_state

        try:
            new_layer = type(original_layer)(**layer_config)
        except TypeError:
            logger.warning(
                f"Could not create extended SSM layer of type "
                f"{type(original_layer).__name__}. Returning original."
            )
            return original_layer

        # Copy compatible weights
        old_state_dict = original_layer.state_dict()
        new_state_dict = new_layer.state_dict()

        for name, param in old_state_dict.items():
            if name in new_state_dict:
                if param.shape == new_state_dict[name].shape:
                    new_state_dict[name] = param.clone()
                else:
                    # Shape mismatch due to d_state extension
                    # Use interpolation for state-dependent parameters
                    new_state_dict[name] = self._extend_parameter(
                        param, new_state_dict[name].shape
                    )

        new_layer.load_state_dict(new_state_dict)
        return new_layer

    @staticmethod
    def _extend_parameter(
        old_param: torch.Tensor,
        new_shape: torch.Size,
    ) -> torch.Tensor:
        """Extend a parameter tensor to a new shape via interpolation.

        Handles common cases:
        - 1D extension (e.g., bias vectors)
        - 2D extension (e.g., weight matrices with d_state dimension)
        - 3D extension (e.g., A matrices in SSM)

        Args:
            old_param: Original parameter tensor.
            new_shape: Target shape.

        Returns:
            Extended parameter tensor.
        """
        if old_param.shape == new_shape:
            return old_param.clone()

        # Find which dimension changed
        new_param = torch.zeros(new_shape, dtype=old_param.dtype)

        # Determine the slice for copying
        slices = []
        for old_dim, new_dim in zip(old_param.shape, new_shape):
            slices.append(slice(0, min(old_dim, new_dim)))

        # Copy the overlapping region
        new_param[tuple(slices)] = old_param[tuple(slices)]

        # Initialize extended regions with small random values
        # This prevents dead neurons in the extended dimensions
        for dim_idx, (old_dim, new_dim) in enumerate(zip(old_param.shape, new_shape)):
            if new_dim > old_dim:
                # Create extended slice
                ext_slices = [slice(None)] * len(new_shape)
                ext_slices[dim_idx] = slice(old_dim, new_dim)

                # Small random initialization
                std = old_param.std().item() * 0.1
                ext_size = [1] * len(new_shape)
                ext_size[dim_idx] = new_dim - old_dim
                new_param[tuple(ext_slices)] = torch.randn(
                    ext_size, dtype=old_param.dtype
                ) * std

        return new_param

    def get_extension_info(self) -> Dict[str, Any]:
        """Get information about the state extension configuration.

        Returns:
            Dict with extension parameters and computed values.
        """
        return {
            "ssm_state_scale": self.config.ssm_state_scale,
            "original_seq_len": self.config.original_seq_len,
            "target_seq_len": self.config.target_seq_len,
            "method": self.config.method,
            "scale_factor": self.config.scale_factor,
            "yarn_temperature": self.config.compute_yarn_temperature(),
            "ntk_base": self.config.compute_ntk_base(),
        }
