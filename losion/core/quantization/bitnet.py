"""
BitNet 1.58-bit Quantization for Losion Framework v0.4.

Upgrade #3: Ternary weight quantization {-1, 0, +1} for ~6x memory reduction
with minimal quality loss.

Key components:
1. BitNetConfig — Configuration for quantization schedule (gradual quantization)
2. BitNetLinear — Drop-in replacement for nn.Linear with 1.58-bit weights
3. Absmean quantization — Quantize full-precision weights to ternary values
4. Int2 weight storage — 3 ternary values packed into 2-bit integers
5. Straight-through estimator — Gradient flow through non-differentiable quantization

Background:
    BitNet b1.58 (Wang et al., 2024) quantizes weights to exactly three values:
    {-1, 0, +1}, which requires only ~1.58 bits per weight (log2(3) ≈ 1.585).
    This provides approximately 6x memory reduction compared to FP32 and
    ~2x compared to INT8, with surprisingly small quality degradation.

    The key insight is that the quantization scale factor preserves the
    magnitude information: W_quantized = round(W / absmean(W)) * absmean(W),
    where absmean(W) = mean(|W|).

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
No custom kernels required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BitNetConfig — Quantization schedule configuration
# ---------------------------------------------------------------------------

@dataclass
class BitNetConfig:
    """
    Configuration for BitNet 1.58-bit quantization schedule.

    Supports gradual quantization during training: the quantization ratio
    increases from ``initial_quant_ratio`` to 1.0 over
    ``warmup_steps`` training steps. This prevents training instability
    from abruptly switching to ternary weights.

    Attributes:
        enabled:              Master switch for BitNet quantization.
        warmup_steps:         Number of training steps over which
                              quantization is gradually introduced.
                              0 = immediate full quantization.
        initial_quant_ratio:  Fraction of weights quantized at step 0.
                              Must be in [0, 1].  0 = no quantization;
                              1 = full quantization from the start.
        threshold:            Percentile threshold for the zero bin in
                              absmean quantization.  Weights with magnitude
                              below ``threshold`` percentile of |W| are
                              quantized to 0.  Default 0.0 means no
                              explicit zero threshold (pure rounding).
        weight_decay:         Optional weight decay on the full-precision
                              latent weights during training.
        STE_mode:             Straight-through estimator variant.
                              "identity"  — gradient passes through unchanged
                              "atan"      — softer gradient using atan
                                            approximation (Better than
                                            Hardtanh, Bai et al. 2023)
        quantize_on_forward:  If True, quantize on every forward pass.
                              If False, use pre-quantized cached weights
                              (faster inference, no STE).
    """

    enabled: bool = True
    warmup_steps: int = 2000
    initial_quant_ratio: float = 0.0
    threshold: float = 0.0
    weight_decay: float = 0.0
    STE_mode: str = "identity"      # "identity" or "atan"
    quantize_on_forward: bool = True

    def get_quant_ratio(self, global_step: int) -> float:
        """
        Compute the current quantization ratio given the training step.

        Linearly interpolates from ``initial_quant_ratio`` to 1.0 over
        ``warmup_steps`` steps.

        Args:
            global_step: Current training step.

        Returns:
            Quantization ratio in [0, 1].
        """
        if not self.enabled:
            return 0.0
        if self.warmup_steps <= 0:
            return 1.0
        if global_step >= self.warmup_steps:
            return 1.0
        progress = global_step / self.warmup_steps
        return self.initial_quant_ratio + (1.0 - self.initial_quant_ratio) * progress


# ---------------------------------------------------------------------------
# Absmean quantization — core quantization primitive
# ---------------------------------------------------------------------------

def absmean_quantize(
    weight: torch.Tensor,
    quant_ratio: float = 1.0,
    threshold_percentile: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a weight tensor to ternary values {-1, 0, +1} using absmean.

    The quantization rule is::

        scale = mean(|W|)          # per-group or per-tensor
        W_q   = round(W / scale)   # in {-1, 0, +1}
        W_approx = W_q * scale

    When ``quant_ratio < 1``, only a random fraction ``quant_ratio`` of
    the weight elements are quantized; the rest retain their original
    values (smooth ramp-up during training).

    When ``threshold_percentile > 0``, weights whose absolute value
    falls below the given percentile of |W| are forced to 0 before
    rounding.  This controls the sparsity of the quantized weights.

    Args:
        weight:               Full-precision weight tensor.
        quant_ratio:          Fraction of weights to quantize (0–1).
        threshold_percentile: Percentile below which weights → 0.

    Returns:
        (quantized_weight, scale):
            quantized_weight: Same shape as *weight*, ternary * scale.
            scale:            Per-output-channel scale, shape ``(out_features,)``.
    """
    # Per output-channel absmean scale
    abs_mean = weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)  # (out, 1)
    scale = abs_mean.squeeze(-1)  # (out,)

    # Normalise
    w_norm = weight / abs_mean  # in roughly [-3, 3] for Gaussian weights

    # Optional threshold: zero out small weights
    if threshold_percentile > 0.0:
        flat_abs = weight.abs().flatten()
        k = max(1, int(threshold_percentile / 100.0 * flat_abs.numel()))
        thresh_val = flat_abs.kthvalue(k).values
        zero_mask = weight.abs() < thresh_val
        w_norm = w_norm.masked_fill(zero_mask, 0.0)

    # Round to ternary
    w_ternary = torch.round(w_norm).clamp(-1, 1)

    # Gradual quantization: blend original and quantized
    if quant_ratio < 1.0:
        mask = torch.rand_like(weight) < quant_ratio
        w_ternary = torch.where(mask, w_ternary, w_norm)

    quantized = w_ternary * abs_mean
    return quantized, scale


# ---------------------------------------------------------------------------
# Int2 weight packing / unpacking utilities
# ---------------------------------------------------------------------------

def pack_ternary_to_int2(ternary: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weight values {-1, 0, +1} into 2-bit integers.

    Mapping:
        -1 → 0b00 (0)
         0 → 0b01 (1)
        +1 → 0b10 (2)
        0b11 (3) is unused/padding.

    Each 2-bit slot stores one ternary value.  We pack 16 values per
    int32 element (32 bits / 2 bits = 16 slots).

    Args:
        ternary: Integer tensor with values in {-1, 0, +1}.

    Returns:
        Packed int32 tensor with ceil(numel / 16) elements.
    """
    # Map {-1, 0, +1} → {0, 1, 2}
    mapped = (ternary + 1).to(torch.int8)  # -1→0, 0→1, +1→2

    flat = mapped.flatten()
    n = flat.numel()
    pad_len = (16 - n % 16) % 16
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.int8, device=flat.device)])

    packed_len = flat.numel() // 16
    packed = torch.zeros(packed_len, dtype=torch.int32, device=flat.device)

    for i in range(16):
        slot_vals = flat[i::16].to(torch.int32)  # every 16th element
        packed |= (slot_vals << (2 * i))

    return packed


def unpack_int2_to_ternary(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """
    Unpack 2-bit integers back to ternary values {-1, 0, +1}.

    Reverse mapping:
        0b00 (0) → -1
        0b01 (1) →  0
        0b10 (2) → +1
        0b11 (3) → +1  (unused slot, treated as +1 for safety)

    Args:
        packed: int32 tensor from :func:`pack_ternary_to_int2`.
        numel:  Original number of ternary elements before packing.

    Returns:
        Tensor with values in {-1, 0, +1}, shape ``(numel,)``.
    """
    slots = []
    for i in range(16):
        slot_vals = ((packed >> (2 * i)) & 0b11).to(torch.int8)
        slots.append(slot_vals)

    # Interleave
    interleaved = torch.stack(slots, dim=1).flatten()  # (packed_len * 16,)

    # Map {0, 1, 2, 3} → {-1, 0, +1, +1}
    result = interleaved - 1  # 0→-1, 1→0, 2→+1, 3→+2
    result = result.clamp(-1, 1)

    return result[:numel]


# ---------------------------------------------------------------------------
# BitNetLinear — Drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------

class BitNetLinear(nn.Module):
    """
    Linear layer with BitNet 1.58-bit ternary weight quantization.

    During **training**:
        - Full-precision latent weights are maintained.
        - On each forward pass, weights are quantized to {-1, 0, +1}
          using absmean quantization.
        - The straight-through estimator (STE) allows gradients to flow
          through the non-differentiable quantization op.
        - Quantization can be gradually introduced via :class:`BitNetConfig`.

    During **inference** (``quantize_on_forward=False``):
        - Pre-quantized ternary weights are stored in int2 packed format,
          providing ~6x memory reduction.
        - The forward pass unpacks int2 weights, multiplies by the
          per-channel scale, and computes the linear transform.

    Memory savings:
        - FP32 weight:  32 bits/weight
        - BitNet int2:   2 bits/weight  → 16x reduction
        - Effective:    ~1.58 bits/weight (log2(3)) → ~6x vs FP32

    Args:
        in_features:    Size of each input sample.
        out_features:   Size of each output sample.
        bias:           If True, adds a learnable bias (default False).
        config:         BitNetConfig controlling quantization behaviour.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[BitNetConfig] = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.config = config or BitNetConfig()

        # ---- Full-precision latent weight (used during training) ----
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        # Kaiming-style init
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="linear")

        # ---- Per-channel scale (learned during training, fixed for inference) ----
        self.register_buffer(
            "scale",
            torch.ones(out_features, dtype=torch.float32)
        )

        # ---- Bias ----
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # ---- Int2 packed weight for inference mode ----
        # Created lazily when ``finalize()`` is called.
        self.register_buffer(
            "_packed_weight",
            torch.zeros(1, dtype=torch.int32)
        )
        self._packed_ready = False

        # ---- Training step counter ----
        self.register_buffer("_global_step", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------

    def _quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Quantize weight using absmean with straight-through estimator.

        The STE replaces the gradient of the round/clip operation with
        the identity (or a soft approximation), so gradients flow
        through the quantization boundary.

        Args:
            weight: Full-precision weight ``(out_features, in_features)``.

        Returns:
            Quantized weight (same shape), ternary * scale.
        """
        quant_ratio = self.config.get_quant_ratio(self._global_step.item())

        if quant_ratio <= 0.0:
            # No quantization yet
            return weight

        # Compute absmean scale per output channel
        abs_mean = weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)

        # Normalise
        w_norm = weight / abs_mean

        # Optional threshold for sparsity
        if self.config.threshold > 0.0:
            flat_abs = weight.abs().flatten()
            k = max(1, int(self.config.threshold / 100.0 * flat_abs.numel()))
            thresh_val = flat_abs.kthvalue(k).values
            zero_mask = weight.abs() < thresh_val
            w_norm = w_norm.masked_fill(zero_mask, 0.0)

        # ---- Straight-through estimator ----
        w_ternary = self._apply_ste(w_norm, quant_ratio)

        # Reconstruct: ternary * scale
        quantized = w_ternary * abs_mean.detach()  # scale detached for stability

        return quantized

    def _apply_ste(
        self, w_norm: torch.Tensor, quant_ratio: float
    ) -> torch.Tensor:
        """
        Apply straight-through estimator to the rounding operation.

        In the forward pass, values are rounded and clamped to {-1, 0, +1}.
        In the backward pass, the gradient passes through as if the
        operation were the identity (or a soft approximation).

        When ``quant_ratio < 1``, only a random fraction of elements
        are quantized, enabling gradual ramp-up.

        Args:
            w_norm:       Normalised weight ``(out, in)``.
            quant_ratio:  Fraction of elements to quantize.

        Returns:
            Ternary tensor with STE gradient.
        """
        # Hard quantization (forward)
        w_round = torch.round(w_norm).clamp(-1, 1)

        # STE: backward pass uses the original (pre-round) values
        if self.config.STE_mode == "atan":
            # Soft STE using atan approximation (better gradient flow)
            # atan(x) has bounded gradient, reducing gradient explosion
            w_ste = w_round + (w_norm - w_norm.detach()) * (1.0 / (1.0 + (w_norm * math.pi).pow(2)))
        else:
            # Identity STE (standard)
            w_ste = w_round + (w_norm - w_norm.detach())

        # Gradual: only quantize a fraction of weights
        if quant_ratio < 1.0:
            mask = (torch.rand_like(w_norm) < quant_ratio).float()
            # mask.detach() so gradient still flows to all weights
            w_ste = mask * w_ste + (1.0 - mask.detach()) * w_norm

        return w_ste

    # ------------------------------------------------------------------
    # Finalize: convert to inference mode with int2 packing
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """
        Finalize the layer for inference.

        Quantizes the full-precision weight to ternary values, packs them
        into int2 format, and marks the layer as ready for inference-only
        forward passes.  After calling this, the full-precision weight
        parameter is no longer needed for forward computation.

        This should be called after training is complete.
        """
        with torch.no_grad():
            abs_mean = self.weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)

            # Quantize to ternary
            w_norm = self.weight / abs_mean
            if self.config.threshold > 0.0:
                flat_abs = self.weight.abs().flatten()
                k = max(1, int(self.config.threshold / 100.0 * flat_abs.numel()))
                thresh_val = flat_abs.kthvalue(k).values
                zero_mask = self.weight.abs() < thresh_val
                w_norm = w_norm.masked_fill(zero_mask, 0.0)

            w_ternary = torch.round(w_norm).clamp(-1, 1).to(torch.int8)

            # Pack into int2
            self._packed_weight = pack_ternary_to_int2(w_ternary)

            # Store scale
            self.scale = abs_mean.squeeze(-1).clone()

            self._packed_ready = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Training mode: quantize weight on-the-fly with STE.
        Inference mode: use pre-packed int2 weights if available.

        Args:
            input: ``(batch, ..., in_features)``.

        Returns:
            ``(batch, ..., out_features)``.
        """
        if self.config.quantize_on_forward and self.training:
            # ---- Training: quantize with STE ----
            w = self._quantize_weight(self.weight)
            return F.linear(input, w, self.bias)

        elif self._packed_ready and not self.training:
            # ---- Inference: unpack int2 weights ----
            numel = self.out_features * self.in_features
            w_ternary = unpack_int2_to_ternary(
                self._packed_weight, numel
            ).to(input.dtype)
            w_ternary = w_ternary.view(self.out_features, self.in_features)
            w = w_ternary * self.scale.unsqueeze(-1)
            return F.linear(input, w, self.bias)

        else:
            # ---- Fallback: use full-precision weights ----
            return F.linear(input, self.weight, self.bias)

    # ------------------------------------------------------------------
    # Step counter
    # ------------------------------------------------------------------

    def increment_step(self, steps: int = 1) -> None:
        """Increment the global step counter (for gradual quantization)."""
        self._global_step.add_(steps)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def memory_footprint_bytes(self, inference: bool = False) -> int:
        """
        Return the memory footprint of the weight in bytes.

        Args:
            inference: If True, compute the int2 footprint; otherwise
                       the full-precision footprint.

        Returns:
            Number of bytes.
        """
        numel = self.out_features * self.in_features
        if inference and self._packed_ready:
            return self._packed_weight.numel() * 4 + self.scale.numel() * 4
        else:
            return numel * 4  # float32

    def compression_ratio(self) -> float:
        """Return the compression ratio vs FP32."""
        return self.memory_footprint_bytes(inference=False) / max(
            1, self.memory_footprint_bytes(inference=True)
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quant_step={self._global_step.item()}, "
            f"packed={self._packed_ready}"
        )


# ---------------------------------------------------------------------------
# BitNet weight conversion utilities
# ---------------------------------------------------------------------------

def convert_linear_to_bitnet(
    module: nn.Module,
    config: Optional[BitNetConfig] = None,
    exclude_names: Optional[List[str]] = None,
) -> nn.Module:
    """
    Recursively convert all nn.Linear layers in a model to BitNetLinear.

    This is a convenience function that walks the module tree and replaces
    every nn.Linear with a BitNetLinear of the same dimensions.

    Args:
        module:        Root module to convert.
        config:        BitNetConfig for the new layers.
        exclude_names: List of dotted names to skip (e.g. ["lm_head"]).

    Returns:
        The same module (modified in-place) with BitNetLinear layers.
    """
    config = config or BitNetConfig()
    exclude_names = exclude_names or []

    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        # Check exclusion
        if any(name.startswith(excl) for excl in exclude_names):
            continue

        # Create BitNetLinear replacement
        bitnet_layer = BitNetLinear(
            in_features=child.in_features,
            out_features=child.out_features,
            bias=child.bias is not None,
            config=config,
        )

        # Copy existing weights
        with torch.no_grad():
            bitnet_layer.weight.copy_(child.weight)
            if child.bias is not None:
                bitnet_layer.bias.copy_(child.bias)

        # Replace in parent module
        # Walk to the parent and set the attribute
        parts = name.split(".")
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], bitnet_layer)

    return module


def finalize_bitnet_model(module: nn.Module) -> None:
    """
    Finalize all BitNetLinear layers in a model for inference.

    Calls :meth:`BitNetLinear.finalize` on every BitNetLinear submodule.

    Args:
        module: Root module.
    """
    for child in module.modules():
        if isinstance(child, BitNetLinear):
            child.finalize()


def increment_bitnet_step(module: nn.Module, steps: int = 1) -> None:
    """
    Increment the global step counter on all BitNetLinear layers.

    Should be called once per training step.

    Args:
        module: Root module.
        steps:  Number of steps to increment.
    """
    for child in module.modules():
        if isinstance(child, BitNetLinear):
            child.increment_step(steps)


# ---------------------------------------------------------------------------
# BitNet-aware weight decay
# ---------------------------------------------------------------------------

def bitnet_weight_decay_loss(
    module: nn.Module,
    decay: float = 0.01,
) -> torch.Tensor:
    """
    Compute L2 regularisation on the latent (full-precision) weights of
    all BitNetLinear layers.

    Standard weight decay applied to the quantised weights has limited
    effect because the weights are discrete.  This function applies decay
    to the underlying continuous latent weights instead.

    Args:
        module: Root module containing BitNetLinear layers.
        decay:  Weight decay coefficient.

    Returns:
        Scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=next(module.parameters()).device)
    count = 0
    for child in module.modules():
        if isinstance(child, BitNetLinear):
            loss = loss + child.weight.pow(2).sum()
            count += 1
    if count > 0:
        loss = loss * (decay / count)
    return loss
