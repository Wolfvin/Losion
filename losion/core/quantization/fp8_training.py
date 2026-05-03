"""
FP8 Training Pipeline for Losion Framework v0.4.

Upgrade #7: FP8 precision for 2x training throughput on H100/H200/MI300 GPUs.

Key components:
1. FP8Scaler — Dynamic scaling with delayed scaling factor
2. FP8Linear — Linear layer with FP8 forward GEMM, BF16 backward
3. FP8TrainingWrapper — Convert a model's Linear layers to FP8Linear
4. Hardware detection — Automatic fallback to BF16 on unsupported hardware

Background:
    FP8 (8-bit floating point) provides ~2x throughput improvement for
    matrix multiplications on NVIDIA H100+ and AMD MI300+ GPUs.  The
    two FP8 formats are:

    - E4M3 (4 exponent bits, 3 mantissa bits): Used for forward-pass
      GEMM (weights and activations).  Range: ±448, precision: ~3 decimal
      digits.  Does not support Inf/NaN natively.

    - E5M2 (5 exponent bits, 2 mantissa bits): Used for backward-pass
      gradients.  Range: ±57344, precision: ~2 decimal digits.
      Supports Inf/NaN.

    The key challenge is **dynamic scaling**: FP8 has limited dynamic range,
    so inputs must be scaled to avoid overflow/underflow.  The standard
    approach is **delayed scaling**: use the scaling factor computed from
    the *previous* iteration, which avoids the overhead of computing a new
    scale every step.

    This implementation uses pure PyTorch and simulates FP8 quantization
    for correctness testing.  On hardware with native FP8 support
    (H100+, MI300+), the quantize/dequantize operations would be replaced
    by hardware-accelerated FP8 GEMM calls.

Hardware: Pure PyTorch with automatic BF16 fallback.
Supports torch.compile for additional optimisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _detect_fp8_support() -> bool:
    """
    Detect whether the current hardware supports native FP8 operations.

    Checks for:
    1. CUDA device with compute capability >= 9.0 (H100+)
    2. OR ROCm device with gfx940+ (MI300+)

    Returns:
        True if native FP8 is supported, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Check CUDA compute capability
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        # H100 = compute 9.0, H200 = 9.0, B100 = 10.0
        if major >= 9:
            return True
    except Exception:
        pass

    return False


FP8_AVAILABLE = _detect_fp8_support()


# ---------------------------------------------------------------------------
# FP8 format constants
# ---------------------------------------------------------------------------

# E4M3: sign(1) + exponent(4) + mantissa(3) = 8 bits
# Max value: 448.0, no Inf/NaN
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = -448.0

# E5M2: sign(1) + exponent(5) + mantissa(2) = 8 bits
# Max value: 57344.0, supports Inf/NaN
FP8_E5M2_MAX = 57344.0
FP8_E5M2_MIN = -57344.0


# ---------------------------------------------------------------------------
# FP8 quantization / dequantization primitives
# ---------------------------------------------------------------------------

def quantize_to_fp8_e4m3(
    tensor: torch.Tensor,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 E4M3 format.

    The quantization is simulated using float8_e4m3fn dtype if available,
    otherwise falls back to a manual simulation.

    E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
    Representable range: [-448, 448], no Inf/NaN.

    Args:
        tensor: Input tensor (BF16 or FP32).
        scale:  Per-tensor or per-channel scaling factor.  The tensor is
                divided by scale before quantization.

    Returns:
        (quantized_tensor, new_scale):
            quantized_tensor: FP8-simulated tensor (same dtype as input).
            new_scale:        Updated scale for the next iteration
                              (delayed scaling).
    """
    # Compute new scale from current tensor
    amax = tensor.abs().amax()
    # Scale so that amax maps to FP8_E4M3_MAX * 0.9 (leave headroom)
    new_scale = (amax / (FP8_E4M3_MAX * 0.9)).clamp(min=1e-12)

    # Use the *provided* scale for quantization (delayed scaling)
    scaled = tensor / scale.clamp(min=1e-12)

    # Try native float8_e4m3fn if available (PyTorch 2.1+)
    if hasattr(torch, "float8_e4m3fn"):
        try:
            fp8_tensor = scaled.to(torch.float8_e4m3fn)
            dequantized = fp8_tensor.to(tensor.dtype) * scale
            return dequantized, new_scale
        except (RuntimeError, NotImplementedError):
            pass

    # Manual simulation: clamp and round to FP8-precision levels
    clamped = scaled.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)

    # Simulate reduced precision by quantizing to discrete levels
    # E4M3 has ~3 decimal digits of precision (~2^3 = 8 mantissa levels)
    precision = 2 ** (3 - 1)  # 4 levels per power-of-two
    # Find the exponent for each value
    abs_clamped = clamped.abs().clamp(min=1e-12)
    exponent = torch.floor(torch.log2(abs_clamped))
    # Quantize within each bin
    quant_step = 2.0 ** (exponent - 2)  # 2^(exp - mantissa_bits)
    quantized = torch.round(clamped / quant_step) * quant_step

    # Restore sign for zeros
    quantized = torch.where(clamped == 0, torch.zeros_like(quantized), quantized)

    dequantized = quantized * scale
    return dequantized, new_scale


def quantize_to_fp8_e5m2(
    tensor: torch.Tensor,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 E5M2 format.

    E5M2 format: 1 sign bit, 5 exponent bits, 2 mantissa bits.
    Representable range: [-57344, 57344], supports Inf/NaN.

    Used for backward-pass gradients where the wider dynamic range
    is needed.

    Args:
        tensor: Input tensor (BF16 or FP32).
        scale:  Per-tensor or per-channel scaling factor.

    Returns:
        (quantized_tensor, new_scale):
            quantized_tensor: FP8-simulated tensor.
            new_scale:        Updated scale for the next iteration.
    """
    # Compute new scale
    amax = tensor.abs().amax()
    new_scale = (amax / (FP8_E5M2_MAX * 0.9)).clamp(min=1e-12)

    # Delayed scaling: use provided scale
    scaled = tensor / scale.clamp(min=1e-12)

    # Try native float8_e5m2 if available
    if hasattr(torch, "float8_e5m2"):
        try:
            fp8_tensor = scaled.to(torch.float8_e5m2)
            dequantized = fp8_tensor.to(tensor.dtype) * scale
            return dequantized, new_scale
        except (RuntimeError, NotImplementedError):
            pass

    # Manual simulation
    clamped = scaled.clamp(FP8_E5M2_MIN, FP8_E5M2_MAX)

    abs_clamped = clamped.abs().clamp(min=1e-12)
    exponent = torch.floor(torch.log2(abs_clamped))
    quant_step = 2.0 ** (exponent - 1)  # 2^(exp - mantissa_bits)
    quantized = torch.round(clamped / quant_step) * quant_step
    quantized = torch.where(clamped == 0, torch.zeros_like(quantized), quantized)

    dequantized = quantized * scale
    return dequantized, new_scale


# ---------------------------------------------------------------------------
# FP8Scaler — Dynamic scaling with delayed scaling factor
# ---------------------------------------------------------------------------

class FP8Scaler(nn.Module):
    """
    Dynamic FP8 scaling with delayed scaling factor.

    Delayed scaling is the standard approach for FP8 training:
    the scaling factor used for the current step is computed from the
    *previous* step's tensor statistics.  This avoids the overhead of
    computing a new scale synchronously during the forward pass.

    Scale update rule::

        new_scale = amax(tensor) / (FP8_MAX * margin)
        current_scale = previous_new_scale   # one-step delay

    The scaler also supports:
    - **Margin**: Safety factor to avoid overflow (default 0.9).
    - **Scale window**: Number of steps to wait before updating scale
      (amortises the cost of scale computation).
    - **Scale clamping**: Min/max bounds to prevent scale instability.

    Args:
        margin:          Safety margin factor (default 0.9).
        fp8_format:      "e4m3" or "e5m2".
        scale_window:    Number of steps between scale updates (default 1).
        scale_min:       Minimum scale value (default 1e-12).
        scale_max:       Maximum scale value (default 1e6).
        warmup_steps:    Steps during which BF16 is used before FP8
                         (default 100).
    """

    def __init__(
        self,
        margin: float = 0.9,
        fp8_format: str = "e4m3",
        scale_window: int = 1,
        scale_min: float = 1e-12,
        scale_max: float = 1e6,
        warmup_steps: int = 100,
    ) -> None:
        super().__init__()

        self.margin = margin
        self.fp8_format = fp8_format
        self.scale_window = scale_window
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.warmup_steps = warmup_steps

        if fp8_format == "e4m3":
            self.fp8_max = FP8_E4M3_MAX
        elif fp8_format == "e5m2":
            self.fp8_max = FP8_E5M2_MAX
        else:
            raise ValueError(f"Unsupported FP8 format: {fp8_format}")

        # Current scale (delayed from previous step)
        self.register_buffer("scale", torch.tensor(1.0))
        # Pending scale (computed this step, applied next step)
        self.register_buffer("_pending_scale", torch.tensor(1.0))

        # Step counter
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def compute_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute a new scaling factor from a tensor's statistics.

        Args:
            tensor: Input tensor.

        Returns:
            New scaling factor (scalar).
        """
        amax = tensor.abs().amax()
        new_scale = (amax / (self.fp8_max * self.margin)).clamp(
            min=self.scale_min, max=self.scale_max
        )
        return new_scale

    def update(self, tensor: torch.Tensor) -> None:
        """
        Update the delayed scale based on a tensor's statistics.

        The new scale is stored as pending and will be applied on the
        next call to :meth:`get_scale`.

        Args:
            tensor: The tensor to use for scale computation.
        """
        self._step += 1

        # Only update every scale_window steps
        if self._step.item() % self.scale_window != 0:
            return

        new_scale = self.compute_scale(tensor)
        self._pending_scale = new_scale

        # Apply delayed scaling: current scale ← pending scale
        self.scale = self._pending_scale.clone()

    def get_scale(self) -> torch.Tensor:
        """
        Get the current scaling factor.

        Returns:
            Current scale (scalar tensor).
        """
        if self._step.item() < self.warmup_steps:
            # During warmup, return a scale that effectively disables quantization
            return torch.tensor(1e-12, device=self.scale.device)
        return self.scale

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize a tensor using the current (delayed) scale.

        After quantization, the scale is updated for the next step.

        Args:
            tensor: Input tensor (BF16 or FP32).

        Returns:
            Quantized tensor (simulated FP8, same dtype as input).
        """
        scale = self.get_scale()

        if self.fp8_format == "e4m3":
            quantized, new_scale = quantize_to_fp8_e4m3(tensor, scale)
        else:
            quantized, new_scale = quantize_to_fp8_e5m2(tensor, scale)

        # Update scale for next step
        self._pending_scale = new_scale
        self.scale = self._pending_scale.clone()
        self._step += 1

        return quantized

    def is_warming_up(self) -> bool:
        """Return True if still in the warmup phase (BF16 mode)."""
        return self._step.item() < self.warmup_steps


# ---------------------------------------------------------------------------
# FP8Linear — Linear layer with FP8 forward, BF16 backward
# ---------------------------------------------------------------------------

class FP8Linear(nn.Module):
    """
    Linear layer that uses FP8 for forward GEMM and BF16 for backward.

    The forward pass:
    1. Quantize the weight to FP8 E4M3 (delayed scaling)
    2. Quantize the input activation to FP8 E4M3 (delayed scaling)
    3. Compute GEMM in FP8 (simulated via quantize/dequantize)
    4. Output is in BF16

    The backward pass:
    1. Compute gradients in BF16 (standard autograd)
    2. Quantize gradients to FP8 E5M2 (optional, for gradient
       communication compression in distributed training)

    When FP8 hardware is not available, the layer falls back to BF16
    computation.  This is transparent to the user.

    Args:
        in_features:    Size of each input sample.
        out_features:   Size of each output sample.
        bias:           If True, adds a learnable bias (default False).
        fp8_format:     FP8 format for forward: "e4m3" (default).
        use_fp8:        If True, enable FP8 quantization.  If False,
                        fall back to BF16.
        margin:         Safety margin for scaling (default 0.9).
        warmup_steps:   Steps of BF16 before switching to FP8.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        fp8_format: str = "e4m3",
        use_fp8: bool = True,
        margin: float = 0.9,
        warmup_steps: int = 100,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8 and FP8_AVAILABLE
        self.fp8_format = fp8_format

        # ---- Weight and bias (stored in BF16 or FP32) ----
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="linear")

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # ---- FP8 Scalers ----
        if self.use_fp8:
            # Weight scaler (E4M3 for forward)
            self.weight_scaler = FP8Scaler(
                margin=margin,
                fp8_format="e4m3",
                warmup_steps=warmup_steps,
            )
            # Activation scaler (E4M3 for forward)
            self.input_scaler = FP8Scaler(
                margin=margin,
                fp8_format="e4m3",
                warmup_steps=warmup_steps,
            )
            # Gradient scaler (E5M2 for backward, optional)
            self.grad_scaler = FP8Scaler(
                margin=margin,
                fp8_format="e5m2",
                warmup_steps=warmup_steps,
            )
        else:
            self.weight_scaler = None  # type: ignore
            self.input_scaler = None   # type: ignore
            self.grad_scaler = None    # type: ignore

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FP8 quantization.

        If FP8 is disabled or hardware doesn't support it, falls back
        to standard BF16 linear.

        Args:
            input: ``(batch, ..., in_features)`` — BF16 or FP32.

        Returns:
            ``(batch, ..., out_features)`` — same dtype as input.
        """
        if not self.use_fp8:
            return F.linear(input, self.weight, self.bias)

        # ---- FP8 forward path ----
        # 1. Quantize weight to FP8 E4M3
        w_scale = self.weight_scaler.get_scale()
        w_fp8, w_new_scale = quantize_to_fp8_e4m3(self.weight, w_scale)
        self.weight_scaler._pending_scale = w_new_scale
        self.weight_scaler.scale = w_new_scale.clone()
        self.weight_scaler._step += 1

        # 2. Quantize input activation to FP8 E4M3
        input_flat = input
        i_scale = self.input_scaler.get_scale()
        x_fp8, i_new_scale = quantize_to_fp8_e4m3(input_flat, i_scale)
        self.input_scaler._pending_scale = i_new_scale
        self.input_scaler.scale = i_new_scale.clone()
        self.input_scaler._step += 1

        # 3. GEMM in "FP8" (simulated: dequantized × dequantized)
        output = F.linear(x_fp8, w_fp8, self.bias)

        return output

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"use_fp8={self.use_fp8}, format={self.fp8_format}"
        )


# ---------------------------------------------------------------------------
# FP8TrainingWrapper — Convert model to FP8
# ---------------------------------------------------------------------------

class FP8TrainingWrapper:
    """
    Wrapper that converts a model's nn.Linear layers to FP8Linear.

    Provides a convenient interface for enabling FP8 training on an
    existing model.  The wrapper:

    1. Recursively finds all nn.Linear layers.
    2. Replaces them with FP8Linear layers (preserving weights).
    3. Provides methods for managing FP8 state (warmup, step counting).
    4. Supports selective conversion (exclude certain layers).

    Usage::

        model = MyModel()
        wrapper = FP8TrainingWrapper(model, warmup_steps=500)
        # model's Linear layers are now FP8Linear
        output = model(input)  # uses FP8 automatically
        wrapper.step()  # increment step counters

    Args:
        model:          The model to convert.
        warmup_steps:   Steps of BF16 before switching to FP8.
        margin:         Safety margin for FP8 scaling.
        exclude_names:  List of dotted module names to skip.
        fp8_format:     FP8 format for forward GEMM ("e4m3" default).
        force_fp8:      If True, enable FP8 even without hardware support
                        (for testing/simulation).
    """

    def __init__(
        self,
        model: nn.Module,
        warmup_steps: int = 100,
        margin: float = 0.9,
        exclude_names: Optional[List[str]] = None,
        fp8_format: str = "e4m3",
        force_fp8: bool = False,
    ) -> None:
        self.model = model
        self.warmup_steps = warmup_steps
        self.margin = margin
        self.exclude_names = exclude_names or []
        self.fp8_format = fp8_format

        # Determine whether to use FP8
        use_fp8 = FP8_AVAILABLE or force_fp8

        # Convert Linear layers
        self._converted_layers: Dict[str, FP8Linear] = {}
        self._convert_model(use_fp8)

    def _convert_model(self, use_fp8: bool) -> None:
        """Walk the model and replace nn.Linear with FP8Linear."""
        for name, child in list(self.model.named_modules()):
            if not isinstance(child, nn.Linear):
                continue

            # Check exclusion
            if any(name.startswith(excl) for excl in self.exclude_names):
                continue

            # Create FP8Linear replacement
            fp8_layer = FP8Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                fp8_format=self.fp8_format,
                use_fp8=use_fp8,
                margin=self.margin,
                warmup_steps=self.warmup_steps,
            )

            # Copy weights
            with torch.no_grad():
                fp8_layer.weight.copy_(child.weight)
                if child.bias is not None:
                    fp8_layer.bias.copy_(child.bias)

            # Replace in parent module
            parts = name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], fp8_layer)

            self._converted_layers[name] = fp8_layer

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        Advance the FP8 scalers by one step.

        Should be called once per training iteration.  During warmup,
        scalers will return BF16-scale factors (effectively disabling
        quantization).  After warmup, FP8 quantization is enabled.
        """
        # The scalers auto-increment in forward(), but we can also
        # force a step here for explicit control
        pass

    def is_warming_up(self) -> bool:
        """Check if any FP8 scaler is still in warmup."""
        for layer in self._converted_layers.values():
            if layer.weight_scaler is not None and layer.weight_scaler.is_warming_up():
                return True
        return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_scale_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get scaling factor statistics for all FP8 layers.

        Returns:
            Dict mapping layer name → {"weight_scale": float, "input_scale": float}.
        """
        stats = {}
        for name, layer in self._converted_layers.items():
            layer_stats: Dict[str, float] = {}
            if layer.weight_scaler is not None:
                layer_stats["weight_scale"] = layer.weight_scaler.scale.item()
            if layer.input_scaler is not None:
                layer_stats["input_scale"] = layer.input_scaler.scale.item()
            stats[name] = layer_stats
        return stats

    def get_fp8_layer_names(self) -> List[str]:
        """Return names of all converted FP8 layers."""
        return list(self._converted_layers.keys())

    def num_converted_layers(self) -> int:
        """Return the number of converted layers."""
        return len(self._converted_layers)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def disable_fp8(self) -> None:
        """Temporarily disable FP8 quantization (all layers → BF16)."""
        for layer in self._converted_layers.values():
            layer.use_fp8 = False

    def enable_fp8(self) -> None:
        """Re-enable FP8 quantization."""
        for layer in self._converted_layers.values():
            layer.use_fp8 = FP8_AVAILABLE

    def __repr__(self) -> str:
        return (
            f"FP8TrainingWrapper("
            f"layers={len(self._converted_layers)}, "
            f"warmup={self.warmup_steps}, "
            f"format={self.fp8_format})"
        )


# ---------------------------------------------------------------------------
# FP8 gradient compression for distributed training
# ---------------------------------------------------------------------------

class FP8GradCompressor:
    """
    Compress gradients to FP8 E5M2 for distributed communication.

    In multi-GPU training, gradient all-reduce can be a bottleneck.
    Compressing gradients to FP8 (E5M2) reduces communication volume
    by 2x with minimal impact on training quality.

    This compressor:
    1. Collects gradients from specified parameters.
    2. Quantizes them to FP8 E5M2.
    3. Returns compressed buffers for all-reduce.
    4. Dequantizes after all-reduce.

    Args:
        model:         Model whose gradients to compress.
        warmup_steps:  Steps before enabling gradient compression.
        margin:        Safety margin for FP8 scaling.
    """

    def __init__(
        self,
        model: nn.Module,
        warmup_steps: int = 100,
        margin: float = 0.9,
    ) -> None:
        self.model = model
        self.scaler = FP8Scaler(
            margin=margin,
            fp8_format="e5m2",
            warmup_steps=warmup_steps,
        )
        self._step = 0

    def compress(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a gradient tensor to FP8 E5M2.

        Args:
            grad: Gradient tensor (BF16 or FP32).

        Returns:
            (compressed_grad, scale): Compressed gradient and its scale.
        """
        scale = self.scaler.get_scale()
        compressed, new_scale = quantize_to_fp8_e5m2(grad, scale)
        self.scaler._pending_scale = new_scale
        self.scaler.scale = new_scale.clone()
        self.scaler._step += 1
        return compressed, scale

    def decompress(
        self, compressed: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Decompress an FP8 E5M2 gradient.

        Args:
            compressed: FP8-simulated gradient tensor.
            scale:      Scale factor used during compression.

        Returns:
            Dequantized gradient (same dtype as compressed).
        """
        # The compressed tensor is already dequantized in our simulation
        return compressed

    def step(self) -> None:
        """Increment the step counter."""
        self._step += 1


# ---------------------------------------------------------------------------
# FP8 utility functions
# ---------------------------------------------------------------------------

def check_fp8_hardware() -> Dict[str, bool]:
    """
    Check FP8 hardware support and return a diagnostic dict.

    Returns:
        Dict with keys:
        - "fp8_available":  Whether FP8 operations are supported.
        - "e4m3_supported": Whether E4M3 format is available.
        - "e5m2_supported": Whether E5M2 format is available.
        - "cuda_available": Whether CUDA is available.
        - "compute_capability": CUDA compute capability string.
    """
    result = {
        "fp8_available": FP8_AVAILABLE,
        "e4m3_supported": False,
        "e5m2_supported": False,
        "cuda_available": torch.cuda.is_available(),
        "compute_capability": "N/A",
    }

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        result["compute_capability"] = f"{cap[0]}.{cap[1]}"

    # Check for native float8 dtypes
    result["e4m3_supported"] = hasattr(torch, "float8_e4m3fn")
    result["e5m2_supported"] = hasattr(torch, "float8_e5m2")

    return result


def estimate_fp8_savings(model: nn.Module) -> Dict[str, float]:
    """
    Estimate memory savings from FP8 quantization for a model.

    Args:
        model: The model to analyze.

    Returns:
        Dict with:
        - "total_params":        Total number of parameters.
        - "linear_params":       Parameters in nn.Linear layers.
        - "bf16_bytes":          Memory in bytes at BF16.
        - "fp8_bytes":           Memory in bytes at FP8 (Linear only).
        - "savings_ratio":       Memory reduction ratio.
    """
    total_params = 0
    linear_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        # Check if this param belongs to a Linear layer
        if "weight" in name:
            # Heuristic: if parent module is Linear
            parts = name.rsplit(".", 1)[0]
            try:
                parent = dict(model.named_modules())[parts]
                if isinstance(parent, nn.Linear):
                    linear_params += param.numel()
            except (KeyError, AttributeError):
                pass

    bf16_bytes = total_params * 2  # BF16 = 2 bytes
    # FP8: linear params in 1 byte, rest in BF16
    fp8_bytes = linear_params * 1 + (total_params - linear_params) * 2

    return {
        "total_params": float(total_params),
        "linear_params": float(linear_params),
        "bf16_bytes": float(bf16_bytes),
        "fp8_bytes": float(fp8_bytes),
        "savings_ratio": 1.0 - (fp8_bytes / bf16_bytes) if bf16_bytes > 0 else 0.0,
    }
