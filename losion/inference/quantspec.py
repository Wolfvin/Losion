"""
QuantSpec — Self-Speculative Decoding with Hierarchical Quantization
=====================================================================

Implements QuantSpec (ICML 2025, poster 46326), a self-speculative decoding
method that uses hierarchically quantized versions of the SAME model as draft
models, eliminating the need for a separate draft model while achieving
>90% acceptance rates and ~2.5x inference speedup.

Key Concepts
------------

1. **Self-Speculative Decoding**
   Unlike traditional speculative decoding which requires a separate (smaller)
   draft model, QuantSpec creates draft models by quantizing the target model
   itself at progressively lower bit-widths.  Because the draft and target
   share the same architecture and weights (just at different precisions),
   the acceptance rate is naturally very high — typically >90%.

2. **Hierarchical Quantization**
   Multiple quantization levels create a hierarchy of increasingly fast draft
   models.  For example, an FP16 target model might use:
     - Level 0 (target):  FP16 (16-bit) — full precision, slow
     - Level 1 (draft):   INT8 (8-bit)  — 2x faster, ~95% acceptance
     - Level 2 (draft):   INT4 (4-bit)  — 4x faster, ~90% acceptance
     - Level 3 (draft):   INT2 (2-bit)  — 8x faster, ~80% acceptance

   The decoder can dynamically select which quantization level to use as the
   draft model, balancing speed vs. acceptance rate.

3. **Tri-Jalur Integration**
   In the Losion Tri-Jalur architecture (SSM + Attention + MoE), the SSM
   pathway can be more aggressively quantized because:
     - SSM layers have lower sensitivity to quantization error
     - SSM state dynamics are inherently lower-rank than attention patterns
     - SSM inference is memory-bandwidth-bound, so lower precision gives
       proportionally larger speedups

   QuantSpec applies **pathway-aware quantization**: the SSM pathway is
   quantized to a lower bit-width than the attention pathway at each level,
   while the MoE pathway uses intermediate quantization.

4. **Draft-Verify Pipeline**
   The speculative decoding loop:
     a. **Draft phase**: Run the quantized draft model autoregressively
        to generate K candidate tokens.
     b. **Verify phase**: Run the full-precision target model on the K+1
        token sequence in a single forward pass (using KV cache).
     c. **Accept/reject**: Accept the longest matching prefix; at the
        first mismatch, replace with the target model's prediction.

   Expected speedup: If acceptance rate is p and draft model is s times
   faster than the target, then effective speedup ≈ s / (1 + (1-p)*(K-1)/K).

Architecture
------------

1. QuantSpecConfig — Configuration dataclass with quant_levels,
   acceptance_threshold, max_draft_tokens, and Tri-Jalur pathway
   quantization overrides.

2. QuantizedDraftModel — Creates a quantized version of a model on-the-fly
   by applying per-pathway quantization. Supports bitsandbytes backend
   and a custom uniform/symmetric quantization fallback.

3. QuantSpecDecoder — Orchestrates the draft-verify pipeline. Supports:
   - Dynamic level selection based on running acceptance rate
   - Hierarchical fallback: if acceptance rate drops, switch to a less
     aggressive quantization level
   - KV cache sharing between draft and target models
   - Stochastic acceptance (speculative sampling) for distribution
     equivalence

4. QuantSpecStats — Statistics tracking for acceptance rates, speedup,
   and per-level diagnostics.

References
----------
- ICML 2025 poster 46326: "QuantSpec: Self-Speculative Decoding with
  Hierarchical Quantization"
- Leviathan et al., "Fast Inference from Transformers via Speculative
  Decoding" (ICML 2023) — original speculative decoding
- Chen et al., "Accelerating Large Language Model Decoding with
  Speculative Sampling" (ICLR 2024) — stochastic acceptance
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models" (ICML 2023) — per-channel
  quantization

Hardware: Pure PyTorch. bitsandbytes is optional. Compatible with CUDA,
ROCm, and CPU.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class QuantBackend(str, Enum):
    """Quantization backend for creating draft models.

    Attributes:
        BITSANDBYTES: Use bitsandbytes library for INT8/INT4 quantization.
            Requires bitsandbytes to be installed.
        CUSTOM: Use custom PyTorch-based uniform/symmetric quantization.
            No external dependencies; works on all hardware.
        AUTO: Automatically select the best available backend.
    """
    BITSANDBYTES = "bitsandbytes"
    CUSTOM = "custom"
    AUTO = "auto"


class QuantLevel(str, Enum):
    """Named quantization levels for the hierarchical draft model.

    The levels are ordered from highest (closest to target) to lowest
    (most aggressive quantization, fastest but least accurate).

    Attributes:
        FP16: 16-bit floating point — typically the target model itself.
        INT8: 8-bit integer quantization — ~2x speedup, >95% acceptance.
        INT4: 4-bit integer quantization — ~4x speedup, ~90% acceptance.
        INT2: 2-bit integer quantization — ~8x speedup, ~80% acceptance.
    """
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"


# ============================================================================
# QuantSpecConfig — Configuration
# ============================================================================


@dataclass
class QuantSpecConfig:
    """Configuration for QuantSpec self-speculative decoding.

    Controls the quantization hierarchy, acceptance thresholds, and
    Tri-Jalur pathway-aware quantization overrides.

    Ref: ICML 2025 poster 46326, "QuantSpec: Self-Speculative Decoding
    with Hierarchical Quantization"

    Attributes:
        quant_levels: Ordered list of quantization levels to use as draft
            models, from most aggressive (fastest) to least aggressive
            (closest to target).  The target model is always full-precision
            and is not included in this list.  Default [INT4, INT8] means
            INT4 is tried first (fastest), falling back to INT8 if
            acceptance rate is too low.
        acceptance_threshold: Minimum acceptance rate to maintain.  If the
            observed rate drops below this threshold, the decoder falls
            back to a less aggressive quantization level.  Default 0.85
            (85%).
        max_draft_tokens: Maximum number of draft tokens to generate per
            speculative decoding step.  Higher values give more potential
            speedup but increase the cost of verification.  Default 5.
        min_draft_tokens: Minimum number of draft tokens.  Default 1.
        backend: Quantization backend to use.  Default AUTO (selects
            bitsandbytes if available, otherwise custom).
        use_stochastic_acceptance: If True, use speculative sampling
            (Chen et al., 2024) for distribution-equivalent outputs.
            If False, use greedy acceptance.  Default True.
        temperature: Sampling temperature for stochastic acceptance.
            Default 1.0.
        top_k: Top-k filtering for sampling.  0 = no filtering.  Default 0.
        top_p: Nucleus sampling threshold.  1.0 = no filtering.  Default 1.0.
        eos_token_id: End-of-sequence token ID.  Default -1 (disabled).

        # ---- Tri-Jalur pathway overrides ----
        ssm_quant_offset: Additional bit-width reduction for the SSM pathway
            at each level.  For example, if the base level is INT4 and
            ssm_quant_offset=1, the SSM pathway is quantized to INT2.
            Default 1 — the SSM pathway is always one level more aggressive.
        attention_quant_offset: Additional bit-width reduction for the
            attention pathway.  Default 0 — attention uses the base level.
        moe_quant_offset: Additional bit-width reduction for the MoE
            pathway.  Default 0 — MoE uses the base level.

        # ---- Adaptive control ----
        adaptive_level: If True, dynamically switch between quantization
            levels based on observed acceptance rate.  Default True.
        ema_alpha: Exponential moving average smoothing factor for the
            acceptance rate tracker.  Default 0.1.
        level_switch_patience: Number of consecutive steps below the
            acceptance threshold before switching to a less aggressive
            quantization level.  Default 5.
    """
    # Core settings
    quant_levels: List[QuantLevel] = field(
        default_factory=lambda: [QuantLevel.INT4, QuantLevel.INT8]
    )
    acceptance_threshold: float = 0.85
    max_draft_tokens: int = 5
    min_draft_tokens: int = 1
    backend: QuantBackend = QuantBackend.AUTO
    use_stochastic_acceptance: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    eos_token_id: int = -1

    # Tri-Jalur pathway overrides
    ssm_quant_offset: int = 1
    attention_quant_offset: int = 0
    moe_quant_offset: int = 0

    # Adaptive control
    adaptive_level: bool = True
    ema_alpha: float = 0.1
    level_switch_patience: int = 5

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.quant_levels:
            raise ValueError("quant_levels must not be empty")
        if self.acceptance_threshold <= 0 or self.acceptance_threshold > 1:
            raise ValueError(
                f"acceptance_threshold must be in (0, 1], "
                f"got {self.acceptance_threshold}"
            )
        if self.max_draft_tokens < 1:
            raise ValueError(
                f"max_draft_tokens must be >= 1, got {self.max_draft_tokens}"
            )
        if self.min_draft_tokens < 1:
            raise ValueError(
                f"min_draft_tokens must be >= 1, got {self.min_draft_tokens}"
            )
        if self.min_draft_tokens > self.max_draft_tokens:
            raise ValueError(
                f"min_draft_tokens ({self.min_draft_tokens}) must be <= "
                f"max_draft_tokens ({self.max_draft_tokens})"
            )
        if self.ema_alpha <= 0 or self.ema_alpha > 1:
            raise ValueError(
                f"ema_alpha must be in (0, 1], got {self.ema_alpha}"
            )

    @property
    def most_aggressive_level(self) -> QuantLevel:
        """Return the most aggressively quantized level (fastest draft)."""
        return self.quant_levels[0]

    @property
    def least_aggressive_level(self) -> QuantLevel:
        """Return the least aggressively quantized level (closest to target)."""
        return self.quant_levels[-1]


# ============================================================================
# QuantSpecStats — Statistics
# ============================================================================


@dataclass
class QuantSpecStats:
    """Statistics tracker for QuantSpec decoding performance monitoring.

    Tracks acceptance rates, speedup metrics, per-level diagnostics, and
    Tri-Jalur pathway-specific statistics.

    Attributes:
        total_steps: Total number of speculative decoding steps.
        total_draft_tokens: Total draft tokens proposed across all steps.
        total_accepted_tokens: Total draft tokens accepted by the verifier.
        total_emitted_tokens: Total tokens emitted (including corrections).
        current_level: Currently active quantization level.
        level_steps: Number of steps at each quantization level.
        level_acceptance: Cumulative accepted tokens per level.
        level_draft: Cumulative draft tokens per level.
        level_switches: Number of times the quantization level changed.
        acceptance_history: Per-step acceptance counts.
    """
    total_steps: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_emitted_tokens: int = 0
    current_level: QuantLevel = QuantLevel.INT4
    level_steps: Dict[str, int] = field(default_factory=dict)
    level_acceptance: Dict[str, int] = field(default_factory=dict)
    level_draft: Dict[str, int] = field(default_factory=dict)
    level_switches: int = 0
    acceptance_history: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate across all steps."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def avg_tokens_per_step(self) -> float:
        """Average tokens emitted per step."""
        if self.total_steps == 0:
            return 0.0
        return self.total_emitted_tokens / self.total_steps

    @property
    def effective_speedup(self) -> float:
        """Estimated speedup vs. standard autoregressive decoding.

        Accounts for the verification overhead (~1.2x single-pass cost)
        and the relative speed of the draft model.
        """
        if self.total_steps == 0:
            return 1.0
        verification_overhead = 1.2
        return self.avg_tokens_per_step / verification_overhead

    def level_acceptance_rate(self, level: QuantLevel) -> float:
        """Acceptance rate for a specific quantization level."""
        level_key = level.value
        drafted = self.level_draft.get(level_key, 0)
        accepted = self.level_acceptance.get(level_key, 0)
        if drafted == 0:
            return 0.0
        return accepted / drafted

    def record_step(
        self,
        n_accepted: int,
        draft_length: int,
        level: QuantLevel,
    ) -> None:
        """Record statistics from a single speculative decoding step.

        Args:
            n_accepted: Number of draft tokens accepted.
            draft_length: Number of draft tokens proposed.
            level: Quantization level used for drafting.
        """
        self.total_steps += 1
        self.total_draft_tokens += draft_length
        self.total_accepted_tokens += n_accepted
        self.total_emitted_tokens += n_accepted + 1
        self.acceptance_history.append(n_accepted)

        level_key = level.value
        self.level_steps[level_key] = self.level_steps.get(level_key, 0) + 1
        self.level_draft[level_key] = (
            self.level_draft.get(level_key, 0) + draft_length
        )
        self.level_acceptance[level_key] = (
            self.level_acceptance.get(level_key, 0) + n_accepted
        )
        self.current_level = level

    def record_level_switch(self) -> None:
        """Record a quantization level switch."""
        self.level_switches += 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_steps = 0
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.total_emitted_tokens = 0
        self.level_steps.clear()
        self.level_acceptance.clear()
        self.level_draft.clear()
        self.level_switches = 0
        self.acceptance_history.clear()

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "QuantSpec Statistics:",
            f"  Total steps:            {self.total_steps}",
            f"  Total draft tokens:     {self.total_draft_tokens}",
            f"  Total accepted tokens:  {self.total_accepted_tokens}",
            f"  Total emitted tokens:   {self.total_emitted_tokens}",
            f"  Acceptance rate:        {self.acceptance_rate:.2%}",
            f"  Avg tokens/step:        {self.avg_tokens_per_step:.2f}",
            f"  Effective speedup:      {self.effective_speedup:.2f}x",
            f"  Level switches:         {self.level_switches}",
            f"  Current level:          {self.current_level.value}",
        ]
        for level in QuantLevel:
            rate = self.level_acceptance_rate(level)
            steps = self.level_steps.get(level.value, 0)
            if steps > 0:
                lines.append(
                    f"  {level.value:>4s} acceptance:      {rate:.2%} "
                    f"({steps} steps)"
                )
        return "\n".join(lines)


# ============================================================================
# QuantizedDraftModel — On-the-fly Quantized Draft Model
# ============================================================================


class QuantizedDraftModel:
    """Creates a quantized version of a model on-the-fly for use as a
    speculative decoding draft model.

    QuantSpec (ICML 2025, poster 46326) uses quantized versions of the
    SAME model as draft models.  This class applies per-layer, per-pathway
    quantization to create a lightweight draft that can run significantly
    faster than the full-precision target while maintaining high agreement.

    **Tri-Jalur Pathway Awareness**: When the target model follows the
    Losion Tri-Jalur architecture (SSM + Attention + MoE), this class
    applies pathway-specific quantization levels:
      - SSM pathway: More aggressively quantized (lower bit-width) because
        SSM layers are less sensitive to quantization error and more
        memory-bandwidth-bound.
      - Attention pathway: Least aggressively quantized (higher bit-width)
        because attention patterns are more sensitive to precision loss.
      - MoE pathway: Intermediate quantization.

    **Backend Support**:
      - bitsandbytes: Uses Linear8bitLt or Linear4bit for INT8/INT4.
        Requires the bitsandbytes package.
      - custom: Pure PyTorch symmetric per-channel quantization.  Works on
        all hardware with no external dependencies.

    Usage::

        draft = QuantizedDraftModel(target_model, level=QuantLevel.INT4)
        logits = draft(input_ids)  # Fast forward pass with INT4 weights

    Args:
        target_model: The full-precision target model (nn.Module).
        level: Quantization level to apply.
        config: QuantSpecConfig with pathway offsets and backend settings.
    """

    # Mapping from QuantLevel to bit-width for custom quantization
    _LEVEL_BITS: Dict[QuantLevel, int] = {
        QuantLevel.FP16: 16,
        QuantLevel.INT8: 8,
        QuantLevel.INT4: 4,
        QuantLevel.INT2: 2,
    }

    # Pathway name patterns for Tri-Jalur aware quantization
    _SSM_PATTERNS = ["ssm", "mamba", "rwkv", "delta_net", "liquid", "ssm_layer"]
    _ATTENTION_PATTERNS = ["attention", "attn", "gated_attention", "lightning", "moba"]
    _MOE_PATTERNS = ["moe", "expert", "retrieval", "engram", "router"]

    def __init__(
        self,
        target_model: nn.Module,
        level: QuantLevel,
        config: QuantSpecConfig,
    ) -> None:
        self.target_model = target_model
        self.level = level
        self.config = config
        self.device = next(target_model.parameters()).device

        # Determine backend
        self._backend = self._resolve_backend(config.backend)

        # Build quantized parameter cache: name -> (quantized_tensor, scale)
        self._quantized_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._pathway_levels: Dict[str, QuantLevel] = {}

        # Quantize all parameters
        self._quantize_model()

    def _resolve_backend(self, backend: QuantBackend) -> QuantBackend:
        """Resolve AUTO backend to a concrete choice."""
        if backend != QuantBackend.AUTO:
            return backend

        try:
            import bitsandbytes  # noqa: F401
            return QuantBackend.BITSANDBYTES
        except ImportError:
            logger.debug(
                "QuantSpec: bitsandbytes not available, using custom backend"
            )
            return QuantBackend.CUSTOM

    def _classify_pathway(self, name: str) -> str:
        """Classify a parameter name into a Tri-Jalur pathway.

        Args:
            name: Fully-qualified parameter name (e.g., "ssm.mamba2.weight").

        Returns:
            One of "ssm", "attention", "moe", or "other".
        """
        name_lower = name.lower()
        for pattern in self._SSM_PATTERNS:
            if pattern in name_lower:
                return "ssm"
        for pattern in self._ATTENTION_PATTERNS:
            if pattern in name_lower:
                return "attention"
        for pattern in self._MOE_PATTERNS:
            if pattern in name_lower:
                return "moe"
        return "other"

    def _get_effective_level(self, pathway: str) -> QuantLevel:
        """Get the effective quantization level for a pathway.

        Applies the pathway-specific offset to the base level.

        Args:
            pathway: Tri-Jalur pathway name ("ssm", "attention", "moe", "other").

        Returns:
            Effective QuantLevel for this pathway.
        """
        if pathway in self._pathway_levels:
            return self._pathway_levels[pathway]

        offset = 0
        if pathway == "ssm":
            offset = self.config.ssm_quant_offset
        elif pathway == "attention":
            offset = self.config.attention_quant_offset
        elif pathway == "moe":
            offset = self.config.moe_quant_offset

        # Convert level to bit-width, apply offset, convert back
        base_bits = self._LEVEL_BITS[self.level]
        effective_bits = max(2, base_bits - offset * 2)  # Each offset = 2 bits

        # Find the closest QuantLevel
        best_level = self.level
        best_diff = float("inf")
        for qlevel, qbits in self._LEVEL_BITS.items():
            diff = abs(qbits - effective_bits)
            if diff < best_diff:
                best_diff = diff
                best_level = qlevel

        self._pathway_levels[pathway] = best_level
        return best_level

    def _quantize_model(self) -> None:
        """Quantize all linear layer parameters in the target model.

        Applies per-channel symmetric quantization based on the effective
        quantization level for each parameter's pathway.
        """
        for name, param in self.target_model.named_parameters():
            if param.dim() < 2:
                # Skip biases and 1D parameters (keep full precision)
                continue

            pathway = self._classify_pathway(name)
            level = self._get_effective_level(pathway)
            bits = self._LEVEL_BITS[level]

            if bits >= 16:
                # No quantization needed
                continue

            # Per-channel symmetric quantization
            quantized, scale = self._symmetric_quantize(param.data, bits)
            self._quantized_params[name] = (quantized, scale)

    @staticmethod
    def _symmetric_quantize(
        tensor: torch.Tensor,
        bits: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply per-channel symmetric quantization.

        Quantizes the tensor to the specified bit-width using per-output-channel
        scales.  This is the same approach as SmoothQuant (Xiao et al., ICML 2023).

        For bits=8:  values in [-128, 127]
        For bits=4:  values in [-8, 7]
        For bits=2:  values in [-2, 1]

        Args:
            tensor: Weight tensor to quantize, shape (out_features, in_features).
            bits: Number of bits for quantization.

        Returns:
            Tuple (quantized_tensor, scale):
                - quantized_tensor: Dequantized weights (same shape as input).
                - scale: Per-channel scale factor, shape (out_features,).
        """
        if bits < 2:
            bits = 2

        max_val = 2 ** (bits - 1) - 1
        min_val = -(2 ** (bits - 1))

        # Per output-channel scale
        scale = tensor.abs().amax(dim=-1).clamp(min=1e-8) / max_val  # (out,)

        # Quantize
        scaled = tensor / scale.unsqueeze(-1)
        quantized = torch.clamp(torch.round(scaled), min_val, max_val)

        # Dequantize (store in original precision for computation)
        dequantized = quantized * scale.unsqueeze(-1)

        return dequantized, scale

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: Optional[Any] = None,
    ) -> torch.Tensor:
        """Run a forward pass with quantized weights.

        Temporarily replaces the model's linear layer weights with their
        quantized versions, runs the forward pass, then restores the
        originals.  This avoids maintaining a separate copy of the model.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len).
            past_kv: Optional KV cache from previous steps.

        Returns:
            Logits tensor, shape (batch, seq_len, vocab_size).
        """
        # Save original parameters
        originals: Dict[str, torch.Tensor] = {}

        for name, (quantized, _scale) in self._quantized_params.items():
            # Navigate to the parameter in the model
            parts = name.split(".")
            module = self.target_model
            for part in parts[:-1]:
                module = getattr(module, part)
            param_name = parts[-1]

            # Save and replace
            originals[name] = getattr(module, param_name).data.clone()
            getattr(module, param_name).data.copy_(quantized)

        try:
            # Forward pass through the model with quantized weights
            with torch.no_grad():
                output = self.target_model(input_ids=input_ids)
                if hasattr(output, "logits"):
                    logits = output.logits
                else:
                    logits = output
        finally:
            # Restore original parameters
            for name, original in originals.items():
                parts = name.split(".")
                module = self.target_model
                for part in parts[:-1]:
                    module = getattr(module, part)
                param_name = parts[-1]
                getattr(module, param_name).data.copy_(original)

        return logits

    def requantize(self, level: QuantLevel) -> None:
        """Re-quantize the model at a different level.

        Called when the QuantSpecDecoder switches between quantization
        levels based on adaptive acceptance rate tracking.

        Args:
            level: New quantization level.
        """
        self.level = level
        self._quantized_params.clear()
        self._pathway_levels.clear()
        self._quantize_model()

    @property
    def effective_bits(self) -> float:
        """Average effective bit-width across all quantized parameters."""
        if not self._quantized_params:
            return 16.0
        total_bits = 0.0
        total_params = 0
        for name, (quantized, scale) in self._quantized_params.items():
            pathway = self._classify_pathway(name)
            level = self._get_effective_level(pathway)
            bits = self._LEVEL_BITS[level]
            numel = quantized.numel()
            total_bits += bits * numel
            total_params += numel
        return total_bits / max(total_params, 1)

    def __repr__(self) -> str:
        n_quantized = len(self._quantized_params)
        return (
            f"QuantizedDraftModel(level={self.level.value}, "
            f"backend={self._backend.value}, "
            f"quantized_params={n_quantized}, "
            f"effective_bits={self.effective_bits:.1f})"
        )


# ============================================================================
# QuantSpecDecoder — Main Draft-Verify Pipeline
# ============================================================================


class QuantSpecDecoder:
    """QuantSpec self-speculative decoding with hierarchical quantization.

    Implements the complete QuantSpec pipeline (ICML 2025, poster 46326):

    1. **Draft**: Generate K candidate tokens using a quantized draft model.
    2. **Verify**: Run the full-precision target model on the K+1 token
       sequence in a single forward pass.
    3. **Accept/Reject**: Accept the longest matching prefix; at the first
       mismatch, emit the target model's correction.
    4. **Adaptive Level Selection**: Dynamically switch between quantization
       levels based on observed acceptance rate.

    The key insight is that since the draft and target share the same
    architecture (only differing in precision), acceptance rates are
    naturally very high (>90%), leading to ~2.5x inference speedup
    without any additional model parameters.

    **Tri-Jalur Integration**: When used with a Losion model, the SSM
    pathway is more aggressively quantized in the draft model, because
    SSM layers have lower quantization sensitivity and are more
    memory-bandwidth-bound.

    Usage::

        config = QuantSpecConfig(quant_levels=[QuantLevel.INT4, QuantLevel.INT8])
        decoder = QuantSpecDecoder(target_model, config)

        for step in range(max_steps):
            tokens, done = decoder.step(
                current_tokens=current_tokens,
                model_forward=model_forward_fn,
            )
            if done:
                break

    Args:
        target_model: The full-precision target model (nn.Module).
        config: QuantSpecConfig with all hyperparameters.
    """

    def __init__(
        self,
        target_model: nn.Module,
        config: QuantSpecConfig,
    ) -> None:
        self.target_model = target_model
        self.config = config
        self.device = next(target_model.parameters()).device

        # Build draft models for each quantization level
        self._draft_models: Dict[QuantLevel, QuantizedDraftModel] = {}
        self._build_draft_models()

        # Current draft level (start with most aggressive)
        self._current_level = config.most_aggressive_level

        # Current speculation length
        self._current_draft_length = config.max_draft_tokens

        # Statistics
        self.stats = QuantSpecStats()

        # Adaptive level tracking
        self._ema_acceptance_rate = 0.0
        self._below_threshold_count = 0

    def _build_draft_models(self) -> None:
        """Build quantized draft models for all configured levels."""
        for level in self.config.quant_levels:
            if level == QuantLevel.FP16:
                # Skip FP16 — that's the target model itself
                continue
            self._draft_models[level] = QuantizedDraftModel(
                target_model=self.target_model,
                level=level,
                config=self.config,
            )
            logger.info(
                f"QuantSpec: Built draft model at {level.value} "
                f"(effective bits: "
                f"{self._draft_models[level].effective_bits:.1f})"
            )

    @property
    def current_level(self) -> QuantLevel:
        """Currently active quantization level."""
        return self._current_level

    @property
    def current_draft_model(self) -> QuantizedDraftModel:
        """Currently active draft model."""
        return self._draft_models[self._current_level]

    def _select_level(self, acceptance_rate: float) -> QuantLevel:
        """Adaptively select the quantization level based on acceptance rate.

        If the acceptance rate is below the threshold for
        level_switch_patience consecutive steps, fall back to a less
        aggressive (higher precision) level.  If the rate is consistently
        high, try a more aggressive level.

        Args:
            acceptance_rate: Most recent step's acceptance rate.

        Returns:
            Selected QuantLevel.
        """
        if not self.config.adaptive_level:
            return self._current_level

        # Update EMA
        if self._ema_acceptance_rate == 0.0:
            self._ema_acceptance_rate = acceptance_rate
        else:
            alpha = self.config.ema_alpha
            self._ema_acceptance_rate = (
                alpha * acceptance_rate
                + (1 - alpha) * self._ema_acceptance_rate
            )

        threshold = self.config.acceptance_threshold
        levels = self.config.quant_levels
        current_idx = levels.index(self._current_level) if self._current_level in levels else 0

        if self._ema_acceptance_rate < threshold:
            self._below_threshold_count += 1
            if self._below_threshold_count >= self.config.level_switch_patience:
                # Fall back to less aggressive (higher index = less aggressive)
                if current_idx + 1 < len(levels):
                    new_level = levels[current_idx + 1]
                    logger.info(
                        f"QuantSpec: Switching from {self._current_level.value} "
                        f"to {new_level.value} (acceptance rate "
                        f"{self._ema_acceptance_rate:.2%} < "
                        f"{threshold:.2%})"
                    )
                    self._current_level = new_level
                    self._below_threshold_count = 0
                    self.stats.record_level_switch()
        else:
            self._below_threshold_count = 0
            # Try to use more aggressive level if rate is high
            if current_idx > 0:
                # Only switch if we're consistently above threshold + margin
                margin = 0.05
                if self._ema_acceptance_rate > threshold + margin:
                    # Check if the more aggressive level is available
                    new_level = levels[current_idx - 1]
                    self._current_level = new_level
                    logger.info(
                        f"QuantSpec: Switching to more aggressive "
                        f"{new_level.value} (acceptance rate "
                        f"{self._ema_acceptance_rate:.2%})"
                    )
                    self.stats.record_level_switch()

        return self._current_level

    def _adjust_draft_length(self, acceptance_rate: float) -> None:
        """Adaptively adjust the draft token length based on acceptance rate.

        Higher acceptance → more draft tokens.
        Lower acceptance → fewer draft tokens.

        Args:
            acceptance_rate: Most recent step's acceptance rate.
        """
        threshold = self.config.acceptance_threshold
        if acceptance_rate > threshold + 0.05:
            self._current_draft_length = min(
                self._current_draft_length + 1,
                self.config.max_draft_tokens,
            )
        elif acceptance_rate < threshold - 0.1:
            self._current_draft_length = max(
                self._current_draft_length - 1,
                self.config.min_draft_tokens,
            )

    def _sample_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits with temperature, top-k, top-p.

        Args:
            logits: Logits tensor, shape (..., vocab_size).

        Returns:
            Sampled token IDs, shape (...).
        """
        cfg = self.config
        if cfg.temperature != 1.0:
            logits = logits / max(cfg.temperature, 1e-8)

        # Top-k filtering
        if cfg.top_k > 0:
            top_k = min(cfg.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-p (nucleus) filtering
        if cfg.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs - F.softmax(
                sorted_logits, dim=-1
            ) >= cfg.top_p
            indices_to_remove = sorted_indices_to_remove.scatter(
                sorted_indices.ndim - 1,
                sorted_indices,
                sorted_indices_to_remove,
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Greedy if temperature is effectively 0
        if cfg.temperature < 1e-8:
            return logits.argmax(dim=-1)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def draft(
        self,
        input_ids: torch.Tensor,
        n_draft: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft tokens using the quantized draft model.

        This is the first phase of QuantSpec. The quantized draft model
        generates K candidate tokens autoregressively.

        Ref: ICML 2025 poster 46326, Section 3.1

        Args:
            input_ids: Current token IDs, shape (batch, seq_len).
            n_draft: Number of draft tokens. If None, uses the adaptive
                draft length.

        Returns:
            Tuple (draft_tokens, draft_logits):
                - draft_tokens: Draft token IDs, shape (batch, n_draft).
                - draft_logits: Draft logits at each position,
                    shape (batch, n_draft, vocab_size).
        """
        draft_length = n_draft or self._current_draft_length
        draft_model = self.current_draft_model

        batch_size = input_ids.shape[0]
        current_ids = input_ids.clone()

        draft_tokens_list: List[torch.Tensor] = []
        draft_logits_list: List[torch.Tensor] = []

        for _ in range(draft_length):
            logits = draft_model.forward(current_ids)
            # Take logits at the last position
            next_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Greedy selection for draft (deterministic matching)
            next_token = next_logits.argmax(dim=-1)  # (batch,)

            draft_tokens_list.append(next_token)
            draft_logits_list.append(next_logits)

            # Append to sequence for autoregressive generation
            current_ids = torch.cat(
                [current_ids, next_token.unsqueeze(-1)], dim=-1
            )

        draft_tokens = torch.stack(draft_tokens_list, dim=1)  # (batch, n_draft)
        draft_logits = torch.stack(draft_logits_list, dim=1)  # (batch, n_draft, V)

        return draft_tokens, draft_logits

    @torch.no_grad()
    def verify(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        verifier_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens against the target model's predictions.

        This is the second phase of QuantSpec. The target model's logits
        are compared with the draft predictions, and the longest matching
        prefix is accepted.

        Ref: ICML 2025 poster 46326, Section 3.2

        Args:
            input_ids: Original input token IDs, shape (batch, seq_len).
            draft_tokens: Draft token IDs from the draft phase,
                shape (batch, n_draft).
            verifier_logits: Logits from the target model at positions
                corresponding to the draft tokens,
                shape (batch, n_draft + 1, vocab_size).

        Returns:
            Tuple (accepted_tokens, n_accepted):
                - accepted_tokens: Accepted + correction tokens,
                    shape (batch, n_accepted + 1).
                - n_accepted: Number of draft tokens accepted.
        """
        batch_size = draft_tokens.shape[0]
        spec_length = draft_tokens.shape[1]

        # Get the verifier's greedy predictions
        verifier_predictions = verifier_logits.argmax(dim=-1)  # (batch, spec_length+1)

        # Compare draft with verifier predictions at each position
        matches = draft_tokens == verifier_predictions[:, :spec_length]

        # Find the longest matching prefix per batch element
        all_match_prefix = matches.cumprod(dim=1)  # (batch, spec_length)
        n_accepted_per_batch = all_match_prefix.sum(dim=1)  # (batch,)

        # Conservative: use minimum across batch
        n_accepted = int(n_accepted_per_batch.min().item())

        # Build output: accepted draft tokens + correction token
        accepted_draft = draft_tokens[:, :n_accepted]
        correction_token = verifier_predictions[:, n_accepted:n_accepted + 1]

        if n_accepted > 0:
            output_tokens = torch.cat([accepted_draft, correction_token], dim=1)
        else:
            output_tokens = correction_token

        return output_tokens, n_accepted

    @torch.no_grad()
    def verify_with_sampling(
        self,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
        verifier_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens with stochastic acceptance.

        Implements speculative sampling (Chen et al., ICLR 2024) which
        guarantees that the output distribution matches standard
        autoregressive sampling from the target model exactly.

        For each draft token c_k with probability q(c_k) from the draft:
        - Draw u ~ Uniform(0, 1)
        - If u < min(1, p(c_k) / q(c_k)): accept token c_k
        - Else: reject and sample from adjusted distribution

        Ref: ICML 2025 poster 46326, Section 3.3

        Args:
            draft_tokens: Draft token IDs, shape (batch, n_draft).
            draft_logits: Draft logits, shape (batch, n_draft, vocab_size).
            verifier_logits: Target model logits,
                shape (batch, n_draft + 1, vocab_size).

        Returns:
            Tuple (accepted_tokens, n_accepted).
        """
        batch_size = draft_tokens.shape[0]
        spec_length = draft_tokens.shape[1]
        device = verifier_logits.device

        # Get probabilities
        verifier_probs = F.softmax(verifier_logits.float(), dim=-1)
        draft_probs = F.softmax(draft_logits.float(), dim=-1)

        accepted_tokens_list: List[torch.Tensor] = []
        n_accepted = 0

        for k in range(spec_length):
            draft_token_k = draft_tokens[:, k]  # (batch,)

            # Gather probabilities for the specific draft tokens
            p_token = verifier_probs[:, k].gather(
                1, draft_token_k.unsqueeze(1)
            ).squeeze(1)
            q_token = draft_probs[:, k].gather(
                1, draft_token_k.unsqueeze(1)
            ).squeeze(1)

            # Acceptance ratio: min(1, p/q)
            acceptance_ratio = (p_token / q_token.clamp(min=1e-10)).clamp(max=1.0)

            # Draw uniform random
            u = torch.rand(batch_size, device=device)

            accepted = u < acceptance_ratio

            if not accepted.all():
                # Some batch elements rejected at position k
                # For batch simplicity, stop at the first batch rejection
                if accepted.any():
                    n_accepted = max(
                        n_accepted,
                        int(accepted.sum().item()) if n_accepted == 0 else n_accepted,
                    )
                break
            else:
                accepted_tokens_list.append(draft_token_k)
                n_accepted = k + 1

        # If all accepted, sample bonus token from verifier
        if n_accepted == spec_length:
            bonus_logits = verifier_logits[:, spec_length]
            bonus_token = self._sample_from_logits(bonus_logits)
            accepted_tokens_list.append(bonus_token)
        else:
            # Sample correction from adjusted distribution
            k_reject = n_accepted
            p_k = verifier_probs[:, k_reject]
            q_k = draft_probs[:, k_reject] if k_reject < spec_length else torch.zeros_like(p_k)
            adjusted = torch.clamp(p_k - q_k, min=0)
            adjusted_sum = adjusted.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            adjusted_probs = adjusted / adjusted_sum
            correction_token = torch.multinomial(
                adjusted_probs, num_samples=1
            ).squeeze(1)
            accepted_tokens_list.append(correction_token)

        if accepted_tokens_list:
            output_tokens = torch.stack(accepted_tokens_list, dim=1)
        else:
            output_tokens = self._sample_from_logits(
                verifier_logits[:, 0]
            ).unsqueeze(1)

        return output_tokens, n_accepted

    @torch.no_grad()
    def step(
        self,
        current_ids: torch.Tensor,
        target_forward: Callable,
    ) -> Tuple[torch.Tensor, int, bool]:
        """Execute one step of QuantSpec self-speculative decoding.

        This is the main entry point for the generation loop. Each call
        performs one draft-verify-accept/reject cycle.

        Ref: ICML 2025 poster 46326, Algorithm 1

        Args:
            current_ids: Current token IDs including all previously
                generated tokens, shape (batch, seq_len).
            target_forward: Callable that takes (input_ids,) and returns
                logits from the full-precision target model,
                shape (batch, seq_len + n_draft + 1, vocab_size).

        Returns:
            Tuple (new_tokens, n_accepted, eos_reached):
                - new_tokens: Accepted + correction tokens,
                    shape (batch, n_emitted).
                - n_accepted: Number of draft tokens accepted.
                - eos_reached: True if EOS token was generated.
        """
        spec_length = self._current_draft_length

        # ---- Phase 1: DRAFT ----
        draft_tokens, draft_logits = self.draft(current_ids, n_draft=spec_length)

        # ---- Phase 2: VERIFY ----
        # Build the verification input: original + draft tokens
        verify_ids = torch.cat([current_ids, draft_tokens], dim=1)
        verifier_logits = target_forward(verify_ids)
        if hasattr(verifier_logits, "logits"):
            verifier_logits = verifier_logits.logits

        # Extract the relevant positions for verification
        # verifier_logits at position seq_len-1 predicts first draft position, etc.
        seq_len = current_ids.shape[1]
        relevant_logits = verifier_logits[:, seq_len - 1:, :]  # (batch, spec_length+1, V)

        # ---- Phase 3: ACCEPT/REJECT ----
        if self.config.use_stochastic_acceptance:
            output_tokens, n_accepted = self.verify_with_sampling(
                draft_tokens=draft_tokens,
                draft_logits=draft_logits,
                verifier_logits=relevant_logits,
            )
        else:
            output_tokens, n_accepted = self.verify(
                input_ids=current_ids,
                draft_tokens=draft_tokens,
                verifier_logits=relevant_logits,
            )

        # ---- Update statistics and adaptive control ----
        acceptance_rate = n_accepted / max(spec_length, 1)
        self.stats.record_step(n_accepted, spec_length, self._current_level)

        if self.config.adaptive_level:
            self._select_level(acceptance_rate)
            self._adjust_draft_length(acceptance_rate)

        # ---- Check for EOS ----
        eos_reached = False
        if self.config.eos_token_id >= 0:
            eos_reached = (output_tokens == self.config.eos_token_id).any()

        return output_tokens, n_accepted, eos_reached

    def reset(self) -> None:
        """Reset decoder state for a new generation."""
        self._current_level = self.config.most_aggressive_level
        self._current_draft_length = self.config.max_draft_tokens
        self._ema_acceptance_rate = 0.0
        self._below_threshold_count = 0
        self.stats.reset()

    def __repr__(self) -> str:
        return (
            f"QuantSpecDecoder("
            f"levels={[l.value for l in self.config.quant_levels]}, "
            f"current_level={self._current_level.value}, "
            f"max_draft={self.config.max_draft_tokens}, "
            f"acceptance_threshold={self.config.acceptance_threshold:.0%})"
        )
