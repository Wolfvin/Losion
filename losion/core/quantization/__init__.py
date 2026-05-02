"""
Losion — Quantization Modules.

v0.4 additions:
  BitNetLinear      — 1.58-bit ternary weight quantization ({-1, 0, +1})
  BitNetConfig      — Configuration for gradual quantization schedule
  FP8TrainingPipeline — FP8 mixed-precision training pipeline
  absmean_quantize  — Core absmean quantization primitive
  convert_linear_to_bitnet — Convert nn.Linear layers to BitNetLinear
"""

from losion.core.quantization.bitnet import (
    BitNetConfig,
    BitNetLinear,
    absmean_quantize,
    pack_ternary_to_int2,
    unpack_int2_to_ternary,
    convert_linear_to_bitnet,
    finalize_bitnet_model,
    increment_bitnet_step,
    bitnet_weight_decay_loss,
)
from losion.core.quantization.fp8_training import FP8TrainingPipeline

__all__ = [
    "BitNetConfig",
    "BitNetLinear",
    "absmean_quantize",
    "pack_ternary_to_int2",
    "unpack_int2_to_ternary",
    "convert_linear_to_bitnet",
    "finalize_bitnet_model",
    "increment_bitnet_step",
    "bitnet_weight_decay_loss",
    "FP8TrainingPipeline",
]
