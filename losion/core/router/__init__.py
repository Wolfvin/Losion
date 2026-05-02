"""
Losion — Adaptive Router Module

Router adaptif untuk Tri-Jalur Architecture yang menggabungkan
BiasRouter (DeepSeek-V3) dan ThinkingToggle (Qwen3).
"""

from .bias_router import BiasRouter, PathwayRoutingInfo
from .thinking_toggle import (
    ThinkingToggle,
    ThinkingAssessment,
    ThinkingMode,
    TaskType,
)
from .router import AdaptiveRouter, AdaptiveRoutingOutput

__all__ = [
    "BiasRouter",
    "PathwayRoutingInfo",
    "ThinkingToggle",
    "ThinkingAssessment",
    "ThinkingMode",
    "TaskType",
    "AdaptiveRouter",
    "AdaptiveRoutingOutput",
]
