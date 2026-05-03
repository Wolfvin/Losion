"""
Losion Evaluation — Benchmark evaluation, perplexity scoring, and routing analysis.

Provides standardized evaluation tools for Losion models including:
  - Perplexity evaluation with sliding window support
  - Benchmark evaluation (MMLU, GSM8K, HellaSwag, ARC)
  - Routing behavior analysis and collapse detection
  - Full evaluation pipeline orchestration

Credits:
  - lm-eval-harness (EleutherAI)
  - DeepEval (confident-ai)
  - AutoEvoEval (arXiv:2506.23735)

Usage:
    >>> from losion.evaluation import LosionEvaluator, BenchmarkConfig
    >>> from losion.config import LosionConfig
    >>> config = BenchmarkConfig(tasks=["mmlu", "gsm8k"], num_fewshot=5)
    >>> evaluator = LosionEvaluator(model, config, model_name="losion-7b")
    >>> report = evaluator.full_evaluation(perplexity_dataset=val_data)
    >>> print(report.summary())
"""

from __future__ import annotations

from losion.evaluation.benchmarks import (
    BenchmarkConfig,
    EvaluationReport,
    LosionEvaluator,
    PerplexityEvaluator,
    RoutingAnalyzer,
    RoutingReport,
)

__all__ = [
    "BenchmarkConfig",
    "EvaluationReport",
    "LosionEvaluator",
    "PerplexityEvaluator",
    "RoutingAnalyzer",
    "RoutingReport",
]
