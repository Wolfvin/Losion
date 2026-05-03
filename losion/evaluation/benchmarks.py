"""
Losion Evaluation — Benchmark evaluation, perplexity scoring, and routing analysis.

Credits:
  - lm-eval-harness (EleutherAI) — standardized LM evaluation framework
  - DeepEval (confident-ai) — LLM evaluation with metrics
  - AutoEvoEval (arXiv:2506.23735) — automated evolutionary evaluation
  - MMLU, GSM8K, HellaSwag, ARC — standard NLP benchmarks

Provides:
  BenchmarkConfig     — Configuration for evaluation runs
  PerplexityEvaluator — Computes perplexity with sliding window support
  RoutingAnalyzer     — Analyzes Tri-Jalur routing behavior and expert utilization
  LosionEvaluator     — Orchestrates full evaluation pipeline
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.config import LosionConfig


# ============================================================================
# Benchmark Configuration
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation.

    Attributes:
        tasks: List of benchmark task names to evaluate.
            Supported: "mmlu", "gsm8k", "hellaswag", "arc", "winogrande",
            "truthfulqa", "humaneval", "bpp".
        num_fewshot: Number of few-shot examples for in-context learning.
        batch_size: Batch size for evaluation forward passes.
        max_seq_len: Maximum sequence length for evaluation.
        limit: Maximum number of examples per task (0 = all).
        seed: Random seed for reproducibility.
        sliding_window: Window size for sliding-window perplexity (0 = full).
        sliding_stride: Stride for sliding-window evaluation.
        device: Device for evaluation ("auto", "cuda", "cpu").
        dtype: Data type for evaluation computation.
    """

    tasks: List[str] = field(default_factory=lambda: ["mmlu", "gsm8k", "hellaswag"])
    num_fewshot: int = 0
    batch_size: int = 8
    max_seq_len: int = 4096
    limit: int = 0
    seed: int = 42
    sliding_window: int = 0
    sliding_stride: int = 512
    device: str = "auto"
    dtype: str = "auto"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_tasks = {
            "mmlu", "gsm8k", "hellaswag", "arc",
            "winogrande", "truthfulqa", "humaneval", "bpp",
        }
        for task in self.tasks:
            if task not in valid_tasks:
                raise ValueError(
                    f"Unknown benchmark task: {task!r}. "
                    f"Valid tasks: {sorted(valid_tasks)}"
                )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.sliding_window < 0:
            raise ValueError(
                f"sliding_window must be non-negative, got {self.sliding_window}"
            )


# ============================================================================
# Perplexity Evaluator
# ============================================================================


class PerplexityEvaluator:
    """Computes perplexity on a dataset with optional sliding window.

    Supports evaluation of language models on arbitrary text datasets.
    For long sequences, a sliding window approach can be used to avoid
    memory issues while still covering the full context.

    Args:
        config: BenchmarkConfig with evaluation parameters.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._total_nll = 0.0
        self._total_tokens = 0

    def evaluate(
        self,
        model: nn.Module,
        dataset: List[Dict[str, torch.Tensor]],
    ) -> float:
        """Compute perplexity on a dataset.

        Args:
            model: Language model with forward() returning logits.
            dataset: List of dicts with "input_ids" tensor (1D, variable length).

        Returns:
            Perplexity score (lower is better).
        """
        self._total_nll = 0.0
        self._total_tokens = 0

        model.eval()
        device = self._resolve_device(model)
        dtype = self._resolve_dtype(model)

        with torch.no_grad():
            for example in dataset:
                input_ids = example["input_ids"].to(device)
                seq_len = input_ids.shape[0]

                if seq_len < 2:
                    continue

                if self.config.sliding_window > 0 and seq_len > self.config.sliding_window:
                    nll, n_tokens = self._evaluate_sliding_window(
                        model, input_ids, device, dtype
                    )
                else:
                    nll, n_tokens = self._evaluate_full(
                        model, input_ids, device, dtype
                    )

                self._total_nll += nll
                self._total_tokens += n_tokens

        if self._total_tokens == 0:
            return float("inf")

        avg_nll = self._total_nll / self._total_tokens
        perplexity = math.exp(avg_nll)
        return perplexity

    def _evaluate_full(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[float, int]:
        """Evaluate a single sequence without sliding window.

        Args:
            model: Language model.
            input_ids: 1D token IDs tensor.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple (total_nll, num_tokens).
        """
        input_ids = input_ids.unsqueeze(0)  # (1, seq_len)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            logits = model(input_ids)  # (1, seq_len, vocab_size)

        if isinstance(logits, tuple):
            logits = logits[0]

        # Shift: predict next token
        shift_logits = logits[:, :-1, :]  # (1, seq_len-1, vocab_size)
        shift_labels = input_ids[:, 1:]   # (1, seq_len-1)

        # Compute per-token NLL
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )

        return nll.item(), shift_labels.numel()

    def _evaluate_sliding_window(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[float, int]:
        """Evaluate a long sequence with sliding window.

        Uses overlapping windows to compute perplexity. Each token's
        NLL is computed as the average over all windows containing it.

        Args:
            model: Language model.
            input_ids: 1D token IDs tensor.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple (total_nll, num_tokens).
        """
        window = self.config.sliding_window
        stride = self.config.sliding_stride
        seq_len = input_ids.shape[0]

        nll_accum = torch.zeros(seq_len, device=device, dtype=torch.float64)
        count_accum = torch.zeros(seq_len, device=device, dtype=torch.float64)

        for start in range(0, seq_len - 1, stride):
            end = min(start + window, seq_len)
            chunk = input_ids[start:end].unsqueeze(0)

            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                logits = model(chunk)

            if isinstance(logits, tuple):
                logits = logits[0]

            shift_logits = logits[:, :-1, :]
            shift_labels = chunk[:, 1:]

            # Per-token NLL
            per_token_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            )

            token_indices = torch.arange(start + 1, end, device=device)
            valid_len = min(len(token_indices), per_token_nll.shape[0])

            if valid_len > 0:
                nll_accum[token_indices[:valid_len]] += per_token_nll[:valid_len].double()
                count_accum[token_indices[:valid_len]] += 1.0

        # Average NLL per token
        mask = count_accum > 0
        if not mask.any():
            return 0.0, 0

        avg_nll = nll_accum[mask] / count_accum[mask]
        total_nll = avg_nll.sum().item()
        total_tokens = mask.sum().item()

        return total_nll, total_tokens

    def _resolve_device(self, model: nn.Module) -> torch.device:
        """Resolve the device for evaluation."""
        if self.config.device != "auto":
            return torch.device(self.config.device)
        try:
            param = next(model.parameters())
            return param.device
        except StopIteration:
            return torch.device("cpu")

    def _resolve_dtype(self, model: nn.Module) -> torch.dtype:
        """Resolve the dtype for evaluation."""
        if self.config.dtype != "auto":
            dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
            return dtype_map.get(self.config.dtype, torch.float32)
        try:
            param = next(model.parameters())
            return param.dtype
        except StopIteration:
            return torch.float32


# ============================================================================
# Routing Report
# ============================================================================


@dataclass
class RoutingReport:
    """Analysis report for Tri-Jalur routing behavior.

    Attributes:
        pathway_utilization: Mean routing weight per pathway.
            Keys: "ssm", "attention", "retrieval". Values: float in [0, 1].
        expert_specialization: Per-expert activation frequency.
            Dict mapping expert_id -> activation_rate.
        routing_entropy: Mean entropy of routing distributions.
            Lower = more specialized, Higher = more uniform.
        max_entropy: Maximum possible entropy (log(num_pathways)).
        normalized_entropy: routing_entropy / max_entropy in [0, 1].
        routing_collapse: Whether routing has collapsed (one pathway dominates).
        collapse_pathway: Which pathway dominates if collapsed (None if not).
        collapse_threshold: Threshold for collapse detection.
        layer_analysis: Per-layer routing statistics.
        expert_overlap: Pairwise expert overlap coefficient.
    """

    pathway_utilization: Dict[str, float] = field(default_factory=dict)
    expert_specialization: Dict[int, float] = field(default_factory=dict)
    routing_entropy: float = 0.0
    max_entropy: float = 0.0
    normalized_entropy: float = 0.0
    routing_collapse: bool = False
    collapse_pathway: Optional[str] = None
    collapse_threshold: float = 0.9
    layer_analysis: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    expert_overlap: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary of the routing report."""
        lines = [
            "=== Losion Routing Analysis Report ===",
            "",
            "Pathway Utilization:",
        ]
        for pathway, util in self.pathway_utilization.items():
            bar = "█" * int(util * 40) + "░" * (40 - int(util * 40))
            lines.append(f"  {pathway:>12s}: {util:.4f} |{bar}|")

        lines.append("")
        lines.append(f"Routing Entropy:     {self.routing_entropy:.4f}")
        lines.append(f"Max Entropy:         {self.max_entropy:.4f}")
        lines.append(f"Normalized Entropy:  {self.normalized_entropy:.4f}")
        lines.append("")

        if self.routing_collapse:
            lines.append(f"⚠ ROUTING COLLAPSE DETECTED on pathway: {self.collapse_pathway}")
        else:
            lines.append("✓ No routing collapse detected")

        lines.append("")
        lines.append(f"Expert Specialization (top 10):")
        sorted_experts = sorted(
            self.expert_specialization.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for eid, rate in sorted_experts:
            lines.append(f"  Expert {eid:>3d}: {rate:.4f}")

        if self.expert_overlap:
            lines.append("")
            lines.append("Expert Overlap (top 5 highest):")
            sorted_overlap = sorted(
                self.expert_overlap.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for (e1, e2), overlap in sorted_overlap:
                lines.append(f"  Expert {e1} <-> Expert {e2}: {overlap:.4f}")

        return "\n".join(lines)


# ============================================================================
# Routing Analyzer
# ============================================================================


class RoutingAnalyzer:
    """Analyzes routing behavior across layers and inputs.

    Examines the Tri-Jalur Router's decisions across layers and inputs,
    computing pathway utilization, expert specialization scores, and
    routing entropy. Detects routing collapse where all tokens are
    routed to the same pathway.

    Args:
        collapse_threshold: Pathway weight threshold for collapse detection.
            If any pathway weight exceeds this on average, collapse is flagged.
    """

    PATHWAY_NAMES = ["ssm", "attention", "retrieval"]

    def __init__(self, collapse_threshold: float = 0.9) -> None:
        self.collapse_threshold = collapse_threshold

    def analyze(
        self,
        model: nn.Module,
        dataloader: Any,
        max_batches: int = 100,
    ) -> RoutingReport:
        """Analyze routing behavior on a dataset.

        Args:
            model: LosionModel with routing_info support.
            dataloader: Iterable yielding batches with "input_ids".
            max_batches: Maximum number of batches to analyze.

        Returns:
            RoutingReport with comprehensive routing analysis.
        """
        model.eval()
        report = RoutingReport(collapse_threshold=self.collapse_threshold)

        # Accumulators
        pathway_weights_accum: List[torch.Tensor] = []
        expert_indices_accum: List[torch.Tensor] = []
        layer_routing_accum: Dict[int, List[torch.Tensor]] = {}

        num_pathways = 3  # SSM, Attention, Retrieval

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                input_ids = batch["input_ids"]
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)

                # Forward pass with routing info
                try:
                    output = model(
                        input_ids,
                        return_routing_info=True,
                    )
                except TypeError:
                    # Fallback if model doesn't support return_routing_info
                    output = model(input_ids)

                # Extract routing info
                routing_info = getattr(output, "routing_info", None)
                if routing_info is None:
                    continue

                for layer_idx, layer_info in enumerate(routing_info):
                    if layer_info is None:
                        continue

                    # Extract route weights
                    route_weights = layer_info.get("route_weights", None)
                    if route_weights is not None:
                        if layer_idx not in layer_routing_accum:
                            layer_routing_accum[layer_idx] = []
                        layer_routing_accum[layer_idx].append(route_weights.cpu())
                        pathway_weights_accum.append(route_weights.cpu())

                    # Extract expert indices from MoE
                    ret_aux = layer_info.get("retrieval_aux", {})
                    expert_indices = ret_aux.get("expert_indices", None)
                    if expert_indices is not None:
                        expert_indices_accum.append(expert_indices.cpu())

        # Compute pathway utilization
        if pathway_weights_accum:
            all_weights = torch.cat(pathway_weights_accum, dim=0)
            # all_weights: (total_tokens, seq_len, 3) or similar
            mean_weights = all_weights.mean(dim=tuple(range(all_weights.dim() - 1)))
            for i, name in enumerate(self.PATHWAY_NAMES):
                report.pathway_utilization[name] = mean_weights[i].item()

        # Compute routing entropy
        if pathway_weights_accum:
            all_weights = torch.cat(pathway_weights_accum, dim=0)
            report.max_entropy = math.log(num_pathways)
            clamped = all_weights.clamp(min=1e-8)
            entropy = -(clamped * clamped.log()).sum(dim=-1)
            report.routing_entropy = entropy.mean().item()
            report.normalized_entropy = (
                report.routing_entropy / report.max_entropy
                if report.max_entropy > 0
                else 0.0
            )

        # Detect routing collapse
        if report.pathway_utilization:
            for name, util in report.pathway_utilization.items():
                if util > self.collapse_threshold:
                    report.routing_collapse = True
                    report.collapse_pathway = name
                    break

        # Compute expert specialization
        if expert_indices_accum:
            all_indices = torch.cat(expert_indices_accum, dim=0).flatten()
            unique_experts, counts = torch.unique(all_indices, return_counts=True)
            total = counts.sum().float()
            for eid, cnt in zip(unique_experts.tolist(), counts.tolist()):
                report.expert_specialization[int(eid)] = cnt / total

        # Compute per-layer analysis
        for layer_idx, weight_list in layer_routing_accum.items():
            layer_weights = torch.cat(weight_list, dim=0)
            layer_mean = layer_weights.mean(dim=tuple(range(layer_weights.dim() - 1)))
            clamped = layer_weights.clamp(min=1e-8)
            layer_entropy = -(clamped * clamped.log()).sum(dim=-1).mean().item()

            report.layer_analysis[layer_idx] = {
                "mean_weights": {
                    name: layer_mean[i].item()
                    for i, name in enumerate(self.PATHWAY_NAMES)
                },
                "entropy": layer_entropy,
                "dominant_pathway": self.PATHWAY_NAMES[layer_mean.argmax().item()],
            }

        # Compute expert overlap (Jaccard-like coefficient)
        if report.expert_specialization and len(report.expert_specialization) > 1:
            sorted_experts = sorted(report.expert_specialization.items(), key=lambda x: x[1], reverse=True)[:20]
            for i in range(min(len(sorted_experts), 10)):
                for j in range(i + 1, min(len(sorted_experts), 10)):
                    e1, r1 = sorted_experts[i]
                    e2, r2 = sorted_experts[j]
                    # Overlap coefficient: min(r1, r2) / max(r1, r2)
                    denom = max(r1, r2)
                    overlap = min(r1, r2) / denom if denom > 0 else 0.0
                    report.expert_overlap[(e1, e2)] = overlap

        return report


# ============================================================================
# Evaluation Report
# ============================================================================


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for a Losion model.

    Attributes:
        model_name: Name/identifier of the evaluated model.
        perplexity: Perplexity score (None if not evaluated).
        benchmark_scores: Dict mapping task_name -> score.
        routing_report: Routing analysis report (None if not analyzed).
        total_eval_tokens: Total tokens evaluated.
        config: BenchmarkConfig used for evaluation.
    """

    model_name: str = ""
    perplexity: Optional[float] = None
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    routing_report: Optional[RoutingReport] = None
    total_eval_tokens: int = 0
    config: Optional[BenchmarkConfig] = None

    def summary(self) -> str:
        """Generate a human-readable evaluation summary."""
        lines = [
            "=== Losion Evaluation Report ===",
            f"Model: {self.model_name or 'unnamed'}",
            "",
        ]

        if self.perplexity is not None:
            lines.append(f"Perplexity: {self.perplexity:.4f}")
        else:
            lines.append("Perplexity: Not evaluated")

        if self.benchmark_scores:
            lines.append("")
            lines.append("Benchmark Scores:")
            max_name_len = max(len(name) for name in self.benchmark_scores)
            for name, score in sorted(self.benchmark_scores.items()):
                lines.append(f"  {name:>{max_name_len}s}: {score:.4f}")

        if self.routing_report is not None:
            lines.append("")
            lines.append(self.routing_report.summary())

        lines.append(f"\nTotal tokens evaluated: {self.total_eval_tokens:,}")
        return "\n".join(lines)


# ============================================================================
# Losion Evaluator — Orchestration
# ============================================================================


class LosionEvaluator:
    """Orchestrates evaluation of a Losion model.

    Combines perplexity evaluation, benchmark scoring, and routing
    analysis into a unified evaluation pipeline.

    Args:
        model: LosionModel to evaluate.
        config: BenchmarkConfig with evaluation parameters.
        model_name: Name/identifier for the model.
    """

    def __init__(
        self,
        model: nn.Module,
        config: BenchmarkConfig,
        model_name: str = "losion-model",
    ) -> None:
        self.model = model
        self.config = config
        self.model_name = model_name
        self._perplexity_evaluator = PerplexityEvaluator(config)
        self._routing_analyzer = RoutingAnalyzer()

    def evaluate_perplexity(
        self,
        dataset: List[Dict[str, torch.Tensor]],
    ) -> float:
        """Evaluate perplexity on a dataset.

        Args:
            dataset: List of dicts with "input_ids" tensor.

        Returns:
            Perplexity score.
        """
        return self._perplexity_evaluator.evaluate(self.model, dataset)

    def evaluate_benchmarks(
        self,
        datasets: Optional[Dict[str, List[Dict[str, torch.Tensor]]]] = None,
    ) -> Dict[str, float]:
        """Evaluate on standard benchmarks.

        For each task in config.tasks, computes a task-specific score
        (perplexity for language modeling tasks, accuracy for QA tasks).

        Args:
            datasets: Optional dict mapping task_name -> dataset.
                If None, only validates config and returns empty scores.

        Returns:
            Dict mapping task_name -> score.
        """
        scores: Dict[str, float] = {}

        if datasets is None:
            return scores

        for task_name in self.config.tasks:
            if task_name not in datasets:
                continue

            dataset = datasets[task_name]
            if not dataset:
                continue

            if task_name in ("mmlu", "arc"):
                # Multiple-choice: compute accuracy via log-likelihood
                score = self._evaluate_multiple_choice(dataset)
            elif task_name == "gsm8k":
                # Math: compute pass@1 accuracy
                score = self._evaluate_generation(dataset)
            elif task_name == "hellaswag":
                # Sentence completion: log-likelihood comparison
                score = self._evaluate_multiple_choice(dataset)
            else:
                # Default: perplexity-based
                ppl = self._perplexity_evaluator.evaluate(self.model, dataset)
                scores[task_name] = ppl
                continue

            scores[task_name] = score

        return scores

    def analyze_routing(
        self,
        dataloader: Any,
        max_batches: int = 100,
    ) -> RoutingReport:
        """Analyze routing behavior on a dataset.

        Args:
            dataloader: Iterable yielding batches with "input_ids".
            max_batches: Maximum number of batches to analyze.

        Returns:
            RoutingReport with comprehensive routing analysis.
        """
        return self._routing_analyzer.analyze(
            self.model, dataloader, max_batches=max_batches
        )

    def full_evaluation(
        self,
        perplexity_dataset: Optional[List[Dict[str, torch.Tensor]]] = None,
        benchmark_datasets: Optional[Dict[str, List[Dict[str, torch.Tensor]]]] = None,
        routing_dataloader: Optional[Any] = None,
    ) -> EvaluationReport:
        """Run full evaluation pipeline.

        Args:
            perplexity_dataset: Dataset for perplexity evaluation.
            benchmark_datasets: Dict mapping task_name -> dataset.
            routing_dataloader: Dataloader for routing analysis.

        Returns:
            EvaluationReport with all metrics.
        """
        report = EvaluationReport(
            model_name=self.model_name,
            config=self.config,
        )

        # Perplexity
        if perplexity_dataset is not None:
            report.perplexity = self.evaluate_perplexity(perplexity_dataset)

        # Benchmarks
        if benchmark_datasets is not None:
            report.benchmark_scores = self.evaluate_benchmarks(benchmark_datasets)

        # Routing analysis
        if routing_dataloader is not None:
            report.routing_report = self.analyze_routing(routing_dataloader)

        return report

    def _evaluate_multiple_choice(
        self,
        dataset: List[Dict[str, torch.Tensor]],
    ) -> float:
        """Evaluate multiple-choice benchmark (MMLU, ARC, HellaSwag).

        Computes accuracy by comparing log-likelihoods of answer choices.

        Args:
            dataset: List of dicts with "input_ids" and "label" tensors.

        Returns:
            Accuracy score in [0, 1].
        """
        self.model.eval()
        device = self._get_device()
        correct = 0
        total = 0

        with torch.no_grad():
            for example in dataset:
                input_ids = example["input_ids"].to(device).unsqueeze(0)
                label = example.get("label", None)
                if label is None:
                    continue

                logits = self.model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]

                # Get logit at last position for the label token
                last_logits = logits[0, -1, :]
                predicted = last_logits.argmax().item()
                target = label.item() if isinstance(label, torch.Tensor) else label

                if predicted == target:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def _evaluate_generation(
        self,
        dataset: List[Dict[str, torch.Tensor]],
    ) -> float:
        """Evaluate generation benchmark (GSM8K).

        Computes pass@1 accuracy for mathematical reasoning tasks.

        Args:
            dataset: List of dicts with "input_ids" and "answer" tensors.

        Returns:
            Accuracy score in [0, 1].
        """
        self.model.eval()
        device = self._get_device()
        correct = 0
        total = 0

        with torch.no_grad():
            for example in dataset:
                input_ids = example["input_ids"].to(device).unsqueeze(0)
                answer = example.get("answer", None)
                if answer is None:
                    continue

                # Greedy generation (simplified)
                generated = input_ids
                max_gen_len = min(256, self.config.max_seq_len - input_ids.shape[1])

                for _ in range(max_gen_len):
                    logits = self.model(generated)
                    if isinstance(logits, tuple):
                        logits = logits[0]

                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)

                # Check if the answer appears in generated text
                gen_tokens = generated[0].tolist()
                target_answer = str(answer.item() if isinstance(answer, torch.Tensor) else answer)
                gen_text = str(gen_tokens)  # Simplified check
                if target_answer in gen_text:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def _get_device(self) -> torch.device:
        """Get the device of the model."""
        try:
            param = next(self.model.parameters())
            return param.device
        except StopIteration:
            return torch.device("cpu")
