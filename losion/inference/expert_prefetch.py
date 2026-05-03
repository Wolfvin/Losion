"""
Losion Expert Prefetch — Speculating Experts for MoE inference acceleration.

Implements the "Speculating Experts" technique for accelerating Mixture-of-Experts
inference by predicting which experts will be needed in subsequent layers and
prefetching them ahead of time, thereby overlapping expert loading latency with
ongoing computation.

Reference:
    "Speculating Experts Accelerates Inference for MoE"
    arXiv:2603.19289, March 2026

Key Insight
-----------
In MoE models, each transformer layer contains a router that selects a sparse
subset of experts for each token.  During inference, loading expert weights
from off-chip memory (or across devices in expert-parallel setups) introduces
significant latency — often dominating per-layer compute time.

The core observation is that the hidden states computed at layer L contain
rich information about the token's semantic trajectory, which strongly
correlates with the routing decisions at layer L+1.  By training a tiny
predictor network that maps layer-L hidden states to layer-(L+1) expert
indices, we can *speculate* which experts will be needed *before* the
router at layer L+1 is actually computed, and start loading them early.

Architecture Overview
---------------------
1. **PrefetchConfig** — Dataclass controlling all prefetch hyperparameters:
   predictor architecture, prefetch horizon, temperature scheduling,
   and accuracy tracking options.

2. **LightweightPredictor** — A small 2-layer MLP per layer that maps
   hidden states from layer L to a probability distribution over experts
   at layer L+1.  Parameter count is <1% of a single expert, making the
   overhead negligible.  Supports both finite MoE (discrete expert indices)
   and ∞-MoE (continuous expert code prediction).

3. **ExpertPrefetcher** — The main orchestrator that manages per-layer
   predictors, drives prefetch predictions, and coordinates with the
   MoE layer for expert loading.  Operates in two modes:
     - *Training mode*: learns predictors from ground-truth routing data.
     - *Inference mode*: uses predictors to speculate and prefetch experts.

4. **PrefetchAccuracyTracker** — Monitors prediction quality in real-time
   during inference, tracking per-layer and aggregate metrics (precision,
   recall, hit rate, coverage) to guide adaptive prefetch strategies.

Prefetch Pipeline (Inference)
-----------------------------
For each layer L during inference::

    1. Receive hidden_states_L from the attention sublayer.
    2. Feed hidden_states_L into predictor[L] → predicted expert set for L+1.
    3. Issue async prefetch for predicted experts (overlap with expert compute
       at layer L).
    4. When layer L+1 begins, check if needed experts are already loaded.
       - Hit: use prefetched expert (zero loading latency).
       - Miss: load on-demand (fallback, same as no-prefetch baseline).

The predictor is trained to maximise top-k recall: if the actual router
at L+1 selects experts {e1, e2, ..., ek}, we want the predictor's top-k
predictions to overlap with this set as much as possible.

Compatibility
-------------
- **Finite MoE** (e.g., Switch Transformer, Mixtral): Predictor outputs
  logits over discrete expert indices {0, 1, ..., N-1}.
- **Infinite MoE** (∞-MoE, continuous expert space): Predictor outputs
  predicted expert *codes* in R^d_code, and nearby codes in the continuous
  space are prefetched via the ExpertCodeClusterer.
- Works with any MoE implementation in Losion that exposes routing
  decisions (expert indices or expert codes) during training.

Credits:
    - "Speculating Experts Accelerates Inference for MoE"
      (arXiv:2603.19289, March 2026)
    - ∞-MoE continuous expert space (arXiv:2601.17680)
    - DeepSeek-V2 expert-parallel communication hiding

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PrefetchConfig — Configuration for expert prefetching
# ============================================================================


@dataclass
class PrefetchConfig:
    """Configuration for the Speculating Experts prefetch system.

    Controls predictor architecture, prefetch strategy, training
    hyperparameters, and accuracy monitoring.

    Attributes:
        n_layers: Number of transformer layers in the model.
        d_model: Model hidden dimension (input to predictors).
        n_experts: Number of experts per MoE layer (finite MoE).
            Set to 0 for ∞-MoE (continuous) mode.
        top_k: Number of experts activated per token per layer.
            Predictors are trained to recall the top-k experts.
        predictor_hidden_dim: Hidden dimension of the 2-layer predictor
            MLP.  Kept small (<1% of expert params) for minimal overhead.
        predictor_dropout: Dropout rate in the predictor MLP.
        prefetch_budget: Maximum number of experts to prefetch per layer.
            Can exceed top_k to improve recall at the cost of more
            prefetch traffic.  Set to 0 to use ``top_k`` directly.
        prediction_temperature: Temperature for sampling from predictor
            logits during inference.  Higher temperature → more diverse
            (exploratory) prefetching.  Set to 0.0 for greedy top-k.
        adaptive_temperature: If True, dynamically adjust prediction
            temperature based on recent prefetch accuracy.  When accuracy
            is high, lower temperature (exploit); when low, raise it
            (explore).
        adaptive_temp_min: Minimum temperature for adaptive scheduling.
        adaptive_temp_max: Maximum temperature for adaptive scheduling.
        adaptive_temp_decay: Exponential decay factor for adaptive
            temperature when accuracy is high.
        infinite_moe_mode: If True, predictors output continuous expert
            codes instead of discrete expert indices.  Used with ∞-MoE.
        code_dim: Dimensionality of expert codes in ∞-MoE mode.
            Ignored when ``infinite_moe_mode`` is False.
        prefetch_code_radius: L2 distance radius for prefetching nearby
            expert codes in ∞-MoE mode.  All codes within this radius of
            a predicted code are prefetched.
        train_predictor_lr: Learning rate for predictor training.
        train_predictor_weight_decay: Weight decay for predictor training.
        train_loss_type: Loss function for predictor training.
            - ``"ce"``: Cross-entropy on expert indices (finite MoE).
            - ``"bce"``: Multi-label binary cross-entropy (finite MoE).
            - ``"mse"``: MSE on expert codes (∞-MoE).
            - ``"cosine"``: Cosine embedding loss on expert codes (∞-MoE).
        track_accuracy: If True, maintain a PrefetchAccuracyTracker
            during inference for monitoring and adaptive strategies.
        accuracy_window: Number of recent tokens to consider for
            rolling accuracy statistics.
    """

    n_layers: int = 32
    d_model: int = 768
    n_experts: int = 64
    top_k: int = 2
    predictor_hidden_dim: int = 128
    predictor_dropout: float = 0.0
    prefetch_budget: int = 0
    prediction_temperature: float = 0.0
    adaptive_temperature: bool = False
    adaptive_temp_min: float = 0.5
    adaptive_temp_max: float = 2.0
    adaptive_temp_decay: float = 0.99
    infinite_moe_mode: bool = False
    code_dim: int = 64
    prefetch_code_radius: float = 1.0
    train_predictor_lr: float = 1e-3
    train_predictor_weight_decay: float = 1e-4
    train_loss_type: str = "bce"
    track_accuracy: bool = True
    accuracy_window: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        if not self.infinite_moe_mode and self.n_experts <= 0:
            raise ValueError(
                f"n_experts must be > 0 in finite MoE mode, got {self.n_experts}"
            )
        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")
        if self.predictor_hidden_dim <= 0:
            raise ValueError(
                f"predictor_hidden_dim must be > 0, got {self.predictor_hidden_dim}"
            )
        if self.train_loss_type not in ("ce", "bce", "mse", "cosine"):
            raise ValueError(
                f"train_loss_type must be 'ce', 'bce', 'mse', or 'cosine', "
                f"got '{self.train_loss_type}'"
            )
        if self.infinite_moe_mode:
            if self.code_dim <= 0:
                raise ValueError(
                    f"code_dim must be > 0 in infinite MoE mode, got {self.code_dim}"
                )
            if self.train_loss_type in ("ce", "bce"):
                raise ValueError(
                    f"infinite_moe_mode requires 'mse' or 'cosine' loss, "
                    f"got '{self.train_loss_type}'"
                )
        else:
            if self.train_loss_type in ("mse", "cosine"):
                raise ValueError(
                    f"finite MoE mode requires 'ce' or 'bce' loss, "
                    f"got '{self.train_loss_type}'"
                )

    @property
    def effective_prefetch_budget(self) -> int:
        """Return the effective number of experts to prefetch per layer."""
        return self.prefetch_budget if self.prefetch_budget > 0 else self.top_k


# ============================================================================
# LightweightPredictor — Per-layer expert prediction network
# ============================================================================


class LightweightPredictor(nn.Module):
    """Lightweight predictor that maps layer-L hidden states to layer-(L+1)
    expert predictions.

    For finite MoE, outputs logits over discrete expert indices.
    For ∞-MoE, outputs predicted expert codes in continuous space.

    Architecture: a 2-layer MLP with SiLU activation::

        hidden_states → Linear(d_model, hidden_dim) → SiLU → Dropout
                     → Linear(hidden_dim, output_dim)

    where ``output_dim`` is ``n_experts`` (finite) or ``top_k * code_dim``
    (infinite).  The total parameter count is kept under 1% of a single
    expert's parameters.

    Args:
        d_model: Model hidden dimension (input size).
        hidden_dim: Predictor MLP hidden dimension.
        n_experts: Number of experts (finite MoE).  0 for ∞-MoE.
        top_k: Number of experts per token.
        code_dim: Expert code dimension (∞-MoE only).
        dropout: Dropout rate.
        infinite_moe_mode: If True, output expert codes instead of logits.

    References:
        "Speculating Experts Accelerates Inference for MoE"
        (arXiv:2603.19289, Section 3.1: Lightweight Predictor Design)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_experts: int = 0,
        top_k: int = 2,
        code_dim: int = 64,
        dropout: float = 0.0,
        infinite_moe_mode: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.code_dim = code_dim
        self.infinite_moe_mode = infinite_moe_mode

        # Output dimension
        if infinite_moe_mode:
            # Predict top_k expert codes, each of dimension code_dim
            output_dim = top_k * code_dim
        else:
            # Predict logits over discrete experts
            output_dim = n_experts

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize with small weights for stable training start."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Scale down final layer for small initial predictions
        last_linear = self.mlp[-1]
        if isinstance(last_linear, nn.Linear):
            with torch.no_grad():
                last_linear.weight.mul_(0.01)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Predict expert routing for the next layer.

        Args:
            hidden_states: Layer-L hidden states.
                Shape ``[batch_size, seq_len, d_model]``.

        Returns:
            Finite MoE mode:
                Expert logits of shape ``[batch_size, seq_len, n_experts]``.
                Higher values indicate higher predicted routing probability.
            ∞-MoE mode:
                Predicted expert codes of shape
                ``[batch_size, seq_len, top_k, code_dim]``.
        """
        output = self.mlp(hidden_states)

        if self.infinite_moe_mode:
            batch_size, seq_len, _ = output.shape
            output = output.view(batch_size, seq_len, self.top_k, self.code_dim)

        return output


# ============================================================================
# PrefetchAccuracyTracker — Real-time monitoring of prediction quality
# ============================================================================


class PrefetchAccuracyTracker:
    """Tracks the accuracy of expert prefetch predictions during inference.

    Maintains rolling statistics over a configurable window to monitor
    prediction quality and support adaptive strategies (e.g., temperature
    scheduling).

    Metrics computed per-layer and aggregated:
      - **Precision@k**: Fraction of prefetched experts that were actually
        used (among top-k prefetch predictions).
      - **Recall@k**: Fraction of actually-used experts that were correctly
        prefetched.
      - **Hit Rate**: Fraction of tokens where at least one prefetched
        expert was used.
      - **Coverage**: Fraction of tokens where *all* used experts were
        correctly prefetched (perfect prediction).
      - **Over-prefetch Ratio**: Average ratio of prefetched-but-unused
        experts to total prefetched experts.

    Args:
        n_layers: Number of transformer layers.
        window: Number of recent tokens for rolling statistics.
        top_k: Number of experts activated per token.

    References:
        "Speculating Experts Accelerates Inference for MoE"
        (arXiv:2603.19289, Section 4: Evaluation Methodology)
    """

    def __init__(
        self,
        n_layers: int,
        window: int = 1000,
        top_k: int = 2,
    ) -> None:
        self.n_layers = n_layers
        self.window = window
        self.top_k = top_k

        # Per-layer rolling records: each entry is a dict of metric values
        self._records: Dict[int, List[Dict[str, float]]] = {
            layer: [] for layer in range(n_layers)
        }

        # Aggregated counters
        self._total_predictions = 0
        self._total_hits = 0
        self._total_perfect = 0

    def update(
        self,
        layer_idx: int,
        predicted_experts: torch.Tensor,
        actual_experts: torch.Tensor,
    ) -> Dict[str, float]:
        """Record a prediction-actual pair and compute metrics.

        For finite MoE mode with discrete expert indices.

        Args:
            layer_idx: Layer index (0-based).
            predicted_experts: Predicted expert indices.
                Shape ``[batch_size, seq_len, prefetch_budget]`` or
                ``[batch_size, prefetch_budget]``.
            actual_experts: Ground-truth expert indices from the router.
                Shape ``[batch_size, seq_len, top_k]`` or
                ``[batch_size, top_k]``.

        Returns:
            Dictionary of per-layer metrics for this update.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )

        # Flatten batch and sequence dimensions
        pred_flat = predicted_experts.reshape(-1, predicted_experts.shape[-1])
        actual_flat = actual_experts.reshape(-1, actual_experts.shape[-1])

        n_tokens = pred_flat.shape[0]
        budget = pred_flat.shape[1]

        precisions: List[float] = []
        recalls: List[float] = []
        hits = 0
        perfect = 0

        for t in range(n_tokens):
            pred_set = set(pred_flat[t].tolist())
            actual_set = set(actual_flat[t].tolist())

            # Intersection
            intersection = pred_set & actual_set
            n_intersection = len(intersection)

            # Precision: fraction of predictions that are correct
            precision = n_intersection / len(pred_set) if pred_set else 0.0
            precisions.append(precision)

            # Recall: fraction of actual experts that were predicted
            recall = n_intersection / len(actual_set) if actual_set else 0.0
            recalls.append(recall)

            # Hit: at least one correct prediction
            if n_intersection > 0:
                hits += 1

            # Perfect: all actual experts were predicted
            if actual_set.issubset(pred_set):
                perfect += 1

        # Aggregate metrics for this update
        metrics = {
            "precision": sum(precisions) / n_tokens if n_tokens > 0 else 0.0,
            "recall": sum(recalls) / n_tokens if n_tokens > 0 else 0.0,
            "hit_rate": hits / n_tokens if n_tokens > 0 else 0.0,
            "coverage": perfect / n_tokens if n_tokens > 0 else 0.0,
            "over_prefetch_ratio": 1.0 - (sum(precisions) / n_tokens) if n_tokens > 0 else 0.0,
        }

        # Record
        self._records[layer_idx].append(metrics)
        if len(self._records[layer_idx]) > self.window:
            self._records[layer_idx] = self._records[layer_idx][-self.window:]

        self._total_predictions += n_tokens
        self._total_hits += hits
        self._total_perfect += perfect

        return metrics

    def update_infinite_moe(
        self,
        layer_idx: int,
        predicted_codes: torch.Tensor,
        actual_codes: torch.Tensor,
        radius: float = 1.0,
    ) -> Dict[str, float]:
        """Record a prediction-actual pair for ∞-MoE (continuous) mode.

        A predicted code is considered a "hit" if its L2 distance to an
        actual expert code is within ``radius``.

        Args:
            layer_idx: Layer index (0-based).
            predicted_codes: Predicted expert codes.
                Shape ``[batch_size, seq_len, prefetch_budget, code_dim]``.
            actual_codes: Ground-truth expert codes from the router.
                Shape ``[batch_size, seq_len, top_k, code_dim]``.
            radius: L2 distance threshold for considering a code as matched.

        Returns:
            Dictionary of per-layer metrics for this update.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )

        B, S, K_pred, D = predicted_codes.shape
        K_actual = actual_codes.shape[2]

        # Flatten
        pred_flat = predicted_codes.reshape(-1, K_pred, D)  # [N, K_pred, D]
        actual_flat = actual_codes.reshape(-1, K_actual, D)  # [N, K_actual, D]
        N = pred_flat.shape[0]

        precisions: List[float] = []
        recalls: List[float] = []
        hits = 0
        perfect = 0

        for t in range(N):
            # Compute pairwise L2 distances between predicted and actual codes
            # pred: [K_pred, D], actual: [K_actual, D]
            dists = torch.cdist(
                pred_flat[t].unsqueeze(0), actual_flat[t].unsqueeze(0)
            ).squeeze(0)  # [K_pred, K_actual]

            # A predicted code is a hit if it's within radius of any actual code
            pred_hits = (dists.min(dim=1).values < radius).sum().item()
            # An actual code is covered if it's within radius of any predicted code
            actual_covered = (dists.min(dim=0).values < radius).sum().item()

            precision = pred_hits / K_pred if K_pred > 0 else 0.0
            recall = actual_covered / K_actual if K_actual > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)

            if pred_hits > 0:
                hits += 1
            if actual_covered >= K_actual:
                perfect += 1

        metrics = {
            "precision": sum(precisions) / N if N > 0 else 0.0,
            "recall": sum(recalls) / N if N > 0 else 0.0,
            "hit_rate": hits / N if N > 0 else 0.0,
            "coverage": perfect / N if N > 0 else 0.0,
            "over_prefetch_ratio": 1.0 - (sum(precisions) / N) if N > 0 else 0.0,
        }

        self._records[layer_idx].append(metrics)
        if len(self._records[layer_idx]) > self.window:
            self._records[layer_idx] = self._records[layer_idx][-self.window:]

        self._total_predictions += N
        self._total_hits += hits
        self._total_perfect += perfect

        return metrics

    def get_layer_metrics(self, layer_idx: int) -> Dict[str, float]:
        """Get rolling-average metrics for a specific layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Dictionary of averaged metrics over the rolling window.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )

        records = self._records[layer_idx]
        if not records:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "hit_rate": 0.0,
                "coverage": 0.0,
                "over_prefetch_ratio": 0.0,
            }

        avg = {}
        for key in records[0]:
            avg[key] = sum(r[key] for r in records) / len(records)
        return avg

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics across all layers.

        Returns:
            Dictionary of globally-averaged metrics.
        """
        all_records = []
        for layer_records in self._records.values():
            all_records.extend(layer_records)

        if not all_records:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "hit_rate": 0.0,
                "coverage": 0.0,
                "over_prefetch_ratio": 0.0,
                "total_predictions": 0,
            }

        avg = {}
        for key in all_records[0]:
            avg[key] = sum(r[key] for r in all_records) / len(all_records)
        avg["total_predictions"] = self._total_predictions
        return avg

    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all tracking data.

        Returns:
            Dictionary with per-layer metrics, aggregate metrics,
            and global counters.
        """
        return {
            "per_layer": {
                f"layer_{i}": self.get_layer_metrics(i)
                for i in range(self.n_layers)
            },
            "aggregate": self.get_aggregate_metrics(),
            "total_predictions": self._total_predictions,
            "total_hits": self._total_hits,
            "total_perfect": self._total_perfect,
            "global_hit_rate": (
                self._total_hits / self._total_predictions
                if self._total_predictions > 0
                else 0.0
            ),
            "global_coverage": (
                self._total_perfect / self._total_predictions
                if self._total_predictions > 0
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all tracking data."""
        self._records = {layer: [] for layer in range(self.n_layers)}
        self._total_predictions = 0
        self._total_hits = 0
        self._total_perfect = 0


# ============================================================================
# ExpertPrefetcher — Main orchestrator for expert prefetching
# ============================================================================


class ExpertPrefetcher(nn.Module):
    """Speculating Experts — Orchestrates expert prefetching for MoE
    inference acceleration.

    Manages a collection of LightweightPredictor modules (one per layer
    transition) and coordinates their use during both training and
    inference.  During training, predictors learn to predict the routing
    decisions of the next layer from the current layer's hidden states.
    During inference, predictors are used to speculate which experts
    will be needed, enabling early (prefetch) loading.

    Usage — Training
    -----------------
    ::

        prefetcher = ExpertPrefetcher(config)

        for batch in dataloader:
            # Run the MoE model forward
            outputs = model(batch)

            # For each layer, train the predictor
            for layer_idx in range(config.n_layers - 1):
                hidden_states_l = model.get_layer_hidden_states(layer_idx)
                routing_info_l1 = model.get_layer_routing(layer_idx + 1)

                loss = prefetcher.train_step(
                    layer_idx, hidden_states_l, routing_info_l1
                )
                loss.backward()

    Usage — Inference
    ------------------
    ::

        prefetcher = ExpertPrefetcher(config)
        prefetcher.eval()

        for layer_idx in range(config.n_layers):
            hidden_states = model.get_layer_hidden_states(layer_idx)

            if layer_idx < config.n_layers - 1:
                # Speculate experts for the next layer
                predicted = prefetcher.predict(layer_idx, hidden_states)

                # Issue async prefetch
                model.prefetch_experts(layer_idx + 1, predicted)

            # ... compute current layer (prefetch overlaps with this) ...

    Args:
        config: :class:`PrefetchConfig` instance.

    References:
        "Speculating Experts Accelerates Inference for MoE"
        (arXiv:2603.19289, Sections 3–4)
    """

    def __init__(self, config: PrefetchConfig) -> None:
        super().__init__()
        self.config = config

        # Create per-layer predictors (layer L predicts routing for L+1)
        # We don't need a predictor for the last layer
        self.predictors = nn.ModuleList()
        for _ in range(config.n_layers - 1):
            self.predictors.append(
                LightweightPredictor(
                    d_model=config.d_model,
                    hidden_dim=config.predictor_hidden_dim,
                    n_experts=config.n_experts if not config.infinite_moe_mode else 0,
                    top_k=config.top_k,
                    code_dim=config.code_dim,
                    dropout=config.predictor_dropout,
                    infinite_moe_mode=config.infinite_moe_mode,
                )
            )

        # Accuracy tracker
        self._tracker: Optional[PrefetchAccuracyTracker] = None
        if config.track_accuracy:
            self._tracker = PrefetchAccuracyTracker(
                n_layers=config.n_layers,
                window=config.accuracy_window,
                top_k=config.top_k,
            )

        # Adaptive temperature state (per-layer)
        self._current_temperatures: Optional[List[float]] = None
        if config.adaptive_temperature:
            self._current_temperatures = [
                config.prediction_temperature
                for _ in range(config.n_layers - 1)
            ]

        # Predictor optimizer (created lazily on first train_step)
        self._optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        """Lazily create the predictor optimizer."""
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self.predictors.parameters(),
                lr=self.config.train_predictor_lr,
                weight_decay=self.config.train_predictor_weight_decay,
            )
        return self._optimizer

    def compute_loss(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        target_routing: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the predictor training loss for a single layer transition.

        For **finite MoE** (discrete expert indices):
          - ``"ce"`` (cross-entropy): Treats expert prediction as a
            multi-class classification problem.  ``target_routing`` is a
            tensor of expert indices ``[batch, seq, top_k]``.
          - ``"bce"`` (binary cross-entropy): Multi-label classification
            where each expert is an independent binary target.
            ``target_routing`` is a binary indicator tensor
            ``[batch, seq, n_experts]``.

        For **∞-MoE** (continuous expert codes):
          - ``"mse"``: Mean squared error between predicted and actual
            expert codes.
          - ``"cosine"``: Cosine embedding loss encouraging predicted
            codes to align with actual codes in direction.

        Args:
            layer_idx: Index of the source layer (predicts routing for
                layer ``layer_idx + 1``).  Must be in ``[0, n_layers - 2]``.
            hidden_states: Hidden states from layer ``layer_idx``.
                Shape ``[batch_size, seq_len, d_model]``.
            target_routing: Ground-truth routing information for layer
                ``layer_idx + 1``.
                Finite MoE (``"ce"``): ``[batch, seq, top_k]`` long tensor.
                Finite MoE (``"bce"``): ``[batch, seq, n_experts]`` float
                tensor with 1.0 for selected experts.
                ∞-MoE (``"mse"``/``"cosine"``): ``[batch, seq, top_k, code_dim]``
                float tensor of expert codes.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: If ``layer_idx`` is out of range.
        """
        if layer_idx < 0 or layer_idx >= len(self.predictors):
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {len(self.predictors)})"
            )

        predictor = self.predictors[layer_idx]
        predictions = predictor(hidden_states)

        loss_type = self.config.train_loss_type

        if loss_type == "ce":
            # Cross-entropy over discrete experts
            # predictions: [B, S, n_experts] — logits
            # target_routing: [B, S, top_k] — expert indices
            B, S, n_experts = predictions.shape
            logits = predictions.reshape(B * S, n_experts)
            targets = target_routing.reshape(B * S, -1)  # [BS, top_k]

            # Multi-target cross-entropy: average CE over all top_k targets
            loss = F.cross_entropy(logits, targets[:, 0], reduction="mean")
            for k in range(1, targets.shape[1]):
                loss = loss + F.cross_entropy(
                    logits, targets[:, k], reduction="mean"
                )
            loss = loss / targets.shape[1]

        elif loss_type == "bce":
            # Multi-label binary cross-entropy
            # predictions: [B, S, n_experts] — logits
            # target_routing: [B, S, n_experts] — binary indicators
            loss = F.binary_cross_entropy_with_logits(
                predictions, target_routing.float(), reduction="mean"
            )

        elif loss_type == "mse":
            # MSE between predicted and actual expert codes
            # predictions: [B, S, top_k, code_dim]
            # target_routing: [B, S, top_k, code_dim]
            loss = F.mse_loss(predictions, target_routing, reduction="mean")

        elif loss_type == "cosine":
            # Cosine embedding loss
            # predictions: [B, S, top_k, code_dim]
            # target_routing: [B, S, top_k, code_dim]
            B, S, K, D = predictions.shape
            pred_flat = predictions.reshape(-1, D)
            target_flat = target_routing.reshape(-1, D)
            # Cosine loss: 1 - cos_sim (minimised when aligned)
            cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1)
            loss = (1.0 - cos_sim).mean()

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return loss

    def train_step(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        target_routing: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single training step for a layer's predictor.

        Computes the loss, backpropagates, and updates predictor weights.

        Args:
            layer_idx: Source layer index.
            hidden_states: Hidden states from layer ``layer_idx``.
                Shape ``[batch_size, seq_len, d_model]``.
            target_routing: Ground-truth routing for layer ``layer_idx + 1``.

        Returns:
            Scalar loss value (detached).
        """
        optimizer = self._ensure_optimizer()

        loss = self.compute_loss(layer_idx, hidden_states, target_routing)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach()

    def train_step_all_layers(
        self,
        hidden_states_per_layer: List[torch.Tensor],
        routing_per_layer: List[torch.Tensor],
    ) -> Dict[str, float]:
        """Train predictors for all layer transitions in one call.

        Convenience method that iterates over all layers and trains
        each predictor.

        Args:
            hidden_states_per_layer: List of hidden states for layers
                0 through ``n_layers - 2``.  Each tensor has shape
                ``[batch_size, seq_len, d_model]``.
            routing_per_layer: List of routing targets for layers
                1 through ``n_layers - 1``.  Each tensor format depends
                on the loss type (see :meth:`compute_loss`).

        Returns:
            Dictionary mapping ``"layer_{i}"`` to the loss value for
            each layer transition, plus ``"total"`` for the sum.
        """
        optimizer = self._ensure_optimizer()
        total_loss = torch.tensor(0.0, device=hidden_states_per_layer[0].device)
        losses: Dict[str, float] = {}

        for i in range(len(self.predictors)):
            loss = self.compute_loss(
                i, hidden_states_per_layer[i], routing_per_layer[i + 1]
            )
            total_loss = total_loss + loss
            losses[f"layer_{i}"] = loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses["total"] = total_loss.item()
        return losses

    # ------------------------------------------------------------------
    # Inference — Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Predict which experts will be needed at layer ``layer_idx + 1``.

        For **finite MoE**, returns predicted expert indices.
        For **∞-MoE**, returns predicted expert codes.

        Args:
            layer_idx: Source layer index (predicts for ``layer_idx + 1``).
                Must be in ``[0, n_layers - 2]``.
            hidden_states: Hidden states from layer ``layer_idx``.
                Shape ``[batch_size, seq_len, d_model]``.
            temperature: Override prediction temperature.  If None, uses
                the configured temperature (or adaptive temperature if
                enabled).

        Returns:
            Finite MoE mode:
                Predicted expert indices of shape
                ``[batch_size, seq_len, prefetch_budget]``.
            ∞-MoE mode:
                Predicted expert codes of shape
                ``[batch_size, seq_len, top_k, code_dim]``.

        Raises:
            ValueError: If ``layer_idx`` is out of range.
            RuntimeError: If called in training mode.
        """
        if self.training:
            raise RuntimeError(
                "predict() should only be called during inference. "
                "Use compute_loss() or train_step() during training."
            )
        if layer_idx < 0 or layer_idx >= len(self.predictors):
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {len(self.predictors)})"
            )

        # Determine temperature
        temp = temperature
        if temp is None:
            if self.config.adaptive_temperature and self._current_temperatures is not None:
                temp = self._current_temperatures[layer_idx]
            else:
                temp = self.config.prediction_temperature

        predictor = self.predictors[layer_idx]
        output = predictor(hidden_states)

        if self.config.infinite_moe_mode:
            # ∞-MoE: output is already [B, S, top_k, code_dim]
            return output
        else:
            # Finite MoE: output is logits [B, S, n_experts]
            budget = self.config.effective_prefetch_budget

            if temp is not None and temp > 0.0:
                # Sample from scaled softmax (Gumbel-like)
                scaled_logits = output / temp
                probs = F.softmax(scaled_logits, dim=-1)
                # Sample budget experts per token
                B, S, n_experts = output.shape
                sampled_indices = torch.multinomial(
                    probs.reshape(B * S, n_experts),
                    num_samples=budget,
                    replacement=False,
                )  # [B*S, budget]
                return sampled_indices.reshape(B, S, budget)
            else:
                # Greedy top-k
                _, top_indices = output.topk(budget, dim=-1)
                return top_indices  # [B, S, budget]

    @torch.no_grad()
    def predict_all_layers(
        self,
        hidden_states_per_layer: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Predict experts for all layer transitions.

        Args:
            hidden_states_per_layer: Hidden states for layers
                0 through ``n_layers - 2``.

        Returns:
            List of prediction tensors, one per layer transition.
        """
        predictions = []
        for i in range(len(self.predictors)):
            pred = self.predict(i, hidden_states_per_layer[i])
            predictions.append(pred)
        return predictions

    # ------------------------------------------------------------------
    # Inference — Accuracy tracking
    # ------------------------------------------------------------------

    def record_accuracy(
        self,
        layer_idx: int,
        predicted_experts: torch.Tensor,
        actual_experts: torch.Tensor,
    ) -> Optional[Dict[str, float]]:
        """Record prediction accuracy for monitoring and adaptive strategies.

        Must be called during inference after both the prediction and the
        actual routing decision are available.

        Args:
            layer_idx: Source layer index (predicts for ``layer_idx + 1``).
            predicted_experts: Predicted expert indices or codes.
            actual_experts: Ground-truth expert indices or codes.

        Returns:
            Dictionary of metrics for this update, or None if tracking
            is disabled.
        """
        if self._tracker is None:
            return None

        if self.config.infinite_moe_mode:
            return self._tracker.update_infinite_moe(
                layer_idx,
                predicted_experts,
                actual_experts,
                radius=self.config.prefetch_code_radius,
            )
        else:
            return self._tracker.update(
                layer_idx, predicted_experts, actual_experts
            )

    def update_adaptive_temperature(self, layer_idx: int) -> None:
        """Update the adaptive temperature for a layer based on recent
        prediction accuracy.

        When recent accuracy is high, temperature is reduced (exploit
        confident predictions).  When accuracy is low, temperature is
        increased (explore more diverse experts).

        Args:
            layer_idx: Source layer index.

        Raises:
            ValueError: If adaptive temperature is not enabled.
        """
        if not self.config.adaptive_temperature:
            raise ValueError("Adaptive temperature is not enabled in config.")
        if self._current_temperatures is None:
            return
        if self._tracker is None:
            return

        metrics = self._tracker.get_layer_metrics(layer_idx)
        recall = metrics.get("recall", 0.5)

        cfg = self.config

        if recall > 0.8:
            # High accuracy: reduce temperature (exploit)
            self._current_temperatures[layer_idx] = max(
                cfg.adaptive_temp_min,
                self._current_temperatures[layer_idx] * cfg.adaptive_temp_decay,
            )
        elif recall < 0.5:
            # Low accuracy: increase temperature (explore)
            self._current_temperatures[layer_idx] = min(
                cfg.adaptive_temp_max,
                self._current_temperatures[layer_idx] / cfg.adaptive_temp_decay,
            )

    # ------------------------------------------------------------------
    # Inference — Prefetch coordination
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_prefetch_set(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Get the set of experts to prefetch for layer ``layer_idx + 1``.

        This is the primary entry point for the inference pipeline.
        After computing hidden states at layer L, call this method to
        determine which experts to start loading for layer L+1.

        The returned set can be fed directly into an expert-loading
        mechanism (e.g., a CUDA stream for async expert weight transfer,
        or an RPC for expert-parallel communication).

        Args:
            layer_idx: Current layer index (will prefetch for ``layer_idx + 1``).
            hidden_states: Current layer's hidden states.
                Shape ``[batch_size, seq_len, d_model]``.

        Returns:
            Finite MoE mode:
                Expert indices to prefetch, shape
                ``[batch_size, seq_len, prefetch_budget]``.
                Contains unique expert IDs to load.
            ∞-MoE mode:
                Expert codes to prefetch, shape
                ``[batch_size, seq_len, top_k, code_dim]``.
                Nearby codes in the continuous space should also be
                prefetched using ``prefetch_code_radius``.
        """
        return self.predict(layer_idx, hidden_states)

    def get_unique_prefetch_experts(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> Set[int]:
        """Get the unique set of expert IDs to prefetch for a layer
        (finite MoE mode only).

        Deduplicates across batch and sequence dimensions, returning
        only the unique expert IDs that need to be loaded.  This is
        useful for batched inference where multiple tokens may need
        the same expert.

        Args:
            layer_idx: Current layer index.
            hidden_states: Hidden states from the current layer.
                Shape ``[batch_size, seq_len, d_model]``.

        Returns:
            Set of unique expert IDs to prefetch.

        Raises:
            ValueError: If called in ∞-MoE mode.
        """
        if self.config.infinite_moe_mode:
            raise ValueError(
                "get_unique_prefetch_experts() is only for finite MoE mode. "
                "Use predict() for ∞-MoE mode."
            )

        predictions = self.predict(layer_idx, hidden_states)
        # predictions: [B, S, budget]
        unique_ids = set(predictions.reshape(-1).tolist())
        return unique_ids

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_prefetch_overlap_estimate(
        self,
        layer_idx: int,
        compute_time_ms: float,
        expert_load_time_ms: float,
    ) -> Dict[str, float]:
        """Estimate the latency savings from prefetching for a layer.

        Compares the baseline (sequential compute + load) with the
        overlapped (compute || prefetch) scenario, factoring in
        prediction accuracy.

        Args:
            layer_idx: Layer index.
            compute_time_ms: Time for expert computation at this layer (ms).
            expert_load_time_ms: Time to load one expert from off-chip (ms).

        Returns:
            Dictionary with estimated timing information:
                - ``"baseline_ms"``: No-prefetch latency (compute + load).
                - ``"prefetched_ms"``: Latency with perfect prefetching.
                - ``"estimated_ms"``: Latency with predicted accuracy.
                - ``"savings_ms"``: Estimated savings vs baseline.
                - ``"savings_pct"``: Savings as percentage of baseline.
        """
        budget = self.config.effective_prefetch_budget
        top_k = self.config.top_k

        # Baseline: compute then load top_k experts
        baseline_ms = compute_time_ms + top_k * expert_load_time_ms

        # Perfect prefetch: all experts loaded during compute
        prefetched_ms = max(compute_time_ms, budget * expert_load_time_ms)

        # Estimated with accuracy
        if self._tracker is not None:
            metrics = self._tracker.get_layer_metrics(layer_idx)
            recall = metrics.get("recall", 0.0)
        else:
            recall = 0.5  # Default estimate

        # Fraction of experts already loaded (hits), rest loaded on-demand
        hits = recall * top_k
        misses = top_k - hits
        estimated_ms = max(
            compute_time_ms,  # Compute time (overlap with prefetch)
            budget * expert_load_time_ms,  # Prefetch time (overlapped)
        ) + misses * expert_load_time_ms  # On-demand loads for misses

        savings_ms = baseline_ms - estimated_ms
        savings_pct = (savings_ms / baseline_ms * 100) if baseline_ms > 0 else 0.0

        return {
            "baseline_ms": baseline_ms,
            "prefetched_ms": prefetched_ms,
            "estimated_ms": estimated_ms,
            "savings_ms": savings_ms,
            "savings_pct": savings_pct,
        }

    def parameter_count(self) -> Dict[str, int]:
        """Count predictor parameters.

        Returns:
            Dictionary with per-layer and total parameter counts.
        """
        counts: Dict[str, int] = {}
        total = 0
        for i, predictor in enumerate(self.predictors):
            n = sum(p.numel() for p in predictor.parameters())
            counts[f"layer_{i}"] = n
            total += n
        counts["total"] = total
        return counts

    def memory_footprint_bytes(self) -> int:
        """Estimate total memory footprint of all predictors in bytes.

        Returns:
            Total number of bytes used by predictor parameters.
        """
        total_params = sum(
            p.numel() for predictor in self.predictors for p in predictor.parameters()
        )
        # Assume float32 for parameters
        return total_params * 4

    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the prefetch system state.

        Returns:
            Dictionary with configuration, parameter counts, memory
            footprint, and accuracy tracking data.
        """
        summary: Dict[str, Any] = {
            "config": {
                "n_layers": self.config.n_layers,
                "d_model": self.config.d_model,
                "n_experts": self.config.n_experts,
                "top_k": self.config.top_k,
                "prefetch_budget": self.config.effective_prefetch_budget,
                "infinite_moe_mode": self.config.infinite_moe_mode,
                "prediction_temperature": self.config.prediction_temperature,
                "adaptive_temperature": self.config.adaptive_temperature,
                "loss_type": self.config.train_loss_type,
            },
            "parameters": self.parameter_count(),
            "memory_bytes": self.memory_footprint_bytes(),
            "memory_mb": self.memory_footprint_bytes() / (1024 * 1024),
        }

        if self._tracker is not None:
            summary["accuracy"] = self._tracker.get_summary()

        if self._current_temperatures is not None:
            summary["adaptive_temperatures"] = {
                f"layer_{i}": t for i, t in enumerate(self._current_temperatures)
            }

        return summary
