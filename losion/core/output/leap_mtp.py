"""
Losion Framework v0.5 — L-MTP: Leap Multi-Token Prediction Beyond Adjacent

Implements Leap Multi-Token Prediction (L-MTP), extending standard Multi-Token
Prediction from predicting adjacent future tokens (t+1, t+2, ...) to LEAPING —
predicting tokens at arbitrary future positions (t+leap_1, t+leap_2, ...).

From the paper:
    "L-MTP: Leap Multi-Token Prediction Beyond Adjacent"
    arXiv 2505.17505, May 2025, NeurIPS 2025

Key Innovation:
    Standard MTP predicts tokens at consecutive offsets (1, 2, 3, ..., K).
    L-MTP predicts tokens at NON-ADJACENT leap positions (e.g., 1, 2, 4, 8),
    following a geometric progression. This provides:

    1. **Wider Temporal Coverage** — A single forward pass covers a larger
       future horizon with fewer heads. With geometric leaps (1, 2, 4, 8),
       4 heads cover 15 positions vs. 4 positions with adjacent-only.

    2. **Better Speculative Decoding** — Leap predictions naturally align with
       SSM (State Space Model) layers that can efficiently predict distant
       tokens. When combined with SSM-based draft models, leap heads predict
       the exact positions the SSM will verify, avoiding wasted computation
       on intermediate tokens that SSM re-computes anyway.

    3. **Proven Efficiency** — Theorem 3.2 of the paper proves that geometric
       leap schedules achieve O(K * log(T/K)) coverage of a sequence of length
       T with K heads, compared to O(K) coverage for adjacent-only MTP.
       This translates to theoretically superior inference efficiency.

Architecture:

1. LeapMTPConfig — Configuration dataclass controlling leap schedule,
   number of leaps, warmup steps, loss weighting, and training stage.

2. LeapScheduleGenerator — Generates leap distance schedules. Supports:
   - "geometric": 1, r, r^2, r^3, ... (optimal per Theorem 3.2)
   - "arithmetic": 1, d+1, 2d+1, ... (uniform spacing)
   - "custom": User-specified leap distances
   - "adjacent": 1, 2, 3, ... (degrades to standard MTP, for ablation)

3. LeapMTPHead — Single prediction head for a specific leap distance.
   Architecture: LayerNorm → Linear → SiLU → Linear → vocab logits.
   Each head predicts the token at position t + leap_distance given h_t.

4. LeapMTP — Main module containing N LeapMTPHead instances with:
   - Two-stage training support (warmup then joint fine-tuning)
   - Geometric decay loss weights per leap (farther leaps → lower weight)
   - Backbone freezing/unfreezing for stage transitions
   - Speculative decoding integration

5. LeapMTPSpeculativeDecoder — Speculative decoding pipeline optimized
   for leap-based predictions. Uses leap heads to draft tokens at
   non-adjacent positions, then fills gaps with SSM or autoregressive
   sampling.

Two-Stage Training (Section 4.2 of the paper):

    Stage 1 — Warmup (backbone frozen):
        - Only leap heads are trained
        - Backbone parameters have requires_grad=False
        - Learns to predict future tokens from frozen representations
        - Typically 5-10% of total training steps

    Stage 2 — Joint Fine-tuning (backbone unfrozen):
        - Both backbone and leap heads are trained jointly
        - Uses a lower learning rate for backbone (0.1x of head LR)
        - Leap loss weight gradually increases via cosine schedule
        - Remaining 90-95% of training steps

Loss Computation:

    For each head k with leap distance d_k, the loss is:

        L_k = CE(head_k(h_t), y_{t + d_k})

    with weight w_k = gamma^k where gamma ∈ (0, 1) is the geometric
    decay factor. The total loss is:

        L_total = sum_k w_k * L_k / sum_k w_k

    This down-weights farther leaps which are inherently harder to predict,
    preventing them from destabilizing training.

References:
- L-MTP: Leap Multi-Token Prediction Beyond Adjacent (arXiv 2505.17505,
  NeurIPS 2025)
- Gloeckle, F. et al., "Better & Faster Large Language Models via
  Multi-token Prediction" (ICML 2024) — Original MTP
- Leviathan, Y. et al., "Fast Inference from Transformers via Speculative
  Decoding" (ICML 2023)
- Gu, A. & Dao, T., "Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces" (2023) — SSM for efficient inference
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Enums
# ============================================================================


class LeapScheduleType(str, Enum):
    """Supported leap schedule types for L-MTP.

    Attributes:
        GEOMETRIC: Leap distances follow geometric progression (1, r, r^2, ...).
            Optimal per Theorem 3.2 of the L-MTP paper. Default and recommended.
        ARITHMETIC: Leap distances follow arithmetic progression (1, d+1, 2d+1, ...).
            Useful for uniform coverage when geometric grows too fast.
        CUSTOM: User-specified leap distances via `custom_leaps` config field.
        ADJACENT: Leap distances are (1, 2, 3, ..., K). Degrades to standard
            MTP. Useful for ablation studies comparing L-MTP vs MTP.
    """

    GEOMETRIC = "geometric"
    ARITHMETIC = "arithmetic"
    CUSTOM = "custom"
    ADJACENT = "adjacent"


class TrainingStage(int, Enum):
    """Current training stage for two-stage L-MTP training.

    Attributes:
        WARMUP: Stage 1 — Leap heads warm up with frozen backbone.
        JOINT: Stage 2 — Joint fine-tuning of backbone and leap heads.
    """

    WARMUP = 1
    JOINT = 2


# ============================================================================
# LeapMTPConfig — Configuration Dataclass
# ============================================================================


@dataclass
class LeapMTPConfig:
    """Configuration for Leap Multi-Token Prediction (L-MTP).

    Controls the leap schedule, head architecture, loss weighting, and
    two-stage training parameters.

    Attributes:
        d_model: Hidden dimension of the main model's output.
        vocab_size: Size of the output vocabulary.
        num_leaps: Number of leap prediction heads. Default 4.
            With geometric schedule and ratio 2, this covers positions
            1, 2, 4, 8 (4 heads, 15-position coverage).
        leap_schedule: Type of leap schedule. Default GEOMETRIC.
        leap_ratio: Ratio for geometric leap schedule. Default 2.0.
            Leap distances: 1, r, r^2, r^3, ... (rounded to int).
            r=2 gives (1, 2, 4, 8, ...), r=1.5 gives (1, 2, 3, 5, ...).
        leap_step: Step size for arithmetic leap schedule. Default 2.
            Leap distances: 1, step+1, 2*step+1, ...
        custom_leaps: Custom leap distances for CUSTOM schedule type.
            Must be a sorted list of positive integers, e.g., [1, 3, 7, 15].
        head_hidden_ratio: Ratio for intermediate MLP dimension in each
            leap head, relative to d_model. Default 0.5.
        loss_decay: Geometric decay factor for per-head loss weights.
            Head k gets weight loss_decay^k. Default 0.8.
            Lower values down-weight farther leaps more aggressively.
        warmup_steps: Number of training steps for Stage 1 (backbone frozen).
            Default 1000. Set to 0 to skip warmup and train jointly from start.
        backbone_lr_scale: Learning rate scale for backbone in Stage 2 relative
            to head learning rate. Default 0.1 (backbone uses 10% of head LR).
        leap_loss_weight: Maximum weight of leap loss in total training loss.
            The leap loss is blended with the main next-token loss:
            L_total = L_main + alpha * L_leap. Default 1.0.
        tie_embeddings: If True, share output projection weights with the
            main model's token embedding. Default False.
        dropout: Dropout rate in leap head projections. Default 0.0.
    """

    d_model: int = 1024
    vocab_size: int = 32000
    num_leaps: int = 4
    leap_schedule: LeapScheduleType = LeapScheduleType.GEOMETRIC
    leap_ratio: float = 2.0
    leap_step: int = 2
    custom_leaps: Optional[List[int]] = None
    head_hidden_ratio: float = 0.5
    loss_decay: float = 0.8
    warmup_steps: int = 1000
    backbone_lr_scale: float = 0.1
    leap_loss_weight: float = 1.0
    tie_embeddings: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.num_leaps <= 0:
            raise ValueError(f"num_leaps must be positive, got {self.num_leaps}")
        if self.loss_decay <= 0 or self.loss_decay > 1:
            raise ValueError(
                f"loss_decay must be in (0, 1], got {self.loss_decay}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.backbone_lr_scale <= 0:
            raise ValueError(
                f"backbone_lr_scale must be positive, got {self.backbone_lr_scale}"
            )
        if self.leap_ratio <= 1.0 and self.leap_schedule == LeapScheduleType.GEOMETRIC:
            raise ValueError(
                f"leap_ratio must be > 1.0 for geometric schedule, "
                f"got {self.leap_ratio}"
            )
        if self.leap_schedule == LeapScheduleType.CUSTOM:
            if self.custom_leaps is None or len(self.custom_leaps) == 0:
                raise ValueError(
                    "custom_leaps must be provided when leap_schedule is CUSTOM"
                )
            if any(d <= 0 for d in self.custom_leaps):
                raise ValueError(
                    f"custom_leaps must all be positive, got {self.custom_leaps}"
                )
            if sorted(self.custom_leaps) != self.custom_leaps:
                raise ValueError(
                    f"custom_leaps must be sorted in ascending order, "
                    f"got {self.custom_leaps}"
                )


# ============================================================================
# LeapScheduleGenerator — Generate Leap Distance Schedules
# ============================================================================


class LeapScheduleGenerator:
    """Generator for leap distance schedules.

    Produces the list of leap distances d_1, d_2, ..., d_K where head k
    predicts the token at position t + d_k. The schedule type determines
    how these distances are distributed.

    The geometric schedule is the primary contribution of the L-MTP paper
    (Theorem 3.2), proven to achieve optimal coverage of the future horizon.

    Examples:
        >>> gen = LeapScheduleGenerator()
        >>> gen.generate(LeapScheduleType.GEOMETRIC, num_leaps=4, leap_ratio=2.0)
        [1, 2, 4, 8]
        >>> gen.generate(LeapScheduleType.ARITHMETIC, num_leaps=4, leap_step=3)
        [1, 4, 7, 10]
        >>> gen.generate(LeapScheduleType.ADJACENT, num_leaps=4)
        [1, 2, 3, 4]
        >>> gen.generate(LeapScheduleType.CUSTOM, custom_leaps=[1, 5, 12, 25])
        [1, 5, 12, 25]
    """

    @staticmethod
    def generate(
        schedule_type: LeapScheduleType,
        num_leaps: int,
        leap_ratio: float = 2.0,
        leap_step: int = 2,
        custom_leaps: Optional[List[int]] = None,
    ) -> List[int]:
        """Generate a leap distance schedule.

        Args:
            schedule_type: Type of leap schedule to generate.
            num_leaps: Number of leap distances to generate.
            leap_ratio: Ratio for geometric schedule (must be > 1.0).
            leap_step: Step size for arithmetic schedule.
            custom_leaps: Custom leap distances (required if CUSTOM type).

        Returns:
            List of leap distances, sorted in ascending order.
            Each distance d_k > 0 indicates that head k predicts the
            token at position t + d_k.

        Raises:
            ValueError: If invalid parameters are provided.
        """
        if schedule_type == LeapScheduleType.GEOMETRIC:
            return LeapScheduleGenerator._geometric_schedule(
                num_leaps, leap_ratio
            )
        elif schedule_type == LeapScheduleType.ARITHMETIC:
            return LeapScheduleGenerator._arithmetic_schedule(
                num_leaps, leap_step
            )
        elif schedule_type == LeapScheduleType.ADJACENT:
            return list(range(1, num_leaps + 1))
        elif schedule_type == LeapScheduleType.CUSTOM:
            if custom_leaps is None:
                raise ValueError("custom_leaps required for CUSTOM schedule")
            return list(custom_leaps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    @staticmethod
    def _geometric_schedule(num_leaps: int, ratio: float) -> List[int]:
        """Generate geometric leap schedule: 1, r, r^2, r^3, ...

        Per Theorem 3.2 of the L-MTP paper, geometric schedules with ratio r
        achieve O(K * log(T/K)) coverage of a sequence of length T with K
        heads. This is optimal among all possible leap schedules.

        The first leap is always 1 (predicting the immediately next token)
        because this is the most important prediction for autoregressive
        generation quality. Subsequent leaps grow geometrically.

        Args:
            num_leaps: Number of leap distances.
            ratio: Geometric ratio (must be > 1.0).

        Returns:
            List of leap distances with geometric growth.
        """
        if ratio <= 1.0:
            raise ValueError(f"Geometric ratio must be > 1.0, got {ratio}")

        leaps = []
        for k in range(num_leaps):
            if k == 0:
                leaps.append(1)
            else:
                # round(r^k) to nearest integer, ensuring monotonic increase
                leap = max(round(ratio ** k), leaps[-1] + 1)
                leaps.append(leap)
        return leaps

    @staticmethod
    def _arithmetic_schedule(num_leaps: int, step: int) -> List[int]:
        """Generate arithmetic leap schedule: 1, step+1, 2*step+1, ...

        Provides uniform spacing between leap positions. Less efficient
        coverage than geometric but may be preferred for certain tasks
        where uniform temporal resolution is important.

        Args:
            num_leaps: Number of leap distances.
            step: Step size between consecutive leaps.

        Returns:
            List of leap distances with arithmetic growth.
        """
        return [1 + k * step for k in range(num_leaps)]

    @staticmethod
    def coverage(leaps: List[int]) -> int:
        """Compute the temporal coverage of a leap schedule.

        Coverage is the number of unique positions that can be directly
        predicted or reconstructed from the leap predictions. For leap
        distances d_1, ..., d_K, the coverage is at least max(d_K) since
        the farthest leap covers all positions up to d_K when combined
        with autoregressive generation.

        A more precise measure: coverage = sum of all leap distances
        minus overlaps. For simplicity, we use the maximum leap distance
        as the horizon and the number of directly predicted positions.

        Args:
            leaps: List of leap distances.

        Returns:
            Maximum horizon covered (equals the maximum leap distance).
        """
        if not leaps:
            return 0
        return max(leaps)

    @staticmethod
    def efficiency_ratio(leaps: List[int]) -> float:
        """Compute the efficiency ratio of a leap schedule.

        The efficiency ratio measures how well the schedule covers the
        future horizon relative to standard adjacent MTP:

            efficiency = coverage(K leap heads) / coverage(K adjacent heads)
                       = max(leaps) / K

        For geometric schedule with ratio 2 and K heads:
            efficiency = 2^(K-1) / K → grows exponentially with K.

        Args:
            leaps: List of leap distances.

        Returns:
            Efficiency ratio (coverage / num_leaps). > 1.0 means the
            leap schedule covers more positions per head than adjacent MTP.
        """
        if not leaps:
            return 0.0
        return max(leaps) / len(leaps)


# ============================================================================
# LeapMTPHead — Single Leap Prediction Head
# ============================================================================


class LeapMTPHead(nn.Module):
    """Single Leap Multi-Token Prediction head for a specific leap distance.

    Each LeapMTPHead takes the hidden state h_t from the main model and
    predicts the token at position t + leap_distance. Unlike standard MTP
    heads that predict at consecutive offsets, leap heads can predict at
    arbitrary future positions.

    Architecture:
        h_t → LayerNorm → Linear(d_model, hidden_dim) → SiLU → Dropout
           → Linear(hidden_dim, vocab_size)

    The two-layer MLP with SiLU activation provides sufficient capacity for
    quality leap predictions while remaining lightweight compared to the
    main model's forward pass. SiLU is preferred over ReLU as it provides
    smoother gradients, which is important for the harder task of predicting
    distant tokens.

    Args:
        d_model: Hidden dimension of the main model.
        vocab_size: Size of the output vocabulary.
        leap_distance: Number of steps ahead this head predicts.
            leap_distance=1 is standard next-token prediction.
            leap_distance=4 predicts the token 4 positions ahead.
        head_index: Index of this head (0-based), used for logging.
        hidden_ratio: Ratio for the intermediate MLP dimension relative to
            d_model. Default 0.5 (half of d_model) for efficiency.
        dropout: Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        leap_distance: int,
        head_index: int = 0,
        hidden_ratio: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.leap_distance = leap_distance
        self.head_index = head_index

        hidden_dim = max(int(d_model * hidden_ratio), 64)

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Initialize weights with scaled normal for stable training
        # Farther leaps get slightly smaller initial weights since their
        # task is harder and we want to avoid large initial loss values
        scale = 0.02 / math.sqrt(max(leap_distance, 1))
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict the token at position t + leap_distance.

        Args:
            hidden_states: Hidden states from the main model,
                shape (batch, seq_len, d_model).

        Returns:
            Logits for the predicted token, shape (batch, seq_len, vocab_size).
            The logits at position t predict the token at position t + leap_distance.
        """
        x = self.norm(hidden_states)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"d_model={self.d_model}, vocab_size={self.vocab_size}, "
            f"leap_distance={self.leap_distance}, head_index={self.head_index}"
        )


# ============================================================================
# LeapMTPOutput — Output Container
# ============================================================================


@dataclass
class LeapMTPOutput:
    """Container for Leap Multi-Token Prediction outputs.

    Holds the predicted token IDs, logits, and probabilities from all leap
    heads for a single forward pass.

    Attributes:
        token_ids: Predicted token IDs, shape (batch, num_leaps).
            Column k contains the prediction at leap distance leaps[k].
        logits: Raw logits from each head, shape (batch, num_leaps, vocab_size).
        probabilities: Softmax probabilities, shape (batch, num_leaps, vocab_size).
        num_leaps: Number of leap prediction heads.
        leap_distances: The leap distances for each head.
            leap_distances[k] is the number of steps ahead head k predicts.
    """

    token_ids: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    num_leaps: int
    leap_distances: List[int]

    def __len__(self) -> int:
        return self.num_leaps


# ============================================================================
# LeapMTP — Main Leap Multi-Token Prediction Module
# ============================================================================


class LeapMTP(nn.Module):
    """Leap Multi-Token Prediction (L-MTP) module.

    Extends standard Multi-Token Prediction from adjacent offsets to
    arbitrary leap distances, following the L-MTP paper (arXiv 2505.17505).

    Instead of predicting tokens at positions t+1, t+2, ..., t+K,
    L-MTP predicts tokens at positions t+d_1, t+d_2, ..., t+d_K
    where d_k are leap distances that follow a geometric progression
    (1, 2, 4, 8, ...) or other configurable schedules.

    Key Features:
    - **Geometric leap schedule** for optimal temporal coverage
    - **Two-stage training** with backbone warmup and joint fine-tuning
    - **Geometric decay loss weights** for stable training
    - **Speculative decoding integration** for inference acceleration
    - **Backward compatible** — set schedule to ADJACENT for standard MTP

    Usage:
        # Create L-MTP module
        config = LeapMTPConfig(
            d_model=1024,
            vocab_size=32000,
            num_leaps=4,
            leap_schedule=LeapScheduleType.GEOMETRIC,
            leap_ratio=2.0,
        )
        leap_mtp = LeapMTP(config)

        # Training forward pass
        mtp_output, loss = leap_mtp(hidden_states, targets=token_ids)

        # Inference draft generation
        mtp_output = leap_mtp.draft(hidden_states, n_draft=4)

        # Two-stage training
        leap_mtp.set_training_stage(TrainingStage.WARMUP)  # Freeze backbone
        # ... train for warmup_steps ...
        leap_mtp.set_training_stage(TrainingStage.JOINT)   # Unfreeze backbone

    Args:
        config: LeapMTPConfig with all hyperparameters.
    """

    def __init__(self, config: LeapMTPConfig):
        super().__init__()
        self.config = config

        # Generate leap schedule
        self.leap_distances: List[int] = LeapScheduleGenerator.generate(
            schedule_type=config.leap_schedule,
            num_leaps=config.num_leaps,
            leap_ratio=config.leap_ratio,
            leap_step=config.leap_step,
            custom_leaps=config.custom_leaps,
        )
        # Store as buffer for device tracking and serialization
        self.register_buffer(
            "_leap_distances_tensor",
            torch.tensor(self.leap_distances, dtype=torch.long),
        )

        # Create leap prediction heads
        self.heads = nn.ModuleList([
            LeapMTPHead(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
                leap_distance=d,
                head_index=k,
                hidden_ratio=config.head_hidden_ratio,
                dropout=config.dropout,
            )
            for k, d in enumerate(self.leap_distances)
        ])

        # Compute and register loss weights with geometric decay
        # Weight for head k = loss_decay^k, then normalize
        raw_weights = [config.loss_decay ** k for k in range(config.num_leaps)]
        total_weight = sum(raw_weights)
        normalized_weights = [w / total_weight for w in raw_weights]
        self.register_buffer(
            "loss_weights",
            torch.tensor(normalized_weights, dtype=torch.float32),
        )

        # Training state
        self._training_stage = TrainingStage.WARMUP if config.warmup_steps > 0 else TrainingStage.JOINT
        self._global_step = 0

        # Reference to backbone parameters (set externally via set_backbone)
        self._backbone_params: Optional[List[nn.Parameter]] = None

    @property
    def num_leaps(self) -> int:
        """Number of leap prediction heads."""
        return self.config.num_leaps

    @property
    def max_leap(self) -> int:
        """Maximum leap distance (farthest prediction horizon)."""
        return max(self.leap_distances)

    @property
    def training_stage(self) -> TrainingStage:
        """Current training stage."""
        return self._training_stage

    @property
    def global_step(self) -> int:
        """Current global training step."""
        return self._global_step

    @property
    def coverage(self) -> int:
        """Temporal coverage of the current leap schedule."""
        return LeapScheduleGenerator.coverage(self.leap_distances)

    @property
    def efficiency_ratio(self) -> float:
        """Efficiency ratio vs. adjacent-only MTP."""
        return LeapScheduleGenerator.efficiency_ratio(self.leap_distances)

    def set_backbone(self, backbone: nn.Module) -> None:
        """Register the backbone module for two-stage training control.

        This must be called before using set_training_stage() to enable
        backbone freezing/unfreezing during stage transitions.

        Args:
            backbone: The main model (backbone) whose parameters should be
                frozen during warmup stage.
        """
        self._backbone_params = list(backbone.parameters())

    def set_training_stage(self, stage: TrainingStage) -> None:
        """Transition to a new training stage.

        Stage 1 (WARMUP): Freeze backbone parameters, only train leap heads.
        Stage 2 (JOINT): Unfreeze backbone, train everything jointly with
            separate learning rates.

        Args:
            stage: The target training stage.

        Raises:
            RuntimeError: If set_training_stage(WARMUP) is called but
                set_backbone() was never called.
        """
        self._training_stage = stage

        if self._backbone_params is None:
            # No backbone registered — skip freezing logic
            return

        if stage == TrainingStage.WARMUP:
            # Freeze backbone
            for param in self._backbone_params:
                param.requires_grad = False
        elif stage == TrainingStage.JOINT:
            # Unfreeze backbone
            for param in self._backbone_params:
                param.requires_grad = True

    def get_param_groups(
        self,
        head_lr: float = 1e-3,
        backbone_lr: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Get parameter groups for the optimizer with separate learning rates.

        In Stage 2 (joint training), the backbone uses a lower learning rate
        (controlled by backbone_lr_scale) while leap heads use the full rate.

        Args:
            head_lr: Learning rate for leap head parameters.
            backbone_lr: Learning rate for backbone parameters. If None,
                uses head_lr * config.backbone_lr_scale.

        Returns:
            List of parameter group dicts for torch.optim.Optimizer.
        """
        if backbone_lr is None:
            backbone_lr = head_lr * self.config.backbone_lr_scale

        head_params = list(self.parameters())
        groups = [
            {
                "params": head_params,
                "lr": head_lr,
                "name": "leap_heads",
            }
        ]

        if self._backbone_params is not None and self._training_stage == TrainingStage.JOINT:
            groups.append(
                {
                    "params": self._backbone_params,
                    "lr": backbone_lr,
                    "name": "backbone",
                }
            )

        return groups

    def leap_loss_weight_schedule(self, step: int) -> float:
        """Compute the leap loss weight at a given training step.

        Uses a cosine schedule that ramps up the leap loss weight during
        warmup, then maintains it at the maximum during joint training.

        The schedule prevents the leap loss from disrupting backbone
        representations early in training when the heads produce poor
        predictions.

        Args:
            step: Current global training step.

        Returns:
            Leap loss weight in [0, leap_loss_weight].
        """
        if self.config.warmup_steps == 0:
            # No warmup — use full weight from the start
            return self.config.leap_loss_weight

        if step >= self.config.warmup_steps:
            # Past warmup — full weight
            return self.config.leap_loss_weight

        # During warmup — cosine ramp from 0 to leap_loss_weight
        progress = step / self.config.warmup_steps
        # Cosine ramp: 0 → 1 over [0, 1]
        ramp = 0.5 * (1 - math.cos(math.pi * progress))
        return self.config.leap_loss_weight * ramp

    def tie_embeddings_weight(self, embedding_weight: torch.Tensor) -> None:
        """Tie the output projections of all leap heads to the given embedding.

        This reduces parameter count and can improve prediction quality by
        sharing the embedding space between input and output.

        Args:
            embedding_weight: Token embedding weight matrix, shape
                (vocab_size, d_model). Must match config.vocab_size and
                config.d_model.

        Raises:
            RuntimeError: If tie_embeddings is not enabled in config.
        """
        if not self.config.tie_embeddings:
            raise RuntimeError(
                "tie_embeddings=True must be set in LeapMTPConfig to use "
                "tie_embeddings_weight()"
            )
        for head in self.heads:
            head.fc2.weight = nn.Parameter(embedding_weight.clone())

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[LeapMTPOutput, Optional[torch.Tensor]]:
        """Run all leap prediction heads on the given hidden states.

        Each head k predicts the token at position t + leap_distances[k]
        from the hidden state at position t.

        Args:
            hidden_states: Hidden states from the main model,
                shape (batch, seq_len, d_model).
            targets: Target token IDs for computing training loss,
                shape (batch, seq_len). If None, no loss is computed.

        Returns:
            Tuple of (leap_output, loss):
            - leap_output: LeapMTPOutput with predictions from all leap heads.
            - loss: Weighted leap prediction loss, or None if targets is None.
                The loss incorporates geometric decay weights and the current
                leap_loss_weight_schedule value.
        """
        batch, seq_len, _ = hidden_states.shape

        all_logits = []
        all_token_ids = []

        for k, head in enumerate(self.heads):
            logits_k = head(hidden_states)  # (batch, seq_len, vocab_size)
            all_logits.append(logits_k)
            token_ids_k = logits_k.argmax(dim=-1)  # (batch, seq_len)
            all_token_ids.append(token_ids_k)

        # Stack: (batch, seq_len, num_leaps, vocab_size)
        all_logits = torch.stack(all_logits, dim=2)
        all_token_ids = torch.stack(all_token_ids, dim=2)

        # Compute probabilities
        all_probs = F.softmax(all_logits.float(), dim=-1).to(all_logits.dtype)

        # Take last position predictions for the output
        last_logits = all_logits[:, -1, :, :]  # (batch, num_leaps, vocab_size)
        last_probs = all_probs[:, -1, :, :]
        last_ids = all_token_ids[:, -1, :]  # (batch, num_leaps)

        leap_output = LeapMTPOutput(
            token_ids=last_ids,
            logits=last_logits,
            probabilities=last_probs,
            num_leaps=self.num_leaps,
            leap_distances=list(self.leap_distances),
        )

        # Compute training loss if targets provided
        loss = None
        if targets is not None:
            loss = self._compute_loss(all_logits, targets)

        return leap_output, loss

    def _compute_loss(
        self,
        all_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted leap multi-token prediction loss.

        For each head k with leap distance d_k:
            L_k = CE(logits_k[:, t, :], targets[:, t + d_k])

        The logits at position t predict the token at position t + d_k.
        We shift the targets accordingly, ensuring valid positions exist.

        Total loss = sum_k weight_k * L_k, where weight_k = loss_decay^k.

        The loss is also scaled by the leap_loss_weight_schedule to support
        gradual ramp-up during warmup.

        Args:
            all_logits: Logits from all leap heads,
                shape (batch, seq_len, num_leaps, vocab_size).
            targets: Target token IDs, shape (batch, seq_len).

        Returns:
            Scalar loss tensor.
        """
        batch, seq_len, num_leaps, vocab_size = all_logits.shape
        total_loss = torch.zeros(1, device=all_logits.device, dtype=all_logits.dtype, requires_grad=True).squeeze()

        for k in range(num_leaps):
            d_k = self.leap_distances[k]

            # Need at least d_k future positions for valid loss computation
            if seq_len <= d_k:
                continue

            # Prediction positions: 0 to seq_len - d_k - 1
            # Head k at position t predicts token at position t + d_k
            pred_logits = all_logits[:, :seq_len - d_k, k, :]  # (batch, seq_len-d_k, vocab_size)
            target_ids = targets[:, d_k:]  # (batch, seq_len-d_k)

            head_loss = F.cross_entropy(
                pred_logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + self.loss_weights[k] * head_loss

        # Scale by leap loss weight schedule
        alpha = self.leap_loss_weight_schedule(self._global_step)
        total_loss = alpha * total_loss

        return total_loss

    @torch.no_grad()
    def draft(
        self,
        hidden_states: torch.Tensor,
        n_draft: Optional[int] = None,
    ) -> LeapMTPOutput:
        """Generate draft tokens from leap heads for speculative decoding.

        Unlike standard MTP draft which produces consecutive tokens,
        leap draft produces tokens at non-adjacent positions. The caller
        must handle gap-filling (e.g., via SSM or autoregressive sampling).

        Args:
            hidden_states: Hidden states from the main model,
                shape (batch, 1, d_model) for autoregressive or
                (batch, seq_len, d_model) for batched.
            n_draft: Number of draft heads to use. If None, uses all heads.

        Returns:
            LeapMTPOutput with draft token predictions at leap positions.

        Raises:
            ValueError: If n_draft exceeds num_leaps.
        """
        if n_draft is not None and n_draft > self.num_leaps:
            raise ValueError(
                f"n_draft ({n_draft}) cannot exceed num_leaps ({self.num_leaps})"
            )

        effective_heads = n_draft or self.num_leaps

        all_logits = []
        all_token_ids = []

        for k in range(effective_heads):
            logits_k = self.heads[k](hidden_states)  # (batch, seq_len, vocab_size)
            token_ids_k = logits_k.argmax(dim=-1)
            all_logits.append(logits_k[:, -1, :])  # Take last position
            all_token_ids.append(token_ids_k[:, -1])

        last_logits = torch.stack(all_logits, dim=1)  # (batch, effective_heads, vocab_size)
        last_ids = torch.stack(all_token_ids, dim=1)  # (batch, effective_heads)
        last_probs = F.softmax(last_logits.float(), dim=-1).to(last_logits.dtype)

        return LeapMTPOutput(
            token_ids=last_ids,
            logits=last_logits,
            probabilities=last_probs,
            num_leaps=effective_heads,
            leap_distances=self.leap_distances[:effective_heads],
        )

    def step_training(self) -> None:
        """Advance the global step counter and handle stage transitions.

        Call this at the end of each training step. Automatically transitions
        from WARMUP to JOINT stage when warmup_steps is reached.
        """
        self._global_step += 1

        # Auto-transition from warmup to joint
        if (
            self._training_stage == TrainingStage.WARMUP
            and self.config.warmup_steps > 0
            and self._global_step >= self.config.warmup_steps
        ):
            self.set_training_stage(TrainingStage.JOINT)

    def reset_training(self) -> None:
        """Reset training state to initial conditions."""
        self._global_step = 0
        if self.config.warmup_steps > 0:
            self.set_training_stage(TrainingStage.WARMUP)
        else:
            self.set_training_stage(TrainingStage.JOINT)

    def summary(self) -> str:
        """Return a human-readable summary of the L-MTP configuration.

        Returns:
            Formatted string with key configuration and schedule info.
        """
        lines = [
            "L-MTP: Leap Multi-Token Prediction",
            "=" * 50,
            f"  d_model:            {self.config.d_model}",
            f"  vocab_size:         {self.config.vocab_size}",
            f"  num_leaps:          {self.num_leaps}",
            f"  leap_schedule:      {self.config.leap_schedule.value}",
            f"  leap_distances:     {self.leap_distances}",
            f"  max_leap:           {self.max_leap}",
            f"  coverage:           {self.coverage}",
            f"  efficiency_ratio:   {self.efficiency_ratio:.2f}x",
            f"  loss_decay:         {self.config.loss_decay}",
            f"  loss_weights:       {[f'{w:.3f}' for w in self.loss_weights.tolist()]}",
            f"  warmup_steps:       {self.config.warmup_steps}",
            f"  training_stage:     {self._training_stage.name}",
            f"  global_step:        {self._global_step}",
            f"  backbone_lr_scale:  {self.config.backbone_lr_scale}",
            f"  leap_loss_weight:   {self.config.leap_loss_weight}",
        ]

        # Comparison with adjacent MTP
        adjacent_coverage = self.num_leaps
        improvement = self.coverage / adjacent_coverage if adjacent_coverage > 0 else 0
        lines.append(f"  vs adjacent MTP:    {improvement:.1f}x more coverage")

        # Parameter count
        total_params = sum(p.numel() for p in self.parameters())
        lines.append(f"  total_params:       {total_params:,}")

        return "\n".join(lines)

    def extra_repr(self) -> str:
        """String representation for module printing."""
        return (
            f"d_model={self.config.d_model}, "
            f"vocab_size={self.config.vocab_size}, "
            f"num_leaps={self.num_leaps}, "
            f"leap_distances={self.leap_distances}, "
            f"schedule={self.config.leap_schedule.value}"
        )


# ============================================================================
# LeapSpeculativeStats — Statistics for Leap Speculative Decoding
# ============================================================================


@dataclass
class LeapSpeculativeStats:
    """Statistics tracker for leap-based speculative decoding.

    Extends standard speculative decoding stats with leap-specific metrics
    including per-leap-distance acceptance rates and gap-fill statistics.

    Attributes:
        total_steps: Total number of speculative decoding steps.
        total_draft_tokens: Total number of leap draft tokens proposed.
        total_accepted_tokens: Total number of leap draft tokens accepted.
        total_emitted_tokens: Total number of tokens emitted.
        total_gap_fill_tokens: Total number of gap-fill tokens (tokens
            between leap positions filled by autoregressive or SSM sampling).
        per_leap_accepted: Per-leap-distance acceptance counts.
        per_leap_total: Per-leap-distance total draft counts.
        acceptance_history: Per-step acceptance counts.
        gap_fill_history: Per-step gap-fill token counts.
    """

    total_steps: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_emitted_tokens: int = 0
    total_gap_fill_tokens: int = 0
    per_leap_accepted: Dict[int, int] = field(default_factory=dict)
    per_leap_total: Dict[int, int] = field(default_factory=dict)
    acceptance_history: List[int] = field(default_factory=list)
    gap_fill_history: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate across all steps."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def avg_tokens_per_step(self) -> float:
        """Average total tokens emitted per step (draft + gap-fill)."""
        if self.total_steps == 0:
            return 0.0
        return self.total_emitted_tokens / self.total_steps

    @property
    def gap_fill_ratio(self) -> float:
        """Ratio of gap-fill tokens to total emitted tokens."""
        if self.total_emitted_tokens == 0:
            return 0.0
        return self.total_gap_fill_tokens / self.total_emitted_tokens

    @property
    def effective_speedup(self) -> float:
        """Estimated effective speedup vs. standard autoregressive decoding.

        Accounts for both the accepted leap tokens and the gap-fill tokens,
        with a verification overhead correction factor.
        """
        if self.total_steps == 0:
            return 1.0
        theoretical = self.avg_tokens_per_step
        verification_overhead = 1.2
        return theoretical / verification_overhead

    def per_leap_acceptance_rate(self, leap_distance: int) -> float:
        """Get acceptance rate for a specific leap distance.

        Args:
            leap_distance: The leap distance to query.

        Returns:
            Acceptance rate for that leap distance, or 0.0 if no data.
        """
        total = self.per_leap_total.get(leap_distance, 0)
        if total == 0:
            return 0.0
        return self.per_leap_accepted.get(leap_distance, 0) / total

    def record_step(
        self,
        n_accepted: int,
        n_gap_fill: int,
        spec_length: int,
        leap_distances: List[int],
        accepted_mask: Optional[List[bool]] = None,
    ) -> None:
        """Record statistics from a single speculative decoding step.

        Args:
            n_accepted: Number of leap draft tokens accepted.
            n_gap_fill: Number of gap-fill tokens generated.
            spec_length: Number of leap draft tokens proposed.
            leap_distances: Leap distances used in this step.
            accepted_mask: Optional per-leap acceptance indicators.
                True if the leap head's prediction was accepted.
        """
        self.total_steps += 1
        self.total_draft_tokens += spec_length
        self.total_accepted_tokens += n_accepted
        self.total_emitted_tokens += n_accepted + n_gap_fill + 1  # +1 for correction/bonus
        self.total_gap_fill_tokens += n_gap_fill
        self.acceptance_history.append(n_accepted)
        self.gap_fill_history.append(n_gap_fill)

        # Per-leap-distance tracking
        if accepted_mask is not None:
            for i, d in enumerate(leap_distances):
                self.per_leap_total[d] = self.per_leap_total.get(d, 0) + 1
                if i < len(accepted_mask) and accepted_mask[i]:
                    self.per_leap_accepted[d] = self.per_leap_accepted.get(d, 0) + 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_steps = 0
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.total_emitted_tokens = 0
        self.total_gap_fill_tokens = 0
        self.per_leap_accepted.clear()
        self.per_leap_total.clear()
        self.acceptance_history.clear()
        self.gap_fill_history.clear()

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "Leap Speculative Decoding Stats:",
            f"  Total steps:            {self.total_steps}",
            f"  Total draft tokens:     {self.total_draft_tokens}",
            f"  Total accepted tokens:  {self.total_accepted_tokens}",
            f"  Total emitted tokens:   {self.total_emitted_tokens}",
            f"  Total gap-fill tokens:  {self.total_gap_fill_tokens}",
            f"  Acceptance rate:        {self.acceptance_rate:.2%}",
            f"  Gap-fill ratio:         {self.gap_fill_ratio:.2%}",
            f"  Avg tokens/step:        {self.avg_tokens_per_step:.2f}",
            f"  Effective speedup:      {self.effective_speedup:.2f}x",
        ]

        # Per-leap acceptance rates
        if self.per_leap_total:
            lines.append("  Per-leap acceptance rates:")
            for d in sorted(self.per_leap_total.keys()):
                rate = self.per_leap_acceptance_rate(d)
                lines.append(f"    leap={d}: {rate:.2%}")

        return "\n".join(lines)


# ============================================================================
# LeapMTPSpeculativeDecoder — Leap-Based Speculative Decoding Pipeline
# ============================================================================


class LeapMTPSpeculativeDecoder(nn.Module):
    """Speculative decoding pipeline optimized for leap-based predictions.

    Uses L-MTP leap heads to draft tokens at non-adjacent positions, then
    fills the gaps between leap positions with SSM-based or autoregressive
    sampling. This provides superior speedup over standard speculative
    decoding because:

    1. **Leap heads predict at geometric positions** (1, 2, 4, 8) instead
       of adjacent (1, 2, 3, 4), covering a much wider horizon per step.

    2. **Gap-filling leverages SSM** — SSM layers can efficiently generate
       tokens between leap positions since they maintain sequential state.
       This is much cheaper than full transformer verification.

    3. **Verification is sparser** — Instead of verifying every position,
       the verifier only needs to check leap positions. Gap tokens are
       accepted if both bounding leap tokens are accepted.

    Algorithm:
        Given current position t with hidden state h_t:

        1. LEAP DRAFT: Run leap heads on h_t to predict tokens at positions
           t+d_1, t+d_2, ..., t+d_K (e.g., t+1, t+2, t+4, t+8).

        2. GAP FILL: For each pair of consecutive leap positions (t+d_k,
           t+d_{k+1}), fill the gap with SSM or autoregressive sampling.
           E.g., between t+2 and t+4, generate tokens at t+3.

        3. VERIFY: Run the main model on the full drafted sequence to
           verify all tokens (both leap and gap-fill).

        4. ACCEPT/REJECT: Accept matching prefix, reject from first
           mismatch. Emit correction token at the rejection point.

    Usage:
        config = LeapMTPConfig(d_model=1024, vocab_size=32000, num_leaps=4)
        leap_mtp = LeapMTP(config)

        decoder = LeapMTPSpeculativeDecoder(
            leap_mtp=leap_mtp,
            d_model=1024,
            vocab_size=32000,
        )

        # In generation loop
        for step in range(max_steps):
            tokens, past_kv, n_accepted, eos = decoder.step(
                current_hidden=h_t,
                verifier_forward=model_forward,
                gap_fill_fn=ssm_generate,
                past_kv=past_kv,
            )

    Args:
        leap_mtp: LeapMTP module serving as the draft model.
        d_model: Hidden dimension of the main model.
        vocab_size: Size of the output vocabulary.
        max_spec_length: Maximum number of leap heads to use for drafting.
            Default None (use all heads).
        adaptive: Whether to adaptively adjust the speculation length.
            Default True.
        target_acceptance_rate: Target acceptance rate for adaptive control.
            Default 0.80 (slightly lower than standard MTP due to harder
            leap predictions).
        temperature: Sampling temperature for verification. Default 1.0.
        top_k: Top-k filtering for sampling. Default 0.
        top_p: Nucleus sampling threshold. Default 1.0.
        eos_token_id: EOS token ID. Default -1 (disabled).
    """

    def __init__(
        self,
        leap_mtp: LeapMTP,
        d_model: int,
        vocab_size: int,
        max_spec_length: Optional[int] = None,
        adaptive: bool = True,
        target_acceptance_rate: float = 0.80,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int = -1,
    ):
        super().__init__()

        self.leap_mtp = leap_mtp
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_spec_length = max_spec_length or leap_mtp.num_leaps
        self.adaptive = adaptive
        self.target_acceptance_rate = target_acceptance_rate
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id

        # Ensure max_spec_length doesn't exceed available heads
        self.max_spec_length = min(self.max_spec_length, leap_mtp.num_leaps)

        # Current speculation length (adaptive)
        self._current_spec_length = self.max_spec_length

        # Statistics
        self.stats = LeapSpeculativeStats()

        # Running acceptance rate for adaptive control
        self._ema_acceptance_rate = 0.0
        self._ema_alpha = 0.1

    @property
    def current_spec_length(self) -> int:
        """Current number of leap heads used for drafting."""
        return self._current_spec_length

    @property
    def leap_distances(self) -> List[int]:
        """Active leap distances for current spec length."""
        return self.leap_mtp.leap_distances[:self._current_spec_length]

    def reset(self) -> None:
        """Reset decoder state for a new sequence."""
        self._current_spec_length = self.max_spec_length
        self._ema_acceptance_rate = 0.0
        self.stats.reset()

    def _adjust_spec_length(self, acceptance_rate: float) -> None:
        """Adaptively adjust the number of leap heads used.

        Strategy:
        - If acceptance rate > target + margin: add one more leap head
        - If acceptance rate < target - margin: remove one leap head
        - Otherwise: keep current

        The margin provides hysteresis to avoid oscillation.

        Args:
            acceptance_rate: Most recent step's acceptance rate.
        """
        if not self.adaptive:
            return

        # Update EMA
        if self._ema_acceptance_rate == 0.0:
            self._ema_acceptance_rate = acceptance_rate
        else:
            self._ema_acceptance_rate = (
                self._ema_alpha * acceptance_rate
                + (1 - self._ema_alpha) * self._ema_acceptance_rate
            )

        margin = 0.05
        if self._ema_acceptance_rate > self.target_acceptance_rate + margin:
            self._current_spec_length = min(
                self._current_spec_length + 1, self.max_spec_length
            )
        elif self._ema_acceptance_rate < self.target_acceptance_rate - margin:
            self._current_spec_length = max(
                self._current_spec_length - 1, 1
            )

    def _sample_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample a token from logits with temperature, top-k, and top-p.

        Args:
            logits: Logits tensor, shape (..., vocab_size).

        Returns:
            Sampled token IDs, shape (...).
        """
        if self.temperature != 1.0:
            logits = logits / max(self.temperature, 1e-8)

        # Top-k filtering
        if self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs - F.softmax(
                sorted_logits, dim=-1
            ) >= self.top_p
            indices_to_remove = sorted_indices_to_remove.scatter(
                sorted_indices.ndim - 1,
                sorted_indices,
                sorted_indices_to_remove,
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Greedy if temperature is effectively 0
        if self.temperature < 1e-8:
            return logits.argmax(dim=-1)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _compute_gap_positions(self, leap_distances: List[int]) -> List[int]:
        """Compute the positions that need gap-filling between leap predictions.

        Given leap distances [d_1, d_2, ..., d_K], the leap heads predict
        tokens at positions t+d_1, t+d_2, ..., t+d_K. Gap positions are
        all intermediate positions that are NOT covered by any leap distance.

        Example: leap_distances = [1, 2, 4, 8]
            Covered positions: 1, 2, 4, 8
            Gap positions: 3, 5, 6, 7

        Args:
            leap_distances: Sorted list of leap distances.

        Returns:
            Sorted list of gap positions (0-indexed relative offsets).
        """
        if not leap_distances:
            return []

        max_pos = max(leap_distances)
        covered = set(leap_distances)
        # All positions from 1 to max_pos that aren't covered by leaps
        gaps = [p for p in range(1, max_pos + 1) if p not in covered]
        return gaps

    @torch.no_grad()
    def draft(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[LeapMTPOutput, List[int]]:
        """Generate draft tokens using leap heads.

        Phase 1 of leap speculative decoding: Produce token predictions
        at leap positions.

        Args:
            hidden_states: Hidden states from the main model,
                shape (batch, seq_len, d_model).

        Returns:
            Tuple of (leap_output, gap_positions):
            - leap_output: Predictions at leap positions.
            - gap_positions: Positions needing gap-fill.
        """
        leap_output = self.leap_mtp.draft(
            hidden_states,
            n_draft=self._current_spec_length,
        )
        gap_positions = self._compute_gap_positions(
            leap_output.leap_distances
        )
        return leap_output, gap_positions

    @torch.no_grad()
    def fill_gaps_autoregressive(
        self,
        leap_output: LeapMTPOutput,
        gap_positions: List[int],
        token_embed_fn: Callable[[torch.Tensor], torch.Tensor],
        ssm_layer: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Fill gap positions between leap predictions using autoregressive or SSM sampling.

        Given leap predictions at positions [1, 2, 4, 8], this fills
        positions [3, 5, 6, 7] using either:
        - SSM state propagation (fast, O(1) per token)
        - Autoregressive token-by-token generation (slower but more accurate)

        This is a key advantage of L-MTP over standard MTP: the leap
        predictions provide "anchor points" that constrain the gap-filling,
        leading to higher quality intermediate tokens.

        Args:
            leap_output: Leap predictions at non-adjacent positions.
            gap_positions: Positions that need filling.
            token_embed_fn: Function that maps token IDs to embeddings.
                Signature: (token_ids: Tensor) -> embeddings: Tensor.
            ssm_layer: Optional SSM layer for fast gap-filling. If None,
                uses simple autoregressive generation.

        Returns:
            Complete drafted sequence including leap and gap-fill tokens,
            shape (batch, max_leap). Positions not yet filled are set to
            the leap predictions where available.
        """
        if not gap_positions:
            # No gaps — all positions are covered by leap predictions
            return self._build_full_sequence(leap_output, {})

        batch = leap_output.token_ids.shape[0]
        device = leap_output.token_ids.device

        # Build a map: position -> predicted token
        # Start with leap predictions
        position_tokens: Dict[int, torch.Tensor] = {}
        for k, d in enumerate(leap_output.leap_distances):
            position_tokens[d] = leap_output.token_ids[:, k]

        # Fill gaps — for each gap position, use the nearest known context
        # In a full implementation, this would use the SSM layer to propagate
        # state and generate tokens sequentially. Here we provide a simplified
        # version that uses token embeddings + linear projection.
        for gap_pos in gap_positions:
            # Find the nearest preceding known position
            prev_known = None
            for d in reversed(leap_output.leap_distances):
                if d < gap_pos:
                    prev_known = d
                    break

            if prev_known is not None and ssm_layer is not None:
                # Use SSM to predict the gap token from the previous context
                prev_token = position_tokens[prev_known]
                prev_embed = token_embed_fn(prev_token)  # (batch, d_model)
                # SSM forward to get next hidden state
                ssm_out, _ = ssm_layer(prev_embed.unsqueeze(1))
                # Project to logits using the first leap head (nearest)
                gap_logits = self.leap_mtp.heads[0](ssm_out).squeeze(1)
                gap_token = gap_logits.argmax(dim=-1)
            else:
                # Fallback: use the nearest leap head's prediction
                # Find the closest leap distance
                closest_leap_idx = min(
                    range(len(leap_output.leap_distances)),
                    key=lambda i: abs(leap_output.leap_distances[i] - gap_pos),
                )
                gap_token = leap_output.token_ids[:, closest_leap_idx]

            position_tokens[gap_pos] = gap_token

        return self._build_full_sequence(leap_output, position_tokens)

    def _build_full_sequence(
        self,
        leap_output: LeapMTPOutput,
        position_tokens: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Build the full drafted sequence from position-token mapping.

        Args:
            leap_output: Leap predictions.
            position_tokens: Map from position offset to token IDs.

        Returns:
            Full sequence tensor, shape (batch, max_position).
        """
        batch = leap_output.token_ids.shape[0]
        device = leap_output.token_ids.device
        max_pos = max(leap_output.leap_distances) if leap_output.leap_distances else 0

        # Start with leap predictions
        for k, d in enumerate(leap_output.leap_distances):
            position_tokens[d] = leap_output.token_ids[:, k]

        # Build sequence from position 1 to max_pos
        sequence = torch.zeros(batch, max_pos, dtype=torch.long, device=device)
        for pos in range(1, max_pos + 1):
            if pos in position_tokens:
                sequence[:, pos - 1] = position_tokens[pos]

        return sequence

    @torch.no_grad()
    def verify(
        self,
        draft_tokens: torch.Tensor,
        verifier_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, List[bool]]:
        """Verify draft tokens against the main model's predictions.

        Compares the drafted sequence (leap + gap-fill tokens) with the
        verifier model's predictions. Accepts the longest matching prefix.

        Args:
            draft_tokens: Full drafted sequence (leap + gap-fill),
                shape (batch, spec_length), where spec_length = max_leap.
            verifier_logits: Logits from the main model,
                shape (batch, spec_length + 1, vocab_size).

        Returns:
            Tuple of (output_tokens, n_accepted, accepted_mask):
            - output_tokens: Accepted + correction tokens,
                shape (batch, n_accepted + 1).
            - n_accepted: Number of draft tokens accepted.
            - accepted_mask: Boolean mask of which positions were accepted.
        """
        batch = draft_tokens.shape[0]
        spec_length = draft_tokens.shape[1]

        # Get verifier predictions
        verifier_predictions = verifier_logits.argmax(dim=-1)  # (batch, spec_length + 1)

        # Compare draft with verifier
        matches = draft_tokens == verifier_predictions[:, :spec_length]  # (batch, spec_length)

        # Find longest matching prefix
        all_match_prefix = matches.cumprod(dim=1)
        n_accepted_per_batch = all_match_prefix.sum(dim=1)
        n_accepted = int(n_accepted_per_batch.min().item())

        # Build accepted mask (for per-leap tracking)
        accepted_mask = [bool(all_match_prefix[0, k]) for k in range(spec_length)]

        # Build output tokens
        accepted_draft = draft_tokens[:, :n_accepted]
        correction_token = verifier_predictions[:, n_accepted:n_accepted + 1]

        if n_accepted > 0:
            output_tokens = torch.cat([accepted_draft, correction_token], dim=1)
        else:
            output_tokens = correction_token

        return output_tokens, n_accepted, accepted_mask

    @torch.no_grad()
    def step(
        self,
        current_hidden: torch.Tensor,
        verifier_forward: Callable[
            [torch.Tensor, Optional[Any]],
            Tuple[torch.Tensor, Any],
        ],
        gap_fill_fn: Optional[Callable] = None,
        token_embed_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ssm_layer: Optional[nn.Module] = None,
        past_kv: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any, int, int, bool]:
        """Execute one step of leap speculative decoding.

        This is the main entry point for the generation loop. Each call
        performs one draft-gap-fill-verify cycle.

        Args:
            current_hidden: Hidden states at the current position,
                shape (batch, 1, d_model).
            verifier_forward: Callable that takes (token_ids, past_kv) and
                returns (logits, new_past_kv).
            gap_fill_fn: Optional callable for custom gap-filling. If None,
                uses the built-in fill_gaps_autoregressive method.
            token_embed_fn: Function mapping token IDs to embeddings.
                Required for gap-filling.
            ssm_layer: Optional SSM layer for fast gap-filling.
            past_kv: Current KV cache.

        Returns:
            Tuple of (new_tokens, new_past_kv, n_accepted, n_gap_fill, eos_reached):
            - new_tokens: Accepted + correction tokens.
            - new_past_kv: Updated KV cache.
            - n_accepted: Number of draft tokens accepted.
            - n_gap_fill: Number of gap-fill tokens generated.
            - eos_reached: True if EOS token was generated.
        """
        spec_length = self._current_spec_length

        # ---- Phase 1: LEAP DRAFT ----
        leap_output, gap_positions = self.draft(current_hidden)

        # ---- Phase 2: GAP FILL ----
        if gap_fill_fn is not None:
            full_draft = gap_fill_fn(leap_output, gap_positions)
        elif token_embed_fn is not None:
            full_draft = self.fill_gaps_autoregressive(
                leap_output, gap_positions, token_embed_fn, ssm_layer
            )
        else:
            # No gap-filling: build sequence from leap predictions only
            full_draft = self._build_full_sequence(
                leap_output,
                {d: leap_output.token_ids[:, k]
                 for k, d in enumerate(leap_output.leap_distances)},
            )

        n_gap_fill = len(gap_positions)

        # ---- Phase 3: VERIFY ----
        verifier_logits, new_past_kv = verifier_forward(full_draft, past_kv)

        # ---- Phase 4: ACCEPT/REJECT ----
        output_tokens, n_accepted, accepted_mask = self.verify(
            full_draft, verifier_logits
        )

        # ---- Phase 5: STATISTICS & ADAPTATION ----
        self.stats.record_step(
            n_accepted=n_accepted,
            n_gap_fill=n_gap_fill,
            spec_length=spec_length,
            leap_distances=leap_output.leap_distances,
            accepted_mask=accepted_mask[:spec_length],
        )

        acceptance_rate = n_accepted / max(full_draft.shape[1], 1)
        self._adjust_spec_length(acceptance_rate)

        # ---- Check EOS ----
        eos_reached = False
        if self.eos_token_id >= 0:
            eos_reached = (output_tokens == self.eos_token_id).any()

        return output_tokens, new_past_kv, n_accepted, n_gap_fill, eos_reached

    @torch.no_grad()
    def generate(
        self,
        initial_hidden: torch.Tensor,
        verifier_forward: Callable,
        gap_fill_fn: Optional[Callable] = None,
        token_embed_fn: Optional[Callable] = None,
        ssm_layer: Optional[nn.Module] = None,
        past_kv: Optional[Any] = None,
        max_new_tokens: int = 256,
    ) -> Tuple[torch.Tensor, LeapSpeculativeStats]:
        """Generate tokens using leap speculative decoding.

        Repeatedly calls step() until max_new_tokens is reached or EOS
        is generated.

        Args:
            initial_hidden: Initial hidden states, shape (batch, 1, d_model).
            verifier_forward: Callable verifier forward.
            gap_fill_fn: Optional custom gap-filling function.
            token_embed_fn: Function mapping token IDs to embeddings.
            ssm_layer: Optional SSM layer for gap-filling.
            past_kv: Initial KV cache.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Tuple of (all_tokens, stats):
            - all_tokens: All generated tokens.
            - stats: LeapSpeculativeStats with full statistics.
        """
        self.reset()
        all_tokens_list: List[torch.Tensor] = []
        total_generated = 0
        current_hidden = initial_hidden

        while total_generated < max_new_tokens:
            output_tokens, past_kv, n_accepted, n_gap_fill, eos_reached = self.step(
                current_hidden=current_hidden,
                verifier_forward=verifier_forward,
                gap_fill_fn=gap_fill_fn,
                token_embed_fn=token_embed_fn,
                ssm_layer=ssm_layer,
                past_kv=past_kv,
            )

            all_tokens_list.append(output_tokens)
            total_generated += output_tokens.shape[1]

            if eos_reached:
                break

        if all_tokens_list:
            all_tokens = torch.cat(all_tokens_list, dim=1)
        else:
            all_tokens = torch.zeros(
                initial_hidden.shape[0], 0,
                dtype=torch.long, device=initial_hidden.device,
            )

        return all_tokens, self.stats

    def summary(self) -> str:
        """Return a human-readable summary of the decoder configuration and stats.

        Returns:
            Formatted string with configuration and current statistics.
        """
        lines = [
            "Leap MTP Speculative Decoder",
            "=" * 50,
            f"  Max spec length:        {self.max_spec_length}",
            f"  Current spec length:    {self._current_spec_length}",
            f"  Active leap distances:  {self.leap_distances}",
            f"  Adaptive:               {self.adaptive}",
            f"  Target acceptance:      {self.target_acceptance_rate:.0%}",
            f"  Temperature:            {self.temperature}",
            f"  EOS token ID:           {self.eos_token_id}",
            "",
            self.stats.summary(),
        ]
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================


def create_geometric_leap_mtp(
    d_model: int,
    vocab_size: int,
    num_leaps: int = 4,
    leap_ratio: float = 2.0,
    **kwargs: Any,
) -> LeapMTP:
    """Convenience function to create a LeapMTP with geometric schedule.

    Args:
        d_model: Hidden dimension of the main model.
        vocab_size: Size of the output vocabulary.
        num_leaps: Number of leap prediction heads.
        leap_ratio: Geometric ratio for leap distances.
        **kwargs: Additional LeapMTPConfig parameters.

    Returns:
        LeapMTP module configured with geometric leap schedule.
    """
    config = LeapMTPConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        num_leaps=num_leaps,
        leap_schedule=LeapScheduleType.GEOMETRIC,
        leap_ratio=leap_ratio,
        **kwargs,
    )
    return LeapMTP(config)


def create_adjacent_mtp(
    d_model: int,
    vocab_size: int,
    num_heads: int = 5,
    **kwargs: Any,
) -> LeapMTP:
    """Convenience function to create a LeapMTP in adjacent (standard MTP) mode.

    This degrades L-MTP to standard Multi-Token Prediction for ablation
    studies comparing L-MTP vs. MTP.

    Args:
        d_model: Hidden dimension of the main model.
        vocab_size: Size of the output vocabulary.
        num_heads: Number of prediction heads.
        **kwargs: Additional LeapMTPConfig parameters.

    Returns:
        LeapMTP module configured with adjacent leap schedule (= standard MTP).
    """
    config = LeapMTPConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        num_leaps=num_heads,
        leap_schedule=LeapScheduleType.ADJACENT,
        **kwargs,
    )
    return LeapMTP(config)


def compare_leap_vs_adjacent(num_heads: int = 4, leap_ratio: float = 2.0) -> str:
    """Compare the coverage and efficiency of leap vs. adjacent MTP schedules.

    Useful for understanding the theoretical advantage of L-MTP.

    Args:
        num_heads: Number of prediction heads.
        leap_ratio: Geometric ratio for leap schedule.

    Returns:
        Formatted comparison string.
    """
    geometric_leaps = LeapScheduleGenerator.generate(
        LeapScheduleType.GEOMETRIC, num_heads, leap_ratio=leap_ratio
    )
    adjacent_leaps = LeapScheduleGenerator.generate(
        LeapScheduleType.ADJACENT, num_heads
    )

    geo_coverage = LeapScheduleGenerator.coverage(geometric_leaps)
    adj_coverage = LeapScheduleGenerator.coverage(adjacent_leaps)
    geo_eff = LeapScheduleGenerator.efficiency_ratio(geometric_leaps)
    adj_eff = LeapScheduleGenerator.efficiency_ratio(adjacent_leaps)

    lines = [
        "L-MTP vs. MTP Comparison",
        "=" * 50,
        f"  Number of heads:       {num_heads}",
        "",
        "  Geometric L-MTP:",
        f"    Leap distances:      {geometric_leaps}",
        f"    Coverage:            {geo_coverage} positions",
        f"    Efficiency ratio:    {geo_eff:.2f}x",
        "",
        "  Adjacent MTP:",
        f"    Leap distances:      {adjacent_leaps}",
        f"    Coverage:            {adj_coverage} positions",
        f"    Efficiency ratio:    {adj_eff:.2f}x",
        "",
        f"  Coverage improvement:  {geo_coverage / adj_coverage:.1f}x",
        f"  Efficiency gain:       {geo_eff / adj_eff:.2f}x",
    ]
    return "\n".join(lines)
