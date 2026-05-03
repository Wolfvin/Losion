"""
Losion Training Recipe — Complete 4-Phase Training Configuration
================================================================

Implements the complete Losion-specific training recipe including the
WSD learning rate schedule, 4-phase curriculum methodology, training
state management, and scaling recipes for different model sizes.

Credits:
- WSD learning rate schedule (ICLR 2025, 46 citations): Warmup-Stable-Decay
  schedule that decouples the stable training phase from the decay phase,
  enabling more flexible training control
- WSM decay-free schedule (arXiv:2507.17634): Warmup-Stable-Mean approach
  that replaces the decay phase with stochastic weight averaging for
  smoother convergence
- Losion 4-phase curriculum: Phase 1 (Individual), Phase 2 (Joint),
  Phase 3 (RL), Phase 4 (Advanced) training methodology
- DeepSeek training methodology: Bias-based routing, GRPO optimization,
  and expert specialization techniques
- TACO (Compute-Aligned Training): Aligns compute budget across pathways
- ETR (Entropy Trend Reward): Reduces wasteful thinking tokens up to 40%
- JEPA (Joint-Embedding Predictive Architecture): Future state prediction
  for principled auxiliary training signal
- DAPO (Yu et al., arXiv 2503.14476, 2025): Decoupled Clip & Dynamic
  Sampling Policy Optimization — improves over GRPO with asymmetric
  clipping, dynamic sampling, token-level loss, and overlong filtering
- RLVR (NeurIPS 2025, arXiv 2601.05607): Reinforcement Learning with
  Verifiable Rewards — objective, programmable reward functions
- L-MTP (arXiv 2505.17505, NeurIPS 2025): Leap Multi-Token Prediction
  Beyond Adjacent — geometric leap schedules for wider temporal coverage

Architecture:
1. WSDLRScheduler — Warmup-Stable-Decay learning rate schedule
2. LosionTrainingRecipe — Complete 4-phase training configuration
3. LosionTrainingState — Training state tracking & phase transitions
4. ScalingRecipe — Pre-configured recipes for losion_1b, losion_7b, losion_48b

Hardware: Pure Python + PyTorch. Compatible with CUDA, ROCm, and CPU.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class TrainingPhase(str, Enum):
    """Losion 4-phase training methodology.

    PHASE_1_INDIVIDUAL: Pre-training each pathway separately
    PHASE_2_JOINT: Joint fine-tuning of all pathways
    PHASE_3_RL: End-to-end RL with GRPO
    PHASE_4_ADVANCED: Advanced optimization (distillation, quantization)
    """
    PHASE_1_INDIVIDUAL = "phase_1_individual"
    PHASE_2_JOINT = "phase_2_joint"
    PHASE_3_RL = "phase_3_rl"
    PHASE_4_ADVANCED = "phase_4_advanced"


class DecayType(str, Enum):
    """Learning rate decay types for WSD schedule."""
    COSINE = "cosine"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    NONE = "none"


class PathwayType(str, Enum):
    """Losion tri-jalur (three-pathway) types."""
    SSM = "ssm"
    ATTENTION = "attention"
    MOE = "moe"
    ROUTER = "router"


# ============================================================================
# WSDLRScheduler — Warmup-Stable-Decay LR Schedule
# ============================================================================


@dataclass
class WSDConfig:
    """Configuration for the WSD learning rate schedule.

    The Warmup-Stable-Decay schedule has three phases:
    1. Warmup: Linear ramp from 0 to peak_lr over warmup_steps
    2. Stable: Constant peak_lr for stable_steps
    3. Decay: Reduce lr over decay_steps according to decay_type

    WSM mode replaces the decay phase with stochastic weight averaging,
    which can achieve smoother convergence without explicit decay.

    Attributes:
        peak_lr: Peak (maximum) learning rate.
        warmup_steps: Number of warmup steps.
        stable_steps: Number of steps at peak learning rate.
        decay_steps: Number of decay steps.
        decay_type: Type of decay ("cosine", "linear", "exponential").
        min_lr_ratio: Minimum lr as a fraction of peak_lr.
        wsm_mode: Whether to use WSM (stochastic weight averaging)
            instead of decay.
        wsm_ema_decay: EMA decay rate for WSM weight averaging.
        decay_from_step: Step at which to trigger early decay. If -1,
            decay starts automatically after warmup + stable.
        exponential_decay_rate: Decay rate for exponential decay type.
    """
    peak_lr: float = 3e-4
    warmup_steps: int = 2000
    stable_steps: int = 80000
    decay_steps: int = 10000
    decay_type: DecayType = DecayType.COSINE
    min_lr_ratio: float = 0.1
    wsm_mode: bool = False
    wsm_ema_decay: float = 0.999
    decay_from_step: int = -1
    exponential_decay_rate: float = 0.95


class WSDLRScheduler:
    """Warmup-Stable-Decay learning rate schedule for Losion training.

    Implements the WSD schedule (ICLR 2025) which decouples the stable
    training phase from the decay phase, enabling more flexible training
    control compared to traditional cosine schedules.

    Key features:
    - Configurable warmup, stable, and decay durations
    - Multiple decay types (cosine, linear, exponential)
    - WSM mode: replaces decay with stochastic weight averaging
    - Supports early decay trigger (can start decay from any point)

    Example:
        >>> config = WSDConfig(peak_lr=3e-4, warmup_steps=2000,
        ...                    stable_steps=80000, decay_steps=10000)
        >>> scheduler = WSDLRScheduler(config)
        >>> lr = scheduler.get_lr(step=50000)  # During stable phase

    Args:
        config: WSDConfig with schedule parameters.
    """

    def __init__(self, config: Optional[WSDConfig] = None) -> None:
        self.config = config or WSDConfig()
        self._current_step = 0
        self._decay_triggered = False
        self._decay_start_step: Optional[int] = None

        # WSM: store averaged weights
        self._swa_weights: Optional[Dict[str, Any]] = None
        self._swa_count = 0

    def get_lr(self, step: int) -> float:
        """Compute the learning rate for a given training step.

        The schedule follows:
        - Step < warmup_steps: Linear warmup from 0 to peak_lr
        - warmup_steps <= step < warmup_steps + stable_steps: Constant peak_lr
        - step >= warmup_steps + stable_steps: Decay from peak_lr to min_lr

        If decay_from_step is set (>= 0), decay begins at that step
        regardless of the stable phase duration.

        Args:
            step: Current training step.

        Returns:
            Learning rate for the given step.
        """
        cfg = self.config
        min_lr = cfg.peak_lr * cfg.min_lr_ratio

        # Check for early decay trigger
        if cfg.decay_from_step >= 0 and step >= cfg.decay_from_step:
            if not self._decay_triggered:
                self._decay_triggered = True
                self._decay_start_step = step
            return self._compute_decay_lr(step, min_lr)

        # Phase 1: Warmup
        if step < cfg.warmup_steps:
            if cfg.warmup_steps <= 0:
                return cfg.peak_lr
            warmup_factor = step / cfg.warmup_steps
            return cfg.peak_lr * warmup_factor

        # Phase 2: Stable
        stable_end = cfg.warmup_steps + cfg.stable_steps
        if step < stable_end:
            return cfg.peak_lr

        # Phase 3: Decay (or WSM)
        if cfg.wsm_mode:
            # WSM: maintain peak_lr, use SWA for convergence
            return cfg.peak_lr

        return self._compute_decay_lr(step, min_lr)

    def _compute_decay_lr(self, step: int, min_lr: float) -> float:
        """Compute the decayed learning rate.

        Args:
            step: Current training step.
            min_lr: Minimum learning rate.

        Returns:
            Decayed learning rate.
        """
        cfg = self.config

        # Determine effective decay start
        if self._decay_start_step is not None:
            decay_start = self._decay_start_step
        else:
            decay_start = cfg.warmup_steps + cfg.stable_steps

        if step <= decay_start:
            return cfg.peak_lr

        # Progress through decay phase [0, 1]
        decay_progress = min(
            1.0, (step - decay_start) / max(cfg.decay_steps, 1)
        )

        # Apply decay type
        if cfg.decay_type == DecayType.COSINE:
            # Cosine decay: smooth transition to min_lr
            lr = min_lr + 0.5 * (cfg.peak_lr - min_lr) * (
                1.0 + math.cos(math.pi * decay_progress)
            )
        elif cfg.decay_type == DecayType.LINEAR:
            # Linear decay from peak_lr to min_lr
            lr = cfg.peak_lr - (cfg.peak_lr - min_lr) * decay_progress
        elif cfg.decay_type == DecayType.EXPONENTIAL:
            # Exponential decay
            lr = cfg.peak_lr * (
                cfg.exponential_decay_rate ** decay_progress
            )
            lr = max(lr, min_lr)
        else:
            lr = cfg.peak_lr

        return max(lr, min_lr)

    def trigger_decay(self, step: int) -> None:
        """Manually trigger the decay phase at the current step.

        Useful for triggering decay based on validation metrics
        (e.g., loss plateau detection) rather than fixed step counts.

        Args:
            step: Current training step when decay is triggered.
        """
        if not self._decay_triggered:
            self._decay_triggered = True
            self._decay_start_step = step
            logger.info(
                f"WSDLRScheduler: Decay triggered at step {step} "
                f"(type={self.config.decay_type.value})"
            )

    def update_swa(self, model_params: Dict[str, Any]) -> None:
        """Update stochastic weight averaging (WSM mode).

        Performs EMA update on model parameters for weight averaging.
        Only effective when wsm_mode is True.

        Args:
            model_params: Dictionary of model parameter name → tensor.
        """
        if not self.config.wsm_mode:
            return

        if self._swa_weights is None:
            # Initialize SWA weights
            self._swa_weights = {
                name: param.clone().detach()
                for name, param in model_params.items()
            }
        else:
            # EMA update
            decay = self.config.wsm_ema_decay
            for name, param in model_params.items():
                if name in self._swa_weights:
                    self._swa_weights[name].mul_(decay).add_(
                        param.detach(), alpha=1.0 - decay
                    )

        self._swa_count += 1

    def get_swa_weights(self) -> Optional[Dict[str, Any]]:
        """Return the SWA-averaged weights (WSM mode).

        Returns:
            Dictionary of averaged parameter tensors, or None if SWA
            hasn't been used.
        """
        return self._swa_weights

    def get_state(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing.

        Returns:
            Dictionary containing scheduler state.
        """
        return {
            "current_step": self._current_step,
            "decay_triggered": self._decay_triggered,
            "decay_start_step": self._decay_start_step,
            "swa_count": self._swa_count,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state from a checkpoint.

        Args:
            state: Dictionary containing scheduler state.
        """
        self._current_step = state.get("current_step", 0)
        self._decay_triggered = state.get("decay_triggered", False)
        self._decay_start_step = state.get("decay_start_step", None)
        self._swa_count = state.get("swa_count", 0)

    def __repr__(self) -> str:
        return (
            f"WSDLRScheduler(peak_lr={self.config.peak_lr}, "
            f"warmup={self.config.warmup_steps}, "
            f"stable={self.config.stable_steps}, "
            f"decay={self.config.decay_steps}, "
            f"decay_type={self.config.decay_type.value}, "
            f"wsm={self.config.wsm_mode})"
        )


# ============================================================================
# LosionTrainingRecipe — Complete 4-Phase Training Configuration
# ============================================================================


@dataclass
class PhaseRecipe:
    """Complete training configuration for a single phase.

    Attributes:
        phase: Training phase identifier.
        budget_fraction: Fraction of total training budget (0.0-1.0).
        lr: Learning rate for this phase.
        batch_size: Batch size for this phase.
        data_config: Data difficulty configuration.
        loss_config: Loss function weights and configurations.
        frozen_modules: Module names to freeze during this phase.
        active_modules: Module names that are actively trained.
        router_weights: Tri-jalur router weights [ssm, attention, moe].
        router_frozen: Whether the router is frozen.
        use_grpo: Whether GRPO optimization is active.
        use_jepa: Whether JEPA loss is active.
        use_etr: Whether ETR reward is active.
        use_early_exit: Whether early exit (MoD) is active.
        use_distillation: Whether generation-focused distillation is active.
        use_bit_distill: Whether BitDistill quantization is active.
        use_flow_matching: Whether flow matching refinement is active.
        use_taco: Whether TACO compute alignment is active.
        use_symbolic_moe: Whether Symbolic-MoE for macro routing is active.
        use_dapo: Whether DAPO optimization is active (v0.8, replaces GRPO
            when enabled).
        use_rlvr: Whether RLVR verifiable rewards are active (v0.8).
        use_leap_mtp: Whether L-MTP Leap Multi-Token Prediction is active
            (v0.8).
    """
    phase: TrainingPhase = TrainingPhase.PHASE_1_INDIVIDUAL
    budget_fraction: float = 0.3
    lr: float = 3e-4
    batch_size: int = 32
    data_config: Dict[str, Any] = field(default_factory=dict)
    loss_config: Dict[str, float] = field(default_factory=dict)
    frozen_modules: List[str] = field(default_factory=list)
    active_modules: List[str] = field(default_factory=list)
    router_weights: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    router_frozen: bool = True
    use_grpo: bool = False
    use_jepa: bool = False
    use_etr: bool = False
    use_early_exit: bool = False
    use_distillation: bool = False
    use_bit_distill: bool = False
    use_flow_matching: bool = False
    use_taco: bool = False
    use_symbolic_moe: bool = False
    use_dapo: bool = False
    use_rlvr: bool = False
    use_leap_mtp: bool = False


class LosionTrainingRecipe:
    """Complete training configuration for Losion models.

    Implements the Losion 4-phase training methodology with per-phase
    configurations for learning rate, data difficulty, loss functions,
    frozen modules, and active modules.

    Phase 1 (Pre-Training Individual, 0-30%):
        Each pathway trained separately with frozen router.
        - SSM: JEPA loss for future state prediction
        - Attention: Standard LM loss + Gated Attention warmup
        - MoE: Expert specialization loss + load balancing
        - LR: WSD with long stable phase
        - Data: Easy, short sequences

    Phase 2 (Joint Fine-Tuning, 30-60%):
        All pathways trained together, router still frozen.
        - Equal pathway weights (0.33/0.33/0.33)
        - Standard LM loss for all pathways
        - TACO compute alignment starts
        - LR: WSD stable phase
        - Data: Mixed difficulty

    Phase 3 (End-to-End RL, 60-90%):
        Router unfrozen, GRPO/DAPO optimization.
        - GRPO (default) or DAPO (v0.8) policy optimization
        - RLVR verifiable rewards when enabled (v0.8)
        - ETR Entropy Trend Reward for thinking tokens
        - Router bias updates (DeepSeek-V3 style)
        - Symbolic-MoE for macro routing
        - LR: WSD decay phase starts
        - Data: Hard, reasoning-heavy

    Phase 4 (Advanced Optimization, 90-100%):
        Early exit, distillation, quantization.
        - Early exit training (Mixture of Depths)
        - Generation-focused distillation
        - BitDistill quantization
        - Flow matching refinement
        - DAPO/RLVR carried forward from Phase 3 (v0.8)
        - LR: WSD final decay
        - Data: Specialized, domain-specific

    Args:
        total_steps: Total number of training steps.
        model_name: Model name identifier.
        peak_lr: Peak learning rate across all phases.
        config: Optional LosionConfig for v0.8 technique detection (DAPO,
            RLVR, L-MTP).  When provided, the recipe automatically enables
            these techniques in the appropriate phases.
    """

    def __init__(
        self,
        total_steps: int = 100000,
        model_name: str = "losion-base",
        peak_lr: float = 3e-4,
        config: Optional[Any] = None,
    ) -> None:
        self.total_steps = total_steps
        self.model_name = model_name
        self.peak_lr = peak_lr
        self.config = config  # Optional LosionConfig for v0.8 technique detection

        # Build per-phase recipes
        self.phases: Dict[TrainingPhase, PhaseRecipe] = self._build_phases()

        # LR scheduler (WSD)
        self.lr_scheduler = WSDLRScheduler(WSDConfig(
            peak_lr=peak_lr,
            warmup_steps=int(total_steps * 0.02),
            stable_steps=int(total_steps * 0.58),
            decay_steps=int(total_steps * 0.40),
            decay_type=DecayType.COSINE,
        ))

    def _build_phases(self) -> Dict[TrainingPhase, PhaseRecipe]:
        """Build the complete 4-phase training configuration.

        Returns:
            Dictionary mapping TrainingPhase to PhaseRecipe.
        """
        phases = {}

        # =====================================================================
        # Phase 1: Pre-Training Individual (0-30%)
        # =====================================================================
        # Detect v0.8 techniques from LosionConfig
        _use_dapo = False
        _use_rlvr = False
        _use_leap_mtp = False
        if self.config is not None:
            if hasattr(self.config, "dapo") and hasattr(self.config.dapo, "enabled"):
                _use_dapo = self.config.dapo.enabled
            if hasattr(self.config, "rlvr") and hasattr(self.config.rlvr, "enabled"):
                _use_rlvr = self.config.rlvr.enabled
            if hasattr(self.config, "output") and hasattr(self.config.output, "use_leap_mtp"):
                _use_leap_mtp = self.config.output.use_leap_mtp

        # =====================================================================
        # Phase 1: Pre-Training Individual (0-30%)
        # =====================================================================
        phase1_loss_config: Dict[str, float] = {
            "lm_loss": 1.0,
            "jepa_loss": 0.1,       # SSM: JEPA for future state prediction
            "expert_spec_loss": 0.05,  # MoE: Expert specialization
            "load_balance_loss": 0.01,  # MoE: Load balancing
            "gated_attn_warmup": 0.1,   # Attention: Gated Attention warmup
        }
        if _use_leap_mtp:
            phase1_loss_config["leap_mtp_loss"] = 0.2  # L-MTP leap prediction loss

        phases[TrainingPhase.PHASE_1_INDIVIDUAL] = PhaseRecipe(
            phase=TrainingPhase.PHASE_1_INDIVIDUAL,
            budget_fraction=0.30,
            lr=self.peak_lr,
            batch_size=32,
            data_config={
                "difficulty": "easy",
                "max_seq_len": 2048,
                "domain_weights": {
                    "web": 0.6, "books": 0.2, "code": 0.05,
                    "academic": 0.05, "reasoning": 0.1,
                },
            },
            loss_config=phase1_loss_config,
            frozen_modules=["router", "attention.gated_attention"],
            active_modules=["ssm", "attention.base", "moe"],
            router_weights=(0.8, 0.1, 0.1),  # SSM dominant
            router_frozen=True,
            use_grpo=False,
            use_jepa=True,     # SSM pathway uses JEPA
            use_etr=False,
            use_early_exit=False,
            use_distillation=False,
            use_bit_distill=False,
            use_flow_matching=False,
            use_taco=False,
            use_symbolic_moe=False,
            use_dapo=False,
            use_rlvr=False,
            use_leap_mtp=_use_leap_mtp,  # v0.8: L-MTP in Phase 1
        )

        # =====================================================================
        # Phase 2: Joint Fine-Tuning (30-60%)
        # =====================================================================
        phases[TrainingPhase.PHASE_2_JOINT] = PhaseRecipe(
            phase=TrainingPhase.PHASE_2_JOINT,
            budget_fraction=0.30,
            lr=self.peak_lr * 0.5,
            batch_size=64,
            data_config={
                "difficulty": "mixed",
                "max_seq_len": 4096,
                "domain_weights": {
                    "web": 0.4, "books": 0.15, "code": 0.15,
                    "academic": 0.15, "reasoning": 0.15,
                },
            },
            loss_config={
                "lm_loss": 1.0,
                "jepa_loss": 0.05,       # Reduced JEPA weight
                "expert_spec_loss": 0.02,
                "load_balance_loss": 0.01,
                "pathway_align_loss": 0.05,  # Cross-pathway alignment
            },
            frozen_modules=["router"],
            active_modules=["ssm", "attention", "moe"],
            router_weights=(0.33, 0.33, 0.33),  # Equal pathway weights
            router_frozen=True,
            use_grpo=False,
            use_jepa=True,
            use_etr=False,
            use_early_exit=False,
            use_distillation=False,
            use_bit_distill=False,
            use_flow_matching=False,
            use_taco=True,      # TACO compute alignment starts
            use_symbolic_moe=False,
            use_dapo=False,
            use_rlvr=False,
            use_leap_mtp=False,
        )

        # =====================================================================
        # Phase 3: End-to-End RL (60-90%)
        # =====================================================================
        # v0.8: DAPO replaces GRPO when config.dapo.enabled;
        # RLVR verifiable rewards supplement the reward signal.
        phase3_loss_config: Dict[str, float] = {
            "lm_loss": 0.5,
            "etr_reward": 0.3,      # ETR for thinking token efficiency
            "load_balance_loss": 0.01,
            "router_entropy_loss": 0.02,  # Router exploration bonus
        }
        if _use_dapo:
            phase3_loss_config["dapo_loss"] = 1.0    # DAPO policy optimization
            phase3_loss_config["rlvr_reward"] = 0.5  # RLVR verifiable rewards
            phase3_loss_config["etr_reward"] = 0.2   # ETR reduced when DAPO+RLVR active
            phase3_loss_config["grpo_loss"] = 0.3    # GRPO as auxiliary
        else:
            phase3_loss_config["grpo_loss"] = 1.0    # GRPO policy optimization

        phases[TrainingPhase.PHASE_3_RL] = PhaseRecipe(
            phase=TrainingPhase.PHASE_3_RL,
            budget_fraction=0.30,
            lr=self.peak_lr * 0.1,
            batch_size=128,
            data_config={
                "difficulty": "hard",
                "max_seq_len": 8192,
                "domain_weights": {
                    "web": 0.2, "books": 0.15, "code": 0.2,
                    "academic": 0.2, "reasoning": 0.25,
                },
            },
            loss_config=phase3_loss_config,
            frozen_modules=[],
            active_modules=["ssm", "attention", "moe", "router"],
            router_weights=(0.33, 0.33, 0.33),  # GRPO/DAPO will adjust
            router_frozen=False,  # Router unfrozen
            use_grpo=True,        # GRPO optimization active (fallback)
            use_jepa=False,
            use_etr=True,         # ETR reward for thinking tokens
            use_early_exit=False,
            use_distillation=False,
            use_bit_distill=False,
            use_flow_matching=False,
            use_taco=True,
            use_symbolic_moe=True,   # Symbolic-MoE for macro routing
            use_dapo=_use_dapo,      # v0.8: DAPO replaces GRPO when enabled
            use_rlvr=_use_rlvr,      # v0.8: RLVR verifiable rewards
            use_leap_mtp=False,
        )

        # =====================================================================
        # Phase 4: Advanced Optimization (90-100%)
        # =====================================================================
        phases[TrainingPhase.PHASE_4_ADVANCED] = PhaseRecipe(
            phase=TrainingPhase.PHASE_4_ADVANCED,
            budget_fraction=0.10,
            lr=self.peak_lr * 0.01,
            batch_size=64,
            data_config={
                "difficulty": "specialized",
                "max_seq_len": 8192,
                "domain_weights": {
                    "web": 0.1, "books": 0.1, "code": 0.25,
                    "academic": 0.3, "reasoning": 0.25,
                },
            },
            loss_config={
                "lm_loss": 0.3,
                "distill_loss": 0.5,     # Generation-focused distillation
                "bit_distill_loss": 0.3,  # BitDistill quantization
                "flow_matching_loss": 0.2,  # Flow matching refinement
                "early_exit_loss": 0.1,   # Mixture of Depths
                "grpo_loss": 0.3,
                "etr_reward": 0.2,
                "dapo_loss": 0.3 if _use_dapo else 0.0,
                "rlvr_reward": 0.2 if _use_rlvr else 0.0,
            },
            frozen_modules=[],
            active_modules=["ssm", "attention", "moe", "router", "output"],
            router_weights=(0.33, 0.33, 0.33),  # Fully adaptive
            router_frozen=False,
            use_grpo=True,
            use_jepa=False,
            use_etr=True,
            use_early_exit=True,     # Mixture of Depths
            use_distillation=True,   # Generation-focused distillation
            use_bit_distill=True,    # BitDistill quantization
            use_flow_matching=True,  # Flow matching refinement
            use_taco=True,
            use_symbolic_moe=True,
            use_dapo=_use_dapo,      # v0.8: DAPO carried into Phase 4
            use_rlvr=_use_rlvr,      # v0.8: RLVR carried into Phase 4
            use_leap_mtp=False,
        )

        return phases

    def get_phase_recipe(self, phase: TrainingPhase) -> PhaseRecipe:
        """Get the training recipe for a specific phase.

        Args:
            phase: Training phase to get the recipe for.

        Returns:
            PhaseRecipe for the requested phase.
        """
        return self.phases[phase]

    def get_current_phase(self, step: int) -> TrainingPhase:
        """Determine the current training phase based on step.

        Args:
            step: Current training step.

        Returns:
            The current TrainingPhase.
        """
        progress = step / max(self.total_steps, 1)

        if progress < 0.30:
            return TrainingPhase.PHASE_1_INDIVIDUAL
        elif progress < 0.60:
            return TrainingPhase.PHASE_2_JOINT
        elif progress < 0.90:
            return TrainingPhase.PHASE_3_RL
        else:
            return TrainingPhase.PHASE_4_ADVANCED

    def get_phase_steps(self, phase: TrainingPhase) -> Tuple[int, int]:
        """Get the start and end steps for a phase.

        Args:
            phase: Training phase.

        Returns:
            Tuple of (start_step, end_step).
        """
        cumulative = 0.0
        for p in TrainingPhase:
            recipe = self.phases[p]
            start = int(cumulative * self.total_steps)
            cumulative += recipe.budget_fraction
            end = int(cumulative * self.total_steps)
            if p == phase:
                return start, end
        return 0, self.total_steps

    def get_lr(self, step: int) -> float:
        """Get the learning rate for a given step.

        Adjusts the base WSD schedule by the phase-specific LR multiplier.

        Args:
            step: Current training step.

        Returns:
            Learning rate for the given step.
        """
        phase = self.get_current_phase(step)
        recipe = self.phases[phase]
        base_lr = self.lr_scheduler.get_lr(step)
        phase_lr_ratio = recipe.lr / self.peak_lr
        return base_lr * phase_lr_ratio

    def get_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all phases.

        Returns:
            List of dictionaries with phase summaries.
        """
        summary = []
        for phase in TrainingPhase:
            recipe = self.phases[phase]
            start, end = self.get_phase_steps(phase)
            summary.append({
                "phase": phase.value,
                "budget_fraction": recipe.budget_fraction,
                "start_step": start,
                "end_step": end,
                "lr": recipe.lr,
                "batch_size": recipe.batch_size,
                "router_weights": recipe.router_weights,
                "router_frozen": recipe.router_frozen,
                "frozen_modules": recipe.frozen_modules,
                "active_modules": recipe.active_modules,
                "loss_config": recipe.loss_config,
                "data_config": recipe.data_config,
                "techniques": {
                    "grpo": recipe.use_grpo,
                    "jepa": recipe.use_jepa,
                    "etr": recipe.use_etr,
                    "early_exit": recipe.use_early_exit,
                    "distillation": recipe.use_distillation,
                    "bit_distill": recipe.use_bit_distill,
                    "flow_matching": recipe.use_flow_matching,
                    "taco": recipe.use_taco,
                    "symbolic_moe": recipe.use_symbolic_moe,
                    "dapo": recipe.use_dapo,
                    "rlvr": recipe.use_rlvr,
                    "leap_mtp": recipe.use_leap_mtp,
                },
            })
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the recipe to a dictionary.

        Returns:
            Dictionary representation of the complete training recipe.
        """
        return {
            "model_name": self.model_name,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "phases": {
                phase.value: {
                    "budget_fraction": recipe.budget_fraction,
                    "lr": recipe.lr,
                    "batch_size": recipe.batch_size,
                    "data_config": recipe.data_config,
                    "loss_config": recipe.loss_config,
                    "frozen_modules": recipe.frozen_modules,
                    "active_modules": recipe.active_modules,
                    "router_weights": list(recipe.router_weights),
                    "router_frozen": recipe.router_frozen,
                }
                for phase, recipe in self.phases.items()
            },
        }

    def __repr__(self) -> str:
        return (
            f"LosionTrainingRecipe(model={self.model_name}, "
            f"total_steps={self.total_steps:,}, "
            f"peak_lr={self.peak_lr})"
        )


# ============================================================================
# LosionTrainingState — Training State Tracking & Phase Transitions
# ============================================================================


@dataclass
class GradientStats:
    """Gradient statistics for monitoring training health.

    Attributes:
        grad_norm: Average gradient norm across all parameters.
        max_grad_norm: Maximum gradient norm.
        grad_scale: Loss scaling factor (for mixed precision).
        num_nan_grads: Number of NaN gradients detected.
    """
    grad_norm: float = 0.0
    max_grad_norm: float = 0.0
    grad_scale: float = 1.0
    num_nan_grads: int = 0


@dataclass
class ActivationStats:
    """Activation statistics for monitoring training health.

    Attributes:
        mean_activation: Mean activation value across layers.
        std_activation: Standard deviation of activations.
        dead_neuron_fraction: Fraction of neurons with zero activation.
        entropy: Average entropy of activation distributions.
    """
    mean_activation: float = 0.0
    std_activation: float = 1.0
    dead_neuron_fraction: float = 0.0
    entropy: float = 0.0


class LosionTrainingState:
    """Tracks training state, phase transitions, and checkpoint management.

    Maintains a comprehensive record of training progress including:
    - Current phase and global step
    - Best loss tracking per phase
    - Phase transition logic (loss plateau + step count)
    - Gradient and activation statistics
    - Checkpoint management (save best per phase)

    Phase transitions are triggered by either:
    1. Step threshold (primary): Reaching the configured step boundary
    2. Loss plateau (secondary): No improvement for plateau_patience steps
    3. Manual override: Explicitly setting the phase

    Args:
        recipe: LosionTrainingRecipe defining the training configuration.
        plateau_patience: Number of steps without improvement before
            considering loss plateaued.
        plateau_threshold: Minimum relative improvement to reset patience.
    """

    def __init__(
        self,
        recipe: LosionTrainingRecipe,
        plateau_patience: int = 1000,
        plateau_threshold: float = 0.001,
    ) -> None:
        self.recipe = recipe
        self.plateau_patience = plateau_patience
        self.plateau_threshold = plateau_threshold

        # ---- Core state ----
        self.current_phase = TrainingPhase.PHASE_1_INDIVIDUAL
        self.global_step = 0
        self.best_loss: float = float("inf")
        self.learning_rate: float = recipe.peak_lr

        # ---- Per-phase tracking ----
        self.phase_best_losses: Dict[TrainingPhase, float] = {
            phase: float("inf") for phase in TrainingPhase
        }
        self.phase_steps: Dict[TrainingPhase, int] = {
            phase: 0 for phase in TrainingPhase
        }

        # ---- Loss plateau detection ----
        self._plateau_counter = 0
        self._last_best_step = 0

        # ---- Statistics ----
        self.gradient_stats = GradientStats()
        self.activation_stats = ActivationStats()
        self._loss_history: List[float] = []

        # ---- Checkpoint tracking ----
        self._phase_checkpoints: Dict[TrainingPhase, str] = {}

        # ---- Transition history ----
        self._transitions: List[Dict[str, Any]] = []

    def update(
        self,
        step: int,
        loss: float,
        grad_stats: Optional[GradientStats] = None,
        act_stats: Optional[ActivationStats] = None,
    ) -> TrainingPhase:
        """Update training state and check for phase transitions.

        Args:
            step: Current training step.
            loss: Current training loss.
            grad_stats: Optional gradient statistics.
            act_stats: Optional activation statistics.

        Returns:
            Current training phase (may have transitioned).
        """
        self.global_step = step
        self.phase_steps[self.current_phase] += 1
        self._loss_history.append(loss)

        # Update best loss
        if loss < self.best_loss:
            self.best_loss = loss
            self._last_best_step = step
            self._plateau_counter = 0
        else:
            self._plateau_counter += 1

        # Update per-phase best loss
        if loss < self.phase_best_losses[self.current_phase]:
            self.phase_best_losses[self.current_phase] = loss

        # Update statistics
        if grad_stats is not None:
            self.gradient_stats = grad_stats
        if act_stats is not None:
            self.activation_stats = act_stats

        # Update learning rate
        self.learning_rate = self.recipe.get_lr(step)

        # Check for phase transition
        return self._check_phase_transition(step, loss)

    def _check_phase_transition(
        self, step: int, loss: float
    ) -> TrainingPhase:
        """Check whether a phase transition should occur.

        Transitions are triggered by:
        1. Step boundary: step >= phase end step
        2. Loss plateau: no improvement for plateau_patience steps
           AND minimum phase steps completed

        Args:
            step: Current training step.
            loss: Current training loss.

        Returns:
            Current training phase after potential transition.
        """
        # Step-based transition
        current_recipe = self.recipe.phases[self.current_phase]
        _, phase_end = self.recipe.get_phase_steps(self.current_phase)

        if step >= phase_end:
            next_phase = self._get_next_phase(self.current_phase)
            if next_phase is not None:
                self._transition_to(
                    next_phase,
                    reason=f"step_boundary (step={step}, boundary={phase_end})",
                )
                return self.current_phase

        # Loss plateau transition (only if minimum steps in phase completed)
        min_phase_steps = 500  # Minimum steps before plateau-based transition
        if (
            self._plateau_counter >= self.plateau_patience
            and self.phase_steps[self.current_phase] >= min_phase_steps
        ):
            next_phase = self._get_next_phase(self.current_phase)
            if next_phase is not None:
                self._transition_to(
                    next_phase,
                    reason=f"loss_plateau (no improvement for "
                           f"{self.plateau_patience} steps)",
                )
                return self.current_phase

        return self.current_phase

    def _get_next_phase(
        self, current: TrainingPhase
    ) -> Optional[TrainingPhase]:
        """Get the next training phase.

        Args:
            current: Current training phase.

        Returns:
            Next training phase, or None if at the last phase.
        """
        phase_order = list(TrainingPhase)
        try:
            idx = phase_order.index(current)
            if idx + 1 < len(phase_order):
                return phase_order[idx + 1]
        except ValueError:
            pass
        return None

    def _transition_to(
        self, next_phase: TrainingPhase, reason: str
    ) -> None:
        """Execute a phase transition.

        Args:
            next_phase: Phase to transition to.
            reason: Reason for the transition.
        """
        old_phase = self.current_phase
        self.current_phase = next_phase
        self._plateau_counter = 0

        # Record transition
        self._transitions.append({
            "from_phase": old_phase.value,
            "to_phase": next_phase.value,
            "step": self.global_step,
            "reason": reason,
            "best_loss": self.best_loss,
            "phase_best_loss": self.phase_best_losses[old_phase],
        })

        logger.info(
            f"LosionTrainingState: Phase transition "
            f"{old_phase.value} → {next_phase.value} "
            f"(reason: {reason}, step: {self.global_step}, "
            f"best_loss: {self.best_loss:.4f})"
        )

    def set_phase(self, phase: TrainingPhase) -> None:
        """Manually set the training phase.

        Args:
            phase: Phase to set.
        """
        if phase != self.current_phase:
            self._transition_to(
                phase, reason="manual_override"
            )

    def should_save_checkpoint(self) -> bool:
        """Check if a checkpoint should be saved.

        Saves checkpoints at:
        - Best loss improvements
        - Phase transitions
        - Every save_interval steps (configured externally)

        Returns:
            True if a checkpoint should be saved.
        """
        return self._plateau_counter == 0  # Just hit a new best loss

    def get_progress(self) -> Dict[str, Any]:
        """Get training progress information.

        Returns:
            Dictionary with progress details.
        """
        phase_start, phase_end = self.recipe.get_phase_steps(
            self.current_phase
        )
        phase_budget = phase_end - phase_start
        phase_progress = 0.0
        if phase_budget > 0:
            phase_progress = min(
                1.0, (self.global_step - phase_start) / phase_budget
            )

        return {
            "current_phase": self.current_phase.value,
            "global_step": self.global_step,
            "total_steps": self.recipe.total_steps,
            "total_progress": self.global_step / max(self.recipe.total_steps, 1),
            "phase_progress": phase_progress,
            "best_loss": self.best_loss,
            "learning_rate": self.learning_rate,
            "phase_best_loss": self.phase_best_losses[self.current_phase],
            "plateau_counter": self._plateau_counter,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Serialize the training state for checkpointing.

        Returns:
            Dictionary containing all training state.
        """
        return {
            "current_phase": self.current_phase.value,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "learning_rate": self.learning_rate,
            "phase_best_losses": {
                p.value: l for p, l in self.phase_best_losses.items()
            },
            "phase_steps": {
                p.value: s for p, s in self.phase_steps.items()
            },
            "gradient_stats": {
                "grad_norm": self.gradient_stats.grad_norm,
                "max_grad_norm": self.gradient_stats.max_grad_norm,
                "grad_scale": self.gradient_stats.grad_scale,
                "num_nan_grads": self.gradient_stats.num_nan_grads,
            },
            "activation_stats": {
                "mean_activation": self.activation_stats.mean_activation,
                "std_activation": self.activation_stats.std_activation,
                "dead_neuron_fraction": self.activation_stats.dead_neuron_fraction,
                "entropy": self.activation_stats.entropy,
            },
            "transitions": self._transitions,
            "loss_history_len": len(self._loss_history),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore training state from a checkpoint.

        Args:
            state: Dictionary containing training state.
        """
        self.current_phase = TrainingPhase(state["current_phase"])
        self.global_step = state["global_step"]
        self.best_loss = state["best_loss"]
        self.learning_rate = state["learning_rate"]

        for phase_val, loss_val in state.get("phase_best_losses", {}).items():
            self.phase_best_losses[TrainingPhase(phase_val)] = loss_val

        for phase_val, steps in state.get("phase_steps", {}).items():
            self.phase_steps[TrainingPhase(phase_val)] = steps

        grad_state = state.get("gradient_stats", {})
        self.gradient_stats = GradientStats(**grad_state)

        act_state = state.get("activation_stats", {})
        self.activation_stats = ActivationStats(**act_state)

        self._transitions = state.get("transitions", [])


# ============================================================================
# ScalingRecipe — Pre-configured Recipes for Different Model Sizes
# ============================================================================


@dataclass
class ModelScaleConfig:
    """Model architecture configuration for a specific scale.

    Attributes:
        name: Model scale name (e.g., "losion_1b").
        n_layers: Number of transformer layers.
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        num_experts: Number of MoE experts.
        num_active_experts: Number of active experts per token.
        d_state: SSM state dimension.
        d_ff: Feed-forward intermediate dimension.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
    """
    name: str = "losion_1b"
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 8
    num_experts: int = 16
    num_active_experts: int = 2
    d_state: int = 64
    d_ff: int = 3072
    vocab_size: int = 32000
    max_seq_len: int = 4096


# Pre-configured scale definitions
_SCALE_CONFIGS: Dict[str, ModelScaleConfig] = {
    "losion_1b": ModelScaleConfig(
        name="losion_1b",
        n_layers=12,
        d_model=768,
        n_heads=8,
        num_experts=16,
        num_active_experts=2,
        d_state=64,
        d_ff=3072,
        vocab_size=32000,
        max_seq_len=4096,
    ),
    "losion_7b": ModelScaleConfig(
        name="losion_7b",
        n_layers=24,
        d_model=2048,
        n_heads=16,
        num_experts=64,
        num_active_experts=4,
        d_state=128,
        d_ff=8192,
        vocab_size=64000,
        max_seq_len=8192,
    ),
    "losion_48b": ModelScaleConfig(
        name="losion_48b",
        n_layers=48,
        d_model=4096,
        n_heads=32,
        num_experts=256,
        num_active_experts=8,
        d_state=256,
        d_ff=16384,
        vocab_size=128000,
        max_seq_len=16384,
    ),
}


# Scale-specific training hyperparameters
_SCALE_TRAINING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "losion_1b": {
        "total_steps": 100_000,
        "peak_lr": 3e-4,
        "batch_size_phase1": 32,
        "batch_size_phase2": 64,
        "batch_size_phase3": 128,
        "batch_size_phase4": 64,
        "warmup_ratio": 0.02,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
    },
    "losion_7b": {
        "total_steps": 300_000,
        "peak_lr": 1.5e-4,
        "batch_size_phase1": 64,
        "batch_size_phase2": 128,
        "batch_size_phase3": 256,
        "batch_size_phase4": 128,
        "warmup_ratio": 0.02,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
    },
    "losion_48b": {
        "total_steps": 1_000_000,
        "peak_lr": 5e-5,
        "batch_size_phase1": 128,
        "batch_size_phase2": 256,
        "batch_size_phase3": 512,
        "batch_size_phase4": 256,
        "warmup_ratio": 0.01,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
    },
}


class ScalingRecipe:
    """Pre-configured training recipes for different Losion model sizes.

    Provides LosionConfig and LosionTrainingRecipe instances for three
    predefined model scales:

    - losion_1b: 12 layers, 768 dim, 8 heads, 16 experts (~1B params)
    - losion_7b: 24 layers, 2048 dim, 16 heads, 64 experts (~7B params)
    - losion_48b: 48 layers, 4096 dim, 32 heads, 256 experts (~48B params)

    Each scale comes with:
    - Optimized architecture configuration (LosionConfig)
    - Scale-appropriate training hyperparameters (LosionTrainingRecipe)
    - Batch size, learning rate, and step count tuned for the scale

    Example:
        >>> config, recipe = ScalingRecipe.get("losion_7b")
        >>> state = LosionTrainingState(recipe)
        >>> print(f"Model: {config.model_name}, "
        ...       f"Params: ~{config.estimated_parameters() / 1e9:.1f}B")

    """

    AVAILABLE_SCALES = ["losion_1b", "losion_7b", "losion_48b"]

    @classmethod
    def get(
        cls, scale: str = "losion_1b"
    ) -> Tuple[Any, LosionTrainingRecipe]:
        """Get the LosionConfig and LosionTrainingRecipe for a model scale.

        Args:
            scale: Model scale name. One of "losion_1b", "losion_7b",
                "losion_48b".

        Returns:
            Tuple of (LosionConfig, LosionTrainingRecipe).

        Raises:
            ValueError: If scale name is not recognized.
        """
        if scale not in _SCALE_CONFIGS:
            raise ValueError(
                f"Unknown scale '{scale}'. "
                f"Available: {cls.AVAILABLE_SCALES}"
            )

        scale_config = _SCALE_CONFIGS[scale]
        train_config = _SCALE_TRAINING_CONFIGS[scale]

        # Build LosionConfig
        config = cls._build_config(scale_config)

        # Build LosionTrainingRecipe
        recipe = cls._build_recipe(scale, train_config)

        return config, recipe

    @classmethod
    def _build_config(cls, scale_config: ModelScaleConfig) -> Any:
        """Build a LosionConfig from scale configuration.

        Args:
            scale_config: ModelScaleConfig with architecture parameters.

        Returns:
            LosionConfig instance.
        """
        try:
            from losion.config import (
                LosionConfig,
                SSMConfig,
                AttentionConfig,
                RetrievalConfig,
                RouterConfig,
                TrainingConfig,
            )

            config = LosionConfig(
                model_name=scale_config.name,
                d_model=scale_config.d_model,
                n_layers=scale_config.n_layers,
                vocab_size=scale_config.vocab_size,
                max_seq_len=scale_config.max_seq_len,
                ssm=SSMConfig(
                    d_state=scale_config.d_state,
                    d_conv=4,
                    expand=2,
                ),
                attention=AttentionConfig(
                    n_heads=scale_config.n_heads,
                    d_kv=scale_config.d_model // scale_config.n_heads,
                    mla_latent_dim=min(
                        scale_config.d_model // 3,
                        scale_config.d_model,
                    ),
                ),
                retrieval=RetrievalConfig(
                    num_experts=scale_config.num_experts,
                    num_active_experts=scale_config.num_active_experts,
                    d_ff=scale_config.d_ff,
                ),
                router=RouterConfig(
                    top_k_pathways=2,
                ),
                training=TrainingConfig(
                    batch_size=scale_config.d_model,  # Will be overridden
                    learning_rate=3e-4,  # Will be overridden by recipe
                ),
            )
            return config

        except ImportError:
            logger.warning(
                "LosionConfig not available. Returning scale_config as-is."
            )
            return scale_config

    @classmethod
    def _build_recipe(
        cls,
        scale: str,
        train_config: Dict[str, Any],
    ) -> LosionTrainingRecipe:
        """Build a LosionTrainingRecipe from scale training configuration.

        Args:
            scale: Model scale name.
            train_config: Training hyperparameters for this scale.

        Returns:
            LosionTrainingRecipe instance.
        """
        recipe = LosionTrainingRecipe(
            total_steps=train_config["total_steps"],
            model_name=scale,
            peak_lr=train_config["peak_lr"],
        )

        # Override phase batch sizes
        recipe.phases[TrainingPhase.PHASE_1_INDIVIDUAL].batch_size = (
            train_config["batch_size_phase1"]
        )
        recipe.phases[TrainingPhase.PHASE_2_JOINT].batch_size = (
            train_config["batch_size_phase2"]
        )
        recipe.phases[TrainingPhase.PHASE_3_RL].batch_size = (
            train_config["batch_size_phase3"]
        )
        recipe.phases[TrainingPhase.PHASE_4_ADVANCED].batch_size = (
            train_config["batch_size_phase4"]
        )

        return recipe

    @classmethod
    def list_scales(cls) -> List[str]:
        """List all available model scales.

        Returns:
            List of scale name strings.
        """
        return cls.AVAILABLE_SCALES

    @classmethod
    def get_scale_info(cls, scale: str) -> Dict[str, Any]:
        """Get detailed information about a model scale.

        Args:
            scale: Model scale name.

        Returns:
            Dictionary with scale architecture and training details.
        """
        if scale not in _SCALE_CONFIGS:
            raise ValueError(
                f"Unknown scale '{scale}'. "
                f"Available: {cls.AVAILABLE_SCALES}"
            )

        scale_config = _SCALE_CONFIGS[scale]
        train_config = _SCALE_TRAINING_CONFIGS[scale]

        # Estimate parameter count
        d = scale_config.d_model
        n = scale_config.n_layers
        v = scale_config.vocab_size
        e = scale_config.num_experts
        d_ff = scale_config.d_ff

        # Rough parameter estimate
        emb_params = v * d
        ssm_params = 4 * d * d * 2  # SSM with expand=2
        attn_params = 4 * d * d
        moe_params = e * 3 * d * d_ff + d * d_ff  # experts + shared
        layer_params = ssm_params + attn_params + moe_params + 4 * d
        total_params = emb_params + n * layer_params + v * d

        return {
            "name": scale_config.name,
            "architecture": {
                "n_layers": scale_config.n_layers,
                "d_model": scale_config.d_model,
                "n_heads": scale_config.n_heads,
                "num_experts": scale_config.num_experts,
                "num_active_experts": scale_config.num_active_experts,
                "d_state": scale_config.d_state,
                "d_ff": scale_config.d_ff,
                "vocab_size": scale_config.vocab_size,
                "max_seq_len": scale_config.max_seq_len,
            },
            "estimated_parameters": total_params,
            "estimated_parameters_billions": total_params / 1e9,
            "training": train_config,
        }

    @classmethod
    def get_all_scales_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get detailed information about all model scales.

        Returns:
            Dictionary mapping scale names to their info dictionaries.
        """
        return {
            scale: cls.get_scale_info(scale)
            for scale in cls.AVAILABLE_SCALES
        }
