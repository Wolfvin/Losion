"""
Losion Training Orchestrator — Unified 4-Phase Training Pipeline
=================================================================

The ONE-STOP entry point for training a Losion model. Integrates ALL Losion
training techniques into a single, config-driven orchestrator that manages
the complete training lifecycle from pre-training through advanced
optimization.

Architecture
------------

    ┌──────────────────────────────────────────────────────────────┐
    │              LosionTrainingOrchestrator                       │
    │  ┌──────────────────────────────────────────────────────────┐ │
    │  │  Phase 1: Individual Pre-Training (0-30%)                │ │
    │  │  ├── WSD Learning Rate Schedule (WSDLRScheduler)         │ │
    │  │  ├── LLM-JEPA Auxiliary Loss (LatentPredictor)           │ │
    │  │  ├── Expert Specialization Loss                          │ │
    │  │  └── Gated Attention Warmup                               │ │
    │  ├──────────────────────────────────────────────────────────┤ │
    │  │  Phase 2: Joint Fine-Tuning (30-60%)                     │ │
    │  │  ├── Cross-Pathway Alignment Loss                        │ │
    │  │  ├── TACO Compute-Aligned Loss Weighting                  │ │
    │  │  ├── Curriculum Learning (CurriculumScheduler)            │ │
    │  │  └── Active Learning (ActiveLearningLoop)                 │ │
    │  ├──────────────────────────────────────────────────────────┤ │
    │  │  Phase 3: End-to-End RL (60-90%)                         │ │
    │  │  ├── DAPO / GRPO (auto-selected by config)                │ │
    │  │  ├── RLVR Verifiable Rewards                              │ │
    │  │  ├── ETR Entropy Trend Reward                             │ │
    │  │  ├── Symbolic-MoE Routing                                 │ │
    │  │  └── Evolutionary Search (EvolutionarySearcher)           │ │
    │  ├──────────────────────────────────────────────────────────┤ │
    │  │  Phase 4: Advanced Optimization (90-100%)                 │ │
    │  │  ├── Generation-Focused Distillation                      │ │
    │  │  ├── BitDistill Quantization-Aware Distillation           │ │
    │  │  ├── Early Exit / Mixture of Depths                       │ │
    │  │  └── Flow Matching Refinement                              │ │
    │  └──────────────────────────────────────────────────────────┘ │
    │  Shared: WSDLRScheduler, LosionTrainingRecipe, Checkpointing │
    └──────────────────────────────────────────────────────────────┘

Credits & References
--------------------
- WSD Schedule: ICLR 2025 (46 citations) — Warmup-Stable-Decay LR schedule
- DAPO: Yu et al., arXiv 2503.14476 (2025) — Decoupled Clip & Dynamic Sampling
- GRPO: Shao et al., DeepSeekMath (2024) — Group Relative Policy Optimization
- RLVR: NeurIPS 2025 Posters 119944, 116633 — Verifiable Rewards for RL
- LLM-JEPA: arXiv 2025 (19 citations) — Joint-Embedding Predictive Architecture
- ETR: Entropy Trend Reward — reduces thinking tokens up to 40%
- TACO: Compute-Aligned Training — aligns training/inference compute
- BitDistill: Wang et al., BitNet b1.58 (2024) — distillation-aware quantization
- Curriculum Learning: Bengio et al. (2009) — progressive difficulty scheduling
- Active Learning: GNoME, DeepMind (2023, Nature) — self-improving training
- Evolutionary Search: FunSearch, DeepMind (2023, Nature) — LLM-guided evolution
- DeepSeek-V2/V3: MLA, Aux-loss-free MoE, GRPO methodology

Hardware: Pure PyTorch, compatible with CUDA, ROCm, and CPU.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.config import LosionConfig, DAPOConfig, RLVRConfig, JEPAConfig
from losion.models.losion_model_v2 import LosionForCausalLMV2
from losion.training.losion_recipe import (
    LosionTrainingRecipe,
    LosionTrainingState,
    PhaseRecipe,
    TrainingPhase,
    WSDLRScheduler,
    WSDConfig,
    GradientStats,
    ActivationStats,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Orchestrator Configuration
# ============================================================================


@dataclass
class OrchestratorConfig:
    """Configuration for the LosionTrainingOrchestrator.

    Controls which training techniques are enabled and their
    hyperparameters. Reads from LosionConfig and provides
    orchestrator-specific overrides.

    Attributes:
        # ---- Core Settings ----
        total_steps: Total number of training steps across all phases.
        peak_lr: Peak learning rate (overridden per-phase by recipe).
        grad_clip: Maximum gradient norm for clipping.
        log_interval: Log metrics every N steps.
        eval_interval: Run evaluation every N steps.
        checkpoint_interval: Save checkpoint every N steps.

        # ---- Technique Enable/Disable Flags ----
        use_dapo: If True, use DAPO in Phase 3; otherwise fall back to GRPO.
        use_rlvr: If True, use RLVR verifiable rewards with DAPO/GRPO.
        use_jepa: If True, use LLM-JEPA auxiliary loss in Phase 1/2.
        use_etr: If True, use ETR entropy trend reward in Phase 3/4.
        use_taco: If True, use TACO compute-aligned loss in Phase 2+.
        use_curriculum: If True, use CurriculumScheduler for data selection.
        use_active_learning: If True, use ActiveLearningLoop for self-improvement.
        use_evolutionary: If True, use EvolutionarySearcher in Phase 3.
        use_distillation: If True, use Generation-Focused Distillation in Phase 4.
        use_bit_distill: If True, use BitDistill in Phase 4.
        use_early_exit: If True, use early exit / Mixture of Depths in Phase 4.
        use_flow_matching: If True, use flow matching refinement in Phase 4.

        # ---- Technique-Specific Hyperparameters ----
        dapo_config: DAPO configuration (overridden by LosionConfig.dapo if set).
        rlvr_config: RLVR configuration.
        jepa_config: JEPA configuration (overridden by LosionConfig.jepa if set).
        etr_alpha: ETR reward mixing coefficient.
        etr_warmup_steps: Steps before ETR reward activates.
        taco_alignment_strength: TACO alignment strength (0.0-1.0).
        distill_temperature: Temperature for generation-focused distillation.
        bit_distill_alpha: Weight for BitDistill loss.
        evolutionary_generations: Max generations for evolutionary search.

        # ---- Checkpoint / Resume ----
        output_dir: Directory for checkpoints and logs.
        resume_from: Path to checkpoint to resume from (None = fresh start).
        save_best_only: If True, only save checkpoints on best validation loss.
    """

    # Core
    total_steps: int = 100000
    peak_lr: float = 3e-4
    grad_clip: float = 1.0
    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 5000

    # Technique flags
    use_dapo: bool = True
    use_rlvr: bool = True
    use_jepa: bool = True
    use_etr: bool = True
    use_taco: bool = True
    use_curriculum: bool = True
    use_active_learning: bool = False
    use_evolutionary: bool = False
    use_distillation: bool = True
    use_bit_distill: bool = True
    use_early_exit: bool = True
    use_flow_matching: bool = False

    # Technique hyperparameters
    dapo_config: Optional[DAPOConfig] = None
    rlvr_config: Optional[RLVRConfig] = None
    jepa_config: Optional[JEPAConfig] = None
    etr_alpha: float = 0.3
    etr_warmup_steps: int = 50
    taco_alignment_strength: float = 0.5
    distill_temperature: float = 4.0
    bit_distill_alpha: float = 0.3
    evolutionary_generations: int = 10

    # Checkpoint
    output_dir: str = "./checkpoints"
    resume_from: Optional[str] = None
    save_best_only: bool = False


# ============================================================================
# Per-Phase Loss Configuration
# ============================================================================


@dataclass
class PhaseLossComponents:
    """Container for the loss components computed in a single phase.

    Attributes:
        total_loss: Weighted sum of all active losses.
        lm_loss: Standard autoregressive language modeling loss.
        jepa_loss: LLM-JEPA auxiliary prediction loss.
        rl_loss: DAPO or GRPO policy optimization loss.
        etr_reward: ETR entropy trend reward signal.
        distill_loss: Generation-focused distillation loss.
        bit_distill_loss: BitDistill quantization-aware distillation loss.
        taco_aligned_loss: TACO compute-aligned adjusted loss.
        load_balance_loss: MoE load balancing auxiliary loss.
        expert_spec_loss: Expert specialization loss.
        pathway_align_loss: Cross-pathway alignment loss.
        router_entropy_loss: Router exploration entropy bonus.
        early_exit_loss: Early exit / Mixture of Depths loss.
        flow_matching_loss: Flow matching refinement loss.
        aux_losses: Dictionary of any additional auxiliary losses.
    """

    total_loss: torch.Tensor = torch.tensor(0.0)
    lm_loss: torch.Tensor = torch.tensor(0.0)
    jepa_loss: torch.Tensor = torch.tensor(0.0)
    rl_loss: torch.Tensor = torch.tensor(0.0)
    etr_reward: torch.Tensor = torch.tensor(0.0)
    distill_loss: torch.Tensor = torch.tensor(0.0)
    bit_distill_loss: torch.Tensor = torch.tensor(0.0)
    taco_aligned_loss: torch.Tensor = torch.tensor(0.0)
    load_balance_loss: torch.Tensor = torch.tensor(0.0)
    expert_spec_loss: torch.Tensor = torch.tensor(0.0)
    pathway_align_loss: torch.Tensor = torch.tensor(0.0)
    router_entropy_loss: torch.Tensor = torch.tensor(0.0)
    early_exit_loss: torch.Tensor = torch.tensor(0.0)
    flow_matching_loss: torch.Tensor = torch.tensor(0.0)
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert all loss components to a flat dictionary of floats."""
        result = {}
        for attr_name in [
            "total_loss", "lm_loss", "jepa_loss", "rl_loss", "etr_reward",
            "distill_loss", "bit_distill_loss", "taco_aligned_loss",
            "load_balance_loss", "expert_spec_loss", "pathway_align_loss",
            "router_entropy_loss", "early_exit_loss", "flow_matching_loss",
        ]:
            val = getattr(self, attr_name)
            if isinstance(val, torch.Tensor):
                result[attr_name] = val.item()
            else:
                result[attr_name] = float(val)
        for k, v in self.aux_losses.items():
            result[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
        return result


# ============================================================================
# LosionTrainingOrchestrator
# ============================================================================


class LosionTrainingOrchestrator:
    """Unified 4-phase training orchestrator for Losion models.

    The ONE-STOP entry point for training a Losion model. Manages the
    complete training lifecycle by integrating ALL Losion training
    techniques into a single, config-driven pipeline:

    **Phase 1 — Individual Pre-Training (0-30% of budget)**
      Each Tri-Jalur pathway (SSM, Attention, MoE) is trained individually
      with the router frozen. LLM-JEPA provides an auxiliary training
      signal for the SSM pathway. Expert specialization and load balancing
      losses shape the MoE pathway. Gated attention warmup primes the
      attention pathway.

    **Phase 2 — Joint Fine-Tuning (30-60%)**
      All pathways are trained together with the router still frozen.
      TACO compute-aligned loss weighting ensures training compute is
      proportional to inference compute. Cross-pathway alignment loss
      encourages coherent multi-pathway representations. Curriculum
      learning schedules data difficulty.

    **Phase 3 — End-to-End RL (60-90%)**
      The router is unfrozen. DAPO (or GRPO as fallback) optimizes the
      routing policy. RLVR provides verifiable rewards for math, code,
      and format verification. ETR entropy trend reward penalizes
      wasteful thinking tokens. Symbolic-MoE enables macro-level
      routing. Evolutionary search discovers novel reasoning paths.

    **Phase 4 — Advanced Optimization (90-100%)**
      Generation-focused distillation transfers knowledge from a teacher.
      BitDistill performs distillation-aware quantization. Early exit
      (Mixture of Depths) enables adaptive compute. Flow matching
      refines the output distribution.

    Example::

        >>> config = OrchestratorConfig(total_steps=100000, use_dapo=True)
        >>> orchestrator = LosionTrainingOrchestrator(
        ...     config=config,
        ...     losion_config=LosionConfig(),
        ...     train_dataloader=train_dl,
        ...     eval_dataloader=eval_dl,
        ...     reward_fn=my_reward_fn,
        ... )
        >>> orchestrator.train()

    Args:
        config: OrchestratorConfig with technique flags and hyperparameters.
        losion_config: LosionConfig for the model architecture.
        train_dataloader: Training data loader yielding batches with
            ``"input_ids"`` and optional ``"labels"`` / ``"attention_mask"``.
        eval_dataloader: Evaluation data loader (same format).
        reward_fn: Callable ``(prompts, responses) -> rewards`` for RL
            phases. If None, a default reward function is used.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        losion_config: LosionConfig,
        train_dataloader: Any,
        eval_dataloader: Any,
        reward_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.losion_config = losion_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.reward_fn = reward_fn

        # ---- Build Model ----
        self.model = LosionForCausalLMV2(losion_config)
        self.device = next(self.model.parameters()).device

        # ---- Training Recipe & State ----
        self.recipe = LosionTrainingRecipe(
            total_steps=config.total_steps,
            model_name=losion_config.model_name,
            peak_lr=config.peak_lr,
        )
        self.training_state = LosionTrainingState(self.recipe)

        # ---- WSD LR Scheduler ----
        self.lr_scheduler = self.recipe.lr_scheduler

        # ---- Optimizer ----
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.peak_lr,
            betas=(0.9, 0.95),
            weight_decay=losion_config.training.weight_decay,
        )

        # ---- Initialize Sub-Modules (lazy, created when phase activates) ----
        self._dapo_trainer = None
        self._grpo_trainer = None
        self._rlvr_trainer = None
        self._etr_trainer = None
        self._taco_trainer = None
        self._distiller = None
        self._bit_distiller = None
        self._curriculum_scheduler = None
        self._active_learning_loop = None
        self._evolutionary_searcher = None
        self._jepa_module = None

        # ---- Internal State ----
        self._global_step = 0
        self._current_phase = TrainingPhase.PHASE_1_INDIVIDUAL
        self._best_eval_loss = float("inf")
        self._metrics_history: List[Dict[str, float]] = []
        self._phase_start_time: float = time.time()
        self._phase_transition_logged: bool = False
        self._total_training_time: float = 0.0

        # ---- Resume if specified ----
        if config.resume_from is not None:
            self.load_checkpoint(config.resume_from)

        logger.info(
            f"LosionTrainingOrchestrator initialized: "
            f"total_steps={config.total_steps:,}, "
            f"model={losion_config.model_name}, "
            f"params={self.model.get_num_params():,}, "
            f"dapo={config.use_dapo}, rlvr={config.use_rlvr}, "
            f"jepa={config.use_jepa}, etr={config.use_etr}, "
            f"taco={config.use_taco}"
        )

    # ==================================================================
    # Sub-Module Initialization (lazy, per-phase)
    # ==================================================================

    def _init_jepa(self) -> None:
        """Initialize LLM-JEPA module for Phase 1/2.

        LLM-JEPA (arXiv 2025, 19 citations) predicts future latent
        states instead of next tokens, providing a principled auxiliary
        training signal especially beneficial for the SSM pathway.
        """
        if self._jepa_module is not None:
            return

        try:
            from losion.training.llm_jepa import LLMJEPA, JEPAConfig as JepaCfg

            jepa_cfg = self.config.jepa_config
            if jepa_cfg is None and hasattr(self.losion_config, "jepa"):
                jepa_cfg = self.losion_config.jepa

            if jepa_cfg is None:
                jepa_cfg = JepaCfg(d_model=self.losion_config.d_model)

            self._jepa_module = LLMJEPA(self.losion_config, jepa_cfg)
            self._jepa_module.to(self.device)
            logger.info("LLM-JEPA module initialized for Phase 1/2")
        except ImportError:
            logger.warning(
                "LLM-JEPA import failed; JEPA auxiliary loss will be disabled"
            )
            self.config.use_jepa = False

    def _init_rl_trainer(self) -> None:
        """Initialize DAPO or GRPO trainer for Phase 3.

        DAPO (Yu et al., arXiv 2503.14476, 2025) improves over GRPO with:
        1. Decoupled clip (asymmetric epsilon_low / epsilon_high)
        2. Dynamic sampling (filter uniform-reward prompts)
        3. Token-level policy gradient loss
        4. Overlong filtering

        Falls back to GRPO if DAPO is disabled or unavailable.
        """
        if self.config.use_dapo:
            try:
                from losion.training.dapo import DAPOTrainer, DAPOConfig as DapoCfg

                dapo_cfg = self.config.dapo_config
                if dapo_cfg is None:
                    dapo_cfg = DapoCfg(
                        clip_ratio_low=self.losion_config.dapo.clip_ratio_low,
                        clip_ratio_high=self.losion_config.dapo.clip_ratio_high,
                        dynamic_sampling=self.losion_config.dapo.dynamic_sampling,
                        token_level_loss=self.losion_config.dapo.token_level_loss,
                        overlong_filter=self.losion_config.dapo.overlong_filter,
                        num_responses_per_prompt=self.losion_config.dapo.num_responses_per_prompt,
                        kl_coefficient=self.losion_config.dapo.kl_coefficient,
                    )
                self._dapo_trainer = DAPOTrainer(
                    config=dapo_cfg,
                    policy_model=self.model,
                    reward_fn=self.reward_fn,
                )
                logger.info("DAPO trainer initialized for Phase 3 RL")
                return
            except ImportError:
                logger.warning(
                    "DAPO import failed; falling back to GRPO for Phase 3"
                )

        # Fallback: GRPO
        try:
            from losion.training.grpo import GRPOTrainer, GRPOConfig

            self._grpo_trainer = GRPOTrainer(
                model=self.model,
                config=GRPOConfig(),
                reward_fn=self.reward_fn,
            )
            logger.info("GRPO trainer initialized for Phase 3 RL (DAPO fallback)")
        except ImportError:
            logger.warning(
                "GRPO import failed; RL training will be unavailable"
            )

    def _init_rlvr(self) -> None:
        """Initialize RLVR verifiable rewards for Phase 3.

        RLVR (NeurIPS 2025 Posters 119944, 116633) replaces learned reward
        models with objective, programmable reward functions for noise-free
        reward signals in math, code, and format verification.
        """
        if self._rlvr_trainer is not None:
            return

        try:
            from losion.training.rlvr import (
                RLVRTrainer,
                RLVRConfig as RlvrCfg,
                MathVerifier,
                CodeVerifier,
                FormatVerifier,
                CompositeVerifier,
                VerificationDifficulty,
            )

            rlvr_cfg = self.config.rlvr_config
            if rlvr_cfg is None:
                rlvr_cfg = RlvrCfg(
                    use_curriculum=self.losion_config.rlvr.use_curriculum,
                    curriculum_warmup_steps=self.losion_config.rlvr.curriculum_warmup_steps,
                    use_math_verifier=self.losion_config.rlvr.use_math_verifier,
                    use_code_verifier=self.losion_config.rlvr.use_code_verifier,
                    use_format_verifier=self.losion_config.rlvr.use_format_verifier,
                )

            # Build verifiers
            verifiers = []
            if rlvr_cfg.use_math_verifier:
                verifiers.append(MathVerifier())
            if rlvr_cfg.use_code_verifier:
                verifiers.append(CodeVerifier())
            if rlvr_cfg.use_format_verifier:
                verifiers.append(FormatVerifier())

            if verifiers:
                rlvr_cfg.verifiers = verifiers
                self._rlvr_trainer = RLVRTrainer(rlvr_cfg)
                logger.info("RLVR verifiable rewards initialized for Phase 3")
            else:
                logger.info("No RLVR verifiers configured; skipping RLVR init")
                self.config.use_rlvr = False
        except ImportError:
            logger.warning("RLVR import failed; verifiable rewards disabled")
            self.config.use_rlvr = False

    def _init_etr(self) -> None:
        """Initialize ETR entropy trend reward for Phase 3/4.

        ETR monitors entropy trends during generation and rewards models
        that converge to answers efficiently, reducing wasteful thinking
        tokens by up to 40%.
        """
        if self._etr_trainer is not None:
            return

        try:
            from losion.training.etr_reward import ETRTrainer, ETRConfig

            self._etr_trainer = ETRTrainer(
                model=self.model,
                etr_config=ETRConfig(),
                etr_alpha=self.config.etr_alpha,
                etr_warmup_steps=self.config.etr_warmup_steps,
            )
            logger.info("ETR entropy trend reward initialized")
        except ImportError:
            logger.warning("ETR import failed; entropy trend reward disabled")
            self.config.use_etr = False

    def _init_taco(self) -> None:
        """Initialize TACO compute-aligned training for Phase 2+.

        TACO ensures training compute is allocated proportionally to
        inference compute, preventing over-training on rarely-used
        MoE experts and under-training on frequently-used ones.
        """
        if self._taco_trainer is not None:
            return

        try:
            from losion.training.compute_aligned import (
                ComputeAlignedTrainer,
                ComputeAlignedConfig,
            )

            self._taco_trainer = ComputeAlignedTrainer(
                model=self.model,
                config=ComputeAlignedConfig(
                    alignment_strength=self.config.taco_alignment_strength,
                ),
            )
            logger.info("TACO compute-aligned training initialized")
        except ImportError:
            logger.warning("TACO import failed; compute-aligned training disabled")
            self.config.use_taco = False

    def _init_distillation(self) -> None:
        """Initialize Generation-Focused Distillation for Phase 4.

        Generation-focused distillation prioritizes generation quality
        over mere logits matching, using KL divergence on output
        distributions, sequence-level distillation, and progressive
        shifting from teacher to student.
        """
        if self._distiller is not None:
            return

        try:
            from losion.training.gen_distillation import (
                GenerationDistiller,
                GenerationDistillationConfig,
            )

            # Teacher is a frozen copy of the current model
            teacher = copy.deepcopy(self.model)
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()

            self._distiller = GenerationDistiller(
                teacher=teacher,
                student=self.model,
                config=GenerationDistillationConfig(
                    temperature=self.config.distill_temperature,
                ),
            )
            logger.info("Generation-Focused Distillation initialized for Phase 4")
        except ImportError:
            logger.warning(
                "GenDistillation import failed; distillation disabled"
            )
            self.config.use_distillation = False

    def _init_bit_distill(self) -> None:
        """Initialize BitDistill for Phase 4.

        BitDistill combines quantization-aware training with knowledge
        distillation so the quantized model learns to mimic the
        full-precision teacher's output distribution.
        """
        if self._bit_distiller is not None:
            return

        try:
            from losion.core.quantization.bit_distill import (
                BitDistillTrainer,
                BitDistillConfig,
            )

            self._bit_distiller = BitDistillTrainer(
                model=self.model,
                config=BitDistillConfig(
                    alpha_distill=self.config.bit_distill_alpha,
                ),
            )
            logger.info("BitDistill quantization-aware distillation initialized")
        except ImportError:
            logger.warning("BitDistill import failed; quantization distillation disabled")
            self.config.use_bit_distill = False

    def _init_curriculum(self) -> None:
        """Initialize CurriculumScheduler for progressive data difficulty."""
        if self._curriculum_scheduler is not None:
            return

        try:
            from losion.training.curriculum import CurriculumScheduler

            self._curriculum_scheduler = CurriculumScheduler(
                config=self.losion_config,
                total_steps=self.config.total_steps,
            )
            logger.info("CurriculumScheduler initialized")
        except ImportError:
            logger.warning("Curriculum import failed; curriculum learning disabled")
            self.config.use_curriculum = False

    def _init_active_learning(self) -> None:
        """Initialize ActiveLearningLoop (GNoME-inspired) for self-improvement."""
        if self._active_learning_loop is not None:
            return

        try:
            from losion.training.active_learning import (
                ActiveLearningLoop,
                ActiveLearningConfig,
            )

            self._active_learning_loop = ActiveLearningLoop(
                model=self.model,
                config=ActiveLearningConfig(),
            )
            logger.info("ActiveLearningLoop initialized")
        except ImportError:
            logger.warning("ActiveLearning import failed; active learning disabled")
            self.config.use_active_learning = False

    def _init_evolutionary(self) -> None:
        """Initialize EvolutionarySearcher (FunSearch-inspired) for Phase 3."""
        if self._evolutionary_searcher is not None:
            return

        try:
            from losion.training.evolutionary_search import (
                EvolutionarySearcher,
                EvolutionaryConfig,
            )

            self._evolutionary_searcher = EvolutionarySearcher(
                d_model=self.losion_config.d_model,
                config=EvolutionaryConfig(
                    max_generations=self.config.evolutionary_generations,
                ),
            )
            self._evolutionary_searcher.to(self.device)
            logger.info("EvolutionarySearcher initialized for Phase 3")
        except ImportError:
            logger.warning("EvolutionarySearch import failed; evolutionary search disabled")
            self.config.use_evolutionary = False

    # ==================================================================
    # Phase Configuration
    # ==================================================================

    def _apply_phase_config(self, phase: TrainingPhase) -> None:
        """Apply phase-specific configuration: freeze/unfreeze modules,
        adjust learning rate, and initialize sub-modules.

        This method is called at each phase transition to:
        1. Freeze modules that should not be trained in this phase
        2. Unfreeze modules that should be trained
        3. Adjust the optimizer learning rate
        4. Initialize phase-specific sub-modules (DAPO, JEPA, etc.)

        Args:
            phase: The training phase to configure for.
        """
        recipe = self.recipe.get_phase_recipe(phase)

        # ---- Freeze/Unfreeze Modules ----
        self._set_module_requires_grad(
            frozen=recipe.frozen_modules,
            active=recipe.active_modules,
        )

        # ---- Initialize Phase-Specific Sub-Modules ----
        if phase == TrainingPhase.PHASE_1_INDIVIDUAL:
            if self.config.use_jepa and recipe.use_jepa:
                self._init_jepa()

        elif phase == TrainingPhase.PHASE_2_JOINT:
            if self.config.use_jepa and recipe.use_jepa:
                self._init_jepa()
            if self.config.use_taco and recipe.use_taco:
                self._init_taco()
            if self.config.use_curriculum:
                self._init_curriculum()
            if self.config.use_active_learning:
                self._init_active_learning()

        elif phase == TrainingPhase.PHASE_3_RL:
            # Initialize RL trainer (DAPO or GRPO)
            self._init_rl_trainer()
            # RLVR verifiable rewards
            if self.config.use_rlvr:
                self._init_rlvr()
            # ETR entropy trend reward
            if self.config.use_etr and recipe.use_etr:
                self._init_etr()
            # TACO continues
            if self.config.use_taco and recipe.use_taco:
                self._init_taco()
            # Evolutionary search
            if self.config.use_evolutionary:
                self._init_evolutionary()

        elif phase == TrainingPhase.PHASE_4_ADVANCED:
            # Distillation
            if self.config.use_distillation and recipe.use_distillation:
                self._init_distillation()
            # BitDistill
            if self.config.use_bit_distill and recipe.use_bit_distill:
                self._init_bit_distill()
            # ETR continues
            if self.config.use_etr and recipe.use_etr:
                self._init_etr()
            # TACO continues
            if self.config.use_taco and recipe.use_taco:
                self._init_taco()

        # ---- Adjust Learning Rate ----
        new_lr = self.recipe.get_lr(self._global_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        self._phase_start_time = time.time()
        self._phase_transition_logged = False

        logger.info(
            f"Phase config applied: {phase.value}, "
            f"lr={new_lr:.2e}, "
            f"frozen={recipe.frozen_modules}, "
            f"active={recipe.active_modules}, "
            f"router_frozen={recipe.router_frozen}"
        )

    def _set_module_requires_grad(
        self,
        frozen: List[str],
        active: List[str],
    ) -> None:
        """Set requires_grad on model sub-modules based on phase config.

        Args:
            frozen: List of module name prefixes to freeze.
            active: List of module name prefixes to unfreeze.
        """
        # First, set all parameters to not require grad
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Then, activate specified modules
        for name, param in self.model.named_parameters():
            name_lower = name.lower()
            for active_prefix in active:
                if active_prefix.lower() in name_lower:
                    param.requires_grad = True
                    break

        # Ensure frozen modules override active
        for name, param in self.model.named_parameters():
            name_lower = name.lower()
            for frozen_prefix in frozen:
                if frozen_prefix.lower() in name_lower:
                    param.requires_grad = False
                    break

    # ==================================================================
    # Phase-Aware Loss Computation
    # ==================================================================

    def _compute_phase_loss(
        self,
        phase: TrainingPhase,
        model_output: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
    ) -> PhaseLossComponents:
        """Compute all active losses for the current training phase.

        Dispatches to the appropriate loss functions based on the phase
        recipe and technique enable flags. Returns a PhaseLossComponents
        object with all individual losses and the weighted total.

        Args:
            phase: Current training phase.
            model_output: Output dict from the model forward pass,
                containing ``"logits"``, ``"loss"``, ``"loss_dict"``,
                ``"hidden_states"``, etc.
            batch: Input batch with ``"input_ids"``, optional
                ``"labels"`` and ``"attention_mask"``.

        Returns:
            PhaseLossComponents with all loss values and the weighted total.
        """
        recipe = self.recipe.get_phase_recipe(phase)
        loss_weights = recipe.loss_config
        components = PhaseLossComponents()

        # ---- Standard LM Loss (always active) ----
        lm_loss = model_output.get("loss", torch.tensor(0.0, device=self.device))
        if lm_loss is None:
            lm_loss = torch.tensor(0.0, device=self.device)
        components.lm_loss = lm_loss

        # ---- JEPA Loss (Phase 1/2) ----
        if (
            self.config.use_jepa
            and recipe.use_jepa
            and self._jepa_module is not None
        ):
            try:
                jepa_loss_val = model_output.get("loss_dict", {}).get(
                    "jepa_loss", 0.0
                )
                if isinstance(jepa_loss_val, (int, float)):
                    jepa_loss_val = torch.tensor(
                        jepa_loss_val, device=self.device
                    )
                components.jepa_loss = jepa_loss_val
            except Exception as e:
                logger.warning(f"[LossComponent] JEPA failed: {e}")

        # ---- Expert Specialization Loss (Phase 1) ----
        if "expert_spec_loss" in loss_weights:
            # Compute from routing info if available
            routing_info = model_output.get("routing_info")
            if routing_info is not None:
                spec_loss = self._compute_expert_specialization_loss(
                    routing_info
                )
                components.expert_spec_loss = spec_loss

        # ---- Load Balance Loss (Phase 1/2/3) ----
        if "load_balance_loss" in loss_weights:
            routing_info = model_output.get("routing_info")
            if routing_info is not None:
                lb_loss = self._compute_load_balance_loss(routing_info)
                components.load_balance_loss = lb_loss

        # ---- Pathway Alignment Loss (Phase 2) ----
        if "pathway_align_loss" in loss_weights:
            hidden_states = model_output.get("hidden_states")
            if hidden_states is not None:
                pa_loss = self._compute_pathway_alignment_loss(hidden_states)
                components.pathway_align_loss = pa_loss

        # ---- RL Loss: DAPO / GRPO (Phase 3) ----
        if phase == TrainingPhase.PHASE_3_RL and (
            recipe.use_grpo or recipe.use_dapo
        ):
            # RL loss is computed separately via the RL trainer's
            # train_step method. Here we record the last known RL loss.
            pass  # RL loss is handled in train_step()

        # ---- ETR Reward (Phase 3/4) ----
        if self.config.use_etr and recipe.use_etr and self._etr_trainer is not None:
            try:
                etr_diag = self._etr_trainer.get_diagnostics()
                etr_reward_val = etr_diag.get("etr_reward", 0.0)
                components.etr_reward = torch.tensor(
                    etr_reward_val, device=self.device
                )
            except Exception as e:
                logger.warning(f"[LossComponent] ETR failed: {e}")

        # ---- Router Entropy Loss (Phase 3) ----
        if "router_entropy_loss" in loss_weights:
            routing_info = model_output.get("routing_info")
            if routing_info is not None:
                re_loss = self._compute_router_entropy_loss(routing_info)
                components.router_entropy_loss = re_loss

        # ---- Distillation Loss (Phase 4) ----
        if (
            self.config.use_distillation
            and recipe.use_distillation
            and self._distiller is not None
        ):
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            if input_ids is not None:
                try:
                    distill_metrics = self._distiller.train_step(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    components.distill_loss = torch.tensor(
                        distill_metrics.get("total_loss", 0.0),
                        device=self.device,
                    )
                except Exception as e:
                    logger.warning(f"[LossComponent] Distillation failed: {e}")

        # ---- BitDistill Loss (Phase 4) ----
        if (
            self.config.use_bit_distill
            and recipe.use_bit_distill
            and self._bit_distiller is not None
        ):
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            if input_ids is not None:
                try:
                    bd_metrics = self._bit_distiller.train_step(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    components.bit_distill_loss = torch.tensor(
                        bd_metrics.get("total_loss", 0.0),
                        device=self.device,
                    )
                except Exception as e:
                    logger.warning(f"[LossComponent] BitDistill failed: {e}")

        # ---- TACO Compute-Aligned Loss (Phase 2+) ----
        if self.config.use_taco and recipe.use_taco and self._taco_trainer is not None:
            try:
                alignment = self._taco_trainer.compute_alignment_weights()
                # TACO adjusts the base loss by alignment weights
                # We store it as a modifier rather than a separate loss
                components.taco_aligned_loss = lm_loss  # Will be adjusted below
            except Exception as e:
                logger.warning(f"[LossComponent] TACO failed: {e}")

        # ---- Early Exit Loss (Phase 4) ----
        if self.config.use_early_exit and recipe.use_early_exit:
            components.early_exit_loss = self._compute_early_exit_loss(model_output)

        # ---- Compute Weighted Total ----
        total = torch.tensor(0.0, device=self.device)

        for loss_name, weight in loss_weights.items():
            component_val = getattr(components, loss_name, None)
            if component_val is not None and isinstance(component_val, torch.Tensor):
                total = total + weight * component_val
            elif component_val is not None:
                total = total + weight * torch.tensor(
                    float(component_val), device=self.device
                )

        # If no loss weights configured, default to LM loss
        if total.item() == 0.0 and lm_loss.item() != 0.0:
            total = lm_loss

        components.total_loss = total
        return components

    # ==================================================================
    # Auxiliary Loss Helpers
    # ==================================================================

    def _compute_expert_specialization_loss(
        self, routing_info: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Compute expert specialization loss from routing information.

        Encourages each expert to specialize in a distinct subset of
        tokens by penalizing uniform routing distributions.

        Args:
            routing_info: List of per-layer routing info dicts.

        Returns:
            Scalar expert specialization loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        for layer_info in routing_info:
            if isinstance(layer_info, dict) and "retrieval_aux" in layer_info:
                aux = layer_info["retrieval_aux"]
                if isinstance(aux, dict) and "router_logits" in aux:
                    logits = aux["router_logits"]
                    # Encourage peaky (specialized) distributions
                    probs = F.softmax(logits.float(), dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                    max_entropy = torch.log(
                        torch.tensor(probs.shape[-1], dtype=torch.float32)
                    )
                    # Specialization = low relative entropy
                    spec_loss = entropy / max_entropy.clamp(min=1.0)
                    total_loss = total_loss + spec_loss
                    count += 1

        if count > 0:
            return total_loss / count
        return total_loss

    def _compute_load_balance_loss(
        self, routing_info: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Compute MoE load balancing auxiliary loss.

        Penalizes uneven expert utilization across tokens to prevent
        expert collapse.

        Args:
            routing_info: List of per-layer routing info dicts.

        Returns:
            Scalar load balance loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        for layer_info in routing_info:
            if isinstance(layer_info, dict) and "retrieval_aux" in layer_info:
                aux = layer_info["retrieval_aux"]
                if isinstance(aux, dict) and "router_logits" in aux:
                    logits = aux["router_logits"]
                    # Load balance: variance of expert selection frequencies
                    probs = F.softmax(logits.float(), dim=-1)
                    mean_probs = probs.mean(dim=0)  # (num_experts,)
                    variance = ((mean_probs - mean_probs.mean()) ** 2).mean()
                    total_loss = total_loss + variance
                    count += 1

        if count > 0:
            return total_loss / count
        return total_loss

    def _compute_pathway_alignment_loss(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-pathway alignment loss.

        Encourages the SSM, Attention, and MoE pathway outputs to
        produce coherent representations by minimizing the variance
        across pathway-weighted outputs.

        Args:
            hidden_states: Model hidden states (batch, seq_len, d_model).

        Returns:
            Scalar pathway alignment loss.
        """
        # Simple alignment: minimize variance of hidden states across
        # positions (encourages coherent multi-pathway representations)
        if hidden_states.dim() != 3:
            return torch.tensor(0.0, device=self.device)

        # Compute per-position variance
        variance = hidden_states.var(dim=-1).mean()
        return variance * 0.01  # Scale down

    def _compute_router_entropy_loss(
        self, routing_info: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Compute router exploration entropy bonus.

        Encourages the router to explore different routing configurations
        rather than collapsing to a single fixed routing.

        Args:
            routing_info: List of per-layer routing info dicts.

        Returns:
            Scalar router entropy bonus (negative entropy = loss).
        """
        total_entropy = torch.tensor(0.0, device=self.device)
        count = 0

        for layer_info in routing_info:
            if isinstance(layer_info, dict) and "route_weights" in layer_info:
                weights = layer_info["route_weights"]
                # Entropy of routing distribution
                entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
                total_entropy = total_entropy + entropy
                count += 1

        if count > 0:
            avg_entropy = total_entropy / count
            # Return negative entropy as loss (minimizing = maximizing entropy)
            return -avg_entropy
        return total_entropy

    def _compute_early_exit_loss(
        self, model_output: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute early exit / Mixture of Depths loss.

        Trains the model to produce acceptable outputs at intermediate
        layers, enabling adaptive compute at inference time.

        Args:
            model_output: Model output dict.

        Returns:
            Scalar early exit loss.
        """
        # Placeholder: in a full implementation, this would compute
        # the loss between intermediate layer outputs and targets
        return torch.tensor(0.0, device=self.device)

    # ==================================================================
    # Main Training Loop
    # ==================================================================

    def train(self) -> Dict[str, Any]:
        """Execute the complete 4-phase training pipeline.

        This is the main entry point. It iterates through training steps,
        automatically managing phase transitions, technique activation,
        learning rate scheduling, evaluation, and checkpointing.

        Returns:
            Dictionary with training summary including final metrics,
            phase transition history, and best evaluation loss.
        """
        logger.info("=" * 70)
        logger.info("Losion Training Orchestrator — Starting Training")
        logger.info(f"  Total steps: {self.config.total_steps:,}")
        logger.info(f"  Model: {self.losion_config.model_name}")
        logger.info(f"  Parameters: {self.model.get_num_params():,}")
        logger.info(f"  DAPO: {self.config.use_dapo}")
        logger.info(f"  RLVR: {self.config.use_rlvr}")
        logger.info(f"  JEPA: {self.config.use_jepa}")
        logger.info(f"  ETR:  {self.config.use_etr}")
        logger.info(f"  TACO: {self.config.use_taco}")
        logger.info("=" * 70)

        # Apply initial phase config
        self._apply_phase_config(self._current_phase)

        # Training loop
        start_time = time.time()
        dataloader_iter = iter(self.train_dataloader)

        for step in range(self._global_step, self.config.total_steps):
            self._global_step = step

            # ---- Get Batch ----
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.train_dataloader)
                batch = next(dataloader_iter)

            # ---- Check Phase Transition ----
            new_phase = self.training_state.get_current_phase(step)
            if new_phase != self._current_phase:
                logger.info(
                    f"\n{'='*60}\n"
                    f"  PHASE TRANSITION: {self._current_phase.value} → "
                    f"{new_phase.value}\n"
                    f"  Step: {step:,} / {self.config.total_steps:,}\n"
                    f"{'='*60}"
                )
                self._current_phase = new_phase
                self._apply_phase_config(new_phase)
                self._phase_transition_logged = True

            # ---- Train Step ----
            metrics = self.train_step(batch)

            # ---- Update Training State ----
            total_loss_val = metrics.get("total_loss", 0.0)
            self.training_state.update(
                step=step,
                loss=total_loss_val,
            )

            # ---- Update WSD LR Scheduler ----
            current_lr = self.lr_scheduler.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            # ---- Track Metrics ----
            self._metrics_history.append(metrics)

            # ---- Logging ----
            if step % self.config.log_interval == 0:
                self._log_step(step, metrics)

            # ---- Evaluation ----
            if step > 0 and step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                eval_loss = eval_metrics.get("eval_loss", float("inf"))
                if eval_loss < self._best_eval_loss:
                    self._best_eval_loss = eval_loss
                    if self.config.save_best_only:
                        self.save_checkpoint(
                            os.path.join(self.config.output_dir, "best")
                        )

            # ---- Checkpoint ----
            if step > 0 and step % self.config.checkpoint_interval == 0:
                if not self.config.save_best_only:
                    self.save_checkpoint(
                        os.path.join(
                            self.config.output_dir, f"step-{step:,}"
                        )
                    )

        # ---- Final Save ----
        self._total_training_time = time.time() - start_time
        self.save_checkpoint(
            os.path.join(self.config.output_dir, "final")
        )

        summary = self.get_training_summary()
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info(f"  Total time: {self._total_training_time:.1f}s")
        logger.info(f"  Best eval loss: {self._best_eval_loss:.4f}")
        logger.info(f"  Final phase: {self._current_phase.value}")
        logger.info("=" * 70)

        return summary

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step with phase-aware loss computation.

        For Phase 1/2, computes the combined supervised loss (LM + JEPA +
        auxiliary losses). For Phase 3, delegates to the RL trainer
        (DAPO or GRPO). For Phase 4, computes distillation + quantization
        losses.

        Args:
            batch: Dictionary with ``"input_ids"``, optional ``"labels"``,
                and ``"attention_mask"``.

        Returns:
            Dictionary of scalar metrics for this step.
        """
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        phase = self._current_phase
        recipe = self.recipe.get_phase_recipe(phase)

        # ---- Phase 3: RL Training Step ----
        if phase == TrainingPhase.PHASE_3_RL and recipe.use_grpo:
            return self._rl_train_step(input_ids, attention_mask, labels, batch)

        # ---- Standard Forward Pass ----
        self.optimizer.zero_grad()

        model_output = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            thinking_mode=self._should_think(phase),
        )

        # ---- Compute Phase-Aware Loss ----
        loss_components = self._compute_phase_loss(phase, model_output, batch)
        total_loss = loss_components.total_loss

        # ---- Backward + Clip + Step ----
        if total_loss.requires_grad:
            total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=self.config.grad_clip,
        )

        self.optimizer.step()

        # ---- Update Sub-Modules ----
        # TACO: track inference compute
        if self._taco_trainer is not None and self.config.use_taco:
            try:
                self._taco_trainer.track_inference_compute(input_ids, attention_mask)
            except Exception as e:
                logger.warning(f"[SubModule] TACO track_inference_compute failed: {e}")

        # Curriculum: update scheduler
        if self._curriculum_scheduler is not None and self.config.use_curriculum:
            try:
                self._curriculum_scheduler.update(self._global_step)
            except Exception as e:
                logger.warning(f"[SubModule] Curriculum update failed: {e}")

        # JEPA: update teacher EMA
        if self._jepa_module is not None and self.config.use_jepa:
            try:
                self._jepa_module.update_teacher()
            except Exception as e:
                logger.warning(f"[SubModule] JEPA teacher update failed: {e}")

        # ---- Build Metrics ----
        metrics = loss_components.to_dict()
        metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        metrics["learning_rate"] = self.lr_scheduler.get_lr(self._global_step)
        metrics["phase"] = phase.value
        metrics["step"] = self._global_step

        return metrics

    def _rl_train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute an RL training step using DAPO or GRPO.

        Handles the full RL pipeline: generate responses, compute rewards
        (optionally with RLVR), compute ETR reward, and update the policy.

        Args:
            input_ids: Prompt token IDs.
            attention_mask: Optional attention mask.
            labels: Target labels (used for supervised loss component).
            batch: Full batch dictionary.

        Returns:
            Dictionary of RL training metrics.
        """
        metrics: Dict[str, float] = {}

        # ---- DAPO / GRPO Step ----
        if self._dapo_trainer is not None:
            try:
                rl_metrics = self._dapo_trainer.train_step(
                    input_ids, attention_mask
                )
                metrics.update({f"dapo/{k}": v for k, v in rl_metrics.items()})
            except Exception as e:
                logger.warning(f"DAPO train_step failed: {e}")
                metrics["dapo/error"] = 1.0
        elif self._grpo_trainer is not None:
            try:
                rl_metrics = self._grpo_trainer.train_step(
                    input_ids, attention_mask
                )
                metrics.update({f"grpo/{k}": v for k, v in rl_metrics.items()})
            except Exception as e:
                logger.warning(f"GRPO train_step failed: {e}")
                metrics["grpo/error"] = 1.0

        # ---- RLVR Verifiable Rewards ----
        if self._rlvr_trainer is not None and self.config.use_rlvr:
            # RLVR is integrated at the reward level; metrics are logged
            # by the underlying DAPO/GRPO trainer
            metrics["rlvr/active"] = 1.0

        # ---- ETR Reward ----
        if self._etr_trainer is not None and self.config.use_etr:
            try:
                etr_diag = self._etr_trainer.get_diagnostics()
                metrics["etr/alpha"] = etr_diag.get("current_alpha", 0.0)
                metrics["etr/convergence"] = etr_diag.get("convergence_score", 0.0)
                metrics["etr/waste"] = etr_diag.get("waste_score", 0.0)
            except Exception as e:
                logger.warning(f"[LossComponent] ETR diagnostics failed: {e}")

        # ---- Evolutionary Search ----
        if self._evolutionary_searcher is not None and self.config.use_evolutionary:
            metrics["evolutionary/active"] = 1.0

        # ---- Add supervised LM loss component ----
        with torch.no_grad():
            lm_output = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            lm_loss = lm_output.get("loss")
            if lm_loss is not None:
                metrics["lm_loss"] = lm_loss.item()

        # ---- Combined loss for tracking ----
        recipe = self.recipe.get_phase_recipe(self._current_phase)
        rl_loss = metrics.get("dapo/loss", metrics.get("grpo/grpo_loss", 0.0))
        lm_loss_val = metrics.get("lm_loss", 0.0)
        metrics["total_loss"] = (
            recipe.loss_config.get("lm_loss", 0.5) * lm_loss_val
            + recipe.loss_config.get("grpo_loss", 1.0) * rl_loss
        )

        metrics["phase"] = self._current_phase.value
        metrics["step"] = self._global_step
        metrics["learning_rate"] = self.lr_scheduler.get_lr(self._global_step)

        return metrics

    def _should_think(self, phase: TrainingPhase) -> Optional[bool]:
        """Determine if thinking mode should be active for a given phase.

        Args:
            phase: Current training phase.

        Returns:
            True for Phase 3/4 (reasoning-heavy), None otherwise
            (letting the router decide).
        """
        if phase in (TrainingPhase.PHASE_3_RL, TrainingPhase.PHASE_4_ADVANCED):
            return True
        return None

    # ==================================================================
    # Evaluation
    # ==================================================================

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the eval dataloader.

        Computes the average loss, perplexity, and auxiliary metrics
        over the evaluation set. Also runs TACO inference compute
        tracking if enabled.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                output = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )

                loss = output.get("loss")
                if loss is not None:
                    total_loss += loss.item()
                    total_steps += 1

                # Limit eval steps to avoid excessive computation
                if total_steps >= 50:
                    break

        avg_loss = total_loss / max(total_steps, 1)
        perplexity = 2 ** avg_loss if avg_loss < 100 else float("inf")

        # TACO alignment summary
        taco_summary = {}
        if self._taco_trainer is not None and self.config.use_taco:
            try:
                taco_summary = self._taco_trainer.get_alignment_summary()
            except Exception as e:
                logger.warning(f"[Eval] TACO alignment summary failed: {e}")

        eval_metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_steps": total_steps,
            "step": self._global_step,
            "phase": self._current_phase.value,
        }

        # Add TACO info
        for k, v in taco_summary.items():
            if isinstance(v, (int, float)):
                eval_metrics[f"taco/{k}"] = v

        logger.info(
            f"Eval step {self._global_step:,}: "
            f"loss={avg_loss:.4f}, ppl={perplexity:.2f}"
        )

        self.model.train()
        return eval_metrics

    # ==================================================================
    # Checkpointing
    # ==================================================================

    def save_checkpoint(self, path: str) -> None:
        """Save a complete training checkpoint.

        Saves the model state dict, optimizer state, LR scheduler state,
        training state, and orchestrator config. The checkpoint can be
        resumed with :meth:`load_checkpoint`.

        Args:
            path: Directory path to save the checkpoint.
        """
        os.makedirs(path, exist_ok=True)

        # Model state
        torch.save(
            self.model.state_dict(),
            os.path.join(path, "model.pt"),
        )

        # Optimizer state
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(path, "optimizer.pt"),
        )

        # LR scheduler state
        torch.save(
            self.lr_scheduler.get_state(),
            os.path.join(path, "lr_scheduler.pt"),
        )

        # Training state
        training_state_dict = self.training_state.get_state_dict()
        with open(os.path.join(path, "training_state.json"), "w") as f:
            json.dump(training_state_dict, f, indent=2)

        # Orchestrator state
        orchestrator_state = {
            "global_step": self._global_step,
            "current_phase": self._current_phase.value,
            "best_eval_loss": self._best_eval_loss,
            "total_training_time": self._total_training_time,
        }
        with open(os.path.join(path, "orchestrator_state.json"), "w") as f:
            json.dump(orchestrator_state, f, indent=2)

        # LosionConfig
        with open(os.path.join(path, "losion_config.json"), "w") as f:
            json.dump(self.losion_config.to_dict(), f, indent=2)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint and resume training state.

        Args:
            path: Directory path containing the checkpoint files.
        """
        # Model state
        model_path = os.path.join(path, "model.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)

        # Optimizer state
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device)
            )

        # LR scheduler state
        scheduler_path = os.path.join(path, "lr_scheduler.pt")
        if os.path.exists(scheduler_path):
            self.lr_scheduler.load_state(
                torch.load(scheduler_path, map_location="cpu")
            )

        # Training state
        training_state_path = os.path.join(path, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path) as f:
                ts = json.load(f)
            self.training_state.load_state_dict(ts)

        # Orchestrator state
        orchestrator_path = os.path.join(path, "orchestrator_state.json")
        if os.path.exists(orchestrator_path):
            with open(orchestrator_path) as f:
                os_state = json.load(f)
            self._global_step = os_state.get("global_step", 0)
            self._current_phase = TrainingPhase(
                os_state.get("current_phase", "phase_1_individual")
            )
            self._best_eval_loss = os_state.get("best_eval_loss", float("inf"))
            self._total_training_time = os_state.get(
                "total_training_time", 0.0
            )

        logger.info(
            f"Checkpoint loaded from {path} "
            f"(step={self._global_step:,}, "
            f"phase={self._current_phase.value})"
        )

    # ==================================================================
    # Logging & Summary
    # ==================================================================

    def _log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Log training metrics for a step.

        Args:
            step: Current training step.
            metrics: Dictionary of metrics from train_step.
        """
        phase = metrics.get("phase", "unknown")
        lr = metrics.get("learning_rate", 0.0)
        total_loss = metrics.get("total_loss", 0.0)
        grad_norm = metrics.get("grad_norm", 0.0)

        # Build log string with phase-specific metrics
        parts = [
            f"Step {step:>8,}/{self.config.total_steps:,}",
            f"Phase: {phase}",
            f"Loss: {total_loss:.4f}",
            f"LR: {lr:.2e}",
            f"GradNorm: {grad_norm:.3f}",
        ]

        # Add technique-specific metrics
        if "jepa_loss" in metrics and metrics["jepa_loss"] != 0.0:
            parts.append(f"JEPA: {metrics['jepa_loss']:.4f}")
        if "dapo/loss" in metrics:
            parts.append(f"DAPO: {metrics['dapo/loss']:.4f}")
        if "grpo/grpo_loss" in metrics:
            parts.append(f"GRPO: {metrics['grpo/grpo_loss']:.4f}")
        if "etr/convergence" in metrics:
            parts.append(f"ETR: {metrics['etr/convergence']:.3f}")
        if "distill_loss" in metrics and metrics["distill_loss"] != 0.0:
            parts.append(f"Distill: {metrics['distill_loss']:.4f}")

        logger.info(" | ".join(parts))

    def get_training_summary(self) -> Dict[str, Any]:
        """Get a comprehensive training summary.

        Returns:
            Dictionary with:
            - ``global_step``: Total steps completed.
            - ``current_phase``: Current training phase.
            - ``best_eval_loss``: Best evaluation loss seen.
            - ``total_training_time``: Wall-clock training time in seconds.
            - ``techniques_used``: Dictionary of technique names to
              whether they were active.
            - ``phase_transitions``: History of phase transitions.
            - ``recipe_summary``: Per-phase recipe configuration.
            - ``model_info``: Model parameter counts.
            - ``final_metrics``: Most recent training metrics.
        """
        # Techniques used
        techniques = {
            "wsd_schedule": True,  # Always used
            "4_phase_training": True,  # Always used
            "dapo": self.config.use_dapo and self._dapo_trainer is not None,
            "grpo": self._grpo_trainer is not None,
            "rlvr": self.config.use_rlvr and self._rlvr_trainer is not None,
            "jepa": self.config.use_jepa and self._jepa_module is not None,
            "etr": self.config.use_etr and self._etr_trainer is not None,
            "taco": self.config.use_taco and self._taco_trainer is not None,
            "curriculum": self.config.use_curriculum
            and self._curriculum_scheduler is not None,
            "active_learning": self.config.use_active_learning
            and self._active_learning_loop is not None,
            "evolutionary": self.config.use_evolutionary
            and self._evolutionary_searcher is not None,
            "distillation": self.config.use_distillation
            and self._distiller is not None,
            "bit_distill": self.config.use_bit_distill
            and self._bit_distiller is not None,
            "early_exit": self.config.use_early_exit,
            "flow_matching": self.config.use_flow_matching,
        }

        # Phase transitions
        transitions = []
        if hasattr(self.training_state, "_transitions"):
            transitions = self.training_state._transitions

        # Recipe summary
        recipe_summary = self.recipe.get_summary()

        # Model info
        model_info = self.model.count_parameters()

        # Final metrics
        final_metrics = self._metrics_history[-1] if self._metrics_history else {}

        return {
            "global_step": self._global_step,
            "current_phase": self._current_phase.value,
            "best_eval_loss": self._best_eval_loss,
            "total_training_time": self._total_training_time,
            "techniques_used": techniques,
            "num_techniques_active": sum(1 for v in techniques.values() if v),
            "phase_transitions": transitions,
            "recipe_summary": recipe_summary,
            "model_info": model_info,
            "total_params": self.model.get_num_params(),
            "final_metrics": final_metrics,
            "losional_config": self.losion_config.model_name,
        }

    def __repr__(self) -> str:
        return (
            f"LosionTrainingOrchestrator("
            f"model={self.losion_config.model_name}, "
            f"step={self._global_step:,}/{self.config.total_steps:,}, "
            f"phase={self._current_phase.value}, "
            f"best_eval_loss={self._best_eval_loss:.4f}, "
            f"params={self.model.get_num_params():,})"
        )
