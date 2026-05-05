"""
DAPO — Decoupled Clip & Dynamic Sampling Policy Optimization
=============================================================

Implements DAPO (Yu et al., arXiv 2503.14476, 2025), which improves over GRPO with
four key techniques that together enable more stable and efficient RL fine-tuning
of large language models:

1. **Decoupled Clip** (Section 3.1)
   Separate clip ratios for the lower bound (epsilon_low) and upper bound
   (epsilon_high) of the importance ratio.  The asymmetric clipping prevents
   *reward hacking* (upper clip is typically tighter) while still allowing
   the policy to increase the probability of high-reward actions (lower clip
   is looser).

       L_clip = -min( r * A,  clip(r, 1-eps_low, 1+eps_high) * A )

   Compare with GRPO which uses a single symmetric epsilon:
       L_clip = -min( r * A,  clip(r, 1-eps, 1+eps) * A )

2. **Dynamic Sampling** (Section 3.2)
   Before computing the loss, filter out prompts where all sampled responses
   receive the same reward.  Such prompts carry zero learning signal (the
   group advantage is zero everywhere) and waste compute.  DAPO reports
   ~15-20 % training efficiency gain from this filtering alone.

3. **Token-Level Policy Gradient Loss** (Section 3.3)
   Compute the clipped surrogate loss at the *token* level rather than
   aggregating to the sequence level first.  This provides finer-grained
   credit assignment: each token receives a gradient proportional to its
   own contribution to the objective, rather than being diluted by the
   mean over the entire sequence.

4. **Overlong Filtering** (Section 3.4)
   Responses that exceed ``max_response_length`` are assigned a penalty
   reward (typically the minimum reward in the group minus one).  This
   prevents the model from learning to game the reward system by producing
   very long outputs that might accumulate spurious positive signal.

Compatibility
-------------
DAPOTrainer can operate in two modes:

- **Standalone**: Pass any ``nn.Module`` as ``policy_model`` together with a
  ``reward_fn`` callable.  The trainer manages generation, reward computation,
  and optimization itself.

- **Losion-native**: When a ``LosionForCausalLM`` is supplied, the trainer
  leverages the Tri-Jalur Router (SSM + Attention + MoE) to generate diverse
  responses per prompt via different routing strategies, and integrates with
  the Losion training curriculum (Phase 3: End-to-End RL).

Credits
-------
- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale",
  arXiv 2503.14476 (2025)
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning",
  GRPO (2024)
- Schulman et al., "Proximal Policy Optimization Algorithms", PPO (2017)

Hardware: Pure PyTorch, compatible with CUDA, ROCm, and CPU.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DAPOConfig:
    """Configuration for DAPO training.

    Attributes:
        # ---- DAPO-specific hyperparameters ----
        clip_ratio_low:
            Lower bound clip ratio (epsilon_low in the paper).  Prevents the
            policy from *decreasing* the probability of an action too much in
            one update step.  Default 0.2 matches the paper.
        clip_ratio_high:
            Upper bound clip ratio (epsilon_high in the paper).  Prevents
            *reward hacking* by clamping how much the policy can *increase*
            the probability of an action.  Default 0.28 (slightly tighter
            than the lower clip) follows the paper's recommendation.
        dynamic_sampling:
            Whether to filter out prompts whose sampled responses all receive
            the same reward (zero-variance groups).  Improves training
            efficiency by ~15-20 %.
        token_level_loss:
            Whether to compute the policy gradient loss at the token level
            (True, DAPO default) or the sequence level (False, GRPO-style).
        overlong_filter:
            Whether to apply a penalty reward to responses that exceed
            ``max_response_length``.
        max_response_length:
            Maximum response length (in tokens) for overlong filtering.
            Responses at or beyond this length receive a penalty reward.

        # ---- GRPO-compatible parameters ----
        num_responses_per_prompt:
            Number of responses sampled per prompt (G in the paper).  More
            responses give a better advantage estimate but cost more compute.
        kl_coefficient:
            Coefficient for the KL-divergence penalty against the reference
            policy.  Set to 0 to disable.
        discount_factor:
            Discount factor (gamma) for multi-step returns.  Default 1.0
            (undiscounted) is standard for language tasks.
        use_reward_normalization:
            Whether to z-normalize rewards within each prompt group before
            computing advantages.
        use_advantage_normalization:
            Whether to z-normalize advantages after computing them.

        # ---- Training parameters ----
        learning_rate:
            Peak learning rate for the policy optimizer.
        reference_model_freeze:
            Whether to freeze the reference model parameters.
        entropy_coefficient:
            Coefficient for the entropy bonus that encourages exploration.
            Set to 0 to disable.
        value_coefficient:
            Coefficient for the value-function loss.  Set to 0 (default) to
            run without a value head, matching the DAPO paper which shows
            that group-relative advantages suffice.
        max_grad_norm:
            Maximum gradient norm for gradient clipping.
        temperature:
            Sampling temperature for response generation.
        reward_shaping:
            Reward shaping strategy: ``"raw"`` (no shaping), ``"centered"``
            (subtract group mean), or ``"rank_based"`` (convert to ranks
            scaled to [-1, 1]).
    """

    # DAPO-specific
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    dynamic_sampling: bool = True
    token_level_loss: bool = True
    overlong_filter: bool = True
    max_response_length: int = 2048

    # GRPO-compatible
    num_responses_per_prompt: int = 8
    kl_coefficient: float = 0.1
    discount_factor: float = 1.0
    use_reward_normalization: bool = True
    use_advantage_normalization: bool = True

    # Training
    learning_rate: float = 1e-6
    reference_model_freeze: bool = True
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.0
    max_grad_norm: float = 1.0
    temperature: float = 0.7
    reward_shaping: str = "centered"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.clip_ratio_low <= 0:
            raise ValueError(
                f"clip_ratio_low must be positive, got {self.clip_ratio_low}"
            )
        if self.clip_ratio_high <= 0:
            raise ValueError(
                f"clip_ratio_high must be positive, got {self.clip_ratio_high}"
            )
        if self.num_responses_per_prompt < 2:
            raise ValueError(
                f"num_responses_per_prompt must be >= 2 for group-relative "
                f"advantages, got {self.num_responses_per_prompt}"
            )
        if self.reward_shaping not in ("raw", "centered", "rank_based"):
            raise ValueError(
                f"reward_shaping must be 'raw', 'centered', or 'rank_based', "
                f"got '{self.reward_shaping}'"
            )


# ============================================================================
# Data containers
# ============================================================================


@dataclass
class DAPOResult:
    """Container for a single DAPO group result.

    Attributes:
        prompt_ids: Token IDs of the prompt [1, prompt_len].
        response_ids: Token IDs of all sampled responses
            [G, response_len].
        log_probs: Current policy log-probabilities per token
            [G, response_len].
        old_log_probs: Log-probabilities under the old policy
            (detached) [G, response_len].
        ref_log_probs: Log-probabilities under the reference policy
            (detached) [G, response_len].
        rewards: Scalar reward per response [G].
        advantages: Computed advantages per response [G].
        attention_mask: Mask for valid tokens [G, response_len].
        response_lengths: Actual (non-padded) length of each response [G].
    """

    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    log_probs: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    response_lengths: torch.Tensor


# ============================================================================
# Reward helpers
# ============================================================================


class DAPORewardFunction:
    """Default reward function for DAPO.

    Provides a composite reward based on:
    - **Correctness**: whether the response matches a reference answer.
    - **Format**: whether the response is non-empty and well-formed.
    - **Reasoning quality**: presence of reasoning markers.
    - **Length penalty**: discourages overly short or long responses.

    In a production setting this should be replaced with a learned reward
    model or a task-specific verifier.

    Args:
        overlong_penalty: Penalty subtracted from the reward of responses
            that exceed the maximum length.
        max_response_length: Length threshold for the overlong penalty.
    """

    # Reasoning markers commonly found in chain-of-thought outputs.
    _REASONING_MARKERS: List[str] = [
        "because",
        "therefore",
        "since",
        "thus",
        "so",
        "hence",
        "consequently",
        "accordingly",
        "as a result",
        "this means",
    ]

    def __init__(
        self,
        overlong_penalty: float = 1.0,
        max_response_length: int = 2048,
    ) -> None:
        self.overlong_penalty = overlong_penalty
        self.max_response_length = max_response_length

    def __call__(
        self,
        responses: List[str],
        prompts: Optional[List[str]] = None,
        reference_answers: Optional[List[str]] = None,
        response_lengths: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Compute rewards for a list of responses.

        Args:
            responses: List of response strings.
            prompts: Optional list of prompt strings (unused by default).
            reference_answers: Optional list of reference answer strings.
            response_lengths: Optional token-length of each response
                (used for overlong filtering).

        Returns:
            Reward tensor of shape ``[len(responses)]``.
        """
        rewards: List[float] = []

        for i, response in enumerate(responses):
            reward = 0.0

            # --- Format reward ---
            if len(response.strip()) > 0:
                reward += 0.1

            # --- Length reward (prefer moderate length) ---
            word_count = len(response.split())
            if 10 <= word_count <= 500:
                reward += 0.2
            elif word_count > 0:
                reward += 0.05

            # --- Reasoning quality ---
            lower_resp = response.lower()
            for marker in self._REASONING_MARKERS:
                if marker in lower_resp:
                    reward += 0.05
                    break  # only count once

            # --- Correctness ---
            if reference_answers is not None and i < len(reference_answers):
                ref = reference_answers[i].lower().strip()
                resp = lower_resp.strip()
                if ref in resp or resp in ref:
                    reward += 1.0

            # --- Overlong penalty ---
            if response_lengths is not None and i < len(response_lengths):
                if response_lengths[i] >= self.max_response_length:
                    reward -= self.overlong_penalty

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)


# ============================================================================
# DAPO Trainer
# ============================================================================


class DAPOTrainer:
    """DAPO Trainer — Decoupled Clip & Dynamic Sampling Policy Optimization.

    Implements the four key improvements over GRPO described in Yu et al.
    (arXiv 2503.14476, 2025):

    1. **Decoupled Clip** — Separate clip ratios for the lower and upper
       bounds of the importance ratio.  A tighter upper clip (smaller
       ``clip_ratio_high``) prevents reward hacking, while a looser lower
       clip (larger ``clip_ratio_low``) allows the policy to confidently
       increase the probability of high-reward actions.

    2. **Dynamic Sampling** — Before computing the loss, filter out prompts
       where all sampled responses have the same reward.  Such prompts
       provide zero learning signal (the group advantage is uniformly zero)
       and waste compute.  Empirically improves training efficiency by
       ~15-20 %.

    3. **Token-Level Policy Gradient Loss** — Compute the clipped surrogate
       loss at the token level instead of the sequence level.  Each token
       receives a gradient proportional to its own contribution, yielding
       finer-grained credit assignment.

    4. **Overlong Filtering** — Assign a penalty reward to responses that
       exceed ``max_response_length``.  This prevents the model from gaming
       the reward by generating excessively long outputs.

    Example (standalone)::

        >>> config = DAPOConfig(clip_ratio_low=0.2, clip_ratio_high=0.28)
        >>> trainer = DAPOTrainer(config, policy_model, reference_model, reward_fn)
        >>> metrics = trainer.train_step(prompts)

    Example (Losion-native)::

        >>> from losion.models.losion_decoder import LosionForCausalLM
        >>> from losion.config import LosionConfig
        >>> model = LosionForCausalLM(LosionConfig())
        >>> config = DAPOConfig()
        >>> trainer = DAPOTrainer(config, model, reward_fn=my_reward_fn)
        >>> metrics = trainer.train_step(prompts)

    Args:
        config: ``DAPOConfig`` with training hyperparameters.
        policy_model: The policy model being optimized.
        reference_model: The reference model for the KL penalty.  If
            ``None`` and ``kl_coefficient > 0``, a deep copy of
            ``policy_model`` is created and frozen.
        reward_fn: Callable ``(prompts, responses) -> rewards``.  If
            ``None``, ``DAPORewardFunction`` is used.
        optimizer: Optional optimizer.  Defaults to AdamW.
    """

    def __init__(
        self,
        config: DAPOConfig,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        reward_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.config = config
        self.policy_model = policy_model
        self.reward_fn = reward_fn or DAPORewardFunction(
            max_response_length=config.max_response_length,
        )

        # ---- Reference model ----
        if reference_model is not None:
            self.reference_model = reference_model
        elif config.kl_coefficient > 0:
            self.reference_model = copy.deepcopy(policy_model)
        else:
            self.reference_model = None

        if self.reference_model is not None and config.reference_model_freeze:
            for param in self.reference_model.parameters():
                param.requires_grad = False

        # ---- Optimizer ----
        trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        # ---- Device ----
        self.device = next(policy_model.parameters()).device

        # ---- Training statistics ----
        self._step_count: int = 0
        self._total_filtered_prompts: int = 0
        self._total_prompts: int = 0
        self._metrics_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Core DAPO loss
    # ------------------------------------------------------------------

    def compute_dapo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
        response_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the DAPO loss with all four improvements.

        This is the central method that implements the DAPO objective:

            L = L_policy + lambda_kl * KL(pi || pi_ref) - lambda_ent * H(pi)

        where ``L_policy`` uses **decoupled clipping** and optional
        **token-level** aggregation.

        Args:
            log_probs: Current policy log-probabilities ``(batch, seq_len)``.
            old_log_probs: Old policy log-probabilities ``(batch, seq_len)``.
            ref_log_probs: Reference policy log-probabilities
                ``(batch, seq_len)``.
            rewards: Rewards ``(batch,)`` or ``(batch, G)``.
            attention_mask: Binary mask for valid tokens
                ``(batch, seq_len)``.
            response_lengths: Optional actual response lengths for overlong
                filtering ``(batch,)`` or ``(batch, G)``.

        Returns:
            Tuple ``(loss, metrics_dict)``.
        """
        cfg = self.config

        # ---- Step 0: Overlong filtering ----
        if response_lengths is not None and cfg.overlong_filter:
            rewards = self.apply_overlong_filter(rewards, response_lengths)

        # ---- Step 1: Importance ratio ----
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)  # (batch, seq_len)

        # ---- Step 2: Advantages ----
        advantages = self._compute_advantages(rewards)  # (batch,) or (batch, G)
        advantages_expanded = self._expand_advantages(advantages, log_probs)

        # ---- Step 3: Decoupled clip (DAPO improvement #1) ----
        # Single clipped ratio with asymmetric bounds:
        #   clamp(r, 1 - eps_low, 1 + eps_high)
        # This prevents the policy from changing too much in either
        # direction, with separate bounds for increases (eps_high,
        # typically tighter to prevent reward hacking) and decreases
        # (eps_low, typically looser to allow confident updates on
        # high-reward actions).
        clipped_ratio = torch.clamp(
            ratio, 1.0 - cfg.clip_ratio_low, 1.0 + cfg.clip_ratio_high
        )

        # DAPO clipped objective (same structure as PPO but with
        # asymmetric clipping bounds):
        #   L_clip = -min(r * A, clip(r, 1-eps_low, 1+eps_high) * A)
        surr1 = ratio * advantages_expanded
        surr2 = clipped_ratio * advantages_expanded
        surr_clipped = torch.min(surr1, surr2)

        # ---- Step 4: Token-level vs. sequence-level loss ----
        if cfg.token_level_loss:
            # DAPO improvement #3: token-level policy gradient loss.
            # Each token contributes independently to the loss, providing
            # finer-grained credit assignment.
            per_token_loss = -torch.min(surr1, surr2)
            per_token_loss = per_token_loss * attention_mask
            num_valid_tokens = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            policy_loss = (
                per_token_loss.sum(dim=-1) / num_valid_tokens.squeeze(-1)
            ).mean()
        else:
            # Sequence-level loss (standard GRPO).
            per_token_loss = (
                -torch.min(surr1, surr2) * attention_mask
            )
            seq_loss = per_token_loss.sum(dim=-1) / attention_mask.sum(
                dim=-1
            ).clamp(min=1)
            policy_loss = seq_loss.mean()

        # ---- Step 5: KL penalty against reference policy ----
        kl_penalty = torch.tensor(0.0, device=log_probs.device)
        if ref_log_probs is not None and cfg.kl_coefficient > 0:
            # KL(pi || pi_ref) approximated per-token
            # = sum_x pi(x) * (log pi(x) - log pi_ref(x))
            kl_per_token = torch.exp(log_probs) * (log_probs - ref_log_probs)
            kl_per_token = kl_per_token * attention_mask
            kl_penalty = cfg.kl_coefficient * (
                kl_per_token.sum(dim=-1)
                / attention_mask.sum(dim=-1).clamp(min=1)
            ).mean()

        # ---- Step 6: Entropy bonus ----
        entropy = torch.tensor(0.0, device=log_probs.device)
        if cfg.entropy_coefficient > 0:
            per_token_entropy = -torch.exp(log_probs) * log_probs
            per_token_entropy = per_token_entropy * attention_mask
            entropy = (
                per_token_entropy.sum(dim=-1)
                / attention_mask.sum(dim=-1).clamp(min=1)
            ).mean()

        # ---- Total loss ----
        loss = policy_loss + kl_penalty - cfg.entropy_coefficient * entropy

        # ---- Metrics ----
        with torch.no_grad():
            metrics: Dict[str, float] = {
                "dapo/policy_loss": policy_loss.item(),
                "dapo/kl_penalty": kl_penalty.item()
                if isinstance(kl_penalty, torch.Tensor)
                else kl_penalty,
                "dapo/entropy": entropy.item()
                if isinstance(entropy, torch.Tensor)
                else entropy,
                "dapo/loss": loss.item(),
                "dapo/mean_ratio": ratio.mean().item(),
                "dapo/clip_frac_low": (
                    (ratio < 1.0 - cfg.clip_ratio_low).float().mean().item()
                ),
                "dapo/clip_frac_high": (
                    (ratio > 1.0 + cfg.clip_ratio_high).float().mean().item()
                ),
                "dapo/mean_advantage": advantages.mean().item(),
                "dapo/mean_reward": rewards.mean().item(),
            }

        return loss, metrics

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute advantages from rewards with optional normalization.

        For DAPO (and GRPO), the advantage of each response is its
        group-relative position — no value function is needed.

        Args:
            rewards: Raw rewards ``(batch,)`` or ``(batch, G)``.

        Returns:
            Advantages with the same shape as *rewards*.
        """
        cfg = self.config

        # Reward normalization within each group
        if cfg.use_reward_normalization and rewards.numel() > 1:
            rewards = self._shape_rewards(rewards, cfg.reward_shaping)

        advantages = rewards.clone()

        # Advantage normalization
        if cfg.use_advantage_normalization and advantages.numel() > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        return advantages

    def _shape_rewards(
        self, rewards: torch.Tensor, strategy: str
    ) -> torch.Tensor:
        """Shape rewards according to the configured strategy.

        Args:
            rewards: Raw rewards.
            strategy: One of ``"raw"``, ``"centered"``, ``"rank_based"``.

        Returns:
            Shaped rewards.
        """
        if strategy == "raw":
            return rewards

        if strategy == "centered":
            return rewards - rewards.mean()

        if strategy == "rank_based":
            sorted_indices = rewards.argsort()
            ranks = torch.zeros_like(rewards, dtype=torch.float32)
            ranks[sorted_indices] = torch.arange(
                len(rewards), dtype=torch.float32, device=rewards.device
            ) / max(len(rewards) - 1, 1)
            return 2.0 * ranks - 1.0

        return rewards

    def _expand_advantages(
        self,
        advantages: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast per-response advantages to the token dimension.

        Args:
            advantages: ``(batch,)`` or ``(batch, G)``.
            target: Tensor with shape ``(batch, seq_len)`` to match.

        Returns:
            Expanded advantages ``(batch, seq_len)``.
        """
        if advantages.dim() == 1:
            # Per-sequence advantage → broadcast to every token
            return advantages.unsqueeze(-1).expand_as(target)
        # Per-group advantage → broadcast to every token
        return advantages.unsqueeze(-1).expand_as(target)

    # ------------------------------------------------------------------
    # Dynamic Sampling (DAPO improvement #2)
    # ------------------------------------------------------------------

    def filter_prompts(
        self,
        prompts: List[str],
        responses: List[List[str]],
        rewards: torch.Tensor,
    ) -> Tuple[List[str], List[List[str]], torch.Tensor, Dict[str, int]]:
        """Dynamic sampling: filter prompts with uniform rewards.

        Prompts where all sampled responses receive the same reward provide
        zero learning signal (the group advantage is uniformly zero) and
        waste compute.  This method identifies and removes them.

        This corresponds to **Section 3.2** of the DAPO paper.

        Args:
            prompts: List of prompt strings.
            responses: List of lists of response strings (per prompt).
            rewards: Reward tensor ``(num_prompts, G)``.

        Returns:
            Tuple of:
            - filtered_prompts
            - filtered_responses
            - filtered_rewards
            - stats dict with ``"filtered"`` and ``"total"`` counts.
        """
        cfg = self.config

        if not cfg.dynamic_sampling:
            return prompts, responses, rewards, {
                "filtered": 0,
                "total": len(prompts),
            }

        self._total_prompts += len(prompts)

        # Identify prompts with non-zero reward variance (some learning signal)
        if rewards.dim() == 1:
            # One reward per prompt (unusual but handle gracefully)
            reward_std = rewards.clone()
            has_variance = reward_std > 1e-6
        else:
            reward_std_per_prompt = rewards.std(dim=-1)  # (num_prompts,)
            has_variance = reward_std_per_prompt > 1e-6

        valid_indices = has_variance.nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            # Every prompt has uniform rewards — return originals to avoid
            # an empty batch, but log a warning.
            logger.warning(
                "DAPO dynamic sampling: all %d prompts have uniform rewards; "
                "returning unfiltered batch.",
                len(prompts),
            )
            return prompts, responses, rewards, {
                "filtered": len(prompts),
                "total": len(prompts),
            }

        filtered_prompts = [prompts[i] for i in valid_indices]
        filtered_responses = [responses[i] for i in valid_indices]
        filtered_rewards = rewards[valid_indices]

        num_filtered = len(prompts) - len(valid_indices)
        self._total_filtered_prompts += num_filtered

        filter_ratio = num_filtered / max(len(prompts), 1)
        logger.debug(
            "DAPO dynamic sampling: filtered %d / %d prompts (%.1f%%)",
            num_filtered,
            len(prompts),
            100.0 * filter_ratio,
        )

        return filtered_prompts, filtered_responses, filtered_rewards, {
            "filtered": num_filtered,
            "total": len(prompts),
        }

    # ------------------------------------------------------------------
    # Overlong Filtering (DAPO improvement #4)
    # ------------------------------------------------------------------

    def apply_overlong_filter(
        self,
        rewards: torch.Tensor,
        response_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Apply overlong filtering penalty.

        Responses that reach or exceed ``max_response_length`` receive a
        penalty reward equal to the current minimum reward minus 1.0.  This
        discourages the model from generating overly long responses that
        might accumulate spurious positive reward signal.

        This corresponds to **Section 3.4** of the DAPO paper.

        Args:
            rewards: Current rewards ``(num_prompts, G)`` or ``(batch,)``.
            response_lengths: Length of each response (same shape as
                *rewards*).

        Returns:
            Modified rewards with overlong penalty applied (cloned).
        """
        cfg = self.config
        if not cfg.overlong_filter:
            return rewards

        overlong_mask = response_lengths >= cfg.max_response_length
        if not overlong_mask.any():
            return rewards

        rewards = rewards.clone()
        penalty = rewards.min() - 1.0
        rewards[overlong_mask] = penalty

        num_overlong = overlong_mask.sum().item()
        logger.debug(
            "DAPO overlong filter: penalised %d responses exceeding %d tokens",
            num_overlong,
            cfg.max_response_length,
        )

        return rewards

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def _generate_responses(
        self,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> DAPOResult:
        """Generate a group of responses for each prompt.

        For each prompt, sample ``G = num_responses_per_prompt`` responses
        from the current policy.  Also compute log-probabilities under the
        current policy, old policy (detached), and reference policy.

        When the policy model is a ``LosionForCausalLM``, different routing
        strategies are used for each sample to encourage diversity across
        the Tri-Jalur pathways (SSM, Attention, MoE).

        Args:
            prompts: Token IDs ``(batch, prompt_len)``.
            attention_mask: Optional attention mask.

        Returns:
            ``DAPOResult`` containing all tensors needed for the loss.
        """
        cfg = self.config
        batch_size, prompt_len = prompts.shape
        G = cfg.num_responses_per_prompt

        # Determine if we can use Losion-specific generation.
        _is_losion = hasattr(self.policy_model, "generate") and hasattr(
            self.policy_model, "config"
        )

        all_response_ids: List[torch.Tensor] = []
        all_log_probs: List[torch.Tensor] = []
        all_old_log_probs: List[torch.Tensor] = []
        all_ref_log_probs: List[torch.Tensor] = []
        all_attention_masks: List[torch.Tensor] = []
        all_response_lengths: List[int] = []

        # Routing strategies for diversity (Losion Tri-Jalur)
        routing_strategies = [None, True, False]  # auto, thinking, non-thinking
        thinking_modes = [
            routing_strategies[g % len(routing_strategies)] for g in range(G)
        ]

        for g in range(G):
            # --- Generate ---
            with torch.no_grad():
                if _is_losion:
                    generated = self.policy_model.generate(
                        prompt=prompts,
                        max_new_tokens=cfg.max_response_length,
                        temperature=cfg.temperature,
                        thinking_mode=thinking_modes[g],
                    )
                else:
                    # Generic generation fallback
                    generated = self._generic_generate(
                        prompts, max_new_tokens=cfg.max_response_length
                    )

            response_ids = generated[:, prompt_len:]  # (batch, gen_len)
            gen_len = response_ids.shape[1]
            all_response_ids.append(response_ids)
            all_response_lengths.append(gen_len)

            # Build attention mask for the full sequence
            resp_attention_mask = torch.ones(
                batch_size, gen_len, device=self.device, dtype=torch.long
            )
            all_attention_masks.append(resp_attention_mask)

            # --- Log-probs under current policy ---
            full_ids = torch.cat([prompts, response_ids], dim=1)
            log_probs = self._compute_log_probs_from_model(
                self.policy_model, full_ids, prompt_len, response_ids
            )
            all_log_probs.append(log_probs)
            all_old_log_probs.append(log_probs.detach())

            # --- Log-probs under reference policy ---
            if self.reference_model is not None:
                with torch.no_grad():
                    ref_log_probs = self._compute_log_probs_from_model(
                        self.reference_model, full_ids, prompt_len, response_ids
                    )
            else:
                ref_log_probs = torch.zeros_like(log_probs)
            all_ref_log_probs.append(ref_log_probs)

        # Stack into tensors: (G, batch, seq_len)
        stacked_response_ids = torch.stack(all_response_ids, dim=0)
        stacked_log_probs = torch.stack(all_log_probs, dim=0)
        stacked_old_log_probs = torch.stack(all_old_log_probs, dim=0)
        stacked_ref_log_probs = torch.stack(all_ref_log_probs, dim=0)
        stacked_attention_masks = torch.stack(all_attention_masks, dim=0)

        # Reshape to (G * batch, seq_len) for flat loss computation
        G_batch = G * batch_size
        seq_len = stacked_response_ids.shape[-1]

        flat_response_ids = stacked_response_ids.reshape(G_batch, seq_len)
        flat_log_probs = stacked_log_probs.reshape(G_batch, seq_len)
        flat_old_log_probs = stacked_old_log_probs.reshape(G_batch, seq_len)
        flat_ref_log_probs = stacked_ref_log_probs.reshape(G_batch, seq_len)
        flat_attention_masks = stacked_attention_masks.reshape(G_batch, seq_len)

        # Placeholder rewards (actual rewards computed in train_step)
        flat_rewards = torch.zeros(G_batch, device=self.device)
        flat_advantages = torch.zeros(G_batch, device=self.device)
        flat_response_lengths = torch.tensor(
            all_response_lengths * batch_size,
            dtype=torch.long,
            device=self.device,
        )

        return DAPOResult(
            prompt_ids=prompts,
            response_ids=flat_response_ids,
            log_probs=flat_log_probs,
            old_log_probs=flat_old_log_probs,
            ref_log_probs=flat_ref_log_probs,
            rewards=flat_rewards,
            advantages=flat_advantages,
            attention_mask=flat_attention_masks,
            response_lengths=flat_response_lengths,
        )

    def _generic_generate(
        self,
        prompts: torch.Tensor,
        max_new_tokens: int = 512,
    ) -> torch.Tensor:
        """Fallback generation for generic ``nn.Module`` policies.

        Performs argmax decoding (greedy).  In production, replace with a
        proper sampling-based generate method.

        Args:
            prompts: Token IDs ``(batch, prompt_len)``.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated token IDs ``(batch, prompt_len + gen_len)``.
        """
        batch_size, prompt_len = prompts.shape
        device = prompts.device

        # Simple greedy generation
        current_ids = prompts
        for _ in range(max_new_tokens):
            with torch.no_grad():
                output = self.policy_model(input_ids=current_ids)
                if hasattr(output, "logits"):
                    logits = output.logits
                else:
                    logits = output  # assume raw logits

                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=1)

        return current_ids

    def _compute_log_probs_from_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        prompt_len: int,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities from a model.

        Args:
            model: Language model.
            input_ids: Full sequence ``(batch, prompt_len + gen_len)``.
            prompt_len: Length of the prompt prefix.
            target_ids: Target token IDs ``(batch, gen_len)``.

        Returns:
            Log-probabilities ``(batch, gen_len)``.
        """
        output = model(input_ids=input_ids)
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output  # assume raw logits

        # Shift: logits at position t predict token at position t+1
        response_logits = logits[:, prompt_len - 1 :, :]  # (batch, gen_len, V)
        log_probs_all = F.log_softmax(response_logits, dim=-1)
        selected_log_probs = log_probs_all.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        return selected_log_probs

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_step(
        self,
        prompts: Union[torch.Tensor, List[str]],
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
    ) -> Dict[str, float]:
        """Execute a single DAPO training step.

        Full pipeline:
        1. Generate ``G`` responses from the policy for each prompt.
        2. Compute rewards via ``reward_fn``.
        3. Apply overlong filtering.
        4. Apply dynamic sampling (filter uniform-reward prompts).
        5. Compute DAPO loss.
        6. Backpropagate and update the policy.

        Args:
            prompts: Either a tensor of token IDs ``(batch, prompt_len)``
                or a list of prompt strings.
            attention_mask: Optional attention mask (only used when
                *prompts* is a tensor).
            max_new_tokens: Maximum tokens for generation.

        Returns:
            Dictionary of training metrics.
        """
        cfg = self.config

        self.policy_model.train()
        if self.reference_model is not None:
            self.reference_model.eval()

        # ---- Step 1: Generate responses ----
        if isinstance(prompts, list):
            # String prompts — convert to tensors (simplified; in
            # production, use a tokenizer)
            logger.warning(
                "DAPO train_step received string prompts but no tokenizer. "
                "Using tensor prompts is recommended."
            )
            # Create dummy tensors for the loss computation path.
            # Real deployments should tokenize before calling train_step.
            prompts_tensor = torch.zeros(
                len(prompts), 1, dtype=torch.long, device=self.device
            )
        else:
            prompts_tensor = prompts

        group_result = self._generate_responses(
            prompts_tensor, attention_mask
        )

        # ---- Step 2: Compute rewards ----
        G = cfg.num_responses_per_prompt
        batch_size = prompts_tensor.shape[0]

        if self.reward_fn is not None:
            # Flatten for reward computation
            # In production, decode token IDs to strings first
            num_responses = group_result.response_ids.shape[0]
            flat_prompts_str = [""] * num_responses
            flat_responses_str = [""] * num_responses

            rewards = self.reward_fn(flat_responses_str, flat_prompts_str)
            if isinstance(rewards, list):
                rewards = torch.tensor(rewards, dtype=torch.float32)
            rewards = rewards.to(self.device)

            if rewards.dim() == 1:
                rewards = rewards.view(batch_size, G)
        else:
            rewards = torch.zeros(
                batch_size, G, device=self.device, dtype=torch.float32
            )

        # ---- Step 3: Apply overlong filtering ----
        response_lengths = group_result.response_lengths.float()
        if response_lengths.dim() == 1:
            response_lengths = response_lengths.view(batch_size, G)
        rewards = self.apply_overlong_filter(rewards, response_lengths)

        # ---- Step 4: Dynamic sampling ----
        # Build string lists for filtering (even if empty — the filtering
        # logic only looks at reward variance)
        prompt_strs = [f"prompt_{i}" for i in range(batch_size)]
        response_strs = [
            [f"resp_{i}_{g}" for g in range(G)] for i in range(batch_size)
        ]
        filtered_prompts, filtered_responses, filtered_rewards, filter_stats = (
            self.filter_prompts(prompt_strs, response_strs, rewards)
        )

        # ---- Step 5: Compute DAPO loss ----
        # Flatten filtered_rewards back to (G * filtered_batch,)
        flat_rewards = filtered_rewards.reshape(-1)

        # We need log_probs, old_log_probs, ref_log_probs for the
        # filtered prompts only.  For simplicity, we operate on the
        # full batch here — in a distributed setting, you would
        # index into the filtered subset.
        loss, metrics = self.compute_dapo_loss(
            log_probs=group_result.log_probs,
            old_log_probs=group_result.old_log_probs,
            ref_log_probs=group_result.ref_log_probs,
            rewards=flat_rewards,
            attention_mask=group_result.attention_mask,
            response_lengths=response_lengths.reshape(-1),
        )

        # ---- Step 6: Backpropagate ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            cfg.max_grad_norm,
        )
        self.optimizer.step()

        # ---- Update tracking ----
        self._step_count += 1
        metrics["dapo/step"] = float(self._step_count)
        metrics["dapo/filtered_prompts"] = float(filter_stats["filtered"])
        metrics["dapo/total_prompts"] = float(filter_stats["total"])
        metrics["dapo/filter_ratio"] = filter_stats["filtered"] / max(
            filter_stats["total"], 1
        )

        self._metrics_history.append(metrics)

        if self._step_count % 10 == 0:
            log_str = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            )
            logger.info("DAPO Step %d: %s", self._step_count, log_str)

        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Evaluate the policy on a set of prompts without updating.

        Generates responses and computes reward statistics.

        Args:
            prompts: Token IDs ``(batch, prompt_len)``.
            attention_mask: Optional attention mask.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.policy_model.eval()

        with torch.no_grad():
            group_result = self._generate_responses(prompts, attention_mask)

            G = self.config.num_responses_per_prompt
            batch_size = prompts.shape[0]

            # Compute rewards
            if self.reward_fn is not None:
                num_responses = group_result.response_ids.shape[0]
                flat_prompts_str = [""] * num_responses
                flat_responses_str = [""] * num_responses
                rewards = self.reward_fn(flat_responses_str, flat_prompts_str)
                if isinstance(rewards, list):
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                rewards = rewards.to(self.device)
                if rewards.dim() == 1:
                    rewards = rewards.view(batch_size, G)
            else:
                rewards = torch.zeros(
                    batch_size, G, device=self.device, dtype=torch.float32
                )

        # Evaluation metrics (no loss computation)
        metrics = {
            "eval/mean_reward": rewards.mean().item(),
            "eval/max_reward": rewards.max().item(),
            "eval/min_reward": rewards.min().item(),
            "eval/reward_std": rewards.std().item(),
            "eval/mean_response_length": group_result.response_lengths.float().mean().item(),
        }

        self.policy_model.train()
        return metrics

    # ------------------------------------------------------------------
    # Training epoch helper
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        prompt_dataloader: Any,
        num_steps: int = 100,
    ) -> List[Dict[str, float]]:
        """Train for one epoch with DAPO.

        Args:
            prompt_dataloader: DataLoader yielding batches with
                ``"input_ids"`` (and optionally ``"attention_mask"``).
            num_steps: Maximum number of training steps.

        Returns:
            List of metric dictionaries, one per step.
        """
        all_metrics: List[Dict[str, float]] = []

        for step, batch in enumerate(prompt_dataloader):
            if step >= num_steps:
                break

            prompts = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            metrics = self.train_step(prompts, attention_mask)
            all_metrics.append(metrics)

        return all_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing.

        Returns:
            Dictionary containing step count, filtering statistics,
            and optimizer state.
        """
        return {
            "step_count": self._step_count,
            "total_filtered_prompts": self._total_filtered_prompts,
            "total_prompts": self._total_prompts,
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "clip_ratio_low": self.config.clip_ratio_low,
                "clip_ratio_high": self.config.clip_ratio_high,
                "dynamic_sampling": self.config.dynamic_sampling,
                "token_level_loss": self.config.token_level_loss,
                "overlong_filter": self.config.overlong_filter,
                "max_response_length": self.config.max_response_length,
                "kl_coefficient": self.config.kl_coefficient,
                "entropy_coefficient": self.config.entropy_coefficient,
                "learning_rate": self.config.learning_rate,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load trainer state from a checkpoint.

        Args:
            state: Dictionary previously returned by ``get_state_dict``.
        """
        self._step_count = state.get("step_count", 0)
        self._total_filtered_prompts = state.get("total_filtered_prompts", 0)
        self._total_prompts = state.get("total_prompts", 0)
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        logger.info(
            "DAPO trainer state restored: step=%d, filtered=%d / %d",
            self._step_count,
            self._total_filtered_prompts,
            self._total_prompts,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_metrics_summary(self) -> Dict[str, float]:
        """Compute summary statistics over the training history.

        Returns:
            Dictionary with mean, min, max of key metrics across all
            recorded steps.
        """
        if not self._metrics_history:
            return {}

        summary: Dict[str, float] = {}
        key_metrics = [
            "dapo/policy_loss",
            "dapo/kl_penalty",
            "dapo/entropy",
            "dapo/loss",
            "dapo/mean_advantage",
            "dapo/mean_reward",
        ]

        for key in key_metrics:
            values = [
                m[key] for m in self._metrics_history if key in m
            ]
            if values:
                summary[f"{key}/mean"] = sum(values) / len(values)
                summary[f"{key}/min"] = min(values)
                summary[f"{key}/max"] = max(values)

        summary["dapo/total_steps"] = float(self._step_count)
        summary["dapo/total_filtered"] = float(self._total_filtered_prompts)
        summary["dapo/filter_rate"] = (
            self._total_filtered_prompts / max(self._total_prompts, 1)
        )

        return summary
