"""
GRPOTrainer — Group Relative Policy Optimization
==================================================

RL framework dari DeepSeek-R1 yang TIDAK memerlukan value function
(berbeda dari PPO). Lebih efisien dan stabil untuk reasoning training.

Algoritma:
1. Sample group of responses untuk setiap prompt
2. Score setiap response menggunakan reward function
3. Compute relative advantage dalam group
4. Update policy menggunakan clipped objective

Digunakan di Fase 3 untuk mengoptimalkan routing policy.

Keunggulan GRPO vs PPO:
- Tidak perlu value function → hemat ~50% parameter dan memori
- Relative advantage lebih stabil → tidak perlu baseline
- Group sampling → estimasi advantage yang lebih akurat
- Clipping → mencegah update yang terlalu besar

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losion.config import LosionConfig
from losion.models.losion_decoder import LosionForCausalLM

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class GRPOConfig:
    """Konfigurasi GRPO.

    Attributes:
        group_size: Jumlah response yang di-sample per prompt
        clip_range: Range clipping untuk policy update (ε)
        kl_coeff: Koefisien penalty KL divergence
        entropy_coeff: Koefisien bonus entropy (untuk eksplorasi)
        max_new_tokens: Maksimum token yang digenerate per response
        temperature: Sampling temperature untuk generasi
        gamma: Discount factor (biasanya 1.0 untuk language)
        use_advantage_normalization: Normalisasi advantage dalam group
        reward_shaping: Bentuk reward: "raw", "centered", "rank_based"
        policy_loss_type: Tipe policy loss: "clipped", "surrogate", "unclipped"
    """

    group_size: int = 8
    clip_range: float = 0.2
    kl_coeff: float = 0.05
    entropy_coeff: float = 0.01
    max_new_tokens: int = 512
    temperature: float = 0.7
    gamma: float = 1.0
    use_advantage_normalization: bool = True
    reward_shaping: str = "centered"
    policy_loss_type: str = "clipped"


@dataclass
class GRPOGroupResult:
    """Hasil dari satu group GRPO.

    Attributes:
        prompt_ids: Token IDs prompt [1, prompt_len]
        response_ids: Token IDs responses [group_size, response_len]
        log_probs: Log probabilities [group_size, response_len]
        rewards: Reward per response [group_size]
        advantages: Computed advantages [group_size]
        old_log_probs: Old log probs (untuk importance sampling) [group_size, response_len]
    """

    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor


# ============================================================================
# Reward Functions
# ============================================================================


class RewardFunction:
    """
    Fungsi reward default untuk GRPO.

    Memberikan reward berdasarkan:
    - Correctness: Apakah jawaban benar
    - Format: Apakah format sesuai
    - Reasoning quality: Kualitas reasoning (panjang, koheren)
    - Router efficiency: Efisiensi penggunaan jalur
    """

    def __call__(
        self,
        responses: List[str],
        prompts: Optional[List[str]] = None,
        reference_answers: Optional[List[str]] = None,
        routing_info: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
        """
        Hitung reward untuk setiap response.

        Args:
            responses: List response string
            prompts: List prompt string (opsional)
            reference_answers: List jawaban referensi (opsional)
            routing_info: Routing info dari model (opsional)

        Returns:
            Reward tensor [group_size]
        """
        rewards = []

        for i, response in enumerate(responses):
            reward = 0.0

            # Format reward (apakah response tidak kosong)
            if len(response.strip()) > 0:
                reward += 0.1

            # Length reward (moderate length preferred)
            length = len(response.split())
            if 10 <= length <= 500:
                reward += 0.2
            elif length > 0:
                reward += 0.05

            # Reasoning quality (sederhana: berdasarkan keberadaan
            # penanda reasoning)
            reasoning_markers = ["because", "therefore", "since", "thus", "so"]
            for marker in reasoning_markers:
                if marker in response.lower():
                    reward += 0.05
                    break

            # Correctness reward (jika ada reference)
            if reference_answers is not None and i < len(reference_answers):
                ref = reference_answers[i].lower().strip()
                resp = response.lower().strip()
                if ref in resp or resp in ref:
                    reward += 1.0

            # Router efficiency reward (jika ada routing info)
            if routing_info is not None and i < len(routing_info):
                info = routing_info[i]
                if "mean_weights" in info:
                    # Reward distribusi yang merata (entropy tinggi)
                    weights = list(info["mean_weights"].values())
                    entropy = -sum(w * math.log(max(w, 1e-8)) for w in weights if w > 0)
                    max_entropy = math.log(3)  # 3 jalur
                    normalized_entropy = entropy / max_entropy
                    reward += 0.1 * normalized_entropy

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)


# ============================================================================
# GRPOTrainer
# ============================================================================


class GRPOTrainer:
    """
    GRPO — Group Relative Policy Optimization (dari DeepSeek-R1).

    RL framework yang TIDAK memerlukan value function (berbeda dari PPO).
    Lebih efisien dan stabil untuk reasoning training.

    Algoritma:
    1. Sample group of responses untuk setiap prompt
    2. Score setiap response menggunakan reward function
    3. Compute relative advantage dalam group
    4. Update policy menggunakan clipped objective

    Digunakan di Fase 3 untuk mengoptimalkan routing policy.

    Args:
        model: LosionForCausalLM yang akan dioptimasi
        config: GRPOConfig
        reward_fn: Fungsi reward (default: RewardFunction)
    """

    def __init__(
        self,
        model: LosionForCausalLM,
        config: Optional[GRPOConfig] = None,
        reward_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.config = config or GRPOConfig()
        self.reward_fn = reward_fn or RewardFunction()

        # ---- Reference model (untuk KL penalty) ----
        # Simpan salinan parameter sebagai referensi
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # ---- Optimizer ----
        # Hanya optimasi parameter yang requires_grad
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-5,  # LR kecil untuk RL
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

        # ---- Device ----
        self.device = next(model.parameters()).device

    def train_step(
        self,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Satu langkah training GRPO.

        Alur:
        1. Generate group of responses
        2. Hitung reward untuk setiap response
        3. Compute relative advantage
        4. Update policy

        Args:
            prompts: Token IDs prompt [batch, prompt_len]
            attention_mask: Mask attention opsional

        Returns:
            Dictionary berisi metrics (loss, reward, kl, entropy)
        """
        self.model.train()
        group_size = self.config.group_size

        # ---- Step 1: Generate group of responses ----
        group_result = self._generate_group(
            prompts, attention_mask, group_size
        )

        # ---- Step 2: Hitung reward ----
        # Decode responses untuk reward computation
        rewards = group_result.rewards  # [group_size]

        # Reward shaping
        rewards = self._shape_rewards(rewards)

        # ---- Step 3: Compute relative advantage ----
        advantages = self._compute_advantages(rewards)
        group_result.advantages = advantages

        # ---- Step 4: Update policy ----
        metrics = self._update_policy(group_result)

        return metrics

    def _generate_group(
        self,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        group_size: int,
    ) -> GRPOGroupResult:
        """
        Generate group of responses untuk setiap prompt.

        Args:
            prompts: Token IDs prompt [batch, prompt_len]
            attention_mask: Mask attention
            group_size: Jumlah response per prompt

        Returns:
            GRPOGroupResult dengan responses, log_probs, dan rewards
        """
        batch_size, prompt_len = prompts.shape
        device = prompts.device

        all_response_ids: List[torch.Tensor] = []
        all_log_probs: List[torch.Tensor] = []
        all_rewards: List[float] = []

        for g in range(group_size):
            # Generate response
            with torch.no_grad():
                generated = self.model.generate(
                    prompt=prompts,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    thinking_mode=True,  # Aktifkan thinking untuk reasoning
                )

            # Ambil hanya token yang digenerate (tanpa prompt)
            response_ids = generated[:, prompt_len:]  # [batch, gen_len]

            # Hitung log probabilities (dengan gradient)
            full_ids = torch.cat([prompts, response_ids], dim=1)
            output = self.model(input_ids=full_ids, attention_mask=attention_mask)
            logits = output.logits[:, prompt_len - 1:, :]  # [batch, gen_len, vocab]

            # Log probs dari generated tokens
            log_probs = self._compute_log_probs(logits, response_ids)

            all_response_ids.append(response_ids)
            all_log_probs.append(log_probs)

        # Stack: [group_size, batch, gen_len] → flatten ke [group_size * batch]
        stacked_response_ids = torch.stack(all_response_ids, dim=0)  # [group, batch, gen_len]
        stacked_log_probs = torch.stack(all_log_probs, dim=0)  # [group, batch, gen_len]

        # Decode responses dan hitung reward sebenarnya
        if self.reward_fn is not None:
            # Decode each response to text for reward computation
            generated_texts = []
            prompt_texts = []
            for g in range(group_size):
                resp_ids = all_response_ids[g]
                # Simple decode: convert token ids to string representation
                generated_texts.append(str(resp_ids.tolist()))
                prompt_texts.append(str(prompts.tolist()))
            rewards = self.reward_fn(responses=generated_texts, prompts=prompt_texts)
            if isinstance(rewards, list):
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
            else:
                rewards = rewards.to(device)
        else:
            rewards = torch.zeros(group_size, device=device)

        return GRPOGroupResult(
            prompt_ids=prompts,
            response_ids=stacked_response_ids[:, 0, :] if batch_size == 1 else stacked_response_ids.reshape(group_size, -1),
            log_probs=stacked_log_probs[:, 0, :] if batch_size == 1 else stacked_log_probs.reshape(group_size, -1),
            rewards=rewards,
            advantages=torch.zeros_like(rewards),  # Akan di-compute nanti
            old_log_probs=stacked_log_probs[:, 0, :].detach() if batch_size == 1 else stacked_log_probs.reshape(group_size, -1).detach(),
        )

    def _compute_log_probs(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hitung log probability dari token yang dipilih.

        Args:
            logits: Logits [batch, seq, vocab_size]
            targets: Target token IDs [batch, seq]

        Returns:
            Log probabilities [batch, seq]
        """
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log probs untuk token yang dipilih
        selected_log_probs = log_probs.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        return selected_log_probs

    def _shape_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Shape rewards berdasarkan konfigurasi.

        Args:
            rewards: Raw rewards [group_size]

        Returns:
            Shaped rewards [group_size]
        """
        if self.config.reward_shaping == "raw":
            return rewards

        elif self.config.reward_shaping == "centered":
            # Center rewards: subtract mean
            return rewards - rewards.mean()

        elif self.config.reward_shaping == "rank_based":
            # Rank-based: convert rewards to ranks
            sorted_indices = rewards.argsort()
            ranks = torch.zeros_like(rewards, dtype=torch.float32)
            ranks[sorted_indices] = torch.arange(
                len(rewards), dtype=torch.float32, device=rewards.device
            ) / max(len(rewards) - 1, 1)
            # Scale ke [-1, 1]
            return 2 * ranks - 1

        return rewards

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute relative advantages dalam group.

        Keunggulan GRPO: advantage relatif dalam group,
        tidak perlu value function.

        Args:
            rewards: Shaped rewards [group_size]

        Returns:
            Advantages [group_size]
        """
        # Relative advantage: (r - mean) / std
        mean_reward = rewards.mean()
        std_reward = rewards.std()

        if std_reward < 1e-8:
            # Jika semua reward sama, advantage = 0
            return torch.zeros_like(rewards)

        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        # Optional: normalisasi ke [-1, 1]
        if self.config.use_advantage_normalization:
            max_abs = advantages.abs().max()
            if max_abs > 1e-8:
                advantages = advantages / max_abs

        return advantages

    def _update_policy(self, group_result: GRPOGroupResult) -> Dict[str, float]:
        """
        Update policy menggunakan GRPO clipped objective.

        L_clip = -min(r * A, clip(r, 1-ε, 1+ε) * A)
        dimana r = π_new / π_old (probability ratio)
        dan A = advantage

        Args:
            group_result: Hasil group GRPO

        Returns:
            Dictionary berisi metrics
        """
        self.optimizer.zero_grad()

        advantages = group_result.advantages  # [group_size]
        old_log_probs = group_result.old_log_probs  # [group_size, seq_len]

        # Re-run forward pass with current model to get new log probs
        # (This is essential: the stored log_probs are from generation time,
        #  we need fresh log_probs from the current model parameters so that
        #  the ratio r = exp(new - old) is not always 1.0)
        prompt_ids = group_result.prompt_ids  # [batch, prompt_len]
        response_ids = group_result.response_ids  # [group_size, seq_len]
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        prompt_len = prompt_ids.shape[-1]
        with torch.enable_grad():
            new_outputs = self.model(input_ids=full_ids)
            # Extract per-token log probs from the new forward pass
            new_all_logits = new_outputs.logits  # (batch, seq_len, vocab_size)
            # Compute log probs for the actions taken
            new_log_probs = F.log_softmax(new_all_logits[:, prompt_len - 1:, :], dim=-1).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

        # Probability ratio: r = exp(new_log_prob - old_log_prob)
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)  # [group_size, seq_len]

        # Untuk per-sequence ratio, average over sequence
        if ratio.dim() > 1:
            # Mean ratio per sequence
            ratio_per_seq = ratio.mean(dim=-1)  # [group_size]
        else:
            ratio_per_seq = ratio

        # Expand advantages untuk broadcasting
        if advantages.dim() == 1:
            advantages_expanded = advantages.unsqueeze(-1)
        else:
            advantages_expanded = advantages

        # ---- Policy loss (clipped) ----
        if self.config.policy_loss_type == "clipped":
            # L_CLIP = -min(r * A, clip(r, 1-ε, 1+ε) * A)
            clip_low = 1.0 - self.config.clip_range
            clip_high = 1.0 + self.config.clip_range

            clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

            if ratio.dim() > 1:
                advantages_for_clipped = advantages.unsqueeze(-1).expand_as(ratio)
                surr1 = ratio * advantages_for_clipped
                surr2 = clipped_ratio * advantages_for_clipped
            else:
                surr1 = ratio * advantages
                surr2 = clipped_ratio * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

        elif self.config.policy_loss_type == "surrogate":
            # Simple surrogate loss
            policy_loss = -(ratio_per_seq * advantages).mean()

        else:
            # Unclipped
            policy_loss = -(ratio_per_seq * advantages).mean()

        # ---- KL penalty ----
        kl_penalty = self._compute_kl_penalty(new_log_probs, old_log_probs)

        # ---- Entropy bonus ----
        entropy = self._compute_entropy(new_log_probs)

        # ---- Total loss ----
        total_loss = (
            policy_loss
            + self.config.kl_coeff * kl_penalty
            - self.config.entropy_coeff * entropy
        )

        # ---- Backward ----
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )

        self.optimizer.step()

        # ---- Metrics ----
        with torch.no_grad():
            metrics = {
                "grpo_loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "kl_penalty": kl_penalty.item(),
                "entropy": entropy.item(),
                "mean_reward": group_result.rewards.mean().item(),
                "mean_advantage": advantages.mean().item(),
                "max_advantage": advantages.max().item(),
                "min_advantage": advantages.min().item(),
            }

        return metrics

    def _compute_kl_penalty(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hitung KL divergence penalty.

        KL(π_old || π_new) ≈ E[log(π_old / π_new)]

        Args:
            new_log_probs: Log probs dari policy baru
            old_log_probs: Log probs dari policy lama (reference)

        Returns:
            Scalar KL penalty
        """
        # Approximate KL
        kl = old_log_probs - new_log_probs
        return kl.mean()

    def _compute_entropy(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Hitung entropy dari distribusi policy.

        H(π) = -E[log(π)]

        Args:
            log_probs: Log probabilities

        Returns:
            Scalar entropy
        """
        return -(log_probs.exp() * log_probs).mean()

    def train_epoch(
        self,
        prompt_dataloader: Any,
        num_steps: int = 100,
    ) -> List[Dict[str, float]]:
        """
        Train satu epoch dengan GRPO.

        Args:
            prompt_dataloader: DataLoader yang menghasilkan prompts
            num_steps: Jumlah langkah training

        Returns:
            List metrics per langkah
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

            if step % 10 == 0:
                log_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in metrics.items()
                )
                logger.info(f"GRPO Step {step}: {log_str}")

        return all_metrics
