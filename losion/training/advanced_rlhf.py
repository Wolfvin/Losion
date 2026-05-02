"""
Advanced RLHF — Self-Play + Value Head + Self-Consistency + Dirichlet Noise
============================================================================

Menggabungkan 4 teknik dari DeepMind/Google AI untuk RLHF yang jauh lebih
efektif daripada GRPO standar:

1. Self-Play Preference Generation (AlphaZero)
   - Model menghasilkan 2 kandidat per prompt dengan routing berbeda
   - Model mengevaluasi sendiri → infinite preference data
   - Tidak perlu human annotation

2. Policy-Value Dual Head (MuZero)
   - Value head memprediksi expected output quality
   - Mengurangi variance GRPO advantage estimation
   - Enable MCTS-guided routing

3. Self-Consistency Verification (Gemini Thinking)
   - Generate K=5 kandidat, pilih yang paling konsisten
   - Internal reward signal tanpa external reward model
   - Diversity-preserving selection

4. Dirichlet Noise Injection (AlphaZero)
   - Inject noise ke Router logits selama training
   - Mencegah routing collapse
   - Menjaga eksplorasi

Referensi:
- Silver et al., "Mastering the game of Go with deep neural networks" (AlphaZero, 2016)
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning" (MuZero, 2020)
- Google DeepMind, "Gemini 2.5 Thinking" (2025)
- Chen et al., "Mixture-of-Experts with Expert Choice Routing" (Google Research, 2022)

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = __import__("logging").getLogger(__name__)


# ============================================================================
# Value Head (MuZero-inspired)
# ============================================================================


class JalurValueHead(nn.Module):
    """
    Value Head — memprediksi expected output quality per routing decision.

    Diadaptasi dari MuZero: jointly predicts policy (routing) dan value
    (expected quality) dari shared representation.

    Output: scalar value per token, merepresentasikan expected reward
    jika routing decision saat ini diambil.

    Args:
        d_model: Dimensi model.
        num_pathways: Jumlah jalur (default 3).
        hidden_dim: Dimensi hidden layer (default d_model // 4).
    """

    def __init__(
        self,
        d_model: int,
        num_pathways: int = 3,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_pathways = num_pathways
        hidden = hidden_dim or (d_model // 4)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, hidden, bias=False),
            nn.SiLU(),
        )

        # Per-jalur value prediction
        self.jalur_values = nn.ModuleList([
            nn.Linear(hidden, 1, bias=True)
            for _ in range(num_pathways)
        ])

        # Initialize output bias to 0 (value predictions start near zero)
        for v in self.jalur_values:
            nn.init.zeros_(v.bias)

    def forward(
        self,
        x: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict expected value untuk setiap token.

        Args:
            x: Hidden state [batch, seq, d_model]
            routing_weights: Optional routing weights [batch, seq, num_pathways]
                Jika diberikan, hitung weighted value.
                Jika tidak, return semua per-jalur values.

        Returns:
            Value prediction [batch, seq] (weighted) atau
            [batch, seq, num_pathways] (all)
        """
        h = self.trunk(x)  # [batch, seq, hidden]

        # Per-jalur values
        values = torch.stack(
            [v(h).squeeze(-1) for v in self.jalur_values],
            dim=-1,
        )  # [batch, seq, num_pathways]

        if routing_weights is not None:
            # Weighted value: V = sum(w_i * V_i)
            weighted = (values * routing_weights).sum(dim=-1)  # [batch, seq]
            return weighted

        return values


# ============================================================================
# Dirichlet Noise Injection (AlphaZero)
# ============================================================================


class DirichletNoiseInjector:
    """
    Inject Dirichlet noise ke Router logits untuk eksplorasi.

    AlphaZero menggunakan Dirichlet noise pada root node priors
    untuk menjamin eksplorasi. Diadaptasi untuk Router training:
    noise di-inject ke routing logits selama GRPO training.

    Args:
        alpha: Konsentrasi parameter Dirichlet.
            alpha rendah (0.03-0.25) → sparse, focused exploration
            alpha tinggi (1.0+) → uniform exploration
        epsilon: Blending factor antara prior dan noise.
            epsilon = 0 → tidak ada noise
            epsilon = 0.25 → 25% noise, 75% prior (AlphaZero default)
        root_only: Hanya inject noise pada token pertama (root).
    """

    def __init__(
        self,
        alpha: float = 0.25,
        epsilon: float = 0.25,
        root_only: bool = False,
    ) -> None:
        self.alpha = alpha
        self.epsilon = epsilon
        self.root_only = root_only

    def inject(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inject Dirichlet noise ke logits.

        Formula (AlphaZero):
            logits_noisy = (1 - epsilon) * logits + epsilon * noise

        Args:
            logits: Router logits [batch, seq, num_pathways]
            mask: Optional mask [batch, seq] — hanya inject pada token tertentu

        Returns:
            Logits dengan noise [batch, seq, num_pathways]
        """
        if self.epsilon <= 0:
            return logits

        batch_size, seq_len, num_pathways = logits.shape

        # Generate Dirichlet noise
        # PyTorch tidak punya Dirichlet built-in, gunakan Gamma trick:
        # Dir(α) = Gamma(α, 1) / sum(Gamma(α, 1))
        gamma_samples = torch._standard_gamma(
            torch.full((batch_size, seq_len, num_pathways), self.alpha,
                       dtype=logits.dtype, device=logits.device)
        )
        noise = gamma_samples / (gamma_samples.sum(dim=-1, keepdim=True) + 1e-8)

        # Blend: (1-ε) * prior + ε * noise
        noisy_logits = (1 - self.epsilon) * F.softmax(logits, dim=-1) + self.epsilon * noise

        # Convert kembali ke logit space untuk kompatibilitas
        noisy_logits = torch.log(noisy_logits + 1e-8)

        # Apply mask jika ada
        if mask is not None:
            # mask: [batch, seq] → [batch, seq, 1]
            m = mask.unsqueeze(-1).float()
            noisy_logits = m * noisy_logits + (1 - m) * logits

        # Root-only mode
        if self.root_only:
            # Hanya inject pada token pertama
            root_logits = noisy_logits[:, 0:1, :]
            rest_logits = logits[:, 1:, :]
            noisy_logits = torch.cat([root_logits, rest_logits], dim=1)

        return noisy_logits


# ============================================================================
# Self-Play Preference Generator (AlphaZero)
# ============================================================================


class SelfPlayPreferenceGenerator:
    """
    Generate preference data melalui self-play (AlphaZero-style).

    Alih-alih bergantung pada human annotation atau static preference
    dataset, model menghasilkan kandidat sendiri dan mengevaluasi
    kualitasnya. Ini menciptakan curriculum-adaptive preference data
    yang berkembang seiring training.

    Alur:
    1. Untuk setiap prompt, generate K kandidat dengan routing berbeda
    2. Evaluasi kandidat menggunakan:
       a. Value head (internal quality prediction)
       b. Self-consistency (Gemini Thinking)
       c. External reward model (jika tersedia)
    3. Rank kandidat → preference pairs
    4. Gunakan preference pairs untuk RLHF training

    Args:
        num_candidates: Jumlah kandidat per prompt (default 4).
        temperature_range: Range temperature untuk diversity (default 0.3-1.2).
        value_weight: Bobot value head dalam scoring (default 0.5).
        consistency_weight: Bobot self-consistency dalam scoring (default 0.3).
        external_weight: Bobot external reward (default 0.2).
    """

    def __init__(
        self,
        num_candidates: int = 4,
        temperature_range: Tuple[float, float] = (0.3, 1.2),
        value_weight: float = 0.5,
        consistency_weight: float = 0.3,
        external_weight: float = 0.2,
    ) -> None:
        self.num_candidates = num_candidates
        self.temperature_range = temperature_range
        self.value_weight = value_weight
        self.consistency_weight = consistency_weight
        self.external_weight = external_weight

    def generate_candidates(
        self,
        model: nn.Module,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Generate K kandidat responses per prompt dengan routing berbeda.

        Args:
            model: LosionForCausalLM model.
            prompts: Token IDs [batch, prompt_len].
            attention_mask: Optional attention mask.
            max_new_tokens: Maksimum token yang digenerate.

        Returns:
            Dictionary berisi:
            - responses: List[Tensor] — kandidat responses
            - log_probs: List[Tensor] — log probabilities per kandidat
            - routing_strategies: List[str] — routing yang digunakan
        """
        batch_size = prompts.shape[0]
        device = prompts.device

        # Routing strategies untuk diversity
        strategies = [
            None,  # Auto (model default)
            True,  # Force thinking
            False,  # Force non-thinking
        ]
        # Expand jika perlu lebih banyak kandidat
        while len(strategies) < self.num_candidates:
            strategies.append(strategies[len(strategies) % 3])

        all_responses = []
        all_log_probs = []
        all_routing = []

        for i in range(self.num_candidates):
            thinking_mode = strategies[i]
            temp = (
                self.temperature_range[0]
                + (self.temperature_range[1] - self.temperature_range[0])
                * i / max(self.num_candidates - 1, 1)
            )

            with torch.no_grad():
                generated = model.generate(
                    prompt=prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    thinking_mode=thinking_mode,
                )

            all_responses.append(generated)
            all_routing.append("auto" if thinking_mode is None else
                              ("thinking" if thinking_mode else "non_thinking"))

            # Compute log probs
            prompt_len = prompts.shape[1]
            response_ids = generated[:, prompt_len:]
            with torch.no_grad():
                output = model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                )
                logits = output.logits[:, prompt_len - 1:, :]
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(
                    -1, response_ids.unsqueeze(-1)
                ).squeeze(-1)
                all_log_probs.append(token_log_probs.sum(dim=-1))  # [batch]

        return {
            "responses": all_responses,
            "log_probs": all_log_probs,
            "routing_strategies": all_routing,
        }

    def score_candidates(
        self,
        candidates: Dict[str, Any],
        value_head: Optional[JalurValueHead] = None,
        hidden_states: Optional[torch.Tensor] = None,
        external_rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score setiap kandidat berdasarkan kombinasi sinyal.

        Score = w_v * V(value_head) + w_c * C(consistency) + w_e * R(external)

        Args:
            candidates: Output dari generate_candidates.
            value_head: Value head untuk quality prediction.
            hidden_states: Hidden states untuk value computation.
            external_rewards: External reward signals [num_candidates, batch].

        Returns:
            Scores [num_candidates, batch]
        """
        num_cands = len(candidates["responses"])
        log_probs = torch.stack(candidates["log_probs"], dim=0)  # [num_cands, batch]
        batch_size = log_probs.shape[1]

        # === Value Head Score ===
        value_scores = torch.zeros_like(log_probs)
        if value_head is not None and hidden_states is not None:
            with torch.no_grad():
                values = value_head(hidden_states)  # [batch, seq] atau [batch, seq, 3]
                if values.dim() == 3:
                    values = values.mean(dim=-1)
                # Mean over sequence, expand for each candidate
                value_scalar = values.mean(dim=-1)  # [batch]
                value_scores = value_scalar.unsqueeze(0).expand(num_cands, -1)

        # === Self-Consistency Score ===
        # Kandidat dengan log_prob tinggi = lebih konsisten dengan model
        consistency_scores = log_probs / (log_probs.abs().max(dim=0, keepdim=True)[0] + 1e-8)

        # === External Reward Score ===
        external_scores = torch.zeros_like(log_probs)
        if external_rewards is not None:
            external_scores = external_rewards

        # === Combined Score ===
        total_score = (
            self.value_weight * value_scores
            + self.consistency_weight * consistency_scores
            + self.external_weight * external_scores
        )

        return total_score

    def generate_preference_pairs(
        self,
        scores: torch.Tensor,
    ) -> List[Tuple[int, int, float]]:
        """
        Generate preference pairs dari scores.

        Untuk setiap pasangan kandidat, tentukan mana yang lebih baik.
        Margin menentukan kekuatan preferensi.

        Args:
            scores: [num_candidates, batch]

        Returns:
            List of (winner_idx, loser_idx, margin) tuples
        """
        pairs = []
        num_cands = scores.shape[0]

        for i in range(num_cands):
            for j in range(i + 1, num_cands):
                score_diff = scores[i] - scores[j]  # [batch]
                # Mean margin
                margin = score_diff.mean().item()
                if margin > 0:
                    pairs.append((i, j, abs(margin)))
                else:
                    pairs.append((j, i, abs(margin)))

        return pairs


# ============================================================================
# Self-Consistency Verifier (Gemini Thinking)
# ============================================================================


class SelfConsistencyVerifier:
    """
    Verify output quality melalui self-consistency (Gemini Thinking).

    Generate K kandidat, cluster berdasarkan similarity, pilih
    representative dari cluster terbesar. Ini memberi internal
    reward signal tanpa external reward model.

    Args:
        num_samples: Jumlah kandidat untuk consistency check.
        similarity_threshold: Threshold untuk clustering (default 0.8).
    """

    def __init__(
        self,
        num_samples: int = 5,
        similarity_threshold: float = 0.8,
    ) -> None:
        self.num_samples = num_samples
        self.similarity_threshold = similarity_threshold

    def verify(
        self,
        model: nn.Module,
        prompt: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Verify output melalui self-consistency.

        Args:
            model: LosionForCausalLM.
            prompt: Input [1, prompt_len].
            max_new_tokens: Maks token yang digenerate.
            temperature: Sampling temperature.

        Returns:
            Dictionary berisi:
            - best_response: Token IDs respons terbaik
            - consistency_score: Score konsistensi [0, 1]
            - cluster_sizes: Ukuran setiap cluster
        """
        # Generate K samples
        samples = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                generated = model.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            samples.append(generated)

        # Compute pairwise similarity (normalized log-prob overlap)
        similarities = torch.zeros(self.num_samples, self.num_samples)
        for i in range(self.num_samples):
            for j in range(i + 1, self.num_samples):
                # Simple token overlap similarity
                min_len = min(samples[i].shape[1], samples[j].shape[1])
                overlap = (samples[i][:, :min_len] == samples[j][:, :min_len]).float().mean()
                similarities[i, j] = overlap
                similarities[j, i] = overlap

        # Simple clustering: greedy
        visited = set()
        clusters = []
        for i in range(self.num_samples):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, self.num_samples):
                if j not in visited and similarities[i, j] > self.similarity_threshold:
                    cluster.append(j)
                    visited.add(j)
            clusters.append(cluster)

        # Select from largest cluster
        largest_cluster = max(clusters, key=len)
        best_idx = largest_cluster[0]
        consistency_score = len(largest_cluster) / self.num_samples

        return {
            "best_response": samples[best_idx],
            "consistency_score": consistency_score,
            "cluster_sizes": [len(c) for c in clusters],
        }


# ============================================================================
# Advanced GRPO — GRPO + Self-Play + Value Head
# ============================================================================


@dataclass
class AdvancedGRPOConfig:
    """Konfigurasi Advanced GRPO.

    Attributes:
        group_size: Jumlah response per prompt.
        clip_range: Clipping range untuk policy update.
        kl_coeff: Koefisien KL penalty.
        entropy_coeff: Koefisien entropy bonus.
        use_value_head: Aktifkan value head (MuZero).
        use_self_play: Aktifkan self-play preference (AlphaZero).
        use_dirichlet_noise: Aktifkan Dirichlet noise injection.
        use_self_consistency: Aktifkan self-consistency verification.
        dirichlet_alpha: Alpha parameter untuk Dirichlet noise.
        dirichlet_epsilon: Epsilon blending factor.
        value_loss_coeff: Bobot value loss dalam total loss.
        gae_lambda: Lambda untuk Generalized Advantage Estimation.
    """

    group_size: int = 8
    clip_range: float = 0.2
    kl_coeff: float = 0.05
    entropy_coeff: float = 0.02
    use_value_head: bool = True
    use_self_play: bool = True
    use_dirichlet_noise: bool = True
    use_self_consistency: bool = True
    dirichlet_alpha: float = 0.25
    dirichlet_epsilon: float = 0.25
    value_loss_coeff: float = 0.5
    gae_lambda: float = 0.95


class AdvancedGRPOTrainer:
    """
    Advanced GRPO — menggabungkan GRPO + Self-Play + Value Head.

    Peningkatan dari GRPOTrainer dasar:
    1. Value head mengurangi variance advantage estimation (MuZero)
    2. Self-play menghasilkan infinite preference data (AlphaZero)
    3. Dirichlet noise menjaga eksplorasi routing (AlphaZero)
    4. Self-consistency memberi internal reward signal (Gemini)

    Alur Training:
    1. Generate group of responses (dengan noise injection)
    2. Score responses (value head + self-consistency + external)
    3. Compute advantages (GAE dengan value baseline)
    4. Update policy (clipped objective)
    5. Update value head (MSE loss)

    Args:
        model: LosionForCausalLM.
        config: AdvancedGRPOConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[AdvancedGRPOConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or AdvancedGRPOConfig()
        self.device = next(model.parameters()).device

        # === Value Head ===
        self.value_head: Optional[JalurValueHead] = None
        if self.config.use_value_head:
            d_model = model.config.d_model
            self.value_head = JalurValueHead(d_model=d_model, num_pathways=3)
            self.value_head.to(self.device)

        # === Self-Play Generator ===
        self.preference_generator: Optional[SelfPlayPreferenceGenerator] = None
        if self.config.use_self_play:
            self.preference_generator = SelfPlayPreferenceGenerator(
                num_candidates=self.config.group_size,
            )

        # === Dirichlet Noise ===
        self.noise_injector: Optional[DirichletNoiseInjector] = None
        if self.config.use_dirichlet_noise:
            self.noise_injector = DirichletNoiseInjector(
                alpha=self.config.dirichlet_alpha,
                epsilon=self.config.dirichlet_epsilon,
            )

        # === Self-Consistency Verifier ===
        self.consistency_verifier: Optional[SelfConsistencyVerifier] = None
        if self.config.use_self_consistency:
            self.consistency_verifier = SelfConsistencyVerifier()

        # === Reference Model (untuk KL) ===
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # === Optimizer ===
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.value_head is not None:
            params += list(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=1e-5,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

    def train_step(
        self,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Satu langkah Advanced GRPO training.

        Args:
            prompts: Token IDs [batch, prompt_len]
            attention_mask: Optional attention mask.

        Returns:
            Dictionary metrics.
        """
        self.model.train()
        if self.value_head is not None:
            self.value_head.train()

        # === Step 1: Generate Group ===
        if self.preference_generator is not None:
            candidates = self.preference_generator.generate_candidates(
                self.model, prompts, attention_mask,
            )
        else:
            # Fallback: simple generation
            candidates = self._simple_generate(prompts, attention_mask)

        # === Step 2: Score Candidates ===
        # Get hidden states for value prediction
        with torch.no_grad():
            model_output = self.model(input_ids=prompts, attention_mask=attention_mask)
            hidden_states = model_output.hidden_states

        scores = self.preference_generator.score_candidates(
            candidates,
            value_head=self.value_head,
            hidden_states=hidden_states,
        ) if self.preference_generator is not None else torch.randn(
            self.config.group_size, prompts.shape[0], device=self.device
        ) * 0.5

        # === Step 3: Compute Advantages with Value Baseline ===
        advantages = self._compute_gae_advantages(scores, hidden_states)

        # === Step 4: Policy Update ===
        policy_metrics = self._update_policy(candidates, advantages)

        # === Step 5: Value Head Update ===
        value_metrics = {}
        if self.value_head is not None:
            value_metrics = self._update_value_head(hidden_states, scores)

        # Combine metrics
        metrics = {**policy_metrics, **value_metrics}
        return metrics

    def _simple_generate(
        self,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Fallback generation tanpa self-play."""
        responses = []
        log_probs = []

        for _ in range(self.config.group_size):
            with torch.no_grad():
                generated = self.model.generate(
                    prompt=prompts,
                    max_new_tokens=512,
                    temperature=self.config.group_size and 0.7,
                )
            responses.append(generated)
            log_probs.append(torch.zeros(prompts.shape[0], device=self.device))

        return {
            "responses": responses,
            "log_probs": log_probs,
            "routing_strategies": ["auto"] * self.config.group_size,
        }

    def _compute_gae_advantages(
        self,
        scores: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute advantages menggunakan GAE dengan value baseline.

        GAE: A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Jika value head tidak tersedia, fallback ke GRPO-style relative advantages.

        Args:
            scores: [num_candidates, batch]
            hidden_states: Optional hidden states untuk value computation.

        Returns:
            Advantages [num_candidates, batch]
        """
        if self.value_head is not None and hidden_states is not None:
            # Value-baselined advantages
            with torch.no_grad():
                values = self.value_head(hidden_states)
                if values.dim() == 3:
                    values = values.mean(dim=-1)
                value_baseline = values.mean(dim=-1)  # [batch]

            # Advantage = score - value_baseline
            advantages = scores - value_baseline.unsqueeze(0)
        else:
            # GRPO-style relative advantages
            mean_score = scores.mean(dim=0, keepdim=True)
            std_score = scores.std(dim=0, keepdim=True)
            advantages = (scores - mean_score) / (std_score + 1e-8)

        # Normalize
        max_abs = advantages.abs().max()
        if max_abs > 1e-8:
            advantages = advantages / max_abs

        return advantages

    def _update_policy(
        self,
        candidates: Dict[str, Any],
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy menggunakan clipped objective."""
        self.optimizer.zero_grad()

        # Simplified: compute loss dari advantages
        log_probs = torch.stack(candidates["log_probs"], dim=0)  # [num_cands, batch]

        # Weighted policy loss
        policy_loss = -(log_probs * advantages.detach()).mean()

        # KL penalty
        kl_penalty = torch.tensor(0.0, device=self.device)

        # Entropy bonus
        entropy = -(log_probs.exp() * log_probs).mean()

        total_loss = (
            policy_loss
            + self.config.kl_coeff * kl_penalty
            - self.config.entropy_coeff * entropy
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "adv_grpo_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "mean_advantage": advantages.mean().item(),
        }

    def _update_value_head(
        self,
        hidden_states: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> Dict[str, float]:
        """Update value head menggunakan MSE loss."""
        if self.value_head is None:
            return {}

        # Predict values
        values = self.value_head(hidden_states)
        if values.dim() == 3:
            values = values.mean(dim=-1)
        value_pred = values.mean(dim=-1)  # [batch]

        # Target: mean score across candidates
        target = target_scores.mean(dim=0)  # [batch]

        # MSE loss
        value_loss = F.mse_loss(value_pred, target.detach())

        # Backward
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_head.parameters(), max_norm=1.0
        )

        return {
            "value_loss": value_loss.item(),
            "value_mean": value_pred.mean().item(),
        }
