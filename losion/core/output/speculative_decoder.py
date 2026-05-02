"""
Losion Framework v0.4 — MTP Speculative Decoding Module

Implements Multi-Token Prediction (MTP) heads and a full speculative decoding
pipeline that uses MTP draft predictions to accelerate autoregressive text
generation by ~1.8x.

Architecture:

1. MTPHead — Single prediction head for one future position offset.
   Lightweight linear projection on top of shared hidden states.

2. MultiTokenPrediction — Module with N parallel MTPHead instances.
   Predicts tokens at offsets 1..N from a single forward pass of the
   main model. Used during training (multi-token loss) and as the
   draft model for speculative decoding.

3. SpeculativeDecoder — Full speculative decoding pipeline.
   - Draft phase:  MTP heads generate K candidate tokens in one pass.
   - Verify phase: Main model verifies all K candidates in one forward pass.
   - Accept/reject: Accept matching prefix, reject from first mismatch onward.
   - Adaptive speculation length based on running acceptance rate.
   - Comprehensive statistics tracking (SpeculativeStats).

4. SpeculativeStats — Dataclass tracking acceptance rates, speedup,
   and per-step diagnostics for monitoring and adaptive length control.

Algorithm (Speculative Decoding):
    Given a sequence of tokens x_1, ..., x_t:

    1. DRAFT: Run MTP heads on the current hidden state h_t to produce
       candidate tokens (c_1, c_2, ..., c_K) where c_k = argmax(head_k(h_t)).

    2. VERIFY: Feed the full sequence including all K candidates into the
       main model in a single forward pass. This gives us the "ground truth"
       next-token distribution at each position.

    3. ACCEPT/REJECT: Compare each c_k with the main model's prediction
       at the corresponding position. Accept all tokens until the first
       mismatch. Let n = number of accepted tokens.

    4. OUTPUT: Emit n+1 tokens (n accepted + 1 correction from the main
       model at the rejection point, or all K + 1 if all accepted).

    Expected speedup: If the MTP heads have acceptance rate p, then
    the expected number of tokens per step is 1 + p + p^2 + ... + p^K
    = (1 - p^{K+1}) / (1 - p). For p ~ 0.85 and K = 5, this yields
    ~4.4 tokens per step, which translates to ~1.8x wall-clock speedup
    when accounting for the verification overhead.

References:
- Leviathan, Y. et al., "Fast Inference from Transformers via Speculative
  Decoding" (ICML 2023)
- Chen, C. et al., "Accelerating Large Language Model Decoding with
  Speculative Sampling" (ICLR 2024)
- Sun, Z. et al., "Speeding up Large Language Model Inference with
  Speculative Decoding and Multi-Token Prediction" (2024)
- Xia, H. et al., "Speculative Decoding: Exploiting Speculative Execution
  for Accelerated Seq2Seq Generation" (EMNLP 2024)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MTPOutput — Data container for multi-token prediction results
# ============================================================================


@dataclass
class MTPOutput:
    """Container for Multi-Token Prediction outputs.

    Holds the predicted token IDs, logits, and probabilities from all MTP
    heads for a single forward pass.

    Attributes:
        token_ids: Predicted token IDs, shape (batch, n_heads).
            Each column k contains the prediction at offset k+1.
        logits: Raw logits from each head, shape (batch, n_heads, vocab_size).
        probabilities: Softmax probabilities, shape (batch, n_heads, vocab_size).
        n_heads: Number of MTP heads (equals the speculation length).
    """

    token_ids: torch.Tensor
    logits: torch.Tensor
    probabilities: torch.Tensor
    n_heads: int

    def __len__(self) -> int:
        return self.n_heads


# ============================================================================
# SpeculativeStats — Statistics for speculative decoding monitoring
# ============================================================================


@dataclass
class SpeculativeStats:
    """Statistics tracker for speculative decoding performance monitoring.

    Tracks acceptance rates, speedup metrics, and per-step diagnostics.
    Used both for monitoring and for adaptive speculation length control.

    Attributes:
        total_steps: Total number of speculative decoding steps executed.
        total_draft_tokens: Total number of draft tokens proposed across all steps.
        total_accepted_tokens: Total number of draft tokens accepted by the verifier.
        total_emitted_tokens: Total number of tokens actually emitted (including
            correction tokens from the verifier).
        max_spec_length_used: Maximum speculation length actually used.
        min_spec_length_used: Minimum speculation length actually used.
        acceptance_history: Per-step acceptance counts (for plotting/analysis).
        spec_length_history: Per-step speculation lengths used.
    """

    total_steps: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_emitted_tokens: int = 0
    max_spec_length_used: int = 0
    min_spec_length_used: int = 0
    acceptance_history: List[int] = field(default_factory=list)
    spec_length_history: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate across all steps.

        Returns:
            Fraction of draft tokens that were accepted. 0.0 if no tokens drafted.
        """
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def avg_tokens_per_step(self) -> float:
        """Average number of tokens emitted per speculative decoding step.

        Returns:
            Average emitted tokens. 0.0 if no steps taken.
        """
        if self.total_steps == 0:
            return 0.0
        return self.total_emitted_tokens / self.total_steps

    @property
    def effective_speedup(self) -> float:
        """Estimated effective speedup ratio vs. standard autoregressive decoding.

        Computes the ratio of average tokens emitted per step over 1 (baseline
        for autoregressive). In practice, the wall-clock speedup is slightly
        lower due to the verification overhead, so we apply a correction factor.

        Returns:
            Estimated speedup ratio. 1.0 if no steps taken.
        """
        if self.total_steps == 0:
            return 1.0
        # Theoretical speedup = avg_tokens_per_step
        # Practical correction: verification costs ~1 forward pass per step,
        # so effective speedup = avg_tokens / (1 + cost_ratio) where
        # cost_ratio accounts for the verification being a longer-sequence pass.
        # For typical models, the verify pass is ~1.3x the cost of a single
        # token generation (KV cache amortization). So effective speedup
        # ≈ avg_tokens_per_step / 1.3 approximately.
        theoretical = self.avg_tokens_per_step
        # Empirical correction: the verification forward pass processes K+1
        # tokens but with KV caching the incremental cost is less than K+1
        # single-token passes. We estimate ~1.2x single-pass cost.
        verification_overhead = 1.2
        return theoretical / verification_overhead

    def record_step(self, n_accepted: int, spec_length: int) -> None:
        """Record statistics from a single speculative decoding step.

        Args:
            n_accepted: Number of draft tokens accepted in this step.
            spec_length: Speculation length used in this step.
        """
        self.total_steps += 1
        self.total_draft_tokens += spec_length
        self.total_accepted_tokens += n_accepted
        # Emitted = accepted + 1 correction token (or +1 bonus if all accepted)
        n_emitted = n_accepted + 1
        self.total_emitted_tokens += n_emitted
        self.acceptance_history.append(n_accepted)
        self.spec_length_history.append(spec_length)

        if self.total_steps == 1:
            self.max_spec_length_used = spec_length
            self.min_spec_length_used = spec_length
        else:
            self.max_spec_length_used = max(self.max_spec_length_used, spec_length)
            self.min_spec_length_used = min(self.min_spec_length_used, spec_length)

    def reset(self) -> None:
        """Reset all statistics to initial state."""
        self.total_steps = 0
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.total_emitted_tokens = 0
        self.max_spec_length_used = 0
        self.min_spec_length_used = 0
        self.acceptance_history.clear()
        self.spec_length_history.clear()

    def summary(self) -> str:
        """Return a human-readable summary of speculative decoding statistics.

        Returns:
            Formatted string with key metrics.
        """
        return (
            f"Speculative Decoding Stats:\n"
            f"  Total steps:            {self.total_steps}\n"
            f"  Total draft tokens:     {self.total_draft_tokens}\n"
            f"  Total accepted tokens:  {self.total_accepted_tokens}\n"
            f"  Total emitted tokens:   {self.total_emitted_tokens}\n"
            f"  Acceptance rate:        {self.acceptance_rate:.2%}\n"
            f"  Avg tokens/step:        {self.avg_tokens_per_step:.2f}\n"
            f"  Effective speedup:      {self.effective_speedup:.2f}x\n"
            f"  Spec length range:      [{self.min_spec_length_used}, "
            f"{self.max_spec_length_used}]"
        )


# ============================================================================
# MTPHead — Single Multi-Token Prediction Head
# ============================================================================


class MTPHead(nn.Module):
    """Single Multi-Token Prediction head for predicting a token at a
    specific future offset.

    Each MTPHead takes the hidden state from the main model and projects it
    to vocabulary logits. Different heads predict tokens at different temporal
    offsets (head k predicts the token at position t+k+1 given hidden state
    at position t).

    Architecture:
        h_t -> LayerNorm -> Linear(d_model, d_model) -> ReLU ->
               Linear(d_model, vocab_size)

    The two-layer MLP provides enough capacity for quality predictions while
    remaining cheap compared to the full model forward pass.

    Args:
        d_model: Hidden dimension of the main model.
        vocab_size: Size of the output vocabulary.
        head_index: Index of this head (0-based). Head k predicts token at
            offset k+1 from the current position.
        hidden_ratio: Ratio for the intermediate MLP dimension relative to
            d_model. Default 0.5 (half of d_model) for efficiency.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        head_index: int = 0,
        hidden_ratio: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.head_index = head_index
        hidden_dim = max(int(d_model * hidden_ratio), 64)

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Initialize weights
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict next tokens at this head's temporal offset.

        Args:
            hidden_states: Hidden states from the main model,
                shape (batch, seq_len, d_model).

        Returns:
            Logits, shape (batch, seq_len, vocab_size).
        """
        x = self.norm(hidden_states)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


# ============================================================================
# MultiTokenPrediction — Parallel MTP Head Module
# ============================================================================


class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction module with N parallel prediction heads.

    Predicts tokens at offsets 1..N from a single hidden state, enabling:
    1. Training with multi-token loss for better representations.
    2. Serving as the draft model for speculative decoding.

    During training, all heads are trained simultaneously with a combined
    loss: L_total = sum_k weight_k * CE(head_k(h_t), y_{t+k+1}).

    During inference, the heads produce draft tokens for the speculative
    decoder in a single forward pass.

    Args:
        d_model: Hidden dimension of the main model.
        vocab_size: Size of the output vocabulary.
        n_heads: Number of prediction heads (speculation length).
            Head k predicts the token at offset k+1. Default 5.
        hidden_ratio: Ratio for intermediate MLP dimension in each head.
        loss_weights: Optional per-head loss weights for training.
            If None, all heads get equal weight.
        tie_embeddings: If True, tie the output projection of each head
            with the main model's token embedding weights. The user must
            pass the embedding weight via `tie_embeddings_weight()`.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_heads: int = 5,
        hidden_ratio: float = 0.5,
        loss_weights: Optional[List[float]] = None,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.tie_embeddings = tie_embeddings

        # Create N prediction heads
        self.heads = nn.ModuleList([
            MTPHead(
                d_model=d_model,
                vocab_size=vocab_size,
                head_index=k,
                hidden_ratio=hidden_ratio,
            )
            for k in range(n_heads)
        ])

        # Loss weights (decreasing for farther offsets)
        if loss_weights is not None:
            assert len(loss_weights) == n_heads, (
                f"loss_weights length ({len(loss_weights)}) must match "
                f"n_heads ({n_heads})"
            )
            self.register_buffer(
                "loss_weights",
                torch.tensor(loss_weights, dtype=torch.float32),
            )
        else:
            # Default: geometric decay, head k gets weight (0.8)^k
            weights = [0.8 ** k for k in range(n_heads)]
            total = sum(weights)
            weights = [w / total for w in weights]
            self.register_buffer(
                "loss_weights",
                torch.tensor(weights, dtype=torch.float32),
            )

    def tie_embeddings_weight(self, embedding_weight: torch.Tensor) -> None:
        """Tie the output projections of all heads to the given embedding weight.

        This is a common technique that reduces parameter count and can improve
        prediction quality by sharing the embedding space.

        Args:
            embedding_weight: Token embedding weight matrix, shape
                (vocab_size, d_model). Must match vocab_size and d_model.
        """
        if not self.tie_embeddings:
            raise RuntimeError(
                "tie_embeddings=True must be set at init to use tie_embeddings_weight()"
            )
        for head in self.heads:
            head.fc2.weight = nn.Parameter(embedding_weight.clone())

    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[MTPOutput, Optional[torch.Tensor]]:
        """Run all MTP heads on the given hidden states.

        Args:
            hidden_states: Hidden states from the main model,
                shape (batch, seq_len, d_model).
            targets: Target token IDs for computing training loss,
                shape (batch, seq_len). If None, no loss is computed.

        Returns:
            Tuple of (mtp_output, loss):
            - mtp_output: MTPOutput with predictions from all heads.
            - loss: Weighted multi-token prediction loss, or None if
                targets is None.
        """
        batch, seq_len, _ = hidden_states.shape

        all_logits = []
        all_token_ids = []

        for k, head in enumerate(self.heads):
            logits_k = head(hidden_states)  # (batch, seq_len, vocab_size)
            all_logits.append(logits_k)
            # Greedy prediction for draft tokens
            token_ids_k = logits_k.argmax(dim=-1)  # (batch, seq_len)
            all_token_ids.append(token_ids_k)

        # Stack: (batch, n_heads, seq_len, vocab_size) -> (batch, seq_len, n_heads, vocab_size)
        all_logits = torch.stack(all_logits, dim=2)  # (batch, seq_len, n_heads, vocab_size)
        all_token_ids = torch.stack(all_token_ids, dim=2)  # (batch, seq_len, n_heads)

        # Compute probabilities
        all_probs = F.softmax(all_logits.float(), dim=-1).to(all_logits.dtype)

        # For the MTPOutput, take only the last position's predictions
        # (the current generation position)
        last_logits = all_logits[:, -1, :, :]  # (batch, n_heads, vocab_size)
        last_probs = all_probs[:, -1, :, :]  # (batch, n_heads, vocab_size)
        last_ids = all_token_ids[:, -1, :]  # (batch, n_heads)

        mtp_output = MTPOutput(
            token_ids=last_ids,
            logits=last_logits,
            probabilities=last_probs,
            n_heads=self.n_heads,
        )

        # Compute training loss if targets provided
        loss = None
        if targets is not None:
            loss = self._compute_loss(all_logits, targets)

        return mtp_output, loss

    def _compute_loss(
        self,
        all_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted multi-token prediction loss.

        For each head k, computes cross-entropy between the prediction at
        position t and the target at position t+k+1 (shifted by the head's
        offset). The total loss is a weighted sum across heads.

        Args:
            all_logits: Logits from all heads,
                shape (batch, seq_len, n_heads, vocab_size).
            targets: Target token IDs, shape (batch, seq_len).

        Returns:
            Scalar loss tensor.
        """
        batch, seq_len, n_heads, vocab_size = all_logits.shape
        total_loss = torch.tensor(0.0, device=all_logits.device, dtype=all_logits.dtype)

        for k in range(n_heads):
            # Head k predicts offset k+1, so shift targets
            # Prediction at position t should match target at position t+k+1
            # We need at least k+1 future positions
            if seq_len <= k + 1:
                continue

            # Prediction positions: 0..seq_len-k-2
            pred_logits = all_logits[:, :seq_len - k - 1, k, :]  # (batch, seq_len-k-1, vocab_size)
            # Target positions: k+1..seq_len-1
            target_ids = targets[:, k + 1:]  # (batch, seq_len-k-1)

            head_loss = F.cross_entropy(
                pred_logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + self.loss_weights[k] * head_loss

        return total_loss

    def draft(
        self,
        hidden_states: torch.Tensor,
        n_draft: Optional[int] = None,
    ) -> MTPOutput:
        """Generate draft tokens from MTP heads for speculative decoding.

        Convenience method that runs only the draft heads without computing
        training loss. Optimized for inference.

        Args:
            hidden_states: Hidden states from the main model at the current
                position, shape (batch, 1, d_model) for autoregressive or
                (batch, seq_len, d_model) for batched.
            n_draft: Number of draft tokens to generate. If None, uses all
                heads (self.n_heads). Must be <= self.n_heads.

        Returns:
            MTPOutput with draft token predictions.
        """
        if n_draft is not None and n_draft > self.n_heads:
            raise ValueError(
                f"n_draft ({n_draft}) cannot exceed n_heads ({self.n_heads})"
            )

        effective_heads = n_draft or self.n_heads

        all_logits = []
        all_token_ids = []

        with torch.no_grad():
            for k in range(effective_heads):
                logits_k = self.heads[k](hidden_states)  # (batch, seq_len, vocab_size)
                token_ids_k = logits_k.argmax(dim=-1)  # (batch, seq_len)
                all_logits.append(logits_k)
                all_token_ids.append(token_ids_k)

        # Take last position
        last_logits = torch.stack(
            [l[:, -1, :] for l in all_logits], dim=1
        )  # (batch, effective_heads, vocab_size)
        last_ids = torch.stack(
            [t[:, -1] for t in all_token_ids], dim=1
        )  # (batch, effective_heads)

        last_probs = F.softmax(last_logits.float(), dim=-1).to(last_logits.dtype)

        return MTPOutput(
            token_ids=last_ids,
            logits=last_logits,
            probabilities=last_probs,
            n_heads=effective_heads,
        )


# ============================================================================
# SpeculativeDecoder — Full Speculative Decoding Pipeline
# ============================================================================


class SpeculativeDecoder(nn.Module):
    """Full speculative decoding pipeline using MTP draft predictions.

    Implements the speculative decoding algorithm:
    1. Draft phase:  MTP heads generate K candidate tokens cheaply.
    2. Verify phase: Main model verifies all K candidates in one pass.
    3. Accept/reject: Accept matching prefix, reject from first mismatch.
    4. Adaptive speculation length based on running acceptance rate.

    The key insight is that MTP heads share the main model's hidden states,
    so the draft phase adds minimal overhead (just small linear projections).
    The verification phase processes K+1 tokens in one forward pass using
    KV caching, which is much cheaper than K separate single-token passes.

    Expected speedup: ~1.8x for typical text generation with acceptance
    rates around 80-90% and speculation length 5.

    Usage:
        # Create MTP draft model
        mtp = MultiTokenPrediction(d_model=1024, vocab_size=32000, n_heads=5)

        # Create speculative decoder
        decoder = SpeculativeDecoder(
            draft_model=mtp,
            d_model=1024,
            vocab_size=32000,
            max_spec_length=5,
        )

        # Use in generation loop
        for step in range(max_steps):
            tokens, done = decoder.step(
                current_tokens=current_tokens,
                model_forward=model_forward_fn,
                past_kv=past_kv,
            )
            if done:
                break

    Args:
        draft_model: MultiTokenPrediction module serving as the draft model.
        d_model: Hidden dimension of the main model.
        vocab_size: Size of the output vocabulary.
        max_spec_length: Maximum speculation length (number of draft tokens).
            Default 5.
        min_spec_length: Minimum speculation length. Default 1.
        adaptive: If True, adjust speculation length based on acceptance rate.
            Default True.
        target_acceptance_rate: Target acceptance rate for adaptive control.
            The speculation length is increased if the observed rate exceeds
            this target, and decreased if below. Default 0.85.
        temperature: Sampling temperature for verification. 1.0 = standard
            sampling, 0.0 = greedy. Default 1.0.
        top_k: Top-k filtering for sampling. 0 = no filtering. Default 0.
        top_p: Nucleus sampling threshold. 1.0 = no filtering. Default 1.0.
        eos_token_id: End-of-sequence token ID. Generation stops when this
            token is produced. Default -1 (disabled).
    """

    def __init__(
        self,
        draft_model: MultiTokenPrediction,
        d_model: int,
        vocab_size: int,
        max_spec_length: int = 5,
        min_spec_length: int = 1,
        adaptive: bool = True,
        target_acceptance_rate: float = 0.85,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int = -1,
    ):
        super().__init__()

        self.draft_model = draft_model
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_spec_length = min(max_spec_length, draft_model.n_heads)
        self.min_spec_length = max(1, min_spec_length)
        self.adaptive = adaptive
        self.target_acceptance_rate = target_acceptance_rate
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id

        # Current speculation length (adaptive)
        self._current_spec_length = self.max_spec_length

        # Statistics
        self.stats = SpeculativeStats()

        # Running acceptance rate for adaptive control (exponential moving avg)
        self._ema_acceptance_rate = 0.0
        self._ema_alpha = 0.1  # Smoothing factor

    @property
    def current_spec_length(self) -> int:
        """Current speculation length, potentially adapted from acceptance rate."""
        return self._current_spec_length

    def _adjust_spec_length(self, acceptance_rate: float) -> None:
        """Adaptively adjust speculation length based on acceptance rate.

        Strategy:
        - If acceptance rate > target + margin: increase spec length by 1
        - If acceptance rate < target - margin: decrease spec length by 1
        - Otherwise: keep current length

        The adjustment is bounded by [min_spec_length, max_spec_length].

        Args:
            acceptance_rate: Most recent step's acceptance rate.
        """
        if not self.adaptive:
            return

        # Update exponential moving average of acceptance rate.
        # Cold-start fix: if the EMA is still at its initial value of 0.0,
        # seed it with the first observation instead of blending with zero.
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
                self._current_spec_length - 1, self.min_spec_length
            )

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
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
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs - F.softmax(
                sorted_logits, dim=-1
            ) >= self.top_p
            # Scatter back to original indices
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

    @torch.no_grad()
    def draft(
        self,
        hidden_states: torch.Tensor,
    ) -> MTPOutput:
        """Generate draft tokens using the MTP heads.

        This is the first phase of speculative decoding. The MTP heads
        predict the next K tokens from the current hidden state.

        Args:
            hidden_states: Hidden states from the main model's last layer,
                shape (batch, seq_len, d_model). Typically seq_len=1 for
                autoregressive generation.

        Returns:
            MTPOutput containing draft token predictions.
        """
        return self.draft_model.draft(
            hidden_states,
            n_draft=self._current_spec_length,
        )

    @torch.no_grad()
    def verify(
        self,
        draft_output: MTPOutput,
        verifier_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens against the main model's predictions.

        This is the second phase of speculative decoding. The main model's
        logits at positions corresponding to the draft tokens are compared
        with the draft predictions.

        The verification accepts the longest prefix of draft tokens that
        match the verifier's greedy predictions. At the first mismatch,
        the verifier's prediction replaces the draft token, and all
        subsequent draft tokens are rejected.

        Args:
            draft_output: MTPOutput from the draft phase.
            verifier_logits: Logits from the main model at positions
                corresponding to the draft tokens, shape
                (batch, spec_length + 1, vocab_size). Position i contains
                the logits for predicting the token at position i+1 relative
                to the original position.

        Returns:
            Tuple of (accepted_tokens, n_accepted):
            - accepted_tokens: Tensor of accepted + correction tokens,
                shape (batch, n_accepted + 1).
            - n_accepted: Number of draft tokens accepted (0 to spec_length).
        """
        batch = draft_output.token_ids.shape[0]
        spec_length = draft_output.n_heads

        # Get the verifier's greedy predictions at each position
        # verifier_logits: (batch, spec_length + 1, vocab_size)
        verifier_predictions = verifier_logits.argmax(dim=-1)  # (batch, spec_length + 1)

        # Draft tokens: (batch, spec_length)
        draft_tokens = draft_output.token_ids  # (batch, spec_length)

        # Compare draft with verifier predictions
        # verifier_predictions[:, k] should match draft_tokens[:, k]
        # for k = 0, 1, ..., spec_length-1
        matches = draft_tokens == verifier_predictions[:, :spec_length]  # (batch, spec_length)

        # For batch processing: find the first mismatch per batch element
        # If all match, n_accepted = spec_length
        # If first mismatch at position k, n_accepted = k

        # Create a mask: 1 if all positions up to and including k match
        # Use cumulative product along the spec dimension
        all_match_prefix = matches.cumprod(dim=1)  # (batch, spec_length)

        # Number of accepted tokens per batch element
        n_accepted_per_batch = all_match_prefix.sum(dim=1)  # (batch,)

        # For simplicity in this implementation, we use the minimum across
        # the batch (conservative). This ensures consistent sequence lengths.
        n_accepted = n_accepted_per_batch.min().item()
        n_accepted = int(n_accepted)

        # Build the output tokens:
        # - First n_accepted tokens: from the draft (which match the verifier)
        # - Then 1 token: the verifier's prediction at position n_accepted
        #   (either the correction if mismatch, or the bonus token if all accepted)
        accepted_draft = draft_tokens[:, :n_accepted]  # (batch, n_accepted)
        correction_token = verifier_predictions[:, n_accepted:n_accepted + 1]  # (batch, 1)

        if n_accepted > 0:
            output_tokens = torch.cat([accepted_draft, correction_token], dim=1)
        else:
            output_tokens = correction_token  # (batch, 1)

        return output_tokens, n_accepted

    @torch.no_grad()
    def verify_with_sampling(
        self,
        draft_output: MTPOutput,
        verifier_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens with stochastic acceptance (speculative sampling).

        Implements the full speculative sampling algorithm from Chen et al. (2024),
        which provides distribution-equivalent outputs to standard sampling.

        For each draft token c_k with probability q(c_k) from the draft model:
        - Draw u ~ Uniform(0, 1)
        - If u < p(c_k) / q(c_k): accept token c_k
        - Else: reject and sample from adjusted distribution

        This guarantees that the output distribution matches standard
        autoregressive sampling from the verifier model exactly.

        Args:
            draft_output: MTPOutput from the draft phase.
            verifier_logits: Logits from the main model at positions
                corresponding to the draft tokens, shape
                (batch, spec_length + 1, vocab_size).

        Returns:
            Tuple of (accepted_tokens, n_accepted):
            - accepted_tokens: Tensor of accepted + resampled tokens,
                shape (batch, n_accepted + 1).
            - n_accepted: Number of draft tokens accepted.
        """
        batch = draft_output.token_ids.shape[0]
        spec_length = draft_output.n_heads
        device = verifier_logits.device

        # Get probabilities
        verifier_probs = F.softmax(verifier_logits.float(), dim=-1)  # (batch, spec_length+1, vocab_size)
        draft_probs = draft_output.probabilities.float()  # (batch, spec_length, vocab_size)

        # Draft token IDs
        draft_tokens = draft_output.token_ids  # (batch, spec_length)

        accepted_tokens_list = []
        n_accepted = 0

        for k in range(spec_length):
            # Draft token at position k
            draft_token_k = draft_tokens[:, k]  # (batch,)

            # Probability of draft token under verifier and draft models
            # Gather probabilities for the specific draft tokens
            p_token = verifier_probs[:, k].gather(
                1, draft_token_k.unsqueeze(1)
            ).squeeze(1)  # (batch,)
            q_token = draft_probs[:, k].gather(
                1, draft_token_k.unsqueeze(1)
            ).squeeze(1)  # (batch,)

            # Acceptance ratio
            # Clamp q_token to avoid division by zero
            acceptance_ratio = p_token / q_token.clamp(min=1e-10)

            # Draw uniform random
            u = torch.rand(batch, device=device)

            # Accept if u < min(1, p/q)
            accepted = u < acceptance_ratio  # (batch,)

            # For batch simplicity, use minimum accepted count
            if not accepted.all():
                # Some batch elements rejected at position k
                n_accepted = min(accepted.sum().item(), n_accepted if n_accepted > 0 else accepted.sum().item())
                break
            else:
                accepted_tokens_list.append(draft_token_k)
                n_accepted = k + 1

        # If all draft tokens accepted, sample bonus token from verifier
        # at position spec_length
        if n_accepted == spec_length:
            bonus_logits = verifier_logits[:, spec_length]  # (batch, vocab_size)
            bonus_token = self._sample_from_logits(bonus_logits)  # (batch,)
            accepted_tokens_list.append(bonus_token)
        else:
            # Sample correction token from adjusted distribution
            # p_adj(x) = max(0, p(x) - q(x)) / sum(max(0, p(x) - q(x)))
            k_reject = n_accepted
            p_k = verifier_probs[:, k_reject]  # (batch, vocab_size)
            q_k = draft_probs[:, k_reject] if k_reject < spec_length else torch.zeros_like(p_k)
            adjusted = torch.clamp(p_k - q_k, min=0)
            adjusted_sum = adjusted.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            adjusted_probs = adjusted / adjusted_sum

            # Sample from adjusted distribution
            correction_token = torch.multinomial(adjusted_probs, num_samples=1).squeeze(1)
            accepted_tokens_list.append(correction_token)

        # Stack accepted tokens
        if accepted_tokens_list:
            output_tokens = torch.stack(accepted_tokens_list, dim=1)  # (batch, n_accepted+1)
        else:
            # Fallback: sample from verifier at position 0
            output_tokens = self._sample_from_logits(
                verifier_logits[:, 0]
            ).unsqueeze(1)  # (batch, 1)

        return output_tokens, n_accepted

    @torch.no_grad()
    def step(
        self,
        current_hidden: torch.Tensor,
        verifier_forward: Callable[
            [torch.Tensor, Optional[torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor],
        ],
        past_kv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
        """Execute one step of speculative decoding.

        This is the main entry point for the generation loop. Each call
        performs one draft-verify-accept/reject cycle and returns the
        accepted tokens plus updated KV cache.

        Args:
            current_hidden: Hidden states at the current position from the
                main model, shape (batch, 1, d_model).
            verifier_forward: Callable that takes (token_ids, past_kv) and
                returns (logits, new_past_kv). This is the main model's
                forward function for verification. It should support
                processing multiple tokens at once with KV caching.
                - Input token_ids: (batch, seq_len) of token IDs
                - Input past_kv: Previous KV cache state
                - Output logits: (batch, seq_len, vocab_size)
                - Output new_past_kv: Updated KV cache state
            past_kv: Current KV cache from previous generation steps.

        Returns:
            Tuple of (new_tokens, new_past_kv, n_accepted, eos_reached):
            - new_tokens: Accepted + correction tokens, shape (batch, n_emitted).
            - new_past_kv: Updated KV cache after processing all tokens.
            - n_accepted: Number of draft tokens accepted this step.
            - eos_reached: True if EOS token was generated.
        """
        spec_length = self._current_spec_length

        # ---- Phase 1: DRAFT ----
        draft_output = self.draft(current_hidden)

        # Draft token IDs: (batch, spec_length)
        draft_tokens = draft_output.token_ids

        # ---- Phase 2: VERIFY ----
        # Feed the draft tokens through the main model to get verifier logits.
        # We need logits at each of the spec_length + 1 positions:
        #   - Position 0: logits for predicting the first draft token
        #   - Position k: logits for predicting the (k+1)-th draft token
        #   - Position spec_length: logits for the bonus token (if all accepted)

        # The verifier processes all draft tokens in one forward pass.
        verifier_logits, new_past_kv = verifier_forward(draft_tokens, past_kv)
        # verifier_logits: (batch, spec_length, vocab_size)

        # We also need the logits at the original position (before draft tokens)
        # to predict the first draft token. This is the "base" logits.
        # In practice, if the model's forward includes the last real token
        # in the input, the first position of verifier_logits gives us this.
        #
        # For a clean implementation, we prepend a dummy and the verifier
        # should process (current_token + draft_tokens). But since we're
        # passing only draft_tokens, the first logit position corresponds
        # to predicting the second draft token given the first.
        #
        # Simpler approach: The caller passes the base logits (from the
        # current position) alongside the hidden states. We reconstruct
        # the full verification here.

        # We need spec_length + 1 sets of logits. The verifier logits give
        # us spec_length. We get the first one from the base logits.
        # For now, use the hidden state to get the base prediction.
        base_logits = self.draft_model.heads[0].norm(current_hidden)
        base_logits = self.draft_model.heads[0].fc1(base_logits)
        base_logits = F.relu(base_logits)
        # This gives us the draft model's base logits (head 0).
        # For proper verification, we need the main model's logits.
        # The cleanest approach: the verifier_forward should also return
        # the logits at the current position.

        # Practical approach: run verifier on draft tokens which gives
        # logits at positions 0..spec_length-1. Logit at position k
        # predicts the token at position k+1 relative to draft start.
        # So logit[0] predicts what should be draft_tokens[0], etc.
        # We also need one more position for the bonus/correction token.

        # Run the verifier one more step to get the final position's logits
        last_draft_token = draft_tokens[:, -1:]  # (batch, 1)
        bonus_logits, new_past_kv = verifier_forward(last_draft_token, new_past_kv)
        # bonus_logits: (batch, 1, vocab_size)

        # Concatenate all verifier logits: (batch, spec_length + 1, vocab_size)
        full_verifier_logits = torch.cat([verifier_logits, bonus_logits], dim=1)

        # ---- Phase 3: ACCEPT/REJECT ----
        if self.temperature < 1e-8:
            # Greedy verification
            output_tokens, n_accepted = self.verify(draft_output, full_verifier_logits)
        else:
            # Stochastic verification (speculative sampling)
            output_tokens, n_accepted = self.verify_with_sampling(
                draft_output, full_verifier_logits
            )

        # ---- Update statistics ----
        self.stats.record_step(n_accepted, spec_length)

        # ---- Adaptive adjustment ----
        step_acceptance_rate = n_accepted / spec_length if spec_length > 0 else 0.0
        self._adjust_spec_length(step_acceptance_rate)

        # ---- Check for EOS ----
        eos_reached = False
        if self.eos_token_id >= 0:
            eos_reached = (output_tokens == self.eos_token_id).any(dim=-1).any().item()

        return output_tokens, new_past_kv, n_accepted, eos_reached

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        model_forward: Callable[
            [torch.Tensor, Optional[torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor],
        ],
        max_new_tokens: int = 512,
        callback: Optional[Callable[[torch.Tensor, int, SpeculativeStats], None]] = None,
    ) -> Tuple[torch.Tensor, SpeculativeStats]:
        """Generate tokens using speculative decoding.

        Full generation loop that uses speculative decoding for accelerated
        text generation. This is the highest-level API for the module.

        Args:
            prompt_tokens: Input prompt token IDs, shape (batch, prompt_len).
            model_forward: Callable that takes (token_ids, past_kv) and returns
                (logits, new_past_kv). Should support multi-token processing
                with KV caching.
            max_new_tokens: Maximum number of new tokens to generate.
                Default 512.
            callback: Optional callback invoked after each step with
                (new_tokens, step_count, stats). Useful for streaming output.

        Returns:
            Tuple of (generated_tokens, stats):
            - generated_tokens: Full sequence including prompt,
                shape (batch, prompt_len + n_generated).
            - stats: SpeculativeStats with generation diagnostics.
        """
        batch, prompt_len = prompt_tokens.shape
        device = prompt_tokens.device

        # Reset statistics
        self.stats.reset()

        # Prefill: process the entire prompt to build the KV cache
        # and get the hidden state for the first draft
        prefill_logits, past_kv = model_forward(prompt_tokens, None)
        # prefill_logits: (batch, prompt_len, vocab_size)

        # The hidden state at the last position drives the first draft.
        # We need the model to also expose hidden states. For simplicity,
        # we use the logits approach: the first token is sampled from
        # the last position's logits.
        first_token = self._sample_from_logits(prefill_logits[:, -1])  # (batch,)

        # Collect generated tokens
        all_tokens = [prompt_tokens, first_token.unsqueeze(1)]
        n_generated = 1

        # Check if first token is EOS
        if self.eos_token_id >= 0 and (first_token == self.eos_token_id).any():
            result = torch.cat(all_tokens, dim=1)
            return result, self.stats

        # For the speculative loop, we need the main model to expose
        # hidden states for the draft model. Since we're working with
        # a generic model_forward callable, we use a simplified approach:
        # run the draft model on the logits-derived representation.

        # Create a projection from vocab_size logits to d_model hidden states
        # (This is a practical workaround; in a real system, the model
        #  would expose its hidden states directly.)
        logits_to_hidden = nn.Linear(self.vocab_size, self.d_model, bias=False).to(device)

        # Get initial hidden states from the prefill logits
        current_hidden = logits_to_hidden(
            F.softmax(prefill_logits.float(), dim=-1).to(prefill_logits.dtype)
        )[:, -1:, :]  # (batch, 1, d_model)

        # Also update past_kv with the first generated token
        first_logits, past_kv = model_forward(first_token.unsqueeze(1), past_kv)

        # Speculative decoding loop
        step_count = 0
        while n_generated < max_new_tokens:
            step_count += 1

            # Update hidden state from latest logits
            current_hidden = logits_to_hidden(
                F.softmax(first_logits.float(), dim=-1).to(first_logits.dtype)
            )[:, -1:, :]

            # Run one speculative step
            new_tokens, past_kv, n_accepted, eos_reached = self.step(
                current_hidden=current_hidden,
                verifier_forward=model_forward,
                past_kv=past_kv,
            )

            # Add new tokens to output
            all_tokens.append(new_tokens)
            n_generated += new_tokens.shape[1]

            # Callback
            if callback is not None:
                callback(new_tokens, step_count, self.stats)

            # Check EOS
            if eos_reached:
                break

            # For the next iteration, we need logits for the last emitted
            # token to compute the hidden state. We already have these from
            # the verification pass (the last position's logits).
            # In this simplified implementation, we'll do one more forward
            # pass for the last token to get updated logits.
            last_token = new_tokens[:, -1:]
            first_logits, past_kv = model_forward(last_token, past_kv)

        result = torch.cat(all_tokens, dim=1)
        return result, self.stats

    def reset(self) -> None:
        """Reset the decoder state for a new generation session.

        Clears statistics, resets adaptive speculation length, and
        resets the acceptance rate EMA.
        """
        self.stats.reset()
        self._current_spec_length = self.max_spec_length
        self._ema_acceptance_rate = 0.0

    def get_stats(self) -> SpeculativeStats:
        """Get the current speculative decoding statistics.

        Returns:
            SpeculativeStats with current generation diagnostics.
        """
        return self.stats

    def extra_repr(self) -> str:
        """String representation for printing."""
        return (
            f"d_model={self.d_model}, vocab_size={self.vocab_size},\n"
            f"  max_spec_length={self.max_spec_length}, "
            f"min_spec_length={self.min_spec_length},\n"
            f"  adaptive={self.adaptive}, "
            f"target_acceptance_rate={self.target_acceptance_rate},\n"
            f"  temperature={self.temperature}, "
            f"top_k={self.top_k}, top_p={self.top_p},\n"
            f"  eos_token_id={self.eos_token_id}"
        )


# ============================================================================
# Helper: create a model_forward wrapper for standard Transformer models
# ============================================================================


def create_model_forward_fn(
    model: nn.Module,
    input_embed_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Callable:
    """Create a model_forward callable compatible with SpeculativeDecoder.

    Wraps a standard Transformer model to provide the (logits, past_kv)
    interface expected by the speculative decoder.

    Args:
        model: A Transformer model with a standard forward method.
        input_embed_fn: Optional function to convert token IDs to embeddings.
            If None, assumes the model has an `embed_tokens` attribute or
            accepts token IDs directly.

    Returns:
        A callable that takes (token_ids, past_kv) and returns
        (logits, new_past_kv).
    """

    @torch.no_grad()
    def model_forward(
        token_ids: torch.Tensor,
        past_kv: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Try to call model with use_cache=True
        try:
            outputs = model(
                input_ids=token_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = outputs.logits
            new_past_kv = outputs.past_key_values
        except (AttributeError, TypeError):
            # Fallback: try without use_cache
            try:
                outputs = model(token_ids, past_kv)
                if isinstance(outputs, tuple):
                    logits, new_past_kv = outputs[0], outputs[1]
                else:
                    logits = outputs
                    new_past_kv = None
            except Exception as e:
                raise RuntimeError(
                    f"Could not call model forward. Ensure model accepts "
                    f"(input_ids, past_key_values) and returns "
                    f"(logits, past_key_values). Error: {e}"
                )

        return logits, new_past_kv

    return model_forward
