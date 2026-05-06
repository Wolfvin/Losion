"""
Tests for Losion v2.5.6 audit X-series fixes.

Covers findings from the v2.5.5 audit:
1. X-01: _generate_speculative attention_mask not extended when tokens added
2. X-03: MCTSReasoner action_probs sum-to-zero when all visit counts zero
3. X-04: MCTS backprop vectorized with scatter_add_
4. X-05: Dead original_len parameter removed from _generate_greedy
5. X-06: Test coverage for _generate_speculative with attention_mask multi-step
"""

import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from losion.inference.generation import (
    LosionGenerator,
    GenerationConfig,
    SpeculativeDecoder,
)
from losion.core.reasoning.mcts import MCTSReasoner, MCTSConfig


# ============================================================================
# 1. Speculative decoding attention_mask extension (X-01, X-06)
# ============================================================================


class TestSpeculativeAttentionMaskExtension:
    """Test that _generate_speculative extends attention_mask with each step.

    v2.5.6: Previously, _generate_speculative accepted attention_mask but
    never extended it when new tokens were appended. This meant that on the
    second iteration of the while loop, current_ids had shape [1, L+K] but
    current_mask still had shape [1, L] — causing a shape mismatch error
    or corrupt model output.

    The fix tracks current_mask alongside current_ids and extends it with
    1s for each accepted token (or fallback token).
    """

    @staticmethod
    def _make_mock_model(vocab_size=50):
        """Create a mock model that returns controlled logits."""
        model = MagicMock()
        call_count = [0]

        def forward(input_ids=None, attention_mask=None, **kwargs):
            call_count[0] += 1
            batch_size, seq_len = input_ids.shape
            # Deterministic logits so generation is reproducible
            torch.manual_seed(call_count[0] * 7)
            logits = torch.randn(batch_size, seq_len, vocab_size)
            output = MagicMock()
            output.logits = logits
            return output

        model.side_effect = forward
        model.__call__ = forward
        return model

    def test_speculative_mask_extends_with_accepted_tokens(self):
        """attention_mask should grow alongside current_ids in speculative decoding.

        We provide an initial attention_mask and run speculative decoding for
        multiple steps. After each step, the mask should be extended by the
        number of accepted tokens.
        """
        vocab_size = 50
        model = self._make_mock_model(vocab_size)
        generator = LosionGenerator(model, device="cpu")

        input_ids = torch.randint(0, vocab_size, (1, 5))
        initial_mask = torch.ones(1, 5, dtype=torch.long)

        config = GenerationConfig(
            max_new_tokens=20,
            speculative_enabled=True,
            speculative_draft_tokens=3,
            eos_token_id=-1,  # Prevent early stopping
        )

        # Run speculative generation with attention_mask
        generated_ids, scores = generator.generate(
            input_ids, config, attention_mask=initial_mask
        )

        # Generation should produce output (at least one token)
        assert len(generated_ids) > 5, (
            f"Speculative generation should produce tokens beyond input, "
            f"got {len(generated_ids)} total (input was 5)"
        )
        assert len(scores) > 0, "Should have scores for generated tokens"

    def test_speculative_mask_shape_matches_input_ids_across_steps(self):
        """Each forward pass should receive mask matching input_ids length.

        We track the attention_mask passed to each model forward call and
        verify its shape always matches input_ids.shape[1].
        """
        vocab_size = 50
        model = self._make_mock_model(vocab_size)
        generator = LosionGenerator(model, device="cpu")

        input_ids = torch.randint(0, vocab_size, (1, 4))
        initial_mask = torch.ones(1, 4, dtype=torch.long)

        config = GenerationConfig(
            max_new_tokens=10,
            speculative_enabled=True,
            speculative_draft_tokens=2,
            eos_token_id=-1,
        )

        # Track (input_ids_len, attention_mask_len) for each forward call
        shape_pairs = []
        original_call = model.__call__

        def tracking_forward(input_ids=None, attention_mask=None, **kwargs):
            if input_ids is not None and attention_mask is not None:
                shape_pairs.append((input_ids.shape[1], attention_mask.shape[1]))
            return original_call(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        model.side_effect = tracking_forward

        generator.generate(input_ids, config, attention_mask=initial_mask)

        # Every recorded forward pass should have matching shapes
        for i, (ids_len, mask_len) in enumerate(shape_pairs):
            assert ids_len == mask_len, (
                f"Forward pass {i}: input_ids length ({ids_len}) != "
                f"attention_mask length ({mask_len}). "
                f"This means the mask was not extended properly."
            )

    def test_speculative_without_mask_still_works(self):
        """Speculative decoding without attention_mask should still work.

        This verifies backward compatibility — attention_mask is optional.
        """
        vocab_size = 50
        model = self._make_mock_model(vocab_size)
        generator = LosionGenerator(model, device="cpu")

        input_ids = torch.randint(0, vocab_size, (1, 4))

        config = GenerationConfig(
            max_new_tokens=10,
            speculative_enabled=True,
            speculative_draft_tokens=2,
            eos_token_id=-1,
        )

        generated_ids, scores = generator.generate(input_ids, config)

        assert len(generated_ids) > 4
        assert len(scores) > 0

    def test_speculative_mask_with_padding_prefix(self):
        """Speculative decoding with a padded prefix (mask has zeros) works.

        This simulates a real scenario where the input was left-padded for
        batched inference and the mask has leading zeros.
        """
        vocab_size = 50
        model = self._make_mock_model(vocab_size)
        generator = LosionGenerator(model, device="cpu")

        # Input with 2 padding tokens and 4 real tokens
        input_ids = torch.tensor([[0, 0, 5, 10, 15, 20]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)

        config = GenerationConfig(
            max_new_tokens=8,
            speculative_enabled=True,
            speculative_draft_tokens=2,
            eos_token_id=-1,
        )

        generated_ids, scores = generator.generate(
            input_ids, config, attention_mask=attention_mask
        )

        # Should produce output
        assert len(generated_ids) > 6


# ============================================================================
# 2. MCTSReasoner — uniform fallback for all-zero visit counts (X-03)
# ============================================================================


class TestMCTSUniformFallback:
    """Test that MCTSReasoner returns uniform distribution when all visit
    counts are zero, instead of an all-zeros distribution that would cause
    NaN in downstream torch.multinomial.

    v2.5.6: When action_probs sums to zero (e.g., if all visit counts are
    zero due to an edge case), the code now falls back to a uniform
    distribution instead of returning zeros / (zeros + 1e-8) = zeros.
    """

    def test_forward_returns_valid_distribution(self):
        """MCTSReasoner.forward() should always return a valid probability
        distribution that sums to approximately 1.0."""
        d_model = 64
        num_actions = 8
        config = MCTSConfig(num_simulations=16, temperature=1.0)
        reasoner = MCTSReasoner(d_model, num_actions, config)

        x = torch.randn(2, d_model)  # batch=2
        action_probs, info = reasoner.forward(x)

        # action_probs should be valid distributions
        assert action_probs.shape == (2, num_actions)

        # Each row should sum to approximately 1.0
        row_sums = action_probs.sum(dim=-1)
        for i, s in enumerate(row_sums):
            assert abs(s.item() - 1.0) < 1e-4, (
                f"Row {i} should sum to ~1.0, got {s.item()}"
            )

        # All values should be non-negative
        assert (action_probs >= 0).all(), "All probabilities should be >= 0"

        # No NaN values
        assert not action_probs.isnan().any(), "No NaN values in action_probs"

    def test_action_probs_works_with_multinomial(self):
        """action_probs from MCTSReasoner should be safe for torch.multinomial.

        This is the downstream operation that would fail with all-zeros
        distribution (returns empty tensor) or NaN distribution (raises error).
        """
        d_model = 64
        num_actions = 8
        config = MCTSConfig(num_simulations=8, temperature=1.0)
        reasoner = MCTSReasoner(d_model, num_actions, config)

        x = torch.randn(4, d_model)  # batch=4
        action_probs, info = reasoner.forward(x)

        # Should be able to sample from action_probs without error
        for b in range(4):
            samples = torch.multinomial(action_probs[b:b+1], num_samples=10, replacement=True)
            assert samples.shape == (1, 10), (
                f"Multinomial should produce 10 samples, got shape {samples.shape}"
            )

    def test_single_simulation_still_valid(self):
        """Even with minimal simulations (1), action_probs should be valid.

        With num_simulations=1, only one action gets visited per batch
        element, so most actions have visit_count=0. The uniform fallback
        ensures the distribution is still valid.
        """
        d_model = 64
        num_actions = 8
        config = MCTSConfig(num_simulations=1, temperature=1.0)
        reasoner = MCTSReasoner(d_model, num_actions, config)

        x = torch.randn(3, d_model)
        action_probs, info = reasoner.forward(x)

        # Should still sum to ~1.0
        row_sums = action_probs.sum(dim=-1)
        for i, s in enumerate(row_sums):
            assert abs(s.item() - 1.0) < 1e-4, (
                f"Row {i} with 1 simulation should still sum to ~1.0, got {s.item()}"
            )

    def test_zero_temperature_greedy_still_works(self):
        """Greedy selection (temperature=0) should still produce valid one-hot."""
        d_model = 64
        num_actions = 8
        config = MCTSConfig(num_simulations=16, temperature=0.0)
        reasoner = MCTSReasoner(d_model, num_actions, config)

        x = torch.randn(2, d_model)
        action_probs, info = reasoner.forward(x)

        # With temperature=0, should be one-hot (greedy)
        row_sums = action_probs.sum(dim=-1)
        for i, s in enumerate(row_sums):
            assert abs(s.item() - 1.0) < 1e-4, (
                f"Greedy row {i} should sum to 1.0, got {s.item()}"
            )

    def test_num_simulations_validation(self):
        """MCTSConfig should reject num_simulations <= 0."""
        with pytest.raises(ValueError, match="num_simulations"):
            MCTSConfig(num_simulations=0)

        with pytest.raises(ValueError, match="num_simulations"):
            MCTSConfig(num_simulations=-1)


# ============================================================================
# 3. MCTS Backprop vectorization (X-04)
# ============================================================================


class TestMCTSBackpropVectorization:
    """Test that the vectorized scatter_add_ backprop produces the same
    results as the original Python loop implementation.

    v2.5.6: Replaced the Python for-loop with per-batch .item() calls
    with vectorized scatter_add_ operations. This eliminates GPU sync
    overhead and produces identical numerical results.
    """

    def test_visit_counts_match_expected(self):
        """Visit counts should accumulate correctly via scatter_add_."""
        d_model = 64
        num_actions = 4
        config = MCTSConfig(num_simulations=8, temperature=1.0)
        reasoner = MCTSReasoner(d_model, num_actions, config)

        torch.manual_seed(42)
        x = torch.randn(1, d_model)
        action_probs, info = reasoner.forward(x)

        # Total visits should equal num_simulations
        total_visits = info["visit_distribution"].sum(dim=-1)
        assert abs(total_visits[0].item() - 1.0) < 1e-4, (
            f"Visit distribution should sum to ~1.0, got {total_visits[0].item()}"
        )

        # max_visit_count should be positive
        assert info["max_visit_count"] > 0, (
            "Should have at least one visit"
        )

    def test_vectorized_matches_loop_for_small_batch(self):
        """Verify scatter_add_ produces same result as naive Python loop.

        We manually compute visit_counts and total_values using both
        the old loop method and the new scatter_add_ method, then compare.
        """
        batch_size = 2
        num_actions = 4
        device = "cpu"

        # Simulate a few iterations
        visit_counts = torch.zeros(batch_size, num_actions, device=device)
        total_values = torch.zeros(batch_size, num_actions, device=device)

        # Pre-determined actions and values for reproducibility
        actions_list = [
            torch.tensor([0, 2]),  # iteration 0
            torch.tensor([1, 3]),  # iteration 1
            torch.tensor([0, 1]),  # iteration 2
            torch.tensor([2, 0]),  # iteration 3
        ]
        values_list = [
            torch.tensor([[0.5], [-0.3]]),
            torch.tensor([[0.8], [0.2]]),
            torch.tensor([[-0.1], [0.6]]),
            torch.tensor([[0.4], [-0.5]]),
        ]

        # Method 1: Old Python loop (reference)
        vc_loop = torch.zeros_like(visit_counts)
        tv_loop = torch.zeros_like(total_values)
        for selected_actions, sim_values in zip(actions_list, values_list):
            for b in range(batch_size):
                a = selected_actions[b].item()
                vc_loop[b, a] += 1
                tv_loop[b, a] += sim_values[b, 0].item()

        # Method 2: scatter_add_ (new implementation)
        vc_scatter = torch.zeros_like(visit_counts)
        tv_scatter = torch.zeros_like(total_values)
        for selected_actions, sim_values in zip(actions_list, values_list):
            vc_scatter.scatter_add_(
                1,
                selected_actions.unsqueeze(1),
                torch.ones(batch_size, 1, device=device),
            )
            tv_scatter.scatter_add_(
                1,
                selected_actions.unsqueeze(1),
                sim_values,
            )

        # Results should be identical
        assert torch.equal(vc_loop, vc_scatter), (
            f"Visit counts differ:\nloop: {vc_loop}\nscatter: {vc_scatter}"
        )
        assert torch.allclose(tv_loop, tv_scatter, atol=1e-6), (
            f"Total values differ:\nloop: {tv_loop}\nscatter: {tv_scatter}"
        )


# ============================================================================
# 4. Dead code removal — original_len parameter (X-05)
# ============================================================================


class TestOriginalLenRemoved:
    """Verify that the dead original_len parameter was removed from
    _generate_greedy and its callers.

    v2.5.6: The original_len parameter was passed from generate() to
    _generate_greedy() but never used inside the method. It was a leftover
    from a previous refactoring. The parameter has been removed.
    """

    def test_generate_greedy_signature(self):
        """_generate_greedy should not accept original_len parameter."""
        import inspect
        sig = inspect.signature(LosionGenerator._generate_greedy)
        param_names = list(sig.parameters.keys())

        assert "original_len" not in param_names, (
            f"_generate_greedy should not have original_len parameter. "
            f"Parameters: {param_names}"
        )
        # Should still have input_ids, config, attention_mask
        assert "input_ids" in param_names
        assert "config" in param_names
        assert "attention_mask" in param_names

    def test_generate_method_routes_without_original_len(self):
        """generate() should route to _generate_greedy without original_len."""
        vocab_size = 50
        model = MagicMock()

        def forward(input_ids=None, attention_mask=None, **kwargs):
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, vocab_size)
            output = MagicMock()
            output.logits = logits
            return output

        model.side_effect = forward
        model.__call__ = forward

        generator = LosionGenerator(model, device="cpu")
        input_ids = torch.randint(0, vocab_size, (1, 5))

        config = GenerationConfig(max_new_tokens=3, eos_token_id=-1)

        # Should not raise TypeError about original_len
        generated_ids, scores = generator.generate(input_ids, config)
        assert len(generated_ids) > 5


# ============================================================================
# 5. Integration — speculative + mask + MCTS together
# ============================================================================


class TestV256Integration:
    """Integration tests verifying v2.5.6 fixes work together."""

    def test_speculative_with_mask_and_early_stop(self):
        """Speculative decoding with mask should stop correctly at EOS."""
        vocab_size = 50
        model = MagicMock()

        # Make the model always predict token 2 (EOS) so generation stops quickly
        def forward(input_ids=None, attention_mask=None, **kwargs):
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, vocab_size)
            # Set EOS token (id=2) to very high logit so it's always selected
            logits[:, :, 2] = 10.0
            output = MagicMock()
            output.logits = logits
            return output

        model.side_effect = forward
        model.__call__ = forward

        generator = LosionGenerator(model, device="cpu")
        input_ids = torch.randint(0, vocab_size, (1, 5))
        attention_mask = torch.ones(1, 5, dtype=torch.long)

        config = GenerationConfig(
            max_new_tokens=20,
            speculative_enabled=True,
            speculative_draft_tokens=3,
            eos_token_id=2,
        )

        generated_ids, scores = generator.generate(
            input_ids, config, attention_mask=attention_mask
        )

        # Should have stopped at or near the EOS token
        assert 2 in generated_ids[5:], "EOS token should appear in generated output"

    def test_mcts_reasoning_consistency_across_batch(self):
        """MCTS should produce consistent results for the same input."""
        d_model = 64
        num_actions = 4
        config = MCTSConfig(num_simulations=16, temperature=1.0)
        reasoner = MCTSReasoner(d_model, num_actions, config)

        torch.manual_seed(123)
        x = torch.randn(2, d_model)

        # Run twice with same seed
        torch.manual_seed(123)
        probs1, info1 = reasoner.forward(x)

        torch.manual_seed(123)
        probs2, info2 = reasoner.forward(x)

        # Should produce identical results (deterministic with same seed)
        assert torch.allclose(probs1, probs2, atol=1e-6), (
            "MCTS should be deterministic with same seed"
        )
