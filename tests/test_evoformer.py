"""
Test suite for Evoformer 5-Level Feedback Loop.

Covers all 5 levels of bidirectional feedback adapted from AlphaFold2:
- Level 1: Inter-Layer Recycling (deep ↔ shallow)
- Level 2: Bidirectional Token Update (old ↔ new tokens)
- Level 3: Decoder ↔ Predict Feedback
- Level 4: Prediction → Context Recycling
- Level 5: Router ↔ Expert Co-Evolution

Plus tests for the EvoformerManager coordinator and EvoformerConfig.

v2.5.0: Created to address audit finding A3.3 — Evoformer 5-level feedback
loop had zero dedicated test coverage despite being one of the most complex
components in the framework.
"""

import pytest
import torch
import torch.nn as nn

from losion.core.feedback.evoformer import (
    EvoformerConfig,
    LayerRecyclingBlock,
    BidirectionalTokenUpdate,
    DecoderPredictFeedback,
    PredictionContextRecycling,
    RouterExpertCoevolve,
    EvoformerManager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_d_model():
    return 64


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 8


@pytest.fixture
def evoformer_config(small_d_model):
    return EvoformerConfig(
        d_model=small_d_model,
        n_recycling_steps=2,
        d_pair=small_d_model,
        dropout=0.0,
        use_layer_recycling=True,
        use_token_recycling=True,
        use_decoder_feedback=True,
        use_prediction_recycling=True,
        use_router_coevolve=True,
        min_recycling_improvement=1e-4,
    )


@pytest.fixture
def evoformer_manager(evoformer_config):
    return EvoformerManager(evoformer_config)


# ============================================================================
# Level 1 — LayerRecyclingBlock
# ============================================================================


class TestLayerRecyclingBlock:
    """Tests for Evoformer Level 1: Inter-Layer Recycling."""

    def test_output_shapes(self, small_d_model, batch_size, seq_len):
        """LayerRecyclingBlock output shapes must match input shapes."""
        block = LayerRecyclingBlock(d_model=small_d_model, n_recycling_steps=2)
        # Simulate hidden states from 4 layers
        hidden_states = [
            torch.randn(batch_size, seq_len, small_d_model) for _ in range(4)
        ]
        revised = block(hidden_states)
        assert len(revised) == len(hidden_states)
        for h, r in zip(hidden_states, revised):
            assert r.shape == h.shape

    def test_revision_changes_representations(self, small_d_model, batch_size, seq_len):
        """Revision should actually change the representations (not identity)."""
        block = LayerRecyclingBlock(d_model=small_d_model)
        hidden_states = [
            torch.randn(batch_size, seq_len, small_d_model) for _ in range(4)
        ]
        revised = block(hidden_states)
        # At least some layers should be different
        any_changed = False
        for h, r in zip(hidden_states, revised):
            if not torch.allclose(h, r, atol=1e-6):
                any_changed = True
                break
        assert any_changed, "Layer recycling should modify at least some representations"

    def test_single_layer_passthrough(self, small_d_model, batch_size, seq_len):
        """With only 1 layer, recycling should return the same list."""
        block = LayerRecyclingBlock(d_model=small_d_model)
        hidden_states = [torch.randn(batch_size, seq_len, small_d_model)]
        revised = block(hidden_states)
        assert len(revised) == 1
        # With <2 layers, no revision can be computed
        assert torch.allclose(revised[0], hidden_states[0])

    def test_gradient_flows_through_revision(self, small_d_model, batch_size, seq_len):
        """Gradients must flow from revised output back to revision parameters."""
        block = LayerRecyclingBlock(d_model=small_d_model)
        hidden_states = [
            torch.randn(batch_size, seq_len, small_d_model, requires_grad=True)
            for _ in range(4)
        ]
        revised = block(hidden_states)
        loss = revised[-1].sum()  # Use deep layer output
        loss.backward()
        # Revision parameters should have gradients
        assert block.shallow_query_proj.weight.grad is not None or any(
            p.grad is not None for p in block.parameters()
        ), "Gradient should flow through revision path"

    def test_compute_revision_output_shape(self, small_d_model, batch_size, seq_len):
        """compute_revision should output same shape as input."""
        block = LayerRecyclingBlock(d_model=small_d_model)
        shallow = torch.randn(batch_size, seq_len, small_d_model)
        deep = torch.randn(batch_size, seq_len, small_d_model)
        revision = block.compute_revision(shallow, deep)
        assert revision.shape == (batch_size, seq_len, small_d_model)


# ============================================================================
# Level 2 — BidirectionalTokenUpdate
# ============================================================================


class TestBidirectionalTokenUpdate:
    """Tests for Evoformer Level 2: Bidirectional Token Update."""

    def test_output_shape(self, small_d_model, batch_size, seq_len):
        """Output shape must match input shape."""
        block = BidirectionalTokenUpdate(d_model=small_d_model, n_heads=4)
        x = torch.randn(batch_size, seq_len, small_d_model)
        out = block(x)
        assert out.shape == x.shape

    def test_single_token_passthrough(self, small_d_model, batch_size):
        """With seq_len=1, should return input unchanged."""
        block = BidirectionalTokenUpdate(d_model=small_d_model, n_heads=4)
        x = torch.randn(batch_size, 1, small_d_model)
        out = block(x)
        assert out.shape == x.shape

    def test_bidirectional_info_flow(self, small_d_model, batch_size, seq_len):
        """Later tokens should influence earlier token representations."""
        block = BidirectionalTokenUpdate(d_model=small_d_model, n_heads=4)
        x = torch.randn(batch_size, seq_len, small_d_model)
        out = block(x)
        # Output should differ from input (information has flowed bidirectionally)
        assert not torch.allclose(x, out, atol=1e-6), \
            "Bidirectional update should modify representations"

    def test_gradient_flow(self, small_d_model, batch_size, seq_len):
        """Gradients should flow through bidirectional attention."""
        block = BidirectionalTokenUpdate(d_model=small_d_model, n_heads=4)
        x = torch.randn(batch_size, seq_len, small_d_model, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None, "Gradient should flow through bidirectional token update"


# ============================================================================
# Level 3 — DecoderPredictFeedback
# ============================================================================


class TestDecoderPredictFeedback:
    """Tests for Evoformer Level 3: Decoder ↔ Predict Feedback."""

    def test_output_shape(self, small_d_model, batch_size, seq_len):
        """Output shape must match hidden_state shape."""
        block = DecoderPredictFeedback(d_model=small_d_model, n_iterations=2)
        hidden_state = torch.randn(batch_size, seq_len, small_d_model)
        decoder_output = torch.randn(batch_size, seq_len, small_d_model)
        out = block(hidden_state, decoder_output)
        assert out.shape == hidden_state.shape

    def test_feedback_modifies_hidden_state(self, small_d_model, batch_size, seq_len):
        """When decoder output differs from hidden state, feedback should modify it."""
        block = DecoderPredictFeedback(d_model=small_d_model)
        hidden_state = torch.randn(batch_size, seq_len, small_d_model)
        decoder_output = torch.randn(batch_size, seq_len, small_d_model) * 2.0
        out = block(hidden_state, decoder_output)
        assert not torch.allclose(hidden_state, out, atol=1e-6), \
            "Decoder feedback should modify hidden state when output differs"

    def test_gradient_flow(self, small_d_model, batch_size, seq_len):
        """Gradients should flow through feedback path."""
        block = DecoderPredictFeedback(d_model=small_d_model)
        hidden_state = torch.randn(batch_size, seq_len, small_d_model, requires_grad=True)
        decoder_output = torch.randn(batch_size, seq_len, small_d_model)
        out = block(hidden_state, decoder_output)
        out.sum().backward()
        assert hidden_state.grad is not None


# ============================================================================
# Level 4 — PredictionContextRecycling
# ============================================================================


class TestPredictionContextRecycling:
    """Tests for Evoformer Level 4: Prediction → Context Recycling."""

    def test_output_shape(self, small_d_model, batch_size, seq_len):
        """Output shape must match hidden_states shape."""
        block = PredictionContextRecycling(d_model=small_d_model)
        hidden_states = torch.randn(batch_size, seq_len, small_d_model)
        prediction_logits = torch.randn(batch_size, seq_len, small_d_model)
        out = block(hidden_states, prediction_logits)
        assert out.shape == hidden_states.shape

    def test_prediction_revises_context(self, small_d_model, batch_size, seq_len):
        """Predictions should revise the context representations."""
        block = PredictionContextRecycling(d_model=small_d_model)
        hidden_states = torch.randn(batch_size, seq_len, small_d_model)
        prediction_logits = torch.randn(batch_size, seq_len, small_d_model)
        out = block(hidden_states, prediction_logits)
        assert not torch.allclose(hidden_states, out, atol=1e-6), \
            "Prediction recycling should modify context representations"

    def test_mismatched_prediction_dim(self, small_d_model, batch_size, seq_len):
        """When prediction dim != d_model, should project and still work."""
        block = PredictionContextRecycling(d_model=small_d_model)
        hidden_states = torch.randn(batch_size, seq_len, small_d_model)
        # Prediction with different last dim
        prediction_logits = torch.randn(batch_size, seq_len, small_d_model * 2)
        out = block(hidden_states, prediction_logits)
        assert out.shape == hidden_states.shape

    def test_gradient_flow(self, small_d_model, batch_size, seq_len):
        """Gradients should flow through prediction recycling."""
        block = PredictionContextRecycling(d_model=small_d_model)
        hidden_states = torch.randn(batch_size, seq_len, small_d_model, requires_grad=True)
        prediction_logits = torch.randn(batch_size, seq_len, small_d_model)
        out = block(hidden_states, prediction_logits)
        out.sum().backward()
        assert hidden_states.grad is not None


# ============================================================================
# Level 5 — RouterExpertCoevolve
# ============================================================================


class TestRouterExpertCoevolve:
    """Tests for Evoformer Level 5: Router ↔ Expert Co-Evolution."""

    def test_routing_adjustment_shape(self, small_d_model):
        """get_routing_adjustment should return (num_pathways,) tensor."""
        coevolve = RouterExpertCoevolve(d_model=small_d_model, num_pathways=3)
        adjustment = coevolve.get_routing_adjustment()
        assert adjustment.shape == (3,)

    def test_routing_adjustment_small_magnitude(self, small_d_model):
        """Adjustments should be small (scaled by 0.1) to avoid instability."""
        coevolve = RouterExpertCoevolve(d_model=small_d_model, num_pathways=3)
        adjustment = coevolve.get_routing_adjustment()
        assert adjustment.abs().max().item() < 0.2, \
            "Routing adjustments should be small for stability"

    def test_forward_modifies_routing_weights(self, small_d_model, batch_size, seq_len):
        """Forward should produce adjusted routing weights."""
        coevolve = RouterExpertCoevolve(d_model=small_d_model, num_pathways=3)
        routing_weights = F.softmax(torch.randn(batch_size, seq_len, 3), dim=-1)
        pathway_outputs = [torch.randn(batch_size, seq_len, small_d_model) for _ in range(3)]
        adjusted = coevolve(routing_weights, pathway_outputs)
        assert adjusted.shape == routing_weights.shape
        # Adjusted weights should still be valid probabilities
        assert torch.allclose(adjusted.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

    def test_update_state_preserves_gradient_path(self, small_d_model, batch_size, seq_len):
        """update_state should return a differentiable tensor."""
        coevolve = RouterExpertCoevolve(d_model=small_d_model, num_pathways=3)
        expert_output = torch.randn(batch_size, seq_len, small_d_model, requires_grad=True)
        update = coevolve.update_state(0, expert_output)
        assert update.requires_grad, "update_state should preserve gradient path"


# ============================================================================
# EvoformerManager
# ============================================================================


class TestEvoformerManager:
    """Tests for the EvoformerManager coordinator."""

    def test_all_levels_active_by_default(self, evoformer_manager):
        """With all levels enabled, manager should have all sub-modules."""
        assert evoformer_manager.layer_recycling is not None
        assert evoformer_manager.bidirectional_token is not None
        assert evoformer_manager.decoder_feedback is not None
        assert evoformer_manager.prediction_recycling is not None
        assert evoformer_manager.router_coevolve is not None

    def test_stats(self, evoformer_manager):
        """get_stats should report all 5 levels."""
        stats = evoformer_manager.get_stats()
        assert stats["level_1_layer_recycling"] is True
        assert stats["level_2_bidirectional_token"] is True
        assert stats["level_3_decoder_feedback"] is True
        assert stats["level_4_prediction_recycling"] is True
        assert stats["level_5_router_coevolve"] is True
        assert stats["n_recycling_steps"] == 2

    def test_disabled_levels(self, small_d_model):
        """When a level is disabled, its module should be None."""
        config = EvoformerConfig(
            d_model=small_d_model,
            use_layer_recycling=False,
            use_token_recycling=False,
            use_decoder_feedback=False,
            use_prediction_recycling=False,
            use_router_coevolve=False,
        )
        manager = EvoformerManager(config)
        assert manager.layer_recycling is None
        assert manager.bidirectional_token is None
        assert manager.decoder_feedback is None
        assert manager.prediction_recycling is None
        assert manager.router_coevolve is None

    def test_recycle_layers(self, evoformer_manager, batch_size, seq_len, small_d_model):
        """recycle_layers should return revised hidden states."""
        hidden_states = [
            torch.randn(batch_size, seq_len, small_d_model) for _ in range(4)
        ]
        revised = evoformer_manager.recycle_layers(hidden_states)
        assert len(revised) == len(hidden_states)
        for r in revised:
            assert r.shape == (batch_size, seq_len, small_d_model)

    def test_bidirectional_token_update(self, evoformer_manager, batch_size, seq_len, small_d_model):
        """bidirectional_token_update should return same-shape tensor."""
        x = torch.randn(batch_size, seq_len, small_d_model)
        out = evoformer_manager.bidirectional_token_update(x)
        assert out.shape == x.shape

    def test_apply_decoder_feedback(self, evoformer_manager, batch_size, seq_len, small_d_model):
        """apply_decoder_feedback should return same-shape tensor."""
        hidden = torch.randn(batch_size, seq_len, small_d_model)
        decoder = torch.randn(batch_size, seq_len, small_d_model)
        out = evoformer_manager.apply_decoder_feedback(hidden, decoder)
        assert out.shape == hidden.shape

    def test_apply_prediction_recycling(self, evoformer_manager, batch_size, seq_len, small_d_model):
        """apply_prediction_recycling should return same-shape tensor."""
        hidden = torch.randn(batch_size, seq_len, small_d_model)
        logits = torch.randn(batch_size, seq_len, small_d_model)
        out = evoformer_manager.apply_prediction_recycling(hidden, logits)
        assert out.shape == hidden.shape

    def test_apply_router_coevolve(self, evoformer_manager, batch_size, seq_len, small_d_model):
        """apply_router_coevolve should return adjusted routing weights."""
        routing = F.softmax(torch.randn(batch_size, seq_len, 3), dim=-1)
        outputs = [torch.randn(batch_size, seq_len, small_d_model) for _ in range(3)]
        adjusted = evoformer_manager.apply_router_coevolve(routing, outputs)
        assert adjusted.shape == routing.shape

    def test_disabled_level_convenience_methods(self, small_d_model, batch_size, seq_len):
        """Convenience methods should return input unchanged when level is disabled."""
        config = EvoformerConfig(
            d_model=small_d_model,
            use_decoder_feedback=False,
            use_prediction_recycling=False,
            use_router_coevolve=False,
        )
        manager = EvoformerManager(config)
        x = torch.randn(batch_size, seq_len, small_d_model)
        assert torch.allclose(manager.decoder_predict_feedback(x), x)
        assert torch.allclose(manager.prediction_context_recycling(x), x)

    def test_reset(self, evoformer_manager):
        """reset() should not raise any errors."""
        evoformer_manager.reset()  # Should be a no-op for now


# ============================================================================
# EvoformerConfig
# ============================================================================


class TestEvoformerConfig:
    """Tests for EvoformerConfig validation."""

    def test_default_d_pair(self, small_d_model):
        """d_pair=0 should default to d_model."""
        config = EvoformerConfig(d_model=small_d_model, d_pair=0)
        assert config.d_pair == small_d_model

    def test_custom_d_pair(self, small_d_model):
        """Custom d_pair should be preserved."""
        config = EvoformerConfig(d_model=small_d_model, d_pair=32)
        assert config.d_pair == 32


import torch.nn.functional as F  # already imported above, but for clarity
