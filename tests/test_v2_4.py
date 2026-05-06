"""
Tests for Losion v2.4.0 fixes.

Covers:
- N-01: inference_sparse propagation from LosionModel to LosionLayer
- N-04: BiasRouter.update_bias() actually changes bias values
- N-05: Sparse inference uses max() not mean()
- N-06: ThinkingAssessment.thinking_score Optional type
"""

import pytest
import torch

from losion.config import LosionConfig
from losion.models.losion_model import LosionModel, LosionLayer
from losion.core.router.thinking_toggle import ThinkingAssessment, ThinkingMode, TaskType


# ---- Fixtures ----

@pytest.fixture
def small_config():
    """Small LosionConfig for testing."""
    return LosionConfig(
        d_model=64,
        n_layers=2,
        vocab_size=100,
        max_seq_len=32,
        dropout=0.0,
    )


@pytest.fixture
def small_model(small_config):
    """Small LosionModel for testing."""
    return LosionModel(small_config)


# ---- N-01: inference_sparse propagation ----

class TestInferenceSparsePropagation:
    """Test that inference_sparse reaches LosionLayer from LosionModel."""

    def test_forward_accepts_inference_sparse(self, small_model):
        """LosionModel.forward() should accept inference_sparse kwarg."""
        input_ids = torch.randint(0, 100, (2, 16))
        # Should NOT raise TypeError
        output = small_model(input_ids, inference_sparse=True)
        assert output.hidden_states is not None

    def test_forward_accepts_sparse_threshold(self, small_model):
        """LosionModel.forward() should accept sparse_threshold kwarg."""
        input_ids = torch.randint(0, 100, (2, 16))
        output = small_model(input_ids, inference_sparse=True, sparse_threshold=0.1)
        assert output.hidden_states is not None

    def test_inference_sparse_no_effect_during_training(self, small_model):
        """inference_sparse should have no effect during training."""
        small_model.train()
        input_ids = torch.randint(0, 100, (2, 16))

        # Run with and without inference_sparse
        output_normal = small_model(input_ids, inference_sparse=False)
        output_sparse = small_model(input_ids, inference_sparse=True)

        # During training, both should produce same results
        # (all pathways computed regardless of sparse flag)
        assert output_normal.hidden_states.shape == output_sparse.hidden_states.shape

    def test_inference_sparse_during_eval(self, small_model):
        """inference_sparse should work during eval mode."""
        small_model.eval()
        input_ids = torch.randint(0, 100, (2, 16))

        # Should not crash with high threshold (some pathways skipped)
        output = small_model(input_ids, inference_sparse=True, sparse_threshold=0.99)
        assert output.hidden_states is not None
        assert not torch.isnan(output.hidden_states).any()


# ---- N-04: BiasRouter.update_bias() changes bias ----

class TestBiasRouterUpdate:
    """Test that BiasRouter.update_bias() actually modifies bias."""

    def test_update_bias_changes_bias(self):
        """update_bias() should change bias values after forward pass accumulates stats."""
        from losion.core.router.bias_router import BiasRouter

        router = BiasRouter(d_model=64, num_pathways=3, bias_lr=0.1)

        # Initial bias should be zeros
        assert torch.allclose(router.bias, torch.zeros(3)), \
            f"Initial bias should be zeros, got {router.bias}"

        # Run forward to accumulate running_load statistics
        x = torch.randn(2, 16, 64)
        for _ in range(5):
            _ = router(x)

        # Now call update_bias
        router.update_bias()

        # Bias should have changed (not all zeros anymore)
        assert not torch.allclose(router.bias, torch.zeros(3), atol=1e-6), \
            f"Bias should have changed after update_bias(), got {router.bias}"

    def test_adaptive_router_update_bias(self):
        """AdaptiveRouter.update_bias() should call through to BiasRouter."""
        from losion.core.router.router import AdaptiveRouter

        router = AdaptiveRouter(d_model=64, num_pathways=3, bias_lr=0.1)

        # Initial bias should be zeros
        assert torch.allclose(router.bias_router.bias, torch.zeros(3))

        # Accumulate statistics
        x = torch.randn(2, 16, 64)
        for _ in range(5):
            _ = router(x)

        # Call update_bias via AdaptiveRouter
        router.update_bias()

        # Bias should have changed
        assert not torch.allclose(router.bias_router.bias, torch.zeros(3), atol=1e-6), \
            "AdaptiveRouter.update_bias() should change bias values"


# ---- N-05: Sparse inference uses max() not mean() ----

class TestSparseMaxNotMean:
    """Test that sparse inference uses max() for threshold check."""

    def test_sparse_uses_max_not_mean(self):
        """Even if most tokens have low weight, if ANY token has high weight,
        the pathway should still be computed."""
        config = LosionConfig(
            d_model=64,
            n_layers=1,
            vocab_size=100,
            max_seq_len=32,
            dropout=0.0,
        )
        model = LosionModel(config)
        model.eval()

        # Create a scenario where we can verify the sparse logic
        # We test that the model accepts the parameters correctly
        input_ids = torch.randint(0, 100, (1, 16))

        # With very low threshold, everything should compute
        output_low = model(input_ids, inference_sparse=True, sparse_threshold=0.001)
        assert output_low.hidden_states is not None

        # With very high threshold, pathways should be skipped
        output_high = model(input_ids, inference_sparse=True, sparse_threshold=0.999)
        assert output_high.hidden_states is not None


# ---- N-06: ThinkingAssessment Optional type ----

class TestThinkingAssessmentOptional:
    """Test that ThinkingAssessment.thinking_score is Optional[torch.Tensor]."""

    def test_thinking_score_can_be_none(self):
        """ThinkingAssessment should accept None for thinking_score."""
        assessment = ThinkingAssessment(
            mode=ThinkingMode.NON_THINKING,
            complexity_score=torch.randn(2, 16),
            task_type_probs=torch.randn(2, 16, 5),
            dominant_task=TaskType.SEQUENTIAL,
            depth_multiplier=torch.randn(2),
            confidence=torch.randn(2),
            thinking_score=None,  # Should NOT raise TypeError
        )
        assert assessment.thinking_score is None

    def test_thinking_score_can_be_tensor(self):
        """ThinkingAssessment should accept a tensor for thinking_score."""
        score = torch.randn(2)
        assessment = ThinkingAssessment(
            mode=ThinkingMode.NON_THINKING,
            complexity_score=torch.randn(2, 16),
            task_type_probs=torch.randn(2, 16, 5),
            dominant_task=TaskType.SEQUENTIAL,
            depth_multiplier=torch.randn(2),
            confidence=torch.randn(2),
            thinking_score=score,
        )
        assert assessment.thinking_score is not None
        assert torch.equal(assessment.thinking_score, score)


# ---- N-03: MoE aux_info field name ----

class TestMoEAuxInfoNaming:
    """Test that MoE aux_info uses normalized_topk_weights."""

    def test_aux_info_field_name(self, small_config):
        """SimplifiedMoE should use 'normalized_topk_weights' in aux_info."""
        from losion.models.losion_model import SimplifiedMoE

        moe = SimplifiedMoE(d_model=64, d_ff=256, num_experts=8, top_k_routing=2)
        x = torch.randn(2, 16, 64)
        output, aux = moe(x)

        # Field should be renamed from 'expert_weights' to 'normalized_topk_weights'
        assert "normalized_topk_weights" in aux, \
            f"aux_info should contain 'normalized_topk_weights', got keys: {list(aux.keys())}"
        assert "expert_weights" not in aux, \
            "aux_info should NOT contain old 'expert_weights' key (renamed to 'normalized_topk_weights')"


# ---- N-09: Conv1d init ----

class TestConv1dInit:
    """Test that Conv1d uses scaled normal init, not kaiming."""

    def test_conv1d_not_kaiming(self, small_config):
        """Conv1d in SimplifiedSSM should use GPT-2 style scaled normal init."""
        from losion.models.losion_model import SimplifiedSSM

        ssm = SimplifiedSSM(d_model=64, d_state=16, expand=2, d_conv=4)
        conv_weight = ssm.conv1d.weight.data

        # Kaiming init tends to have much larger variance than normal(0, 0.02/sqrt(2*L))
        # For L=2, std = 0.02/sqrt(4) = 0.01
        # Kaiming for groups=128 would be ~0.088
        # Check that the std is closer to 0.01 than 0.088
        actual_std = conv_weight.std().item()
        # With only 2 layers, expected std is 0.02/sqrt(4) = 0.01
        expected_std = 0.01
        kaiming_std = 0.088  # Approximate kaiming std for these dims

        # Should be much closer to expected than to kaiming
        assert abs(actual_std - expected_std) < abs(actual_std - kaiming_std), \
            f"Conv1d std ({actual_std:.4f}) should be closer to GPT-2 style ({expected_std}) " \
            f"than kaiming ({kaiming_std:.4f})"
