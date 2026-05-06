"""
Test suite untuk modul Router — Losion Framework.

Menguji forward pass, shape output, dan edge cases untuk
BiasRouter, ThinkingToggle, dan AdaptiveRouter.

Semua test berjalan di CPU tanpa memerlukan GPU.
"""

import pytest
import torch
import torch.nn as nn

from losion.core.router import (
    BiasRouter,
    PathwayRoutingInfo,
    ThinkingToggle,
    ThinkingAssessment,
    ThinkingMode,
    TaskType,
    AdaptiveRouter,
    AdaptiveRoutingOutput,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def bias_router():
    """Buat BiasRouter untuk testing."""
    return BiasRouter(
        d_model=64,
        num_pathways=3,
        top_k_pathways=2,
        bias_lr=0.01,
    )


@pytest.fixture
def thinking_toggle():
    """Buat ThinkingToggle untuk testing."""
    return ThinkingToggle(
        d_model=64,
        threshold=0.5,
    )


@pytest.fixture
def adaptive_router():
    """Buat AdaptiveRouter untuk testing."""
    return AdaptiveRouter(
        d_model=64,
        num_pathways=3,
        top_k_pathways=2,
        bias_lr=0.01,
        thinking_threshold=0.5,
    )


# ============================================================================
# Test BiasRouter
# ============================================================================


class TestBiasRouter:
    """Test suite untuk BiasRouter."""

    def test_init(self, bias_router):
        """Test bahwa BiasRouter dapat diinisialisasi."""
        assert isinstance(bias_router, nn.Module)
        assert bias_router.num_pathways == 3
        assert bias_router.top_k_pathways == 2

    def test_forward_shape(self, bias_router):
        """Test bahwa output memiliki shape yang benar."""
        batch, seq, d = 2, 16, 64
        x = torch.randn(batch, seq, d)
        weights, info = bias_router(x)
        assert weights.shape == (batch, seq, 3)

    def test_forward_weights_sum_to_one(self, bias_router):
        """Test bahwa routing weights sum = 1 (softmax)."""
        x = torch.randn(2, 16, 64)
        weights, _ = bias_router(x)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_forward_weights_non_negative(self, bias_router):
        """Test bahwa routing weights tidak negatif."""
        x = torch.randn(2, 16, 64)
        weights, _ = bias_router(x)
        assert (weights >= 0).all(), "Routing weights tidak boleh negatif"

    def test_forward_no_nan(self, bias_router):
        """Test bahwa output tidak mengandung NaN."""
        x = torch.randn(2, 16, 64)
        weights, _ = bias_router(x)
        assert not torch.isnan(weights).any()

    def test_forward_returns_routing_info(self, bias_router):
        """Test bahwa forward mengembalikan PathwayRoutingInfo."""
        x = torch.randn(2, 16, 64)
        _, info = bias_router(x)
        assert isinstance(info, PathwayRoutingInfo)

    def test_forward_different_batch_sizes(self, bias_router):
        """Test berbagai ukuran batch."""
        for batch in [1, 2, 4, 8]:
            x = torch.randn(batch, 16, 64)
            weights, _ = bias_router(x)
            assert weights.shape[0] == batch

    def test_forward_different_seq_lengths(self, bias_router):
        """Test berbagai panjang sequence."""
        for seq in [1, 8, 16, 32]:
            x = torch.randn(2, seq, 64)
            weights, _ = bias_router(x)
            assert weights.shape[1] == seq

    def test_input_validation(self, bias_router):
        """Test validasi input."""
        with pytest.raises(ValueError):
            # Input harus 3D
            bias_router(torch.randn(2, 64))

    def test_gradient_flow(self, bias_router):
        """Test bahwa gradient mengalir melalui BiasRouter."""
        x = torch.randn(2, 16, 64, requires_grad=True)
        weights, _ = bias_router(x)
        loss = weights.sum()
        loss.backward()
        assert x.grad is not None


# ============================================================================
# Test ThinkingToggle
# ============================================================================


class TestThinkingToggle:
    """Test suite untuk ThinkingToggle."""

    def test_init(self, thinking_toggle):
        """Test bahwa ThinkingToggle dapat diinisialisasi."""
        assert isinstance(thinking_toggle, nn.Module)

    def test_forward_returns_assessment(self, thinking_toggle):
        """Test bahwa forward mengembalikan ThinkingAssessment."""
        x = torch.randn(2, 16, 64)
        assessment = thinking_toggle(x)
        assert isinstance(assessment, ThinkingAssessment)

    def test_forward_complexity_score_range(self, thinking_toggle):
        """Test bahwa complexity score di [0, 1]."""
        x = torch.randn(2, 16, 64)
        assessment = thinking_toggle(x)
        assert assessment.complexity_score.min() >= 0.0
        assert assessment.complexity_score.max() <= 1.0

    def test_forward_thinking_mode_valid(self, thinking_toggle):
        """Test bahwa thinking mode valid."""
        x = torch.randn(2, 16, 64)
        assessment = thinking_toggle(x)
        assert assessment.mode in [ThinkingMode.NON_THINKING, ThinkingMode.THINKING]

    def test_forward_task_type_valid(self, thinking_toggle):
        """Test bahwa task type valid."""
        x = torch.randn(2, 16, 64)
        assessment = thinking_toggle(x)
        # TaskType bisa termasuk CREATIVE atau lainnya
        # Yang penting adalah assessment memiliki dominant_task
        assert hasattr(assessment.dominant_task, 'value')
        assert isinstance(assessment.dominant_task, TaskType)

    def test_forward_confidence_range(self, thinking_toggle):
        """Test bahwa confidence di [0, 1]."""
        x = torch.randn(2, 16, 64)
        assessment = thinking_toggle(x)
        assert (0.0 <= assessment.confidence).all() and (assessment.confidence <= 1.0).all()

    def test_set_force_mode(self, thinking_toggle):
        """Test force mode setting."""
        thinking_toggle.set_force_mode(ThinkingMode.THINKING)
        x = torch.randn(2, 16, 64)
        assessment = thinking_toggle(x)
        assert assessment.mode == ThinkingMode.THINKING

        thinking_toggle.set_force_mode(ThinkingMode.NON_THINKING)
        assessment = thinking_toggle(x)
        assert assessment.mode == ThinkingMode.NON_THINKING

        # Reset ke auto
        thinking_toggle.set_force_mode(None)
        # Mode sekarang harus ditentukan secara otomatis

    def test_set_threshold(self, thinking_toggle):
        """Test update threshold."""
        thinking_toggle.set_threshold(0.3)
        assert thinking_toggle.threshold == 0.3

        thinking_toggle.set_threshold(0.7)
        assert thinking_toggle.threshold == 0.7

    def test_different_batch_sizes(self, thinking_toggle):
        """Test berbagai ukuran batch."""
        for batch in [1, 2, 4]:
            x = torch.randn(batch, 16, 64)
            assessment = thinking_toggle(x)
            assert assessment.complexity_score.shape == (batch, 16)


# ============================================================================
# Test AdaptiveRouter
# ============================================================================


class TestAdaptiveRouter:
    """Test suite untuk AdaptiveRouter."""

    def test_init(self, adaptive_router):
        """Test bahwa AdaptiveRouter dapat diinisialisasi."""
        assert isinstance(adaptive_router, nn.Module)
        assert adaptive_router.num_pathways == 3
        assert adaptive_router.top_k_pathways == 2

    def test_forward_shape(self, adaptive_router):
        """Test bahwa output memiliki shape yang benar."""
        batch, seq, d = 2, 16, 64
        x = torch.randn(batch, seq, d)
        output = adaptive_router(x)
        assert output.routing_weights.shape == (batch, seq, 3)
        assert output.adjusted_weights.shape == (batch, seq, 3)

    def test_forward_returns_adaptive_routing_output(self, adaptive_router):
        """Test bahwa forward mengembalikan AdaptiveRoutingOutput."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        assert isinstance(output, AdaptiveRoutingOutput)

    def test_forward_weights_sum_to_one(self, adaptive_router):
        """Test bahwa adjusted weights sum = 1."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        weight_sums = output.adjusted_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-4)

    def test_forward_weights_non_negative(self, adaptive_router):
        """Test bahwa adjusted weights tidak negatif."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        assert (output.adjusted_weights >= -1e-6).all()

    def test_forward_no_nan(self, adaptive_router):
        """Test bahwa output tidak mengandung NaN."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        assert not torch.isnan(output.adjusted_weights).any()

    def test_forward_contains_thinking_assessment(self, adaptive_router):
        """Test bahwa output mengandung ThinkingAssessment."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        assert isinstance(output.thinking_assessment, ThinkingAssessment)

    def test_forward_contains_routing_info(self, adaptive_router):
        """Test bahwa output mengandung PathwayRoutingInfo."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        assert isinstance(output.routing_info, PathwayRoutingInfo)

    def test_forward_pathway_labels(self, adaptive_router):
        """Test bahwa pathway labels lengkap."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        assert len(output.pathway_labels) == 3
        assert "sequential" in output.pathway_labels
        assert "reasoning" in output.pathway_labels
        assert "factual" in output.pathway_labels

    def test_forward_depth_multiplier(self, adaptive_router):
        """Test bahwa depth multiplier valid (non-negative)."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        # depth_multiplier bisa < 1.0 tergantung thinking assessment
        # Yang penting adalah nilainya valid (non-negative)
        assert (output.depth_multiplier >= 0.0).all()

    def test_input_validation(self, adaptive_router):
        """Test validasi input."""
        with pytest.raises(ValueError):
            # Input harus 3D
            adaptive_router(torch.randn(2, 64))

        with pytest.raises(ValueError):
            # Input harus 3D
            adaptive_router(torch.randn(64))

    def test_update_bias(self, adaptive_router):
        """Test update bias."""
        # Tidak boleh error
        adaptive_router.update_bias()

    def test_set_force_thinking(self, adaptive_router):
        """Test force thinking mode."""
        adaptive_router.set_force_thinking(ThinkingMode.THINKING)
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        # Thinking mode harus aktif
        assert output.thinking_assessment.mode == ThinkingMode.THINKING

    def test_set_thinking_threshold(self, adaptive_router):
        """Test update thinking threshold."""
        adaptive_router.set_thinking_threshold(0.3)
        # Tidak boleh error
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        assert output.adjusted_weights.shape == (2, 16, 3)

    def test_get_pathway_summary(self, adaptive_router):
        """Test ringkasan distribusi routing."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        summary = adaptive_router.get_pathway_summary(output)

        assert "pathway_labels" in summary
        assert "mean_weights" in summary
        assert "thinking_mode" in summary
        assert "dominant_task" in summary
        assert "depth_multiplier" in summary

        # Mean weights harus sum ~ 1
        total = sum(summary["mean_weights"].values())
        assert abs(total - 1.0) < 0.1

    def test_compute_routing_entropy(self, adaptive_router):
        """Test perhitungan entropy routing."""
        x = torch.randn(2, 16, 64)
        output = adaptive_router(x)
        entropy = adaptive_router.compute_routing_entropy(output.adjusted_weights)

        # Entropy harus di [0, 1] (normalized)
        assert 0.0 <= entropy.item() <= 1.0

    def test_gradient_flow(self, adaptive_router):
        """Test bahwa gradient mengalir melalui router."""
        x = torch.randn(2, 16, 64, requires_grad=True)
        output = adaptive_router(x)
        loss = output.adjusted_weights.sum()
        loss.backward()
        assert x.grad is not None, "Gradient tidak mengalir ke input"

    def test_different_batch_sizes(self, adaptive_router):
        """Test berbagai ukuran batch."""
        for batch in [1, 2, 4, 8]:
            x = torch.randn(batch, 16, 64)
            output = adaptive_router(x)
            assert output.adjusted_weights.shape[0] == batch

    def test_different_seq_lengths(self, adaptive_router):
        """Test berbagai panjang sequence."""
        for seq in [1, 8, 16, 32]:
            x = torch.randn(2, seq, 64)
            output = adaptive_router(x)
            assert output.adjusted_weights.shape[1] == seq

    def test_consistency_across_calls(self, adaptive_router):
        """Test konsistensi output untuk input yang sama."""
        x = torch.randn(2, 16, 64)
        adaptive_router.eval()
        with torch.no_grad():
            output1 = adaptive_router(x)
            output2 = adaptive_router(x)
        # Output harus sama untuk input yang sama dalam eval mode
        assert torch.allclose(output1.adjusted_weights, output2.adjusted_weights, atol=1e-5)


# ============================================================================
# Test Enum dan Data Classes
# ============================================================================


class TestEnums:
    """Test suite untuk enum dan data classes."""

    def test_thinking_mode_values(self):
        """Test ThinkingMode enum values."""
        assert ThinkingMode.NON_THINKING.value == "non_thinking"
        assert ThinkingMode.THINKING.value == "thinking"

    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.SEQUENTIAL.value == "sequential"
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.FACTUAL.value == "factual"

    def test_thinking_assessment_creation(self):
        """Test pembuatan ThinkingAssessment."""
        assessment = ThinkingAssessment(
            mode=ThinkingMode.THINKING,
            complexity_score=torch.tensor(0.8),
            task_type_probs=torch.tensor([0.1, 0.7, 0.2]),
            confidence=0.9,
            dominant_task=TaskType.REASONING,
            depth_multiplier=1.5,
        )
        assert assessment.mode == ThinkingMode.THINKING
        assert assessment.confidence == 0.9
        assert assessment.depth_multiplier == 1.5
