"""
Test suite untuk modul Attention (Jalur 2) — Losion Framework.

Menguji forward pass, shape output, dan edge cases untuk
komponen Attention: MLA, iRoPE, AdaptiveInterleaving,
dan AttentionKompresiLayer.

Semua test berjalan di CPU tanpa memerlukan GPU.
"""

import pytest
import torch
import torch.nn as nn

from losion.core.attention import (
    MLA,
    MLAKVCache,
    InterleavedRoPE,
    AdaptiveInterleaving,
    AttentionKompresiLayer,
    AttentionState,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_mla():
    """Buat MLA layer kecil untuk testing."""
    return MLA(
        d_model=64,
        n_heads=4,
        d_kv=16,
        mla_latent_dim=32,
    )


@pytest.fixture
def small_irope():
    """Buat InterleavedRoPE untuk testing."""
    return InterleavedRoPE(
        dim=16,
        base=10000.0,
        ratio=3,
    )


@pytest.fixture
def small_attention_layer():
    """Buat AttentionKompresiLayer kecil untuk testing."""
    return AttentionKompresiLayer(
        d_model=64,
        n_heads=4,
        d_kv=16,
        mla_latent_dim=32,
        irope_ratio=3,
        base_interleaving_ratio=5,
        thinking_interleaving_ratio=2,
        sliding_window_size=32,
        ffn_dim_multiplier=4,
    )


# ============================================================================
# Test MLA
# ============================================================================


class TestMLA:
    """Test suite untuk MLA (Multi-head Latent Attention)."""

    def test_init(self, small_mla):
        """Test bahwa MLA dapat diinisialisasi."""
        assert isinstance(small_mla, nn.Module)

    def test_forward_shape(self, small_mla):
        """Test bahwa output MLA memiliki shape yang benar."""
        batch, seq, d = 2, 16, 64
        x = torch.randn(batch, seq, d)
        output, kv_latent, kv_cache = small_mla(x)
        assert output.shape == (batch, seq, d)

    def test_forward_no_nan(self, small_mla):
        """Test bahwa output tidak mengandung NaN."""
        x = torch.randn(2, 16, 64)
        output, _, _ = small_mla(x)
        assert not torch.isnan(output).any(), "MLA output mengandung NaN"

    def test_kv_latent_shape(self, small_mla):
        """Test bahwa KV latent memiliki shape yang benar."""
        x = torch.randn(2, 16, 64)
        _, kv_latent, _ = small_mla(x)
        assert kv_latent.shape == (2, 16, 32)  # mla_latent_dim=32

    def test_memory_savings(self, small_mla):
        """Test bahwa MLA mengkompresi representasi KV."""
        # Full KV: 2 * n_heads * d_kv = 2 * 4 * 16 = 128
        # MLA latent: mla_latent_dim = 32
        assert small_mla.memory_savings_ratio > 0.5  # Minimal 50% savings

    def test_forward_with_cache(self, small_mla):
        """Test forward dengan KV cache."""
        x = torch.randn(2, 16, 64)
        _, kv_latent, kv_cache = small_mla(x)
        # kv_cache bisa None jika tidak dibuat secara eksplisit
        # Yang penting kv_latent harus ada
        assert kv_latent is not None

    def test_forward_inference(self, small_mla):
        """Test inference mode MLA."""
        x = torch.randn(2, 1, 64)
        # Buat cache kosong
        kv_cache = small_mla.create_kv_cache(
            batch_size=2, max_seq_len=64, dtype=torch.float32, device=torch.device("cpu")
        )
        output, kv_cache = small_mla.forward_inference(
            x, kv_cache=kv_cache, start_pos=0, rope_enabled=True
        )
        assert output.shape == (2, 1, 64)


# ============================================================================
# Test InterleavedRoPE
# ============================================================================


class TestInterleavedRoPE:
    """Test suite untuk InterleavedRoPE (iRoPE)."""

    def test_init(self, small_irope):
        """Test bahwa iRoPE dapat diinisialisasi."""
        assert isinstance(small_irope, InterleavedRoPE)

    def test_should_use_rope_pattern(self, small_irope):
        """Test pola RoPE/NoPE."""
        # Dengan ratio 3:1, 3 dari 4 layer menggunakan RoPE
        n_layers = 8
        rope_count = sum(1 for i in range(n_layers) if small_irope.should_use_rope(i))
        nope_count = n_layers - rope_count
        # Harus ada baik RoPE maupun NoPE layers
        assert rope_count > 0, "Tidak ada layer yang menggunakan RoPE"
        assert nope_count > 0, "Tidak ada layer yang menggunakan NoPE"

    def test_get_layer_pattern(self, small_irope):
        """Test pola layer iRoPE."""
        n_layers = 12
        pattern = small_irope.get_layer_pattern(n_layers)
        assert len(pattern) == n_layers
        assert all(isinstance(p, bool) for p in pattern)

    def test_ratio_consistency(self):
        """Test bahwa rasio RoPE konsisten."""
        for ratio in [1, 2, 3, 5]:
            irope = InterleavedRoPE(dim=16, base=10000.0, ratio=ratio)
            n_layers = ratio * 4  # Enough layers for pattern
            pattern = irope.get_layer_pattern(n_layers)
            rope_fraction = sum(pattern) / n_layers
            # RoPE fraction harus mendekati ratio/(ratio+1)
            expected_fraction = ratio / (ratio + 1)
            assert abs(rope_fraction - expected_fraction) < 0.2, \
                f"RoPE fraction {rope_fraction:.2f} jauh dari expected {expected_fraction:.2f}"


# ============================================================================
# Test AttentionKompresiLayer
# ============================================================================


class TestAttentionKompresiLayer:
    """Test suite untuk AttentionKompresiLayer."""

    def test_init(self, small_attention_layer):
        """Test bahwa layer dapat diinisialisasi."""
        assert isinstance(small_attention_layer, nn.Module)
        assert small_attention_layer.d_model == 64
        assert small_attention_layer.n_heads == 4

    def test_forward_shape(self, small_attention_layer):
        """Test bahwa output memiliki shape yang benar."""
        batch, seq, d = 2, 16, 64
        x = torch.randn(batch, seq, d)
        output, state = small_attention_layer(x, layer_idx=0)
        assert output.shape == (batch, seq, d)

    def test_forward_no_nan(self, small_attention_layer):
        """Test bahwa output tidak mengandung NaN."""
        x = torch.randn(2, 16, 64)
        output, _ = small_attention_layer(x, layer_idx=0)
        assert not torch.isnan(output).any(), "Attention output mengandung NaN"

    def test_forward_no_inf(self, small_attention_layer):
        """Test bahwa output tidak mengandung Inf."""
        x = torch.randn(2, 16, 64)
        output, _ = small_attention_layer(x, layer_idx=0)
        assert not torch.isinf(output).any(), "Attention output mengandung Inf"

    def test_forward_returns_attention_state(self, small_attention_layer):
        """Test bahwa forward mengembalikan AttentionState."""
        x = torch.randn(2, 16, 64)
        _, state = small_attention_layer(x, layer_idx=0)
        assert isinstance(state, AttentionState)

    def test_forward_thinking_mode(self, small_attention_layer):
        """Test forward dengan thinking mode aktif."""
        x = torch.randn(2, 16, 64)
        output, state = small_attention_layer(x, layer_idx=0, thinking_mode=True)
        assert output.shape == (2, 16, 64)
        assert state.thinking_mode == True

    def test_forward_different_layers(self, small_attention_layer):
        """Test forward di layer berbeda (iRoPE pattern)."""
        x = torch.randn(2, 16, 64)
        for layer_idx in range(8):
            output, state = small_attention_layer(x, layer_idx=layer_idx)
            assert output.shape == (2, 16, 64)

    def test_forward_with_routing_weights(self, small_attention_layer):
        """Test forward dengan routing weights."""
        x = torch.randn(2, 16, 64)
        routing_weights = torch.tensor(0.5)
        output, state = small_attention_layer(
            x, layer_idx=0, routing_weights=routing_weights
        )
        assert output.shape == (2, 16, 64)

    def test_forward_with_attention_state(self, small_attention_layer):
        """Test forward dengan state dari layer sebelumnya."""
        x = torch.randn(2, 16, 64)
        _, state = small_attention_layer(x, layer_idx=0)
        output2, state2 = small_attention_layer(x, layer_idx=1, attention_state=state)
        assert output2.shape == (2, 16, 64)

    def test_forward_empty_sequence(self, small_attention_layer):
        """Test edge case: sequence kosong."""
        x = torch.randn(2, 0, 64)
        output, state = small_attention_layer(x, layer_idx=0)
        assert output.shape == (2, 0, 64)

    def test_create_kv_cache(self, small_attention_layer):
        """Test pembuatan KV cache."""
        cache = small_attention_layer.create_kv_cache(
            batch_size=2,
            max_seq_len=128,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        assert cache is not None

    def test_get_layer_info(self, small_attention_layer):
        """Test informasi layer untuk debugging."""
        info = small_attention_layer.get_layer_info(layer_idx=0)
        assert "layer_idx" in info
        assert "attention_type" in info
        assert "rope_enabled" in info

    def test_gradient_flow(self, small_attention_layer):
        """Test bahwa gradient mengalir melalui layer."""
        x = torch.randn(2, 16, 64, requires_grad=True)
        output, _ = small_attention_layer(x, layer_idx=0)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None, "Gradient tidak mengalir ke input"

    def test_compute_model_stats(self, small_attention_layer):
        """Test statistik model."""
        stats = small_attention_layer.compute_model_stats(n_layers=12)
        assert "n_layers" in stats
        assert "irope" in stats
        assert "mla" in stats
        assert stats["n_layers"] == 12


# ============================================================================
# Test AttentionState
# ============================================================================


class TestAttentionState:
    """Test suite untuk AttentionState."""

    def test_default_state(self):
        """Test state default."""
        state = AttentionState()
        assert state.kv_latent is None
        assert state.kv_cache is None
        assert state.layer_type == "local"
        assert state.rope_used == True
        assert state.thinking_mode == False

    def test_update_state(self):
        """Test update state."""
        state = AttentionState()
        state.update(layer_type="global", rope_used=False, thinking_mode=True)
        assert state.layer_type == "global"
        assert state.rope_used == False
        assert state.thinking_mode == True

    def test_partial_update(self):
        """Test update parsial."""
        state = AttentionState(layer_type="local")
        state.update(thinking_mode=True)
        assert state.layer_type == "local"  # Tidak berubah
        assert state.thinking_mode == True  # Berubah

    def test_update_returns_self(self):
        """Test bahwa update mengembalikan self untuk chaining."""
        state = AttentionState()
        result = state.update(layer_type="global")
        assert result is state
