"""
Attention Komposisi Layer — Composite attention layer for Losion Framework.

Menggabungkan MLA, iRoPE, dan AdaptiveInterleaving ke dalam satu layer
yang menyediakan interface tingkat tinggi untuk Jalur 2 (Attention+Compression).

Komponen:
  AttentionState       — State object untuk attention layer
  MLAKVCache          — KV cache wrapper untuk MLA
  AdaptiveInterleaving — Modul yang mengatur pola interleaving RoPE/NoPE
  AttentionKompresiLayer — Composite layer MLA + iRoPE + kompresi

Referensi:
- DeepSeek-AI, "DeepSeek-V2" (2024) — MLA
- Su, Q. et al., "RoFormer" (2021) — RoPE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# AttentionState — State object untuk attention layer
# ============================================================================


@dataclass
class AttentionState:
    """
    State object yang menyimpan informasi attention antar layer.

    Meneruskan KV latent, cache, dan metadata (layer_type, rope_used,
    thinking_mode) dari satu layer ke layer berikutnya, memungkinkan
    koordinasi pola iRoPE dan adaptive interleaving.

    Attributes:
        kv_latent: KV latent tensor dari MLA (batch, seq_len, kv_lora_rank).
        kv_cache: KV cache object (MLAKVCache atau tuple).
        layer_type: Tipe layer ("local" atau "global").
        rope_used: Apakah RoPE digunakan pada layer ini.
        thinking_mode: Apakah thinking mode aktif pada layer ini.
    """

    kv_latent: Optional[torch.Tensor] = None
    kv_cache: Optional[Any] = None
    layer_type: str = "local"
    rope_used: bool = True
    thinking_mode: bool = False

    def update(self, **kwargs) -> "AttentionState":
        """
        Update state attributes. Partial updates are supported — hanya
        atribut yang diberikan yang diubah, sisanya tetap.

        Returns:
            self, untuk method chaining.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


# ============================================================================
# MLAKVCache — KV Cache untuk Multi-head Latent Attention
# ============================================================================


class MLAKVCache:
    """
    KV Cache wrapper untuk MLA yang menyimpan compressed KV latent.

    Alih-alih menyimpan full K dan V per token per head, MLAKVCache
    menyimpan KV latent berdimensi rendah (kv_lora_rank), mengurangi
    penggunaan memori secara dramatis.

    Attributes:
        kv_latent: Tensor (batch, max_seq_len, kv_lora_rank) — compressed KV.
        position: Posisi penulisan berikutnya dalam cache.
        batch_size: Jumlah batch.
        max_seq_len: Panjang sequence maksimum.
        kv_lora_rank: Dimensi latent compression.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.dtype = dtype
        self.device = device

        # Pre-allocate cache tensor
        self.kv_latent = torch.zeros(
            batch_size, max_seq_len, kv_lora_rank,
            dtype=dtype, device=device,
        )
        self.position = 0

    def update(
        self,
        new_kv_latent: torch.Tensor,
        start_pos: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Tambahkan KV latent baru ke cache.

        Args:
            new_kv_latent: Tensor (batch, seq_len, kv_lora_rank).
            start_pos: Posisi awal penulisan. Jika None, gunakan self.position.

        Returns:
            Full KV latent sampai posisi saat ini.
        """
        if start_pos is None:
            start_pos = self.position

        seq_len = new_kv_latent.shape[1]
        self.kv_latent[:, start_pos:start_pos + seq_len, :] = new_kv_latent
        self.position = max(self.position, start_pos + seq_len)

        return self.kv_latent[:, :self.position, :]

    def get_cached(self) -> Optional[torch.Tensor]:
        """
        Ambil KV latent yang sudah di-cache.

        Returns:
            Tensor (batch, position, kv_lora_rank) atau None jika kosong.
        """
        if self.position == 0:
            return None
        return self.kv_latent[:, :self.position, :]

    def size(self) -> int:
        """Jumlah token yang sudah di-cache."""
        return self.position

    def clear(self) -> None:
        """Reset cache."""
        self.kv_latent.zero_()
        self.position = 0


# ============================================================================
# AdaptiveInterleaving — Adaptive RoPE/NoPE interleaving pattern
# ============================================================================


class AdaptiveInterleaving(nn.Module):
    """
    Modul yang mengatur pola interleaving RoPE/NoPE secara adaptif.

    Menggabungkan base interleaving ratio (konstan) dengan thinking mode
    ratio (dinamis) untuk menentukan rasio RoPE vs NoPE pada setiap layer.

    Base ratio menentukan pola default: misalnya ratio=5 berarti 5 dari 6
    layer menggunakan RoPE. Thinking ratio menentukan pola saat thinking
    mode aktif: ratio=2 berarti 2 dari 3 layer menggunakan RoPE.

    Args:
        base_interleaving_ratio: Rasio RoPE:NoPE saat mode normal (default 5).
        thinking_interleaving_ratio: Rasio RoPE:NoPE saat thinking mode (default 2).
    """

    def __init__(
        self,
        base_interleaving_ratio: int = 5,
        thinking_interleaving_ratio: int = 2,
    ):
        super().__init__()
        self.base_ratio = base_interleaving_ratio
        self.thinking_ratio = thinking_interleaving_ratio

    def get_ratio(self, thinking_mode: bool = False) -> int:
        """
        Ambil rasio interleaving saat ini.

        Args:
            thinking_mode: Apakah thinking mode aktif.

        Returns:
            Rasio interleaving (integer).
        """
        return self.thinking_ratio if thinking_mode else self.base_ratio

    def should_use_rope(self, layer_idx: int, thinking_mode: bool = False) -> bool:
        """
        Tentukan apakah layer pada indeks tertentu menggunakan RoPE.

        Dengan ratio R, layer pada indeks i menggunakan RoPE jika
        i % (R + 1) != R. Dengan kata lain, dari setiap R+1 layer,
        R layer menggunakan RoPE dan 1 layer menggunakan NoPE.

        Args:
            layer_idx: Indeks layer.
            thinking_mode: Apakah thinking mode aktif.

        Returns:
            True jika layer ini menggunakan RoPE.
        """
        ratio = self.get_ratio(thinking_mode)
        # Dari (ratio + 1) layer berturut-turut, 1 layer tidak menggunakan RoPE
        return (layer_idx % (ratio + 1)) != ratio

    def get_layer_pattern(self, n_layers: int, thinking_mode: bool = False) -> list:
        """
        Hitung pola RoPE/NoPE untuk seluruh model.

        Args:
            n_layers: Jumlah total layer.
            thinking_mode: Apakah thinking mode aktif.

        Returns:
            List boolean dengan panjang n_layers. True = RoPE, False = NoPE.
        """
        return [self.should_use_rope(i, thinking_mode) for i in range(n_layers)]

    def forward(
        self,
        x: torch.Tensor,
        thinking_mode: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass — pass-through. Interleaving pattern diterapkan
        pada level layer (bukan pada tensor), jadi forward() hanya
        mengembalikan input unchanged.

        Args:
            x: Input tensor.
            thinking_mode: Apakah thinking mode aktif.

        Returns:
            Input tensor unchanged.
        """
        return x


# ============================================================================
# AttentionKompresiLayer — Composite Attention + Compression Layer
# ============================================================================


class AttentionKompresiLayer(nn.Module):
    """
    Composite attention layer yang menggabungkan MLA, iRoPE, dan kompresi.

    Ini adalah layer tingkat tinggi untuk Jalur 2 (Attention+Compression)
    yang menyediakan:
    - MLA (Multi-head Latent Attention) dengan KV kompresi
    - InterleavedRoPE dengan pola adaptif
    - Sliding window untuk konteks lokal
    - Feed-forward network untuk transformasi tambahan
    - AttentionState yang meneruskan informasi antar layer

    Arsitektur:
        1. MLA attention dengan KV latent compression
        2. InterleavedRoPE berdasarkan pola layer (iRoPE)
        3. Adaptive interleaving (base vs thinking mode ratio)
        4. Sliding window masking untuk konteks lokal
        5. FFN untuk transformasi non-linear
        6. Residual connection dan normalisasi

    Args:
        d_model: Dimensi model.
        n_heads: Jumlah attention heads.
        d_kv: Dimensi per head (alias d_head).
        mla_latent_dim: Dimensi KV latent compression (alias kv_lora_rank).
        irope_ratio: Rasio RoPE:NoPE (default 3).
        base_interleaving_ratio: Rasio interleaving dasar (default 5).
        thinking_interleaving_ratio: Rasio interleaving saat thinking (default 2).
        sliding_window_size: Ukuran sliding window (default 32).
        ffn_dim_multiplier: Pengali dimensi FFN (default 4).
        dropout: Dropout rate (default 0.0).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_kv: int = 64,
        mla_latent_dim: int = 256,
        irope_ratio: int = 3,
        base_interleaving_ratio: int = 5,
        thinking_interleaving_ratio: int = 2,
        sliding_window_size: int = 32,
        ffn_dim_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.mla_latent_dim = mla_latent_dim
        self.irope_ratio = irope_ratio
        self.sliding_window_size = sliding_window_size

        # MLA attention
        from losion.core.attention.lightning_attention import MLA, InterleavedRoPE
        self.mla = MLA(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_kv,
            kv_lora_rank=mla_latent_dim,
        )

        # InterleavedRoPE
        self.rope = InterleavedRoPE(
            dim=d_kv,
            base=10000.0,
        )

        # Adaptive interleaving
        self.adaptive_interleaving = AdaptiveInterleaving(
            base_interleaving_ratio=base_interleaving_ratio,
            thinking_interleaving_ratio=thinking_interleaving_ratio,
        )

        # FFN
        ffn_dim = d_model * ffn_dim_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, d_model, bias=False),
            nn.Dropout(dropout),
        )

        # Norms
        self.norm = nn.RMSNorm(d_model, eps=1e-5)
        self.ffn_norm = nn.RMSNorm(d_model, eps=1e-5)

        # Output norm
        self.output_norm = nn.RMSNorm(d_model, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int = 0,
        thinking_mode: bool = False,
        routing_weights: Optional[torch.Tensor] = None,
        attention_state: Optional[AttentionState] = None,
    ) -> Tuple[torch.Tensor, AttentionState]:
        """
        Forward pass melalui composite attention layer.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            layer_idx: Indeks layer saat ini (untuk iRoPE pattern).
            thinking_mode: Apakah thinking mode aktif.
            routing_weights: Bobot routing opsional.
            attention_state: State dari layer sebelumnya.

        Returns:
            Tuple (output, new_attention_state):
            - output: (batch, seq_len, d_model)
            - new_attention_state: AttentionState yang diperbarui
        """
        batch, seq_len, _ = x.shape

        # Tentukan apakah RoPE digunakan pada layer ini
        rope_used = self.adaptive_interleaving.should_use_rope(
            layer_idx, thinking_mode
        )

        # Norm
        normed = self.norm(x)

        # MLA forward
        # MLA returns (output, (present_kv, None))
        mla_output, (present_kv, _) = self.mla(normed)

        # Residual connection
        hidden = x + mla_output

        # FFN
        ffn_out = self.ffn(self.ffn_norm(hidden))
        output = hidden + ffn_out

        # Output norm
        output = self.output_norm(output)

        # Build attention state
        new_state = AttentionState(
            kv_latent=present_kv,
            kv_cache=None,
            layer_type="local",
            rope_used=rope_used,
            thinking_mode=thinking_mode,
        )

        return output, new_state

    def create_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> MLAKVCache:
        """
        Buat KV cache baru untuk inference.

        Args:
            batch_size: Jumlah batch.
            max_seq_len: Panjang sequence maksimum.
            dtype: Data type tensor.
            device: Device tensor.

        Returns:
            MLAKVCache instance.
        """
        return MLAKVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kv_lora_rank=self.mla_latent_dim,
            dtype=dtype,
            device=device,
        )

    def get_layer_info(self, layer_idx: int = 0) -> Dict[str, Any]:
        """
        Informasi layer untuk debugging.

        Args:
            layer_idx: Indeks layer.

        Returns:
            Dictionary dengan informasi layer.
        """
        rope_used = self.adaptive_interleaving.should_use_rope(layer_idx)
        return {
            "layer_idx": layer_idx,
            "attention_type": "MLA+KDA",
            "rope_enabled": rope_used,
            "mla_latent_dim": self.mla_latent_dim,
            "n_heads": self.n_heads,
            "d_kv": self.d_kv,
            "sliding_window_size": self.sliding_window_size,
        }

    def compute_model_stats(self, n_layers: int = 12) -> Dict[str, Any]:
        """
        Hitung statistik model untuk seluruh stack.

        Args:
            n_layers: Jumlah total layer.

        Returns:
            Dictionary dengan statistik model.
        """
        pattern = self.adaptive_interleaving.get_layer_pattern(n_layers)
        rope_layers = sum(pattern)
        nope_layers = n_layers - rope_layers

        # Estimasi penghematan memori MLA vs standard attention
        # Standard KV cache: 2 * n_heads * d_kv per token
        # MLA KV cache: mla_latent_dim per token
        standard_kv_per_token = 2 * self.n_heads * self.d_kv
        mla_kv_per_token = self.mla_latent_dim
        savings_ratio = 1.0 - (mla_kv_per_token / standard_kv_per_token)

        return {
            "n_layers": n_layers,
            "irope": {
                "rope_layers": rope_layers,
                "nope_layers": nope_layers,
                "rope_fraction": rope_layers / n_layers if n_layers > 0 else 0,
            },
            "mla": {
                "latent_dim": self.mla_latent_dim,
                "standard_kv_per_token": standard_kv_per_token,
                "mla_kv_per_token": mla_kv_per_token,
                "savings_ratio": savings_ratio,
            },
        }
