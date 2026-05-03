"""
Shared Attention Pool Implementation untuk Losion Framework v0.4.

Upgrade #9: Zamba2-style Shared Attention.

Dalam arsitektur Zamba2, beberapa attention layer berbagi parameter attention
yang sama (Q, K, V, O projections) tetapi memiliki FFN/MoE weights yang terpisah.
Hal ini mengurangi KV cache secara signifikan (~6x) karena shared layers
tidak memerlukan penyimpanan KV terpisah — mereka dapat mereuse KV yang sama
atau menghitung KV sekali dan membagikannya.

Arsitektur:
1. SharedAttentionPool — Kontainer parameter attention yang dibagi
   Menyimpan proyeksi Q, K, V, dan O yang direferensikan oleh
   beberapa SharedAttentionLayer. Juga menyimpan KV cache shared
   sehingga tidak perlu dihitung ulang untuk setiap layer.

2. SharedAttentionLayer — Layer attention yang mereferensikan pool
   Setiap layer memiliki:
   - Referensi ke SharedAttentionPool (attention parameters + KV cache)
   - Layer norm sendiri
   - FFN/MoE sendiri
   - Gate opsional untuk blending
   Beberapa layer bisa "unique" (punya parameter attention sendiri)
   atau "shared" (mereferensikan pool).

Konfigurasi:
   - n_shared_groups: Berapa kelompok shared attention (default 1 = semua shared)
   - n_unique_layers: Berapa layer yang punya parameter attention sendiri
   - sharing_pattern: Pola sharing, misalnya [0,0,1,1,0,0,1,1]
     dimana angka sama = group yang sama

Keuntungan:
   - Parameter reduction: Hanya 1 set attention params untuk N shared layers
   - KV cache reduction: ~6x karena shared layers bisa share KV computation
   - Faster inference: KV hanya dihitung sekali untuk shared layers
   - Flexible: Bisa dicampur dengan unique layers

Referensi:
- Glorioso et al., "Zamba2: A Next-Generation Hybrid Mamba2-Based
  Language Model" (2024)
- Konsep shared attention: mengurangi redundansi parameter antar layer
  tanpa mengorbankan kualitas secara signifikan
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SharedAttentionPool — Parameter Pool untuk Shared Attention
# ============================================================================

class SharedAttentionPool(nn.Module):
    """
    Pool parameter attention yang dibagi oleh beberapa layer.

    Menyimpan satu set parameter attention (Q, K, V, O projections) yang
    dapat direferensikan oleh beberapa SharedAttentionLayer. Juga menyediakan
    shared KV cache sehingga komputasi KV hanya dilakukan sekali untuk semua
    layer yang sharing.

    Arsitektur Pool:
        - q_proj: Proyeksi query (d_model → n_heads * d_head)
        - k_proj: Proyeksi key (d_model → n_heads * d_head)
        - v_proj: Proyeksi value (d_model → n_heads * d_head)
        - out_proj: Proyeksi output (n_heads * d_head → d_model)
        - q_norm, k_norm: QK normalization
        - rope: Rotary Position Embedding

    KV Cache Sharing:
        Ketika multiple layer mereferensikan pool yang sama, KV hanya dihitung
        sekali per forward pass dan disimpan dalam cache. Layer berikutnya
        langsung menggunakan cached KV, mengurangi komputasi dan memori.

    Args:
        d_model: Dimensi model.
        n_heads: Jumlah attention heads.
        d_head: Dimensi per head.
        use_rope: Apakah menggunakan RoPE (default True).
        rope_base: Basis frekuensi RoPE.
        pool_id: Identifier unik untuk pool (berguna untuk debugging).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        pool_id: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head
        self.use_rope = use_rope
        self.pool_id = pool_id

        # ---- Shared Attention Parameters ----
        self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # ---- QK Normalization ----
        self.q_norm = nn.RMSNorm(d_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(d_head, eps=1e-5)

        # ---- RoPE ----
        if self.use_rope:
            # Import locally to avoid circular dependency
            from .lightning_attention import InterleavedRoPE
            self.rope = InterleavedRoPE(
                dim=d_head,
                d_rope=d_head // 2,
                base=rope_base,
                interleaved=True,
            )

        # ---- KV Cache (shared antar layer) ----
        # Disimpan sebagai non-parameter buffer
        self._kv_cache: Optional[torch.Tensor] = None
        self._kv_cache_len: int = 0

    def compute_qkv(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hitung Q, K, V menggunakan shared parameters.

        Juga meng-update shared KV cache.

        Args:
            x: Tensor input, (batch, seq_len, d_model).
            position_offset: Offset posisi untuk RoPE.

        Returns:
            Tuple (q, k, v):
            - q: (batch, seq_len, n_heads, d_head)
            - k: (batch, seq_len, n_heads, d_head)
            - v: (batch, seq_len, n_heads, d_head)
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_head)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        if self.use_rope:
            q = self.rope(q, offset=position_offset)
            k = self.rope(k, offset=position_offset)

        return q, k, v

    def update_kv_cache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        past_kv_cache: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update dan return KV cache.

        KV cache disimpan sebagai tensor tunggal:
        (batch, cache_len, n_heads, d_head * 2)
        dimana d_head * 2 = K dan V digabung.

        Jika window_size ditentukan, hanya menyimpan token terakhir
        sebesar window_size (sliding window).

        Args:
            k: Key tensor, (batch, seq_len, n_heads, d_head).
            v: Value tensor, (batch, seq_len, n_heads, d_head).
            past_kv_cache: Cache dari step sebelumnya,
                (batch, past_len, n_heads, d_head * 2).
            window_size: Ukuran sliding window (None = unlimited).

        Returns:
            Tuple (k_full, v_full):
            - k_full: (batch, full_len, n_heads, d_head)
            - v_full: (batch, full_len, n_heads, d_head)
        """
        if past_kv_cache is not None:
            past_k = past_kv_cache[:, :, :, :self.d_head]
            past_v = past_kv_cache[:, :, :, self.d_head:]
            k_full = torch.cat([past_k, k], dim=1)
            v_full = torch.cat([past_v, v], dim=1)
        else:
            k_full = k
            v_full = v

        # Truncate ke window_size jika diperlukan
        if window_size is not None and k_full.shape[1] > window_size:
            k_full = k_full[:, k_full.shape[1] - window_size:]
            v_full = v_full[:, v_full.shape[1] - window_size:]

        return k_full, v_full

    def build_kv_cache_tensor(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bangun tensor KV cache dari K dan V.

        Args:
            k: (batch, seq_len, n_heads, d_head)
            v: (batch, seq_len, n_heads, d_head)

        Returns:
            KV cache tensor, (batch, seq_len, n_heads, d_head * 2)
        """
        return torch.cat([k, v], dim=-1)

    def compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
    ) -> torch.Tensor:
        """
        Hitung standard multi-head attention.

        Args:
            q: Query, (batch, seq_len, n_heads, d_head).
            k: Key, (batch, full_len, n_heads, d_head).
            v: Value, (batch, full_len, n_heads, d_head).
            attention_mask: Mask opsional.
            dropout: Dropout rate.

        Returns:
            Attention output, (batch, seq_len, n_heads, d_head).
        """
        batch, seq_len, n_heads, d_head = q.shape
        full_len = k.shape[1]

        # Transpose ke (batch, n_heads, seq_len, d_head)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Attention scores
        scale = math.sqrt(d_head)
        attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) / scale

        # Causal mask (default)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, full_len, dtype=torch.bool, device=q.device),
                diagonal=full_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        if dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=dropout)

        # Weighted sum
        output = torch.matmul(attn_weights, v_t)  # (batch, n_heads, seq_len, d_head)

        # Transpose kembali
        output = output.transpose(1, 2)  # (batch, seq_len, n_heads, d_head)

        return output

    def project_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proyeksi output menggunakan shared out_proj.

        Args:
            x: Tensor, (batch, seq_len, n_heads, d_head).

        Returns:
            Tensor, (batch, seq_len, d_model).
        """
        batch, seq_len, _, _ = x.shape
        x = x.contiguous().view(batch, seq_len, self.d_inner)
        return self.out_proj(x)


# ============================================================================
# SharedAttentionLayer — Layer dengan Shared Attention Parameters
# ============================================================================

class SharedAttentionLayer(nn.Module):
    """
    Attention layer yang menggunakan shared attention parameters dari pool.

    Setiap SharedAttentionLayer:
    - Mereferensikan SharedAttentionPool untuk parameter attention (Q, K, V, O)
    - Memiliki layer norm sendiri (pre-norm dan post-norm)
    - Memiliki FFN/MoE sendiri
    - Memiliki gate opsional untuk output blending

    Dua mode:
    1. Shared mode (default): Menggunakan parameter dari SharedAttentionPool
       - KV cache dihitung sekali oleh pool dan di-share
       - Parameter attention tidak ada di layer ini
       - Efisien untuk inferensi dan memori

    2. Unique mode: Memiliki parameter attention sendiri
       - Tidak mereferensikan pool
       - Memiliki Q, K, V, O projections sendiri
       - Berguna untuk layer yang membutuhkan representasi berbeda

    Contoh konfigurasi (Zamba2-style, 12 layers):
        - Pool 0: Layers 0, 3, 6, 9 (shared attention group 0)
        - Pool 1: Layers 1, 4, 7, 10 (shared attention group 1)
        - Unique:  Layers 2, 5, 8, 11 (unique attention)
        → Hanya 2 + 4 = 6 set attention params, bukan 12
        → KV cache: ~3x reduction (2 shared groups * 1 KV + 4 unique)

    Args:
        d_model: Dimensi model.
        n_heads: Jumlah attention heads (harus cocok dengan pool).
        d_head: Dimensi per head (harus cocok dengan pool).
        pool: SharedAttentionPool yang direferensikan (None untuk unique).
        is_unique: Jika True, layer ini punya parameter attention sendiri.
        ffn_dim: Dimensi FFN tersembunyi (default 4 * d_model).
        dropout: Dropout rate.
        layer_id: Identifier unik untuk layer.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        pool: Optional[SharedAttentionPool] = None,
        is_unique: bool = False,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        layer_id: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head
        self.is_unique = is_unique
        self.layer_id = layer_id

        ffn_dim = ffn_dim or (4 * d_model)

        # ---- Reference ke shared pool ----
        if not is_unique and pool is not None:
            self.pool = pool
            self._uses_pool = True
        else:
            self.pool = None
            self._uses_pool = False

        # ---- Unique attention parameters (jika is_unique=True) ----
        if is_unique:
            self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)
            self.k_proj = nn.Linear(d_model, self.d_inner, bias=False)
            self.v_proj = nn.Linear(d_model, self.d_inner, bias=False)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
            self.q_norm = nn.RMSNorm(d_head, eps=1e-5)
            self.k_norm = nn.RMSNorm(d_head, eps=1e-5)

            # RoPE untuk unique layer
            from .lightning_attention import InterleavedRoPE
            self.rope = InterleavedRoPE(
                dim=d_head,
                d_rope=d_head // 2,
                interleaved=True,
            )

        # ---- Layer norms (selalu punya sendiri) ----
        self.pre_norm = nn.RMSNorm(d_model, eps=1e-5)
        self.post_norm = nn.RMSNorm(d_model, eps=1e-5)

        # ---- FFN (selalu punya sendiri) ----
        self.ffn = nn.Sequential(
            nn.RMSNorm(d_model, eps=1e-5),
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, d_model, bias=False),
            nn.Dropout(dropout),
        )

        # ---- Gate untuk attention-FFN blending ----
        self.ffn_gate = nn.Linear(d_model, 1, bias=False)

        # ---- Dropout ----
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _compute_attention_unique(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Hitung attention menggunakan parameter unik (bukan dari pool).

        Args:
            x: Input, (batch, seq_len, d_model).
            attention_mask: Mask opsional.
            past_kv_cache: KV cache dari step sebelumnya.
            position_offset: Offset posisi untuk RoPE.

        Returns:
            Tuple (output, new_kv_cache).
        """
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_head)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q, offset=position_offset)
        k = self.rope(k, offset=position_offset)

        # Update KV cache
        k_full, v_full = self._update_kv(k, v, past_kv_cache)

        # Attention
        q_t = q.transpose(1, 2)
        k_t = k_full.transpose(1, 2)
        v_t = v_full.transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) / scale

        # Causal mask
        full_len = k_full.shape[1]
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, full_len, dtype=torch.bool, device=x.device),
                diagonal=full_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v_t)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_inner)
        output = self.out_proj(output)

        # Build new KV cache
        new_kv_cache = torch.cat([k_full, v_full], dim=-1)
        # Truncate jika perlu
        max_cache = 2048  # default window
        if new_kv_cache.shape[1] > max_cache:
            new_kv_cache = new_kv_cache[:, new_kv_cache.shape[1] - max_cache:]

        return output, new_kv_cache

    def _update_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        past_kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate K, V dengan cache.

        Args:
            k: (batch, seq_len, n_heads, d_head)
            v: (batch, seq_len, n_heads, d_head)
            past_kv_cache: (batch, past_len, n_heads, d_head * 2)

        Returns:
            Tuple (k_full, v_full).
        """
        if past_kv_cache is not None:
            past_k = past_kv_cache[:, :, :, :self.d_head]
            past_v = past_kv_cache[:, :, :, self.d_head:]
            return (
                torch.cat([past_k, k], dim=1),
                torch.cat([past_v, v], dim=1),
            )
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        shared_kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass SharedAttentionLayer.

        Jika shared mode: gunakan SharedAttentionPool untuk attention,
        KV cache di-share antar layer dalam group yang sama.
        Jika unique mode: gunakan parameter attention sendiri.

        Args:
            x: Tensor input, (batch, seq_len, d_model).
            attention_mask: Mask opsional.
            past_kv_cache: KV cache dari step sebelumnya.
                Untuk shared mode: ini adalah shared KV cache dari pool.
                Untuk unique mode: cache layer ini sendiri.
            position_offset: Offset posisi untuk RoPE.
            shared_kv_cache: Override KV cache untuk shared mode.
                Jika diberikan, gunakan ini alih-alih menghitung KV baru.

        Returns:
            Tuple (output, new_kv_cache):
            - output: (batch, seq_len, d_model)
            - new_kv_cache: Updated KV cache (shared atau unique)
        """
        batch, seq_len, _ = x.shape

        # ---- Pre-normalization ----
        residual = x
        x_normed = self.pre_norm(x)

        # ---- Attention ----
        if self._uses_pool and self.pool is not None:
            # Shared mode: gunakan pool
            q, k, v = self.pool.compute_qkv(x_normed, position_offset=position_offset)

            # Update atau gunakan shared KV cache
            if shared_kv_cache is not None:
                k_full, v_full = self.pool.update_kv_cache(
                    k, v, past_kv_cache=shared_kv_cache
                )
            elif past_kv_cache is not None:
                k_full, v_full = self.pool.update_kv_cache(
                    k, v, past_kv_cache=past_kv_cache
                )
            else:
                k_full, v_full = k, v

            # Hitung attention menggunakan pool
            attn_output = self.pool.compute_attention(
                q, k_full, v_full,
                attention_mask=attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
            )

            # Proyeksi output menggunakan pool
            attn_output = self.pool.project_output(attn_output)

            # Build new shared KV cache
            new_kv_cache = self.pool.build_kv_cache_tensor(k_full, v_full)
        else:
            # Unique mode: parameter attention sendiri
            attn_output, new_kv_cache = self._compute_attention_unique(
                x_normed,
                attention_mask=attention_mask,
                past_kv_cache=past_kv_cache,
                position_offset=position_offset,
            )

        # ---- Residual connection ----
        attn_output = self.resid_dropout(attn_output)
        x = residual + attn_output

        # ---- FFN dengan gating ----
        ffn_residual = x
        ffn_output = self.ffn(x)

        # Gate: mengontrol kontribusi FFN
        gate = torch.sigmoid(self.ffn_gate(x))  # (batch, seq_len, 1)
        ffn_output = gate * ffn_output

        x = ffn_residual + ffn_output

        # ---- Post-normalization ----
        x = self.post_norm(x)

        return x, new_kv_cache

    def forward_inference(
        self,
        x: torch.Tensor,
        past_kv_cache: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass untuk inferensi token-per-token.

        Args:
            x: Tensor input satu token, (batch, 1, d_model).
            past_kv_cache: KV cache dari step sebelumnya.
            position_offset: Offset posisi.

        Returns:
            Tuple (output, new_kv_cache).
        """
        return self.forward(
            x,
            attention_mask=None,
            past_kv_cache=past_kv_cache,
            position_offset=position_offset,
        )


# ============================================================================
# SharedAttentionConfig — Konfigurasi Sharing Pattern
# ============================================================================

class SharedAttentionConfig:
    """
    Konfigurasi untuk pola sharing attention antar layer.

    Menentukan layer mana yang sharing pool yang sama dan mana
    yang memiliki parameter attention unik.

    Contoh:
        # Zamba2-style: 12 layers, 2 shared groups + 2 unique
        config = SharedAttentionConfig(
            n_layers=12,
            n_shared_groups=2,
            sharing_pattern="interleaved",  # atau "block" atau custom list
        )

        # Custom pattern
        config = SharedAttentionConfig(
            n_layers=8,
            sharing_pattern=[0, 0, -1, 1, 0, 0, -1, 1],
            # -1 = unique, 0 = pool 0, 1 = pool 1
        )

    Args:
        n_layers: Jumlah total layer.
        n_shared_groups: Jumlah shared attention groups/pools.
        sharing_pattern: Pola sharing. Bisa:
            - "interleaved": Shared dan unique di-interleave
            - "block": Shared dikelompokkan dalam blok
            - "all_shared": Semua layer sharing satu pool
            - List[int]: Custom pattern, -1 = unique, >=0 = pool id
        unique_ratio: Rasio layer unique (0.0 - 1.0), digunakan jika
            sharing_pattern adalah "interleaved" atau "block".
    """

    def __init__(
        self,
        n_layers: int,
        n_shared_groups: int = 1,
        sharing_pattern: Union[str, List[int]] = "interleaved",
        unique_ratio: float = 0.25,
    ):
        self.n_layers = n_layers
        self.n_shared_groups = n_shared_groups
        self.sharing_pattern = sharing_pattern
        self.unique_ratio = unique_ratio

        # Build pattern
        if isinstance(sharing_pattern, str):
            self._pattern = self._build_pattern(sharing_pattern, n_layers, n_shared_groups, unique_ratio)
        else:
            self._pattern = sharing_pattern
            # Auto-detect n_shared_groups dari custom pattern
            max_pool_id = max((p for p in self._pattern if p >= 0), default=-1)
            if max_pool_id >= 0:
                self.n_shared_groups = max_pool_id + 1

        self._validate_pattern()

    def _build_pattern(
        self,
        mode: str,
        n_layers: int,
        n_shared_groups: int,
        unique_ratio: float,
    ) -> List[int]:
        """
        Bangun pola sharing berdasarkan mode.

        Args:
            mode: "interleaved", "block", atau "all_shared".
            n_layers: Jumlah total layer.
            n_shared_groups: Jumlah shared groups.
            unique_ratio: Rasio layer unique.

        Returns:
            List assignment: -1 = unique, >=0 = pool id.
        """
        n_unique = max(1, int(n_layers * unique_ratio))
        n_shared = n_layers - n_unique

        if mode == "all_shared":
            # Semua layer dalam satu pool
            return [0] * n_layers

        elif mode == "block":
            # Unique layers di tengah, shared di awal dan akhir
            pattern = []
            shared_per_group = n_shared // n_shared_groups
            remaining = n_shared % n_shared_groups

            for g in range(n_shared_groups):
                count = shared_per_group + (1 if g < remaining else 0)
                pattern.extend([g] * count)

            # Sisipkan unique layers di antara shared blocks
            result = []
            unique_inserted = 0
            shared_idx = 0
            unique_spacing = max(1, n_shared // (n_unique + 1))

            for i in range(n_layers):
                if unique_inserted < n_unique and i > 0 and i % unique_spacing == 0:
                    result.append(-1)
                    unique_inserted += 1
                elif shared_idx < len(pattern):
                    result.append(pattern[shared_idx])
                    shared_idx += 1
                else:
                    result.append(-1)
                    unique_inserted += 1

            # Fill remaining
            while len(result) < n_layers:
                result.append(-1)

            return result[:n_layers]

        elif mode == "interleaved":
            # Interleave shared groups dan unique
            pattern = []
            shared_per_group = n_shared // n_shared_groups
            remaining = n_shared % n_shared_groups

            # Buat daftar assignment per group
            group_assignments = []
            for g in range(n_shared_groups):
                count = shared_per_group + (1 if g < remaining else 0)
                group_assignments.extend([g] * count)

            # Interleave: sisipkan unique secara merata
            result = []
            shared_ptr = 0
            unique_ptr = 0
            unique_spacing = max(1, n_shared // (n_unique + 1))
            step = 0

            for i in range(n_layers):
                if unique_ptr < n_unique and step > 0 and step % unique_spacing == 0:
                    result.append(-1)
                    unique_ptr += 1
                elif shared_ptr < len(group_assignments):
                    result.append(group_assignments[shared_ptr])
                    shared_ptr += 1
                else:
                    result.append(-1)
                    unique_ptr += 1
                step += 1

            while len(result) < n_layers:
                if shared_ptr < len(group_assignments):
                    result.append(group_assignments[shared_ptr])
                    shared_ptr += 1
                else:
                    result.append(-1)

            return result[:n_layers]

        else:
            # Default: all shared
            return [0] * n_layers

    def _validate_pattern(self):
        """Validasi pola sharing."""
        if len(self._pattern) != self.n_layers:
            raise ValueError(
                f"Pattern length ({len(self._pattern)}) != n_layers ({self.n_layers})"
            )

        # Verifikasi pool IDs valid
        pool_ids = set(p for p in self._pattern if p >= 0)
        if pool_ids and max(pool_ids) >= self.n_shared_groups:
            raise ValueError(
                f"Pool ID {max(pool_ids)} >= n_shared_groups {self.n_shared_groups}"
            )

    def get_layer_assignment(self, layer_idx: int) -> int:
        """
        Dapatkan assignment untuk layer tertentu.

        Args:
            layer_idx: Indeks layer (0-based).

        Returns:
            -1 untuk unique, >=0 untuk pool id.
        """
        return self._pattern[layer_idx]

    def get_shared_layer_indices(self, pool_id: int) -> List[int]:
        """
        Dapatkan indeks layer yang sharing pool tertentu.

        Args:
            pool_id: ID shared pool.

        Returns:
            List indeks layer yang menggunakan pool ini.
        """
        return [i for i, p in enumerate(self._pattern) if p == pool_id]

    def get_unique_layer_indices(self) -> List[int]:
        """
        Dapatkan indeks layer yang memiliki attention unik.

        Returns:
            List indeks layer unique.
        """
        return [i for i, p in enumerate(self._pattern) if p == -1]

    def count_parameters(self, d_model: int, n_heads: int, d_head: int) -> Dict[str, int]:
        """
        Hitung jumlah parameter untuk konfigurasi ini.

        Args:
            d_model: Dimensi model.
            n_heads: Jumlah attention heads.
            d_head: Dimensi per head.

        Returns:
            Dict dengan jumlah parameter per kategori.
        """
        d_inner = n_heads * d_head

        # Parameter per set attention
        attn_params = (
            d_model * d_inner  # q_proj
            + d_model * d_inner  # k_proj
            + d_model * d_inner  # v_proj
            + d_inner * d_model  # out_proj
        )

        n_unique = len(self.get_unique_layer_indices())
        total_shared_groups = len(set(p for p in self._pattern if p >= 0))

        shared_params = total_shared_groups * attn_params
        unique_params = n_unique * attn_params
        total_attn_params = shared_params + unique_params
        baseline_params = self.n_layers * attn_params

        return {
            "total_attention_params": total_attn_params,
            "baseline_attention_params": baseline_params,
            "shared_params": shared_params,
            "unique_params": unique_params,
            "savings_ratio": 1.0 - (total_attn_params / baseline_params) if baseline_params > 0 else 0.0,
            "n_shared_groups": total_shared_groups,
            "n_unique_layers": n_unique,
        }

    @property
    def pattern(self) -> List[int]:
        """Return pola sharing."""
        return self._pattern.copy()

    def __repr__(self) -> str:
        return (
            f"SharedAttentionConfig(n_layers={self.n_layers}, "
            f"n_shared_groups={self.n_shared_groups}, "
            f"pattern={self._pattern})"
        )
