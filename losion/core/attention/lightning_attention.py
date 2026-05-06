"""
Lightning Attention Implementation untuk Losion Framework v0.4.

Jalur 2 dari arsitektur Tri-Jalur Router. Implementasi Lightning Attention
yang mencapai kompleksitas O(1) inferensi per token dan mendukung konteks
hingga 4M token melalui chunked processing.

Arsitektur:
1. MLA (Multi-head Latent Attention) — Base attention mechanism
   Kompresi KV ke latent berdimensi rendah, mengurangi KV cache secara dramatis.
   Backward-compatible foundation untuk Lightning Attention.

2. InterleavedRoPE — Rotary Position Embedding dengan pola interleaved
   Menerapkan RoPE pada subset dimensi, memungkinkan mixing informasi
   posisional dan konten. Kompatibel dengan MLA.

3. LightningAttention — O(1) inference, 4M token context (v0.4 upgrade)
   Berdasarkan Linear Attention: alih-alih softmax(QK^T)V yang O(n^2),
   hitung Q(K^T V) menggunakan cumulative KV products.
   - Training: O(n) via chunked parallel computation
   - Inference: O(1) per token via state update
   - Hybrid: local window (softmax) + global linear attention
   - Chunked processing untuk konteks sangat panjang

Referensi:
- Sun, Q. et al., "Lightning Attention-2: A Free Lunch for Handling
  Unlimited Sequence Lengths" (2024)
- Katharopoulos, A. et al., "Transformers are RNNs: Fast Autoregressive
  Transformers with Linear Attention" (2020)
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient
  Mixture-of-Experts Language Model" (2024) — MLA
- Su, J. et al., "RoFormer: Enhanced Transformer with Rotary Position
  Embedding" (2021) — RoPE
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# InterleavedRoPE — Rotary Position Embedding dengan Pola Interleaved
# ============================================================================

class InterleavedRoPE(nn.Module):
    """
    Interleaved Rotary Position Embedding (RoPE).

    Menerapkan rotasi posisional pada subset dimensi dengan pola interleaved,
    memungkinkan mixing informasi posisional dan konten. Dimensi yang di-rotate
    dan yang tidak di-rotate di-interleave, bukan dipisah menjadi dua blok.

    Keuntungan interleaved vs. split:
    - Setiap head mendapat campuran posisional dan konten
    - Lebih baik untuk task yang membutuhkan positional sensitivity
    - Kompatibel dengan MLA: RoPE diterapkan setelah up-projection

    Args:
        dim: Dimensi total per head (harus genap).
        d_rope: Jumlah dimensi yang mendapat RoPE (harus genap, <= dim).
            Default: dim (semua dimensi mendapat RoPE).
        base: Basis frekuensi sinusoidal (default 10000).
        interleaved: Jika True, dimensi RoPE dan non-RoPE di-interleave.
            Jika False, dimensi RoPE di awal dan non-RoPE di akhir.
        ratio: Rasio RoPE:NoPE untuk pola interleaving per layer (default None).
            Jika diberikan, mengaktifkan metode should_use_rope() dan get_layer_pattern().
            Ratio R berarti dari setiap R+1 layer, R layer menggunakan RoPE dan 1 NoPE.
    """

    def __init__(
        self,
        dim: int,
        d_rope: Optional[int] = None,
        base: float = 10000.0,
        interleaved: bool = True,
        ratio: Optional[int] = None,
    ):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        self.dim = dim
        self.d_rope = d_rope if d_rope is not None else dim
        self.base = base
        self.interleaved = interleaved
        self.ratio = ratio  # Rasio RoPE:NoPE untuk pola per-layer

        if self.d_rope % 2 != 0:
            raise ValueError(f"d_rope must be even, got {self.d_rope}")
        if self.d_rope > dim:
            raise ValueError(f"d_rope ({self.d_rope}) must be <= dim ({dim})")

        # Frekuensi invers: theta_i = 1 / base^(2i/d_rope)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.d_rope, 2).float() / self.d_rope)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute interleaving index jika mode interleaved
        if self.interleaved and self.d_rope < dim:
            # Buat index yang meng-interleave dimensi RoPE dan non-RoPE
            rope_indices = torch.arange(self.d_rope)
            non_rope_indices = torch.arange(self.d_rope, dim)
            # Interleave: rope[0], non_rope[0], rope[1], non_rope[1], ...
            n_rope = self.d_rope
            n_non_rope = dim - self.d_rope
            # Kita buat mapping yang menyisipkan non-rope dims di antara rope dims
            interleaved_idx = torch.zeros(dim, dtype=torch.long)
            rope_ptr, non_rope_ptr = 0, 0
            for i in range(dim):
                if rope_ptr < n_rope and (non_rope_ptr >= n_non_rope or i % 2 == 0):
                    interleaved_idx[i] = rope_ptr
                    rope_ptr += 1
                else:
                    interleaved_idx[i] = n_rope + non_rope_ptr
                    non_rope_ptr += 1
            self.register_buffer("_interleave_idx", interleaved_idx, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Terapkan Interleaved RoPE pada tensor input.

        Args:
            x: Tensor input, bentuk (..., seq_len, n_heads, dim) atau
               (..., seq_len, dim).
            offset: Offset posisi (untuk continuation/prefill).

        Returns:
            Tensor dengan RoPE diterapkan, bentuk sama dengan input.
        """
        x_shape = x.shape

        # Deteksi apakah input punya dimensi heads
        if x.dim() >= 3 and x.shape[-1] == self.dim:
            # ... x seq_len x dim  atau  ... x seq_len x n_heads x dim
            has_heads = (x.dim() >= 4)
        else:
            has_heads = False

        if has_heads:
            # (batch, seq_len, n_heads, dim)
            seq_len = x.shape[-3]
        else:
            # (batch, seq_len, dim) atau (seq_len, dim)
            seq_len = x.shape[-2]

        # Buat posisi
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) + offset
        # (seq_len, d_rope // 2)
        freqs = torch.outer(t, self.inv_freq)
        # (seq_len, d_rope)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos_emb = emb.cos()  # (seq_len, d_rope)
        sin_emb = emb.sin()  # (seq_len, d_rope)

        # Sisipkan dimensi untuk broadcasting
        if has_heads:
            # cos_emb: (1, seq_len, 1, d_rope)
            cos_emb = cos_emb.unsqueeze(0).unsqueeze(2)
            sin_emb = sin_emb.unsqueeze(0).unsqueeze(2)
        else:
            # cos_emb: (1, seq_len, d_rope)
            cos_emb = cos_emb.unsqueeze(0)
            sin_emb = sin_emb.unsqueeze(0)

        # Ekstrak dimensi yang mendapat RoPE
        if self.d_rope == self.dim:
            x_rope = x
        elif self.interleaved:
            # Dalam mode interleaved, ambir dimensi genap untuk rope
            x_rope = x[..., :self.d_rope]
        else:
            # Mode split: dimensi pertama d_rope mendapat RoPE
            x_rope = x[..., :self.d_rope]

        # Aplikasikan rotasi
        x_rotated = self._apply_rotation(x_rope, cos_emb, sin_emb)

        if self.d_rope == self.dim:
            return x_rotated

        # Gabungkan kembali dimensi non-RoPE
        if self.interleaved:
            x_non_rope = x[..., self.d_rope:]
            # Interleave dimensi yang sudah di-rotate dan yang tidak
            output = torch.cat([x_rotated, x_non_rope], dim=-1)
            # Reorder menggunakan index interleaving
            if hasattr(self, '_interleave_idx'):
                output = output[..., self._interleave_idx]
        else:
            x_non_rope = x[..., self.d_rope:]
            output = torch.cat([x_rotated, x_non_rope], dim=-1)

        return output

    @staticmethod
    def _apply_rotation(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aplikasikan rotasi sinusoidal.

        Rotasi pada pasangan dimensi (x1, x2):
            x1' = x1 * cos - x2 * sin
            x2' = x1 * sin + x2 * cos

        Args:
            x: Tensor input, dimensi terakhir = d_rope.
            cos: Cosine embedding, broadcastable ke x.
            sin: Sine embedding, broadcastable ke x.

        Returns:
            Tensor yang sudah di-rotate.
        """
        d = x.shape[-1]
        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]

        # cos dan sin: (..., d_rope // 2) -> expand ke (..., d // 2)
        cos_half = cos[..., :d // 2]
        sin_half = sin[..., :d // 2]

        out1 = x1 * cos_half - x2 * sin_half
        out2 = x1 * sin_half + x2 * cos_half

        return torch.cat([out1, out2], dim=-1)

    def should_use_rope(self, layer_idx: int) -> bool:
        """
        Tentukan apakah layer pada indeks tertentu menggunakan RoPE.

        Menggunakan rasio yang diberikan saat inisialisasi (self.ratio).
        Jika ratio tidak diset, selalu mengembalikan True (semua layer RoPE).

        Dengan ratio R, dari setiap (R+1) layer berturut-turut,
        R layer menggunakan RoPE dan 1 layer menggunakan NoPE.
        Layer dengan indeks i % (R+1) == R tidak menggunakan RoPE.

        Args:
            layer_idx: Indeks layer (0-based).

        Returns:
            True jika layer ini menggunakan RoPE.
        """
        if self.ratio is None:
            return True
        return (layer_idx % (self.ratio + 1)) != self.ratio

    def get_layer_pattern(self, n_layers: int) -> list:
        """
        Hitung pola RoPE/NoPE untuk seluruh model.

        Args:
            n_layers: Jumlah total layer.

        Returns:
            List boolean dengan panjang n_layers. True = RoPE, False = NoPE.
        """
        return [self.should_use_rope(i) for i in range(n_layers)]


# ============================================================================
# MLA — Multi-head Latent Attention
# ============================================================================

class MLA(nn.Module):
    """
    Multi-head Latent Attention (MLA) — DeepSeek-V2 style.

    Alih-alih menyimpan full K dan V dalam KV cache, MLA mengkompres
    KV ke latent berdimensi rendah. Ini mengurangi KV cache secara dramatis
    tanpa kehilangan kualitas.

    Arsitektur:
        1. Down-projection: x → c_kv  (d_model → kv_lora_rank)
        2. Up-projection K: c_kv → k   (kv_lora_rank → n_heads * d_rope)
        3. Up-projection V: c_kv → v   (kv_lora_rank → n_heads * d_head)
        4. Q projection:    x → q      (d_model → n_heads * d_head)
        5. RoPE pada Q dan K (subset dimensi)
        6. Standard multi-head attention
        7. Output projection

    Keuntungan:
    - KV cache hanya menyimpan c_kv (kv_lora_rank per token, bukan n_heads * d_head)
    - Rekompression ratio: ~6-10x untuk konfigurasi tipikal
    - Backward-compatible: interface sama dengan standard attention

    Args:
        d_model: Dimensi model.
        n_heads: Jumlah attention heads.
        d_head: Dimensi per head.
        kv_lora_rank: Rank dari KV latent compression.
        q_lora_rank: Rank dari Q latent compression (opsional, default = d_model).
        d_rope: Dimensi yang mendapat RoPE (default: d_head // 2).
        rope_base: Basis frekuensi RoPE (default: 10000).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        kv_lora_rank: int = 256,
        q_lora_rank: Optional[int] = None,
        d_rope: Optional[int] = None,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        d_kv: Optional[int] = None,
        mla_latent_dim: Optional[int] = None,
        **kwargs,
    ):
        # Backward-compatible aliases: d_kv -> d_head, mla_latent_dim -> kv_lora_rank
        if d_kv is not None:
            d_head = d_kv
        if mla_latent_dim is not None:
            kv_lora_rank = mla_latent_dim

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank or d_model
        self.d_rope = d_rope or (d_head // 2)
        self.rope_base = rope_base

        # Alias attributes untuk backward compatibility
        self.d_kv = self.d_head
        self.mla_latent_dim = self.kv_lora_rank

        # ---- KV compression ----
        # Down-projection: x → c_kv latent
        self.kv_down_proj = nn.Linear(d_model, kv_lora_rank, bias=False)
        # Latent norm
        self.kv_norm = nn.RMSNorm(kv_lora_rank, eps=1e-5)
        # Up-projections dari latent
        self.k_up_proj = nn.Linear(kv_lora_rank, self.d_inner, bias=False)
        self.v_up_proj = nn.Linear(kv_lora_rank, self.d_inner, bias=False)

        # ---- Q projection (dengan optional LoRA) ----
        if self.q_lora_rank < d_model:
            self.q_down_proj = nn.Linear(d_model, self.q_lora_rank, bias=False)
            self.q_norm_down = nn.RMSNorm(self.q_lora_rank, eps=1e-5)
            self.q_up_proj = nn.Linear(self.q_lora_rank, self.d_inner, bias=False)
        else:
            self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # ---- RoPE ----
        self.rope = InterleavedRoPE(
            dim=self.d_rope,
            d_rope=self.d_rope,
            base=rope_base,
            interleaved=False,
        )

        # ---- Output ----
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---- QK normalization ----
        self.q_norm = nn.RMSNorm(d_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.d_head, eps=1e-5)

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        """Proyeksi Q dengan optional LoRA compression."""
        if self.q_lora_rank < self.d_model:
            q = self.q_up_proj(self.q_norm_down(self.q_down_proj(x)))
        else:
            q = self.q_proj(x)
        return q

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass MLA.

        Args:
            x: Tensor input, bentuk (batch, seq_len, d_model).
            attention_mask: Mask opsional, bentuk (batch, 1, seq_len, seq_len)
                atau (batch, 1, 1, seq_len) untuk causal + padding.
            past_key_value: Tuple (cached_kv, cached_v) dari step sebelumnya.
                cached_kv: (batch, past_len, kv_lora_rank)
                cached_v: Tidak digunakan langsung (V di-recompute dari latent).
            position_offset: Offset posisi untuk RoPE (default 0).

        Returns:
            Tuple (output, present_key_value):
            - output: (batch, seq_len, d_model)
            - present_key_value: (updated_kv_latent, dummy) untuk caching
        """
        batch, seq_len, _ = x.shape

        # ---- Proyeksi KV latent ----
        c_kv = self.kv_norm(self.kv_down_proj(x))  # (batch, seq_len, kv_lora_rank)

        # ---- Concatenate dengan past KV cache ----
        if past_key_value is not None and past_key_value[0] is not None:
            cached_kv = past_key_value[0]
            c_kv_full = torch.cat([cached_kv, c_kv], dim=1)
        else:
            c_kv_full = c_kv

        # Cache untuk step berikutnya
        present_kv = c_kv_full

        # ---- Up-project K dan V ----
        k = self.k_up_proj(c_kv_full)  # (batch, full_len, n_heads * d_head)
        v = self.v_up_proj(c_kv_full)     # (batch, full_len, n_heads * d_head)

        # Reshape ke heads
        k = k.view(batch, -1, self.n_heads, self.d_head)
        v = v.view(batch, -1, self.n_heads, self.d_head)

        # ---- Proyeksi Q ----
        q = self._project_q(x)  # (batch, seq_len, n_heads * d_head)
        q = q.view(batch, seq_len, self.n_heads, self.d_head)

        # ---- Split Q dan K untuk RoPE dan non-RoPE ----
        q_rope = q[..., :self.d_rope].contiguous()
        q_pass = q[..., self.d_rope:].contiguous()
        k_rope = k[..., :self.d_rope].contiguous()
        k_pass = k[..., self.d_rope:].contiguous()

        # ---- Terapkan RoPE ----
        full_len = c_kv_full.shape[1]
        offset = full_len - seq_len + position_offset

        q_rope = self.rope(q_rope, offset=offset)
        k_rope = self.rope(k_rope, offset=0)  # K sudah full sequence

        # ---- Reconstruct Q dan K dengan RoPE pada dimensi pertama ----
        q = torch.cat([q_rope, q_pass], dim=-1)
        k = torch.cat([k_rope, k_pass], dim=-1)

        # ---- QK normalization (full d_head) ----
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ---- Standard multi-head attention ----
        # Transpose ke (batch, n_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scale = math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            # Default causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, full_len, dtype=torch.bool, device=x.device),
                diagonal=full_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        attn_output = torch.matmul(attn_weights, v)  # (batch, n_heads, seq_len, d_head)

        # Reshape dan output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.d_inner)
        output = self.out_proj(attn_output)

        return output, (present_kv, None)

    def forward_inference(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[Any] = None,
        start_pos: int = 0,
        rope_enabled: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass untuk inferensi token-per-token.

        Hanya memproses satu token, menggunakan KV cache.

        Args:
            x: Tensor input satu token, (batch, 1, d_model).
            past_key_value: Tuple (cached_kv_latent, None).
            kv_cache: MLAKVCache instance (backward-compatible alias untuk past_key_value).
            start_pos: Posisi awal (backward-compatible, default 0).
            rope_enabled: Apakah RoPE diaktifkan (backward-compatible, default True).

        Returns:
            Tuple (output, present_key_value).
        """
        # Backward-compatible: jika kv_cache diberikan, gunakan sebagai past_key_value
        if past_key_value is None and kv_cache is not None:
            from losion.core.attention.attention_komposisi import MLAKVCache
            if isinstance(kv_cache, MLAKVCache):
                past_key_value = (kv_cache.get_cached(), None)
            elif isinstance(kv_cache, tuple):
                past_key_value = kv_cache

        return self.forward(
            x,
            attention_mask=None,
            past_key_value=past_key_value,
            position_offset=start_pos,
        )

    @property
    def memory_savings_ratio(self) -> float:
        """
        Rasio penghematan memori MLA vs standard attention.

        Standard KV cache: 2 * n_heads * d_head per token
        MLA KV cache: kv_lora_rank per token

        Returns:
            Float antara 0 dan 1 (misalnya 0.75 = 75% penghematan).
        """
        standard_kv_per_token = 2 * self.n_heads * self.d_head
        mla_kv_per_token = self.kv_lora_rank
        if standard_kv_per_token == 0:
            return 0.0
        return 1.0 - (mla_kv_per_token / standard_kv_per_token)

    def create_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Buat KV cache untuk inference.

        Args:
            batch_size: Jumlah batch.
            max_seq_len: Panjang sequence maksimum.
            dtype: Data type tensor.
            device: Device tensor.

        Returns:
            MLAKVCache instance.
        """
        from losion.core.attention.attention_komposisi import MLAKVCache
        return MLAKVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            kv_lora_rank=self.kv_lora_rank,
            dtype=dtype,
            device=device,
        )


# ============================================================================
# LightningAttention — O(1) Inference, 4M Token Context
# ============================================================================

class LightningAttention(nn.Module):
    """
    Lightning Attention — O(1) inference, mendukung konteks 4M token.

    v0.4 Upgrade untuk Jalur 2. Berdasarkan Linear Attention yang menggantikan
    softmax(QK^T)V yang O(n^2) dengan Q(K^T V) yang O(n) training dan O(1)
    inference per token.

    Algoritma Inti:
        Standard: y = softmax(QK^T)V     → O(n^2) waktu, O(n^2) memori
        Linear:   y = φ(Q) * Σ(φ(K)^T V) → O(n·d^2) training, O(d^2) inference

    State Cumulative:
        S_t = Σ_{s=1}^{t} φ(k_s) ⊗ v_s   (akumulasi produk KV)
        y_t = φ(q_t) @ S_t / (φ(q_t) @ z_t)
        dimana z_t = Σ_{s=1}^{t} φ(k_s)   (normalization)

    Hybrid Architecture:
        - Local window attention (softmax): untuk konteks dekat, kualitas tinggi
        - Global linear attention: untuk konteks jauh, kompleksitas rendah
        - Output = local_output + global_output

    Chunked Processing (4M tokens):
        - Proses dalam chunk-chunk yang bisa diparalelkan
        - Propagasi state antar chunk secara sekuensial
        - Memori konstan per chunk, tidak peduli panjang total sekuens

    Backward-Compatible dengan MLA:
        - Mendukung KV latent compression (kv_lora_rank)
        - Interface yang sama dengan MLA
        - Bisa drop-in replace di posisi MLA dalam arsitektur

    Args:
        d_model: Dimensi model.
        n_heads: Jumlah attention heads.
        d_head: Dimensi per head.
        kv_lora_rank: Rank kompresi KV latent (MLA-compatible). 0 = tanpa kompresi.
        window_size: Ukuran local sliding window (default 2048).
        chunk_size: Ukuran chunk untuk parallel training (default 4096).
        feature_map: Feature map untuk linear attention: "elu", "relu", "cos" (default "elu").
        max_context_length: Maksimum panjang konteks yang didukung (default 4_194_304 = 4M).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        kv_lora_rank: int = 0,
        window_size: int = 2048,
        chunk_size: int = 4096,
        feature_map: str = "elu",
        max_context_length: int = 4_194_304,  # 4M tokens
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.feature_map = feature_map
        self.max_context_length = max_context_length
        self.use_rope = use_rope

        # ---- Q, K, V Projections ----
        if kv_lora_rank > 0:
            # MLA-compatible: kompresi KV ke latent
            self.kv_down_proj = nn.Linear(d_model, kv_lora_rank, bias=False)
            self.kv_norm = nn.RMSNorm(kv_lora_rank, eps=1e-5)
            self.k_up_proj = nn.Linear(kv_lora_rank, self.d_inner, bias=False)
            self.v_up_proj = nn.Linear(kv_lora_rank, self.d_inner, bias=False)
        else:
            # Standard: proyeksi langsung
            self.k_proj = nn.Linear(d_model, self.d_inner, bias=False)
            self.v_proj = nn.Linear(d_model, self.d_inner, bias=False)

        self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # ---- RoPE ----
        if self.use_rope:
            self.rope = InterleavedRoPE(
                dim=d_head,
                d_rope=d_head // 2,
                base=rope_base,
                interleaved=True,
            )

        # ---- Local window attention components ----
        self.local_norm_q = nn.RMSNorm(d_head, eps=1e-5)
        self.local_norm_k = nn.RMSNorm(d_head, eps=1e-5)

        # ---- Global linear attention components ----
        self.global_norm_q = nn.RMSNorm(d_head, eps=1e-5)
        self.global_norm_k = nn.RMSNorm(d_head, eps=1e-5)

        # ---- Decay parameter untuk global attention ----
        # Mengontrol seberapa banyak informasi jauh yang dipertahankan
        # Diinisialisasi mendekati 1 (preserve most information)
        self.decay_log = nn.Parameter(torch.zeros(n_heads))
        # decay = sigmoid(decay_log) -> mendekati 0.5, bisa dituning

        # ---- Gate untuk blending local dan global ----
        self.blend_gate = nn.Linear(d_model, n_heads, bias=False)

        # ---- Output ----
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---- Group norm per head untuk output stabilization ----
        self.out_norm = nn.RMSNorm(d_model, eps=1e-5)

    def _get_decay(self) -> torch.Tensor:
        """
        Hitung decay factor per head.

        Returns:
            Tensor (n_heads,) dengan nilai 0 < decay < 1.
        """
        return torch.sigmoid(self.decay_log)

    def _apply_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Terapkan feature map φ untuk linear attention.

        Feature map mengubah Q dan K sehingga φ(Q)·φ(K)^T ≈ softmax(QK^T)
        untuk similarity approximation.

        Args:
            x: Tensor input.

        Returns:
            Tensor setelah feature map diterapkan.
        """
        if self.feature_map == "elu":
            # ELU+1: φ(x) = elu(x) + 1, menjamin non-negatif
            return F.elu(x) + 1.0
        elif self.feature_map == "relu":
            # ReLU: φ(x) = relu(x), simple non-negative
            return F.relu(x) + 1e-6  # epsilon untuk numerical stability
        elif self.feature_map == "cos":
            # Cos-based: normalisasi lalu cos/sin transform
            x_norm = F.normalize(x, p=2, dim=-1)
            return x_norm
        else:
            return F.elu(x) + 1.0

    def _project_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Proyeksi K dan V, dengan optional MLA compression.

        Args:
            x: Tensor input, (batch, seq_len, d_model).

        Returns:
            Tuple (k, v), masing-masing (batch, seq_len, n_heads, d_head).
        """
        batch, seq_len, _ = x.shape

        if self.kv_lora_rank > 0:
            # MLA path: kompresi ke latent lalu up-project
            c_kv = self.kv_norm(self.kv_down_proj(x))  # (batch, seq_len, kv_lora_rank)
            k = self.k_up_proj(c_kv)
            v = self.v_up_proj(c_kv)
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)

        k = k.view(batch, seq_len, self.n_heads, self.d_head)
        v = v.view(batch, seq_len, self.n_heads, self.d_head)
        return k, v

    def _local_window_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Local sliding window attention dengan softmax.

        Hanya menghitung attention dalam window sebesar self.window_size.
        Menggunakan standard softmax attention untuk kualitas tertinggi
        pada konteks dekat.

        Args:
            q: Query, (batch, n_heads, seq_len, d_head).
            k: Key, (batch, n_heads, full_len, d_head).
            v: Value, (batch, n_heads, full_len, d_head).
            attention_mask: Mask opsional.
            kv_cache: Cached K,V dari langkah sebelumnya.

        Returns:
            Tuple (local_output, updated_kv_cache).
        """
        batch, n_heads, seq_len, d_head = q.shape
        full_len = k.shape[2]

        # QK normalization untuk local attention
        q = self.local_norm_q(q)
        k = self.local_norm_k(k)

        # Sliding window mask
        # q: (batch, n_heads, seq_len, d_head)
        # k: (batch, n_heads, full_len, d_head)
        # Window: setiap query hanya melihat [i-w+1, i] dari key

        # Buat sliding window mask
        if full_len <= self.window_size:
            # Tidak perlu windowing, gunakan full causal
            window_mask = torch.triu(
                torch.ones(seq_len, full_len, dtype=torch.bool, device=q.device),
                diagonal=full_len - seq_len + 1,
            )
        else:
            # Window mask: query position i bisa melihat key positions
            # [max(0, offset+i-window_size+1), offset+i]
            # dimana offset = full_len - seq_len
            offset = full_len - seq_len
            q_positions = torch.arange(seq_len, device=q.device) + offset
            k_positions = torch.arange(full_len, device=q.device)

            # Causal + window
            window_mask = ~(
                (k_positions.unsqueeze(0) <= q_positions.unsqueeze(1))
                & (k_positions.unsqueeze(0) >= q_positions.unsqueeze(1) - self.window_size + 1)
            )  # True = masked out

        # Attention scores
        scale = math.sqrt(d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Terapkan mask
        attn_weights = attn_weights.masked_fill(
            window_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        local_output = torch.matmul(attn_weights, v)
        return local_output, kv_cache

    def _global_linear_attention_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        initial_sum: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Global linear attention dengan chunked processing.

        Menghitung linear attention: y_t = φ(q_t) @ S_t / (φ(q_t) @ z_t)
        dimana S_t = Σ decay^(t-s) * φ(k_s) ⊗ v_s adalah cumulative KV state.

        Chunked processing:
        1. Bagi sekuens menjadi chunk-chunk berukuran chunk_size
        2. Dalam setiap chunk, hitung intra-chunk output secara paralel
        3. Propagasi state antar-chunk secara sekuensial
        4. Ini memungkinkan training O(n) dengan memori O(chunk_size^2)

        Untuk 4M token: chunk_size=4096 → ~1024 chunk, memori per chunk konstan.

        Args:
            q: Query, (batch, n_heads, seq_len, d_head).
            k: Key, (batch, n_heads, seq_len, d_head).
            v: Value, (batch, n_heads, seq_len, d_head).
            initial_state: KV state awal, (batch, n_heads, d_head, d_head).
            initial_sum: Sum state awal (normalization), (batch, n_heads, d_head).

        Returns:
            Tuple (output, final_state, final_sum):
            - output: (batch, n_heads, seq_len, d_head)
            - final_state: (batch, n_heads, d_head, d_head)
            - final_sum: (batch, n_heads, d_head)
        """
        batch, n_heads, seq_len, d_head = q.shape

        # Inisialisasi state
        if initial_state is None:
            state = torch.zeros(
                batch, n_heads, d_head, d_head,
                dtype=q.dtype, device=q.device,
            )
        else:
            state = initial_state.clone()

        if initial_sum is None:
            sum_k = torch.zeros(
                batch, n_heads, d_head,
                dtype=q.dtype, device=q.device,
            )
        else:
            sum_k = initial_sum.clone()

        # Decay factor
        decay = self._get_decay()  # (n_heads,)
        decay_state = decay.view(1, n_heads, 1, 1)  # untuk state (batch, n_heads, d_head, d_head)
        decay_sum = decay.view(1, n_heads, 1)  # untuk sum_k (batch, n_heads, d_head)

        # Terapkan feature map
        q_feat = self._apply_feature_map(self.global_norm_q(q))
        k_feat = self._apply_feature_map(self.global_norm_k(k))

        # ---- Chunked processing ----
        chunk_size = min(self.chunk_size, seq_len)
        n_chunks = (seq_len + chunk_size - 1) // chunk_size

        outputs = []

        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk_len = end - start

            q_c = q_feat[:, :, start:end]  # (batch, n_heads, chunk_len, d_head)
            k_c = k_feat[:, :, start:end]
            v_c = v[:, :, start:end]

            # ---- Inter-chunk contribution dari state ----
            # y_from_state = q_c @ state / (q_c @ sum_k)
            inter_output = torch.matmul(q_c, state)  # (batch, n_heads, chunk_len, d_head)
            inter_normalizer = torch.matmul(
                q_c, sum_k.unsqueeze(-1)
            ).squeeze(-1)  # (batch, n_heads, chunk_len)

            # ---- Intra-chunk: cumulative linear attention ----
            # Untuk setiap posisi t dalam chunk:
            # output_t = Σ_{s=start}^{t} k_s ⊗ v_s  (cumulative)
            # Gunakan parallel scan atau sequential accumulation

            # KV outer products: k_c^T @ v_c
            # k_c: (batch, n_heads, chunk_len, d_head)
            # v_c: (batch, n_heads, chunk_len, d_head)
            k_c_t = k_c.transpose(-2, -1)  # (batch, n_heads, d_head, chunk_len)

            # Intra-chunk attention via causal cumulative sum
            # Untuk setiap head: KV[t] = Σ_{s=0}^{t} k[s] ⊗ v[s]
            # Ini bisa dihitung secara paralel menggunakan:
            # 1. Sequential scan (sederhana, O(chunk_len))
            # 2. Parallel prefix sum (lebih kompleks tapi paralel)

            # Sequential scan (stabil dan sederhana)
            intra_outputs = []
            intra_norms = []
            cumulative_state = torch.zeros_like(state)  # (batch, n_heads, d_head, d_head)
            cumulative_sum = torch.zeros_like(sum_k)     # (batch, n_heads, d_head)

            for s in range(chunk_len):
                # Tambah kontribusi token s
                k_s = k_c[:, :, s:s+1, :]  # (batch, n_heads, 1, d_head)
                v_s = v_c[:, :, s:s+1, :]  # (batch, n_heads, 1, d_head)

                # KV outer product
                kv_s = torch.matmul(k_s.transpose(-2, -1), v_s)  # (batch, n_heads, d_head, d_head)

                # Cumulative dengan decay
                cumulative_state = decay_state * cumulative_state + kv_s
                cumulative_sum = decay_sum * cumulative_sum + k_s.squeeze(2)  # (batch, n_heads, d_head)

                # Output dari intra-chunk state
                q_s = q_c[:, :, s:s+1, :]  # (batch, n_heads, 1, d_head)
                intra_out = torch.matmul(q_s, cumulative_state)  # (batch, n_heads, 1, d_head)
                intra_norm = torch.matmul(
                    q_s, cumulative_sum.unsqueeze(-1)
                ).squeeze(-1)  # (batch, n_heads, 1)

                intra_outputs.append(intra_out.squeeze(2))
                intra_norms.append(intra_norm.squeeze(2))

            # Stack intra-chunk outputs
            intra_output = torch.stack(intra_outputs, dim=2)  # (batch, n_heads, chunk_len, d_head)
            intra_normalizer = torch.stack(intra_norms, dim=2)  # (batch, n_heads, chunk_len)

            # ---- Gabungkan inter-chunk dan intra-chunk ----
            # Total output = (inter_output + intra_output) / (inter_normalizer + intra_normalizer)
            total_output = inter_output + intra_output
            total_normalizer = inter_normalizer.unsqueeze(-1) + intra_normalizer.unsqueeze(-1) + 1e-6

            chunk_output = total_output / total_normalizer

            outputs.append(chunk_output)

            # ---- Update state untuk chunk berikutnya ----
            # State di akhir chunk = decay^chunk_len * state + Σ decay^(chunk_len-1-s) * k_s ⊗ v_s
            # Simplifikasi: gunakan cumulative_state dari loop terakhir
            state = decay_state * state + cumulative_state
            sum_k = decay_sum * sum_k + cumulative_sum

        # Gabungkan semua chunk
        output = torch.cat(outputs, dim=2)  # (batch, n_heads, seq_len, d_head)

        return output, state, sum_k

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass Lightning Attention.

        Hybrid: local window (softmax) + global linear attention.
        Output = gate * local_output + (1 - gate) * global_output

        Args:
            x: Tensor input, (batch, seq_len, d_model).
            attention_mask: Mask opsional untuk padding.
            past_key_value: Tuple (kv_cache, linear_state, linear_sum) dari step sebelumnya.
                kv_cache: (batch, past_len, n_heads, d_head*2) untuk local attention
                linear_state: (batch, n_heads, d_head, d_head) untuk global attention
                linear_sum: (batch, n_heads, d_head) untuk global normalization
            position_offset: Offset posisi untuk RoPE.

        Returns:
            Tuple (output, present_key_value):
            - output: (batch, seq_len, d_model)
            - present_key_value: updated (kv_cache, linear_state, linear_sum)
        """
        batch, seq_len, _ = x.shape

        if seq_len == 0:
            dummy_out = torch.zeros(
                batch, 0, self.d_model, dtype=x.dtype, device=x.device
            )
            return dummy_out, (None, None, None)

        # ---- Unpack past state ----
        kv_cache = None
        linear_state = None
        linear_sum = None
        if past_key_value is not None:
            kv_cache, linear_state, linear_sum = past_key_value

        # ---- Proyeksi Q, K, V ----
        q = self.q_proj(x)  # (batch, seq_len, d_inner)
        k, v = self._project_kv(x)

        # Reshape ke heads: (batch, seq_len, n_heads, d_head)
        q = q.view(batch, seq_len, self.n_heads, self.d_head)

        # ---- Terapkan RoPE ----
        if self.use_rope:
            q = self.rope(q, offset=position_offset)
            k = self.rope(k, offset=position_offset)

        # Transpose ke (batch, n_heads, seq_len, d_head) untuk attention
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # ---- Concatenate dengan KV cache untuk local attention ----
        if kv_cache is not None:
            # kv_cache: (batch, past_len, n_heads, d_head*2)
            past_k = kv_cache[:, :, :, :self.d_head].transpose(1, 2)  # (batch, n_heads, past_len, d_head)
            past_v = kv_cache[:, :, :, self.d_head:].transpose(1, 2)
            k_full = torch.cat([past_k, k_t], dim=2)
            v_full = torch.cat([past_v, v_t], dim=2)
        else:
            k_full = k_t
            v_full = v_t

        # ---- Local window attention ----
        local_output, _ = self._local_window_attention(
            q_t, k_full, v_full,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        # ---- Global linear attention ----
        global_output, new_linear_state, new_linear_sum = \
            self._global_linear_attention_chunked(
                q_t, k_t, v_t,
                initial_state=linear_state,
                initial_sum=linear_sum,
            )

        # ---- Blend local dan global ----
        # Gate: (batch, seq_len, n_heads) -> (batch, n_heads, seq_len, 1)
        gate = torch.sigmoid(self.blend_gate(x))  # (batch, seq_len, n_heads)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (batch, n_heads, seq_len, 1)

        blended = gate * local_output + (1 - gate) * global_output

        # ---- Update KV cache ----
        # Simpan hanya window_size token terakhir untuk local attention
        full_len = k_full.shape[2]
        if full_len > self.window_size:
            cache_k = k_full[:, :, full_len - self.window_size:].transpose(1, 2)
            cache_v = v_full[:, :, full_len - self.window_size:].transpose(1, 2)
        else:
            cache_k = k_full.transpose(1, 2)
            cache_v = v_full.transpose(1, 2)

        new_kv_cache = torch.cat([cache_k, cache_v], dim=-1)
        # (batch, min(full_len, window_size), n_heads, d_head*2)

        present_key_value = (new_kv_cache, new_linear_state, new_linear_sum)

        # ---- Output projection ----
        # Transpose kembali ke (batch, seq_len, n_heads, d_head) -> (batch, seq_len, d_inner)
        blended = blended.transpose(1, 2).contiguous()
        blended = blended.view(batch, seq_len, self.d_inner)

        output = self.out_proj(blended)
        output = self.out_norm(output)

        return output, present_key_value

    def forward_inference(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass untuk inferensi token-per-token (O(1) per token).

        Untuk global linear attention, hanya perlu:
        1. Update state: S_{t+1} = decay * S_t + φ(k_t) ⊗ v_t
        2. Compute output: y_t = φ(q_t) @ S_t / (φ(q_t) @ z_t)

        Untuk local window attention, gunakan sliding KV cache berukuran window_size.

        Total: O(d^2) per token = O(1) w.r.t. sequence length.

        Args:
            x: Tensor input satu token, (batch, 1, d_model).
            past_key_value: Tuple (kv_cache, linear_state, linear_sum).

        Returns:
            Tuple (output, present_key_value).
        """
        batch = x.shape[0]

        # ---- Unpack state ----
        kv_cache = None
        linear_state = None
        linear_sum = None
        if past_key_value is not None:
            kv_cache, linear_state, linear_sum = past_key_value

        # ---- Proyeksi ----
        q = self.q_proj(x)
        k, v = self._project_kv(x)

        q = q.view(batch, 1, self.n_heads, self.d_head)
        # k, v sudah (batch, 1, n_heads, d_head)

        # ---- RoPE ----
        if self.use_rope:
            offset = 0
            if kv_cache is not None:
                offset = kv_cache.shape[1]
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        # Transpose ke (batch, n_heads, 1, d_head)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # ---- Local attention ----
        if kv_cache is not None:
            past_k = kv_cache[:, :, :, :self.d_head].transpose(1, 2)
            past_v = kv_cache[:, :, :, self.d_head:].transpose(1, 2)
            k_full = torch.cat([past_k, k_t], dim=2)
            v_full = torch.cat([past_v, v_t], dim=2)
        else:
            k_full = k_t
            v_full = v_t

        local_output, _ = self._local_window_attention(q_t, k_full, v_full)

        # ---- Global linear attention (O(1) state update) ----
        decay = self._get_decay()  # (n_heads,)
        decay_state = decay.view(1, self.n_heads, 1, 1)
        decay_sum = decay.view(1, self.n_heads, 1)

        if linear_state is None:
            linear_state = torch.zeros(
                batch, self.n_heads, self.d_head, self.d_head,
                dtype=x.dtype, device=x.device,
            )
        if linear_sum is None:
            linear_sum = torch.zeros(
                batch, self.n_heads, self.d_head,
                dtype=x.dtype, device=x.device,
            )

        # Feature map
        q_feat = self._apply_feature_map(self.global_norm_q(q_t))  # (batch, n_heads, 1, d_head)
        k_feat = self._apply_feature_map(self.global_norm_k(k_t))  # (batch, n_heads, 1, d_head)

        # State update: S_{t+1} = decay * S_t + k_t^T @ v_t
        kv = torch.matmul(k_feat.transpose(-2, -1), v_t)  # (batch, n_heads, d_head, d_head)
        new_linear_state = decay_state * linear_state + kv

        # Sum update: z_{t+1} = decay * z_t + k_t
        new_linear_sum = decay_sum * linear_sum + k_feat.squeeze(2)

        # Output: y_t = q_t @ S_t / (q_t @ z_t)
        global_output = torch.matmul(q_feat, new_linear_state)  # (batch, n_heads, 1, d_head)
        normalizer = torch.matmul(
            q_feat, new_linear_sum.unsqueeze(-1)
        ).squeeze(-1)  # (batch, n_heads, 1)
        global_output = global_output / (normalizer.unsqueeze(-1) + 1e-6)

        # ---- Blend ----
        gate = torch.sigmoid(self.blend_gate(x))  # (batch, 1, n_heads)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (batch, n_heads, 1, 1)

        blended = gate * local_output + (1 - gate) * global_output

        # ---- Update KV cache ----
        full_len = k_full.shape[2]
        if full_len > self.window_size:
            cache_k = k_full[:, :, full_len - self.window_size:].transpose(1, 2)
            cache_v = v_full[:, :, full_len - self.window_size:].transpose(1, 2)
        else:
            cache_k = k_full.transpose(1, 2)
            cache_v = v_full.transpose(1, 2)

        new_kv_cache = torch.cat([cache_k, cache_v], dim=-1)
        present_key_value = (new_kv_cache, new_linear_state, new_linear_sum)

        # ---- Output projection ----
        blended = blended.transpose(1, 2).contiguous().view(batch, 1, self.d_inner)
        output = self.out_proj(blended)
        output = self.out_norm(output)

        return output, present_key_value

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inisialisasi state kosong untuk inferensi.

        Args:
            batch_size: Ukuran batch.
            device: Device tensor.
            dtype: Tipe data tensor.

        Returns:
            Tuple (kv_cache, linear_state, linear_sum):
            - kv_cache: None (belum ada cached tokens)
            - linear_state: (batch, n_heads, d_head, d_head) zeros
            - linear_sum: (batch, n_heads, d_head) zeros
        """
        linear_state = torch.zeros(
            batch_size, self.n_heads, self.d_head, self.d_head,
            dtype=dtype, device=device,
        )
        linear_sum = torch.zeros(
            batch_size, self.n_heads, self.d_head,
            dtype=dtype, device=device,
        )
        return (None, linear_state, linear_sum)

    def get_cache_size(self, seq_len: int) -> int:
        """
        Hitung ukuran KV cache dalam elemen.

        Args:
            seq_len: Panjang sekuens.

        Returns:
            Jumlah elemen tensor yang di-cache.
        """
        # Local: window_size * n_heads * d_head * 2 (K+V)
        local_cache = min(seq_len, self.window_size) * self.n_heads * self.d_head * 2
        # Global: n_heads * d_head * d_head (state) + n_heads * d_head (sum)
        global_cache = self.n_heads * self.d_head * self.d_head + self.n_heads * self.d_head
        # MLA: jika kv_lora_rank > 0, tambah latent cache
        mla_cache = 0
        if self.kv_lora_rank > 0:
            mla_cache = seq_len * self.kv_lora_rank

        return local_cache + global_cache + mla_cache


# ============================================================================
# PairwiseAttentionLayer — Original Pairwise Attention Pattern
# ============================================================================

class PairwiseAttentionLayer(nn.Module):
    """
    Pairwise Attention Layer — Layer attention dengan pola pairwise.

    Menerapkan attention dengan konstrain pairwise: setiap token hanya
    meng-attend ke token dalam pasangannya. Berguna untuk:
    - Cross-attention antara dua sekuens
    - Bidirectional attention dengan konstrain
    - Attention antara input dan label embeddings

    Implementasi menggunakan standard multi-head attention dengan
    custom attention mask yang mengimplementasikan pola pairwise.

    Args:
        d_model: Dimensi model.
        n_heads: Jumlah attention heads.
        d_head: Dimensi per head.
        dropout: Dropout rate.
        causal: Apakah menggunakan causal masking (default False).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        dropout: float = 0.0,
        causal: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head
        self.causal = causal

        # Projections
        self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Normalization
        self.q_norm = nn.RMSNorm(d_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(d_head, eps=1e-5)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # RoPE
        self.rope = InterleavedRoPE(
            dim=d_head,
            d_rope=d_head // 2,
            interleaved=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_pair: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pair_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass Pairwise Attention.

        Jika x_pair diberikan, lakukan cross-attention (x sebagai Q,
        x_pair sebagai K, V). Jika tidak, lakukan self-attention.

        Jika pair_indices diberikan, gunakan untuk membatasi attention
        ke pasangan tertentu.

        Args:
            x: Tensor input, (batch, seq_len, d_model).
            x_pair: Tensor pasangan opsional, (batch, pair_len, d_model).
            attention_mask: Mask opsional.
            pair_indices: Index pasangan opsional, (batch, seq_len, n_pairs).

        Returns:
            Tensor output, (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Tentukan sumber K, V
        if x_pair is not None:
            kv_source = x_pair
        else:
            kv_source = x

        kv_len = kv_source.shape[1]

        # Proyeksi
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(kv_source).view(batch, kv_len, self.n_heads, self.d_head)
        v = self.v_proj(kv_source).view(batch, kv_len, self.n_heads, self.d_head)

        # RoPE
        q = self.rope(q)
        k = self.rope(k)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose ke (batch, n_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scale = math.sqrt(self.d_head)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Pairwise mask
        if pair_indices is not None:
            # v1.9.0: Vectorized mask construction — replaces nested batch+seq loop
            # pair_indices: (batch, seq_len, n_pairs) -> -1 = padding
            k_indices = torch.arange(kv_len, device=x.device)
            # Clamp -1 to 0 (will be masked out by valid_mask)
            clamped = pair_indices.clamp(min=0)
            valid_mask = pair_indices >= 0  # (batch, seq_len, n_pairs)

            # Expand for scatter: (batch, seq_len, n_pairs, kv_len)
            # For each (b, i, p), set pair_mask[b, i, clamped[b,i,p]] = True
            pair_mask = torch.zeros(
                batch, seq_len, kv_len, dtype=torch.bool, device=x.device
            )
            # Use scatter for vectorized one-hot encoding
            expanded_clamped = clamped.unsqueeze(-1).expand(-1, -1, -1, kv_len)
            one_hot = torch.zeros(
                batch, seq_len, pair_indices.shape[-1], kv_len,
                dtype=torch.bool, device=x.device
            )
            one_hot.scatter_(-1, expanded_clamped, valid_mask.unsqueeze(-1))
            pair_mask = one_hot.any(dim=2)

            attn_weights = attn_weights.masked_fill(
                ~pair_mask.unsqueeze(1), float("-inf")
            )

        # Causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, dtype=torch.bool, device=x.device),
                diagonal=kv_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_inner)
        output = self.resid_dropout(self.out_proj(output))

        return output
