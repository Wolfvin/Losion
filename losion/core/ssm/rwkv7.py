"""
RWKV-7 WKV (Weighted Key-Value) Implementation untuk Losion Framework.

Implementasi layer RWKV-7 WKV berbasis pure PyTorch.
Mendukung mode training (paralel) dan inference (sekuensial O(1) per token).

Referensi:
- Peng, B. et al., "RWKV-7: The Next Generation RWKV Architecture" (2025)
- Mekanisme WKV menghitung recurrence data-dependent yang memungkinkan
  pemrosesan sekuensial O(1) per token saat inferensi.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# WKV Core: Weighted Key-Value Recurrence
# ---------------------------------------------------------------------------

def wkv_forward_parallel(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Komputasi WKV paralel untuk training.

    Mekanisme WKV menghitung recurrence data-dependent:
    - wkv_state: akumulator weighted key-value
    - sum_state: akumulator bobot normalisasi

    Untuk setiap token t:
    1. Decay state: state = state * exp(w_t)
    2. Tambahkan kontribusi token saat ini: state += k_t * v_t
    3. Hitung output: y_t = r_t @ (state + u * k_t * v_t) / (sum + u * k_t^2 + eps)

    Args:
        r: Receptance (query), bentuk (batch, seq_len, d_inner).
        k: Key, bentuk (batch, seq_len, d_head).
        v: Value, bentuk (batch, seq_len, d_head).
        w: Decay weight per token, bentuk (batch, seq_len, d_head).
           Nilai negatif (semakin negatif = decay lebih cepat).
        u: Bonus per-posisi (learned), bentuk (d_head,).
        initial_state: Tuple opsional (wkv_state, sum_state) dari step sebelumnya.

    Returns:
        Tuple (output, final_state):
        - output: bentuk (batch, seq_len, d_inner)
        - final_state: (wkv_state, sum_state) tuple
    """
    batch, seq_len, d_inner = r.shape
    d_head = k.shape[-1]

    # Inisialisasi state
    if initial_state is not None:
        wkv_state, sum_state = initial_state
    else:
        wkv_state = torch.zeros(
            batch, d_head, dtype=r.dtype, device=r.device
        )
        sum_state = torch.zeros(
            batch, d_head, dtype=r.dtype, device=r.device
        )

    # Decay: w dalam domain log, konversi ke linear
    # w < 0, jadi decay = exp(w) ∈ (0, 1)
    decay = torch.exp(w)  # (batch, seq_len, d_head)

    # ---- Sequential WKV scan (paralel across batch) ----
    outputs = []

    for t in range(seq_len):
        # Current token
        r_t = r[:, t, :]  # (batch, d_inner)
        k_t = k[:, t, :]  # (batch, d_head)
        v_t = v[:, t, :]  # (batch, d_head)
        d_t = decay[:, t, :]  # (batch, d_head)

        # Decay state
        wkv_state = wkv_state * d_t  # (batch, d_head)
        sum_state = sum_state * d_t  # (batch, d_head)

        # WKV computation:
        # numerator = wkv_state + u * k_t * v_t
        # denominator = sum_state + u * k_t^2 + eps
        # output = r_t @ (numerator / denominator)

        # Simple linear attention style:
        # wkv = (wkv_state + u * k_t * v_t) / (sum_state + u * k_t^2 + eps)
        # y_t = r_t @ wkv

        # Untuk d_inner != d_head, gunakan proyeksi
        # Simplifikasi: gunakan element-wise dengan broadcast
        kv_t = k_t * v_t  # (batch, d_head)
        k_sq = k_t * k_t  # (batch, d_head)

        numerator = wkv_state + u.unsqueeze(0) * kv_t  # (batch, d_head)
        denominator = sum_state + u.unsqueeze(0) * k_sq + 1e-8  # (batch, d_head)

        wkv_val = numerator / denominator  # (batch, d_head)

        # Output: y_t = r_t * wkv_val (element-wise jika d_inner == d_head)
        if d_inner == d_head:
            y_t = r_t * wkv_val  # (batch, d_inner)
        else:
            # Truncate atau pad wkv_val ke d_inner
            if d_head > d_inner:
                y_t = r_t * wkv_val[..., :d_inner]
            else:
                wkv_padded = F.pad(wkv_val, (0, d_inner - d_head))
                y_t = r_t * wkv_padded

        outputs.append(y_t)

        # Update state: tambahkan kontribusi token saat ini
        wkv_state = wkv_state + kv_t  # (batch, d_head)
        sum_state = sum_state + k_sq  # (batch, d_head)

    output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
    final_state = (wkv_state, sum_state)

    return output, final_state


def wkv_forward_inference(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    state: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Komputasi WKV sekuensial O(1) per token untuk inferensi.

    Args:
        r: Receptance, bentuk (batch, 1, d_inner).
        k: Key, bentuk (batch, 1, d_head).
        v: Value, bentuk (batch, 1, d_head).
        w: Decay, bentuk (batch, 1, d_head).
        u: Bonus posisi, bentuk (d_head,).
        state: (wkv_state, sum_state) dari step sebelumnya.

    Returns:
        Tuple (output, new_state).
    """
    # Squeeze sequence dim
    r_t = r.squeeze(1)  # (batch, d_inner)
    k_t = k.squeeze(1)  # (batch, d_head)
    v_t = v.squeeze(1)  # (batch, d_head)
    d_t = torch.exp(w.squeeze(1))  # (batch, d_head)

    wkv_state, sum_state = state

    # Decay
    wkv_state = wkv_state * d_t
    sum_state = sum_state * d_t

    # WKV computation
    kv_t = k_t * v_t
    k_sq = k_t * k_t

    numerator = wkv_state + u.unsqueeze(0) * kv_t
    denominator = sum_state + u.unsqueeze(0) * k_sq + 1e-8

    wkv_val = numerator / denominator

    d_inner = r.shape[-1]
    d_head = k.shape[-1]

    if d_inner == d_head:
        y = r_t * wkv_val
    elif d_head > d_inner:
        y = r_t * wkv_val[..., :d_inner]
    else:
        wkv_padded = F.pad(wkv_val, (0, d_inner - d_head))
        y = r_t * wkv_padded

    # Update state
    wkv_state = wkv_state + kv_t
    sum_state = sum_state + k_sq

    return y.unsqueeze(1), (wkv_state, sum_state)


# ---------------------------------------------------------------------------
# RWKV7WKV Layer
# ---------------------------------------------------------------------------

class RWKV7WKV(nn.Module):
    """
    RWKV-7 style WKV (Weighted Key-Value) recurrence layer.

    Fitur utama:
    - Rekurensi WKV data-dependent (O(1) per token)
    - Evolusi state dinamis
    - Token-shift simplifikasi dari RWKV-7
    - Representasi state yang ekspresif

    Mekanisme WKV menghitung:
        wkv_t = (a_t * wkv_{t-1} + b_t * v_t) / (a_t * sum_{t-1} + b_t)
    dimana a_t, b_t adalah bobot data-dependent.

    Hardware: Pure PyTorch, bekerja di CUDA/ROCm/CPU.
    """

    def __init__(
        self,
        d_model: int,
        d_head: int = 64,
        n_heads: int = None,
        d_inner: int = None,
        use_bias: bool = True,
        **kwargs,
    ):
        """
        Inisialisasi RWKV7WKV layer.

        Args:
            d_model: Dimensi model input.
            d_head: Dimensi per head WKV (default 64).
            n_heads: Jumlah head (default: d_model // d_head).
            d_inner: Dimensi inner (default: n_heads * d_head).
            use_bias: Apakah menggunakan bias.
        """
        super().__init__()

        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = n_heads or max(1, d_model // d_head)
        self.d_inner = d_inner or (self.n_heads * d_head)

        # ---- Proyeksi key, value, receptance ----
        self.key_proj = nn.Linear(d_model, self.n_heads * d_head, bias=use_bias)
        self.value_proj = nn.Linear(d_model, self.n_heads * d_head, bias=use_bias)
        self.receptance_proj = nn.Linear(d_model, self.d_inner, bias=use_bias)

        # ---- Decay parameter ----
        # w: data-dependent decay per head per token
        self.decay_proj = nn.Linear(d_model, self.n_heads * d_head, bias=False)

        # Inisialisasi decay: mulai dari nilai kecil negatif
        # Agar decay mendekati 1 (long memory)
        with torch.no_grad():
            init_w = -torch.randn(self.n_heads * d_head) * 0.1 - 5.0
        self.init_decay = nn.Parameter(init_w)

        # ---- Bonus posisi u ----
        # u: bobot bonus untuk posisi saat ini
        self.u = nn.Parameter(torch.randn(self.n_heads, d_head) * 0.1)

        # ---- Output gating ----
        self.gate_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # ---- Proyeksi output ----
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)

        # ---- Normalisasi output ----
        self.norm = nn.RMSNorm(self.d_inner, eps=1e-5)

    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass RWKV-7 WKV.

        Args:
            input: Tensor input, bentuk (batch, seq_len, d_model).
            initial_state: State awal opsional, tuple (wkv_state, sum_state).

        Returns:
            Tuple (output, final_state):
            - output: bentuk (batch, seq_len, d_model)
            - final_state: (wkv_state, sum_state) tuple
        """
        batch, seq_len, _ = input.shape

        # Handle edge cases
        if seq_len == 0:
            dummy_out = torch.zeros(
                batch, 0, self.d_model, dtype=input.dtype, device=input.device
            )
            if initial_state is not None:
                return dummy_out, initial_state
            else:
                dummy_state = (
                    torch.zeros(batch, self.d_head, dtype=input.dtype, device=input.device),
                    torch.zeros(batch, self.d_head, dtype=input.dtype, device=input.device),
                )
                return dummy_out, dummy_state

        # ---- Token-shift mixing ----
        if seq_len > 1:
            shifted = torch.cat(
                [torch.zeros(batch, 1, self.d_model, dtype=input.dtype, device=input.device), input[:, :-1, :]],
                dim=1,
            )
        else:
            shifted = torch.zeros_like(input)

        # Mix: additive mixing (RWKV-7 simplifikasi)
        mixed = input + shifted

        # ---- Proyeksi ----
        k = self.key_proj(mixed)  # (batch, seq_len, n_heads * d_head)
        v = self.value_proj(input)  # (batch, seq_len, n_heads * d_head)
        r = self.receptance_proj(mixed)  # (batch, seq_len, d_inner)

        # Decay: data-dependent
        w = self.decay_proj(mixed) + self.init_decay.unsqueeze(0).unsqueeze(0)
        # w: (batch, seq_len, n_heads * d_head) — nilai negatif

        # ---- Reshape ke heads ----
        k = rearrange(k, "b s (h d) -> b s h d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.n_heads)
        w = rearrange(w, "b s (h d) -> b s h d", h=self.n_heads)

        # ---- WKV Computation per head ----
        # Flatten heads untuk komputasi
        k_flat = rearrange(k, "b s h d -> b s (h d)")
        v_flat = rearrange(v, "b s h d -> b s (h d)")
        w_flat = rearrange(w, "b s h d -> b s (h d)")

        # Expand initial state jika perlu
        if initial_state is not None:
            init_state_flat = initial_state
        else:
            init_state_flat = None

        # Untuk single-token, gunakan inference mode
        if seq_len == 1:
            if initial_state is None:
                initial_state = (
                    torch.zeros(batch, self.n_heads * self.d_head, dtype=input.dtype, device=input.device),
                    torch.zeros(batch, self.n_heads * self.d_head, dtype=input.dtype, device=input.device),
                )
            y, final_state = wkv_forward_inference(
                r, k_flat, v_flat, w_flat, self.u.flatten(), initial_state
            )
        else:
            y, final_state = wkv_forward_parallel(
                r, k_flat, v_flat, w_flat, self.u.flatten(), init_state_flat
            )

        # ---- Output gating ----
        gate = torch.sigmoid(self.gate_proj(input))  # (batch, seq_len, d_inner)
        y = y * gate

        # ---- Normalisasi ----
        y = self.norm(y)

        # ---- Proyeksi output ----
        output = self.out_proj(y)  # (batch, seq_len, d_model)

        return output, final_state

    def forward_inference(
        self,
        input: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass untuk inferensi token-per-token O(1).

        Args:
            input: Tensor input satu token, bentuk (batch, 1, d_model).
            state: State WKV, tuple (wkv_state, sum_state).

        Returns:
            Tuple (output, new_state).
        """
        return self.forward(input, initial_state=state)
