"""
Gated DeltaNet Implementation untuk Losion Framework.

Implementasi layer Gated DeltaNet berbasis pure PyTorch.
Terinspirasi oleh Qwen3-Next, menggabungkan linear attention
dengan gating mechanism dan delta rule untuk incremental updates.

Referensi:
- Yang, S. et al., "Gated Linear Attention" (2024)
- Schlag, I. et al., "Linear Transformers Are Secretly Fast Weight Programmers" (2021)
- Delta rule: delta_t = beta_t * (q_t @ k_t.T - alpha_t * prev_state)
  menghasilkan pembaruan incremental yang lebih stabil daripada
  full attention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# DeltaNet Core: Delta Rule dengan Linear Attention
# ---------------------------------------------------------------------------

def delta_net_forward_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Komputasi Gated DeltaNet secara paralel untuk training.

    Delta rule:
        delta_t = beta_t * (q_t @ k_t.T - alpha_t * prev_state)
        new_state = prev_state + delta_t
        output_t = new_state @ v_t

    Diimplementasikan dengan chunk-based computation untuk efisiensi:
    1. Hitung KV products per chunk secara paralel
    2. Propagasi state antar-chunk via sequential scan
    3. Hitung output per chunk

    Args:
        q: Query, bentuk (batch, seq_len, n_heads, d_head).
        k: Key, bentuk (batch, seq_len, n_heads, d_head).
        v: Value, bentuk (batch, seq_len, n_heads, d_head).
        beta: Gate parameter, bentuk (batch, seq_len, n_heads).
              Mengontrol seberapa besar update delta.
        alpha: Decay parameter, bentuk (batch, seq_len, n_heads).
               Mengontrol seberapa banyak state sebelumnya dipertahankan.
        initial_state: State awal opsional, bentuk (batch, n_heads, d_head, d_head).
        chunk_size: Ukuran chunk untuk komputasi paralel.

    Returns:
        Tuple (output, final_state):
        - output: bentuk (batch, seq_len, n_heads, d_head)
        - final_state: bentuk (batch, n_heads, d_head, d_head)
    """
    batch, seq_len, n_heads, d_head = q.shape

    # Inisialisasi state
    if initial_state is None:
        state = torch.zeros(
            batch, n_heads, d_head, d_head,
            dtype=q.dtype, device=q.device,
        )
    else:
        state = initial_state.clone()

    # Padding untuk chunk
    pad_len = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))
        alpha = F.pad(alpha, (0, 0, 0, pad_len))

    padded_len = q.shape[1]
    n_chunks = padded_len // chunk_size

    # Reshape ke chunks
    q_c = rearrange(q, "b (c s) h d -> b c s h d", c=n_chunks)
    k_c = rearrange(k, "b (c s) h d -> b c s h d", c=n_chunks)
    v_c = rearrange(v, "b (c s) h d -> b c s h d", c=n_chunks)
    beta_c = rearrange(beta, "b (c s) h -> b c s h", c=n_chunks)
    alpha_c = rearrange(alpha, "b (c s) h -> b c s h", c=n_chunks)

    outputs = []

    for c in range(n_chunks):
        # Proses setiap chunk (paralel within chunk via matmul)
        q_chunk = q_c[:, c]  # (batch, chunk_size, n_heads, d_head)
        k_chunk = k_c[:, c]
        v_chunk = v_c[:, c]
        beta_chunk = beta_c[:, c]  # (batch, chunk_size, n_heads)
        alpha_chunk = alpha_c[:, c]

        # ---- Intra-chunk: Linear Attention ----
        # Hitung intra-chunk attention scores
        # q @ k^T: (batch, chunk_size, n_heads, d_head) @ (batch, chunk_size, n_heads, d_head).T
        # Untuk linear attention: gunakan kernel feature map
        # Simplifikasi: gunakan dot-product attention dengan causal mask

        chunk_s = chunk_size
        # Causal mask untuk intra-chunk
        causal_mask = torch.triu(
            torch.ones(chunk_s, chunk_s, dtype=torch.bool, device=q.device),
            diagonal=1,
        )  # True = masked

        # Elu-based feature map (linear attention kernel)
        q_feat = F.elu(q_chunk) + 1  # (batch, chunk_s, n_heads, d_head)
        k_feat = F.elu(k_chunk) + 1

        # Intra-chunk linear attention
        # Score: q_feat @ k_feat^T per head
        # Reshape: (batch, n_heads, chunk_s, d_head)
        q_h = rearrange(q_feat, "b s h d -> b h s d")
        k_h = rearrange(k_feat, "b s h d -> b h s d")
        v_h = rearrange(v_chunk, "b s h d -> b h s d")

        # Attention scores: (batch, n_heads, chunk_s, chunk_s)
        attn_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(d_head)

        # Terapkan causal mask
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Intra-chunk output: (batch, n_heads, chunk_s, d_head)
        intra_output = torch.matmul(attn_weights, v_h)

        # ---- Delta rule update untuk state ----
        # Store intermediate states so each position uses its own state
        intermediate_states = []
        for s in range(chunk_s):
            # Decay state
            a_t = alpha_chunk[:, s].unsqueeze(-1).unsqueeze(-1)  # (batch, n_heads, 1, 1)
            b_t = beta_chunk[:, s].unsqueeze(-1).unsqueeze(-1)  # (batch, n_heads, 1, 1)

            k_t = k_h[:, :, s:s+1, :]  # (batch, n_heads, 1, d_head)
            v_t = v_h[:, :, s:s+1, :]  # (batch, n_heads, 1, d_head)

            # Delta rule: new_state = alpha * prev_state + beta * k^T @ v
            # Full delta: delta = beta * (k^T @ v - alpha * prev_state)
            # Simplifikasi: new_state = alpha * state + beta * k^T @ v
            kv_outer = torch.matmul(k_t.transpose(-2, -1), v_t)  # (batch, n_heads, d_head, d_head)
            state = a_t * state + b_t * kv_outer
            intermediate_states.append(state.clone())

        # ---- Inter-chunk output dari state ----
        # Each position s uses the state at position s (not the final state)
        # q_h: (batch, n_heads, chunk_s, d_head)
        # intermediate_states[s]: (batch, n_heads, d_head, d_head)
        state_outputs = torch.cat([
            torch.matmul(q_h[:, :, s:s+1, :], intermediate_states[s])
            for s in range(chunk_s)
        ], dim=2)  # (batch, n_heads, chunk_s, d_head)

        # Gabungkan intra-chunk dan state output
        # Interpolasi: gunakan gating berdasarkan posisi
        # Awal chunk: lebih bergantung pada state
        # Akhir chunk: lebih bergantung pada intra-chunk attention
        position_weight = torch.linspace(
            0, 1, chunk_s, device=q.device, dtype=q.dtype
        ).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, chunk_s, 1)

        y_chunk = (1 - position_weight) * state_outputs + position_weight * intra_output
        # (batch, n_heads, chunk_s, d_head)

        y_chunk = rearrange(y_chunk, "b h s d -> b s h d")
        outputs.append(y_chunk)

    # Gabungkan semua chunk
    y = torch.cat(outputs, dim=1)  # (batch, padded_len, n_heads, d_head)

    # Hapus padding
    if pad_len > 0:
        y = y[:, :seq_len]

    return y, state


# ---------------------------------------------------------------------------
# GatedDeltaNet Layer
# ---------------------------------------------------------------------------

class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet layer (terinspirasi oleh Qwen3-Next).

    Fitur utama:
    - Linear attention dengan gating mechanism
    - In-context learning yang lebih baik dibanding SSM standar
    - Delta rule untuk pembaruan incremental
    - Kompleksitas O(n)

    Delta rule: sebagai ganti full attention, menggunakan:
        delta_t = beta_t * (q_t @ k_t.T - alpha_t * prev_state)
        new_state = prev_state + delta_t
        output = new_state @ v_t

    Gating mengontrol:
    - beta: seberapa besar update terhadap state
    - alpha: seberapa banyak state sebelumnya dipertahankan (decay)

    Hardware: Pure PyTorch, kompatibel CUDA/ROCm/CPU.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        chunk_size: int = 256,
        use_bias: bool = False,
        gate_fn: str = "swish",
        **kwargs,
    ):
        """
        Inisialisasi GatedDeltaNet layer.

        Args:
            d_model: Dimensi model input.
            n_heads: Jumlah attention heads.
            d_head: Dimensi per head.
            chunk_size: Ukuran chunk untuk komputasi paralel.
            use_bias: Apakah menggunakan bias di proyeksi.
            gate_fn: Fungsi gating ("swish", "sigmoid", "relu").
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_inner = n_heads * d_head
        self.chunk_size = chunk_size
        self.gate_fn = gate_fn

        # ---- Proyeksi Q, K, V ----
        self.q_proj = nn.Linear(d_model, self.d_inner, bias=use_bias)
        self.k_proj = nn.Linear(d_model, self.d_inner, bias=use_bias)
        self.v_proj = nn.Linear(d_model, self.d_inner, bias=use_bias)

        # ---- Gating parameters ----
        # beta: gate untuk update magnitude
        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)
        # alpha: gate untuk decay rate
        self.alpha_proj = nn.Linear(d_model, n_heads, bias=False)

        # Inisialisasi alpha agar mendekati 1 (preserve state)
        with torch.no_grad():
            nn.init.zeros_(self.alpha_proj.weight)
            # Bias towards keeping state
        self.alpha_offset = nn.Parameter(torch.ones(n_heads) * 0.9)

        # ---- Output gating ----
        self.gate_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # ---- Output proyeksi ----
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)

        # ---- Normalisasi ----
        self.norm = nn.RMSNorm(self.d_inner, eps=1e-5)
        self.q_norm = nn.RMSNorm(d_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(d_head, eps=1e-5)

    def _apply_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Terapkan fungsi gating."""
        if self.gate_fn == "swish":
            return F.silu(x)
        elif self.gate_fn == "sigmoid":
            return torch.sigmoid(x)
        elif self.gate_fn == "relu":
            return F.relu(x)
        else:
            return F.silu(x)

    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass Gated DeltaNet.

        Args:
            input: Tensor input, bentuk (batch, seq_len, d_model).
            initial_state: State awal opsional, bentuk (batch, n_heads, d_head, d_head).

        Returns:
            Tuple (output, final_state):
            - output: bentuk (batch, seq_len, d_model)
            - final_state: bentuk (batch, n_heads, d_head, d_head)
        """
        batch, seq_len, _ = input.shape

        # Handle edge cases
        if seq_len == 0:
            dummy_out = torch.zeros(
                batch, 0, self.d_model, dtype=input.dtype, device=input.device
            )
            dummy_state = (
                initial_state
                if initial_state is not None
                else torch.zeros(
                    batch, self.n_heads, self.d_head, self.d_head,
                    dtype=input.dtype, device=input.device,
                )
            )
            return dummy_out, dummy_state

        # ---- Proyeksi Q, K, V ----
        q = self.q_proj(input)  # (batch, seq_len, d_inner)
        k = self.k_proj(input)
        v = self.v_proj(input)

        # Reshape ke heads
        q = rearrange(q, "b s (h d) -> b s h d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.n_heads)

        # Normalisasi Q dan K (QK-norm)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ---- Gating parameters ----
        # beta: seberapa besar update (0-1)
        beta = torch.sigmoid(self.beta_proj(input))  # (batch, seq_len, n_heads)
        # alpha: seberapa banyak state dipertahankan (0-1)
        alpha = torch.sigmoid(
            self.alpha_proj(input) + self.alpha_offset.unsqueeze(0).unsqueeze(0)
        )  # (batch, seq_len, n_heads)

        # ---- DeltaNet Computation ----
        if seq_len == 1:
            # Inference mode: O(1) update
            y, final_state = self._forward_single_token(
                q, k, v, beta, alpha, initial_state
            )
        else:
            # Training mode: parallel chunk computation
            y, final_state = delta_net_forward_parallel(
                q, k, v, beta, alpha, initial_state, self.chunk_size
            )

        # ---- Output gating ----
        y = rearrange(y, "b s h d -> b s (h d)")  # (batch, seq_len, d_inner)
        gate = self._apply_gate(self.gate_proj(input))
        y = y * gate

        # ---- Normalisasi dan proyeksi output ----
        y = self.norm(y)
        output = self.out_proj(y)  # (batch, seq_len, d_model)

        return output, final_state

    def _forward_single_token(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        alpha: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass untuk token tunggal (inference mode).

        O(1) per token: update state dan hitung output.

        Args:
            q, k, v: Masing-masing (batch, 1, n_heads, d_head).
            beta, alpha: (batch, 1, n_heads).
            state: State sebelumnya, (batch, n_heads, d_head, d_head).

        Returns:
            Tuple (output, new_state).
        """
        batch = q.shape[0]

        if state is None:
            state = torch.zeros(
                batch, self.n_heads, self.d_head, self.d_head,
                dtype=q.dtype, device=q.device,
            )

        # Squeeze sequence dimension
        q_t = q.squeeze(1)  # (batch, n_heads, d_head)
        k_t = k.squeeze(1)
        v_t = v.squeeze(1)
        beta_t = beta.squeeze(1)  # (batch, n_heads)
        alpha_t = alpha.squeeze(1)

        # Delta rule update:
        # new_state = alpha * prev_state + beta * k^T @ v
        a = alpha_t.unsqueeze(-1).unsqueeze(-1)  # (batch, n_heads, 1, 1)
        b = beta_t.unsqueeze(-1).unsqueeze(-1)  # (batch, n_heads, 1, 1)

        kv = torch.matmul(
            k_t.unsqueeze(-1), v_t.unsqueeze(-2)
        )  # (batch, n_heads, d_head, d_head)

        new_state = a * state + b * kv

        # Output: q @ new_state
        # q: (batch, n_heads, d_head), state: (batch, n_heads, d_head, d_head)
        y = torch.matmul(
            q_t.unsqueeze(-2), new_state
        ).squeeze(-2)  # (batch, n_heads, d_head)

        # Reshape ke (batch, 1, n_heads, d_head)
        y = y.unsqueeze(1)

        return y, new_state

    def forward_inference(
        self,
        input: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass untuk inferensi token-per-token.

        Args:
            input: Tensor input satu token, bentuk (batch, 1, d_model).
            state: State opsional, bentuk (batch, n_heads, d_head, d_head).

        Returns:
            Tuple (output, new_state).
        """
        return self.forward(input, initial_state=state)
