"""
FG2-GDN — Fine-Grained Gated DeltaNet untuk Losion Framework.

Enhanced Gated DeltaNet dengan fine-grained gating yang menyediakan
in-context learning yang lebih baik.

Perbedaan dengan GatedDeltaNet standar:
- GatedDeltaNet: gate(hidden_state) → satu gate value per head
- FG2-GDN: gate(hidden_state, position, head) → gate value per-head,
  per-position, memberikan kontrol retensi yang jauh lebih granular

Mengapa fine-grained gating?
1. In-context learning memerlukan kontrol retensi yang presisi
2. Gate yang sama untuk semua posisi terlalu kaku — beberapa token
   penting dan harus dipertahankan, yang lain bisa dilupakan
3. Per-head gating memungkinkan head yang berbeda mengkhususkan diri
   pada pola retensi yang berbeda
4. Learnable temperature per head memungkinkan head yang berbeda
   memiliki "selectivity" yang berbeda

Arsitektur FG2-GDN:
- Per-head, per-position gating (bukan hanya per-head)
- Learnable gate temperature per head
- Mendukung sigmoid dan softmax gating
- Same interface sebagai GatedDeltaNet (drop-in replacement)
- Kompatibel dengan SSMTerpaduLayer interleaving

Delta rule (sama seperti GatedDeltaNet):
    delta_t = beta_t * (q_t @ k_t.T - alpha_t * prev_state)
    new_state = prev_state + delta_t
    output_t = new_state @ v_t

Fine-grained gating:
    beta_t[h, s] = gate_fn(hidden_t[s] @ W_beta + b_beta[h]) / temperature[h]
    alpha_t[h, s] = gate_fn(hidden_t[s] @ W_alpha + b_alpha[h]) / temperature[h]

Dimana h = head index, s = position index.

Komponen:
1. FineGrainedGate — Fine-grained gating mechanism
2. FG2GDN — Fine-Grained Gated DeltaNet layer

Referensi:
- Yang, S. et al., "Gated Linear Attention" (2024)
- Schlag, I. et al., "Linear Transformers Are Secretly Fast Weight Programmers" (2021)
- Losion Framework — GatedDeltaNet implementation

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# FineGrainedGate — Per-head, per-position gating
# ---------------------------------------------------------------------------

class FineGrainedGate(nn.Module):
    """
    Fine-grained gating mechanism untuk DeltaNet.

    Menghasilkan gate values per-head, per-position (bukan hanya per-head
    seperti GatedDeltaNet standar). Ini memungkinkan kontrol retensi
    yang lebih granular: beberapa posisi dalam sequence bisa memiliki
    gate yang berbeda, memungkinkan model mempertahankan informasi
    penting dan melupakan yang tidak penting secara selektif.

    Gate types:
    - "sigmoid": Standard sigmoid gating, output di (0, 1).
      Cocok untuk kontrol retensi yang smooth.
    - "softmax": Softmax gating, output di (0, 1) dan sum = 1.
      Cocok untuk attention-style selective retention.

    Temperature:
    - Learnable temperature per head mengontrol "selectivity" gating.
    - Temperature rendah → gate lebih sharp (lebih selektif).
    - Temperature tinggi → gate lebih smooth (lebih uniform).

    Args:
        d_model: Dimensi model input.
        n_heads: Jumlah attention heads.
        gate_type: Tipe gating ("sigmoid" atau "softmax").
        init_temperature: Initial temperature untuk semua heads.
        use_position_bias: Jika True, tambahkan learnable position bias
            ke gate values.
        max_seq_len: Panjang sequence maksimum (untuk position bias).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        gate_type: str = "sigmoid",
        init_temperature: float = 1.0,
        use_position_bias: bool = True,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.gate_type = gate_type
        self.max_seq_len = max_seq_len

        # ---- Gate projections ----
        # Beta gate: mengontrol seberapa besar update
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        # Alpha gate: mengontrol decay rate
        self.alpha_proj = nn.Linear(d_model, n_heads, bias=True)

        # ---- Learnable temperature per head ----
        # Di-log space untuk memastikan positif
        self.log_beta_temperature = nn.Parameter(
            torch.full((n_heads,), math.log(init_temperature))
        )
        self.log_alpha_temperature = nn.Parameter(
            torch.full((n_heads,), math.log(init_temperature))
        )

        # ---- Position bias (opsional) ----
        self.use_position_bias = use_position_bias
        if use_position_bias:
            # Position bias untuk beta dan alpha
            self.beta_pos_bias = nn.Parameter(torch.zeros(max_seq_len, n_heads))
            self.alpha_pos_bias = nn.Parameter(torch.zeros(max_seq_len, n_heads))

        # ---- Alpha offset: inisialisasi agar alpha mendekati 1 ----
        self.alpha_offset = nn.Parameter(torch.ones(n_heads) * 0.9)

    @property
    def beta_temperature(self) -> torch.Tensor:
        """Temperature untuk beta gate (selalu positif)."""
        return torch.exp(self.log_beta_temperature)

    @property
    def alpha_temperature(self) -> torch.Tensor:
        """Temperature untuk alpha gate (selalu positif)."""
        return torch.exp(self.log_alpha_temperature)

    def _apply_gate_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Terapkan fungsi gating."""
        if self.gate_type == "sigmoid":
            return torch.sigmoid(x)
        elif self.gate_type == "softmax":
            # Softmax gating: normalize per position
            return F.softmax(x, dim=-1)
        else:
            return torch.sigmoid(x)

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hitung fine-grained gate values.

        Menghasilkan per-head, per-position gate values untuk beta
        (update magnitude) dan alpha (decay rate).

        Args:
            hidden: Hidden state, (batch, seq_len, d_model).

        Returns:
            Tuple (beta, alpha):
            - beta: (batch, seq_len, n_heads) — gate untuk update magnitude.
            - alpha: (batch, seq_len, n_heads) — gate untuk decay rate.
        """
        batch, seq_len, _ = hidden.shape

        # ---- Beta gate: per-position, per-head ----
        beta_logits = self.beta_proj(hidden)  # (batch, seq_len, n_heads)

        # Tambahkan position bias
        if self.use_position_bias:
            pos_indices = torch.arange(
                seq_len, device=hidden.device, dtype=torch.long
            )
            pos_indices = pos_indices.clamp(max=self.max_seq_len - 1)
            beta_logits = beta_logits + self.beta_pos_bias[pos_indices].unsqueeze(0)

        # Terapkan temperature scaling
        beta_temperature = self.beta_temperature.unsqueeze(0).unsqueeze(0)  # (1, 1, n_heads)
        beta = self._apply_gate_fn(beta_logits / beta_temperature)  # (batch, seq_len, n_heads)

        # ---- Alpha gate: per-position, per-head ----
        alpha_logits = self.alpha_proj(hidden)  # (batch, seq_len, n_heads)

        # Tambahkan alpha offset (bias toward keeping state)
        alpha_logits = alpha_logits + self.alpha_offset.unsqueeze(0).unsqueeze(0)

        # Tambahkan position bias
        if self.use_position_bias:
            alpha_logits = alpha_logits + self.alpha_pos_bias[pos_indices].unsqueeze(0)

        # Terapkan temperature scaling
        alpha_temperature = self.alpha_temperature.unsqueeze(0).unsqueeze(0)  # (1, 1, n_heads)
        alpha = self._apply_gate_fn(alpha_logits / alpha_temperature)  # (batch, seq_len, n_heads)

        return beta, alpha


# ---------------------------------------------------------------------------
# FG2-GDN Parallel Computation
# ---------------------------------------------------------------------------

def fg2_gdn_forward_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Komputasi FG2-GDN secara paralel untuk training.

    Sama seperti delta_net_forward_parallel tetapi menggunakan
    fine-grained beta dan alpha (per-position, per-head).

    Delta rule dengan fine-grained gating:
        new_state = alpha[t,h] * prev_state[h] + beta[t,h] * k[t,h]^T @ v[t,h]
        output[t,h] = q[t,h] @ new_state[h]

    Implementasi chunk-based untuk efisiensi:
    1. Hitung intra-chunk attention secara paralel
    2. Propagasi state antar-chunk via sequential scan
    3. Hitung output per chunk dengan fine-grained gating

    Args:
        q: Query, (batch, seq_len, n_heads, d_head).
        k: Key, (batch, seq_len, n_heads, d_head).
        v: Value, (batch, seq_len, n_heads, d_head).
        beta: Fine-grained beta gate, (batch, seq_len, n_heads).
        alpha: Fine-grained alpha gate, (batch, seq_len, n_heads).
        initial_state: State awal opsional, (batch, n_heads, d_head, d_head).
        chunk_size: Ukuran chunk untuk komputasi paralel.

    Returns:
        Tuple (output, final_state):
        - output: (batch, seq_len, n_heads, d_head)
        - final_state: (batch, n_heads, d_head, d_head)
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
        q_chunk = q_c[:, c]
        k_chunk = k_c[:, c]
        v_chunk = v_c[:, c]
        beta_chunk = beta_c[:, c]
        alpha_chunk = alpha_c[:, c]

        # ---- Intra-chunk: Linear Attention ----
        chunk_s = chunk_size
        causal_mask = torch.triu(
            torch.ones(chunk_s, chunk_s, dtype=torch.bool, device=q.device),
            diagonal=1,
        )

        q_feat = F.elu(q_chunk) + 1
        k_feat = F.elu(k_chunk) + 1

        q_h = rearrange(q_feat, "b s h d -> b h s d")
        k_h = rearrange(k_feat, "b s h d -> b h s d")
        v_h = rearrange(v_chunk, "b s h d -> b h s d")

        attn_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(d_head)
        attn_scores = attn_scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        intra_output = torch.matmul(attn_weights, v_h)

        # ---- Delta rule update dengan fine-grained gating ----
        # Store intermediate states so each position uses its own state
        intermediate_states = []
        for s in range(chunk_s):
            # Per-head, per-position alpha dan beta
            a_t = alpha_chunk[:, s].unsqueeze(-1).unsqueeze(-1)  # (batch, n_heads, 1, 1)
            b_t = beta_chunk[:, s].unsqueeze(-1).unsqueeze(-1)   # (batch, n_heads, 1, 1)

            k_t = k_h[:, :, s:s+1, :]
            v_t = v_h[:, :, s:s+1, :]

            kv_outer = torch.matmul(k_t.transpose(-2, -1), v_t)

            # Fine-grained delta rule:
            # new_state = alpha * prev_state + beta * k^T @ v
            state = a_t * state + b_t * kv_outer
            intermediate_states.append(state.clone())

        # ---- Inter-chunk output dari state ----
        # Each position s uses the state at position s (not the final state)
        state_outputs = torch.cat([
            torch.matmul(q_h[:, :, s:s+1, :], intermediate_states[s])
            for s in range(chunk_s)
        ], dim=2)  # (batch, n_heads, chunk_s, d_head)

        # Gabungkan dengan position-dependent weighting
        position_weight = torch.linspace(
            0, 1, chunk_s, device=q.device, dtype=q.dtype
        ).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        y_chunk = (1 - position_weight) * state_outputs + position_weight * intra_output
        y_chunk = rearrange(y_chunk, "b h s d -> b s h d")
        outputs.append(y_chunk)

    y = torch.cat(outputs, dim=1)

    if pad_len > 0:
        y = y[:, :seq_len]

    return y, state


# ---------------------------------------------------------------------------
# FG2GDN — Fine-Grained Gated DeltaNet
# ---------------------------------------------------------------------------

class FG2GDN(nn.Module):
    """
    Fine-Grained Gated DeltaNet (FG2-GDN).

    Enhanced version dari GatedDeltaNet dengan fine-grained gating:
    - Per-head, per-position gate values (bukan hanya per-head)
    - Learnable temperature per head
    - Better in-context learning through finer retention control
    - Same interface sebagai GatedDeltaNet (drop-in replacement)

    Fine-grained gating memungkinkan model:
    - Mempertahankan informasi penting secara selektif per posisi
    - Melupakan informasi yang tidak relevan secara selektif
    - Head yang berbeda memiliki pola retensi yang berbeda
    - Temperature per head mengontrol selectivity

    Kompatibel dengan SSMTerpaduLayer interleaving:
    - Same forward() interface: (input, initial_state) → (output, final_state)
    - Same forward_inference() interface
    - Can be used as drop-in replacement for GatedDeltaNet

    Args:
        d_model: Dimensi model input.
        n_heads: Jumlah attention heads.
        d_head: Dimensi per head.
        chunk_size: Ukuran chunk untuk komputasi paralel.
        use_bias: Apakah menggunakan bias di proyeksi.
        gate_type: Tipe gating ("sigmoid" atau "softmax").
        use_position_bias: Apakah menggunakan position bias di gating.
        init_temperature: Initial temperature untuk gate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        chunk_size: int = 256,
        use_bias: bool = False,
        gate_type: str = "sigmoid",
        use_position_bias: bool = True,
        init_temperature: float = 1.0,
        gate_fn: str = "swish",
        **kwargs,
    ) -> None:
        """
        Inisialisasi FG2-GDN layer.

        Args:
            d_model: Dimensi model input.
            n_heads: Jumlah attention heads.
            d_head: Dimensi per head.
            chunk_size: Ukuran chunk untuk komputasi paralel.
            use_bias: Apakah menggunakan bias di proyeksi.
            gate_type: Tipe gating ("sigmoid" atau "softmax").
            use_position_bias: Apakah menggunakan position bias di gating.
            init_temperature: Initial temperature untuk gate.
            gate_fn: Fungsi output gating ("swish", "sigmoid", "relu").
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

        # ---- Fine-grained gating mechanism ----
        self.fine_gate = FineGrainedGate(
            d_model=d_model,
            n_heads=n_heads,
            gate_type=gate_type,
            init_temperature=init_temperature,
            use_position_bias=use_position_bias,
        )

        # ---- Output gating ----
        self.gate_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # ---- Output proyeksi ----
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)

        # ---- Normalisasi ----
        self.norm = nn.RMSNorm(self.d_inner, eps=1e-5)
        self.q_norm = nn.RMSNorm(d_head, eps=1e-5)
        self.k_norm = nn.RMSNorm(d_head, eps=1e-5)

    def _apply_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Terapkan fungsi output gating."""
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
        Forward pass FG2-GDN.

        Args:
            input: Tensor input, (batch, seq_len, d_model).
            initial_state: State awal opsional, (batch, n_heads, d_head, d_head).

        Returns:
            Tuple (output, final_state):
            - output: (batch, seq_len, d_model)
            - final_state: (batch, n_heads, d_head, d_head)
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
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        # Reshape ke heads
        q = rearrange(q, "b s (h d) -> b s h d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.n_heads)

        # Normalisasi Q dan K (QK-norm)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ---- Fine-grained gating ----
        beta, alpha = self.fine_gate(input)
        # beta: (batch, seq_len, n_heads) — per-position, per-head
        # alpha: (batch, seq_len, n_heads) — per-position, per-head

        # ---- DeltaNet Computation ----
        if seq_len == 1:
            # Inference mode: O(1) update dengan fine-grained gates
            y, final_state = self._forward_single_token(
                q, k, v, beta, alpha, initial_state
            )
        else:
            # Training mode: parallel chunk computation
            y, final_state = fg2_gdn_forward_parallel(
                q, k, v, beta, alpha, initial_state, self.chunk_size
            )

        # ---- Output gating ----
        y = rearrange(y, "b s h d -> b s (h d)")
        gate = self._apply_gate(self.gate_proj(input))
        y = y * gate

        # ---- Normalisasi dan proyeksi output ----
        y = self.norm(y)
        output = self.out_proj(y)

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

        O(1) per token: update state dan hitung output
        dengan fine-grained per-head gating.

        Args:
            q, k, v: Masing-masing (batch, 1, n_heads, d_head).
            beta: Fine-grained beta, (batch, 1, n_heads).
            alpha: Fine-grained alpha, (batch, 1, n_heads).
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
        alpha_t = alpha.squeeze(1)  # (batch, n_heads)

        # Fine-grained delta rule update:
        # new_state[h] = alpha[h] * prev_state[h] + beta[h] * k[h]^T @ v[h]
        a = alpha_t.unsqueeze(-1).unsqueeze(-1)  # (batch, n_heads, 1, 1)
        b = beta_t.unsqueeze(-1).unsqueeze(-1)   # (batch, n_heads, 1, 1)

        kv = torch.matmul(
            k_t.unsqueeze(-1), v_t.unsqueeze(-2)
        )  # (batch, n_heads, d_head, d_head)

        new_state = a * state + b * kv

        # Output: q @ new_state
        y = torch.matmul(
            q_t.unsqueeze(-2), new_state
        ).squeeze(-2)  # (batch, n_heads, d_head)

        y = y.unsqueeze(1)  # (batch, 1, n_heads, d_head)

        return y, new_state

    def forward_inference(
        self,
        input: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass untuk inferensi token-per-token.

        Drop-in compatible dengan GatedDeltaNet.forward_inference().

        Args:
            input: Tensor input satu token, (batch, 1, d_model).
            state: State opsional, (batch, n_heads, d_head, d_head).

        Returns:
            Tuple (output, new_state).
        """
        return self.forward(input, initial_state=state)
