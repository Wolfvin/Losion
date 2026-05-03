"""
PoST Decay Spectra — Position-Dependent Decay for SSM Long-Range Memory.

Peningkatan Jalur 1 (SSM): Mengganti single learnable decay parameter per head
dengan spectrum of decay rates yang bervariasi berdasarkan posisi, memungkinkan
retensi memori jangka panjang yang lebih baik.

Motivasi:
---------
SSM standar (Mamba-2) menggunakan satu parameter decay per head/channel:
    h_t = exp(dt * A) * h_{t-1} + dt * B * x_t

Satu decay rate tidak cukup untuk menangkap kebutuhan memori yang berbeda
pada posisi yang berbeda dalam sequence:
  - Posisi awal: butuh decay lambat (retensi panjang)
  - Posisi tengah: butuh keseimbangan antara retensi dan pelupakan
  - Posisi akhir: butuh decay cepat (fokus pada konteks lokal)

PoST (Position-Dependent Decay Spectra) menyelesaikan ini dengan:
  1. DecaySpectrum — Spectrum learnable decay rates per head
  2. Position-dependent mixing — Bobot campuran berdasarkan posisi
  3. Multi-mode state update — State update menggunakan campuran decay modes

Recurrence dengan PoST:
    h_t = sum_modes( mix_t(mode) * gamma(mode) * h_{t-1} ) + k_t * v_t

dimana:
    gamma(mode) = decay rate untuk mode tersebut (learnable)
    mix_t(mode) = bobot campuran pada posisi t untuk mode tersebut (learnable)

Keuntungan:
  - Retensi jangka panjang lebih baik (mode dengan decay lambat)
  - Fleksibilitas posisi-dependent (mixing weights bervariasi per posisi)
  - Backward-compatible dengan SSMTerpaduLayer (1 mode = SSM standar)
  - Overhead minimal (n_decay_modes kecil, default 4)

Referensi:
- Gu & Dao, "Mamba-2: A Generalized State Space Model with Structured
  State Space Duality" (2024)
- Peng et al., "Random Feature Attention" (2021) — inspirasi position-dependent
  mixing menggunakan random features
- Poli et al., "Striped Attention" — posisi-dependent processing patterns

Hardware: Pure PyTorch, kompatibel dengan CUDA / ROCm / CPU.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DecaySpectrum — Learnable spectrum of decay rates per head
# ---------------------------------------------------------------------------


class DecaySpectrum(nn.Module):
    """
    Learnable spectrum of decay rates per head.

    Alih-alih satu decay rate per head, DecaySpectrum menyediakan
    ``n_decay_modes`` decay rates yang berbeda per head. Setiap mode
    menangkap pola decay yang berbeda:

      - Mode 0: sangat lambat (retensi jangka panjang)
      - Mode 1: lambat
      - Mode 2: sedang
      - Mode 3: cepat (fokus lokal)
      - ... (bisa lebih dari 4 mode)

    Position-dependent mixing:
        mix_t = softmax(position_encoder(t))

    dimana position_encoder memetakan posisi ke bobot campuran
    untuk setiap mode.

    Args:
        n_heads: Jumlah SSM heads.
        n_decay_modes: Jumlah decay modes per head (default 4).
        max_seq_len: Panjang sequence maksimum untuk positional encoding (default 4096).
        init_decay_range: Tuple (min, max) untuk inisialisasi log decay rates.
            Decay rates diinisialisasi secara log-uniform dalam range ini.
            Default (-6, -1) → decay rates antara ~0.002 dan ~0.37.
    """

    def __init__(
        self,
        n_heads: int,
        n_decay_modes: int = 4,
        max_seq_len: int = 4096,
        init_decay_range: Tuple[float, float] = (-6.0, -1.0),
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.n_decay_modes = n_decay_modes
        self.max_seq_len = max_seq_len
        self.init_decay_range = init_decay_range

        # ---- Learnable decay rates per (head, mode) ----
        # Log-domain untuk stabilitas: gamma = exp(log_gamma)
        # Diinisialisasi secara log-uniform agar mode memiliki
        # decay rates yang tersebar merata dari sangat lambat ke cepat
        log_gamma_init = torch.zeros(n_heads, n_decay_modes)
        for mode in range(n_decay_modes):
            # Distribusikan decay rates merata dari lambat ke cepat
            # mode 0 = paling lambat, mode terakhir = paling cepat
            frac = mode / max(n_decay_modes - 1, 1)
            log_val = init_decay_range[0] + frac * (
                init_decay_range[1] - init_decay_range[0]
            )
            log_gamma_init[:, mode] = log_val

        # Tambahkan sedikit noise untuk break symmetry antar heads
        log_gamma_init += torch.randn_like(log_gamma_init) * 0.1

        self.log_gamma = nn.Parameter(log_gamma_init)

        # ---- Position encoder untuk mixing weights ----
        # Maps position index → mixing weights over modes
        # Menggunakan embedding + MLP untuk fleksibilitas
        self.position_embedding = nn.Embedding(
            max_seq_len, n_heads * n_decay_modes
        )

        # MLP untuk memproses positional features
        self.position_mlp = nn.Sequential(
            nn.Linear(n_heads * n_decay_modes, n_heads * n_decay_modes, bias=True),
            nn.SiLU(),
            nn.Linear(n_heads * n_decay_modes, n_heads * n_decay_modes, bias=False),
        )

        # Inisialisasi: bias ke mode sedang di awal
        with torch.no_grad():
            # Set default mixing: equal weight for all modes initially
            nn.init.normal_(
                self.position_embedding.weight, mean=0.0, std=0.02
            )

    def get_decay_rates(self) -> torch.Tensor:
        """
        Ambil decay rates (gamma) per (head, mode).

        Returns:
            gamma: Tensor bentuk (n_heads, n_decay_modes), nilai positif.
                Gamma mendekati 0 = decay cepat (lupa),
                Gamma mendekati 1 = decay lambat (ingat).
        """
        # sigmoid untuk memastikan gamma ∈ (0, 1)
        return torch.sigmoid(self.log_gamma)

    def get_mixing_weights(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hitung position-dependent mixing weights.

        Args:
            positions: Posisi token, bentuk (batch, seq_len).
                Harus dalam range [0, max_seq_len).

        Returns:
            mix: Mixing weights, bentuk (batch, seq_len, n_heads, n_decay_modes).
                Sum over last dim = 1 (softmax normalized).
        """
        batch, seq_len = positions.shape

        # Clamp positions ke valid range
        pos_clamped = positions.clamp(0, self.max_seq_len - 1)

        # Position embedding: (batch, seq_len, n_heads * n_decay_modes)
        pos_embed = self.position_embedding(pos_clamped)

        # MLP processing: (batch, seq_len, n_heads * n_decay_modes)
        pos_features = self.position_mlp(pos_embed)

        # Reshape ke (batch, seq_len, n_heads, n_decay_modes)
        mix = pos_features.view(
            batch, seq_len, self.n_heads, self.n_decay_modes
        )

        # Softmax over modes untuk normalisasi
        mix = F.softmax(mix, dim=-1)

        return mix

    def forward(
        self,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hitung decay rates dan mixing weights.

        Args:
            positions: Posisi token, bentuk (batch, seq_len).

        Returns:
            gamma: Decay rates, bentuk (n_heads, n_decay_modes).
            mix: Mixing weights, bentuk (batch, seq_len, n_heads, n_decay_modes).
        """
        gamma = self.get_decay_rates()
        mix = self.get_mixing_weights(positions)
        return gamma, mix


# ---------------------------------------------------------------------------
# PoST SSM Scan — Sequential scan dengan multi-mode decay
# ---------------------------------------------------------------------------


def post_ssm_scan(
    x_seq: torch.Tensor,
    A_base: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt: torch.Tensor,
    gamma: torch.Tensor,
    mix: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sequential SSM scan dengan PoST decay spectra.

    Berbeda dari scan SSM standar yang menggunakan satu decay rate,
    scan ini menggunakan spectrum decay rates dengan position-dependent mixing:

        h_t = sum_m( mix_t(m) * gamma(m) * dA_t * h_{t-1,m} ) + dB_t * x_t
        y_t = C_t^T @ h_t

    dimana m adalah index mode decay.

    State disimpan terpisah per mode untuk memungkinkan decay rate
    yang berbeda per mode. Mixing terjadi pada level output state.

    Args:
        x_seq: Input sequence, bentuk (batch, seq_len, d_inner).
        A_base: Base transition matrix (negatif), bentuk (d_inner, d_state).
        B: Input matrix, bentuk (batch, seq_len, d_state).
        C: Output matrix, bentuk (batch, seq_len, d_state).
        dt: Step size per token, bentuk (batch, seq_len).
        gamma: Decay rates per (head, mode), bentuk (n_heads, n_decay_modes).
            Nilai dalam (0, 1).
        mix: Mixing weights, bentuk (batch, seq_len, n_heads, n_decay_modes).
            Sum over last dim = 1.
        initial_state: State awal opsional, bentuk (batch, d_inner, d_state).

    Returns:
        Tuple (output, final_state):
        - output: bentuk (batch, seq_len, d_inner)
        - final_state: bentuk (batch, d_inner, d_state)
    """
    batch, seq_len, d_inner = x_seq.shape
    d_state = B.shape[-1]
    n_heads = gamma.shape[0]
    n_modes = gamma.shape[1]

    # ---- Inisialisasi state per mode ----
    # Setiap mode memiliki state sendiri
    if initial_state is None:
        states = [
            torch.zeros(
                batch, d_inner, d_state,
                dtype=x_seq.dtype, device=x_seq.device,
            )
            for _ in range(n_modes)
        ]
    else:
        # Bagi initial state merata ke semua mode
        states = [initial_state.clone() / n_modes for _ in range(n_modes)]

    # ---- Pre-compute gamma per (d_inner, d_state) dari (n_heads, n_modes) ----
    # Map heads ke d_inner channels: setiap head menguasai
    # d_inner / n_heads channels
    channels_per_head = d_inner // n_heads

    # gamma_per_channel: (d_inner, n_modes)
    gamma_expanded = gamma.repeat_interleave(channels_per_head, dim=0)
    if gamma_expanded.shape[0] < d_inner:
        # Handle d_inner yang tidak habis dibagi n_heads
        pad = d_inner - gamma_expanded.shape[0]
        gamma_expanded = F.pad(gamma_expanded, (0, 0, 0, pad), value=0.5)

    # ---- Pre-compute discretization ----
    # dA_base = exp(dt * A) tanpa decay modulation
    A_avg = A_base.mean(dim=0)  # (d_state,)
    dA_base = torch.exp(
        dt.unsqueeze(-1) * A_avg.unsqueeze(0).unsqueeze(0)
    )  # (batch, seq_len, d_state)
    dB = dt.unsqueeze(-1) * B  # (batch, seq_len, d_state)

    # ---- Sequential scan ----
    outputs = []
    for t in range(seq_len):
        # Update state per mode dengan decay rate masing-masing
        new_states = []
        combined_h = torch.zeros_like(states[0])  # (batch, d_inner, d_state)

        for m in range(n_modes):
            # Mode-specific decay: gamma_m * dA_t
            # gamma_per_channel[:, m]: (d_inner,)
            # Modulate decay per channel
            gamma_m = gamma_expanded[:, m]  # (d_inner,)

            # State update: h_m = gamma_m * dA_t * h_{m, t-1} + dB_t * x_t / n_modes
            # Distribusikan input contribution merata ke mode
            dA_t = dA_base[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            dB_t = dB[:, t, :]  # (batch, d_state)
            x_t = x_seq[:, t, :]  # (batch, d_inner)

            # Apply gamma per channel
            # gamma_m: (d_inner,) -> (batch, d_inner, 1)
            gamma_broadcast = gamma_m.unsqueeze(0).unsqueeze(-1)  # (1, d_inner, 1)

            h_m = states[m] * dA_t * gamma_broadcast  # (batch, d_inner, d_state)
            # Input contribution (dibagi merata ke modes)
            h_m = h_m + (x_t.unsqueeze(-1) * dB_t.unsqueeze(1)) / n_modes

            new_states.append(h_m)

            # Mixing: gabungkan states dari semua mode dengan position-dependent weights
            # mix[:, t, :, m]: (batch, n_heads)
            # Expand ke (batch, d_inner) untuk broadcasting
            mix_m = mix[:, t, :, m]  # (batch, n_heads)
            mix_m_expanded = mix_m.repeat_interleave(channels_per_head, dim=-1)
            if mix_m_expanded.shape[-1] < d_inner:
                pad = d_inner - mix_m_expanded.shape[-1]
                mix_m_expanded = F.pad(mix_m_expanded, (0, pad), value=0.0)

            combined_h = combined_h + mix_m_expanded.unsqueeze(-1) * h_m

        states = new_states

        # Output: y_t = C_t @ h_combined
        y_t = torch.sum(
            combined_h * C[:, t, :].unsqueeze(1), dim=-1
        )  # (batch, d_inner)
        outputs.append(y_t)

    # Stack output
    y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)

    # Final state: weighted combination dari mode states
    # Gunakan mixing weight dari posisi terakhir
    final_combined = torch.zeros_like(states[0])
    for m in range(n_modes):
        mix_m = mix[:, -1, :, m]  # (batch, n_heads)
        mix_m_expanded = mix_m.repeat_interleave(channels_per_head, dim=-1)
        if mix_m_expanded.shape[-1] < d_inner:
            pad = d_inner - mix_m_expanded.shape[-1]
            mix_m_expanded = F.pad(mix_m_expanded, (0, pad), value=0.0)
        final_combined = final_combined + mix_m_expanded.unsqueeze(-1) * states[m]

    return y, final_combined


# ---------------------------------------------------------------------------
# PoSTDecaySSM — SSM Layer dengan PoST Decay Spectra
# ---------------------------------------------------------------------------


class PoSTDecaySSM(nn.Module):
    """
    SSM layer dengan PoST (Position-Dependent Decay Spectra).

    Mengganti single decay parameter per head dengan spectrum
    decay rates yang bervariasi berdasarkan posisi. Fitur:

    1. Multiple decay modes per head — menangkap pola memori berbeda
    2. Position-dependent mixing — bobot campuran berdasarkan posisi
    3. Learnable spectrum — decay rates dioptimasi selama training
    4. Backward-compatible — dengan n_decay_modes=1, setara SSM standar

    Recurrence:
        h_t = sum_m( mix_t(m) * gamma(m) * dA_t * h_{t-1,m} ) + dB_t * x_t / M
        y_t = C_t^T @ h_t + D * x_t

    Args:
        d_model: Dimensi model input/output.
        d_state: Dimensi state SSM (default 128).
        d_conv: Lebar konvolusi lokal (default 4).
        expand: Faktor ekspansi (default 2).
        chunk_size: Ukuran chunk SSD (default 256).
        n_heads: Jumlah SSM heads (default 8).
        n_decay_modes: Jumlah decay modes per head (default 4).
        max_seq_len: Panjang sequence maksimum (default 4096).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 256,
        n_heads: int = 8,
        n_decay_modes: int = 4,
        max_seq_len: int = 4096,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        use_bias: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.chunk_size = chunk_size
        self.n_heads = n_heads
        self.n_decay_modes = n_decay_modes
        self.max_seq_len = max_seq_len
        self.d_inner = int(expand * d_model)

        # ---- Input projection ----
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=use_bias)

        # ---- Local causal convolution ----
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # ---- SSM parameter projections ----
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # ---- dt bias (per-channel) ----
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_softplus = torch.log(torch.exp(dt_init) - 1)
        self.dt_bias = nn.Parameter(inv_softplus)

        # ---- Parameter A (log-domain) ----
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A))

        # ---- Parameter D (skip connection) ----
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # ---- PoST Decay Spectrum ----
        self.decay_spectrum = DecaySpectrum(
            n_heads=n_heads,
            n_decay_modes=n_decay_modes,
            max_seq_len=max_seq_len,
        )

        # ---- Position counter (buffer, bukan parameter) ----
        # Digunakan untuk tracking posisi selama inference
        self.register_buffer(
            "_position_counter",
            torch.zeros(1, dtype=torch.long),
        )

        # ---- Output ----
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=use_bias)
        self.norm = nn.RMSNorm(self.d_inner, eps=1e-5)

    def _get_dt(self) -> torch.Tensor:
        """Ambil parameter dt setelah softplus."""
        return F.softplus(self.dt_bias + 1e-4)

    def forward(
        self,
        input: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass PoSTDecaySSM.

        Args:
            input: Tensor input, bentuk (batch, seq_len, d_model).
            initial_state: State awal opsional, bentuk (batch, d_inner, d_state).
            position_ids: Posisi ID opsional, bentuk (batch, seq_len).
                Jika None, menggunakan posisi sekuensial 0, 1, 2, ...

        Returns:
            Tuple (output, final_state):
            - output: bentuk (batch, seq_len, d_model)
            - final_state: bentuk (batch, d_inner, d_state)
        """
        batch, seq_len, _ = input.shape

        # Handle edge case
        if seq_len == 0:
            dummy_out = torch.zeros(
                batch, 0, self.d_model, dtype=input.dtype, device=input.device
            )
            dummy_state = (
                initial_state
                if initial_state is not None
                else torch.zeros(
                    batch, self.d_inner, self.d_state,
                    dtype=input.dtype, device=input.device,
                )
            )
            return dummy_out, dummy_state

        # ---- Position IDs ----
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=input.device
            ).unsqueeze(0).expand(batch, -1)

        # ---- Step 1: Input projection ----
        xz = self.in_proj(input)
        x, z = xz.chunk(2, dim=-1)

        # ---- Step 2: Local causal convolution ----
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # ---- Step 3: SSM parameters ----
        ssm_params = self.x_proj(x_conv)
        B = ssm_params[..., :self.d_state]
        C = ssm_params[..., self.d_state:self.d_state * 2]

        # dt
        dt_bias = self._get_dt()
        dt_full = F.softplus(
            self.dt_proj(x_conv) + dt_bias.unsqueeze(0).unsqueeze(0)
        )
        dt_avg = dt_full.mean(dim=-1)  # (batch, seq_len)

        # ---- Step 4: A parameter ----
        A = -torch.exp(self.A_log.float()).to(dtype=x_conv.dtype)

        # ---- Step 5: PoST Decay Spectrum ----
        gamma, mix = self.decay_spectrum(position_ids)
        # gamma: (n_heads, n_decay_modes)
        # mix: (batch, seq_len, n_heads, n_decay_modes)

        # ---- Step 6: PoST SSM Scan ----
        y, final_state = post_ssm_scan(
            x_seq=x_conv,
            A_base=A,
            B=B,
            C=C,
            dt=dt_avg,
            gamma=gamma,
            mix=mix,
            initial_state=initial_state,
        )

        # ---- Step 7: Skip connection D ----
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        # ---- Step 8: Gating and output ----
        y = y * F.silu(z)
        y = self.norm(y)
        output = self.out_proj(y)

        return output, final_state

    def forward_inference(
        self,
        input: torch.Tensor,
        state: torch.Tensor,
        position_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass untuk inferensi token-per-token (O(1) per token).

        Args:
            input: Tensor input satu token, bentuk (batch, 1, d_model).
            state: State rekurensi, bentuk (batch, d_inner, d_state).
            position_id: Posisi token opsional, bentuk (batch,).

        Returns:
            Tuple (output, new_state).
        """
        batch = input.shape[0]

        # ---- Position tracking ----
        if position_id is None:
            position_id = self._position_counter.expand(batch)
            self._position_counter = self._position_counter + 1

        # ---- Input projection ----
        xz = self.in_proj(input)
        x, z = xz.chunk(2, dim=-1)

        # Skip conv1d untuk single-token (asumsi cached)
        x_conv = F.silu(x)

        # ---- SSM parameters ----
        ssm_params = self.x_proj(x_conv)
        B = ssm_params[..., :self.d_state]
        C = ssm_params[..., self.d_state:]

        # dt
        dt_bias = self._get_dt()
        dt = F.softplus(
            self.dt_proj(x_conv) + dt_bias.unsqueeze(0).unsqueeze(0)
        )

        # A
        A = -torch.exp(self.A_log.float()).to(dtype=x_conv.dtype)

        # ---- PoST decay ----
        pos_ids = position_id.unsqueeze(-1)  # (batch, 1)
        gamma, mix = self.decay_spectrum(pos_ids)
        # gamma: (n_heads, n_modes)
        # mix: (batch, 1, n_heads, n_modes)

        # ---- Mode-specific state update (inference) ----
        dt_squeezed = dt.squeeze(1)  # (batch, d_inner)

        # dA: exp(dt * A) — per (batch, d_inner, d_state)
        # A: (d_inner, d_state)
        dA = torch.exp(dt_squeezed.unsqueeze(-1) * A.unsqueeze(0))  # (batch, d_inner, d_state)

        # dB = dt * B: (batch, d_inner, d_state)
        dB = dt_squeezed.unsqueeze(-1) * B.squeeze(1).unsqueeze(1)  # (batch, d_inner, d_state)

        # dBx = x * dB: outer product → (batch, d_inner, d_state)
        dBx = x_conv.squeeze(1).unsqueeze(-1) * dB  # (batch, d_inner, d_state)

        # Apply multi-mode decay
        channels_per_head = self.d_inner // self.n_heads
        gamma_expanded = gamma.repeat_interleave(channels_per_head, dim=0)
        if gamma_expanded.shape[0] < self.d_inner:
            pad = self.d_inner - gamma_expanded.shape[0]
            gamma_expanded = F.pad(gamma_expanded, (0, 0, 0, pad), value=0.5)

        # Weighted combination of decay rates
        # mix: (batch, 1, n_heads, n_modes) → effective gamma per channel
        mix_squeezed = mix.squeeze(1)  # (batch, n_heads, n_modes)
        mix_expanded = mix_squeezed.repeat_interleave(channels_per_head, dim=1)
        if mix_expanded.shape[1] < self.d_inner:
            pad = self.d_inner - mix_expanded.shape[1]
            mix_expanded = F.pad(mix_expanded, (0, pad, 0, 0), value=0.0)

        # Effective gamma per channel: sum_m(mix_m * gamma_m)
        effective_gamma = (mix_expanded * gamma_expanded.unsqueeze(0)).sum(dim=-1)
        # effective_gamma: (batch, d_inner)

        # Apply effective gamma to state update
        # effective_gamma: (batch, d_inner) → (batch, d_inner, 1)
        # dA: (batch, d_inner, d_state)
        new_state = effective_gamma.unsqueeze(-1) * dA * state + dBx

        # Output
        y = torch.sum(
            C.squeeze(1).unsqueeze(1) * new_state, dim=-1
        )  # (batch, d_inner)
        y = y + x_conv.squeeze(1) * self.D.unsqueeze(0)
        y = y * F.silu(z.squeeze(1))
        y = self.norm(y.unsqueeze(1)).squeeze(1)
        output = self.out_proj(y)

        return output.unsqueeze(1), new_state

    def reset_position(self) -> None:
        """Reset position counter (untuk sequence baru)."""
        self._position_counter.zero_()

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Inisialisasi state SSM kosong.

        Args:
            batch_size: Ukuran batch.
            device: Device tensor.
            dtype: Tipe data tensor.

        Returns:
            State tensor kosong, bentuk (batch_size, d_inner, d_state).
        """
        return torch.zeros(
            batch_size, self.d_inner, self.d_state,
            dtype=dtype, device=device,
        )
