"""
SSMTerpaduLayer — Layer SSM Terpadu untuk Losion Framework.

Jalur 1 dari arsitektur Tri-Jalur Router. Menggabungkan tiga inovasi
SSM dalam satu layer yang koheren:
1. Mamba-2 SSD — pemrosesan sekuensial paralel, GPU-aware
2. RWKV-7 WKV — evolusi state dinamis, inferensi O(1)
3. Gated DeltaNet — kemampuan in-context learning

Interleaving pattern yang dapat dikonfigurasi memungkinkan router
mengontrol rasio penggunaan masing-masing sub-layer secara dinamis.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba2 import Mamba2SSD
from .rwkv7 import RWKV7WKV
from .delta_net import GatedDeltaNet


# ---------------------------------------------------------------------------
# Interleaving Scheduler
# ---------------------------------------------------------------------------

class InterleavingScheduler:
    """
    Penjadwal interleaving untuk SSMTerpaduLayer.

    Menentukan urutan sub-layer mana yang diproses pada setiap posisi
    berdasarkan rasio interleaving dan bobot routing opsional.

    Contoh dengan rasio (4, 1, 1):
    - Blok 0: SSD (Mamba-2)
    - Blok 1: SSD (Mamba-2)
    - Blok 2: SSD (Mamba-2)
    - Blok 3: SSD (Mamba-2)
    - Blok 4: WKV (RWKV-7)
    - Blok 5: Delta (Gated DeltaNet)
    """

    SSD = "ssd"
    WKV = "wkv"
    DELTA = "delta"

    def __init__(
        self,
        ratios: Tuple[int, int, int] = (4, 1, 1),
    ):
        """
        Inisialisasi scheduler.

        Args:
            ratios: Tuple (ssd_ratio, wkv_ratio, delta_ratio).
        """
        self.ratios = ratios
        self._schedule = self._build_schedule(ratios)
        self._total_blocks = len(self._schedule)

    @staticmethod
    def _build_schedule(ratios: Tuple[int, int, int]) -> List[str]:
        """
        Bangun jadwal interleaving dari rasio.

        Menggunakan round-robin terinterleave untuk distribusi merata.
        Misalnya (4, 1, 1) -> [ssd, wkv, delta, ssd, ssd, ssd]
        yang menyebarkan WKV dan Delta di antara blok SSD.

        Args:
            ratios: Tuple rasio (ssd, wkv, delta).

        Returns:
            List nama sub-layer dalam urutan eksekusi.
        """
        ssd_r, wkv_r, delta_r = ratios
        total = ssd_r + wkv_r + delta_r

        if total == 0:
            return [InterleavingScheduler.SSD]  # Fallback

        # Strategi interleaving: sebarkan blok non-SSD secara merata
        schedule = []

        # Buat daftar blok per tipe
        ssd_blocks = [InterleavingScheduler.SSD] * ssd_r
        wkv_blocks = [InterleavingScheduler.WKV] * wkv_r
        delta_blocks = [InterleavingScheduler.DELTA] * delta_r

        # Interleave: sisipkan blok non-SSD secara merata di antara SSD
        # Pertama, tentukan posisi untuk blok non-SSD
        non_ssd = wkv_blocks + delta_blocks
        n_non_ssd = len(non_ssd)

        if n_non_ssd == 0:
            return ssd_blocks

        # Hitung jarak antar blok non-SSD
        n_ssd = len(ssd_blocks)

        if n_ssd == 0:
            return non_ssd

        # Distribusikan blok non-SSD secara merata
        # Setiap blok non-SSD disisipkan setelah blok SSD tertentu
        # Jarak kira-kira: n_ssd / (n_non_ssd + 1)
        insert_positions = []
        spacing = n_ssd / (n_non_ssd + 1)
        for i in range(n_non_ssd):
            pos = int((i + 1) * spacing)
            insert_positions.append(min(pos, n_ssd))

        # Bangun jadwal final
        ssd_idx = 0
        non_ssd_idx = 0
        for pos in sorted(set(insert_positions)):
            # Tambahkan blok SSD sebelum posisi sisipan
            while ssd_idx < pos:
                schedule.append(ssd_blocks[ssd_idx])
                ssd_idx += 1
            # Tambahkan blok non-SSD
            schedule.append(non_ssd[non_ssd_idx])
            non_ssd_idx += 1

        # Tambahkan sisa blok SSD
        while ssd_idx < n_ssd:
            schedule.append(ssd_blocks[ssd_idx])
            ssd_idx += 1

        # Tambahkan sisa blok non-SSD (jika ada)
        while non_ssd_idx < n_non_ssd:
            schedule.append(non_ssd[non_ssd_idx])
            non_ssd_idx += 1

        return schedule

    def get_schedule(self) -> List[str]:
        """Ambil jadwal interleaving saat ini."""
        return self._schedule.copy()

    def get_total_blocks(self) -> int:
        """Ambil jumlah total blok dalam satu siklus interleaving."""
        return self._total_blocks

    def get_block_type(self, block_idx: int) -> str:
        """
        Ambil tipe sub-layer untuk indeks blok tertentu.

        Args:
            block_idx: Indeks blok (dimodulo total_blocks).

        Returns:
            Nama sub-layer: "ssd", "wkv", atau "delta".
        """
        return self._schedule[block_idx % self._total_blocks]


# ---------------------------------------------------------------------------
# SSM State Container
# ---------------------------------------------------------------------------

class SSMState:
    """
    Kontainer state gabungan untuk semua sub-layer SSM.

    Menyimpan state dari Mamba-2 SSD, RWKV-7 WKV, dan Gated DeltaNet
    secara terpisah, memungkinkan manajemen state yang fleksibel.
    """

    def __init__(
        self,
        ssd_state: Optional[torch.Tensor] = None,
        wkv_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        delta_state: Optional[torch.Tensor] = None,
    ):
        """
        Inisialisasi SSMState.

        Args:
            ssd_state: State Mamba-2 SSD, bentuk (batch, d_inner, d_state).
            wkv_state: State RWKV-7 WKV, tuple (wkv, sum).
            delta_state: State Gated DeltaNet, bentuk (batch, n_heads, d_head, d_head).
        """
        self.ssd_state = ssd_state
        self.wkv_state = wkv_state
        self.delta_state = delta_state

    def is_empty(self) -> bool:
        """Periksa apakah semua state adalah None."""
        return (
            self.ssd_state is None
            and self.wkv_state is None
            and self.delta_state is None
        )


# ---------------------------------------------------------------------------
# SSMTerpaduLayer
# ---------------------------------------------------------------------------

class SSMTerpaduLayer(nn.Module):
    """
    Jalur 1: SSM Terpadu Layer — menggabungkan Mamba-2 SSD, RWKV-7 WKV,
    dan Gated DeltaNet dalam satu layer yang koheren.

    Interleaving pattern (dapat dikonfigurasi, default 4:1:1):
    - 4 blok Mamba-2 SSD (pemrosesan sekuensial paralel, GPU-aware)
    - 1 blok RWKV-7 WKV (evolusi state dinamis, inferensi O(1))
    - 1 blok Gated DeltaNet (kemampuan in-context learning)

    Semua blok berbagi dimensi yang sama dan bisa di-interleave.
    Router mengontrol rasio interleaving secara dinamis.

    Args:
        d_model: Dimensi model
        d_state: Dimensi state SSM (default 128)
        d_conv: Lebar konvolusi lokal (default 4)
        expand: Faktor ekspansi (default 2)
        chunk_size: Ukuran chunk SSD (default 256)
        interleaving_ratios: Tuple rasio (ssd, wkv, delta) (default (4, 1, 1))
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 256,
        n_heads: int = 8,
        d_head: int = 64,
        interleaving_ratios: Tuple[int, int, int] = (4, 1, 1),
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        Inisialisasi SSMTerpaduLayer.

        Args:
            d_model: Dimensi model input/output.
            d_state: Dimensi state SSM.
            d_conv: Lebar konvolusi lokal untuk Mamba-2.
            expand: Faktor ekspansi untuk Mamba-2.
            chunk_size: Ukuran chunk untuk komputasi paralel.
            n_heads: Jumlah attention heads (untuk WKV dan DeltaNet).
            d_head: Dimensi per head (untuk WKV dan DeltaNet).
            interleaving_ratios: Tuple rasio (ssd, wkv, delta).
            dropout: Dropout rate.
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.chunk_size = chunk_size
        self.n_heads = n_heads
        self.d_head = d_head
        self.interleaving_ratios = interleaving_ratios
        self.dropout_rate = dropout

        # ---- Buat sub-layer ----
        self.ssd = Mamba2SSD(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size,
        )

        self.wkv = RWKV7WKV(
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
        )

        self.delta = GatedDeltaNet(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            chunk_size=chunk_size,
        )

        # ---- Interleaving scheduler ----
        self.scheduler = InterleavingScheduler(interleaving_ratios)

        # ---- LayerNorm sebelum setiap sub-layer (Pre-norm) ----
        total_blocks = self.scheduler.get_total_blocks()
        self.layer_norms = nn.ModuleList(
            [nn.RMSNorm(d_model, eps=1e-5) for _ in range(total_blocks)]
        )

        # ---- Dropout ----
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---- Routing gate (opsional, untuk dynamic routing) ----
        # Menghasilkan bobot routing per blok dari input
        self.routing_gate = nn.Linear(d_model, 3, bias=False)
        # 3 output: weight untuk SSD, WKV, Delta

        # ---- Output integration ----
        self.output_norm = nn.RMSNorm(d_model, eps=1e-5)

    def forward(
        self,
        input: torch.Tensor,
        ssm_state: Optional[SSMState] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SSMState]:
        """
        Forward pass SSMTerpaduLayer.

        Memproses input melalui sub-layer SSM sesuai interleaving pattern.
        Jika routing_weights diberikan, menyesuaikan pola interleaving
        secara dinamis.

        Args:
            input: Tensor input, bentuk (batch, seq_len, d_model).
            ssm_state: State SSM dari step sebelumnya (opsional).
            routing_weights: Bobot routing dari adaptive router (opsional).
                Bentuk: (batch, 3) atau (batch, seq_len, 3).
                Dimensi terakhir: [ssd_weight, wkv_weight, delta_weight].

        Returns:
            Tuple (output, new_ssm_state):
            - output: bentuk (batch, seq_len, d_model)
            - new_ssm_state: SSMState yang berisi state terkini
        """
        batch, seq_len, _ = input.shape

        # Handle edge case
        if seq_len == 0:
            dummy_out = torch.zeros(
                batch, 0, self.d_model, dtype=input.dtype, device=input.device
            )
            return dummy_out, SSMState()

        # ---- Inisialisasi state ----
        if ssm_state is None:
            ssd_state = None
            wkv_state = None
            delta_state = None
        else:
            ssd_state = ssm_state.ssd_state
            wkv_state = ssm_state.wkv_state
            delta_state = ssm_state.delta_state

        # ---- Hitung routing weights jika tidak diberikan ----
        if routing_weights is None:
            # Gunakan default interleaving schedule
            # Routing weights = 1 untuk scheduled block, 0 untuk lainnya
            use_dynamic_routing = False
        else:
            use_dynamic_routing = True
            # Normalisasi routing weights
            if routing_weights.dim() == 2:
                # (batch, 3) -> (batch, 1, 3)
                routing_weights = routing_weights.unsqueeze(1)
            routing_weights = F.softmax(routing_weights, dim=-1)
            # routing_weights: (batch, seq_len, 3) atau (batch, 1, 3)

        # ---- Proses melalui interleaving blocks ----
        hidden = input
        total_blocks = self.scheduler.get_total_blocks()

        for block_idx in range(total_blocks):
            # Pre-norm
            residual = hidden
            hidden = self.layer_norms[block_idx](hidden)

            if use_dynamic_routing:
                # ---- Dynamic routing: blend outputs dari semua sub-layer ----
                block_weights = routing_weights  # (batch, seq_len, 3) atau (batch, 1, 3)

                # Proses melalui semua sub-layer dan blend
                ssd_out, ssd_state = self.ssd(hidden, ssd_state)
                wkv_out, wkv_state = self.wkv(hidden, wkv_state)
                delta_out, delta_state = self.delta(hidden, delta_state)

                # Stack outputs: (batch, seq_len, 3, d_model)
                stacked = torch.stack([ssd_out, wkv_out, delta_out], dim=2)

                # Blend dengan routing weights
                # weights: (batch, seq_len, 3) -> (batch, seq_len, 3, 1)
                w = block_weights.unsqueeze(-1)

                # Pastikan dimensi cocok
                if w.shape[1] == 1 and stacked.shape[1] > 1:
                    w = w.expand(-1, stacked.shape[1], -1, -1)

                blended = (stacked * w).sum(dim=2)  # (batch, seq_len, d_model)

                # Residual connection
                hidden = residual + self.dropout(blended)
            else:
                # ---- Fixed interleaving: gunakan satu sub-layer per blok ----
                block_type = self.scheduler.get_block_type(block_idx)

                if block_type == InterleavingScheduler.SSD:
                    block_out, ssd_state = self.ssd(hidden, ssd_state)
                elif block_type == InterleavingScheduler.WKV:
                    block_out, wkv_state = self.wkv(hidden, wkv_state)
                elif block_type == InterleavingScheduler.DELTA:
                    block_out, delta_state = self.delta(hidden, delta_state)
                else:
                    # Fallback ke SSD
                    block_out, ssd_state = self.ssd(hidden, ssd_state)

                # Residual connection
                hidden = residual + self.dropout(block_out)

        # ---- Output normalization ----
        output = self.output_norm(hidden)

        # ---- Kemas state baru ----
        new_state = SSMState(
            ssd_state=ssd_state,
            wkv_state=wkv_state,
            delta_state=delta_state,
        )

        return output, new_state

    def forward_inference(
        self,
        input: torch.Tensor,
        ssm_state: Optional[SSMState] = None,
        routing_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SSMState]:
        """
        Forward pass untuk inferensi token-per-token.

        Menggunakan mode O(1) per sub-layer yang mendukungnya.

        Args:
            input: Tensor input satu token, bentuk (batch, 1, d_model).
            ssm_state: State SSM dari step sebelumnya.
            routing_weights: Bobot routing opsional, bentuk (batch, 3).

        Returns:
            Tuple (output, new_ssm_state).
        """
        batch = input.shape[0]

        # Inisialisasi state
        if ssm_state is None:
            ssd_state = None
            wkv_state = None
            delta_state = None
        else:
            ssd_state = ssm_state.ssd_state
            wkv_state = ssm_state.wkv_state
            delta_state = ssm_state.delta_state

        # Default: gunakan SSD untuk inference (paling efisien O(1))
        if routing_weights is None:
            # Gunakan SSD untuk single-token inference
            hidden = input
            total_blocks = self.scheduler.get_total_blocks()

            for block_idx in range(total_blocks):
                residual = hidden
                hidden = self.layer_norms[block_idx](hidden)
                block_type = self.scheduler.get_block_type(block_idx)

                if block_type == InterleavingScheduler.SSD:
                    block_out, ssd_state = self.ssd.forward_inference(
                        hidden, ssd_state
                    )
                elif block_type == InterleavingScheduler.WKV:
                    block_out, wkv_state = self.wkv.forward_inference(
                        hidden, wkv_state
                    )
                elif block_type == InterleavingScheduler.DELTA:
                    block_out, delta_state = self.delta.forward_inference(
                        hidden, delta_state
                    )
                else:
                    block_out, ssd_state = self.ssd.forward_inference(
                        hidden, ssd_state
                    )

                hidden = residual + block_out

            output = self.output_norm(hidden)
        else:
            # Dynamic routing: blend semua sub-layer
            if routing_weights.dim() == 1:
                routing_weights = routing_weights.unsqueeze(0)
            routing_weights = F.softmax(routing_weights, dim=-1)
            # routing_weights: (batch, 3)

            hidden = input

            # Proses melalui setiap sub-layer
            # Untuk inference, kita hanya perlu satu blok dengan blending
            residual = hidden
            hidden_normed = self.layer_norms[0](hidden)

            ssd_out, ssd_state = self.ssd.forward_inference(hidden_normed, ssd_state)
            wkv_out, wkv_state = self.wkv.forward_inference(hidden_normed, wkv_state)
            delta_out, delta_state = self.delta.forward_inference(hidden_normed, delta_state)

            # Stack dan blend
            stacked = torch.stack([ssd_out, wkv_out, delta_out], dim=1)  # (batch, 3, d_model)
            w = routing_weights.unsqueeze(-1)  # (batch, 3, 1)
            blended = (stacked * w).sum(dim=1)  # (batch, d_model)

            hidden = residual + blended
            output = self.output_norm(hidden.unsqueeze(1)).squeeze(1)
            output = output.unsqueeze(1)  # (batch, 1, d_model)

        new_state = SSMState(
            ssd_state=ssd_state,
            wkv_state=wkv_state,
            delta_state=delta_state,
        )

        return output, new_state

    def get_routing_logits(self, input: torch.Tensor) -> torch.Tensor:
        """
        Hitung routing logits dari input.

        Berguna untuk router adaptif yang ingin menentukan bobot
        interleaving berdasarkan karakteristik input.

        Args:
            input: Tensor input, bentuk (batch, seq_len, d_model).

        Returns:
            Routing logits, bentuk (batch, seq_len, 3).
            Dimensi terakhir: [ssd, wkv, delta].
        """
        return self.routing_gate(input)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> SSMState:
        """
        Inisialisasi state SSM kosong.

        Args:
            batch_size: Ukuran batch.
            device: Device tensor.
            dtype: Tipe data tensor.

        Returns:
            SSMState yang baru diinisialisasi.
        """
        d_inner = int(self.expand * self.d_model)

        ssd_state = torch.zeros(
            batch_size, d_inner, self.d_state,
            dtype=dtype, device=device,
        )

        n_heads = self.n_heads
        d_head = self.d_head
        h_dim = n_heads * d_head

        wkv_state = (
            torch.zeros(batch_size, h_dim, dtype=dtype, device=device),
            torch.zeros(batch_size, h_dim, dtype=dtype, device=device),
        )

        delta_state = torch.zeros(
            batch_size, n_heads, d_head, d_head,
            dtype=dtype, device=device,
        )

        return SSMState(
            ssd_state=ssd_state,
            wkv_state=wkv_state,
            delta_state=delta_state,
        )
