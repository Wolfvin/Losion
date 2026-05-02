"""
Path-Lock Expert (PLE) — Architectural Reasoning Control (arXiv:2604.27201).

Modul reasoning control yang mengunci jalur expert tertentu untuk tipe
reasoning spesifik, tanpa menambah FLOPs. PLE memodifikasi probabilitas
routing pada MoE layers untuk memaksa expert tertentu aktif untuk input
dengan pola reasoning tertentu.

Konsep Inti:
Dalam model MoE, token dirutekan ke experts berdasarkan affinity scores.
Tanpa kontrol, routing bersifat "serendipitous" — tidak ada jaminan bahwa
token reasoning akan dirutekan ke expert yang terspecialisasi reasoning.

PLE mengatasi ini dengan:
1. Mendeteksi tipe input (reasoning, factual, creative, dll.)
2. Menerapkan path-lock masks pada routing logits
3. Memaksa expert tertentu aktif untuk tipe reasoning tertentu
4. Zero additional FLOPs — hanya mengubah routing probabilities

Arsitektur:
1. PathLockConfig — Konfigurasi untuk path-locking behavior
   Menentukan mapping antara tipe reasoning dan expert mask.
   - lock_patterns: Dict mapping reasoning_type → expert_mask
   - soft_lock: Bool (soft constraint vs hard lock)
   - lock_strength: Float (0.0 = no lock, 1.0 = hard lock)

2. PathLockExpert — Path-Lock Expert module
   - Mendeteksi tipe input (reasoning, factual, creative, dll.)
   - Menerapkan path-lock masks pada routing logits
   - Soft lock: menambah bias pada expert tertentu
   - Hard lock: memaksa expert tertentu aktif

3. PathLockLayer — Wrapper yang menambah PLE ke MoE layer
   - Drop-in compatible dengan existing MoE modules
   - Menambah path-lock behavior tanpa mengubah interface

Mode Operasi:
- Reasoning mode: Memaksa expert reasoning aktif
  Berguna untuk mathematical/logical reasoning di mana konsistensi penting
- Factual mode: Memaksa expert factual aktif
  Berguna untuk knowledge retrieval di mana akurasi faktual penting
- Creative mode: Memperluas routing diversity
  Berguna untuk generative tasks di mana variasi diinginkan
- Auto mode: Deteksi otomatis tipe input

Keuntungan:
- Zero additional FLOPs: hanya mengubah routing weights
- Zero additional parameters (minimal): hanya classification head kecil
- Meningkatkan konsistensi reasoning: expert yang sama untuk tipe input yang sama
- Mengurangi "reasoning leakage": reasoning dan factual terisolasi
- Compatible dengan MoE manapun: drop-in wrapper

Referensi:
- arXiv:2604.27201 — Path-Lock Expert: Architectural Reasoning Control
- DeepSeek-V3 — Expert specialization via MoE
- Losion MCTS + Parallel Thinking — Reasoning modules yang diintegrasikan

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Enums dan Data Classes
# ============================================================================

class ReasoningType(Enum):
    """Tipe reasoning yang didukung oleh PLE."""
    REASONING = "reasoning"      # Mathematical/logical reasoning
    FACTUAL = "factual"          # Knowledge retrieval/factual QA
    CREATIVE = "creative"        # Creative generation
    CODE = "code"                # Code generation/understanding
    ANALYSIS = "analysis"        # Analytical thinking
    AUTO = "auto"                # Auto-detect dari input


@dataclass
class PathLockConfig:
    """Konfigurasi untuk path-locking behavior.

    Menentukan bagaimana path-lock diterapkan pada MoE routing.
    Bisa dikonfigurasi per-layer atau secara global.

    Attributes:
        lock_patterns: Dict mapping reasoning_type → expert_mask.
            expert_mask: Tensor boolean [num_experts], True = expert terkunci
            untuk tipe reasoning ini.
        soft_lock: Jika True, menerapkan soft constraint (bias pada logits).
            Jika False, menerapkan hard lock (force expert selection).
        lock_strength: Kekuatan lock (0.0 = no lock, 1.0 = hard lock).
            Hanya relevan jika soft_lock=True.
        default_reasoning_type: Tipe reasoning default jika tidak terdeteksi.
        auto_detect: Jika True, deteksi tipe input secara otomatis.
        per_layer: Jika True, setiap layer bisa punya konfigurasi berbeda.
    """

    lock_patterns: Dict[str, torch.Tensor] = field(default_factory=dict)
    soft_lock: bool = True
    lock_strength: float = 0.5
    default_reasoning_type: str = "factual"
    auto_detect: bool = True
    per_layer: bool = False

    def get_mask(self, reasoning_type: str) -> Optional[torch.Tensor]:
        """
        Dapatkan expert mask untuk tipe reasoning tertentu.

        Args:
            reasoning_type: Nama tipe reasoning.

        Returns:
            Expert mask tensor atau None jika tidak ada pattern.
        """
        return self.lock_patterns.get(reasoning_type)

    def add_pattern(
        self,
        reasoning_type: str,
        expert_indices: List[int],
        num_experts: int,
    ) -> None:
        """
        Tambahkan lock pattern baru.

        Args:
            reasoning_type: Nama tipe reasoning.
            expert_indices: List indeks expert yang dikunci untuk tipe ini.
            num_experts: Jumlah total experts.
        """
        mask = torch.zeros(num_experts, dtype=torch.bool)
        for idx in expert_indices:
            if 0 <= idx < num_experts:
                mask[idx] = True
        self.lock_patterns[reasoning_type] = mask

    @classmethod
    def create_default(
        cls,
        num_experts: int,
        soft_lock: bool = True,
        lock_strength: float = 0.5,
    ) -> "PathLockConfig":
        """
        Buat konfigurasi default dengan pattern yang telah ditentukan.

        Distribusi expert:
        - reasoning: expert 0 hingga num_experts//4
        - factual: expert num_experts//4 hingga num_experts//2
        - creative: expert num_experts//2 hingga 3*num_experts//4
        - code: expert 3*num_experts//4 hingga num_experts

        Args:
            num_experts: Jumlah total experts.
            soft_lock: Gunakan soft constraint.
            lock_strength: Kekuatan lock.

        Returns:
            PathLockConfig dengan default patterns.
        """
        config = cls(
            soft_lock=soft_lock,
            lock_strength=lock_strength,
            auto_detect=True,
        )

        n = num_experts
        q = max(n // 4, 1)

        config.add_pattern("reasoning", list(range(0, q)), n)
        config.add_pattern("factual", list(range(q, 2 * q)), n)
        config.add_pattern("creative", list(range(2 * q, 3 * q)), n)
        config.add_pattern("code", list(range(3 * q, n)), n)
        config.add_pattern("analysis", list(range(0, 2 * q)), n)

        return config


@dataclass
class PathLockOutput:
    """Output dari PathLockExpert.

    Attributes:
        modified_logits: Routing logits setelah path-lock diterapkan.
        detected_type: Tipe reasoning yang terdeteksi (atau default).
        lock_applied: Apakah path-lock diterapkan.
        expert_affinity: Affinity scores per expert setelah lock.
    """

    modified_logits: torch.Tensor
    detected_type: str
    lock_applied: bool
    expert_affinity: torch.Tensor


# ============================================================================
# PathLockExpert — Path-Lock Expert Module
# ============================================================================

class PathLockExpert(nn.Module):
    """
    Path-Lock Expert module — Architectural reasoning control.

    Menerapkan path-lock masks pada MoE routing logits untuk memaksa
    expert tertentu aktif untuk tipe reasoning tertentu.

    Mekanisme Deteksi Tipe Input:
    1. Input hidden state → classification head → tipe reasoning
    2. Classification head: lightweight MLP (d_model → n_types)
    3. Tipe terdeteksi digunakan untuk memilih lock pattern

    Mekanisme Path-Lock:
    - Soft lock: menambah bias pada routing logits
      logits += lock_strength * expert_bias (untuk locked experts)
      expert_bias > 0 untuk experts yang diinginkan
      expert_bias < 0 untuk experts yang tidak diinginkan
    - Hard lock: memaksa expert tertentu aktif
      Setelah softmax, zero-out probabilitas experts yang tidak diinginkan
      Renormalize probabilitas yang tersisa

    Zero Additional FLOPs:
    - Deteksi: satu forward pass kecil pada classification head
    - Lock: hanya mengubah routing logits (add/multiply, O(num_experts))
    - Tidak menambah komputasi pada expert FFNs

    Args:
        d_model: Dimensi model.
        num_experts: Jumlah total experts dalam MoE.
        config: Konfigurasi path-lock. Jika None, gunakan default.
        n_reasoning_types: Jumlah tipe reasoning yang didukung (default 5).
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        config: Optional[PathLockConfig] = None,
        n_reasoning_types: int = 5,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.n_reasoning_types = n_reasoning_types

        # Konfigurasi path-lock
        if config is None:
            self.config = PathLockConfig.create_default(num_experts)
        else:
            self.config = config

        # ===================================================================
        # Input Type Classifier
        # ===================================================================
        # Lightweight MLP untuk mendeteksi tipe input
        self.type_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 4, n_reasoning_types, bias=False),
        )

        # Norm untuk input classifier
        self.classifier_norm = nn.RMSNorm(d_model, eps=1e-5)

        # ===================================================================
        # Expert Bias Parameters (untuk soft lock)
        # ===================================================================
        # Satu bias per expert per reasoning type
        # Shape: (n_reasoning_types, num_experts)
        self.expert_bias = nn.Parameter(
            torch.zeros(n_reasoning_types, num_experts)
        )

        # ===================================================================
        # Reasoning Type Mapping
        # ===================================================================
        # Mapping dari class index ke reasoning type name
        self._type_names = ["reasoning", "factual", "creative", "code", "analysis"]
        # Extend jika ada lebih dari 5 types
        for i in range(5, n_reasoning_types):
            self._type_names.append(f"type_{i}")

        # Register lock patterns sebagai buffers (untuk persistence)
        for rtype, mask in self.config.lock_patterns.items():
            self.register_buffer(
                f"_lock_mask_{rtype}",
                mask.to(torch.float32),  # Simpan sebagai float untuk operasi
            )

        # Mode saat ini (bisa di-set manual)
        self.register_buffer(
            "_current_mode",
            torch.tensor(0, dtype=torch.long),  # Index ke _type_names
        )
        self._manual_mode: Optional[str] = None

    def detect_reasoning_type(
        self,
        x: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:
        """
        Deteksi tipe reasoning dari input.

        Menggunakan classification head untuk menentukan tipe input.
        Bisa juga di-override secara manual via set_mode().

        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Tuple (type_name, type_probs):
            - type_name: Nama tipe reasoning yang terdeteksi
            - type_probs: Probabilitas per tipe [batch, seq_len, n_types]
        """
        # Jika mode manual di-set, gunakan itu
        if self._manual_mode is not None:
            dummy_probs = torch.zeros(
                x.shape[0], x.shape[1], self.n_reasoning_types,
                dtype=x.dtype, device=x.device,
            )
            if self._manual_mode in self._type_names:
                idx = self._type_names.index(self._manual_mode)
                dummy_probs[:, :, idx] = 1.0
            return self._manual_mode, dummy_probs

        # Auto-detect via classifier
        normed = self.classifier_norm(x)
        # Pool seq_len dimension untuk klasifikasi
        pooled = normed.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        type_logits = self.type_classifier(pooled)  # (batch, 1, n_types)
        type_probs = F.softmax(type_logits, dim=-1)  # (batch, 1, n_types)

        # Expand ke seq_len
        type_probs = type_probs.expand(-1, x.shape[1], -1)  # (batch, seq_len, n_types)

        # Ambil tipe dominan
        dominant_idx = type_probs[0, 0].argmax().item()
        type_name = self._type_names[dominant_idx] if dominant_idx < len(self._type_names) else "factual"

        return type_name, type_probs

    def apply_soft_lock(
        self,
        logits: torch.Tensor,
        reasoning_type: str,
        type_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Terapkan soft lock pada routing logits.

        Soft lock menambah bias pada expert tertentu berdasarkan
        tipe reasoning. Expert yang dikunci untuk tipe ini mendapat
        bias positif, yang tidak dikunci mendapat bias negatif.

        Args:
            logits: Routing logits [batch, seq_len, num_experts].
            reasoning_type: Nama tipe reasoning.
            type_probs: Probabilitas per tipe (opsional, untuk weighted lock).

        Returns:
            Modified logits [batch, seq_len, num_experts].
        """
        # Dapatkan expert bias untuk tipe ini
        if reasoning_type in self._type_names:
            type_idx = self._type_names.index(reasoning_type)
        else:
            type_idx = 0

        # Bias untuk tipe ini: (num_experts,)
        bias = self.expert_bias[type_idx]  # (num_experts,)

        # Terapkan lock pattern jika ada
        lock_mask_key = f"_lock_mask_{reasoning_type}"
        if hasattr(self, lock_mask_key):
            lock_mask = getattr(self, lock_mask_key)  # (num_experts,) float
            # Locked experts: bias positif
            # Non-locked experts: bias negatif (proporsional)
            n_locked = lock_mask.sum().item()
            n_unlocked = self.num_experts - n_locked

            if n_locked > 0 and n_unlocked > 0:
                # Hitung bias adjustment
                locked_bias = self.config.lock_strength
                unlocked_bias = -locked_bias * (n_locked / n_unlocked)

                # Gabungkan dengan learned bias
                adjusted_bias = (
                    lock_mask * (locked_bias + bias * self.config.lock_strength)
                    + (1 - lock_mask) * (unlocked_bias + bias * self.config.lock_strength * 0.1)
                )
            else:
                adjusted_bias = bias * self.config.lock_strength
        else:
            # Tidak ada lock pattern, gunakan learned bias saja
            adjusted_bias = bias * self.config.lock_strength

        # Terapkan bias pada logits
        modified = logits + adjusted_bias.unsqueeze(0).unsqueeze(0)

        return modified

    def apply_hard_lock(
        self,
        probs: torch.Tensor,
        reasoning_type: str,
    ) -> torch.Tensor:
        """
        Terapkan hard lock pada routing probabilities.

        Hard lock memaksa expert tertentu aktif dengan men-zero-out
        probabilitas experts yang tidak diinginkan dan renormalize.

        Args:
            probs: Routing probabilities [batch, seq_len, num_experts].
            reasoning_type: Nama tipe reasoning.

        Returns:
            Modified probabilities [batch, seq_len, num_experts].
        """
        lock_mask_key = f"_lock_mask_{reasoning_type}"
        if not hasattr(self, lock_mask_key):
            return probs

        lock_mask = getattr(self, lock_mask_key)  # (num_experts,) float

        # Zero-out non-locked experts
        mask = lock_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, num_experts)
        masked_probs = probs * mask

        # Renormalize
        sum_probs = masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = masked_probs / sum_probs

        return normalized

    def forward(
        self,
        logits: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        reasoning_type: Optional[str] = None,
    ) -> PathLockOutput:
        """
        Terapkan path-lock pada routing logits.

        Args:
            logits: Routing logits dari router, [batch, seq_len, num_experts].
            hidden_states: Hidden states untuk deteksi tipe input,
                [batch, seq_len, d_model]. Diperlukan jika auto_detect=True
                dan reasoning_type tidak diberikan.
            reasoning_type: Override tipe reasoning. Jika diberikan,
                skip auto-detection.

        Returns:
            PathLockOutput dengan modified logits dan informasi routing.
        """
        lock_applied = False
        type_probs = None

        # Deteksi atau gunakan tipe reasoning yang diberikan
        if reasoning_type is not None:
            detected_type = reasoning_type
        elif self.config.auto_detect and hidden_states is not None:
            detected_type, type_probs = self.detect_reasoning_type(hidden_states)
        else:
            detected_type = self.config.default_reasoning_type

        # Cek apakah ada lock pattern untuk tipe ini
        lock_mask_key = f"_lock_mask_{detected_type}"
        has_lock = hasattr(self, lock_mask_key)

        if has_lock or detected_type in self._type_names:
            if self.config.soft_lock:
                # Soft lock: modifikasi logits
                modified_logits = self.apply_soft_lock(
                    logits, detected_type, type_probs
                )
                lock_applied = True
            else:
                # Hard lock: modifikasi setelah softmax
                # Pertama, hitung probs, terapkan hard lock, lalu kembali ke logits
                probs = F.softmax(logits, dim=-1)
                modified_probs = self.apply_hard_lock(probs, detected_type)
                # Konversi kembali ke logits (approximation via log)
                modified_logits = torch.log(modified_probs + 1e-10)
                lock_applied = True
        else:
            modified_logits = logits

        # Hitung expert affinity setelah lock
        expert_affinity = F.softmax(modified_logits, dim=-1).mean(dim=(0, 1))

        return PathLockOutput(
            modified_logits=modified_logits,
            detected_type=detected_type,
            lock_applied=lock_applied,
            expert_affinity=expert_affinity,
        )

    def forward_inference(
        self,
        logits: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass untuk inferensi — hanya mengembalikan modified logits.

        Args:
            logits: Routing logits [batch, seq_len, num_experts].
            hidden_states: Hidden states [batch, seq_len, d_model].

        Returns:
            Modified logits [batch, seq_len, num_experts].
        """
        output = self.forward(logits, hidden_states)
        return output.modified_logits

    def set_mode(self, mode: str) -> None:
        """
        Set mode reasoning secara manual.

        Berguna untuk inference di mana kita ingin mengontrol
        tipe reasoning secara eksplisit.

        Args:
            mode: Nama tipe reasoning ("reasoning", "factual", "creative",
                "code", "analysis", atau "auto" untuk auto-detect).
        """
        if mode == "auto":
            self._manual_mode = None
        else:
            self._manual_mode = mode

    def get_expert_affinity_report(
        self,
        reasoning_type: Optional[str] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Dapatkan laporan expert affinity per tipe reasoning.

        Berguna untuk memahami bagaimana PLE mengubah distribusi routing.

        Args:
            reasoning_type: Tipe reasoning (opsional, default: auto-detect).
            hidden_states: Hidden states untuk auto-detection.

        Returns:
            Dictionary per tipe reasoning dengan expert affinity scores.
        """
        report = {}

        types_to_check = [reasoning_type] if reasoning_type else self._type_names
        dummy_logits = torch.zeros(
            1, 1, self.num_experts, dtype=torch.float32, device=next(self.parameters()).device
        )

        for rtype in types_to_check:
            output = self.forward(dummy_logits, reasoning_type=rtype)
            affinity = output.expert_affinity.detach().cpu()

            report[rtype] = {
                f"expert_{i}": affinity[i].item()
                for i in range(self.num_experts)
            }
            report[rtype]["top_expert"] = affinity.argmax().item()
            report[rtype]["lock_applied"] = output.lock_applied

        return report

    def add_reasoning_type(
        self,
        type_name: str,
        expert_indices: List[int],
    ) -> None:
        """
        Tambahkan tipe reasoning baru beserta lock pattern.

        Args:
            type_name: Nama tipe reasoning.
            expert_indices: List indeks expert yang dikunci untuk tipe ini.
        """
        # Tambah ke type names
        if type_name not in self._type_names:
            self._type_names.append(type_name)

        # Tambah lock pattern ke config
        self.config.add_pattern(type_name, expert_indices, self.num_experts)

        # Register buffer untuk persistence
        mask = torch.zeros(self.num_experts, dtype=torch.float32)
        for idx in expert_indices:
            if 0 <= idx < self.num_experts:
                mask[idx] = 1.0
        self.register_buffer(f"_lock_mask_{type_name}", mask)


# ============================================================================
# PathLockLayer — Wrapper yang Menambah PLE ke MoE Layer
# ============================================================================

class PathLockLayer(nn.Module):
    """
    Wrapper yang menambah Path-Lock Expert behavior ke MoE layer manapun.

    Drop-in compatible dengan existing MoE modules. Membungkus MoE layer
    dan menyisipkan path-lock logic ke dalam routing process.

    Cara kerja:
    1. Intercept routing logits dari MoE router
    2. Terapkan path-lock masks berdasarkan tipe reasoning
    3. Lanjutkan ke standard MoE dispatch + combine

    Penggunaan:
        # Bungkus existing MoE layer
        moe_layer = ExpertChoiceMoE(d_model=512, d_ff=2048, num_experts=64)
        ple_layer = PathLockLayer(moe_layer, d_model=512, num_experts=64)

        # Forward pass — transparent wrapper
        output, routing_info = ple_layer(x)

        # Set reasoning mode
        ple_layer.set_mode("reasoning")

    Kompatibilitas:
    - ExpertChoiceMoE: Compatible
    - HeterogeneousMoE: Compatible
    - MatryoshkaMoE: Compatible
    - GradientRoutedMoE: Compatible
    - AuxFreeMoE: Compatible (dari modul ini)

    Args:
        moe_layer: MoE layer yang dibungkus. Harus memiliki:
            - router attribute dengan gate_proj yang menghasilkan logits
            - experts attribute (ModuleList)
            - forward() method yang menerima (x) dan mengembalikan (output, info)
        d_model: Dimensi model.
        num_experts: Jumlah total experts.
        ple_config: Konfigurasi PathLockExpert. Jika None, gunakan default.
    """

    def __init__(
        self,
        moe_layer: nn.Module,
        d_model: int,
        num_experts: int,
        ple_config: Optional[PathLockConfig] = None,
    ):
        super().__init__()

        self.moe_layer = moe_layer
        self.d_model = d_model
        self.num_experts = num_experts

        # Path-Lock Expert
        self.ple = PathLockExpert(
            d_model=d_model,
            num_experts=num_experts,
            config=ple_config,
        )

        # Flag untuk mengaktifkan/menonaktifkan PLE
        self.register_buffer(
            "ple_enabled",
            torch.tensor(True, dtype=torch.bool),
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, object]:
        """
        Forward pass dengan path-lock enhancement.

        Mendelegasikan ke MoE layer yang dibungkus, tetapi menyisipkan
        path-lock logic ke dalam routing jika PLE diaktifkan.

        Implementasi:
        PLE bekerja dengan meng-override routing logits sebelum dispatch.
        Karena berbagai MoE implementations memiliki interface berbeda,
        kita menggunakan hook-based approach:
        1. Register forward hook pada router's gate_proj
        2. Hook menerapkan path-lock pada logits
        3. Forward pass MoE berjalan normal dengan modified logits
        4. Remove hook setelah forward pass

        Args:
            x: Input tensor [batch, seq_len, d_model].
            **kwargs: Argumen tambahan untuk MoE layer.

        Returns:
            Tuple (output, routing_info) dari MoE layer.
        """
        if not self.ple_enabled:
            # PLE disabled — pass through tanpa modifikasi
            return self.moe_layer(x, **kwargs)

        # Deteksi tipe reasoning
        detected_type, _ = self.ple.detect_reasoning_type(x)

        # Hook untuk meng-override routing logits
        hook_handle = None

        def path_lock_hook(module, input, output):
            """Forward hook yang menerapkan path-lock pada router output."""
            # output: routing logits [batch, seq, num_experts]
            if isinstance(output, torch.Tensor) and output.dim() == 3:
                ple_output = self.ple(output, hidden_states=None, reasoning_type=detected_type)
                return ple_output.modified_logits
            return output

        # Cari router module dan pasang hook
        router = getattr(self.moe_layer, 'router', None)
        if router is not None:
            gate_proj = getattr(router, 'gate_proj', None)
            if gate_proj is not None:
                # gate_proj bisa nn.Linear atau nn.Sequential
                if isinstance(gate_proj, nn.Sequential):
                    # Pasang hook pada layer terakhir dari sequential
                    hook_handle = gate_proj[-1].register_forward_hook(path_lock_hook)
                else:
                    hook_handle = gate_proj.register_forward_hook(path_lock_hook)

        # Jalankan MoE forward pass
        try:
            result = self.moe_layer(x, **kwargs)
        finally:
            # Selalu remove hook setelah forward pass
            if hook_handle is not None:
                hook_handle.remove()

        return result

    def forward_inference(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, object]:
        """
        Forward pass untuk inferensi.

        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Tuple (output, routing_info).
        """
        return self.forward(x)

    def set_mode(self, mode: str) -> None:
        """
        Set mode reasoning PLE.

        Args:
            mode: "reasoning", "factual", "creative", "code", "analysis",
                atau "auto" untuk auto-detect.
        """
        self.ple.set_mode(mode)

    def enable_ple(self) -> None:
        """Aktifkan Path-Lock Expert."""
        self.ple_enabled.fill_(True)

    def disable_ple(self) -> None:
        """Nonaktifkan Path-Lock Expert (passthrough)."""
        self.ple_enabled.fill_(False)

    def get_expert_affinity_report(self) -> Dict[str, Dict[str, float]]:
        """
        Dapatkan laporan expert affinity per tipe reasoning.

        Returns:
            Dictionary per tipe reasoning dengan expert affinity scores.
        """
        return self.ple.get_expert_affinity_report()

    @property
    def is_ple_enabled(self) -> bool:
        """Apakah PLE sedang aktif."""
        return self.ple_enabled.item()
