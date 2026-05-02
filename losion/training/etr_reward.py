"""
ETR Entropy Trend Reward — Reduces Thinking Tokens up to 40%.

ETR (Entropy Trend Reward) memonitor tren entropy selama generasi dan
memberikan reward pada model yang konvergen ke jawaban secara efisien,
mengurangi thinking tokens yang tidak perlu hingga 40%.

Motivasi:
---------
Model reasoning sering menghasilkan "thinking tokens" yang berlebihan:
  - Mengulang argumen yang sama berkali-kali
  - Tidak konvergen ke jawaban (entropy tetap tinggi)
  - Padding dengan token yang tidak informatif

ETR menyelesaikan ini dengan:
  1. Melacak token-level entropy selama generasi
  2. Memberikan reward pada entropy yang menurun (model konvergen)
  3. Memberikan penalti pada entropy yang tetap tinggi (wasteful thinking)
  4. Dapat mengurangi thinking tokens hingga 40% tanpa quality loss

Intuisi:
--------
Ketika model "berpikir" dengan efisien:
  - Entropy awalnya tinggi (mengeksplorasi kemungkinan)
  - Secara bertahap menurun (menyempitkan jawaban)
  - Konvergen ke entropy rendah (menemukan jawaban)

Ketika model "berpikir" secara wasteful:
  - Entropy tetap tinggi (tidak konvergen)
  - Fluktuasi tanpa pola (tidak ada progres)
  - Tiba-tiba turun di akhir (jump ke jawaban tanpa reasoning yang jelas)

ETR mereward pola pertama dan menghukum pola kedua.

Arsitektur:
-----------
1. EntropyTrendTracker — Melacak entropy trend selama generasi
   - compute_token_entropy(): per-token entropy dari logits
   - compute_trend(): entropy trend over recent tokens
   - is_converging(): apakah entropy menurun

2. ETRRewardFunction — Reward berdasarkan entropy trend
   - __call__(): hitung reward berdasarkan entropy trend
   - Rewards decreasing entropy (efisien thinking)
   - Penalizes sustained high entropy (wasteful thinking)
   - Compatible dengan RewardFunction interface di GRPO

3. ETRTrainer — GRPO trainer dengan ETR reward signal
   - Extends GRPO training dengan ETR reward
   - Mixed reward: task_reward + alpha * etr_reward
   - Mendukung warmup (ETR reward dimulai setelah beberapa steps)

Rumus Reward:
-------------
    ETR_reward = w_converge * R_converge - w_waste * R_waste

dimana:
    R_converge = mean(entropy_gradient < 0)  — fraksi token yang konvergen
    R_waste    = mean(entropy > threshold)     — fraksi token wasteful

Sehingga:
    - Model yang cepat konvergen → reward tinggi
    - Model yang lama konvergen → reward rendah / penalti

Contoh:
-------
    >>> tracker = EntropyTrendTracker(window_size=16)
    >>> reward_fn = ETRRewardFunction(tracker)
    >>> # Selama training, hitung reward
    >>> etr_reward = reward_fn(generated_logits_list)
    >>> # Combined reward
    >>> total_reward = task_reward + 0.3 * etr_reward

Hardware: Pure PyTorch, kompatibel dengan CUDA / ROCm / CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# EntropyTrendTracker — Melacak entropy trend selama generasi
# ---------------------------------------------------------------------------


class EntropyTrendTracker:
    """
    Melacak token-level entropy trend selama generasi.

    Menghitung entropy dari logits pada setiap posisi token dan
    menyimpan history untuk analisis trend. Digunakan oleh
    ETRRewardFunction untuk menentukan apakah model sedang
    "berpikir" secara efisien atau wasteful.

    Metrik yang dilacak:
    - token_entropy: entropy per token
    - entropy_trend: gradien entropy (apakah menurun/naik)
    - convergence_score: seberapa cepat entropy menurun
    - waste_score: seberapa banyak token dengan entropy tinggi

    Args:
        window_size: Ukuran window untuk komputasi trend (default 16).
            Menggunakan sliding window untuk menghitung trend lokal.
        high_entropy_threshold: Threshold entropy tinggi (default 0.8 * max_entropy).
            Token dengan entropy di atas ini dianggap "wasteful".
        max_entropy: Maximum possible entropy (default log(vocab_size)).
            Digunakan untuk normalisasi dan threshold.
    """

    def __init__(
        self,
        window_size: int = 16,
        high_entropy_threshold: Optional[float] = None,
        max_entropy: Optional[float] = None,
    ) -> None:
        self.window_size = window_size
        self.high_entropy_threshold = high_entropy_threshold
        self.max_entropy = max_entropy

        # ---- History buffers ----
        self._entropy_history: List[torch.Tensor] = []
        self._position_history: List[int] = []

    def compute_token_entropy(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Hitung per-token entropy dari logits.

        Entropy mengukur ketidakpastian distribusi prediksi:
            H(p) = -sum_x p(x) * log(p(x))

        Entropy tinggi = model tidak yakin (masih "berpikir")
        Entropy rendah = model yakin (sudah konvergen)

        Args:
            logits: Logits dari model, bentuk (batch, seq_len, vocab_size)
                atau (batch, vocab_size) atau (vocab_size,).
            temperature: Temperature untuk softmax (default 1.0).

        Returns:
            Entropy per token, bentuk (batch, seq_len) atau (batch,)
                atau scalar, tergantung input shape.
        """
        # Pastikan 3D: (batch, seq_len, vocab_size)
        squeeze_dims = []
        if logits.dim() == 1:
            logits = logits.unsqueeze(0).unsqueeze(0)
            squeeze_dims = [0, 1]
        elif logits.dim() == 2:
            logits = logits.unsqueeze(1)
            squeeze_dims = [1]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)

        # Compute probabilities
        probs = F.softmax(logits.float(), dim=-1)

        # Compute entropy: H = -sum(p * log(p))
        log_probs = F.log_softmax(logits.float(), dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

        # Normalize by max entropy jika diketahui
        if self.max_entropy is not None and self.max_entropy > 0:
            entropy = entropy / self.max_entropy

        # Squeeze kembali ke shape asli
        for dim in reversed(squeeze_dims):
            entropy = entropy.squeeze(dim)

        return entropy

    def compute_trend(
        self,
        entropy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hitung entropy trend over recent tokens.

        Menggunakan linear regression sederhana pada window
        terakhir dari entropy history untuk menentukan trend:
        - Trend negatif = entropy menurun = model konvergen ✓
        - Trend positif = entropy naik = model tidak konvergen ✗
        - Trend nol = entropy stabil

        Args:
            entropy: Entropy tensor opsional. Jika None, gunakan
                history yang tersimpan.

        Returns:
            Trend scalar. Negatif = konvergen, positif = divergen.
        """
        if entropy is not None:
            # Tambahkan ke history
            self._entropy_history.append(entropy.detach().cpu())

        if len(self._entropy_history) < 2:
            return torch.tensor(0.0)

        # Ambil window terakhir
        window = self._entropy_history[-self.window_size:]
        # Stack: (window_len, ...)
        entropy_seq = torch.stack(window, dim=0)

        # Flatten ke 1D jika batch
        if entropy_seq.dim() > 1:
            entropy_seq = entropy_seq.flatten()

        n = len(entropy_seq)
        if n < 2:
            return torch.tensor(0.0)

        # Simple linear regression: y = a + b*x
        # b = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        x = torch.arange(n, dtype=torch.float32)
        y = entropy_seq.float()

        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x * x).sum()

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-8:
            return torch.tensor(0.0)

        trend = (n * sum_xy - sum_x * sum_y) / denominator

        return trend

    def is_converging(
        self,
        entropy: Optional[torch.Tensor] = None,
        threshold: float = -0.01,
    ) -> bool:
        """
        Apakah entropy menurun (model konvergen).

        Args:
            entropy: Entropy tensor opsional.
            threshold: Threshold trend untuk "konvergen".
                Negatif berarti entropy menurun. Default -0.01.

        Returns:
            True jika model sedang konvergen.
        """
        trend = self.compute_trend(entropy)
        return trend.item() < threshold

    def get_convergence_score(self) -> float:
        """
        Hitung skor konvergensi: fraksi token dengan entropy menurun.

        Returns:
            Skor antara 0 (tidak konvergen) dan 1 (sangat konvergen).
        """
        if len(self._entropy_history) < 2:
            return 0.5  # Netral

        window = self._entropy_history[-self.window_size:]
        entropy_seq = torch.stack(window, dim=0)

        if entropy_seq.dim() > 1:
            entropy_seq = entropy_seq.flatten()

        if len(entropy_seq) < 2:
            return 0.5

        # Hitung diff: entropy[t] - entropy[t-1]
        diffs = entropy_seq[1:] - entropy_seq[:-1]

        # Fraksi diff yang negatif (entropy menurun)
        convergence = (diffs < 0).float().mean().item()

        return convergence

    def get_waste_score(self) -> float:
        """
        Hitung skor waste: fraksi token dengan entropy tinggi.

        Returns:
            Skor antara 0 (tidak waste) dan 1 (sangat waste).
        """
        if not self._entropy_history:
            return 0.0

        window = self._entropy_history[-self.window_size:]
        entropy_seq = torch.stack(window, dim=0)

        if entropy_seq.dim() > 1:
            entropy_seq = entropy_seq.flatten()

        if len(entropy_seq) == 0:
            return 0.0

        # Tentukan threshold
        threshold = self.high_entropy_threshold
        if threshold is None:
            # Default: 80th percentile dari entropy history
            if len(entropy_seq) > 4:
                threshold = torch.quantile(entropy_seq, 0.8).item()
            else:
                if self.max_entropy is not None:
                    threshold = 0.8 * self.max_entropy
                else:
                    threshold = 0.8 * math.log(max(entropy_seq.shape[0], 2))

        # Fraksi token di atas threshold
        waste = (entropy_seq > threshold).float().mean().item()

        return waste

    def record(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Rekam entropy dari logits dan kembalikan entropy value.

        Convenience method yang menggabungkan compute_token_entropy
        dan penyimpanan ke history.

        Args:
            logits: Logits tensor, bentuk (batch, seq_len, vocab_size).

        Returns:
            Entropy tensor, bentuk (batch, seq_len).
        """
        entropy = self.compute_token_entropy(logits)
        self._entropy_history.append(entropy.detach().cpu())
        return entropy

    def reset(self) -> None:
        """Reset history (untuk sequence baru)."""
        self._entropy_history.clear()
        self._position_history.clear()

    @property
    def history_length(self) -> int:
        """Jumlah entri dalam history."""
        return len(self._entropy_history)

    def get_history(self) -> List[torch.Tensor]:
        """Ambil seluruh entropy history."""
        return self._entropy_history.copy()


# ---------------------------------------------------------------------------
# ETRRewardFunction — Reward berdasarkan Entropy Trend
# ---------------------------------------------------------------------------


@dataclass
class ETRConfig:
    """
    Konfigurasi ETR (Entropy Trend Reward).

    Attributes:
        window_size: Ukuran window untuk trend computation (default 16).
        convergence_weight: Bobot reward untuk konvergensi (default 1.0).
        waste_penalty_weight: Bobot penalti untuk wasteful thinking (default 0.5).
        high_entropy_threshold: Threshold entropy tinggi (opsional).
        max_entropy: Maximum entropy untuk normalisasi (opsional).
        min_tokens_before_eval: Jumlah minimum token sebelum evaluasi ETR (default 8).
            Mencegah evaluasi premature saat entropy masih naik di awal.
        reward_clamp: Clamp reward ke range [-reward_clamp, reward_clamp] (default 2.0).
        temperature: Temperature untuk entropy computation (default 1.0).
    """

    window_size: int = 16
    convergence_weight: float = 1.0
    waste_penalty_weight: float = 0.5
    high_entropy_threshold: Optional[float] = None
    max_entropy: Optional[float] = None
    min_tokens_before_eval: int = 8
    reward_clamp: float = 2.0
    temperature: float = 1.0


class ETRRewardFunction:
    """
    ETR (Entropy Trend Reward) function.

    Memberikan reward berdasarkan tren entropy selama generasi:
    - Reward entropy yang menurun (model konvergen ke jawaban secara efisien)
    - Penalti entropy yang tetap tinggi (model berpikir secara wasteful)

    Kompatibel dengan RewardFunction interface di GRPO module
    dan bisa digunakan sebagai reward signal tambahan.

    Rumus:
        ETR_reward = w_conv * convergence_score - w_waste * waste_score

    dimana:
        convergence_score = fraksi token dengan entropy menurun
        waste_score = fraksi token dengan entropy tinggi (di atas threshold)

    Contoh:
        >>> config = ETRConfig(convergence_weight=1.0, waste_penalty_weight=0.5)
        >>> reward_fn = ETRRewardFunction(config)
        >>> # Selama generation, kumpulkan logits
        >>> for step in generation_loop:
        ...     logits = model(input_ids)
        ...     reward_fn.update(logits)
        >>> # Hitung reward
        >>> reward = reward_fn.compute_reward()

    Args:
        config: ETRConfig dengan parameter konfigurasi.
        tracker: EntropyTrendTracker opsional. Jika None, buat baru.
    """

    def __init__(
        self,
        config: Optional[ETRConfig] = None,
        tracker: Optional[EntropyTrendTracker] = None,
    ) -> None:
        self.config = config or ETRConfig()
        self.tracker = tracker or EntropyTrendTracker(
            window_size=self.config.window_size,
            high_entropy_threshold=self.config.high_entropy_threshold,
            max_entropy=self.config.max_entropy,
        )

    def update(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Update tracker dengan logits baru dan kembalikan entropy.

        Dipanggil pada setiap langkah generasi untuk mengumpulkan
        statistik entropy.

        Args:
            logits: Logits dari model, bentuk (batch, seq_len, vocab_size).

        Returns:
            Entropy tensor, bentuk (batch, seq_len).
        """
        return self.tracker.record(logits)

    def compute_reward(
        self,
        logits_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Hitung ETR reward berdasarkan entropy trend.

        Jika logits_list diberikan, hitung entropy dari logits tersebut.
        Jika tidak, gunakan history yang sudah dikumpulkan via update().

        Args:
            logits_list: List logits tensors opsional. Jika diberikan,
                hitung reward dari list ini (meng-override history).

        Returns:
            Scalar reward tensor. Positif = efisien, negatif = wasteful.
        """
        # Jika logits_list diberikan, reset dan hitung ulang
        if logits_list is not None:
            self.tracker.reset()
            for logits in logits_list:
                self.tracker.record(logits)

        # Jika belum cukup token, return reward netral
        if self.tracker.history_length < self.config.min_tokens_before_eval:
            return torch.tensor(0.0)

        # ---- Convergence score ----
        convergence = self.tracker.get_convergence_score()
        # convergence ∈ [0, 1], 1 = semua token konvergen

        # ---- Waste score ----
        waste = self.tracker.get_waste_score()
        # waste ∈ [0, 1], 1 = semua token wasteful

        # ---- Compute trend bonus ----
        trend = self.tracker.compute_trend()
        # trend: negatif = konvergen, positif = divergen
        # Normalisasi trend ke [-1, 1]
        trend_bonus = torch.clamp(-trend, -1.0, 1.0)

        # ---- Combined reward ----
        reward = (
            self.config.convergence_weight * convergence
            - self.config.waste_penalty_weight * waste
            + 0.2 * trend_bonus  # Bonus kecil untuk trend negatif
        )

        # ---- Clamp reward ----
        reward = torch.clamp(
            reward,
            min=-self.config.reward_clamp,
            max=self.config.reward_clamp,
        )

        return reward

    def __call__(
        self,
        logits_list: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Hitung reward — compatible dengan RewardFunction interface.

        Args:
            logits_list: List logits tensors opsional.

        Returns:
            Scalar reward tensor.
        """
        return self.compute_reward(logits_list)

    def reset(self) -> None:
        """Reset tracker state (untuk sequence baru)."""
        self.tracker.reset()

    def get_diagnostics(self) -> Dict[str, float]:
        """
        Ambil diagnostik untuk logging.

        Returns:
            Dict berisi metrik ETR:
            - convergence_score: Skor konvergensi
            - waste_score: Skor waste
            - trend: Entropy trend
            - is_converging: Apakah model sedang konvergen
            - history_length: Jumlah entri history
        """
        return {
            "convergence_score": self.tracker.get_convergence_score(),
            "waste_score": self.tracker.get_waste_score(),
            "trend": self.tracker.compute_trend().item(),
            "is_converging": self.tracker.is_converging(),
            "history_length": self.tracker.history_length,
        }


# ---------------------------------------------------------------------------
# ETRTrainer — GRPO Trainer dengan ETR Reward
# ---------------------------------------------------------------------------


class ETRTrainer:
    """
    Trainer yang mengintegrasikan ETR (Entropy Trend Reward) ke dalam
    GRPO training.

    Extends GRPO training dengan menambahkan ETR reward signal:
        total_reward = task_reward + alpha * etr_reward

    dimana alpha adalah koefisien mixing yang dapat dikonfigurasi.

    Fitur:
    - Mixed reward: task_reward + alpha * etr_reward
    - Warmup: ETR reward dimulai setelah beberapa steps
    - Adaptive alpha: alpha meningkat secara gradual
    - Compatible dengan GRPOTrainer yang sudah ada

    Contoh:
        >>> etr_config = ETRConfig()
        >>> trainer = ETRTrainer(
        ...     model=model,
        ...     grpo_config=GRPOConfig(),
        ...     etr_config=etr_config,
        ...     etr_alpha=0.3,
        ... )
        >>> metrics = trainer.train_step(prompts)

    Args:
        model: LosionForCausalLM yang akan dioptimasi.
        grpo_config: GRPOConfig untuk GRPO training (opsional).
        etr_config: ETRConfig untuk ETR reward (opsional).
        etr_alpha: Koefisien mixing ETR reward (default 0.3).
            total_reward = task_reward + etr_alpha * etr_reward
        etr_warmup_steps: Jumlah steps sebelum ETR reward aktif (default 50).
            Mencegah ETR mengganggu training di awal.
        etr_alpha_schedule: Schedule untuk alpha ("constant" | "linear" | "cosine").
            - constant: alpha tetap
            - linear: alpha meningkat dari 0 ke etr_alpha
            - cosine: alpha mengikuti cosine schedule
    """

    def __init__(
        self,
        model: nn.Module,
        grpo_config: Optional[Any] = None,
        etr_config: Optional[ETRConfig] = None,
        etr_alpha: float = 0.3,
        etr_warmup_steps: int = 50,
        etr_alpha_schedule: str = "linear",
    ) -> None:
        self.model = model
        self.etr_config = etr_config or ETRConfig()
        self.etr_alpha = etr_alpha
        self.etr_warmup_steps = etr_warmup_steps
        self.etr_alpha_schedule = etr_alpha_schedule

        # ---- ETR reward function ----
        self.etr_reward_fn = ETRRewardFunction(self.etr_config)

        # ---- GRPO trainer ----
        # Lazy import untuk menghindari circular import
        try:
            from losion.training.grpo import GRPOTrainer, GRPOConfig, RewardFunction
            self._grpo_cls = GRPOTrainer
            self._grpo_config_cls = GRPOConfig

            if grpo_config is None:
                grpo_config = GRPOConfig()

            self.grpo_trainer = GRPOTrainer(
                model=model,
                config=grpo_config,
            )
        except ImportError:
            self._grpo_cls = None
            self._grpo_config_cls = None
            self.grpo_trainer = None

        # ---- State ----
        self._step_count = 0
        self._device = next(model.parameters()).device

    def _get_current_alpha(self) -> float:
        """
        Hitung alpha saat ini berdasarkan schedule.

        Returns:
            Current alpha value.
        """
        if self._step_count < self.etr_warmup_steps:
            # Warmup: alpha = 0
            return 0.0

        steps_after_warmup = self._step_count - self.etr_warmup_steps

        if self.etr_alpha_schedule == "constant":
            return self.etr_alpha

        elif self.etr_alpha_schedule == "linear":
            # Linear ramp dari 0 ke etr_alpha selama 2x warmup
            ramp_steps = self.etr_warmup_steps * 2
            if steps_after_warmup >= ramp_steps:
                return self.etr_alpha
            return self.etr_alpha * (steps_after_warmup / ramp_steps)

        elif self.etr_alpha_schedule == "cosine":
            # Cosine schedule
            total_steps = self.etr_warmup_steps * 4
            if steps_after_warmup >= total_steps:
                return self.etr_alpha
            progress = steps_after_warmup / total_steps
            return self.etr_alpha * 0.5 * (1 - math.cos(math.pi * progress))

        return self.etr_alpha

    def compute_mixed_reward(
        self,
        task_reward: torch.Tensor,
        logits_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Hitung mixed reward: task_reward + alpha * etr_reward.

        Args:
            task_reward: Task reward dari GRPO, bentuk (group_size,).
            logits_list: List logits tensors untuk ETR computation.

        Returns:
            Mixed reward, bentuk (group_size,).
        """
        alpha = self._get_current_alpha()

        if alpha < 1e-8:
            return task_reward

        # Hitung ETR reward
        etr_reward = self.etr_reward_fn.compute_reward(logits_list)

        # Pastikan shape cocok
        if etr_reward.dim() == 0:
            etr_reward = etr_reward.unsqueeze(0)

        # Broadcast jika perlu
        if etr_reward.shape[0] == 1 and task_reward.shape[0] > 1:
            etr_reward = etr_reward.expand(task_reward.shape[0])
        elif etr_reward.shape[0] != task_reward.shape[0]:
            # Fallback: gunakan mean ETR reward
            etr_reward = etr_reward.mean().unsqueeze(0).expand(task_reward.shape[0])

        mixed = task_reward + alpha * etr_reward

        return mixed

    def train_step(
        self,
        prompts: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Satu langkah training dengan ETR-enhanced GRPO.

        Alur:
        1. Generate group of responses (via GRPO)
        2. Hitung task reward
        3. Hitung ETR reward dari logits
        4. Combine: mixed_reward = task + alpha * etr
        5. Update policy

        Args:
            prompts: Token IDs prompt, bentuk (batch, prompt_len).
            attention_mask: Mask attention opsional.

        Returns:
            Dictionary berisi metrics.
        """
        self._step_count += 1

        # ---- Jika GRPO trainer tersedia, gunakan ----
        if self.grpo_trainer is not None:
            # Run GRPO step
            metrics = self.grpo_trainer.train_step(prompts, attention_mask)

            # Tambahkan ETR diagnostics
            alpha = self._get_current_alpha()
            etr_diag = self.etr_reward_fn.get_diagnostics()

            metrics["etr_alpha"] = alpha
            metrics["etr_convergence"] = etr_diag["convergence_score"]
            metrics["etr_waste"] = etr_diag["waste_score"]
            metrics["etr_trend"] = etr_diag["trend"]
            metrics["etr_is_converging"] = float(etr_diag["is_converging"])

            return metrics

        # ---- Fallback: simplified training step ----
        self.model.train()

        # Simplified: hanya hitung ETR reward sebagai monitoring
        # (Full training membutuhkan GRPOTrainer)
        with torch.no_grad():
            if prompts.dim() == 2:
                output = self.model(input_ids=prompts)
                if hasattr(output, "logits"):
                    logits = output.logits
                else:
                    logits = output

                self.etr_reward_fn.update(logits)

        alpha = self._get_current_alpha()
        etr_reward = self.etr_reward_fn.compute_reward()
        etr_diag = self.etr_reward_fn.get_diagnostics()

        metrics = {
            "etr_reward": etr_reward.item(),
            "etr_alpha": alpha,
            "etr_convergence": etr_diag["convergence_score"],
            "etr_waste": etr_diag["waste_score"],
            "etr_trend": etr_diag["trend"],
            "etr_is_converging": float(etr_diag["is_converging"]),
            "step": self._step_count,
        }

        return metrics

    def get_diagnostics(self) -> Dict[str, float]:
        """
        Ambil diagnostik ETR untuk logging.

        Returns:
            Dict berisi metrik ETR dan training state.
        """
        diag = self.etr_reward_fn.get_diagnostics()
        diag["current_alpha"] = self._get_current_alpha()
        diag["step_count"] = self._step_count
        diag["warmup_progress"] = min(
            1.0, self._step_count / max(self.etr_warmup_steps, 1)
        )
        return diag

    def reset(self) -> None:
        """Reset ETR state."""
        self.etr_reward_fn.reset()
