"""
CurriculumScheduler — Penjadwal Curriculum Learning untuk Losion
=================================================================

Mengatur transisi antar fase training berdasarkan:
- Step count
- Validation metrics
- Load balance quality

Fase 1: Hanya Jalur 1 SSM yang aktif (equal router weights = [0.8, 0.1, 0.1])
Fase 2: Semua jalur aktif, router masih frozen (weights = [0.33, 0.33, 0.33])
Fase 3: Router di-unfreeze, GRPO mulai
Fase 4: Full optimization, early exit, distillation

Transisi antar fase dapat berdasarkan:
1. Step threshold (default)
2. Validation loss threshold
3. Load balance quality metric
4. Manual override

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from losion.config import LosionConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Enum — Fase Training
# ============================================================================


class TrainingPhase(str, Enum):
    """Fase training Losion.

    Setiap fase memiliki tujuan dan konfigurasi yang berbeda:

    - PHASE_1_INDIVIDUAL: Pre-training individual per jalur
    - PHASE_2_JOINT: Joint fine-tuning semua jalur
    - PHASE_3_RL: End-to-end RL dengan GRPO
    - PHASE_4_ADVANCED: Advanced optimization
    """

    PHASE_1_INDIVIDUAL = "phase_1_individual"
    PHASE_2_JOINT = "phase_2_joint"
    PHASE_3_RL = "phase_3_rl"
    PHASE_4_ADVANCED = "phase_4_advanced"


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class PhaseConfig:
    """Konfigurasi untuk satu fase training.

    Attributes:
        phase: Nama fase
        budget_fraction: Fraksi budget training (0.0-1.0)
        start_step: Langkah awal (opsional, override budget)
        end_step: Langkah akhir (opsional, override budget)
        router_weights: Bobot routing untuk fase ini [w1, w2, w3]
        router_frozen: Apakah router di-freeze
        learning_rate: Learning rate untuk fase ini
        target_pathways: Jalur yang dilatih di fase ini
        use_grpo: Apakah GRPO digunakan di fase ini
        use_early_exit: Apakah early exit digunakan
        use_distillation: Apakah distillation digunakan
        validation_threshold: Threshold validation loss untuk transisi (opsional)
    """

    phase: TrainingPhase
    budget_fraction: float
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    router_weights: Tuple[float, float, float] = (0.33, 0.33, 0.33)
    router_frozen: bool = True
    learning_rate: float = 3e-4
    target_pathways: List[str] = field(default_factory=lambda: ["ssm", "attention", "retrieval"])
    use_grpo: bool = False
    use_early_exit: bool = False
    use_distillation: bool = False
    validation_threshold: Optional[float] = None


@dataclass
class PhaseTransition:
    """Catatan transisi antar fase.

    Attributes:
        from_phase: Fase sebelumnya
        to_phase: Fase selanjutnya
        step: Langkah saat transisi
        reason: Alasan transisi
        from_metrics: Metrics dari fase sebelumnya
    """

    from_phase: TrainingPhase
    to_phase: TrainingPhase
    step: int
    reason: str
    from_metrics: Optional[Dict[str, float]] = None


# ============================================================================
# CurriculumScheduler
# ============================================================================


class CurriculumScheduler:
    """
    Curriculum Learning untuk Losion 4-Fase Training.

    Mengatur transisi antar fase berdasarkan:
    - Step count
    - Validation metrics
    - Load balance quality

    Fase 1: Hanya Jalur 1 SSM yang aktif (equal router weights = [0.8, 0.1, 0.1])
    Fase 2: Semua jalur aktif, router masih frozen (weights = [0.33, 0.33, 0.33])
    Fase 3: Router di-unfreeze, GRPO mulai
    Fase 4: Full optimization, early exit, distillation

    Args:
        config: LosionConfig
        total_steps: Total langkah training (jika None, dihitung dari config)
    """

    def __init__(
        self,
        config: LosionConfig,
        total_steps: Optional[int] = None,
    ) -> None:
        self.config = config
        self.total_steps = total_steps or config.training.max_steps

        # ---- Current state ----
        self.current_phase = TrainingPhase.PHASE_1_INDIVIDUAL
        self.current_step = 0
        self.pathway_index = 0  # Untuk fase 1: jalur mana yang sedang dilatih

        # ---- Phase configurations ----
        self.phase_configs = self._build_phase_configs()

        # ---- Transition history ----
        self.transition_history: List[PhaseTransition] = []

        # ---- Phase-specific counters ----
        self.phase_step_counters: Dict[TrainingPhase, int] = {
            phase: 0 for phase in TrainingPhase
        }

        # ---- Validation metrics tracking ----
        self.validation_metrics: Dict[str, List[float]] = {
            "loss": [],
            "perplexity": [],
        }

        # ---- Load balance tracking ----
        self.load_balance_history: List[Dict[str, float]] = []

    def _build_phase_configs(self) -> Dict[TrainingPhase, PhaseConfig]:
        """
        Bangun konfigurasi untuk setiap fase.

        Budget distribution:
        - Fase 1: 0-30% (Pre-Training Individual)
        - Fase 2: 30-60% (Joint Fine-Tuning)
        - Fase 3: 60-90% (End-to-End RL)
        - Fase 4: 90-100% (Advanced Optimization)

        Returns:
            Dictionary mapping TrainingPhase ke PhaseConfig
        """
        configs = {
            TrainingPhase.PHASE_1_INDIVIDUAL: PhaseConfig(
                phase=TrainingPhase.PHASE_1_INDIVIDUAL,
                budget_fraction=0.30,
                router_weights=(0.8, 0.1, 0.1),  # Dominasi SSM
                router_frozen=True,
                learning_rate=self.config.training.learning_rate,
                target_pathways=["ssm"],  # Mulai dari SSM saja
                use_grpo=False,
                use_early_exit=False,
                use_distillation=False,
            ),
            TrainingPhase.PHASE_2_JOINT: PhaseConfig(
                phase=TrainingPhase.PHASE_2_JOINT,
                budget_fraction=0.30,
                router_weights=(0.33, 0.33, 0.33),  # Equal weights
                router_frozen=True,
                learning_rate=self.config.training.learning_rate * 0.5,  # LR lebih kecil
                target_pathways=["ssm", "attention", "retrieval"],  # Semua jalur
                use_grpo=False,
                use_early_exit=False,
                use_distillation=False,
            ),
            TrainingPhase.PHASE_3_RL: PhaseConfig(
                phase=TrainingPhase.PHASE_3_RL,
                budget_fraction=0.30,
                router_weights=(0.33, 0.33, 0.33),  # Akan di-adjust oleh GRPO
                router_frozen=False,  # Router di-unfreeze
                learning_rate=self.config.training.learning_rate * 0.1,  # LR kecil untuk RL
                target_pathways=["ssm", "attention", "retrieval", "router"],
                use_grpo=True,  # GRPO aktif
                use_early_exit=False,
                use_distillation=False,
            ),
            TrainingPhase.PHASE_4_ADVANCED: PhaseConfig(
                phase=TrainingPhase.PHASE_4_ADVANCED,
                budget_fraction=0.10,
                router_weights=(0.33, 0.33, 0.33),  # Fully adaptive
                router_frozen=False,
                learning_rate=self.config.training.learning_rate * 0.01,  # LR sangat kecil
                target_pathways=["ssm", "attention", "retrieval", "router"],
                use_grpo=True,
                use_early_exit=True,  # Early exit aktif
                use_distillation=True,  # Distillation aktif
            ),
        }

        # Hitung start/end steps berdasarkan budget
        cumulative_budget = 0.0
        for phase in TrainingPhase:
            cfg = configs[phase]
            cfg.start_step = int(cumulative_budget * self.total_steps)
            cumulative_budget += cfg.budget_fraction
            cfg.end_step = int(cumulative_budget * self.total_steps)

        return configs

    def set_phase(self, phase: TrainingPhase) -> None:
        """
        Set fase training secara manual.

        Berguna untuk testing atau override otomatis.

        Args:
            phase: Fase yang diinginkan
        """
        old_phase = self.current_phase
        self.current_phase = phase

        if old_phase != phase:
            self.transition_history.append(
                PhaseTransition(
                    from_phase=old_phase,
                    to_phase=phase,
                    step=self.current_step,
                    reason="manual_override",
                )
            )
            logger.info(
                f"Curriculum: Fase diubah manual dari {old_phase.value} ke {phase.value}"
            )

    def update(
        self,
        step: int,
        validation_loss: Optional[float] = None,
        load_balance: Optional[Dict[str, float]] = None,
    ) -> TrainingPhase:
        """
        Update scheduler dan tentukan fase saat ini.

        Memeriksa apakah transisi fase diperlukan berdasarkan:
        1. Step threshold (primary)
        2. Validation loss threshold (secondary)
        3. Load balance quality (tertiary)

        Args:
            step: Langkah training saat ini
            validation_loss: Validation loss terbaru (opsional)
            load_balance: Load balance metrics (opsional)

        Returns:
            Fase training saat ini
        """
        self.current_step = step
        self.phase_step_counters[self.current_phase] += 1

        # Track validation metrics
        if validation_loss is not None:
            self.validation_metrics["loss"].append(validation_loss)
            if validation_loss > 0:
                self.validation_metrics["perplexity"].append(
                    float("inf") if validation_loss > 100 else 2 ** validation_loss
                )

        # Track load balance
        if load_balance is not None:
            self.load_balance_history.append(load_balance)

        # ---- Cek transisi berdasarkan step threshold ----
        current_config = self.phase_configs[self.current_phase]

        if current_config.end_step is not None and step >= current_config.end_step:
            next_phase = self._get_next_phase(self.current_phase)
            if next_phase is not None:
                self._transition_to(
                    next_phase,
                    reason=f"step_threshold_reached (step={step}, threshold={current_config.end_step})",
                )
                return self.current_phase

        # ---- Cek transisi berdasarkan validation loss ----
        if validation_loss is not None:
            next_config = self._get_next_phase_config(self.current_phase)
            if (
                next_config is not None
                and next_config.validation_threshold is not None
                and validation_loss < next_config.validation_threshold
            ):
                next_phase = self._get_next_phase(self.current_phase)
                if next_phase is not None:
                    self._transition_to(
                        next_phase,
                        reason=f"validation_threshold_met (loss={validation_loss:.4f})",
                    )
                    return self.current_phase

        return self.current_phase

    def _transition_to(self, next_phase: TrainingPhase, reason: str) -> None:
        """
        Transisi ke fase berikutnya.

        Args:
            next_phase: Fase tujuan
            reason: Alasan transisi
        """
        old_phase = self.current_phase

        # Simpan metrics sebelum transisi
        from_metrics = {}
        if self.validation_metrics["loss"]:
            from_metrics["last_val_loss"] = self.validation_metrics["loss"][-1]
        if self.load_balance_history:
            from_metrics["last_load_balance"] = self.load_balance_history[-1]

        self.transition_history.append(
            PhaseTransition(
                from_phase=old_phase,
                to_phase=next_phase,
                step=self.current_step,
                reason=reason,
                from_metrics=from_metrics,
            )
        )

        self.current_phase = next_phase

        logger.info(
            f"Curriculum: Transisi dari {old_phase.value} → {next_phase.value} "
            f"(alasan: {reason}, step: {self.current_step})"
        )

    def _get_next_phase(self, current: TrainingPhase) -> Optional[TrainingPhase]:
        """
        Dapatkan fase berikutnya setelah fase saat ini.

        Args:
            current: Fase saat ini

        Returns:
            Fase berikutnya, atau None jika sudah di fase terakhir
        """
        phase_order = list(TrainingPhase)
        try:
            idx = phase_order.index(current)
            if idx + 1 < len(phase_order):
                return phase_order[idx + 1]
        except ValueError:
            pass
        return None

    def _get_next_phase_config(self, current: TrainingPhase) -> Optional[PhaseConfig]:
        """
        Dapatkan konfigurasi fase berikutnya.

        Args:
            current: Fase saat ini

        Returns:
            PhaseConfig fase berikutnya, atau None
        """
        next_phase = self._get_next_phase(current)
        if next_phase is not None:
            return self.phase_configs.get(next_phase)
        return None

    # ---- Metode akses konfigurasi ----

    def get_current_config(self) -> PhaseConfig:
        """
        Ambil konfigurasi fase saat ini.

        Returns:
            PhaseConfig untuk fase saat ini
        """
        return self.phase_configs[self.current_phase]

    def get_router_weights(self) -> Tuple[float, float, float]:
        """
        Ambil bobot routing untuk fase saat ini.

        Returns:
            Tuple (w1, w2, w3) bobot routing
        """
        return self.phase_configs[self.current_phase].router_weights

    def is_router_frozen(self) -> bool:
        """
        Periksa apakah router di-freeze di fase saat ini.

        Returns:
            True jika router di-freeze
        """
        return self.phase_configs[self.current_phase].router_frozen

    def get_learning_rate(self) -> float:
        """
        Ambil learning rate untuk fase saat ini.

        Returns:
            Learning rate
        """
        return self.phase_configs[self.current_phase].learning_rate

    def get_current_target_pathway(self) -> str:
        """
        Ambil jalur target saat ini (untuk Fase 1).

        Di Fase 1, training dilakukan per jalur secara bergantian.
        Method ini mengembalikan jalur mana yang sedang dilatih.

        Returns:
            Nama jalur: "ssm", "attention", atau "retrieval"
        """
        if self.current_phase == TrainingPhase.PHASE_1_INDIVIDUAL:
            # Rotasi jalur di fase 1
            pathways = ["ssm", "attention", "retrieval"]
            phase_config = self.phase_configs[TrainingPhase.PHASE_1_INDIVIDUAL]
            phase_steps = self.phase_step_counters[TrainingPhase.PHASE_1_INDIVIDUAL]

            # Ganti jalur setiap 1/3 dari fase 1
            phase_budget = (
                phase_config.end_step - phase_config.start_step
                if phase_config.end_step is not None and phase_config.start_step is not None
                else self.total_steps // 3
            )
            sub_phase_length = phase_budget // 3

            if sub_phase_length > 0:
                pathway_index = min(phase_steps // sub_phase_length, 2)
            else:
                pathway_index = 0

            return pathways[pathway_index]

        # Di fase lain, semua jalur aktif
        return "ssm"

    def get_progress(self) -> Dict[str, float]:
        """
        Ambil progres training saat ini.

        Returns:
            Dictionary berisi informasi progres
        """
        phase_config = self.phase_configs[self.current_phase]
        phase_start = phase_config.start_step or 0
        phase_end = phase_config.end_step or self.total_steps
        phase_budget = phase_end - phase_start

        phase_progress = 0.0
        if phase_budget > 0:
            phase_progress = min(
                (self.current_step - phase_start) / phase_budget, 1.0
            )

        total_progress = self.current_step / max(self.total_steps, 1)

        return {
            "total_progress": total_progress,
            "current_phase": self.current_phase.value,
            "phase_progress": phase_progress,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
        }

    def get_schedule_summary(self) -> List[Dict[str, object]]:
        """
        Ambil ringkasan jadwal training.

        Returns:
            List dictionary berisi ringkasan setiap fase
        """
        summary = []
        for phase in TrainingPhase:
            config = self.phase_configs[phase]
            is_current = phase == self.current_phase

            summary.append({
                "phase": phase.value,
                "is_current": is_current,
                "budget_fraction": config.budget_fraction,
                "start_step": config.start_step,
                "end_step": config.end_step,
                "router_weights": config.router_weights,
                "router_frozen": config.router_frozen,
                "learning_rate": config.learning_rate,
                "target_pathways": config.target_pathways,
                "use_grpo": config.use_grpo,
                "use_early_exit": config.use_early_exit,
                "use_distillation": config.use_distillation,
            })

        return summary
