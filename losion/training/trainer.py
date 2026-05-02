"""
LosionTrainer — Trainer Utama untuk Framework Losion
=====================================================

Mengimplementasikan training 4-fase:

Fase 1: Pre-Training Individual (0-30% budget)
- Setiap jalur dilatih terpisah (frozen router)
- Hanya jalur yang ditarget yang di-update

Fase 2: Joint Fine-Tuning (30-60% budget)
- Ketiga jalur dilatih bersama, frozen router
- Bridge mechanism dilatih

Fase 3: End-to-End RL (60-90% budget)
- Router di-unfreeze
- GRPO untuk optimasi routing

Fase 4: Advanced Optimization (90-100% budget)
- Early exit, flow matching, distillation

Mendukung:
- Distributed training (DDP, FSDP)
- Mixed precision (bf16, fp8)
- Gradient accumulation
- Checkpoint saving/resuming
- Wandb logging

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losion.config import LosionConfig
from losion.models.losion_decoder import LosionForCausalLM
from losion.training.curriculum import CurriculumScheduler, TrainingPhase
from losion.training.utils import (
    count_parameters,
    estimate_training_memory,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    load_checkpoint,
    save_checkpoint,
    setup_distributed,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Konfigurasi Trainer
# ============================================================================


@dataclass
class TrainerConfig:
    """Konfigurasi spesifik untuk LosionTrainer.

    Attributes:
        output_dir: Direktori output untuk checkpoint dan log
        num_train_epochs: Jumlah epoch pelatihan
        max_train_steps: Maksimum langkah pelatihan (override epochs)
        gradient_accumulation_steps: Langkah akumulasi gradien
        max_grad_norm: Norma gradien maksimum untuk clipping
        fp16: Aktifkan FP16 mixed precision
        bf16: Aktifkan BF16 mixed precision
        logging_steps: Interval logging (dalam langkah)
        save_steps: Interval penyimpanan checkpoint (dalam langkah)
        eval_steps: Interval evaluasi (dalam langkah)
        warmup_ratio: Rasio langkah warmup terhadap total langkah
        weight_decay: Weight decay untuk optimizer
        learning_rate: Learning rate awal
        lr_scheduler_type: Tipe scheduler ("cosine", "linear", "constant")
        use_wandb: Aktifkan Wandb logging
        wandb_project: Nama proyek Wandb
        seed: Random seed
        dataloader_num_workers: Jumlah worker untuk dataloader
        dataloader_pin_memory: Pin memory untuk dataloader
        resume_from_checkpoint: Path ke checkpoint untuk resume
        use_fsdp: Aktifkan FSDP (Fully Sharded Data Parallel)
        use_ddp: Aktifkan DDP (Distributed Data Parallel)
    """

    output_dir: str = "./checkpoints"
    num_train_epochs: int = 1
    max_train_steps: int = -1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    learning_rate: float = 3e-4
    lr_scheduler_type: str = "cosine"
    use_wandb: bool = False
    wandb_project: str = "losion"
    seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    resume_from_checkpoint: Optional[str] = None
    use_fsdp: bool = False
    use_ddp: bool = False


# ============================================================================
# LosionTrainer
# ============================================================================


class LosionTrainer:
    """
    Losion Trainer — mengimplementasikan training 4-fase.

    Fase 1: Pre-Training Individual (0-30% budget)
    - Setiap jalur dilatih terpisah (frozen router)

    Fase 2: Joint Fine-Tuning (30-60% budget)
    - Ketiga jalur dilatih bersama, frozen router
    - Bridge mechanism dilatih

    Fase 3: End-to-End RL (60-90% budget)
    - Router di-unfreeze
    - GRPO untuk optimasi routing

    Fase 4: Advanced Optimization (90-100% budget)
    - Early exit, flow matching, distillation

    Mendukung:
    - Distributed training (DDP, FSDP)
    - Mixed precision (bf16, fp8)
    - Gradient accumulation
    - Checkpoint saving/resuming
    - Wandb logging

    Args:
        model_config: LosionConfig untuk model
        trainer_config: TrainerConfig untuk trainer
    """

    def __init__(
        self,
        model_config: LosionConfig,
        trainer_config: Optional[TrainerConfig] = None,
    ) -> None:
        self.model_config = model_config
        self.trainer_config = trainer_config or TrainerConfig()

        # ---- Distributed setup ----
        self.local_rank, self.world_size, self.is_distributed = setup_distributed()

        # ---- Device ----
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}" if self.is_distributed else "cuda")
        else:
            self.device = torch.device("cpu")

        # ---- Seed ----
        torch.manual_seed(self.trainer_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.trainer_config.seed)

        # ---- Build model ----
        self.model = LosionForCausalLM(model_config)
        self.model.to(self.device)

        # ---- Curriculum scheduler ----
        self.curriculum = CurriculumScheduler(model_config)

        # ---- Training state ----
        self.global_step = 0
        self.current_epoch = 0
        self.total_steps = 0
        self.best_eval_loss = float("inf")
        self.training_history: List[Dict[str, Any]] = []

        # ---- Mixed precision ----
        self.scaler: Optional[torch.amp.GradScaler] = None
        if self.trainer_config.fp16:
            self.scaler = torch.amp.GradScaler("cuda")

        # ---- Wandb ----
        self.wandb_run = None
        if self.trainer_config.use_wandb and self._is_main_process():
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.trainer_config.wandb_project,
                    config={
                        "model": str(model_config),
                        "trainer": str(self.trainer_config),
                    },
                )
            except ImportError:
                logger.warning("wandb tidak terinstal. Logging ke wandb dinonaktifkan.")

        # ---- Apply initial phase ----
        self.curriculum.set_phase(TrainingPhase.PHASE_1_INDIVIDUAL)
        self._apply_phase_to_model(TrainingPhase.PHASE_1_INDIVIDUAL)

        # ---- Log model info ----
        if self._is_main_process():
            param_counts = self.model.count_parameters()
            total_params = param_counts.get("total", 0)
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logger.info(
                f"LosionForCausalLM: {total_params:,} total parameters, "
                f"{trainable_params:,} trainable ({trainable_params/total_params:.1%})"
            )
            vram_estimate = estimate_training_memory(model_config)
            logger.info(f"Estimated VRAM needed: {vram_estimate / (1024**3):.1f} GB")

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Jalankan training 4-fase.

        Fase ditentukan oleh CurriculumScheduler berdasarkan
        progres training (step count dan validation metrics).

        Args:
            train_dataloader: DataLoader untuk data pelatihan
            eval_dataloader: DataLoader untuk data evaluasi (opsional)

        Returns:
            Dictionary berisi ringkasan training
        """
        # ---- Hitung total steps ----
        if self.trainer_config.max_train_steps > 0:
            self.total_steps = self.trainer_config.max_train_steps
        else:
            num_update_steps_per_epoch = len(train_dataloader) // self.trainer_config.gradient_accumulation_steps
            self.total_steps = self.trainer_config.num_train_epochs * num_update_steps

        # ---- Setup optimizer dan scheduler ----
        optimizer, scheduler = self._create_optimizer_and_scheduler()

        # ---- Resume dari checkpoint jika ada ----
        if self.trainer_config.resume_from_checkpoint:
            self._resume_from_checkpoint(
                self.trainer_config.resume_from_checkpoint, optimizer, scheduler
            )

        # ---- Wrap model untuk distributed training ----
        model = self._wrap_model_for_distributed()

        # ---- Training loop ----
        logger.info("===== Memulai Training Losion 4-Fase =====")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {'amp=' + self.model_config.training.amp_dtype if self.model_config.training.use_amp else 'bf16' if self.trainer_config.bf16 else 'fp16' if self.trainer_config.fp16 else 'fp32'}")

        model.train()
        start_time = time.time()

        for epoch in range(self.trainer_config.num_train_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_dataloader):
                # ---- Cek dan transisi fase ----
                current_phase = self.curriculum.current_phase
                self.curriculum.update(self.global_step, epoch_loss if num_batches > 0 else None)

                if self.curriculum.current_phase != current_phase:
                    self._apply_phase_to_model(self.curriculum.current_phase)
                    # Recreate optimizer untuk parameter yang berubah
                    optimizer, scheduler = self._create_optimizer_and_scheduler()
                    logger.info(
                        f"Fase berubah: {current_phase.value} → {self.curriculum.current_phase.value}"
                    )

                # ---- Pindahkan batch ke device ----
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # ---- Forward pass dengan mixed precision ----
                # Determine autocast settings from model_config.training (LosionConfig)
                # and fallback to trainer_config for backward compatibility
                use_amp = self.model_config.training.use_amp
                if use_amp:
                    amp_dtype_str = self.model_config.training.amp_dtype
                    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
                    autocast_device = self.device.type if self.device.type == "cuda" else "cpu"
                else:
                    # Fallback ke logic lama dari trainer_config
                    use_bf16 = self.trainer_config.bf16 and self.device.type == "cuda"
                    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
                    autocast_device = "cuda"
                    use_amp = use_bf16 or self.trainer_config.fp16

                with torch.amp.autocast(autocast_device, dtype=amp_dtype, enabled=use_amp):
                    # Tentukan thinking mode berdasarkan fase
                    thinking_mode = None
                    if self.curriculum.current_phase == TrainingPhase.PHASE_3_RL:
                        thinking_mode = True  # Aktifkan thinking di fase RL
                    elif self.curriculum.current_phase == TrainingPhase.PHASE_4_ADVANCED:
                        thinking_mode = True

                    # Tentukan apakah menggunakan evo recycling
                    use_evo = (
                        self.curriculum.current_phase == TrainingPhase.PHASE_4_ADVANCED
                    )

                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        thinking_mode=thinking_mode,
                        use_evo_recycling=use_evo,
                    )

                    loss = output.loss
                    if loss is None:
                        continue

                    # Gradient accumulation
                    loss = loss / self.trainer_config.gradient_accumulation_steps

                # ---- Backward pass ----
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # ---- Optimizer step (dengan gradient accumulation) ----
                if (batch_idx + 1) % self.trainer_config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.trainer_config.max_grad_norm,
                    )

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()

                    self.global_step += 1

                    # ---- Logging ----
                    epoch_loss += loss.item() * self.trainer_config.gradient_accumulation_steps
                    num_batches += 1

                    if self.global_step % self.trainer_config.logging_steps == 0:
                        avg_loss = epoch_loss / max(num_batches, 1)
                        lr = scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        steps_per_sec = self.global_step / max(elapsed, 1)

                        log_dict = {
                            "step": self.global_step,
                            "epoch": epoch,
                            "loss": avg_loss,
                            "lr": lr,
                            "phase": self.curriculum.current_phase.value,
                            "steps_per_sec": steps_per_sec,
                        }

                        if output.ar_loss is not None:
                            log_dict["ar_loss"] = output.ar_loss.item()
                        if output.mtp_loss is not None:
                            log_dict["mtp_loss"] = output.mtp_loss.item()

                        self.training_history.append(log_dict)
                        self._log(log_dict)

                    # ---- Evaluation ----
                    if (
                        eval_dataloader is not None
                        and self.global_step % self.trainer_config.eval_steps == 0
                    ):
                        eval_loss = self.evaluate(eval_dataloader)
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self._save_checkpoint("best")

                    # ---- Save checkpoint ----
                    if self.global_step % self.trainer_config.save_steps == 0:
                        self._save_checkpoint(f"step-{self.global_step}")

                # ---- Cek batas langkah ----
                if 0 < self.total_steps <= self.global_step:
                    break

            # ---- End of epoch ----
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                f"Epoch {epoch + 1}/{self.trainer_config.num_train_epochs} — "
                f"Loss: {avg_epoch_loss:.4f} — Phase: {self.curriculum.current_phase.value}"
            )

            if 0 < self.total_steps <= self.global_step:
                break

        # ---- Training selesai ----
        total_time = time.time() - start_time
        logger.info(f"Training selesai dalam {total_time:.1f} detik ({self.global_step} langkah)")

        # Simpan checkpoint terakhir
        self._save_checkpoint("final")

        return {
            "total_steps": self.global_step,
            "total_time": total_time,
            "best_eval_loss": self.best_eval_loss,
            "training_history": self.training_history,
            "final_phase": self.curriculum.current_phase.value,
        }

    def evaluate(self, eval_dataloader: DataLoader) -> float:
        """
        Evaluasi model pada dataset.

        Args:
            eval_dataloader: DataLoader untuk evaluasi

        Returns:
            Rata-rata loss pada dataset evaluasi
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                use_amp = self.model_config.training.use_amp
                if use_amp:
                    amp_dtype_str = self.model_config.training.amp_dtype
                    amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
                    autocast_device = self.device.type if self.device.type == "cuda" else "cpu"
                else:
                    use_bf16 = self.trainer_config.bf16 and self.device.type == "cuda"
                    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
                    autocast_device = "cuda"
                    use_amp = use_bf16 or self.trainer_config.fp16

                with torch.amp.autocast(autocast_device, dtype=amp_dtype, enabled=use_amp):
                    output = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                    )

                if output.loss is not None:
                    total_loss += output.loss.item()
                    num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.model.train()

        logger.info(f"Evaluasi: Loss = {avg_loss:.4f}")
        self._log({"eval_loss": avg_loss})

        return avg_loss

    def _apply_phase_to_model(self, phase: TrainingPhase) -> None:
        """
        Terapkan konfigurasi fase ke model.

        Mengatur parameter mana yang frozen/unfrozen berdasarkan fase:
        - Fase 1: Hanya jalur target yang unfrozen
        - Fase 2: Semua jalur unfrozen, router frozen
        - Fase 3: Router di-unfreeze
        - Fase 4: Semua unfrozen

        Args:
            phase: Fase training saat ini
        """
        model = self.model

        if phase == TrainingPhase.PHASE_1_INDIVIDUAL:
            # Freeze semua, lalu unfreeze jalur yang ditarget
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze target pathway (Jalur 1 SSM)
            target_pathway = self.curriculum.get_current_target_pathway()
            self._unfreeze_pathway(target_pathway)

            # Unfreeze embedding dan LM head (selalu diperlukan)
            for param in model.model.token_embedding.parameters():
                param.requires_grad = True
            for param in model.lm_head.parameters():
                param.requires_grad = True

        elif phase == TrainingPhase.PHASE_2_JOINT:
            # Unfreeze semua jalur, freeze router
            for param in model.parameters():
                param.requires_grad = True

            # Freeze router
            for layer in model.model.layers:
                for param in layer.router.parameters():
                    param.requires_grad = False

        elif phase == TrainingPhase.PHASE_3_RL:
            # Unfreeze semua termasuk router
            for param in model.parameters():
                param.requires_grad = True

        elif phase == TrainingPhase.PHASE_4_ADVANCED:
            # Semua unfrozen
            for param in model.parameters():
                param.requires_grad = True

    def _unfreeze_pathway(self, pathway: str) -> None:
        """
        Unfreeze parameter jalur tertentu di semua layer.

        Args:
            pathway: "ssm", "attention", atau "retrieval"
        """
        for layer in self.model.model.layers:
            if pathway == "ssm":
                target = layer.ssm_layer
            elif pathway == "attention":
                target = layer.attn_layer
            elif pathway == "retrieval":
                target = layer.retrieval_layer
            else:
                continue

            for param in target.parameters():
                param.requires_grad = True

    def _create_optimizer_and_scheduler(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Buat optimizer dan learning rate scheduler.

        Menggunakan AdamW dengan weight decay yang berbeda
        untuk parameter bias dan weight.

        Returns:
            Tuple (optimizer, scheduler)
        """
        # Pisahkan parameter berdasarkan weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "LayerNorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.trainer_config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.trainer_config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Learning rate scheduler
        warmup_steps = int(self.total_steps * self.trainer_config.warmup_ratio)

        if self.trainer_config.lr_scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.trainer_config.lr_scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_steps,
            )
        else:
            # Constant LR
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=self.total_steps,
            )

        return optimizer, scheduler

    def _wrap_model_for_distributed(self) -> nn.Module:
        """
        Wrap model untuk distributed training.

        Returns:
            Model yang sudah di-wrap
        """
        if self.trainer_config.use_fsdp and self.is_distributed:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            return FSDP(self.model)

        if self.trainer_config.use_ddp and self.is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            return DDP(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
            )

        return self.model

    def _save_checkpoint(self, step_name: str) -> None:
        """
        Simpan checkpoint.

        Args:
            step_name: Nama checkpoint (misalnya "step-1000", "best", "final")
        """
        if not self._is_main_process():
            return

        save_dir = os.path.join(self.trainer_config.output_dir, f"checkpoint-{step_name}")
        os.makedirs(save_dir, exist_ok=True)

        save_checkpoint(
            model=self.model,
            optimizer=None,  # Tidak simpan optimizer untuk hemat ruang
            scheduler=None,
            global_step=self.global_step,
            save_dir=save_dir,
        )

        # Simpan juga konfigurasi training
        self.model.save_pretrained(save_dir)

        logger.info(f"Checkpoint disimpan: {save_dir}")

    def _resume_from_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ) -> None:
        """
        Resume training dari checkpoint.

        Args:
            checkpoint_path: Path ke direktori checkpoint
            optimizer: Optimizer yang akan di-load
            scheduler: Scheduler yang akan di-load
        """
        checkpoint_data = load_checkpoint(checkpoint_path, self.device)

        if checkpoint_data is not None:
            self.model.load_state_dict(checkpoint_data["model_state_dict"])

            if "optimizer_state_dict" in checkpoint_data and optimizer is not None:
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint_data and scheduler is not None:
                scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])

            if "global_step" in checkpoint_data:
                self.global_step = checkpoint_data["global_step"]

            logger.info(f"Resumed dari checkpoint: {checkpoint_path} (step {self.global_step})")

    def _is_main_process(self) -> bool:
        """Periksa apakah ini proses utama (rank 0)."""
        if not self.is_distributed:
            return True
        return self.local_rank == 0

    def _log(self, log_dict: Dict[str, Any]) -> None:
        """
        Log metrics ke console dan wandb.

        Args:
            log_dict: Dictionary berisi metrics
        """
        # Console logging
        if self._is_main_process():
            log_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in log_dict.items())
            logger.info(log_str)

        # Wandb logging
        if self.wandb_run is not None and self._is_main_process():
            try:
                import wandb
                wandb.log(log_dict, step=self.global_step)
            except Exception as e:
                logger.warning(f"Gagal log ke wandb: {e}")
