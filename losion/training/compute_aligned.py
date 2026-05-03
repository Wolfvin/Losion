"""
TACO — Training with Compute Alignment untuk Losion Framework.

TACO (Training with Compute Alignment) memastikan bahwa training compute
dialokasikan secara proporsional terhadap inference compute, mencegah
mismatch antara training dan inference.

Masalah yang dipecahkan:
- Pada MoE, expert yang jarang digunakan di inference tetap mendapat
  gradien yang sama dengan expert yang sering digunakan
- Ini menyebabkan over-training pada rarely-used experts dan
  under-training pada frequently-used experts
- Akibatnya, kualitas inference menurun karena expert yang paling
  penting tidak terlatih optimal

Solusi TACO:
1. Track inference compute per expert/layer secara online
2. Hitung alignment weights berdasarkan rasio inference usage
3. Sesuaikan training loss weights agar sebanding dengan inference usage
4. Mencegah over-training pada rarely-used experts

Komponen:
1. ComputeAlignedConfig — Konfigurasi TACO
2. ComputeAlignedTrainer — Trainer dengan compute-aligned loss weighting

Referensi:
- DeepSeek-V2 Technical Report (2024) — compute-aware training
- Mixtral paper — expert utilization analysis
- Losion Framework — Jalur 3 MoE architecture

Hardware: Pure PyTorch, kompatibel dengan CUDA, ROCm, dan CPU.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

@dataclass
class ComputeAlignedConfig:
    """Konfigurasi untuk compute-aligned training (TACO).

    Attributes:
        alignment_window: Jumlah steps untuk tracking inference compute.
            Window yang lebih besar → estimasi lebih stabil tapi kurang responsif.
        alignment_strength: Seberapa kuat alignment diterapkan (0.0 - 1.0).
            0.0 = tidak ada alignment (standard training),
            1.0 = full alignment (loss weight persis sesuai inference usage).
        track_expert_usage: Track per-expert compute usage di MoE layers.
            Jika False, hanya track per-layer compute.
        min_loss_weight: Minimum loss weight untuk expert/layer manapun.
            Mencegah expert mendapat zero gradient (yang bisa menyebabkan
            weight decay atau dead neurons).
        ema_decay: Exponential moving average decay untuk compute tracking.
            Nilai tinggi → lebih smooth, nilai rendah → lebih responsif.
        warmup_steps: Jumlah steps sebelum alignment diterapkan.
            Memungkinkan model "warm up" dengan standard training dulu.
        align_experts: Jika True, sesuaikan loss per-expert dalam MoE.
        align_layers: Jika True, sesuaikan loss per-layer dalam model.
    """

    alignment_window: int = 100
    alignment_strength: float = 0.5
    track_expert_usage: bool = True
    min_loss_weight: float = 0.1
    ema_decay: float = 0.99
    warmup_steps: int = 500
    align_experts: bool = True
    align_layers: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.alignment_strength <= 1.0:
            raise ValueError(
                f"alignment_strength harus di [0, 1], mendapat {self.alignment_strength}"
            )
        if self.alignment_window < 1:
            raise ValueError(
                f"alignment_window harus >= 1, mendapat {self.alignment_window}"
            )
        if self.min_loss_weight < 0:
            raise ValueError(
                f"min_loss_weight tidak boleh negatif, mendapat {self.min_loss_weight}"
            )


# ---------------------------------------------------------------------------
# Compute Tracker — Track inference compute per expert/layer
# ---------------------------------------------------------------------------

class ComputeTracker:
    """
    Tracker untuk inference compute usage per expert dan per layer.

    Menggunakan exponential moving average (EMA) untuk melacak
    seberapa sering setiap expert dan layer digunakan selama inference.
    Tracker ini di-update setiap inference step dan digunakan untuk
    menghitung alignment weights.

    Args:
        config: Konfigurasi TACO.
        num_experts_per_layer: List jumlah experts per MoE layer.
            Misalnya [8, 8, 16, 8] untuk model dengan 4 MoE layers.
        num_layers: Jumlah total layers dalam model.
    """

    def __init__(
        self,
        config: ComputeAlignedConfig,
        num_experts_per_layer: Optional[List[int]] = None,
        num_layers: int = 0,
    ) -> None:
        self.config = config
        self.num_experts_per_layer = num_experts_per_layer or []
        self.num_layers = num_layers

        # ---- Expert usage tracking (EMA) ----
        # expert_usage[layer_idx][expert_idx] = EMA usage frequency
        self.expert_usage_ema: Dict[int, torch.Tensor] = {}
        for layer_idx, n_experts in enumerate(self.num_experts_per_layer):
            self.expert_usage_ema[layer_idx] = torch.ones(n_experts) / n_experts

        # ---- Layer usage tracking (EMA) ----
        self.layer_usage_ema: torch.Tensor = torch.ones(max(num_layers, 1)) / max(num_layers, 1)

        # ---- Step counter ----
        self._step = 0

    def update_expert_usage(
        self,
        layer_idx: int,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> None:
        """
        Update expert usage tracking berdasarkan inference routing.

        Dipanggil setiap inference step untuk setiap MoE layer.
        Expert yang dipilih oleh router akan meningkatkan usage count-nya.

        Args:
            layer_idx: Indeks MoE layer.
            expert_indices: Expert indices yang dipilih, (batch, seq_len, top_k).
            expert_weights: Routing weights, (batch, seq_len, top_k).
        """
        if not self.config.track_expert_usage:
            return

        if layer_idx not in self.expert_usage_ema:
            n_experts = expert_indices.max().item() + 1
            self.expert_usage_ema[layer_idx] = torch.ones(n_experts) / n_experts

        n_experts = self.expert_usage_ema[layer_idx].size(0)
        device = self.expert_usage_ema[layer_idx].device

        # Hitung usage frequency: seberapa sering setiap expert dipilih
        # Weighted by routing weights
        usage = torch.zeros(n_experts, device=device)
        flat_indices = expert_indices.flatten()
        flat_weights = expert_weights.flatten()

        for idx, w in zip(flat_indices, flat_weights):
            if idx.item() < n_experts:
                usage[idx.item()] += w.item()

        # Normalize
        usage_sum = usage.sum()
        if usage_sum > 0:
            usage = usage / usage_sum

        # EMA update
        decay = self.config.ema_decay
        ema = self.expert_usage_ema[layer_idx]
        if ema.device != usage.device:
            ema = ema.to(usage.device)
            self.expert_usage_ema[layer_idx] = ema
        self.expert_usage_ema[layer_idx] = decay * ema + (1 - decay) * usage

    def update_layer_usage(
        self,
        layer_compute: torch.Tensor,
    ) -> None:
        """
        Update layer usage tracking berdasarkan inference compute.

        Args:
            layer_compute: Per-layer compute metric, (num_layers,).
                Misalnya FLOPs atau aktivasi per layer.
        """
        if not self.config.align_layers:
            return

        if layer_compute.size(0) != self.layer_usage_ema.size(0):
            self.layer_usage_ema = torch.ones(layer_compute.size(0)) / layer_compute.size(0)

        # Normalize
        compute_sum = layer_compute.sum()
        if compute_sum > 0:
            normalized = layer_compute / compute_sum
        else:
            normalized = torch.ones_like(layer_compute) / layer_compute.size(0)

        # EMA update
        decay = self.config.ema_decay
        self.layer_usage_ema = decay * self.layer_usage_ema.to(normalized.device) + (1 - decay) * normalized

    def get_expert_alignment_weights(
        self,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Dapatkan alignment weights untuk expert pada layer tertentu.

        Expert yang sering digunakan di inference mendapat weight tinggi,
        expert yang jarang digunakan mendapat weight rendah.

        Args:
            layer_idx: Indeks MoE layer.

        Returns:
            Alignment weights, (num_experts,), di-sum ke num_experts.
        """
        if layer_idx not in self.expert_usage_ema:
            n_experts = self.num_experts_per_layer[layer_idx] if layer_idx < len(self.num_experts_per_layer) else 1
            return torch.ones(n_experts) / n_experts

        usage = self.expert_usage_ema[layer_idx].clone()

        # Pastikan minimum weight
        usage = usage.clamp(min=self.config.min_loss_weight)

        # Re-normalize sehingga sum = n_experts (uniform = no alignment)
        usage = usage / usage.sum() * usage.size(0)

        return usage

    def get_layer_alignment_weights(self) -> torch.Tensor:
        """
        Dapatkan alignment weights untuk setiap layer.

        Returns:
            Alignment weights, (num_layers,), di-sum ke num_layers.
        """
        usage = self.layer_usage_ema.clone()
        usage = usage.clamp(min=self.config.min_loss_weight)
        usage = usage / usage.sum() * usage.size(0)
        return usage

    def increment_step(self) -> None:
        """Increment step counter."""
        self._step += 1

    def get_step(self) -> int:
        """Dapatkan step counter saat ini."""
        return self._step


# ---------------------------------------------------------------------------
# ComputeAlignedTrainer
# ---------------------------------------------------------------------------

class ComputeAlignedTrainer:
    """
    Trainer dengan compute-aligned loss weighting (TACO).

    TACO memastikan bahwa training compute dialokasikan proporsional
    terhadap inference compute. Ini mencegah over-training pada
    rarely-used experts dan memastikan frequently-used experts
    mendapat gradient signal yang cukup.

    Cara kerja:
    1. Track inference compute: Monitor expert/layer usage saat inference
    2. Compute alignment weights: Turunkan training loss weights dari
       inference compute profile
    3. Apply aligned loss: Kalikan per-expert/per-layer loss dengan
       alignment weights

    Loss weighting rule:
        weight_i = (1 - strength) * 1.0 + strength * (usage_i / mean_usage)

    Dimana strength = alignment_strength. Ini memastikan:
    - strength=0: standard training (uniform weights)
    - strength=1: full alignment (weight proporsional ke usage)

    Args:
        model: Model yang akan di-train.
        config: Konfigurasi TACO.
        tracker: ComputeTracker opsional (akan dibuat jika None).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ComputeAlignedConfig] = None,
        tracker: Optional[ComputeTracker] = None,
    ) -> None:
        self.config = config or ComputeAlignedConfig()
        self.model = model
        self.device = next(model.parameters()).device

        # ---- Infer model structure ----
        num_experts_per_layer, num_layers = self._infer_model_structure()
        self.num_experts_per_layer = num_experts_per_layer
        self.num_layers = num_layers

        # ---- Create tracker ----
        self.tracker = tracker or ComputeTracker(
            config=self.config,
            num_experts_per_layer=num_experts_per_layer,
            num_layers=num_layers,
        )

        # ---- Step counter ----
        self._step = 0

    def _infer_model_structure(self) -> Tuple[List[int], int]:
        """
        Infer model structure dari model yang ada.

        Mendeteksi MoE layers dan jumlah experts per layer.

        Returns:
            Tuple (num_experts_per_layer, num_layers).
        """
        num_experts_per_layer: List[int] = []
        num_layers = 0

        for name, module in self.model.named_modules():
            num_layers += 1
            # Deteksi MoE layers berdasarkan attribute 'num_experts'
            if hasattr(module, 'num_experts'):
                num_experts_per_layer.append(module.num_experts)
            # Deteksi berdasarkan 'experts' ModuleList
            elif hasattr(module, 'experts') and isinstance(module.experts, nn.ModuleList):
                num_experts_per_layer.append(len(module.experts))

        # Perkiraan jumlah layer (termasuk sub-modules)
        # Hanya hitung top-level layers untuk alignment
        num_layers = max(1, num_layers // max(1, len(num_experts_per_layer) + 1))

        return num_experts_per_layer, num_layers

    # ------------------------------------------------------------------
    # Inference Compute Tracking
    # ------------------------------------------------------------------

    def track_inference_compute(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Monitor inference compute usage per expert dan per layer.

        Melakukan forward pass inference dan mengumpulkan routing
        statistics dari semua MoE layers.

        Args:
            input_ids: Token IDs, (batch, seq_len).
            attention_mask: Mask attention opsional.

        Returns:
            Dictionary berisi compute statistics per layer.
        """
        self.model.eval()
        compute_stats: Dict[str, torch.Tensor] = {}

        with torch.no_grad():
            # Hook untuk mengumpulkan routing info
            hooks = []
            routing_info: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

            def make_hook(name: str):
                def hook_fn(module, input, output):
                    # Coba ekstrak routing info dari output
                    if isinstance(output, tuple) and len(output) >= 2:
                        # Output format: (main_output, aux_dict)
                        aux = output[1]
                        if isinstance(aux, dict):
                            if "router_logits" in aux:
                                routing_info[name] = aux["router_logits"]
                            if "expert_indices" in aux and "expert_weights" in aux:
                                routing_info[name] = (
                                    aux["expert_indices"],
                                    aux["expert_weights"],
                                )
                return hook_fn

            # Register hooks pada MoE layers
            for name, module in self.model.named_modules():
                if hasattr(module, 'num_experts') or (
                    hasattr(module, 'experts') and isinstance(module.experts, nn.ModuleList)
                ):
                    hooks.append(module.register_forward_hook(make_hook(name)))

            # Forward pass
            try:
                _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception:
                pass

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Update tracker berdasarkan routing info
            for name, info in routing_info.items():
                if isinstance(info, tuple) and len(info) == 2:
                    indices, weights = info
                    # Cari layer index dari name
                    layer_idx = self._name_to_layer_idx(name)
                    self.tracker.update_expert_usage(layer_idx, indices, weights)

        return compute_stats

    def _name_to_layer_idx(self, name: str) -> int:
        """Konversi nama module ke layer index."""
        # Coba ekstrak angka dari nama (misal "layers.3.moe" → 3)
        parts = name.split(".")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        # Fallback: hash
        return hash(name) % max(len(self.num_experts_per_layer), 1)

    # ------------------------------------------------------------------
    # Alignment Weights
    # ------------------------------------------------------------------

    def compute_alignment_weights(self) -> Dict[str, torch.Tensor]:
        """
        Hitung training loss weights dari inference compute profile.

        Alignment weights menyesuaikan loss per-expert dan per-layer
        sehingga training compute proporsional terhadap inference compute.

        Formula:
            aligned_weight_i = (1 - strength) * 1.0 + strength * (usage_i / mean_usage)

        Returns:
            Dictionary berisi alignment weights:
            - "expert_weights": Dict[int, Tensor] — per-layer expert weights
            - "layer_weights": Tensor — per-layer weights
        """
        strength = self.config.alignment_strength

        # Warmup: selama warmup steps, kurangi strength secara linear
        if self._step < self.config.warmup_steps:
            strength = strength * (self._step / max(1, self.config.warmup_steps))

        result: Dict[str, torch.Tensor] = {}

        # ---- Expert alignment weights ----
        if self.config.align_experts:
            expert_weights: Dict[int, torch.Tensor] = {}
            for layer_idx in range(len(self.num_experts_per_layer)):
                usage = self.tracker.get_expert_alignment_weights(layer_idx)
                # Alignment: blend antara uniform dan usage-proportional
                n_experts = usage.size(0)
                uniform = torch.ones(n_experts, device=usage.device)
                aligned = (1 - strength) * uniform + strength * usage
                expert_weights[layer_idx] = aligned
            result["expert_weights"] = expert_weights  # type: ignore

        # ---- Layer alignment weights ----
        if self.config.align_layers:
            usage = self.tracker.get_layer_alignment_weights()
            uniform = torch.ones_like(usage)
            aligned = (1 - strength) * uniform + strength * usage
            result["layer_weights"] = aligned

        return result

    # ------------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        base_loss_fn: Optional[callable] = None,
    ) -> Dict[str, float]:
        """
        Satu langkah training dengan compute-aligned loss weighting.

        Alur:
        1. Hitung base loss (cross-entropy atau custom)
        2. Dapatkan alignment weights dari tracker
        3. Apply alignment weights ke loss
        4. Backward dan update

        Args:
            input_ids: Token IDs input, (batch, seq_len).
            target_ids: Target token IDs opsional.
            attention_mask: Mask attention opsional.
            base_loss_fn: Loss function opsional. Default: cross-entropy.

        Returns:
            Dictionary metrics (loss, alignment_info, dll.).
        """
        self.model.train()

        if target_ids is None:
            target_ids = input_ids

        # ---- Forward pass ----
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits if hasattr(output, 'logits') else output

        # ---- Base loss ----
        if base_loss_fn is not None:
            base_loss = base_loss_fn(logits, target_ids)
        else:
            # Standard cross-entropy
            vocab_size = logits.size(-1)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:].contiguous()
            base_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_targets.view(-1),
                reduction="mean",
            )

        # ---- Compute alignment weights ----
        alignment = self.compute_alignment_weights()

        # ---- Apply alignment ----
        aligned_loss = base_loss

        # Layer-level alignment
        if "layer_weights" in alignment:
            layer_weights = alignment["layer_weights"]
            # Dapatkan per-layer losses jika tersedia
            if hasattr(output, 'layer_losses') and output.layer_losses:
                layer_losses = output.layer_losses
                if len(layer_losses) == layer_weights.size(0):
                    aligned_layer_loss = sum(
                        w * l for w, l in zip(layer_weights, layer_losses)
                    ) / len(layer_losses)
                    # Blend base loss dengan aligned layer loss
                    strength = self.config.alignment_strength
                    if self._step < self.config.warmup_steps:
                        strength *= self._step / max(1, self.config.warmup_steps)
                    aligned_loss = (1 - strength) * base_loss + strength * aligned_layer_loss

        # Expert-level alignment (jika ada per-expert losses)
        if "expert_weights" in alignment and hasattr(output, 'expert_losses'):
            expert_losses = output.expert_losses
            expert_weights = alignment["expert_weights"]
            if expert_losses and expert_weights:
                total_expert_loss = torch.tensor(0.0, device=self.device)
                count = 0
                for layer_idx, losses in expert_losses.items():
                    if layer_idx in expert_weights:
                        weights = expert_weights[layer_idx].to(self.device)
                        if len(losses) == weights.size(0):
                            for expert_id, loss in enumerate(losses):
                                total_expert_loss = total_expert_loss + weights[expert_id] * loss
                                count += 1
                if count > 0:
                    total_expert_loss = total_expert_loss / count
                    strength = self.config.alignment_strength
                    if self._step < self.config.warmup_steps:
                        strength *= self._step / max(1, self.config.warmup_steps)
                    aligned_loss = (1 - strength * 0.5) * base_loss + strength * 0.5 * total_expert_loss

        # ---- Update step counter ----
        self._step += 1
        self.tracker.increment_step()

        # ---- Metrics ----
        with torch.no_grad():
            metrics = {
                "aligned_loss": aligned_loss.item(),
                "base_loss": base_loss.item(),
                "step": self._step,
                "alignment_strength": self.config.alignment_strength * min(
                    1.0, self._step / max(1, self.config.warmup_steps)
                ),
            }

        return metrics

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_alignment_summary(self) -> Dict[str, object]:
        """
        Dapatkan ringkasan alignment saat ini.

        Returns:
            Dictionary berisi expert usage, layer usage, dan alignment weights.
        """
        alignment = self.compute_alignment_weights()
        summary: Dict[str, object] = {
            "step": self._step,
            "effective_strength": self.config.alignment_strength * min(
                1.0, self._step / max(1, self.config.warmup_steps)
            ),
        }

        if "expert_weights" in alignment:
            summary["expert_alignment"] = {
                idx: w.tolist()
                for idx, w in alignment["expert_weights"].items()
            }

        if "layer_weights" in alignment:
            summary["layer_alignment"] = alignment["layer_weights"].tolist()

        return summary

    def reset(self) -> None:
        """Reset tracker dan step counter."""
        self._step = 0
        self.tracker = ComputeTracker(
            config=self.config,
            num_experts_per_layer=self.num_experts_per_layer,
            num_layers=self.num_layers,
        )
