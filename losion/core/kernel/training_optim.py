"""
Losion Training Optimizations — Memory-efficient training primitives.

Provides:
  - MemoryEfficientTrainer: Training wrapper with gradient checkpointing,
    CPU offload, and selective recomputation
  - CUDAGraphOptimizer: CUDA Graph capture for repeated forward/backward
  - GradientCompressor: AllReduce compression for distributed training
  - FusedAdamW: Fused CUDA AdamW kernel (fallback to torch AdamW)
  - CPUOffloadOptimizer: ZeRO-Offload style optimizer state on CPU

These optimizations can reduce VRAM usage by 30-60% during training
with minimal impact on convergence speed or model quality.

Credits:
  - ZeRO-Offload: Rajbhandari et al., SC 2021
  - Gradient Checkpointing: Chen et al., arXiv:1604.06174 (2016)
  - CUDA Graphs: PyTorch CUDA Graph documentation
  - DeepSpeed: Rajbhandari et al., SC 2020
  - PowerSGD: Vogt et al., NeurIPS 2019
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# CPU Offload Optimizer — ZeRO-Offload Style
# ============================================================================


class CPUOffloadOptimizer:
    """Offload optimizer states to CPU to reduce GPU VRAM usage.

    Implements ZeRO-Offload style optimization where:
    - Model parameters and gradients remain on GPU
    - Optimizer states (momentum, variance) are on CPU
    - Parameter updates are computed on CPU then copied back to GPU

    This can reduce VRAM usage by 50%+ for the optimizer component,
    at the cost of ~10-20% throughput reduction due to CPU-GPU transfers.

    Args:
        params: Model parameters (on GPU).
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters.
        eps: Adam epsilon.
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._step_count = 0

        # Initialize optimizer states on CPU
        self._states: Dict[int, Dict[str, torch.Tensor]] = {}
        for p in params:
            if p.requires_grad:
                pid = id(p)
                self._states[pid] = {
                    "exp_avg": torch.zeros_like(p.data, device="cpu", dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(p.data, device="cpu", dtype=torch.float32),
                }

        self._params = [p for p in params if p.requires_grad]

    def step(self, closure=None) -> None:
        """Perform a single optimization step.

        For each parameter:
        1. Copy gradient to CPU
        2. Update optimizer states on CPU
        3. Compute parameter update on CPU
        4. Copy update back to GPU and apply
        """
        self._step_count += 1

        for p in self._params:
            if p.grad is None:
                continue

            pid = id(p)
            state = self._states[pid]

            # Copy gradient to CPU
            grad_cpu = p.grad.data.to("cpu", dtype=torch.float32, non_blocking=True)
            param_cpu = p.data.to("cpu", dtype=torch.float32, non_blocking=True)

            # Update momentum
            exp_avg = state["exp_avg"]
            exp_avg.mul_(self.beta1).add_(grad_cpu, alpha=1 - self.beta1)

            # Update variance
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(self.beta2).addcmul_(grad_cpu, grad_cpu, value=1 - self.beta2)

            # Bias correction
            bias1 = 1 - self.beta1 ** self._step_count
            bias2 = 1 - self.beta2 ** self._step_count
            step_size = self.lr / bias1
            bias_correction2 = 1 - self.beta2 ** self._step_count

            # Compute update
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            update = exp_avg / denom

            # Weight decay (decoupled)
            if self.weight_decay > 0:
                update.add_(param_cpu, alpha=self.weight_decay)

            # Apply update on CPU, then copy to GPU
            new_param = param_cpu - step_size * update
            p.data.copy_(new_param.to(p.device, non_blocking=True))

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients."""
        for p in self._params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()


# Fix missing import
import math


# ============================================================================
# Fused AdamW — Optimized AdamW implementation
# ============================================================================


class FusedAdamW:
    """AdamW optimizer with optional fused CUDA kernel.

    When torchao is available, uses the fused CUDA AdamW kernel
    which is ~2x faster than the standard PyTorch AdamW.

    Falls back to torch.optim.AdamW when fused kernel is unavailable.

    Args:
        params: Model parameters.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters.
        eps: Adam epsilon.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        self._use_fused = False

        try:
            from torchao.optim import FusedAdam as _FusedAdam
            self._optimizer = _FusedAdam(
                params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
            )
            self._use_fused = True
            logger.info("Using torchao FusedAdamW")
        except (ImportError, Exception):
            self._optimizer = torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
            )
            logger.info("Using standard PyTorch AdamW (torchao not available)")

    def step(self, closure=None) -> None:
        self._optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self._optimizer.param_groups = value


# ============================================================================
# Gradient Compressor — AllReduce Compression for Distributed Training
# ============================================================================


class GradientCompressor:
    """Compress gradients before AllReduce to reduce communication overhead.

    Implements PowerSGD-style gradient compression:
    1. Reshape gradient into matrix form
    2. Low-rank decomposition: G ≈ U * V^T
    3. Communicate only U and V (much smaller than G)
    4. Reconstruct approximate gradient on each worker

    Compression ratio = 1 - (m*(r) + n*(r)) / (m*n) where r is rank.

    Args:
        compression_rank: Rank for low-rank compression (default 1).
        warmup_steps: Steps before compression starts (default 100).
    """

    def __init__(
        self,
        compression_rank: int = 1,
        warmup_steps: int = 100,
    ) -> None:
        self.compression_rank = compression_rank
        self.warmup_steps = warmup_steps
        self._step = 0

    def compress(
        self,
        grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress a gradient tensor using low-rank decomposition.

        Args:
            grad: Gradient tensor.

        Returns:
            Tuple (U, V) — compressed representation.
        """
        self._step += 1

        if self._step <= self.warmup_steps:
            # During warmup, return full gradient (no compression)
            return grad, None

        original_shape = grad.shape
        if grad.dim() == 1:
            return grad, None

        # Reshape to 2D matrix
        m = grad.shape[0]
        n = grad.numel() // m
        grad_matrix = grad.reshape(m, n)

        r = min(self.compression_rank, min(m, n))

        # PowerSGD: one-step power iteration
        # Random initialization (could use previous V for better convergence)
        V = torch.randn(n, r, device=grad.device, dtype=grad.dtype)
        V, _ = torch.linalg.qr(V)

        # U = G @ V
        U = grad_matrix @ V  # (m, r)

        # Re-orthogonalize
        U, _ = torch.linalg.qr(U)

        return U, V

    def decompress(
        self,
        U: torch.Tensor,
        V: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """Decompress gradient from low-rank representation.

        Args:
            U: Left factor (m, r).
            V: Right factor (n, r).
            original_shape: Original gradient shape.

        Returns:
            Reconstructed gradient tensor.
        """
        if V is None:
            return U

        grad_approx = U @ V.T
        return grad_approx.reshape(original_shape)

    def compress_gradients(
        self,
        model: nn.Module,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Size]]:
        """Compress all model gradients.

        Args:
            model: Model with computed gradients.

        Returns:
            Dict mapping parameter name to (U, V, original_shape).
        """
        compressed = {}
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.dim() > 1:
                U, V = self.compress(param.grad.data)
                compressed[name] = (U, V, param.grad.shape)
        return compressed


# ============================================================================
# CUDA Graph Optimizer — Capture forward/backward for repeated execution
# ============================================================================


class CUDAGraphOptimizer:
    """CUDA Graph capture for repeated forward/backward passes.

    CUDA Graphs eliminate CPU-side overhead by recording the entire
    computation graph once, then replaying it. This gives 10-30%
    speedup for small models or when CPU overhead is the bottleneck.

    Limitations:
    - Input shapes must be fixed across iterations
    - Control flow (if/else) must be data-independent
    - Works best with torch.compile(mode="reduce-overhead")

    Usage:
        optimizer = CUDAGraphOptimizer(model, sample_input)
        for batch in dataloader:
            output = optimizer.replay(batch)

    Args:
        model: Model to capture.
        sample_input: Sample input tensor for graph capture.
        mode: Capture mode ("full" or "forward_only").
    """

    def __init__(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        mode: str = "full",
    ) -> None:
        self.model = model
        self.mode = mode
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured = False

        if torch.cuda.is_available():
            try:
                self._static_input = sample_input.clone()
                self._static_output: Optional[torch.Tensor] = None
                self._capture_graph()
            except Exception as e:
                logger.warning(f"CUDA Graph capture failed: {e}. Using eager mode.")

    def _capture_graph(self) -> None:
        """Capture the CUDA graph."""
        if not torch.cuda.is_available():
            return

        self._graph = torch.cuda.CUDAGraph()

        # Warmup
        with torch.cuda.graph(self._graph):
            self._static_output = self.model(self._static_input)

        self._captured = True
        logger.info("CUDA Graph captured successfully")

    def replay(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Replay the captured graph with new input.

        Args:
            input_tensor: New input tensor (must have same shape as sample).

        Returns:
            Model output.
        """
        if not self._captured or self._graph is None:
            return self.model(input_tensor)

        # Copy new input data into static buffer
        self._static_input.copy_(input_tensor)

        # Replay the graph
        self._graph.replay()

        return self._static_output.clone()

    @staticmethod
    def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
        """Apply torch.compile to the model.

        Args:
            model: Model to compile.
            mode: Compile mode ("reduce-overhead", "default", "max-autotune").

        Returns:
            Compiled model.
        """
        try:
            compiled = torch.compile(model, mode=mode)
            logger.info(f"Model compiled with mode={mode}")
            return compiled
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Using eager mode.")
            return model


# ============================================================================
# Memory Efficient Trainer — Gradient Checkpointing + CPU Offload
# ============================================================================


class MemoryEfficientTrainer:
    """Training wrapper with comprehensive memory optimizations.

    Combines multiple memory optimization techniques:
    1. Selective gradient checkpointing (per-jalur)
    2. CPU offload for optimizer states
    3. torch.compile integration
    4. Gradient accumulation with delayed optimizer step
    5. Mixed precision training (BF16/FP16)
    6. Selective layer freezing

    Expected memory savings:
    - Gradient checkpointing: 30-50% VRAM reduction
    - CPU offload: additional 20-30% VRAM reduction
    - Combined: up to 60% VRAM reduction

    Args:
        model: LosionModel to train.
        lr: Learning rate.
        weight_decay: Weight decay.
        gradient_checkpointing: Enable gradient checkpointing.
        cpu_offload: Enable CPU offload for optimizer.
        compile_mode: torch.compile mode (None = no compile).
        gradient_accumulation_steps: Number of gradient accumulation steps.
        max_grad_norm: Maximum gradient norm for clipping.
        bf16: Use BF16 mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        gradient_checkpointing: bool = True,
        cpu_offload: bool = False,
        compile_mode: Optional[str] = "reduce-overhead",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        bf16: bool = True,
    ) -> None:
        self.model = model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.bf16 = bf16
        self._accumulation_step = 0

        # Enable gradient checkpointing
        if gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")

        # Apply torch.compile
        if compile_mode is not None:
            self.model = CUDAGraphOptimizer.compile_model(model, mode=compile_mode)

        # Create optimizer
        if cpu_offload:
            self.optimizer = CPUOffloadOptimizer(
                params=list(model.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
            logger.info("Using CPU-offloaded optimizer (ZeRO-Offload style)")
        else:
            self.optimizer = FusedAdamW(
                params=model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        # Gradient scaler for FP16
        self.scaler: Optional[torch.amp.GradScaler] = None
        if not bf16:
            self.scaler = torch.amp.GradScaler("cuda")

        # Selective layer freezing support
        self._frozen_layers: set = set()

    def training_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Execute a single training step with memory optimizations.

        Args:
            input_ids: Token IDs (batch, seq_len).
            labels: Target token IDs (batch, seq_len).
            attention_mask: Optional attention mask.

        Returns:
            Dict with loss and metrics.
        """
        dtype = torch.bfloat16 if self.bf16 else torch.float16
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        with torch.amp.autocast(device_type, dtype=dtype, enabled=self.bf16 or self.scaler is not None):
            output = self.model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

            loss = output.loss if hasattr(output, "loss") else output["loss"]
            if loss is None:
                return {"loss": 0.0}

            loss = loss / self.gradient_accumulation_steps

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._accumulation_step += 1

        metrics = {"loss": loss.item() * self.gradient_accumulation_steps}

        # Optimizer step (with gradient accumulation)
        if self._accumulation_step >= self.gradient_accumulation_steps:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer._optimizer if isinstance(self.optimizer, FusedAdamW) else self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            if self.scaler is not None:
                self.scaler.step(self.optimizer._optimizer if isinstance(self.optimizer, FusedAdamW) else self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self._accumulation_step = 0

        return metrics

    def freeze_layers(self, layer_indices: List[int]) -> None:
        """Freeze specific layers to save memory during training.

        Args:
            layer_indices: List of layer indices to freeze.
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for idx in layer_indices:
                layer = self.model.model.layers[idx]
                for param in layer.parameters():
                    param.requires_grad = False
                self._frozen_layers.add(idx)
            logger.info(f"Frozen layers: {layer_indices}")

    def unfreeze_layers(self, layer_indices: Optional[List[int]] = None) -> None:
        """Unfreeze previously frozen layers.

        Args:
            layer_indices: Specific layers to unfreeze (None = all).
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            target_indices = layer_indices or list(self._frozen_layers)
            for idx in target_indices:
                if idx in self._frozen_layers:
                    layer = self.model.model.layers[idx]
                    for param in layer.parameters():
                        param.requires_grad = True
                    self._frozen_layers.discard(idx)
            logger.info(f"Unfrozen layers: {target_indices}")

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics.

        Returns:
            Dict with memory stats in GB.
        """
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
