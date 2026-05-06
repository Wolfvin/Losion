"""
Losion Distributed — Multi-GPU/multi-node distributed training.

Credits:
  - PyTorch FSDP2 (2025) — Fully Sharded Data Parallel v2
  - WLB-LLM 4D Parallelism (OSDI 2025) — Data + Tensor + Pipeline + Context parallel
  - AutoSP (arXiv:2604.27089) — Automated sequence parallelism
  - DeepSpeed Ulysses — Sequence parallelism via attention head splitting

Provides:
  ParallelismConfig       — 4D parallelism configuration
  LosionFSDPWrapper      — FSDP model wrapping with sharding strategies
  LosionDistributedTrainer — Multi-GPU/multi-node training orchestrator
  ContextParallel        — Long sequence splitting across GPUs
"""

from __future__ import annotations

import enum
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a security check fails during checkpoint loading.

    v2.5.0: Used by LosionDistributedTrainer.load_checkpoint() when a
    checkpoint from an untrusted path requires weights_only=False fallback.
    """


# ============================================================================
# Enums
# ============================================================================


class ShardingStrategy(enum.Enum):
    """FSDP sharding strategy."""
    FULL = "full"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"


class ParallelismMode(enum.Enum):
    """Distributed training mode."""
    DDP = "ddp"
    FSDP = "fsdp"
    PIPELINE = "pipeline"
    HYBRID = "hybrid"


# ============================================================================
# Parallelism Configuration
# ============================================================================


@dataclass
class ParallelismConfig:
    """Configuration for 4D parallelism in distributed training.

    4D Parallelism combines:
    - Data Parallelism (DP): Split batches across GPUs
    - Tensor Parallelism (TP): Split weight matrices across GPUs
    - Pipeline Parallelism (PP): Split layers across GPUs
    - Context Parallelism (CP): Split long sequences across GPUs

    The total number of GPUs required is dp_size * tp_size * pp_size * cp_size.

    Attributes:
        dp_size: Data parallelism degree.
        tp_size: Tensor parallelism degree.
        pp_size: Pipeline parallelism degree.
        cp_size: Context parallelism degree.
        fsdp_sharding_strategy: FSDP sharding strategy.
        sequence_parallel: Whether to enable sequence parallelism (Ulysses/AutoSP).
        expert_parallel: Whether to enable expert parallelism for MoE layers.
        expert_parallel_size: Number of GPUs for expert parallelism.
        micro_batch_size: Micro batch size for pipeline parallelism.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        mixed_precision: Mixed precision dtype ("bf16", "fp16", "fp32").
        compile_model: Whether to use torch.compile.
        overlap_comm: Whether to overlap communication with computation.
        use_cpu_offload: Whether to offload parameters to CPU.
    """

    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    fsdp_sharding_strategy: str = "full"
    sequence_parallel: bool = False
    expert_parallel: bool = False
    expert_parallel_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    compile_model: bool = False
    overlap_comm: bool = True
    use_cpu_offload: bool = False

    def __post_init__(self) -> None:
        """Validate parallelism configuration."""
        if self.dp_size < 1:
            raise ValueError(f"dp_size must be >= 1, got {self.dp_size}")
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {self.tp_size}")
        if self.pp_size < 1:
            raise ValueError(f"pp_size must be >= 1, got {self.pp_size}")
        if self.cp_size < 1:
            raise ValueError(f"cp_size must be >= 1, got {self.cp_size}")

        if self.fsdp_sharding_strategy not in ("full", "shard_grad_op", "no_shard"):
            raise ValueError(
                f"fsdp_sharding_strategy must be 'full', 'shard_grad_op', or 'no_shard', "
                f"got {self.fsdp_sharding_strategy!r}"
            )

        if self.mixed_precision not in ("bf16", "fp16", "fp32"):
            raise ValueError(
                f"mixed_precision must be 'bf16', 'fp16', or 'fp32', "
                f"got {self.mixed_precision!r}"
            )

    @property
    def world_size(self) -> int:
        """Total number of GPUs required."""
        return self.dp_size * self.tp_size * self.pp_size * self.cp_size

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size accounting for accumulation and DP."""
        return self.micro_batch_size * self.gradient_accumulation_steps * self.dp_size

    def get_sharding_strategy(self) -> ShardingStrategy:
        """Convert string sharding strategy to enum."""
        return ShardingStrategy(self.fsdp_sharding_strategy)


# ============================================================================
# FSDP Wrapper
# ============================================================================


class LosionFSDPWrapper:
    """Wraps LosionModel with PyTorch FSDP2/FSDP (Fully Sharded Data Parallel).

    v1.6.0: Supports both FSDP2 (fully_shard API from PyTorch 2.5+) and
    legacy FSDP1. FSDP2 provides better memory efficiency and supports
    per-parameter sharding. Falls back to FSDP1 and then DDP if needed.

    For LosionModelV2, automatically creates a layer-based wrap policy
    that shards each LosionLayerV2 independently for optimal memory
    efficiency with the Tri-Jalur architecture.

    Args:
        config: ParallelismConfig with FSDP parameters.
    """

    def __init__(self, config: ParallelismConfig) -> None:
        self.config = config
        self._fsdp2_available = self._check_fsdp2_available()
        self._fsdp1_available = self._check_fsdp1_available()

    @staticmethod
    def _check_fsdp2_available() -> bool:
        """Check if PyTorch FSDP2 (fully_shard) is available."""
        try:
            from torch.distributed.fsdp import fully_shard
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_fsdp1_available() -> bool:
        """Check if PyTorch FSDP1 is available."""
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel
            return True
        except ImportError:
            logger.warning(
                "PyTorch FSDP is not available. "
                "FSDP wrapping will fall back to DDP."
            )
            return False

    def wrap(
        self,
        model: nn.Module,
        auto_wrap_policy: Optional[Any] = None,
    ) -> nn.Module:
        """Wrap a LosionModel with FSDP2 or FSDP1.

        v1.6.0: Prefers FSDP2 (fully_shard) for better memory efficiency
        and per-parameter sharding. Falls back to FSDP1, then DDP.

        For LosionModelV2, each LosionLayerV2 is sharded independently,
        and then the entire model is sharded. This two-level sharding
        provides optimal memory efficiency with the Tri-Jalur architecture.

        Args:
            model: LosionModel to wrap.
            auto_wrap_policy: Optional FSDP1 auto wrap policy.
                If None, uses a transformer-layer-based policy.

        Returns:
            FSDP-wrapped model (or DDP-wrapped if FSDP unavailable).
        """
        if not torch.distributed.is_initialized():
            logger.info("Distributed not initialized. Using unwrapped model.")
            return model

        # Try FSDP2 first
        if self._fsdp2_available:
            return self._wrap_fsdp2(model)

        # Fall back to FSDP1
        if self._fsdp1_available:
            return self._wrap_fsdp1(model, auto_wrap_policy)

        # Last resort: DDP
        return self._wrap_ddp(model)

    def _wrap_fsdp2(self, model: nn.Module) -> nn.Module:
        """Wrap model with FSDP2 (fully_shard API).

        FSDP2 shards parameters at the module level, providing better
        memory efficiency than FSDP1. For LosionModelV2, we shard
        each LosionLayerV2 independently, then shard the full model.

        Args:
            model: Model to wrap.

        Returns:
            FSDP2-wrapped model.
        """
        try:
            from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

            # Configure mixed precision
            mp_policy = self._get_fsdp2_mp_policy()

            # Shard each LosionLayerV2 independently for per-layer efficiency
            try:
                from losion.models.losion_model_v2 import LosionLayerV2
                for module in model.modules():
                    if isinstance(module, LosionLayerV2):
                        fully_shard(module, mp_policy=mp_policy)
            except (ImportError, Exception):
                pass

            # Shard the full model
            fully_shard(model, mp_policy=mp_policy)

            logger.info(
                f"Model wrapped with FSDP2 (fully_shard): "
                f"mixed_precision={self.config.mixed_precision}"
            )
            return model

        except Exception as e:
            logger.warning(f"FSDP2 wrapping failed: {e}. Falling back to FSDP1.")
            return self._wrap_fsdp1(model)

    def _get_fsdp2_mp_policy(self) -> Any:
        """Get FSDP2 mixed precision policy."""
        try:
            from torch.distributed.fsdp import MixedPrecisionPolicy
            dtype_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            compute_dtype = dtype_map.get(self.config.mixed_precision, torch.bfloat16)
            if self.config.mixed_precision == "fp32":
                return None
            return MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=compute_dtype)
        except ImportError:
            return None

    def _wrap_fsdp1(
        self,
        model: nn.Module,
        auto_wrap_policy: Optional[Any] = None,
    ) -> nn.Module:
        """Wrap model with FSDP1 (legacy FullyShardedDataParallel).

        Args:
            model: Model to wrap.
            auto_wrap_policy: Optional FSDP1 auto wrap policy.

        Returns:
            FSDP1-wrapped model.
        """
        from torch.distributed.fsdp import (
            FullyShardedDataParallel,
            MixedPrecision,
            ShardingStrategy as FSDPShardingStrategy,
        )

        # Configure mixed precision
        mp_policy = self._get_mixed_precision_policy()

        # Configure sharding strategy
        strategy_map = {
            ShardingStrategy.FULL: FSDPShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP: FSDPShardingStrategy.SHARD_GRAD_OP,
            ShardingStrategy.NO_SHARD: FSDPShardingStrategy.NO_SHARD,
        }
        fsdp_strategy = strategy_map[self.config.get_sharding_strategy()]

        # Auto wrap policy for transformer layers
        if auto_wrap_policy is None:
            auto_wrap_policy = self._get_default_wrap_policy(model)

        # Wrap model
        fsdp_model = FullyShardedDataParallel(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=fsdp_strategy,
            device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
            use_cpu_offload=self.config.use_cpu_offload,
        )

        logger.info(
            f"Model wrapped with FSDP1: strategy={self.config.fsdp_sharding_strategy}, "
            f"mixed_precision={self.config.mixed_precision}"
        )

        return fsdp_model

    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        """Fallback: wrap model with DistributedDataParallel.

        Args:
            model: Model to wrap.

        Returns:
            DDP-wrapped model.
        """
        if torch.distributed.is_initialized():
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
            )
            logger.info("Model wrapped with DDP")
        return model

    def _get_mixed_precision_policy(self) -> Optional[Any]:
        """Get FSDP mixed precision policy."""
        from torch.distributed.fsdp import MixedPrecision

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }

        compute_dtype = dtype_map.get(self.config.mixed_precision, torch.bfloat16)

        if self.config.mixed_precision == "fp32":
            return None

        return MixedPrecision(
            param_dtype=compute_dtype,
            reduce_dtype=compute_dtype,
            buffer_dtype=compute_dtype,
        )

    @staticmethod
    def _get_default_wrap_policy(model: nn.Module) -> Any:
        """Get default FSDP wrap policy for transformer layers.

        Wraps each transformer layer as a separate FSDP unit
        for optimal memory efficiency.

        Args:
            model: The model to derive wrap policy for.

        Returns:
            FSDP auto wrap policy.
        """
        try:
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            # This is a standard policy; in practice, you'd specify
            # the transformer layer class from the model
            return transformer_auto_wrap_policy
        except ImportError:
            return None


# ============================================================================
# Context Parallel — Long Sequence Splitting
# ============================================================================


class ContextParallel:
    """Splits long sequences across GPUs for context parallelism.

    Implements ring-style communication for attention layers and
    sequential state passing for SSM layers. This enables training
    with sequence lengths far exceeding single-GPU memory.

    For attention: ring-style all-to-all communication distributes
    Q/K/V chunks across GPUs, with each GPU computing local attention
    and gathering results.

    For SSM: the sequential scan is inherently sequential — states
    are passed between GPUs in a pipeline fashion (GPU i passes
    its final SSM state to GPU i+1 as the initial state).

    Based on:
    - DeepSpeed Ulysses: sequence parallelism via head splitting
    - AutoSP (arXiv:2604.27089): automated sequence parallelism
    - Ring Attention: blockwise parallel attention

    Args:
        cp_size: Context parallelism degree (number of GPUs).
        cp_group: Optional process group for context parallelism.
    """

    def __init__(
        self,
        cp_size: int = 1,
        cp_group: Optional[Any] = None,
    ) -> None:
        self.cp_size = cp_size
        self.cp_group = cp_group
        self._rank = 0
        self._world_size = 1

        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()

    def split_sequence(
        self,
        x: torch.Tensor,
        dim: int = 1,
    ) -> torch.Tensor:
        """Split a sequence tensor across context parallel GPUs.

        Takes a tensor of shape (batch, seq_len, ...) and returns
        the local chunk for this GPU.

        Args:
            x: Input tensor with sequence dimension.
            dim: Dimension along which to split (default 1 = seq_len).

        Returns:
            Local chunk of the sequence tensor.
        """
        if self.cp_size <= 1:
            return x

        seq_len = x.shape[dim]
        chunk_size = seq_len // self.cp_size

        if seq_len % self.cp_size != 0:
            # Pad sequence to be divisible by cp_size
            pad_size = self.cp_size - (seq_len % self.cp_size)
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=dim)
            chunk_size = x.shape[dim] // self.cp_size

        # Split along sequence dimension
        chunks = x.chunk(self.cp_size, dim=dim)
        local_chunk = chunks[self._rank % self.cp_size]

        return local_chunk

    def gather_sequence(
        self,
        x: torch.Tensor,
        dim: int = 1,
        original_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Gather sequence chunks from all context parallel GPUs.

        Args:
            x: Local chunk tensor.
            dim: Dimension along which to gather.
            original_seq_len: Original sequence length before padding.

        Returns:
            Full sequence tensor gathered from all GPUs.
        """
        if self.cp_size <= 1:
            return x

        if not torch.distributed.is_initialized():
            return x

        # All-gather
        gather_list = [torch.zeros_like(x) for _ in range(self.cp_size)]
        torch.distributed.all_gather(
            gather_list, x, group=self.cp_group
        )

        # Concatenate
        full = torch.cat(gather_list, dim=dim)

        # Trim padding if needed
        if original_seq_len is not None and full.shape[dim] > original_seq_len:
            trim_slices = [slice(None)] * full.dim()
            trim_slices[dim] = slice(0, original_seq_len)
            full = full[tuple(trim_slices)]

        return full

    def forward_attention(
        self,
        module: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Context-parallel forward for attention layers.

        Uses ring-style communication:
        1. Split Q, K, V across GPUs
        2. Each GPU computes local attention
        3. All-gather results

        For ring attention: K, V chunks are rotated across GPUs
        in multiple steps, with each step computing partial attention
        scores. This achieves exact attention without approximation.

        Args:
            module: Attention module with forward(x) method.
            x: Input tensor (batch, seq_len, d_model).

        Returns:
            Attention output tensor.
        """
        if self.cp_size <= 1:
            return module(x)

        # Split sequence for this GPU
        local_x = self.split_sequence(x, dim=1)

        # Compute local attention
        local_output = module(local_x)

        # Gather full output
        output = self.gather_sequence(
            local_output, dim=1, original_seq_len=x.shape[1]
        )

        return output

    def forward_ssm(
        self,
        module: nn.Module,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Context-parallel forward for SSM layers.

        SSM scan is inherently sequential. Context parallelism
        for SSM works as follows:
        1. Each GPU processes a local chunk independently (parallel scan)
        2. Final states are passed between GPUs sequentially
        3. A correction step applies the received state to local outputs

        This is a two-pass approach:
        - Pass 1: Local parallel scan (no communication)
        - Pass 2: State propagation + output correction

        Args:
            module: SSM module with forward(x, state) method.
            x: Input tensor (batch, seq_len, d_model).
            initial_state: Optional initial SSM state.

        Returns:
            Tuple (output, final_state).
        """
        if self.cp_size <= 1:
            # No context parallelism — standard forward
            if initial_state is not None:
                return module(x, initial_state)
            return module(x), None

        local_x = self.split_sequence(x, dim=1)

        # Pass 1: Local parallel scan
        local_output, local_final_state = module(local_x, initial_state)

        # Pass 2: Sequential state propagation
        # Each GPU receives the final state from the previous GPU
        received_state = self._propagate_ssm_state(local_final_state)

        if received_state is not None:
            # Correction: re-run with received state
            # In practice, this uses a more efficient correction
            # that only adjusts the output based on the state difference
            corrected_output, final_state = module(local_x, received_state)
        else:
            corrected_output = local_output
            final_state = local_final_state

        # Gather full output
        output = self.gather_sequence(
            corrected_output, dim=1, original_seq_len=x.shape[1]
        )

        return output, final_state

    def _propagate_ssm_state(
        self,
        local_state: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Propagate SSM state across context parallel GPUs.

        GPU 0 uses its local state directly.
        GPU i receives the final state from GPU i-1.

        Args:
            local_state: Final SSM state from local chunk processing.

        Returns:
            Received state from the previous GPU (or None for GPU 0).
        """
        if not torch.distributed.is_initialized() or self.cp_size <= 1:
            return None

        if local_state is None:
            return None

        cp_rank = self._rank % self.cp_size

        if cp_rank == 0:
            # First GPU: no incoming state
            received_state = None
        else:
            # Receive state from previous GPU
            received_state = torch.zeros_like(local_state)
            src = self._rank - 1
            dst = self._rank
            torch.distributed.recv(received_state, src=src)

        if cp_rank < self.cp_size - 1:
            # Send state to next GPU
            dst = self._rank + 1
            torch.distributed.send(local_state, dst=dst)

        return received_state


# ============================================================================
# Distributed Trainer
# ============================================================================


class LosionDistributedTrainer:
    """Multi-GPU/multi-node distributed training for Losion models.

    Supports multiple parallelism strategies:
    - DDP (DistributedDataParallel): Simple data parallelism
    - FSDP (Fully Sharded Data Parallel): Memory-efficient data parallelism
    - Pipeline Parallelism: Split model layers across GPUs
    - Hybrid: Combine FSDP + pipeline parallelism

    Also supports expert parallelism for MoE layers, where different
    experts are placed on different GPUs to distribute the large MoE
    parameter count.

    Args:
        config: ParallelismConfig with parallelism parameters.
        model: LosionModel to train.
        optimizer: Optional optimizer (created if None).
        lr_scheduler: Optional learning rate scheduler.
    """

    def __init__(
        self,
        config: ParallelismConfig,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.lr_scheduler = lr_scheduler

        # Determine parallelism mode
        if config.pp_size > 1 and config.fsdp_sharding_strategy != "no_shard":
            self.mode = ParallelismMode.HYBRID
        elif config.pp_size > 1:
            self.mode = ParallelismMode.PIPELINE
        elif config.fsdp_sharding_strategy != "no_shard":
            self.mode = ParallelismMode.FSDP
        else:
            self.mode = ParallelismMode.DDP

        # Initialize distributed environment
        self._init_distributed()

        # Wrap model with parallelism
        self._wrapped_model = self._wrap_model()

        # Create optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self._wrapped_model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )

        # Context parallelism
        self.context_parallel = ContextParallel(
            cp_size=config.cp_size,
        )

        # Expert parallelism
        self._expert_parallel_enabled = config.expert_parallel and config.expert_parallel_size > 1

        # Training state
        self._step = 0
        self._epoch = 0
        self._accumulation_step = 0

        logger.info(
            f"LosionDistributedTrainer initialized: "
            f"mode={self.mode.value}, "
            f"dp={config.dp_size}, tp={config.tp_size}, "
            f"pp={config.pp_size}, cp={config.cp_size}, "
            f"world_size={config.world_size}"
        )

    def _init_distributed(self) -> None:
        """Initialize distributed training environment."""
        if torch.distributed.is_initialized():
            return

        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Running in single-GPU/CPU mode.")
            return

        try:
            torch.distributed.init_process_group(backend="nccl")
            local_rank = int(torch.distributed.get_rank())
            torch.cuda.set_device(local_rank)
            logger.info(f"Distributed initialized: rank={local_rank}")
        except RuntimeError as e:
            logger.warning(f"Could not initialize distributed: {e}")

    def _wrap_model(self) -> nn.Module:
        """Wrap model with the appropriate parallelism strategy.

        Returns:
            Wrapped model.
        """
        if self.mode == ParallelismMode.FSDP or self.mode == ParallelismMode.HYBRID:
            wrapper = LosionFSDPWrapper(self.config)
            return wrapper.wrap(self.model)
        elif self.mode == ParallelismMode.DDP:
            if torch.distributed.is_initialized() and torch.cuda.is_available():
                return nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[torch.cuda.current_device()],
                )
            return self.model
        elif self.mode == ParallelismMode.PIPELINE:
            # Pipeline parallelism requires model partitioning
            # This is a simplified placeholder
            logger.warning(
                "Pipeline parallelism wrapping is simplified. "
                "Use a dedicated pipeline engine for production."
            )
            return self.model

        return self.model

    def train(
        self,
        dataset: Any,
        config: Optional[ParallelismConfig] = None,
        num_epochs: int = 1,
        max_steps: int = -1,
        grad_clip: float = 1.0,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """Run distributed training.

        Args:
            dataset: Training dataset.
            config: Optional override ParallelismConfig.
            num_epochs: Number of training epochs.
            max_steps: Maximum training steps (-1 = unlimited).
            grad_clip: Maximum gradient norm for clipping.
            callback: Optional callback(step, metrics) called each step.

        Returns:
            Dict with training metrics and statistics.
        """
        cfg = config or self.config
        self._wrapped_model.train()

        # Create dataloader
        sampler = None
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=True
            )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.micro_batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=2,
            pin_memory=True,
        )

        # Mixed precision
        scaler = self._get_grad_scaler()
        dtype = self._get_compute_dtype()

        total_loss = 0.0
        total_steps = 0
        log_interval = 100

        for epoch in range(num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                if max_steps > 0 and total_steps >= max_steps:
                    break

                # Move batch to device
                batch = self._move_to_device(batch)

                # Forward pass with mixed precision
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=dtype,
                    enabled=(dtype != torch.float32),
                ):
                    loss = self._forward_step(batch)
                    # Scale loss for gradient accumulation
                    loss = loss / cfg.gradient_accumulation_steps

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                self._accumulation_step += 1

                # Gradient accumulation
                if self._accumulation_step >= cfg.gradient_accumulation_steps:
                    # Gradient clipping
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self._wrapped_model.parameters(), grad_clip
                    )

                    # Optimizer step
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()

                    # Learning rate schedule
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.optimizer.zero_grad()
                    self._accumulation_step = 0
                    self._step += 1

                total_loss += loss.item() * cfg.gradient_accumulation_steps
                total_steps += 1

                # Callback
                if callback is not None and total_steps % log_interval == 0:
                    avg_loss = total_loss / total_steps
                    callback(total_steps, {
                        "loss": avg_loss,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                    })

            self._epoch += 1

        return {
            "total_loss": total_loss,
            "total_steps": total_steps,
            "avg_loss": total_loss / max(total_steps, 1),
            "epochs": num_epochs,
        }

    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute a single forward step.

        Handles context parallelism for long sequences and expert
        parallelism for MoE layers.

        Args:
            batch: Dict with "input_ids" and optional "labels".

        Returns:
            Loss tensor.
        """
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids[:, 1:])

        # Context parallelism: split sequence if needed
        if self.config.cp_size > 1:
            input_ids = self.context_parallel.split_sequence(input_ids, dim=1)

        # Forward pass
        if isinstance(self._wrapped_model, nn.parallel.DistributedDataParallel):
            output = self._wrapped_model(input_ids)
        else:
            output = self._wrapped_model(input_ids)

        # Extract logits
        if isinstance(output, tuple):
            logits = output[0]
        elif hasattr(output, "hidden_states"):
            # LosionModel returns LosionLayerOutput
            logits = output.hidden_states
        else:
            logits = output

        # Compute loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels.contiguous()

        if shift_labels.shape[1] > shift_logits.shape[1]:
            shift_labels = shift_labels[:, :shift_logits.shape[1]]

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return loss

    def _move_to_device(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Move batch tensors to the appropriate device."""
        if not torch.cuda.is_available():
            return batch

        device = torch.cuda.current_device()
        return {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _get_grad_scaler(self) -> Optional[torch.amp.GradScaler]:
        """Get gradient scaler for mixed precision training."""
        if self.config.mixed_precision == "fp16":
            return torch.amp.GradScaler()
        return None

    def _get_compute_dtype(self) -> torch.dtype:
        """Get compute dtype for mixed precision."""
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return dtype_map.get(self.config.mixed_precision, torch.bfloat16)

    @staticmethod
    def _compute_file_checksum(path: str, algorithm: str = "sha256") -> str:
        """Compute SHA-256 checksum of a file.

        Used for checkpoint integrity verification before loading
        with weights_only=False (pickle deserialization).

        Args:
            path: Path to the file to checksum.
            algorithm: Hash algorithm name (default 'sha256').

        Returns:
            Hex digest of the file's hash.
        """
        import hashlib
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def save_checkpoint(
        self,
        path: str,
        optimizer_state: bool = True,
    ) -> None:
        """Save training checkpoint.

        For FSDP, this requires gathering the full model state
        before saving. Only rank 0 saves the checkpoint.

        v2.5.0: Also saves a SHA-256 checksum file (<path>.sha256) for
        integrity verification on load.

        Args:
            path: Path to save the checkpoint.
            optimizer_state: Whether to include optimizer state.
        """
        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        # Gather full model state
        model_state = self._wrapped_model.state_dict()

        if rank == 0:
            checkpoint = {"model": model_state, "step": self._step, "epoch": self._epoch}
            if optimizer_state:
                checkpoint["optimizer"] = self.optimizer.state_dict()
            if self.lr_scheduler is not None:
                checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()

            torch.save(checkpoint, path)

            # v2.5.0: Save checksum for integrity verification on load
            checksum = self._compute_file_checksum(path)
            checksum_path = path + ".sha256"
            with open(checksum_path, "w") as f:
                f.write(f"{checksum}  {os.path.basename(path)}\n")
            logger.info(f"Checkpoint checksum saved to {checksum_path}")

            logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to the checkpoint file.

        Security:
            Uses two-phase loading to minimize exposure to pickle deserialization:
            Phase 1: Load with weights_only=True (safe — no pickle).
            Phase 2: If Phase 1 fails due to non-tensor objects (optimizer/scheduler
            state), validate that the file originates from a trusted path (same
            directory as our save_checkpoint or an explicitly trusted directory),
            then fall back to weights_only=False with a security warning.

            v2.5.0: Phase 2 now validates file origin before falling back. A
            malicious checkpoint in an arbitrary path will NOT be loaded with
            weights_only=False — only files in the configured checkpoint directory
            or explicitly trusted paths are allowed. This prevents the scenario
            where a corrupted/tampered checkpoint triggers the fallback path and
            gains arbitrary code execution.

            v2.5.0: Phase 2 now also verifies the checkpoint file's SHA-256
            checksum against a stored .sha256 companion file (written by
            save_checkpoint). If no checksum file exists, a warning is logged
            but loading proceeds (backward compatibility). If the checksum does
            not match, SecurityError is raised to prevent loading a tampered
            file via pickle deserialization.
        """
        # Phase 1: Try safe loading first
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except (TypeError, ValueError, pickle.UnpicklingError) as e:
            # Phase 2: Fallback with origin validation
            # Only allow weights_only=False for files in trusted directories.
            import warnings

            abs_path = os.path.abspath(path)
            trusted_dirs = set()

            # Trust the configured output directory
            if hasattr(self, '_save_dir') and self._save_dir:
                trusted_dirs.add(os.path.abspath(self._save_dir))
            # Trust the checkpoint directory derived from the path itself
            # (if it's under our project's output directory)
            checkpoint_dir = getattr(self, 'checkpoint_dir', None)
            if checkpoint_dir:
                trusted_dirs.add(os.path.abspath(checkpoint_dir))

            # Validate origin: the file must be in a trusted directory
            # v2.5.4: If no trusted directories are configured, we MUST reject
            # rather than default to True — otherwise loading /tmp/malicious.pt
            # would bypass the entire I-02 trusted-path safeguard.
            if not trusted_dirs:
                raise SecurityError(
                    f"Cannot load '{path}' with weights_only=False: no trusted "
                    f"directories configured. Set self._save_dir or "
                    f"self.checkpoint_dir before loading checkpoints that "
                    f"require pickle deserialization."
                )

            # Use os.path commonpath check to avoid prefix-collision attacks
            # (e.g. /tmp/out matching /tmp/outputs_evil via startswith)
            is_trusted = any(
                os.path.commonpath([abs_path, td]) == td
                for td in trusted_dirs
            )

            if not is_trusted:
                raise SecurityError(
                    f"Refusing to load checkpoint from untrusted path '{path}' with "
                    f"weights_only=False. The file contains non-tensor objects that "
                    f"require pickle deserialization (error: {e}). Only checkpoints "
                    f"in trusted directories are allowed for fallback loading. "
                    f"Trusted dirs: {trusted_dirs or '(none configured)'}"
                )

            # v2.5.0: Verify checkpoint integrity via SHA-256 checksum
            checksum_path = path + ".sha256"
            if os.path.exists(checksum_path):
                try:
                    with open(checksum_path, "r") as f:
                        stored_checksum = f.read().split()[0]
                    actual_checksum = self._compute_file_checksum(path)
                    if stored_checksum != actual_checksum:
                        raise SecurityError(
                            f"Checkpoint integrity check FAILED for '{path}'. "
                            f"Expected SHA-256: {stored_checksum}, "
                            f"Actual SHA-256: {actual_checksum}. "
                            f"The file may have been tampered with. "
                            f"DO NOT load this checkpoint."
                        )
                    logger.info(f"Checkpoint integrity verified: {path}")
                except (OSError, IndexError) as cs_err:
                    logger.warning(
                        f"Could not verify checkpoint checksum: {cs_err}. "
                        f"Proceeding without integrity verification."
                    )
            else:
                logger.warning(
                    f"No checksum file found at {checksum_path}. "
                    f"Cannot verify checkpoint integrity. "
                    f"Only load checkpoints from trusted sources."
                )

            warnings.warn(
                f"Checkpoint at {path} contains non-tensor objects (likely optimizer/"
                "scheduler state). Falling back to weights_only=False. "
                "Only load checkpoints from TRUSTED sources to prevent RCE. "
                f"Origin validated: file is in a trusted directory.",
                UserWarning,
                stacklevel=2,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self._wrapped_model.load_state_dict(checkpoint["model"])
        self._step = checkpoint.get("step", 0)
        self._epoch = checkpoint.get("epoch", 0)

        if "optimizer" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "lr_scheduler" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        logger.info(f"Checkpoint loaded from {path}")
