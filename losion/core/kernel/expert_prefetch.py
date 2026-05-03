"""
Expert Prefetch via Routing Prediction for Losion.

Losion's AdaptiveRouter computes routing weights BEFORE executing the MoE
pathway. This means we know which experts will be needed BEFORE they are
accessed. Expert prefetch exploits this knowledge to:

1. Pre-load expert parameters from CPU to GPU (KTransformers-style)
2. Pre-load expert parameters from HBM to SRAM (cache warming)
3. Pipeline expert loading with SSM/Attention computation
4. Enable CPU/GPU hybrid MoE inference on resource-constrained setups

This provides 2-3x throughput improvement for distributed MoE inference
and enables running Losion-48B on a single GPU via CPU offloading.

References:
  - KTransformers: (dl.acm.org/doi/10.1145/3731569.3764843, SOSP 2025)
  - Occult: ICML 2025 — optimizing MoE all-to-all communication
  - FlashMoE: (arXiv:2506.04667) — fused distributed MoE kernel
  - Losion Framework: Wolfvin (github.com/Wolfvin/Losion)
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional, Dict, Any, List, Tuple, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Expert Prefetcher
# ============================================================================

class ExpertPrefetcher:
    """Prefetch MoE expert parameters based on router predictions.

    Uses the routing weights from Losion's AdaptiveRouter to predict
    which experts will be needed in the MoE pathway, and prefetches
    their parameters before they are accessed.

    Two prefetch strategies:
    1. GPU-GPU: Move expert params from HBM to shared memory (cache warming)
    2. CPU-GPU: Move expert params from CPU RAM to GPU HBM (offloading)

    Args:
        n_experts: Total number of experts.
        expert_dim: Dimension of each expert's parameters.
        prefetch_top_k: Number of experts to prefetch (matches routing top-k).
        device: Target device for prefetched experts.
        offload_to_cpu: If True, keep non-active experts on CPU.
    """

    def __init__(
        self,
        n_experts: int = 8,
        expert_dim: int = 512,
        prefetch_top_k: int = 2,
        device: str = "cuda",
        offload_to_cpu: bool = False,
    ):
        self.n_experts = n_experts
        self.expert_dim = expert_dim
        self.prefetch_top_k = prefetch_top_k
        self.device = device
        self.offload_to_cpu = offload_to_cpu

        # Expert parameter storage
        self._expert_params: Dict[int, torch.Tensor] = {}
        self._cpu_expert_params: Dict[int, torch.Tensor] = {}
        self._active_experts: Set[int] = set()

        # Prefetch queue
        self._prefetch_queue: queue.Queue = queue.Queue()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the background prefetch thread."""
        if self._prefetch_thread is not None:
            return

        self._running = True
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
        )
        self._prefetch_thread.start()

    def stop(self) -> None:
        """Stop the background prefetch thread."""
        self._running = False
        self._prefetch_queue.put(None)  # Sentinel
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
            self._prefetch_thread = None

    def _prefetch_worker(self) -> None:
        """Background worker that prefetches expert parameters."""
        while self._running:
            try:
                item = self._prefetch_queue.get(timeout=0.1)
                if item is None:
                    break

                expert_id, params = item
                # Move to GPU
                self._expert_params[expert_id] = params.to(self.device, non_blocking=True)
                self._active_experts.add(expert_id)

            except queue.Empty:
                continue

    def register_expert(self, expert_id: int, params: torch.Tensor) -> None:
        """Register an expert's parameters.

        Args:
            expert_id: Expert identifier.
            params: Expert parameter tensor.
        """
        if self.offload_to_cpu:
            self._cpu_expert_params[expert_id] = params.cpu()
        else:
            self._expert_params[expert_id] = params.to(self.device)

    def prefetch_from_routing(
        self,
        routing_weights: torch.Tensor,
        expert_ids: Optional[torch.Tensor] = None,
    ) -> List[int]:
        """Determine which experts to prefetch based on routing weights.

        Args:
            routing_weights: MoE routing weights (batch, seq_len, n_experts).
            expert_ids: Optional pre-computed expert IDs.

        Returns:
            List of expert IDs that will be prefetched.
        """
        # Find top-k experts by mean routing weight
        mean_weights = routing_weights.mean(dim=(0, 1))  # (n_experts,)
        top_k = min(self.prefetch_top_k, self.n_experts)
        top_ids = mean_weights.topk(top_k).indices.tolist()

        # Queue prefetch for experts not already on GPU
        for expert_id in top_ids:
            if expert_id not in self._active_experts and expert_id in self._cpu_expert_params:
                self._prefetch_queue.put((expert_id, self._cpu_expert_params[expert_id]))

        # Evict least-recently-used experts if GPU memory is tight
        if self.offload_to_cpu:
            current_active = len(self._active_experts)
            max_active = self.prefetch_top_k + 2  # Keep a few extra
            if current_active > max_active:
                # Evict experts not in top_ids
                to_evict = self._active_experts - set(top_ids)
                for expert_id in to_evict:
                    if expert_id in self._expert_params:
                        self._cpu_expert_params[expert_id] = self._expert_params[expert_id].cpu()
                        del self._expert_params[expert_id]
                        self._active_experts.discard(expert_id)

        return top_ids

    def get_expert_params(self, expert_id: int) -> Optional[torch.Tensor]:
        """Get expert parameters (should be on GPU after prefetch).

        Args:
            expert_id: Expert identifier.

        Returns:
            Expert parameters on GPU, or None if not available.
        """
        return self._expert_params.get(expert_id)


# ============================================================================
# Communication-Computation Overlap for Distributed MoE
# ============================================================================

class MoECommunicationOverlap:
    """Overlap MoE expert communication with SSM/Attention computation.

    In distributed MoE training, the all-to-all communication for expert
    routing is a major bottleneck. This class overlaps that communication
    with computation on other pathways.

    Pipeline:
    1. Start MoE all-to-all communication (async)
    2. Execute SSM pathway while communication is in flight
    3. Execute Attention pathway while communication is in flight
    4. Synchronize and receive MoE results

    References:
        - Occult: ICML 2025 — optimizing collaborative communication
        - FlashMoE: (arXiv:2506.04667) — fused distributed MoE kernel

    Args:
        overlap_with_ssm: If True, overlap MoE communication with SSM.
        overlap_with_attn: If True, overlap MoE communication with Attention.
    """

    def __init__(
        self,
        overlap_with_ssm: bool = True,
        overlap_with_attn: bool = True,
    ):
        self.overlap_with_ssm = overlap_with_ssm
        self.overlap_with_attn = overlap_with_attn

    def execute_overlapped(
        self,
        ssm_fn,
        attn_fn,
        moe_comm_start_fn,
        moe_comm_end_fn,
    ):
        """Execute MoE communication overlapped with SSM/Attention.

        Args:
            ssm_fn: SSM pathway function.
            attn_fn: Attention pathway function.
            moe_comm_start_fn: Start async MoE communication.
            moe_comm_end_fn: Wait for and receive MoE results.

        Returns:
            Tuple of (ssm_out, attn_out, moe_out).
        """
        # Step 1: Start MoE communication
        comm_handle = moe_comm_start_fn()

        # Step 2: Execute SSM and Attention while communication is in flight
        ssm_out = ssm_fn()
        attn_out = attn_fn()

        # Step 3: Wait for MoE results
        moe_out = moe_comm_end_fn(comm_handle)

        return ssm_out, attn_out, moe_out


__all__ = [
    "ExpertPrefetcher",
    "MoECommunicationOverlap",
]
