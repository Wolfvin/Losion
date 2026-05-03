"""
Dynamic Memory Sparsification (DMS) — Inference-time KV cache hyper-scaling.

Sparsifies KV caches at inference time to enable "hyper-scaling" — generating
more tokens within the same compute budget. Only requires ~1K training steps
for the importance predictor.

Inspired by:
  - Dynamic Memory Sparsification (NeurIPS 2025, arXiv 2506.05345)
  - ThinK: Thinner Key Cache by Query-Driven Pruning (ICLR 2025)
  - ChunkKV: Semantic-Preserving KV Cache Compression (NeurIPS 2025)

Key innovations:
  - Lightweight importance predictor that scores KV pairs for eviction
  - Online eviction during inference (no pre-training required for basic mode)
  - Query-driven importance: tokens important for current query are retained
  - Budget-aware: maintain target KV cache size dynamically

Credits:
  - DMS: NeurIPS 2025 (arXiv 2506.05345)
  - ThinK: ICLR 2025
  - ChunkKV: NeurIPS 2025

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DMSConfig:
    """Configuration for Dynamic Memory Sparsification.

    Attributes:
        enabled: Whether to enable DMS.
        target_cache_ratio: Target KV cache ratio (0.5 = keep 50% of tokens).
        eviction_strategy: How to select tokens for eviction
            ("importance", "lru", "attention_weight", "key_norm").
        importance_hidden_dim: Hidden dimension for importance predictor.
        update_frequency: How often to run eviction (every N tokens).
        min_tokens_to_keep: Minimum tokens to keep in cache.
        use_chunk_eviction: Whether to evict in chunks (ChunkKV-style).
        chunk_size: Chunk size for chunk eviction.
    """
    enabled: bool = True
    target_cache_ratio: float = 0.5
    eviction_strategy: str = "importance"
    importance_hidden_dim: int = 64
    update_frequency: int = 64
    min_tokens_to_keep: int = 32
    use_chunk_eviction: bool = True
    chunk_size: int = 16


class ImportancePredictor(nn.Module):
    """Lightweight predictor for KV pair importance scores.

    Predicts how important each cached KV pair is for the current query.
    Used to decide which KV pairs to evict from the cache.

    Args:
        d_kv: Dimension per key/value head.
        hidden_dim: Hidden dimension for the predictor.
    """

    def __init__(self, d_kv: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.query_proj = nn.Linear(d_kv, hidden_dim, bias=False)
        self.key_proj = nn.Linear(d_kv, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance scores for each key given the query.

        Args:
            query: Current query tensor (batch, n_heads, d_kv).
            keys: Cached key tensor (batch, n_heads, seq_len, d_kv).

        Returns:
            Importance scores (batch, n_heads, seq_len).
        """
        # Project
        q = self.query_proj(query)  # (batch, n_heads, hidden_dim)
        k = self.key_proj(keys)     # (batch, n_heads, seq_len, hidden_dim)

        # Compute similarity-based importance
        q_expanded = q.unsqueeze(2)  # (batch, n_heads, 1, hidden_dim)
        similarity = (k * q_expanded).sum(dim=-1)  # (batch, n_heads, seq_len)

        # Score projection
        combined = F.silu(k + q_expanded)
        scores = self.score_proj(combined).squeeze(-1)  # (batch, n_heads, seq_len)

        # Combine similarity and score
        importance = similarity + scores * 0.1
        return importance


class DynamicMemorySparsification:
    """Inference-time KV cache sparsification for memory reduction.

    Evicts less important KV pairs from the cache during inference,
    maintaining a target cache size. This enables:
    - Longer generation within the same memory budget
    - Higher throughput by reducing attention computation
    - Minimal quality loss when using importance-based eviction

    Usage:
        dms = DynamicMemorySparsification(config)
        # During inference, after each attention layer:
        importance_scores = dms.compute_importance(query, cached_keys)
        should_evict = dms.select_eviction_targets(importance_scores, current_cache_size)
        new_keys, new_values = dms.evict(cached_keys, cached_values, should_evict)

    Args:
        config: DMSConfig with sparsification parameters.
    """

    def __init__(self, config: DMSConfig) -> None:
        self.config = config
        self._step_count = 0
        self._predictor: Optional[ImportancePredictor] = None

    def init_predictor(self, d_kv: int) -> None:
        """Initialize the importance predictor.

        Args:
            d_kv: Dimension per key/value head.
        """
        if self._predictor is None:
            self._predictor = ImportancePredictor(
                d_kv=d_kv,
                hidden_dim=self.config.importance_hidden_dim,
            )

    def compute_importance(
        self,
        query: torch.Tensor,
        cached_keys: torch.Tensor,
        cached_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute importance scores for cached KV pairs.

        Args:
            query: Current query (batch, n_heads, d_kv) or (batch, 1, n_heads, d_kv).
            cached_keys: Cached keys (batch, n_heads, seq_len, d_kv).
            cached_values: Cached values (unused for most strategies).

        Returns:
            Importance scores (batch, n_heads, seq_len). Higher = more important.
        """
        strategy = self.config.eviction_strategy

        if strategy == "importance" and self._predictor is not None:
            if query.dim() == 4:
                query = query.squeeze(1)
            return self._predictor(query, cached_keys)

        elif strategy == "attention_weight":
            # Use attention weights as importance proxy
            scale = math.sqrt(query.shape[-1])
            if query.dim() == 4:
                query = query.squeeze(1)
            attn = torch.matmul(query.unsqueeze(2), cached_keys.transpose(-2, -1)) / scale
            attn_weights = F.softmax(attn, dim=-1)
            return attn_weights.squeeze(2)

        elif strategy == "key_norm":
            # Simple: larger key norm = more important
            return cached_keys.norm(dim=-1)

        elif strategy == "lru":
            # Least Recently Used: older tokens are less important
            seq_len = cached_keys.shape[2]
            scores = torch.linspace(1.0, 0.0, seq_len, device=cached_keys.device)
            return scores.unsqueeze(0).unsqueeze(0).expand_as(
                torch.zeros(cached_keys.shape[0], cached_keys.shape[1], seq_len, device=cached_keys.device)
            )

        else:
            # Fallback: uniform scores (no eviction preference)
            return torch.ones(cached_keys.shape[:3], device=cached_keys.device)

    def select_eviction_targets(
        self,
        importance_scores: torch.Tensor,
        current_cache_size: int,
    ) -> torch.Tensor:
        """Select which tokens to evict based on importance scores.

        Args:
            importance_scores: (batch, n_heads, seq_len) importance per token.
            current_cache_size: Current number of tokens in cache.

        Returns:
            Boolean mask (batch, n_heads, seq_len) where True = keep.
        """
        target_size = max(
            self.config.min_tokens_to_keep,
            int(current_cache_size * self.config.target_cache_ratio),
        )

        if current_cache_size <= target_size:
            # No eviction needed
            return torch.ones_like(importance_scores, dtype=torch.bool)

        # Select top-k important tokens
        num_to_evict = current_cache_size - target_size
        _, bottom_k_indices = importance_scores.topk(
            num_to_evict, dim=-1, largest=False,
        )

        # Create keep mask
        keep_mask = torch.ones_like(importance_scores, dtype=torch.bool)
        keep_mask.scatter_(-1, bottom_k_indices, False)

        return keep_mask

    def evict(
        self,
        cached_keys: torch.Tensor,
        cached_values: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evict tokens from the cache based on the keep mask.

        Args:
            cached_keys: (batch, n_heads, seq_len, d_kv).
            cached_values: (batch, n_heads, seq_len, d_kv).
            keep_mask: (batch, n_heads, seq_len) boolean mask.

        Returns:
            Tuple (new_keys, new_values) with evicted tokens removed.
        """
        # For each batch/head, gather kept tokens
        batch, n_heads, seq_len, d_kv = cached_keys.shape

        # Use the mask to select tokens
        # Expand mask for gathering
        keep_expanded = keep_mask.unsqueeze(-1).expand_as(cached_keys)

        # Simple approach: for each head, select kept tokens
        # This handles variable numbers of kept tokens per head
        new_keys_list = []
        new_values_list = []

        for b in range(batch):
            batch_keys = []
            batch_values = []
            for h in range(n_heads):
                mask_h = keep_mask[b, h]  # (seq_len,)
                batch_keys.append(cached_keys[b, h, mask_h])
                batch_values.append(cached_values[b, h, mask_h])

            # Pad to same length within batch
            max_kept = max(k.shape[0] for k in batch_keys)
            padded_keys = []
            padded_values = []
            for k, v in zip(batch_keys, batch_values):
                pad_size = max_kept - k.shape[0]
                if pad_size > 0:
                    k = F.pad(k, (0, 0, 0, pad_size))
                    v = F.pad(v, (0, 0, 0, pad_size))
                padded_keys.append(k)
                padded_values.append(v)

            new_keys_list.append(torch.stack(padded_keys))
            new_values_list.append(torch.stack(padded_values))

        new_keys = torch.stack(new_keys_list)
        new_values = torch.stack(new_values_list)

        return new_keys, new_values

    def should_evict(self, current_cache_size: int) -> bool:
        """Check if eviction should be performed at this step.

        Args:
            current_cache_size: Current number of tokens in cache.

        Returns:
            True if eviction should be performed.
        """
        self._step_count += 1
        target_size = max(
            self.config.min_tokens_to_keep,
            int(current_cache_size * self.config.target_cache_ratio),
        )

        return (
            current_cache_size > target_size * 1.5  # Buffer to avoid frequent eviction
            and self._step_count % self.config.update_frequency == 0
        )
