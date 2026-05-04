"""
Losion KV Cache — Paged KV Cache Manager with INT4 quantization.

Provides memory-efficient key-value cache management for inference and
training evaluation:
  - PagedKVCacheManager: Pages-based KV cache with dynamic allocation
  - INT4KVCacheQuantizer: 4-bit quantization for KV cache compression
  - KVEvictionManager: LRU-based eviction for long sequences

Paged allocation avoids the memory fragmentation of static pre-allocation
and allows variable-length sequences without padding waste.

Credits:
  - vLLM PagedAttention: Kwon et al., SOSP 2023
  - KV Cache Compression: Liu et al., arXiv:2310.01877 (2023)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class INT4KVCacheQuantizer(nn.Module):
    """4-bit quantization for KV cache compression.

    Quantizes float16/bfloat16 KV pairs to 4-bit integers with
    per-head scale factors, achieving ~4x compression ratio.

    The quantization uses symmetric per-head quantization:
        k_int4 = round(k_float / scale)
        k_float = k_int4 * scale

    where scale = max(|k_float|) / 7 (4-bit range: -8 to 7)

    Args:
        n_heads: Number of attention heads.
        d_kv: Dimension per key/value head.
    """

    def __init__(self, n_heads: int = 8, d_kv: int = 64) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_kv = d_kv
        # Per-head scale factors
        self.register_buffer("k_scale", torch.ones(n_heads, d_kv))
        self.register_buffer("v_scale", torch.ones(n_heads, d_kv))

    def quantize_k(self, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize key tensor to INT4.

        Args:
            k: Key tensor (batch, n_heads, seq_len, d_kv).

        Returns:
            Tuple (k_int4, scale) where k_int4 is int8-stored 4-bit values.
        """
        scale = k.abs().amax(dim=(0, 2), keepdim=False).clamp(min=1e-5) / 7.0
        self.k_scale.copy_(scale.squeeze(0) if scale.dim() > 2 else scale)

        k_scaled = k / scale.unsqueeze(0).unsqueeze(2)
        k_int4 = torch.clamp(torch.round(k_scaled), -8, 7).to(torch.int8)
        return k_int4, scale

    def quantize_v(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize value tensor to INT4.

        Args:
            v: Value tensor (batch, n_heads, seq_len, d_kv).

        Returns:
            Tuple (v_int4, scale) where v_int4 is int8-stored 4-bit values.
        """
        scale = v.abs().amax(dim=(0, 2), keepdim=False).clamp(min=1e-5) / 7.0
        self.v_scale.copy_(scale.squeeze(0) if scale.dim() > 2 else scale)

        v_scaled = v / scale.unsqueeze(0).unsqueeze(2)
        v_int4 = torch.clamp(torch.round(v_scaled), -8, 7).to(torch.int8)
        return v_int4, scale

    def dequantize_k(self, k_int4: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT4 key back to float.

        Args:
            k_int4: Quantized key (int8-stored).
            scale: Per-head scale factor.

        Returns:
            Dequantized key tensor.
        """
        return k_int4.float() * scale.unsqueeze(0).unsqueeze(2)

    def dequantize_v(self, v_int4: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT4 value back to float.

        Args:
            v_int4: Quantized value (int8-stored).
            scale: Per-head scale factor.

        Returns:
            Dequantized value tensor.
        """
        return v_int4.float() * scale.unsqueeze(0).unsqueeze(2)


class KVEvictionManager:
    """LRU-based eviction for KV cache on long sequences.

    When the cache is full, evicts the least-recently-used pages
    to make room for new tokens. This enables processing sequences
    longer than the physical cache capacity.

    Args:
        max_pages: Maximum number of pages in the cache.
        page_size: Number of tokens per page.
    """

    def __init__(self, max_pages: int = 1024, page_size: int = 16) -> None:
        self.max_pages = max_pages
        self.page_size = page_size
        self._lru_counter = 0
        self._page_access: Dict[int, int] = {}

    def touch(self, page_id: int) -> None:
        """Mark a page as recently accessed."""
        self._lru_counter += 1
        self._page_access[page_id] = self._lru_counter

    def evict(self, n_pages: int) -> List[int]:
        """Select n_pages least-recently-used pages for eviction.

        Args:
            n_pages: Number of pages to evict.

        Returns:
            List of page IDs to evict.
        """
        if len(self._page_access) <= n_pages:
            return list(self._page_access.keys())

        sorted_pages = sorted(self._page_access.items(), key=lambda x: x[1])
        evict_ids = [pid for pid, _ in sorted_pages[:n_pages]]
        for pid in evict_ids:
            del self._page_access[pid]

        return evict_ids


class PagedKVCacheManager:
    """Paged KV cache with dynamic allocation and INT4 quantization.

    Instead of pre-allocating a contiguous block of memory for the KV
    cache, this manager allocates pages (small fixed-size blocks) on
    demand and tracks them with a page table. This eliminates memory
    fragmentation and allows variable-length sequences without waste.

    The cache supports:
    - Dynamic page allocation per sequence
    - INT4 quantization for 4x compression
    - LRU eviction for sequences longer than cache capacity
    - Efficient page table lookups

    Args:
        n_heads: Number of attention heads.
        d_kv: Dimension per key/value head.
        page_size: Number of tokens per page (default 16).
        max_pages: Maximum number of pages (default 1024).
        dtype: Data type for cache storage.
        use_int4: Whether to use INT4 quantization.
    """

    def __init__(
        self,
        n_heads: int = 8,
        d_kv: int = 64,
        page_size: int = 16,
        max_pages: int = 1024,
        dtype: torch.dtype = torch.bfloat16,
        use_int4: bool = False,
    ) -> None:
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.page_size = page_size
        self.max_pages = max_pages
        self.dtype = dtype
        self.use_int4 = use_int4

        # Page table: maps (seq_id, page_idx) -> physical page
        self._page_table: Dict[Tuple[int, int], int] = {}
        self._free_pages: List[int] = list(range(max_pages))
        self._next_page_id = 0

        # INT4 quantizer
        self.quantizer = INT4KVCacheQuantizer(n_heads, d_kv) if use_int4 else None

        # Eviction manager
        self.eviction = KVEvictionManager(max_pages, page_size)

        # Physical cache storage (lazy allocation)
        self._k_cache: Optional[torch.Tensor] = None
        self._v_cache: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None

    def _ensure_cache_allocated(self, device: torch.device) -> None:
        """Allocate physical cache storage on first use."""
        if self._k_cache is None or self._device != device:
            self._k_cache = torch.zeros(
                self.max_pages, self.n_heads, self.page_size, self.d_kv,
                dtype=torch.int8 if self.use_int4 else self.dtype,
                device=device,
            )
            self._v_cache = torch.zeros(
                self.max_pages, self.n_heads, self.page_size, self.d_kv,
                dtype=torch.int8 if self.use_int4 else self.dtype,
                device=device,
            )
            self._device = device

    def allocate(self, seq_id: int, n_tokens: int) -> List[int]:
        """Allocate pages for a sequence.

        Args:
            seq_id: Sequence identifier.
            n_tokens: Number of tokens to allocate.

        Returns:
            List of physical page IDs allocated.
        """
        n_pages_needed = (n_tokens + self.page_size - 1) // self.page_size
        allocated = []

        for page_idx in range(n_pages_needed):
            if not self._free_pages:
                # Evict LRU pages
                evicted = self.eviction.evict(1)
                self._free_pages.extend(evicted)

            page_id = self._free_pages.pop(0)
            self._page_table[(seq_id, page_idx)] = page_id
            self.eviction.touch(page_id)
            allocated.append(page_id)

        return allocated

    def write(
        self,
        seq_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Write KV data to the cache.

        Args:
            seq_id: Sequence identifier.
            k: Key tensor (1, n_heads, seq_len, d_kv).
            v: Value tensor (1, n_heads, seq_len, d_kv).
            positions: Token positions (1, seq_len).
        """
        self._ensure_cache_allocated(k.device)
        seq_len = k.shape[2]

        for t in range(seq_len):
            pos = positions[0, t].item()
            page_idx = pos // self.page_size
            offset = pos % self.page_size

            key = (seq_id, page_idx)
            if key not in self._page_table:
                self.allocate(seq_id, (page_idx + 1) * self.page_size)

            page_id = self._page_table[key]
            self._k_cache[page_id, :, offset, :] = k[0, :, t, :]
            self._v_cache[page_id, :, offset, :] = v[0, :, t, :]
            self.eviction.touch(page_id)

    def read(
        self,
        seq_id: int,
        n_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read KV data from the cache.

        Args:
            seq_id: Sequence identifier.
            n_tokens: Number of tokens to read.

        Returns:
            Tuple (k, v) tensors of shape (1, n_heads, n_tokens, d_kv).
        """
        self._ensure_cache_allocated(self._device or torch.device("cpu"))
        n_pages = (n_tokens + self.page_size - 1) // self.page_size

        k_parts = []
        v_parts = []

        for page_idx in range(n_pages):
            key = (seq_id, page_idx)
            if key in self._page_table:
                page_id = self._page_table[key]
                k_parts.append(self._k_cache[page_id])
                v_parts.append(self._v_cache[page_id])

        if k_parts:
            k = torch.cat(k_parts, dim=1)[:, :n_tokens, :]
            v = torch.cat(v_parts, dim=1)[:, :n_tokens, :]
            return k.unsqueeze(0), v.unsqueeze(0)

        # Return empty cache
        empty_k = torch.zeros(1, self.n_heads, n_tokens, self.d_kv, device=self._device)
        empty_v = torch.zeros(1, self.n_heads, n_tokens, self.d_kv, device=self._device)
        return empty_k, empty_v

    def free(self, seq_id: int) -> None:
        """Free all pages for a sequence.

        Args:
            seq_id: Sequence identifier.
        """
        keys_to_remove = [(sid, pidx) for sid, pidx in self._page_table if sid == seq_id]
        for key in keys_to_remove:
            page_id = self._page_table.pop(key)
            self._free_pages.append(page_id)
            if page_id in self.eviction._page_access:
                del self.eviction._page_access[page_id]
