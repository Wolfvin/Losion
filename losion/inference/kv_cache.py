"""
Losion KV Cache — Efficient key-value cache for autoregressive generation.

Implements multiple caching strategies for the Losion Tri-Jalur architecture:
  - KVCache: Standard per-layer KV cache with MLA compressed mode
  - PagedKVCache: vLLM-style paged attention cache for memory-efficient serving
  - KVCacheCompressor: ChunkKV + EvolKV style cache compression

Credits:
  - vLLM PagedAttention (github.com/vllm-project/vllm)
  - ChunkKV (NeurIPS 2025)
  - DeepSeek-V2 MLA KV compression

Hardware: Pure PyTorch, compatible with CUDA / ROCm / CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================================
# KVCacheEntry — Single layer cache entry
# ============================================================================


@dataclass
class KVCacheEntry:
    """A single layer's KV cache entry.

    Supports two modes:
      - Standard: stores separate key and value tensors
      - MLA compressed: stores only the c_kv latent representation

    Attributes:
        key: Key tensor [batch, n_heads, seq_len, d_kv] (standard mode).
        value: Value tensor [batch, n_heads, seq_len, d_kv] (standard mode).
        c_kv: MLA compressed KV latent [batch, seq_len, mla_latent_dim]
              (MLA mode — key and value are None).
        mla_mode: Whether this entry uses MLA compressed cache.
    """

    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    c_kv: Optional[torch.Tensor] = None
    mla_mode: bool = False

    def __post_init__(self) -> None:
        """Validate consistency of the cache entry."""
        if self.mla_mode:
            if self.c_kv is None and self.key is None and self.value is None:
                # Empty MLA entry — fine, will be filled on first update
                return
            if self.c_kv is not None and (self.key is not None or self.value is not None):
                raise ValueError(
                    "MLA mode entry must have c_kv but not key/value tensors"
                )
        else:
            if self.c_kv is not None:
                raise ValueError(
                    "Non-MLA mode entry must not have c_kv tensor"
                )

    @property
    def seq_len(self) -> int:
        """Return the current sequence length stored in this entry."""
        if self.mla_mode:
            if self.c_kv is None:
                return 0
            return self.c_kv.shape[1]
        else:
            if self.key is None:
                return 0
            return self.key.shape[2]

    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes.

        Returns:
            Estimated memory footprint of this cache entry.
        """
        if self.mla_mode:
            if self.c_kv is None:
                return 0
            return self.c_kv.nelement() * self.c_kv.element_size()
        else:
            total = 0
            if self.key is not None:
                total += self.key.nelement() * self.key.element_size()
            if self.value is not None:
                total += self.value.nelement() * self.value.element_size()
            return total

    def slice(self, start: int, end: int) -> "KVCacheEntry":
        """Return a sliced view of this entry [start:end].

        Args:
            start: Start position (inclusive).
            end: End position (exclusive).

        Returns:
            New KVCacheEntry with the sliced tensors.
        """
        if self.mla_mode:
            return KVCacheEntry(
                c_kv=self.c_kv[:, start:end, :] if self.c_kv is not None else None,
                mla_mode=True,
            )
        else:
            return KVCacheEntry(
                key=self.key[:, :, start:end, :] if self.key is not None else None,
                value=self.value[:, :, start:end, :] if self.value is not None else None,
                mla_mode=False,
            )


# ============================================================================
# KVCache — Per-layer KV cache manager
# ============================================================================


class KVCache:
    """Per-layer, per-head KV cache for autoregressive generation.

    Supports both standard (separate K/V) and MLA compressed modes.
    In MLA mode, only the c_kv latent is stored, saving significant memory
    for long sequences.

    Args:
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.
        d_kv: Dimension per key/value head.
        mla_latent_dim: MLA latent compression dimension (0 = standard mode).
        dtype: Data type for cache tensors (default float16).
        device: Device for cache tensors (default "cpu").
        max_seq_len: Maximum sequence length (pre-allocation hint, 0 = dynamic).
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_kv: int,
        mla_latent_dim: int = 0,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
        max_seq_len: int = 0,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.mla_latent_dim = mla_latent_dim
        self.mla_mode = mla_latent_dim > 0
        self.dtype = dtype
        self.device = device
        self.max_seq_len = max_seq_len

        # Per-layer cache entries
        self._entries: List[KVCacheEntry] = [
            KVCacheEntry(mla_mode=self.mla_mode) for _ in range(n_layers)
        ]

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> KVCacheEntry:
        """Update the cache for a given layer with new key/value tensors.

        Appends new K/V to the existing cache along the sequence dimension.
        For MLA mode, expects new_k to be the c_kv latent [batch, seq, mla_latent_dim]
        and new_v is ignored.

        Args:
            layer_idx: Layer index to update.
            new_k: New key tensor [batch, n_heads, new_seq, d_kv]
                   or c_kv latent [batch, new_seq, mla_latent_dim] in MLA mode.
            new_v: New value tensor [batch, n_heads, new_seq, d_kv]
                   (ignored in MLA mode).

        Returns:
            Updated KVCacheEntry for this layer.

        Raises:
            ValueError: If layer_idx is out of range.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )

        entry = self._entries[layer_idx]

        if self.mla_mode:
            # MLA: new_k is actually c_kv latent
            c_kv = new_k  # [batch, new_seq, mla_latent_dim]

            if entry.c_kv is None:
                updated = KVCacheEntry(
                    c_kv=c_kv.to(dtype=self.dtype, device=self.device),
                    mla_mode=True,
                )
            else:
                # Concatenate along sequence dimension
                existing = entry.c_kv
                updated = KVCacheEntry(
                    c_kv=torch.cat([existing, c_kv.to(dtype=self.dtype, device=self.device)], dim=1),
                    mla_mode=True,
                )
        else:
            # Standard mode: separate K and V
            if entry.key is None:
                updated = KVCacheEntry(
                    key=new_k.to(dtype=self.dtype, device=self.device),
                    value=new_v.to(dtype=self.dtype, device=self.device),
                    mla_mode=False,
                )
            else:
                existing_k = entry.key
                existing_v = entry.value
                updated = KVCacheEntry(
                    key=torch.cat([
                        existing_k,
                        new_k.to(dtype=self.dtype, device=self.device),
                    ], dim=2),
                    value=torch.cat([
                        existing_v,
                        new_v.to(dtype=self.dtype, device=self.device),
                    ], dim=2),
                    mla_mode=False,
                )

        self._entries[layer_idx] = updated
        return updated

    def get(self, layer_idx: int) -> KVCacheEntry:
        """Get the cache entry for a given layer.

        Args:
            layer_idx: Layer index.

        Returns:
            KVCacheEntry for this layer.

        Raises:
            ValueError: If layer_idx is out of range.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )
        return self._entries[layer_idx]

    def clear(self) -> None:
        """Clear all cache entries, releasing memory."""
        self._entries = [
            KVCacheEntry(mla_mode=self.mla_mode) for _ in range(self.n_layers)
        ]

    def seq_len(self, layer_idx: int) -> int:
        """Get the current sequence length for a given layer.

        Args:
            layer_idx: Layer index.

        Returns:
            Current sequence length in the cache for this layer.
        """
        return self.get(layer_idx).seq_len

    def max_seq_len_across_layers(self) -> int:
        """Get the maximum sequence length across all layers.

        Returns:
            Maximum sequence length in the cache.
        """
        return max(entry.seq_len for entry in self._entries)

    def memory_bytes(self) -> int:
        """Estimate total memory usage across all layers.

        Returns:
            Total memory in bytes used by the cache.
        """
        return sum(entry.memory_bytes() for entry in self._entries)

    def memory_summary(self) -> Dict[str, Any]:
        """Get a detailed memory usage summary.

        Returns:
            Dictionary with per-layer and total memory statistics.
        """
        per_layer = {}
        for i, entry in enumerate(self._entries):
            per_layer[f"layer_{i}"] = {
                "seq_len": entry.seq_len,
                "memory_mb": entry.memory_bytes() / (1024 * 1024),
                "mla_mode": entry.mla_mode,
            }
        total_mb = self.memory_bytes() / (1024 * 1024)
        return {
            "total_memory_mb": total_mb,
            "n_layers": self.n_layers,
            "mla_mode": self.mla_mode,
            "per_layer": per_layer,
        }

    def truncate(self, layer_idx: int, new_len: int) -> None:
        """Truncate the cache to a shorter sequence length.

        Useful for rolling back after rejected speculative tokens.

        Args:
            layer_idx: Layer index.
            new_len: New sequence length (must be <= current length).
        """
        entry = self.get(layer_idx)
        if entry.seq_len <= new_len:
            return

        if self.mla_mode:
            if entry.c_kv is not None:
                self._entries[layer_idx] = KVCacheEntry(
                    c_kv=entry.c_kv[:, :new_len, :],
                    mla_mode=True,
                )
        else:
            if entry.key is not None:
                self._entries[layer_idx] = KVCacheEntry(
                    key=entry.key[:, :, :new_len, :],
                    value=entry.value[:, :, :new_len, :],
                    mla_mode=False,
                )

    def truncate_all(self, new_len: int) -> None:
        """Truncate all layers to a shorter sequence length.

        Args:
            new_len: New sequence length.
        """
        for i in range(self.n_layers):
            self.truncate(i, new_len)

    def pad_for_batch(
        self,
        seq_lens: List[int],
    ) -> List[KVCacheEntry]:
        """Create padded entries for batched generation with different seq lengths.

        For sequences shorter than the max length in the batch, the cache is
        left-aligned (padding is on the left), matching causal attention semantics.

        Args:
            seq_lens: List of sequence lengths for each item in the batch.

        Returns:
            List of padded KVCacheEntry, one per layer.
        """
        max_len = max(seq_lens)
        padded_entries: List[KVCacheEntry] = []

        for layer_idx in range(self.n_layers):
            entry = self._entries[layer_idx]

            if self.mla_mode:
                if entry.c_kv is None:
                    padded_entries.append(KVCacheEntry(mla_mode=True))
                    continue

                batch_size = entry.c_kv.shape[0]
                latent_dim = entry.c_kv.shape[2]
                padded = torch.zeros(
                    batch_size, max_len, latent_dim,
                    dtype=self.dtype, device=self.device,
                )
                for b, slen in enumerate(seq_lens):
                    offset = max_len - slen
                    padded[b, offset:, :] = entry.c_kv[b, :slen, :]
                padded_entries.append(KVCacheEntry(c_kv=padded, mla_mode=True))
            else:
                if entry.key is None:
                    padded_entries.append(KVCacheEntry(mla_mode=False))
                    continue

                batch_size = entry.key.shape[0]
                padded_k = torch.zeros(
                    batch_size, self.n_heads, max_len, self.d_kv,
                    dtype=self.dtype, device=self.device,
                )
                padded_v = torch.zeros_like(padded_k)
                for b, slen in enumerate(seq_lens):
                    offset = max_len - slen
                    padded_k[b, :, offset:, :] = entry.key[b, :, :slen, :]
                    padded_v[b, :, offset:, :] = entry.value[b, :, :slen, :]
                padded_entries.append(KVCacheEntry(
                    key=padded_k, value=padded_v, mla_mode=False,
                ))

        return padded_entries


# ============================================================================
# PagedKVCache — vLLM-style paged KV cache
# ============================================================================


@dataclass
class PageTable:
    """Page table mapping for a single sequence.

    Maps logical token positions to physical page indices.

    Attributes:
        page_ids: List of physical page IDs for this sequence.
        num_tokens_in_last_page: Number of valid tokens in the last page.
    """

    page_ids: List[int] = field(default_factory=list)
    num_tokens_in_last_page: int = 0


class PagedKVCache:
    """vLLM-style paged KV cache for memory-efficient serving.

    Divides the KV cache into fixed-size pages, allowing:
      - On-demand page allocation (no pre-allocation for max seq len)
      - Prefix caching (shared system prompt pages across requests)
      - Efficient memory management with page free/alloc

    Inspired by vLLM PagedAttention (github.com/vllm-project/vllm).

    Args:
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        d_kv: Dimension per key/value head.
        page_size: Number of tokens per page (default 16).
        num_pages: Total number of physical pages available.
        dtype: Data type for page tensors.
        device: Device for page tensors.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_kv: int,
        page_size: int = 16,
        num_pages: int = 65536,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_kv = d_kv
        self.page_size = page_size
        self.num_pages = num_pages
        self.dtype = dtype
        self.device = device

        # Physical page storage: [num_pages, n_heads, page_size, d_kv] per layer
        self._k_pages: List[torch.Tensor] = []
        self._v_pages: List[torch.Tensor] = []
        for _ in range(n_layers):
            k_store = torch.zeros(
                num_pages, n_heads, page_size, d_kv,
                dtype=dtype, device=device,
            )
            v_store = torch.zeros_like(k_store)
            self._k_pages.append(k_store)
            self._v_pages.append(v_store)

        # Free page pool
        self._free_pages: List[int] = list(range(num_pages))

        # Per-sequence page tables (seq_id -> PageTable)
        self._page_tables: Dict[int, PageTable] = {}

        # Prefix cache: hash of prefix tokens -> list of page IDs
        self._prefix_cache: Dict[int, List[int]] = {}

    @property
    def num_free_pages(self) -> int:
        """Number of free pages available for allocation."""
        return len(self._free_pages)

    @property
    def num_used_pages(self) -> int:
        """Number of pages currently in use."""
        return self.num_pages - self.num_free_pages

    def allocate_page(self) -> int:
        """Allocate a single free page.

        Returns:
            Physical page ID of the allocated page.

        Raises:
            MemoryError: If no free pages are available.
        """
        if not self._free_pages:
            raise MemoryError(
                "PagedKVCache out of memory: no free pages available. "
                f"Total pages: {self.num_pages}, all in use."
            )
        return self._free_pages.pop(0)

    def free_page(self, page_id: int) -> None:
        """Free a previously allocated page.

        Args:
            page_id: Physical page ID to free.

        Raises:
            ValueError: If the page_id is out of range or already free.
        """
        if page_id < 0 or page_id >= self.num_pages:
            raise ValueError(
                f"page_id {page_id} out of range [0, {self.num_pages})"
            )
        if page_id in self._free_pages:
            raise ValueError(f"Double free detected for page {page_id}")
        self._free_pages.append(page_id)
        # Sort free list for better allocation patterns
        self._free_pages.sort()

    def copy_page(self, src_page_id: int) -> int:
        """Copy a page to a new physical page.

        Used for copy-on-write semantics when sequences diverge
        from a shared prefix.

        Args:
            src_page_id: Source page ID to copy from.

        Returns:
            Physical page ID of the new copy.
        """
        dst_page_id = self.allocate_page()
        for layer_idx in range(self.n_layers):
            self._k_pages[layer_idx][dst_page_id].copy_(
                self._k_pages[layer_idx][src_page_id]
            )
            self._v_pages[layer_idx][dst_page_id].copy_(
                self._v_pages[layer_idx][src_page_id]
            )
        return dst_page_id

    def add_sequence(self, seq_id: int) -> None:
        """Register a new sequence in the page table.

        Args:
            seq_id: Unique sequence identifier.

        Raises:
            ValueError: If seq_id already exists.
        """
        if seq_id in self._page_tables:
            raise ValueError(f"Sequence {seq_id} already registered")
        self._page_tables[seq_id] = PageTable()

    def remove_sequence(self, seq_id: int) -> None:
        """Remove a sequence and free all its pages.

        Args:
            seq_id: Sequence identifier to remove.
        """
        if seq_id not in self._page_tables:
            return

        table = self._page_tables[seq_id]
        for page_id in table.page_ids:
            self.free_page(page_id)
        del self._page_tables[seq_id]

    def update(
        self,
        seq_id: int,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        """Write new K/V data into the paged cache for a sequence.

        Handles page allocation automatically when a new page is needed.

        Args:
            seq_id: Sequence identifier.
            layer_idx: Layer index.
            new_k: New key tensor [n_heads, new_seq, d_kv].
            new_v: New value tensor [n_heads, new_seq, d_kv].
        """
        if seq_id not in self._page_tables:
            self.add_sequence(seq_id)

        table = self._page_tables[seq_id]
        new_seq_len = new_k.shape[1]

        for pos in range(new_seq_len):
            # Compute the global position in the sequence
            current_total = (
                len(table.page_ids) * self.page_size
                - self.page_size
                + table.num_tokens_in_last_page
            ) if table.page_ids else 0
            global_pos = current_total + pos

            # Determine which page this position falls into
            page_index = global_pos // self.page_size
            pos_in_page = global_pos % self.page_size

            # Allocate new page if needed
            if page_index >= len(table.page_ids):
                page_id = self.allocate_page()
                table.page_ids.append(page_id)
                table.num_tokens_in_last_page = 0

            # Write to the physical page
            current_page_id = table.page_ids[page_index]
            self._k_pages[layer_idx][current_page_id, :, pos_in_page, :] = new_k[:, pos, :]
            self._v_pages[layer_idx][current_page_id, :, pos_in_page, :] = new_v[:, pos, :]

            table.num_tokens_in_last_page = pos_in_page + 1

    def get_page_table_tensor(
        self,
        seq_ids: List[int],
        max_num_pages: Optional[int] = None,
    ) -> torch.Tensor:
        """Get a batched page table tensor for GPU attention kernels.

        Args:
            seq_ids: List of sequence IDs in the batch.
            max_num_pages: Maximum number of pages per sequence (padded).
                If None, uses the maximum across the batch.

        Returns:
            Page table tensor [batch_size, max_num_pages] with physical page IDs.
            -1 indicates padding (no page).
        """
        if max_num_pages is None:
            max_num_pages = max(
                len(self._page_tables[sid].page_ids)
                for sid in seq_ids
            )
        max_num_pages = max(max_num_pages, 1)

        batch_size = len(seq_ids)
        table_tensor = torch.full(
            (batch_size, max_num_pages), -1,
            dtype=torch.long, device=self.device,
        )

        for b, sid in enumerate(seq_ids):
            pt = self._page_tables[sid]
            for i, pid in enumerate(pt.page_ids):
                if i < max_num_pages:
                    table_tensor[b, i] = pid

        return table_tensor

    def register_prefix(
        self,
        prefix_hash: int,
        prefix_token_ids: torch.Tensor,
        k_per_layer: List[torch.Tensor],
        v_per_layer: List[torch.Tensor],
    ) -> List[int]:
        """Register a shared prefix (e.g., system prompt) for reuse.

        Allocates pages and writes the prefix KV data. Returns page IDs
        that can be shared by multiple sequences starting with the same prefix.

        Args:
            prefix_hash: Hash of the prefix token IDs for cache lookup.
            prefix_token_ids: Prefix token IDs [seq_len].
            k_per_layer: List of key tensors per layer [n_heads, seq_len, d_kv].
            v_per_layer: List of value tensors per layer [n_heads, seq_len, d_kv].

        Returns:
            List of physical page IDs holding the prefix KV data.
        """
        seq_len = prefix_token_ids.shape[0]
        num_pages_needed = math.ceil(seq_len / self.page_size)
        page_ids: List[int] = []

        for _ in range(num_pages_needed):
            page_ids.append(self.allocate_page())

        # Write KV data into pages
        for layer_idx in range(self.n_layers):
            k = k_per_layer[layer_idx]  # [n_heads, seq_len, d_kv]
            v = v_per_layer[layer_idx]
            for page_local_idx, pid in enumerate(page_ids):
                start = page_local_idx * self.page_size
                end = min(start + self.page_size, seq_len)
                page_len = end - start
                self._k_pages[layer_idx][pid, :, :page_len, :] = k[:, start:end, :]
                self._v_pages[layer_idx][pid, :, :page_len, :] = v[:, start:end, :]

        self._prefix_cache[prefix_hash] = page_ids
        return page_ids

    def lookup_prefix(self, prefix_hash: int) -> Optional[List[int]]:
        """Look up a previously registered prefix by hash.

        Args:
            prefix_hash: Hash of the prefix token IDs.

        Returns:
            List of physical page IDs if found, None otherwise.
        """
        return self._prefix_cache.get(prefix_hash)

    def memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary.

        Returns:
            Dictionary with memory statistics.
        """
        page_bytes = (
            self.n_layers
            * 2  # K + V
            * self.n_heads
            * self.page_size
            * self.d_kv
            * torch.tensor([], dtype=self.dtype).element_size()
        )
        total_bytes = self.num_pages * page_bytes
        used_bytes = self.num_used_pages * page_bytes

        return {
            "total_memory_mb": total_bytes / (1024 * 1024),
            "used_memory_mb": used_bytes / (1024 * 1024),
            "free_memory_mb": (total_bytes - used_bytes) / (1024 * 1024),
            "num_pages_total": self.num_pages,
            "num_pages_used": self.num_used_pages,
            "num_pages_free": self.num_free_pages,
            "num_sequences": len(self._page_tables),
            "num_cached_prefixes": len(self._prefix_cache),
            "page_size": self.page_size,
        }


# ============================================================================
# KVCacheCompressor — Cache compression (ChunkKV + EvolKV style)
# ============================================================================


class KVCacheCompressor:
    """Compresses KV cache for long-context efficiency.

    Implements two complementary strategies:
      - ChunkKV-style: Groups KV pairs into chunks and scores/removes
        low-importance chunks based on attention weights.
      - EvolKV-style: Layer-wise adaptive budget allocation that
        distributes compression budget across layers based on their
        sensitivity.

    Credits:
      - ChunkKV (NeurIPS 2025): chunk-level importance scoring
      - EvolKV: layer-wise adaptive budget allocation

    Args:
        chunk_size: Number of tokens per chunk for ChunkKV scoring.
        importance_metric: How to score chunk importance
            ("attention_sum", "attention_max", "key_norm").
        budget_allocation: How to allocate compression budget across layers
            ("uniform" or "adaptive").
    """

    def __init__(
        self,
        chunk_size: int = 64,
        importance_metric: str = "attention_sum",
        budget_allocation: str = "adaptive",
    ) -> None:
        self.chunk_size = chunk_size
        self.importance_metric = importance_metric
        self.budget_allocation = budget_allocation

        # Per-layer sensitivity scores (for adaptive budget allocation)
        self._layer_sensitivity: Optional[List[float]] = None

    def score_chunks(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score KV chunks by importance.

        Groups tokens into chunks and computes an importance score for each.
        Higher score = more important = should be kept.

        Args:
            key: Key tensor [batch, n_heads, seq_len, d_kv].
            value: Value tensor [batch, n_heads, seq_len, d_kv].
            attention_weights: Optional attention weight tensor
                [batch, n_heads, seq_len, seq_len] for importance scoring.

        Returns:
            Chunk importance scores [batch, n_chunks].
        """
        batch, n_heads, seq_len, d_kv = key.shape
        n_chunks = math.ceil(seq_len / self.chunk_size)

        # Pad sequence to multiple of chunk_size
        pad_len = n_chunks * self.chunk_size - seq_len
        if pad_len > 0:
            key_padded = F.pad(key, (0, 0, 0, pad_len))
        else:
            key_padded = key

        # Reshape into chunks: [batch, n_heads, n_chunks, chunk_size, d_kv]
        key_chunks = key_padded.reshape(
            batch, n_heads, n_chunks, self.chunk_size, d_kv
        )

        if self.importance_metric == "attention_sum" and attention_weights is not None:
            # Sum of attention weights received by each chunk
            # attention_weights: [batch, n_heads, seq_len, seq_len]
            # Sum over query positions for each key chunk
            attn_to_keys = attention_weights.sum(dim=2)  # [batch, n_heads, seq_len]
            # Pad and reshape
            if pad_len > 0:
                attn_to_keys = F.pad(attn_to_keys, (0, pad_len))
            attn_chunks = attn_to_keys.reshape(
                batch, n_heads, n_chunks, self.chunk_size
            )
            # Average over heads and positions within chunk
            scores = attn_chunks.mean(dim=(1, 3))  # [batch, n_chunks]

        elif self.importance_metric == "attention_max" and attention_weights is not None:
            # Max attention weight received by any token in the chunk
            attn_to_keys = attention_weights.max(dim=2).values  # [batch, n_heads, seq_len]
            if pad_len > 0:
                attn_to_keys = F.pad(attn_to_keys, (0, pad_len))
            attn_chunks = attn_to_keys.reshape(
                batch, n_heads, n_chunks, self.chunk_size
            )
            scores = attn_chunks.max(dim=3).values.mean(dim=1)  # [batch, n_chunks]

        elif self.importance_metric == "key_norm":
            # Norm of key vectors — tokens with larger key norms
            # tend to be more important for attention
            key_norms = key_chunks.norm(dim=-1)  # [batch, n_heads, n_chunks, chunk_size]
            scores = key_norms.mean(dim=(1, 3))  # [batch, n_chunks]

        else:
            # Fallback: uniform scoring (no compression preference)
            scores = torch.ones(
                batch, n_chunks, device=key.device, dtype=key.dtype
            )

        return scores

    def compute_layer_sensitivity(
        self,
        cache: KVCache,
        attention_weights_per_layer: List[torch.Tensor],
    ) -> List[float]:
        """Compute per-layer sensitivity scores for adaptive budget allocation.

        Layers where attention is more concentrated (lower entropy) are more
        sensitive to KV compression and should retain more tokens.

        Args:
            cache: The KVCache to analyze.
            attention_weights_per_layer: Attention weights for each layer
                [batch, n_heads, seq_len, seq_len].

        Returns:
            List of sensitivity scores (higher = more sensitive = keep more).
        """
        sensitivities: List[float] = []

        for layer_idx in range(cache.n_layers):
            attn = attention_weights_per_layer[layer_idx]
            # Compute entropy of attention distribution per layer
            # Lower entropy = more concentrated = more sensitive
            eps = 1e-10
            entropy = -(attn * (attn + eps).log()).sum(dim=-1).mean().item()
            # Sensitivity is inverse of entropy (concentrated attn = high sensitivity)
            sensitivity = 1.0 / (entropy + 1.0)
            sensitivities.append(sensitivity)

        self._layer_sensitivity = sensitivities
        return sensitivities

    def _allocate_budget_adaptive(
        self,
        total_budget: int,
        n_layers: int,
    ) -> List[int]:
        """Allocate compression budget across layers adaptively.

        More sensitive layers get a larger share of the budget
        (i.e., they keep more tokens).

        Args:
            total_budget: Total tokens to keep across all layers.
            n_layers: Number of layers.

        Returns:
            List of token budgets per layer.
        """
        if self._layer_sensitivity is None or len(self._layer_sensitivity) != n_layers:
            # Fallback to uniform
            per_layer = total_budget // n_layers
            return [per_layer] * n_layers

        # Weight by sensitivity
        total_sensitivity = sum(self._layer_sensitivity)
        budgets: List[int] = []
        allocated = 0

        for i in range(n_layers):
            weight = self._layer_sensitivity[i] / total_sensitivity
            budget = int(total_budget * weight)
            budgets.append(budget)
            allocated += budget

        # Distribute remainder
        remainder = total_budget - allocated
        for i in range(remainder):
            budgets[i % n_layers] += 1

        return budgets

    def compress(
        self,
        cache: KVCache,
        budget_ratio: float,
        attention_weights_per_layer: Optional[List[torch.Tensor]] = None,
    ) -> KVCache:
        """Compress the KV cache according to the given budget ratio.

        Args:
            cache: The KVCache to compress.
            budget_ratio: Fraction of tokens to retain (0.0 to 1.0).
                E.g., 0.5 means keep 50% of tokens.
            attention_weights_per_layer: Optional attention weights for
                importance scoring. If None, uses key_norm metric.

        Returns:
            New compressed KVCache with reduced sequence lengths.
        """
        if budget_ratio <= 0.0 or budget_ratio >= 1.0:
            # No compression needed
            return cache

        # Determine total budget
        max_seq = cache.max_seq_len_across_layers()
        if max_seq == 0:
            return cache

        total_budget = max(1, int(max_seq * budget_ratio))

        # Allocate per-layer budgets
        if self.budget_allocation == "adaptive" and attention_weights_per_layer is not None:
            self.compute_layer_sensitivity(cache, attention_weights_per_layer)
            layer_budgets = self._allocate_budget_adaptive(total_budget, cache.n_layers)
        else:
            per_layer = max(1, total_budget // cache.n_layers)
            layer_budgets = [per_layer] * cache.n_layers

        # Create compressed cache with same config
        compressed = KVCache(
            n_layers=cache.n_layers,
            n_heads=cache.n_heads,
            d_kv=cache.d_kv,
            mla_latent_dim=cache.mla_latent_dim if cache.mla_mode else 0,
            dtype=cache.dtype,
            device=cache.device,
        )

        # Compress each layer
        for layer_idx in range(cache.n_layers):
            entry = cache.get(layer_idx)
            budget = layer_budgets[layer_idx]

            if cache.mla_mode:
                if entry.c_kv is None:
                    continue
                compressed._entries[layer_idx] = self._compress_mla_entry(
                    entry, budget,
                )
            else:
                if entry.key is None:
                    continue
                attn = (
                    attention_weights_per_layer[layer_idx]
                    if attention_weights_per_layer is not None
                    else None
                )
                compressed._entries[layer_idx] = self._compress_standard_entry(
                    entry, budget, attn,
                )

        return compressed

    def _compress_standard_entry(
        self,
        entry: KVCacheEntry,
        budget: int,
        attention_weights: Optional[torch.Tensor],
    ) -> KVCacheEntry:
        """Compress a standard (non-MLA) cache entry.

        Args:
            entry: The KVCacheEntry to compress.
            budget: Number of tokens to keep.
            attention_weights: Optional attention weights for scoring.

        Returns:
            Compressed KVCacheEntry.
        """
        key = entry.key  # [batch, n_heads, seq_len, d_kv]
        value = entry.value
        seq_len = key.shape[2]

        if seq_len <= budget:
            return entry

        # Score chunks
        scores = self.score_chunks(key, value, attention_weights)
        n_chunks = scores.shape[1]
        n_chunks_to_keep = max(1, int(n_chunks * budget / seq_len))

        # Select top-k chunks
        _, top_indices = scores.topk(n_chunks_to_keep, dim=-1)
        top_indices = top_indices.sort(dim=-1).values

        # Gather tokens from selected chunks
        batch = key.shape[0]
        n_heads = key.shape[1]
        d_kv = key.shape[3]

        kept_keys = []
        kept_values = []

        for b in range(batch):
            token_indices: List[int] = []
            for chunk_idx in top_indices[b]:
                start = chunk_idx.item() * self.chunk_size
                end = min(start + self.chunk_size, seq_len)
                token_indices.extend(range(start, end))

            # Truncate to budget
            token_indices = token_indices[:budget]
            idx_tensor = torch.tensor(
                token_indices, device=key.device, dtype=torch.long
            )
            kept_keys.append(key[b, :, idx_tensor, :])
            kept_values.append(value[b, :, idx_tensor, :])

        # Pad to same length within batch
        max_kept = max(k.shape[1] for k in kept_keys)
        padded_keys = []
        padded_values = []
        for k, v in zip(kept_keys, kept_values):
            pad_size = max_kept - k.shape[1]
            if pad_size > 0:
                k = F.pad(k, (0, 0, 0, pad_size))
                v = F.pad(v, (0, 0, 0, pad_size))
            padded_keys.append(k)
            padded_values.append(v)

        new_key = torch.stack(padded_keys)
        new_value = torch.stack(padded_values)

        return KVCacheEntry(key=new_key, value=new_value, mla_mode=False)

    def _compress_mla_entry(
        self,
        entry: KVCacheEntry,
        budget: int,
    ) -> KVCacheEntry:
        """Compress an MLA mode cache entry.

        For MLA mode, we apply simpler compression since we only have the
        c_kv latent. Uses norm-based scoring of latent vectors.

        Args:
            entry: The MLA KVCacheEntry to compress.
            budget: Number of tokens to keep.

        Returns:
            Compressed KVCacheEntry in MLA mode.
        """
        c_kv = entry.c_kv  # [batch, seq_len, mla_latent_dim]
        seq_len = c_kv.shape[1]

        if seq_len <= budget:
            return entry

        # Score by latent norm
        norms = c_kv.norm(dim=-1)  # [batch, seq_len]
        _, top_indices = norms.topk(budget, dim=-1)  # [batch, budget]
        top_indices = top_indices.sort(dim=-1).values

        # Gather
        batch = c_kv.shape[0]
        gathered = torch.stack([
            c_kv[b, top_indices[b], :] for b in range(batch)
        ])

        return KVCacheEntry(c_kv=gathered, mla_mode=True)
