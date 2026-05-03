"""
Losion Inference — Efficient inference subsystem for the Losion framework.

Provides KV caching, generation pipeline, and batched serving:
  - KVCache: Standard and MLA-compressed per-layer KV cache
  - PagedKVCache: vLLM-style paged attention cache
  - KVCacheCompressor: ChunkKV + EvolKV style cache compression
  - GenerationConfig: Comprehensive generation parameters
  - LogitsProcessor: Temperature, top-k, top-p, repetition penalty
  - SpeculativeDecoder: SSM-pathway draft with full-model verification
  - ContinuousBatcher: Iteration-level scheduling for concurrent requests
  - LosionGenerator: Main generation API (greedy, sampling, beam, speculative)

Credits:
  - vLLM PagedAttention (github.com/vllm-project/vllm)
  - ChunkKV (NeurIPS 2025)
  - DeepSeek-V2 MLA KV compression
  - EAGLE-3 (speculative decoding, Li et al. 2025)
  - HuggingFace generate() API design
"""

from losion.inference.kv_cache import (
    KVCache,
    KVCacheCompressor,
    KVCacheEntry,
    PageTable,
    PagedKVCache,
)
from losion.inference.generation import (
    ContinuousBatcher,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    LogitsProcessor,
    LosionGenerator,
    SpeculativeDecoder,
)

__all__ = [
    # KV Cache
    "KVCache",
    "KVCacheCompressor",
    "KVCacheEntry",
    "PageTable",
    "PagedKVCache",
    # Generation
    "ContinuousBatcher",
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResult",
    "GenerationStatus",
    "LogitsProcessor",
    "LosionGenerator",
    "SpeculativeDecoder",
]
