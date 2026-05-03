"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 1.4.0 — "Kernel Optimization Release"

v1.4.0 Kernel Optimization Improvements:
  - [CRITICAL] New `losion.core.kernel` module with 12 sub-modules providing
    comprehensive GPU kernel optimizations for all three pathways:
    * sdpa_compat — Unified SDPA/Flash Attention interface with auto-detection
      (FlashAttention-2/3, PyTorch SDPA, math fallback)
    * ssm_kernels — Parallel associative scan + chunk-parallel scan + Triton SSM
      kernels. Replaces O(n) Python for-loops with O(log n) parallel reduction.
    * flash_attn — FlashAttention 2/3/4 integration with automatic version
      selection and KV cache support.
    * triton_kernels — Custom Triton kernels for fused tri-pathway blend,
      fused norm+gate, and MoE dispatch/combine.
    * compile_utils — torch.compile custom FX graph optimization passes,
      FusedPathwayNorms, FusedRoutingBlend, selective compilation.
    * fp8_native — Native FP8 training via torchao and NVIDIA Transformer Engine.
      DeepSeek-V3 fine-grained (1x128/128x128) tile-wise FP8 scaling.
    * speculative — Speculative decoding with SSM-as-drafter. Losion's SSM
      pathway naturally serves as a fast drafter for 2-3x inference speedup.
    * early_exit — Pathway-level and layer-level early exit via router
      confidence. 30-60% compute reduction for easy inputs.
    * paged_kv — PagedAttention (vLLM-style), E8 lattice VQ (10-33x KV
      compression), INT4 quantized KV cache, block-wise eviction.
    * parallel_pathway — Parallel pathway execution using CUDA streams
      (Nemotron-3 style). All three pathways compute simultaneously.
    * expert_prefetch — Expert prefetch via routing prediction. CPU/GPU hybrid
      MoE inference (KTransformers-style). MoE communication overlap.
    * fsdp_utils — FSDP2 + FP8 combined training pipeline, native Tensor
      Parallelism, router separate LR + entropy regularization.

  - [CRITICAL] Replaced manual `torch.matmul + F.softmax` attention with
    F.scaled_dot_product_attention (SDPA) in 6 additional modules:
    * SharedAttentionPool.compute_attention() — 2-4x speedup
    * SharedAttentionLayer._compute_attention_unique() — 2-4x speedup
    * SparseAttentionExpert (MoSA) — auto SDPA with sparse fallback
    * EvoformerLevel._compute_revision() — SDPA for feedback
    * All new kernel modules use SDPA exclusively

  - [HIGH] Early exit / conditional routing in LosionLayer.forward():
    During inference, pathways with mean routing weight <5% are skipped,
    reducing unnecessary computation when one pathway dominates.

  - [HIGH] Router gradient collapse fixes:
    * get_param_groups() — 10x higher LR for router parameters
    * compute_entropy_regularization() — entropy-based routing balance
    * compute_load_balancing_loss() — Switch Transformer style load balance

  - [MEDIUM] torch.compile integration improvements:
    * compile_losion_model() — model-level compilation with optimal settings
    * compile_pathway() — per-pathway selective compilation
    * FusedPathwayNorms — fuses 3 RMSNorm calls into single kernel
    * FusedRoutingBlend — fuses routing + weighted sum

  - [MEDIUM] FP8 native training support:
    * torchao-based FP8 — simple, works on any hardware
    * Transformer Engine FP8 — maximum performance on H100+
    * DeepSeek-V3 fine-grained FP8 — tile-wise (1x128) scaling
    * setup_fp8_fsdp2_training() — combined FP8 + FSDP2 pipeline

  - [MEDIUM] Speculative decoding with SSM-as-drafter:
    * SSMDraftModel — extracts SSM pathway as fast drafter
    * SpeculativeDecoder — verifies draft tokens with full model
    * Unique to Losion: SSM pathway naturally serves as drafter

  - [MEDIUM] PagedAttention + KV cache compression:
    * PagedKVCacheManager — vLLM-style paged KV cache (<4% waste)
    * E8LatticeQuantizer — 10-33x KV cache compression
    * INT4KVCacheQuantizer — 4x compression with minimal quality loss
    * KVEvictionManager — structured block-wise KV cache pruning

  - [LOW] Parallel pathway execution (Nemotron-3 style):
    * ParallelPathwayExecutor — CUDA streams for concurrent pathways
    * FusedPathwayModule — fused module for parallel execution

  - [LOW] Expert prefetch via routing prediction:
    * ExpertPrefetcher — prefetch MoE experts before they are needed
    * MoECommunicationOverlap — overlap MoE all-to-all with SSM/Attn

References for v1.4.0:
  - FlashAttention-2: Dao (arXiv:2307.08691)
  - FlashAttention-3: Dao et al. (arXiv:2407.08608)
  - FlashAttention-4: (arXiv:2603.05451)
  - Mamba-2 SSD: Gu & Dao (arXiv:2405.21075)
  - Mamba-3: (arXiv:2603.15569)
  - PyTorch Mamba2 Kernel Fusion: pytorch.org/blog/accelerating-mamba2-with-kernel-fusion
  - FlashMoE: (arXiv:2506.04667)
  - DeepSeek-V3 FP8: (arXiv:2412.19437)
  - NVIDIA Transformer Engine: github.com/NVIDIA/TransformerEngine
  - PagedAttention / vLLM: Kwon et al. (SOSP 2023)
  - PagedEviction: (arXiv:2509.04377)
  - E8 Lattice VQ: vLLM Issue #39241
  - Triton Anatomy: (arXiv:2511.11581)
  - Warp Specialization in Triton: PyTorch Blog 2025
  - torch.compile FX passes: blog.ezyang.com/2024/11/ways-to-use-torch-compile
  - Ring Attention / Striped Attention: (DistFlashAttn)
  - SpecForge: (arXiv:2603.18567)
  - SwiftSpec: (dl.acm.org/doi/10.1145/3779212.3790246)
  - KTransformers: (dl.acm.org/doi/10.1145/3731569.3764843, SOSP 2025)
  - AutoKernel: (arXiv:2603.21331)
  - Nemotron 3: (arXiv:2604.12374)
  - Routing Mamba: NeurIPS 2025
  - Occult: ICML 2025
  - INT4 Decoding GQA: PyTorch Blog
  - FSDP2+FP8: pytorch.org/blog/training-using-float8-fsdp2
  - SimpleFSDP: (ResearchGate 385510534)
  - Early Exit Survey: (arXiv:2501.07670)
  - Losion Framework: Wolfvin (github.com/Wolfvin/Losion)

v1.3.0 Performance & Scalability Improvements:
  - [CRITICAL] Replaced all manual matmul+softmax attention with
    F.scaled_dot_product_attention (SDPA).
  - [CRITICAL] Replaced Python for-loops in SSM scans with chunk-parallel
    computation.
  - [HIGH] Added torch.compile integration to LosionGenerator.
  - [HIGH] Per-pathway gradient checkpointing.
  - [HIGH] Router gradient collapse fixes.
  - [MEDIUM] FSDP2 migration.
  - [MEDIUM] FP8 training via torchao.
  - [MEDIUM] Flash Attention auto-detection.
  - [MEDIUM] Early exit / conditional routing.

v1.2.1 Packaging & Tooling Fixes:
  - Added einops>=0.7.0 to pyproject.toml and setup.py dependencies.
  - Regenerated test_results.json.
  - Added MoBA as alias for MoBAAttention.
  - Added GitHub Actions CI workflow.

v1.2.0 Bug Fixes & Improvements:
  - GatedAttention dim mismatch, MoBA indexing, SymbolicMoERouter API mismatch,
    ThinkingToggle dead gradients, MoE nested loop, version mismatch,
    gradient checkpointing routing_info.

v1.0.0 End-to-End Verified:
  All 40+ components tested with forward+backward passes.

Losion combines three complementary computational pathways into a single
adaptive architecture:

  Jalur 1 (SSM):           Mamba-2 SSD + RWKV-7 WKV + Gated DeltaNet
                            + Mamba-3 + Routing Mamba + Liquid SSM
                            + PoST Decay + FG2-GDN + Structured Sparse SSM
  Jalur 2 (Attention):     MLA + iRoPE + Lightning Attention + RoPE
                            + Gated Attention + MoBA + KDA+MLA
                            + Cross-Jalur Attention-MoE Routing
                            + AttnRes (Attention Residuals, MoonshotAI 2026)
                            + Child-3W (QKV-level MoE routing)
  Jalur 3 (Retrieval):     MoE + Engram Memory + Expert Choice
                            + S'MoRE + Symbolic-MoE + AuxFreeMoE
                            + ∞-MoE + MoHGE (Heterogeneous Grouped Experts)

Router:  Adaptive (BiasRouter + ThinkingToggle + Symbolic-MoE), GRPO/DAPO-trained.
         + Router ↔ Expert Co-Evolution (Evoformer Level 5)
"""

__version__ = "1.4.0"
__author__ = "Losion Contributors"
__license__ = "MIT"
