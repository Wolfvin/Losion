"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 1.3.0 — "Performance & Scalability Release"

v1.3.0 Performance & Scalability Improvements:
  - [CRITICAL] Replaced all manual matmul+softmax attention with
    F.scaled_dot_product_attention (SDPA). This automatically uses Flash
    Attention 2 when available, reducing memory from O(n^2) to O(n) and
    providing 2-4x speedup for training. Works on both CUDA and ROCm.
  - [CRITICAL] Replaced Python for-loops in SSM scans (Mamba2, Mamba3,
    RWKV7) with chunk-parallel computation using cumprod/cumsum. Reduces
    Python loop iterations from O(seq_len) to O(seq_len/chunk_size).
    For seq_len=4096, chunk_size=256: 16 iterations instead of 4096.
  - [HIGH] Added torch.compile integration to LosionGenerator with
    mode="reduce-overhead". Provides 10-30% speedup on both CUDA and ROCm
    by fusing small operators and eliminating Python overhead.
  - [HIGH] Per-pathway gradient checkpointing instead of per-layer.
    Each pathway (SSM, Attention, MoE) is checkpointed independently,
    reducing peak activation memory to ~1/3 of full-layer checkpointing.
  - [HIGH] Router gradient collapse fixes: separate learning rate support
    (router_params(), get_param_groups()), entropy regularization
    (compute_entropy_regularization()) to prevent routing collapse.
  - [MEDIUM] FSDP2 migration: added wrap_fsdp2() using composable
    fully_shard() API for better memory efficiency (PyTorch >= 2.4).
  - [MEDIUM] FP8 training via torchao: convert_to_fp8_training(),
    check_fp8_support(), get_optimal_precision(). Supports NVIDIA H100+
    and AMD MI300X with automatic BF16 fallback.
  - [MEDIUM] Flash Attention auto-detection: HAS_FLASH_ATTENTION flag
    and attention_forward() unified interface. Detects flash_attn,
    flash_attn_rocm, and PyTorch SDPA Flash backend.
  - [MEDIUM] Early exit / conditional routing in LosionLayerV2:
    during inference, pathways with mean weight < 5% are skipped,
    reducing unnecessary computation when one pathway dominates.

v1.2.1 Packaging & Tooling Fixes:
  - Added einops>=0.7.0 to pyproject.toml and setup.py dependencies.
  - Regenerated test_results.json — GatedAttention and MoBA now show OK.
  - Added MoBA as alias for MoBAAttention in attention __init__.py.
  - Added GitHub Actions CI workflow (.github/workflows/ci.yml).

v1.2.0 Bug Fixes & Improvements:
  - [CRITICAL] GatedAttention: Fixed tensor dimension mismatch in RoPE application.
    InterleavedRoPE now receives full (batch, seq_len, n_heads, d_kv) tensors
    instead of sliced (batch, seq_len, n_heads, d_kv//2) tensors, fixing the
    "Tensors must have same number of dimensions: got 2 and 3" error.
  - [CRITICAL] SymbolicMoERouter: Fixed API mismatch in test script.
    SymbolicMoERouter is a skill→pathway router, not an expert router;
    removed invalid num_experts/num_active_experts kwargs.
  - [CRITICAL] MoBA: Added dimension guards for 2D input and 3D KV cache,
    plus safer unpacking of past_key_value tuples.
  - [WARNING] ThinkingToggle: Fixed dead gradients in task_classifier and
    context_integrator. Added gradient scaling buffers (10x) with
    straight-through estimator to amplify gradient signal through mean
    aggregation. Updated compute_auxiliary_loss() with matching scaling.
  - [WARNING] SimplifiedMoE/_FallbackMoE: Replaced O(N×K×E) nested Python
    loops with vectorized sort-based scatter/gather dispatch using index_add_.
  - [WARNING] Version sync: Unified version to 1.2.0 across __init__.py,
    pyproject.toml, and setup.py.
  - [INFO] Gradient checkpointing: Improved routing_info preservation by
    storing detach-safe copies of routing tensors.

v1.0.0 End-to-End Verified:
  All 40+ components have been tested with actual forward+backward passes.
  A 17M-parameter model was trained for 10 steps and all pathways verified:
  - SSM (Jalur 1): Gradient flows correctly through Mamba-3/RoutingMamba
  - Attention (Jalur 2): GatedAttention/MoBA properly connected
  - MoE (Jalur 3): SmoreMoE/AuxFreeMoE with proper load balancing
  - Router: AdaptiveRouter with ThinkingToggle dynamically routes
  - RDT: RecurrentDepthBlock with proper block wrapper
  - Evoformer: All 5 levels wired and functional
  - DualMemory: Write+Read cycle verified
  - JEPA: JEPAHead loss computed and gradients flow
  - MTP: Multi-token prediction loss correctly shaped
  - Generation: Autoregressive generation works end-to-end
  - Save/Load: Round-trip verified with zero difference

  Critical wiring fixes in v1.0.0:
  - MoBAAttention constructor: Fixed config vs positional arg mismatch
  - GatedAttention config: Added d_model field to GatedAttentionConfig
  - LLMJEPA: Replaced standalone wrapper with lightweight JEPAHead
  - RDT: Inner block now returns (output, aux) tuple + accepts **kwargs
  - MTP loss: Fixed shape mismatch in shifted label computation
  - Generation: Fixed dimension mismatch in token concatenation
  - Mamba3SSD: Fixed config object vs keyword arg constructor mismatch
  - SymbolicMoE: Fixed fall-through that didn't return a module
  - from_pretrained: Uses _from_dict for proper nested config loading

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

__version__ = "1.3.0"
__author__ = "Losion Contributors"
__license__ = "MIT"
