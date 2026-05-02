# Changelog

All notable changes to the Losion project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-05-02 — "Lightning & Liquid"

### Added — HIGH Priority

- **Lightning Attention** (`core/attention/lightning_attention.py`): O(1) inference per token,
  4M token context via hybrid local-window (softmax) + global linear attention with chunked
  processing. Backward-compatible MLA integration with KV latent compression.

- **Parallel-Head Mode** (`models/parallel_head.py`): Eliminates routing overhead for the
  Losion-1B model by running all three pathways in parallel and blending outputs with a
  learned gate. Suitable for deployment scenarios where routing latency matters.

- **BitNet 1.58-bit Quantization** (`core/quantization/bitnet.py`): Ternary weight quantization
  {-1, 0, +1} with absmean scaling, straight-through estimator (STE), gradual quantization
  schedule, and int2 weight packing for ~6x memory reduction at inference.

### Added — MEDIUM Priority

- **Heterogeneous MoE** (`core/retrieval/heterogeneous_moe.py`): Variable-size experts with
  learned capacity allocation, allowing experts to specialize at different granularities.

- **Matryoshka MoE** (`core/retrieval/matryoshka_moe.py`): Elastic expert count with nested
  Matryoshka-style routing — supports variable active-expert counts at inference for
  compute-quality tradeoffs.

- **Gradient-Routed MoE** (`core/retrieval/gradient_routed_moe.py`): Loss-aligned routing
  that uses gradient signals to improve expert-token affinity, reducing routing collapse.

- **FP8 Training Pipeline** (`core/quantization/fp8_training.py`): Mixed FP8/BF16 training
  with dynamic scaling for ~2x throughput on H100/H200 GPUs.

- **Post-Training NAS** (`core/nas/layer_search.py`): DARTS-style differentiable architecture
  search for post-training layer optimization — identifies which layers benefit from
  attention vs. SSM vs. MoE.

### Added — LOW Priority

- **Shared Attention** (`core/attention/shared_attention.py`): Zamba2-style shared attention
  parameter pool with configurable sharing patterns. ~6x KV cache reduction when multiple
  layers share the same attention parameters.

- **MTP Speculative Decoding** (`core/output/speculative_decoder.py`): Multi-Token Prediction
  speculative decoding for ~1.8x inference speedup. Drafts multiple tokens per step and
  verifies against the full model.

- **Asymmetric MoE Placement** (`core/retrieval/asymmetric_placement.py`): Selective MoE
  placement with layer-wise sparsity — only places MoE in layers where it's most beneficial,
  reducing compute in early/late layers.

### Added — LONG-TERM

- **Liquid SSM** (`core/ssm/liquid_ssm.py`): Adaptive compute depth SSM with per-token
  complexity estimation via ComplexityGate. Tokens assessed as "easy" early-exit after a
  single SSD pass (depth 1), while complex tokens receive full multi-layer treatment (depth 3).
  LiquidSSD provides input-adaptive time constants that modulate state decay.

### Changed

- Updated all YAML configs (`losion-1b.yaml`, `losion-7b.yaml`, `losion-48b.yaml`) with
  v0.4 feature flags and new parameters.
- Updated `__init__.py` across all core submodules to export v0.4 classes.

---

## [0.3.0] — 2026-04-15 — "Tri-Jalur"

### Added

- **Tri-Jalur Router Architecture**: Three-pathway design (SSM, Attention+Compression, Retrieval)
  with bias-based aux-loss-free routing and GRPO training.
- **Jalur 1 (SSM)**: Mamba-2 SSD + RWKV-7 WKV + Gated DeltaNet with 4:1:1 interleaving.
- **Jalur 2 (Attention+Compression)**: MLA + iRoPE + Pairformer with 8x KV compression.
- **Jalur 3 (Retrieval)**: MoE + Engram Memory + Expert Choice routing (16–256 experts).
- **Adaptive Router**: BiasRouter (DeepSeek-style) + ThinkingToggle (Qwen3-style).
- **Reasoning**: MCTS, Neuro-symbolic, Parallel Thinking modules.
- **Output**: Flow Matching, Diffusion Refinement.
- **Elastic**: Matryoshka dimension elasticity.
- **Training**: Full trainer, GRPO, Curriculum, RLHF, Active Learning.
- **Models**: LosionModel, LosionForCausalLM with 1B/7B/48B configs.
- **6 Novel Contributions**: SSD-DeltaNet-MLA Trinity, Adaptive iRoPE-5:1, GRPO Router,
  RWKV+MTP, Meta-State MoE, Jamba++.
