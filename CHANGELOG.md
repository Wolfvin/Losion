# Changelog

All notable changes to the Losion project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] — 2026-05-03 — "Mythos & Mamba"

### Added — Recurrent-Depth Transformer (OpenMythos / Claude Mythos)

- **Recurrent-Depth Transformer (RDT)** (`core/recurrent/rdt.py`): Looped transformer blocks
  with shared weights for 2-3x parameter efficiency. Inspired by OpenMythos reconstruction
  of Claude Mythos architecture. Includes LTI-Stable Injection (spectral radius constraint
  for training stability), Adaptive Computation Time (variable-depth halting), loop-index
  positional embeddings, and depth-wise LoRA per iteration (Relaxed Recursive Transformers).
  Credits: Universal Transformers (Dehghani 2019), OpenMythos (Kye Gomez 2026),
  Relaxed Recursive Transformers (Bae 2024), ACT (Graves 2016).

### Added — Attention Improvements (NeurIPS 2025)

- **Gated Attention** (`core/attention/gated_attention.py`): Sigmoid gate after softmax
  attention from Qwen (NeurIPS 2025 Best Paper). Eliminates attention sinks, adds soft
  per-head sparsity, synergizes with MoE routing. Near-identity initialization.
  Credits: Qwen Team (NeurIPS 2025 Best Paper).

- **MoBA — Mixture of Block Attention** (`core/attention/moba.py`): MoE routing applied
  directly to attention blocks (Moonshot AI, NeurIPS 2025). Routes attention computation
  sparsely to relevant blocks instead of full O(n²). Supports MLA compression, hard/soft
  routing, load balancing.
  Credits: Moonshot AI (NeurIPS 2025).

### Added — SSM Improvements

- **Mamba-3 SSD** (`core/ssm/mamba3.py`): Half the state size of Mamba-2 (d_state=32 vs 64)
  with comparable perplexity. Dual token shift (RWKV-inspired), inference-first dt
  discretization with clamped exponential for stability.
  Credits: arXiv:2603.15569 (Mamba-3, 2026).

- **Routing Mamba (RoM)** (`core/ssm/routing_mamba.py`): MoE routing over SSM linear
  projections (Microsoft Research, NeurIPS 2025). Multiple expert-specific B/C/dt with
  shared A matrix. DeepSeek-V3 style bias-based load balancing. Drop-in for Mamba2SSD.
  Credits: Microsoft Research (NeurIPS 2025).

### Added — MoE Improvements

- **S'MoRE** (`core/retrieval/smore.py`): Sub-tree MoE with Residual Experts from Meta
  (NeurIPS 2025). Composes experts from shared residual sub-trees for ~50% parameter
  savings vs standard MoE. Soft composition weights + expert-specific residual branch.
  Credits: Meta Research (NeurIPS 2025).

- **Symbolic-MoE** (`core/retrieval/symbolic_moe.py`): Skill-based discrete routing with
  two-stage approach: SkillClassifier → SymbolicRoutingRule. Maps skill types (REASONING,
  NARRATIVE, KNOWLEDGE, etc.) to pathway allocation weights. Can combine with BiasRouter.
  Credits: Symbolic-MoE (2025).

### Added — Training Improvements

- **LLM-JEPA** (`training/llm_jepa.py`): Joint-Embedding Predictive Architecture for LLMs.
  Predicts future latent states instead of next tokens. VICReg loss prevents collapse,
  EMA target encoder provides stable targets. Natural fit for SSM state transitions.
  Credits: LeCun (JEPA 2022), I-JEPA (Assran 2023), LLM-JEPA (2025).

### Changed

- Updated `config.py` with new sub-configurations: `RecurrentConfig`, `JEPAConfig`, and
  new fields in `SSMConfig`, `AttentionConfig`, `RetrievalConfig`.
- Updated `__init__.py` version to 0.6.0.
- Updated all `__init__.py` in core submodules to export v0.6 classes.
- Updated `CREDITS.md` with 8 new component references and additional research influences.

---

## [0.5.0] — 2026-05-02 — "KDA & Aux-Free"

### Added — Priority 1 Architecture Improvements

- **KDA+MLA Hybrid Attention** (`core/attention/kda_mla.py`): Key-Direction Attention
  combined with Multi-head Latent Attention for ~75% KV cache reduction.
- **Aux-Loss-Free MoE + MTP** (`core/retrieval/aux_free_moe.py`): DeepSeek-V3 style
  bias-based load balancing with Multi-Token Prediction heads.
- **Path-Lock Expert** (`core/reasoning/path_lock_expert.py`): Architectural reasoning
  control with zero additional FLOPs.

### Added — Priority 2 Efficiency Improvements

- **PoST Decay Spectra** (`core/ssm/post_decay.py`): Position-dependent decay spectrum
  with multiple decay modes per head.
- **HyLo Upcycling** (`utils/upcycling.py`): Dense-to-MoE checkpoint conversion.
- **Mirror Speculative Decoding** (`core/output/mirror_speculative.py`): SSM pathway as
  draft model for speculative decoding.
- **ETR Entropy Trend Reward** (`training/etr_reward.py`): Rewards efficient thinking
  token usage during GRPO training.

### Added — Priority 3 Training Improvements

- **Generation-Focused Distillation** (`training/gen_distillation.py`): KL + sequence-level
  + hidden state matching distillation.
- **TACO** (`training/compute_aligned.py`): Training with Compute Alignment.
- **BitDistill** (`core/quantization/bit_distill.py`): Joint quantization + distillation.
- **Attention-Preferred LoRA** (`core/elastic/attn_lora.py`): Asymmetric LoRA ranks.
- **FG2-GDN** (`core/ssm/fg2_gdn.py`): Fine-Grained Gated DeltaNet.

---

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
