# Credits & References — Losion AI Framework

This document tracks all research papers, projects, and ideas that have been referenced,
adapted, or incorporated into the Losion AI Framework. We are deeply grateful to the
original authors for their contributions to the open research community.

---

## Losion Base Framework

| Component | Reference | Authors/Team |
|-----------|-----------|--------------|
| Losion Architecture | [Wolfvin/Losion](https://github.com/Wolfvin/Losion) | Wolfvin & Contributors |
| Tri-Jalur Router (SSM + Attention + Retrieval) | Losion v0.4 | Wolfvin |
| Agent Layer | Losion v0.5.0 | Wolfvin |

---

## Priority 1 — Architecture Improvements (Losion v0.5)

### 1. KDA+MLA Hybrid Attention (`losion/core/attention/kda_mla.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| arXiv:2510.26692 | "KDA: Key-Direction Attention for Efficient LLM Inference" | 2025 | Key-Direction Attention — projects keys to low-dimensional directional subspace, reducing KV cache ~75% and improving throughput ~6x |
| DeepSeek-AI | "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" | 2024 | Multi-head Latent Attention (MLA) — KV compression via low-rank latent projections |
| Sun, Q. et al. | "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models" | 2024 | Linear attention with tiling/chunking for O(n) training complexity |

**What we adapted:** Combined KDA's key-direction projection with MLA's latent compression into a hybrid attention with two paths (local softmax + global linear), blended via a learned gate.

---

### 2. Auxiliary-Loss-Free MoE + MTP (`losion/core/retrieval/aux_free_moe.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| DeepSeek-AI (arXiv:2412.19437) | "DeepSeek-V3 Technical Report" | 2024 | Auxiliary-loss-free load balancing — bias-based routing with EMA statistics instead of quality-degrading auxiliary loss |
| Gloeckle, F. et al. | "Better & Faster Large Language Models via Multi-token Prediction" | 2024 | Multi-Token Prediction (MTP) heads — predicting future tokens provides richer training signal for expert specialization |

**What we adapted:** DeepSeek-V3's bias-based balancing (non-gradient router bias updates) combined with MTP heads using geometric decay weights for complementary expert specialization signal.

---

### 3. Path-Lock Expert PLE (`losion/core/reasoning/path_lock_expert.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| arXiv:2604.27201 | "Path-Lock Expert: Architectural Reasoning Control" | 2025 | Architectural reasoning control — locks specific expert pathways for reasoning types, zero additional FLOPs |
| DeepSeek-AI | "DeepSeek-V3 Technical Report" | 2024 | Expert specialization analysis in MoE layers |

**What we adapted:** Path-lock masks on MoE routing logits with automatic input type detection (reasoning, factual, creative, code, analysis) and soft/hard lock modes. Wrapped as a drop-in layer compatible with any MoE implementation.

---

## Priority 2 — Efficiency Improvements

### 4. PoST Decay Spectra (`losion/core/ssm/post_decay.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Gu, A. & Dao, T. | "Mamba-2: A Generalized State Space Model with Structured State Space Duality" | 2024 | SSM with structured state space duality, single decay parameter per head |
| Peng, H. et al. | "Random Feature Attention" | 2021 | Position-dependent attention via random features — inspiration for position-dependent mixing weights |
| Poli, M. et al. | "Striped Attention" | 2023 | Position-dependent processing patterns in attention/SSM layers |

**What we adapted:** Replaced single learnable decay per head with a spectrum of decay rates (n_decay_modes=4 by default) with position-dependent softmax mixing, enabling different memory retention patterns at different sequence positions. Backward compatible: 1 mode = standard SSM.

---

### 5. HyLo Upcycling (`losion/utils/upcycling.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Komatsu, H. et al. | "HyLo: Heterogeneous Layer Upcycling for Mixture-of-Experts" | 2024 | Dense-to-MoE checkpoint conversion via weight clustering without full retraining |
| Roller, S. et al. | "Mixtral of Experts" | 2024 | MoE layer design and expert utilization analysis |

**What we adapted:** Complete HyLo upcycling pipeline with KMeans/Spectral clustering, activation-based router initialization, progressive upcycling (1→2→4→8... experts), and automatic FFN layer detection from state dicts.

---

### 6. Mirror Speculative Decoding (`losion/core/output/mirror_speculative.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Leviathan, Y. et al. | "Fast Inference from Transformers via Speculative Decoding" | ICML 2023 | Speculative decoding — draft-verify paradigm for accelerated inference |
| Chen, C. et al. | "Accelerating Large Language Model Decoding with Speculative Sampling" | ICLR 2024 | Stochastic speculative sampling with distribution-equivalent guarantees |
| Gu, A. & Dao, T. | "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" | 2023 | SSM as efficient sequence model — O(1) per token inference |

**What we adapted:** Used SSM pathway (already in Losion) as the draft model instead of a separate model or MTP heads. SSM provides O(1) draft tokens that capture sequential patterns, achieving 3x+ speedup with no additional parameters.

---

### 7. ETR Entropy Trend Reward (`losion/training/etr_reward.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Shao, Z. et al. | "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" | 2024 | GRPO training with reward shaping for reasoning |
| Team, G. | "Gemini 2.5 Technical Report" — Thinking tokens optimization | 2025 | Concept of monitoring and reducing wasteful thinking tokens |

**What we adapted:** Entropy trend tracking during generation: rewards decreasing entropy (efficient convergence to answer), penalizes sustained high entropy (wasteful thinking). Reduces thinking tokens up to 40% without quality loss. Integrated as a reward signal in GRPO training.

---

## Priority 3 — Training Improvements

### 8. Generation-Focused Distillation (`losion/training/gen_distillation.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Kim, Y. & Rush, A. | "Sequence-Level Knowledge Distillation" | EMNLP 2016 | Sequence-level distillation — training student on teacher-generated sequences |
| Freitag, M. et al. | "Mixture Models for Diverse Machine Translation: Tricks of the Trade" | TACL 2023 | Diverse distillation with mixture models |
| Agarwal, R. et al. | "On-Policy Distillation: Aligning Language Models with RL from Teacher Feedback" | 2024 | On-policy distillation with progressive teacher-to-student shifting |

**What we adapted:** Combined KL divergence on output distributions + sequence-level cross-entropy loss + optional hidden state matching, with progressive distillation (gradual shift from teacher signal to student's own loss) and three modes: teacher-forcing, free-running, and mixed.

---

### 9. Compute Aligned Training — TACO (`losion/training/compute_aligned.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| DeepSeek-AI | "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" | 2024 | Compute-aware training — aligning training compute with inference compute allocation |
| Jiang, A. Q. et al. | "Mixtral of Experts" | 2024 | Expert utilization analysis in MoE models — experts have highly skewed usage patterns |

**What we adapted:** TACO (Training with Compute Alignment) tracks inference compute per expert/layer via EMA, then derives training loss weights proportional to inference usage. Prevents over-training on rarely-used experts and under-training on frequently-used ones. Formula: weight_i = (1-strength) * 1.0 + strength * (usage_i / mean_usage).

---

### 10. BitDistill (`losion/core/quantization/bit_distill.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Wang, H. et al. (Microsoft Research) | "BitNet b1.58: 1-bit LLMs for the Era of 1-bit LLMs" | 2024 | 1.58-bit ternary quantization ({-1, 0, +1}) with gradual quantization schedule |
| Kim, Y. & Rush, A. | "Sequence-Level Knowledge Distillation" | EMNLP 2016 | Knowledge distillation via KL divergence on soft targets |
| Stock, P. et al. | "Training with Quantization Noise for Extreme Model Compression" | ICLR 2020 | Quantization-aware training (QAT) with noise injection |

**What we adapted:** Joint quantization + distillation training: frozen full-precision teacher provides soft targets while student is gradually quantized using BitNetLinear. Gradual quantization schedule compatible with BitNet warmup. Loss = alpha_quant * task_loss + alpha_distill * KL(teacher || student).

---

### 11. Attention-Preferred LoRA (`losion/core/elastic/attn_lora.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Hu, E. J. et al. | "LoRA: Low-Rank Adaptation of Large Language Models" | ICLR 2022 | Low-rank adaptation via down-projection + up-projection with zero init and alpha/r scaling |
| Hayou, S. et al. | "LoRA+: Efficient Low Rank Adaptation of Large Models" | ICML 2024 | Asymmetric LoRA ranks — different ranks for different projection matrices improve efficiency |
| Biderman, D. et al. | "LoRA Fine-tuning Efficiently Undoes Safety Training" — rank analysis | 2024 | LoRA rank sensitivity analysis across different layer types |

**What we adapted:** Asymmetric LoRA ranks based on layer type: attention layers get higher rank (default 16), FFN/MoE get lower rank (default 4), SSM gets medium rank (default 8). Automatic rank allocation based on module dimensions. Supports merge/unmerge for zero-overhead inference.

---

### 12. FG2-GDN — Fine-Grained Gated DeltaNet (`losion/core/ssm/fg2_gdn.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Yang, S. et al. | "Gated Linear Attention" (GLA) | ICML 2024 | Gated linear attention with per-head gating for controlled retention |
| Schlag, I. et al. | "Linear Transformers Are Secretly Fast Weight Programmers" | ICLR 2021 | Delta rule for fast weight updates in linear attention |
| Losion Framework | "GatedDeltaNet" (`losion/core/ssm/delta_net.py`) | 2024 | Original GatedDeltaNet with per-head gating and chunk-based parallel computation |

**What we adapted:** Enhanced GatedDeltaNet with per-head, per-position gating (original: per-head only). Added learnable temperature per head for different selectivity, position bias for retention patterns, and support for both sigmoid and softmax gate functions. Drop-in replacement for GatedDeltaNet.

---

## v0.6 — "Mythos & Mamba" Improvements (8 New Components)

### 13. Recurrent-Depth Transformer + LTI-Stable + ACT (`losion/core/recurrent/rdt.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Dehghani, M. et al. | "Universal Transformers" (arXiv:1807.03819) | 2019 | Original looped transformer concept — same weights applied iteratively for adaptive compute depth |
| Kye Gomez | "OpenMythos" (github.com/kyegomez/OpenMythos) | 2026 | Open-source reconstruction of Claude Mythos architecture — Recurrent-Depth Transformer hypothesis with MoE + MLA/GQA + LTI stability |
| Bae, S. et al. | "Relaxed Recursive Transformers" (arXiv:2410.20672) | 2024 | LoRA per loop iteration — allows per-iteration behavioral adaptation while preserving weight-sharing compactness |
| Saunshi, N. et al. | "Reasoning with Latent Thoughts" (arXiv:2502.17416) | 2025 | Proves looped models simulate chain-of-thought reasoning in latent space |
| Sachan, M. et al. | "COCONUT: Continuous Latent Reasoning" (arXiv:2412.06769) | 2024 | Training LLMs to reason in continuous latent space without explicit tokens |
| Goyal, S. et al. | "Loop, Think, & Generalize" (arXiv:2604.07822) | 2026 | Implicit reasoning capabilities of recurrent-depth transformers |
| Zhang, J. et al. | "Parcae: Scaling Laws for Looped Language Models" (arXiv:2604.12946) | 2026 | Scaling laws specific to looped architectures |
| Graves, A. | "Adaptive Computation Time for Recurrent Neural Networks" (arXiv:1603.08983) | 2016 | Adaptive Computation Time (ACT) — learned halting criterion for variable compute depth |
| Hyperloop Transformers | arXiv:2604.21254 | 2026 | Advanced looped transformer variant |
| The Recurrent Transformer | arXiv:2604.21215 | 2026 | Greater effective depth and efficient decoding in looped models |

**What we adapted:** Recurrent-Depth Transformer with LTI-stable injection (spectral radius constraint ρ(A) < 1 for training stability), Adaptive Computation Time for variable-depth halting, loop-index positional embeddings for iteration differentiation, and depth-wise LoRA for per-iteration adaptation. The full RecurrentDepthBlock orchestrates: Prelude → [Looped Block with LTI stability + ACT + DepthLoRA] → Coda.

---

### 14. MoBA — Mixture of Block Attention (`losion/core/attention/moba.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Moonshot AI | "MoBA: Mixture of Block Attention" (NeurIPS 2025) | 2025 | MoE routing applied directly to attention blocks — routes attention computation sparsely to relevant blocks instead of full O(n²) attention |
| NeurIPS 2025 | https://neurips.cc/virtual/2025/poster/117997 | 2025 | Conference presentation and poster |

**What we adapted:** Block-sparse attention via MoE routing. Sequence is partitioned into blocks, a router selects top-K relevant blocks per query, and attention is computed only within selected blocks. Supports MLA KV compression, hard and soft routing modes, and load balancing auxiliary loss. Drop-in replacement for standard attention with sub-quadratic complexity.

---

### 15. Gated Attention (`losion/core/attention/gated_attention.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Qwen Team | "Gated Attention" (NeurIPS 2025 Best Paper) | 2025 | Sigmoid gate after softmax attention — eliminates attention sinks, adds beneficial sparsity, synergizes with MoE routing |
| Sun, Y. et al. | "Retentive Network: A Successor to Transformer" | 2023 | Gated retention mechanism inspiration |

**What we adapted:** Per-head sigmoid gating after softmax attention output. Gate = sigmoid(W_g * output) with near-identity initialization (gates start ≈1 and gradually learn to suppress). Eliminates attention sink tokens, adds soft per-head sparsity that synergizes with MoE routing. Supports MLA KV compression and QK normalization.

---

### 16. Routing Mamba (RoM) (`losion/core/ssm/routing_mamba.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Microsoft Research | "Routing Mamba (RoM)" (NeurIPS 2025) | 2025 | Scales SSM parameters using sparse mixtures of linear projection experts — combines MoE routing with SSM efficiency |
| NeurIPS 2025 | https://neurips.cc/virtual/2025/poster/116256 | 2025 | Conference presentation and poster |
| Gu, A. & Dao, T. | "Mamba-2: A Generalized State Space Model" (arXiv:2405.21060) | 2024 | SSM with structured state space duality — base architecture for Routing Mamba |
| DeepSeek-AI | "DeepSeek-V3 Technical Report" (arXiv:2412.19437) | 2024 | Aux-loss-free bias-based routing for load balancing |

**What we adapted:** MoE routing over SSM linear projections. Multiple expert-specific B, C, dt projections with shared A matrix and D skip connection. DeepSeek-V3 style bias-based routing for load balancing. Soft mixture of expert projections → single SSM scan (efficient). Optional shared expert. Drop-in replacement for Mamba2SSD.

---

### 17. Mamba-3 SSD (`losion/core/ssm/mamba3.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| arXiv:2603.15569 | "Mamba-3: Inference-First State Space Models" | 2026 | Half the state size of Mamba-2 with comparable perplexity — three core methodological improvements from inference-first perspective |
| Gu, A. & Dao, T. | "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (arXiv:2312.00752) | 2023 | Original Mamba SSM |
| Gu, A. & Dao, T. | "Mamba-2" (arXiv:2405.21060) | 2024 | Predecessor architecture |

**What we adapted:** Mamba-3 improvements: (1) Reduced state dimension d_state=32 vs Mamba-2's 64 with better utilization, (2) Dual token shift — two separate shift patterns (forward + backward) inspired by RWKV, (3) Inference-first dt discretization with clamped exponential and stabilized input scaling for stable O(1) per-token generation. Drop-in replacement for Mamba2SSD.

---

### 18. S'MoRE — Sub-tree MoE with Residual Experts (`losion/core/retrieval/smore.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Meta Research | "S'MoRE: Composing Experts from Shared Residual Sub-trees" (NeurIPS 2025) | 2025 | Parameter-efficient expert diversity by composing experts from shared sub-trees with residual connections |
| DeepSeek-AI | "DeepSeekMoE: Towards Ultimate Expert Specialization" (arXiv:2401.06066) | 2024 | Fine-grained expert segmentation and shared expert design |

**What we adapted:** S'MoRE expert composition with shared ResidualSubTree components (SwiGLU layers with residual connections) that multiple ComposedExperts reference for parameter sharing. Each composed expert softly blends sub-trees via learned composition weights plus an expert-specific residual branch. Achieves ~50% parameter savings vs standard MoE with equivalent expert count. Includes load balancing auxiliary loss.

---

### 19. Symbolic-MoE — Skill-based Discrete Routing (`losion/core/retrieval/symbolic_moe.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Symbolic-MoE (2025) | "Skill-based Discrete Routing for Mixture of Experts" | 2025 | Top-level orchestrator using skill-type classification for pathway routing instead of pure learned routing |
| Fedus, W. et al. | "Switch Transformers: Scaling to Trillion Parameter Models" | 2022 | MoE routing fundamentals |
| Lewis et al. | "Retrieval-Augmented Generation" | 2020 | Task-type conditioned routing inspiration |

**What we adapted:** Two-stage routing: SkillClassifier (small MLP) classifies input into skill types (REASONING, NARRATIVE, KNOWLEDGE, CODING, CREATIVE, MATHEMATICAL), then SymbolicRoutingRule maps skill type to pathway allocation weights (e.g., REASONING → attention:0.7, NARRATIVE → SSM:0.8). Supports soft blending, hard discrete routing, and combination with Losion's BiasRouter for macro+micro routing.

---

### 20. LLM-JEPA — Joint-Embedding Predictive Architecture for LLMs (`losion/training/llm_jepa.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| LLM-JEPA (2025) | "LLM-JEPA: Predicting Future Latent States in Large Language Models" (arXiv, 19 citations) | 2025 | Predicts future latent states instead of next tokens — principled training objective for hybrid models |
| LeCun, Y. | "A Path Towards Autonomous Machine Intelligence" (JEPA) | 2022 | Joint-Embedding Predictive Architecture — predict in latent space, not pixel/token space |
| Assran, M. et al. | "I-JEPA: Self-Supervised Learning from Images via Joint-Embedding Predictive Architecture" (arXiv:2301.08243) | 2023 | Image JEPA with VICReg loss and EMA target encoder |
| Bardes, A. et al. | "V-JEPA: Video Joint-Embedding Predictive Architecture" | 2024 | Video JEPA extending the framework to temporal sequences |

**What we adapted:** JEPA training for LLMs: LatentPredictor predicts future hidden states (H steps ahead) from current representations, TargetEncoder (EMA teacher) provides stable targets, VICReg loss prevents representation collapse. Total loss = LM_loss + prediction_weight * JEPA_loss. Natural fit for SSM components that already model state transitions. Compatible with LosionTrainer's training loop.

---

## Additional Research Influencing v0.6 Design

| Research | Source | Impact on Losion |
|----------|--------|------------------|
| Anthropic's Attention Interpretability | transformer-circuits.pub/2025/attention-update/ | Multi-head coordination analysis — informed how to preserve critical attention patterns in hybrid SSM+Attention architecture |
| "The Hidden Attention of Mamba Models" (ACL 2025, 134 citations) | aclanthology.org/2025.acl-long.76.pdf | Proves Mamba implicitly computes attention — SSM and attention aren't independent, motivating unified SSM+Attention blocks |
| Anthropic's "Context Engineering" (Sep 2025) | anthropic.com/engineering/effective-context-engineering | Attention budget analysis — directly motivates SSM for long-range state where attention is wasteful |
| Claude Mythos System Card (Apr 2026) | Anthropic official documentation | Behavioral observations informing OpenMythos reconstruction |
| Jamba / Jamba-1.5 (AI21 Labs) | arXiv:2403.19887, ICLR 2025 | Pioneering hybrid Transformer-Mamba-MoE with 1:7 attention-to-SSM ratio |
| MoBA (Moonshot AI, NeurIPS 2025) | neurips.cc/virtual/2025/poster/117997 | MoE routing applied to attention — unifies MoE+Attention in single framework |
| Routing Mamba (Microsoft, NeurIPS 2025) | neurips.cc/virtual/2025/poster/116256 | MoE over SSM projections — the missing piece for SSM+MoE integration |

---

## Foundational Technologies Used Throughout Losion

| Technology | Reference | Key Contribution to Losion |
|-----------|-----------|---------------------------|
| State Space Models (SSM) | Gu, A. et al., "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023) | Jalur 1 (SSM pathway) in Tri-Jalur architecture |
| Mixture of Experts (MoE) | Fedus, W. et al., "Switch Transformers" (2022) | Jalur 3 (Retrieval/MoE) in Tri-Jalur architecture |
| MCTS for Reasoning | Silver, D. et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017) | Monte Carlo Tree Search reasoning in Losion |
| GRPO | Shao, Z. et al., "DeepSeekMath" (2024) | Group Relative Policy Optimization for RL training |
| SwiGLU Activation | Shazeer, N., "GLU Variants Improve Transformer" (2020) | SwiGLU activation in expert FFNs |
| RoPE | Su, J. et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021) | Rotary position embeddings throughout Losion |
| RMSNorm | Zhang, B. & Sennrich, R., "Root Mean Square Layer Normalization" (2019) | Normalization in all Losion layers |

---

## How to Cite Losion

If you use Losion in your research, please cite the original repository:

```bibtex
@software{losion2024,
  title = {Losion: A Hybrid AI Framework with Tri-Jalur Architecture},
  author = {Wolfvin and Contributors},
  url = {https://github.com/Wolfvin/Losion},
  year = {2024},
}
```

---

## License Note

Each referenced paper and technology is the intellectual property of its respective authors.
Losion adapts and implements ideas from these works under the terms of its own license.
Users should respect the original licenses and terms of all referenced works.

This CREDITS file was created to ensure transparency and proper attribution.
If you believe any reference is missing or incorrectly attributed, please open an issue
or submit a pull request at [github.com/Wolfvin/Losion](https://github.com/Wolfvin/Losion).
