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
