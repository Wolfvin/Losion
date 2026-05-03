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

## Agent Layer — Losion v0.5.0+ (Based on 40+ Research Papers)

### A1. Signal Extraction — Tool Use & Confidence Routing (`losion/agent/signals.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Schick, T. et al. (Meta AI) | "Toolformer: Language Models Can Teach Themselves to Use Tools" | 2023 | Self-supervised tool use learning — perplexity-based filtering for when to call APIs |
| Patil, S. et al. (UC Berkeley) | "Gorilla: Connected to Massive APIs" | 2023 | Fine-tuning LLMs for API calls with AST-based evaluation and retrieval-augmented training |
| Qin, Y. et al. (Tsinghua) | "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs" | 2023 | DFSDT (Depth-First Search Decision Tree) for tool exploration with backtracking |
| — | "SMART: Self-Aware Agent for Tool Overuse Mitigation" | 2025 | Self-awareness mechanism to prevent unnecessary tool calls when parametric knowledge is sufficient |
| — | "Paradigm Routing as Inference-Time Optimization" | 2024 | Different reasoning paradigms should be selected per-task by a learned router |

**What we adapted:** Multi-signal fusion (confidence + routing weights + thinking mode + task type) for agent signal extraction. Added SMART-style knowledge sufficiency check using Tri-Jalur routing weights. Added Toolformer-style perplexity-based confidence estimation. Added paradigm routing hints for the orchestrator.

---

### A2. Orchestrator — Agentic Frameworks (`losion/agent/orchestrator.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Yao, S. et al. (Princeton) | "ReAct: Synergizing Reasoning and Acting in Language Models" | 2023 | Interleaving reasoning (Chain-of-Thought) with acting (tool use) in a single loop |
| Shen, Y. et al. (Zhejiang) | "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace" | 2023 | LLM as orchestrator: Plan → Route → Execute → Synthesize pipeline |
| Shinn, N. et al. (Northeastern) | "Reflexion: Language Agents with Verbal Reinforcement Learning" | 2023 | Agents learn from failures via self-reflection stored in memory |
| Chen, W. et al. (Tsinghua) | "AgentVerse: Facilitating Multi-Agent Collaboration" | 2023 | Dynamic agent recruitment based on task requirements |

**What we adapted:** ReAct-style interleaved agent loop with HuggingGPT-style pipeline separation. Reflexion-inspired reflection after each action. Orchestrator NEVER modifies the model — only uses model signals and feeds action results back as context.

---

### A3. Paradigm Router — Reasoning Paradigm Selection (`losion/agent/planning/paradigm_router.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| — | "SMART: Self-Aware Agent for Tool Overuse Mitigation" | 2025 | Knowledge sufficiency check — prevents tool overuse when parametric knowledge is sufficient |
| — | "Paradigm Routing as Inference-Time Optimization" | 2024 | Per-task paradigm selection (Direct, CoT, ReAct, RAG, Multi-Agent) by learned router |
| Losion Tri-Jalur Router | Losion v0.4 | 2024 | Model-level routing between SSM, Attention, and Retrieval — agent-level extension |

**What we adapted:** Five reasoning paradigms (Direct, CoT, ReAct, RAG, MCTS) with SMART-style knowledge sufficiency check. Uses Tri-Jalur routing weights to determine if parametric knowledge is sufficient. Domain-aware and calibration-aware adjustments.

---

### A4. MCTS Agent Loop — Tree-Structured Action Exploration (`losion/agent/planning/mcts_agent.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Zhou, A. et al. | "LATS: Language Agent Tree Search" | ICML 2024 | Unifies reasoning, acting, and planning in single MCTS framework — LM self-evaluation as value function |
| Qin, Y. et al. (Tsinghua) | "ToolLLM: DFSDT" | 2023 | Depth-First Search Decision Tree — backtracking when tool paths fail |
| — | "ExACT: Reflective MCTS for Agent Decision-Making" | 2024 | Combines reflection with tree search for improved agent decisions |

**What we adapted:** LATS-style MCTS agent loop with full Select→Expand→Evaluate→Simulate→Backpropagate cycle. DFSDT-style backtracking when actions reduce confidence. UCB1 for exploration-exploitation balance. Confidence changes propagated as reward signals.

---

### A5. DEPS Planner — Failure Recovery (`losion/agent/planning/deps_planner.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Wang, Z. et al. (KAIST) | "DEPS: Describe, Explain, Plan, Select for Interactive Planning" | 2023, cited 447× | Structured recovery: Describe failure → Explain why → Plan alternatives → Select best |
| Zhu, X. et al. (Tsinghua) | "GITM: Ghost in the Minecraft" | 2023 | Sub-goal tree decomposition — high-level goals decomposed into tree of sub-goals |

**What we adapted:** Full DEPS pipeline: Describe → Explain → Plan → Select for structured failure recovery. Maps to Losion's existing components: SignalExtractor (Describe), ReflectionEngine (Explain), MCTS (Plan), Parallel Thinking (Select). Seven failure types with tailored recovery strategies and fallback chains.

---

### A6. Agentic Retriever — Multi-Round Retrieval (`losion/agent/retrieval/agentic_retriever.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| — | "CRP-RAG: CRP-based Reasoning Graph for RAG" | 2024, cited 36× | Reasoning graphs for complex query reasoning instead of single-query retrieval |
| — | "OPEN-RAG: Enhanced Retrieval-Augmented Reasoning" | 2024, cited 79× | Self-reflection on retrieval quality for better generation |
| — | "SR-RAG: Selective Retrieval RAG" | 2024 | Don't always retrieve — use parametric knowledge when sufficient |
| — | "AU-RAG: Agent-based Universal RAG" | 2024 | Dynamic search across diverse pools with descriptive metadata |

**What we adapted:** Multi-round retrieval with confidence-based query refinement: Initial search → Quality assessment → Query reformulation → Re-search → Synthesis. Five refinement strategies (add_context, rephrase, decompose, narrow, broaden). Quality scoring using result count, relevance, query coverage, and content richness.

---

### A7. Risk Simulator — Pre-Execution Safety Assessment (`losion/agent/safety/risk_simulator.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Ruan, J. et al. | "ToolEmu: Identifying Risks of LM Agents with an LM Emulator" | ICLR 2024 Spotlight, cited 326× | Use LM to emulate tool execution for scalable risk testing without executing dangerous tools |

**What we adapted:** ToolEmu-style three-layer risk assessment: (1) Static analysis with pattern matching for dangerous commands, (2) Dynamic simulation of predicted outcomes, (3) Experience-based assessment from episodic memory. Five risk levels (SAFE→CRITICAL) with approval routing.

---

### A8. Self-Reflection (`losion/agent/reflection.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Shinn, N. et al. (Northeastern) | "Reflexion: Language Agents with Verbal Reinforcement Learning" | 2023 | Agents learn from verbal feedback rather than parameter updates, storing reflections for future decisions |
| Madaan, A. et al. | "Self-Refine: Iterative Refinement with Self-Feedback" | 2023 | Iterative self-improvement through structured feedback loops |
| — | "ExACT: Reflective MCTS" | 2024 | Combines reflection with tree search |

**What we adapted:** Reflexion-inspired verbal feedback: after each action, evaluate outcome quality and generate structured reflection with lesson learned. Self-Refine-inspired strategy corrections when confidence drops. Six reflection types including action success/failure, strategy correction, tool trust updates, skill refinement, and confidence recalibration.

---

### A9. Adaptive Calibration (`losion/agent/calibration.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| — | "ATTC: Adaptive Tool Trust Calibration for LLMs" | 2026 | Guides models to adaptively choose between tools vs. answering directly based on confidence scoring |
| — | "Alignment for Efficient Tool Calling of LLMs" | EMNLP 2025 | Gradual decline in tool usage as model accuracy increases |

**What we adapted:** Dynamic threshold calibration with three signals: domain profiles (7 domain-specific threshold sets), tool trust scores (EMA-based reliability tracking per tool per domain), and episodic experience. Successful actions lower thresholds (use more eagerly); failed actions raise them (use more cautiously).

---

### A10. Episodic Memory with Forgetting (`losion/agent/memory.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Park, J. et al. (Stanford/Google) | "Generative Agents: Interactive Simulacra of Human Behavior" | 2023, landmark | Multi-factor retrieval: recency × importance × relevance scoring for memory |
| Zhong, W. et al. | "MemoryBank: Enhancing Large Language Models with Long-Term Memory" | 2023, cited 790× | Ebbinghaus forgetting curve — memories decay over time, reinforced by access |
| — | "Synapse: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation" | 2024 | Spreading activation — related memories activated with decreasing strength |
| — | "MemP: Exploring Agent Procedural Memory" | 2025 | Procedural memory is separate from semantic memory |
| — | "A-MEM: Agentic Memory" | 2025 | Agent-native memory with structured attributes and LLM-curated knowledge |

**What we adapted:** Four-layer memory architecture (Working, Semantic/Engram, Episodic, Procedural/SkillStore). Ebbinghaus forgetting curve with access reinforcement. Multi-factor retrieval (recency × importance × relevance × effective_strength). Spreading activation for generalization. Periodic consolidation: merge similar episodes, discard weak ones.

---

### A11. Meta-Skill System (`losion/agent/meta_skills.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| — | "CASCADE: Cumulative Agentic Skill Creation through Autonomous Development and Evolution" | 2025 | Meta-skills: the ability to learn HOW to learn skills, not just individual skills |
| — | "SoK: Beyond Tool Use in LLM Agents — Agentic Skills" | 2026 | Skill abstraction layer distinct from tools with applicability, composability, and security |
| Wang, L. et al. (NVIDIA) | "Voyager: An Open-Ended Embodied Agent with Large Language Models" | 2023, cited 2,174× | Skill library storing reusable executable programs with iterative refinement |
| — | "CREATOR: Disentangling Abstract and Concrete Reasoning for Tool Creation" | 2023, cited 127× | Two-phase tool creation: abstract documentation first, then concrete implementation |
| Cai, T. et al. | "LATM: Large Language Models As Tool Makers" | 2023, cited 293× | Closed-loop: tool maker creates, tool user applies, tools persist across tasks |

**What we adapted:** Three meta-skills: (1) SkillSynthesis — multi-query search with cross-referencing and test case generation, (2) SkillVerification — Bayesian confidence updates from test results, (3) SkillComposition — decompose complex tasks into skill chains with compatibility checking. Voyager-style executable skills with preconditions/postconditions/error patterns.

---

### A12. Self-Improving Agents — Training Integration

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Chen, Z. et al. | "FireAct: Toward Language Agent Fine-tuning" | 2023, cited 218× | Fine-tuning on agent trajectories improves both agent performance and general LLM capabilities |
| Zeng, A. et al. (Tsinghua) | "AgentTuning: Enabling Generalized Agent Abilities for LLMs" | 2023 | Mixing agent trajectories with general instructions at ~50% ratio prevents catastrophic forgetting |

**What we adapted:** Episodic memory as source for agent fine-tuning data. AgentTuning-style data mixing: successful episodes + general training data at ~50% ratio. FireAct-inspired trajectory collection from agent interactions.

---

## v0.7 — "Integrated & Complete" Improvements

### 21. LosionModelV2 — Config-Driven Integration (`losion/models/losion_model_v2.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Su, J. et al. | "RoFormer: Enhanced Transformer with Rotary Position Embedding" (arXiv:2104.09864) | 2021 | Rotary position embeddings replacing learned position embeddings |
| Losion Framework | Wolfvin & Contributors (github.com/Wolfvin/Losion) | 2024-2026 | Config-driven module selection replacing Simplified* placeholders |
| DeepSeek-V2/V3 | arXiv:2405.04434, arXiv:2412.19437 | 2024 | MLA, aux-loss-free MoE integrated into production model |

**What we adapted:** Complete rewrite of the production model. RoPE replaces learned position embeddings. Config-driven factory pattern selects SSM/Attention/MoE modules based on config flags (e.g., `use_mamba3` → Mamba3SSD, `use_gated_attention` → GatedMultiHeadAttention). AdaptiveRouter replaces nn.Linear. MTP heads and JEPA loss integrated into forward pass.

---

### 22. KV Cache + Inference Optimization (`losion/inference/`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| vLLM Team | "Efficient Memory Management for Large Language Model Serving with PagedAttention" | 2023 | PagedAttention — virtual memory for KV cache with page tables |
| ChunkKV | NeurIPS 2025 (neurips.cc/virtual/2025/poster/120181) | 2025 | Semantic-preserving chunk-level KV cache compression |
| EvolKV | EMNLP Findings 2025 (arXiv:2509.08315) | 2025 | Adaptive layer-wise KV cache compression via evolutionary optimization |
| EAGLE-3 | Li et al., 2025 | 2025 | Multi-layer feature fusion speculative decoding |
| HuggingFace | generate() API | 2024 | Generation API design patterns |

**What we adapted:** Three-tier KV cache: standard (full K/V), MLA compressed (latent c_kv only), and paged (vLLM-style page allocation with prefix caching). ChunkKV + EvolKV compression. Full generation pipeline with temperature/top-k/top-p, speculative decoding (SSM as draft model), continuous batching, and streaming support.

---

### 23. Data Pipeline (`losion/data/`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| OpenAI | tiktoken tokenizer | 2023 | BPE tokenizer backend |
| Google | SentencePiece | 2018 | Language-independent tokenizer |
| Bit-level BPE | arXiv:2506.07541 | 2025 | Sub-byte tokenization for shorter sequences |
| FineWeb2 | ICLR 2025 (87 citations) | 2025 | Adaptive data curation pipeline |
| ADAPT | ICLR 2026 | 2026 | Online reweighting for data curation |
| MinHash LSH | Broder, A. (1997) | 1997 | Scalable near-duplicate detection |

**What we adapted:** Unified tokenizer wrapping tiktoken/sentencepiece with thinking tokens (<think_start>, <think_end>). Memory-mapped dataset with packed sequences. Data curation pipeline: quality filtering + MinHash LSH dedup + PII removal + domain mixing. Curriculum data loader with phase-aware difficulty progression.

---

### 24. Losion Training Recipe (`losion/training/losion_recipe.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| WSD Schedule | ICLR 2025 (46 citations, openreview.net/forum?id=m51BgoqvbP) | 2025 | Warmup-Stable-Decay LR schedule — high-quality intermediate models from any point |
| WSM | arXiv:2507.17634 | 2025 | Decay-free schedule via checkpoint merging (stochastic weight averaging) |
| DeepSeek Training | DeepSeek-V2/V3 Technical Reports | 2024 | Multi-stage training methodology |
| TACO | DeepSeek-V2 (2024) | 2024 | Training with Compute Alignment |
| ETR Reward | DeepSeekMath (2024), Gemini 2.5 (2025) | 2024-2025 | Entropy trend reward for efficient thinking tokens |

**What we adapted:** Complete Losion-specific 4-phase training methodology with per-phase hyperparameters, loss configurations, and data difficulty settings. WSD LR schedule with WSM weight averaging. Scaling recipes for 1B/7B/48B with pre-configured LosionConfig + TrainingRecipe.

---

### 25. Evaluation Framework (`losion/evaluation/benchmarks.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| EleutherAI | lm-eval-harness | 2023 | Standard LLM evaluation harness |
| DeepEval | confident-ai/deepeval | 2024 | Pytest-like LLM evaluation with specialized metrics |
| AutoEvoEval | arXiv:2506.23735 | 2025 | Evolution-based evaluation that finds model weaknesses |

**What we adapted:** Perplexity evaluator (full + sliding window). Benchmark evaluation for MMLU, GSM8K, HellaSwag, ARC. Routing behavior analyzer with utilization, specialization, entropy, and collapse detection metrics.

---

### 26. Safety & Alignment (`losion/safety/alignment.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Anthropic | "Constitutional AI: Harmlessness from AI Feedback" | 2022 | Constitutional AI with critique-revise training |
| R-CAI | arXiv:2604.17769 | 2026 | Reverse Constitutional AI for automated red teaming |
| AlphaDPO | ICML 2025 (icml.cc/virtual/2025/poster/45946) | 2025 | Adaptive reward margin in DPO |
| DRO | OpenReview 2025 (openreview.net/forum?id=5EqAAgBMWZ) | 2025 | Direct Reward Optimization without pairwise preferences |

**What we adapted:** 15 constitutional principles with auto-categorization. Safety classifier (binary safe/unsafe + multi-label toxicity/violence/hate/sexual/self-harm). Constitutional trainer with generate→evaluate→critique→revise loop. Red teamer with R-CAI adversarial prompt generation.

---

### 27. Distributed Training (`losion/distributed/parallel.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| PyTorch FSDP2 | docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html | 2025 | Fully Sharded Data Parallel v2 with implicit prefetching |
| WLB-LLM | OSDI 2025 (arXiv:2503.17924, 24 citations) | 2025 | Workload-balanced 4D parallelism for LLM training |
| AutoSP | arXiv:2604.27089 | 2026 | Compiler-based automated sequence parallelism |
| DeepSpeed Ulysses | emergentmind.com/topics/deepspeed-ulysses | 2025 | Scalable sequence and head parallelism |

**What we adapted:** 4D parallelism (DP+TP+PP+CP) with configurable FSDP sharding. Context parallelism with ring-style attention communication and sequential SSM state propagation. Expert parallelism for MoE layers. Distributed training loop with gradient accumulation and mixed precision.

---

### 28. Long Context Extension (`losion/core/attention/context_extension.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| YaRN | arXiv:2309.00071 | 2024 | Yet Another RoPE Extension — NTK-by-parts interpolation |
| NTK-Aware Scaling | 2023-2025, widely adopted | 2023 | Dimension-wise rotary frequency adjustment for context extension |
| Scaling RNN State | ACL 2025 (aclanthology.org/2025.acl-long.564) | 2025 | Efficient scaling of RNN/SSM state size for longer contexts |

**What we adapted:** Four RoPE extension methods: YaRN (mixed interpolation + temperature), NTK-aware (base scaling), linear (frequency division), dynamic NTK (progressive base). SSM state extension via interpolation + zero-padding for longer SSM context windows.

---

## v0.8 — "Next-Gen Training & Infinite Experts" Improvements

### 29. DAPO — Decoupled Clip & Dynamic Sampling Policy Optimization (`losion/training/dapo.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Yu et al. | "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (arXiv 2503.14476) | 2025 | 4 improvements over GRPO: decoupled clip, dynamic sampling, token-level loss, overlong filtering |
| Shao, Z. et al. | "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" | 2024 | GRPO baseline that DAPO improves upon |

**What we adapted:** Complete DAPO implementation replacing GRPO as the default RL optimizer. Decoupled clip with separate low/high ratios (0.2/0.28) prevents both policy collapse and reward hacking. Dynamic sampling filters prompts with zero-variance rewards for ~15-20% efficiency gain. Token-level loss for finer credit assignment. Overlong filtering penalizes excessively long responses.

---

### 30. ∞-MoE — Infinite Mixture of Experts (`losion/core/retrieval/infinite_moe.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| arXiv 2601.17680 | "∞-MoE: Generalizing Mixture of Experts to Infinite Experts" | 2026 | Extends MoE from finite discrete set to continuous expert space via hypernetwork parameterization |

**What we adapted:** Continuous expert parameterization using ExpertCodeRouter (produces expert codes + routing logits in continuous space) + ContinuousExpertGenerator (hypernetwork that generates expert weights from codes). Shared base expert + code-conditioned scaling/bias/low-rank residual modifications. ExpertCodeClusterer for inference efficiency. Drop-in replacement for discrete MoE layers.

---

### 31. L-MTP — Leap Multi-Token Prediction (`losion/core/output/leap_mtp.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| arXiv 2505.17505, NeurIPS 2025 (poster 120311) | "L-MTP: Leap Multi-Token Prediction Beyond Adjacent" | 2025 | Extends MTP to predict tokens at arbitrary future positions (leaping), improving speculative decoding efficiency |

**What we adapted:** Geometric leap schedule (1, 2, 4, 8 steps) for 4 leap heads covering 8 positions (2x vs adjacent MTP's 4). Two-stage training: warm-up heads with frozen backbone, then joint fine-tuning. Geometric decay loss weights per leap distance. LeapSpeculativeDecoder with gap-filling via SSM pathway. Backward compatible: `ADJACENT` schedule = standard MTP.

---

### 32. Cross-Jalur Attention-MoE Routing (`losion/core/retrieval/cross_jalur_routing.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| arXiv 2505.00792 | "Improving Routing in Sparse MoE with Graph of Tokens" | 2025 | Uses attention matrix to guide MoE routing — tokens that attend to each other should route to similar experts |

**What we adapted:** Cross-jalur bridge between Attention (Jalur 2) and MoE (Jalur 3). AttentionGraphBuilder constructs sparse token affinity graph from attention weights. CrossJalurRouter performs graph convolution to propagate routing logits across attended tokens. RoutingSmoother blends original and attention-informed logits with learnable gate. Drop-in module compatible with any Losion MoE implementation.

---

### 33. RLVR — Reinforcement Learning with Verifiable Rewards (`losion/training/rlvr.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| NeurIPS 2025 (posters 119944, 116633) | Multiple RLVR papers | 2025 | Scales RL training using objective, programmable reward functions instead of learned reward models |
| arXiv 2601.05607 | RLVR theoretical framework | 2026 | Proves RLVR consistently improves policy performance over time |
| arXiv 2603.22117 | RLVR with verifiable math/code rewards | 2026 | Verifiable rewards for math and code tasks |

**What we adapted:** Multiple verifier types: MathVerifier (numeric + symbolic), CodeVerifier (sandboxed execution), FormatVerifier (regex + length + JSON), ExactMatchVerifier (exact/fuzzy matching). CompositeVerifier with curriculum difficulty scheduling (EASY→MEDIUM→HARD). Integrates with DAPO/GRPO as the reward function provider.

---

### 34. Expert Prefetching — Speculating Experts (`losion/inference/expert_prefetch.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| arXiv 2603.19289 | "Speculating Experts Accelerates Inference for MoE" | 2026 | Uses computed representations to predict which MoE experts are needed in subsequent layers, enabling prefetching |

**What we adapted:** LightweightPredictor (2-layer MLP, <1% parameter overhead) per layer that predicts expert routing for the next layer. Supports both finite MoE (discrete expert index prediction) and ∞-MoE (continuous code prediction with L2 distance matching). PrefetchAccuracyTracker with rolling-window precision/recall. Adaptive temperature scheduling based on recent recall. ExpertPrefetcher orchestrator for multi-layer coordination.

---

### 35. Losion Training Orchestrator (`losion/training/losion_orchestrator.py`)

| Reference | Paper | Year | Key Contribution |
|-----------|-------|------|-----------------|
| Losion Framework | Unified training methodology | 2026 | One-stop training orchestrator integrating ALL 13+ Losion training techniques |
| WSD Schedule | ICLR 2025 | 2025 | Warmup-Stable-Decay LR schedule |
| DAPO | arXiv 2503.14476 | 2025 | Decoupled clip policy optimization |
| RLVR | NeurIPS 2025 | 2025 | Verifiable rewards |
| LLM-JEPA | arXiv 2509.14252 | 2025 | Future state prediction |
| TACO | DeepSeek-V2 | 2024 | Compute-aligned training |
| ETR Reward | DeepSeekMath + Gemini 2.5 | 2024-2025 | Entropy trend reward |
| Gen. Distillation | Kim & Rush (2016), Agarwal et al. (2024) | 2016-2024 | Sequence-level distillation |
| BitDistill | Wang et al. (2024) | 2024 | Joint quantization + distillation |
| Curriculum Learning | Multiple sources | 2024-2025 | Phase-aware data difficulty |

**What we adapted:** Comprehensive training orchestrator that manages the complete 4-phase Losion training pipeline. Phase 1: WSD + JEPA + expert specialization. Phase 2: JEPA (reduced) + TACO + curriculum + active learning. Phase 3: DAPO/GRPO (auto-selected) + RLVR + ETR + TACO + evolutionary search. Phase 4: Gen distillation + BitDistill + ETR + early exit. Full checkpoint save/resume with all training state.

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
