# Losion Training Guide

> The definitive reference for training models with the Losion Tri-Jalur (Three-Pathway)
> Router framework. Covers pre-training through advanced RLHF, memory optimization,
> data pipelines, distributed training, and hyperparameter tuning.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Pre-Training](#2-pre-training)
3. [Training 4-Fase](#3-training-4-fase)
4. [Curriculum Learning](#4-curriculum-learning)
5. [GRPO Deep-Dive](#5-grpo-deep-dive)
6. [Advanced RLHF](#6-advanced-rlhf)
7. [DAPO — Decoupled Clip & Dynamic Sampling (v0.8)](#7-dapo--decoupled-clip--dynamic-sampling-v08)
8. [RLVR — Reinforcement Learning with Verifiable Rewards (v0.8)](#8-rlvr--reinforcement-learning-with-verifiable-rewards-v08)
9. [LLM-JEPA — Joint-Embedding Predictive Architecture (v0.6)](#9-llm-jepa--joint-embedding-predictive-architecture-v06)
10. [Losion Training Orchestrator (v0.8)](#10-losion-training-orchestrator-v08)
11. [Advanced Training Techniques](#11-advanced-training-techniques)
12. [Memory Optimization](#12-memory-optimization)
13. [Data Pipeline](#13-data-pipeline)
14. [Hyperparameter Reference](#14-hyperparameter-reference)
15. [Distributed Training](#15-distributed-training)
16. [Monitoring](#16-monitoring)
17. [Common Issues & Solutions](#17-common-issues--solutions)
18. [Evaluation & Benchmarks](#18-evaluation--benchmarks)

---

## 1. Quick Start

### Minimal Training Commands

```bash
# Single GPU — Losion-1B (fits on RTX 4090 / A10G)
python scripts/train.py --config configs/losion-1b.yaml

# Custom data directory
python scripts/train.py --config configs/losion-1b.yaml --data_dir ./my-data

# Resume from checkpoint
python scripts/train.py --config configs/losion-1b.yaml --resume checkpoints/step-5000
```

### Multi-GPU Training

```bash
# 4 GPUs with DDP (Losion-7B)
torchrun --nproc_per_node=4 scripts/train.py --config configs/losion-7b.yaml

# 8 GPUs with FSDP (Losion-48B)
torchrun --nproc_per_node=8 scripts/train.py --config configs/losion-48b.yaml

# Multi-node (2 nodes × 8 GPUs)
torchrun --nnodes=2 --nproc_per_node=8 \
    --master_addr=10.0.0.1 --master_port=29500 \
    scripts/train.py --config configs/losion-48b.yaml
```

### AMD ROCm

```bash
# Single AMD GPU
HIP_VISIBLE_DEVICES=0 python scripts/train.py --config configs/losion-1b.yaml

# Multi-GPU AMD
HIP_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train.py --config configs/losion-7b.yaml
```

### Verify Training is Running

After launching, you should see output like:

```
===== Memulai Training Losion 4-Fase =====
Total steps: 100000
Device: cuda:0
Mixed precision: bf16
LosionForCausalLMV2: 924,312,576 total parameters, 924,312,576 trainable (100.0%)
Estimated VRAM needed: 18.2 GB
Step 10 | loss: 10.5432 | lr: 0.000030 | phase: phase_1_individual | steps_per_sec: 2.3
Step 20 | loss: 9.8765 | lr: 0.000060 | phase: phase_1_individual | steps_per_sec: 2.4
```

> **Agent Context**
> ```
> Key files: scripts/train.py, configs/losion-{1b,7b,48b}.yaml
> Entry point: LosionTrainer.train() in losion/training/trainer.py
> Default precision: bf16 (no scaler needed)
> First step: verify loss decreases within first 50 steps
> ```

---

## 2. Pre-Training

Pre-training is the foundation of the Losion model. Before the 4-Fase training loop
begins, you must prepare data, select a tokenizer, and size your training corpus
according to compute-optimal scaling laws.

### 2.1 Data Collection & Curation

**Goal**: Assemble a large, diverse, high-quality text corpus.

| Step | Action | Tooling |
|------|--------|---------|
| 1 | Collect raw text from web crawls, books, code repos, scientific papers | Common Crawl, GitHub dumps, arXiv |
| 2 | Remove personally identifiable information (PII) | Presidio, regex filters |
| 3 | De-duplicate at document and substring level | MinHash + LSH (0.7 threshold) |
| 4 | Filter toxic / low-quality content | classifier-based quality filters |
| 5 | Language identification & balancing | fasttext lid.176, target ≤70% English |
| 6 | Domain stratification — ensure code, math, multilingual, factual coverage | domain-tagged sampling |

**Quality filters (recommended thresholds)**:

```yaml
curation:
  min_doc_length: 200          # characters
  max_doc_length: 1000000      # characters
  perplexity_filter:
    enabled: true
    max_perplexity: 500        # reject docs above this
  repetition_filter:
    max_dup_line_ratio: 0.3    # reject if >30% duplicate lines
  language_filter:
    min_confidence: 0.8        # fasttext lid confidence
```

### 2.2 Tokenizer Selection

Losion supports BPE (Byte Pair Encoding) tokenizers with vocabulary sizes from 32K to 128K.

| Model Size | Recommended Vocab | Rationale |
|-----------|-------------------|-----------|
| 1B | 32K | Smaller vocab → more data per token, faster convergence |
| 7B | 128K | Larger vocab → better multilingual & code coverage |
| 48B | 128K | Same as 7B; parameter budget supports large embedding |

```python
from losion.data import Tokenizer

# Load a pre-trained tokenizer
tokenizer = Tokenizer.from_file("tokenizer-128k.json")

# Or train from scratch on your corpus
tokenizer = Tokenizer.train(
    files=["data/corpus-part-*.txt"],
    vocab_size=128256,
    model_type="BPE",
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    min_frequency=2,
)
```

**Tokenizer requirements for Losion**:
- Must include `<think_start>` and `<think_end>` special tokens for thinking mode.
- Must include `<|route_ssm|>`, `<|route_attn|>`, `<|route_ret|>` tokens (optional, for debugging).
- Recommended: add code-specific tokens (`\n`, `\t`, `    `) if using a 128K vocab.

### 2.3 Data Preprocessing Pipeline

```
Raw Data → Cleaning → Tokenization → Packing → DataLoader
```

1. **Cleaning**: Strip HTML, normalize whitespace, filter short docs (<200 chars).
2. **Tokenization**: Convert text to token IDs using BPE tokenizer.
3. **Packing**: Concatenate short sequences into fixed-length blocks for efficiency.
4. **DataLoader**: Batch, shuffle, prefetch with configurable workers.

```python
from losion.data import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    tokenizer_path="tokenizer-128k.json",
    max_seq_len=8192,
    pack_sequences=True,        # Concatenate short docs
    pack_separator="<s>",       # Document separator token
    num_workers=8,
    shuffle_buffer_size=100_000,
)

# Process raw JSONL files
dataset = pipeline.process(
    input_pattern="data/cleaned/*.jsonl",
    output_dir="data/tokenized/",
    overwrite=False,
)
```

### 2.4 Recommended Datasets per Phase

| Phase | Datasets | Approximate Tokens | Purpose |
|-------|----------|--------------------|---------|
| Phase 1 (0–30%) | RefinedWeb, The Pile, C4 | 500B–1T | General language modeling, base capabilities |
| Phase 2 (30–60%) | + RedPajama-v1, StarCoder, Wikipedia | +200B | Diverse quality, code, structured knowledge |
| Phase 3 (60–90%) | + GSM8K, MATH, CodeContests, OpenAssistant | +50B | Reasoning, math, instruction-following |
| Phase 4 (90–100%) | + Domain-specific, RLHF preference data | +10B | Specialized tasks, alignment |

**Dataset mix ratios for Phase 2 (recommended)**:

```yaml
data_mix:
  web_text: 0.50
  code: 0.20
  scientific: 0.10
  books: 0.10
  wikipedia: 0.05
  math: 0.05
```

### 2.5 Chinchilla-Optimal Data Sizing

**Key principle** (Hoffmann et al., 2022): Train compute-optimally with ~20 tokens per parameter.

For Losion's MoE architecture, use **active parameters**, not total parameters:

```python
from losion.training.advanced_memory_data import ChinchillaDataSizer

sizer = ChinchillaDataSizer(moe_active_experts=4, moe_total_experts=64)

# For a 48B total parameter model with ~30% MoE params
result = sizer.compute_optimal_dataset_size(
    total_params=48_000_000_000,
    moe_params_fraction=0.30,
)
print(f"Active params: {result['active_params']:,}")
print(f"Optimal tokens: {result['optimal_tokens']:,}")
print(f"Optimal dataset: {result['optimal_dataset_gb']:.1f} GB")
# Active params:  38,400,000,000
# Optimal tokens: 768,000,000,000 (768B)
# Optimal dataset: ~2,868 GB
```

### 2.6 Active Parameters vs Total Parameters (MoE)

In a Mixture-of-Experts model, only a fraction of parameters are active per token:

```
active_params = non_moe_params + (num_active_experts / num_total_experts) * moe_params
```

| Model | Total Params | Active Params | Active Ratio |
|-------|-------------|---------------|-------------|
| Losion-1B | 1.2B | 1.2B | 100% (16 experts, top-2 = 12.5% MoE active) |
| Losion-7B | 9.8B | 7.2B | ~73% |
| Losion-48B | 52B | 38B | ~73% |

> **Agent Context**
> ```
> Chinchilla scaling is per-jalur in Losion, not per-model.
> Use ChinchillaScaler in losion/training/advanced_backprop.py
> for per-pathway FLOP budget allocation.
> Key formula: C = 6 * N * D, optimal D/N ≈ 20
> MoE: always use active_params for data sizing
> ```

### 2.7 Token-to-Parameter Ratio Calculator

```python
from losion.training.advanced_memory_data import ChinchillaDataSizer

def calculate_required_data(total_params, moe_active=6, moe_total=64, moe_fraction=0.3):
    """Calculate Chinchilla-optimal dataset size for a Losion model."""
    sizer = ChinchillaDataSizer(moe_active, moe_total)
    result = sizer.compute_optimal_dataset_size(total_params, moe_fraction)
    ratio = result['optimal_tokens'] / result['active_params']
    return {
        "total_params_B": total_params / 1e9,
        "active_params_B": result['active_params'] / 1e9,
        "optimal_tokens_B": result['optimal_tokens'] / 1e9,
        "token_to_param_ratio": ratio,
        "dataset_size_GB": result['optimal_dataset_gb'],
        "sufficient": ratio >= 18,  # Minimum acceptable ratio
    }

# Example: Losion-7B
print(calculate_required_data(9_800_000_000, moe_active=4, moe_total=64))
# {'total_params_B': 9.8, 'active_params_B': 7.2, 'optimal_tokens_B': 144.0,
#  'token_to_param_ratio': 20.0, 'dataset_size_GB': 537.0, 'sufficient': True}
```

---

## 3. Training 4-Fase

Losion uses a purpose-built 4-phase training paradigm designed for the Tri-Jalur Router
architecture. Each phase has distinct objectives, frozen/unfrozen parameter sets,
learning rates, and data strategies.

### Phase 1: Individual Pre-Training (0–30% budget)

**Objective**: Train each pathway independently so it functions well on its own.

**Strategy**:
- Router is **frozen** (no gradient updates)
- Only one pathway is trained at a time (rotating: SSM → Attention → Retrieval)
- Embeddings and LM head are always unfrozen (shared across pathways)

**Parameter schedule**:

| Cycle | Target Pathway | Active Parameters | FLOP Share |
|-------|---------------|-------------------|-----------|
| 1 (0–10%) | SSM | embedding + SSM + lm_head (~30% total) | ~20% |
| 2 (10–20%) | Attention | embedding + Attention + lm_head (~35% total) | ~50% |
| 3 (20–30%) | Retrieval (MoE) | embedding + MoE + lm_head (~20% total) | ~30% |

**Why train pathways separately?** If all three pathways train simultaneously from
random initialization, the dominant pathway (usually Attention) can suppress gradient
signals to the others. Separate pre-training ensures each pathway develops independent
capability before learning to coordinate.

**Phase 1 configuration**:

```yaml
training:
  phase_1:
    router_frozen: true
    router_weights: [0.8, 0.1, 0.1]  # Heavy SSM bias initially
    learning_rate: 3e-4
    warmup_steps: 500
    grad_clip: 1.0
    data: "general_text"               # RefinedWeb, Pile
    seq_len: 512
```

**Automatic transition**: The `CurriculumScheduler` detects when the current
pathway's loss has plateaued and rotates to the next pathway.

```python
from losion.training.curriculum import CurriculumScheduler, TrainingPhase
from losion.config import LosionConfig

scheduler = CurriculumScheduler(LosionConfig(), total_steps=100000)
print(scheduler.get_current_target_pathway())  # "ssm"
print(scheduler.is_router_frozen())             # True
print(scheduler.get_learning_rate())            # 3e-4
```

### Phase 2: Joint Fine-Tuning (30–60% budget)

**Objective**: Train all three pathways to work together coherently.

**Strategy**:
- All three pathways **unfrozen**
- Router remains **frozen** — uses weights learned during Phase 1
- Bridge/merge mechanisms are trained to harmonize pathway outputs
- Learning rate reduced to 50% of Phase 1

**Critical details**:
- Tight gradient clipping (`max_grad_norm=0.5`) for stability
- Short warmup (200 steps) since parameters are already initialized
- Monitor routing weights — if one pathway dominates >90%, rebalancing needed
- Modality-Aware Loss Weighting (Gemini) can be activated here

```yaml
training:
  phase_2:
    router_frozen: true
    router_weights: [0.33, 0.33, 0.33]  # Equal weights
    learning_rate: 1.5e-4
    warmup_steps: 200
    grad_clip: 0.5
    data: "+code,+scientific"
    seq_len: 2048
    use_modality_weighting: true
```

### Phase 3: End-to-End RL (60–90% budget)

**Objective**: Optimize routing decisions and model quality using reinforcement learning.

**Strategy**:
- Router **unfrozen** — now learns optimal routing
- DAPO (Decoupled Clip & Dynamic Sampling Policy Optimization, v0.8+) replaces GRPO
  as the default RL optimizer. GRPO remains available as a fallback.
- Thinking toggle activated — model learns when to "think deeper"
- Auxiliary rewards: routing entropy, load balancing, coherence
- RLVR provides verifiable reward signals for math and code tasks

**DAPO details for Phase 3** (recommended, v0.8+):
- Decoupled clip: separate low/high ratios (0.2/0.28) prevent policy collapse and reward hacking
- Dynamic sampling: filters prompts with zero-variance rewards for ~15-20% efficiency gain
- Token-level loss for finer credit assignment
- Overlong filtering penalizes excessively long responses
- Group size: 8 samples per prompt
- KL penalty coefficient: 0.05

```yaml
training:
  phase_3:
    router_frozen: false
    learning_rate: 5e-5
    warmup_steps: 500
    grad_clip: 1.0
    use_dapo: true          # DAPO is now default (v0.8+)
    use_grpo: false         # Set true for legacy GRPO
    dapo:
      group_size: 8
      kl_coefficient: 0.05
      clip_ratio_low: 0.2
      clip_ratio_high: 0.28
      dynamic_sampling: true
      token_level_loss: true
      overlong_filter: true
      entropy_coefficient: 0.01
      max_new_tokens: 512
      temperature: 0.7
    data: "+math,+reasoning"
    seq_len: 8192
```

**Using GRPO programmatically**:

```python
from losion.training.grpo import GRPOTrainer, GRPOConfig
from losion.models.losion_model_v2 import LosionForCausalLMV2
from losion.config import LosionConfig

config = LosionConfig()
model = LosionForCausalLMV2(config)

grpo_config = GRPOConfig(
    group_size=8,
    clip_range=0.2,
    kl_coeff=0.05,
    entropy_coeff=0.01,
    reward_shaping="centered",
)

trainer = GRPOTrainer(model, config=grpo_config)

# Single GRPO step
metrics = trainer.train_step(prompts=input_ids, attention_mask=mask)
print(f"Policy loss: {metrics['policy_loss']:.4f}")
print(f"KL penalty: {metrics['kl_penalty']:.4f}")
print(f"Mean reward: {metrics['mean_reward']:.4f}")
```

### Phase 4: Advanced Optimization (90–100% budget)

**Objective**: Final optimization for maximum quality and efficiency.

**Strategy**:
- All parameters unfrozen
- Evoformer recycling enabled (2–3 iterations of refinement)
- Flow matching enabled (if configured, recommended for 48B+)
- Early exit training — model learns when it can stop processing early
- Learning rate very low (1/10 of Phase 3)
- Optional: knowledge distillation from a larger model

```yaml
training:
  phase_4:
    router_frozen: false
    learning_rate: 5e-6
    warmup_steps: 100
    grad_clip: 0.5
    use_grpo: true
    use_early_exit: true
    use_distillation: true
    use_evo_recycling: true
    use_flow_matching: true
    data: "+long_context,+rl_tasks"
    seq_len: 32768
```

**Phase 4 outputs**:
- Production-ready model
- Multiple checkpoints for quality/speed trade-offs
- Evaluated on standard benchmarks

> **Agent Context**
> ```
> Phase transitions are automatic via CurriculumScheduler.
> Override manually: scheduler.set_phase(TrainingPhase.PHASE_2_JOINT)
> Recreate optimizer after phase transition (parameter groups change).
> Phase boundaries: 30% / 60% / 90% of total_steps by default.
> ```

---

## 4. Curriculum Learning

Curriculum Learning in Losion controls not only phase transitions but also progressive
increases in sequence length, data complexity, and active pathway count.

### Curriculum Dimensions

| Dimension | Start | End | Strategy |
|-----------|-------|-----|----------|
| Sequence length | 512 tokens | Max seq len (32K–1M) | Doubling per phase |
| Data complexity | Simple text | Complex text + reasoning | Difficulty-based sampling |
| Batch size | Large | Moderate | Adjusted with sequence length |
| Learning rate | High | Low | Cosine decay |
| Active pathways | 1 | 3 | Progressive unfreeze |
| Thinking mode | Off | Triggered | Activated in Phase 3+ |

### Example Progression (Losion-7B, 300K steps)

```
Phase 1 (0–30K steps):
  Seq length: 512 → 2048
  Data: Wikipedia, Books (simple text)
  Active pathway: rotating (SSM → Attn → Retrieval)
  LR: 2e-4 → 1e-4 (cosine)
  Thinking: OFF

Phase 2 (30K–60K steps):
  Seq length: 2048 → 8192
  Data: + Code, + Scientific papers
  Active pathways: all 3 (joint)
  LR: 1e-4 → 5e-5 (cosine)
  Thinking: OFF

Phase 3 (60K–90K steps):
  Seq length: 8192 → 32768
  Data: + Math, + Reasoning tasks
  Router: unfrozen + GRPO
  LR: 5e-5 → 1e-5 (cosine)
  Thinking: TRIGGERED

Phase 4 (90K–100K steps):
  Seq length: 32768 → 131072
  Data: + Long-context tasks, + RL tasks
  All optimizations active
  LR: 1e-5 → 1e-6 (cosine)
  Thinking: TRIGGERED + Early Exit
```

### Customizing the Schedule

```python
from losion.training.curriculum import CurriculumScheduler, TrainingPhase, PhaseConfig
from losion.config import LosionConfig

config = LosionConfig()
scheduler = CurriculumScheduler(config, total_steps=300000)

# Override Phase 2 start (advance it earlier)
scheduler.phase_configs[TrainingPhase.PHASE_2_JOINT].learning_rate = 8e-5
scheduler.phase_configs[TrainingPhase.PHASE_2_JOINT].validation_threshold = 3.5

# Get schedule summary
for entry in scheduler.get_schedule_summary():
    print(f"{entry['phase']}: steps {entry['start_step']}-{entry['end_step']}, "
          f"LR={entry['learning_rate']:.2e}, GRPO={entry['use_grpo']}")
```

> **Agent Context**
> ```
> CurriculumScheduler is in losion/training/curriculum.py
> Phase transitions: step-based (primary), validation loss (secondary)
| Manual override: scheduler.set_phase(TrainingPhase.PHASE_3_RL)
> Track progress: scheduler.get_progress() → {"total_progress": 0.45, ...}
> ```

---

## 5. GRPO Deep-Dive

### 5.1 What is GRPO?

GRPO (Group Relative Policy Optimization) is a variant of PPO that uses **relative
comparisons within a group** rather than absolute reward values. This is more stable
for language model training because it eliminates the need for an accurate absolute
reward signal.

**Key advantages over PPO**:
- No value function required → saves ~50% parameters and memory
- Relative advantage is more stable → no baseline needed
- Group sampling → more accurate advantage estimation
- Clipping → prevents destructively large updates

### 5.2 Algorithm

```
1. For each prompt x:
   a. Generate G samples: {y_1, y_2, ..., y_G}
   b. Compute reward for each sample: {r_1, r_2, ..., r_G}
   c. Compute relative advantage:
      A_i = (r_i - mean(r)) / std(r)
   d. Update policy:
      L = -E[A_i * log(π(y_i|x))]
        + β * KL(π(·|x) || π_ref(·|x))
        - η * H(π(·|x))
```

### 5.3 GRPO for Losion — Routing-Specific Rewards

In Losion, GRPO optimizes **routing decisions** alongside output quality.

**Reward Components**:

| Component | Weight | Description |
|-----------|--------|-------------|
| Task accuracy | 0.50 | Is the output correct? |
| Routing entropy | 0.15 | Is the routing distribution balanced? |
| Compute efficiency | 0.15 | Is the routing efficient? |
| Load balance | 0.10 | Are all experts used evenly? |
| Coherence | 0.10 | Is the output coherent? |

**Routing-Specific Reward Implementation**:

```python
import math

def routing_entropy_reward(routing_weights):
    """Reward balanced routing distribution."""
    entropy = -sum(w * math.log(max(w, 1e-8)) for w in routing_weights if w > 0)
    max_entropy = math.log(3)  # 3 pathways
    normalized = entropy / max_entropy
    return 1.0 if 0.5 < normalized < 0.9 else 0.0

def load_balance_reward(expert_usage, total_tokens):
    """Reward even expert utilization (MoE)."""
    expert_load = expert_usage / total_tokens
    return -expert_load.std().item()  # Lower std = more balanced

def compute_efficiency_reward(routing_weights, pathway_costs):
    """Reward compute-efficient routing."""
    active_compute = sum(w * c for w, c in zip(routing_weights, pathway_costs))
    return -active_compute  # Less compute = better (within quality bounds)
```

### 5.4 GRPO Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `group_size` | 8 | 4–16 | Samples per prompt |
| `kl_coeff` | 0.05 | 0.01–0.2 | KL penalty (β) |
| `clip_range` | 0.2 | 0.1–0.3 | PPO-style clipping (ε) |
| `entropy_coeff` | 0.01 | 0.001–0.05 | Entropy bonus (η) |
| `reward_shaping` | "centered" | raw/centered/rank_based | Reward normalization |
| `policy_loss_type` | "clipped" | clipped/surrogate/unclipped | Loss formulation |
| `max_new_tokens` | 512 | 128–2048 | Generation length per sample |
| `temperature` | 0.7 | 0.3–1.2 | Sampling temperature |

> **Agent Context**
> ```
> GRPOTrainer class: losion/training/grpo.py
> Config: GRPOConfig dataclass
> Used in Phase 3 (routing optimization) and Phase 4 (full RL)
> Reference model is deep-copied at init for KL computation
> Advantage normalization is ON by default (use_advantage_normalization=True)
> ```

---

## 6. Advanced RLHF

Losion combines four techniques from DeepMind/Google AI for RLHF that is far more
effective than standard GRPO:

1. **Self-Play Preference Generation** (AlphaZero) — infinite preference data
2. **Value Head** (MuZero) — reduced variance advantage estimation
3. **Self-Consistency Verification** (Gemini Thinking) — internal reward signal
4. **Dirichlet Noise Injection** (AlphaZero) — exploration guarantee

### 6.1 Self-Play Preference Generation (AlphaZero)

The model generates 2+ candidates per prompt with different routing strategies,
then evaluates them itself. This creates infinite, curriculum-adaptive preference
data without human annotation.

```python
from losion.training.advanced_rlhf import SelfPlayPreferenceGenerator

generator = SelfPlayPreferenceGenerator(
    num_candidates=4,
    temperature_range=(0.3, 1.2),
    value_weight=0.5,
    consistency_weight=0.3,
    external_weight=0.2,
)

# Generate candidates for a batch of prompts
candidates = generator.generate_candidates(
    model, prompts=input_ids, max_new_tokens=256,
)
print(f"Generated {len(candidates['responses'])} candidates")
print(f"Strategies: {candidates['routing_strategies']}")

# Score candidates using value head + self-consistency + external reward
scores = generator.score_candidates(
    candidates,
    value_head=value_head,
    hidden_states=hidden_states,
    external_rewards=external_rewards,  # optional
)

# Generate preference pairs from scores
pairs = generator.generate_preference_pairs(scores)
```

### 6.2 Value Head (MuZero)

A policy-value dual head that predicts expected output quality for each routing
decision. This reduces GRPO advantage estimation variance by providing a learned
baseline.

```python
from losion.training.advanced_rlhf import JalurValueHead

value_head = JalurValueHead(
    d_model=2048,
    num_pathways=3,
    hidden_dim=512,
)

# Forward: predict value per token
values = value_head(hidden_states, routing_weights=weights)  # [batch, seq]
# Or per-pathway values:
all_values = value_head(hidden_states)  # [batch, seq, 3]
```

**Value head loss**: MSE between predicted value and actual reward, used as an
auxiliary training signal alongside the GRPO policy loss.

### 6.3 Self-Consistency Verification (Gemini Thinking)

Generate K=5 candidates, cluster by similarity, select the representative from
the largest cluster. Provides an internal reward signal without an external reward model.

```python
from losion.training.advanced_rlhf import SelfConsistencyVerifier

verifier = SelfConsistencyVerifier(
    num_samples=5,
    similarity_threshold=0.8,
)

result = verifier.verify(
    model, prompt=input_ids,
    max_new_tokens=128, temperature=0.7,
)
print(f"Consistency score: {result['consistency_score']:.2f}")
print(f"Cluster sizes: {result['cluster_sizes']}")
```

### 6.4 Dirichlet Noise Injection (AlphaZero)

Injects Dirichlet noise into Router logits during training to guarantee exploration
and prevent routing collapse.

**Formula** (AlphaZero):
```
logits_noisy = (1 - ε) * softmax(logits) + ε * Dir(α)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.25 | Concentration — low = sparse, high = uniform |
| `epsilon` | 0.25 | Blending factor — 0 = no noise, 1 = pure random |
| `root_only` | False | Inject only at first token (root) |

```python
from losion.training.advanced_rlhf import DirichletNoiseInjector

injector = DirichletNoiseInjector(
    alpha=0.25,    # Sparse exploration (AlphaZero default for Go)
    epsilon=0.25,  # 25% noise, 75% prior
    root_only=False,
)

# Inject noise into routing logits during training
noisy_logits = injector.inject(router_logits)  # [batch, seq, 3]
```

### 6.5 AdvancedGRPOTrainer — Putting It All Together

```python
from losion.training.advanced_rlhf import AdvancedGRPOTrainer, AdvancedGRPOConfig
from losion.models.losion_model_v2 import LosionForCausalLMV2
from losion.config import LosionConfig

model = LosionForCausalLMV2(LosionConfig())

adv_config = AdvancedGRPOConfig(
    group_size=8,
    clip_range=0.2,
    kl_coeff=0.05,
    entropy_coeff=0.02,
    use_value_head=True,       # MuZero value prediction
    use_self_play=True,        # AlphaZero self-play
    use_dirichlet_noise=True,  # AlphaZero exploration
    use_self_consistency=True, # Gemini verification
    dirichlet_alpha=0.25,
    dirichlet_epsilon=0.25,
    value_loss_coeff=0.5,
    gae_lambda=0.95,
)

adv_trainer = AdvancedGRPOTrainer(model, config=adv_config)

# Train step combines all four techniques
metrics = adv_trainer.train_step(prompts=input_ids, attention_mask=mask)
print(f"Total loss: {metrics['adv_grpo_loss']:.4f}")
print(f"Value loss: {metrics.get('value_loss', 'N/A')}")
print(f"Mean advantage: {metrics['mean_advantage']:.4f}")
```

> **Agent Context**
> ```
> All advanced RLHF classes: losion/training/advanced_rlhf.py
> AdvancedGRPOConfig: controls which techniques are active
> AdvancedGRPOTrainer: inherits from none, composes all four
> Value head parameters are included in the optimizer automatically
> Reference model is deep-copied at init for KL computation
> ```

---

## 7. DAPO — Decoupled Clip & Dynamic Sampling (v0.8)

DAPO (Decoupled Clip & Dynamic Sampling Policy Optimization) replaces GRPO as the
default RL optimizer in Losion v0.8+. It provides four key improvements over GRPO:

1. **Decoupled clip**: Separate low/high clip ratios (0.2/0.28) prevent both policy
   collapse and reward hacking
2. **Dynamic sampling**: Filters prompts with zero-variance rewards for ~15-20%
   efficiency gain
3. **Token-level loss**: Finer credit assignment per token instead of per-sequence
4. **Overlong filtering**: Penalizes excessively long responses to prevent reward hacking

```python
from losion.training.dapo import DAPOTrainer, DAPOConfig
from losion.models.losion_model_v2 import LosionForCausalLMV2
from losion.config import LosionConfig

config = LosionConfig()
model = LosionForCausalLMV2(config)

dapo_config = DAPOConfig(
    group_size=8,
    clip_ratio_low=0.2,
    clip_ratio_high=0.28,
    dynamic_sampling=True,
    token_level_loss=True,
    overlong_filter=True,
    kl_coeff=0.05,
    entropy_coeff=0.01,
)

trainer = DAPOTrainer(model, config=dapo_config)
metrics = trainer.train_step(prompts=input_ids, attention_mask=mask)
print(f"Policy loss: {metrics['policy_loss']:.4f}")
print(f"Mean reward: {metrics['mean_reward']:.4f}")
```

DAPO integrates with RLVR as the reward function provider, enabling verifiable
rewards for math and code tasks without a learned reward model.

---

## 8. RLVR — Reinforcement Learning with Verifiable Rewards (v0.8)

RLVR (Reinforcement Learning with Verifiable Rewards) scales RL training using
objective, programmable reward functions instead of learned reward models. This is
especially effective for math and code tasks where correctness can be verified automatically.

**Verifier types**:

| Verifier | Description | Use Case |
|----------|-------------|----------|
| `MathVerifier` | Numeric + symbolic verification | Math reasoning |
| `CodeVerifier` | Sandboxed execution | Code generation |
| `FormatVerifier` | Regex + length + JSON validation | Structured output |
| `ExactMatchVerifier` | Exact/fuzzy matching | Factual QA |
| `CompositeVerifier` | Curriculum difficulty scheduling | Multi-task |

```python
from losion.training.rlvr import RLVRTrainer, MathVerifier, CodeVerifier
from losion.training.dapo import DAPOTrainer, DAPOConfig

# Set up verifiable rewards
verifiers = {
    "math": MathVerifier(),
    "code": CodeVerifier(timeout=30),
}

# DAPO + RLVR integration
dapo_config = DAPOConfig(group_size=8, dynamic_sampling=True)
trainer = DAPOTrainer(model, config=dapo_config, reward_fn=verifiers)
```

---

## 9. LLM-JEPA — Joint-Embedding Predictive Architecture (v0.6)

LLM-JEPA replaces the standard next-token prediction objective with predicting
future latent states. Instead of predicting token IDs, the model learns to predict
the *representation* of future tokens, providing a richer training signal.

**Key benefits**:
- Richer training signal than next-token prediction alone
- Better representation learning for downstream tasks
- Compatible with standard autoregressive training (auxiliary loss)
- Particularly effective in Phase 1–2 of the Losion training recipe

```python
from losion.training.llm_jepa import LLMJEPALoss, JEPAConfig

jepa_config = JEPAConfig(
    predictor_depth=2,
    prediction_horizon=4,        # Predict 4 steps ahead
    loss_weight=0.3,             # Auxiliary loss weight
    target_decay=0.99,           # EMA decay for target encoder
)

jepa_loss = LLMJEPALoss(config=jepa_config)

# In training loop:
total_loss = ar_loss + jepa_weight * jepa_loss(hidden_states, future_states)
```

---

## 10. Losion Training Orchestrator (v0.8)

The Losion Training Orchestrator (`losion/training/losion_orchestrator.py`) is a
one-stop training manager that integrates ALL 13+ Losion training techniques into
a single 4-phase pipeline:

- **Phase 1**: WSD LR + JEPA + expert specialization
- **Phase 2**: JEPA (reduced) + TACO + curriculum + active learning
- **Phase 3**: DAPO/GRPO (auto-selected) + RLVR + ETR + TACO + evolutionary search
- **Phase 4**: Gen distillation + BitDistill + ETR + early exit

```python
from losion.training.losion_orchestrator import LosionOrchestrator, OrchestratorConfig

orch_config = OrchestratorConfig(
    model_config=LosionConfig(),
    total_steps=300000,
    use_dapo=True,           # v0.8+ (fallback to GRPO if False)
    use_rlvr=True,           # v0.8+
    use_jepa=True,           # v0.6+
    use_taco=True,           # v0.5+
    use_etr=True,            # v0.5+
    use_distillation=True,   # v0.5+
    use_bitdistill=True,     # v0.5+
    use_evolutionary=True,   # v0.7+
)

orchestrator = LosionOrchestrator(config=orch_config)
orchestrator.train()
```

Full checkpoint save/resume with all training state is supported.

---

## 11. Advanced Training Techniques

### 11.1 Chinchilla Per-Jalur Scaling

Each pathway in Losion has different FLOP costs. Chinchilla scaling must be applied
**per pathway**, not uniformly across the model.

```python
from losion.training.advanced_backprop import ChinchillaScaler

scaler = ChinchillaScaler(
    total_flops_budget=1e22,
    jalur_flop_ratios=(0.2, 0.5, 0.3),  # SSM=20%, Attn=50%, MoE=30%
)

# Compute optimal parameter/data allocation per pathway
result = scaler.compute_optimal_scaling(moe_active_ratio=0.0625)
for i, (p, d) in enumerate(zip(result.jalur_params, result.jalur_data)):
    names = ["SSM", "Attention", "MoE"]
    print(f"Jalur {i+1} ({names[i]}): {p:,.0f} params, {d:,.0f} tokens")

# Validate your config against Chinchilla scaling
from losion.config import LosionConfig
validation = scaler.validate_config(LosionConfig())
print(validation['recommendation'])
```

### 11.2 Per-Jalur Learning Rate Schedules

Each pathway has different training dynamics: SSM is cheap per step (fast warmup,
fast decay), Attention is expensive (slow warmup, slow decay), MoE is medium.
Using a single LR for all pathways causes cheap-branch oscillation.

```python
from losion.training.advanced_backprop import PerJalurLRScheduler

lr_scheduler = PerJalurLRScheduler(
    base_lr=3e-4,
    total_steps=100000,
    warmup_ratios=(0.03, 0.06, 0.04),  # SSM=3%, Attn=6%, MoE=4%
    decay_rates=(0.8, 0.5, 0.6),        # SSM=sharp, Attn=smooth, MoE=medium
)

# Get LR for each pathway at step 50K
lrs = lr_scheduler.get_all_lrs(50000)
print(f"SSM LR: {lrs[0]:.2e}, Attn LR: {lrs[1]:.2e}, MoE LR: {lrs[2]:.2e}")
```

### 11.3 Logit Soft Capping (Gemma 2)

Prevents logit divergence during training without hard clipping, which can cause
mode collapse. Applied to AR output logits, flow matching velocity predictions,
and MTP auxiliary head logits.

**Formula**: `capped = cap * tanh(logits / cap)`

```python
from losion.training.advanced_backprop import LogitSoftCapper

capper = LogitSoftCapper(cap_value=50.0)  # Gemma 2 default

# Apply during training
capped_logits = capper(output_logits)  # [batch, seq, vocab_size]
```

**Why soft capping?** Hard clipping (e.g., `torch.clamp(logits, -50, 50)`)
creates a flat gradient region where the model cannot learn to pull logits back
from the boundary. Soft capping via `tanh` provides smooth gradients everywhere.

### 11.4 Scheduled Sampling (GraphCast)

Bridges the teacher-forcing / autoregressive gap. During early training, use
ground truth as input. Gradually replace with model's own predictions.

```python
from losion.training.advanced_backprop import ScheduledSampler

sampler = ScheduledSampler(
    total_steps=100000,
    max_scheduled_ratio=0.5,   # Up to 50% model predictions
    warmup_steps=1000,         # Pure teacher forcing for first 1K steps
    schedule_type="linear",    # linear / exponential / inverse_sigmoid
)

# At training step
p = sampler.get_sampling_probability(step=50000)  # e.g., 0.25
input_tokens = sampler.sample_input(ground_truth, model_prediction, step=50000)
```

### 11.5 Confidence Heads (AlphaFold 3)

Three auxiliary confidence heads provide dense training signals without affecting
inference (can be distilled away after training):

1. **Routing Confidence**: Is the routing decision correct?
2. **Prediction Difficulty**: How hard is the next token?
3. **Diffusion Quality**: Will flow matching produce good output?

```python
from losion.training.advanced_backprop import ConfidenceHeads

heads = ConfidenceHeads(d_model=2048, num_confidence_types=3)

# Forward: predict confidence scores
predictions = heads(hidden_states)
print(predictions.keys())
# dict_keys(['routing_confidence', 'prediction_difficulty', 'diffusion_quality'])

# Compute auxiliary loss (used as regularization)
aux_loss = heads.compute_auxiliary_loss(
    hidden_states=hidden_states,
    ar_loss_per_token=per_token_loss,
    routing_entropy=routing_entropy,
)
total_loss = main_loss + 0.1 * aux_loss  # Weight as regularization
```

### 11.6 Parallel Attention + FFN (PaLM)

Computes attention and FFN in **parallel** rather than sequentially, effectively
doubling the model's "depth" within the same latency budget.

**Formula**: `output = x + Attention(LN(x)) + FFN(LN(x))`

```python
from losion.training.advanced_backprop import ParallelAttentionFFN

layer = ParallelAttentionFFN(
    d_model=2048,
    n_heads=16,
    d_kv=128,
    mla_latent_dim=512,
    ffn_dim_multiplier=4,
)

# Forward pass — attention and FFN computed in parallel
output = layer(x, attention_mask=mask)  # [batch, seq, d_model]
```

**When to use**: Apply in Jalur 2 (Attention pathway) where MLA attention and
FFN/compression can run in parallel. Particularly beneficial for large models (7B+).

### 11.7 Gradient Overlapping (PaLM 2)

Overlaps gradient synchronization with backward pass computation, hiding 40–60%
of communication latency.

```python
from losion.training.advanced_backprop import GradientOverlapScheduler

overlap = GradientOverlapScheduler(
    num_jalurs=3,
    overlap_strategy="flops_weighted",  # or "round_robin"
)

# Create overlap plan for one backward pass
plan = overlap.create_overlap_plan()
for step in plan:
    print(f"Step {step['step']}: {step['description']}")
# Step 0: Backward Jalur 0 + Sync gradient Jalur 1
# Step 1: Backward Jalur 1 + Sync gradient Jalur 2
# Step 2: Backward Jalur 2 + Sync gradient Jalur 0
```

**Implementation note**: Requires dual CUDA streams for compute and communication.
The `GradientOverlapScheduler` provides the scheduling logic; actual overlapping
requires distributed training setup with NCCL.

> **Agent Context**
> ```
> All advanced techniques: losion/training/advanced_backprop.py
DAPO: losion/training/dapo.py
RLVR: losion/training/rlvr.py
LLM-JEPA: losion/training/llm_jepa.py
Orchestrator: losion/training/losion_orchestrator.py
> ChinchillaScaler: per-jalur FLOP budget allocation
> PerJalurLRScheduler: different warmup/decay per pathway
> LogitSoftCapper: stabilizes training, no mode collapse
> ScheduledSampler: bridges teacher-forcing gap
> ConfidenceHeads: auxiliary signals (can be distilled away)
> ParallelAttentionFFN: doubles effective depth
> GradientOverlapScheduler: hides 40-60% communication latency
> ```

---

## 12. Memory Optimization

### 12.1 Progressive KV Compression (Gemini LC)

Position-dependent KV cache compression — newer tokens get full fidelity, older
tokens are compressed more aggressively.

| Token Age | Compression | Memory |
|-----------|------------|--------|
| Recent (last 4K) | 1:1 (full) | 100% |
| Medium (4K–64K) | 4:1 | 25% |
| Old (64K+) | 16:1 | 6.25% |

**Overall**: ~10× memory reduction for 1M context compared to uniform storage.

```python
from losion.training.advanced_memory_data import ProgressiveKVCompressor

compressor = ProgressiveKVCompressor(
    recent_window=4096,
    medium_window=65536,
    recent_ratio=1.0,
    medium_ratio=0.25,
    old_ratio=0.0625,
)

# Compress KV cache
compressed_k, compressed_v = compressor.compress_kv(keys, values, current_length=100000)

# Estimate memory savings for a sequence
savings = compressor.estimate_memory_savings(seq_len=100000, bytes_per_element=2)
print(f"Memory savings: {savings['savings_ratio']:.1%}")
# Memory savings: 87.5%
```

### 12.2 Attention Sinks (Gemini LC)

Reserve 4 "sink tokens" at the start of every sequence. Sink tokens are never
evicted from the KV cache and receive disproportionate attention weight, stabilizing
streaming inference.

```python
from losion.training.advanced_memory_data import AttentionSinkManager

sink_manager = AttentionSinkManager(num_sink_tokens=4)

# Create eviction mask (True = can evict)
eviction_mask = sink_manager.get_eviction_mask(seq_len=10000, device=device)

# Modify attention mask to always attend to sinks
modified_mask = sink_manager.modify_attention_mask(attention_mask)
```

### 12.3 Dynamic Expert Buffer Allocation (GShard)

Instead of over-provisioning fixed buffers for every MoE expert (causing 30–50%
memory waste), allocate dynamically based on predicted load.

```python
from losion.training.advanced_memory_data import DynamicExpertBufferAllocator

allocator = DynamicExpertBufferAllocator(
    num_experts=64,
    base_buffer_size=256,
    safety_margin=0.10,
)

# Allocate buffers based on predicted load
predicted_loads = router.get_predicted_loads()  # [num_experts]
buffers = allocator.allocate_buffers(predicted_loads, total_tokens=4096)

# Compare vs fixed allocation
savings = allocator.compute_memory_savings(predicted_loads, total_tokens=4096)
print(f"Memory savings: {savings['memory_savings_percent']:.1f}%")
```

### 12.4 Gradient Checkpointing

Reduces memory by ~60% at the cost of ~30% slower training (recomputation during
backward pass instead of storing all activations).

```yaml
training:
  gradient_checkpointing: true
  # Checkpoint every other layer for best speed/memory trade-off
  checkpoint_every_n_layers: 2
```

### 12.5 FP8 Training

Reduces memory by ~40% and increases throughput on H100/MI300X hardware.

```yaml
training:
  fp8_enabled: true
  precision: bf16  # FP8 for matmuls, bf16 for accumulation

# Only supported on:
# - NVIDIA H100, H200, B200
# - AMD MI300X, MI325
```

**FP8 limitations**:
- Not supported on consumer GPUs (RTX 4090, etc.)
- Requires `transformer_engine` or `torch._scaled_mm` backend
- Some operations still use bf16 (layer norm, softmax)

### 12.6 FSDP Sharding

Fully Sharded Data Parallel distributes model parameters, gradients, and optimizer
states across all GPUs.

```yaml
training:
  use_fsdp: true
  fsdp_config:
    sharding_strategy: FULL_SHARD  # or SHARD_GRAD_OP
    mixed_precision: bf16
    activation_checkpointing: true
    sync_module_states: true
```

### 12.7 OOM Troubleshooting Checklist

| # | Action | Memory Saved | Difficulty |
|---|--------|-------------|-----------|
| 1 | Reduce batch size 2× | ~40% | Easy |
| 2 | Enable gradient accumulation | Enables #1 | Easy |
| 3 | Reduce sequence length | Proportional | Easy |
| 4 | Enable gradient checkpointing | ~60% | Easy |
| 5 | Switch to FSDP | ~70% per GPU | Medium |
| 6 | Enable FP8 (H100/MI300X only) | ~40% | Medium |
| 7 | Use Progressive KV Compression | ~90% for long ctx | Medium |
| 8 | Dynamic Expert Buffer Allocation | ~30-50% MoE | Medium |
| 9 | Reduce `num_active_experts` | Proportional | Easy |
| 10 | Use `torch.compile()` with `max_autotune` | ~10-15% | Hard |

> **Agent Context**
> ```
> Memory optimization classes: losion/training/advanced_memory_data.py
> ProgressiveKVCompressor: position-dependent KV compression
> AttentionSinkManager: stabilize streaming inference
> DynamicExpertBufferAllocator: reduce MoE memory waste
> MemoryEfficientBackprop: losion/training/advanced_backprop.py
> Enable FP8: config.training.fp8_enabled = True
> Enable FSDP: trainer_config.use_fsdp = True
> ```

---

## 13. Data Pipeline

### 13.1 Modality-Aware Loss Weighting (Gemini)

Dynamic per-pathway loss weighting based on inverse perplexity. If a pathway has
low perplexity (well-trained), reduce its loss weight and increase the weight for
pathways that need more training.

```python
from losion.training.advanced_memory_data import ModalityAwareLossWeighter

weighter = ModalityAwareLossWeighter(
    num_jalurs=3,
    temperature=1.0,
    ema_decay=0.99,
    min_weight=0.10,  # Never fully ignore a pathway
)

# Update with per-pathway losses
weighter.update_perplexities([ssm_loss, attn_loss, moe_loss])

# Get weights for current step
weights = weighter.compute_weights()
print(f"SSM: {weights[0]:.3f}, Attn: {weights[1]:.3f}, MoE: {weights[2]:.3f}")

# Apply to loss
total_loss = weights[0] * ssm_loss + weights[1] * attn_loss + weights[2] * moe_loss
```

### 13.2 Chinchilla Data Sizing

See [Section 2.5](#25-chinchilla-optimal-data-sizing) for the full calculator.
Key rule: **20 tokens per active parameter**.

```python
from losion.training.advanced_memory_data import ChinchillaDataSizer

sizer = ChinchillaDataSizer(moe_active_experts=4, moe_total_experts=64)
result = sizer.compute_optimal_dataset_size(
    total_params=9_800_000_000, moe_params_fraction=0.30,
)
# optimal_tokens: 144B, dataset: ~537 GB
```

### 13.3 Sample-then-Filter (AlphaCode)

Generate K=64 candidates, then filter using AR log-probability, consistency
classification, and diversity clustering. Dramatically improves output quality
at K× compute cost.

```python
from losion.training.advanced_memory_data import SampleFilterPipeline

pipeline = SampleFilterPipeline(
    num_samples=64,
    ar_weight=0.4,
    consistency_weight=0.3,
    diversity_clusters=8,
    top_k_final=1,
)

# Generate and filter
best_output = pipeline.generate_and_filter(
    model, prompt=input_ids,
    max_new_tokens=128, temperature=0.8,
)
```

### 13.4 Template-Based Conditional Routing (AlphaCode)

Condition routing on the expected output type. When the Router detects structured
output patterns (code, math, formal language), inject a "template bias" into the
routing logits.

```python
from losion.training.advanced_memory_data import TemplateConditionalRouter

router = TemplateConditionalRouter(
    custom_biases={
        "code": [-0.1, 0.2, 0.1],    # Code → more Attention (precise)
        "math": [-0.1, 0.3, 0.0],     # Math → more Attention (reasoning)
        "creative": [0.1, -0.1, 0.1], # Creative → more SSM + MoE
        "factual": [-0.1, -0.1, 0.3], # Factual → more Retrieval
    }
)

# Detect output type from input
output_type = router.detect_output_type(input_ids, tokenizer=tokenizer)

# Apply bias to routing logits
biased_logits = router.apply_template_bias(routing_logits, input_ids, tokenizer)
```

### 9.5 Active Learning Loop (GNoME Style)

Self-improving training cycle: train → predict → verify → augment → retrain.

```python
from losion.training.active_learning import ActiveLearningLoop, ActiveLearningConfig

al_config = ActiveLearningConfig(
    confidence_threshold=0.90,
    consistency_threshold=0.80,
    max_new_samples=10000,
    retrain_epochs=1,
    num_iterations=5,
    use_curriculum=True,
)

al_loop = ActiveLearningLoop(model, config=al_config)

# Run active learning iterations
for iteration in range(al_config.num_iterations):
    # Train on current data
    al_loop.train_iteration(train_data)
    # Predict on unlabeled data and filter by confidence
    new_data = al_loop.predict_and_filter(unlabeled_data)
    # Augment training data
    al_loop.augment_training_data(new_data)
    # Print summary
    print(al_loop.get_iteration_summary())
```

### 9.6 Evolutionary Search (FunSearch Style)

Use the LLM as a mutator in evolutionary search to discover novel solutions.
Population-based: mutate best solutions, evaluate, select survivors.

```python
from losion.training.evolutionary_search import EvolutionarySearcher, EvolutionaryConfig

evo_config = EvolutionaryConfig(
    population_size=16,
    num_elites=4,
    mutation_rate=0.7,
    crossover_rate=0.3,
    max_generations=10,
    score_threshold=0.95,
    diversity_weight=0.1,
)

searcher = EvolutionarySearcher(d_model=2048, config=evo_config)

# Run evolutionary search
best_solution, info = searcher.forward(
    seed=initial_embedding,
    external_evaluator=reward_function,  # optional
)
print(f"Best score: {info['best_score']:.4f}")
print(f"Generations: {info['generations']}")
print(f"Converged: {info['converged']}")
```

> **Agent Context**
> ```
> Data pipeline classes: losion/training/advanced_memory_data.py
> Active learning: losion/training/active_learning.py
> Evolutionary search: losion/training/evolutionary_search.py
> ModalityAwareLossWeighter: dynamic per-pathway weighting
> SampleFilterPipeline: AlphaCode-style filtering
> TemplateConditionalRouter: output-type-aware routing
> ActiveLearningLoop: GNoME-style self-improvement
> EvolutionarySearcher: FunSearch-style solution discovery
> ```

---

## 14. Hyperparameter Reference

### 10.1 Learning Rates

| Model Size | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|-----------|---------|---------|---------|---------|
| 1B | 3e-4 | 1.5e-4 | 5e-5 | 5e-6 |
| 7B | 2e-4 | 1e-4 | 5e-5 | 1e-6 |
| 48B | 1e-4 | 5e-5 | 2e-5 | 5e-7 |

**Scheduler**: Cosine decay with linear warmup. Warmup = 2–5% of total steps.

**LR tuning tips**:
- Loss spiking early? → Decrease LR 2–5×
- Loss decreasing too slowly? → Increase LR 2×
- Phase 3 instability? → Reduce router `bias_lr` from 0.01 to 0.001

### 10.2 Batch Sizes

| Model Size | Single GPU | 4 GPU | 8 GPU | Effective Target |
|-----------|-----------|-------|-------|-----------------|
| 1B | 16–32 | 64–128 | 128–256 | 256–512 |
| 7B | 2–4 | 16–32 | 32–64 | 512–1024 |
| 48B | — | 4–8 | 16–32 | 1024–2048 |

**Formula**: `effective_batch = batch_size × grad_accum_steps × num_gpus`

### 10.3 Weight Decay

| Component | Weight Decay |
|-----------|-------------|
| Weight matrices | 0.1 |
| Bias | 0.0 |
| LayerNorm / RMSNorm | 0.0 |
| Embedding | 0.0 |
| Router bias | 0.0 |

### 10.4 Router Hyperparameters

| Parameter | 1B | 7B | 48B | Range |
|-----------|-----|------|------|-------|
| `bias_lr` | 0.01 | 0.01 | 0.01 | 0.001–0.1 |
| `aux_loss_weight` | 0.0 | 0.0 | 0.0 | 0.0–0.01 |
| `thinking_threshold` | 0.5 | 0.5 | 0.5 | 0.3–0.7 |
| `top_k_pathways` | 2 | 2 | 2 | 1–3 |

### 10.5 MoE Hyperparameters

| Parameter | 1B | 7B | 48B |
|-----------|-----|------|-------|
| `num_experts` | 16 | 64 | 256 |
| `num_active_experts` | 2 | 4 | 8 |
| `engram_dim` | 128 | 256 | 512 |
| `shared_expert` | yes | yes | yes |
| `routing_strategy` | expert_choice | expert_choice | expert_choice |

### 10.6 Complete YAML Reference (Losion-7B)

```yaml
model:
  d_model: 2048
  n_layers: 24
  vocab_size: 128256
  max_seq_len: 131072

  ssm:
    d_state: 128
    d_conv: 4
    expand: 2
    chunk_size: 256
    use_wkv: true
    use_delta_net: true
    interleaving_ratios: [4, 1, 1]

  attention:
    n_heads: 16
    d_kv: 128
    mla_latent_dim: 512
    use_irope: true
    irope_ratio: 3
    base_interleaving_ratio: 5
    thinking_interleaving_ratio: 2

  retrieval:
    num_experts: 64
    num_active_experts: 4
    d_ff: 4096
    use_engram: true
    engram_dim: 256
    use_shared_expert: true

  router:
    top_k_pathways: 2
    use_thinking_toggle: true
    bias_lr: 0.01
    aux_loss_weight: 0.0  # aux-loss-free!

  output:
    use_mtp: true
    mtp_num_tokens: 3
    use_flow_matching: false

training:
  batch_size: 128
  learning_rate: 2.0e-4
  weight_decay: 0.1
  warmup_steps: 4000
  max_steps: 300000
  grad_clip: 1.0
  fp8_enabled: false
  precision: bf16

hardware:
  device: auto
  backend: auto
  compile_model: true
```

---

## 15. Distributed Training

### 11.1 DDP vs FSDP

| Aspect | DDP | FSDP |
|--------|-----|------|
| Model replication | Full copy per GPU | Sharded parameters |
| Memory per GPU | Full model | Model ÷ num_gpus |
| Communication | Gradient AllReduce | Parameter AllGather + Gradient ReduceScatter |
| Setup complexity | Low | Medium |
| Recommended for | Models that fit on 1 GPU | Large models (48B+) |
| `torch.compile()` | Compatible | Experimental |

### 11.2 Single-Node Multi-GPU

```bash
# 4× A100 80GB, Losion-7B with DDP
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/losion-7b.yaml
```

```yaml
# config override for 4-GPU DDP
training:
  batch_size: 64          # Per-GPU
  gradient_accumulation_steps: 2  # Effective: 64 × 2 × 4 = 512
  use_ddp: true

hardware:
  compile_model: true     # Safe with DDP
```

### 11.3 Multi-Node Setup

```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=8 \
    --master_addr=10.0.0.1 --master_port=29500 \
    --node_rank=0 \
    scripts/train.py --config configs/losion-48b.yaml

# Node 1 (worker)
torchrun --nnodes=2 --nproc_per_node=8 \
    --master_addr=10.0.0.1 --master_port=29500 \
    --node_rank=1 \
    scripts/train.py --config configs/losion-48b.yaml
```

```yaml
# 8× H100 80GB, Losion-48B with FSDP
training:
  batch_size: 8           # Per-GPU
  gradient_accumulation_steps: 8  # Effective: 8 × 8 × 8 = 512
  use_fsdp: true

hardware:
  compile_model: false    # Not yet stable with FSDP
```

### 11.4 Gradient Overlapping Configuration

```python
from losion.training.advanced_backprop import GradientOverlapScheduler

# Flops-weighted: synchronize cheapest pathway during backward of most expensive
overlap = GradientOverlapScheduler(num_jalurs=3, overlap_strategy="flops_weighted")
plan = overlap.create_overlap_plan()
# When backward Jalur 2 (Attention, most expensive), sync Jalur 1 (SSM, cheapest)
```

> **Agent Context**
> ```
> DDP: torch.nn.parallel.DistributedDataParallel
> FSDP: torch.distributed.fsdp.FullyShardedDataParallel
> Setup: losion/training/utils.py :: setup_distributed()
> LosionTrainer._wrap_model_for_distributed() handles wrapping
> compile_model=True is safe with DDP, experimental with FSDP
> ```

---

## 16. Monitoring

### 12.1 Console Logging

Losion prints metrics every `logging_steps` steps:

```
Step 100 | loss: 8.2341 | ar_loss: 7.8912 | mtp_loss: 1.1432 | lr: 0.000150 | phase: phase_1_individual
Step 200 | loss: 7.6534 | ar_loss: 7.3214 | mtp_loss: 0.9856 | lr: 0.000200 | phase: phase_1_individual
```

### 12.2 Wandb Integration

```yaml
training:
  use_wandb: true
  wandb_project: "losion-experiments"
```

**Metrics logged to Wandb**:

| Metric | Description |
|--------|-------------|
| `loss` | Total loss |
| `ar_loss` | Autoregressive cross-entropy loss |
| `mtp_loss` | Multi-token prediction auxiliary loss |
| `lr` | Current learning rate |
| `phase` | Active training phase |
| `steps_per_sec` | Training throughput |
| `eval_loss` | Validation loss |
| `routing_weights/mean` | Mean routing weight per pathway |
| `routing_entropy` | Routing distribution entropy |
| `expert_utilization` | MoE expert usage distribution |
| `gpu_memory_used` | GPU memory usage |
| `gpu_utilization` | GPU compute utilization |

### 12.3 Custom Monitoring with Routing Info

```python
from losion import LosionForCausalLMV2, LosionConfig

config = LosionConfig()
model = LosionForCausalLMV2(config)

# Forward with routing info
output = model(input_ids, labels=labels, return_routing_info=True)

if output.routing_info:
    for layer_idx, routing in enumerate(output.routing_info):
        summary = {
            "layer": layer_idx,
            "weights": {
                "ssm": routing.adjusted_weights[:, :, 0].mean().item(),
                "attention": routing.adjusted_weights[:, :, 1].mean().item(),
                "retrieval": routing.adjusted_weights[:, :, 2].mean().item(),
            },
            "thinking_mode": routing.thinking_assessment.mode.value,
            "complexity": routing.thinking_assessment.complexity_score.mean().item(),
        }
        print(f"Layer {layer_idx}: {summary}")
```

### 12.4 TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir checkpoints/

# In config
training:
  use_tensorboard: true
  tensorboard_dir: "./tb_logs"
```

> **Agent Context**
> ```
> Wandb init: LosionTrainer.__init__() with trainer_config.use_wandb=True
> Custom metrics: access output.routing_info after forward pass
> Logging interval: trainer_config.logging_steps (default 10)
> ```

---

## 17. Common Issues & Solutions

### 13.1 Loss Not Decreasing

**Symptoms**: Loss stagnates or increases after a few thousand steps.

| Cause | Solution |
|-------|---------|
| LR too high | Decrease LR 2–5× |
| LR too low | Increase LR 2× |
| Batch size too small | Increase batch size or gradient accumulation |
| Data issues | Check dataset (tokenization, duplicates, corruption) |
| Gradient explosion | Reduce `max_grad_norm` to 0.5 |
| One pathway dominates | Check routing weights, may need manual rebalance |

### 13.2 Routing Collapse

**Symptoms**: One pathway dominates (>95% routing weight).

```python
# Check routing distribution
output = model(input_ids, return_routing_info=True)
for layer_routing in output.routing_info:
    weights = layer_routing.adjusted_weights.mean(dim=(0, 1))
    print(f"SSM: {weights[0]:.3f}, Attn: {weights[1]:.3f}, Retr: {weights[2]:.3f}")

# Solutions:
# 1. Lower thinking threshold
config.router.thinking_threshold = 0.3  # from 0.5

# 2. Temporarily add aux loss
config.router.aux_loss_weight = 0.01  # from 0.0; remove after rebalancing

# 3. Re-initialize router weights
for layer in model.model.layers:
    nn.init.xavier_uniform_(layer.router.bias_router.routing_weight)
```

### 13.3 OOM (Out of Memory)

See [Section 8.7](#87-oom-troubleshooting-checklist) for the complete checklist.

Quick fix priority:
1. Reduce batch size 2× (~40% VRAM saved)
2. Enable gradient checkpointing (~60% VRAM saved, 30% slower)
3. Switch to FSDP (~70% VRAM per GPU saved)
4. Enable FP8 on supported hardware (~40% VRAM saved)

### 13.4 Phase 3 Training Instability

**Symptoms**: Loss fluctuates wildly after router is unfrozen.

**Solutions**:
- Reduce router LR: `config.router.bias_lr = 0.001` (from 0.01)
- Increase KL penalty in GRPO: `kl_coefficient = 0.1` (from 0.05)
- Freeze router for additional steps before unfreezing
- Tighten gradient clipping: `max_grad_norm = 0.5`
- Use AdvancedGRPOTrainer with Dirichlet noise for exploration stability

### 13.5 AuxFreeMoE MTP Loss Not Propagated — FIXED in v2.0.0

**Symptoms** (pre-v2.0.0): `mtp_loss` computed by `MTPMoEHead` inside `AuxFreeMoE` but never added to the model's total loss. This meant **32.2% of model parameters** (`MTPMoEHead.pred_heads`) received zero gradient — they were dead weight during training.

**Fix** (v2.0.0): `LosionForCausalLMV2.forward()` now extracts `mtp_loss` from each layer's `routing_info["retrieval_aux"]`, averages across layers, and adds it to the total loss. All model parameters now receive training gradients.

```python
# v2.0.0: Verify MTP loss is being propagated
output = model(input_ids)
# Check that moe_mtp_loss appears in loss breakdown
print(output.loss)  # Should include MoE MTP contribution
```

### 13.6 MoE Expert Underutilization

**Symptoms**: Some experts never receive tokens.

**Solutions**:
- Bias-based routing should handle this (biases are updated by gradients)
- If still occurring, temporarily add auxiliary load balancing loss:
  `config.router.aux_loss_weight = 0.01`
- Once load balancing stabilizes, return to `0.0` (aux-loss-free)
- Consider switching to `expert_choice` routing strategy (guaranteed balance)

### 13.7 Loss Spikes During Long-Context Training

**Symptoms**: Sudden loss spikes when sequence length increases.

**Solutions**:
- Apply Logit Soft Capping (`cap_value=50.0`) before softmax
- Reduce LR when doubling sequence length (scale by 0.5×)
- Enable Attention Sinks for stable streaming
- Use Progressive KV Compression to reduce memory pressure

> **Agent Context**
> ```
> Routing collapse: check output.routing_info
> OOM: see Section 8 checklist, most common fix is gradient_checkpointing
> Phase 3 instability: reduce bias_lr, increase kl_coeff
> Expert underutilization: try expert_choice routing strategy
> Loss spikes: enable LogitSoftCapper, reduce LR at seq_len transitions
> ```

---

## 18. Evaluation & Benchmarks

### 14.1 Standard Benchmarks

| Category | Benchmark | Metric | 1B Target | 7B Target | 48B Target |
|----------|-----------|--------|-----------|-----------|-----------|
| General | MMLU | Accuracy (5-shot) | 30–40% | 55–65% | 75–85% |
| Reasoning | GSM8K | Accuracy | 20–30% | 60–75% | 85–95% |
| Code | HumanEval | pass@1 | 10–20% | 30–45% | 60–75% |
| Math | MATH | Accuracy | 5–10% | 20–35% | 45–60% |
| Long Context | LongBench | F1/Accuracy | — | 40–55% | 60–75% |
| Multi-Language | XWinograd | Accuracy | 55–65% | 70–80% | 85–90% |

### 14.2 Running Evaluations

```bash
# Evaluate on specific benchmarks
python scripts/evaluate.py --config configs/losion-7b.yaml \
    --checkpoint checkpoints/step-50000 \
    --benchmark mmlu,gsm8k,humaneval

# Evaluate all standard benchmarks
python scripts/evaluate.py --config configs/losion-7b.yaml \
    --checkpoint checkpoints/step-50000 \
    --benchmark all
```

### 14.3 Evaluation During Training

```yaml
training:
  eval_steps: 500          # Run eval every 500 steps
  eval_benchmarks: mmlu    # Quick benchmark during training
  save_steps: 2000         # Save checkpoint every 2000 steps
```

### 14.4 Interpreting Results

| Metric | Good | Acceptable | Needs Improvement |
|--------|------|-----------|-------------------|
| MMLU (5-shot) | >65% | 50–65% | <50% |
| GSM8K | >70% | 50–70% | <50% |
| HumanEval | >45% | 25–45% | <25% |
| Training loss (1B) | <2.5 | 2.5–4.0 | >4.0 |
| Training loss (7B) | <2.0 | 2.0–3.5 | >3.5 |
| Training loss (48B) | <1.5 | 1.5–3.0 | >3.0 |
| Routing entropy | 0.8–1.0 | 0.5–0.8 | <0.5 (collapse) |

### 14.5 Losion-Specific Metrics

| Metric | What It Measures | Healthy Range |
|--------|-----------------|--------------|
| `routing_entropy` | Balance of pathway usage | 0.8–1.0 (normalized) |
| `expert_utilization` | MoE expert load distribution | CoV < 0.3 |
| `thinking_activation_rate` | % of tokens using thinking mode | 10–30% |
| `pathway_agreement` | How often pathways agree on output | >0.6 after Phase 2 |

> **Agent Context**
> ```
> Evaluation script: scripts/evaluate.py
> Benchmarks: MMLU, GSM8K, HumanEval, MATH, LongBench, XWinograd
> During training: eval_steps in trainer_config
> Losion-specific: check routing_entropy, expert_utilization, thinking_activation_rate
> ```

---

## Appendix: Full Training Config Templates

### Losion-1B (Single GPU Prototype)

```yaml
model:
  d_model: 768
  n_layers: 12
  vocab_size: 32000
  max_seq_len: 32768
  ssm:
    d_state: 64
    d_conv: 4
    expand: 2
    use_wkv: true
    use_delta_net: true
  attention:
    n_heads: 12
    d_kv: 64
    mla_latent_dim: 128
    use_irope: true
  retrieval:
    num_experts: 16
    num_active_experts: 2
    use_engram: true
    engram_dim: 128
  router:
    top_k_pathways: 2
    use_thinking_toggle: true
    bias_lr: 0.01
    aux_loss_weight: 0.0
  output:
    use_mtp: true
    mtp_num_tokens: 2
    use_flow_matching: false
training:
  batch_size: 32
  learning_rate: 3.0e-4
  weight_decay: 0.1
  warmup_steps: 2000
  max_steps: 100000
  grad_clip: 1.0
  precision: bf16
hardware:
  device: auto
  compile_model: true
```

### Losion-48B (Production Scale)

```yaml
model:
  d_model: 4096
  n_layers: 48
  vocab_size: 128256
  max_seq_len: 1048576
  ssm:
    d_state: 256
    d_conv: 4
    expand: 2
    use_wkv: true
    use_delta_net: true
  attention:
    n_heads: 32
    d_kv: 128
    mla_latent_dim: 1024
    use_irope: true
  retrieval:
    num_experts: 256
    num_active_experts: 8
    use_engram: true
    engram_dim: 512
  router:
    top_k_pathways: 2
    use_thinking_toggle: true
    bias_lr: 0.01
    aux_loss_weight: 0.0
  output:
    use_mtp: true
    mtp_num_tokens: 4
    use_flow_matching: true
    fm_num_steps: 4
training:
  batch_size: 512
  learning_rate: 1.0e-4
  weight_decay: 0.1
  warmup_steps: 8000
  max_steps: 500000
  grad_clip: 1.0
  fp8_enabled: true
  precision: bf16
hardware:
  device: auto
  compile_model: true
```

---

*This documentation covers Losion v2.0.0. Hyperparameters and recommendations
may evolve with ongoing development. For questions, see the [ARCHITECTURE.md](ARCHITECTURE.md)
or open an issue on GitHub.*
