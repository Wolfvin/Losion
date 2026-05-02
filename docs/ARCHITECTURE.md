# Losion Architecture — Tri-Jalur Router

> **Comprehensive technical reference** for the Losion open-source AI framework.
> This document covers every component in detail, with mathematical foundations,
> implementation specifics, and design rationale — written for both human
> researchers and AI agents.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Jalur 1: SSM Terpadu](#3-jalur-1-ssm-terpadu)
4. [Jalur 2: Attention + Compression](#4-jalur-2-attention--compression)
5. [Jalur 3: Specialized Retrieval](#5-jalur-3-specialized-retrieval)
6. [Adaptive Router](#6-adaptive-router)
7. [Reasoning Engine](#7-reasoning-engine)
8. [Elastic Inference](#8-elastic-inference)
9. [Output Pipeline](#9-output-pipeline)
10. [Evoformer Integration](#10-evoformer-integration)
11. [Advanced Training Techniques](#11-advanced-training-techniques)
12. [Advanced Memory & Data Pipeline](#12-advanced-memory--data-pipeline)
13. [Advanced RLHF](#13-advanced-rlhf)
14. [Parameter Computation](#14-parameter-computation)
15. [Design Decisions & Justification](#15-design-decisions--justification)
16. [Comparison with Other Architectures](#16-comparison-with-other-architectures)

---

## 1. Overview

Losion is a generative AI architecture built on the **Tri-Jalur Router** paradigm — three complementary computational pathways whose contributions are dynamically weighted per-token by an adaptive router. The name *Tri-Jalur* (Indonesian: "Three Pathways") reflects the core insight that no single computational primitive — whether attention, state-space modeling, or retrieval — is optimal for all tokens in all contexts.

### Why Three Pathways?

Traditional transformer architectures (e.g., GPT) apply a single uniform mechanism — self-attention — to all tokens. This is suboptimal because:

1. **Sequential dependencies** (syntax, temporal patterns) do not require O(n²) attention; they are better modeled by O(n) state-space models.
2. **Reasoning** (logical inference, multi-step comparison) genuinely benefits from full attention but is computationally expensive.
3. **Factual knowledge** is better served by explicit retrieval (MoE + engram) than by distributing facts across dense attention weights.

Losion resolves this by assigning each type of computation to a pathway optimized for it:

| Pathway | Name | Optimized For | Complexity | Key Innovation |
|---------|------|---------------|------------|-----------------|
| 1 | SSM Terpadu | Long-range sequential dependencies | O(n) | Mamba-2 + RWKV-7 + Gated DeltaNet interleaved |
| 2 | Attention + Compression | Reasoning with memory efficiency | O(n·d_latent) | MLA compression (8× savings), iRoPE, Pairformer |
| 3 | Specialized Retrieval | Factual & domain-specific knowledge | O(n·k) sparse | MoE + Engram Memory with Expert Choice routing |

### Core Design Principles

- **Hardware-Agnostic**: Pure PyTorch — runs on NVIDIA (CUDA) and AMD (ROCm) without code changes. `torch.compile()` provides graph optimization without custom kernels.
- **Aux-Loss-Free**: Router uses bias-based routing (DeepSeek-V3 style) — no auxiliary loss needed for load balancing.
- **Adaptive Computation**: Router adjusts compute per-token based on complexity; thinking mode activates deeper processing.
- **Memory-Efficient**: MLA KV compression (8× reduction) + SSM linear recurrence + progressive KV compression.
- **Scalable**: From 1B (prototype) to 48B+ (production) with identical architecture, varying only `d_model`, `n_layers`, and expert counts.
- **Inference-Scalable**: MCTS, parallel thinking, and neuro-symbolic verification allow trading compute for quality at inference time.

> **Agent Context:** Losion is configured via `LosionConfig` (see `losion/config.py`). Sub-configs: `SSMConfig`, `AttentionConfig`, `RetrievalConfig`, `RouterConfig`, `ReasoningConfig`, `ElasticConfig`, `OutputConfig`, `TrainingConfig`, `HardwareConfig`. Entry point: `LosionModel` (see `losion/models/losion_model.py`).

---

## 2. Architecture Diagram

Complete data flow through a single `LosionLayer`:

```
                            INPUT x [B, S, d_model]
                                       │
                      ┌────────────────┼────────────────┐
                      │                │                 │
                ┌─────▼─────┐   ┌─────▼─────┐   ┌──────▼──────┐
                │  SSM Norm  │   │ Attn Norm  │   │ Retr. Norm  │
                │  (RMSNorm) │   │ (RMSNorm)  │   │  (RMSNorm)  │
                └─────┬─────┘   └─────┬─────┘   └──────┬──────┘
                      │                │                 │
           ┌──────────▼──────────┐     │                 │
           │   JALUR 1: SSM      │     │                 │
           │   Terpadu Layer     │     │                 │
           │  ┌───────────────┐  │     │                 │
           │  │ Interleaving  │  │     │                 │
           │  │ Scheduler     │  │     │                 │
           │  │ (4:1:1 ratio) │  │     │                 │
           │  └───────┬───────┘  │     │                 │
           │          │          │     │                 │
           │  ┌───────▼───────┐  │     │                 │
           │  │  ┌─────────┐  │  │     │                 │
           │  │  │ Mamba-2  │  │  │     │                 │
           │  │  │   SSD    │──┤  │     │                 │
           │  │  └─────────┘  │  │     │                 │
           │  │  ┌─────────┐  │  │     │                 │
           │  │  │ RWKV-7  │  │  │     │                 │
           │  │  │   WKV   │──┤  │     │                 │
           │  │  └─────────┘  │  │     │                 │
           │  │  ┌─────────┐  │  │     │                 │
           │  │  │ Gated   │  │  │     │                 │
           │  │  │DeltaNet │──┤  │     │                 │
           │  │  └─────────┘  │  │     │                 │
           │  └───────┬───────┘  │     │                 │
           │          │          │     │                 │
           │    ssm_out [B,S,D]  │     │                 │
           └──────────┬──────────┘     │                 │
                      │                │                 │
                      │    ┌───────────▼───────────┐     │
                      │    │  JALUR 2: Attention +  │     │
                      │    │     Compression        │     │
                      │    │  ┌─────────────────┐  │     │
                      │    │  │  Adaptive       │  │     │
                      │    │  │  Interleaving   │  │     │
                      │    │  │  (Local/Global) │  │     │
                      │    │  └────────┬────────┘  │     │
                      │    │           │           │     │
                      │    │  ┌────────▼────────┐  │     │
                      │    │  │       MLA       │  │     │
                      │    │  │  (KV Latent     │  │     │
                      │    │  │   Compression)  │  │     │
                      │    │  └────────┬────────┘  │     │
                      │    │           │           │     │
                      │    │  ┌────────▼────────┐  │     │
                      │    │  │      iRoPE      │  │     │
                      │    │  │  (Interleaved   │  │     │
                      │    │  │   RoPE/NoPE)    │  │     │
                      │    │  └────────┬────────┘  │     │
                      │    │           │           │     │
                      │    │  ┌────────▼────────┐  │     │
                      │    │  │  Pairformer     │  │     │
                      │    │  │  (thinking only)│  │     │
                      │    │  └────────┬────────┘  │     │
                      │    │           │           │     │
                      │    │  ┌────────▼────────┐  │     │
                      │    │  │  SwiGLU FFN     │  │     │
                      │    │  └────────┬────────┘  │     │
                      │    │           │           │     │
                      │    │   attn_out [B,S,D]    │     │
                      │    └───────────┬───────────┘     │
                      │                │                 │
                      │                │    ┌────────────▼────────────┐
                      │                │    │  JALUR 3: Specialized   │
                      │                │    │      Retrieval          │
                      │                │    │  ┌──────────────────┐   │
                      │                │    │  │  Engram Memory   │   │
                      │                │    │  │  (Hash-based     │   │
                      │                │    │  │   Fact Store)    │   │
                      │                │    │  └────────┬─────────┘   │
                      │                │    │           │             │
                      │                │    │  ┌────────▼─────────┐   │
                      │                │    │  │  MoE Specialist  │   │
                      │                │    │  │  Pool            │   │
                      │                │    │  │  (Expert Choice  │   │
                      │                │    │  │   or Token Choice│   │
                      │                │    │  │   routing)       │   │
                      │                │    │  └────────┬─────────┘   │
                      │                │    │           │             │
                      │                │    │  ┌────────▼─────────┐   │
                      │                │    │  │  Gated Fusion    │   │
                      │                │    │  │  (Engram + MoE)  │   │
                      │                │    │  └────────┬─────────┘   │
                      │                │    │           │             │
                      │                │    │  retrieval_out [B,S,D] │
                      │                │    └───────────┬─────────────┘
                      │                │                │
                ┌─────▼────────────────▼────────────────▼─────┐
                │              MERGE + ROUTING                  │
                │                                              │
                │  ┌─────────────────────────────────────┐    │
                │  │        Adaptive Router               │    │
                │  │  ┌──────────┐  ┌──────────────────┐  │    │
                │  │  │  Bias    │  │  Thinking        │  │    │
                │  │  │  Router  │  │  Toggle          │  │    │
                │  │  └────┬─────┘  └────┬─────────────┘  │    │
                │  │       │             │                 │    │
                │  │       └──────┬──────┘                 │    │
                │  │              │                        │    │
                │  │     routing_weights [B, S, 3]         │    │
                │  │     [w_ssm, w_attn, w_retr]           │    │
                │  └──────────────┬────────────────────────┘    │
                │                 │                              │
                │  merged = w_ssm·ssm_out +                     │
                │          w_attn·attn_out +                     │
                │          w_retr·retrieval_out                  │
                │                                              │
                │  output = x + merged  (residual)              │
                │  output = PostMergeNorm(output)               │
                └──────────────────────┬────────────────────────┘
                                       │
                            OUTPUT [B, S, d_model]
                                       │
                      ┌────────────────┼────────────────┐
                      │                │                 │
                ┌─────▼─────┐   ┌─────▼─────┐   ┌──────▼──────┐
                │  Reasoning │   │  Elastic   │   │   Output    │
                │  Engine    │   │  Inference │   │   Pipeline  │
                │ (MCTS +    │   │(Matryoshka)│   │(MTP + FM +  │
                │  Parallel  │   │            │   │ Diffusion)  │
                │  Thinking +│   │            │   │             │
                │  NeuroSym) │   │            │   │             │
                └───────────┘   └───────────┘   └──────┬──────┘
                                                       │
                                            LOGITS [B, S, V]
```

> **Agent Context:** The `LosionLayer` class in `losion/models/losion_model.py` implements this entire flow. Each layer instantiates `SSMTerpaduLayer`, `AttentionKompresiLayer`, `RetrievalTerpaduLayer`, and `AdaptiveRouter`. The `LosionModel` stacks `n_layers` of these and adds embedding + final norm + output pipeline.

---

## 3. Jalur 1: SSM Terpadu

Jalur 1 is the sequential processing backbone of Losion. Instead of using a single SSM variant, Losion combines three innovations in one coherent `SSMTerpaduLayer` with a configurable interleaving pattern.

### 3.1 Mamba-2 SSD (State Space Duality)

Mamba-2 SSD implements the **State Space Duality** principle from Gu & Dao (2024). The core insight is the duality between recurrent and parallel representations of state-space models.

**Discrete SSM formulation:**

$$h_t = \bar{A} \cdot h_{t-1} + \bar{B} \cdot x_t \quad \text{(state transition)}$$

$$y_t = \bar{C} \cdot h_t \quad \text{(output projection)}$$

where:
- $h_t \in \mathbb{R}^N$ is the hidden state at time $t$
- $\bar{A} \in \mathbb{R}^{N \times N}$ is the diagonal transition matrix
- $\bar{B} \in \mathbb{R}^{N \times 1}$ is the input matrix
- $\bar{C} \in \mathbb{R}^{1 \times N}$ is the output matrix
- $x_t \in \mathbb{R}$ is the scalar input (per channel)

**Parallel-Recurrent Duality:**

In parallel mode (training), the output is computed as a structured convolution:

$$Y = \text{SSM}(\bar{A}, \bar{B}, \bar{C}) * X$$

where $\text{SSM}(\cdot)$ forms a structured lower-triangular matrix:

$$S = \begin{bmatrix} \bar{C}\bar{A}^0\bar{B} & 0 & 0 & \cdots \\ \bar{C}\bar{A}^1\bar{B} & \bar{C}\bar{A}^0\bar{B} & 0 & \cdots \\ \bar{C}\bar{A}^2\bar{B} & \bar{C}\bar{A}^1\bar{B} & \bar{C}\bar{A}^0\bar{B} & \cdots \\ \vdots & \vdots & \vdots & \ddots \end{bmatrix}$$

In recurrent mode (inference), computation is O(1) per token.

**Discretization with step size $dt$:**

$$\bar{A} = \exp(dt \cdot A), \quad \bar{B} = dt \cdot B$$

**Chunking for efficiency:**

SSD uses chunk-based computation to combine the advantages of both modes:
- **Within chunk**: parallel computation via GPU-efficient matrix multiplication
- **Across chunks**: recurrent state passing

Default chunk size: 256 tokens (balances parallelism and memory usage).

```python
# Simplified SSD chunk scan (from lossion/core/ssm/mamba2.py)
dA = torch.exp(dt.unsqueeze(-1) * A)   # discrete transition
dB = dt.unsqueeze(-1) * B               # discrete input

for t in range(seq_len):
    h = h * dA[:, t, :]                           # decay state
    h = h + x_seq[:, t, :].unsqueeze(-1) * dB[:, t, :]  # add input
    y_t = torch.sum(h * C[:, t, :].unsqueeze(1), dim=-1) # output
```

> **Agent Context:** Implemented in `Mamba2SSD` class (`lossion/core/ssm/mamba2.py`). Key params: `d_model`, `d_state` (default 128), `d_conv` (default 4), `expand` (default 2), `chunk_size` (default 256). Supports `forward()` (training) and `forward_inference()` (O(1) per token).

### 3.2 RWKV-7 WKV (Weighted Key-Value)

RWKV-7 implements **WKV recurrence** — an evolution of attention that replaces softmax scoring with an exponentially-weighted moving average.

**WKV recurrence math:**

$$\text{wkv}_t = \frac{a_t \cdot \text{wkv}_{t-1} + u \cdot k_t \odot v_t}{a_t \cdot \text{sum}_{t-1} + u \cdot k_t \odot k_t + \epsilon}$$

$$\text{sum}_t = a_t \cdot \text{sum}_{t-1} + k_t \odot k_t$$

where:
- $a_t = \exp(w_t)$ is the data-dependent decay factor ($w_t < 0$, so $a_t \in (0,1)$)
- $u$ is the learned position bonus (current-token emphasis)
- $k_t, v_t$ are key and value vectors per head
- $\epsilon = 10^{-8}$ for numerical stability

**Key advantages:**
- **O(1) inference**: Only stores `(wkv_state, sum_state)` — not the entire KV cache
- **Unbounded context**: State accumulates without sequence length limit
- **Explicit forgetting**: Decay factor $a_t$ allows selective "forgetting" of old information
- **Token-shift mixing**: Input is mixed with its predecessor (`mixed = input + shifted`) for temporal smoothness

```python
# WKV parallel computation (from lossion/core/ssm/rwkv7.py)
decay = torch.exp(w)  # w < 0, so decay ∈ (0, 1)
wkv_state = wkv_state * d_t    # decay old state
sum_state = sum_state * d_t
kv_t = k_t * v_t
numerator = wkv_state + u * kv_t
denominator = sum_state + u * (k_t * k_t) + 1e-8
wkv_val = numerator / denominator
```

> **Agent Context:** Implemented in `RWKV7WKV` class (`lossion/core/ssm/rwkv7.py`). Key params: `d_model`, `d_head` (default 64), `n_heads`. Outputs `(output, (wkv_state, sum_state))` tuple. The `u` parameter is learned per-head.

### 3.3 Gated DeltaNet

Gated DeltaNet implements **in-context learning** via the delta rule. Unlike standard attention that only adds new information, DeltaNet can **correct** previously stored associations.

**State update formula (delta rule):**

$$V_t = \alpha_t \cdot V_{t-1} + \beta_t \cdot (k_t^\top \cdot v_t)$$

where:
- $V_t \in \mathbb{R}^{d_h \times d_h}$ is the value matrix state (per head)
- $k_t, v_t$ are current key and value vectors
- $\beta_t = \sigma(\text{beta\_proj}(x))$ is the update gate $\in (0,1)$
- $\alpha_t = \sigma(\text{alpha\_proj}(x) + \alpha_\text{offset})$ is the decay gate $\in (0,1)$

This formula says: **blend the previous state with a new key-value association**, where the blend ratio is controlled by learned gates.

**Output with gating:**

$$\text{output}_t = g_t \odot (q_t^\top \cdot V_t)$$

where $g_t$ is the output gate controlling how much DeltaNet information is used.

**Chunk-based parallel computation:**

For training, DeltaNet uses chunk-based computation:
1. Intra-chunk: linear attention with causal mask
2. Inter-chunk: state propagation via delta rule
3. Interpolation: position-weighted blend of state output and intra-chunk attention

```python
# Delta rule update (from lossion/core/ssm/delta_net.py)
kv_outer = torch.matmul(k_t.transpose(-2, -1), v_t)  # [B, H, D, D]
new_state = alpha * state + beta * kv_outer
y = torch.matmul(q_t, new_state)  # query the updated state
```

> **Agent Context:** Implemented in `GatedDeltaNet` class (`lossion/core/ssm/delta_net.py`). Key params: `d_model`, `n_heads`, `d_head`, `chunk_size`. Uses QK-norm (RMSNorm on queries and keys) for training stability. Alpha initialized near 1.0 (preserve state) via `alpha_offset`.

### 3.4 Interleaving Pattern

The three SSM sub-layers are interleaved with a default **4:1:1** ratio:
- **4 blocks Mamba-2 SSD**: Primary parallel sequential processing (GPU-aware)
- **1 block RWKV-7 WKV**: Dynamic state evolution with explicit forgetting
- **1 block Gated DeltaNet**: In-context learning and state correction

The `InterleavingScheduler` distributes WKV and DeltaNet blocks evenly among SSD blocks:

```
Block:  0    1    2    3    4    5
Type:   SSD  SSD  WKV  SSD  SSD  Delta
```

**Dynamic routing mode**: When routing weights are provided, all three sub-layers process the input simultaneously and outputs are blended:

$$\text{blended} = w_\text{ssd} \cdot y_\text{ssd} + w_\text{wkv} \cdot y_\text{wkv} + w_\text{delta} \cdot y_\text{delta}$$

This allows the adaptive router to dynamically adjust the SSM computation mix per token.

> **Agent Context:** `InterleavingScheduler` in `lossion/core/ssm/ssm_layer.py`. Schedule is built from the ratio tuple via `_build_schedule()`. The `SSMTerpaduLayer` manages the interleaving and delegates to sub-layers. State is tracked via `SSMState(ssd_state, wkv_state, delta_state)`.

---

## 4. Jalur 2: Attention + Compression

Jalur 2 is the reasoning engine of Losion. It combines MLA (memory-efficient attention), iRoPE (interleaved position encoding), adaptive local/global interleaving, and Pairformer (triangular attention for deep reasoning).

### 4.1 MLA (Multi-head Latent Attention)

MLA, adapted from DeepSeek-V3, compresses KV-cache representations into a **lower-dimensional latent space**, dramatically reducing memory requirements without significant quality loss.

**KV Compression math:**

Instead of storing full KV pairs:

$$K = [k_1, \ldots, k_T] \in \mathbb{R}^{T \times n_h \times d_{kv}}, \quad V = [v_1, \ldots, v_T] \in \mathbb{R}^{T \times n_h \times d_{kv}}$$

MLA stores a **latent representation**:

$$c_{kv} = W_\text{kv\_compress}(x) \in \mathbb{R}^{d_\text{latent}}$$

When needed, key and value are reconstructed:

$$k = W_\text{k\_up}(c_{kv}) \in \mathbb{R}^{n_h \times d_{kv}}, \quad v = W_\text{v\_up}(c_{kv}) \in \mathbb{R}^{n_h \times d_{kv}}$$

**Memory savings:**

For Losion-7B with `n_heads=16`, `d_kv=128`, `mla_latent_dim=512`:
- Full KV: $2 \times 16 \times 128 = 4{,}096$ dimensions per token per layer
- MLA latent: $512$ dimensions per token per layer
- **Compression: 8×**

The savings ratio is computed as:

$$\text{savings} = 1 - \frac{d_\text{latent}}{2 \cdot n_h \cdot d_{kv}}$$

**MLA also applies QK-norm** (RMSNorm on queries and keys before attention) for training stability, and conditional RoPE application (controlled by iRoPE).

```python
# MLA compression + reconstruction (from lossion/core/attention/mla.py)
kv_latent = self.W_kv_compress(x)           # compress to latent
k = self.W_k_up(full_kv_latent)             # reconstruct key
v = self.W_v_up(full_kv_latent)             # reconstruct value
k = self.k_norm(k.transpose(1,2)).transpose(1,2)  # QK-norm
```

> **Agent Context:** Implemented in `MLA` class (`lossion/core/attention/mla.py`). KV cache is `MLAKVCache` storing latent vectors. Key params: `d_model`, `n_heads`, `d_kv`, `mla_latent_dim`, `use_rope`, `use_flash_attn`. Memory savings computed as `self.memory_savings_ratio`.

### 4.2 iRoPE (Interleaved Rotary Position Embeddings)

iRoPE, adapted from Llama 4, interleaves between layers using **RoPE** (explicit positional encoding) and **NoPE** (no positional encoding, relying on learned attention bias).

**Pattern with 3:1 ratio (default):**

```
Layer:  0    1    2    3    4    5    6    7    8    ...
RoPE:   ✓    ✓    ✓    ✗    ✓    ✓    ✓    ✗    ✓    ...
```

**Justification:**
1. **RoPE** provides relative positional information critical for reasoning about token order
2. **NoPE** allows the model to rely on learned attention biases, which are more flexible for long contexts and avoid positional extrapolation issues
3. **Interleaving** provides both benefits simultaneously — position-aware reasoning where needed, position-agnostic flexibility elsewhere

**RoPE math:**

$$\text{RoPE}(x, \theta) = x \odot \cos(\theta) + \text{rotate\_half}(x) \odot \sin(\theta)$$

where $\theta$ is the position-dependent frequency vector.

> **Agent Context:** `InterleavedRoPE` class in `lossion/core/attention/irope.py`. Method `should_use_rope(layer_idx)` returns boolean. Frequencies precomputed via `precompute_rope_freqs()`.

### 4.3 Adaptive Interleaving (Local/Global Attention)

Adaptive Interleaving controls the ratio between **local attention** (sliding window) and **global attention** (full sequence), adapted from Gemma 3.

**Interleaving ratios:**

| Mode | Local:Global | Description |
|------|-------------|-------------|
| Base | 5:1 | 5 local layers per 1 global layer — memory efficient |
| Thinking | 2:1 | 2 local layers per 1 global layer — deeper reasoning |

In **non-thinking** mode, most layers use local attention (sliding window 1024 tokens), with only 1 in 6 using global attention. This saves memory significantly.

In **thinking** mode, the ratio shifts to 2:1, providing more global attention layers for full-context understanding during complex reasoning.

> **Agent Context:** `AdaptiveInterleaving` class in `lossion/core/attention/interleaving.py`. Method `is_global_layer(layer_idx, thinking_mode)` determines layer type. Method `get_effective_ratio()` supports dynamic adjustment via routing weights.

### 4.4 Pairformer — AlphaFold3-style Triangular Attention

The Pairformer, adapted from AlphaFold3's Pairformer module, models **pairwise relationships** between tokens using triangular attention. It activates only during thinking mode.

**Architecture:**

1. **Pair Representation**: For each pair $(i, j)$, compute a feature vector from the concatenation of single representations:

$$z_{ij} = W_\text{pair}([s_i; s_j]) + b_\text{pair}$$

2. **Triangular Updates** (outgoing and incoming):
   - **Outgoing**: Update $z_{ij}$ based on $z_{ik} \odot z_{jk}$ for all $k$ — captures transitive relationships
   - **Incoming**: Update $z_{ij}$ based on $z_{ki} \odot z_{kj}$ for all $k$

3. **Single ← Pair Conditioning**: Aggregate pair information and inject into single representation:

$$s_i \leftarrow s_i + W_\text{p2s}(\text{mean}_j(z_{ij}))$$

4. **Pair-biased Attention**: Pair representation biases the attention scores:

$$\text{attn\_weights} = QK^T / \sqrt{d} + 0.1 \cdot \text{mean}(z, \text{dim}=-1)$$

**Complexity**: $O(n^2 \cdot d_\text{pair})$ — mitigated via chunking (`pairformer_chunk_size`).

> **Agent Context:** `PairwiseAttentionLayer` in `lossion/core/attention/pairformer.py`. Sub-components: `PairRepresentation`, `TriangularUpdate`. Only active during thinking mode. Config: `d_pair` (default 64), `pairformer_chunk_size` (default 256).

### 4.5 Thinking Mode

Thinking mode is activated by the router when it detects that input requires deep reasoning. Effects:

1. Interleaving ratio changes from 5:1 to 2:1 (more global attention)
2. Pairformer activates for pairwise relationship modeling
3. KV-cache retains more context for cross-referencing
4. Reasoning engine (MCTS + parallel thinking) may engage

---

## 5. Jalur 3: Specialized Retrieval

Jalur 3 handles **factual knowledge** and **domain-specific knowledge** through a layered architecture combining MoE (Mixture of Experts) with Engram Memory.

### 5.1 MoE (Mixture of Experts)

MoE uses **sparse gating** — only a fraction of experts are activated per token.

**Token-choice routing math (DeepSeek-V3 style):**

$$g(x) = \text{TopK}(\text{softmax}(W_\text{gate} \cdot x + b))$$

$$\text{output} = \sum_i g_i(x) \cdot \text{Expert}_i(x)$$

where:
- $W_\text{gate} \in \mathbb{R}^{N_\text{experts} \times d_\text{model}}$ is the routing matrix
- $b \in \mathbb{R}^{N_\text{experts}}$ is the learnable bias (updated directly by gradients, not aux loss)
- TopK retains only `num_active_experts` experts

**Expert-choice routing (Google Research style):**

Instead of tokens choosing experts, experts choose tokens:

$$S_{ij} = \text{sim}(\text{token}_i, \text{expert}_j)$$

For each expert $j$: select top-$K$ tokens with highest affinity scores.

**Guaranteed load balancing**: Every expert processes exactly $K$ tokens — no auxiliary loss needed.

**Configuration per model size:**

| Model | `num_experts` | `num_active_experts` | Sparsity |
|-------|-------------|---------------------|----------|
| 1B | 16 | 2 | 87.5% |
| 7B | 64 | 4 | 93.75% |
| 48B | 256 | 8 | 96.88% |

**Shared expert**: Always active, providing baseline general knowledge independent of routing.

> **Agent Context:** Two routing strategies: `MoERetrieval` (token-choice, `lossion/core/retrieval/moe.py`) and `ExpertChoiceMoE` (expert-choice, `lossion/core/retrieval/expert_choice.py`). Config field `routing_strategy` selects between them. Shared expert controlled by `use_shared_expert`.

### 5.2 Engram Memory

Engram Memory is a **hash-based fact store** that stores factual knowledge explicitly, analogous to human associative memory.

**Architecture:**

$$\text{subject\_string} \xrightarrow{\text{hash}} \text{bucket\_index} \xrightarrow{\text{embedding\_lookup}} \text{retrieval}$$

Components:
1. **Hash Function**: Converts subject string to bucket index
2. **Embedding Table**: `nn.Embedding(num_buckets, embedding_dim)` — stores engram vectors
3. **Retrieval**: Look up matching embedding and combine with input representation

**Advantages:**
- **O(1) retrieval**: Hash-based access, no linear search
- **Explicit storage**: Factual knowledge stored explicitly, not distributed across weights
- **Updateable**: New facts can be inserted without retraining via `insert_fact()` and `insert_facts_batch()`

> **Agent Context:** `EngramMemory` class in `lossion/core/retrieval/engram.py`. Key methods: `retrieve(x, subject_strings)`, `insert(subject, embedding)`, `insert_batch(subjects, embeddings)`, `get_stats()`. Config: `num_buckets` (default 1,000,000), `engram_embedding_dim` (default 256).

### 5.3 Gated Fusion

Engram and MoE outputs are combined via **gated fusion**:

$$\text{gate\_logits} = W_\text{fusion}([\text{engram\_out}; \text{moe\_out}])$$

$$w = \text{softmax}(\text{gate\_logits}) \quad \in \mathbb{R}^2$$

$$\text{fused} = w_0 \cdot \text{engram\_out} + w_1 \cdot \text{moe\_out}$$

The gate learns when to rely on Engram (static facts) vs MoE (dynamic/domain-specific knowledge).

Three fusion modes are supported:
- **"gated"** (default): Learned gate controls fusion — most flexible
- **"additive"**: Simple weighted sum with learned scalar $\alpha$
- **"concat"**: Concatenation + projection — most expressive but most parameters

> **Agent Context:** Fusion implemented in `RetrievalTerpaduLayer._fuse()` (`lossion/core/retrieval/retrieval_layer.py`). Mode controlled by `engram_fusion_mode` config. Output: `RetrievalOutput` dataclass with `fusion_weights [B, S, 2]`.

---

## 6. Adaptive Router

The Adaptive Router is the "brain" of the Tri-Jalur architecture, combining BiasRouter (computational allocation) with ThinkingToggle (complexity detection).

### 6.1 BiasRouter — Aux-Loss-Free Routing

BiasRouter uses **learnable bias** updated directly by gradients, rather than auxiliary loss for load balancing.

**Math:**

$$\text{logits} = W_\text{router} \cdot x + b_\text{bias}$$

$$\text{weights} = \text{softmax}(\text{logits})$$

**Bias update (per training step):**

$$b_\text{bias} \leftarrow b_\text{bias} - \eta_\text{bias} \cdot \frac{\partial L}{\partial b_\text{bias}}$$

The bias is updated directly by the main loss gradient — no auxiliary loss term needed.

**Why no aux loss?**

Auxiliary loss (as in Switch Transformer or GShard) adds a separate loss term to encourage load balancing. Problems:
1. Requires hyperparameter tuning (aux loss weight)
2. Can interfere with the main loss optimization
3. Doesn't always produce optimal load balancing

Bias-based routing relies on the main loss gradient flowing directly to the bias, providing a more natural signal. This approach is adapted from DeepSeek-V3.

> **Agent Context:** `BiasRouter` class in `lossion/core/router/bias_router.py`. Key config: `d_model`, `num_pathways` (default 3), `top_k_pathways` (default 2), `bias_lr`. Output: `PathwayRoutingInfo` with routing weights and selected pathways.

### 6.2 ThinkingToggle — Complexity Detection

ThinkingToggle analyzes each token and determines whether deep reasoning is required.

**Mechanism:**

$$\text{complexity} = \sigma(W_\text{complexity} \cdot x) \quad \in [0, 1]$$

$$\text{task\_type} = \arg\max(W_\text{task} \cdot x) \quad \in \{\text{sequential}, \text{reasoning}, \text{factual}\}$$

$$\text{depth\_multiplier} = \begin{cases} 1.0 + \text{complexity} & \text{if complexity} > \text{threshold} \\ 1.0 & \text{otherwise} \end{cases}$$

**Effects of Thinking Mode:**

| Aspect | Non-Thinking | Thinking |
|--------|-------------|----------|
| Routing weights | Jalur 1 dominant | Jalur 2+3 activated |
| Interleaving ratio | 5:1 (local:global) | 2:1 (local:global) |
| Pairformer | Inactive | Active |
| Depth multiplier | 1.0 | 1.0–2.0 |
| Example tokens | "the", "is", "," | "therefore", "because", "?" |

> **Agent Context:** `ThinkingToggle` class in `lossion/core/router/thinking_toggle.py`. Output: `ThinkingAssessment` with `mode` (THINKING/NON_THINKING), `complexity_score`, `dominant_task`, `confidence`, `depth_multiplier`. Can be forced via `set_force_mode()`.

### 6.3 Thinking-Weight Adjustment

After BiasRouter and ThinkingToggle produce their respective outputs, routing weights are adjusted:

```python
# 1. BiasRouter → routing_weights [B, S, 3]
# 2. ThinkingToggle → complexity_score [B, S]
# 3. Adjustment via learned network:
adjuster_input = cat([routing_weights, complexity_score.unsqueeze(-1)], dim=-1)
adjustment = thinking_adjuster(adjuster_input)     # [B, S, 3]
adjusted = routing_weights + 0.1 * adjustment      # residual with small scale
adjusted = softmax(adjusted, dim=-1)

# 4. Mode-specific boost:
if non_thinking:
    adjusted += [0.3, -0.15, -0.15]   # Boost Jalur 1
else:
    adjusted += [-0.15, 0.15, 0.15]   # Boost Jalur 2+3
adjusted = softmax(adjusted, dim=-1)
```

The adjustment uses a residual connection with a small scale factor (0.1) to prevent the thinking signal from overwhelming the base routing. Mode-specific boosts are applied in logit space before final softmax normalization.

> **Agent Context:** Full routing in `AdaptiveRouter` class (`lossion/core/router/router.py`). The `thinking_adjuster` is an `nn.Sequential(Linear(4,6), SiLU, Linear(6,3))`. Pathway priors registered as buffer: `[0.4, 0.3, 0.3]`.

---

## 7. Reasoning Engine

Losion integrates three DeepMind-inspired reasoning techniques for inference-time compute scaling.

### 7.1 MCTS (Monte Carlo Tree Search)

AlphaZero-inspired tree search for reasoning. Instead of relying solely on single-pass generation, MCTS explores multiple reasoning paths and selects the best.

**UCB (Upper Confidence Bound) for selection:**

$$\text{UCB}(s, a) = Q(s, a) + c_\text{puct} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

where:
- $Q(s, a)$ = average value of action $a$ at state $s$
- $P(s, a)$ = prior probability from policy network
- $N(s)$ = total visit count of parent state
- $N(s, a)$ = visit count of action $a$
- $c_\text{puct}$ = exploration constant (default 1.5)

**MCTS cycle:**
1. **Selection**: Traverse tree via UCB from root to leaf
2. **Expansion**: Add children from leaf node using policy network priors
3. **Evaluation**: Value network estimates quality of leaf state
4. **Backpropagation**: Update visit counts and values from leaf to root

**Adaptive compute budget:**

$$\text{budget} = \text{base} + (\text{max} - \text{base}) \cdot \text{complexity}^2$$

More complex inputs get exponentially more simulations.

> **Agent Context:** `MCTSReasoner` class in `lossion/core/reasoning/mcts.py`. Sub-components: `ValueNetwork` (3-layer MLP → tanh), `PolicyNetwork` (2-layer → logits). Config: `MCTSConfig` with `num_simulations` (default 64), `c_puct` (default 1.5), `max_depth` (default 10).

### 7.2 Parallel Thinking — Gemini Deep Think Style

Explores multiple reasoning paths simultaneously, then selects the best.

**Architecture:**

1. **Path Diversification**: Each path receives a unique learned perturbation:

$$\text{path}_i = x + 0.1 \cdot \text{path\_embedding}_i$$

2. **Path Evaluation**: Each path is scored on three dimensions:
   - **Value**: Quality of solution (via `PathEvaluator`)
   - **Confidence**: Model's certainty
   - **Novelty**: Unique perspective contribution

3. **Consistency Check**: Cross-path consistency measured via learned projection:

$$\text{consistency}_i = \text{ConsistencyChecker}(\text{path}_i, \text{all\_paths})$$

4. **Final Score**:

$$\text{score}_i = 0.5 \cdot \text{value}_i + 0.35 \cdot \text{consistency}_i + 0.15 \cdot \text{novelty}_i$$

**Selection strategies:**
- `BEST_OF_N`: Select path with highest final score
- `MAJORITY_VOTE`: Select path with highest consistency (self-consistency)
- `WEIGHTED_MERGE`: Softmax-weighted merge of all paths
- `TOURNAMENT`: Elimination tournament between paths

**Adaptive budget:**

$$\text{num\_paths} = \text{min} + (\text{max} - \text{min}) \cdot \text{complexity}^{1.5}$$

> **Agent Context:** `ParallelThinker` class in `lossion/core/reasoning/parallel_thinking.py`. Sub-components: `PathEvaluator`, `ConsistencyChecker`. Learned `path_embeddings` parameter of shape `(num_paths, d_model)`. Strategies in `ThinkingStrategy` enum.

### 7.3 Neuro-Symbolic Verification — AlphaProof Style

Combines neural generation with symbolic verification for formally correct reasoning outputs.

**Architecture:**

1. **Symbolic Rule Engine**: `num_rules` (default 16) learned rule embeddings, each with an applicability checker and verification network:

$$\text{ver\_score}_r = \sigma(W_\text{verifier}([h_\text{output}; \text{rule\_embed}_r]))$$

2. **Error Localization**: Per-token error probability:

$$\text{error\_prob}_t = \sigma(W_\text{locator}([h_t; \text{rule\_embed}_\text{worst}]))$$

3. **Feedback Generation**: Correction signal for failed verification:

$$\text{feedback} = W_\text{feedback}([h_\text{output}; \text{rule\_embed}_\text{worst}])$$

4. **Iterative Revision Loop** (up to `max_revision_iterations`):
   - If VERIFIED → return
   - If NEEDS_REVISION → `new_output = RevisionNet([current; feedback])`
   - If PARTIAL → `new_output = current + 0.3 * RevisionNet([current; feedback])`

**Task gating**: A learned gate determines whether verification is needed for the current task type (math, code, logic → verify; creative, translation → skip).

> **Agent Context:** `NeuroSymbolicVerifier` class in `lossion/core/reasoning/neuro_symbolic.py`. Sub-component: `SymbolicRuleEngine`. Output: `VerificationResult` with `status` (VERIFIED/FAILED/PARTIAL/UNSURE/NEEDS_REVISION), `confidence`, `feedback`. Config: `num_rules` (default 16), `max_revision_iterations` (default 3), `verification_threshold` (default 0.8).

---

## 8. Elastic Inference

### 8.1 Matryoshka / MatFormer — One Weight Set, Multiple Submodels

Adapted from Gemma 3n / MatFormer (Google DeepMind, 2025), Matryoshka Nested Transformer enables one weight set to produce multiple valid submodels of different sizes.

**Nested FFN structure:**

Full FFN: $W_\text{gate} \in \mathbb{R}^{d_\text{ff} \times d}$, $W_\text{up} \in \mathbb{R}^{d_\text{ff} \times d}$, $W_\text{down} \in \mathbb{R}^{d \times d_\text{ff}}$

Submodel with factor $f$: $W_\text{gate}^{(f)} = W_\text{gate}[:f \cdot d_\text{ff}, :]$, etc.

Each submatrix is a valid model — all submodels are trained simultaneously via Matryoshka loss:

$$\mathcal{L}_\text{matryoshka} = \frac{w_\text{mat}}{|\mathcal{F}|} \sum_{f \in \mathcal{F}} \mathcal{L}(\text{submodel}_f, \text{target}_f)$$

where $\mathcal{F}$ is the set of granularity factors (default: `[0.25, 0.5, 0.75, 1.0]`).

**Adaptive sizing per token:**

A learned `size_selector` network predicts a score $\in [0, 1]$ from input, which is mapped to the nearest granularity factor. Simple tokens (function words, punctuation) can use smaller submodels; complex tokens (reasoning, rare words) use the full model.

**Mix'n'Match**: Different layers can use different granularity factors — e.g., early layers small (0.25), middle layers medium (0.5), late layers full (1.0).

```python
# Matryoshka forward (from losion/core/elastic/matryoshka.py)
if factor >= 1.0:
    gate = F.silu(self.gate_proj(x))
    up = self.up_proj(x)
    output = self.down_proj(gate * up)
else:
    gate_w = self.gate_proj.weight[:d_ff_active, :]
    up_w = self.up_proj.weight[:d_ff_active, :]
    down_w = self.down_proj.weight[:, :d_ff_active]
    gate = F.silu(F.linear(x, gate_w))
    up = F.linear(x, up_w)
    output = F.linear(gate * up, down_w)
```

> **Agent Context:** `MatryoshkaLayer` in `lossion/core/elastic/matryoshka.py`. `ElasticExtractor` utility for extracting submodels. Config: `MatryoshkaConfig` with `granularity_factors`, `matryoshka_loss_weight`, `use_adaptive`.

---

## 9. Output Pipeline

The output pipeline transforms hidden states into final token predictions through a multi-stage process.

### 9.1 MTP (Multi-Token Prediction)

MTP trains multiple prediction heads, each predicting at a different offset:

$$\text{Head}_k: \text{logits}_k = \text{LM\_head}_k(\text{hidden\_states}) \rightarrow \text{predict token } t+k+1$$

**Training benefit:** Forces hidden states to contain information about future tokens, not just the next token, producing richer representations.

**Inference benefit:** Speculative decoding — MTP heads predict multiple tokens simultaneously, which are then verified by the main model. This provides 2–3× speedup.

**MTP Loss:**

$$\mathcal{L}_\text{MTP} = \sum_{k=0}^{K-1} \frac{1}{k+1} \cdot \text{CE}(\text{logits}_k, \text{labels}[t+k+1:])$$

Loss weight decreases for farther offsets (1/1, 1/2, 1/3, ...).

> **Agent Context:** `MultiTokenPrediction` class in `lossion/core/output/mtp.py`. Config: `mtp_n_tokens` (default 3). Method `generate_speculative()` for speculative decoding.

### 9.2 Flow Matching

Flow Matching is an alternative generation method that smoothly interpolates from noise to data distribution:

$$x_t = (1-t) \cdot x_0 + t \cdot x_1 \quad \text{(linear interpolation)}$$

$$v_t = \frac{dx_t}{dt} = x_1 - x_0 \quad \text{(velocity field)}$$

The model learns the velocity field $v_\theta(x_t, t)$ and uses it for sampling:

$$x_{t+dt} = x_t + dt \cdot v_\theta(x_t, t)$$

Flow Matching is only activated on large models (48B+) due to significant computational overhead.

### 9.3 Diffusion Refinement — AlphaFold3-Inspired

Iterative denoising inspired by AlphaFold3's diffusion module. Instead of producing the final output directly, the model generates a coarse output and refines it.

**Forward process (noise addition):**

$$x_t = \alpha(t) \cdot x_0 + \sigma(t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\alpha(t) = \cos^2(t \cdot \pi/2)$ (cosine schedule) and $\sigma(t) = \sqrt{1 - \alpha(t)^2}$.

**Reverse process (denoising):** Iterative application of `DenoiserBlock` conditioned on context and time step:

$$x_{t-\Delta t} = \text{DenoiserBlock}(x_t, \text{context}, t)$$

**Training loss (epsilon prediction):**

$$\mathcal{L} = \|\epsilon - \hat{\epsilon}\|^2$$

**Integration with Losion**: Applied to the output of the Tri-Jalur pipeline before vocabulary projection. Starts from lightly noised model output (t=0.3) rather than pure noise, then iteratively refines.

```python
# Diffusion refinement (from lossion/core/output/diffusion_refinement.py)
t_start = torch.ones(batch_size, 1, device=device) * 0.3
current = self.add_noise(x, t_start)  # Light noise on model output
for step in range(n_steps):
    t = timesteps[step].expand(batch_size, 1)
    current = self.denoisers[step % len(self.denoisers)](current, context, t)
refined = self.output_proj(current)
refined = 0.7 * refined + 0.3 * x  # Residual blend
```

> **Agent Context:** `DiffusionRefinement` in `lossion/core/output/diffusion_refinement.py`. Sub-components: `NoiseScheduler`, `DenoiserBlock`. Config: `DiffusionRefinementConfig` with `num_steps` (default 4), `schedule_type` ("cosine"/"linear"). The full pipeline is orchestrated by `OutputPipeline` in `lossion/core/output/output_pipeline.py`.

---

## 10. Evoformer Integration

Losion integrates **Evoformer** concepts from AlphaFold2 across 5 levels of feedback:

### Level 1: Inter-Layer Routing Feedback
- Routing weights from layer $L-1$ are projected and fed back to layer $L$
- Projection: `evo_feedback_proj` (Linear → SiLU → Linear) maps 3-dim routing weights to $d_\text{model}$
- Gated combination: `evo_gate` (Linear → Sigmoid) controls feedback strength
- Implementation: `x = x + gate * feedback`

### Level 2: Cross-Pathway Communication
- Each pathway can access partial outputs from other pathways within the same layer
- Implemented via lightweight attention between pathway representations

### Level 3: Output Recycling
- Logits from the first forward pass are fed back to hidden states
- Hidden states are refined to produce better logits
- Default: 2 recycling iterations

### Level 4: Iterative Refinement
- Multiple rounds of output recycling (2–3 iterations)
- Each iteration improves output coherence
- Trade-off: quality vs. inference speed

### Level 5: Full Evoformer
- Bidirectional feedback between all components
- Still experimental — not fully implemented

> **Agent Context:** Level 1 is implemented directly in `LosionLayer` (`lossion/models/losion_model.py`) via `evo_feedback_proj` and `evo_gate`. Levels 2–5 are planned extensions.

---

## 11. Advanced Training Techniques

Seven techniques from DeepMind/Google AI research for more efficient training.

### 11.1 Chinchilla Per-Jalur Scaling

Chinchilla (Hoffmann et al., 2022) shows that for compute budget $C$, optimal parameters $N$ and data $D$ follow $C \approx 6ND$ with ratio $D/N \approx 20$.

Losion applies this **per pathway**:
- Each pathway has a different FLOP budget (SSM cheap, Attention expensive)
- Parameters allocated proportional to FLOP budget
- MoE: only count active parameters (not total experts)

$$N_j = \sqrt{\frac{C_j}{6 \times 20}}, \quad D_j = 20 \cdot N_j$$

Default FLOP ratios: Jalur 1 = 20%, Jalur 2 = 50%, Jalur 3 = 30%.

> **Agent Context:** `ChinchillaScaler` in `lossion/training/advanced_backprop.py`. Method `compute_optimal_scaling()` returns `ChinchillaScalingResult`. Method `validate_config()` checks LosionConfig against Chinchilla-optimal ratios.

### 11.2 Per-Jalur Learning Rate Schedules

Each pathway has different training dynamics:
- **Jalur 1 (SSM)**: Cheap per step → LR peaks fast, decays fast (warmup 3%, sharp decay 0.8)
- **Jalur 2 (Attention)**: Expensive per step → LR peaks slow, decays slow (warmup 6%, soft decay 0.5)
- **Jalur 3 (MoE)**: Medium → moderate schedule (warmup 4%, decay 0.6)

$$\text{LR}_j(t) = \begin{cases} \text{LR}_0 \cdot t / t_\text{warmup,j} & t < t_\text{warmup,j} \\ \text{LR}_0 \cdot \frac{1}{2}(1 + \cos(\pi \cdot p^{\gamma_j})) & t \geq t_\text{warmup,j} \end{cases}$$

where $p$ is training progress and $\gamma_j$ is the decay rate for pathway $j$.

> **Agent Context:** `PerJalurLRScheduler` in `lossion/training/advanced_backprop.py`. Method `get_lr(step, jalur_idx)`.

### 11.3 Logit Soft Capping (Gemma 2)

Prevents logit divergence during training without hard clipping:

$$\text{capped} = c \cdot \tanh(x / c)$$

where $c$ is the cap value (default 50.0). Applied to AR output logits, flow matching velocity predictions, and MTP auxiliary head logits.

> **Agent Context:** `LogitSoftCapper` in `lossion/training/advanced_backprop.py`.

### 11.4 Scheduled Sampling (GraphCast)

Bridges the teacher-forcing/autoregressive gap by gradually replacing ground truth with model predictions:

$$p(t) = \min\left(1, \frac{t - t_\text{warmup}}{T - t_\text{warmup}}\right) \cdot r_\text{max}$$

Supports linear, exponential, and inverse-sigmoid schedules.

> **Agent Context:** `ScheduledSampler` in `lossion/training/advanced_backprop.py`.

### 11.5 Confidence Heads (AlphaFold3)

Three auxiliary prediction heads providing dense supervisory signals:
1. **Routing Confidence**: Is the routing decision correct?
2. **Prediction Difficulty**: How hard is the next token?
3. **Diffusion Quality**: Will flow matching produce good output?

These heads are trained with auxiliary loss using AR loss and routing entropy as targets, providing rich gradient signal without affecting inference (can be distilled away).

> **Agent Context:** `ConfidenceHeads` in `lossion/training/advanced_backprop.py`. Method `compute_auxiliary_loss()`.

### 11.6 Parallel Attention + FFN (PaLM)

PaLM-style parallel formulation computes attention and FFN simultaneously:

$$\text{output} = x + \text{Attention}(\text{LN}(x)) + \text{FFN}(\text{LN}(x))$$

This effectively doubles "depth" within the same latency budget. Applied to Jalur 2.

> **Agent Context:** `ParallelAttentionFFN` in `lossion/training/advanced_backprop.py`.

### 11.7 Gradient Communication Overlapping (PaLM 2)

Overlaps gradient synchronization with backward computation:
- When computing gradients for Jalur 1, simultaneously synchronize Jalur 2 gradients
- When computing Jalur 2, synchronize Jalur 3
- Reduces communication overhead by 40–60%

> **Agent Context:** `GradientOverlapScheduler` in `lossion/training/advanced_backprop.py`. Method `get_communication_schedule(current_jalur)`.

### 11.8 Memory-Efficient Backpropagation

Combines:
1. **Gradient Checkpointing** (active in `LosionModel`)
2. **Expert Gradient Accumulation** (GShard-style: accumulate K micro-batches before all-reduce)
3. **Selective Gradient Computation** (only compute gradients for active experts)

> **Agent Context:** `MemoryEfficientBackprop` in `lossion/training/advanced_backprop.py`.

---

## 12. Advanced Memory & Data Pipeline

Seven techniques for memory optimization and data pipeline efficiency.

### 12.1 Progressive KV Compression (Gemini LC)

Position-dependent KV cache compression:
- Recent tokens (last 4K): full fidelity (1:1)
- Medium tokens (4K–64K): 4:1 compression
- Old tokens (64K+): 16:1 compression

Achieves ~10× memory reduction for 1M context vs. uniform compression.

> **Agent Context:** `ProgressiveKVCompressor` in `lossion/training/advanced_memory_data.py`.

### 12.2 Attention Sinks (Gemini LC)

Reserves 4 "sink tokens" at the start of the sequence that are never evicted from KV cache, stabilizing streaming inference by preventing attention drift.

> **Agent Context:** `AttentionSinkManager` in `lossion/training/advanced_memory_data.py`.

### 12.3 Dynamic Expert Buffer Allocation (GShard)

Instead of over-provisioning buffer for every expert (30–50% memory waste), allocates buffers dynamically based on router's predicted load per batch.

> **Agent Context:** `DynamicExpertBufferAllocator` in `lossion/training/advanced_memory_data.py`.

### 12.4 Modality-Aware Loss Weighting (Gemini)

Per-Jalur loss weighting based on inverse perplexity — pathways with high perplexity get more training weight. Uses EMA tracking of per-pathway perplexity.

> **Agent Context:** `ModalityAwareLossWeighter` in `lossion/training/advanced_memory_data.py`.

### 12.5 Chinchilla Token-to-Parameter Ratio

Ensures dataset size ≥ 20 × active_parameters. For MoE, only active expert parameters are counted.

> **Agent Context:** `ChinchillaDataSizer` in `lossion/training/advanced_memory_data.py`.

### 12.6 Sample-then-Filter (AlphaCode)

Generate K=64 candidate continuations, score by AR log-probability + consistency, select best. Dramatically improves output quality at K× compute cost.

> **Agent Context:** `SampleFilterPipeline` in `lossion/training/advanced_memory_data.py`.

### 12.7 Template-Based Conditional Routing (AlphaCode)

When the router detects structured output patterns (code, math, formal language), injects "template bias" into routing:

| Output Type | Routing Bias `[J1, J2, J3]` |
|------------|---------------------------|
| Code | `[-0.1, 0.2, 0.1]` — more to Jalur 2 (precise) |
| Math | `[-0.1, 0.3, 0.0]` — more to Jalur 2 (reasoning) |
| Creative | `[0.1, -0.1, 0.1]` — more to Jalur 1+3 |
| Factual | `[-0.1, -0.1, 0.3]` — more to Jalur 3 (retrieval) |

> **Agent Context:** `TemplateConditionalRouter` in `lossion/training/advanced_memory_data.py`.

---

## 13. Advanced RLHF

Four DeepMind/Google AI techniques for RLHF far more effective than standard GRPO.

### 13.1 GRPO (DeepSeek-R1)

Group Relative Policy Optimization: generates a group of responses per prompt, scores them, and updates policy using relative advantages within the group.

### 13.2 Self-Play Preference Generation (AlphaZero)

Model generates multiple candidates per prompt with different routing strategies (auto/thinking/non-thinking), evaluates them via value head + self-consistency, and creates preference pairs. No human annotation needed — infinite preference data.

$$\text{score} = w_v \cdot V + w_c \cdot C + w_e \cdot R$$

where $V$ = value head score, $C$ = consistency score, $R$ = external reward.

### 13.3 Policy-Value Dual Head (MuZero)

`JalurValueHead` jointly predicts policy (routing) and value (expected quality) per pathway. Value head reduces variance in advantage estimation:

$$V = \sum_{i=1}^{3} w_i \cdot V_i$$

where $w_i$ are routing weights and $V_i$ are per-pathway value predictions.

### 13.4 Self-Consistency Verification (Gemini Thinking)

Generate K=5 candidates, cluster by similarity, select from largest cluster. Provides internal reward signal without external reward model.

### 13.5 Dirichlet Noise Injection (AlphaZero)

Injects Dirichlet noise into router logits during training to prevent routing collapse:

$$\text{logits}_\text{noisy} = (1 - \epsilon) \cdot \text{softmax}(\text{logits}) + \epsilon \cdot \text{Dir}(\alpha)$$

Default: $\alpha = 0.25$, $\epsilon = 0.25$ (AlphaZero standard).

> **Agent Context:** All components in `lossion/training/advanced_rlhf.py`. Key classes: `JalurValueHead`, `DirichletNoiseInjector`, `SelfPlayPreferenceGenerator`, `SelfConsistencyVerifier`, `AdvancedGRPOTrainer`. Config: `AdvancedGRPOConfig`.

---

## 14. Parameter Computation

### Base Formulas

**Embedding:**
$$P_\text{embed} = V \cdot d$$

**SSM per Layer:**
$$P_\text{ssm} = 2 \cdot d \cdot (d \cdot e) + N_s \cdot d + 3d$$

where $e$ = expand factor, $N_s$ = d_state.

**Attention per Layer:**
$$P_\text{attn} = 4d^2 + d \cdot d_\text{mla} + d \cdot d_\text{pair} \cdot 2 + d_\text{pair} \cdot 32 \cdot 2 + d \cdot 4d$$

**Retrieval per Layer:**
$$P_\text{retr} = k_\text{active} \cdot 2 \cdot d \cdot d_\text{ff} + d \cdot N_e + d \cdot d_\text{engram} + 2 \cdot d \cdot d_\text{ff}$$

where $k_\text{active}$ = active experts, $N_e$ = total experts.

**Router per Layer:**
$$P_\text{router} = 3d + 3d + 4 \cdot 6 + 6 \cdot 3 + (d/4) \cdot 2 + 2d$$

### Estimates per Model Size

| Component | 1B | 7B | 48B |
|-----------|-----|------|-------|
| `d_model` | 768 | 2,048 | 4,096 |
| `n_layers` | 12 | 24 | 48 |
| `num_experts` | 16 | 64 | 256 |
| `num_active_experts` | 2 | 4 | 8 |
| Embedding | ~24M | ~262M | ~524M |
| SSM layers | ~14M | ~201M | ~1.6B |
| Attention layers | ~38M | ~540M | ~4.3B |
| Retrieval layers | ~8M | ~180M | ~2.9B |
| Router + Evo | ~2M | ~12M | ~95M |
| Output (MTP+FM+Diff) | ~1M | ~6M | ~25M |
| Reasoning Engine | ~2M | ~15M | ~120M |
| **Total (estimate)** | **~0.9B** | **~7.2B** | **~47B** |
| **Active params** | ~0.6B | ~4.1B | ~18B |

> **Agent Context:** `LosionConfig.estimated_parameters()` provides runtime computation. `LosionModel.count_parameters()` returns per-component breakdown. Config files: `configs/losion-1b.yaml`, `configs/losion-7b.yaml`, `configs/losion-48b.yaml`.

---

## 15. Design Decisions & Justification

### 1. Three Pathways Instead of One

**Decision**: Combine SSM + Attention + Retrieval in one model.
**Justification**: Each primitive is optimal for a different computation type. SSM for sequential dependencies (O(n)), Attention for reasoning (full context), MoE+Engram for factual knowledge (explicit retrieval). No single primitive covers all needs efficiently.

### 2. Aux-Loss-Free Routing

**Decision**: Use bias-based routing without auxiliary loss.
**Justification**: Aux loss requires hyperparameter tuning, can interfere with the main loss, and doesn't guarantee optimal load balancing. Bias-based routing (DeepSeek-V3) gets signals directly from the main loss gradient — simpler, more stable, and empirically effective.

### 3. Interleaving Patterns (Not Sequential)

**Decision**: Interleave SSM sub-layers (4:1:1) and local/global attention (5:1/2:1).
**Justification**: Sequential processing (all SSD, then all WKV, etc.) causes gradient vanishing for later sub-layers. Interleaving distributes computational variety evenly, following Gemma 3 and Llama 4.

### 4. MLA Compression (Not Standard KV-Cache)

**Decision**: Compress KV-cache to latent space via MLA.
**Justification**: For 1M token sequences, full KV-cache requires ~128GB per layer. MLA reduces this by 8× without significant quality degradation, as demonstrated by DeepSeek-V3.

### 5. Engram Memory (Not Full Parameterized Knowledge)

**Decision**: Add Engram Memory for explicit factual knowledge.
**Justification**: MoE stores knowledge distributed across expert weights. Specific facts (names, dates, definitions) are better stored explicitly. Engram provides O(1) retrieval for known facts. The hybrid approach (Engram + MoE) is more flexible than either alone.

### 6. Hardware-Agnostic Design

**Decision**: Pure PyTorch without custom CUDA kernels.
**Justification**: Custom CUDA kernels only run on NVIDIA, not AMD. PyTorch supports both CUDA and ROCm. Trade-off: slightly slower than custom kernels, but far more portable. `torch.compile()` provides significant graph optimization without manual kernel writing.

### 7. Expert Choice Routing Option

**Decision**: Support both token-choice and expert-choice routing.
**Justification**: Token-choice (DeepSeek-V3) gives tokens control over which experts to use. Expert-choice (Google Research 2022) guarantees load balancing automatically. Providing both options lets users choose based on their priorities.

### 8. Diffusion Refinement at Output

**Decision**: Add AlphaFold3-style diffusion refinement to the output pipeline.
**Justification**: Single-pass output generation can produce artifacts and inconsistencies. Iterative denoising (even just 4 steps) significantly improves coherence, especially for structured outputs (code, math proofs).

---

## 16. Comparison with Other Architectures

### Losion vs GPT (Standard Transformer)

| Aspect | GPT | Losion |
|--------|-----|--------|
| Primary mechanism | Self-attention | Tri-Jalur (SSM + Attn + Retrieval) |
| Complexity | O(n²) | O(n) + O(n·d_latent) + O(n·k) sparse |
| Adaptive | No (uniform) | Yes (per-token routing) |
| KV-cache memory | Full | MLA compressed (8× reduction) |
| Factual knowledge | Distributed | Engram + MoE |
| Thinking mode | No | Yes (adaptive depth) |
| Multi-token prediction | No | MTP (2–4 tokens) |
| Inference-time scaling | No | MCTS + parallel thinking |
| Elastic inference | No | Matryoshka submodels |
| Diffusion refinement | No | Yes (optional) |

**When GPT is better**: Tasks that always require deep reasoning (math, code), when GPU memory is not a constraint.

**When Losion is better**: Mixed-complexity tasks, very long contexts (>32K tokens), constrained hardware deployment.

### Losion vs Mamba (SSM-only)

| Aspect | Mamba | Losion |
|--------|-------|--------|
| Primary mechanism | SSM (S6) | Tri-Jalur |
| Reasoning | Limited | Yes (Jalur 2 Attention) |
| Factual knowledge | Distributed | Engram + MoE |
| Interleaving | No | Yes (SSD:WKV:Delta = 4:1:1) |
| Routing | No | Adaptive Router |
| Context length | Unbounded (recurrent) | Unbounded (SSM) + long (Attention) |

**When Mamba is better**: Pure sequential tasks (time series, genomics), when inference speed is the absolute priority.

**When Losion is better**: Tasks requiring reasoning and factual knowledge, general NLP, chat assistants.

### Losion vs Jamba (SSM + Attention Hybrid)

| Aspect | Jamba | Losion |
|--------|-------|--------|
| Architecture | SSM + Attention (sequential layers) | Tri-Jalur (parallel + routing) |
| Routing | No | Adaptive (per-token) |
| Attention | Standard | MLA (compressed) |
| Retrieval | No | MoE + Engram |
| Thinking mode | No | Yes |
| Output pipeline | Standard | MTP + Flow Matching + Diffusion |

**When Jamba is better**: Simpler implementation, faster inference for simple tasks.

**When Losion is better**: When adaptivity is needed (varied tasks), factual knowledge matters, and very long contexts.

### Losion vs DeepSeek-V3

| Aspect | DeepSeek-V3 | Losion |
|--------|-------------|--------|
| Architecture | Attention + MoE | Tri-Jalur (SSM + Attn + MoE) |
| SSM pathway | No | Yes (Mamba-2 + RWKV-7 + DeltaNet) |
| Attention | MLA | MLA + iRoPE + Pairformer |
| MoE routing | Token-choice (bias-based) | Token-choice + Expert-choice option |
| Engram memory | No | Yes (hash-based fact store) |
| Thinking mode | No | Yes (adaptive) |
| Elastic inference | No | Matryoshka |
| Diffusion refinement | No | Yes |

**When DeepSeek-V3 is better**: When you need maximum attention capacity and don't need SSM pathways or explicit factual retrieval.

**When Losion is better**: When long-context efficiency (SSM pathway), explicit factual retrieval (Engram), or adaptive compute (thinking mode + elastic inference) are priorities.

### Losion vs Gemma 3

| Aspect | Gemma 3 | Losion |
|--------|---------|--------|
| Architecture | Standard transformer | Tri-Jalur |
| KV compression | Standard KV | MLA (8× compression) |
| Interleaving | 5:1 local/global | 5:1/2:1 adaptive |
| MoE | No | Yes (16–256 experts) |
| SSM pathway | No | Yes (Mamba-2 + RWKV-7 + DeltaNet) |
| Elastic inference | Matryoshka | Matryoshka |
| Diffusion refinement | No | Yes |
| Neuro-symbolic verification | No | Yes |

**When Gemma 3 is better**: When you want a well-tested, production-ready transformer with simple deployment.

**When Losion is better**: When you need the efficiency of SSM, the memory savings of MLA, or the factual accuracy of Engram + MoE retrieval.

---

*This document was written for Losion v0.1.0. For questions and discussion, please open an issue on the GitHub repository.*

*Indonesian notes (catatan): "Tri-Jalur" = Three Pathways; "Terpadu" = Integrated/Unified; "Engram" = jejak memori; "Jalur" = Pathway/Channel.*
