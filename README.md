<h1 align="center">Losion</h1>

<p align="center">
  <strong>Hybrid AI Framework with Tri-Jalur Router Architecture</strong>
</p>

<p align="center">
  <a href="https://github.com/Wolfvin/Losion/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/></a>
  <a href="https://github.com/Wolfvin/Losion/releases"><img src="https://img.shields.io/badge/version-2.5.1-brightgreen.svg" alt="Version 2.5.1"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1%2B-red.svg" alt="PyTorch"/></a>
</p>

<p align="center">
  <a href="#what-is-losion">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#version-history">Version History</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#model-configs">Configs</a> •
  <a href="#documentation">Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## What is Losion?

Losion is an open-source hybrid AI framework that combines **three complementary computational pathways** into a single adaptive architecture called the **Tri-Jalur Router** (Three-Pathway Router). After six major development cycles (v0.4–v0.9), version 1.0.0 unifies **40+ research-backed components** into a single production-ready system.

Each pathway excels at different aspects of language understanding and generation:

| Pathway | Name | Mechanism | Best For |
|---------|------|-----------|----------|
| **Jalur 1** | SSM | 9 SSM variants with structured transitions | Sequential processing, long-range dependencies |
| **Jalur 2** | Attention+Compression | 12 attention mechanisms with learned residuals | Reasoning, precise retrieval, O(1) inference |
| **Jalur 3** | Retrieval | 11 MoE variants with infinite expert scaling | Factual recall, knowledge-intensive tasks |

The **Adaptive Router** dynamically routes each token to the optimal pathway(s) based on input complexity, using a bias-based aux-loss-free mechanism trained with GRPO/DAPO. A **Thinking Toggle** (inspired by Qwen3) detects when deeper reasoning is needed and activates additional compute depth. **Evoformer feedback loops** (5 levels, AlphaFold-inspired) enable bidirectional information flow between layers, tokens, decoders, predictions, and the router itself.

### Why Tri-Jalur?

No single architecture excels at everything. Pure attention models struggle with long contexts (O(n²) scaling). Pure SSM models lose fine-grained retrieval capability. Pure MoE models lack sequential coherence. Losion's Tri-Jalur architecture combines the strengths while mitigating the weaknesses:

- **SSM pathway** provides linear-time sequential processing with constant-time inference
- **Attention pathway** provides precise token-level retrieval and reasoning with compression
- **Retrieval pathway** provides knowledge access through sparse expert activation

The router learns to allocate compute where it matters most, achieving both efficiency and quality.

### All Components by Pathway

#### Jalur 1 — SSM (9 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **Mamba-2 SSD** | v0.1 | Structured State Space Duality, parallel scan |
| **RWKV-7 WKV** | v0.1 | Eagle kernel, linear attention alternative |
| **Gated DeltaNet** | v0.1 | Gated delta networks for fast mixing |
| **Liquid SSM** | v0.4 | Adaptive compute depth per token |
| **PoST Decay Spectra** | v0.5 | Power-law decay spectra for SSM transitions |
| **FG2-GDN** | v0.5 | Fine-grained gated delta network |
| **Mamba-3 SSD** | v0.6 | Inference-first SSM, half state dimension |
| **Routing Mamba** | v0.6 | MoE over SSM projections (Microsoft, NeurIPS '25) |
| **Structured Sparse SSM** | v0.8 | Structured sparse transition matrices (NeurIPS '25) |

#### Jalur 2 — Attention+Compression (12 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **MLA** | v0.1 | Multi-head Latent Attention (DeepSeek-V2) |
| **iRoPE** | v0.1 | Interleaved Rotary Position Embedding |
| **Lightning Attention** | v0.4 | O(1) inference, 4M token context |
| **Shared Attention** | v0.4 | Zamba2-style, ~6x KV cache reduction |
| **KDA+MLA Hybrid** | v0.5 | Kernelized density-aware MLA attention |
| **Gated Attention** | v0.6 | Qwen NeurIPS '25 Best Paper |
| **MoBA** | v0.6 | Mixture of Block Attention (Moonshot AI, NeurIPS '25) |
| **RoPE** | v0.7 | Rotary Position Embedding with YaRN/NTK scaling |
| **Context Extension** | v0.7 | YaRN, NTK-aware, SSM-based length extension |
| **AttnRes** | v0.9 | Attention Residuals, learned aggregation (MoonshotAI 2026) |
| **Child-3W** | v0.9 | MoE at QKV level, representation-level specialization |
| **Cross-Jalur Routing** | v0.8 | Attention-MoE cross-pathway routing |

#### Jalur 3 — Retrieval/MoE (11 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **Expert Choice MoE** | v0.1 | Token-choice routing (Google Research) |
| **Engram Memory** | v0.1 | Persistent expert memory embeddings |
| **Heterogeneous MoE** | v0.4 | Variable-size expert dimensions |
| **Matryoshka MoE** | v0.4 | Elastic expert count |
| **Gradient-routed MoE** | v0.4 | Loss-aligned routing signals |
| **Asymmetric Placement** | v0.4 | Selective MoE sparsity per layer |
| **Aux-Loss-Free MoE** | v0.5 | DeepSeek-V3 style, bias-based balancing |
| **S'MoRE** | v0.6 | Sub-tree MoE with Residual Experts (Meta, NeurIPS '25) |
| **Symbolic-MoE** | v0.6 | Skill-based discrete routing |
| **∞-MoE (Infinite MoE)** | v0.8 | Continuous expert space via codebook+hypernetwork |
| **MoHGE** | v0.8 | Heterogeneous Grouped Experts |

#### Router & Feedback (4 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **Adaptive Router / BiasRouter** | v0.1 | Dynamic pathway selection with bias-based balancing |
| **ThinkingToggle** | v0.1 | Qwen3-style deeper reasoning activation |
| **Evoformer** | v0.9 | 5-level AlphaFold-inspired feedback loops |
| **Dual Memory System** | v0.9 | Working memory (ring buffer) + Long-term memory (AttnRes state) |

#### Output Heads (6 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **Flow Matching** | v0.3 | Continuous normalizing flow for output |
| **Diffusion Refinement** | v0.3 | Denoising refinement for token distribution |
| **MTP Speculative Decoding** | v0.4 | Multi-token prediction, ~1.8x speedup |
| **Mirror Speculative Decoding** | v0.5 | Draft-then-verify with mirror model |
| **L-MTP (Leap MTP)** | v0.8 | Non-adjacent multi-token prediction (NeurIPS '25) |
| **Anchored Diffusion Decoder** | v0.9 | Continuous vector pipeline + lightweight diffusion |

#### Recurrent & Self-Supervised (2 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **RDT (Recurrent-Depth Transformer)** | v0.6 | Looped transformer blocks with shared weights (OpenMythos) |
| **LLM-JEPA** | v0.6 | Predict future latent states instead of next tokens |

#### Training (16 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **GRPO** | v0.3 | Group Relative Policy Optimization |
| **Curriculum Learning** | v0.3 | Difficulty-scheduled training |
| **Active Learning** | v0.3 | Uncertainty-based data selection |
| **Advanced RLHF** | v0.3 | Multi-objective reward optimization |
| **ETR Reward** | v0.5 | Expert-Teacher Reward for RL |
| **Gen Distillation** | v0.5 | Generation-level knowledge distillation |
| **TACO** | v0.5 | Training with Adaptive Consistency Optimization |
| **Losion Recipe** | v0.7 | 4-phase training pipeline (Pretrain → SFT → RL → Align) |
| **Compute Aligned** | v0.7 | Compute-optimal scaling laws |
| **Evolutionary Search** | v0.7 | Population-based hyperparameter search |
| **Advanced Backprop** | v0.7 | Gradient checkpointing & memory optimization |
| **Advanced Memory Data** | v0.7 | Memory-efficient data loading |
| **DAPO** | v0.8 | Decoupled Clip & Dynamic Sampling Policy Optimization |
| **RLVR** | v0.8 | Reinforcement Learning with Verifiable Rewards |
| **Training Orchestrator** | v0.8 | Multi-phase training coordination |
| **HyLo Upcycling** | v0.5 | Dense-to-sparse upcycling for SSM pathways |

#### Quantization & Compression (5 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **BitNet 1.58-bit** | v0.4 | ~6x memory reduction |
| **FP8 Training** | v0.4 | ~2x training throughput |
| **BitDistill** | v0.5 | Distillation-aware quantization |
| **Attention-Preferred LoRA** | v0.5 | LoRA with attention-aware rank allocation |
| **QuantSpec** | v0.8 | Quantization-aware speculative decoding |

#### Elastic & NAS (3 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **Matryoshka Elasticity** | v0.3 | Dimension-level model elasticity |
| **Post-Training NAS** | v0.4 | DARTS-style layer-wise architecture search |
| **Path-Lock Expert** | v0.5 | Lock critical expert pathways during fine-tuning |

#### Agent Layer (12 components, v0.5)

| Component | Module | Description |
|-----------|--------|-------------|
| **Orchestrator** | `agent/orchestrator.py` | Central agent coordination |
| **MCTS Agent** | `agent/planning/mcts_agent.py` | Monte Carlo Tree Search planning |
| **Dependency Planner** | `agent/planning/deps_planner.py` | Task dependency resolution |
| **Paradigm Router** | `agent/planning/paradigm_router.py` | Planning paradigm selection |
| **Skill Manager** | `agent/skills/manager.py` | Skill lifecycle management |
| **Skill Creator** | `agent/skills/creator.py` | Dynamic skill generation |
| **Skill Store** | `agent/skills/store.py` | Skill persistence and retrieval |
| **Meta Skills** | `agent/meta_skills.py` | Higher-order skill composition |
| **Agentic Retriever** | `agent/retrieval/agentic_retriever.py` | Agent-driven retrieval |
| **Calibration** | `agent/calibration.py` | Confidence calibration |
| **Reflection** | `agent/reflection.py` | Self-reflection and correction |
| **Tool Registry / Terminal / Web Search** | `agent/tools/` | Tool use infrastructure |
| **Risk Simulator** | `agent/safety/risk_simulator.py` | Safety risk assessment |
| **Agent Memory** | `agent/memory.py` | Persistent agent state |
| **Signals** | `agent/signals.py` | Inter-component signaling |

#### Reasoning (4 components)

| Component | Version | Description |
|-----------|---------|-------------|
| **MCTS** | v0.1 | Monte Carlo Tree Search for reasoning |
| **Neuro-symbolic** | v0.1 | Neural-symbolic integration |
| **Parallel Thinking** | v0.1 | Multi-path reasoning |
| **Path-Lock Expert** | v0.5 | Expert pathway locking for reasoning chains |

---

## Architecture

```
                              Input Token(s)
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Token Embedding        │
                    │         + RoPE / iRoPE        │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
               ┌───────────────────────────────────────┐
               │         Adaptive Router (BiasRouter     │
               │     + ThinkingToggle + Symbolic-MoE)    │
               │         → routing_weights[3]            │
               └───┬──────────────┬──────────────┬──────┘
                   │              │              │
          ┌────────▼──┐   ┌──────▼───────┐  ┌──▼──────────┐
          │  Jalur 1   │   │   Jalur 2    │  │   Jalur 3    │
          │    SSM     │   │  Attn+Compr  │  │  Retrieval   │
          │            │   │              │  │              │
          │ Mamba-2    │   │ KDA+MLA      │  │ AuxFreeMoE   │
          │ Mamba-3    │   │ Gated Attn   │  │ ∞-MoE        │
          │ RWKV-7     │   │ MoBA         │  │ S'MoRE       │
          │ DeltaNet   │   │ Lightning    │  │ Symbolic-MoE │
          │ Liquid SSM │   │ Shared Attn  │  │ MoHGE        │
          │ PoST Decay │   │ Child-3W     │  │ ExpertChoice │
          │ FG2-GDN    │   │ AttnRes      │  │ Engram Mem   │
          │ Routing M  │   │ Cross-Jalur  │  │ Hetero MoE   │
          │ StructSp   │   │ RoPE/YaRN    │  │ Matryoshka   │
          │            │   │ Context Ext  │  │ Grad-Routed  │
          └────┬───────┘   └──────┬───────┘  └──┬───────────┘
               │                  │              │
               └────────┬─────────┴──────────────┘
                        ▼
              ┌─────────────────────┐
              │   Blend & RMSNorm   │
              └─────────┬───────────┘
                        │
              ┌─────────▼───────────┐
              │  Evoformer Feedback │ ← Level 1: Inter-layer recycling
              │  (5 levels)         │ ← Level 2: Bidirectional token update
              │                     │ ← Level 3: Decoder ↔ Predict feedback
              │                     │ ← Level 4: Prediction → Context recycling
              │                     │ ← Level 5: Router ↔ Expert co-evolution
              └─────────┬───────────┘
                        │
              ┌─────────▼───────────┐
              │  Dual Memory System │ ← Working: recent, detailed (ring buffer)
              │                     │ ← Long-term: compressed, persistent (AttnRes)
              └─────────┬───────────┘
                        │
              ┌─────────▼──────────────┐
              │  Optional: RDT Blocks  │  (Recurrent-Depth Transformer loops)
              └─────────┬──────────────┘
                        │
                        ▼
            ┌────────────────────────────┐
            │       Output Heads         │
            │  LM Head + MTP / L-MTP    │
            │  Flow Matching / Diffusion │
            │  Anchored Diffusion Decoder│
            │  Mirror Speculative Dec.   │
            └────────────┬───────────────┘
                         │
                         ▼
                  Output Token(s)
            (MTP/L-MTP Speculative)
```

---

## Version History

| Version | Codename | Key Additions | Components Added |
|---------|----------|--------------|-----------------|
| **v0.1** | Foundation | Mamba-2, RWKV-7, DeltaNet, MLA, iRoPE, Expert Choice, Engram, Adaptive Router, ThinkingToggle, MCTS, Neuro-symbolic, Flow Matching, Diffusion, GRPO | 14 |
| **v0.2** | Early Hybrid | Matryoshka Elasticity, Pairformer, Curriculum Learning, Active Learning | 4 |
| **v0.3** | MoE & RL | Advanced RLHF, Pairformer refinement | 2 |
| **v0.4** | Lightning & Liquid | Lightning Attention, Shared Attention, Heterogeneous MoE, Matryoshka MoE, Gradient-routed MoE, FP8 Training, Post-Training NAS, MTP Speculative Decoding, Asymmetric Placement, BitNet 1.58-bit, Liquid SSM, Parallel Head | 12 |
| **v0.5** | KDA & Aux-Free | KDA+MLA Hybrid, Aux-Loss-Free MoE + MTP, Path-Lock Expert, PoST Decay Spectra, HyLo Upcycling, Mirror Speculative Decoding, ETR Reward, Gen Distillation, TACO, BitDistill, Attention-Preferred LoRA, FG2-GDN, Agent Layer (12 components) | 18 |
| **v0.6** | Mythos & Mamba | RDT, Gated Attention (NeurIPS Best Paper), MoBA, Mamba-3 SSD, Routing Mamba, S'MoRE, Symbolic-MoE, LLM-JEPA | 8 |
| **v0.7** | Integrated & Complete | LosionModelV2 (config-driven), RoPE, KV Cache (Standard+MLA+Paged), Generation Pipeline, LosionTokenizer, LosionDataset, Training Recipe (4-phase), Evaluation Framework, Safety & Alignment (Constitutional AI), Distributed Training (4D parallelism), Context Extension (YaRN, NTK, SSM) | 11 |
| **v0.8** | Next-Gen Training & Infinite Experts | DAPO, ∞-MoE, L-MTP, Cross-Jalur Routing, RLVR, Expert Prefetching, Training Orchestrator, Structured Sparse SSM, MoHGE, QuantSpec | 10 |
| **v0.9** | Architecture Document Realized | AttnRes (MoonshotAI 2026), Evoformer (5-level AlphaFold feedback), Child-3W (QKV-level MoE), Anchored Diffusion Decoder, Dual Memory System | 5 |
| **v1.0.0** | Unified & Complete | Full integration of ALL components into production model, version alignment, updated configs, complete documentation | — |
| **v1.5.0** | SDPA & Parallel | SDPA/Flash Attention, RWKV-7 parallel, per-jalur gradient checkpointing | 3 |
| **v1.6.0** | Training Optimization | FP8, FSDP2, LoRA, gradient checkpointing | 4 |
| **v1.6.1** | Gradient Repair | Critical bug fixes, gradient flow repair, SDPA connected | 3 |
| **v1.7.0** | Differentiable Thinking | Differentiable ThinkingToggle, loop-free SSM, entropy regularization | 3 |
| **v1.8.0** | Per-Channel Selectivity | Per-channel selectivity, ThinkingToggle soft-blending, entropy from all layers | 3 |
| **v1.9.0** | Complete Gradient Flow | Evoformer+DualMemory gradient fix, vectorized attention, 10.0/10 score | 3 |
| **v2.0.0** | Alive Gradients | AuxFreeMoE MTP loss propagated (was 32.2% dead params), production ready | 1 |
| **v2.1.0** | Honest Code & Real Kernels | Real Triton kernel, functional use_cache, fixed Evoformer gradient, iRoPE, vectorized scan | 6 |
| **v2.2.0** | Deep Audit & Bug Purge | 40+ bug fixes (10 critical, 14 high, 17 medium, 1 low) across all modules | 40+ |
| **v2.3.0** | Security & Correctness | 12 audit fixes: 3 RCE vulnerabilities, sandbox hardening, vectorized MoE, double softmax fix, sparse inference, LLM init, FSDP thread safety | 12 |

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/Wolfvin/Losion.git
cd Losion
pip install -e ".[all]"
```

### Minimal Install

```bash
pip install -e .
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- PyYAML >= 6.0

Optional dependencies for training, evaluation, inference, and development are listed in `pyproject.toml`.

---

## Quick Start

### Build a Model with LosionForCausalLMV2

```python
import torch
from losion.config import LosionConfig
from losion.models.losion_model_v2 import LosionForCausalLMV2

# Load a pre-defined config (or create your own)
config = LosionConfig.from_yaml("configs/losion-7b.yaml")

# Build the full causal LM — config-driven module selection
model = LosionForCausalLMV2(config)

# Forward pass with labels for training
input_ids = torch.randint(0, config.vocab_size, (2, 128))
outputs = model(input_ids=input_ids, labels=input_ids)
print(f"Loss: {outputs['loss']:.4f}")
print(f"Logits shape: {outputs['logits'].shape}")

# Generate text
prompt = torch.randint(0, config.vocab_size, (1, 32))
generated = model.generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True,
)
print(f"Generated shape: {generated.shape}")
```

### Custom Configuration

```python
from losion.config import LosionConfig

# Create a custom config with specific feature flags
config = LosionConfig(
    d_model=2048,
    n_layers=24,
    vocab_size=128256,
    max_seq_len=131072,
    # Enable v0.6 features
    attention=dict(
        use_gated_attention=True,   # Qwen NeurIPS Best Paper
        use_moba=True,              # Moonshot AI
    ),
    # Enable v0.8 features
    retrieval=dict(
        use_infinite_moe=True,      # ∞-MoE continuous expert space
        infinite_moe_codebook_size=512,
    ),
    # Enable v0.9 features
    attn_res=dict(enabled=True, mode="block"),
    evoformer=dict(enabled=True, use_router_coevolve=True),
    dual_memory=dict(enabled=True),
)
```

### Apply BitNet Quantization

```python
from losion.core.quantization import BitNetConfig, convert_linear_to_bitnet

# Configure gradual quantization
bitnet_config = BitNetConfig(
    enabled=True,
    warmup_steps=2000,
    initial_quant_ratio=0.0,
    STE_mode="identity",
)

# Convert all Linear layers
model = convert_linear_to_bitnet(model, config=bitnet_config, exclude_names=["lm_head"])

# During training, increment step counter
from losion.core.quantization import increment_bitnet_step
increment_bitnet_step(model)

# After training, finalize for inference
from losion.core.quantization import finalize_bitnet_model
finalize_bitnet_model(model)  # ~6x memory reduction
```

### Training

```bash
# Phase-based training with the Losion Recipe
python scripts/train.py --config configs/losion-1b.yaml
python scripts/train.py --config configs/losion-7b.yaml
python scripts/train.py --config configs/losion-48b.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --config configs/losion-7b.yaml --checkpoint checkpoints/losion-7b.pt
```

### Save & Load Checkpoints

```python
# Save
model.save_pretrained("checkpoints/losion-7b-v1/")

# Load
model = LosionForCausalLMV2.from_pretrained("checkpoints/losion-7b-v1/")
```

---

## Model Configs

| Config | Parameters | d_model | Layers | Seq Length | Experts | Key Features | GPU Requirement |
|--------|-----------|---------|--------|------------|---------|-------------|-----------------|
| `losion-1b.yaml` | 1B | 768 | 12 | 32K | 16 | Liquid SSM, Lightning Attn, AuxFree MoE | 1x RTX 4090 / A10G |
| `losion-7b.yaml` | 7B | 2048 | 24 | 131K | 64 | Gated Attn, MoBA, ∞-MoE, AttnRes, Evoformer | 1x A100 80GB / H100 |
| `losion-48b.yaml` | 48B | 4096 | 48 | 1M | 256 | Full feature set, RDT, LLM-JEPA, Dual Memory | 8x H100 80GB |

All configs include feature flags for every component from v0.4 through v0.9. Enable or disable features by toggling config flags — no code changes required.

---

## Project Structure

```
losion/
├── losion/
│   ├── __init__.py
│   ├── config.py                  # LosionConfig + all sub-configs (15+ dataclasses)
│   │
│   ├── core/
│   │   ├── ssm/                   # Jalur 1: SSM Pathway
│   │   │   ├── mamba2.py          #   Mamba-2 SSD
│   │   │   ├── mamba3.py          #   Mamba-3 SSD (v0.6)
│   │   │   ├── rwkv7.py           #   RWKV-7 WKV
│   │   │   ├── delta_net.py       #   Gated DeltaNet
│   │   │   ├── liquid_ssm.py      #   Liquid SSM (v0.4)
│   │   │   ├── post_decay.py      #   PoST Decay Spectra (v0.5)
│   │   │   ├── fg2_gdn.py         #   FG2-GDN (v0.5)
│   │   │   ├── routing_mamba.py   #   Routing Mamba (v0.6)
│   │   │   ├── structured_sparse.py # Structured Sparse SSM (v0.8)
│   │   │   └── ssm_layer.py       #   SSMTerpaduLayer (combined)
│   │   │
│   │   ├── attention/             # Jalur 2: Attention+Compression Pathway
│   │   │   ├── kda_mla.py         #   KDA+MLA Hybrid (v0.5)
│   │   │   ├── gated_attention.py #   Gated Attention (v0.6, NeurIPS Best Paper)
│   │   │   ├── moba.py            #   MoBA (v0.6, Moonshot AI)
│   │   │   ├── lightning_attention.py # Lightning Attention (v0.4)
│   │   │   ├── shared_attention.py #  Shared Attention (v0.4)
│   │   │   ├── attn_res.py        #   Attention Residuals (v0.9, MoonshotAI 2026)
│   │   │   ├── child_3w.py        #   Child-3W QKV-level MoE (v0.9)
│   │   │   ├── context_extension.py # YaRN, NTK, SSM extension (v0.7)
│   │   │   └── __init__.py        #   MLA, iRoPE exports
│   │   │
│   │   ├── retrieval/             # Jalur 3: Retrieval/MoE Pathway
│   │   │   ├── expert_choice.py   #   Expert Choice MoE
│   │   │   ├── engram.py          #   Engram Memory
│   │   │   ├── heterogeneous_moe.py # Heterogeneous MoE (v0.4)
│   │   │   ├── matryoshka_moe.py  #   Matryoshka MoE (v0.4)
│   │   │   ├── gradient_routed_moe.py # Gradient-routed MoE (v0.4)
│   │   │   ├── asymmetric_placement.py # Asymmetric MoE (v0.4)
│   │   │   ├── aux_free_moe.py    #   Aux-Loss-Free MoE (v0.5)
│   │   │   ├── smore.py           #   S'MoRE (v0.6, Meta)
│   │   │   ├── symbolic_moe.py    #   Symbolic-MoE (v0.6)
│   │   │   ├── infinite_moe.py    #   ∞-MoE Infinite MoE (v0.8)
│   │   │   ├── mohge.py           #   MoHGE Grouped Experts (v0.8)
│   │   │   └── cross_jalur_routing.py # Cross-Jalur Routing (v0.8)
│   │   │
│   │   ├── router/                # Adaptive Router
│   │   │   ├── router.py          #   AdaptiveRouter
│   │   │   ├── bias_router.py     #   BiasRouter
│   │   │   └── thinking_toggle.py #   ThinkingToggle (Qwen3-style)
│   │   │
│   │   ├── output/                # Output Heads
│   │   │   ├── flow_matching.py   #   Flow Matching
│   │   │   ├── diffusion_refinement.py # Diffusion Refinement
│   │   │   ├── speculative_decoder.py  # MTP Speculative (v0.4)
│   │   │   ├── mirror_speculative.py   # Mirror Speculative (v0.5)
│   │   │   ├── leap_mtp.py        #   L-MTP Leap MTP (v0.8)
│   │   │   └── anchored_decoder.py #  Anchored Diffusion Decoder (v0.9)
│   │   │
│   │   ├── reasoning/             # Reasoning Modules
│   │   │   ├── mcts.py            #   Monte Carlo Tree Search
│   │   │   ├── neuro_symbolic.py  #   Neuro-symbolic integration
│   │   │   ├── parallel_thinking.py # Parallel Thinking
│   │   │   └── path_lock_expert.py #  Path-Lock Expert (v0.5)
│   │   │
│   │   ├── feedback/              # Evoformer Feedback (v0.9)
│   │   │   └── evoformer.py       #   5-level AlphaFold feedback
│   │   │
│   │   ├── memory/                # Dual Memory System (v0.9)
│   │   │   └── dual_memory.py     #   Working + Long-term memory
│   │   │
│   │   ├── recurrent/             # RDT (v0.6)
│   │   │   └── rdt.py             #   Recurrent-Depth Transformer
│   │   │
│   │   ├── elastic/               # Elastic & LoRA
│   │   │   ├── matryoshka.py      #   Dimension elasticity
│   │   │   └── attn_lora.py       #   Attention-Preferred LoRA (v0.5)
│   │   │
│   │   ├── quantization/          # Quantization
│   │   │   ├── bitnet.py          #   BitNet 1.58-bit (v0.4)
│   │   │   ├── fp8_training.py    #   FP8 Training (v0.4)
│   │   │   └── bit_distill.py     #   BitDistill (v0.5)
│   │   │
│   │   └── nas/                   # Neural Architecture Search
│   │       └── layer_search.py    #   DARTS-style Layer Search (v0.4)
│   │
│   ├── models/                    # Production Models
│   │   ├── losion_model_v2.py     #   LosionForCausalLMV2 (config-driven, v0.7+)
│   │   ├── losion_model.py        #   LosionModel (v0.1 legacy)
│   │   ├── losion_decoder.py      #   LosionDecoder
│   │   └── parallel_head.py       #   ParallelHeadLayer (1B config)
│   │
│   ├── training/                  # Training Pipeline
│   │   ├── trainer.py             #   Core Trainer
│   │   ├── grpo.py                #   GRPO (v0.3)
│   │   ├── dapo.py                #   DAPO (v0.8)
│   │   ├── rlvr.py                #   RLVR (v0.8)
│   │   ├── etr_reward.py          #   ETR Reward (v0.5)
│   │   ├── gen_distillation.py    #   Gen Distillation (v0.5)
│   │   ├── llm_jepa.py            #   LLM-JEPA (v0.6)
│   │   ├── losion_recipe.py       #   4-Phase Training Recipe (v0.7)
│   │   ├── losion_orchestrator.py #   Training Orchestrator (v0.8)
│   │   ├── curriculum.py          #   Curriculum Learning
│   │   ├── active_learning.py     #   Active Learning
│   │   ├── advanced_rlhf.py       #   Advanced RLHF
│   │   ├── compute_aligned.py     #   Compute-Optimal Scaling (v0.7)
│   │   ├── evolutionary_search.py #   Evolutionary Search (v0.7)
│   │   ├── advanced_backprop.py   #   Gradient Optimization (v0.7)
│   │   └── advanced_memory_data.py #  Memory-Efficient Data (v0.7)
│   │
│   ├── agent/                     # Agent Layer (v0.5, 12 components)
│   │   ├── orchestrator.py        #   Central agent coordination
│   │   ├── meta_skills.py         #   Higher-order skill composition
│   │   ├── memory.py              #   Persistent agent state
│   │   ├── calibration.py         #   Confidence calibration
│   │   ├── reflection.py          #   Self-reflection & correction
│   │   ├── signals.py             #   Inter-component signaling
│   │   ├── planning/              #   Planning subsystem
│   │   │   ├── mcts_agent.py      #     MCTS planning
│   │   │   ├── deps_planner.py    #     Dependency resolution
│   │   │   └── paradigm_router.py #     Planning paradigm selection
│   │   ├── skills/                #   Skill subsystem
│   │   │   ├── manager.py         #     Skill lifecycle
│   │   │   ├── creator.py         #     Dynamic skill generation
│   │   │   └── store.py           #     Skill persistence
│   │   ├── retrieval/             #   Agent retrieval
│   │   │   └── agentic_retriever.py
│   │   ├── tools/                 #   Tool use
│   │   │   ├── registry.py        #     Tool registry
│   │   │   ├── terminal.py        #     Terminal tool
│   │   │   ├── creator.py         #     Tool creator
│   │   │   └── web_search.py      #     Web search tool
│   │   └── safety/                #   Agent safety
│   │       └── risk_simulator.py  #     Risk assessment
│   │
│   ├── inference/                 # Inference Engine (v0.7+)
│   │   ├── generation.py          #   Generation Pipeline
│   │   ├── kv_cache.py            #   KV Cache (Standard + MLA + Paged)
│   │   ├── expert_prefetch.py     #   Expert Prefetching (v0.8)
│   │   └── quantspec.py           #   QuantSpec (v0.8)
│   │
│   ├── evaluation/                # Evaluation Framework (v0.7)
│   │   └── benchmarks.py          #   Standard benchmarks
│   │
│   ├── safety/                    # Safety & Alignment (v0.7)
│   │   └── alignment.py           #   Constitutional AI
│   │
│   ├── distributed/               # Distributed Training (v0.7)
│   │   └── parallel.py            #   4D Parallelism (DP, TP, PP, EP)
│   │
│   └── utils/                     # Utilities
│       ├── hardware.py            #   Hardware detection
│       ├── logging.py             #   Logging
│       └── upcycling.py           #   HyLo Upcycling (v0.5)
│
├── configs/                       # YAML configs for 1B/7B/48B
├── tests/                         # Unit tests (core, model, agent, advanced)
├── scripts/                       # Train, evaluate, convert checkpoints
├── docs/                          # Architecture, Training, Hardware, Agent guides
├── .github/workflows/             # CI/CD
├── pyproject.toml
├── requirements.txt
└── setup.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Detailed Tri-Jalur architecture design |
| [Training Guide](docs/TRAINING.md) | How to train Losion models (4-phase recipe) |
| [Hardware Guide](docs/HARDWARE.md) | GPU requirements and optimization |
| [Getting Started](docs/GETTING_STARTED.md) | Quick start guide for new users |
| [Contributing](docs/CONTRIBUTING.md) | How to contribute to Losion |
| [Agent Training](docs/AGENT_TRAINING_TECHNIQUES.md) | Advanced agent training techniques |
| [Agent Architecture](docs/AGENT_ARCHITECTURE_RESEARCH.md) | Agent layer research & design |
| [Agents Guide](docs/AGENTS.md) | Agent system usage |

---

## Key Research References

### SSM & State Space Models
- **Mamba-2 SSD**: Gu & Dao, "SSD: Structured State Space Duality" (2024)
- **Mamba-3**: arXiv:2603.15569 (2026)
- **RWKV-7**: Peng et al., "RWKV-7: Eagle" (2025)
- **Gated DeltaNet**: Yang et al., "Gated Delta Networks" (2024)
- **Routing Mamba**: Microsoft Research, NeurIPS 2025
- **Structured Sparse SSM**: NeurIPS 2025

### Attention Mechanisms
- **MLA / DeepSeek-V2**: DeepSeek-AI, arXiv:2405.04434 (2024)
- **Gated Attention**: Qwen Team, NeurIPS 2025 Best Paper
- **MoBA**: Moonshot AI, NeurIPS 2025
- **Lightning Attention**: Sun et al., "Lightning Attention-2" (2024)
- **RoPE**: Su et al., arXiv:2104.09864 (2021)
- **AttnRes**: MoonshotAI (2026), GPQA-Diamond +7.5
- **Zamba2**: Glorioso et al. (2024)

### Mixture of Experts
- **Expert Choice Routing**: Zhou et al., Google Research (2022)
- **Aux-Loss-Free MoE**: DeepSeek-V3, arXiv:2412.19437 (2024)
- **S'MoRE**: Meta Research, NeurIPS 2025
- **∞-MoE**: Infinite MoE via codebook+hypernetwork (2026)
- **MoHGE**: Heterogeneous Grouped Experts (2026)

### Feedback & Memory
- **Evoformer**: Jumper et al., Nature 2021 (Nobel Prize 2024)
- **AttnRes**: MoonshotAI (2026)

### Training & RL
- **DAPO**: Yu et al., arXiv:2503.14476 (2025)
- **RLVR**: NeurIPS 2025, arXiv:2601.05607, 2603.22117
- **GRPO**: Group Relative Policy Optimization
- **LLM-JEPA**: Predicting future latent states (2026)

### Recurrent & Universal
- **RDT**: OpenMythos reconstruction (2026)
- **Universal Transformers**: Dehghani et al., arXiv:1807.03819 (2019)

### Quantization & Compression
- **BitNet 1.58**: Wang et al. (2024)
- **BitDistill**: Distillation-aware quantization (2025)
- **QuantSpec**: Quantization-aware speculative decoding (2026)

### Context Extension
- **YaRN**: Peng et al., arXiv:2309.00071 (2023)
- **NTK-aware Scaling**: Local NTK-aware interpolation (2023)

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. Fork the repository at [github.com/Wolfvin/Losion](https://github.com/Wolfvin/Losion)
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Losion builds on ideas from many excellent research projects including Mamba, RWKV, DeepSeek, Qwen, Zamba, BitNet, AlphaFold/Evoformer, OpenMythos, MoonshotAI, Meta, Microsoft, and the broader SSM/MoE/hybrid architecture community. We gratefully acknowledge their contributions to open research.
