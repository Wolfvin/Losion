<p align="center">
  <img src="docs/assets/losion-logo.png" alt="Losion" width="200"/>
</p>

<h1 align="center">Losion</h1>

<p align="center">
  <strong>Hybrid AI Framework with Tri-Jalur Router Architecture</strong>
</p>

<p align="center">
  <a href="https://github.com/losion/losion/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/></a>
  <a href="https://github.com/losion/losion/releases"><img src="https://img.shields.io/badge/version-0.4.0-orange.svg" alt="Version"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1%2B-red.svg" alt="PyTorch"/></a>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#v04-upgrades">v0.4 Upgrades</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#model-configs">Configs</a> •
  <a href="#documentation">Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## What is Losion?

Losion is an open-source hybrid AI framework that combines **three complementary computational pathways** into a single adaptive architecture called the **Tri-Jalur Router** (Three-Pathway Router). Each pathway excels at different aspects of language understanding and generation:

| Pathway | Name | Mechanism | Best For |
|---------|------|-----------|----------|
| **Jalur 1** | SSM | Mamba-2 SSD + RWKV-7 WKV + Gated DeltaNet | Sequential processing, long-range dependencies |
| **Jalur 2** | Attention+Compression | MLA + iRoPE + Pairformer + Lightning Attention | Reasoning, precise retrieval, O(1) inference |
| **Jalur 3** | Retrieval | MoE + Engram Memory + Expert Choice | Factual recall, knowledge-intensive tasks |

The **Adaptive Router** dynamically routes each token to the optimal pathway(s) based on input complexity, using a bias-based aux-loss-free mechanism trained with GRPO. A **Thinking Toggle** (inspired by Qwen3) detects when deeper reasoning is needed and activates additional compute depth.

### Why Tri-Jalur?

No single architecture excels at everything. Pure attention models struggle with long contexts (O(n²) scaling). Pure SSM models lose fine-grained retrieval capability. Pure MoE models lack sequential coherence. Losion's Tri-Jalur architecture combines the strengths while mitigating the weaknesses:

- **SSM pathway** provides linear-time sequential processing with constant-time inference
- **Attention pathway** provides precise token-level retrieval and reasoning with compression
- **Retrieval pathway** provides knowledge access through sparse expert activation

The router learns to allocate compute where it matters most, achieving both efficiency and quality.

---

## Architecture

```
Input Token
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                 Adaptive Router                      │
│   BiasRouter + ThinkingToggle → routing_weights[3]  │
└────────┬────────────┬────────────┬──────────────────┘
         │            │            │
    ┌────▼────┐  ┌────▼────┐  ┌───▼─────┐
    │ Jalur 1 │  │ Jalur 2 │  │ Jalur 3 │
    │   SSM   │  │  Attn+  │  │ Retrieval│
    │         │  │  Compr  │  │   MoE    │
    │ Mamba-2 │  │  MLA +  │  │ Expert  │
    │ RWKV-7  │  │ iRoPE + │  │ Choice +│
    │ Delta   │  │Lightning│  │ Engram  │
    │         │  │ Shared  │  │ Hetero  │
    │ Liquid  │  │         │  │ Matryos │
    └────┬────┘  └────┬────┘  └───┬─────┘
         │            │            │
         └──────┬─────┴────────────┘
                ▼
        ┌───────────────┐
        │  Blend & Norm  │
        └───────┬───────┘
                ▼
         Output Token(s)
         (MTP Speculative)
```

---

## v0.4 Upgrades

Version 0.4 ("Lightning & Liquid") introduces 12 research-backed upgrades across four priority tiers:

### HIGH Priority

| Upgrade | Module | Impact |
|---------|--------|--------|
| **Lightning Attention** | `core/attention/lightning_attention.py` | O(1) inference, 4M token context |
| **Parallel-Head Mode (1B)** | `models/parallel_head.py` | Eliminate routing overhead |
| **BitNet 1.58-bit** | `core/quantization/bitnet.py` | ~6x memory reduction |

### MEDIUM Priority

| Upgrade | Module | Impact |
|---------|--------|--------|
| **Heterogeneous MoE** | `core/retrieval/heterogeneous_moe.py` | Variable-size experts |
| **Matryoshka MoE** | `core/retrieval/matryoshka_moe.py` | Elastic expert count |
| **Gradient-Routed MoE** | `core/retrieval/gradient_routed_moe.py` | Loss-aligned routing |
| **FP8 Training** | `core/quantization/fp8_training.py` | ~2x training throughput |
| **Post-Training NAS** | `core/nas/layer_search.py` | Layer-wise architecture optimization |

### LOW Priority

| Upgrade | Module | Impact |
|---------|--------|--------|
| **Shared Attention** | `core/attention/shared_attention.py` | ~6x KV cache reduction (Zamba2-style) |
| **MTP Speculative Decoding** | `core/output/speculative_decoder.py` | ~1.8x inference speedup |
| **Asymmetric MoE Placement** | `core/retrieval/asymmetric_placement.py` | Selective MoE sparsity |

### LONG-TERM

| Upgrade | Module | Impact |
|---------|--------|--------|
| **Liquid SSM** | `core/ssm/liquid_ssm.py` | Adaptive compute depth per token |

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/losion/losion.git
cd losion
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

Optional dependencies for training, evaluation, and development are listed in `pyproject.toml`.

---

## Quick Start

### Build a Model

```python
import torch
from losion.core.ssm import SSMTerpaduLayer, LiquidSSMTerpaduLayer
from losion.core.attention import LightningAttention, SharedAttentionConfig
from losion.core.retrieval import ExpertChoiceMoE, HeterogeneousMoE
from losion.core.router import AdaptiveRouter
from losion.core.quantization import BitNetConfig, convert_linear_to_bitnet

# Initialize the Adaptive Router
router = AdaptiveRouter(d_model=768, top_k_pathways=2)

# Jalur 1: SSM with Liquid adaptive depth
ssm_layer = LiquidSSMTerpaduLayer(
    d_model=768, d_state=64, n_heads=12, use_liquid=True
)

# Jalur 2: Lightning Attention with MLA compression
attn_layer = LightningAttention(
    d_model=768, n_heads=12, d_head=64,
    kv_lora_rank=128, window_size=2048
)

# Jalur 3: Expert Choice MoE with Engram
moe_layer = ExpertChoiceMoE(
    d_model=768, d_ff=1536, num_experts=16, use_shared_expert=True
)

# Forward pass example
x = torch.randn(2, 128, 768)  # [batch, seq, d_model]

# Route
routing = router(x)
print(f"Routing weights: {routing.adjusted_weights.shape}")
print(f"Thinking mode: {routing.thinking_assessment.mode}")
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
python scripts/train.py --config configs/losion-1b.yaml
python scripts/train.py --config configs/losion-7b.yaml
python scripts/train.py --config configs/losion-48b.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --config configs/losion-7b.yaml --checkpoint checkpoints/losion-7b.pt
```

---

## Model Configs

| Config | Parameters | d_model | Layers | Seq Length | Experts | GPU Requirement |
|--------|-----------|---------|--------|------------|---------|-----------------|
| `losion-1b.yaml` | 1B | 768 | 12 | 32K | 16 | 1x RTX 4090 / A10G |
| `losion-7b.yaml` | 7B | 2048 | 24 | 131K | 64 | 1x A100 80GB / H100 |
| `losion-48b.yaml` | 48B | 4096 | 48 | 1M | 256 | 8x H100 80GB |

All configs include v0.4 feature flags for Lightning Attention, Liquid SSM, BitNet, FP8, etc.

---

## Project Structure

```
losion/
├── losion/
│   ├── __init__.py
│   ├── core/
│   │   ├── ssm/           # Jalur 1: Mamba-2, RWKV-7, DeltaNet, Liquid SSM
│   │   ├── attention/     # Jalur 2: MLA, iRoPE, Lightning, Shared Attention
│   │   ├── retrieval/     # Jalur 3: Expert Choice, Heterogeneous, Matryoshka MoE
│   │   ├── router/        # Adaptive Router (BiasRouter + ThinkingToggle)
│   │   ├── reasoning/     # MCTS, Neuro-symbolic, Parallel Thinking
│   │   ├── elastic/       # Matryoshka dimension elasticity
│   │   ├── output/        # Flow Matching, Diffusion, Speculative Decoder
│   │   ├── quantization/  # BitNet 1.58-bit, FP8 Training
│   │   └── nas/           # DARTS-style Neural Architecture Search
│   ├── models/            # ParallelHeadLayer for Losion-1B
│   ├── training/          # Trainer, GRPO, Curriculum, RLHF, Active Learning
│   └── utils/             # Hardware detection, logging
├── configs/               # YAML configs for 1B/7B/48B
├── tests/                 # Unit tests
├── scripts/               # Train, evaluate, convert checkpoints
├── docs/                  # Architecture, Training, Hardware guides
├── .github/workflows/     # CI/CD
├── pyproject.toml
├── requirements.txt
└── setup.py
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Detailed Tri-Jalur architecture design |
| [Training Guide](docs/TRAINING.md) | How to train Losion models |
| [Hardware Guide](docs/HARDWARE.md) | GPU requirements and optimization |
| [Getting Started](docs/GETTING_STARTED.md) | Quick start guide for new users |
| [Contributing](docs/CONTRIBUTING.md) | How to contribute to Losion |
| [Agent Training](docs/AGENT_TRAINING_TECHNIQUES.md) | Advanced agent training techniques |

---

## Benchmark Results (v0.3 vs v0.4)

| Metric | v0.3 | v0.4 | Improvement |
|--------|------|------|-------------|
| Inference speed (tokens/s) | 42 | 76 | +81% |
| Max context length | 131K | 4M | +30x |
| Memory (7B, BF16) | 14.2 GB | 2.4 GB (BitNet) | -83% |
| Training throughput | 1.0x | 1.9x (FP8) | +90% |
| KV cache (7B) | 2.1 GB | 0.35 GB (Shared) | -83% |
| MoE load balance | 0.87 | 0.94 (Gradient-routed) | +8% |

*Full benchmark report: See `Losion_v0.3_vs_v0.4_Benchmark_Report.pdf`*

---

## Key Research References

- **Mamba-2 SSD**: Gu & Dao, "SSD: Structured State Space Duality" (2024)
- **RWKV-7**: Peng et al., "RWKV-7: Eagle" (2025)
- **Gated DeltaNet**: Yang et al., "Gated Delta Networks" (2024)
- **MLA / DeepSeek-V2**: DeepSeek-AI (2024)
- **Expert Choice Routing**: Zhou et al., Google Research (2022)
- **Lightning Attention**: Sun et al., "Lightning Attention-2" (2024)
- **BitNet 1.58**: Wang et al. (2024)
- **Zamba2**: Glorioso et al. (2024)
- **MTP Speculative Decoding**: Inspired by DeepSeek-V3

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Losion builds on ideas from many excellent research projects including Mamba, RWKV, DeepSeek, Qwen, Zamba, BitNet, and the broader SSM/MoE/hybrid architecture community. We gratefully acknowledge their contributions to open research.
