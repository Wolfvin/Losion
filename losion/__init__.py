"""
Losion — Hybrid AI Framework with Tri-Jalur Router Architecture.

Version 0.7.0 — "Integrated & Complete"

Losion combines three complementary computational pathways into a single
adaptive architecture:

  Jalur 1 (SSM):           Mamba-2 SSD + RWKV-7 WKV + Gated DeltaNet
                            + Mamba-3 + Routing Mamba + Liquid SSM
  Jalur 2 (Attention):     MLA + iRoPE + Lightning Attention + RoPE
                            + Gated Attention + MoBA + KDA+MLA
  Jalur 3 (Retrieval):     MoE + Engram Memory + Expert Choice
                            + S'MoRE + Symbolic-MoE + AuxFreeMoE

Router:  Adaptive (BiasRouter + ThinkingToggle + Symbolic-MoE), GRPO-trained.

v0.7 Upgrades — "Integrated & Complete":
  INTEGRATION:  LosionModelV2 — config-driven module selection
                All core modules wired into production model
                AdaptiveRouter replaces nn.Linear router
                RoPE replaces learned position embeddings
                MTP heads + JEPA loss in training
  INFERENCE:    KV Cache (standard + MLA compressed + Paged)
                Full .generate() with temperature/top-k/top-p
                Speculative decoding (SSM as draft model)
                Continuous batching server
                KV cache compression (ChunkKV + EvolKV)
  DATA:         LosionTokenizer (tiktoken/sentencepiece)
                LosionDataset with packed sequences
                Data curation pipeline (quality + dedup + PII)
                Curriculum data loader
  TRAINING:     Losion Training Recipe (4-phase methodology)
                WSD LR schedule with WSM weight averaging
                Scaling recipes for 1B/7B/48B
  EVALUATION:   Perplexity + MMLU/GSM8K/HellaSwag
                Routing behavior analysis
  SAFETY:       Constitutional AI + Safety Classifier + Red Teaming
  DISTRIBUTED:  FSDP + DDP + Pipeline + Context Parallel
  LONG CONTEXT: RoPE extension (YaRN, NTK-aware) + SSM state extension
"""

__version__ = "0.7.0"
__author__ = "Losion Contributors"
__license__ = "MIT"
