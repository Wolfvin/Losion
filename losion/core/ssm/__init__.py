"""
Losion — Jalur 1: State Space Models.

Sub-layers (4:1:1 interleaving):
  Mamba2SSD  — Mamba-2 State Space Dual computation
  RWKV7WKV   — RWKV-7 Weighted Key-Value attention
  GatedDeltaNet — Gated DeltaNet recurrence

v0.4 additions:
  LiquidSSD  — Mamba-2 SSD with liquid (input-adaptive) time constants
  ComplexityGate — Per-token complexity estimation for adaptive depth
  LiquidSSMTerpaduLayer — SSM layer with adaptive compute depth

v0.5 additions:
  FineGrainedGate — Per-head, per-position gating mechanism for DeltaNet
  FG2GDN — Fine-Grained Gated DeltaNet (enhanced in-context learning)
  DecaySpectrum — Learnable spectrum of decay rates per head
  PoSTDecaySSM — SSM layer with position-dependent decay spectra

v0.6 additions:
  Mamba3Config — Configuration dataclass for Mamba-3 SSD
  DualTokenShift — Dual token shift module (RWKV-inspired Mamba-3 improvement)
  Mamba3SSD — Mamba-3 State Space Dual (half state, dual shift, inference-first)

v0.7 additions:
  RoutingMambaConfig — Configuration for Routing Mamba (RoM)
  SSMExpertRouter — DeepSeek-V3 style aux-loss-free MoE router for SSM
  RoutingMamba — Routing Mamba: MoE routing over SSM projections (NeurIPS 2025)
"""

from losion.core.ssm.ssm_layer import SSMTerpaduLayer, SSMState, InterleavingScheduler
from losion.core.ssm.mamba2 import Mamba2SSD
from losion.core.ssm.rwkv7 import RWKV7WKV
from losion.core.ssm.delta_net import GatedDeltaNet
from losion.core.ssm.liquid_ssm import LiquidSSD, ComplexityGate, LiquidSSMTerpaduLayer
from losion.core.ssm.fg2_gdn import FineGrainedGate, FG2GDN
from losion.core.ssm.post_decay import DecaySpectrum, PoSTDecaySSM
from losion.core.ssm.mamba3 import Mamba3Config, DualTokenShift, Mamba3SSD
from losion.core.ssm.routing_mamba import RoutingMambaConfig, SSMExpertRouter, RoutingMamba

__all__ = [
    "SSMTerpaduLayer",
    "SSMState",
    "InterleavingScheduler",
    "Mamba2SSD",
    "RWKV7WKV",
    "GatedDeltaNet",
    "LiquidSSD",
    "ComplexityGate",
    "LiquidSSMTerpaduLayer",
    # FG2-GDN
    "FineGrainedGate",
    "FG2GDN",
    # PoST Decay Spectra
    "DecaySpectrum",
    "PoSTDecaySSM",
    # Mamba-3
    "Mamba3Config",
    "DualTokenShift",
    "Mamba3SSD",
    # Routing Mamba (RoM)
    "RoutingMambaConfig",
    "SSMExpertRouter",
    "RoutingMamba",
]
