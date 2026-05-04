"""
Losion Safety — Constitutional AI alignment and safety classification.

Provides safety tools for Losion models including:
  - Constitutional principle evaluation and compliance checking
  - Safety classification (binary + multi-label) on hidden states
  - Constitutional AI training loop with DPO/AlphaDPO
  - Adversarial red teaming via Reverse Constitutional AI (R-CAI)

Credits:
  - Constitutional AI (Anthropic, 2022)
  - Reverse Constitutional AI (arXiv:2604.17769, 2026)
  - Direct Reward Optimization (DRO, OpenReview 2025)
  - AlphaDPO (ICML 2025)

Usage:
    >>> from losion.safety import Constitution, SafetyClassifier, ConstitutionalTrainer, RedTeamer
    >>> constitution = Constitution()
    >>> classifier = SafetyClassifier(d_model=768)
    >>> red_teamer = RedTeamer(constitution, intensity=3)
    >>> prompts = red_teamer.generate_adversarial_prompts()
"""

from __future__ import annotations

from losion.safety.alignment import (
    Constitution,
    ConstitutionalPrinciple,
    ConstitutionalTrainer,
    RedTeamer,
    SafetyClassifier,
)

__all__ = [
    "Constitution",
    "ConstitutionalPrinciple",
    "ConstitutionalTrainer",
    "RedTeamer",
    "SafetyClassifier",
]
