"""
Losion Agent Layer v2 — Autonomous agent capabilities on top of the Losion model.

This module implements the Agent Layer that sits ABOVE the Tri-Jalur model,
providing skills management, tool execution, web search, and orchestration
capabilities. The model remains a clean neural architecture; this layer
translates model signals (confidence, routing weights, thinking mode) into
agent actions (web search, tool use, skill creation).

v2 Improvements (based on research):
- Reflective Agent Loop (Reflexion, 2023; Self-Refine, 2023)
- Adaptive Confidence Calibration (ATTC, 2026)
- Episodic Memory (Synapse, 2024; MemP, 2025)
- Meta-Skill System (CASCADE, 2025; SoK: Agentic Skills, 2026)

Architecture:
    Model (Tri-Jalur) → Signals → Agent Orchestrator → Actions
                                              ↓
                                    ┌─────────┴──────────┐
                                    │   Skills Manager    │
                                    │   Tools Registry    │
                                    │   Web Search        │
                                    │   Terminal (sandbox)│
                                    │   Reflection (v2)   │
                                    │   Calibration (v2)  │
                                    │   Episodic Mem (v2) │
                                    │   Meta-Skills (v2)  │
                                    └────────────────────┘

Design Principles:
    - Model provides signals; agent responds
    - Skills & tools stored externally (not in model weights)
    - Web search only when model confidence is low
    - Terminal execution in isolated sandbox
    - Everything is optional and configurable
    - Learn from experience via reflection and calibration (v2)
    - Skills can create and verify other skills (v2)
    - Thresholds adapt based on outcomes (v2)
"""

from losion.agent.signals import (
    AgentSignal,
    AgentAction,
    SignalExtractor,
    ConfidenceThreshold,
)
from losion.agent.skills.manager import SkillManager
from losion.agent.skills.store import SkillStore, SkillEntry, SkillMetadata
from losion.agent.skills.creator import SkillCreator
from losion.agent.tools.registry import ToolRegistry, ToolEntry
from losion.agent.tools.terminal import SandboxedTerminal, TerminalResult
from losion.agent.tools.web_search import WebSearchInterface, SearchResult
from losion.agent.tools.creator import ToolCreator
from losion.agent.orchestrator import AgentOrchestrator, AgentConfig, AgentResult
from losion.agent.reflection import ReflectionEngine, Reflection, ReflectionType
from losion.agent.memory import EpisodicMemory, Episode
from losion.agent.calibration import CalibrationEngine, DomainProfile, ToolTrustScore
from losion.agent.meta_skills import (
    SkillSynthesisMetaSkill,
    SkillVerificationMetaSkill,
    SkillCompositionMetaSkill,
    ComposedSkill,
)

__all__ = [
    # Signals
    "AgentSignal",
    "AgentAction",
    "SignalExtractor",
    "ConfidenceThreshold",
    # Skills
    "SkillManager",
    "SkillStore",
    "SkillEntry",
    "SkillMetadata",
    "SkillCreator",
    # Tools
    "ToolRegistry",
    "ToolEntry",
    "SandboxedTerminal",
    "TerminalResult",
    "WebSearchInterface",
    "SearchResult",
    "ToolCreator",
    # Orchestrator
    "AgentOrchestrator",
    "AgentConfig",
    "AgentResult",
    # Reflection (v2)
    "ReflectionEngine",
    "Reflection",
    "ReflectionType",
    # Episodic Memory (v2)
    "EpisodicMemory",
    "Episode",
    # Calibration (v2)
    "CalibrationEngine",
    "DomainProfile",
    "ToolTrustScore",
    # Meta-Skills (v2)
    "SkillSynthesisMetaSkill",
    "SkillVerificationMetaSkill",
    "SkillCompositionMetaSkill",
    "ComposedSkill",
]
