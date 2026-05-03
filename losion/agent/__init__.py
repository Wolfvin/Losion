"""
Losion Agent Layer v3 — Autonomous agent capabilities on top of the Losion model.

This module implements the Agent Layer that sits ABOVE the Tri-Jalur model,
providing skills management, tool execution, web search, and orchestration
capabilities. The model remains a clean neural architecture; this layer
translates model signals (confidence, routing weights, thinking mode) into
agent actions (web search, tool use, skill creation).

v3 Improvements (based on research — 40+ papers):
- LATS-style MCTS Agent Loop (Zhou et al., ICML 2024): Tree-structured
  action exploration with backtracking, replacing the linear agent loop.
- SMART Knowledge Sufficiency Check (2025): Prevents tool overuse when
  parametric knowledge is sufficient, using Tri-Jalur routing weights.
- Paradigm Router: Selects the best reasoning paradigm per query
  (Direct, CoT, ReAct, RAG, MCTS) instead of always using the full loop.
- DEPS Failure Recovery (Wang et al., 2023): Structured recovery via
  Describe-Explain-Plan-Select when actions fail.
- Agentic Multi-round Retrieval: Multi-round web search with query
  refinement, replacing single-round search.
- ToolEmu Risk Simulation (Ruan et al., ICLR 2024): Pre-execution risk
  assessment via simulated outcome prediction.
- Voyager-style Executable Skills (Wang et al., 2023): Skills with
  executable code, preconditions, postconditions, and error patterns.
- Ebbinghaus Memory Decay (MemoryBank, 2023): Forgetting curve with
  access reinforcement and periodic consolidation.
- Multi-factor Retrieval (Generative Agents, 2023): Recency × Importance
  × Relevance × Strength scoring for episodic memory.

Architecture:
    Model (Tri-Jalur) → Signals → Paradigm Router → Agent Loop
                                              ↓
                                    ┌─────────┴──────────┐
                                    │   Skills Manager    │
                                    │   Tools Registry    │
                                    │   Web Search        │
                                    │   Terminal (sandbox)│
                                    │   Reflection (v2)   │
                                    │   Calibration (v2)  │
                                    │   Episodic Mem (v3) │
                                    │   Meta-Skills (v2)  │
                                    │   MCTS Agent (v3)   │
                                    │   DEPS Planner (v3) │
                                    │   Agentic RAG (v3)  │
                                    │   Risk Simulator(v3)│
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
    - Tree-structured action exploration with backtracking (v3)
    - Knowledge sufficiency prevents tool overuse (v3)
    - Multi-round retrieval with query refinement (v3)
    - Pre-execution risk assessment (v3)
    - Executable skills with retry logic (v3)
    - Memory decays over time unless reinforced (v3)
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
# v3 additions
from losion.agent.planning import (
    MCTSAgentLoop,
    AgentState,
    ActionNode,
    ActionEdge,
    ParadigmRouter,
    ReasoningParadigm,
    ParadigmSelection,
    DEPSPlanner,
    FailureDescription,
    RecoveryPlan,
)
from losion.agent.retrieval import (
    AgenticRetriever,
    RetrievalRound,
    RetrievalQuality,
)
from losion.agent.safety import (
    RiskSimulator,
    RiskLevel,
    RiskAssessment,
    SimulationResult,
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
    # Episodic Memory (v2/v3)
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
    # Planning (v3)
    "MCTSAgentLoop",
    "AgentState",
    "ActionNode",
    "ActionEdge",
    "ParadigmRouter",
    "ReasoningParadigm",
    "ParadigmSelection",
    "DEPSPlanner",
    "FailureDescription",
    "RecoveryPlan",
    # Retrieval (v3)
    "AgenticRetriever",
    "RetrievalRound",
    "RetrievalQuality",
    # Safety (v3)
    "RiskSimulator",
    "RiskLevel",
    "RiskAssessment",
    "SimulationResult",
]
