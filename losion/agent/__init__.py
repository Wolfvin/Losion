"""
Losion Agent Layer — Autonomous agent capabilities on top of the Losion model.

This module implements the Agent Layer that sits ABOVE the Tri-Jalur model,
providing skills management, tool execution, web search, and orchestration
capabilities. The model remains a clean neural architecture; this layer
translates model signals (confidence, routing weights, thinking mode) into
agent actions (web search, tool use, skill creation).

Architecture:
    Model (Tri-Jalur) → Signals → Agent Orchestrator → Actions
                                              ↓
                                    ┌─────────┴──────────┐
                                    │   Skills Manager    │
                                    │   Tools Registry    │
                                    │   Web Search        │
                                    │   Terminal (sandbox)│
                                    └────────────────────┘

Design Principles:
    - Model provides signals; agent responds
    - Skills & tools stored externally (not in model weights)
    - Web search only when model confidence is low
    - Terminal execution in isolated sandbox
    - Everything is optional and configurable
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
]
