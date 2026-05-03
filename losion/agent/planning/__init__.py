"""
Agent Planning Subpackage — LATS-style MCTS, Paradigm Routing, DEPS Recovery.

v3 Improvements (based on research):
- LATS (Language Agent Tree Search, Zhou et al., ICML 2024): MCTS-based agent
  loop with tree-structured action exploration and backtracking.
- SMART (Self-Aware Agent for Tool Overuse Mitigation, 2025): Knowledge
  sufficiency check and paradigm routing.
- DEPS (Describe-Explain-Plan-Select, Wang et al., 2023): Failure recovery
  with structured re-planning.
- DFSDT (Depth-First Search Decision Tree, Qin et al., 2023): Backtracking
  in tool use when actions fail.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                  PARADIGM ROUTER                      │
    │  Direct | CoT | ReAct | RAG | MCTS-Tree-Search       │
    └──────────────────────┬──────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼───┐  ┌────▼───┐  ┌────▼────┐
         │ MCTS   │  │ DEPS   │  │ DFSDT   │
         │ Agent  │  │Planner │  │Backtrack│
         │ Loop   │  │        │  │         │
         └────────┘  └────────┘  └─────────┘
"""

from losion.agent.planning.mcts_agent import (
    MCTSAgentLoop,
    AgentState,
    ActionNode,
    ActionEdge,
)
from losion.agent.planning.paradigm_router import (
    ParadigmRouter,
    ReasoningParadigm,
    ParadigmSelection,
)
from losion.agent.planning.deps_planner import (
    DEPSPlanner,
    FailureDescription,
    RecoveryPlan,
)

__all__ = [
    # MCTS Agent Loop
    "MCTSAgentLoop",
    "AgentState",
    "ActionNode",
    "ActionEdge",
    # Paradigm Router
    "ParadigmRouter",
    "ReasoningParadigm",
    "ParadigmSelection",
    # DEPS Planner
    "DEPSPlanner",
    "FailureDescription",
    "RecoveryPlan",
]
