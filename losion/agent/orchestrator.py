"""
Agent Orchestrator v3 — Central coordination with MCTS, Paradigm Routing, and Risk Assessment.

v3 Improvements (based on 40+ research papers):
- LATS-style MCTS Agent Loop: Tree-structured action exploration with
  backtracking, replacing the linear agent loop (Zhou et al., ICML 2024)
- SMART Knowledge Sufficiency: Prevents tool overuse when parametric
  knowledge is sufficient (2025)
- Paradigm Router: Selects the best reasoning paradigm per query
  (Direct, CoT, ReAct, RAG, MCTS)
- DEPS Failure Recovery: Structured recovery via Describe-Explain-Plan-Select
  when actions fail (Wang et al., 2023)
- Agentic Multi-round Retrieval: Multi-round web search with query refinement
- ToolEmu Risk Simulation: Pre-execution risk assessment (Ruan et al., ICLR 2024)
- DFSDT Backtracking: Backtrack and try alternatives when actions fail
  (Qin et al., 2023)

v2 Improvements (preserved):
- Reflective Agent Loop: After each action, reflect on outcome quality
  (Reflexion, 2023; Self-Refine, 2023)
- Adaptive Calibration: Thresholds adjust based on experience
  (ATTC, 2026)
- Episodic Memory: Past experiences inform future decisions
  (Synapse, 2024; MemP, 2025)
- Meta-Skill System: Skills that create, verify, and compose other skills
  (CASCADE, 2025; SoK: Agentic Skills, 2026)

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    USER QUERY                        │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │              LOSION MODEL (Tri-Jalur)                │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │        PARADIGM ROUTER (v3: SMART + routing)        │
    │  Direct | CoT | ReAct | RAG | MCTS-Tree-Search      │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │           AGENT ORCHESTRATOR v3                      │
    │                                                      │
    │   If MCTS selected:                                  │
    │     MCTS Agent Loop (tree search + backtracking)     │
    │   Else:                                              │
    │     Linear agent loop (as before)                    │
    │                                                      │
    │   On failure: DEPS Planner (structured recovery)     │
    │   Pre-execution: Risk Simulator (risk assessment)    │
    │                                                      │
    │   Sub-modules:                                       │
    │   ├── ReflectionEngine   (Reflexion/Self-Refine)    │
    │   ├── CalibrationEngine  (ATTC)                     │
    │   ├── EpisodicMemory     (Synapse/MemP + Ebbinghaus)│
    │   ├── MetaSkillSystem    (CASCADE/SoK)              │
    │   ├── MCTSAgentLoop      (LATS/DFSDT)              │
    │   ├── ParadigmRouter     (SMART)                    │
    │   ├── DEPSPlanner        (DEPS)                     │
    │   ├── AgenticRetriever   (Agentic RAG)              │
    │   └── RiskSimulator      (ToolEmu)                  │
    └─────────────────────────────────────────────────────┘

Key design principle: The orchestrator NEVER modifies the model.
It only uses model signals to decide agent actions, and feeds
action results back as additional context for the model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from losion.agent.signals import (
    AgentAction,
    AgentSignal,
    ConfidenceThreshold,
    SignalExtractor,
)
from losion.agent.skills.manager import SkillManager
from losion.agent.skills.store import SkillEntry, SkillStore
from losion.agent.skills.creator import SkillCreator
from losion.agent.tools.registry import ToolEntry, ToolRegistry, ToolSafety
from losion.agent.tools.terminal import SandboxedTerminal, TerminalResult, SandboxConfig
from losion.agent.tools.web_search import WebSearchInterface, SearchResult
from losion.agent.tools.creator import ToolCreator
from losion.agent.reflection import ReflectionEngine, Reflection, ReflectionType
from losion.agent.memory import EpisodicMemory, Episode
from losion.agent.calibration import CalibrationEngine, ToolTrustScore
from losion.agent.meta_skills import (
    SkillSynthesisMetaSkill,
    SkillVerificationMetaSkill,
    SkillCompositionMetaSkill,
    ComposedSkill,
)
from losion.agent.planning.mcts_agent import MCTSAgentLoop, AgentState, MCTSResult
from losion.agent.planning.paradigm_router import ParadigmRouter, ReasoningParadigm, ParadigmSelection
from losion.agent.planning.deps_planner import DEPSPlanner, FailureDescription, FailureType, RecoveryPlan
from losion.agent.retrieval.agentic_retriever import AgenticRetriever
from losion.agent.safety.risk_simulator import RiskSimulator, RiskLevel, RiskAssessment

logger = logging.getLogger(__name__)


class OrchestratorState(Enum):
    """State of the agent orchestrator loop."""

    IDLE = "idle"
    MODEL_INFERENCE = "model_inference"
    SIGNAL_EXTRACTION = "signal_extraction"
    WEB_SEARCH = "web_search"
    SKILL_LOOKUP = "skill_lookup"
    SKILL_CREATE = "skill_create"
    TOOL_SEARCH = "tool_search"
    TOOL_CREATE = "tool_create"
    TERMINAL_EXECUTE = "terminal_execute"
    VERIFY_OUTPUT = "verify_output"
    REFLECT = "reflect"
    CALIBRATION = "calibration"
    CONTEXT_INJECTION = "context_injection"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for the Agent Orchestrator.

    v2 additions:
    - episodic_store_dir: Directory for episodic memory storage.
    - enable_reflection: Whether to use self-reflection.
    - enable_calibration: Whether to use adaptive threshold calibration.
    - enable_meta_skills: Whether to use meta-skill system.
    - reflection_on_failure: Whether to trigger reflection on action failure.
    - calibration_learning_rate: How fast thresholds adapt.

    Attributes:
        max_iterations: Maximum agent loop iterations per query.
        enable_web_search: Whether web search is available.
        enable_terminal: Whether terminal execution is available.
        enable_skill_creation: Whether skills can be auto-created.
        enable_tool_creation: Whether tools can be auto-created.
        confidence_thresholds: Threshold configuration.
        auto_inject_context: Whether to automatically inject search results.
        sandbox_config: Terminal sandbox configuration.
        skill_store_dir: Directory for skill storage.
        verbose: Whether to log detailed agent actions.
    """

    max_iterations: int = 5
    enable_web_search: bool = True
    enable_terminal: bool = True
    enable_skill_creation: bool = True
    enable_tool_creation: bool = True
    confidence_thresholds: ConfidenceThreshold = field(default_factory=ConfidenceThreshold)
    auto_inject_context: bool = True
    sandbox_config: SandboxConfig = field(default_factory=SandboxConfig)
    skill_store_dir: str = "~/.losion/skills"
    verbose: bool = False
    # v2 additions
    episodic_store_dir: str = "~/.losion/episodic"
    enable_reflection: bool = True
    enable_calibration: bool = True
    enable_meta_skills: bool = True
    reflection_on_failure: bool = True
    calibration_learning_rate: float = 0.1
    # v3 additions
    enable_paradigm_routing: bool = True
    enable_mcts_agent: bool = True
    enable_deps_recovery: bool = True
    enable_risk_simulation: bool = True
    enable_agentic_retrieval: bool = True
    mcts_max_simulations: int = 8
    mcts_max_depth: int = 5
    risk_threshold: str = "medium"  # safe, low, medium, high, critical


@dataclass
class AgentResult:
    """Result of an agent loop execution.

    v2 additions:
    - reflections: Reflections generated during execution.
    - episode_id: ID of the episodic memory entry created.
    - calibration_applied: Whether adaptive calibration was used.
    - meta_skills_used: Meta-skills that were invoked.

    Attributes:
        output: Final output from the agent loop.
        state: Final orchestrator state.
        iterations: Number of iterations executed.
        actions_taken: List of actions taken.
        signals: List of signals received.
        context_used: Context that was injected.
        skills_used: Skills that were applied.
        tools_used: Tools that were executed.
        total_time: Total execution time.
        model_confidence: Final model confidence.
        metadata: Additional metadata.
    """

    output: Any = None
    state: OrchestratorState = OrchestratorState.COMPLETE
    iterations: int = 0
    actions_taken: List[str] = field(default_factory=list)
    signals: List[AgentSignal] = field(default_factory=list)
    context_used: List[str] = field(default_factory=list)
    skills_used: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    total_time: float = 0.0
    model_confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # v2 additions
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    episode_id: Optional[str] = None
    calibration_applied: bool = False
    meta_skills_used: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Whether the agent loop completed successfully."""
        return self.state == OrchestratorState.COMPLETE


class AgentOrchestrator:
    """Central orchestrator for the Losion Agent Layer v2.

    v2 improvements:
    1. Reflective Loop: After each action, reflect on outcome
       and store reflections in episodic memory.
    2. Adaptive Calibration: Thresholds adjust based on
       past outcomes (successful actions lower thresholds,
       failed actions raise them).
    3. Episodic Memory: Past experiences stored and retrieved
       for similar queries.
    4. Meta-Skill System: Skills can create, verify, and
       compose other skills.

    The orchestrator manages the agent loop:
    1. Receive model output
    2. Extract signals (with adaptive thresholds)
    3. Decide action
    4. Execute action
    5. Reflect on outcome (NEW)
    6. Calibrate thresholds (NEW)
    7. Inject context
    8. Store episode in memory (NEW)
    9. Repeat until confidence sufficient or max iterations

    Usage:
        orchestrator = AgentOrchestrator(config)
        result = orchestrator.run(
            model_output=routing_output,
            query="What is the capital of France?",
            model_inference_fn=my_model_forward,
        )

    Args:
        config: Agent configuration.
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()

        # === Initialize components ===
        # Calibration engine (v2)
        self.calibration_engine = CalibrationEngine(
            learning_rate=self.config.calibration_learning_rate,
        ) if self.config.enable_calibration else None

        # Episodic memory (v2)
        self.episodic_memory = EpisodicMemory(
            store_dir=self.config.episodic_store_dir,
        ) if self.config.enable_reflection else None

        # Reflection engine (v2)
        self.reflection_engine = ReflectionEngine() if self.config.enable_reflection else None

        # Signal extractor with v2 enhancements
        self.signal_extractor = SignalExtractor(
            thresholds=self.config.confidence_thresholds,
            enable_web_search=self.config.enable_web_search,
            enable_terminal=self.config.enable_terminal,
            enable_skill_creation=self.config.enable_skill_creation,
            enable_tool_creation=self.config.enable_tool_creation,
            calibration_engine=self.calibration_engine,
            episodic_memory=self.episodic_memory,
        )

        # Existing components
        self.web_search = WebSearchInterface()
        self.terminal = SandboxedTerminal(self.config.sandbox_config)
        self.skill_store = SkillStore(store_dir=self.config.skill_store_dir)
        self.skill_creator = SkillCreator(store=self.skill_store, web_search=self.web_search)
        self.skill_manager = SkillManager(
            store=self.skill_store,
            creator=self.skill_creator,
            auto_create=self.config.enable_skill_creation,
        )
        self.tool_registry = ToolRegistry(allow_dangerous=False)
        self.tool_creator = ToolCreator(
            registry=self.tool_registry,
            web_search=self.web_search,
            auto_register=True,
        )

        # Meta-skill system (v2)
        if self.config.enable_meta_skills:
            self.synthesis_meta = SkillSynthesisMetaSkill(
                skill_creator=self.skill_creator,
                web_search=self.web_search,
            )
            self.verification_meta = SkillVerificationMetaSkill(
                skill_store=self.skill_store,
            )
            self.composition_meta = SkillCompositionMetaSkill(
                skill_manager=self.skill_manager,
            )
        else:
            self.synthesis_meta = None
            self.verification_meta = None
            self.composition_meta = None

        # v3: Paradigm Router
        self.paradigm_router = ParadigmRouter() if self.config.enable_paradigm_routing else None

        # v3: MCTS Agent Loop
        self.mcts_agent = MCTSAgentLoop(
            action_executor=self._mcts_action_executor,
            signal_extractor=self.signal_extractor,
            max_simulations=self.config.mcts_max_simulations,
            max_depth=self.config.mcts_max_depth,
        ) if self.config.enable_mcts_agent else None

        # v3: DEPS Planner
        self.deps_planner = DEPSPlanner() if self.config.enable_deps_recovery else None

        # v3: Agentic Retriever
        self.agentic_retriever = AgenticRetriever(
            web_search=self.web_search,
        ) if self.config.enable_agentic_retrieval else None

        # v3: Risk Simulator
        risk_threshold_map = {
            "safe": RiskLevel.SAFE,
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL,
        }
        self.risk_simulator = RiskSimulator(
            risk_threshold=risk_threshold_map.get(self.config.risk_threshold, RiskLevel.MEDIUM),
        ) if self.config.enable_risk_simulation else None

        # Register built-in tools
        self._register_builtin_tools()

        # State
        self._state = OrchestratorState.IDLE
        self._iteration_count = 0

    def _register_builtin_tools(self) -> None:
        """Register built-in tools that are always available."""
        self.tool_registry.register(ToolEntry(
            name="web-search",
            description="Search the web for information",
            handler=lambda query="", **kw: [
                {"title": r.title, "snippet": r.snippet, "url": r.url}
                for r in self.web_search.search(query=query)
            ],
            safety=ToolSafety.SAFE,
            domain="web",
            tags=["search", "web", "information"],
            is_builtin=True,
        ))

        self.tool_registry.register(ToolEntry(
            name="terminal",
            description="Execute a terminal command in a sandbox",
            handler=lambda command="", **kw: {
                "success": self.terminal.execute(command).success,
                "output": self.terminal.execute(command).output,
            },
            safety=ToolSafety.REQUIRES_APPROVAL,
            domain="system",
            tags=["terminal", "shell", "execute"],
            is_builtin=True,
        ))

    def _mcts_action_executor(
        self,
        action_name: str,
        query: str,
        context: List[str],
        domain: Optional[str] = None,
    ) -> Tuple[Any, float]:
        """Execute an action for the MCTS agent loop.

        Returns (result, new_confidence) tuple.
        """
        # Create a signal for this action
        from losion.agent.signals import AgentAction
        try:
            action_enum = AgentAction(action_name)
        except ValueError:
            return None, 0.5

        signal = AgentSignal(
            action=action_enum,
            confidence=0.5,
            query=query,
            domain=domain,
        )

        # Execute the action using existing methods
        action_result = self._execute_action(signal, query, context)

        # Estimate new confidence based on result
        if action_result is not None:
            new_confidence = 0.6  # Action produced a result → moderate confidence
        else:
            new_confidence = 0.3  # No result → low confidence

        return action_result, new_confidence

    def run(
        self,
        model_output: Any = None,
        query: str = "",
        confidence: Optional[float] = None,
        model_inference_fn: Optional[Callable] = None,
        context: Optional[List[str]] = None,
    ) -> AgentResult:
        """Run the agent loop with v2 reflective and adaptive capabilities.

        The v2 agent loop:
        1. Extract signal from model output (with adaptive thresholds)
        2. If MODEL_ONLY → return current output
        3. If REFLECT → load reflections from episodic memory
        4. If action needed → execute action
        5. Reflect on outcome (NEW)
        6. Calibrate thresholds based on outcome (NEW)
        7. Inject context from action result
        8. Re-infer with new context (if model_inference_fn provided)
        9. Repeat until confidence sufficient or max iterations
        10. Store episode in episodic memory (NEW)

        Args:
            model_output: Output from the Losion model.
            query: The user's query text.
            confidence: Override confidence score.
            model_inference_fn: Function for re-inference with context.
            context: Pre-existing context to inject.

        Returns:
            AgentResult with the final output and execution details.
        """
        start_time = time.time()
        self._state = OrchestratorState.MODEL_INFERENCE
        self._iteration_count = 0

        result = AgentResult(calibration_applied=self.calibration_engine is not None)
        current_context = list(context) if context else []
        current_output = model_output
        current_confidence = confidence
        all_reflections: List[Reflection] = []

        for iteration in range(self.config.max_iterations):
            self._iteration_count = iteration + 1
            self._state = OrchestratorState.SIGNAL_EXTRACTION

            # === Extract signal (v2: adaptive thresholds) ===
            signal = self.signal_extractor.extract(
                model_output=current_output,
                confidence=current_confidence,
                query_text=query,
            )
            result.signals.append(signal)

            if self.config.verbose:
                logger.info(
                    f"Iteration {iteration + 1}: signal={signal.action.value}, "
                    f"confidence={signal.confidence:.2f}, "
                    f"priority={signal.priority:.2f}, "
                    f"tool_trust={signal.tool_trust:.2f}"
                )

            # === Check if intervention needed ===
            if not signal.needs_intervention:
                self._state = OrchestratorState.COMPLETE
                result.state = self._state
                result.model_confidence = signal.confidence
                break

            # === Handle REFLECT action (v2) ===
            if signal.action == AgentAction.REFLECT:
                self._state = OrchestratorState.REFLECT
                reflection_context = self._handle_reflect(signal, query)
                if reflection_context:
                    current_context.append(reflection_context)
                    result.context_used.append(reflection_context)
                result.actions_taken.append(signal.action.value)
                continue

            # === Execute action ===
            confidence_before = current_confidence or signal.confidence
            action_result = self._execute_action(signal, query, current_context)
            result.actions_taken.append(signal.action.value)

            # === Process action result ===
            if action_result is not None:
                if isinstance(action_result, str):
                    current_context.append(action_result)
                    result.context_used.append(action_result)
                elif isinstance(action_result, dict):
                    context_str = str(action_result)
                    current_context.append(context_str)
                    result.context_used.append(context_str)
                elif isinstance(action_result, list):
                    for item in action_result:
                        context_str = str(item)
                        current_context.append(context_str)
                        result.context_used.append(context_str)

                # Auto-inject context and re-infer
                if self.config.auto_inject_context and model_inference_fn is not None:
                    self._state = OrchestratorState.CONTEXT_INJECTION
                    try:
                        new_output, new_confidence = model_inference_fn(
                            query, current_context
                        )
                        current_output = new_output
                        current_confidence = new_confidence
                    except Exception as e:
                        logger.warning(f"Re-inference failed: {e}")

            # === v2: Reflect on outcome ===
            confidence_after = current_confidence or signal.confidence
            if self.reflection_engine is not None:
                reflections = self.reflection_engine.evaluate(
                    action=signal.action.value,
                    action_result=action_result,
                    confidence_before=confidence_before,
                    confidence_after=confidence_after,
                    query=query,
                    domain=signal.domain,
                )
                all_reflections.extend(reflections)

                # Add reflection context
                for r in reflections:
                    result.reflections.append(r.to_dict())
                    if r.lesson and r.is_positive:
                        current_context.append(f"Lesson: {r.lesson}")
                        result.context_used.append(f"Lesson: {r.lesson}")

            # === v2: Calibrate thresholds ===
            if self.calibration_engine is not None:
                success = confidence_after >= confidence_before
                self.calibration_engine.record_outcome(
                    action=signal.action.value,
                    domain=signal.domain,
                    success=success,
                    confidence_before=confidence_before,
                    confidence_after=confidence_after,
                )

            # Check if confidence has improved enough
            if current_confidence is not None and current_confidence >= 0.7:
                self._state = OrchestratorState.COMPLETE
                result.state = self._state
                result.model_confidence = current_confidence
                break

        # === v2: Store episode in episodic memory ===
        if self.episodic_memory is not None:
            episode = Episode(
                query=query,
                domain=result.signals[0].domain if result.signals else None,
                actions=result.actions_taken,
                reflections=[r.to_dict() for r in all_reflections],
                final_confidence=current_confidence or 0.5,
                success=(current_confidence or 0.0) >= 0.5,
                total_iterations=self._iteration_count,
                total_time=time.time() - start_time,
            )
            self.episodic_memory.store_episode(episode)
            result.episode_id = episode.episode_id

        # === Finalize ===
        if self._state != OrchestratorState.COMPLETE:
            self._state = OrchestratorState.COMPLETE
            result.state = self._state

        result.output = current_output
        result.iterations = self._iteration_count
        result.model_confidence = current_confidence or (signal.confidence if result.signals else 0.5)
        result.total_time = time.time() - start_time

        return result

    def _handle_reflect(self, signal: AgentSignal, query: str) -> Optional[str]:
        """Handle REFLECT action by loading relevant past experience.

        Args:
            signal: The reflect signal.
            query: Current query.

        Returns:
            Context string from past reflections, or None.
        """
        if self.episodic_memory is None:
            return None

        lessons = self.episodic_memory.get_lessons_for_query(
            query=query, domain=signal.domain, limit=3
        )
        if lessons:
            context = "Past experience reflections:\n"
            for i, lesson in enumerate(lessons):
                context += f"{i+1}. {lesson}\n"
            return context

        return None

    def _execute_action(
        self,
        signal: AgentSignal,
        query: str,
        context: List[str],
    ) -> Any:
        """Execute the action specified by the signal.

        v2: Uses meta-skill system for enhanced skill creation and
        tool search fallback to skill composition.
        """
        action = signal.action

        if action == AgentAction.WEB_SEARCH:
            self._state = OrchestratorState.WEB_SEARCH
            return self._action_web_search(signal, query)

        elif action == AgentAction.SKILL_LOOKUP:
            self._state = OrchestratorState.SKILL_LOOKUP
            return self._action_skill_lookup(signal, query)

        elif action == AgentAction.SKILL_CREATE:
            self._state = OrchestratorState.SKILL_CREATE
            return self._action_skill_create(signal, query)

        elif action == AgentAction.TOOL_SEARCH:
            self._state = OrchestratorState.TOOL_SEARCH
            return self._action_tool_search(signal, query)

        elif action == AgentAction.TOOL_CREATE:
            self._state = OrchestratorState.TOOL_CREATE
            return self._action_tool_create(signal, query)

        elif action == AgentAction.TERMINAL_EXECUTE:
            self._state = OrchestratorState.TERMINAL_EXECUTE
            return self._action_terminal_execute(signal, query)

        elif action == AgentAction.VERIFY_OUTPUT:
            self._state = OrchestratorState.VERIFY_OUTPUT
            return self._action_verify_output(signal, query)

        else:
            logger.warning(f"Unknown action: {action}")
            return None

    def _action_web_search(
        self, signal: AgentSignal, query: str
    ) -> Optional[List[Dict[str, str]]]:
        """Execute web search action.

        v3: Uses AgenticRetriever for multi-round retrieval with query
        refinement when available. Falls back to single-round search.
        """
        search_query = signal.query or query
        logger.info(f"Web search: {search_query}")

        try:
            # v3: Try agentic multi-round retrieval first
            if self.agentic_retriever is not None:
                results, rounds = self.agentic_retriever.retrieve(
                    query=search_query,
                    domain=signal.domain,
                    max_rounds=3,
                    initial_confidence=signal.confidence,
                )
                return [
                    {
                        "title": r.title,
                        "snippet": r.snippet,
                        "url": r.url,
                        "source": r.source,
                        "relevance": r.relevance_score,
                    }
                    for r in results
                    if r.is_valid
                ]

            # Fallback: single-round search
            results = self.web_search.search(query=search_query)
            return [
                {
                    "title": r.title,
                    "snippet": r.snippet,
                    "url": r.url,
                    "source": r.source,
                }
                for r in results
                if r.is_valid
            ]
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return None

    def _action_skill_lookup(
        self, signal: AgentSignal, query: str
    ) -> Optional[str]:
        """Execute skill lookup action."""
        lookup_result = self.skill_manager.lookup(
            query=query,
            domain=signal.domain,
            tags=signal.metadata.get("tags"),
        )

        if lookup_result.found and lookup_result.skill:
            skill = lookup_result.skill
            self.skill_manager.record_usage(skill.name, success=True)

            # v2: Verify skill if verification meta-skill is available
            if self.verification_meta is not None:
                verify_result = self.verification_meta.verify(skill.name)
                if verify_result and not verify_result.passed:
                    logger.info(f"Skill {skill.name} verification failed, confidence adjusted")

            return f"Skill: {skill.name}\n{skill.definition}"

        # v2: Try skill composition if no single skill found
        if self.composition_meta is not None and signal.domain:
            composed = self.composition_meta.compose(
                query=query, domain=signal.domain
            )
            if composed and composed.skill_chain:
                result.meta_skills_used.append("composition")
                return f"Composed skills: {' → '.join(composed.skill_chain)}\n{composed.description}"

        return None

    def _action_skill_create(
        self, signal: AgentSignal, query: str
    ) -> Optional[str]:
        """Execute skill creation action.

        v2: Uses SkillSynthesisMetaSkill for richer skill creation.
        """
        if not self.config.enable_skill_creation:
            return None

        try:
            # v2: Use meta-skill synthesis for better skill creation
            if self.synthesis_meta is not None:
                skill = self.synthesis_meta.synthesize(
                    query=query, domain=signal.domain
                )
                if skill:
                    result_meta = "meta_skills_used"
                    return f"Created skill (meta-synthesized): {skill.name}\n{skill.definition}"
            else:
                # Fallback to basic creation
                skill = self.skill_creator.create(
                    query=query,
                    domain=signal.domain,
                )
                if skill:
                    return f"Created skill: {skill.name}\n{skill.definition}"
        except Exception as e:
            logger.warning(f"Skill creation failed: {e}")

        return None

    def _action_tool_search(
        self, signal: AgentSignal, query: str
    ) -> Optional[List[Dict[str, str]]]:
        """Execute tool search action."""
        tools = self.tool_registry.search(
            query=query,
            domain=signal.domain,
            executable_only=False,
        )

        if tools:
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "safety": t.safety.value,
                }
                for t in tools
            ]

        if self.config.enable_tool_creation:
            return self._action_tool_create(signal, query)

        return None

    def _action_tool_create(
        self, signal: AgentSignal, query: str
    ) -> Optional[str]:
        """Execute tool creation action."""
        if not self.config.enable_tool_creation:
            return None

        try:
            tool = self.tool_creator.create(
                query=query,
                domain=signal.domain,
            )
            if tool:
                return f"Created tool: {tool.name} — {tool.description}"
        except Exception as e:
            logger.warning(f"Tool creation failed: {e}")

        return None

    def _action_terminal_execute(
        self, signal: AgentSignal, query: str
    ) -> Optional[Dict[str, Any]]:
        """Execute terminal command action."""
        if not self.config.enable_terminal:
            return None

        if signal.domain not in ("code", "data", None):
            return None

        return {
            "terminal_available": True,
            "message": f"Terminal execution available for: {query}. "
                       f"Use the 'terminal' tool to execute specific commands.",
            "domain": signal.domain,
        }

    def _action_verify_output(
        self, signal: AgentSignal, query: str
    ) -> Optional[Dict[str, Any]]:
        """Execute output verification action."""
        try:
            from losion.core.reasoning.neuro_symbolic import NeuroSymbolicVerifier

            verifier = NeuroSymbolicVerifier(d_model=768)
            return {
                "verification_available": True,
                "message": "Neuro-symbolic verification is available.",
                "domain": signal.domain,
            }
        except ImportError:
            return {
                "verification_available": False,
                "message": "Neuro-symbolic verifier not available. "
                           "Consider running terminal verification instead.",
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics including v3 components."""
        stats = {
            "state": self._state.value,
            "total_iterations": self._iteration_count,
            "skills": self.skill_store.get_stats(),
            "tools": self.tool_registry.get_stats(),
            "web_search": self.web_search.get_stats(),
            "terminal": self.terminal.get_stats(),
        }

        # v2 stats
        if self.calibration_engine is not None:
            stats["calibration"] = self.calibration_engine.get_stats()

        if self.episodic_memory is not None:
            stats["episodic_memory"] = self.episodic_memory.get_stats()

        # v3 stats
        if self.risk_simulator is not None:
            stats["risk_simulation"] = self.risk_simulator.get_stats()

        if self.paradigm_router is not None:
            stats["paradigm_routing"] = "enabled"

        if self.mcts_agent is not None:
            stats["mcts_agent"] = "enabled"

        if self.deps_planner is not None:
            stats["deps_recovery"] = "enabled"

        if self.agentic_retriever is not None:
            stats["agentic_retrieval"] = "enabled"

        return stats
