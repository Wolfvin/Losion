"""
Agent Orchestrator — Central coordination for the Losion Agent Layer.

The Orchestrator is the "brain" of the agent layer. It sits between the
Losion model and the agent's capabilities (skills, tools, web search,
terminal), deciding when and how to use each one based on model signals.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    USER QUERY                        │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │              LOSION MODEL (Tri-Jalur)                │
    │   → produces: routing_weights, thinking_mode,       │
    │     confidence, task_type                            │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │            SIGNAL EXTRACTOR                          │
    │   → produces: AgentSignal (action, priority)        │
    └──────────────────────┬──────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │              AGENT ORCHESTRATOR                      │
    │                                                      │
    │   if signal.action == MODEL_ONLY:                    │
    │       return model output                            │
    │   elif signal.action == WEB_SEARCH:                  │
    │       search → inject context → re-infer             │
    │   elif signal.action == SKILL_LOOKUP:                │
    │       find skill → apply → return                    │
    │   elif signal.action == SKILL_CREATE:                │
    │       create skill → apply → return                  │
    │   elif signal.action == TOOL_SEARCH:                 │
    │       find tool → execute → return                   │
    │   elif signal.action == TOOL_CREATE:                 │
    │       create tool → execute → return                 │
    │   elif signal.action == TERMINAL_EXECUTE:            │
    │       sandbox → execute → return                     │
    │   elif signal.action == VERIFY_OUTPUT:               │
    │       verify → possibly revise → return              │
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
    CONTEXT_INJECTION = "context_injection"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for the Agent Orchestrator.

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


@dataclass
class AgentResult:
    """Result of an agent loop execution.

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

    @property
    def success(self) -> bool:
        """Whether the agent loop completed successfully."""
        return self.state == OrchestratorState.COMPLETE


class AgentOrchestrator:
    """Central orchestrator for the Losion Agent Layer.

    The orchestrator manages the agent loop:
    1. Receive model output
    2. Extract signals
    3. Decide action
    4. Execute action
    5. Inject context (if applicable)
    6. Repeat until confidence is sufficient or max iterations reached

    The orchestrator is decoupled from the model — it receives model
    output as input and returns results, never modifying the model.

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
        self.signal_extractor = SignalExtractor(
            thresholds=self.config.confidence_thresholds,
            enable_web_search=self.config.enable_web_search,
            enable_terminal=self.config.enable_terminal,
            enable_skill_creation=self.config.enable_skill_creation,
            enable_tool_creation=self.config.enable_tool_creation,
        )

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

        # Register built-in tools
        self._register_builtin_tools()

        # State
        self._state = OrchestratorState.IDLE
        self._iteration_count = 0

    def _register_builtin_tools(self) -> None:
        """Register built-in tools that are always available."""
        # Web search tool
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

        # Terminal tool
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

    def run(
        self,
        model_output: Any = None,
        query: str = "",
        confidence: Optional[float] = None,
        model_inference_fn: Optional[Callable] = None,
        context: Optional[List[str]] = None,
    ) -> AgentResult:
        """Run the agent loop.

        The agent loop:
        1. Extract signal from model output
        2. If MODEL_ONLY → return current output
        3. If action needed → execute action
        4. Inject context from action result
        5. Re-infer with new context (if model_inference_fn provided)
        6. Repeat until confidence sufficient or max iterations

        Args:
            model_output: Output from the Losion model (AdaptiveRoutingOutput).
            query: The user's query text.
            confidence: Override confidence score.
            model_inference_fn: Function to call for re-inference with context.
                               Signature: fn(query, context_list) → (output, confidence)
            context: Pre-existing context to inject.

        Returns:
            AgentResult with the final output and execution details.
        """
        start_time = time.time()
        self._state = OrchestratorState.MODEL_INFERENCE
        self._iteration_count = 0

        result = AgentResult()
        current_context = list(context) if context else []
        current_output = model_output
        current_confidence = confidence

        for iteration in range(self.config.max_iterations):
            self._iteration_count = iteration + 1
            self._state = OrchestratorState.SIGNAL_EXTRACTION

            # === Extract signal ===
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
                    f"priority={signal.priority:.2f}"
                )

            # === Check if intervention needed ===
            if not signal.needs_intervention:
                self._state = OrchestratorState.COMPLETE
                result.state = self._state
                result.model_confidence = signal.confidence
                break

            # === Execute action ===
            action_result = self._execute_action(signal, query, current_context)
            result.actions_taken.append(signal.action.value)

            # === Process action result ===
            if action_result is not None:
                # Add to context
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
                        # Keep previous output
                        pass
            # Check if confidence has improved enough
            if current_confidence is not None and current_confidence >= 0.7:
                self._state = OrchestratorState.COMPLETE
                result.state = self._state
                result.model_confidence = current_confidence
                break

        # === Finalize ===
        if self._state != OrchestratorState.COMPLETE:
            self._state = OrchestratorState.COMPLETE
            result.state = self._state

        result.output = current_output
        result.iterations = self._iteration_count
        result.model_confidence = current_confidence or signal.confidence
        result.total_time = time.time() - start_time

        return result

    def _execute_action(
        self,
        signal: AgentSignal,
        query: str,
        context: List[str],
    ) -> Any:
        """Execute the action specified by the signal.

        Args:
            signal: AgentSignal with the action to execute.
            query: Original query.
            context: Current context list.

        Returns:
            Action result (context string, dict, or list).
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

        Args:
            signal: The web search signal.
            query: Search query.

        Returns:
            List of search result dicts.
        """
        search_query = signal.query or query
        logger.info(f"Web search: {search_query}")

        try:
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
        """Execute skill lookup action.

        Args:
            signal: The skill lookup signal.
            query: Skill query.

        Returns:
            Skill definition if found.
        """
        lookup_result = self.skill_manager.lookup(
            query=query,
            domain=signal.domain,
            tags=signal.metadata.get("tags"),
        )

        if lookup_result.found and lookup_result.skill:
            skill = lookup_result.skill
            self.skill_manager.record_usage(skill.name, success=True)
            return f"Skill: {skill.name}\n{skill.definition}"

        return None

    def _action_skill_create(
        self, signal: AgentSignal, query: str
    ) -> Optional[str]:
        """Execute skill creation action.

        Args:
            signal: The skill create signal.
            query: Skill query.

        Returns:
            Created skill definition.
        """
        if not self.config.enable_skill_creation:
            return None

        try:
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
        """Execute tool search action.

        Args:
            signal: The tool search signal.
            query: Tool query.

        Returns:
            List of matching tool descriptions.
        """
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

        # If no tool found and creation is enabled, create one
        if self.config.enable_tool_creation:
            return self._action_tool_create(signal, query)

        return None

    def _action_tool_create(
        self, signal: AgentSignal, query: str
    ) -> Optional[str]:
        """Execute tool creation action.

        Args:
            signal: The tool create signal.
            query: Tool query.

        Returns:
            Description of the created tool.
        """
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
        """Execute terminal command action.

        Only executed for code/data domain queries where terminal
        execution is relevant and safe.

        Args:
            signal: The terminal execute signal.
            query: Command or task description.

        Returns:
            Terminal execution result dict.
        """
        if not self.config.enable_terminal:
            return None

        # Only execute if domain is code or data
        if signal.domain not in ("code", "data", None):
            return None

        # For safety, don't auto-execute arbitrary commands
        # Instead, return a message indicating terminal is available
        return {
            "terminal_available": True,
            "message": f"Terminal execution available for: {query}. "
                       f"Use the 'terminal' tool to execute specific commands.",
            "domain": signal.domain,
        }

    def _action_verify_output(
        self, signal: AgentSignal, query: str
    ) -> Optional[Dict[str, Any]]:
        """Execute output verification action.

        This would integrate with Losion's Neuro-Symbolic Verifier
        if available, otherwise provides a basic check.

        Args:
            signal: The verify signal.
            query: The query to verify against.

        Returns:
            Verification result dict.
        """
        try:
            from losion.core.reasoning.neuro_symbolic import NeuroSymbolicVerifier

            # If the model's neuro-symbolic verifier is available, use it
            verifier = NeuroSymbolicVerifier(d_model=768)
            # Note: In practice, you'd pass the actual hidden states
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
        """Get orchestrator statistics."""
        return {
            "state": self._state.value,
            "total_iterations": self._iteration_count,
            "skills": self.skill_store.get_stats(),
            "tools": self.tool_registry.get_stats(),
            "web_search": self.web_search.get_stats(),
            "terminal": self.terminal.get_stats(),
        }
