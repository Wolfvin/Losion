"""
Signal Extraction — Bridge between Losion model output and agent decisions.

This module translates model-internal signals (ThinkingToggle assessment,
routing weights, confidence scores) into actionable agent signals. The model
produces these signals during inference; the agent layer reads them and
decides whether to intervene (web search, tool use, skill lookup) or let
the model continue autonomously.

This is the KEY integration point: the model provides "hints" about what
it needs, and the agent layer responds with the appropriate action.

Design:
    Model Output → SignalExtractor → AgentSignal → Orchestrator → Action

The extractor never modifies model behavior — it only reads model outputs
and produces recommendations for the agent layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Attempt to import model types; fall back to duck-typing if unavailable
try:
    from losion.core.router import (
        AdaptiveRoutingOutput,
        ThinkingMode,
        TaskType,
    )
except ImportError:
    # Allow standalone usage without model imports
    AdaptiveRoutingOutput = None
    ThinkingMode = None
    TaskType = None


class AgentAction(Enum):
    """Possible actions the agent layer can take.

    Each action corresponds to a different type of intervention:
    - MODEL_ONLY: No intervention needed, let model continue
    - SKILL_LOOKUP: Check skill store for relevant skills
    - SKILL_CREATE: Create a new skill (when lookup fails)
    - TOOL_SEARCH: Find a tool for the current task
    - TOOL_CREATE: Create a new tool (when search fails)
    - WEB_SEARCH: Search the web for context/information
    - TERMINAL_EXECUTE: Run a terminal command (sandboxed)
    - VERIFY_OUTPUT: Use neuro-symbolic verification on model output
    """

    MODEL_ONLY = "model_only"
    SKILL_LOOKUP = "skill_lookup"
    SKILL_CREATE = "skill_create"
    TOOL_SEARCH = "tool_search"
    TOOL_CREATE = "tool_create"
    WEB_SEARCH = "web_search"
    TERMINAL_EXECUTE = "terminal_execute"
    VERIFY_OUTPUT = "verify_output"


@dataclass
class ConfidenceThreshold:
    """Threshold configuration for confidence-based signal extraction.

    The agent layer uses these thresholds to decide when to intervene.
    Lower thresholds = more aggressive intervention (more agent actions).
    Higher thresholds = more conservative (let model handle more).

    Attributes:
        web_search: Trigger web search when confidence drops below this.
        skill_lookup: Trigger skill lookup when confidence drops below this.
        tool_search: Trigger tool search when confidence drops below this.
        verify: Trigger verification when confidence drops below this.
        terminal: Trigger terminal execution when confidence drops below this.
    """

    web_search: float = 0.3
    skill_lookup: float = 0.4
    tool_search: float = 0.35
    verify: float = 0.5
    terminal: float = 0.25

    def __post_init__(self) -> None:
        """Validate threshold values."""
        for name, value in [
            ("web_search", self.web_search),
            ("skill_lookup", self.skill_lookup),
            ("tool_search", self.tool_search),
            ("verify", self.verify),
            ("terminal", self.terminal),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"ConfidenceThreshold.{name} must be in [0.0, 1.0], got {value}"
                )


@dataclass
class AgentSignal:
    """A signal from the model to the agent layer.

    This is the output of SignalExtractor — it represents what the agent
    layer should do based on the model's current state.

    Attributes:
        action: The recommended agent action.
        confidence: Model's confidence score [0.0, 1.0].
        reasoning: Why this action was recommended.
        query: Search query or task description (for web_search/skill_lookup).
        domain: Domain classification (for skill/tool routing).
        task_type: Type of task (sequential/reasoning/factual).
        thinking_mode: Whether model is in thinking or non-thinking mode.
        routing_weights: Copy of routing weights for diagnostics.
        metadata: Additional context for the action.
        priority: Priority level (higher = more urgent intervention).
    """

    action: AgentAction = AgentAction.MODEL_ONLY
    confidence: float = 1.0
    reasoning: str = ""
    query: Optional[str] = None
    domain: Optional[str] = None
    task_type: Optional[str] = None
    thinking_mode: Optional[str] = None
    routing_weights: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0

    @property
    def needs_intervention(self) -> bool:
        """Whether this signal requires agent intervention."""
        return self.action != AgentAction.MODEL_ONLY


class SignalExtractor:
    """Extracts agent signals from Losion model output.

    This is the core bridge between the model and the agent layer. It reads
    model outputs (routing weights, thinking assessment, confidence scores)
    and translates them into actionable AgentSignal objects.

    The extractor uses a multi-signal fusion approach:
    1. Confidence signal: Low confidence → need external information
    2. Routing signal: Retrieval-dominant → need lookup/search
    3. Thinking signal: Thinking mode + low confidence → deep intervention
    4. Task type signal: Factual task + no knowledge → web search

    All signals are fused to produce a single, prioritized AgentSignal.

    Args:
        thresholds: Confidence thresholds for different actions.
        enable_web_search: Whether web search is available.
        enable_terminal: Whether terminal execution is available.
        enable_skill_creation: Whether skill auto-creation is enabled.
        enable_tool_creation: Whether tool auto-creation is enabled.
    """

    # Domain keywords for classification
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "math": ["calculate", "equation", "formula", "solve", "integral", "derivative", "proof"],
        "code": ["function", "class", "method", "import", "debug", "compile", "runtime", "syntax"],
        "science": ["experiment", "hypothesis", "theory", "molecule", "reaction", "physics"],
        "history": ["when", "historical", "century", "war", "civilization", "dynasty"],
        "language": ["translate", "grammar", "vocabulary", "meaning", "etymology"],
        "web": ["website", "api", "url", "http", "endpoint", "request"],
        "data": ["dataset", "database", "query", "csv", "json", "schema"],
    }

    def __init__(
        self,
        thresholds: Optional[ConfidenceThreshold] = None,
        enable_web_search: bool = True,
        enable_terminal: bool = True,
        enable_skill_creation: bool = True,
        enable_tool_creation: bool = True,
    ) -> None:
        self.thresholds = thresholds or ConfidenceThreshold()
        self.enable_web_search = enable_web_search
        self.enable_terminal = enable_terminal
        self.enable_skill_creation = enable_skill_creation
        self.enable_tool_creation = enable_tool_creation

    def extract(
        self,
        model_output: Any,
        confidence: Optional[float] = None,
        query_text: Optional[str] = None,
    ) -> AgentSignal:
        """Extract agent signal from model output.

        This is the main entry point. It reads model output and produces
        a single, prioritized AgentSignal.

        Args:
            model_output: Output from AdaptiveRouter.forward() or similar.
                         Expected to have: thinking_assessment, adjusted_weights.
            confidence: Override confidence score (if not in model_output).
            query_text: The original query text (for domain classification).

        Returns:
            AgentSignal with the recommended action.
        """
        # === Extract signals from model output ===
        thinking_mode = self._extract_thinking_mode(model_output)
        routing_weights = self._extract_routing_weights(model_output)
        model_confidence = self._extract_confidence(model_output)
        task_type = self._extract_task_type(model_output)

        # Use override confidence if provided
        effective_confidence = confidence if confidence is not None else model_confidence

        # === Classify domain from query text ===
        domain = self._classify_domain(query_text) if query_text else None

        # === Multi-signal fusion ===
        signals = self._fuse_signals(
            confidence=effective_confidence,
            thinking_mode=thinking_mode,
            routing_weights=routing_weights,
            task_type=task_type,
            domain=domain,
            query_text=query_text,
        )

        # === Return highest-priority signal ===
        if not signals:
            return AgentSignal(
                action=AgentAction.MODEL_ONLY,
                confidence=effective_confidence,
                reasoning="Model confidence sufficient, no intervention needed.",
                thinking_mode=thinking_mode,
                routing_weights=routing_weights,
                task_type=task_type,
                domain=domain,
            )

        # Sort by priority (highest first)
        signals.sort(key=lambda s: s.priority, reverse=True)
        best = signals[0]
        return best

    def _extract_thinking_mode(self, model_output: Any) -> Optional[str]:
        """Extract thinking mode from model output.

        Handles both proper AdaptiveRoutingOutput and duck-typed objects.
        """
        if model_output is None:
            return None

        # Try attribute access (proper model output)
        if hasattr(model_output, "thinking_assessment"):
            assessment = model_output.thinking_assessment
            if hasattr(assessment, "mode"):
                return assessment.mode.value if hasattr(assessment.mode, "value") else str(assessment.mode)

        # Try dictionary access
        if isinstance(model_output, dict):
            assessment = model_output.get("thinking_assessment")
            if assessment and hasattr(assessment, "mode"):
                return assessment.mode.value if hasattr(assessment.mode, "value") else str(assessment.mode)
            return model_output.get("thinking_mode")

        return None

    def _extract_routing_weights(self, model_output: Any) -> Optional[List[float]]:
        """Extract routing weights from model output.

        Returns [w_ssm, w_attn, w_retr] if available.
        """
        if model_output is None:
            return None

        # Try attribute access
        if hasattr(model_output, "adjusted_weights"):
            weights = model_output.adjusted_weights
            if hasattr(weights, "mean"):
                # PyTorch tensor — average over batch and seq dims
                return weights.mean(dim=(0, 1)).tolist()
            return list(weights)

        # Try dictionary access
        if isinstance(model_output, dict):
            weights = model_output.get("adjusted_weights") or model_output.get("routing_weights")
            if weights is not None:
                if hasattr(weights, "mean"):
                    return weights.mean(dim=(0, 1)).tolist()
                return list(weights)

        return None

    def _extract_confidence(self, model_output: Any) -> float:
        """Extract or estimate model confidence.

        Uses multiple heuristics:
        1. Explicit confidence from thinking assessment
        2. Routing entropy (high entropy = uncertain)
        3. Default to moderate confidence
        """
        if model_output is None:
            return 0.5  # Unknown = moderate

        # Try explicit confidence
        if hasattr(model_output, "thinking_assessment"):
            assessment = model_output.thinking_assessment
            if hasattr(assessment, "confidence"):
                return float(assessment.confidence)

        # Try dictionary
        if isinstance(model_output, dict):
            return float(model_output.get("confidence", 0.5))

        # Estimate from routing weights entropy
        weights = self._extract_routing_weights(model_output)
        if weights is not None:
            import math
            total = sum(weights)
            if total > 0:
                normalized = [w / total for w in weights]
                entropy = -sum(w * math.log(w + 1e-8) for w in normalized if w > 0)
                max_entropy = math.log(len(normalized))
                # High entropy = uncertain = low confidence
                if max_entropy > 0:
                    uncertainty = entropy / max_entropy
                    return 1.0 - uncertainty

        return 0.5

    def _extract_task_type(self, model_output: Any) -> Optional[str]:
        """Extract dominant task type from model output."""
        if model_output is None:
            return None

        if hasattr(model_output, "thinking_assessment"):
            assessment = model_output.thinking_assessment
            if hasattr(assessment, "dominant_task"):
                task = assessment.dominant_task
                return task.value if hasattr(task, "value") else str(task)

        if isinstance(model_output, dict):
            return model_output.get("task_type")

        return None

    def _classify_domain(self, query_text: Optional[str]) -> Optional[str]:
        """Classify the domain of a query based on keywords.

        Args:
            query_text: The user's query text.

        Returns:
            Domain string or None if unclassifiable.
        """
        if not query_text:
            return None

        query_lower = query_text.lower()
        domain_scores: Dict[str, int] = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return None

        return max(domain_scores, key=domain_scores.get)

    def _fuse_signals(
        self,
        confidence: float,
        thinking_mode: Optional[str],
        routing_weights: Optional[List[float]],
        task_type: Optional[str],
        domain: Optional[str],
        query_text: Optional[str],
    ) -> List[AgentSignal]:
        """Fuse multiple signals to produce prioritized agent actions.

        This is the core decision logic. It considers all available signals
        and produces a list of candidate actions, each with a priority score.

        Priority scoring:
        - Base priority from confidence deficit (1.0 - confidence)
        - Boosted by thinking mode (thinking + low confidence = urgent)
        - Boosted by retrieval-dominant routing (model knows it needs info)
        - Boosted by factual task type (factual = need external knowledge)

        Args:
            confidence: Model confidence [0.0, 1.0].
            thinking_mode: "thinking" or "non_thinking" or None.
            routing_weights: [w_ssm, w_attn, w_retr] or None.
            task_type: "sequential", "reasoning", or "factual" or None.
            domain: Domain classification or None.
            query_text: Original query text.

        Returns:
            List of AgentSignal candidates, sorted by priority.
        """
        signals: List[AgentSignal] = []
        confidence_deficit = 1.0 - confidence

        # === Signal 1: Web Search ===
        if self.enable_web_search and confidence < self.thresholds.web_search:
            priority = confidence_deficit * 1.0

            # Boost: thinking mode + low confidence = urgent search
            if thinking_mode == "thinking":
                priority *= 1.3

            # Boost: factual task + low confidence = definitely need search
            if task_type == "factual":
                priority *= 1.5

            # Boost: retrieval-dominant routing = model wants external info
            if routing_weights and len(routing_weights) >= 3:
                retrieval_weight = routing_weights[2]  # Jalur 3
                if retrieval_weight > 0.5:
                    priority *= 1.2

            signals.append(AgentSignal(
                action=AgentAction.WEB_SEARCH,
                confidence=confidence,
                reasoning=(
                    f"Model confidence ({confidence:.2f}) below web search threshold "
                    f"({self.thresholds.web_search}). "
                    f"Thinking mode: {thinking_mode}, Task: {task_type}."
                ),
                query=query_text,
                domain=domain,
                task_type=task_type,
                thinking_mode=thinking_mode,
                routing_weights=routing_weights,
                priority=min(priority, 2.0),  # Cap at 2.0
            ))

        # === Signal 2: Skill Lookup ===
        if confidence < self.thresholds.skill_lookup and domain is not None:
            priority = confidence_deficit * 0.9

            # Boost: domain has specialized skills
            if domain in ("math", "code", "science", "data"):
                priority *= 1.3

            signals.append(AgentSignal(
                action=AgentAction.SKILL_LOOKUP,
                confidence=confidence,
                reasoning=(
                    f"Model confidence ({confidence:.2f}) below skill lookup threshold "
                    f"({self.thresholds.skill_lookup}). Domain: {domain}."
                ),
                query=query_text,
                domain=domain,
                task_type=task_type,
                thinking_mode=thinking_mode,
                routing_weights=routing_weights,
                priority=min(priority, 1.8),
            ))

        # === Signal 3: Tool Search ===
        if confidence < self.thresholds.tool_search:
            priority = confidence_deficit * 0.85

            # Boost: code/data domain needs tools
            if domain in ("code", "data", "web"):
                priority *= 1.4

            # Boost: reasoning task might need calculation tools
            if task_type == "reasoning":
                priority *= 1.2

            signals.append(AgentSignal(
                action=AgentAction.TOOL_SEARCH,
                confidence=confidence,
                reasoning=(
                    f"Model confidence ({confidence:.2f}) below tool search threshold "
                    f"({self.thresholds.tool_search}). Domain: {domain}."
                ),
                query=query_text,
                domain=domain,
                task_type=task_type,
                thinking_mode=thinking_mode,
                routing_weights=routing_weights,
                priority=min(priority, 1.6),
            ))

        # === Signal 4: Terminal Execution ===
        if self.enable_terminal and confidence < self.thresholds.terminal:
            priority = confidence_deficit * 0.7

            # Only relevant for code/data tasks
            if domain in ("code", "data"):
                priority *= 1.5
            else:
                priority *= 0.3  # Suppress for non-code tasks

            if priority > 0.2:  # Only include if meaningful
                signals.append(AgentSignal(
                    action=AgentAction.TERMINAL_EXECUTE,
                    confidence=confidence,
                    reasoning=(
                        f"Model confidence ({confidence:.2f}) below terminal threshold "
                        f"({self.thresholds.terminal}). Domain: {domain}. "
                        f"Terminal execution may help verify or execute code."
                    ),
                    query=query_text,
                    domain=domain,
                    task_type=task_type,
                    thinking_mode=thinking_mode,
                    routing_weights=routing_weights,
                    priority=min(priority, 1.4),
                ))

        # === Signal 5: Verify Output ===
        if confidence < self.thresholds.verify:
            priority = confidence_deficit * 0.6

            # Boost: math/logic needs verification
            if domain in ("math", "code") or task_type == "reasoning":
                priority *= 1.3

            signals.append(AgentSignal(
                action=AgentAction.VERIFY_OUTPUT,
                confidence=confidence,
                reasoning=(
                    f"Model confidence ({confidence:.2f}) below verification threshold "
                    f"({self.thresholds.verify}). Neuro-symbolic verification recommended."
                ),
                query=query_text,
                domain=domain,
                task_type=task_type,
                thinking_mode=thinking_mode,
                routing_weights=routing_weights,
                priority=min(priority, 1.2),
            ))

        return signals
