"""
Signal Extraction — Bridge between Losion model output and agent decisions.

This module translates model-internal signals (ThinkingToggle assessment,
routing weights, confidence scores) into actionable agent signals. The model
produces these signals during inference; the agent layer reads them and
decides whether to intervene (web search, tool use, skill lookup) or let
the model continue autonomously.

This is the KEY integration point: the model provides "hints" about what
it needs, and the agent layer responds with the appropriate action.

v2 Improvements (based on research):
- Adaptive confidence thresholds via CalibrationEngine (ATTC, 2026)
- Episodic memory integration for experience-based decisions (Reflexion, 2023)
- Tool trust scores influence signal priority (ATTC, 2026)
- Action recommendations from past experience (EpisodicMemory)
- Domain-specific threshold calibration (DomainProfiles)

Design:
    Model Output → SignalExtractor → AgentSignal → Orchestrator → Action
                        ↑
            CalibrationEngine (adaptive thresholds)
            EpisodicMemory (past experience)
            ToolTrustScores (tool reliability)
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
    - REFLECT: Self-reflect on previous action outcomes (v2)
    """

    MODEL_ONLY = "model_only"
    SKILL_LOOKUP = "skill_lookup"
    SKILL_CREATE = "skill_create"
    TOOL_SEARCH = "tool_search"
    TOOL_CREATE = "tool_create"
    WEB_SEARCH = "web_search"
    TERMINAL_EXECUTE = "terminal_execute"
    VERIFY_OUTPUT = "verify_output"
    REFLECT = "reflect"


@dataclass
class ConfidenceThreshold:
    """Threshold configuration for confidence-based signal extraction.

    v2: These are now the DEFAULT thresholds. The CalibrationEngine
    can override them with adaptive, experience-based values.

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

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ConfidenceThreshold":
        """Create thresholds from a dictionary (e.g., from CalibrationEngine)."""
        return cls(
            web_search=data.get("web_search", 0.3),
            skill_lookup=data.get("skill_lookup", 0.4),
            tool_search=data.get("tool_search", 0.35),
            verify=data.get("verify", 0.5),
            terminal=data.get("terminal", 0.25),
        )


@dataclass
class AgentSignal:
    """A signal from the model to the agent layer.

    This is the output of SignalExtractor — it represents what the agent
    layer should do based on the model's current state.

    v2 additions:
    - tool_trust: Trust score for the recommended action's tool
    - episodic_relevance: How relevant past experience is
    - recommended_from_experience: Whether this action is recommended
      based on past episodes

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
        tool_trust: Trust score for the recommended tool (v2).
        episodic_relevance: Relevance of past experience (v2).
        recommended_from_experience: Whether action is experience-based (v2).
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
    # v2 additions
    tool_trust: float = 0.5
    episodic_relevance: float = 0.0
    recommended_from_experience: bool = False

    @property
    def needs_intervention(self) -> bool:
        """Whether this signal requires agent intervention."""
        return self.action != AgentAction.MODEL_ONLY


class SignalExtractor:
    """Extracts agent signals from Losion model output.

    v2 Improvements:
    - Uses CalibrationEngine for adaptive thresholds
    - Queries EpisodicMemory for experience-based recommendations
    - Incorporates tool trust scores into signal priority
    - Generates REFLECT signals when past experience suggests it

    The extractor uses a multi-signal fusion approach:
    1. Confidence signal: Low confidence → need external information
    2. Routing signal: Retrieval-dominant → need lookup/search
    3. Thinking signal: Thinking mode + low confidence → deep intervention
    4. Task type signal: Factual task + no knowledge → web search
    5. Experience signal (v2): Past episodes recommend specific actions
    6. Trust signal (v2): Tool trust influences action priority

    All signals are fused to produce a single, prioritized AgentSignal.

    Args:
        thresholds: Confidence thresholds for different actions.
        enable_web_search: Whether web search is available.
        enable_terminal: Whether terminal execution is available.
        enable_skill_creation: Whether skill auto-creation is enabled.
        enable_tool_creation: Whether tool auto-creation is enabled.
        calibration_engine: Optional CalibrationEngine for adaptive thresholds.
        episodic_memory: Optional EpisodicMemory for experience-based signals.
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
        calibration_engine: Optional[Any] = None,
        episodic_memory: Optional[Any] = None,
    ) -> None:
        self.thresholds = thresholds or ConfidenceThreshold()
        self.enable_web_search = enable_web_search
        self.enable_terminal = enable_terminal
        self.enable_skill_creation = enable_skill_creation
        self.enable_tool_creation = enable_tool_creation
        self.calibration_engine = calibration_engine
        self.episodic_memory = episodic_memory

    def extract(
        self,
        model_output: Any,
        confidence: Optional[float] = None,
        query_text: Optional[str] = None,
    ) -> AgentSignal:
        """Extract agent signal from model output.

        v2: Uses adaptive thresholds and episodic memory when available.

        Args:
            model_output: Output from AdaptiveRouter.forward() or similar.
            confidence: Override confidence score.
            query_text: The original query text.

        Returns:
            AgentSignal with the recommended action.
        """
        # === Extract signals from model output ===
        thinking_mode = self._extract_thinking_mode(model_output)
        routing_weights = self._extract_routing_weights(model_output)
        model_confidence = self._extract_confidence(model_output)
        task_type = self._extract_task_type(model_output)

        effective_confidence = confidence if confidence is not None else model_confidence
        domain = self._classify_domain(query_text) if query_text else None

        # === v2: Get adaptive thresholds ===
        thresholds = self._get_adaptive_thresholds(domain)

        # === v2: Get experience-based recommendations ===
        experience_recs = self._get_experience_recommendations(query_text, domain)

        # === Multi-signal fusion ===
        signals = self._fuse_signals(
            confidence=effective_confidence,
            thinking_mode=thinking_mode,
            routing_weights=routing_weights,
            task_type=task_type,
            domain=domain,
            query_text=query_text,
            thresholds=thresholds,
            experience_recs=experience_recs,
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

        signals.sort(key=lambda s: s.priority, reverse=True)
        best = signals[0]
        return best

    def _get_adaptive_thresholds(self, domain: Optional[str]) -> ConfidenceThreshold:
        """Get thresholds from CalibrationEngine if available.

        v2: Instead of using static thresholds, query the CalibrationEngine
        for adaptive, experience-based thresholds.
        """
        if self.calibration_engine is not None:
            adaptive = self.calibration_engine.get_thresholds(domain)
            return ConfidenceThreshold.from_dict(adaptive)
        return self.thresholds

    def _get_experience_recommendations(
        self, query_text: Optional[str], domain: Optional[str]
    ) -> Dict[str, float]:
        """Get action recommendations from episodic memory.

        v2: If episodic memory is available, check what actions have
        been successful for similar past queries.
        """
        if self.episodic_memory is None or not query_text:
            return {}

        try:
            return self.episodic_memory.get_action_recommendations(
                query=query_text, domain=domain
            )
        except Exception:
            return {}

    def _fuse_signals(
        self,
        confidence: float,
        thinking_mode: Optional[str],
        routing_weights: Optional[List[float]],
        task_type: Optional[str],
        domain: Optional[str],
        query_text: Optional[str],
        thresholds: Optional[ConfidenceThreshold] = None,
        experience_recs: Optional[Dict[str, float]] = None,
    ) -> List[AgentSignal]:
        """Fuse multiple signals to produce prioritized agent actions.

        v2 additions:
        - Uses adaptive thresholds from CalibrationEngine
        - Incorporates experience-based recommendations
        - Adds tool trust scores to signal metadata
        - Generates REFLECT signals when experience suggests past failures

        Args:
            confidence: Model confidence [0.0, 1.0].
            thinking_mode: "thinking" or "non_thinking" or None.
            routing_weights: [w_ssm, w_attn, w_retr] or None.
            task_type: "sequential", "reasoning", or "factual" or None.
            domain: Domain classification or None.
            query_text: Original query text.
            thresholds: Adaptive thresholds (v2).
            experience_recs: Experience-based recommendations (v2).

        Returns:
            List of AgentSignal candidates, sorted by priority.
        """
        signals: List[AgentSignal] = []
        confidence_deficit = 1.0 - confidence
        effective_thresholds = thresholds or self.thresholds
        recs = experience_recs or {}

        # === Signal 1: Web Search ===
        if self.enable_web_search and confidence < effective_thresholds.web_search:
            priority = confidence_deficit * 1.0

            if thinking_mode == "thinking":
                priority *= 1.3
            if task_type == "factual":
                priority *= 1.5
            if routing_weights and len(routing_weights) >= 3:
                retrieval_weight = routing_weights[2]
                if retrieval_weight > 0.5:
                    priority *= 1.2

            # v2: Boost if experience recommends web search
            if "web_search" in recs and recs["web_search"] > 0:
                priority *= 1.2
                experience_boost = True
            else:
                experience_boost = False

            # v2: Tool trust score
            tool_trust = 0.5
            if self.calibration_engine:
                tool_trust = self.calibration_engine.get_tool_trust("web_search", domain)

            signals.append(AgentSignal(
                action=AgentAction.WEB_SEARCH,
                confidence=confidence,
                reasoning=(
                    f"Model confidence ({confidence:.2f}) below web search threshold "
                    f"({effective_thresholds.web_search:.2f}). "
                    f"Thinking mode: {thinking_mode}, Task: {task_type}."
                    f"{' Experience recommends this action.' if experience_boost else ''}"
                ),
                query=query_text,
                domain=domain,
                task_type=task_type,
                thinking_mode=thinking_mode,
                routing_weights=routing_weights,
                priority=min(priority, 2.0),
                tool_trust=tool_trust,
                episodic_relevance=recs.get("web_search", 0.0),
                recommended_from_experience=experience_boost,
            ))

        # === Signal 2: Skill Lookup ===
        if confidence < effective_thresholds.skill_lookup and domain is not None:
            priority = confidence_deficit * 0.9
            if domain in ("math", "code", "science", "data"):
                priority *= 1.3

            # v2: Experience boost
            if "skill_lookup" in recs and recs["skill_lookup"] > 0:
                priority *= 1.2

            tool_trust = 0.5
            if self.calibration_engine:
                tool_trust = self.calibration_engine.get_tool_trust("skill_lookup", domain)

            signals.append(AgentSignal(
                action=AgentAction.SKILL_LOOKUP,
                confidence=confidence,
                reasoning=(
                    f"Model confidence ({confidence:.2f}) below skill lookup threshold "
                    f"({effective_thresholds.skill_lookup:.2f}). Domain: {domain}."
                ),
                query=query_text,
                domain=domain,
                task_type=task_type,
                thinking_mode=thinking_mode,
                routing_weights=routing_weights,
                priority=min(priority, 1.8),
                tool_trust=tool_trust,
                episodic_relevance=recs.get("skill_lookup", 0.0),
            ))

        # === Signal 3: Tool Search ===
        if confidence < effective_thresholds.tool_search:
            priority = confidence_deficit * 0.85
            if domain in ("code", "data", "web"):
                priority *= 1.4
            if task_type == "reasoning":
                priority *= 1.2

            # v2: Experience boost
            if "tool_search" in recs and recs["tool_search"] > 0:
                priority *= 1.2

            tool_trust = 0.5
            if self.calibration_engine:
                tool_trust = self.calibration_engine.get_tool_trust("tool_search", domain)

            signals.append(AgentSignal(
                action=AgentAction.TOOL_SEARCH,
                confidence=confidence,
                reasoning=(
                    f"Model confidence ({confidence:.2f}) below tool search threshold "
                    f"({effective_thresholds.tool_search:.2f}). Domain: {domain}."
                ),
                query=query_text,
                domain=domain,
                task_type=task_type,
                thinking_mode=thinking_mode,
                routing_weights=routing_weights,
                priority=min(priority, 1.6),
                tool_trust=tool_trust,
                episodic_relevance=recs.get("tool_search", 0.0),
            ))

        # === Signal 4: Terminal Execution ===
        if self.enable_terminal and confidence < effective_thresholds.terminal:
            priority = confidence_deficit * 0.7
            if domain in ("code", "data"):
                priority *= 1.5
            else:
                priority *= 0.3

            if priority > 0.2:
                tool_trust = 0.5
                if self.calibration_engine:
                    tool_trust = self.calibration_engine.get_tool_trust("terminal_execute", domain)

                signals.append(AgentSignal(
                    action=AgentAction.TERMINAL_EXECUTE,
                    confidence=confidence,
                    reasoning=(
                        f"Model confidence ({confidence:.2f}) below terminal threshold "
                        f"({effective_thresholds.terminal:.2f}). Domain: {domain}. "
                        f"Terminal execution may help verify or execute code."
                    ),
                    query=query_text,
                    domain=domain,
                    task_type=task_type,
                    thinking_mode=thinking_mode,
                    routing_weights=routing_weights,
                    priority=min(priority, 1.4),
                    tool_trust=tool_trust,
                    episodic_relevance=recs.get("terminal_execute", 0.0),
                ))

        # === Signal 5: Verify Output ===
        if confidence < effective_thresholds.verify:
            priority = confidence_deficit * 0.6
            if domain in ("math", "code") or task_type == "reasoning":
                priority *= 1.3

            signals.append(AgentSignal(
                action=AgentAction.VERIFY_OUTPUT,
                confidence=confidence,
                reasoning=(
                    f"Model confidence ({confidence:.2f}) below verification threshold "
                    f"({effective_thresholds.verify:.2f}). Neuro-symbolic verification recommended."
                ),
                query=query_text,
                domain=domain,
                task_type=task_type,
                thinking_mode=thinking_mode,
                routing_weights=routing_weights,
                priority=min(priority, 1.2),
            ))

        # === Signal 6: Reflect (v2) ===
        # Trigger reflection when past experience shows similar queries
        # had failures that could be avoided
        if self.episodic_memory and query_text:
            try:
                lessons = self.episodic_memory.get_lessons_for_query(
                    query=query_text, domain=domain, limit=2
                )
                if lessons and confidence < 0.6:
                    signals.append(AgentSignal(
                        action=AgentAction.REFLECT,
                        confidence=confidence,
                        reasoning=(
                            f"Past experience with similar queries suggests reflection "
                            f"may improve the approach. Lessons: {'; '.join(lessons[:2])}"
                        ),
                        query=query_text,
                        domain=domain,
                        task_type=task_type,
                        thinking_mode=thinking_mode,
                        routing_weights=routing_weights,
                        priority=0.4,
                        metadata={"lessons": lessons},
                    ))
            except Exception:
                pass

        return signals

    def _extract_thinking_mode(self, model_output: Any) -> Optional[str]:
        """Extract thinking mode from model output."""
        if model_output is None:
            return None
        if hasattr(model_output, "thinking_assessment"):
            assessment = model_output.thinking_assessment
            if hasattr(assessment, "mode"):
                return assessment.mode.value if hasattr(assessment.mode, "value") else str(assessment.mode)
        if isinstance(model_output, dict):
            assessment = model_output.get("thinking_assessment")
            if assessment and hasattr(assessment, "mode"):
                return assessment.mode.value if hasattr(assessment.mode, "value") else str(assessment.mode)
            return model_output.get("thinking_mode")
        return None

    def _extract_routing_weights(self, model_output: Any) -> Optional[List[float]]:
        """Extract routing weights from model output."""
        if model_output is None:
            return None
        if hasattr(model_output, "adjusted_weights"):
            weights = model_output.adjusted_weights
            if hasattr(weights, "mean"):
                return weights.mean(dim=(0, 1)).tolist()
            return list(weights)
        if isinstance(model_output, dict):
            weights = model_output.get("adjusted_weights") or model_output.get("routing_weights")
            if weights is not None:
                if hasattr(weights, "mean"):
                    return weights.mean(dim=(0, 1)).tolist()
                return list(weights)
        return None

    def _extract_confidence(self, model_output: Any) -> float:
        """Extract or estimate model confidence."""
        if model_output is None:
            return 0.5
        if hasattr(model_output, "thinking_assessment"):
            assessment = model_output.thinking_assessment
            if hasattr(assessment, "confidence"):
                return float(assessment.confidence)
        if isinstance(model_output, dict):
            return float(model_output.get("confidence", 0.5))
        weights = self._extract_routing_weights(model_output)
        if weights is not None:
            import math
            total = sum(weights)
            if total > 0:
                normalized = [w / total for w in weights]
                entropy = -sum(w * math.log(w + 1e-8) for w in normalized if w > 0)
                max_entropy = math.log(len(normalized))
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
        """Classify the domain of a query based on keywords."""
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
