"""
Paradigm Router — SMART-style knowledge sufficiency and reasoning paradigm selection.

Inspired by:
- SMART (Self-Aware Agent for Tool Overuse Mitigation, 2025): Prevents
  unnecessary tool calls when the model's parametric knowledge is sufficient.
- Paradigm Routing as Inference-Time Optimization (2024): Different reasoning
  paradigms (Direct, CoT, ReAct, RAG, MCTS) should be selected per-task,
  not fixed architecturally.
- Losion's Tri-Jalur Router: Already routes between SSM, Attention, and
  Retrieval at the model level. This module extends the same principle
  to the agent level.

The Paradigm Router answers: "Which reasoning paradigm should the agent use
for this specific query?" Instead of always running the full ReAct loop,
it selects the lightest-weight paradigm that's likely to succeed:

1. DIRECT: Model answers directly, no tools (confidence > 0.8)
2. COT: Chain-of-thought reasoning, no tools (confidence 0.5-0.8)
3. REACT: Interleaved thought-action-observation (confidence 0.3-0.5)
4. RAG: Single retrieval + generation (confidence 0.15-0.3)
5. MCTS: Full tree search with backtracking (confidence < 0.15)

Key innovation — Knowledge Sufficiency Check (from SMART):
    The model's Tri-Jalur routing weights tell us whether the model already
    has sufficient parametric knowledge. If retrieval weight (Jalur 3) is LOW
    and attention weight (Jalur 2) is HIGH, the model can reason through
    the answer without tools — even if confidence is moderate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReasoningParadigm(Enum):
    """Available reasoning paradigms for the agent.

    Ordered from lightest to heaviest computational cost:
    - DIRECT: No tools, parametric knowledge only
    - COT: Chain-of-thought reasoning (ThinkingToggle ON), no tools
    - REACT: ReAct-style interleaved thought-action-observation
    - RAG: Single retrieval + generation (one web search)
    - MCTS: Full tree search with backtracking and simulation
    """

    DIRECT = "direct"
    COT = "cot"
    REACT = "react"
    RAG = "rag"
    MCTS = "mcts"

    @property
    def uses_tools(self) -> bool:
        """Whether this paradigm uses external tools."""
        return self in (ReasoningParadigm.REACT, ReasoningParadigm.RAG, ReasoningParadigm.MCTS)

    @property
    def uses_tree_search(self) -> bool:
        """Whether this paradigm uses tree-structured search."""
        return self == ReasoningParadigm.MCTS

    @property
    def computational_cost(self) -> int:
        """Relative computational cost (1=lightest, 5=heaviest)."""
        costs = {
            ReasoningParadigm.DIRECT: 1,
            ReasoningParadigm.COT: 2,
            ReasoningParadigm.REACT: 3,
            ReasoningParadigm.RAG: 3,
            ReasoningParadigm.MCTS: 5,
        }
        return costs[self]


@dataclass
class ParadigmSelection:
    """Result of paradigm routing.

    Attributes:
        paradigm: The selected reasoning paradigm.
        confidence: Model confidence that led to this selection.
        knowledge_sufficient: Whether parametric knowledge is sufficient.
        routing_weights: Tri-Jalur routing weights (if available).
        reasoning: Why this paradigm was selected.
        alternatives: Alternative paradigms that could work (ranked).
        metadata: Additional context for the selection.
    """

    paradigm: ReasoningParadigm = ReasoningParadigm.DIRECT
    confidence: float = 1.0
    knowledge_sufficient: bool = True
    routing_weights: Optional[List[float]] = None
    reasoning: str = ""
    alternatives: List[ReasoningParadigm] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_use_tools(self) -> bool:
        """Whether the selected paradigm uses external tools."""
        return self.paradigm.uses_tools

    @property
    def should_use_mcts(self) -> bool:
        """Whether the selected paradigm uses MCTS tree search."""
        return self.paradigm.uses_tree_search


class ParadigmRouter:
    """Routes queries to the best reasoning paradigm.

    This is the agent-level equivalent of Losion's Tri-Jalur Router.
    While the Tri-Jalur Router selects between SSM, Attention, and Retrieval
    at the model level, the Paradigm Router selects between Direct, CoT,
    ReAct, RAG, and MCTS at the agent level.

    Key Innovation — SMART Knowledge Sufficiency Check:
    Instead of always using the most expensive paradigm, the router first
    checks whether the model's parametric knowledge is sufficient. This
    prevents the common failure mode of "tool overuse" where agents call
    web search for questions they already know.

    The knowledge sufficiency signal comes from:
    1. Tri-Jalur routing weights: If retrieval weight is LOW, the model
       doesn't think it needs external knowledge
    2. Model confidence: If confidence is high, parametric knowledge is
       likely sufficient
    3. Domain expertise: If calibration shows high success for this domain
       without tools, knowledge is likely sufficient

    Usage:
        router = ParadigmRouter()
        selection = router.route(
            confidence=0.6,
            routing_weights=[0.3, 0.5, 0.2],  # [SSM, Attn, Retrieval]
            query="Calculate 2+2",
            domain="math",
        )
        # selection.paradigm == ReasoningParadigm.DIRECT
        # selection.knowledge_sufficient == True

    Args:
        confidence_thresholds: Thresholds for paradigm selection.
        use_knowledge_sufficiency: Whether to use SMART-style knowledge
            sufficiency check.
        retrieval_weight_threshold: If retrieval weight is below this,
            model doesn't need external knowledge.
        attention_weight_threshold: If attention weight is above this,
            reasoning pathway is active.
    """

    def __init__(
        self,
        confidence_thresholds: Optional[Dict[str, float]] = None,
        use_knowledge_sufficiency: bool = True,
        retrieval_weight_threshold: float = 0.2,
        attention_weight_threshold: float = 0.4,
    ) -> None:
        self.thresholds = confidence_thresholds or {
            "direct": 0.8,    # confidence >= 0.8 → direct
            "cot": 0.5,       # 0.5 <= confidence < 0.8 → chain-of-thought
            "react": 0.3,     # 0.3 <= confidence < 0.5 → ReAct loop
            "rag": 0.15,      # 0.15 <= confidence < 0.3 → single retrieval
            # confidence < 0.15 → MCTS tree search
        }
        self.use_knowledge_sufficiency = use_knowledge_sufficiency
        self.retrieval_weight_threshold = retrieval_weight_threshold
        self.attention_weight_threshold = attention_weight_threshold

    def route(
        self,
        confidence: float,
        routing_weights: Optional[List[float]] = None,
        query: Optional[str] = None,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
        thinking_mode: Optional[str] = None,
        calibration_engine: Optional[Any] = None,
    ) -> ParadigmSelection:
        """Route a query to the best reasoning paradigm.

        Uses multi-signal fusion:
        1. Confidence signal: Primary determinant
        2. Knowledge sufficiency signal (SMART): From routing weights
        3. Domain signal: Some domains benefit from specific paradigms
        4. Task type signal: Factual vs reasoning vs sequential
        5. Calibration signal: Past success rates for this domain

        Args:
            confidence: Model confidence [0.0, 1.0].
            routing_weights: Tri-Jalur weights [w_ssm, w_attn, w_retr].
            query: The user query text.
            domain: Domain classification.
            task_type: Task type (sequential, reasoning, factual).
            thinking_mode: Whether model is in thinking mode.
            calibration_engine: Optional CalibrationEngine for adaptive data.

        Returns:
            ParadigmSelection with the recommended paradigm.
        """
        # === Step 1: Confidence-based baseline paradigm ===
        baseline_paradigm = self._confidence_to_paradigm(confidence)

        # === Step 2: SMART Knowledge Sufficiency Check ===
        knowledge_sufficient = False
        if self.use_knowledge_sufficiency and routing_weights is not None:
            knowledge_sufficient = self._check_knowledge_sufficiency(
                confidence, routing_weights
            )

            if knowledge_sufficient and baseline_paradigm.uses_tools:
                # Model has sufficient knowledge — downgrade to lighter paradigm
                if confidence >= 0.5:
                    baseline_paradigm = ReasoningParadigm.COT
                else:
                    baseline_paradigm = ReasoningParadigm.DIRECT

        # === Step 3: Domain adjustments ===
        adjusted_paradigm = self._adjust_for_domain(
            baseline_paradigm, domain, task_type
        )

        # === Step 4: Task type adjustments ===
        adjusted_paradigm = self._adjust_for_task_type(
            adjusted_paradigm, task_type, confidence
        )

        # === Step 5: Thinking mode override ===
        if thinking_mode == "thinking" and adjusted_paradigm == ReasoningParadigm.DIRECT:
            # If model is already thinking, upgrade to at least CoT
            adjusted_paradigm = ReasoningParadigm.COT

        # === Step 6: Calibration adjustments ===
        if calibration_engine is not None and domain is not None:
            adjusted_paradigm = self._adjust_for_calibration(
                adjusted_paradigm, domain, calibration_engine
            )

        # === Build alternatives ===
        alternatives = self._generate_alternatives(adjusted_paradigm, confidence)

        # === Build reasoning ===
        reasoning = self._build_reasoning(
            confidence, routing_weights, knowledge_sufficient,
            domain, task_type, adjusted_paradigm,
        )

        return ParadigmSelection(
            paradigm=adjusted_paradigm,
            confidence=confidence,
            knowledge_sufficient=knowledge_sufficient,
            routing_weights=routing_weights,
            reasoning=reasoning,
            alternatives=alternatives,
            metadata={
                "baseline_paradigm": baseline_paradigm.value,
                "domain": domain,
                "task_type": task_type,
            },
        )

    def _confidence_to_paradigm(self, confidence: float) -> ReasoningParadigm:
        """Map confidence score to baseline paradigm."""
        if confidence >= self.thresholds["direct"]:
            return ReasoningParadigm.DIRECT
        elif confidence >= self.thresholds["cot"]:
            return ReasoningParadigm.COT
        elif confidence >= self.thresholds["react"]:
            return ReasoningParadigm.REACT
        elif confidence >= self.thresholds["rag"]:
            return ReasoningParadigm.RAG
        else:
            return ReasoningParadigm.MCTS

    def _check_knowledge_sufficiency(
        self, confidence: float, routing_weights: List[float]
    ) -> bool:
        """SMART-style knowledge sufficiency check.

        The model's Tri-Jalur routing weights tell us whether it already
        has sufficient parametric knowledge. If:
        - Retrieval weight (Jalur 3) is LOW → model doesn't think it needs
          external knowledge
        - Attention weight (Jalur 2) is HIGH → reasoning pathway is active,
          model can reason through the answer
        - Confidence is moderate → model is not unsure

        Then parametric knowledge is sufficient and tools are unnecessary.
        """
        if len(routing_weights) < 3:
            return False

        retrieval_weight = routing_weights[2]
        attention_weight = routing_weights[1]

        # Model doesn't need retrieval AND has active reasoning AND is somewhat confident
        parametric_sufficient = (
            retrieval_weight < self.retrieval_weight_threshold
            and attention_weight > self.attention_weight_threshold
            and confidence > 0.5
        )

        return parametric_sufficient

    def _adjust_for_domain(
        self,
        paradigm: ReasoningParadigm,
        domain: Optional[str],
        task_type: Optional[str],
    ) -> ReasoningParadigm:
        """Adjust paradigm based on domain characteristics.

        Some domains benefit from specific paradigms:
        - Math: Often solvable with CoT (no tools needed for reasoning)
        - Code: Benefits from RAG (APIs change, need current info)
        - History: Benefits from RAG (facts need verification)
        - Web: Benefits from REACT (multi-step web interactions)
        - Data: Benefits from REACT (multi-step data processing)
        """
        if domain is None:
            return paradigm

        domain_adjustments = {
            "math": {
                # Math is often solvable by reasoning alone
                ReasoningParadigm.RAG: ReasoningParadigm.COT,
                ReasoningParadigm.REACT: ReasoningParadigm.COT,
            },
            "code": {
                # Code benefits from actual tool execution
                ReasoningParadigm.COT: ReasoningParadigm.REACT,
            },
            "history": {
                # Historical facts need verification
                ReasoningParadigm.COT: ReasoningParadigm.RAG,
            },
            "web": {
                # Web tasks need active search
                ReasoningParadigm.COT: ReasoningParadigm.REACT,
                ReasoningParadigm.RAG: ReasoningParadigm.REACT,
            },
            "data": {
                # Data tasks need tool execution
                ReasoningParadigm.COT: ReasoningParadigm.REACT,
            },
        }

        domain_map = domain_adjustments.get(domain, {})
        return domain_map.get(paradigm, paradigm)

    def _adjust_for_task_type(
        self,
        paradigm: ReasoningParadigm,
        task_type: Optional[str],
        confidence: float,
    ) -> ReasoningParadigm:
        """Adjust paradigm based on task type."""
        if task_type == "factual" and paradigm == ReasoningParadigm.COT:
            # Factual questions don't need deep reasoning — try RAG
            return ReasoningParadigm.RAG if confidence < 0.6 else ReasoningParadigm.DIRECT

        if task_type == "reasoning" and paradigm == ReasoningParadigm.RAG:
            # Reasoning tasks need CoT, not just retrieval
            return ReasoningParadigm.REACT

        if task_type == "sequential" and paradigm in (
            ReasoningParadigm.COT, ReasoningParadigm.RAG
        ):
            # Sequential tasks need step-by-step execution
            return ReasoningParadigm.REACT

        return paradigm

    def _adjust_for_calibration(
        self,
        paradigm: ReasoningParadigm,
        domain: str,
        calibration_engine: Any,
    ) -> ReasoningParadigm:
        """Adjust paradigm based on calibration data.

        If past experience shows that tools rarely help for this domain,
        prefer lighter paradigms. If tools consistently help, prefer
        heavier paradigms.
        """
        try:
            tool_trust = calibration_engine.get_tool_trust("web_search", domain)
            if tool_trust < 0.3 and paradigm.uses_tools:
                # Web search is unreliable for this domain → prefer CoT
                if paradigm == ReasoningParadigm.RAG:
                    return ReasoningParadigm.COT
                if paradigm == ReasoningParadigm.REACT:
                    return ReasoningParadigm.COT

            if tool_trust > 0.7 and not paradigm.uses_tools and paradigm == ReasoningParadigm.COT:
                # Web search is very reliable → upgrade to RAG
                return ReasoningParadigm.RAG

        except Exception:
            pass

        return paradigm

    def _generate_alternatives(
        self,
        selected: ReasoningParadigm,
        confidence: float,
    ) -> List[ReasoningParadigm]:
        """Generate ranked alternative paradigms.

        Returns paradigms that could also work, ranked by
        likely effectiveness.
        """
        all_paradigms = list(ReasoningParadigm)
        alternatives = [p for p in all_paradigms if p != selected]
        # Sort by distance from selected paradigm
        cost_diffs = {p: abs(p.computational_cost - selected.computational_cost) for p in alternatives}
        alternatives.sort(key=lambda p: cost_diffs[p])
        return alternatives[:3]

    def _build_reasoning(
        self,
        confidence: float,
        routing_weights: Optional[List[float]],
        knowledge_sufficient: bool,
        domain: Optional[str],
        task_type: Optional[str],
        paradigm: ReasoningParadigm,
    ) -> str:
        """Build a human-readable explanation of the paradigm selection."""
        parts = [f"Selected {paradigm.value} paradigm (confidence={confidence:.2f}"]

        if knowledge_sufficient:
            parts.append("SMART: parametric knowledge sufficient")
            if routing_weights:
                parts.append(
                    f"routing=[SSM={routing_weights[0]:.2f}, "
                    f"Attn={routing_weights[1]:.2f}, "
                    f"Retr={routing_weights[2]:.2f}]"
                )
        else:
            parts.append("external tools needed")

        if domain:
            parts.append(f"domain={domain}")
        if task_type:
            parts.append(f"task_type={task_type}")

        return "; ".join(parts) + ")"
