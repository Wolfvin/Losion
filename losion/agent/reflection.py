"""
Self-Reflection Module — Reflexion + Self-Refine for the Agent Layer.

Inspired by:
- Reflexion (Shinn et al., 2023): Agents learn from verbal feedback
  rather than parameter updates, storing reflections for future decisions.
- Self-Refine (Madaan et al., 2023): Iterative refinement with self-feedback.
- ExACT (2024): Reflective MCTS combines reflection with tree search.

This module enables the agent to:
1. Evaluate the quality of its own actions after execution
2. Generate verbal reflections on what went wrong/right
3. Store reflections in EpisodicMemory for future reference
4. Use reflections to improve subsequent action decisions

Architecture:
    Action Result → ReflectionEngine.evaluate() → Reflection
                                                    ↓
                                          EpisodicMemory.store()
                                                    ↓
    Next Action Decision ← SignalExtractor reads ← EpisodicMemory

The key insight from Reflexion: "Agents that can reflect on their failures
and successes can improve rapidly without parameter updates, using only
verbal feedback stored in a memory buffer."
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReflectionType(Enum):
    """Types of reflections an agent can produce."""

    ACTION_SUCCESS = "action_success"
    ACTION_FAILURE = "action_failure"
    STRATEGY_CORRECTION = "strategy_correction"
    TOOL_TRUST_UPDATE = "tool_trust_update"
    SKILL_REFINEMENT = "skill_refinement"
    CONFIDENCE_RECALIBRATION = "confidence_recalibration"


@dataclass
class Reflection:
    """A single reflection on an agent action.

    Inspired by Reflexion: verbal feedback stored as structured text,
    enabling the agent to learn from experience without parameter updates.

    Attributes:
        reflection_type: Category of this reflection.
        action_taken: What action was taken.
        outcome: What happened as a result.
        assessment: Self-assessment of the outcome quality.
        lesson: What was learned (the key verbal feedback).
        confidence_before: Agent confidence before the action.
        confidence_after: Agent confidence after seeing the result.
        domain: Domain context for this reflection.
        timestamp: When this reflection was created.
        metadata: Additional context.
    """

    reflection_type: ReflectionType
    action_taken: str = ""
    outcome: str = ""
    assessment: str = ""
    lesson: str = ""
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    domain: Optional[str] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def confidence_delta(self) -> float:
        """How much confidence changed after this action."""
        return self.confidence_after - self.confidence_before

    @property
    def is_positive(self) -> bool:
        """Whether this reflection represents a successful outcome."""
        return self.reflection_type in (
            ReflectionType.ACTION_SUCCESS,
            ReflectionType.STRATEGY_CORRECTION,
        )

    @property
    def hash_key(self) -> str:
        """Hash key for O(1) lookup in episodic memory."""
        content = f"{self.action_taken}:{self.domain}:{self.lesson[:50]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = asdict(self)
        d["reflection_type"] = self.reflection_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reflection":
        """Deserialize from dictionary."""
        data["reflection_type"] = ReflectionType(data["reflection_type"])
        return cls(**data)


class ReflectionEngine:
    """Generates reflections on agent actions.

    This is the core self-evaluation mechanism. After each action,
    the engine assesses the outcome and generates structured verbal
    feedback that can be stored in episodic memory.

    The assessment criteria are inspired by:
    - Reflexion: "Was the action helpful? Did it achieve the goal?"
    - Self-Refine: "Can the output be improved? How?"
    - ATTC: "Should I trust this tool result?"

    Assessment heuristics:
    1. Result quality: Did the action produce useful content?
    2. Confidence change: Did confidence improve after the action?
    3. Goal alignment: Is the result relevant to the original query?
    4. Efficiency: Was this the most efficient path?

    Args:
        min_lesson_length: Minimum characters for a meaningful lesson.
        max_reflections_per_action: Max reflections per action evaluation.
    """

    # Heuristic patterns for evaluating action outcomes
    SUCCESS_INDICATORS = {
        "web_search": ["found", "retrieved", "relevant", "results"],
        "skill_lookup": ["matched", "applied", "found skill"],
        "tool_search": ["available", "registered", "executable"],
        "terminal_execute": ["success", "completed", "exit_code=0"],
        "verify_output": ["verified", "correct", "consistent"],
    }

    FAILURE_INDICATORS = {
        "web_search": ["no results", "failed", "timeout", "irrelevant"],
        "skill_lookup": ["not found", "no match", "low confidence"],
        "tool_search": ["no tool", "unavailable", "no handler"],
        "terminal_execute": ["error", "failed", "timed out", "exit_code!=0"],
        "verify_output": ["incorrect", "inconsistent", "failed"],
    }

    def __init__(
        self,
        min_lesson_length: int = 10,
        max_reflections_per_action: int = 3,
    ) -> None:
        self.min_lesson_length = min_lesson_length
        self.max_reflections_per_action = max_reflections_per_action

    def evaluate(
        self,
        action: str,
        action_result: Any,
        confidence_before: float,
        confidence_after: float,
        query: str = "",
        domain: Optional[str] = None,
    ) -> List[Reflection]:
        """Evaluate an action and generate reflections.

        This is the main entry point. After an action is executed,
        call this method to generate self-evaluations.

        Args:
            action: The action that was taken (e.g., "web_search").
            action_result: The result of the action.
            confidence_before: Agent confidence before the action.
            confidence_after: Agent confidence after seeing the result.
            query: The original user query.
            domain: Domain classification.

        Returns:
            List of Reflection objects.
        """
        reflections: List[Reflection] = []

        # === Primary reflection: Success/Failure assessment ===
        success = self._assess_success(action, action_result, confidence_delta=confidence_after - confidence_before)
        reflection_type = ReflectionType.ACTION_SUCCESS if success else ReflectionType.ACTION_FAILURE

        assessment = self._generate_assessment(action, action_result, success)
        lesson = self._generate_lesson(action, action_result, success, domain)

        reflections.append(Reflection(
            reflection_type=reflection_type,
            action_taken=action,
            outcome=self._summarize_outcome(action_result),
            assessment=assessment,
            lesson=lesson,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            domain=domain,
        ))

        # === Secondary reflection: Strategy correction if needed ===
        if not success and confidence_after < confidence_before:
            strategy_lesson = self._generate_strategy_correction(action, query, domain)
            if len(strategy_lesson) >= self.min_lesson_length:
                reflections.append(Reflection(
                    reflection_type=ReflectionType.STRATEGY_CORRECTION,
                    action_taken=action,
                    outcome="Action decreased confidence",
                    assessment="Current strategy is not effective for this query type.",
                    lesson=strategy_lesson,
                    confidence_before=confidence_before,
                    confidence_after=confidence_after,
                    domain=domain,
                ))

        # === Tertiary reflection: Tool trust update ===
        if action in ("web_search", "tool_search", "terminal_execute"):
            trust_update = self._evaluate_tool_trust(action, action_result, success)
            if trust_update:
                reflections.append(Reflection(
                    reflection_type=ReflectionType.TOOL_TRUST_UPDATE,
                    action_taken=action,
                    outcome=self._summarize_outcome(action_result),
                    assessment=trust_update,
                    lesson=f"Tool trust for {action}: {'increased' if success else 'decreased'}",
                    confidence_before=confidence_before,
                    confidence_after=confidence_after,
                    domain=domain,
                    metadata={"tool": action, "trust_delta": 0.1 if success else -0.1},
                ))

        return reflections[:self.max_reflections_per_action]

    def _assess_success(
        self,
        action: str,
        action_result: Any,
        confidence_delta: float = 0.0,
    ) -> bool:
        """Assess whether an action was successful.

        Uses multiple heuristics:
        1. Result content analysis (success/failure indicators)
        2. Confidence change (positive delta = success)
        3. Result presence (None = failure)

        Args:
            action: The action taken.
            action_result: The result of the action.
            confidence_delta: Change in confidence.

        Returns:
            True if the action appears successful.
        """
        # No result = failure
        if action_result is None:
            return False

        # Confidence improved = likely success
        if confidence_delta > 0.05:
            return True

        # Confidence dropped significantly = likely failure
        if confidence_delta < -0.1:
            return False

        # Check result content for success/failure indicators
        result_str = str(action_result).lower()
        success_words = self.SUCCESS_INDICATORS.get(action, [])
        failure_words = self.FAILURE_INDICATORS.get(action, [])

        success_count = sum(1 for w in success_words if w in result_str)
        failure_count = sum(1 for w in failure_words if w in result_str)

        if failure_count > success_count:
            return False
        if success_count > 0:
            return True

        # For dict results, check success field
        if isinstance(action_result, dict):
            if "success" in action_result:
                return bool(action_result["success"])

        # Default: result exists = partial success
        return bool(result_str.strip())

    def _generate_assessment(
        self, action: str, action_result: Any, success: bool
    ) -> str:
        """Generate a verbal assessment of the action outcome."""
        if success:
            return (
                f"The {action} action produced useful results. "
                f"The agent's confidence in the current approach is maintained or improved."
            )
        else:
            return (
                f"The {action} action did not produce the expected results. "
                f"Consider alternative approaches or adjusting the query strategy."
            )

    def _generate_lesson(
        self,
        action: str,
        action_result: Any,
        success: bool,
        domain: Optional[str],
    ) -> str:
        """Generate a lesson learned from this action.

        This is the core of Reflexion: verbal feedback that captures
        what the agent learned, stored for future reference.
        """
        if success:
            lessons = {
                "web_search": f"Web search was effective for this query. The search returned relevant results that can be used as context.",
                "skill_lookup": f"A matching skill was found and applied successfully. This domain has established skills.",
                "tool_search": f"A suitable tool was found for this task. The tool registry has relevant capabilities.",
                "terminal_execute": f"Terminal execution produced valid output. The command was safe and effective.",
                "verify_output": f"Output verification confirmed correctness. The model's reasoning was sound.",
            }
            return lessons.get(action, f"The {action} action was effective for this type of query.")
        else:
            lessons = {
                "web_search": f"Web search did not return useful results. Consider: refining the query, using different keywords, or trying a different search strategy.",
                "skill_lookup": f"No suitable skill was found. Consider: creating a new skill, broadening the search, or using a different domain classification.",
                "tool_search": f"No suitable tool was found. Consider: creating a new tool, composing existing tools, or using terminal execution as fallback.",
                "terminal_execute": f"Terminal execution failed. Consider: checking the command syntax, ensuring dependencies are available, or using a different approach.",
                "verify_output": f"Output verification failed. Consider: revising the reasoning, checking for logical errors, or seeking additional information.",
            }
            return lessons.get(action, f"The {action} action was not effective. Consider alternative strategies.")

    def _generate_strategy_correction(
        self,
        action: str,
        query: str,
        domain: Optional[str],
    ) -> str:
        """Generate a strategy correction when actions reduce confidence.

        This is the Self-Refine component: when an action makes things
        worse, suggest a better approach.
        """
        corrections = {
            "web_search": "Web search reduced confidence. Try: (1) more specific query terms, (2) domain-specific search, (3) check if the information is available in existing skills before searching.",
            "skill_lookup": "Skill lookup failed and reduced confidence. Try: (1) check if a skill needs to be created, (2) try a broader domain search, (3) use the model's own knowledge first.",
            "tool_search": "Tool search reduced confidence. Try: (1) create a new tool, (2) use terminal execution as a generic tool, (3) decompose the task into simpler sub-tasks.",
        }
        correction = corrections.get(action, f"The {action} action reduced confidence. Try a different approach or rely on model-only inference.")

        if domain:
            correction += f" For the {domain} domain, consider domain-specific strategies."

        return correction

    def _evaluate_tool_trust(
        self, action: str, action_result: Any, success: bool
    ) -> str:
        """Evaluate tool trustworthiness based on outcome.

        Inspired by ATTC (Adaptive Tool Trust Calibration):
        "Guides the model to adaptively choose between using tools
        vs. answering directly based on confidence scoring."
        """
        if success:
            return f"Tool {action} produced reliable results. Trust level should be increased for similar queries."
        else:
            return f"Tool {action} produced unreliable results. Trust level should be decreased. Consider using model knowledge or alternative tools."

    def _summarize_outcome(self, action_result: Any) -> str:
        """Create a brief summary of the action result."""
        if action_result is None:
            return "No result produced"

        if isinstance(action_result, dict):
            if "success" in action_result:
                return f"Result: success={action_result['success']}"
            if "error" in action_result:
                return f"Error: {action_result['error']}"
            return f"Dict result with {len(action_result)} keys"

        if isinstance(action_result, list):
            return f"List result with {len(action_result)} items"

        result_str = str(action_result)
        if len(result_str) > 200:
            return result_str[:200] + "..."
        return result_str
