"""
Adaptive Confidence Calibration — Dynamic threshold adjustment for agent actions.

Inspired by:
- ATTC (2026): "Adaptive Tool Trust Calibration For LLMs" — guides models
  to adaptively choose between using tools vs. answering directly based on
  confidence scoring.
- "Alignment for Efficient Tool Calling of LLMs" (EMNLP 2025) — gradual
  decline in tool usage as model accuracy increases, indicating adaptive
  tool invocation based on knowledge confidence.

Current Losion uses static confidence thresholds (hardcoded at 0.3, 0.4, etc.).
This module replaces them with adaptive thresholds that:
1. Learn from experience (episodic memory)
2. Adjust per-domain (math tasks need different thresholds than code tasks)
3. Consider tool trustworthiness (some tools are more reliable than others)
4. Calibrate based on outcome history (past successes/failures)

Architecture:
    SignalExtractor reads → CalibrationEngine.get_thresholds()
                                   ↓
                          EpisodicMemory (past outcomes)
                          ToolTrustScores (tool reliability)
                          DomainProfiles (domain-specific tuning)

Key innovation: The thresholds START at reasonable defaults but ADAPT
based on actual experience. If web search consistently fails for a domain,
the threshold goes up (less likely to trigger web search). If it consistently
helps, the threshold goes down (more likely to trigger web search).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DomainProfile:
    """Domain-specific calibration profile.

    Different domains have different optimal thresholds. For example:
    - Math: High threshold for web search (model is usually right)
    - Code: Low threshold for tool search (tools are very helpful)
    - History: Low threshold for web search (facts change, need updates)

    Attributes:
        domain: Domain name.
        web_search_threshold: Adaptive threshold for web search.
        skill_lookup_threshold: Adaptive threshold for skill lookup.
        tool_search_threshold: Adaptive threshold for tool search.
        verify_threshold: Adaptive threshold for verification.
        terminal_threshold: Adaptive threshold for terminal execution.
        sample_count: Number of observations used for calibration.
        last_updated: Timestamp of last update.
    """

    domain: str
    web_search_threshold: float = 0.3
    skill_lookup_threshold: float = 0.4
    tool_search_threshold: float = 0.35
    verify_threshold: float = 0.5
    terminal_threshold: float = 0.25
    sample_count: int = 0
    last_updated: float = 0.0

    def __post_init__(self) -> None:
        if self.last_updated == 0.0:
            self.last_updated = time.time()


@dataclass
class ToolTrustScore:
    """Trust score for a specific tool/action.

    Inspired by ATTC: instead of trusting all tools equally,
    we track how reliable each tool has been in practice.

    Attributes:
        tool_name: Name of the tool/action.
        trust_score: Current trust level [0.0, 1.0].
        total_uses: Number of times this tool has been used.
        success_count: Number of successful uses.
        failure_count: Number of failed uses.
        last_used: Timestamp of last use.
        domain_scores: Per-domain trust scores.
    """

    tool_name: str = ""
    trust_score: float = 0.5  # Start neutral
    total_uses: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0
    domain_scores: Dict[str, float] = field(default_factory=dict)

    def record_outcome(self, success: bool, domain: Optional[str] = None) -> None:
        """Record a tool usage outcome.

        Uses exponential moving average for trust updates:
        trust = alpha * new_evidence + (1 - alpha) * old_trust

        Args:
            success: Whether the tool use was successful.
            domain: Domain context for this usage.
        """
        self.total_uses += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # EMA update (alpha = 0.3 for moderate adaptation speed)
        alpha = 0.3
        new_evidence = 1.0 if success else 0.0
        self.trust_score = alpha * new_evidence + (1 - alpha) * self.trust_score
        self.last_used = time.time()

        # Per-domain update
        if domain:
            current = self.domain_scores.get(domain, 0.5)
            self.domain_scores[domain] = alpha * new_evidence + (1 - alpha) * current

    @property
    def reliability(self) -> float:
        """Reliability based on usage history."""
        if self.total_uses == 0:
            return 0.5
        return self.success_count / self.total_uses


class CalibrationEngine:
    """Adaptive confidence calibration for agent actions.

    This engine dynamically adjusts the confidence thresholds that
    determine when the agent should take actions (web search, tool use,
    skill lookup, etc.).

    The calibration is based on three signals:
    1. **Domain profiles**: Per-domain threshold adjustments
    2. **Tool trust**: Per-tool reliability scores
    3. **Episodic experience**: Past outcomes for similar queries

    Usage:
        engine = CalibrationEngine()

        # Get adaptive thresholds for a domain
        thresholds = engine.get_thresholds(domain="math")

        # Record an outcome for future calibration
        engine.record_outcome(
            action="web_search",
            domain="math",
            success=False,
            confidence_before=0.2,
            confidence_after=0.15,
        )

        # Next time, thresholds for "math" will be adjusted

    Args:
        learning_rate: How fast thresholds adapt (0.0-1.0).
        min_threshold: Minimum possible threshold value.
        max_threshold: Maximum possible threshold value.
        min_samples: Minimum samples before adapting thresholds.
    """

    # Default thresholds (same as original ConfidenceThreshold)
    DEFAULT_THRESHOLDS = {
        "web_search": 0.3,
        "skill_lookup": 0.4,
        "tool_search": 0.35,
        "verify": 0.5,
        "terminal": 0.25,
    }

    # Domain-specific initial profiles based on research
    DOMAIN_PROFILES = {
        "math": DomainProfile(
            domain="math",
            web_search_threshold=0.45,    # Math: model is usually right, search less
            skill_lookup_threshold=0.35,  # Math skills are very useful
            tool_search_threshold=0.3,    # Calculator/tools helpful
            verify_threshold=0.4,         # Verification important for math
            terminal_threshold=0.2,       # Terminal useful for computation
        ),
        "code": DomainProfile(
            domain="code",
            web_search_threshold=0.25,    # Code: APIs change, search more
            skill_lookup_threshold=0.35,  # Code skills very useful
            tool_search_threshold=0.2,    # Tools extremely useful for code
            verify_threshold=0.35,        # Testing important
            terminal_threshold=0.15,      # Terminal critical for code
        ),
        "science": DomainProfile(
            domain="science",
            web_search_threshold=0.25,    # Science: need latest research
            skill_lookup_threshold=0.4,
            tool_search_threshold=0.35,
            verify_threshold=0.4,
            terminal_threshold=0.3,
        ),
        "history": DomainProfile(
            domain="history",
            web_search_threshold=0.2,     # History: facts need verification
            skill_lookup_threshold=0.45,
            tool_search_threshold=0.4,
            verify_threshold=0.45,
            terminal_threshold=0.35,      # Terminal less useful
        ),
        "language": DomainProfile(
            domain="language",
            web_search_threshold=0.35,
            skill_lookup_threshold=0.4,
            tool_search_threshold=0.4,
            verify_threshold=0.5,
            terminal_threshold=0.35,
        ),
        "web": DomainProfile(
            domain="web",
            web_search_threshold=0.2,     # Web: search is native
            skill_lookup_threshold=0.35,
            tool_search_threshold=0.2,    # API tools very useful
            verify_threshold=0.4,
            terminal_threshold=0.2,
        ),
        "data": DomainProfile(
            domain="data",
            web_search_threshold=0.35,
            skill_lookup_threshold=0.35,
            tool_search_threshold=0.2,    # Data tools critical
            verify_threshold=0.35,
            terminal_threshold=0.15,      # Terminal useful for data
        ),
    }

    def __init__(
        self,
        learning_rate: float = 0.1,
        min_threshold: float = 0.1,
        max_threshold: float = 0.8,
        min_samples: int = 3,
    ) -> None:
        self.learning_rate = learning_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_samples = min_samples

        # Domain profiles (start from defaults, adapt over time)
        self._profiles: Dict[str, DomainProfile] = {
            name: DomainProfile(**{k: v for k, v in prof.__dict__.items()})
            for name, prof in self.DOMAIN_PROFILES.items()
        }

        # Tool trust scores
        self._tool_trust: Dict[str, ToolTrustScore] = {}

        # Outcome history for calibration: domain → action → [(success, confidence_delta)]
        self._outcome_history: Dict[str, Dict[str, List[Tuple[bool, float]]]] = {}

        # Thread safety: protect shared mutable dicts
        self._lock = threading.RLock()

    def get_thresholds(self, domain: Optional[str] = None) -> Dict[str, float]:
        """Get adaptive thresholds for the given domain.

        If the domain has a profile, use it. Otherwise, use defaults.
        Tool trust scores are used to further adjust thresholds.

        The logic:
        - If a tool has high trust → lower threshold (use it more)
        - If a tool has low trust → higher threshold (use it less)
        - Domain profile provides the base thresholds

        Args:
            domain: Domain classification.

        Returns:
            Dictionary of threshold_name → threshold_value.
        """
        with self._lock:
            # Get base thresholds from domain profile
            if domain and domain in self._profiles:
                profile = self._profiles[domain]
                thresholds = {
                    "web_search": profile.web_search_threshold,
                    "skill_lookup": profile.skill_lookup_threshold,
                    "tool_search": profile.tool_search_threshold,
                    "verify": profile.verify_threshold,
                    "terminal": profile.terminal_threshold,
                }
            else:
                thresholds = dict(self.DEFAULT_THRESHOLDS)

            # Adjust based on tool trust scores
            tool_action_map = {
                "web_search": "web_search",
                "skill_lookup": "skill_lookup",
                "tool_search": "tool_search",
                "terminal": "terminal_execute",
                "verify": "verify_output",
            }

            for action, tool_name in tool_action_map.items():
                trust = self._tool_trust.get(tool_name)
                if trust and trust.total_uses >= self.min_samples:
                    # High trust → lower threshold (use more eagerly)
                    # Low trust → higher threshold (use more cautiously)
                    trust_delta = (trust.trust_score - 0.5) * 0.2  # Max ±0.1 adjustment
                    domain_trust = trust.domain_scores.get(domain or "", trust.trust_score)
                    domain_delta = (domain_trust - 0.5) * 0.1

                    adjusted = thresholds[action] - trust_delta - domain_delta
                    thresholds[action] = max(self.min_threshold, min(self.max_threshold, adjusted))

            return thresholds

    def record_outcome(
        self,
        action: str,
        domain: Optional[str],
        success: bool,
        confidence_before: float,
        confidence_after: float,
    ) -> None:
        """Record an action outcome for future calibration.

        This is the feedback loop: after an action is executed and
        evaluated, the result is fed back to adjust future thresholds.

        Calibration logic:
        - If action was successful AND confidence improved:
          → This action is helpful → lower threshold (use more)
        - If action failed AND confidence dropped:
          → This action is harmful → raise threshold (use less)
        - If action was successful but confidence didn't improve:
          → Action was neutral → minimal adjustment

        Args:
            action: The action taken (e.g., "web_search").
            domain: Domain classification.
            success: Whether the action was successful.
            confidence_before: Confidence before the action.
            confidence_after: Confidence after the action.
        """
        with self._lock:
            # Update tool trust
            if action not in self._tool_trust:
                self._tool_trust[action] = ToolTrustScore(tool_name=action)
            self._tool_trust[action].record_outcome(success, domain)

            # Record in outcome history
            if domain not in self._outcome_history:
                self._outcome_history[domain] = {}
            if action not in self._outcome_history[domain]:
                self._outcome_history[domain][action] = []

            confidence_delta = confidence_after - confidence_before
            self._outcome_history[domain][action].append((success, confidence_delta))

            # Keep history bounded
            if len(self._outcome_history[domain][action]) > 100:
                self._outcome_history[domain][action] = self._outcome_history[domain][action][-100:]

            # Adapt domain profile if enough samples
            if domain and domain in self._profiles:
                self._adapt_profile(domain, action, success, confidence_delta)

    def _adapt_profile(
        self,
        domain: str,
        action: str,
        success: bool,
        confidence_delta: float,
    ) -> None:
        """Adapt a domain profile based on a new outcome.

        Uses a conservative learning rate to avoid overfitting to
        recent outcomes.

        Args:
            domain: Domain name.
            action: Action taken.
            success: Whether the action was successful.
            confidence_delta: Change in confidence.
        """
        profile = self._profiles[domain]
        profile.sample_count += 1
        profile.last_updated = time.time()

        # Only adapt after minimum samples
        if profile.sample_count < self.min_samples:
            return

        # Determine threshold field name
        threshold_map = {
            "web_search": "web_search_threshold",
            "skill_lookup": "skill_lookup_threshold",
            "tool_search": "tool_search_threshold",
            "verify": "verify_threshold",
            "terminal_execute": "terminal_threshold",
        }

        field_name = threshold_map.get(action)
        if not field_name:
            return

        current = getattr(profile, field_name)

        # If action was successful and improved confidence → lower threshold
        # If action failed and reduced confidence → raise threshold
        if success and confidence_delta > 0:
            # Action helped → make it easier to trigger
            adjustment = -self.learning_rate * abs(confidence_delta)
        elif not success and confidence_delta < 0:
            # Action hurt → make it harder to trigger
            adjustment = self.learning_rate * abs(confidence_delta)
        else:
            # Mixed signal → small adjustment
            adjustment = self.learning_rate * 0.1 * confidence_delta

        new_value = current + adjustment
        new_value = max(self.min_threshold, min(self.max_threshold, new_value))
        setattr(profile, field_name, new_value)

        logger.debug(
            f"Calibration: {domain}/{action} threshold "
            f"{current:.3f} → {new_value:.3f} "
            f"(success={success}, delta={confidence_delta:.3f})"
        )

    def get_tool_trust(self, action: str, domain: Optional[str] = None) -> float:
        """Get trust score for a specific tool/action.

        Args:
            action: Action name.
            domain: Optional domain for domain-specific trust.

        Returns:
            Trust score [0.0, 1.0].
        """
        with self._lock:
            trust = self._tool_trust.get(action)
            if trust is None:
                return 0.5  # Neutral

            if domain and domain in trust.domain_scores:
                return trust.domain_scores[domain]

            return trust.trust_score

    def get_stats(self) -> Dict[str, Any]:
        """Get calibration statistics."""
        with self._lock:
            return {
                "domain_profiles": {
                    name: {
                        "web_search": prof.web_search_threshold,
                        "skill_lookup": prof.skill_lookup_threshold,
                        "tool_search": prof.tool_search_threshold,
                        "sample_count": prof.sample_count,
                    }
                    for name, prof in self._profiles.items()
                },
                "tool_trust": {
                    name: {
                        "trust": score.trust_score,
                        "reliability": score.reliability,
                        "uses": score.total_uses,
                    }
                    for name, score in self._tool_trust.items()
                },
                "learning_rate": self.learning_rate,
            }
