"""
Risk Simulator — ToolEmu-style pre-execution risk assessment.

Inspired by:
- ToolEmu (Ruan et al., ICLR 2024 Spotlight, cited 326×): Use an LM to
  EMULATE tool execution, enabling scalable risk testing of LM agents
  without actually executing dangerous tools.
- Execution Isolation Best Practices: Container-based isolation, network
  isolation, filesystem isolation, time limits, audit logging.

Current Losion SandboxedTerminal validates commands with static rules
(blocked commands, patterns). This module adds DYNAMIC risk assessment:

1. Pre-execution simulation: Before running a command, predict the outcome
2. Risk classification: Categorize the predicted risk level
3. Approval routing: Route risky actions to appropriate approval level
4. Audit trail: Full logging of risk assessments

Key design:
    Action → RiskSimulator.assess() → RiskAssessment → Execute/Block/Request Approval

Integration with Losion:
- Uses Losion model (when available) for outcome prediction
- Uses CalibrationEngine tool trust for risk probability estimation
- Uses EpisodicMemory for past failure patterns
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for agent actions.

    Ordered from lowest to highest risk:
    - SAFE: No risk, can execute immediately
    - LOW: Minor risk, can execute with logging
    - MEDIUM: Moderate risk, may need approval
    - HIGH: Significant risk, requires explicit approval
    - CRITICAL: Dangerous, should be blocked or require admin approval
    """

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def requires_approval(self) -> bool:
        """Whether this risk level requires approval before execution."""
        return self in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)

    @property
    def should_block(self) -> bool:
        """Whether this risk level should block execution entirely."""
        return self == RiskLevel.CRITICAL


@dataclass
class SimulationResult:
    """Result of a pre-execution risk simulation.

    Attributes:
        predicted_outcome: What the simulator predicts will happen.
        predicted_risk_level: Assessed risk level from simulation.
        confidence_in_prediction: How confident the simulator is [0.0, 1.0].
        potential_damages: List of potential negative outcomes.
        safety_mitigations: Suggested mitigations to reduce risk.
        similar_past_incidents: Past failures similar to this action.
    """

    predicted_outcome: str = ""
    predicted_risk_level: RiskLevel = RiskLevel.SAFE
    confidence_in_prediction: float = 0.5
    potential_damages: List[str] = field(default_factory=list)
    safety_mitigations: List[str] = field(default_factory=list)
    similar_past_incidents: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Complete risk assessment for an agent action.

    Combines static analysis (command patterns) with dynamic simulation
    (predicted outcome) and experience-based assessment (past failures).

    Attributes:
        action: The action being assessed.
        risk_level: Overall risk level.
        static_risk: Risk from static analysis (command patterns).
        simulation_risk: Risk from pre-execution simulation.
        experience_risk: Risk from past experience (episodic memory).
        simulation: Detailed simulation result.
        recommendation: What to do (execute, block, request approval).
        reasoning: Why this risk level was assigned.
        timestamp: When the assessment was made.
    """

    action: str = ""
    risk_level: RiskLevel = RiskLevel.SAFE
    static_risk: RiskLevel = RiskLevel.SAFE
    simulation_risk: RiskLevel = RiskLevel.SAFE
    experience_risk: RiskLevel = RiskLevel.SAFE
    simulation: Optional[SimulationResult] = None
    recommendation: str = "execute"
    reasoning: str = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def should_execute(self) -> bool:
        """Whether the action should be executed."""
        return not self.risk_level.should_block

    @property
    def needs_approval(self) -> bool:
        """Whether the action needs approval before execution."""
        return self.risk_level.requires_approval


class RiskSimulator:
    """Pre-execution risk assessment for agent actions.

    This simulator adds ToolEmu-style dynamic risk assessment on top
    of Losion's existing static command validation. Before executing
    any action, the simulator:

    1. Performs static analysis (command patterns, file paths, etc.)
    2. Simulates the predicted outcome
    3. Checks past experience for similar failures
    4. Combines all signals into a single risk assessment
    5. Routes to appropriate execution path

    Static Analysis (command-level):
    - Pattern matching against known dangerous commands
    - File path analysis (system files, user data)
    - Network access detection
    - Resource usage estimation

    Dynamic Simulation (ToolEmu-style):
    - Predicts what a command will do before executing it
    - Uses model-based prediction when available
    - Falls back to heuristic prediction otherwise

    Experience-Based:
    - Checks episodic memory for similar past failures
    - Adjusts risk based on tool trust scores

    Usage:
        simulator = RiskSimulator()
        assessment = simulator.assess(
            action="terminal_execute",
            command="rm -rf /tmp/test",
            domain="code",
        )
        if assessment.should_execute:
            result = terminal.execute(command)
        elif assessment.needs_approval:
            if get_user_approval():
                result = terminal.execute(command)

    Args:
        enable_simulation: Whether to run pre-execution simulation.
        enable_experience: Whether to check episodic memory.
        risk_threshold: Risk level above which approval is required.
    """

    # Static risk patterns for terminal commands
    CRITICAL_PATTERNS: List[str] = [
        r"rm\s+-rf\s+/",
        r"mkfs",
        r"dd\s+(if|of)=",
        r":\(\)\{\s*:\|:&\s*\}",
        r"format\s+[A-Z]:",
        r">\s*/dev/sd",
        r"chmod\s+777\s+/",
        r"curl\s+\S+\s*\|\s*(sh|bash)",
        r"wget\s+\S+\s*\|\s*(sh|bash)",
    ]

    HIGH_PATTERNS: List[str] = [
        r"sudo\s+",
        r"su\s+",
        r"chmod\s+",
        r"chown\s+",
        r"iptables",
        r"sysctl",
        r"systemctl\s+(stop|disable|mask)",
        r"nmap",
        r"netcat",
        r"nc\s+-l",
    ]

    MEDIUM_PATTERNS: List[str] = [
        r"pip\s+install",
        r"npm\s+install\s+-g",
        r"apt\s+install",
        r"yum\s+install",
        r"docker\s+run",
        r"kubectl\s+",
        r"git\s+push\s+--force",
    ]

    # File paths that are high-risk to modify
    PROTECTED_PATHS: List[str] = [
        "/etc/",
        "/usr/",
        "/bin/",
        "/sbin/",
        "/boot/",
        "/root/",
        "/var/log/",
    ]

    def __init__(
        self,
        enable_simulation: bool = True,
        enable_experience: bool = True,
        risk_threshold: RiskLevel = RiskLevel.MEDIUM,
    ) -> None:
        self.enable_simulation = enable_simulation
        self.enable_experience = enable_experience
        self.risk_threshold = risk_threshold
        self._assessment_history: List[RiskAssessment] = []

    def assess(
        self,
        action: str,
        command: Optional[str] = None,
        domain: Optional[str] = None,
        tool_name: Optional[str] = None,
        calibration_engine: Optional[Any] = None,
        episodic_memory: Optional[Any] = None,
    ) -> RiskAssessment:
        """Assess the risk of an agent action before execution.

        Combines three assessment layers:
        1. Static analysis: Pattern matching on command/action
        2. Dynamic simulation: Predicted outcome analysis
        3. Experience-based: Past failure patterns

        Args:
            action: The agent action being assessed.
            command: The specific command (for terminal actions).
            domain: Domain classification.
            tool_name: Name of the tool being used.
            calibration_engine: Optional CalibrationEngine for trust scores.
            episodic_memory: Optional EpisodicMemory for past failures.

        Returns:
            RiskAssessment with the combined risk level and recommendation.
        """
        # === Layer 1: Static Analysis ===
        static_risk = self._static_analysis(action, command)

        # === Layer 2: Dynamic Simulation (ToolEmu-style) ===
        simulation_result = None
        simulation_risk = RiskLevel.SAFE

        if self.enable_simulation and command is not None:
            simulation_result = self._simulate_outcome(action, command, domain)
            simulation_risk = simulation_result.predicted_risk_level

        # === Layer 3: Experience-Based Assessment ===
        experience_risk = RiskLevel.SAFE
        if self.enable_experience and episodic_memory is not None:
            experience_risk = self._experience_assessment(
                action, command, domain, episodic_memory
            )

        # === Combine risk levels (take the maximum) ===
        risk_order = {
            RiskLevel.SAFE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }
        all_risks = [static_risk, simulation_risk, experience_risk]
        overall_risk = max(all_risks, key=lambda r: risk_order[r])

        # === Adjust with calibration data ===
        if calibration_engine is not None and tool_name:
            trust = calibration_engine.get_tool_trust(tool_name, domain)
            if trust < 0.3:
                # Low trust → increase risk level by one
                risk_levels = list(RiskLevel)
                current_idx = risk_levels.index(overall_risk)
                if current_idx < len(risk_levels) - 1:
                    overall_risk = risk_levels[current_idx + 1]

        # === Build recommendation ===
        recommendation = self._build_recommendation(overall_risk)

        # === Build reasoning ===
        reasoning = self._build_reasoning(
            action, command, static_risk, simulation_risk,
            experience_risk, overall_risk,
        )

        # === Create assessment ===
        assessment = RiskAssessment(
            action=action,
            risk_level=overall_risk,
            static_risk=static_risk,
            simulation_risk=simulation_risk,
            experience_risk=experience_risk,
            simulation=simulation_result,
            recommendation=recommendation,
            reasoning=reasoning,
        )

        # Record assessment
        self._assessment_history.append(assessment)

        return assessment

    def _static_analysis(
        self, action: str, command: Optional[str]
    ) -> RiskLevel:
        """Perform static risk analysis on the action/command.

        Uses pattern matching to identify known dangerous patterns
        in terminal commands and file operations.
        """
        if command is None:
            # Non-terminal actions are generally safer
            if action in ("web_search", "skill_lookup", "verify_output"):
                return RiskLevel.SAFE
            if action in ("tool_search", "skill_create"):
                return RiskLevel.LOW
            return RiskLevel.LOW

        command_lower = command.lower().strip()

        # Check critical patterns
        for pattern in self.CRITICAL_PATTERNS:
            if re.search(pattern, command_lower):
                return RiskLevel.CRITICAL

        # Check high-risk patterns
        for pattern in self.HIGH_PATTERNS:
            if re.search(pattern, command_lower):
                return RiskLevel.HIGH

        # Check medium-risk patterns
        for pattern in self.MEDIUM_PATTERNS:
            if re.search(pattern, command_lower):
                return RiskLevel.MEDIUM

        # Check protected file paths
        for path in self.PROTECTED_PATHS:
            if path in command_lower:
                return RiskLevel.HIGH

        # Check for network operations
        if any(kw in command_lower for kw in ["curl", "wget", "nc ", "netcat", "ssh"]):
            return RiskLevel.MEDIUM

        # Check for file operations on user data
        if any(kw in command_lower for kw in ["rm ", "del ", "erase "]):
            return RiskLevel.MEDIUM

        # Simple commands are generally safe
        if any(kw in command_lower for kw in ["echo", "ls", "cat", "pwd", "whoami", "date"]):
            return RiskLevel.SAFE

        # Default for unknown commands
        return RiskLevel.LOW

    def _simulate_outcome(
        self,
        action: str,
        command: str,
        domain: Optional[str],
    ) -> SimulationResult:
        """Simulate the predicted outcome of an action (ToolEmu-style).

        Instead of executing the command, predict what would happen
        based on heuristics and pattern analysis. In a full implementation,
        this would use the Losion model for prediction.
        """
        result = SimulationResult()

        # Heuristic-based prediction
        command_lower = command.lower()

        # Predicted outcome
        if "rm" in command_lower:
            result.predicted_outcome = "Files/directories will be permanently deleted"
            result.potential_damages.append("Irreversible data loss")
            result.safety_mitigations.append("Verify target paths before deletion")
        elif "curl" in command_lower or "wget" in command_lower:
            result.predicted_outcome = "Data will be downloaded from the internet"
            result.potential_damages.append("Malicious code could be downloaded")
            result.potential_damages.append("Sensitive data could be exfiltrated")
            result.safety_mitigations.append("Verify the URL is trusted")
        elif "pip" in command_lower or "npm" in command_lower:
            result.predicted_outcome = "Software packages will be installed"
            result.potential_damages.append("Malicious packages could be installed")
            result.potential_damages.append("Dependency conflicts could break system")
            result.safety_mitigations.append("Use virtual environments")
        elif "python" in command_lower or "node" in command_lower:
            result.predicted_outcome = "Code will be executed"
            result.potential_damages.append("Arbitrary code execution risk")
            result.safety_mitigations.append("Run in sandboxed environment")
        else:
            result.predicted_outcome = "Command will execute normally"
            result.confidence_in_prediction = 0.3

        # Set predicted risk level based on potential damages
        if len(result.potential_damages) >= 2:
            result.predicted_risk_level = RiskLevel.HIGH
        elif len(result.potential_damages) == 1:
            result.predicted_risk_level = RiskLevel.MEDIUM
        else:
            result.predicted_risk_level = RiskLevel.LOW

        return result

    def _experience_assessment(
        self,
        action: str,
        command: Optional[str],
        domain: Optional[str],
        episodic_memory: Any,
    ) -> RiskLevel:
        """Assess risk based on past experience.

        Checks episodic memory for similar actions that failed
        or caused problems in the past.
        """
        try:
            # Look for past failures with similar actions
            query = f"{action} {command or ''}"
            lessons = episodic_memory.get_lessons_for_query(
                query=query, domain=domain, limit=3
            )

            # Count failure-related lessons
            failure_lessons = [
                l for l in lessons
                if any(kw in l.lower() for kw in ["fail", "error", "dangerous", "risk", "avoid"])
            ]

            if len(failure_lessons) >= 2:
                return RiskLevel.HIGH
            elif len(failure_lessons) == 1:
                return RiskLevel.MEDIUM

            # Check action recommendations
            recs = episodic_memory.get_action_recommendations(query, domain)
            action_score = recs.get(action, 0.0)

            if action_score < -0.3:
                return RiskLevel.HIGH
            elif action_score < -0.1:
                return RiskLevel.MEDIUM

        except Exception:
            pass

        return RiskLevel.SAFE

    def _build_recommendation(self, risk_level: RiskLevel) -> str:
        """Build a recommendation based on the risk level."""
        recommendations = {
            RiskLevel.SAFE: "execute",
            RiskLevel.LOW: "execute_with_logging",
            RiskLevel.MEDIUM: "request_approval",
            RiskLevel.HIGH: "require_approval_with_audit",
            RiskLevel.CRITICAL: "block",
        }
        return recommendations.get(risk_level, "request_approval")

    def _build_reasoning(
        self,
        action: str,
        command: Optional[str],
        static_risk: RiskLevel,
        simulation_risk: RiskLevel,
        experience_risk: RiskLevel,
        overall_risk: RiskLevel,
    ) -> str:
        """Build a human-readable reasoning for the risk assessment."""
        parts = [f"Risk assessment for '{action}'"]

        if command:
            parts.append(f"command='{command[:50]}'")

        parts.append(f"static={static_risk.value}")
        parts.append(f"simulation={simulation_risk.value}")
        parts.append(f"experience={experience_risk.value}")
        parts.append(f"overall={overall_risk.value}")

        return "; ".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get risk simulation statistics."""
        total = len(self._assessment_history)
        risk_counts = {}
        for assessment in self._assessment_history:
            key = assessment.risk_level.value
            risk_counts[key] = risk_counts.get(key, 0) + 1

        return {
            "total_assessments": total,
            "risk_distribution": risk_counts,
            "blocked_actions": sum(
                1 for a in self._assessment_history
                if a.risk_level.should_block
            ),
            "approval_required": sum(
                1 for a in self._assessment_history
                if a.risk_level.requires_approval
            ),
        }
