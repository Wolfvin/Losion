"""
RLVR — Reinforcement Learning with Verifiable Rewards
=======================================================

Implements RLVR, a training paradigm that replaces learned reward models
with **objective, programmable reward functions** (math verification,
code execution, logic checks, format checking) to provide noise-free
reward signals for reinforcement learning.

The core insight, established across multiple NeurIPS 2025 papers
(posters 119944, 116633) and arXiv 2601.05607 / 2603.22117, is that
for domains where correctness can be verified programmatically, a
verifiable reward is strictly superior to a learned reward model:

1. **No Reward Hacking** — Verifiable rewards are objective; the model
   cannot game them by exploiting reward-model blind spots.

2. **Zero Reward Noise** — Unlike a learned RM with imperfect accuracy,
   verifiers are deterministic and noise-free for the tasks they cover.

3. **Curriculum-Friendly** — Verification strictness can be tuned
   (e.g., partial credit, relaxed matching), enabling automatic
   curriculum scheduling from easy to hard criteria.

4. **Composable** — Multiple verifiers can be combined with learnable
   weights to create rich reward signals that cover different aspects
   of response quality.

Architecture
------------
RLVR consists of five components:

    ┌───────────────────┐
    │   RewardVerifier   │  ← Abstract base class
    ├───────────────────┤
    │  MathVerifier      │  ← SymPy / eval-based math checking
    │  CodeVerifier      │  ← Sandboxed code execution
    │  FormatVerifier    │  ← Regex / structural format checking
    │  ExactMatchVerifier│  ← String-level exact / fuzzy matching
    ├───────────────────┤
    │  CompositeVerifier │  ← Weighted combination of verifiers
    ├───────────────────┤
    │  RLVRTrainer       │  ← Integrates with DAPO / GRPO
    └───────────────────┘

**RewardVerifier** is the abstract base class.  Each concrete verifier
implements ``verify(prompt, response, reference) -> float``.

**CompositeVerifier** combines multiple verifiers with configurable
weights.  It can also apply curriculum-based difficulty scheduling
where verification criteria tighten over training.

**RLVRTrainer** wraps an existing RL trainer (DAPO or GRPO) and
replaces its reward function with verifiable rewards.  It handles:

- Response generation and reward computation.
- Curriculum scheduling (easy → hard verification).
- Per-verifier reward tracking and logging.

Compatibility
-------------
RLVRTrainer is compatible with both ``DAPOTrainer`` and ``GRPOTrainer``
from the Losion training framework.  It can also operate standalone
with any ``nn.Module`` that exposes a ``generate()`` method.

References
----------
- NeurIPS 2025 Poster 119944 — RLVR for mathematical reasoning.
- NeurIPS 2025 Poster 116633 — Curriculum-based verifiable rewards.
- arXiv 2601.05607 — "Verifiable Rewards for RL Fine-Tuning".
- arXiv 2603.22117 — "Scaling RLVR to Multi-Domain Verification".
- Yu et al., "DAPO" (arXiv 2503.14476, 2025) — RL training method.
- Shao et al., "DeepSeekMath / GRPO" (2024) — RL training method.

Hardware: Pure PyTorch + standard library.  No custom CUDA kernels.
Code execution is sandboxed via ``exec()`` with restricted globals.
"""

from __future__ import annotations

import logging
import math
import re
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class VerificationDifficulty(Enum):
    """Curriculum difficulty levels for verifiable rewards.

    Used by ``CompositeVerifier`` to gradually increase verification
    strictness during training, following the curriculum-learning
    approach from NeurIPS 2025 Poster 116633.

    Levels:
        EASY: Relaxed matching (partial credit, loose format).
        MEDIUM: Standard matching (exact answers, correct format).
        HARD: Strict matching (exact + step verification, no partial).
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class VerificationResult(Enum):
    """Outcome of a single verification check.

    Values:
        CORRECT: The response fully satisfies the verification criterion.
        PARTIAL: The response partially satisfies the criterion
            (eligible for partial credit).
        INCORRECT: The response does not satisfy the criterion.
        ERROR: Verification itself failed (e.g., code execution error).
    """

    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    ERROR = "error"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RLVRConfig:
    """Configuration for RLVR training.

    Attributes:
        # ---- Core RLVR settings ----
        verifiers:
            List of ``RewardVerifier`` instances to use for reward
            computation.  At least one verifier must be provided.
        verifier_weights:
            Optional per-verifier weights for ``CompositeVerifier``.
            If None, equal weights are used.  Must match the length of
            ``verifiers``.
        use_curriculum:
            Whether to use curriculum-based difficulty scheduling.
            When True, verification starts at EASY and progresses to
            HARD over ``curriculum_warmup_steps``.  Default True.
        curriculum_warmup_steps:
            Number of training steps over which the difficulty
            progresses from EASY to HARD.  Default 1000.
        initial_difficulty:
            Starting difficulty for curriculum scheduling.  Default
            ``VerificationDifficulty.EASY``.
        final_difficulty:
            Final difficulty for curriculum scheduling.  Default
            ``VerificationDifficulty.HARD``.
        partial_credit_weight:
            Weight for PARTIAL results relative to CORRECT (e.g., 0.5
            means a partial result receives half the reward).  Default
            0.5.
        error_penalty:
            Reward value assigned when verification returns ERROR.
            Default -1.0.

        # ---- RL training settings ----
        rl_method:
            Which RL method to use: ``"dapo"`` or ``"grpo"``.
            Default ``"dapo"``.
        num_responses_per_prompt:
            Number of responses sampled per prompt for group-relative
            advantage computation.  Default 8.
        clip_ratio_low:
            DAPO lower clip ratio.  Default 0.2.
        clip_ratio_high:
            DAPO upper clip ratio.  Default 0.28.
        kl_coefficient:
            KL divergence penalty coefficient.  Default 0.1.
        entropy_coefficient:
            Entropy bonus coefficient.  Default 0.01.
        learning_rate:
            Peak learning rate for the policy optimizer.  Default 1e-6.
        max_grad_norm:
            Maximum gradient norm for clipping.  Default 1.0.
        temperature:
            Sampling temperature for response generation.  Default 0.7.
        max_response_length:
            Maximum response length in tokens.  Default 2048.

        # ---- Logging ----
        log_verifier_details:
            Whether to log per-verifier reward details at each step.
            Default True.
    """

    # Core RLVR
    verifiers: List[Any] = field(default_factory=list)
    verifier_weights: Optional[List[float]] = None
    use_curriculum: bool = True
    curriculum_warmup_steps: int = 1000
    initial_difficulty: VerificationDifficulty = VerificationDifficulty.EASY
    final_difficulty: VerificationDifficulty = VerificationDifficulty.HARD
    partial_credit_weight: float = 0.5
    error_penalty: float = -1.0

    # RL training
    rl_method: str = "dapo"
    num_responses_per_prompt: int = 8
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    kl_coefficient: float = 0.1
    entropy_coefficient: float = 0.01
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    temperature: float = 0.7
    max_response_length: int = 2048

    # Logging
    log_verifier_details: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.rl_method not in ("dapo", "grpo"):
            raise ValueError(
                f"rl_method must be 'dapo' or 'grpo', got '{self.rl_method}'"
            )
        if self.verifier_weights is not None:
            if len(self.verifier_weights) != len(self.verifiers):
                raise ValueError(
                    f"verifier_weights length ({len(self.verifier_weights)}) "
                    f"must match verifiers length ({len(self.verifiers)})"
                )
        if self.partial_credit_weight < 0 or self.partial_credit_weight > 1:
            raise ValueError(
                f"partial_credit_weight must be in [0, 1], "
                f"got {self.partial_credit_weight}"
            )


# ============================================================================
# RewardVerifier — Abstract Base Class
# ============================================================================


class RewardVerifier(ABC):
    """Abstract base class for verifiable reward functions.

    Each verifier implements a ``verify`` method that takes a prompt,
    a model response, and an optional reference answer, and returns a
    float reward in [0, 1] (or a negative value for errors / penalties).

    Subclasses must implement:
        - ``verify(prompt, response, reference, difficulty)``

    Optionally override:
        - ``verify_batch(prompts, responses, references, difficulty)``

    Args:
        name: Human-readable name for this verifier (used in logging).
        weight: Default weight when used in a CompositeVerifier.
    """

    def __init__(self, name: str = "base_verifier", weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    @abstractmethod
    def verify(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        difficulty: VerificationDifficulty = VerificationDifficulty.MEDIUM,
    ) -> Tuple[float, VerificationResult]:
        """Verify a single response and compute its reward.

        Args:
            prompt: The input prompt string.
            response: The model's response string.
            reference: Optional reference / ground-truth answer.
            difficulty: Verification strictness level.

        Returns:
            Tuple ``(reward, result)`` where:
            - ``reward`` is a float (typically in [0, 1] or negative
              for penalties).
            - ``result`` is a ``VerificationResult`` enum value.
        """
        ...

    def verify_batch(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[Optional[str]]] = None,
        difficulty: VerificationDifficulty = VerificationDifficulty.MEDIUM,
    ) -> Tuple[List[float], List[VerificationResult]]:
        """Verify a batch of responses.

        Default implementation calls ``verify`` in a loop.  Subclasses
        can override for vectorised verification.

        Args:
            prompts: List of prompt strings.
            responses: List of response strings.
            references: Optional list of reference answers.
            difficulty: Verification strictness level.

        Returns:
            Tuple ``(rewards, results)`` — lists of floats and
            VerificationResult values.
        """
        rewards: List[float] = []
        results: List[VerificationResult] = []

        for i, response in enumerate(responses):
            ref = references[i] if references is not None else None
            reward, result = self.verify(
                prompts[i] if i < len(prompts) else "",
                response,
                ref,
                difficulty,
            )
            rewards.append(reward)
            results.append(result)

        return rewards, results


# ============================================================================
# MathVerifier
# ============================================================================


class MathVerifier(RewardVerifier):
    """Verifies mathematical answers by comparing computed results.

    Supports two verification strategies:

    1. **Symbolic verification** (via ``sympy`` if available):  Parses
       both the reference and response as symbolic expressions and
       checks equality.  This handles algebraic equivalence (e.g.,
       ``x**2 + 2*x + 1`` vs ``(x+1)**2``).

    2. **Numeric evaluation**:  Evaluates both expressions numerically
       and compares the results within a tolerance.  This handles cases
       where symbolic parsing fails but the numeric values match.

    Difficulty levels:
        - EASY: Numeric comparison with loose tolerance (1e-2).
        - MEDIUM: Numeric comparison with standard tolerance (1e-4).
        - HARD: Symbolic equality required (falls back to numeric with
          tight tolerance 1e-6).

    The verifier extracts the final answer from the response using
    common patterns like ``\\boxed{...}``, ``Answer: ...``, or the
    last numeric expression on the final line.

    References:
        - arXiv 2601.05607, Section 3.2 — Math verification pipeline.
        - NeurIPS 2025 Poster 119944 — Math RLVR scaling.

    Args:
        name: Verifier name.  Default ``"math"``.
        weight: Default weight.  Default 1.0.
        numeric_tolerance_medium: Tolerance for MEDIUM difficulty.
            Default 1e-4.
        numeric_tolerance_easy: Tolerance for EASY difficulty.
            Default 1e-2.
        numeric_tolerance_hard: Tolerance for HARD difficulty (fallback).
            Default 1e-6.
    """

    # Patterns for extracting the final answer from a response
    _ANSWER_PATTERNS: List[str] = [
        r"\\boxed\{([^}]+)\}",          # LaTeX \boxed{...}
        r"\\boxed\s*\(([^)]+)\)",        # LaTeX \boxed(...)
        r"[Aa]nswer:\s*(.+?)(?:\.|$)",   # Answer: ...
        r"[Tt]he answer is\s*(.+?)(?:\.|$)",  # The answer is ...
        r"=\s*([+-]?\d*\.?\d+)",         # Last = <number>
    ]

    def __init__(
        self,
        name: str = "math",
        weight: float = 1.0,
        numeric_tolerance_medium: float = 1e-4,
        numeric_tolerance_easy: float = 1e-2,
        numeric_tolerance_hard: float = 1e-6,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.numeric_tolerance_medium = numeric_tolerance_medium
        self.numeric_tolerance_easy = numeric_tolerance_easy
        self.numeric_tolerance_hard = numeric_tolerance_hard
        self._sympy_available = False
        try:
            import sympy  # noqa: F401
            self._sympy_available = True
        except ImportError:
            pass

    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract the final answer from a response string.

        Tries multiple patterns in order; returns the first match.

        Args:
            text: Response string.

        Returns:
            Extracted answer string, or None if no pattern matches.
        """
        for pattern in self._ANSWER_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def _numeric_eval(self, expr_str: str) -> Optional[float]:
        """Safely evaluate a numeric expression.

        Uses Python's ``eval()`` with a restricted namespace that only
        includes math functions.  This is NOT a full sandbox — do not
        use with untrusted code.

        Args:
            expr_str: Expression string to evaluate.

        Returns:
            Float result, or None if evaluation fails.
        """
        safe_globals = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }
        try:
            result = eval(expr_str, safe_globals, {})  # noqa: S307
            return float(result)
        except Exception:
            return None

    def verify(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        difficulty: VerificationDifficulty = VerificationDifficulty.MEDIUM,
    ) -> Tuple[float, VerificationResult]:
        """Verify a math response.

        Args:
            prompt: Input prompt (unused by this verifier).
            response: Model's response string.
            reference: Reference answer string.
            difficulty: Verification strictness.

        Returns:
            Tuple ``(reward, result)``.
        """
        if reference is None:
            return 0.0, VerificationResult.ERROR

        # Extract answers
        ref_answer = self._extract_answer(reference) or reference.strip()
        resp_answer = self._extract_answer(response)

        if resp_answer is None:
            # Could not extract an answer from the response
            return 0.0, VerificationResult.INCORRECT

        # ---- Numeric comparison ----
        ref_val = self._numeric_eval(ref_answer)
        resp_val = self._numeric_eval(resp_answer)

        if ref_val is not None and resp_val is not None:
            # Choose tolerance based on difficulty
            if difficulty == VerificationDifficulty.EASY:
                tol = self.numeric_tolerance_easy
            elif difficulty == VerificationDifficulty.HARD:
                tol = self.numeric_tolerance_hard
            else:
                tol = self.numeric_tolerance_medium

            if abs(ref_val - resp_val) <= tol * max(abs(ref_val), 1.0):
                return 1.0, VerificationResult.CORRECT
            # Partial credit for EASY: within 10x tolerance
            if difficulty == VerificationDifficulty.EASY:
                if abs(ref_val - resp_val) <= tol * 10 * max(abs(ref_val), 1.0):
                    return 0.5, VerificationResult.PARTIAL
            return 0.0, VerificationResult.INCORRECT

        # ---- Symbolic comparison (if sympy available) ----
        if self._sympy_available and difficulty == VerificationDifficulty.HARD:
            try:
                import sympy
                ref_sym = sympy.sympify(ref_answer)
                resp_sym = sympy.sympify(resp_answer)
                if sympy.simplify(ref_sym - resp_sym) == 0:
                    return 1.0, VerificationResult.CORRECT
            except Exception:
                pass

        # ---- String-level comparison as fallback ----
        if ref_answer.strip().lower() == resp_answer.strip().lower():
            return 1.0, VerificationResult.CORRECT

        return 0.0, VerificationResult.INCORRECT


# ============================================================================
# CodeVerifier
# ============================================================================


class CodeVerifier(RewardVerifier):
    """Verifies code responses by executing them in a sandboxed environment.

    The verifier runs the model's code response using Python's ``exec()``
    with a restricted set of built-in functions, then checks the output
    against expected test cases or a reference implementation.

    **Security note**: The sandbox restricts builtins but is not
    foolproof.  For production use, consider containerised execution
    (Docker, gVisor) or a remote code execution service.

    Difficulty levels:
        - EASY: Code runs without errors (syntax + runtime check only).
        - MEDIUM: Code produces correct output for basic test cases.
        - HARD: Code produces correct output for all test cases
            including edge cases.

    References:
        - arXiv 2603.22117, Section 4 — Code verification pipeline.
        - NeurIPS 2025 Poster 116633 — Curriculum code verification.

    Args:
        name: Verifier name.  Default ``"code"``.
        weight: Default weight.  Default 1.0.
        timeout_seconds: Maximum execution time per code snippet.
            Default 5.
        max_output_length: Maximum length of captured stdout.  Default
            10000 characters.
        allowed_builtins: Set of builtin names allowed in the sandbox.
    """

    # Safe builtins for sandboxed execution
    _DEFAULT_ALLOWED_BUILTINS: set = {
        "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "hex", "int",
        "isinstance", "len", "list", "map", "max", "min", "oct",
        "ord", "pow", "print", "range", "repr", "round", "set",
        "sorted", "str", "sum", "tuple", "type", "zip",
    }

    # Dangerous dunder attributes that allow sandbox escape
    _BLOCKED_ATTRS: frozenset = frozenset({
        "__class__", "__bases__", "__mro__", "__subclasses__",
        "__globals__", "__code__", "__func__", "__self__",
        "__dict__", "__weakref__", "__module__", "__import__",
    })

    def __init__(
        self,
        name: str = "code",
        weight: float = 1.0,
        timeout_seconds: int = 5,
        max_output_length: int = 10000,
        allowed_builtins: Optional[set] = None,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.timeout_seconds = timeout_seconds
        self.max_output_length = max_output_length
        self.allowed_builtins = allowed_builtins or self._DEFAULT_ALLOWED_BUILTINS

    def _execute_code(
        self,
        code: str,
        test_inputs: Optional[List[Any]] = None,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Execute code in a subprocess sandboxed environment.

        v2.4.0: Replaced the previous regex-based static analysis + in-process
        ``exec()`` sandbox with subprocess isolation. The regex approach was
        fundamentally flawed — it operates on source code as a string, but
        Python can construct attribute names dynamically via string concatenation
        (``getattr(x, '__cla' + 'ss__')``), variables (``attr = '__class__';
        getattr(x, attr)``), or ``chr()`` encoding (``getattr(x, chr(95)*2 +
        'class' + chr(95)*2)``). All three bypasses were verified against the
        previous implementation.

        The new approach runs code in a **separate subprocess** with:
        - CPU time limit (``resource.RLIMIT_CPU``) via ``preexec_fn``
        - Memory limit (``resource.RLIMIT_AS``) to prevent OOM attacks
        - No file descriptor inheritance beyond stdin/stdout/stderr
        - Timeout enforced by both ``subprocess.run(timeout=)`` and ``ulimit``
        - Restricted builtins still applied as a defense-in-depth layer

        For production deployments, Docker/gVisor isolation is still
        recommended for defense against kernel exploits and sophisticated
        attacks. See SECURITY.md for the full threat model.

        Args:
            code: Python code string to execute.
            test_inputs: Optional list of test inputs (passed as
                ``test_input`` variable in the execution namespace).

        Returns:
            Tuple ``(stdout_output, error_message, success)``:
            - ``stdout_output``: Captured stdout (or None on error).
            - ``error_message``: Error traceback (or None on success).
            - ``success``: Whether execution completed without error.
        """
        import json
        import subprocess
        import sys

        # --- Defense in depth: still block obvious dunder attrs ---
        # This catches casual misuse (not malicious bypass). The real
        # security boundary is the subprocess isolation below.
        for attr in self._BLOCKED_ATTRS:
            if attr in code:
                import re
                patterns = [
                    rf"""\.\s*{re.escape(attr)}\b""",
                    rf"""\[\s*['\"]\s*{re.escape(attr)}\s*['\"]\s*\]""",
                    rf"""getattr\s*\([^,]+,\s*['\"]\s*{re.escape(attr)}\s*['\"]""",
                ]
                for pattern in patterns:
                    if re.search(pattern, code):
                        error_msg = (
                            f"Sandbox violation: access to '{attr}' is blocked. "
                            f"This is a security restriction to prevent sandbox escape "
                            f"via Python's object introspection chain."
                        )
                        return None, error_msg, False

        # --- Subprocess sandbox ---
        # Build a wrapper script that:
        # 1. Restricts builtins (defense in depth)
        # 2. Runs the user code
        # 3. Returns results as JSON on stdout
        allowed_builtins_json = json.dumps(sorted(self.allowed_builtins))
        test_inputs_json = json.dumps(test_inputs)

        sandbox_script = f'''
import sys
import json
import io
from contextlib import redirect_stdout

# Restricted builtins (defense in depth — real isolation is the subprocess)
_allowed = set({allowed_builtins_json})
_safe_builtins = {{}}
_b = __builtins__
if isinstance(_b, dict):
    for _name in _allowed:
        if _name in _b:
            _safe_builtins[_name] = _b[_name]
else:
    for _name in _allowed:
        if hasattr(_b, _name):
            _safe_builtins[_name] = getattr(_b, _name)
__builtins__ = _safe_builtins

_test_inputs = {test_inputs_json}

_stdout = io.StringIO()
_local_vars = {{"test_input": _test_inputs}}
_success = False
_output = None
_error = None

try:
    _compiled = compile({repr(code)}, "<rlvr_sandbox>", "exec")
    with redirect_stdout(_stdout):
        exec(_compiled, {{"__builtins__": _safe_builtins}}, _local_vars)
    _output = _stdout.getvalue()[:{self.max_output_length}]
    _success = True
except Exception:
    import traceback
    _error = traceback.format_exc()

_result = {{"success": _success, "output": _output, "error": _error}}
sys.stdout.write(json.dumps(_result))
'''

        def _set_resource_limits():
            """Set resource limits in the child process."""
            try:
                import resource
                # CPU time limit (seconds) — kills process if exceeded
                resource.setrlimit(resource.RLIMIT_CPU, (
                    self.timeout_seconds,
                    self.timeout_seconds + 1
                ))
                # Memory limit (bytes) — 256MB max
                mem_limit = 256 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
                # No core dumps
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except (ImportError, ValueError, OSError):
                pass  # resource module not available on all platforms

        try:
            result = subprocess.run(
                [sys.executable, "-c", sandbox_script],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds + 2,  # Extra margin beyond ulimit
                preexec_fn=_set_resource_limits,
                # Security: close file descriptors beyond stdin/stdout/stderr
                close_fds=True,
            )
        except subprocess.TimeoutExpired:
            return None, f"Execution timed out after {self.timeout_seconds}s", False
        except OSError as e:
            return None, f"Failed to start sandbox process: {e}", False

        if result.returncode != 0:
            # Process was killed (signal) or had a fatal error
            stderr = result.stderr.strip()
            if "resource.RLIMIT_CPU" in stderr or "CPU time limit" in stderr:
                return None, f"Execution timed out after {self.timeout_seconds}s", False
            return None, f"Process error (exit code {result.returncode}): {stderr}", False

        # Parse JSON result from subprocess
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None, f"Failed to parse sandbox output: {result.stdout[:500]}", False

        if parsed.get("success"):
            return parsed.get("output"), None, True
        else:
            return None, parsed.get("error", "Unknown error"), False

    def verify(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        difficulty: VerificationDifficulty = VerificationDifficulty.MEDIUM,
    ) -> Tuple[float, VerificationResult]:
        """Verify a code response.

        Args:
            prompt: Input prompt (may contain the coding problem).
            response: Model's code response string.
            reference: Reference code or expected output.
                If reference starts with ``"test_cases:"``, it is
                parsed as a JSON list of (input, expected_output) pairs.
                Otherwise, it is treated as reference code whose output
                is compared against the response's output.
            difficulty: Verification strictness.

        Returns:
            Tuple ``(reward, result)``.
        """
        # ---- Extract code from response ----
        # Look for code blocks marked with ```python ... ```
        code_blocks = re.findall(
            r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL
        )
        if code_blocks:
            code = code_blocks[0]
        else:
            code = response  # Assume the entire response is code

        # ---- EASY: Just check if code runs ----
        output, error, success = self._execute_code(code)

        if not success:
            if difficulty == VerificationDifficulty.EASY:
                # EASY: partial credit if at least the syntax is valid
                try:
                    compile(code, "<string>", "exec")
                    return 0.3, VerificationResult.PARTIAL
                except SyntaxError:
                    return 0.0, VerificationResult.INCORRECT
            return 0.0, VerificationResult.ERROR

        if difficulty == VerificationDifficulty.EASY:
            # Code runs without errors — that's enough for EASY
            return 1.0, VerificationResult.CORRECT

        # ---- MEDIUM / HARD: Compare output with reference ----
        if reference is None:
            # No reference to compare against; just reward execution success
            return 0.5, VerificationResult.PARTIAL

        # Parse reference as test cases or reference code
        if reference.startswith("test_cases:"):
            # Expected format: test_cases:[{"input": ..., "expected": ...}, ...]
            import json

            try:
                test_cases_str = reference[len("test_cases:"):]
                test_cases = json.loads(test_cases_str)
            except (json.JSONDecodeError, ValueError):
                return 0.0, VerificationResult.ERROR

            correct = 0
            total = len(test_cases)
            for tc in test_cases:
                test_output, _, test_success = self._execute_code(
                    code, test_inputs=tc.get("input")
                )
                if test_success and test_output is not None:
                    expected = str(tc.get("expected", "")).strip()
                    actual = test_output.strip()
                    if actual == expected:
                        correct += 1
                    elif difficulty == VerificationDifficulty.MEDIUM:
                        # Partial: numeric comparison
                        try:
                            if abs(float(actual) - float(expected)) < 1e-4:
                                correct += 1
                        except (ValueError, TypeError):
                            pass

            if total == 0:
                return 0.0, VerificationResult.ERROR

            ratio = correct / total
            if ratio == 1.0:
                return 1.0, VerificationResult.CORRECT
            elif ratio >= 0.5 and difficulty != VerificationDifficulty.HARD:
                return ratio, VerificationResult.PARTIAL
            else:
                return ratio, VerificationResult.INCORRECT
        else:
            # Reference is another code implementation — compare outputs
            ref_output, _, ref_success = self._execute_code(reference)
            if not ref_success:
                return 0.0, VerificationResult.ERROR

            if output is not None and ref_output is not None:
                if output.strip() == ref_output.strip():
                    return 1.0, VerificationResult.CORRECT
                # Try numeric comparison
                try:
                    if abs(float(output.strip()) - float(ref_output.strip())) < 1e-4:
                        return 1.0, VerificationResult.CORRECT
                except (ValueError, TypeError):
                    pass
                if difficulty == VerificationDifficulty.MEDIUM:
                    return 0.3, VerificationResult.PARTIAL

            return 0.0, VerificationResult.INCORRECT


# ============================================================================
# FormatVerifier
# ============================================================================


class FormatVerifier(RewardVerifier):
    """Verifies that responses follow expected structural format.

    Checks formatting rules such as:
    - Presence of required sections (e.g., "Step 1:", "Conclusion:").
    - Correct markdown structure (code blocks, lists, headings).
    - JSON / structured output validity.
    - Custom regex patterns.

    This verifier is essential for RLVR because format compliance is
    a prerequisite for other verifiers to work correctly.  For example,
    ``MathVerifier`` cannot extract a boxed answer if the response does
    not use the expected LaTeX format.

    Difficulty levels:
        - EASY: Only check that the response is non-empty and contains
          at least one required marker.
        - MEDIUM: Check all primary format requirements.
        - HARD: Check all format requirements including whitespace and
          ordering conventions.

    References:
        - arXiv 2601.05607, Section 3.3 — Format verification.
        - NeurIPS 2025 Poster 116633 — Format-aware curriculum.

    Args:
        name: Verifier name.  Default ``"format"``.
        weight: Default weight.  Default 0.5.
        required_patterns: List of regex patterns that must be present
            in the response.
        forbidden_patterns: List of regex patterns that must NOT be
            present in the response.
        min_length: Minimum response length (characters).  Default 10.
        max_length: Maximum response length (characters).  Default
            10000.
        must_be_valid_json: Whether the response must be valid JSON.
            Default False.
    """

    def __init__(
        self,
        name: str = "format",
        weight: float = 0.5,
        required_patterns: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
        min_length: int = 10,
        max_length: int = 10000,
        must_be_valid_json: bool = False,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.required_patterns = required_patterns or []
        self.forbidden_patterns = forbidden_patterns or []
        self.min_length = min_length
        self.max_length = max_length
        self.must_be_valid_json = must_be_valid_json

    def verify(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        difficulty: VerificationDifficulty = VerificationDifficulty.MEDIUM,
    ) -> Tuple[float, VerificationResult]:
        """Verify the format of a response.

        Args:
            prompt: Input prompt (unused by this verifier).
            response: Model's response string.
            reference: Unused.
            difficulty: Verification strictness.

        Returns:
            Tuple ``(reward, result)``.
        """
        score = 0.0
        max_score = 0.0

        # ---- Length check ----
        max_score += 1.0
        if len(response) >= self.min_length and len(response) <= self.max_length:
            score += 1.0
        elif difficulty == VerificationDifficulty.EASY and len(response) > 0:
            score += 0.5

        # ---- Required patterns ----
        if self.required_patterns:
            num_required = len(self.required_patterns)
            if difficulty == VerificationDifficulty.EASY:
                # EASY: at least one required pattern
                max_score += 1.0
                for pattern in self.required_patterns:
                    if re.search(pattern, response):
                        score += 1.0
                        break
            else:
                # MEDIUM / HARD: all required patterns
                max_score += 1.0
                found = 0
                for pattern in self.required_patterns:
                    if re.search(pattern, response):
                        found += 1
                if found == num_required:
                    score += 1.0
                elif difficulty == VerificationDifficulty.MEDIUM and found > 0:
                    score += found / num_required

        # ---- Forbidden patterns ----
        if self.forbidden_patterns:
            max_score += 1.0
            has_forbidden = any(
                re.search(pattern, response)
                for pattern in self.forbidden_patterns
            )
            if not has_forbidden:
                score += 1.0

        # ---- JSON validity ----
        if self.must_be_valid_json:
            max_score += 1.0
            try:
                import json
                json.loads(response)
                score += 1.0
            except (json.JSONDecodeError, ValueError):
                if difficulty == VerificationDifficulty.EASY:
                    # EASY: partial credit if it looks like JSON
                    if response.strip().startswith("{") or response.strip().startswith("["):
                        score += 0.3

        # ---- Compute final reward ----
        if max_score == 0:
            return 1.0, VerificationResult.CORRECT

        ratio = score / max_score
        if ratio >= 1.0:
            return 1.0, VerificationResult.CORRECT
        elif ratio >= 0.5:
            return ratio, VerificationResult.PARTIAL
        else:
            return ratio, VerificationResult.INCORRECT


# ============================================================================
# ExactMatchVerifier
# ============================================================================


class ExactMatchVerifier(RewardVerifier):
    """Verifies responses by exact or fuzzy string matching.

    Supports multiple matching strategies:

    - **Exact match**: The response exactly equals the reference.
    - **Case-insensitive match**: Match after lowercasing.
    - **Contains match**: The reference is a substring of the response.
    - **Fuzzy match**: Normalised edit-distance ratio (Levenshtein).

    Difficulty levels:
        - EASY: Contains match (reference substring in response).
        - MEDIUM: Case-insensitive exact match.
        - HARD: Exact match (including whitespace and case).

    References:
        - arXiv 2601.05607, Section 3.1 — Exact match baselines.

    Args:
        name: Verifier name.  Default ``"exact_match"``.
        weight: Default weight.  Default 1.0.
        fuzzy_threshold: Edit-distance ratio threshold for partial
            credit (0.0 to 1.0).  Default 0.8.
    """

    def __init__(
        self,
        name: str = "exact_match",
        weight: float = 1.0,
        fuzzy_threshold: float = 0.8,
    ) -> None:
        super().__init__(name=name, weight=weight)
        self.fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def _levenshtein_ratio(s1: str, s2: str) -> float:
        """Compute normalised Levenshtein similarity ratio.

        Returns a value in [0, 1] where 1.0 means the strings are
        identical.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Similarity ratio.
        """
        if len(s1) == 0 and len(s2) == 0:
            return 1.0
        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        # Simple dynamic programming implementation
        m, n = len(s1), len(s2)
        # Use a 2-row DP for memory efficiency
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev

        distance = prev[n]
        return 1.0 - distance / max(m, n)

    def verify(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        difficulty: VerificationDifficulty = VerificationDifficulty.MEDIUM,
    ) -> Tuple[float, VerificationResult]:
        """Verify a response by string matching.

        Args:
            prompt: Input prompt (unused by this verifier).
            response: Model's response string.
            reference: Reference answer string.
            difficulty: Verification strictness.

        Returns:
            Tuple ``(reward, result)``.
        """
        if reference is None:
            return 0.0, VerificationResult.ERROR

        resp = response.strip()
        ref = reference.strip()

        # ---- HARD: Exact match ----
        if difficulty == VerificationDifficulty.HARD:
            if resp == ref:
                return 1.0, VerificationResult.CORRECT
            # Partial credit via fuzzy match
            ratio = self._levenshtein_ratio(resp, ref)
            if ratio >= self.fuzzy_threshold:
                return ratio, VerificationResult.PARTIAL
            return 0.0, VerificationResult.INCORRECT

        # ---- MEDIUM: Case-insensitive exact match ----
        if difficulty == VerificationDifficulty.MEDIUM:
            if resp.lower() == ref.lower():
                return 1.0, VerificationResult.CORRECT
            # Partial credit for close fuzzy match
            ratio = self._levenshtein_ratio(resp.lower(), ref.lower())
            if ratio >= self.fuzzy_threshold:
                return ratio, VerificationResult.PARTIAL
            return 0.0, VerificationResult.INCORRECT

        # ---- EASY: Contains match ----
        if ref.lower() in resp.lower():
            return 1.0, VerificationResult.CORRECT
        # Partial credit for fuzzy match
        ratio = self._levenshtein_ratio(resp.lower(), ref.lower())
        if ratio >= 0.5:
            return ratio * 0.5, VerificationResult.PARTIAL
        return 0.0, VerificationResult.INCORRECT


# ============================================================================
# CompositeVerifier
# ============================================================================


class CompositeVerifier(RewardVerifier):
    """Combines multiple verifiers with configurable weights and
    curriculum-based difficulty scheduling.

    The composite reward for a single (prompt, response, reference)
    triple is:

        R = Σ_i  w_i · r_i

    where ``w_i`` is the weight for verifier *i* (normalised to sum
    to 1) and ``r_i`` is the reward from verifier *i*.

    When curriculum scheduling is enabled, the difficulty level
    progresses automatically:

        EASY → MEDIUM → HARD

    over ``curriculum_warmup_steps`` training steps, following the
    approach from NeurIPS 2025 Poster 116633.

    Args:
        verifiers: List of ``RewardVerifier`` instances.
        weights: Per-verifier weights.  If None, equal weights are
            used (proportional to each verifier's ``weight`` attribute).
        use_curriculum: Whether to use difficulty scheduling.
            Default True.
        curriculum_warmup_steps: Steps for EASY → HARD progression.
            Default 1000.
        initial_difficulty: Starting difficulty.  Default EASY.
        final_difficulty: Final difficulty.  Default HARD.
        partial_credit_weight: Weight for PARTIAL results.  Default 0.5.
        error_penalty: Reward for ERROR results.  Default -1.0.
    """

    def __init__(
        self,
        verifiers: List[RewardVerifier],
        weights: Optional[List[float]] = None,
        use_curriculum: bool = True,
        curriculum_warmup_steps: int = 1000,
        initial_difficulty: VerificationDifficulty = VerificationDifficulty.EASY,
        final_difficulty: VerificationDifficulty = VerificationDifficulty.HARD,
        partial_credit_weight: float = 0.5,
        error_penalty: float = -1.0,
    ) -> None:
        super().__init__(name="composite", weight=1.0)
        self.verifiers = verifiers
        self.use_curriculum = use_curriculum
        self.curriculum_warmup_steps = curriculum_warmup_steps
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.partial_credit_weight = partial_credit_weight
        self.error_penalty = error_penalty

        # Normalise weights
        if weights is not None:
            if len(weights) != len(verifiers):
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"verifiers length ({len(verifiers)})"
                )
            raw_weights = weights
        else:
            raw_weights = [v.weight for v in verifiers]

        total = sum(raw_weights)
        self.weights = [w / total for w in raw_weights] if total > 0 else [1.0 / len(verifiers)] * len(verifiers)

        # Training step counter for curriculum
        self._step_count: int = 0

    def get_current_difficulty(self) -> VerificationDifficulty:
        """Get the current verification difficulty based on curriculum.

        Returns:
            Current ``VerificationDifficulty`` level.
        """
        if not self.use_curriculum:
            return self.final_difficulty

        # Linear interpolation from initial to final difficulty
        difficulty_order = [
            VerificationDifficulty.EASY,
            VerificationDifficulty.MEDIUM,
            VerificationDifficulty.HARD,
        ]
        initial_idx = difficulty_order.index(self.initial_difficulty)
        final_idx = difficulty_order.index(self.final_difficulty)

        progress = min(self._step_count / max(self.curriculum_warmup_steps, 1), 1.0)
        target_idx = initial_idx + progress * (final_idx - initial_idx)
        current_idx = min(int(round(target_idx)), len(difficulty_order) - 1)

        return difficulty_order[current_idx]

    def step(self) -> None:
        """Advance the curriculum by one step.

        Call this after each training step to update the difficulty.
        """
        self._step_count += 1

    def verify(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        difficulty: Optional[VerificationDifficulty] = None,
    ) -> Tuple[float, VerificationResult]:
        """Verify using all sub-verifiers and combine rewards.

        Args:
            prompt: Input prompt string.
            response: Model's response string.
            reference: Optional reference answer.
            difficulty: Override difficulty level.  If None, uses the
                current curriculum difficulty.

        Returns:
            Tuple ``(combined_reward, aggregate_result)``.
        """
        current_difficulty = difficulty or self.get_current_difficulty()

        total_reward = 0.0
        results: List[VerificationResult] = []
        per_verifier_rewards: Dict[str, float] = {}

        for verifier, weight in zip(self.verifiers, self.weights):
            reward, result = verifier.verify(
                prompt, response, reference, current_difficulty
            )

            # Apply partial credit / error penalty
            if result == VerificationResult.PARTIAL:
                reward *= self.partial_credit_weight
            elif result == VerificationResult.ERROR:
                reward = self.error_penalty

            total_reward += weight * reward
            results.append(result)
            per_verifier_rewards[verifier.name] = reward

        # Determine aggregate result
        if all(r == VerificationResult.CORRECT for r in results):
            aggregate_result = VerificationResult.CORRECT
        elif any(r == VerificationResult.ERROR for r in results):
            aggregate_result = VerificationResult.ERROR
        elif all(r == VerificationResult.INCORRECT for r in results):
            aggregate_result = VerificationResult.INCORRECT
        else:
            aggregate_result = VerificationResult.PARTIAL

        return total_reward, aggregate_result

    def verify_batch(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[Optional[str]]] = None,
        difficulty: Optional[VerificationDifficulty] = None,
    ) -> Tuple[List[float], List[VerificationResult]]:
        """Verify a batch of responses using all sub-verifiers.

        Args:
            prompts: List of prompt strings.
            responses: List of response strings.
            references: Optional list of reference answers.
            difficulty: Override difficulty level.

        Returns:
            Tuple ``(rewards, results)`` — lists of floats and
            VerificationResult values.
        """
        current_difficulty = difficulty or self.get_current_difficulty()

        rewards: List[float] = []
        results: List[VerificationResult] = []

        for i, response in enumerate(responses):
            ref = references[i] if references is not None else None
            prompt = prompts[i] if i < len(prompts) else ""
            reward, result = self.verify(prompt, response, ref, current_difficulty)
            rewards.append(reward)
            results.append(result)

        return rewards, results


# ============================================================================
# RLVRTrainer
# ============================================================================


class RLVRTrainer:
    """Reinforcement Learning with Verifiable Rewards — main trainer.

    Integrates verifiable rewards from ``CompositeVerifier`` with an RL
    training loop, supporting both DAPO and GRPO as the underlying
    policy optimisation method.

    The trainer replaces the default reward function in DAPO/GRPO with
    verifiable rewards, ensuring that the reward signal is objective
    and noise-free.  It also manages curriculum scheduling by advancing
    the difficulty level of the ``CompositeVerifier`` after each step.

    Example
    -------
    >>> math_verifier = MathVerifier()
    >>> format_verifier = FormatVerifier(
    ...     required_patterns=[r"Step \\d+:"],
    ... )
    >>> config = RLVRConfig(
    ...     verifiers=[math_verifier, format_verifier],
    ...     verifier_weights=[0.7, 0.3],
    ...     use_curriculum=True,
    ...     curriculum_warmup_steps=500,
    ...     rl_method="dapo",
    ... )
    >>> trainer = RLVRTrainer(config, policy_model)
    >>> metrics = trainer.train_step(prompts, references)

    Args:
        config: ``RLVRConfig`` with training hyperparameters.
        policy_model: The policy model being optimized.
        reference_model: Optional reference model for KL penalty.
            If None and ``kl_coefficient > 0``, a deep copy is made.
        tokenizer: Optional tokenizer for encoding/decoding.  If None,
            responses must be provided as strings.
    """

    def __init__(
        self,
        config: RLVRConfig,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.policy_model = policy_model
        self.tokenizer = tokenizer

        # ---- Build CompositeVerifier ----
        self.composite_verifier = CompositeVerifier(
            verifiers=config.verifiers,
            weights=config.verifier_weights,
            use_curriculum=config.use_curriculum,
            curriculum_warmup_steps=config.curriculum_warmup_steps,
            initial_difficulty=config.initial_difficulty,
            final_difficulty=config.final_difficulty,
            partial_credit_weight=config.partial_credit_weight,
            error_penalty=config.error_penalty,
        )

        # ---- Build underlying RL trainer ----
        self._rl_trainer = self._build_rl_trainer(
            config, policy_model, reference_model
        )

        # ---- Training statistics ----
        self._step_count: int = 0
        self._metrics_history: List[Dict[str, float]] = []

    def _build_rl_trainer(
        self,
        config: RLVRConfig,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module],
    ) -> Any:
        """Build the underlying DAPO or GRPO trainer.

        Args:
            config: RLVRConfig.
            policy_model: Policy model.
            reference_model: Optional reference model.

        Returns:
            DAPOTrainer or GRPOTrainer instance.
        """
        # Use the RLVR reward function as the reward_fn
        reward_fn = self._rlvr_reward_fn

        if config.rl_method == "dapo":
            from losion.training.dapo import DAPOTrainer, DAPOConfig

            dapo_config = DAPOConfig(
                clip_ratio_low=config.clip_ratio_low,
                clip_ratio_high=config.clip_ratio_high,
                num_responses_per_prompt=config.num_responses_per_prompt,
                kl_coefficient=config.kl_coefficient,
                entropy_coefficient=config.entropy_coefficient,
                learning_rate=config.learning_rate,
                max_grad_norm=config.max_grad_norm,
                temperature=config.temperature,
                max_response_length=config.max_response_length,
            )
            return DAPOTrainer(
                config=dapo_config,
                policy_model=policy_model,
                reference_model=reference_model,
                reward_fn=reward_fn,
            )
        elif config.rl_method == "grpo":
            from losion.training.grpo import GRPOTrainer

            return GRPOTrainer(
                model=policy_model,
                reward_fn=reward_fn,
            )
        else:
            raise ValueError(
                f"Unsupported rl_method: {config.rl_method}"
            )

    def _rlvr_reward_fn(
        self,
        responses: List[str],
        prompts: Optional[List[str]] = None,
        reference_answers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute verifiable rewards for a list of responses.

        This function is passed to the underlying DAPO/GRPO trainer
        as the ``reward_fn``.

        Args:
            responses: List of response strings.
            prompts: Optional list of prompt strings.
            reference_answers: Optional list of reference answers.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Reward tensor of shape ``[len(responses)]``.
        """
        prompt_list = prompts or [""] * len(responses)

        # Convert reference_answers to Optional[str] list
        ref_list: List[Optional[str]] = (
            list(reference_answers) if reference_answers else [None] * len(responses)
        )

        rewards, results = self.composite_verifier.verify_batch(
            prompts=prompt_list,
            responses=responses,
            references=ref_list,
        )

        if self.config.log_verifier_details:
            # Log per-verifier details
            current_difficulty = self.composite_verifier.get_current_difficulty()
            result_counts = {}
            for r in results:
                result_counts[r.value] = result_counts.get(r.value, 0) + 1
            logger.info(
                "RLVR reward: mean=%.4f, difficulty=%s, results=%s",
                sum(rewards) / max(len(rewards), 1),
                current_difficulty.value,
                result_counts,
            )

        return torch.tensor(rewards, dtype=torch.float32)

    def train_step(
        self,
        prompts: Union[torch.Tensor, List[str]],
        references: Optional[List[Optional[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Execute a single RLVR training step.

        Generates responses, computes verifiable rewards, and updates
        the policy using the underlying RL method (DAPO or GRPO).

        Args:
            prompts: Either a tensor of token IDs or a list of prompt
                strings.
            references: Optional list of reference answers for
                verification.
            attention_mask: Optional attention mask (only used when
                *prompts* is a tensor).

        Returns:
            Dictionary of training metrics.
        """
        # Store references for the reward function
        self._current_references = references

        # Advance curriculum
        self.composite_verifier.step()
        self._step_count += 1

        # Delegate to the underlying RL trainer
        if hasattr(self._rl_trainer, "train_step"):
            metrics = self._rl_trainer.train_step(
                prompts, attention_mask
            )
        else:
            raise RuntimeError(
                "Underlying RL trainer does not have a train_step method"
            )

        # Add RLVR-specific metrics
        current_difficulty = self.composite_verifier.get_current_difficulty()
        metrics["rlvr/difficulty"] = float(
            {"easy": 0.0, "medium": 1.0, "hard": 2.0}[
                current_difficulty.value
            ]
        )
        metrics["rlvr/step"] = float(self._step_count)
        metrics["rlvr/curriculum_progress"] = min(
            self._step_count / max(self.config.curriculum_warmup_steps, 1),
            1.0,
        )

        self._metrics_history.append(metrics)

        return metrics

    def compute_verifiable_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[Optional[str]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute verifiable rewards without training.

        Useful for evaluation and debugging.

        Args:
            prompts: List of prompt strings.
            responses: List of response strings.
            references: Optional list of reference answers.

        Returns:
            Tuple ``(rewards, details)`` where:
            - ``rewards``: ``[len(responses)]`` tensor.
            - ``details``: Dict with per-verifier rewards and results.
        """
        ref_list = references or [None] * len(responses)
        rewards, results = self.composite_verifier.verify_batch(
            prompts=prompts,
            responses=responses,
            references=ref_list,
        )

        # Collect per-verifier details
        current_difficulty = self.composite_verifier.get_current_difficulty()
        per_verifier_details: Dict[str, List[float]] = {}
        for verifier in self.composite_verifier.verifiers:
            verifier_rewards = []
            for i, response in enumerate(responses):
                ref = ref_list[i]
                prompt = prompts[i] if i < len(prompts) else ""
                reward, _ = verifier.verify(
                    prompt, response, ref, current_difficulty
                )
                verifier_rewards.append(reward)
            per_verifier_details[verifier.name] = verifier_rewards

        details = {
            "difficulty": current_difficulty.value,
            "results": [r.value for r in results],
            "per_verifier_rewards": per_verifier_details,
        }

        return torch.tensor(rewards, dtype=torch.float32), details

    def get_curriculum_progress(self) -> float:
        """Get the current curriculum progress (0.0 to 1.0).

        Returns:
            Float in [0, 1] indicating how far through the curriculum
            the trainer has progressed.
        """
        return min(
            self._step_count / max(self.config.curriculum_warmup_steps, 1),
            1.0,
        )

    def get_current_difficulty(self) -> VerificationDifficulty:
        """Get the current verification difficulty level.

        Returns:
            Current ``VerificationDifficulty``.
        """
        return self.composite_verifier.get_current_difficulty()

    def set_difficulty(
        self, difficulty: VerificationDifficulty
    ) -> None:
        """Manually override the curriculum difficulty.

        This disables automatic curriculum progression.

        Args:
            difficulty: Target difficulty level.
        """
        self.composite_verifier.use_curriculum = False
        self.composite_verifier.initial_difficulty = difficulty
        self.composite_verifier.final_difficulty = difficulty

    def train_epoch(
        self,
        prompt_dataloader: Any,
        references: Optional[List[List[Optional[str]]]] = None,
        num_steps: int = 100,
    ) -> List[Dict[str, float]]:
        """Train for one epoch using RLVR.

        Args:
            prompt_dataloader: DataLoader that yields prompts.
            references: Optional per-step reference answers.
            num_steps: Maximum number of training steps.

        Returns:
            List of metrics dictionaries, one per step.
        """
        all_metrics: List[Dict[str, float]] = []

        for step, batch in enumerate(prompt_dataloader):
            if step >= num_steps:
                break

            prompts = batch.get("input_ids", batch.get("prompts", batch))
            attn_mask = batch.get("attention_mask", None)

            step_refs = None
            if references is not None and step < len(references):
                step_refs = references[step]

            metrics = self.train_step(
                prompts=prompts,
                references=step_refs,
                attention_mask=attn_mask,
            )
            all_metrics.append(metrics)

            if step % 10 == 0:
                log_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in metrics.items()
                )
                logger.info(f"RLVR Step {step}: {log_str}")

        return all_metrics
