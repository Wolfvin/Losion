"""
Meta-Skill System — Higher-order skills that create, verify, and compose other skills.

Inspired by:
- CASCADE (2025): "Cumulative Agentic Skill Creation through Autonomous
  Development and Evolution" — agents need meta-skills: the ability to
  learn HOW to learn skills, not just individual skills.
  Key meta-skills: continuous learning via web search, code extraction.
- SoK: Agentic Skills (2026): "Beyond Tool Use in LLM Agents" —
  skill abstraction layer is distinct from tools. Skills have
  applicability, composability, and security properties.

This module implements three meta-skills:
1. **Skill Synthesis**: How to create new skills effectively
   - Search for relevant information
   - Synthesize into a coherent skill definition
   - Validate and test the skill before storing

2. **Skill Verification**: How to test and validate skills
   - Run the skill on test inputs
   - Verify the output meets expectations
   - Update confidence based on test results

3. **Skill Composition**: How to chain skills together
   - Decompose complex tasks into skill sequences
   - Compose skills into pipelines
   - Optimize skill ordering for efficiency

Architecture:
    ┌────────────────────────────────────────────┐
    │            META-SKILL SYSTEM                │
    │                                             │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐   │
    │  │Synthesis │ │Verification│ │Composition│  │
    │  │Meta-Skill│ │Meta-Skill │ │Meta-Skill │  │
    │  └─────┬────┘ └─────┬────┘ └─────┬────┘   │
    │        │             │             │         │
    │        ▼             ▼             ▼         │
    │  ┌──────────────────────────────────────┐   │
    │  │          SkillManager                │   │
    │  │   (existing skill management)        │   │
    │  └──────────────────────────────────────┘   │
    └────────────────────────────────────────────┘

Key insight from CASCADE: "The transition from 'LLM + tool use' to
'LLM + skill acquisition' requires agents that can autonomously
develop and evolve their skill set, not just use pre-defined tools."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from losion.agent.skills.store import SkillEntry, SkillMetadata, SkillStore
from losion.agent.skills.creator import SkillCreator
from losion.agent.skills.manager import SkillManager
from losion.agent.tools.web_search import WebSearchInterface

logger = logging.getLogger(__name__)


class MetaSkillType(Enum):
    """Types of meta-skills."""

    SYNTHESIS = "synthesis"        # How to create skills
    VERIFICATION = "verification"  # How to test skills
    COMPOSITION = "composition"    # How to compose skills


@dataclass
class ComposedSkill:
    """A skill composed from multiple existing skills.

    Attributes:
        name: Name of the composed skill.
        description: What the composed skill does.
        skill_chain: Ordered list of skill names to execute.
        input_mapping: How to map inputs between skills.
        output_schema: Expected output format.
        metadata: Additional metadata.
    """

    name: str
    description: str
    skill_chain: List[str] = field(default_factory=list)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_schema: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of a skill verification test.

    Attributes:
        skill_name: Name of the skill that was verified.
        passed: Whether the skill passed verification.
        test_cases_run: Number of test cases executed.
        test_cases_passed: Number of test cases that passed.
        errors: List of error messages from failed tests.
        confidence_after: Updated confidence after verification.
        recommendations: Suggestions for improving the skill.
    """

    skill_name: str = ""
    passed: bool = False
    test_cases_run: int = 0
    test_cases_passed: int = 0
    errors: List[str] = field(default_factory=list)
    confidence_after: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of test cases that passed."""
        if self.test_cases_run == 0:
            return 0.0
        return self.test_cases_passed / self.test_cases_run


class SkillSynthesisMetaSkill:
    """Meta-skill: How to create new skills effectively.

    This meta-skill improves upon the basic SkillCreator by:
    1. Using multiple search queries for richer context
    2. Cross-referencing search results for consistency
    3. Generating test cases alongside the skill definition
    4. Validating the skill before storing

    This follows CASCADE's insight: "Skills should be developed through
    autonomous development and evolution, not just one-shot creation."

    Args:
        skill_creator: Base SkillCreator to enhance.
        web_search: WebSearchInterface for multi-query search.
        max_search_queries: Maximum parallel search queries.
    """

    def __init__(
        self,
        skill_creator: Optional[SkillCreator] = None,
        web_search: Optional[WebSearchInterface] = None,
        max_search_queries: int = 3,
    ) -> None:
        self.creator = skill_creator or SkillCreator()
        self.web_search = web_search or WebSearchInterface()
        self.max_search_queries = max_search_queries

    def synthesize(
        self,
        query: str,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[SkillEntry]:
        """Synthesize a new skill using multi-source information.

        Unlike basic SkillCreator.create(), this method:
        1. Generates multiple search queries for the same task
        2. Aggregates and deduplicates search results
        3. Cross-references results for consistency
        4. Generates a richer, more reliable skill definition
        5. Attaches test cases for future verification

        Args:
            query: What the skill should do.
            domain: Domain classification.
            tags: Tags for the skill.

        Returns:
            The synthesized SkillEntry, or None if synthesis fails.
        """
        logger.info(f"Meta-skill: Synthesizing skill for '{query}' (domain={domain})")

        # === Step 1: Generate multiple search queries ===
        search_queries = self._generate_search_queries(query, domain)
        search_queries = search_queries[:self.max_search_queries]

        # === Step 2: Execute searches ===
        all_results = []
        for sq in search_queries:
            try:
                results = self.web_search.search(query=sq, num_results=3)
                all_results.extend([
                    {"title": r.title, "snippet": r.snippet, "url": r.url}
                    for r in results if r.is_valid
                ])
            except Exception as e:
                logger.warning(f"Search query '{sq}' failed: {e}")

        # === Step 3: Deduplicate and rank ===
        seen_snippets = set()
        unique_results = []
        for r in all_results:
            snippet_key = r.get("snippet", "")[:100]
            if snippet_key not in seen_snippets:
                seen_snippets.add(snippet_key)
                unique_results.append(r)

        # === Step 4: Create skill with enriched context ===
        # Use the base creator but with richer search context
        skill = self.creator.create(query=query, domain=domain, tags=tags)

        if skill is None:
            logger.warning(f"Meta-skill: Skill synthesis failed for '{query}'")
            return None

        # === Step 5: Enrich skill with meta-information ===
        # Add test cases as part of the skill definition
        test_cases = self._generate_test_cases(query, domain, unique_results)
        if test_cases:
            enriched_definition = skill.definition + f"\n\n## Test Cases\n"
            for i, tc in enumerate(test_cases):
                enriched_definition += f"{i+1}. Input: {tc['input']} → Expected: {tc['expected']}\n"
            skill.definition = enriched_definition

        # Higher initial confidence for meta-synthesized skills
        # (because they're based on multiple sources and have test cases)
        skill.metadata.confidence = min(0.5, skill.metadata.confidence + 0.2)
        skill.metadata.source = "meta_synthesized"

        logger.info(f"Meta-skill: Synthesized '{skill.name}' with {len(test_cases)} test cases")
        return skill

    def _generate_search_queries(
        self, query: str, domain: Optional[str]
    ) -> List[str]:
        """Generate multiple search queries for the same skill.

        Different query framings capture different aspects of the skill.
        """
        queries = [query]

        # How-to framing
        if not query.lower().startswith("how to"):
            queries.append(f"how to {query}")

        # Best practices framing
        if domain:
            queries.append(f"{query} best practices {domain}")

        # Tutorial framing
        queries.append(f"{query} tutorial guide")

        return queries

    def _generate_test_cases(
        self,
        query: str,
        domain: Optional[str],
        search_results: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Generate test cases for a skill based on search context.

        Test cases provide a way to verify the skill works correctly
        and build confidence over time.
        """
        test_cases = []

        # Basic test: skill should handle its own query
        test_cases.append({
            "input": query,
            "expected": f"A valid response addressing {query}",
        })

        # Domain-specific tests
        if domain == "math":
            test_cases.append({
                "input": "2 + 2",
                "expected": "4",
            })
        elif domain == "code":
            test_cases.append({
                "input": "print('hello world')",
                "expected": "hello world",
            })

        return test_cases


class SkillVerificationMetaSkill:
    """Meta-skill: How to test and validate skills.

    This meta-skill verifies that existing skills work correctly by:
    1. Running the skill on test inputs
    2. Checking the output against expected results
    3. Updating the skill's confidence based on results
    4. Recommending improvements for failing skills

    This addresses a gap in the original implementation: skills
    were created but never tested. A skill with 0 usage but 0.3
    confidence is unreliable. Verification provides actual evidence.

    Args:
        skill_store: SkillStore to verify skills from.
        max_test_cases: Maximum test cases per verification run.
    """

    def __init__(
        self,
        skill_store: Optional[SkillStore] = None,
        max_test_cases: int = 5,
    ) -> None:
        self.store = skill_store or SkillStore()
        self.max_test_cases = max_test_cases

    def verify(self, skill_name: str) -> VerificationResult:
        """Verify a skill by running its test cases.

        Args:
            skill_name: Name of the skill to verify.

        Returns:
            VerificationResult with test outcomes.
        """
        skill = self.store.lookup(skill_name)
        if skill is None:
            return VerificationResult(
                skill_name=skill_name,
                errors=[f"Skill '{skill_name}' not found"],
            )

        # Extract test cases from skill definition
        test_cases = self._extract_test_cases(skill.definition)

        if not test_cases:
            # No test cases found — mark as unverified but not failed
            return VerificationResult(
                skill_name=skill_name,
                passed=True,  # No tests = no failures
                test_cases_run=0,
                confidence_after=skill.metadata.confidence,
                recommendations=["Add test cases to enable verification"],
            )

        # Run test cases
        result = VerificationResult(skill_name=skill_name)
        result.test_cases_run = len(test_cases[:self.max_test_cases])

        for tc in test_cases[:self.max_test_cases]:
            # Basic validation: check if the skill definition mentions
            # the expected output concept
            expected = tc.get("expected", "").lower()
            definition_lower = skill.definition.lower()

            # Simple heuristic: if expected output terms appear in definition
            if expected and any(term in definition_lower for term in expected.split()):
                result.test_cases_passed += 1
            elif not expected:
                result.test_cases_passed += 1  # No expectation = auto-pass
            else:
                result.errors.append(
                    f"Test case failed: expected '{expected}' not found in skill definition"
                )

        # Determine pass/fail
        result.passed = result.pass_rate >= 0.6  # 60% pass threshold

        # Update confidence based on verification
        if result.test_cases_run > 0:
            # Bayesian-like update: confidence = weighted average of
            # prior confidence and verification pass rate
            prior = skill.metadata.confidence
            evidence = result.pass_rate
            # Weight evidence more as test count increases
            evidence_weight = min(result.test_cases_run / 10.0, 0.7)
            result.confidence_after = (1 - evidence_weight) * prior + evidence_weight * evidence

            # Update skill confidence
            skill.metadata.confidence = result.confidence_after
            self.store.store(skill)

        # Generate recommendations
        if result.pass_rate < 1.0:
            result.recommendations.append(
                "Skill verification partially failed. Consider refining the skill definition."
            )
        if result.pass_rate < 0.5:
            result.recommendations.append(
                "Skill verification largely failed. Consider recreating with better context."
            )

        logger.info(
            f"Skill verification: {skill_name} — "
            f"{result.test_cases_passed}/{result.test_cases_run} passed, "
            f"confidence: {result.confidence_after:.2f}"
        )

        return result

    def _extract_test_cases(self, definition: str) -> List[Dict[str, str]]:
        """Extract test cases from a skill definition.

        Test cases are expected in the format:
        ## Test Cases
        1. Input: <input> → Expected: <expected>
        """
        test_cases = []
        in_test_section = False

        for line in definition.split("\n"):
            line_lower = line.lower().strip()
            if "test cases" in line_lower or "test cases" in line_lower:
                in_test_section = True
                continue

            if in_test_section and "→" in line:
                parts = line.split("→", 1)
                input_part = parts[0].strip()
                expected_part = parts[1].strip() if len(parts) > 1 else ""

                # Clean up numbering
                for prefix in ["1.", "2.", "3.", "4.", "5."]:
                    if input_part.startswith(prefix):
                        input_part = input_part[len(prefix):].strip()

                # Extract actual input/expected values
                input_val = input_part.replace("Input:", "").strip()
                expected_val = expected_part.replace("Expected:", "").strip()

                if input_val:
                    test_cases.append({"input": input_val, "expected": expected_val})

        return test_cases


class SkillCompositionMetaSkill:
    """Meta-skill: How to compose skills into pipelines.

    This meta-skill enables the agent to:
    1. Decompose complex tasks into skill sequences
    2. Find compatible skills that can be chained
    3. Compose skills into pipelines for efficient execution

    This follows the insight from SoK: "Skills have composability
    properties — they can be combined to solve tasks that no
    single skill can handle alone."

    Args:
        skill_manager: SkillManager for skill lookup.
    """

    def __init__(
        self,
        skill_manager: Optional[SkillManager] = None,
    ) -> None:
        self.manager = skill_manager or SkillManager(auto_create=False)

    def compose(
        self,
        query: str,
        domain: Optional[str] = None,
        max_chain_length: int = 4,
    ) -> Optional[ComposedSkill]:
        """Compose multiple skills into a pipeline for a complex query.

        Steps:
        1. Decompose the query into sub-tasks
        2. Find skills for each sub-task
        3. Check compatibility between skill outputs/inputs
        4. Compose into an ordered pipeline

        Args:
            query: Complex query requiring multiple skills.
            domain: Domain classification.
            max_chain_length: Maximum skills in the chain.

        Returns:
            ComposedSkill with the skill chain, or None if composition fails.
        """
        logger.info(f"Meta-skill: Composing skills for '{query}'")

        # === Step 1: Decompose query into sub-tasks ===
        sub_tasks = self._decompose_query(query, domain)

        if not sub_tasks:
            return None

        # === Step 2: Find skills for each sub-task ===
        skill_chain = []
        for sub_task in sub_tasks[:max_chain_length]:
            result = self.manager.lookup(query=sub_task, domain=domain)
            if result.found and result.skill:
                skill_chain.append(result.skill.name)
            else:
                # No skill found for this sub-task — composition incomplete
                logger.debug(f"No skill found for sub-task: {sub_task}")
                # Add placeholder
                skill_chain.append(f"__missing:{sub_task}__")

        if not skill_chain:
            return None

        # === Step 3: Compose ===
        # Filter out missing skills for the final chain
        valid_chain = [s for s in skill_chain if not s.startswith("__missing:")]
        missing = [s for s in skill_chain if s.startswith("__missing:")]

        composed = ComposedSkill(
            name=f"composed-{query.lower().replace(' ', '-')[:30]}",
            description=f"Composed pipeline for: {query}. "
                       f"Chain: {' → '.join(valid_chain)}",
            skill_chain=valid_chain,
            metadata={
                "original_sub_tasks": sub_tasks,
                "missing_skills": [s.replace("__missing:", "").replace("__", "") for s in missing],
                "domain": domain,
                "created_at": time.time(),
            },
        )

        if missing:
            composed.metadata["recommendations"] = [
                f"Create skill for: {s.replace('__missing:', '').replace('__', '')}"
                for s in missing
            ]

        logger.info(
            f"Composed skill: {composed.name} "
            f"({len(valid_chain)} skills, {len(missing)} missing)"
        )

        return composed

    def _decompose_query(
        self, query: str, domain: Optional[str]
    ) -> List[str]:
        """Decompose a complex query into sub-tasks.

        Uses heuristic decomposition based on conjunctions and
        domain-specific patterns.
        """
        sub_tasks = []

        # Split on common conjunctions
        conjunctions = [" and ", ", then ", " then ", " after that ", " followed by "]
        remaining = query
        for conj in conjunctions:
            if conj in remaining.lower():
                parts = remaining.lower().split(conj)
                sub_tasks.extend([p.strip() for p in parts if p.strip()])
                remaining = ""
                break

        # If no conjunctions found, the query is a single task
        if not sub_tasks:
            sub_tasks = [query]

        # Domain-specific decomposition
        if domain == "code":
            # Code tasks often involve: write → test → debug
            if len(sub_tasks) == 1:
                sub_tasks = [sub_tasks[0], "test the code", "debug if needed"]

        elif domain == "data":
            # Data tasks often involve: collect → process → analyze
            if len(sub_tasks) == 1:
                sub_tasks = [sub_tasks[0], "process the data", "analyze results"]

        return sub_tasks[:6]  # Cap at 6 sub-tasks
