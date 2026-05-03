"""
Skill Creator — Auto-generate skills from web search and synthesis.

When the agent encounters a task for which no suitable skill exists,
the SkillCreator generates one by:
1. Searching the web for relevant information
2. Synthesizing the results into a skill definition
3. Storing the new skill for future use

This is the "build-in skill check → web search → create skill" pipeline
that enables Losion to continuously expand its capabilities.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from losion.agent.skills.store import SkillEntry, SkillMetadata, SkillStore
from losion.agent.tools.web_search import WebSearchInterface

logger = logging.getLogger(__name__)


class SkillCreator:
    """Auto-creates skills from web search results.

    The creation process follows a structured pipeline:
    1. Analyze: Determine what kind of skill is needed
    2. Search: Query the web for relevant information
    3. Synthesize: Combine search results into a skill definition
    4. Validate: Check that the skill is well-formed
    5. Store: Persist the skill for future use

    Skill types that can be auto-created:
    - "prompt": A prompt template for a specific task type
    - "code": A code snippet/function for a specific computation
    - "search_strategy": A strategy for how to search for information
    - "pipeline": A multi-step procedure combining multiple actions

    Args:
        store: SkillStore to save created skills.
        web_search: WebSearchInterface for information retrieval.
        default_skill_type: Default type for auto-created skills.
        max_search_results: Maximum web search results to consider.
        min_relevance_score: Minimum relevance for search results.
    """

    def __init__(
        self,
        store: Optional[SkillStore] = None,
        web_search: Optional[WebSearchInterface] = None,
        default_skill_type: str = "prompt",
        max_search_results: int = 5,
        min_relevance_score: float = 0.3,
    ) -> None:
        self.store = store or SkillStore()
        self.web_search = web_search or WebSearchInterface()
        self.default_skill_type = default_skill_type
        self.max_search_results = max_search_results
        self.min_relevance_score = min_relevance_score

    def create(
        self,
        query: str,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        skill_type: Optional[str] = None,
    ) -> Optional[SkillEntry]:
        """Create a new skill for the given query.

        Args:
            query: What the skill should do.
            domain: Domain classification.
            tags: Tags for the skill.
            skill_type: Override skill type.

        Returns:
            The created SkillEntry, or None if creation failed.
        """
        logger.info(f"Creating skill for: {query} (domain={domain})")

        # === Step 1: Analyze ===
        skill_type = skill_type or self._determine_skill_type(query, domain)
        search_query = self._build_search_query(query, domain)

        # === Step 2: Search ===
        search_results = self._search_for_context(search_query)

        # === Step 3: Synthesize ===
        definition = self._synthesize_skill(query, skill_type, search_results, domain)

        if not definition:
            logger.warning(f"Failed to synthesize skill for: {query}")
            return None

        # === Step 4: Validate ===
        if not self._validate_skill(definition, skill_type):
            logger.warning(f"Skill validation failed for: {query}")
            return None

        # === Step 5: Store ===
        skill_name = self._generate_skill_name(query, domain)
        description = self._generate_description(query, domain, search_results)

        entry = SkillEntry(
            name=skill_name,
            description=description,
            skill_type=skill_type,
            definition=definition,
            metadata=SkillMetadata(
                source="auto_created",
                domain=domain,
                tags=tags or self._extract_tags(query, domain),
                confidence=0.3,  # Low initial confidence for auto-created skills
            ),
            inputs=self._infer_inputs(query, skill_type),
            outputs=self._infer_outputs(query, skill_type),
        )

        self.store.store(entry)
        logger.info(f"Skill created and stored: {skill_name}")

        return entry

    def _determine_skill_type(self, query: str, domain: Optional[str]) -> str:
        """Determine the appropriate skill type based on query and domain.

        Args:
            query: The skill query.
            domain: Domain classification.

        Returns:
            Skill type string.
        """
        query_lower = query.lower()

        # Code-related queries → code skill
        code_keywords = ["function", "implement", "code", "script", "program", "algorithm"]
        if any(kw in query_lower for kw in code_keywords) or domain in ("code", "data"):
            return "code"

        # Search-related queries → search strategy
        search_keywords = ["find", "search", "lookup", "research", "investigate"]
        if any(kw in query_lower for kw in search_keywords):
            return "search_strategy"

        # Multi-step queries → pipeline
        pipeline_keywords = ["pipeline", "workflow", "process", "step-by-step", "automate"]
        if any(kw in query_lower for kw in pipeline_keywords):
            return "pipeline"

        return self.default_skill_type

    def _build_search_query(self, query: str, domain: Optional[str]) -> str:
        """Build a web search query from the skill query.

        Adds domain context and frames the query to find
        actionable information.

        Args:
            query: The skill query.
            domain: Domain classification.

        Returns:
            Search query string.
        """
        parts = [query]

        if domain:
            parts.append(f"in {domain}")

        # Add "how to" framing for actionable results
        if not any(query.lower().startswith(p) for p in ["how to", "what is", "explain"]):
            parts.insert(0, "how to")

        return " ".join(parts)

    def _search_for_context(self, search_query: str) -> List[Dict[str, Any]]:
        """Search the web for context about the skill.

        Args:
            search_query: The search query.

        Returns:
            List of search result dictionaries.
        """
        try:
            results = self.web_search.search(
                query=search_query,
                num_results=self.max_search_results,
            )
            return [
                {
                    "title": r.name,
                    "snippet": r.snippet,
                    "url": r.url,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Web search failed during skill creation: {e}")
            return []

    def _synthesize_skill(
        self,
        query: str,
        skill_type: str,
        search_results: List[Dict[str, Any]],
        domain: Optional[str],
    ) -> str:
        """Synthesize a skill definition from search results.

        This method creates a structured skill definition by combining
        information from web search results. The format depends on the
        skill type:

        - "prompt": Creates a prompt template with placeholders
        - "code": Creates a code snippet with the search-derived logic
        - "search_strategy": Creates a search procedure
        - "pipeline": Creates a multi-step pipeline

        Args:
            query: Original skill query.
            skill_type: Type of skill to create.
            search_results: Web search results for context.
            domain: Domain classification.

        Returns:
            Skill definition string, or empty string if synthesis fails.
        """
        # Extract key information from search results
        context_parts = []
        for result in search_results:
            if result.get("snippet"):
                context_parts.append(result["snippet"])

        context = "\n".join(context_parts)

        if skill_type == "prompt":
            return self._synthesize_prompt(query, context, domain)
        elif skill_type == "code":
            return self._synthesize_code(query, context, domain)
        elif skill_type == "search_strategy":
            return self._synthesize_search_strategy(query, context, domain)
        elif skill_type == "pipeline":
            return self._synthesize_pipeline(query, context, domain)
        else:
            return self._synthesize_prompt(query, context, domain)

    def _synthesize_prompt(
        self, query: str, context: str, domain: Optional[str]
    ) -> str:
        """Synthesize a prompt-type skill.

        Creates a reusable prompt template with placeholders
        for variable inputs.
        """
        domain_str = f" in the domain of {domain}" if domain else ""
        template = (
            f"# Skill: {query}\n"
            f"# Domain: {domain or 'general'}\n"
            f"# Source: auto-created from web context\n"
            f"# Context:\n"
            f"# {context[:500]}\n\n"
            f"Given a task related to {query}{domain_str}, follow this approach:\n\n"
            f"1. Analyze the input to identify key requirements\n"
            f"2. Apply relevant knowledge from the context above\n"
            f"3. Generate a structured response addressing all requirements\n"
            f"4. Verify the output for correctness and completeness\n\n"
            f"Input: {{input}}\n"
            f"Output: "
        )
        return template

    def _synthesize_code(
        self, query: str, context: str, domain: Optional[str]
    ) -> str:
        """Synthesize a code-type skill."""
        domain_str = f"# Domain: {domain}\n" if domain else ""
        template = (
            f"# Code Skill: {query}\n"
            f"{domain_str}"
            f"# Source: auto-created from web context\n"
            f"# Context snippets:\n"
        )
        for line in context.split("\n")[:5]:
            template += f"# {line}\n"

        template += (
            f"\ndef skill_{query.lower().replace(' ', '_').replace('-', '_')}(input_data):\n"
            f"    \"\"\"Auto-generated skill for: {query}\n\n"
            f"    Args:\n"
            f"        input_data: Input for the skill.\n\n"
            f"    Returns:\n"
            f"        Result of applying this skill.\n"
            f"    \"\"\"\n"
            f"    # TODO: Implement based on context above\n"
            f"    # This is a scaffold — refine based on actual requirements\n"
            f"    result = input_data\n"
            f"    return result\n"
        )
        return template

    def _synthesize_search_strategy(
        self, query: str, context: str, domain: Optional[str]
    ) -> str:
        """Synthesize a search strategy skill."""
        domain_str = f" in {domain}" if domain else ""
        template = (
            f"# Search Strategy: {query}\n"
            f"# Domain: {domain or 'general'}\n\n"
            f"## Objective\n"
            f"Find information about {query}{domain_str}\n\n"
            f"## Search Steps\n"
            f"1. Primary query: \"{query}\"\n"
        )
        if domain:
            template += f"2. Domain-specific query: \"{query} {domain} best practices\"\n"
        template += (
            f"3. Broaden if needed: \"{query} tutorial guide\"\n"
            f"4. Narrow if too many results: \"{query} specific implementation\"\n\n"
            f"## Evaluation Criteria\n"
            f"- Relevance to the query\n"
            f"- Recency of information\n"
            f"- Source credibility\n"
            f"- Actionability of content\n\n"
            f"## Context from Previous Searches\n"
            f"{context[:500]}\n"
        )
        return template

    def _synthesize_pipeline(
        self, query: str, context: str, domain: Optional[str]
    ) -> str:
        """Synthesize a pipeline-type skill."""
        template = (
            f"# Pipeline: {query}\n"
            f"# Domain: {domain or 'general'}\n\n"
            f"## Steps\n"
            f"1. **Analyze**: Parse and understand the input for {query}\n"
            f"2. **Search**: Look up relevant information if needed\n"
            f"3. **Process**: Apply the core logic\n"
            f"4. **Validate**: Verify the output\n"
            f"5. **Format**: Structure the final result\n\n"
            f"## Context\n"
            f"{context[:500]}\n\n"
            f"## Input Schema\n"
            f"{{input}}\n\n"
            f"## Output Schema\n"
            f"{{output}}\n"
        )
        return template

    def _validate_skill(self, definition: str, skill_type: str) -> bool:
        """Validate that a skill definition is well-formed.

        Args:
            definition: Skill definition string.
            skill_type: Expected skill type.

        Returns:
            True if valid.
        """
        if not definition or len(definition.strip()) < 20:
            return False
        return True

    def _generate_skill_name(self, query: str, domain: Optional[str]) -> str:
        """Generate a unique skill name from the query.

        Uses kebab-case with domain prefix for namespacing.

        Args:
            query: Original query.
            domain: Domain classification.

        Returns:
            Skill name string.
        """
        # Normalize to kebab-case
        name = query.lower()
        name = name.replace(" ", "-")
        # Remove non-alphanumeric characters (keep hyphens)
        name = "".join(c for c in name if c.isalnum() or c == "-")
        # Remove consecutive hyphens
        while "--" in name:
            name = name.replace("--", "-")
        name = name.strip("-")

        # Add domain prefix
        if domain:
            name = f"{domain}-{name}"

        # Ensure uniqueness
        base_name = name
        counter = 1
        while self.store.lookup(name) is not None:
            name = f"{base_name}-v{counter}"
            counter += 1

        return name

    def _generate_description(
        self,
        query: str,
        domain: Optional[str],
        search_results: List[Dict[str, Any]],
    ) -> str:
        """Generate a human-readable description for the skill."""
        desc = f"Auto-created skill for: {query}"
        if domain:
            desc += f" (domain: {domain})"
        if search_results:
            desc += f". Based on {len(search_results)} web sources."
        return desc

    def _extract_tags(self, query: str, domain: Optional[str]) -> List[str]:
        """Extract tags from query and domain."""
        tags = []
        if domain:
            tags.append(domain)
        # Add significant words as tags
        words = query.lower().split()
        for w in words:
            if len(w) > 3 and w not in ("the", "for", "and", "with", "from"):
                tags.append(w)
        return tags[:5]  # Max 5 tags

    def _infer_inputs(self, query: str, skill_type: str) -> str:
        """Infer input description from query and skill type."""
        if skill_type == "code":
            return "Input data for the code skill"
        elif skill_type == "search_strategy":
            return "Search query or topic"
        elif skill_type == "pipeline":
            return "Task input following the pipeline schema"
        else:
            return f"Input related to {query}"

    def _infer_outputs(self, query: str, skill_type: str) -> str:
        """Infer output description from query and skill type."""
        if skill_type == "code":
            return "Result of code execution"
        elif skill_type == "search_strategy":
            return "Search results and synthesized information"
        elif skill_type == "pipeline":
            return "Pipeline execution result"
        else:
            return f"Response addressing {query}"
