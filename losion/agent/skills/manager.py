"""
Skill Manager — High-level skill management with lookup and auto-creation.

The SkillManager is the main interface for the agent orchestrator to
interact with skills. It provides:
1. Skill lookup with fuzzy matching
2. Automatic skill creation when no suitable skill exists
3. Skill execution tracking
4. Skill ranking and selection

It delegates storage to SkillStore and creation to SkillCreator,
acting as the coordination layer between them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from losion.agent.skills.store import SkillEntry, SkillMetadata, SkillStore
from losion.agent.skills.creator import SkillCreator

logger = logging.getLogger(__name__)


@dataclass
class SkillLookupResult:
    """Result of a skill lookup operation.

    Attributes:
        found: Whether a suitable skill was found.
        skill: The found skill entry (None if not found).
        candidates: List of candidate skills that partially match.
        auto_created: Whether the skill was auto-created during this lookup.
        confidence: Confidence score of the best match.
    """

    found: bool = False
    skill: Optional[SkillEntry] = None
    candidates: List[SkillEntry] = None
    auto_created: bool = False
    confidence: float = 0.0

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []


class SkillManager:
    """High-level skill management for the Losion Agent Layer.

    The SkillManager provides the interface between the agent orchestrator
    and the skill system. It handles:
    - Looking up skills by query, domain, or tags
    - Auto-creating skills when no suitable one exists
    - Tracking skill usage and success rates
    - Ranking skills by relevance and confidence

    Auto-creation flow:
        1. Agent requests a skill for a given query/domain
        2. Manager searches the SkillStore
        3. If no suitable skill found AND auto_creation enabled:
           a. Creator generates skill via web search + synthesis
           b. New skill is stored in SkillStore
           c. Skill is returned to agent
        4. If auto_creation disabled, returns not-found

    Args:
        store: SkillStore instance for persistent storage.
        creator: SkillCreator instance for auto-creation.
        auto_create: Whether to auto-create skills when not found.
        min_match_confidence: Minimum confidence for a skill to be considered a match.
        max_candidates: Maximum number of candidate skills to return.
    """

    def __init__(
        self,
        store: Optional[SkillStore] = None,
        creator: Optional[SkillCreator] = None,
        auto_create: bool = True,
        min_match_confidence: float = 0.3,
        max_candidates: int = 5,
    ) -> None:
        self.store = store or SkillStore()
        self.creator = creator or SkillCreator(store=self.store)
        self.auto_create = auto_create
        self.min_match_confidence = min_match_confidence
        self.max_candidates = max_candidates

    def lookup(
        self,
        query: str,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> SkillLookupResult:
        """Look up a skill for the given query.

        Search flow:
        1. Exact name match (O(1) hash lookup)
        2. Fuzzy search (query + domain + tags)
        3. Auto-create if no match and enabled

        Args:
            query: Skill query (name, description keywords, etc.).
            domain: Optional domain filter.
            tags: Optional tag filters.

        Returns:
            SkillLookupResult with the best matching skill.
        """
        # === Step 1: Exact name match ===
        exact = self.store.lookup(query)
        if exact is not None and exact.metadata.confidence >= self.min_match_confidence:
            logger.info(f"Skill found (exact): {query}")
            return SkillLookupResult(
                found=True,
                skill=exact,
                confidence=exact.metadata.confidence,
            )

        # === Step 2: Fuzzy search ===
        candidates = self.store.search(
            query=query,
            domain=domain,
            tags=tags,
            min_confidence=self.min_match_confidence,
            limit=self.max_candidates,
        )

        if candidates:
            best = candidates[0]
            # Check if best candidate is a strong enough match
            if best.metadata.confidence >= self.min_match_confidence:
                logger.info(f"Skill found (fuzzy): {best.name} (conf={best.metadata.confidence:.2f})")
                return SkillLookupResult(
                    found=True,
                    skill=best,
                    candidates=candidates[1:],
                    confidence=best.metadata.confidence,
                )

        # === Step 3: Auto-create if enabled ===
        if self.auto_create and self.creator is not None:
            logger.info(f"No suitable skill found for '{query}', auto-creating...")
            try:
                new_skill = self.creator.create(
                    query=query,
                    domain=domain,
                    tags=tags,
                )
                if new_skill is not None:
                    return SkillLookupResult(
                        found=True,
                        skill=new_skill,
                        auto_created=True,
                        confidence=new_skill.metadata.confidence,
                    )
            except Exception as e:
                logger.warning(f"Auto-creation failed for '{query}': {e}")

        # === No match ===
        return SkillLookupResult(
            found=False,
            candidates=candidates,
        )

    def register_skill(
        self,
        name: str,
        description: str,
        skill_type: str,
        definition: str,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        inputs: str = "",
        outputs: str = "",
    ) -> SkillEntry:
        """Register a new skill manually.

        Args:
            name: Unique skill name.
            description: What this skill does.
            skill_type: Type ("prompt", "code", "pipeline", "search_strategy").
            definition: The skill content.
            domain: Domain classification.
            tags: Searchable tags.
            inputs: Input description.
            outputs: Output description.

        Returns:
            The created SkillEntry.
        """
        entry = SkillEntry(
            name=name,
            description=description,
            skill_type=skill_type,
            definition=definition,
            metadata=SkillMetadata(
                source="manual",
                domain=domain,
                tags=tags or [],
                confidence=0.5,  # Initial confidence for manually registered skills
            ),
            inputs=inputs,
            outputs=outputs,
        )
        self.store.store(entry)
        logger.info(f"Skill registered: {name}")
        return entry

    def record_usage(self, skill_name: str, success: bool) -> None:
        """Record a skill usage event.

        Args:
            skill_name: Name of the skill that was used.
            success: Whether the usage was successful.
        """
        self.store.record_usage(skill_name, success)

    def get_skill(self, name: str) -> Optional[SkillEntry]:
        """Get a skill by exact name.

        Args:
            name: Skill name.

        Returns:
            SkillEntry or None.
        """
        return self.store.lookup(name)

    def list_skills(
        self,
        domain: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[SkillEntry]:
        """List skills, optionally filtered.

        Args:
            domain: Filter by domain.
            min_confidence: Filter by minimum confidence.

        Returns:
            List of SkillEntry objects.
        """
        return self.store.search(
            domain=domain,
            min_confidence=min_confidence,
            limit=1000,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get skill store statistics."""
        return self.store.get_stats()
