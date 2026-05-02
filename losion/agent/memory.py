"""
Episodic Memory — Experience-based memory for the Agent Layer.

Inspired by:
- Synapse (2024): "Empowering LLM Agents with Episodic-Semantic Memory
  via Spreading Activation" — unified memory with spreading activation.
- MemP (2025): "Exploring Agent Procedural Memory" — procedural memory
  is separate from semantic memory.
- Reflexion (2023): Reflections stored for future decision-making.

This module implements a three-layer memory architecture:
1. **Procedural Memory**: How to do things (stored in SkillStore)
2. **Semantic Memory**: Facts and knowledge (linked to Engram Memory)
3. **Episodic Memory**: Past experiences and outcomes (this module)

Episodic Memory is the KEY addition — it stores the agent's experiences:
- What actions were taken
- What the outcomes were
- What reflections were generated
- What worked and what didn't

This enables the agent to:
- Avoid repeating failed strategies
- Reuse successful strategies for similar queries
- Build up domain expertise over time
- Provide context for the ReflectionEngine

Storage:
    episodic_dir/
    ├── index.json          # Query hash → episode IDs
    ├── episodes/
    │   ├── abc12345.json   # Episode entry
    │   └── ...
    └── stats.json          # Usage statistics

Retrieval uses query similarity (keyword overlap + domain match)
rather than exact hash, enabling generalization across similar queries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """A single episodic memory entry.

    Represents one complete agent interaction — what was queried,
    what actions were taken, what the outcomes were, and what
    was learned.

    Attributes:
        episode_id: Unique identifier (hash-based).
        query: The original user query.
        domain: Domain classification.
        actions: List of actions taken in order.
        reflections: Reflections generated during the interaction.
        final_confidence: Confidence at the end of the interaction.
        success: Whether the interaction achieved its goal.
        total_iterations: Number of agent loop iterations.
        total_time: Total execution time.
        created_at: Creation timestamp.
        metadata: Additional metadata.
    """

    episode_id: str = ""
    query: str = ""
    domain: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    final_confidence: float = 0.0
    success: bool = False
    total_iterations: int = 0
    total_time: float = 0.0
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            content = f"{self.query}:{self.created_at or time.time()}"
            self.episode_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def key_lessons(self) -> List[str]:
        """Extract key lessons from reflections."""
        return [
            r.get("lesson", "")
            for r in self.reflections
            if r.get("lesson")
        ]

    @property
    def successful_actions(self) -> List[str]:
        """Actions that had positive reflections."""
        return [
            r.get("action_taken", "")
            for r in self.reflections
            if r.get("reflection_type") in ("action_success", "strategy_correction")
        ]

    @property
    def failed_actions(self) -> List[str]:
        """Actions that had negative reflections."""
        return [
            r.get("action_taken", "")
            for r in self.reflections
            if r.get("reflection_type") == "action_failure"
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Deserialize from dictionary."""
        return cls(**data)


class EpisodicMemory:
    """Episodic memory store for the agent layer.

    Stores past agent interactions (episodes) and provides retrieval
    based on query similarity. When the agent encounters a new query,
    it can look up similar past episodes and use the lessons learned
    to make better decisions.

    This is the "experience" component of the three-layer memory:
    - Procedural (SkillStore): HOW to do things
    - Semantic (Engram): WHAT facts are known
    - Episodic (this): WHAT HAPPENED in past interactions

    Spreading Activation (from Synapse):
    When an episode is retrieved, related episodes are also activated
    with decreasing strength. This enables the agent to access not
    just the most similar episode, but a network of related experiences.

    Args:
        store_dir: Directory for persistent storage.
        max_episodes: Maximum number of episodes to store (0 = unlimited).
        activation_decay: Decay factor for spreading activation.
        auto_save: Whether to auto-save after every write.
    """

    def __init__(
        self,
        store_dir: str = "~/.losion/episodic",
        max_episodes: int = 0,
        activation_decay: float = 0.5,
        auto_save: bool = True,
    ) -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.max_episodes = max_episodes
        self.activation_decay = activation_decay
        self.auto_save = auto_save

        # In-memory storage
        self._episodes: Dict[str, Episode] = {}
        # Query hash → episode IDs (for fast retrieval)
        self._query_index: Dict[str, Set[str]] = {}
        # Domain → episode IDs
        self._domain_index: Dict[str, Set[str]] = {}
        # Thread lock
        self._lock = threading.Lock()

        # Load from disk
        self._load()

    def store_episode(self, episode: Episode) -> None:
        """Store an episode in memory.

        Args:
            episode: Episode to store.
        """
        with self._lock:
            # Check capacity
            if (
                self.max_episodes > 0
                and episode.episode_id not in self._episodes
                and len(self._episodes) >= self.max_episodes
            ):
                self._evict_oldest()

            # Store
            self._episodes[episode.episode_id] = episode

            # Update indices
            query_hash = self._hash_query(episode.query)
            if query_hash not in self._query_index:
                self._query_index[query_hash] = set()
            self._query_index[query_hash].add(episode.episode_id)

            if episode.domain:
                if episode.domain not in self._domain_index:
                    self._domain_index[episode.domain] = set()
                self._domain_index[episode.domain].add(episode.episode_id)

            if self.auto_save:
                self._save()

    def retrieve_similar(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.1,
    ) -> List[Tuple[Episode, float]]:
        """Retrieve episodes similar to the given query.

        Uses keyword overlap (Jaccard similarity) for matching,
        with domain bonus for same-domain episodes.

        Spreading Activation:
        After finding the top matches, also activate related episodes
        (episodes that share actions or domain) with decayed strength.

        Args:
            query: Query to find similar episodes for.
            domain: Optional domain filter.
            limit: Maximum number of results.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of (Episode, similarity_score) tuples, sorted by similarity.
        """
        with self._lock:
            if not self._episodes:
                return []

            query_terms = set(query.lower().split())
            candidates: List[Tuple[Episode, float]] = []

            for episode in self._episodes.values():
                # Domain filter
                if domain and episode.domain and episode.domain != domain:
                    # Reduce similarity for different domain but don't exclude
                    domain_penalty = 0.3
                else:
                    domain_penalty = 0.0

                # Keyword overlap (Jaccard similarity)
                episode_terms = set(episode.query.lower().split())
                if query_terms and episode_terms:
                    intersection = len(query_terms & episode_terms)
                    union = len(query_terms | episode_terms)
                    similarity = intersection / union if union > 0 else 0.0
                else:
                    similarity = 0.0

                # Domain bonus
                if domain and episode.domain == domain:
                    similarity += 0.2

                # Success bonus — prefer successful episodes
                if episode.success:
                    similarity += 0.1

                # Apply domain penalty
                similarity -= domain_penalty

                if similarity >= min_similarity:
                    candidates.append((episode, similarity))

            # Sort by similarity
            candidates.sort(key=lambda x: x[1], reverse=True)

            # === Spreading Activation ===
            # Activate related episodes with decayed strength
            top_episodes = candidates[:limit]
            if top_episodes and self.activation_decay > 0:
                activated_ids: Set[str] = {ep.episode_id for ep, _ in top_episodes}
                for ep, sim in top_episodes:
                    # Find episodes that share actions
                    for other_id in self._domain_index.get(ep.domain or "", set()):
                        if other_id not in activated_ids:
                            other = self._episodes.get(other_id)
                            if other:
                                decayed_sim = sim * self.activation_decay
                                if decayed_sim >= min_similarity:
                                    candidates.append((other, decayed_sim))
                                    activated_ids.add(other_id)

                # Re-sort
                candidates.sort(key=lambda x: x[1], reverse=True)

            return candidates[:limit]

    def get_lessons_for_query(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 3,
    ) -> List[str]:
        """Get lessons learned from similar past queries.

        This is the primary interface for the signal extractor:
        given a new query, what did we learn from similar past queries?

        Args:
            query: Current query.
            domain: Domain classification.
            limit: Maximum number of lessons.

        Returns:
            List of lesson strings from similar episodes.
        """
        similar = self.retrieve_similar(query, domain, limit=limit)
        lessons = []
        for episode, similarity in similar:
            for lesson in episode.key_lessons:
                if lesson and lesson not in lessons:
                    lessons.append(lesson)
            if len(lessons) >= limit:
                break
        return lessons[:limit]

    def get_action_recommendations(
        self,
        query: str,
        domain: Optional[str] = None,
    ) -> Dict[str, float]:
        """Get action recommendations based on past experience.

        Returns a mapping of action → score, where score represents
        how successful that action has been for similar queries.

        Args:
            query: Current query.
            domain: Domain classification.

        Returns:
            Dictionary mapping action names to success scores.
        """
        similar = self.retrieve_similar(query, domain, limit=10)
        action_scores: Dict[str, List[float]] = {}

        for episode, similarity in similar:
            weight = similarity  # Weight by similarity
            for action in episode.successful_actions:
                if action not in action_scores:
                    action_scores[action] = []
                action_scores[action].append(weight)

            for action in episode.failed_actions:
                if action not in action_scores:
                    action_scores[action] = []
                action_scores[action].append(-weight * 0.5)  # Penalty

        # Aggregate
        recommendations = {}
        for action, scores in action_scores.items():
            recommendations[action] = sum(scores) / max(len(scores), 1)

        return recommendations

    def _hash_query(self, query: str) -> str:
        """Hash a query for indexing."""
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _evict_oldest(self) -> None:
        """Evict the oldest episode."""
        if not self._episodes:
            return
        oldest_id = min(self._episodes, key=lambda eid: self._episodes[eid].created_at)
        self._remove_episode(oldest_id)

    def _remove_episode(self, episode_id: str) -> None:
        """Remove an episode and clean up indices."""
        episode = self._episodes.pop(episode_id, None)
        if episode is None:
            return

        # Clean up query index
        query_hash = self._hash_query(episode.query)
        if query_hash in self._query_index:
            self._query_index[query_hash].discard(episode_id)
            if not self._query_index[query_hash]:
                del self._query_index[query_hash]

        # Clean up domain index
        if episode.domain and episode.domain in self._domain_index:
            self._domain_index[episode.domain].discard(episode_id)
            if not self._domain_index[episode.domain]:
                del self._domain_index[episode.domain]

    def _load(self) -> None:
        """Load episodes from persistent storage."""
        episodes_dir = self.store_dir / "episodes"
        if not episodes_dir.exists():
            return

        index_path = self.store_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                for episode_id in index.get("episode_ids", []):
                    ep_path = episodes_dir / f"{episode_id}.json"
                    if ep_path.exists():
                        with open(ep_path, "r", encoding="utf-8") as f:
                            episode = Episode.from_dict(json.load(f))
                        self._episodes[episode_id] = episode
                        # Rebuild indices
                        query_hash = self._hash_query(episode.query)
                        if query_hash not in self._query_index:
                            self._query_index[query_hash] = set()
                        self._query_index[query_hash].add(episode_id)
                        if episode.domain:
                            if episode.domain not in self._domain_index:
                                self._domain_index[episode.domain] = set()
                            self._domain_index[episode.domain].add(episode_id)
            except (json.JSONDecodeError, KeyError, TypeError):
                self._episodes = {}
                self._query_index = {}
                self._domain_index = {}

    def _save(self) -> None:
        """Save episodes to persistent storage."""
        episodes_dir = self.store_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # Save index
        index = {
            "episode_ids": list(self._episodes.keys()),
            "last_updated": time.time(),
        }
        with open(self.store_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        # Save each episode
        for episode_id, episode in self._episodes.items():
            ep_path = episodes_dir / f"{episode_id}.json"
            with open(ep_path, "w", encoding="utf-8") as f:
                json.dump(episode.to_dict(), f, indent=2, ensure_ascii=False)

        # Save stats (compute inline to avoid deadlock — _save is called within _lock)
        total = len(self._episodes)
        successful = sum(1 for e in self._episodes.values() if e.success)
        domains = {d: len(ids) for d, ids in self._domain_index.items()}
        total_lessons = sum(len(e.key_lessons) for e in self._episodes.values())
        stats = {
            "total_episodes": total,
            "successful_episodes": successful,
            "success_rate": successful / max(total, 1),
            "domains": domains,
            "total_lessons_learned": total_lessons,
            "max_capacity": self.max_episodes if self.max_episodes > 0 else "unlimited",
        }
        with open(self.store_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total = len(self._episodes)
        successful = sum(1 for e in self._episodes.values() if e.success)
        domains = {d: len(ids) for d, ids in self._domain_index.items()}
        total_lessons = sum(len(e.key_lessons) for e in self._episodes.values())

        return {
            "total_episodes": total,
            "successful_episodes": successful,
            "success_rate": successful / max(total, 1),
            "domains": domains,
            "total_lessons_learned": total_lessons,
            "max_capacity": self.max_episodes if self.max_episodes > 0 else "unlimited",
        }

    @property
    def size(self) -> int:
        """Number of episodes in memory."""
        return len(self._episodes)
