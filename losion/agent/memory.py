"""
Episodic Memory — Experience-based memory with Ebbinghaus forgetting.

Inspired by:
- Synapse (2024): "Empowering LLM Agents with Episodic-Semantic Memory
  via Spreading Activation" — unified memory with spreading activation.
- MemP (2025): "Exploring Agent Procedural Memory" — procedural memory
  is separate from semantic memory.
- Reflexion (2023): Reflections stored for future decision-making.
- MemoryBank (Zhong et al., 2023, cited 790×): Human-like long-term memory
  with Ebbinghaus forgetting curve — memories decay over time unless
  reinforced by access.
- Generative Agents (Park et al., 2023): Multi-factor retrieval with
  recency × importance × relevance scoring.

This module implements a three-layer memory architecture:
1. **Procedural Memory**: How to do things (stored in SkillStore)
2. **Semantic Memory**: Facts and knowledge (linked to Engram Memory)
3. **Episodic Memory**: Past experiences and outcomes (this module)

v3 Improvements:
- Ebbinghaus forgetting curve: Memory strength decays over time,
  reinforced by access. Prevents unbounded memory growth and keeps
  the memory focused on relevant experiences.
- Multi-factor retrieval: Recency × Importance × Relevance scoring
  (from Generative Agents), replacing simple Jaccard similarity.
- Periodic consolidation: Merge similar episodes, discard weak ones.

Storage:
    episodic_dir/
    ├── index.json          # Query hash → episode IDs
    ├── episodes/
    │   ├── abc12345.json   # Episode entry
    │   └── ...
    └── stats.json          # Usage statistics

Retrieval uses multi-factor scoring (recency × importance × relevance)
with spreading activation, enabling generalization across similar queries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Encryption at Rest (audit finding A3.5)
# ============================================================================


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 32-byte encryption key from passphrase and salt using PBKDF2.

    Uses hashlib.pbkdf2_hmac with SHA-256 and 100,000 iterations for
    key derivation. This is computationally expensive enough to resist
    brute-force attacks while remaining fast for legitimate use.

    Args:
        passphrase: The encryption passphrase.
        salt: Random salt bytes (16+ bytes recommended).

    Returns:
        32-byte derived key.
    """
    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        iterations=100_000,
    )


def _xor_encrypt_decrypt(data: bytes, key: bytes) -> bytes:
    """XOR-based encrypt/decrypt (symmetric — same operation for both).

    This is a simple but effective encryption for data at rest when
    the threat model is casual file inspection rather than a determined
    cryptanalyst. For production deployments requiring strong encryption,
    replace with AES-256 via the `cryptography` package.

    The XOR cipher is used here because:
    1. No external dependencies (pure Python stdlib)
    2. Symmetric — same function for encrypt and decrypt
    3. Adequate for protecting episodic memory from casual file reads

    Args:
        data: Data to encrypt/decrypt.
        key: Encryption key (will be cycled if shorter than data).

    Returns:
        Encrypted/decrypted bytes.
    """
    # Cycle the key to match data length
    key_cycle = key * (len(data) // len(key) + 1)
    return bytes(a ^ b for a, b in zip(data, key_cycle[:len(data)]))


class _EncryptionManager:
    """Manages encryption at rest for episodic memory.

    Provides a simple encryption layer for JSON episode files stored on disk.
    This prevents casual file inspection from revealing sensitive episode
    data (queries, actions, reflections).

    Security model:
    - Protects against casual file reads (e.g., shared filesystem)
    - Does NOT protect against determined attackers with memory access
    - For production: replace with AES-256 via `cryptography` package

    The encryption key is derived from a passphrase using PBKDF2-HMAC-SHA256
    with 100,000 iterations and a random salt. The salt is stored alongside
    the encrypted data (standard practice — salt is not secret).

    Args:
        passphrase: Encryption passphrase. If None, encryption is disabled.
            Passphrase can also be set via LOSION_MEMORY_PASSPHRASE env var.
    """

    def __init__(self, passphrase: Optional[str] = None) -> None:
        # Check environment variable if no passphrase provided
        if passphrase is None:
            passphrase = os.environ.get("LOSION_MEMORY_PASSPHRASE", None)

        self.enabled = passphrase is not None
        self._key: Optional[bytes] = None
        self._passphrase = passphrase

    def _get_or_derive_key(self, salt: bytes) -> bytes:
        """Get cached key or derive from passphrase."""
        if self._key is None and self._passphrase is not None:
            self._key = _derive_key(self._passphrase, salt)
        return self._key or b""

    def encrypt(self, data: bytes) -> Tuple[bytes, bytes]:
        """Encrypt data, returning (encrypted_data, salt).

        Args:
            data: Plaintext bytes to encrypt.

        Returns:
            Tuple of (encrypted_bytes, salt_bytes). Salt must be stored
            alongside encrypted data for decryption.
        """
        if not self.enabled:
            return data, b""

        salt = os.urandom(16)
        key = _derive_key(self._passphrase, salt)
        encrypted = _xor_encrypt_decrypt(data, key)
        return encrypted, salt

    def decrypt(self, data: bytes, salt: bytes) -> bytes:
        """Decrypt data using the stored salt.

        Args:
            data: Encrypted bytes.
            salt: Salt bytes that were stored alongside the data.

        Returns:
            Decrypted plaintext bytes.
        """
        if not self.enabled:
            return data

        key = _derive_key(self._passphrase, salt)
        return _xor_encrypt_decrypt(data, key)


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
    # v3: Ebbinghaus forgetting curve fields
    strength: float = 1.0          # Memory strength [0.0, 1.0], decays over time
    access_count: int = 0          # Number of times this episode was accessed
    last_accessed_at: float = 0.0  # Timestamp of last access

    def __post_init__(self) -> None:
        if not self.episode_id:
            content = f"{self.query}:{self.created_at or time.time()}"
            self.episode_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_accessed_at == 0.0:
            self.last_accessed_at = self.created_at

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

    @property
    def importance(self) -> float:
        """Importance score based on confidence and success (Generative Agents).

        Higher importance = more likely to be retrieved and retained.
        Successful episodes with high confidence are most important.
        """
        base = self.final_confidence
        if self.success:
            base *= 1.2
        else:
            base *= 0.6
        return min(1.0, base)

    def get_effective_strength(self, current_time: Optional[float] = None) -> float:
        """Compute effective memory strength with Ebbinghaus forgetting curve.

        Memory strength decays over time following the Ebbinghaus curve:
            strength = base_decay^age * reinforcement_from_access

        Where:
        - base_decay = e^(-0.1 * age_in_days) — exponential forgetting
        - reinforcement = 1 + 0.2 * log(1 + access_count) — access strengthens memory

        This prevents stale memories from dominating retrieval while keeping
        frequently-accessed memories strong.
        """
        import math
        now = current_time or time.time()
        age_seconds = now - self.created_at
        age_days = age_seconds / 86400.0

        # Ebbinghaus exponential decay
        decay = math.exp(-0.1 * age_days)

        # Reinforcement from repeated access
        reinforcement = 1.0 + 0.2 * math.log(1 + self.access_count)

        # Combined effective strength
        effective = self.strength * decay * reinforcement
        return min(1.0, max(0.0, effective))

    def reinforce(self) -> None:
        """Reinforce this memory by recording an access.

        Following MemoryBank's Ebbinghaus model: each access strengthens
        the memory, counteracting natural decay.
        """
        self.access_count += 1
        self.last_accessed_at = time.time()
        # Boost strength on access (partial recovery from decay)
        self.strength = min(1.0, self.strength + 0.1)

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
        encryption_passphrase: Optional[str] = None,
    ) -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.max_episodes = max_episodes
        self.activation_decay = activation_decay
        self.auto_save = auto_save

        # Encryption at rest (audit finding A3.5)
        self._encryption = _EncryptionManager(encryption_passphrase)
        if self._encryption.enabled:
            logger.info("EpisodicMemory: encryption at rest ENABLED")

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

        v3: Uses multi-factor retrieval scoring (from Generative Agents):
            score = recency × importance × relevance × effective_strength

        Where:
        - recency: Exponential decay based on time since creation
        - importance: Based on final confidence and success
        - relevance: Jaccard similarity of query terms
        - effective_strength: Ebbinghaus forgetting curve with access reinforcement

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
            now = time.time()
            candidates: List[Tuple[Episode, float]] = []

            for episode in self._episodes.values():
                # === Multi-factor scoring (Generative Agents style) ===

                # 1. Relevance: Jaccard similarity
                episode_terms = set(episode.query.lower().split())
                if query_terms and episode_terms:
                    intersection = len(query_terms & episode_terms)
                    union = len(query_terms | episode_terms)
                    relevance = intersection / union if union > 0 else 0.0
                else:
                    relevance = 0.0

                # Domain bonus
                if domain and episode.domain == domain:
                    relevance += 0.2
                elif domain and episode.domain and episode.domain != domain:
                    relevance -= 0.15

                # 2. Recency: Exponential decay (Park et al.)
                age_hours = (now - episode.created_at) / 3600.0
                recency = math.exp(-0.05 * age_hours)  # Half-life ~14 hours

                # 3. Importance: Based on confidence and success
                importance = episode.importance

                # 4. Effective strength: Ebbinghaus forgetting curve
                effective_strength = episode.get_effective_strength(now)

                # === Composite score: recency × importance × relevance × strength ===
                composite = recency * importance * max(relevance, 0.01) * effective_strength

                if composite >= min_similarity:
                    candidates.append((episode, composite))

            # Sort by composite score
            candidates.sort(key=lambda x: x[1], reverse=True)

            # === Spreading Activation ===
            top_episodes = candidates[:limit]
            if top_episodes and self.activation_decay > 0:
                activated_ids: Set[str] = {ep.episode_id for ep, _ in top_episodes}
                for ep, sim in top_episodes:
                    for other_id in self._domain_index.get(ep.domain or "", set()):
                        if other_id not in activated_ids:
                            other = self._episodes.get(other_id)
                            if other:
                                decayed_sim = sim * self.activation_decay
                                if decayed_sim >= min_similarity:
                                    candidates.append((other, decayed_sim))
                                    activated_ids.add(other_id)

                candidates.sort(key=lambda x: x[1], reverse=True)

            # === Reinforce accessed memories (Ebbinghaus) ===
            for episode, _ in candidates[:limit]:
                episode.reinforce()

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
        """Load episodes from persistent storage.

        On corruption, preserves the broken file for forensic analysis
        (renamed to .corrupt.<timestamp>) and skips bad entries individually
        instead of nuking the full store. Errors are surfaced via logging.
        """
        episodes_dir = self.store_dir / "episodes"
        if not episodes_dir.exists():
            return

        index_path = self.store_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                # Corrupt index — preserve broken file, log, reset
                logger.error(
                    f"EpisodicMemory: corrupt index at {index_path}: {exc!r}. "
                    f"Renaming to .corrupt.<ts> and resetting in-memory store."
                )
                try:
                    corrupt_name = f"index.json.corrupt.{int(time.time())}"
                    index_path.rename(index_path.parent / corrupt_name)
                except OSError:
                    pass
                self._episodes = {}
                self._query_index = {}
                self._domain_index = {}
                return

            # Load each episode individually — skip bad entries instead of
            # nuking the entire store on a single bad file.
            episode_ids = index.get("episode_ids", [])
            for episode_id in episode_ids:
                ep_path = episodes_dir / f"{episode_id}.json"
                salt_path = episodes_dir / f"{episode_id}.salt"
                if ep_path.exists():
                    try:
                        if self._encryption.enabled and salt_path.exists():
                            # Load encrypted episode
                            with open(salt_path, "rb") as sf:
                                salt = sf.read()
                            with open(ep_path, "rb") as f:
                                encrypted_data = f.read()
                            decrypted_data = self._encryption.decrypt(encrypted_data, salt)
                            episode = Episode.from_dict(json.loads(decrypted_data.decode("utf-8")))
                        else:
                            # Load plain JSON episode
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
                    except (json.JSONDecodeError, KeyError, TypeError) as ep_exc:
                        # Skip this bad entry but continue loading others
                        logger.error(
                            f"EpisodicMemory: skipping corrupt episode "
                            f"{episode_id} at {ep_path}: {ep_exc!r}"
                        )
                        continue

    def _save(self) -> None:
        """Save episodes to persistent storage.

        v2.5.0: When encryption is enabled, episode JSON files are encrypted
        at rest using the _EncryptionManager. The salt is stored in a
        separate ``.salt`` file alongside the episode file.
        """
        episodes_dir = self.store_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # Save index
        index = {
            "episode_ids": list(self._episodes.keys()),
            "last_updated": time.time(),
        }
        with open(self.store_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        # Save each episode (with optional encryption)
        for episode_id, episode in self._episodes.items():
            ep_path = episodes_dir / f"{episode_id}.json"
            episode_json = json.dumps(episode.to_dict(), indent=2, ensure_ascii=False)

            if self._encryption.enabled:
                # Encrypt the JSON data
                data_bytes = episode_json.encode("utf-8")
                encrypted_bytes, salt = self._encryption.encrypt(data_bytes)

                # Save salt alongside the encrypted data
                salt_path = episodes_dir / f"{episode_id}.salt"
                with open(salt_path, "wb") as sf:
                    sf.write(salt)

                # Save encrypted data as binary
                with open(ep_path, "wb") as f:
                    f.write(encrypted_bytes)
            else:
                # Save as plain JSON
                with open(ep_path, "w", encoding="utf-8") as f:
                    f.write(episode_json)

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

    def consolidate(self, strength_threshold: float = 0.05, similarity_threshold: float = 0.8) -> Dict[str, int]:
        """Consolidate memory by removing weak and redundant episodes.

        v3: Following MemoryBank's consolidation process:
        1. Remove episodes whose effective strength has decayed below threshold
        2. Merge very similar episodes (keeping the stronger one)

        This prevents unbounded memory growth and keeps the memory
        focused on relevant, well-established experiences.

        Args:
            strength_threshold: Remove episodes with effective strength below this.
            similarity_threshold: Merge episodes with similarity above this.

        Returns:
            Dictionary with consolidation statistics.
        """
        with self._lock:
            removed_weak = 0
            merged = 0
            now = time.time()

            # Step 1: Remove decayed episodes
            to_remove = []
            for eid, episode in self._episodes.items():
                effective = episode.get_effective_strength(now)
                if effective < strength_threshold:
                    to_remove.append(eid)

            for eid in to_remove:
                self._remove_episode(eid)
                removed_weak += 1

            # Step 2: Merge very similar episodes
            episode_list = list(self._episodes.values())
            for i, ep1 in enumerate(episode_list):
                for ep2 in episode_list[i+1:]:
                    if ep1.episode_id not in self._episodes or ep2.episode_id not in self._episodes:
                        continue
                    terms1 = set(ep1.query.lower().split())
                    terms2 = set(ep2.query.lower().split())
                    if terms1 and terms2:
                        intersection = len(terms1 & terms2)
                        union = len(terms1 | terms2)
                        sim = intersection / union if union > 0 else 0.0

                        if sim >= similarity_threshold and ep1.domain == ep2.domain:
                            # Keep the stronger episode
                            s1 = ep1.get_effective_strength(now)
                            s2 = ep2.get_effective_strength(now)
                            if s1 >= s2:
                                self._remove_episode(ep2.episode_id)
                                merged += 1
                            else:
                                self._remove_episode(ep1.episode_id)
                                merged += 1

            if self.auto_save:
                self._save()

            stats = {"removed_weak": removed_weak, "merged": merged, "remaining": len(self._episodes)}
            logger.info(f"Memory consolidation: {stats}")
            return stats
