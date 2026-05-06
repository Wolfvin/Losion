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
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Encryption at Rest (v2.5.1: upgraded from XOR to Fernet/AES)
# ============================================================================

# Try to import cryptography.fernet for proper authenticated encryption.
# Fernet provides AES-128-CBC + HMAC-SHA256 with built-in IV and tamper
# detection — a significant upgrade over the previous XOR cipher which
# was vulnerable to known-plaintext attacks and provided no authentication.
#
# If cryptography is not installed, we fall back to the legacy XOR mode
# with an explicit deprecation warning.
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64 as _base64
    _FERNET_AVAILABLE = True
except ImportError:
    _FERNET_AVAILABLE = False


def _derive_fernet_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a Fernet-compatible key from passphrase and salt.

    Uses PBKDF2-HMAC-SHA256 with 100,000 iterations for key derivation.
    The derived key is base64-encoded to match Fernet's expected format.

    v2.5.2: Removed unreachable else branch that used _base64 (not in scope
    when cryptography is not installed). This function is ONLY called when
    _FERNET_AVAILABLE is True, so the fallback was dead code with a NameError.

    Args:
        passphrase: The encryption passphrase.
        salt: Random salt bytes (16 bytes).

    Returns:
        Base64-encoded 32-byte key suitable for Fernet.

    Raises:
        RuntimeError: If cryptography package is not installed.
    """
    if not _FERNET_AVAILABLE:
        raise RuntimeError(
            "_derive_fernet_key() requires the 'cryptography' package. "
            "Install with: pip install cryptography"
        )
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    key = kdf.derive(passphrase.encode("utf-8"))
    return _base64.urlsafe_b64encode(key)


def _xor_encrypt_decrypt(data: bytes, key: bytes) -> bytes:
    """Legacy XOR-based encrypt/decrypt (DEPRECATED — kept for backward compat).

    XOR with a repeating key is vulnerable to known-plaintext attacks and
    provides no authentication. This function is retained ONLY for reading
    episode files encrypted with the v2.5.0 XOR scheme. New encryptions
    always use Fernet (AES-128-CBC + HMAC-SHA256).

    Args:
        data: Data to encrypt/decrypt.
        key: Encryption key (will be cycled if shorter than data).

    Returns:
        Encrypted/decrypted bytes.
    """
    key_cycle = key * (len(data) // len(key) + 1)
    return bytes(a ^ b for a, b in zip(data, key_cycle[:len(data)]))


class _EncryptionManager:
    """Manages encryption at rest for episodic memory.

    v2.5.1: Upgraded from XOR to Fernet (AES-128-CBC + HMAC-SHA256).
    The previous XOR cipher was vulnerable to known-plaintext attacks:
    episode JSON always starts with a predictable format ("episode_id",
    "query", etc.), so an attacker with access to both the encrypted file
    and the salt could XOR the known plaintext with the ciphertext to
    recover the keystream and decrypt the rest. Fernet eliminates this
    class of attack entirely by using proper AES-CBC with random IV and
    HMAC-SHA256 for authentication (tamper detection).

    Security model (with Fernet):
    - AES-128-CBC encryption — semantic security via random IV per token
    - HMAC-SHA256 authentication — tampering is detected before decryption
    - PBKDF2-HMAC-SHA256 key derivation — 100k iterations resist brute-force
    - Random salt per file — rainbow table attacks infeasible

    Backward compatibility:
    - Can decrypt v2.5.0 XOR-encrypted episode files (auto-detected)
    - New encryptions always use Fernet
    - If `cryptography` package is not installed, falls back to XOR with
      a deprecation warning

    Args:
        passphrase: Encryption passphrase. If None, encryption is disabled.
            Passphrase can also be set via LOSION_MEMORY_PASSPHRASE env var.
    """

    # Magic bytes to identify Fernet-encrypted data (v2.5.1+)
    _FERNET_MAGIC = b"FRNT"

    def __init__(self, passphrase: Optional[str] = None) -> None:
        # Check environment variable if no passphrase provided
        if passphrase is None:
            passphrase = os.environ.get("LOSION_MEMORY_PASSPHRASE", None)

        self.enabled = passphrase is not None
        self._passphrase = passphrase

        if self.enabled and not _FERNET_AVAILABLE:
            logger.warning(
                "EpisodicMemory: 'cryptography' package not installed. "
                "Falling back to legacy XOR encryption (deprecated, insecure). "
                "Install with: pip install cryptography"
            )

    def encrypt(self, data: bytes) -> Tuple[bytes, bytes]:
        """Encrypt data, returning (encrypted_data, salt).

        v2.5.1: Uses Fernet (AES-128-CBC + HMAC-SHA256) when available.
        Falls back to XOR with deprecation warning if cryptography is
        not installed.

        Args:
            data: Plaintext bytes to encrypt.

        Returns:
            Tuple of (encrypted_bytes, salt_bytes). Salt must be stored
            alongside encrypted data for decryption.
        """
        if not self.enabled:
            return data, b""

        salt = os.urandom(16)

        if _FERNET_AVAILABLE:
            # Fernet encryption: AES-128-CBC + HMAC-SHA256
            key = _derive_fernet_key(self._passphrase, salt)
            f = Fernet(key)
            encrypted = f.encrypt(data)
            # Prepend magic bytes so decrypt() can identify the format
            return self._FERNET_MAGIC + encrypted, salt
        else:
            # Legacy XOR fallback (deprecated)
            raw_key = hashlib.pbkdf2_hmac(
                "sha256",
                self._passphrase.encode("utf-8"),
                salt,
                iterations=100_000,
            )
            encrypted = _xor_encrypt_decrypt(data, raw_key)
            return encrypted, salt

    def decrypt(self, data: bytes, salt: bytes) -> bytes:
        """Decrypt data using the stored salt.

        v2.5.1: Auto-detects encryption format:
        - If data starts with _FERNET_MAGIC, uses Fernet decryption
        - Otherwise, falls back to legacy XOR (for v2.5.0 compat)

        Args:
            data: Encrypted bytes.
            salt: Salt bytes that were stored alongside the data.

        Returns:
            Decrypted plaintext bytes.
        """
        if not self.enabled:
            return data

        # Auto-detect: Fernet-encrypted data starts with magic bytes
        if data[:4] == self._FERNET_MAGIC and _FERNET_AVAILABLE:
            key = _derive_fernet_key(self._passphrase, salt)
            f = Fernet(key)
            return f.decrypt(data[4:])  # Strip magic bytes before decryption
        else:
            # Legacy XOR (v2.5.0 compatibility)
            raw_key = hashlib.pbkdf2_hmac(
                "sha256",
                self._passphrase.encode("utf-8"),
                salt,
                iterations=100_000,
            )
            return _xor_encrypt_decrypt(data, raw_key)


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
        strict_mode: bool = False,
    ) -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.max_episodes = max_episodes
        self.activation_decay = activation_decay
        self.auto_save = auto_save
        self.strict_mode = strict_mode

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

        v2.5.2: Moved auto_save (which involves encryption + disk I/O) outside
        the lock to prevent blocking other threads during PBKDF2 key derivation
        (~100-300ms per call). The in-memory store update is still inside the
        lock for thread safety, but I/O is deferred.

        Args:
            episode: Episode to store.
        """
        should_save = False
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

            should_save = self.auto_save

        # v2.5.2: Save OUTSIDE the lock — encryption + disk I/O can take
        # 100-300ms per call, blocking all other threads during that time.
        if should_save:
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

    @staticmethod
    def _compute_checksum(data: bytes) -> str:
        """Compute SHA-256 checksum of raw episode data.

        Used for per-episode integrity verification on load. Stored in
        the index alongside the episode ID, so that corruption or
        tampering of individual episode files is detected without
        nuking the entire store.

        Args:
            data: Raw bytes of the episode JSON (before encryption).

        Returns:
            Hex digest string (first 16 chars of SHA-256).
        """
        return hashlib.sha256(data).hexdigest()[:16]

    def _atomic_write_text(self, path: Path, content: str) -> None:
        """Write text content to a file atomically using temp file + rename.

        Prevents partial writes on crash: if the process is killed mid-write,
        the original file remains intact. The rename is atomic on POSIX.

        Args:
            path: Target file path.
            content: String content to write.
        """
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        """Write binary content to a file atomically using temp file + rename.

        Args:
            path: Target file path.
            data: Bytes to write.
        """
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, prefix=".tmp_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        """Load episodes from persistent storage.

        v2.5.2: Moved decryption outside the lock to avoid blocking during
        PBKDF2 key derivation. Files are read and decrypted first, then the
        lock is acquired briefly to populate the in-memory store.

        v2.5.3: Added per-episode checksum verification. Each episode's
        JSON content is hashed on save and the checksum is stored in the
        index. On load, checksums are verified — corrupted or tampered
        episodes are skipped individually with an error log instead of
        causing a full-store reset.

        On index corruption, preserves the broken file for forensic analysis
        (renamed to .corrupt.<timestamp>) and skips bad entries individually
        instead of nuking the full store. In strict_mode, raises instead
        of silently resetting.
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
                with self._lock:
                    self._episodes = {}
                    self._query_index = {}
                    self._domain_index = {}
                if self.strict_mode:
                    raise RuntimeError(
                        f"EpisodicMemory: corrupt index at {index_path}: {exc!r}. "
                        f"Broken file preserved as .corrupt.<ts>. "
                        f"Strict mode requires explicit handling of data loss."
                    ) from exc
                return

            # === Phase 1: Read and decrypt all files (outside lock) ===
            # This is the slow part — PBKDF2 + Fernet decryption per file.
            loaded_episodes: List[Episode] = []
            episode_ids = index.get("episode_ids", [])
            checksums = index.get("checksums", {})  # episode_id → sha256[:16]
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
                            raw_json = decrypted_data.decode("utf-8")
                            # Verify checksum (if available in index)
                            expected = checksums.get(episode_id)
                            if expected:
                                actual = self._compute_checksum(decrypted_data)
                                if actual != expected:
                                    logger.error(
                                        f"EpisodicMemory: checksum mismatch for "
                                        f"episode {episode_id}: expected {expected}, "
                                        f"got {actual}. Skipping corrupt/tampered entry."
                                    )
                                    continue
                            episode = Episode.from_dict(json.loads(raw_json))
                        else:
                            # Load plain JSON episode
                            raw_json = ep_path.read_text(encoding="utf-8")
                            # Verify checksum (if available in index)
                            expected = checksums.get(episode_id)
                            if expected:
                                actual = self._compute_checksum(raw_json.encode("utf-8"))
                                if actual != expected:
                                    logger.error(
                                        f"EpisodicMemory: checksum mismatch for "
                                        f"episode {episode_id}: expected {expected}, "
                                        f"got {actual}. Skipping corrupt/tampered entry."
                                    )
                                    continue
                            episode = Episode.from_dict(json.loads(raw_json))
                        loaded_episodes.append(episode)
                    except (json.JSONDecodeError, KeyError, TypeError) as ep_exc:
                        # Skip this bad entry but continue loading others
                        logger.error(
                            f"EpisodicMemory: skipping corrupt episode "
                            f"{episode_id} at {ep_path}: {ep_exc!r}"
                        )
                        continue

            # === Phase 2: Populate in-memory store (inside lock, fast) ===
            with self._lock:
                for episode in loaded_episodes:
                    self._episodes[episode.episode_id] = episode
                    query_hash = self._hash_query(episode.query)
                    if query_hash not in self._query_index:
                        self._query_index[query_hash] = set()
                    self._query_index[query_hash].add(episode.episode_id)
                    if episode.domain:
                        if episode.domain not in self._domain_index:
                            self._domain_index[episode.domain] = set()
                        self._domain_index[episode.domain].add(episode.episode_id)

    def _save(self) -> None:
        """Save episodes to persistent storage.

        v2.5.2: Refactored to move encryption and disk I/O outside the lock.
        Previously, _save() was called inside `with self._lock`, which meant
        PBKDF2 key derivation (~100-300ms) blocked all other threads. Now we
        snapshot the state under a brief lock, then perform crypto + I/O
        without holding the lock.

        The trade-off is that concurrent store_episode() calls between the
        snapshot and the disk write may not be reflected in this particular
        save — but they will trigger their own auto_save. This is acceptable
        because the in-memory store is always the source of truth during
        runtime, and disk is a persistent backup.
        """
        # === Phase 1: Snapshot state under lock (fast, no I/O) ===
        with self._lock:
            episodes_snapshot = {
                eid: ep.to_dict() for eid, ep in self._episodes.items()
            }
            # Compute per-episode checksums for integrity verification.
            # The checksum is computed on the SAME serialized bytes that will be
            # written to disk (with indent=2) so that the load verification matches.
            checksums = {}
            for eid, ep_dict in episodes_snapshot.items():
                raw = json.dumps(ep_dict, indent=2, ensure_ascii=False).encode("utf-8")
                checksums[eid] = self._compute_checksum(raw)
            index_data = {
                "episode_ids": list(self._episodes.keys()),
                "checksums": checksums,
                "last_updated": time.time(),
            }
            stats_data = {
                "total_episodes": len(self._episodes),
                "successful_episodes": sum(1 for e in self._episodes.values() if e.success),
                "success_rate": sum(1 for e in self._episodes.values() if e.success) / max(len(self._episodes), 1),
                "domains": {d: len(ids) for d, ids in self._domain_index.items()},
                "total_lessons_learned": sum(len(e.key_lessons) for e in self._episodes.values()),
                "max_capacity": self.max_episodes if self.max_episodes > 0 else "unlimited",
            }

        # === Phase 2: Encrypt + write to disk (outside lock, can be slow) ===
        # Uses atomic writes (temp file + rename) to prevent partial writes on crash.
        # Episode files are written BEFORE the index so that a crash mid-save
        # leaves the old index pointing to a valid mix of old/new episodes.
        episodes_dir = self.store_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # Save each episode (with optional encryption — this is the slow part)
        for episode_id, episode_dict in episodes_snapshot.items():
            ep_path = episodes_dir / f"{episode_id}.json"
            episode_json = json.dumps(episode_dict, indent=2, ensure_ascii=False)

            if self._encryption.enabled:
                # Encrypt the JSON data (PBKDF2 + Fernet happens here)
                data_bytes = episode_json.encode("utf-8")
                encrypted_bytes, salt = self._encryption.encrypt(data_bytes)

                # Save salt alongside the encrypted data (atomic)
                salt_path = episodes_dir / f"{episode_id}.salt"
                self._atomic_write_bytes(salt_path, salt)

                # Save encrypted data as binary (atomic)
                self._atomic_write_bytes(ep_path, encrypted_bytes)
            else:
                # Save as plain JSON (atomic)
                self._atomic_write_text(ep_path, episode_json)

        # Save index LAST (so it references valid episode files)
        self._atomic_write_text(
            self.store_dir / "index.json",
            json.dumps(index_data, indent=2)
        )

        # Save stats
        self._atomic_write_text(
            self.store_dir / "stats.json",
            json.dumps(stats_data, indent=2)
        )

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
