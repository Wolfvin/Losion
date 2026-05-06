"""
Skill Store — Persistent skill storage inspired by Losion's Engram Memory.

This module implements an Engram-like storage system for agent skills.
Just as Engram Memory stores factual knowledge in hash-based buckets for
O(1) retrieval, SkillStore stores agent skills for fast lookup.

Key design decisions:
- Skills are stored as JSON-serializable entries in a persistent directory
- Hash-based indexing for O(1) lookup (same principle as Engram)
- Metadata tracking (creation time, usage count, success rate)
- Thread-safe operations for concurrent access
- Auto-expiration of stale skills

Unlike model-level Engram (which stores embedding vectors), SkillStore
stores executable skill definitions — code, prompts, and configurations
that the agent can invoke.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    """Metadata for a skill entry.

    Tracks usage statistics, creation info, and domain classification
    for efficient skill discovery and ranking.

    Attributes:
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.
        usage_count: Number of times this skill has been invoked.
        success_count: Number of successful invocations.
        last_used_at: Unix timestamp of last invocation.
        source: How this skill was created ("manual", "auto_created", "web_search").
        domain: Domain classification (math, code, science, etc.).
        tags: Free-form tags for search.
        dependencies: List of external dependencies this skill requires.
        confidence: Reliability score [0.0, 1.0] based on success rate.
    """

    created_at: float = 0.0
    updated_at: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    last_used_at: float = 0.0
    source: str = "manual"
    domain: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = self.created_at

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction [0.0, 1.0]."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def record_usage(self, success: bool) -> None:
        """Record a skill invocation.

        Args:
            success: Whether the invocation was successful.
        """
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.last_used_at = time.time()
        self.updated_at = time.time()
        # Update confidence using Bayesian-like update
        self.confidence = self.success_rate

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillMetadata":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class SkillEntry:
    """A single skill entry in the skill store.

    A skill is a reusable capability that the agent can invoke. It contains
    the skill definition (code, prompt, or configuration) along with
    metadata for discovery and ranking.

    v3 (Voyager-style): Skills can now be executable programs with
    preconditions, postconditions, and error patterns, following
    Voyager's skill library design (Wang et al., 2023, NVIDIA).

    Attributes:
        name: Unique skill name (kebab-case, e.g., "python-unit-test").
        description: Human-readable description of what this skill does.
        skill_type: Type of skill ("prompt", "code", "pipeline", "search_strategy").
        definition: The actual skill content (prompt template, code, etc.).
        metadata: Usage and quality metadata.
        version: Skill version number.
        inputs: Description of expected inputs.
        outputs: Description of expected outputs.
        executable_code: Voyager-style: executable Python code string.
        preconditions: Voyager-style: conditions that must be true before execution.
        postconditions: Voyager-style: conditions that should be true after execution.
        error_patterns: Voyager-style: common failures and their fix strategies.
    """

    name: str
    description: str
    skill_type: str  # "prompt", "code", "pipeline", "search_strategy"
    definition: str  # The actual skill content
    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    version: int = 1
    inputs: str = ""
    outputs: str = ""
    # v3: Voyager-style executable skills
    executable_code: str = ""           # Python function source code
    preconditions: List[str] = field(default_factory=list)   # e.g., ["web_search_available"]
    postconditions: List[str] = field(default_factory=list)  # e.g., ["result_is_valid_json"]
    error_patterns: Dict[str, str] = field(default_factory=dict)  # error_type → fix_strategy

    @property
    def domain(self) -> Optional[str]:
        """Convenience access to domain from metadata."""
        return self.metadata.domain

    @property
    def hash_key(self) -> str:
        """Hash key for O(1) lookup (Engram-style)."""
        return hashlib.sha256(self.name.encode()).hexdigest()[:16]

    @property
    def is_executable(self) -> bool:
        """Whether this skill has executable code (Voyager-style)."""
        return bool(self.executable_code.strip())

    def execute(self, **kwargs: Any) -> Any:
        """Execute this skill with Voyager-style retry on known errors.

        If the skill has executable_code, it will be executed with
        automatic retry on known error patterns. This follows Voyager's
        approach of storing skills as executable programs with retry logic.

        Args:
            **kwargs: Arguments to pass to the skill function.

        Returns:
            Result of skill execution.

        Raises:
            RuntimeError: If skill fails after all retry attempts.
        """
        if not self.is_executable:
            # Non-executable skills just return their definition
            return self.definition

        for attempt in range(3):
            try:
                # Execute in a restricted namespace
                namespace: Dict[str, Any] = {"__builtins__": __builtins__}
                exec(self.executable_code, namespace)
                # Find the skill function (convention: skill_<name>)
                func_name = f"skill_{self.name.replace('-', '_').replace(' ', '_')}"
                if func_name in namespace:
                    result = namespace[func_name](**kwargs)
                    if self._verify_postconditions(result):
                        return result
                    else:
                        logger.warning(f"Skill {self.name} postconditions not met on attempt {attempt+1}")
                else:
                    # Try to find any function in namespace
                    for key, val in namespace.items():
                        if callable(val) and not key.startswith("_"):
                            result = val(**kwargs)
                            if self._verify_postconditions(result):
                                return result
                            break
            except Exception as e:
                error_type = type(e).__name__
                fix = self.error_patterns.get(error_type)
                if fix:
                    logger.info(f"Skill {self.name} error: {error_type}. Applying fix: {fix}")
                    # Apply fix (e.g., modify kwargs)
                    if "retry_with_empty_input" in fix and not kwargs:
                        kwargs = {"input_data": ""}
                else:
                    logger.warning(f"Skill {self.name} failed: {e} (attempt {attempt+1})")

        raise RuntimeError(f"Skill {self.name} failed after 3 attempts")

    # Recognized postcondition tokens — unknown tokens must NOT silently pass.
    _KNOWN_POSTCONDITIONS = frozenset({
        "result_is_valid_json",
        "result_is_non_empty",
    })

    def _verify_postconditions(self, result: Any) -> bool:
        """Verify that the result meets postconditions.

        Recognized conditions are evaluated explicitly.
        Unknown condition tokens are treated as a verification failure
        (fail-closed) and logged, preventing silent policy bypass.
        """
        if not self.postconditions:
            return True  # No postconditions = always pass

        result_str = str(result).lower()
        for condition in self.postconditions:
            if condition == "result_is_valid_json":
                try:
                    import json
                    json.loads(str(result))
                except (json.JSONDecodeError, TypeError):
                    return False
            elif condition == "result_is_non_empty":
                if not result or not str(result).strip():
                    return False
            elif condition in self._KNOWN_POSTCONDITIONS:
                # Future-known conditions handled here
                continue
            else:
                # Unknown condition token — fail closed and alert
                logger.error(
                    f"Skill {self.name}: unknown postcondition '{condition}' — "
                    f"treating as verification failure to prevent silent bypass"
                )
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        v3: Includes Voyager-style executable fields (executable_code,
        preconditions, postconditions, error_patterns) to ensure
        behavioral round-trip on persistence reload.
        """
        return {
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type,
            "definition": self.definition,
            "metadata": self.metadata.to_dict(),
            "version": self.version,
            "inputs": self.inputs,
            "outputs": self.outputs,
            # v3: Voyager-style executable fields (must round-trip)
            "executable_code": self.executable_code,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "error_patterns": self.error_patterns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillEntry":
        """Deserialize from dictionary.

        Backward-compatible: if executable fields are missing from older
        serialized data, defaults are used via .get().
        """
        metadata_data = data.pop("metadata", {})
        # Backward-compatible defaults for Voyager-style fields
        executable_code = data.pop("executable_code", "")
        preconditions = data.pop("preconditions", [])
        postconditions = data.pop("postconditions", [])
        error_patterns = data.pop("error_patterns", {})
        return cls(
            metadata=SkillMetadata.from_dict(metadata_data),
            executable_code=executable_code,
            preconditions=preconditions,
            postconditions=postconditions,
            error_patterns=error_patterns,
            **data,
        )


class SkillStore:
    """Persistent skill storage with hash-based O(1) lookup.

    Inspired by Losion's Engram Memory, which uses hash-based buckets
    for O(1) fact retrieval. SkillStore applies the same principle to
    skill definitions: each skill name is hashed to a bucket index,
    enabling constant-time lookup regardless of store size.

    Storage layout:
        store_dir/
        ├── index.json          # Name → hash_key mapping
        ├── meta.json           # Global store metadata
        └── skills/
            ├── abc12345.json   # Skill entry (named by hash prefix)
            ├── def67890.json
            └── ...

    Thread Safety:
        All public methods are protected by a threading.Lock to ensure
        safe concurrent access from multiple agent threads.

    Args:
        store_dir: Directory for persistent storage.
        auto_save: Whether to auto-save after every write operation.
        max_skills: Maximum number of skills to store (0 = unlimited).
        ttl_seconds: Time-to-live for skills in seconds (0 = no expiry).
    """

    def __init__(
        self,
        store_dir: str = "~/.losion/skills",
        auto_save: bool = True,
        max_skills: int = 0,
        ttl_seconds: float = 0,
    ) -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.auto_save = auto_save
        self.max_skills = max_skills
        self.ttl_seconds = ttl_seconds

        # In-memory index: name → SkillEntry
        self._skills: Dict[str, SkillEntry] = {}
        # Hash index: hash_key → name (for O(1) lookup)
        self._hash_index: Dict[str, str] = {}
        # Domain index: domain → set of names
        self._domain_index: Dict[str, Set[str]] = {}
        # Thread lock
        self._lock = threading.Lock()

        # Load from disk if exists
        self._load()

    def _load(self) -> None:
        """Load skills from persistent storage.

        On corruption, preserves the broken file for forensic analysis
        (renamed to .corrupt.<timestamp>) instead of silently discarding data.
        The in-memory store is still reset so the process can continue, but
        the error is surfaced via logging so operators can investigate.
        """
        skills_dir = self.store_dir / "skills"
        if not skills_dir.exists():
            return

        index_path = self.store_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                for name, hash_key in index.items():
                    skill_path = skills_dir / f"{hash_key}.json"
                    if skill_path.exists():
                        with open(skill_path, "r", encoding="utf-8") as f:
                            entry = SkillEntry.from_dict(json.load(f))
                        self._skills[name] = entry
                        self._hash_index[hash_key] = name
                        # Rebuild domain index
                        if entry.domain:
                            if entry.domain not in self._domain_index:
                                self._domain_index[entry.domain] = set()
                            self._domain_index[entry.domain].add(name)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                # Corrupt index — preserve broken file, reset in-memory, log loudly
                logger.error(
                    f"SkillStore: corrupt index at {index_path}: {exc!r}. "
                    f"Renaming to .corrupt.<ts> and resetting in-memory store."
                )
                # Rename broken file for forensic recovery
                try:
                    corrupt_name = f"index.json.corrupt.{int(time.time())}"
                    index_path.rename(index_path.parent / corrupt_name)
                except OSError:
                    pass  # Best-effort rename; don't mask the original error
                self._skills = {}
                self._hash_index = {}
                self._domain_index = {}

    def _save(self) -> None:
        """Save skills to persistent storage."""
        skills_dir = self.store_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        # Save index
        index = {name: entry.hash_key for name, entry in self._skills.items()}
        with open(self.store_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        # Save each skill
        for name, entry in self._skills.items():
            skill_path = skills_dir / f"{entry.hash_key}.json"
            with open(skill_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)

        # Save global metadata
        meta = {
            "total_skills": len(self._skills),
            "domains": list(self._domain_index.keys()),
            "last_updated": time.time(),
        }
        with open(self.store_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def lookup(self, name: str) -> Optional[SkillEntry]:
        """Look up a skill by name (O(1) via hash index).

        Args:
            name: Skill name.

        Returns:
            SkillEntry if found, None otherwise.
        """
        with self._lock:
            entry = self._skills.get(name)
            if entry is None:
                return None

            # Check TTL
            if self.ttl_seconds > 0:
                age = time.time() - entry.metadata.updated_at
                if age > self.ttl_seconds:
                    self._remove_skill(name)
                    return None

            return entry

    def lookup_by_hash(self, hash_key: str) -> Optional[SkillEntry]:
        """Look up a skill by hash key (O(1)).

        This mirrors Engram Memory's hash-based retrieval.

        Args:
            hash_key: Hash key (first 16 chars of SHA256).

        Returns:
            SkillEntry if found, None otherwise.
        """
        with self._lock:
            name = self._hash_index.get(hash_key)
            if name is None:
                return None
            return self._skills.get(name)

    def search(
        self,
        query: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        skill_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[SkillEntry]:
        """Search skills by multiple criteria.

        Args:
            query: Text to match against name and description.
            domain: Domain to filter by.
            tags: Tags to filter by (any match).
            skill_type: Skill type to filter by.
            min_confidence: Minimum confidence threshold.
            limit: Maximum number of results.

        Returns:
            List of matching SkillEntry objects, sorted by confidence.
        """
        # Snapshot BOTH structures atomically under lock to prevent
        # torn reads between candidates and domain_index.
        with self._lock:
            candidates = list(self._skills.values())
            domain_index_snapshot: Dict[str, Set[str]] = {
                k: set(v) for k, v in self._domain_index.items()
            }

        # Filter by domain using snapshot only
        if domain is not None:
            domain_names = domain_index_snapshot.get(domain, set())
            candidates = [s for s in candidates if s.name in domain_names]

        # Filter by skill type
        if skill_type is not None:
            candidates = [s for s in candidates if s.skill_type == skill_type]

        # Filter by tags
        if tags is not None:
            tag_set = set(tags)
            candidates = [
                s for s in candidates
                if tag_set.intersection(s.metadata.tags)
            ]

        # Filter by confidence
        if min_confidence > 0:
            candidates = [s for s in candidates if s.metadata.confidence >= min_confidence]

        # Filter by query text
        if query is not None:
            query_lower = query.lower()
            candidates = [
                s for s in candidates
                if query_lower in s.name.lower()
                or query_lower in s.description.lower()
            ]

        # Sort by confidence (highest first)
        candidates.sort(key=lambda s: s.metadata.confidence, reverse=True)

        return candidates[:limit]

    def store(self, entry: SkillEntry) -> None:
        """Store a skill entry.

        If a skill with the same name already exists, it will be updated.

        Args:
            entry: SkillEntry to store.

        Raises:
            ValueError: If max_skills limit would be exceeded.
        """
        with self._lock:
            # Check capacity
            if (
                self.max_skills > 0
                and entry.name not in self._skills
                and len(self._skills) >= self.max_skills
            ):
                # Evict lowest-confidence skill
                self._evict_lowest_confidence()

            # Update domain index
            if entry.domain:
                if entry.domain not in self._domain_index:
                    self._domain_index[entry.domain] = set()
                self._domain_index[entry.domain].add(entry.name)

            # Store
            self._skills[entry.name] = entry
            self._hash_index[entry.hash_key] = entry.name

            if self.auto_save:
                self._save()

    def remove(self, name: str) -> bool:
        """Remove a skill by name.

        Args:
            name: Skill name to remove.

        Returns:
            True if the skill was found and removed, False otherwise.
        """
        with self._lock:
            return self._remove_skill(name)

    def _remove_skill(self, name: str) -> bool:
        """Internal remove (must be called within lock)."""
        entry = self._skills.pop(name, None)
        if entry is None:
            return False

        # Clean up indices
        self._hash_index.pop(entry.hash_key, None)
        if entry.domain and entry.domain in self._domain_index:
            self._domain_index[entry.domain].discard(name)
            if not self._domain_index[entry.domain]:
                del self._domain_index[entry.domain]

        if self.auto_save:
            self._save()

        return True

    def record_usage(self, name: str, success: bool) -> None:
        """Record a skill usage event.

        Updates the skill's metadata with usage statistics and
        recalculates its confidence score.

        Args:
            name: Skill name.
            success: Whether the invocation was successful.
        """
        with self._lock:
            entry = self._skills.get(name)
            if entry is not None:
                entry.metadata.record_usage(success)
                if self.auto_save:
                    self._save()

    def _evict_lowest_confidence(self) -> None:
        """Evict the skill with the lowest confidence score.

        Must be called within self._lock.
        """
        if not self._skills:
            return

        lowest_name = min(
            self._skills,
            key=lambda n: self._skills[n].metadata.confidence,
        )
        self._remove_skill(lowest_name)

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics.

        Returns:
            Dictionary with store stats.
        """
        with self._lock:
            total_skills = len(self._skills)
            total_usage = sum(s.metadata.usage_count for s in self._skills.values())
            total_success = sum(s.metadata.success_count for s in self._skills.values())
            domains = {d: len(names) for d, names in self._domain_index.items()}

            return {
                "total_skills": total_skills,
                "total_usage": total_usage,
                "total_success": total_success,
                "overall_success_rate": total_success / max(total_usage, 1),
                "domains": domains,
                "max_capacity": self.max_skills if self.max_skills > 0 else "unlimited",
            }

    @property
    def size(self) -> int:
        """Number of skills in the store."""
        return len(self._skills)

    def list_skills(self) -> List[str]:
        """List all skill names."""
        with self._lock:
            return list(self._skills.keys())

    def clear(self) -> None:
        """Remove all skills from the store."""
        with self._lock:
            self._skills.clear()
            self._hash_index.clear()
            self._domain_index.clear()
            if self.auto_save:
                self._save()
