"""
Tool Registry — Dynamic tool discovery and management.

The ToolRegistry manages available tools for the agent layer. Unlike skills
(which are knowledge/procedure templates), tools are executable capabilities
with defined interfaces. Think of skills as "knowing how" and tools as
"being able to do."

Tools in Losion's agent layer:
- Terminal execution (sandboxed)
- Web search
- File I/O
- Code execution
- API calls
- Any callable with a defined interface

Tools are registered with metadata describing their capabilities, inputs,
outputs, and safety requirements. The registry supports:
- O(1) lookup by name
- Search by capability description
- Safety classification (safe, requires_approval, dangerous)
- Dependency tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ToolSafety(Enum):
    """Safety classification for tools.

    - SAFE: Can be executed without any approval
    - REQUIRES_APPROVAL: Needs user/system approval before execution
    - DANGEROUS: Should only be used in extreme cases with full audit trail
    """

    SAFE = "safe"
    REQUIRES_APPROVAL = "requires_approval"
    DANGEROUS = "dangerous"


@dataclass
class ToolEntry:
    """A registered tool entry.

    Attributes:
        name: Unique tool name (kebab-case).
        description: Human-readable description of what this tool does.
        handler: Callable that implements the tool.
        safety: Safety classification.
        inputs_schema: Description of expected inputs.
        outputs_schema: Description of expected outputs.
        domain: Domain this tool is useful for.
        tags: Searchable tags.
        is_builtin: Whether this is a built-in tool.
        created_at: Creation timestamp.
        usage_count: Number of times this tool has been invoked.
        success_count: Number of successful invocations.
    """

    name: str
    description: str
    handler: Optional[Callable] = None
    safety: ToolSafety = ToolSafety.SAFE
    inputs_schema: str = ""
    outputs_schema: str = ""
    domain: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    is_builtin: bool = False
    created_at: float = 0.0
    usage_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction [0.0, 1.0]."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    @property
    def is_executable(self) -> bool:
        """Whether this tool has a callable handler."""
        return self.handler is not None

    def record_usage(self, success: bool) -> None:
        """Record a tool usage event."""
        self.usage_count += 1
        if success:
            self.success_count += 1


class ToolRegistry:
    """Dynamic tool registry for the Losion Agent Layer.

    Provides O(1) lookup by name and fuzzy search by capability.
    Tools are classified by safety level and domain.

    Usage:
        registry = ToolRegistry()

        # Register a tool
        registry.register(ToolEntry(
            name="python-exec",
            description="Execute Python code in a sandbox",
            handler=sandboxed_python_exec,
            safety=ToolSafety.REQUIRES_APPROVAL,
            domain="code",
        ))

        # Look up a tool
        tool = registry.lookup("python-exec")

        # Search for tools
        tools = registry.search(query="execute code", domain="code")

        # Execute a tool
        result = registry.execute("python-exec", code="print('hello')")

    Args:
        allow_dangerous: Whether dangerous tools can be registered.
        require_approval_callback: Callback for tools that require approval.
    """

    def __init__(
        self,
        allow_dangerous: bool = False,
        require_approval_callback: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._tools: Dict[str, ToolEntry] = {}
        self._domain_index: Dict[str, Set[str]] = {}
        self._allow_dangerous = allow_dangerous
        self._approval_callback = require_approval_callback

    def register(self, entry: ToolEntry) -> None:
        """Register a tool.

        Args:
            entry: ToolEntry to register.

        Raises:
            ValueError: If a tool with the same name already exists.
            PermissionError: If the tool is dangerous and allow_dangerous is False.
        """
        if entry.name in self._tools:
            raise ValueError(f"Tool '{entry.name}' already registered")

        if entry.safety == ToolSafety.DANGEROUS and not self._allow_dangerous:
            raise PermissionError(
                f"Tool '{entry.name}' is classified as DANGEROUS. "
                f"Set allow_dangerous=True to register dangerous tools."
            )

        self._tools[entry.name] = entry

        # Update domain index
        if entry.domain:
            if entry.domain not in self._domain_index:
                self._domain_index[entry.domain] = set()
            self._domain_index[entry.domain].add(entry.name)

        logger.info(f"Tool registered: {entry.name} (safety={entry.safety.value})")

    def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name.

        Returns:
            True if the tool was found and removed.
        """
        entry = self._tools.pop(name, None)
        if entry is None:
            return False

        if entry.domain and entry.domain in self._domain_index:
            self._domain_index[entry.domain].discard(name)

        logger.info(f"Tool unregistered: {name}")
        return True

    def lookup(self, name: str) -> Optional[ToolEntry]:
        """Look up a tool by name (O(1)).

        Args:
            name: Tool name.

        Returns:
            ToolEntry or None.
        """
        return self._tools.get(name)

    def search(
        self,
        query: Optional[str] = None,
        domain: Optional[str] = None,
        safety: Optional[ToolSafety] = None,
        tags: Optional[List[str]] = None,
        executable_only: bool = True,
        limit: int = 10,
    ) -> List[ToolEntry]:
        """Search for tools by multiple criteria.

        Args:
            query: Text to match against name and description.
            domain: Domain filter.
            safety: Safety level filter.
            tags: Tags to match (any match).
            executable_only: Only return tools with callable handlers.
            limit: Maximum number of results.

        Returns:
            List of matching ToolEntry objects.
        """
        candidates = list(self._tools.values())

        # Filter by domain
        if domain is not None:
            domain_names = self._domain_index.get(domain, set())
            candidates = [t for t in candidates if t.name in domain_names]

        # Filter by safety
        if safety is not None:
            candidates = [t for t in candidates if t.safety == safety]

        # Filter by tags
        if tags is not None:
            tag_set = set(tags)
            candidates = [
                t for t in candidates
                if tag_set.intersection(t.tags)
            ]

        # Filter by executability
        if executable_only:
            candidates = [t for t in candidates if t.is_executable]

        # Filter by query text
        if query is not None:
            query_lower = query.lower()
            candidates = [
                t for t in candidates
                if query_lower in t.name.lower()
                or query_lower in t.description.lower()
            ]

        # Sort by success rate
        candidates.sort(key=lambda t: t.success_rate, reverse=True)

        return candidates[:limit]

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a registered tool.

        Handles safety checks and approval flow before execution.

        Args:
            name: Tool name.
            **kwargs: Arguments to pass to the tool handler.

        Returns:
            Result of the tool execution.

        Raises:
            KeyError: If tool not found.
            PermissionError: If tool requires approval and is denied.
            RuntimeError: If tool has no handler.
        """
        entry = self._tools.get(name)
        if entry is None:
            raise KeyError(f"Tool '{name}' not found")

        # Safety check
        if entry.safety == ToolSafety.REQUIRES_APPROVAL:
            if self._approval_callback and not self._approval_callback(name):
                raise PermissionError(
                    f"Tool '{name}' requires approval and was denied"
                )

        if entry.safety == ToolSafety.DANGEROUS:
            logger.warning(f"Executing DANGEROUS tool: {name}")
            if self._approval_callback and not self._approval_callback(name):
                raise PermissionError(
                    f"Tool '{name}' is DANGEROUS and was denied"
                )

        # Execute
        if entry.handler is None:
            raise RuntimeError(f"Tool '{name}' has no handler")

        try:
            result = entry.handler(**kwargs)
            entry.record_usage(success=True)
            return result
        except Exception as e:
            entry.record_usage(success=False)
            raise

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        safety_counts = {}
        for entry in self._tools.values():
            key = entry.safety.value
            safety_counts[key] = safety_counts.get(key, 0) + 1

        return {
            "total_tools": len(self._tools),
            "executable_tools": sum(1 for t in self._tools.values() if t.is_executable),
            "domains": {d: len(n) for d, n in self._domain_index.items()},
            "safety_distribution": safety_counts,
        }
