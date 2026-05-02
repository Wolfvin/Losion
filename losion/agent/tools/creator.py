"""
Tool Creator — Auto-generate tools when no suitable tool exists.

When the agent encounters a task for which no suitable tool exists in the
registry, the ToolCreator can generate one. This is the "cari tools yang
relevan, jika tidak maka akan bikin tools" feature.

The creation process:
1. Analyze: Determine what kind of tool is needed
2. Search: Check if any existing tool partially matches
3. Design: Design the tool interface and logic
4. Implement: Generate the tool implementation
5. Register: Add the tool to the registry for future use
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from losion.agent.tools.registry import ToolEntry, ToolRegistry, ToolSafety
from losion.agent.tools.web_search import WebSearchInterface

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Specification for a tool to be created.

    Attributes:
        name: Tool name.
        description: What the tool does.
        domain: Domain classification.
        inputs_schema: Expected inputs description.
        outputs_schema: Expected outputs description.
        safety: Safety classification.
        implementation_hint: Hint for how the tool should be implemented.
    """

    name: str = ""
    description: str = ""
    domain: Optional[str] = None
    inputs_schema: str = ""
    outputs_schema: str = ""
    safety: ToolSafety = ToolSafety.SAFE
    implementation_hint: str = ""


class ToolCreator:
    """Auto-creates tools for the agent layer.

    When no suitable tool exists for a task, the ToolCreator generates
    one based on the task requirements and available context (potentially
    from web search).

    The creation flow:
        1. Analyze the task to determine tool requirements
        2. Search the registry for partial matches
        3. Optionally search the web for implementation ideas
        4. Generate the tool implementation
        5. Register the tool for future use

    Tool types that can be auto-created:
    - Shell command wrappers
    - Python function wrappers
    - API call wrappers
    - File operation tools
    - Data processing tools

    Args:
        registry: ToolRegistry to register created tools.
        web_search: WebSearchInterface for context (optional).
        auto_register: Whether to automatically register created tools.
        default_safety: Default safety level for auto-created tools.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        web_search: Optional[WebSearchInterface] = None,
        auto_register: bool = True,
        default_safety: ToolSafety = ToolSafety.REQUIRES_APPROVAL,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.web_search = web_search or WebSearchInterface()
        self.auto_register = auto_register
        self.default_safety = default_safety

    def create(
        self,
        query: str,
        domain: Optional[str] = None,
        spec: Optional[ToolSpec] = None,
    ) -> Optional[ToolEntry]:
        """Create a new tool for the given requirement.

        Args:
            query: Description of what the tool should do.
            domain: Domain classification.
            spec: Optional pre-defined specification.

        Returns:
            The created ToolEntry, or None if creation failed.
        """
        logger.info(f"Creating tool for: {query} (domain={domain})")

        # === Step 1: Analyze ===
        if spec is None:
            spec = self._analyze_requirements(query, domain)

        if not spec.name:
            logger.warning(f"Could not determine tool spec for: {query}")
            return None

        # === Step 2: Check for existing partial matches ===
        existing = self.registry.search(query=query, domain=domain, executable_only=False)
        if existing:
            best = existing[0]
            logger.info(f"Found partial match: {best.name} — skipping creation")
            return best

        # === Step 3: Search for implementation context ===
        context = self._search_for_context(query, domain)

        # === Step 4: Implement ===
        handler = self._implement_tool(spec, context)

        # === Step 5: Register ===
        entry = ToolEntry(
            name=spec.name,
            description=spec.description,
            handler=handler,
            safety=spec.safety,
            inputs_schema=spec.inputs_schema,
            outputs_schema=spec.outputs_schema,
            domain=spec.domain or domain,
            tags=[domain] if domain else [],
            is_builtin=False,
        )

        if self.auto_register:
            try:
                self.registry.register(entry)
                logger.info(f"Tool created and registered: {spec.name}")
            except ValueError:
                # Tool already exists (race condition)
                existing = self.registry.lookup(spec.name)
                if existing:
                    return existing

        return entry

    def _analyze_requirements(
        self, query: str, domain: Optional[str]
    ) -> ToolSpec:
        """Analyze task requirements to produce a tool specification.

        Args:
            query: What the tool should do.
            domain: Domain context.

        Returns:
            ToolSpec with the analysis results.
        """
        query_lower = query.lower()

        # Determine tool type from query
        is_shell = any(
            kw in query_lower
            for kw in ["run", "execute", "command", "shell", "bash", "terminal"]
        )
        is_api = any(
            kw in query_lower
            for kw in ["api", "endpoint", "request", "http", "fetch", "call"]
        )
        is_file = any(
            kw in query_lower
            for kw in ["file", "read", "write", "save", "load", "path"]
        )
        is_data = any(
            kw in query_lower
            for kw in ["parse", "transform", "convert", "process", "analyze"]
        )

        # Generate name
        name_parts = []
        if domain:
            name_parts.append(domain)
        name_parts.append(query_lower.split()[0] if query_lower.split() else "tool")
        name = "-".join(name_parts)[:40]

        # Determine safety level
        if is_shell:
            safety = ToolSafety.REQUIRES_APPROVAL
        elif is_api or is_file:
            safety = ToolSafety.REQUIRES_APPROVAL
        else:
            safety = self.default_safety

        # Build spec
        spec = ToolSpec(
            name=name,
            description=f"Auto-created tool: {query}",
            domain=domain,
            safety=safety,
        )

        if is_shell:
            spec.implementation_hint = "shell"
            spec.inputs_schema = "command: str — Shell command to execute"
            spec.outputs_schema = "TerminalResult — Command execution result"
        elif is_api:
            spec.implementation_hint = "api"
            spec.inputs_schema = "url: str, method: str, params: dict"
            spec.outputs_schema = "dict — API response"
        elif is_file:
            spec.implementation_hint = "file"
            spec.inputs_schema = "path: str, content: str (for write)"
            spec.outputs_schema = "str — File content or operation result"
        elif is_data:
            spec.implementation_hint = "data"
            spec.inputs_schema = "data: Any, format: str"
            spec.outputs_schema = "Any — Processed data"
        else:
            spec.implementation_hint = "generic"
            spec.inputs_schema = "input_data: Any"
            spec.outputs_schema = "Any — Tool result"

        return spec

    def _search_for_context(
        self, query: str, domain: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Search the web for implementation context.

        Args:
            query: Tool requirement.
            domain: Domain context.

        Returns:
            List of context dictionaries from search results.
        """
        search_query = f"python tool {query}"
        if domain:
            search_query += f" {domain}"

        try:
            results = self.web_search.search(query=search_query, num_results=3)
            return [
                {"title": r.title, "snippet": r.snippet, "url": r.url}
                for r in results
                if r.is_valid
            ]
        except Exception as e:
            logger.warning(f"Web search for tool context failed: {e}")
            return []

    def _implement_tool(
        self, spec: ToolSpec, context: List[Dict[str, Any]]
    ) -> Optional[Callable]:
        """Generate a tool handler function.

        Creates a callable that implements the tool's functionality.
        For safety, auto-created tools are wrapped with error handling
        and logging.

        Args:
            spec: Tool specification.
            context: Web search context.

        Returns:
            Callable handler, or None if implementation fails.
        """
        hint = spec.implementation_hint

        if hint == "shell":
            def handler(command: str = "", **kwargs) -> Dict[str, Any]:
                """Execute a shell command (sandboxed)."""
                from losion.agent.tools.terminal import SandboxedTerminal
                terminal = SandboxedTerminal()
                result = terminal.execute(command)
                return {
                    "success": result.success,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                }
            return handler

        elif hint == "api":
            def handler(url: str = "", method: str = "GET", params: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
                """Make an API request."""
                try:
                    import urllib.request
                    import json
                    req = urllib.request.Request(url, method=method)
                    with urllib.request.urlopen(req, timeout=30) as response:
                        data = response.read().decode("utf-8")
                        try:
                            return {"success": True, "data": json.loads(data)}
                        except json.JSONDecodeError:
                            return {"success": True, "data": data}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            return handler

        elif hint == "file":
            def handler(path: str = "", content: Optional[str] = None, mode: str = "r", **kwargs) -> Dict[str, Any]:
                """Read or write a file."""
                try:
                    if mode == "r" or content is None:
                        with open(path, "r", encoding="utf-8") as f:
                            return {"success": True, "data": f.read()}
                    else:
                        with open(path, mode, encoding="utf-8") as f:
                            f.write(content)
                        return {"success": True, "message": f"Written to {path}"}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            return handler

        elif hint == "data":
            def handler(data: Any = None, format: str = "auto", **kwargs) -> Dict[str, Any]:
                """Process/transform data."""
                # Generic data processing scaffold
                return {"success": True, "data": data, "format": format}
            return handler

        else:
            # Generic tool
            def handler(input_data: Any = None, **kwargs) -> Dict[str, Any]:
                """Generic tool handler."""
                return {
                    "success": True,
                    "result": input_data,
                    "tool": spec.name,
                }
            return handler
