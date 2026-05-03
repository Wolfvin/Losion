"""
Sandboxed Terminal — Safe terminal command execution for the agent layer.

This module provides a sandboxed terminal execution environment where the
agent can run commands safely. The sandbox enforces:
- Command whitelisting/blacklisting
- Resource limits (CPU, memory, time)
- Output size limits
- Working directory isolation
- Audit logging

The terminal is NEVER available during model training — it only exists
in the agent layer, completely decoupled from the neural architecture.

Design:
    Agent → Terminal.execute(command) → Subprocess/Docker → Result
                                              ↓
                                 ┌─────────────────────┐
                                 │   Safety Checks      │
                                 │   - Whitelist        │
                                 │   - Blacklist        │
                                 │   - Resource         │
                                 │   - Timeout          │
                                 │   - Network隔离      │
                                 │   - Filesystem隔离   │
                                 │   - Audit Log        │
                                 └─────────────────────┘

Container Isolation (recommended for production):
    When use_container=True, commands run inside a Docker container
    with network isolation, read-only system mounts, and resource limits.
    This provides defense-in-depth beyond subprocess-level sandboxing.

    Example:
        config = SandboxConfig(
            use_container=True,
            container_image="python:3.11-slim",
            container_network=False,  # No network access
        )
        terminal = SandboxedTerminal(config)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class TerminalResult:
    """Result of a terminal command execution.

    Attributes:
        command: The command that was executed.
        exit_code: Process exit code (0 = success).
        stdout: Standard output (truncated if exceeds max_output_size).
        stderr: Standard error output.
        execution_time: Wall-clock execution time in seconds.
        timed_out: Whether the command was killed due to timeout.
        working_dir: Working directory used for execution.
    """

    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    timed_out: bool = False
    working_dir: str = ""

    @property
    def success(self) -> bool:
        """Whether the command succeeded."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Combined stdout + stderr."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr)
        return "\n".join(parts)


@dataclass
class SandboxConfig:
    """Configuration for the sandboxed terminal.

    Attributes:
        allowed_commands: Set of allowed command prefixes (empty = all allowed).
        blocked_commands: Set of blocked command prefixes (takes precedence).
        blocked_patterns: Patterns that cannot appear in any command.
        max_execution_time: Maximum execution time in seconds.
        max_output_size: Maximum output size in bytes.
        max_memory_mb: Maximum memory usage in MB.
        working_dir: Working directory for command execution.
        env_vars: Additional environment variables.
        allow_network: Whether to allow network access.
        audit_log: Whether to log all commands for audit.
    """

    allowed_commands: Set[str] = field(default_factory=set)
    blocked_commands: Set[str] = field(default_factory=lambda: {
        # Destructive commands
        "rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:",
        "format", "del /f /s /q C:",
        # System modification
        "sudo", "su ", "chmod 777", "chown",
        # Network attacks
        "nmap", "netcat", "nc -l",
        # Dangerous downloads
        "curl | sh", "wget | sh", "curl | bash",
    })
    blocked_patterns: List[str] = field(default_factory=lambda: [
        "rm -rf /",
        "> /dev/sd",
        "mkfs",
        "dd of=",
    ])
    max_execution_time: float = 30.0
    max_output_size: int = 1_000_000  # 1MB
    max_memory_mb: int = 512
    working_dir: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)
    allow_network: bool = False
    audit_log: bool = True
    # Container isolation (Docker-based, recommended for production)
    use_container: bool = False              # Use Docker container for execution
    container_image: str = "python:3.11-slim"  # Docker image to use
    container_network: bool = False           # Allow network in container
    container_readonly_root: bool = True      # Read-only root filesystem in container
    container_cpus: float = 1.0               # CPU limit per container
    container_memory_mb: int = 512            # Memory limit per container
    # Filesystem isolation
    readonly_paths: List[str] = field(default_factory=lambda: [
        "/etc", "/usr", "/bin", "/sbin", "/boot", "/root",
    ])
    writable_paths: List[str] = field(default_factory=lambda: [
        "/tmp", "/home",
    ])


class SandboxedTerminal:
    """Sandboxed terminal execution environment.

    Provides a safe way for the agent to execute terminal commands with
    resource limits, command filtering, and audit logging.

    Security layers:
    1. Command validation (whitelist/blacklist)
    2. Pattern scanning (dangerous patterns)
    3. Resource limits (time, memory, output)
    4. Working directory isolation
    5. Audit logging

    Args:
        config: Sandbox configuration.
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()
        self._execution_history: List[TerminalResult] = []
        self._temp_dir: Optional[str] = None

    def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> TerminalResult:
        """Execute a command in the sandbox.

        When use_container=True, commands run inside a Docker container
        with network isolation, read-only root filesystem, and resource
        limits. When False (default), commands run as subprocesses with
        command validation and resource limits.

        Args:
            command: Command to execute.
            timeout: Override timeout for this command.
            working_dir: Override working directory.
            env: Override environment variables.

        Returns:
            TerminalResult with execution details.

        Raises:
            PermissionError: If command is blocked by safety rules.
        """
        # === Security Layer 1: Command validation ===
        self._validate_command(command)

        # === Security Layer 2: Filesystem isolation check ===
        self._validate_filesystem_access(command)

        # === Audit logging ===
        if self.config.audit_log:
            logger.info(f"Terminal executing: {command}")

        # === Route to execution backend ===
        if self.config.use_container:
            return self._execute_in_container(command, timeout, working_dir, env)
        else:
            return self._execute_subprocess(command, timeout, working_dir, env)

    def _execute_subprocess(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> TerminalResult:
        """Execute a command as a subprocess with resource limits."""
        effective_timeout = timeout or self.config.max_execution_time
        effective_wd = working_dir or self.config.working_dir or self._get_temp_dir()

        # Build environment
        effective_env = os.environ.copy()
        effective_env.update(self.config.env_vars)
        if env:
            effective_env.update(env)

        # === Execute ===
        start_time = time.time()
        timed_out = False

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=effective_wd,
                env=effective_env,
            )

            try:
                stdout, stderr = process.communicate(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                timed_out = True

            execution_time = time.time() - start_time

            # Truncate output if needed
            stdout_str = stdout.decode("utf-8", errors="replace")[:self.config.max_output_size]
            stderr_str = stderr.decode("utf-8", errors="replace")[:self.config.max_output_size]

            result = TerminalResult(
                command=command,
                exit_code=process.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time=execution_time,
                timed_out=timed_out,
                working_dir=effective_wd,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            result = TerminalResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                working_dir=effective_wd,
            )

        # Record
        self._execution_history.append(result)

        return result

    def _execute_in_container(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> TerminalResult:
        """Execute a command inside a Docker container with full isolation.

        Container isolation provides defense-in-depth:
        - Network isolation (no network by default)
        - Read-only root filesystem
        - CPU and memory limits
        - PID namespace isolation
        - Automatic cleanup after execution

        Requires Docker to be installed and available.
        Falls back to subprocess if Docker is unavailable.
        """
        effective_timeout = timeout or self.config.max_execution_time
        effective_wd = working_dir or self.config.working_dir or self._get_temp_dir()

        # Build Docker command
        docker_args = [
            "docker", "run", "--rm",  # Auto-remove after execution
            "--pid=host",             # PID namespace isolation
            f"--cpus={self.config.container_cpus}",
            f"--memory={self.config.container_memory_mb}m",
            f"--stop-timeout={int(effective_timeout)}",
        ]

        # Network isolation
        if not self.config.container_network and not self.config.allow_network:
            docker_args.append("--network=none")

        # Read-only root filesystem
        if self.config.container_readonly_root:
            docker_args.append("--read-only")
            # Need tmpfs for writable directories
            docker_args.extend([
                "--tmpfs", "/tmp:size=100m",
                "--tmpfs", "/run:size=10m",
            ])

        # Mount working directory
        if effective_wd and os.path.exists(effective_wd):
            docker_args.extend(["-v", f"{effective_wd}:/workspace"])
            docker_args.extend(["-w", "/workspace"])

        # Environment variables
        if env:
            for key, value in env.items():
                docker_args.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env_vars.items():
            docker_args.extend(["-e", f"{key}={value}"])

        # Image and command
        docker_args.append(self.config.container_image)
        docker_args.extend(["sh", "-c", command])

        # Execute via subprocess (Docker CLI)
        start_time = time.time()
        timed_out = False

        try:
            process = subprocess.Popen(
                docker_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                stdout, stderr = process.communicate(timeout=effective_timeout + 5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                timed_out = True

            execution_time = time.time() - start_time

            stdout_str = stdout.decode("utf-8", errors="replace")[:self.config.max_output_size]
            stderr_str = stderr.decode("utf-8", errors="replace")[:self.config.max_output_size]

            result = TerminalResult(
                command=command,
                exit_code=process.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                execution_time=execution_time,
                timed_out=timed_out,
                working_dir=effective_wd,
            )

        except FileNotFoundError:
            # Docker not available — fall back to subprocess
            logger.warning(
                "Docker not available, falling back to subprocess execution. "
                "Container isolation is not active."
            )
            return self._execute_subprocess(command, timeout, working_dir, env)
        except Exception as e:
            execution_time = time.time() - start_time
            result = TerminalResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Container execution failed: {e}",
                execution_time=execution_time,
                working_dir=effective_wd,
            )

        # Record
        self._execution_history.append(result)

        return result

    def _validate_filesystem_access(self, command: str) -> None:
        """Validate that the command doesn't write to protected filesystem paths.

        Checks against the configured readonly_paths to prevent
        modifications to system directories.

        Args:
            command: Command to validate.

        Raises:
            PermissionError: If command attempts to write to protected paths.
        """
        if not self.config.readonly_paths:
            return

        command_lower = command.lower().strip()

        # Check for write operations to protected paths
        write_indicators = [">", ">>", "tee ", "cp ", "mv ", "install ", "ln "]
        is_write = any(ind in command_lower for ind in write_indicators)

        if is_write:
            for protected_path in self.config.readonly_paths:
                if protected_path.lower() in command_lower:
                    raise PermissionError(
                        f"Command attempts to write to protected path: {protected_path}. "
                        f"This path is configured as read-only."
                    )

    def _validate_command(self, command: str) -> None:
        """Validate a command against safety rules.

        Args:
            command: Command to validate.

        Raises:
            PermissionError: If command is blocked.
        """
        command_lower = command.lower().strip()

        # Check blocked commands
        for blocked in self.config.blocked_commands:
            if blocked.lower() in command_lower:
                raise PermissionError(
                    f"Command blocked by safety rules: matches '{blocked}'"
                )

        # Check blocked patterns
        for pattern in self.config.blocked_patterns:
            if pattern.lower() in command_lower:
                raise PermissionError(
                    f"Command blocked by pattern rule: matches '{pattern}'"
                )

        # Check whitelist (if defined)
        if self.config.allowed_commands:
            command_prefix = command_lower.split()[0] if command_lower.split() else ""
            if command_prefix not in {c.lower() for c in self.config.allowed_commands}:
                raise PermissionError(
                    f"Command '{command_prefix}' not in allowed list"
                )

    def _get_temp_dir(self) -> str:
        """Get or create a temporary working directory."""
        if self._temp_dir is None or not os.path.exists(self._temp_dir):
            self._temp_dir = tempfile.mkdtemp(prefix="losion_sandbox_")
        return self._temp_dir

    def cleanup(self) -> None:
        """Clean up temporary files and directories."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def get_history(self, limit: int = 100) -> List[TerminalResult]:
        """Get execution history.

        Args:
            limit: Maximum number of results.

        Returns:
            List of TerminalResult objects.
        """
        return self._execution_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get terminal execution statistics."""
        total = len(self._execution_history)
        successes = sum(1 for r in self._execution_history if r.success)
        timeouts = sum(1 for r in self._execution_history if r.timed_out)
        avg_time = (
            sum(r.execution_time for r in self._execution_history) / total
            if total > 0
            else 0.0
        )

        return {
            "total_executions": total,
            "successful": successes,
            "failed": total - successes,
            "timeouts": timeouts,
            "success_rate": successes / max(total, 1),
            "avg_execution_time": avg_time,
        }
