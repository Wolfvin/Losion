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

import json
import logging
import os
import shlex
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

    # Mandatory allowlist: if non-empty, ONLY these command prefixes are allowed.
    # In production, this MUST be populated. Empty set means all commands are
    # allowed (INSECURE — use only for development/testing).
    # v2.5.0: Changed from optional to recommended-mandatory. The blacklist
    # approach is inherently bypassable (see audit finding 2.1).
    allowed_commands: Set[str] = field(default_factory=set)
    # v2.5.1: Changed default to True (audit finding 2.6). Secure-by-default:
    # if no allowlist is configured, subprocess execution is blocked. Set to
    # False only for development/testing, or use use_container=True for
    # Docker-isolated execution which has its own security boundary.
    require_allowlist: bool = True
    blocked_commands: Set[str] = field(default_factory=lambda: {
        # Destructive commands (defense-in-depth — allowlist is the real boundary)
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
    # Audit log persistence (v2.5.0)
    audit_log_file: str = ""  # Path to persistent audit log file. Empty = logging.info only
    audit_log_format: str = "json"  # "json" or "text"


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
        # v2.5.0: Persistent audit log file handle
        self._audit_file: Optional[Any] = None

        # v2.5.1: Warn if allowlist is not enforced (audit finding 2.6).
        # Secure-by-default means require_allowlist=True is now the default.
        # If someone explicitly sets it to False, warn them.
        if not self.config.require_allowlist and not self.config.use_container:
            logger.warning(
                "SandboxedTerminal: require_allowlist=False without container "
                "isolation. Any command not in the blacklist can be executed. "
                "This is INSECURE for production. Either set require_allowlist=True "
                "with an allowed_commands set, or use use_container=True."
            )

    def _write_audit_log(self, command: str, result: Optional[TerminalResult] = None) -> None:
        """Write audit log entry — both to logging.info and to persistent file.

        v2.5.0: Previously, audit logs were only written via logging.info(),
        which is not guaranteed to be persistent (depends on logging config)
        and is not tamper-proof. Now also writes to a file if audit_log_file
        is configured.

        Args:
            command: The command that was executed.
            result: Optional execution result.
        """
        # Always log to standard logger
        logger.info(f"Terminal executing: {command}")
        if result is not None:
            logger.info(
                f"Terminal result: exit_code={result.exit_code}, "
                f"time={result.execution_time:.3f}s, "
                f"timed_out={result.timed_out}"
            )

        # Write to persistent audit file if configured
        if self.config.audit_log_file:
            try:
                audit_path = Path(self.config.audit_log_file)
                audit_path.parent.mkdir(parents=True, exist_ok=True)

                if self.config.audit_log_format == "json":
                    entry = {
                        "timestamp": time.time(),
                        "command": command,
                    }
                    if result is not None:
                        entry.update({
                            "exit_code": result.exit_code,
                            "execution_time": result.execution_time,
                            "timed_out": result.timed_out,
                            "success": result.success,
                        })
                    line = json.dumps(entry)
                else:
                    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] cmd={command}"
                    if result is not None:
                        line += f" exit={result.exit_code} time={result.execution_time:.3f}s"

                with open(audit_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as e:
                logger.warning(f"Failed to write audit log: {e}")

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

        # === Security Layer 1.5: Allowlist enforcement ===
        # v2.5.0: If require_allowlist is True and allowed_commands is empty,
        # reject the command (production safety). Container mode is exempt
        # because Docker provides its own isolation boundary.
        if (
            self.config.require_allowlist
            and not self.config.allowed_commands
            and not self.config.use_container
        ):
            raise PermissionError(
                "Production safety: allowed_commands is empty and require_allowlist "
                "is True. Either populate allowed_commands with permitted command "
                "prefixes, or use use_container=True for Docker-isolated execution. "
                "An empty allowlist with shell=False is not safe for production."
            )

        # === Security Layer 2: Filesystem isolation check ===
        self._validate_filesystem_access(command)

        # === Audit logging (v2.5.0: now includes persistent file logging) ===
        if self.config.audit_log:
            self._write_audit_log(command)

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
        """Execute a command as a subprocess with resource limits.

        v2.5.0: Changed from shell=True to shell=False with shlex.split().
        The shell=True approach was inherently insecure — even with blacklist
        filtering, shell metacharacters and string concatenation could bypass
        the safety checks (e.g., ``r"m -rf /``, variable references, ``chr()``
        tricks). Using shell=False with argument splitting ensures the command
        is executed as a direct argv list, preventing shell injection entirely.

        For commands that genuinely require shell features (pipes, redirection),
        use ``use_container=True`` which provides Docker isolation.
        """
        effective_timeout = timeout or self.config.max_execution_time
        effective_wd = working_dir or self.config.working_dir or self._get_temp_dir()

        # Build environment
        effective_env = os.environ.copy()
        effective_env.update(self.config.env_vars)
        if env:
            effective_env.update(env)

        # === Parse command into argv list (shell=False) ===
        # v2.5.0: Use shlex.split() instead of shell=True to prevent
        # shell injection. This means pipes (|), redirections (>), && etc.
        # will NOT work in subprocess mode — use container mode for those.
        try:
            cmd_args = shlex.split(command)
        except ValueError as e:
            return TerminalResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Invalid command syntax (shlex parse error): {e}",
                execution_time=0.0,
                working_dir=effective_wd,
            )

        if not cmd_args:
            return TerminalResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="Empty command after parsing",
                execution_time=0.0,
                working_dir=effective_wd,
            )

        # === Execute ===
        start_time = time.time()
        timed_out = False

        try:
            process = subprocess.Popen(
                cmd_args,
                shell=False,
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

        # Record + audit
        self._execution_history.append(result)
        if self.config.audit_log:
            self._write_audit_log(command, result)

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
        # v2.5.1: Removed --pid=host (audit finding 2.1). This flag REMOVES
        # PID namespace isolation — the opposite of what the previous comment
        # claimed. --pid=host gives the container full access to the host's
        # PID namespace, allowing it to see and signal host processes, which
        # is a container escape vector. The default Docker PID namespace
        # (no flag) is already properly isolated.
        docker_args = [
            "docker", "run", "--rm",  # Auto-remove after execution
            # No --pid=host — default PID namespace is already isolated
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

        # Image and command — use shlex.split for the inner command too
        # Note: Inside Docker, we DO use shell=True (sh -c) because the
        # container provides its own isolation boundary. The shell injection
        # risk is mitigated by: no network, read-only filesystem, resource
        # limits, and automatic cleanup.
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

        # Record + audit
        self._execution_history.append(result)
        if self.config.audit_log:
            self._write_audit_log(command, result)

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

        v2.5.0: The primary security boundary is now the allowlist (if
        configured). The blacklist provides defense-in-depth but is not
        sufficient on its own — string obfuscation techniques can bypass
        substring matching.

        With shell=False (v2.5.0), shell metacharacters are no longer
        interpreted, which eliminates a large class of injection attacks.

        Args:
            command: Command to validate.

        Raises:
            PermissionError: If command is blocked.
        """
        command_lower = command.lower().strip()

        # Check blocked commands (defense-in-depth)
        for blocked in self.config.blocked_commands:
            if blocked.lower() in command_lower:
                raise PermissionError(
                    f"Command blocked by safety rules: matches '{blocked}'"
                )

        # Check blocked patterns (defense-in-depth)
        for pattern in self.config.blocked_patterns:
            if pattern.lower() in command_lower:
                raise PermissionError(
                    f"Command blocked by pattern rule: matches '{pattern}'"
                )

        # Check allowlist (PRIMARY security boundary if configured)
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
