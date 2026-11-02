"""In-memory stand-in for `langsmith.sandbox.SandboxClient`, for tests: no network, no shell.

    client = LangSmithClient(StubSandboxClient())  # type: ignore[arg-type]

Covers only what `LangSmithClient` calls. Commands understood: the client's own stat and mkdir
helpers, plus whatever `run_handler` answers. Snapshots copy files, like LangSmith's.
"""

from __future__ import annotations

import copy
import itertools
import posixpath
import shlex
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langsmith.sandbox import (
    CommandTimeoutError,
    ExecutionResult,
    ResourceInUseError,
    ResourceNotFoundError,
)

from langchain_space.langsmith.client import _MKDIR_SCRIPT, _STAT_SCRIPT

RunHandler = Callable[["StubSandbox", str], ExecutionResult]
HOME = "/home/user"


def no_shell(sandbox: StubSandbox, command: str) -> ExecutionResult:
    """Default: anything but the client's helpers is unknown."""
    return ExecutionResult(stdout="", stderr=f"stub cannot run {command!r}", exit_code=127)


@dataclass
class StubSnapshot:
    """A captured copy of a stub sandbox's files and run config."""

    id: str
    name: str
    files: dict[str, bytes]
    dirs: set[str]
    run_config: dict[str, Any] | None
    status: str = "ready"


@dataclass
class StubSandbox:
    """One stub sandbox: files, directories, and every command it was asked to run."""

    name: str
    client: StubSandboxClient
    files: dict[str, bytes] = field(default_factory=dict)
    dirs: set[str] = field(default_factory=lambda: {"/", "/home", HOME, "/tmp"})
    run_config: dict[str, Any] | None = None
    status: str = "ready"
    runs: list[dict[str, Any]] = field(default_factory=list)

    def _path(self, path: str) -> str:
        return posixpath.normpath(path if path.startswith("/") else posixpath.join(HOME, path))

    def run(self, command: str, *, timeout: int = 60, **kwargs: Any) -> ExecutionResult:
        self.runs.append({"command": command, "timeout": timeout, **kwargs})
        words = shlex.split(command)
        if words[:3] == ["sh", "-c", _STAT_SCRIPT]:
            path = self._path(words[4])
            if path in self.dirs:
                return ExecutionResult(stdout="directory\n", stderr="", exit_code=0)
            if path in self.files:
                return ExecutionResult(
                    stdout=f"file {len(self.files[path])}\n", stderr="", exit_code=0
                )
            return ExecutionResult(stdout="", stderr="", exit_code=44)
        if words[:3] == ["sh", "-c", _MKDIR_SCRIPT]:
            path = self._path(words[5])
            if path in self.files:
                return ExecutionResult(stdout="", stderr="mkdir: File exists", exit_code=1)
            while path not in self.dirs:
                self.dirs.add(path)
                path = posixpath.dirname(path)
            return ExecutionResult(stdout="", stderr="", exit_code=0)
        return self.client.run_handler(self, command)

    def write(self, path: str, content: bytes | str, **kwargs: Any) -> None:
        normalized = self._path(path)
        parent = posixpath.dirname(normalized)
        while parent not in self.dirs:  # LangSmith creates parent directories
            self.dirs.add(parent)
            parent = posixpath.dirname(parent)
        self.files[normalized] = content.encode() if isinstance(content, str) else content

    def read(self, path: str, **kwargs: Any) -> bytes:
        normalized = self._path(path)
        if normalized not in self.files:
            raise ResourceNotFoundError(f"File '{path}' not found", resource_type="file")
        return self.files[normalized]


class StubSandboxClient:
    """The subset of `SandboxClient` that `LangSmithClient` uses."""

    def __init__(self, run_handler: RunHandler = no_shell) -> None:
        self.run_handler = run_handler
        self.sandboxes: dict[str, StubSandbox] = {}
        self.snapshots: dict[str, StubSnapshot] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.refuse_snapshot_delete: set[str] = set()
        """Snapshot ids whose delete raises `ResourceInUseError` (once)."""
        self.closed = False
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def _record(self, method: str, **args: Any) -> None:
        with self._lock:
            self.calls.append((method, args))

    def calls_to(self, method: str) -> list[dict[str, Any]]:
        """Arguments of every recorded call to `method`."""
        return [args for name, args in self.calls if name == method]

    def create_sandbox(
        self,
        snapshot_id: str | None = None,
        *,
        snapshot: str | None = None,
        run_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StubSandbox:
        self._record(
            "create_sandbox",
            snapshot_id=snapshot_id,
            snapshot=snapshot,
            run_config=run_config,
            **kwargs,
        )
        with self._lock:
            name = f"box-{next(self._ids):04d}"
        sandbox = StubSandbox(name, self, run_config=run_config)
        if snapshot_id is not None:
            source = self.snapshots.get(snapshot_id)
            if source is None:
                raise ResourceNotFoundError("Snapshot not found", resource_type="snapshot")
            sandbox.files = copy.deepcopy(source.files)
            sandbox.dirs = set(source.dirs)
            sandbox.run_config = copy.deepcopy(source.run_config)
        with self._lock:
            self.sandboxes[name] = sandbox
        return sandbox

    def get_sandbox(self, name: str, **kwargs: Any) -> StubSandbox:
        self._record("get_sandbox", name=name)
        sandbox = self.sandboxes.get(name)
        if sandbox is None:
            raise ResourceNotFoundError(f"Sandbox '{name}' not found", resource_type="sandbox")
        return sandbox

    def delete_sandbox(self, name: str, **kwargs: Any) -> None:
        self._record("delete_sandbox", name=name)
        if self.sandboxes.pop(name, None) is None:
            raise ResourceNotFoundError(f"Sandbox '{name}' not found", resource_type="sandbox")

    def stop_sandbox(self, name: str, **kwargs: Any) -> None:
        self._record("stop_sandbox", name=name)
        self.get_sandbox(name).status = "stopped"

    def start_sandbox(self, name: str, **kwargs: Any) -> StubSandbox:
        self._record("start_sandbox", name=name, **kwargs)
        sandbox = self.get_sandbox(name)
        sandbox.status = "ready"
        return sandbox

    def capture_snapshot(
        self,
        sandbox_name: str,
        name: str,
        *,
        run_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StubSnapshot:
        self._record(
            "capture_snapshot", sandbox_name=sandbox_name, name=name, run_config=run_config
        )
        source = self.get_sandbox(sandbox_name)
        merged = {**(source.run_config or {}), **(run_config or {})} or None
        snapshot = StubSnapshot(
            str(uuid.uuid4()), name, copy.deepcopy(source.files), set(source.dirs), merged
        )
        with self._lock:
            self.snapshots[snapshot.id] = snapshot
        return snapshot

    def delete_snapshot(self, snapshot_id: str, **kwargs: Any) -> None:
        """Delete a snapshot; refuses once for ids in `refuse_snapshot_delete`."""
        self._record("delete_snapshot", snapshot_id=snapshot_id)
        if snapshot_id in self.refuse_snapshot_delete:
            self.refuse_snapshot_delete.discard(snapshot_id)
            raise ResourceInUseError("Snapshot is in use")
        if self.snapshots.pop(snapshot_id, None) is None:
            raise ResourceNotFoundError("Snapshot not found", resource_type="snapshot")

    def close(self) -> None:
        self.closed = True


def timing_out(sandbox: StubSandbox, command: str) -> ExecutionResult:
    """A run handler whose every command times out."""
    raise CommandTimeoutError("command timed out", timeout=1)
