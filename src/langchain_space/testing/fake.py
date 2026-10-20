"""In-memory `SpaceClient` for tests. Needs no SPACE, no network and no POSIX shell."""

from __future__ import annotations

import copy
import itertools
import posixpath
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_space.client import CommandResult, FileStat, SpaceError

SandboxState = Literal["running", "suspended", "deleted"]
HOME = "/home/user"


@dataclass
class FakeSandbox:
    """One fake sandbox: a filesystem (files + directories) and a lifecycle state."""

    id: str
    template: str
    state: SandboxState = "running"
    files: dict[str, bytes] = field(default_factory=dict)
    dirs: set[str] = field(default_factory=lambda: {"/", "/home", HOME, "/tmp"})
    unreadable: set[str] = field(default_factory=set)
    env: dict[str, str] = field(default_factory=dict)
    network_policy: dict[str, Any] | None = None
    forked_from: str | None = None

    def normalize(self, path: str) -> str:
        """Resolve `path` the way SPACE does: relative paths live under /home/user."""
        if not path.startswith("/"):
            path = posixpath.join(HOME, path)
        return posixpath.normpath(path).replace("//", "/")

    def read_text(self, path: str) -> str:
        """Convenience for exec handlers: decode a file as UTF-8."""
        return self.files[self.normalize(path)].decode()

    def write_text(self, path: str, text: str) -> None:
        """Convenience for exec handlers: write a file, creating parent directories."""
        normalized = self.normalize(path)
        self._make_parents(normalized)
        self.files[normalized] = text.encode()

    def _make_parents(self, path: str) -> None:
        parent = posixpath.dirname(path)
        while parent not in self.dirs:
            self.dirs.add(parent)
            parent = posixpath.dirname(parent)


ExecHandler = Callable[[FakeSandbox, Sequence[str]], tuple[str, int]]
"""Simulates a command: receives the sandbox and argv, returns (output, exit_code).

Handlers may read and mutate `sandbox.files`. For `["sh", "-c", cmd]` argv, `cmd` is `argv[2]`.
"""


def unsupported_exec(sandbox: FakeSandbox, argv: Sequence[str]) -> tuple[str, int]:
    """Default handler: the fake has no shell."""
    return f"fake sandbox cannot run {list(argv)!r}", 127


@dataclass(frozen=True)
class Call:
    """One recorded client call, for assertions (e.g. no secret was ever sent)."""

    method: str
    args: dict[str, Any]


class FakeSpaceClient:
    """Implements `SpaceClient` in memory.

    Fork deep-copies the filesystem, so branches are isolated exactly like SPACE forks.

    Args:
        exec_handler: Simulates `run_command`. Defaults to `unsupported_exec`.
    """

    def __init__(self, exec_handler: ExecHandler | None = None) -> None:
        self.exec_handler: ExecHandler = exec_handler or unsupported_exec
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.calls: list[Call] = []
        self.credential_references: dict[str, str] = {}
        self.fork_error: SpaceError | None = None
        """When set, every `fork` raises it (to test failure handling)."""
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers for tests

    def sandbox(self, sandbox_id: str) -> FakeSandbox:
        """Return a sandbox by id, raising like SPACE if it does not exist."""
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None or sandbox.state == "deleted":
            raise SpaceError(f"sandbox {sandbox_id} not found", code="not_found", status=404)
        return sandbox

    @property
    def live_sandboxes(self) -> list[FakeSandbox]:
        """Sandboxes that are not deleted."""
        return [s for s in self.sandboxes.values() if s.state != "deleted"]

    def calls_to(self, method: str) -> list[Call]:
        """Recorded calls to one method."""
        return [c for c in self.calls if c.method == method]

    def transcript(self) -> str:
        """Everything ever sent to the fake, as one string (for secret-leak assertions)."""
        return "\n".join(f"{c.method} {c.args!r}" for c in self.calls)

    def _record(self, method: str, **args: Any) -> None:
        with self._lock:
            self.calls.append(Call(method, args))

    def _new_id(self, prefix: str = "sbx") -> str:
        with self._lock:
            return f"{prefix}_fake{next(self._ids):04d}"

    def _running(self, sandbox_id: str) -> FakeSandbox:
        sandbox = self.sandbox(sandbox_id)
        if sandbox.state != "running":
            raise SpaceError(
                f"sandbox {sandbox_id} is {sandbox.state}", code="failed_precondition", status=412
            )
        return sandbox

    # ------------------------------------------------------------------ SpaceClient

    def create_sandbox(
        self,
        template: str,
        *,
        env: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        network_policy: Mapping[str, Any] | None = None,
    ) -> str:
        """See `SpaceClient.create_sandbox`."""
        self._record(
            "create_sandbox",
            template=template,
            env=dict(env or {}),
            metadata=dict(metadata or {}),
            network_policy=network_policy,
        )
        sandbox_id = self._new_id()
        self.sandboxes[sandbox_id] = FakeSandbox(
            sandbox_id,
            template,
            env=dict(env or {}),
            network_policy=dict(network_policy) if network_policy is not None else None,
        )
        return sandbox_id

    def delete_sandbox(self, sandbox_id: str) -> None:
        """See `SpaceClient.delete_sandbox`."""
        self._record("delete_sandbox", sandbox_id=sandbox_id)
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is not None:
            sandbox.state = "deleted"

    def run_command(
        self,
        sandbox_id: str,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """See `SpaceClient.run_command`."""
        self._record(
            "run_command",
            sandbox_id=sandbox_id,
            argv=list(argv),
            cwd=cwd,
            env=dict(env or {}),
            timeout=timeout,
        )
        sandbox = self._running(sandbox_id)
        output, exit_code = self.exec_handler(sandbox, argv)
        return CommandResult(output=output, exit_code=exit_code)

    def write_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        """See `SpaceClient.write_file`."""
        self._record("write_file", sandbox_id=sandbox_id, path=path, data=data)
        sandbox = self._running(sandbox_id)
        normalized = sandbox.normalize(path)
        if normalized in sandbox.dirs:
            raise SpaceError(f"{path} is a directory", code="failed_precondition", status=412)
        if posixpath.dirname(normalized) not in sandbox.dirs:
            raise _path_not_found(path)
        sandbox.files[normalized] = bytes(data)

    def read_file(self, sandbox_id: str, path: str, *, size: int | None = None) -> bytes:
        """See `SpaceClient.read_file`. `size` is recorded but not needed."""
        self._record("read_file", sandbox_id=sandbox_id, path=path, size=size)
        sandbox = self._running(sandbox_id)
        normalized = sandbox.normalize(path)
        if normalized in sandbox.dirs:
            raise SpaceError(f"{path} is a directory", code="failed_precondition", status=412)
        if normalized in sandbox.unreadable:
            raise SpaceError(f"{path}: permission denied", code="forbidden", status=403)
        if normalized not in sandbox.files:
            raise _path_not_found(path)
        return sandbox.files[normalized]

    def stat(self, sandbox_id: str, path: str) -> FileStat:
        """See `SpaceClient.stat`."""
        self._record("stat", sandbox_id=sandbox_id, path=path)
        sandbox = self._running(sandbox_id)
        normalized = sandbox.normalize(path)
        if normalized in sandbox.dirs:
            return FileStat(normalized, "directory", 0)
        if normalized in sandbox.files:
            return FileStat(normalized, "file", len(sandbox.files[normalized]))
        raise _path_not_found(path)

    def mkdir(self, sandbox_id: str, path: str, *, parents: bool = True) -> None:
        """See `SpaceClient.mkdir`."""
        self._record("mkdir", sandbox_id=sandbox_id, path=path, parents=parents)
        sandbox = self._running(sandbox_id)
        normalized = sandbox.normalize(path)
        if normalized in sandbox.files:
            raise SpaceError(f"{path} is a file", code="failed_precondition", status=412)
        if not parents and posixpath.dirname(normalized) not in sandbox.dirs:
            raise _path_not_found(path)
        sandbox._make_parents(normalized)
        sandbox.dirs.add(normalized)

    def fork(self, sandbox_id: str, *, lifetime_seconds: int | None = None) -> str:
        """See `SpaceClient.fork`. Copies files and directories; the child starts running."""
        self._record("fork", sandbox_id=sandbox_id, lifetime_seconds=lifetime_seconds)
        if self.fork_error is not None:
            raise self.fork_error
        source = self.sandbox(sandbox_id)
        child_id = self._new_id()
        self.sandboxes[child_id] = FakeSandbox(
            child_id,
            source.template,
            files=copy.deepcopy(source.files),
            dirs=set(source.dirs),
            unreadable=set(source.unreadable),
            env=dict(source.env),
            network_policy=copy.deepcopy(source.network_policy),
            forked_from=sandbox_id,
        )
        return child_id

    def suspend(self, sandbox_id: str) -> None:
        """See `SpaceClient.suspend`."""
        self._record("suspend", sandbox_id=sandbox_id)
        self.sandbox(sandbox_id).state = "suspended"

    def wake(self, sandbox_id: str) -> None:
        """See `SpaceClient.wake`."""
        self._record("wake", sandbox_id=sandbox_id)
        self.sandbox(sandbox_id).state = "running"

    def register_credential_reference(self, secret_arn: str) -> str:
        """See `SpaceClient.register_credential_reference`."""
        self._record("register_credential_reference", secret_arn=secret_arn)
        reference_id = self._new_id("cred")
        self.credential_references[reference_id] = secret_arn
        return reference_id

    def set_network_policy(self, sandbox_id: str, policy: Mapping[str, Any]) -> None:
        """See `SpaceClient.set_network_policy`."""
        self._record("set_network_policy", sandbox_id=sandbox_id, policy=dict(policy))
        self.sandbox(sandbox_id).network_policy = dict(policy)


def _path_not_found(path: str) -> SpaceError:
    return SpaceError(
        f"{path}: no such file or directory",
        code="not_found",
        status=404,
        details={"resource": "path"},
    )
