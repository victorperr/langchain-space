"""`SpaceClient` on top of LangSmith sandboxes (`langsmith.sandbox.SandboxClient`).

With this client, `SpaceCheckpointer`, `fork_and_select` and `SpaceSandbox` run on LangSmith
instead of SPACE. Sandbox ids are LangSmith sandbox names.

How the snapshot actions map:

- anchor = a LangSmith snapshot captured from the live sandbox (`capture_snapshot`);
- restore = a new sandbox booted from that snapshot (`create_sandbox(snapshot_id=...)`);
- fork (fork-and-select) = a temporary snapshot, a sandbox from it, then the snapshot is deleted.

Not supported on LangSmith: SPACE network policies and credential references (LangSmith has its
own auth proxy, configured on the `SandboxClient`).
"""

from __future__ import annotations

import contextlib
import math
import shlex
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NoReturn

from langsmith.sandbox import (
    CommandTimeoutError,
    QuotaExceededError,
    ResourceAlreadyExistsError,
    ResourceInUseError,
    ResourceNameConflictError,
    ResourceNotFoundError,
    ResourceTimeoutError,
    Sandbox,
    SandboxAuthenticationError,
    SandboxClient,
    SandboxClientError,
    SandboxConnectionError,
    ValidationError,
)

from langchain_space.client import (
    CommandResult,
    FileStat,
    SpaceError,
    write_files_one_by_one,
)

DEFAULT_TEMPLATE = ""
"""Template meaning "LangSmith's default image" (no snapshot)."""

_STAT_SCRIPT = (
    'p=$1; if [ -d "$p" ]; then echo directory; '
    'elif [ -e "$p" ]; then printf \'file \'; wc -c < "$p"; '
    "else exit 44; fi"
)
_MISSING = 44
"""Exit code of `_STAT_SCRIPT` for a missing path."""

_MKDIR_SCRIPT = '[ -d "$2" ] || mkdir $1 -- "$2"'

_CONFLICTS = (ResourceInUseError, ResourceAlreadyExistsError, ResourceNameConflictError)


class LangSmithClient:
    """Implements `SpaceClient` with LangSmith sandboxes and snapshots.

    Args:
        sandbox_client: The LangSmith client (default: `SandboxClient()`, which reads
            `LANGSMITH_API_KEY`). Closed by `close()` only if this instance created it.
        idle_ttl_seconds: Idle time before LangSmith stops a sandbox (multiple of 60;
            `0` never stops it). Longer than LangSmith's 10-minute default, since an agent
            may think for a while between two tool calls.
        delete_after_stop_seconds: Passed to every sandbox (`None`: LangSmith's default).
        default_timeout: Command timeout in seconds when the caller gives none.
        ready_timeout: Seconds to wait for a new sandbox to be ready.
        snapshot_timeout: Seconds to wait for a snapshot to be ready.
        snapshot_prefix: Prefix of the snapshot names this client creates.
    """

    provider = "langsmith"

    def __init__(
        self,
        sandbox_client: SandboxClient | None = None,
        *,
        idle_ttl_seconds: int = 3600,
        delete_after_stop_seconds: int | None = None,
        default_timeout: int = 30 * 60,
        ready_timeout: int = 180,
        snapshot_timeout: int = 300,
        snapshot_prefix: str = "lcs",
    ) -> None:
        self._owns_client = sandbox_client is None
        self._client = sandbox_client if sandbox_client is not None else SandboxClient()
        self._idle_ttl = idle_ttl_seconds
        self._delete_after_stop = delete_after_stop_seconds
        self.default_timeout = default_timeout
        self._ready_timeout = ready_timeout
        self._snapshot_timeout = snapshot_timeout
        self._prefix = snapshot_prefix
        self._sandboxes: dict[str, Sandbox] = {}
        self._env: dict[str, dict[str, str]] = {}
        """Environment each sandbox was created with, so anchors and forks keep it."""
        self._undeleted_snapshots: list[str] = []
        """Temporary fork snapshots LangSmith refused to delete yet; retried by `close()`."""
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle

    def create_sandbox(
        self,
        template: str,
        *,
        env: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        network_policy: Mapping[str, Any] | None = None,
    ) -> str:
        """See `SpaceClient.create_sandbox`.

        `template` is a LangSmith snapshot (`name`, `name:tag` or id), or `""` for the default
        image. `metadata` is ignored: LangSmith sandboxes have none.
        """
        if network_policy is not None:
            _not_supported("network policies")
        run_config = {"env_vars": dict(env)} if env else None
        with _translated():
            sandbox = self._client.create_sandbox(
                snapshot=template or None,
                run_config=run_config,
                **self._create_kwargs(),
            )
        return self._remember(sandbox, env)

    def delete_sandbox(self, sandbox_id: str) -> None:
        """See `SpaceClient.delete_sandbox`."""
        with self._lock:
            self._sandboxes.pop(sandbox_id, None)
            self._env.pop(sandbox_id, None)
        with _translated(), contextlib.suppress(ResourceNotFoundError):
            self._client.delete_sandbox(sandbox_id)

    def suspend(self, sandbox_id: str) -> None:
        """See `SpaceClient.suspend`. Stops the sandbox; its files are kept."""
        with _translated():
            self._client.stop_sandbox(sandbox_id)

    def wake(self, sandbox_id: str) -> None:
        """See `SpaceClient.wake`."""
        with _translated():
            self._client.start_sandbox(sandbox_id, timeout=self._ready_timeout)

    # ------------------------------------------------------------------ snapshots

    def create_anchor(self, sandbox_id: str, *, lifetime_seconds: int | None = None) -> str:
        """See `SpaceClient.create_anchor`. Captures a LangSmith snapshot; returns its id.

        `lifetime_seconds` is ignored: snapshots last until deleted (`delete_anchor`).
        """
        return self._capture(sandbox_id, "anchor")

    def restore_anchor(self, anchor_id: str) -> str:
        """See `SpaceClient.restore_anchor`. Boots a new sandbox from the snapshot."""
        with _translated():
            sandbox = self._client.create_sandbox(snapshot_id=anchor_id, **self._create_kwargs())
        return self._remember(sandbox, None)

    def delete_anchor(self, anchor_id: str) -> None:
        """See `SpaceClient.delete_anchor`. Deletes the snapshot."""
        with _translated(), contextlib.suppress(ResourceNotFoundError):
            self._client.delete_snapshot(anchor_id)

    def fork(self, sandbox_id: str, *, lifetime_seconds: int | None = None) -> str:
        """See `SpaceClient.fork`. Temporary snapshot, new sandbox from it, snapshot deleted.

        `lifetime_seconds` is ignored; the child follows `idle_ttl_seconds`.
        """
        snapshot_id = self._capture(sandbox_id, "fork")
        try:
            return self.restore_anchor(snapshot_id)
        finally:
            try:
                self.delete_anchor(snapshot_id)
            except SpaceError:
                # Possibly refused while the new sandbox still uses it; retried by close().
                with self._lock:
                    self._undeleted_snapshots.append(snapshot_id)

    # ------------------------------------------------------------------ commands and files

    def run_command(
        self,
        sandbox_id: str,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """See `SpaceClient.run_command`. stdout comes first, then stderr.

        `timeout=0` means no deadline; a timed-out command returns exit code 124.
        """
        seconds = self.default_timeout if timeout is None else math.ceil(timeout)
        run_config: dict[str, Any] = {}
        if cwd is not None:
            run_config["work_dir"] = cwd
        if env:
            run_config["env_vars"] = dict(env)
        sandbox = self._sandbox(sandbox_id)
        try:
            with _translated():
                result = sandbox.run(
                    shlex.join(argv), timeout=seconds, run_config=run_config or None
                )
        except SpaceError as error:
            if isinstance(error.__cause__, CommandTimeoutError):
                return CommandResult(output="", exit_code=124, status="timed_out")
            raise
        output = result.stdout or ""
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        return CommandResult(output=output, exit_code=result.exit_code)

    def write_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        """See `SpaceClient.write_file`."""
        sandbox = self._sandbox(sandbox_id)
        with _translated():
            sandbox.write(path, data)

    def write_files(
        self, sandbox_id: str, files: Sequence[tuple[str, bytes]]
    ) -> list[SpaceError | None]:
        """See `SpaceClient.write_files`. One `mkdir` per parent directory, then each write."""
        return write_files_one_by_one(self, sandbox_id, files)

    def read_file(self, sandbox_id: str, path: str, *, size: int | None = None) -> bytes:
        """See `SpaceClient.read_file`. `size` is not needed."""
        sandbox = self._sandbox(sandbox_id)
        with _translated():
            return sandbox.read(path)

    def stat(self, sandbox_id: str, path: str) -> FileStat:
        """See `SpaceClient.stat`. One shell command, since LangSmith's own stat is file-only."""
        result = self.run_command(sandbox_id, ["sh", "-c", _STAT_SCRIPT, "sh", path], timeout=60)
        if result.exit_code == _MISSING:
            raise _path_not_found(path)
        if result.exit_code != 0:
            raise _shell_error(path, result.output)
        kind, _, size = result.output.strip().partition(" ")
        if kind == "directory":
            return FileStat(path, "directory", 0)
        return FileStat(path, "file", int(size.strip() or 0))

    def mkdir(self, sandbox_id: str, path: str, *, parents: bool = True) -> None:
        """See `SpaceClient.mkdir`."""
        flag = "-p" if parents else ""
        argv = ["sh", "-c", _MKDIR_SCRIPT, "sh", flag, path]
        result = self.run_command(sandbox_id, argv, timeout=60)
        if result.exit_code != 0:
            raise _shell_error(path, result.output)

    # ------------------------------------------------------------------ SPACE-only features

    def set_network_policy(self, sandbox_id: str, policy: Mapping[str, Any]) -> None:
        """Not supported: LangSmith has no SPACE network policies."""
        _not_supported("network policies")

    def register_credential_reference(self, secret_arn: str) -> str:
        """Not supported: use LangSmith's auth proxy (`proxy_config`) instead."""
        _not_supported("credential references")

    def retire_credential_reference(self, credential_reference_id: str) -> None:
        """Not supported: use LangSmith's auth proxy (`proxy_config`) instead."""
        _not_supported("credential references")

    # ------------------------------------------------------------------ cleanup

    def close(self) -> None:
        """Retry deleting leftover temporary snapshots, then close the client if owned."""
        with self._lock:
            pending, self._undeleted_snapshots = self._undeleted_snapshots, []
        for snapshot_id in pending:
            with contextlib.suppress(SpaceError):
                self.delete_anchor(snapshot_id)
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LangSmithClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ helpers

    def _create_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "timeout": self._ready_timeout,
            "idle_ttl_seconds": self._idle_ttl,
        }
        if self._delete_after_stop is not None:
            kwargs["delete_after_stop_seconds"] = self._delete_after_stop
        return kwargs

    def _remember(self, sandbox: Sandbox, env: Mapping[str, str] | None) -> str:
        with self._lock:
            self._sandboxes[sandbox.name] = sandbox
            if env:
                self._env[sandbox.name] = dict(env)
        return sandbox.name

    def _sandbox(self, sandbox_id: str) -> Sandbox:
        with self._lock:
            sandbox = self._sandboxes.get(sandbox_id)
        if sandbox is not None:
            return sandbox
        with _translated():
            sandbox = self._client.get_sandbox(sandbox_id)
        with self._lock:
            return self._sandboxes.setdefault(sandbox_id, sandbox)

    def _capture(self, sandbox_id: str, kind: str) -> str:
        name = f"{self._prefix}-{kind}-{uuid.uuid4().hex[:12]}"
        with self._lock:
            env = self._env.get(sandbox_id)
        # A snapshot does not keep the sandbox's own env vars; bake them into its run config
        # so sandboxes restored from it (even in another process) get them back.
        run_config = {"env_vars": env} if env else None
        with _translated():
            snapshot = self._client.capture_snapshot(
                sandbox_id, name, run_config=run_config, timeout=self._snapshot_timeout
            )
        return snapshot.id


@contextlib.contextmanager
def _translated() -> Iterator[None]:
    """Re-raise LangSmith errors as `SpaceError`, so the rest of the package handles them."""
    try:
        yield
    except SandboxClientError as error:
        raise _space_error(error) from error


def _space_error(error: SandboxClientError) -> SpaceError:
    message = str(error)
    if isinstance(error, ResourceNotFoundError):
        # A missing file is a missing *path*; anything else (sandbox, snapshot) is not.
        details = {"resource": "path"} if error.resource_type == "file" else {}
        return SpaceError(message, code="not_found", status=404, details=details)
    if isinstance(error, ResourceTimeoutError | CommandTimeoutError):
        return SpaceError(message, code="timeout")
    if isinstance(error, _CONFLICTS):
        return SpaceError(message, code="conflict", status=409)
    if isinstance(error, ValidationError):
        return SpaceError(message, code="invalid_request", status=400)
    if isinstance(error, QuotaExceededError):
        return SpaceError(message, code="resource_exhausted", status=429, retryable=True)
    if isinstance(error, SandboxAuthenticationError):
        return SpaceError(message, code="unauthenticated", status=401)
    if isinstance(error, SandboxConnectionError):
        return SpaceError(message, code="transport", retryable=True)
    lowered = message.lower()
    if "permission denied" in lowered:
        return SpaceError(message, code="forbidden", status=403)
    if "is a directory" in lowered:
        return SpaceError(message, code="failed_precondition", status=412)
    return SpaceError(message, code="unknown")


def _path_not_found(path: str) -> SpaceError:
    return SpaceError(
        f"{path}: no such file or directory",
        code="not_found",
        status=404,
        details={"resource": "path"},
    )


def _shell_error(path: str, output: str) -> SpaceError:
    """Map a failed `sh` helper's message to the error SPACE's file routes would give."""
    lowered = output.lower()
    if "no such file" in lowered:
        return _path_not_found(path)
    if "permission denied" in lowered:
        return SpaceError(f"{path}: permission denied", code="forbidden", status=403)
    return SpaceError(f"{path}: {output.strip()}", code="failed_precondition", status=412)


def _not_supported(feature: str) -> NoReturn:
    raise SpaceError(f"LangSmith sandboxes do not support SPACE {feature}", code="not_supported")
