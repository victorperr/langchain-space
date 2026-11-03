"""Client for the SPACE API.

This is the only module that knows SPACE URLs and JSON shapes. Everything else depends on the
`SpaceClient` protocol, so tests can swap in `langchain_space.testing.FakeSpaceClient`.

API facts used here are verified against the SPACE source; see the "SPACE API notes" section
of docs/ARCHITECTURE.md for the references.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import logging
import os
import posixpath
import random
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Protocol, Self, runtime_checkable
from urllib.parse import quote, urljoin, urlsplit

import httpx
from httpx_sse import connect_sse

FileType = Literal["file", "directory", "symlink", "other"]

logger = logging.getLogger(__name__)

MIB = 1024 * 1024
_FIRST_POLL_DELAY = 0.05
"""Seconds before the first re-poll; doubles up to `poll_interval` (fast ops finish fast)."""
_TERMINAL_OPERATION_STATES = frozenset({"canceled", "expired", "failed", "succeeded"})
_TERMINAL_SANDBOX_STATES = frozenset({"deleted", "deleting", "failed"})
_SUSPEND_UPLOAD_CONFLICT = "suspend-snapshot upload in flight"
_WRITE_BATCH_MAX_ENTRIES = 256
"""Files per `files:write_batch` (`MAX_WRITE_BATCH_ENTRIES`, SPACE common/core/src/lib.rs)."""
_WRITE_BATCH_MAX_BYTES = 3 * MIB
"""Data plus path bytes per `files:write_batch` (`MAX_FILE_WRITE_BATCH_BYTES`,
SPACE api/src/routes/public/files.rs)."""

_READ_FILES_SCRIPT = """exec 2>/dev/null
budget=$1
shift
for p do
  if [ -d "$p" ]; then echo D
  elif [ ! -e "$p" ]; then echo N
  elif [ ! -r "$p" ]; then echo P
  else
    n=$(wc -c < "$p" | tr -d ' ')
    case $n in ''|*[!0-9]*) echo P; continue;; esac
    if [ "$n" -gt "$budget" ]; then echo "B $n"
    else budget=$((budget - n)); echo "F $n"; base64 < "$p" | tr -d '\\n'; echo
    fi
  fi
done"""
"""Prints one record per path: `D` (directory), `N` (missing), `P` (unreadable), `B <size>`
(over the output budget: read it with `files:read`) or `F <size>` and a base64 line.
Base64 keeps binary files intact through the command's UTF-8 output stream."""
_READ_FILES_MAX_PATHS = 256
_READ_FILES_MAX_PATH_BYTES = 64 * 1024
"""Path bytes per `read_files` command, far below Linux's argument size limit."""


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one command run inside a sandbox."""

    output: str
    """stdout and stderr, interleaved in arrival order."""
    exit_code: int | None
    """Process exit code; 124 on timeout, 128+n when killed by signal n."""
    status: str = "exited"
    """SPACE exit status: `exited`, `signaled`, `timed_out` or `unknown`."""
    truncated: bool = False
    """True when the client dropped output beyond its buffering limit."""


@dataclass(frozen=True)
class FileStat:
    """Metadata for one path inside a sandbox."""

    path: str
    file_type: FileType
    size_bytes: int


class SpaceError(Exception):
    """An error reported by SPACE (or by this client about a SPACE interaction).

    `code` is SPACE's stable classifier (`not_found`, `forbidden`, `conflict`, ...) or a
    client-side one (`timeout`, `transport`, `invalid_response`, `not_configured`).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "unknown",
        status: int | None = None,
        retryable: bool = False,
        details: Mapping[str, str] | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.retryable = retryable
        self.details: dict[str, str] = dict(details or {})
        self.request_id = request_id

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    @property
    def is_path_not_found(self) -> bool:
        """True when a file route says the *path* is missing (not the sandbox)."""
        return self.code == "not_found" and self.details.get("resource") == "path"


@runtime_checkable
class SpaceClient(Protocol):
    """The subset of SPACE that langchain-space depends on.

    All methods are blocking. Methods that change state wait until SPACE reports the change
    as complete, so callers never see a half-created sandbox.
    """

    provider: str
    """Which sandbox provider this client talks to (`"space"`, `"langsmith"`). Stored in every
    checkpoint, so a checkpoint is never restored with another provider's client."""

    def create_sandbox(
        self,
        template: str,
        *,
        env: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        network_policy: Mapping[str, Any] | None = None,
    ) -> str:
        """Create a sandbox from `template`, wait until it is running, return its id."""
        ...

    def delete_sandbox(self, sandbox_id: str) -> None:
        """Delete a sandbox. Deleting an already-deleted sandbox is not an error."""
        ...

    def run_command(
        self,
        sandbox_id: str,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run `argv` (not through a shell) and wait for it to exit.

        `timeout=None` uses the client default; `timeout=0` means no deadline.
        """
        ...

    def write_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        """Create or replace the file at `path`. The parent directory must exist."""
        ...

    def write_files(
        self, sandbox_id: str, files: Sequence[tuple[str, bytes]]
    ) -> list[SpaceError | None]:
        """Create or replace each file, creating missing parent directories, in order.

        Returns one entry per file: None if it was written, else the error for that file.
        """
        ...

    def read_file(self, sandbox_id: str, path: str, *, size: int | None = None) -> bytes:
        """Return the full contents of the file at `path`.

        `size` is the file's size in bytes when the caller already knows it (e.g. from
        `stat`), which saves the client from looking it up again.
        """
        ...

    def read_files(self, sandbox_id: str, paths: Sequence[str]) -> list[bytes | SpaceError]:
        """Return the contents of each file, or the error for that file, in order.

        A directory is a `SpaceError` with code `is_directory`; a missing path is one with
        code `not_found` (`is_path_not_found`).
        """
        ...

    def stat(self, sandbox_id: str, path: str) -> FileStat:
        """Return metadata for `path`; raises `SpaceError` with code `not_found` if missing."""
        ...

    def mkdir(self, sandbox_id: str, path: str, *, parents: bool = True) -> None:
        """Create a directory. Succeeds if it already exists."""
        ...

    def fork(self, sandbox_id: str, *, lifetime_seconds: int | None = None) -> str:
        """Copy the sandbox's disk into a new running sandbox and return the new id.

        The source may be running, paused or suspended and is left unchanged.
        """
        ...

    def suspend(self, sandbox_id: str) -> None:
        """Persist and release a sandbox's resources; it can be forked from or woken later."""
        ...

    def wake(self, sandbox_id: str) -> None:
        """Bring a paused or suspended sandbox back to running."""
        ...

    def create_anchor(self, sandbox_id: str, *, lifetime_seconds: int | None = None) -> str:
        """Save an immutable copy of the sandbox's filesystem (an anchor); return its id.

        The sandbox keeps running unchanged. An anchor is never run, only restored from.
        """
        ...

    def restore_anchor(self, anchor_id: str) -> str:
        """Create a new running sandbox from an anchor and return its id."""
        ...

    def delete_anchor(self, anchor_id: str) -> None:
        """Delete an anchor. Deleting an already-deleted anchor is not an error."""
        ...

    def register_credential_reference(self, secret_arn: str) -> str:
        """Register an AWS Secrets Manager secret by reference; return its reference id."""
        ...

    def retire_credential_reference(self, credential_reference_id: str) -> None:
        """Retire a credential reference. Retiring an unknown one is not an error."""
        ...

    def set_network_policy(self, sandbox_id: str, policy: Mapping[str, Any]) -> None:
        """Replace the sandbox's network policy and wait until it is applied."""
        ...


def write_files_one_by_one(
    client: SpaceClient,
    sandbox_id: str,
    files: Sequence[tuple[str, bytes]],
    *,
    made_dirs: set[str] | None = None,
) -> list[SpaceError | None]:
    """`SpaceClient.write_files` built from `mkdir` and `write_file`.

    Makes one `mkdir` per parent directory. `made_dirs` holds the directories already
    created, shared between calls.
    """
    made = made_dirs if made_dirs is not None else set()
    results: list[SpaceError | None] = []
    for path, data in files:
        try:
            parent = posixpath.dirname(path)
            if parent and parent != "/" and parent not in made:
                client.mkdir(sandbox_id, parent, parents=True)
                made.add(parent)
            client.write_file(sandbox_id, path, data)
        except SpaceError as error:
            results.append(error)
        else:
            results.append(None)
    return results


def read_files_one_by_one(
    client: SpaceClient, sandbox_id: str, paths: Sequence[str]
) -> list[bytes | SpaceError]:
    """`SpaceClient.read_files` built from `stat` and `read_file` (two calls per file)."""
    results: list[bytes | SpaceError] = []
    for path in paths:
        try:
            stat = client.stat(sandbox_id, path)
            if stat.file_type == "directory":
                results.append(SpaceError(f"{path} is a directory", code="is_directory"))
                continue
            results.append(client.read_file(sandbox_id, path, size=stat.size_bytes))
        except SpaceError as error:
            results.append(error)
    return results


@dataclass(frozen=True)
class _Retry:
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 20.0

    def delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        backoff: float = self.base_delay * 2**attempt
        return float(min(backoff, self.max_delay) * random.uniform(0.5, 1.0))


class HttpSpaceClient:
    """`SpaceClient` over SPACE's native HTTP API (plus its E2B-compatible fork route).

    Args:
        api_url: Native API origin, e.g. `http://127.0.0.1:8080`.
        data_plane_url: Data-plane origin serving command event streams, e.g.
            `http://127.0.0.1:8090`.
        token: Session token or API key, sent as `Authorization: Bearer`.
        ecw_api_url: Origin of the E2B-compatible API host. Required only for `fork`.
        ecw_api_key: API key enabled for the `ecw` surface, sent as `X-API-Key`.
        default_command_timeout: Seconds used when `run_command(timeout=None)`.
        default_fork_lifetime: Seconds a forked sandbox may live when not specified (anchors,
            and other forks unless `live_lifetime` is set).
        live_lifetime: Opt-in shorter lifetime, in seconds, for forks that run: sandboxes
            restored from an anchor and `fork_and_select` branches. Limits how long they
            survive a crashed process. Unverified on real SPACE: if the lifetime is a hard
            kill, a thread unused for longer loses its live sandbox.
        idle_timeout: Opt-in: seconds without activity after which SPACE suspends a sandbox
            created from a template (`idle_timeout` + `idle_action: "suspend"`). Unverified
            on real SPACE: whether a command wakes a suspended sandbox.
        suspend_anchors_in_background: Return an anchor as soon as its fork exists and
            suspend it in a background thread (saves the suspend time, about 6 s, per
            snapshot). Restoring or deleting that anchor first waits for its suspend.
        ready_timeout: Seconds to wait for an operation or a sandbox state.
        poll_interval: Longest pause between status polls (polling starts faster).
        max_output_bytes: Command output kept in memory; the rest is read and dropped,
            and the result is marked `truncated`.
        http: Optional preconfigured `httpx.Client` (tests inject a mocked transport).
    """

    provider = "space"

    def __init__(
        self,
        api_url: str,
        data_plane_url: str,
        token: str,
        *,
        ecw_api_url: str | None = None,
        ecw_api_key: str | None = None,
        default_command_timeout: float = 1800,
        default_fork_lifetime: int = 7 * 24 * 3600,
        live_lifetime: int | None = None,
        idle_timeout: int | None = None,
        suspend_anchors_in_background: bool = True,
        ready_timeout: float = 300,
        poll_interval: float = 0.5,
        request_timeout: float = 60,
        chunk_size: int = 4 * MIB,
        max_output_bytes: int = 10 * MIB,
        http: httpx.Client | None = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._data_plane_url = data_plane_url.rstrip("/")
        self._token = token
        self._ecw_api_url = ecw_api_url.rstrip("/") if ecw_api_url else None
        self._ecw_api_key = ecw_api_key
        self.default_command_timeout = default_command_timeout
        self.default_fork_lifetime = default_fork_lifetime
        if live_lifetime is not None and live_lifetime <= 0:
            raise ValueError("live_lifetime must be > 0")
        if idle_timeout is not None and idle_timeout <= 0:
            raise ValueError("idle_timeout must be > 0")
        self.live_lifetime = live_lifetime
        self.idle_timeout = idle_timeout
        self._background_suspend = suspend_anchors_in_background
        self._suspends: dict[str, Future[None]] = {}
        """Anchors whose background suspend may still be running."""
        self._suspend_pool: ThreadPoolExecutor | None = None
        self._suspend_lock = threading.Lock()
        self._ready_timeout = ready_timeout
        self._poll_interval = poll_interval
        self._request_timeout = request_timeout
        if not 0 < chunk_size <= 16 * MIB:
            raise ValueError("chunk_size must be in (0, 16 MiB]")
        self._chunk_size = chunk_size
        if max_output_bytes < 0:
            raise ValueError("max_output_bytes must be >= 0")
        self._max_output_bytes = max_output_bytes
        self._retry = _Retry()
        self._owns_http = http is None
        self._http = http or httpx.Client(
            timeout=request_timeout, follow_redirects=False, trust_env=False
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> HttpSpaceClient:
        """Build a client from `SPACE_API_URL`, `SPACE_DATA_PLANE_URL`, `SPACE_API_TOKEN`,
        and optionally `SPACE_ECW_API_URL` / `SPACE_ECW_API_KEY`."""
        try:
            api_url = os.environ["SPACE_API_URL"]
            data_plane_url = os.environ["SPACE_DATA_PLANE_URL"]
            token = os.environ["SPACE_API_TOKEN"]
        except KeyError as missing:
            raise RuntimeError(f"environment variable {missing} is not set") from None
        return cls(
            api_url,
            data_plane_url,
            token,
            ecw_api_url=os.environ.get("SPACE_ECW_API_URL"),
            ecw_api_key=os.environ.get("SPACE_ECW_API_KEY"),
            **kwargs,
        )

    def close(self) -> None:
        """Finish background suspends, then close the HTTP client if this instance created it."""
        with self._suspend_lock:
            pending = list(self._suspends)
        for anchor_id in pending:
            self._await_suspend(anchor_id)
        if self._suspend_pool is not None:
            self._suspend_pool.shutdown(wait=True)
            self._suspend_pool = None
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ sandboxes

    def create_sandbox(
        self,
        template: str,
        *,
        env: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        network_policy: Mapping[str, Any] | None = None,
    ) -> str:
        """See `SpaceClient.create_sandbox`."""
        body: dict[str, Any] = {"template": template}
        if env:
            body["env"] = dict(env)
        if metadata:
            body["metadata"] = dict(metadata)
        if network_policy is not None:
            body["network_policy"] = dict(network_policy)
        if self.idle_timeout is not None:
            body["idle_timeout"] = f"{self.idle_timeout}s"
            body["idle_action"] = "suspend"
        created = self._json("POST", "/v1/sandboxes", body=body, mutation=True, expect=(202,))
        sandbox_id = _str(created, "sandbox_id")
        try:
            self._wait_operation(_str(created, "operation_id"))
            self._wait_running(sandbox_id)
        except BaseException:
            self._best_effort_delete(sandbox_id)
            raise
        return sandbox_id

    def delete_sandbox(self, sandbox_id: str) -> None:
        """See `SpaceClient.delete_sandbox`."""
        deadline = time.monotonic() + self._ready_timeout
        while True:
            try:
                response = self._json(
                    "DELETE", f"/v1/sandboxes/{_q(sandbox_id)}", mutation=True, expect=(202,)
                )
            except SpaceError as error:
                if error.status == 404:
                    return
                raise
            try:
                self._wait_operation(_str(response, "operation_id"))
                return
            except SpaceError as error:
                # A suspend snapshot still uploading blocks delete; the failed operation is
                # immutable, so retry with a fresh idempotency key.
                if error.code != "conflict" or _SUSPEND_UPLOAD_CONFLICT not in error.message:
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(self._poll_interval)

    def suspend(self, sandbox_id: str) -> None:
        """See `SpaceClient.suspend`."""
        self._lifecycle(sandbox_id, "suspend")
        self._wait_sandbox_state(sandbox_id, "suspended")

    def wake(self, sandbox_id: str) -> None:
        """See `SpaceClient.wake`."""
        self._lifecycle(sandbox_id, "wake")
        self._wait_running(sandbox_id)

    def create_anchor(self, sandbox_id: str, *, lifetime_seconds: int | None = None) -> str:
        """See `SpaceClient.create_anchor`. On SPACE an anchor is a suspended fork.

        With `suspend_anchors_in_background`, the anchor is returned once its fork exists.
        If the background suspend fails, the anchor stays running: still a correct copy
        (nothing runs commands in it), only more expensive.
        """
        lifetime = lifetime_seconds or self.default_fork_lifetime
        anchor_id = self.fork(sandbox_id, lifetime_seconds=lifetime)
        if self._background_suspend:
            with self._suspend_lock:
                if self._suspend_pool is None:
                    self._suspend_pool = ThreadPoolExecutor(thread_name_prefix="space-suspend")
                self._suspends[anchor_id] = self._suspend_pool.submit(self.suspend, anchor_id)
            return anchor_id
        try:
            self.suspend(anchor_id)
        except BaseException:
            self._best_effort_delete(anchor_id)
            raise
        return anchor_id

    def restore_anchor(self, anchor_id: str) -> str:
        """See `SpaceClient.restore_anchor`. Forks the suspended anchor."""
        self._await_suspend(anchor_id)
        return self.fork(anchor_id)

    def delete_anchor(self, anchor_id: str) -> None:
        """See `SpaceClient.delete_anchor`. An anchor is a sandbox on SPACE."""
        self._await_suspend(anchor_id)
        self.delete_sandbox(anchor_id)

    def _await_suspend(self, anchor_id: str) -> None:
        """Wait for the anchor's background suspend, if one is pending. Never raises."""
        with self._suspend_lock:
            pending = self._suspends.pop(anchor_id, None)
        if pending is None:
            return
        try:
            pending.result()
        except Exception:
            # The anchor stays running, which is still a correct copy to fork from.
            logger.warning(
                "suspending anchor %s failed; it stays running", anchor_id, exc_info=True
            )

    def fork(self, sandbox_id: str, *, lifetime_seconds: int | None = None) -> str:
        """See `SpaceClient.fork`. Uses the E2B-compatible route (disk-only fork)."""
        if self._ecw_api_url is None or self._ecw_api_key is None:
            raise SpaceError(
                "fork needs ecw_api_url and ecw_api_key (SPACE's E2B-compatible API)",
                code="not_configured",
            )
        lifetime = lifetime_seconds or self.live_lifetime or self.default_fork_lifetime
        response = self._request(
            "POST",
            f"{self._ecw_api_url}/sandboxes/{_q(_ecw_sandbox_id(sandbox_id))}/fork",
            body={"timeout": lifetime, "count": 1},
            mutation=True,
            expect=(201,),
            headers={"X-API-Key": self._ecw_api_key},
            auth=False,
        )
        results = response.json()
        if not isinstance(results, list) or len(results) != 1:
            raise SpaceError(f"unexpected fork response: {results!r}", code="invalid_response")
        result = results[0]
        if not isinstance(result, dict):
            raise SpaceError(f"unexpected fork result: {result!r}", code="invalid_response")
        if "error" in result:
            err = result["error"]
            if isinstance(err, dict):
                message, code = err.get("message"), err.get("code")
            else:
                message, code = err, None
            raise SpaceError(
                str(message or "fork failed"),
                code=str(code) if code is not None else "fork_failed",
            )
        child_id = _native_sandbox_id(_str(result, "sandboxID"))
        self._wait_running(child_id)
        return child_id

    # ------------------------------------------------------------------ commands

    def run_command(
        self,
        sandbox_id: str,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """See `SpaceClient.run_command`.

        Creating a command does not start it: the process starts when its SSE event stream
        is opened, and only one stream may attach, so the stream is opened exactly once.
        """
        if not argv:
            raise ValueError("argv must not be empty")
        effective_timeout = self.default_command_timeout if timeout is None else timeout
        body: dict[str, Any] = {"command": list(argv)}
        if cwd is not None:
            body["cwd"] = cwd
        if env:
            body["env"] = dict(env)
        if effective_timeout > 0:
            body["timeout"] = f"{effective_timeout:g}s"
        created = self._json(
            "POST",
            f"/v1/sandboxes/{_q(sandbox_id)}/commands",
            body=body,
            mutation=True,
            expect=(201,),
        )
        attach = created.get("attach")
        if not isinstance(attach, dict):
            raise SpaceError("command response has no attach object", code="invalid_response")
        events_url = self._events_url(_str(attach, "events_url"))
        read_timeout = effective_timeout + 30 if effective_timeout > 0 else None
        return self._collect_events(events_url, _str(attach, "token"), read_timeout)

    def _collect_events(
        self, events_url: str, attach_token: str, read_timeout: float | None
    ) -> CommandResult:
        chunks: list[bytes] = []
        kept = 0
        truncated = False
        headers = {"Authorization": f"Bearer {attach_token}", "Accept": "text/event-stream"}
        timeout = httpx.Timeout(self._request_timeout, read=read_timeout)
        with connect_sse(self._http, "GET", events_url, headers=headers, timeout=timeout) as source:
            if source.response.status_code != 200:
                source.response.read()
                raise _error_from_response(source.response)
            for event in source.iter_sse():
                if not event.data:
                    continue
                frame = json.loads(event.data)
                kind = frame.get("type") or event.event
                if kind in ("stdout", "stderr"):
                    data = base64.b64decode(_str(frame, "data_base64"), validate=True)
                    # Keep draining past the cap (the process must reach `process_exited`)
                    # but stop buffering, so a noisy command cannot exhaust host memory.
                    room = self._max_output_bytes - kept
                    if len(data) > room:
                        data, truncated = data[: max(room, 0)], True
                    if data:
                        chunks.append(data)
                        kept += len(data)
                elif kind == "process_exited":
                    exit_code = frame.get("exit_code")
                    return CommandResult(
                        output=b"".join(chunks).decode("utf-8", errors="replace"),
                        exit_code=exit_code if isinstance(exit_code, int) else None,
                        status=str(frame.get("status", "unknown")),
                        truncated=truncated,
                    )
                elif kind in ("error", "resync_required"):
                    raise SpaceError(
                        str(frame.get("message") or frame.get("reason") or kind),
                        code=str(frame.get("code", kind)),
                        retryable=bool(frame.get("retryable", False)),
                    )
        raise SpaceError("command stream ended before process_exited", code="stream_closed")

    def _events_url(self, events_url: str) -> str:
        url = urljoin(f"{self._data_plane_url}/", events_url)
        if _origin(url) != _origin(self._data_plane_url):
            # Never send an attach token to an origin we were not configured with.
            raise SpaceError(f"events URL {url!r} is outside the data plane", code="unsafe_url")
        return url

    # ------------------------------------------------------------------ files

    def write_file(self, sandbox_id: str, path: str, data: bytes) -> None:
        """See `SpaceClient.write_file`. Large files are sent as a write plus appends."""
        offsets = range(0, max(len(data), 1), self._chunk_size)
        for index, offset in enumerate(offsets):
            chunk = data[offset : offset + self._chunk_size]
            self._json(
                "POST",
                f"/v1/sandboxes/{_q(sandbox_id)}/files:write",
                body={
                    "path": path,
                    "data_base64": base64.b64encode(chunk).decode("ascii"),
                    "append": index > 0,
                },
                mutation=True,
            )

    def write_files(
        self, sandbox_id: str, files: Sequence[tuple[str, bytes]]
    ) -> list[SpaceError | None]:
        """See `SpaceClient.write_files`.

        Consecutive small files share one `files:write_batch` request (at most 256 files
        and 3 MiB of data plus paths), which also creates their parent directories. A file
        too large for a batch is written on its own with `write_file`.

        SPACE stops a batch at its first failing file without saying which one failed, so
        the files of a failed batch are written again one by one, each getting its own
        result. Writing a file that already landed again is harmless: every write replaces.
        """
        results: list[SpaceError | None] = []
        made_dirs: set[str] = set()
        batch: list[tuple[str, bytes]] = []
        batch_bytes = 0
        for path, data in files:
            size = len(path.encode()) + len(data)
            full = len(batch) == _WRITE_BATCH_MAX_ENTRIES
            if batch and (full or batch_bytes + size > _WRITE_BATCH_MAX_BYTES):
                results.extend(self._write_batch(sandbox_id, batch, made_dirs))
                batch, batch_bytes = [], 0
            if size > _WRITE_BATCH_MAX_BYTES:
                results.extend(
                    write_files_one_by_one(self, sandbox_id, [(path, data)], made_dirs=made_dirs)
                )
                continue
            batch.append((path, data))
            batch_bytes += size
        if batch:
            results.extend(self._write_batch(sandbox_id, batch, made_dirs))
        return results

    def _write_batch(
        self, sandbox_id: str, batch: list[tuple[str, bytes]], made_dirs: set[str]
    ) -> list[SpaceError | None]:
        entries = [
            {"path": path, "data_base64": base64.b64encode(data).decode("ascii")}
            for path, data in batch
        ]
        try:
            self._json(
                "POST",
                f"/v1/sandboxes/{_q(sandbox_id)}/files:write_batch",
                body={"entries": entries, "parents": True},
                mutation=True,
            )
        except SpaceError as error:
            logger.debug(
                "batch of %d files failed (%s); writing them one by one", len(batch), error
            )
            return write_files_one_by_one(self, sandbox_id, batch, made_dirs=made_dirs)
        return [None] * len(batch)

    def read_file(self, sandbox_id: str, path: str, *, size: int | None = None) -> bytes:
        """See `SpaceClient.read_file`. Reads in chunks of at most `chunk_size` bytes."""
        if size is None:
            size = self.stat(sandbox_id, path).size_bytes
        parts: list[bytes] = []
        offset = 0
        while offset < size:
            length = min(self._chunk_size, size - offset)
            response = self._request(
                "POST",
                f"{self._api_url}/v1/sandboxes/{_q(sandbox_id)}/files:read",
                body={"path": path, "offset": offset, "length": length},
            )
            part = response.content
            if not part:
                break
            parts.append(part)
            offset += len(part)
        return b"".join(parts)

    def read_files(self, sandbox_id: str, paths: Sequence[str]) -> list[bytes | SpaceError]:
        """See `SpaceClient.read_files`.

        Several absolute paths are read by one shell command that prints each file in
        base64, up to about three quarters of `max_output_bytes` in total. Larger files are
        then read with `files:read`. Files the command could not return (unexpected or cut
        output, no shell) are read one by one, so each still gets its own result.
        """
        results: list[bytes | SpaceError | None] = [None] * len(paths)
        group: list[int] = []
        group_bytes = 0
        for index, path in enumerate(paths):
            if not path.startswith("/"):
                continue  # read one by one below, resolved like the file routes resolve it
            size = len(path.encode())
            if group and (
                len(group) == _READ_FILES_MAX_PATHS
                or group_bytes + size > _READ_FILES_MAX_PATH_BYTES
            ):
                self._read_group(sandbox_id, paths, group, results)
                group, group_bytes = [], 0
            group.append(index)
            group_bytes += size
        if len(group) > 1:  # one file costs two requests either way; skip the command
            self._read_group(sandbox_id, paths, group, results)
        rest = [index for index, result in enumerate(results) if result is None]
        for index, result in zip(
            rest, read_files_one_by_one(self, sandbox_id, [paths[i] for i in rest]), strict=True
        ):
            results[index] = result
        return [result for result in results if result is not None]

    def _read_group(
        self,
        sandbox_id: str,
        paths: Sequence[str],
        group: list[int],
        results: list[bytes | SpaceError | None],
    ) -> None:
        """Fill `results` for the paths at `group` that one command could read."""
        budget = max(self._max_output_bytes * 3 // 4 - 64 * 1024, 0)
        argv = ["sh", "-c", _READ_FILES_SCRIPT, "sh", str(budget), *(paths[i] for i in group)]
        try:
            output = self.run_command(sandbox_id, argv, timeout=300).output
        except SpaceError as error:
            logger.debug("reading %d files in one command failed (%s)", len(group), error)
            return
        records = _parse_read_files(output, [paths[i] for i in group])
        for index, record in zip(group, records, strict=True):
            if isinstance(record, int):  # over the budget: its size is known, read it directly
                try:
                    results[index] = self.read_file(sandbox_id, paths[index], size=record)
                except SpaceError as error:
                    results[index] = error
            else:
                results[index] = record

    def stat(self, sandbox_id: str, path: str) -> FileStat:
        """See `SpaceClient.stat`."""
        response = self._json(
            "POST", f"/v1/sandboxes/{_q(sandbox_id)}/files:stat", body={"path": path}
        )
        entry = response.get("entry")
        if not isinstance(entry, dict):
            raise SpaceError("files:stat response has no entry", code="invalid_response")
        file_type = entry.get("file_type", "other")
        return FileStat(
            path=str(entry.get("path", path)),
            file_type=file_type if file_type in ("file", "directory", "symlink") else "other",
            size_bytes=int(entry.get("size_bytes", 0)),
        )

    def mkdir(self, sandbox_id: str, path: str, *, parents: bool = True) -> None:
        """See `SpaceClient.mkdir`."""
        self._json(
            "POST",
            f"/v1/sandboxes/{_q(sandbox_id)}/files:mkdir",
            body={"path": path, "parents": parents},
            mutation=True,
        )

    # ------------------------------------------------------------------ credentials

    def register_credential_reference(self, secret_arn: str) -> str:
        """See `SpaceClient.register_credential_reference`."""
        response = self._json(
            "POST",
            "/v1/credential-references",
            body={
                "provider": "aws_secrets_manager",
                "request_id": uuid.uuid4().hex,
                "secret_arn": secret_arn,
            },
            mutation=True,
            expect=(201,),
        )
        return _str(response, "credential_reference_id")

    def retire_credential_reference(self, credential_reference_id: str) -> None:
        """See `SpaceClient.retire_credential_reference`."""
        try:
            self._request(
                "DELETE",
                f"{self._api_url}/v1/credential-references/{_q(credential_reference_id)}",
                mutation=True,
                expect=(204,),
            )
        except SpaceError as error:
            if error.status != 404:
                raise

    def set_network_policy(self, sandbox_id: str, policy: Mapping[str, Any]) -> None:
        """See `SpaceClient.set_network_policy`."""
        response = self._json(
            "PATCH",
            f"/v1/sandboxes/{_q(sandbox_id)}",
            body={"network_policy": {"mode": "set", "policy": dict(policy)}},
            mutation=True,
            expect=(202,),
        )
        self._wait_operation(_str(response, "operation_id"))

    # ------------------------------------------------------------------ plumbing

    def _lifecycle(self, sandbox_id: str, action: str) -> None:
        response = self._json(
            "POST", f"/v1/sandboxes/{_q(sandbox_id)}:{action}", mutation=True, expect=(202,)
        )
        self._wait_operation(_str(response, "operation_id"))

    def _wait_operation(self, operation_id: str) -> None:
        payload = self._poll(
            f"/v1/operations/{_q(operation_id)}",
            ready="succeeded",
            terminal=_TERMINAL_OPERATION_STATES,
        )
        if payload.get("state") != "succeeded":
            detail = payload.get("terminal_error")
            message = detail.get("message") if isinstance(detail, dict) else None
            raise SpaceError(
                str(message or f"operation {operation_id} ended in {payload.get('state')}"),
                code=str(payload.get("terminal_error_code") or "operation_failed"),
            )

    def _wait_running(self, sandbox_id: str) -> None:
        self._wait_sandbox_state(sandbox_id, "running")
        # `running` can precede guest-agent readiness; listing processes proves the guest
        # answers (same probe as SPACE's own examples).
        self._json("GET", f"/v1/sandboxes/{_q(sandbox_id)}/processes?limit=1")

    def _wait_sandbox_state(self, sandbox_id: str, ready: str) -> None:
        payload = self._poll(
            f"/v1/sandboxes/{_q(sandbox_id)}", ready=ready, terminal=_TERMINAL_SANDBOX_STATES
        )
        if payload.get("state") != ready:
            raise SpaceError(
                f"sandbox {sandbox_id} ended in {payload.get('state')} while waiting for {ready}",
                code="unexpected_state",
            )

    def _poll(self, path: str, *, ready: str, terminal: frozenset[str]) -> dict[str, Any]:
        deadline = time.monotonic() + self._ready_timeout
        delay = min(_FIRST_POLL_DELAY, self._poll_interval)
        while True:
            payload = self._json("GET", path)
            state = payload.get("state")
            if state == ready or state in terminal:
                return payload
            if time.monotonic() >= deadline:
                raise SpaceError(
                    f"{path} did not reach {ready} within {self._ready_timeout:g}s "
                    f"(last state {state!r})",
                    code="timeout",
                )
            time.sleep(delay)
            delay = min(delay * 2, self._poll_interval)

    def _best_effort_delete(self, sandbox_id: str) -> None:
        # Cleanup after a failed create must not mask the original error.
        with contextlib.suppress(Exception):
            self.delete_sandbox(sandbox_id)

    def _json(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        mutation: bool = False,
        expect: tuple[int, ...] = (200,),
    ) -> dict[str, Any]:
        response = self._request(
            method, f"{self._api_url}{path}", body=body, mutation=mutation, expect=expect
        )
        if not response.content:
            return {}
        payload = response.json()
        if not isinstance(payload, dict):
            raise SpaceError(f"expected a JSON object from {path}", code="invalid_response")
        return payload

    def _request(
        self,
        method: str,
        url: str,
        *,
        body: Mapping[str, Any] | None = None,
        mutation: bool = False,
        expect: tuple[int, ...] = (200,),
        headers: Mapping[str, str] | None = None,
        auth: bool = True,
    ) -> httpx.Response:
        request_headers = {"Accept": "application/json"}
        if auth:
            request_headers["Authorization"] = f"Bearer {self._token}"
        if mutation:
            # One key per logical request, reused across retries so SPACE deduplicates them.
            request_headers["Idempotency-Key"] = uuid.uuid4().hex
        request_headers.update(headers or {})
        attempt = 0
        while True:
            try:
                response = self._http.request(
                    method,
                    url,
                    json=dict(body) if body is not None else None,
                    headers=request_headers,
                )
            except httpx.TransportError as transport_error:
                if attempt + 1 >= self._retry.max_attempts:
                    raise SpaceError(
                        f"{method} {url}: {transport_error}", code="transport", retryable=True
                    ) from transport_error
                time.sleep(self._retry.delay(attempt, None))
                attempt += 1
                continue
            if response.status_code in expect:
                return response
            space_error = _error_from_response(response)
            if not space_error.retryable or attempt + 1 >= self._retry.max_attempts:
                raise space_error
            time.sleep(self._retry.delay(attempt, _retry_after(response)))
            attempt += 1


def _error_from_response(response: httpx.Response) -> SpaceError:
    """Map a SPACE error envelope (or a bare HTTP failure) to `SpaceError`."""
    status = response.status_code
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("code"), str):
        raw_details = payload.get("details") or []
        details = {
            str(item.get("key")): str(item.get("value"))
            for item in raw_details
            if isinstance(item, dict)
        }
        return SpaceError(
            str(payload.get("message", "")),
            code=payload["code"],
            status=status,
            retryable=bool(payload.get("retryable", False)),
            details=details,
            request_id=payload.get("request_id"),
        )
    # No valid envelope (e.g. the E2B routes): the HTTP status decides.
    message = payload.get("message") if isinstance(payload, dict) else None
    return SpaceError(
        str(message or response.text[:500] or response.reason_phrase),
        code=_CODE_BY_STATUS.get(status, "http_error"),
        status=status,
        retryable=status in (429, 503) or (status >= 500 and status != 501),
    )


_CODE_BY_STATUS = {
    400: "invalid_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    412: "failed_precondition",
    429: "rate_limited",
    503: "service_unavailable",
}


def _parse_read_files(output: str, paths: Sequence[str]) -> list[bytes | SpaceError | int | None]:
    """Decode `_READ_FILES_SCRIPT`'s records, one per path.

    An int is the size of a file the command left out; None means the output stopped
    making sense there (e.g. it was cut), so that path and the ones after it are unknown.
    """
    lines = output.split("\n")
    records: list[bytes | SpaceError | int | None] = []
    position = 0
    for path in paths:
        if position >= len(lines):
            break
        kind, _, size = lines[position].partition(" ")
        position += 1
        if kind == "D":
            records.append(SpaceError(f"{path} is a directory", code="is_directory"))
        elif kind == "N":
            records.append(
                SpaceError(
                    f"{path}: no such file or directory",
                    code="not_found",
                    details={"resource": "path"},
                )
            )
        elif kind == "P":
            records.append(SpaceError(f"{path}: permission denied", code="forbidden"))
        elif kind == "B" and size.isdigit():
            records.append(int(size))
        elif kind == "F" and size.isdigit() and position < len(lines):
            try:
                data = base64.b64decode(lines[position], validate=True)
            except binascii.Error:
                break
            position += 1
            if len(data) != int(size):
                break
            records.append(data)
        else:
            break
    return records + [None] * (len(paths) - len(records))


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SpaceError(f"response field {key!r} missing or not a string", code="invalid_response")
    return value


def _q(segment: str) -> str:
    return quote(segment, safe="")


# The native API names a sandbox `sbx_<uuid>` (`PublicSandboxId`, which rejects a bare UUID); the
# E2B-compatible API takes and returns the bare UUID (`SandboxId`, which rejects the prefix and
# answers "Sandbox not found"). Same UUID either way. SPACE: crates/libs/public/api/src/lib.rs,
# crates/libs/common/core/src/types/sandbox.rs, crates/libs/api/src/routes/e2b/translate.rs.
_SANDBOX_PREFIX = "sbx_"


def _ecw_sandbox_id(sandbox_id: str) -> str:
    return sandbox_id.removeprefix(_SANDBOX_PREFIX)


def _native_sandbox_id(sandbox_id: str) -> str:
    return sandbox_id if sandbox_id.startswith(_SANDBOX_PREFIX) else _SANDBOX_PREFIX + sandbox_id


def _origin(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    return parts.scheme, parts.netloc
