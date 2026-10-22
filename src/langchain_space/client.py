"""Client for the SPACE API.

This is the only module that knows SPACE URLs and JSON shapes. Everything else depends on the
`SpaceClient` protocol, so tests can swap in `langchain_space.testing.FakeSpaceClient`.

API facts used here are verified against the SPACE source; see the "SPACE API notes" section
of CLAUDE.md for the references.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import random
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Literal, Protocol, Self, runtime_checkable
from urllib.parse import quote, urljoin, urlsplit

import httpx
from httpx_sse import connect_sse

FileType = Literal["file", "directory", "symlink", "other"]

MIB = 1024 * 1024
_FIRST_POLL_DELAY = 0.05
"""Seconds before the first re-poll; doubles up to `poll_interval` (fast ops finish fast)."""
_TERMINAL_OPERATION_STATES = frozenset({"canceled", "expired", "failed", "succeeded"})
_TERMINAL_SANDBOX_STATES = frozenset({"deleted", "deleting", "failed"})
_SUSPEND_UPLOAD_CONFLICT = "suspend-snapshot upload in flight"


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

    def read_file(self, sandbox_id: str, path: str, *, size: int | None = None) -> bytes:
        """Return the full contents of the file at `path`.

        `size` is the file's size in bytes when the caller already knows it (e.g. from
        `stat`), which saves the client from looking it up again.
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

    def register_credential_reference(self, secret_arn: str) -> str:
        """Register an AWS Secrets Manager secret by reference; return its reference id."""
        ...

    def set_network_policy(self, sandbox_id: str, policy: Mapping[str, Any]) -> None:
        """Replace the sandbox's network policy and wait until it is applied."""
        ...


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
        default_fork_lifetime: Seconds a forked sandbox may live when not specified.
        ready_timeout: Seconds to wait for an operation or a sandbox state.
        poll_interval: Longest pause between status polls (polling starts faster).
        max_output_bytes: Command output kept in memory; the rest is read and dropped,
            and the result is marked `truncated`.
        http: Optional preconfigured `httpx.Client` (tests inject a mocked transport).
    """

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
        """Close the underlying HTTP client if this instance created it."""
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

    def fork(self, sandbox_id: str, *, lifetime_seconds: int | None = None) -> str:
        """See `SpaceClient.fork`. Uses the E2B-compatible route (disk-only fork)."""
        if self._ecw_api_url is None or self._ecw_api_key is None:
            raise SpaceError(
                "fork needs ecw_api_url and ecw_api_key (SPACE's E2B-compatible API)",
                code="not_configured",
            )
        lifetime = lifetime_seconds or self.default_fork_lifetime
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
