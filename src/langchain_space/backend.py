"""`SpaceSandbox`: a Deep Agents sandbox backend running on one SPACE sandbox.

This module must not import LangGraph, so it stays usable from a plain script.
"""

from __future__ import annotations

import contextlib
import posixpath
import threading
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from typing import Any

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
)
from deepagents.backends.sandbox import BaseSandbox

from langchain_space.client import SpaceClient, SpaceError

DEFAULT_MAX_OUTPUT_CHARS = 200_000

ContentCheck = Callable[[bytes, str], None]
"""`(content, where) -> None`; raises to block content from being sent into the sandbox."""

_READ_ONLY: ContextVar[bool] = ContextVar("langchain_space_read_only", default=False)
"""Set while a `BaseSandbox` helper that only reads (`ls`, `read`, `grep`, `glob`) runs.

Those helpers are built on `execute`; the flag stops them from bumping `generation`. A
context variable (not a thread-local) so it also reaches the worker thread that
`aexecute` runs `execute` in.
"""


@contextlib.contextmanager
def _read_only() -> Iterator[None]:
    token = _READ_ONLY.set(True)
    try:
        yield
    finally:
        _READ_ONLY.reset(token)


class SpaceSandbox(BaseSandbox):
    """Deep Agents backend bound to one SPACE sandbox.

    `BaseSandbox` derives `ls`, `read`, `write`, `edit`, `grep` and `glob` from the three
    primitives implemented here. The template must provide a POSIX shell and `python3`
    (SPACE's `code-runtime` template does).

    Every call that can change the filesystem (`execute`, `upload_files`) bumps
    `generation`, so snapshot logic can tell whether anything changed since the last
    snapshot without asking SPACE. The read-only helpers `ls`, `read`, `grep` and `glob`
    run commands too, but do not bump it, so an agent step that only reads files does not
    cost a snapshot.

    Args:
        client: Any `SpaceClient` (the HTTP client, or the fake in tests).
        sandbox_id: The SPACE sandbox to drive. It must be running.
        default_timeout: Seconds per command when the caller passes none; `None` defers to
            the client's default.
        max_output_chars: Output beyond this is cut and `truncated` is set.
        owns_sandbox: Whether `terminate()` should delete the sandbox.
        content_check: Called with `(content, where)` before anything is sent into the
            sandbox (commands, uploaded files); raising blocks it. Use
            `langchain_space.credentials.SecretGuard` to keep host secrets out of the VM.
        network_policy: The SPACE network policy this sandbox runs under. Not applied
            here (`create` applies it); recorded so forks of it can re-apply it.
    """

    def __init__(
        self,
        client: SpaceClient,
        sandbox_id: str,
        *,
        default_timeout: int | None = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        owns_sandbox: bool = False,
        content_check: ContentCheck | None = None,
        network_policy: Mapping[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._sandbox_id = sandbox_id
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        self._owns_sandbox = owns_sandbox
        self._content_check = content_check
        self._network_policy = dict(network_policy) if network_policy is not None else None
        self._generation = 0
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        client: SpaceClient,
        template: str,
        *,
        env: Mapping[str, str] | None = None,
        network_policy: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> SpaceSandbox:
        """Create a new SPACE sandbox from `template` and wrap it (owned: `terminate` deletes).

        Raises whatever `content_check` raises if `env` contains blocked content.
        """
        check: ContentCheck | None = kwargs.get("content_check")
        if check is not None:
            for key, value in (env or {}).items():
                check(f"{key}={value}".encode(), "sandbox env")
        sandbox_id = client.create_sandbox(template, env=env, network_policy=network_policy)
        return cls(client, sandbox_id, owns_sandbox=True, network_policy=network_policy, **kwargs)

    def derive(self, sandbox_id: str, *, owns_sandbox: bool = True) -> SpaceSandbox:
        """A backend for another sandbox (e.g. a fork of this one) with the same settings.

        Keeps `content_check`, timeouts, output limit and network policy, so a fork is never
        less guarded than its source. Applying the policy to the new sandbox is the
        caller's job (see `fork_and_select`).
        """
        return type(self)(
            self._client,
            sandbox_id,
            default_timeout=self._default_timeout,
            max_output_chars=self._max_output_chars,
            owns_sandbox=owns_sandbox,
            content_check=self._content_check,
            network_policy=self._network_policy,
        )

    @property
    def id(self) -> str:
        """The SPACE sandbox id (`sbx_...`)."""
        return self._sandbox_id

    @property
    def client(self) -> SpaceClient:
        """The client this backend talks through."""
        return self._client

    @property
    def network_policy(self) -> dict[str, Any] | None:
        """The network policy this sandbox runs under, if one was set (a copy)."""
        return dict(self._network_policy) if self._network_policy is not None else None

    @property
    def generation(self) -> int:
        """Counter bumped by every call that may have changed the filesystem."""
        return self._generation

    def _bump(self) -> None:
        with self._lock:
            self._generation += 1

    def _blocked(self, content: bytes, where: str) -> str | None:
        """The refusal message if `content_check` blocks `content`, else None."""
        if self._content_check is None:
            return None
        try:
            self._content_check(content, where)
        except Exception as error:
            return f"blocked: {error}"
        return None

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run `command` with `sh -c`; stdout and stderr are merged in arrival order.

        `timeout=0` disables the deadline; a timed-out command exits with code 124. A command
        blocked by `content_check` is not sent and returns exit code 126 with the reason.
        """
        blocked = self._blocked(command.encode(), "command")
        if blocked is not None:
            return ExecuteResponse(output=blocked, exit_code=126)
        if not _READ_ONLY.get():
            # Bump before running: a command that fails halfway may still have written files.
            self._bump()
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = self._client.run_command(
            self._sandbox_id, ["sh", "-c", command], timeout=effective_timeout
        )
        output = result.output
        truncated = result.truncated or len(output) > self._max_output_chars
        output = output[: self._max_output_chars]
        return ExecuteResponse(output=output, exit_code=result.exit_code, truncated=truncated)

    # Read-only helpers: same commands as `BaseSandbox`, without marking the sandbox dirty.

    def ls(self, path: str) -> LsResult:
        """See `BaseSandbox.ls`; does not bump `generation`."""
        with _read_only():
            return super().ls(path)

    async def als(self, path: str) -> LsResult:
        """See `BaseSandbox.als`; does not bump `generation`."""
        with _read_only():
            return await super().als(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """See `BaseSandbox.read`; does not bump `generation`."""
        with _read_only():
            return super().read(file_path, offset, limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """See `BaseSandbox.aread`; does not bump `generation`."""
        with _read_only():
            return await super().aread(file_path, offset, limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """See `BaseSandbox.grep`; does not bump `generation`."""
        with _read_only():
            return super().grep(pattern, path, glob, max_count=max_count)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """See `BaseSandbox.agrep`; does not bump `generation`."""
        with _read_only():
            return await super().agrep(pattern, path, glob, max_count=max_count)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """See `BaseSandbox.glob`; does not bump `generation`."""
        with _read_only():
            return super().glob(pattern, path)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """See `BaseSandbox.aglob`; does not bump `generation`."""
        with _read_only():
            return await super().aglob(pattern, path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write each file, creating parent directories. Errors are reported per file."""
        if files:
            self._bump()
        responses: list[FileUploadResponse] = []
        made_dirs: set[str] = set()  # one mkdir per parent directory, not per file
        for path, content in files:
            if not _is_absolute(path):
                responses.append(FileUploadResponse(path, error=INVALID_PATH))
                continue
            blocked = self._blocked(path.encode() + b"\n" + content, f"file {path}")
            if blocked is not None:
                responses.append(FileUploadResponse(path, error=blocked))
                continue
            try:
                parent = posixpath.dirname(path)
                if parent and parent != "/" and parent not in made_dirs:
                    self._client.mkdir(self._sandbox_id, parent, parents=True)
                    made_dirs.add(parent)
                self._client.write_file(self._sandbox_id, path, content)
            except SpaceError as error:
                responses.append(FileUploadResponse(path, error=_file_error(error)))
            else:
                responses.append(FileUploadResponse(path))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read each file. Errors are reported per file."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not _is_absolute(path):
                responses.append(FileDownloadResponse(path, error=INVALID_PATH))
                continue
            try:
                stat = self._client.stat(self._sandbox_id, path)
                if stat.file_type == "directory":
                    responses.append(FileDownloadResponse(path, error=IS_DIRECTORY))
                    continue
                content = self._client.read_file(self._sandbox_id, path, size=stat.size_bytes)
            except SpaceError as error:
                responses.append(FileDownloadResponse(path, error=_file_error(error)))
            else:
                responses.append(FileDownloadResponse(path, content=content))
        return responses

    def terminate(self) -> None:
        """Delete the SPACE sandbox if this backend created it.

        Named `terminate` because `BaseSandbox.delete(path)` already deletes a *file*.
        """
        if self._owns_sandbox:
            self._client.delete_sandbox(self._sandbox_id)


def _is_absolute(path: str) -> bool:
    return path.startswith("/")


def _file_error(error: SpaceError) -> str:
    """Map a SPACE file-route error to Deep Agents' standard codes where one fits."""
    if error.is_path_not_found:
        return FILE_NOT_FOUND
    if error.code == "forbidden":
        return PERMISSION_DENIED
    if error.code == "invalid_request":
        return INVALID_PATH
    # Anything else (sandbox gone, disk full, ...) is not one of the LLM-fixable codes.
    return f"{error.code}: {error.message}"
