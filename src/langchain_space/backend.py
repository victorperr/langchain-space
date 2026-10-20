"""`SpaceSandbox`: a Deep Agents sandbox backend running on one SPACE sandbox.

This module must not import LangGraph, so it stays usable from a plain script.
"""

from __future__ import annotations

import posixpath
import threading
from collections.abc import Callable, Mapping
from typing import Any

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

from langchain_space.client import SpaceClient, SpaceError

DEFAULT_MAX_OUTPUT_CHARS = 200_000

ContentCheck = Callable[[bytes, str], None]
"""`(content, where) -> None`; raises to block content from being sent into the sandbox."""


class SpaceSandbox(BaseSandbox):
    """Deep Agents backend bound to one SPACE sandbox.

    `BaseSandbox` derives `ls`, `read`, `write`, `edit`, `grep` and `glob` from the three
    primitives implemented here. The template must provide a POSIX shell and `python3`
    (SPACE's `code-runtime` template does).

    Every call that can change the filesystem (`execute`, `upload_files`) bumps
    `generation`, so snapshot logic can tell whether anything changed since the last
    snapshot without asking SPACE.

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
    ) -> None:
        self._client = client
        self._sandbox_id = sandbox_id
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars
        self._owns_sandbox = owns_sandbox
        self._content_check = content_check
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
        return cls(client, sandbox_id, owns_sandbox=True, **kwargs)

    @property
    def id(self) -> str:
        """The SPACE sandbox id (`sbx_...`)."""
        return self._sandbox_id

    @property
    def client(self) -> SpaceClient:
        """The client this backend talks through."""
        return self._client

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
        # Bump before running: a command that fails halfway may still have written files.
        self._bump()
        effective_timeout = timeout if timeout is not None else self._default_timeout
        result = self._client.run_command(
            self._sandbox_id, ["sh", "-c", command], timeout=effective_timeout
        )
        output = result.output
        truncated = len(output) > self._max_output_chars
        if truncated:
            output = output[: self._max_output_chars]
        return ExecuteResponse(output=output, exit_code=result.exit_code, truncated=truncated)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write each file, creating parent directories. Errors are reported per file."""
        if files:
            self._bump()
        responses: list[FileUploadResponse] = []
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
                if parent and parent != "/":
                    self._client.mkdir(self._sandbox_id, parent, parents=True)
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
                if self._client.stat(self._sandbox_id, path).file_type == "directory":
                    responses.append(FileDownloadResponse(path, error=IS_DIRECTORY))
                    continue
                content = self._client.read_file(self._sandbox_id, path)
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
