"""The standard langchain-tests sandbox suite, run against a real SPACE sandbox."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_space.backend import SpaceSandbox
from langchain_space.client import HttpSpaceClient

pytestmark = pytest.mark.integration

# langchain-tests 1.1.9 (its latest release, May 2026) still checks the old BaseSandbox
# contract. The deepagents 0.7 BaseSandbox we inherit these methods from now returns absolute
# glob paths, lists files only (its glob script filters with os.path.isfile), and lets `write`
# overwrite an existing file. Every BaseSandbox backend fails these the same way. strict=True:
# once the suite catches up they pass, xfail turns into a failure, and the marker can go.
_STALE_CONTRACT = pytest.mark.xfail(
    strict=True,
    reason="langchain-tests 1.1.9 predates the deepagents 0.7 glob/write contract",
)


class TestSpaceSandboxStandard(SandboxIntegrationTests):
    @pytest.fixture(scope="class")
    def sandbox(
        self, space_client: HttpSpaceClient, space_template: str
    ) -> Iterator[SandboxBackendProtocol]:
        backend = SpaceSandbox.create(space_client, space_template)
        try:
            yield backend
        finally:
            backend.terminate()

    @_STALE_CONTRACT
    def test_glob(self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str) -> None:
        super().test_glob(sandbox_backend, sandbox_test_root)

    @_STALE_CONTRACT
    def test_glob_basic_pattern(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_basic_pattern(sandbox_backend, sandbox_test_root)

    @_STALE_CONTRACT
    def test_glob_with_directories(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_with_directories(sandbox_backend, sandbox_test_root)

    @_STALE_CONTRACT
    def test_glob_hidden_files_explicitly(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_hidden_files_explicitly(sandbox_backend, sandbox_test_root)

    @_STALE_CONTRACT
    def test_glob_with_character_class(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_with_character_class(sandbox_backend, sandbox_test_root)

    @_STALE_CONTRACT
    def test_glob_with_question_mark(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_with_question_mark(sandbox_backend, sandbox_test_root)

    @_STALE_CONTRACT
    def test_write_existing_file_fails(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_write_existing_file_fails(sandbox_backend, sandbox_test_root)
