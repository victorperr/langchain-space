"""The standard langchain-tests sandbox suite, run against a real SPACE sandbox."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_space.backend import SpaceSandbox
from langchain_space.client import HttpSpaceClient

pytestmark = pytest.mark.integration


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
