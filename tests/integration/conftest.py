"""Fixtures for tests against a real SPACE deployment.

Set SPACE_API_URL, SPACE_DATA_PLANE_URL and SPACE_API_TOKEN. Snapshot and fork tests also
need SPACE_ECW_API_URL and SPACE_ECW_API_KEY (SPACE's E2B-compatible API serves forks).
SPACE_TEMPLATE defaults to `code-runtime`.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.client import HttpSpaceClient

REQUIRED_ENV = ("SPACE_API_URL", "SPACE_DATA_PLANE_URL", "SPACE_API_TOKEN")
FORK_ENV = ("SPACE_ECW_API_URL", "SPACE_ECW_API_KEY")


def _require(names: tuple[str, ...]) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.skip(f"SPACE not configured (missing {', '.join(missing)})")


@pytest.fixture(scope="session")
def space_client() -> Iterator[HttpSpaceClient]:
    _require(REQUIRED_ENV)
    with HttpSpaceClient.from_env() as client:
        yield client


@pytest.fixture(scope="session")
def forking_client(space_client: HttpSpaceClient) -> HttpSpaceClient:
    _require(FORK_ENV)
    return space_client


@pytest.fixture(scope="session")
def space_template() -> str:
    return os.environ.get("SPACE_TEMPLATE", "code-runtime")


@pytest.fixture
def space_saver(
    forking_client: HttpSpaceClient, space_template: str
) -> Iterator[SpaceCheckpointer]:
    """A SpaceCheckpointer whose sandboxes *and* anchors are deleted at teardown."""
    saver = SpaceCheckpointer(InMemorySaver(), forking_client, template=space_template)
    try:
        yield saver
    finally:
        saver.close()
        for anchor_id in saver.registry.anchors:
            with contextlib.suppress(Exception):
                forking_client.delete_sandbox(anchor_id)
