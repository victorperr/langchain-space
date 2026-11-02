"""LangSmithClient against real LangSmith sandboxes.

    LANGSMITH_API_KEY=... uv run pytest -m langsmith

Optional: LANGSMITH_TEMPLATE, a snapshot to boot from (default: LangSmith's default image).
Every sandbox and snapshot the tests create is deleted at teardown, also on failure.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.langsmith import LangSmithClient
from langchain_space.testing.graphs import build_edit_graph, run_scenarios

pytestmark = pytest.mark.langsmith


@pytest.fixture(scope="module")
def client() -> Iterator[LangSmithClient]:
    if not os.environ.get("LANGSMITH_API_KEY"):
        pytest.skip("LangSmith not configured (missing LANGSMITH_API_KEY)")
    with LangSmithClient() as client:
        yield client


@pytest.fixture(scope="module")
def template() -> str:
    return os.environ.get("LANGSMITH_TEMPLATE", "")


def test_restored_anchor_has_the_anchored_files(client: LangSmithClient, template: str) -> None:
    sandboxes: list[str] = []
    anchors: list[str] = []
    try:
        source = client.create_sandbox(template)
        sandboxes.append(source)
        client.mkdir(source, "/tmp/lcs")
        client.write_file(source, "/tmp/lcs/a.txt", b"v1")
        anchors.append(client.create_anchor(source))
        client.write_file(source, "/tmp/lcs/a.txt", b"v2")

        restored = client.restore_anchor(anchors[0])
        sandboxes.append(restored)

        assert client.read_file(restored, "/tmp/lcs/a.txt") == b"v1"
        assert client.stat(restored, "/tmp/lcs").file_type == "directory"
        assert client.run_command(restored, ["cat", "/tmp/lcs/a.txt"]).output == "v1"
    finally:
        for sandbox_id in sandboxes:
            with contextlib.suppress(Exception):
                client.delete_sandbox(sandbox_id)
        for anchor_id in anchors:
            with contextlib.suppress(Exception):
                client.delete_anchor(anchor_id)


@pytest.mark.consistency
def test_rewind_restores_the_sandbox(client: LangSmithClient, template: str) -> None:
    saver = SpaceCheckpointer(InMemorySaver(), client, template=template)
    graph = build_edit_graph(saver.backend, saver)
    try:
        observations = run_scenarios(graph)
    finally:
        saver.close()
        for anchor_id in saver.registry.anchors:
            with contextlib.suppress(Exception):
                client.delete_anchor(anchor_id)
    assert [o for o in observations if not o.consistent] == []
