"""The bug: with a plain sandbox backend, rewinding a LangGraph thread does not rewind files.

These tests pass *because* the plain backend is inconsistent. They document the problem
that SpaceCheckpointer solves (see test_space_checkpointer.py).
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langchain_space.backend import SpaceSandbox
from langchain_space.testing import FakeSpaceClient
from langchain_space.testing.graphs import Observation, build_edit_graph, run_scenarios

pytestmark = pytest.mark.consistency


@pytest.fixture(scope="module")
def observations() -> dict[str, Observation]:
    backend = SpaceSandbox.create(FakeSpaceClient(), "code-runtime")
    graph = build_edit_graph(backend, InMemorySaver())
    return {o.scenario: o for o in run_scenarios(graph)}


@pytest.mark.parametrize(
    ("scenario", "stale_content"),
    [
        ("rewind", "v3"),  # rewound to v1, but the file still holds the latest write
        ("replay", "v3"),
        ("branch-b", "a"),  # branch b sees branch a's file
        ("branch-a-resumed", "b"),  # and resuming a sees b's
    ],
)
def test_plain_backend_leaves_sandbox_in_wrong_state(
    observations: dict[str, Observation], scenario: str, stale_content: str
) -> None:
    observation = observations[scenario]
    assert not observation.consistent
    assert observation.observed == stale_content
