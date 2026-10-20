"""The fix: with SpaceCheckpointer, every resumed run sees the files of its checkpoint.

Same scenarios as test_plain_backend.py, under both LangGraph durability modes that save
intermediate checkpoints. `async` is the hard one: checkpoints are saved in the background
while the next step is already writing to the sandbox.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Durability

from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.testing import FakeSpaceClient
from langchain_space.testing.graphs import Observation, build_edit_graph, run_scenarios

pytestmark = pytest.mark.consistency

SCENARIOS = ["rewind", "replay", "branch-a", "branch-b", "branch-a-resumed"]


def observe(durability: Durability) -> dict[str, Observation]:
    saver = SpaceCheckpointer(InMemorySaver(), FakeSpaceClient(), template="code-runtime")
    graph = build_edit_graph(saver.backend, saver)
    return {o.scenario: o for o in run_scenarios(graph, durability=durability)}


@pytest.fixture(scope="module", params=["sync", "async"])
def observations(request: pytest.FixtureRequest) -> dict[str, Observation]:
    return observe(request.param)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_space_checkpointer_keeps_sandbox_consistent(
    observations: dict[str, Observation], scenario: str
) -> None:
    observation = observations[scenario]
    assert observation.consistent, observation
