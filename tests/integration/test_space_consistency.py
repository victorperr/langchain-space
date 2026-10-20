"""The core claim on real SPACE sandboxes: rewinds and forks restore the files."""

from __future__ import annotations

import pytest

from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.testing.graphs import build_edit_graph, run_scenarios

pytestmark = pytest.mark.integration


def test_time_travel_scenarios_are_consistent(space_saver: SpaceCheckpointer) -> None:
    graph = build_edit_graph(space_saver.backend, space_saver)
    observations = run_scenarios(graph, durability="async")
    inconsistent = [o for o in observations if not o.consistent]
    assert not inconsistent, inconsistent


def test_anchors_are_suspended_and_restorable(space_saver: SpaceCheckpointer) -> None:
    graph = build_edit_graph(space_saver.backend, space_saver)
    graph.invoke({"target": "v1"}, {"configurable": {"thread_id": "anchor-check"}})
    [anchor] = space_saver.registry.anchors

    restored = space_saver.client.fork(anchor)
    try:
        assert space_saver.client.read_file(restored, "/home/user/work/app.txt") == b"v1"
    finally:
        space_saver.client.delete_sandbox(restored)
