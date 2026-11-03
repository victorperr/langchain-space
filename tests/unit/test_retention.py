"""Snapshot retention: old snapshots are deleted, newer checkpoints stay restorable."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langchain_space.checkpointer import SnapshotRetention, SpaceCheckpointer
from langchain_space.snapshots import MissingSnapshotError, SnapshotGoneError
from langchain_space.testing import FakeSpaceClient
from langchain_space.testing.graphs import build_edit_graph, find_state, last_observed, thread


@pytest.fixture
def fake() -> FakeSpaceClient:
    return FakeSpaceClient()


def make_saver(fake: FakeSpaceClient, retention: SnapshotRetention | None) -> SpaceCheckpointer:
    saver = SpaceCheckpointer(InMemorySaver(), fake, template="code-runtime", retention=retention)
    # A regression would leave a task waiting for a checkpoint that is never saved; fail fast.
    saver.registry._wait_timeout = 5
    return saver


def alive(fake: FakeSpaceClient, anchors: list[str]) -> list[str]:
    return [a for a in anchors if fake.sandboxes[a].state != "deleted"]


def test_keeps_only_the_newest_snapshots(fake: FakeSpaceClient) -> None:
    saver = make_saver(fake, SnapshotRetention(keep_last=2))
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2", "v3", "v4", "v5"):
        graph.invoke({"target": target}, thread("t"))

    created = [c.args["sandbox_id"] for c in fake.calls_to("suspend")]  # one per snapshot
    assert len(created) == 5
    assert alive(fake, created) == created[-2:]
    assert saver.registry.anchors == created[-2:]


def test_kept_checkpoint_still_restores(fake: FakeSpaceClient) -> None:
    saver = make_saver(fake, SnapshotRetention(keep_last=2))
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2", "v3", "v4", "v5"):
        graph.invoke({"target": target}, thread("t"))

    after_v4 = find_state(graph, "t", target="v4", finished=True)
    assert last_observed(graph.invoke({"target": "x"}, after_v4.config)) == "v4"


@pytest.mark.parametrize("durability", ["sync", "async"])
def test_rewind_to_a_pruned_checkpoint_is_a_clear_error(
    fake: FakeSpaceClient, durability: Any
) -> None:
    saver = make_saver(fake, SnapshotRetention(keep_last=1))
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2", "v3"):
        graph.invoke({"target": target}, thread("t"))
    stored_before = len(list(saver.inner.list(thread("t"))))

    after_v1 = find_state(graph, "t", target="v1", finished=True)
    with pytest.raises(SnapshotGoneError):
        graph.invoke({"target": "x"}, after_v1.config, durability=durability)

    # Write-before-commit: nothing was saved on top of the pruned snapshot.
    assert len(list(saver.inner.list(thread("t")))) == stored_before
    assert issubclass(SnapshotGoneError, MissingSnapshotError)  # existing handlers catch it


def test_keep_protects_marked_checkpoints(fake: FakeSpaceClient) -> None:
    keep = SnapshotRetention(keep_last=1, keep=lambda stored: stored.metadata.get("keep") is True)
    saver = make_saver(fake, keep)
    graph = build_edit_graph(saver.backend, saver)
    graph.invoke({"target": "v1"}, {**thread("t"), "metadata": {"keep": True}})
    for target in ("v2", "v3", "v4"):
        graph.invoke({"target": target}, thread("t"))

    after_v1 = find_state(graph, "t", target="v1", finished=True)
    assert last_observed(graph.invoke({"target": "x"}, after_v1.config)) == "v1"
    after_v2 = find_state(graph, "t", target="v2", finished=True)
    with pytest.raises(SnapshotGoneError):
        graph.invoke({"target": "y"}, after_v2.config)


def test_manual_prune_without_a_policy_on_the_checkpointer(fake: FakeSpaceClient) -> None:
    saver = make_saver(fake, None)
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2", "v3", "v4"):
        graph.invoke({"target": target}, thread("t"))
    created = list(saver.registry.anchors)
    assert alive(fake, created) == created  # nothing is pruned automatically

    deleted = saver.prune_snapshots(SnapshotRetention(keep_last=1))

    assert sorted(deleted) == sorted(created[:-1])
    assert alive(fake, created) == created[-1:]
    with pytest.raises(ValueError, match="SnapshotRetention"):
        saver.prune_snapshots()


def test_snapshot_shared_with_another_thread_is_kept(fake: FakeSpaceClient) -> None:
    saver = make_saver(fake, None)
    graph = build_edit_graph(saver.backend, saver)
    graph.invoke({"target": "v1"}, thread("t"))
    # Copy thread t into u, as a saver's `copy_thread` would: u now ends on v1's snapshot.
    parent: str | None = None
    for stored in reversed(list(saver.inner.list(thread("t")))):
        config = {"configurable": {"thread_id": "u", "checkpoint_ns": "", "checkpoint_id": parent}}
        versions = stored.checkpoint["channel_versions"]
        saver.inner.put(config, stored.checkpoint, stored.metadata, versions)  # type: ignore[arg-type]
        parent = stored.config["configurable"]["checkpoint_id"]
    for target in ("v2", "v3"):
        graph.invoke({"target": target}, thread("t"))

    saver.prune_snapshots(SnapshotRetention(keep_last=1))

    head_u = graph.get_state(thread("u"))
    assert head_u.values["target"] == "v1"
    assert last_observed(graph.invoke({"target": "x"}, head_u.config)) == "v1"


def test_snapshot_pruned_by_another_process_is_a_clear_error(fake: FakeSpaceClient) -> None:
    storage = InMemorySaver()
    first = SpaceCheckpointer(storage, fake, template="code-runtime")
    graph = build_edit_graph(first.backend, first)
    for target in ("v1", "v2", "v3"):
        graph.invoke({"target": target}, thread("t"))
    first.prune_snapshots(SnapshotRetention(keep_last=1))

    # A second process sharing the storage does not know which snapshots were pruned.
    second = SpaceCheckpointer(storage, fake, template="code-runtime")
    other_graph = build_edit_graph(second.backend, second)
    after_v1 = find_state(other_graph, "t", target="v1", finished=True)
    with pytest.raises(SnapshotGoneError, match="no longer exists"):
        other_graph.invoke({"target": "x"}, after_v1.config, durability="sync")


def test_snapshot_a_rewind_is_moving_onto_is_kept(fake: FakeSpaceClient) -> None:
    saver = make_saver(fake, None)
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2", "v3"):
        graph.invoke({"target": target}, thread("t"))
    after_v1 = find_state(graph, "t", target="v1", finished=True)
    v1_anchor = saver.registry.anchors[0]

    # A rewind to v1 has bound its first checkpoint, which is not stored yet.
    saver.registry.bind("t", "", after_v1.config["configurable"]["checkpoint_id"], "next")
    deleted = saver.prune_snapshots(SnapshotRetention(keep_last=1))

    assert v1_anchor not in deleted
    assert len(deleted) == 1  # v2's snapshot
    assert saver.sandbox("t").download_files(["/home/user/work/app.txt"])[0].content == b"v1"


def test_failed_delete_is_retried_on_the_next_prune(
    fake: FakeSpaceClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    saver = make_saver(fake, None)
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2"):
        graph.invoke({"target": target}, thread("t"))
    oldest = saver.registry.anchors[0]
    real_delete = fake.delete_anchor

    def failing(anchor_id: str) -> None:
        raise RuntimeError("busy")

    monkeypatch.setattr(fake, "delete_anchor", failing)
    with pytest.raises(RuntimeError, match="busy"):
        saver.prune_snapshots(SnapshotRetention(keep_last=1))
    assert fake.sandboxes[oldest].state != "deleted"

    monkeypatch.setattr(fake, "delete_anchor", real_delete)
    assert saver.prune_snapshots(SnapshotRetention(keep_last=1)) == [oldest]


def test_keep_last_must_be_positive() -> None:
    with pytest.raises(ValueError, match="keep_last"):
        SnapshotRetention(keep_last=0)
