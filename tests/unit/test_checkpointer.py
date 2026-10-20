"""SpaceCheckpointer's binding rules, on the fake client."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.client import SpaceError
from langchain_space.snapshots import SPACE_METADATA_KEY, MissingSnapshotError
from langchain_space.testing import FakeSpaceClient
from langchain_space.testing.graphs import EDIT_FILE, build_edit_graph, find_state, thread


@pytest.fixture
def fake() -> FakeSpaceClient:
    return FakeSpaceClient()


@pytest.fixture
def saver(fake: FakeSpaceClient) -> SpaceCheckpointer:
    return SpaceCheckpointer(InMemorySaver(), fake, template="code-runtime")


def file_in(fake: FakeSpaceClient, sandbox_id: str) -> bytes:
    return fake.sandbox(sandbox_id).files[EDIT_FILE]


def test_every_checkpoint_is_bound(saver: SpaceCheckpointer) -> None:
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2"):
        graph.invoke({"target": target}, thread("t"))

    stored = list(saver.inner.list(thread("t")))
    assert stored
    assert all(SPACE_METADATA_KEY in t.metadata for t in stored)


def test_no_snapshot_when_sandbox_unchanged(fake: FakeSpaceClient) -> None:
    class S(TypedDict):
        n: int

    saver = SpaceCheckpointer(InMemorySaver(), fake, template="code-runtime")
    builder = StateGraph(S)
    builder.add_node("think", lambda s: {"n": s["n"] + 1})  # never touches the sandbox
    builder.add_edge(START, "think")
    builder.add_edge("think", END)
    graph = builder.compile(checkpointer=saver)

    for _ in range(3):
        graph.invoke({"n": 0}, thread("t"))

    assert fake.calls_to("fork") == []
    assert fake.calls_to("create_sandbox") == []  # sandboxes are created lazily


def _read_mostly_graph(saver: SpaceCheckpointer) -> Any:
    """`START -> work -> END`; `work` writes a file on the first run, then only reads."""

    class S(TypedDict):
        n: int

    def work(state: S) -> S:
        backend = saver.backend
        if state["n"] == 0:
            backend.upload_files([(EDIT_FILE, b"v1")])
        backend.ls("/home/user")
        backend.read(EDIT_FILE)
        backend.grep("v1", "/home/user")
        backend.glob("*.txt", "/home/user")
        return {"n": state["n"] + 1}

    builder = StateGraph(S)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    return builder.compile(checkpointer=saver)


def test_read_only_steps_take_no_snapshot(fake: FakeSpaceClient, saver: SpaceCheckpointer) -> None:
    graph = _read_mostly_graph(saver)
    for n in range(3):
        graph.invoke({"n": n}, thread("t"))
    assert fake.calls_to("run_command")  # the reads did reach the sandbox
    assert len(saver.registry.anchors) == 1  # only the step that wrote was snapshotted


async def test_async_read_only_steps_take_no_snapshot(
    fake: FakeSpaceClient, saver: SpaceCheckpointer
) -> None:
    class S(TypedDict):
        n: int

    async def work(state: S) -> S:
        backend = saver.backend
        if state["n"] == 0:
            await backend.aupload_files([(EDIT_FILE, b"v1")])
        await backend.als("/home/user")
        await backend.aread(EDIT_FILE)
        await backend.agrep("v1", "/home/user")
        await backend.aglob("*.txt", "/home/user")
        return {"n": state["n"] + 1}

    builder = StateGraph(S)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    graph = builder.compile(checkpointer=saver)
    for n in range(3):
        await graph.ainvoke({"n": n}, thread("t"))
    assert fake.calls_to("run_command")
    assert len(saver.registry.anchors) == 1


def test_routed_write_resolves_the_sandbox_once() -> None:
    # `write` is a mkdir command plus an upload; the routed backend forwards it whole.
    fake = FakeSpaceClient(lambda sandbox, argv: ("", 0))
    saver = SpaceCheckpointer(InMemorySaver(), fake, template="code-runtime")
    acquired: list[str | None] = []
    original = saver.registry.acquire

    def counting(thread_id: str, base: str | None) -> Any:
        acquired.append(base)
        return original(thread_id, base)

    saver.registry.acquire = counting  # type: ignore[method-assign]

    class S(TypedDict):
        ok: bool

    def work(state: S) -> S:
        return {"ok": saver.backend.write(EDIT_FILE, "v1").error is None}

    builder = StateGraph(S)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    result = builder.compile(checkpointer=saver).invoke({"ok": False}, thread("t"))

    assert result["ok"]
    assert len(acquired) == 1
    assert fake.calls_to("write_file")


def test_one_snapshot_per_state_change(fake: FakeSpaceClient, saver: SpaceCheckpointer) -> None:
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2", "v3"):
        graph.invoke({"target": target}, thread("t"))
    assert len(saver.registry.anchors) == 3


def test_write_before_commit(fake: FakeSpaceClient, saver: SpaceCheckpointer) -> None:
    graph = build_edit_graph(saver.backend, saver)
    graph.invoke({"target": "v1"}, thread("t"))
    stored_before = len(list(saver.inner.list(thread("t"))))

    fake.fork_error = SpaceError("capture failed", code="internal")
    with pytest.raises(SpaceError, match="capture failed"):
        graph.invoke({"target": "v2"}, thread("t"), durability="sync")

    history = list(graph.get_state_history(thread("t")))
    # The checkpoint after the failed snapshot (v2 written, run finished) was never saved...
    assert not [s for s in history if s.values.get("target") == "v2" and s.next == ()]
    assert len(history) == stored_before + 2  # only run 2's input and pre-edit checkpoints
    # ...and the newest saved one still binds to a real snapshot holding v1.
    newest_anchor = history[0].metadata[SPACE_METADATA_KEY]["anchor_id"]
    assert file_in(fake, newest_anchor) == b"v1"


def test_rewind_forks_new_sandbox_and_leaves_old_one_untouched(
    fake: FakeSpaceClient, saver: SpaceCheckpointer
) -> None:
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2"):
        graph.invoke({"target": target}, thread("t"))
    old_live = saver.registry.live_sandbox_id("t")
    assert old_live is not None

    after_v1 = find_state(graph, "t", target="v1", finished=True)
    graph.invoke({"target": "v9"}, after_v1.config)

    new_live = saver.registry.live_sandbox_id("t")
    assert new_live != old_live
    assert file_in(fake, old_live) == b"v2"  # retired, never mutated
    assert old_live in saver.registry.retired
    assert file_in(fake, new_live) == b"v9"


def test_retired_sandboxes_can_be_deleted_eagerly(fake: FakeSpaceClient) -> None:
    saver = SpaceCheckpointer(
        InMemorySaver(), fake, template="code-runtime", delete_retired_eagerly=True
    )
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2"):
        graph.invoke({"target": target}, thread("t"))
    old_live = saver.registry.live_sandbox_id("t")
    assert old_live is not None

    after_v1 = find_state(graph, "t", target="v1", finished=True)
    graph.invoke({"target": "v9"}, after_v1.config)

    assert fake.sandboxes[old_live].state == "deleted"
    assert saver.registry.retired == []
    assert file_in(fake, saver.registry.live_sandbox_id("t") or "") == b"v9"


def test_failed_retired_delete_is_kept_for_the_next_attempt(
    fake: FakeSpaceClient, saver: SpaceCheckpointer, monkeypatch: pytest.MonkeyPatch
) -> None:
    saver.registry.retired.extend(["sbx_a", "sbx_b"])

    def flaky(sandbox_id: str) -> None:
        if sandbox_id == "sbx_a":
            raise SpaceError("busy", code="conflict")

    monkeypatch.setattr(fake, "delete_sandbox", flaky)
    with pytest.raises(SpaceError, match="busy"):
        saver.registry.delete_retired()
    assert saver.registry.retired == ["sbx_a"]


def test_anchors_are_never_written(fake: FakeSpaceClient, saver: SpaceCheckpointer) -> None:
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2", "v3"):
        graph.invoke({"target": target}, thread("t"))
    graph.invoke({"target": "x"}, find_state(graph, "t", target="v1", finished=True).config)

    anchors = set(saver.registry.anchors)
    writes = fake.calls_to("write_file") + fake.calls_to("run_command")
    assert anchors
    assert not anchors & {c.args["sandbox_id"] for c in writes}
    assert all(fake.sandbox(a).state == "suspended" for a in anchors)


def test_checkpoint_without_binding_is_an_error() -> None:
    class NoSandbox(TypedDict):
        target: str

    plain = InMemorySaver()
    builder = StateGraph(NoSandbox)
    builder.add_node("noop", lambda s: {})
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    builder.compile(checkpointer=plain).invoke({"target": "x"}, thread("legacy"))

    # A thread saved without SpaceCheckpointer has no snapshots: refuse, don't guess.
    wrapped = SpaceCheckpointer(plain, FakeSpaceClient(), template="t")
    with pytest.raises(MissingSnapshotError):
        wrapped.get_tuple(thread("legacy"))


def test_exit_durability_is_rejected(saver: SpaceCheckpointer) -> None:
    graph = build_edit_graph(saver.backend, saver)
    with pytest.raises(ValueError, match="durability"):
        graph.invoke({"target": "v1"}, thread("t"), durability="exit")


def test_backend_outside_a_run_points_to_sandbox_helper(saver: SpaceCheckpointer) -> None:
    with pytest.raises(RuntimeError, match="sandbox\\(thread_id\\)"):
        saver.backend.download_files(["/x"])


def test_a_new_process_restores_from_stored_bindings(fake: FakeSpaceClient) -> None:
    storage = InMemorySaver()
    first = SpaceCheckpointer(storage, fake, template="code-runtime")
    graph = build_edit_graph(first.backend, first)
    for target in ("v1", "v2"):
        graph.invoke({"target": target}, thread("t"))

    # Same storage and SPACE, fresh in-process registry (as after a restart).
    second = SpaceCheckpointer(storage, fake, template="code-runtime")
    graph = build_edit_graph(second.backend, second)
    after_v1 = find_state(graph, "t", target="v1", finished=True)
    result = graph.invoke({"target": "v3"}, after_v1.config)
    assert result["observed"][-1] == "v1"

    # And continuing the latest checkpoint restores the latest files.
    third = SpaceCheckpointer(storage, fake, template="code-runtime")
    result = build_edit_graph(third.backend, third).invoke({"target": "v4"}, thread("t"))
    assert result["observed"][-1] == "v3"


async def test_async_api_is_consistent(saver: SpaceCheckpointer) -> None:
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2"):
        await graph.ainvoke({"target": target}, thread("t"))
    after_v1 = None
    async for snapshot in graph.aget_state_history(thread("t")):
        if snapshot.values.get("target") == "v1" and snapshot.next == ():
            after_v1 = snapshot
            break
    assert after_v1 is not None
    result = await graph.ainvoke({"target": "v3"}, after_v1.config)
    assert result["observed"][-1] == "v1"


def test_unreferenced_anchors_are_deleted(fake: FakeSpaceClient, saver: SpaceCheckpointer) -> None:
    graph = build_edit_graph(saver.backend, saver)
    for target in ("v1", "v2"):
        graph.invoke({"target": target}, thread("t"))
    anchors = list(saver.registry.anchors)

    saver.delete_thread("t")
    deleted = saver.delete_unreferenced_anchors()
    saver.close()

    assert sorted(deleted) == sorted(anchors)
    assert fake.live_sandboxes == []


def test_anchor_of_a_checkpoint_being_saved_is_kept(
    fake: FakeSpaceClient, saver: SpaceCheckpointer
) -> None:
    graph = build_edit_graph(saver.backend, saver)
    graph.invoke({"target": "v1"}, thread("t"))
    latest = saver.get_tuple(thread("t"))
    assert latest is not None

    # Write-before-commit window: the snapshot exists, its checkpoint is not stored yet.
    saver.sandbox("t").upload_files([(EDIT_FILE, b"v2")])
    binding = saver.registry.bind("t", "", latest.config["configurable"]["checkpoint_id"], "next")
    assert binding.anchor_id is not None

    assert saver.delete_unreferenced_anchors() == []
    assert fake.sandbox(binding.anchor_id).files[EDIT_FILE] == b"v2"


def test_anchor_cleanup_ignores_checkpoints_saved_without_snapshots(
    fake: FakeSpaceClient,
) -> None:
    class S(TypedDict):
        n: int

    storage = InMemorySaver()
    builder = StateGraph(S)
    builder.add_node("noop", lambda s: {})
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    builder.compile(checkpointer=storage).invoke({"n": 0}, thread("other-app"))

    saver = SpaceCheckpointer(storage, fake, template="code-runtime")
    graph = build_edit_graph(saver.backend, saver)
    graph.invoke({"target": "v1"}, thread("t"))
    anchors = list(saver.registry.anchors)
    saver.delete_thread("t")

    assert anchors
    assert saver.delete_unreferenced_anchors() == anchors
