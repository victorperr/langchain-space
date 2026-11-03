"""LangSmithClient against an in-memory LangSmith stand-in (no network, runs on Windows)."""

from __future__ import annotations

import shlex
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langsmith.sandbox import ExecutionResult, SandboxClient

from langchain_space import ProviderMismatchError, SpaceClient, SpaceError
from langchain_space.backend import SpaceSandbox
from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.fork import fork_and_select
from langchain_space.langsmith import LangSmithClient
from langchain_space.langsmith.testing import StubSandbox, StubSandboxClient, timing_out
from langchain_space.testing import FakeSpaceClient
from langchain_space.testing.graphs import build_edit_graph, run_scenarios, thread


@pytest.fixture
def stub() -> StubSandboxClient:
    return StubSandboxClient()


@pytest.fixture
def client(stub: StubSandboxClient) -> LangSmithClient:
    return LangSmithClient(cast(SandboxClient, stub))


def test_implements_the_space_client_protocol(client: LangSmithClient) -> None:
    assert isinstance(client, SpaceClient)


# ---------------------------------------------------------------------- lifecycle


def test_create_sandbox_boots_the_template_snapshot(
    client: LangSmithClient, stub: StubSandboxClient
) -> None:
    name = client.create_sandbox("python:3.12", env={"MODE": "test"})
    [call] = stub.calls_to("create_sandbox")
    assert call["snapshot"] == "python:3.12"
    assert call["run_config"] == {"env_vars": {"MODE": "test"}}
    assert call["idle_ttl_seconds"] == 3600
    assert name in stub.sandboxes


def test_empty_template_means_default_image(
    client: LangSmithClient, stub: StubSandboxClient
) -> None:
    client.create_sandbox("")
    assert stub.calls_to("create_sandbox")[0]["snapshot"] is None


def test_network_policy_is_refused(client: LangSmithClient) -> None:
    with pytest.raises(SpaceError) as error:
        client.create_sandbox("", network_policy={"egress": {}})
    assert error.value.code == "not_supported"


def test_delete_is_idempotent(client: LangSmithClient, stub: StubSandboxClient) -> None:
    name = client.create_sandbox("")
    client.delete_sandbox(name)
    client.delete_sandbox(name)
    assert name not in stub.sandboxes


def test_suspend_and_wake_stop_and_start(client: LangSmithClient, stub: StubSandboxClient) -> None:
    name = client.create_sandbox("")
    client.suspend(name)
    assert stub.sandboxes[name].status == "stopped"
    client.wake(name)
    assert stub.sandboxes[name].status == "ready"


# ---------------------------------------------------------------------- snapshots


def test_restored_anchor_has_the_files_of_the_anchored_moment(client: LangSmithClient) -> None:
    name = client.create_sandbox("")
    client.write_file(name, "/home/user/a.txt", b"v1")
    anchor = client.create_anchor(name)
    client.write_file(name, "/home/user/a.txt", b"v2")

    restored = client.restore_anchor(anchor)

    assert restored != name
    assert client.read_file(restored, "/home/user/a.txt") == b"v1"
    assert client.read_file(name, "/home/user/a.txt") == b"v2"  # the source is untouched


def test_anchor_keeps_the_sandbox_env(client: LangSmithClient, stub: StubSandboxClient) -> None:
    name = client.create_sandbox("", env={"MODE": "test"})
    client.create_anchor(name)
    assert stub.calls_to("capture_snapshot")[0]["run_config"] == {"env_vars": {"MODE": "test"}}


def test_delete_anchor_deletes_the_snapshot_and_is_idempotent(
    client: LangSmithClient, stub: StubSandboxClient
) -> None:
    anchor = client.create_anchor(client.create_sandbox(""))
    client.delete_anchor(anchor)
    client.delete_anchor(anchor)
    assert stub.snapshots == {}


def test_fork_leaves_no_snapshot_behind(client: LangSmithClient, stub: StubSandboxClient) -> None:
    name = client.create_sandbox("")
    client.write_file(name, "/home/user/a.txt", b"v1")
    child = client.fork(name)
    assert client.read_file(child, "/home/user/a.txt") == b"v1"
    assert stub.snapshots == {}


def test_snapshot_delete_refused_during_fork_is_retried_on_close(
    client: LangSmithClient, stub: StubSandboxClient
) -> None:
    name = client.create_sandbox("")
    original_capture = stub.capture_snapshot

    def capture_and_refuse(*args: Any, **kwargs: Any) -> Any:
        snapshot = original_capture(*args, **kwargs)
        stub.refuse_snapshot_delete.add(snapshot.id)
        return snapshot

    stub.capture_snapshot = capture_and_refuse  # type: ignore[method-assign]
    client.fork(name)
    assert len(stub.snapshots) == 1

    client.close()
    assert stub.snapshots == {}
    assert not stub.closed  # an injected client belongs to the caller


# ---------------------------------------------------------------------- commands


def test_argv_is_quoted_and_outputs_are_merged(
    client: LangSmithClient, stub: StubSandboxClient
) -> None:
    def echo(sandbox: StubSandbox, command: str) -> ExecutionResult:
        return ExecutionResult(stdout="out", stderr="err", exit_code=3)

    stub.run_handler = echo
    name = client.create_sandbox("")
    result = client.run_command(
        name, ["sh", "-c", "echo 'hi there'"], cwd="/tmp", env={"A": "1"}, timeout=2.5
    )

    assert (result.output, result.exit_code) == ("out\nerr", 3)
    run = stub.sandboxes[name].runs[-1]
    assert shlex.split(run["command"]) == ["sh", "-c", "echo 'hi there'"]
    assert run["timeout"] == 3  # rounded up to whole seconds
    assert run["run_config"] == {"work_dir": "/tmp", "env_vars": {"A": "1"}}


def test_default_and_disabled_timeouts(client: LangSmithClient, stub: StubSandboxClient) -> None:
    stub.run_handler = lambda sandbox, command: ExecutionResult("", "", 0)
    name = client.create_sandbox("")
    client.run_command(name, ["true"])
    client.run_command(name, ["true"], timeout=0)
    assert [r["timeout"] for r in stub.sandboxes[name].runs] == [30 * 60, 0]


def test_timeout_returns_exit_code_124(client: LangSmithClient, stub: StubSandboxClient) -> None:
    stub.run_handler = timing_out
    result = client.run_command(client.create_sandbox(""), ["sleep", "10"], timeout=1)
    assert (result.exit_code, result.status) == (124, "timed_out")


# ---------------------------------------------------------------------- files


def test_missing_file_is_a_missing_path(client: LangSmithClient) -> None:
    name = client.create_sandbox("")
    with pytest.raises(SpaceError) as error:
        client.read_file(name, "/nope.txt")
    assert error.value.is_path_not_found


def test_missing_sandbox_is_not_a_missing_path(client: LangSmithClient) -> None:
    with pytest.raises(SpaceError) as error:
        client.read_file("box-gone", "/a.txt")
    assert error.value.code == "not_found"
    assert not error.value.is_path_not_found


def test_stat_tells_files_directories_and_missing_paths(client: LangSmithClient) -> None:
    name = client.create_sandbox("")
    client.write_file(name, "/home/user/a.txt", b"hello")
    assert client.stat(name, "/home/user/a.txt").file_type == "file"
    assert client.stat(name, "/home/user/a.txt").size_bytes == 5
    assert client.stat(name, "/home/user").file_type == "directory"
    with pytest.raises(SpaceError) as error:
        client.stat(name, "/home/user/missing")
    assert error.value.is_path_not_found


def test_mkdir_creates_parents_and_refuses_files(client: LangSmithClient) -> None:
    name = client.create_sandbox("")
    client.mkdir(name, "/home/user/a/b")
    assert client.stat(name, "/home/user/a").file_type == "directory"
    client.write_file(name, "/home/user/f", b"")
    with pytest.raises(SpaceError):
        client.mkdir(name, "/home/user/f")


def test_write_files_reports_each_file(client: LangSmithClient, stub: StubSandboxClient) -> None:
    name = client.create_sandbox("")
    client.write_file(name, "/home/user/f", b"")
    files = [("/home/user/a/1", b"1"), ("/home/user/f/x", b"2"), ("/home/user/a/2", b"3")]

    results = client.write_files(name, files)

    assert results[0] is None and results[2] is None
    assert isinstance(results[1], SpaceError)
    assert client.read_file(name, "/home/user/a/2") == b"3"
    mkdirs = [r for r in stub.sandboxes[name].runs if "mkdir" in r["command"]]
    assert len(mkdirs) == 2  # one per parent directory: /home/user/a and /home/user/f


def test_backend_upload_and_download_round_trip(client: LangSmithClient) -> None:
    sandbox = SpaceSandbox(client, client.create_sandbox(""))
    [uploaded] = sandbox.upload_files([("/home/user/work/x.txt", b"data")])
    assert uploaded.error is None
    [downloaded] = sandbox.download_files(["/home/user/work/x.txt"])
    assert downloaded.content == b"data"
    [missing] = sandbox.download_files(["/home/user/none.txt"])
    assert missing.error == "file_not_found"
    [directory] = sandbox.download_files(["/home/user/work"])
    assert directory.error == "is_directory"


# ---------------------------------------------------------------------- the core claim on LangSmith


@pytest.mark.consistency
@pytest.mark.parametrize("durability", ["sync", "async"])
def test_rewind_restores_the_sandbox_on_langsmith(client: LangSmithClient, durability: Any) -> None:
    saver = SpaceCheckpointer(InMemorySaver(), client, template="")
    graph = build_edit_graph(saver.backend, saver)
    try:
        observations = run_scenarios(graph, durability=durability)
    finally:
        saver.close()
    assert [o for o in observations if not o.consistent] == []


def test_space_checkpoints_are_refused_with_a_clear_error(client: LangSmithClient) -> None:
    storage = InMemorySaver()
    on_space = SpaceCheckpointer(storage, FakeSpaceClient(), template="code-runtime")
    build_edit_graph(on_space.backend, on_space).invoke({"target": "v1"}, thread("t"))

    on_langsmith = SpaceCheckpointer(storage, client, template="")
    graph = build_edit_graph(on_langsmith.backend, on_langsmith)
    with pytest.raises(ProviderMismatchError, match="saved with space sandboxes"):
        graph.invoke({"target": "v2"}, thread("t"))


def test_fork_and_select_on_langsmith(client: LangSmithClient, stub: StubSandboxClient) -> None:
    def tests_pass_if_fixed(sandbox: StubSandbox, command: str) -> ExecutionResult:
        fixed = sandbox.files.get("/home/user/fix.txt") == b"good"
        return ExecutionResult("", "", 0 if fixed else 1)

    stub.run_handler = tests_pass_if_fixed
    source = SpaceSandbox(client, client.create_sandbox(""))

    def write(content: bytes) -> Any:
        return lambda child: child.upload_files([("/home/user/fix.txt", content)])

    result = fork_and_select(source, [write(b"bad"), write(b"good")], test_command="test")

    assert result.winner is not None and result.winner.index == 1
    assert stub.snapshots == {}  # temporary fork snapshots are gone
    assert set(stub.sandboxes) == {source.id, result.winner.sandbox_id}  # losers deleted
