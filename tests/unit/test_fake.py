"""FakeSpaceClient behaves like SPACE where the rest of the package depends on it."""

from __future__ import annotations

import pytest

from langchain_space.client import SpaceClient, SpaceError
from langchain_space.testing import FakeSpaceClient


def test_fake_satisfies_protocol() -> None:
    assert isinstance(FakeSpaceClient(), SpaceClient)


def test_write_requires_parent_directory() -> None:
    fake = FakeSpaceClient()
    sbx = fake.create_sandbox("t")
    with pytest.raises(SpaceError) as info:
        fake.write_file(sbx, "/a/b.txt", b"x")
    assert info.value.is_path_not_found

    fake.mkdir(sbx, "/a")
    fake.write_file(sbx, "/a/b.txt", b"x")
    assert fake.read_file(sbx, "/a/b.txt") == b"x"


def test_relative_paths_resolve_under_home() -> None:
    fake = FakeSpaceClient()
    sbx = fake.create_sandbox("t")
    fake.write_file(sbx, "notes.txt", b"hi")
    assert fake.stat(sbx, "/home/user/notes.txt").size_bytes == 2


def test_fork_is_an_isolated_copy() -> None:
    fake = FakeSpaceClient()
    parent = fake.create_sandbox("t")
    fake.write_file(parent, "/tmp/f", b"v1")

    child = fake.fork(parent)
    fake.write_file(child, "/tmp/f", b"child")
    fake.write_file(parent, "/tmp/f", b"parent")

    assert fake.read_file(child, "/tmp/f") == b"child"
    assert fake.read_file(parent, "/tmp/f") == b"parent"
    assert fake.sandbox(child).forked_from == parent


def test_suspended_sandbox_can_be_forked_but_not_used() -> None:
    fake = FakeSpaceClient()
    sbx = fake.create_sandbox("t")
    fake.write_file(sbx, "/tmp/f", b"v1")
    fake.suspend(sbx)

    with pytest.raises(SpaceError):
        fake.read_file(sbx, "/tmp/f")
    child = fake.fork(sbx)
    assert fake.read_file(child, "/tmp/f") == b"v1"


def test_deleted_sandbox_is_not_found() -> None:
    fake = FakeSpaceClient()
    sbx = fake.create_sandbox("t")
    fake.delete_sandbox(sbx)
    fake.delete_sandbox(sbx)  # idempotent, like SPACE's 404-means-gone
    with pytest.raises(SpaceError) as info:
        fake.stat(sbx, "/")
    assert info.value.code == "not_found"
    assert fake.live_sandboxes == []


def test_exec_handler_sees_the_sandbox() -> None:
    def handler(sandbox, argv):  # type: ignore[no-untyped-def]
        sandbox.write_text("/tmp/ran", argv[-1])
        return "ok", 0

    fake = FakeSpaceClient(handler)
    sbx = fake.create_sandbox("t")
    result = fake.run_command(sbx, ["sh", "-c", "echo"])
    assert (result.output, result.exit_code) == ("ok", 0)
    assert fake.sandbox(sbx).read_text("/tmp/ran") == "echo"
