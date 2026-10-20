"""SpaceSandbox on the fake client: the three primitives Deep Agents builds on."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse

from langchain_space.backend import SpaceSandbox
from langchain_space.testing import FakeSandbox, FakeSpaceClient


def echo_handler(sandbox: FakeSandbox, argv: Sequence[str]) -> tuple[str, int]:
    command = argv[2]
    if command.startswith("exit "):
        return "", int(command.split()[1])
    return command, 0


@pytest.fixture
def fake() -> FakeSpaceClient:
    return FakeSpaceClient(echo_handler)


@pytest.fixture
def sandbox(fake: FakeSpaceClient) -> SpaceSandbox:
    return SpaceSandbox.create(fake, "code-runtime")


class TestExecute:
    def test_runs_through_sh_and_returns_exit_code(
        self, fake: FakeSpaceClient, sandbox: SpaceSandbox
    ) -> None:
        result = sandbox.execute("exit 3", timeout=12)
        assert result.exit_code == 3
        call = fake.calls_to("run_command")[-1]
        assert call.args["argv"] == ["sh", "-c", "exit 3"]
        assert call.args["timeout"] == 12

    def test_output_is_truncated_with_flag(self, fake: FakeSpaceClient) -> None:
        sandbox = SpaceSandbox.create(fake, "t", max_output_chars=10)
        result = sandbox.execute("0123456789abcdef")
        assert result.truncated
        assert result.output.startswith("0123456789")

    def test_every_execute_bumps_generation(self, sandbox: SpaceSandbox) -> None:
        before = sandbox.generation
        sandbox.execute("ls")
        assert sandbox.generation == before + 1

    def test_read_only_helpers_do_not_bump_generation(
        self, fake: FakeSpaceClient, sandbox: SpaceSandbox
    ) -> None:
        before = sandbox.generation
        sandbox.ls("/home/user")
        sandbox.read("/home/user/a.txt")
        sandbox.grep("needle", "/home/user")
        sandbox.glob("*.py", "/home/user")
        assert len(fake.calls_to("run_command")) == 4  # they did run commands
        assert sandbox.generation == before
        sandbox.execute("touch x")  # the flag does not leak past the helper
        assert sandbox.generation == before + 1

    async def test_async_read_only_helpers_do_not_bump_generation(
        self, fake: FakeSpaceClient, sandbox: SpaceSandbox
    ) -> None:
        before = sandbox.generation
        await sandbox.als("/home/user")
        await sandbox.aread("/home/user/a.txt")
        await sandbox.agrep("needle", "/home/user")
        await sandbox.aglob("*.py", "/home/user")
        assert len(fake.calls_to("run_command")) == 4
        assert sandbox.generation == before
        await sandbox.aexecute("touch x")
        assert sandbox.generation == before + 1

    def test_id_is_the_sandbox_id(self, fake: FakeSpaceClient, sandbox: SpaceSandbox) -> None:
        assert sandbox.id in fake.sandboxes


class TestUpload:
    def test_creates_parent_dirs_and_preserves_order(
        self, fake: FakeSpaceClient, sandbox: SpaceSandbox
    ) -> None:
        responses = sandbox.upload_files([("/w/a/one.txt", b"1"), ("/w/b/two.txt", b"2")])
        assert responses == [
            FileUploadResponse("/w/a/one.txt"),
            FileUploadResponse("/w/b/two.txt"),
        ]
        assert fake.read_file(sandbox.id, "/w/b/two.txt") == b"2"

    def test_relative_path_is_invalid_and_not_sent(
        self, fake: FakeSpaceClient, sandbox: SpaceSandbox
    ) -> None:
        responses = sandbox.upload_files([("rel.txt", b"x")])
        assert responses == [FileUploadResponse("rel.txt", error="invalid_path")]
        assert fake.calls_to("write_file") == []

    def test_partial_success(self, fake: FakeSpaceClient, sandbox: SpaceSandbox) -> None:
        sandbox.upload_files([("/w/file", b"x")])
        responses = sandbox.upload_files([("/w/file/child", b"y"), ("/w/ok", b"z")])
        assert responses[0].error is not None
        assert responses[1] == FileUploadResponse("/w/ok")

    def test_one_mkdir_per_parent_directory(
        self, fake: FakeSpaceClient, sandbox: SpaceSandbox
    ) -> None:
        sandbox.upload_files([("/w/a", b"1"), ("/w/b", b"2"), ("/w/sub/c", b"3")])
        assert [c.args["path"] for c in fake.calls_to("mkdir")] == ["/w", "/w/sub"]

    def test_upload_bumps_generation(self, sandbox: SpaceSandbox) -> None:
        before = sandbox.generation
        sandbox.upload_files([("/w/x", b"1")])
        assert sandbox.generation == before + 1


class TestDownload:
    def test_roundtrip_and_order(self, sandbox: SpaceSandbox) -> None:
        sandbox.upload_files([("/w/a", b"\x00\xff"), ("/w/b", b"b")])
        assert sandbox.download_files(["/w/b", "/w/a"]) == [
            FileDownloadResponse("/w/b", b"b"),
            FileDownloadResponse("/w/a", b"\x00\xff"),
        ]

    @pytest.mark.parametrize(
        ("path", "error"),
        [
            ("relative/path.txt", "invalid_path"),
            ("/w/missing", "file_not_found"),
            ("/w", "is_directory"),
            ("/w/secret", "permission_denied"),
        ],
    )
    def test_error_codes(
        self, fake: FakeSpaceClient, sandbox: SpaceSandbox, path: str, error: str
    ) -> None:
        sandbox.upload_files([("/w/secret", b"s")])
        fake.sandbox(sandbox.id).unreadable.add("/w/secret")
        assert sandbox.download_files([path]) == [FileDownloadResponse(path, None, error)]

    def test_download_does_not_bump_generation(self, sandbox: SpaceSandbox) -> None:
        sandbox.upload_files([("/w/a", b"a")])
        before = sandbox.generation
        sandbox.download_files(["/w/a"])
        assert sandbox.generation == before


def test_terminate_deletes_owned_sandbox(fake: FakeSpaceClient) -> None:
    sandbox = SpaceSandbox.create(fake, "t")
    sandbox.terminate()
    assert fake.live_sandboxes == []
