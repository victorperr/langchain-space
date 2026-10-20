"""HttpSpaceClient against mocked HTTP shaped like SPACE's documented responses."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx

from langchain_space.backend import SpaceSandbox
from langchain_space.client import HttpSpaceClient, SpaceError
from langchain_space.credentials import CredentialRule, RegisteredCredential, build_network_policy

API = "http://space.test"
DATA_PLANE = "http://dp.space.test"
ECW = "http://ecw.space.test"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("langchain_space.client.time.sleep", lambda _s: None)


@pytest.fixture
def client() -> Iterator[HttpSpaceClient]:
    with HttpSpaceClient(
        API, DATA_PLANE, "tok", ecw_api_url=ECW, ecw_api_key="ecw-key", chunk_size=4
    ) as c:
        yield c


@pytest.fixture
def mock() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


def sse(*frames: dict[str, Any]) -> str:
    return "".join(f"event: {f['type']}\ndata: {json.dumps(f)}\n\n" for f in frames)


def out(kind: str, text: str, seq: int) -> dict[str, Any]:
    return {"type": kind, "seq": seq, "data_base64": base64.b64encode(text.encode()).decode()}


def mock_running(mock: respx.MockRouter, sandbox_id: str) -> None:
    mock.get(f"{API}/v1/sandboxes/{sandbox_id}").respond(
        json={"id": sandbox_id, "state": "running"}
    )
    mock.get(f"{API}/v1/sandboxes/{sandbox_id}/processes", params={"limit": "1"}).respond(
        json={"data": []}
    )


def mock_command(mock: respx.MockRouter, body: str, events_url: str = "/v1/sessions/proc_1/events"):  # type: ignore[no-untyped-def]
    created = mock.post(f"{API}/v1/sandboxes/sbx_1/commands").respond(
        201,
        json={
            "session_id": "proc_1",
            "process_id": "proc_1",
            "attach": {"events_url": events_url, "token": "dpt_attach"},
        },
    )
    events = mock.get(f"{DATA_PLANE}/v1/sessions/proc_1/events").respond(
        200, text=body, headers={"content-type": "text/event-stream"}
    )
    return created, events


class TestCreateDelete:
    def test_create_waits_for_operation_and_running(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        create = mock.post(f"{API}/v1/sandboxes").respond(
            202, json={"sandbox_id": "sbx_1", "operation_id": "op_1"}
        )
        op = mock.get(f"{API}/v1/operations/op_1")
        op.side_effect = [
            httpx.Response(200, json={"state": "running"}),
            httpx.Response(200, json={"state": "succeeded"}),
        ]
        mock_running(mock, "sbx_1")

        assert client.create_sandbox("code-runtime", env={"A": "1"}) == "sbx_1"

        request = create.calls.last.request
        assert request.headers["Authorization"] == "Bearer tok"
        assert request.headers["Idempotency-Key"]
        assert json.loads(request.content) == {"template": "code-runtime", "env": {"A": "1"}}
        assert op.call_count == 2

    def test_create_deletes_sandbox_when_operation_fails(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        mock.post(f"{API}/v1/sandboxes").respond(
            202, json={"sandbox_id": "sbx_1", "operation_id": "op_1"}
        )
        mock.get(f"{API}/v1/operations/op_1").respond(
            json={
                "state": "failed",
                "terminal_error_code": "capacity_unavailable",
                "terminal_error": {"message": "no capacity"},
            }
        )
        delete = mock.delete(f"{API}/v1/sandboxes/sbx_1").respond(404)

        with pytest.raises(SpaceError) as info:
            client.create_sandbox("seed")

        assert info.value.code == "capacity_unavailable"
        assert delete.called

    def test_delete_of_missing_sandbox_is_ok(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        mock.delete(f"{API}/v1/sandboxes/sbx_gone").respond(
            404, json={"code": "not_found", "message": "no", "retryable": False, "details": []}
        )
        client.delete_sandbox("sbx_gone")


class TestRunCommand:
    def test_collects_interleaved_output_and_exit_code(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        body = sse(
            {"type": "attached", "session_id": "proc_1"},
            {"type": "process_started", "process_id": "proc_1", "pid": 7},
            out("stdout", "hello ", 1),
            out("stderr", "warn ", 2),
            out("stdout", "world", 3),
            {"type": "process_exited", "process_id": "proc_1", "exit_code": 3, "status": "exited"},
            {"type": "closed", "reason": "exited"},
        )
        created, events = mock_command(mock, body)

        result = client.run_command("sbx_1", ["sh", "-c", "x"], timeout=2.5, cwd="/tmp")

        assert result.output == "hello warn world"
        assert result.exit_code == 3
        assert json.loads(created.calls.last.request.content) == {
            "command": ["sh", "-c", "x"],
            "cwd": "/tmp",
            "timeout": "2.5s",
        }
        # The attach token, not the API token, authenticates the stream.
        assert events.calls.last.request.headers["Authorization"] == "Bearer dpt_attach"

    @pytest.mark.parametrize(("timeout", "expected"), [(None, "1800s"), (0, None), (30, "30s")])
    def test_timeout_encoding(
        self,
        client: HttpSpaceClient,
        mock: respx.MockRouter,
        timeout: float | None,
        expected: str | None,
    ) -> None:
        created, _ = mock_command(mock, sse({"type": "process_exited", "exit_code": 0}))
        client.run_command("sbx_1", ["true"], timeout=timeout)
        assert json.loads(created.calls.last.request.content).get("timeout") == expected

    def test_refuses_events_url_on_foreign_origin(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        mock_command(mock, "", events_url="http://evil.test/steal")
        with pytest.raises(SpaceError) as info:
            client.run_command("sbx_1", ["true"])
        assert info.value.code == "unsafe_url"

    def test_error_frame_raises(self, client: HttpSpaceClient, mock: respx.MockRouter) -> None:
        mock_command(mock, sse({"type": "error", "code": "unavailable", "message": "node lost"}))
        with pytest.raises(SpaceError, match="node lost"):
            client.run_command("sbx_1", ["true"])

    def test_output_beyond_the_cap_is_drained_but_not_kept(self, mock: respx.MockRouter) -> None:
        body = sse(
            out("stdout", "hello ", 1),
            out("stdout", "world", 2),
            out("stdout", "!!!", 3),
            {"type": "process_exited", "exit_code": 0, "status": "exited"},
        )
        mock_command(mock, body)
        with HttpSpaceClient(API, DATA_PLANE, "tok", max_output_bytes=8) as capped:
            result = capped.run_command("sbx_1", ["true"])
        assert result.output == "hello wo"
        assert result.truncated
        assert result.exit_code == 0

    def test_output_within_the_cap_is_not_truncated(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        mock_command(mock, sse(out("stdout", "ok", 1), {"type": "process_exited", "exit_code": 0}))
        assert not client.run_command("sbx_1", ["true"]).truncated

    def test_stream_closing_without_exit_is_an_error(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        mock_command(mock, sse(out("stdout", "partial", 1), {"type": "closed", "reason": "x"}))
        with pytest.raises(SpaceError) as info:
            client.run_command("sbx_1", ["true"])
        assert info.value.code == "stream_closed"


class TestErrorsAndRetries:
    def test_envelope_maps_to_space_error(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        mock.post(f"{API}/v1/sandboxes/sbx_1/files:stat").respond(
            404,
            json={
                "code": "not_found",
                "message": "missing",
                "request_id": "req_1",
                "retryable": False,
                "details": [{"key": "resource", "value": "path"}],
            },
        )
        with pytest.raises(SpaceError) as info:
            client.stat("sbx_1", "/nope")
        assert info.value.is_path_not_found
        assert info.value.request_id == "req_1"

    def test_retryable_error_is_retried_with_same_idempotency_key(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        route = mock.post(f"{API}/v1/sandboxes/sbx_1/files:mkdir")
        route.side_effect = [
            httpx.Response(
                503,
                headers={"Retry-After": "0"},
                json={"code": "service_unavailable", "message": "busy", "retryable": True},
            ),
            httpx.Response(200, json={"ok": True}),
        ]
        client.mkdir("sbx_1", "/work")
        keys = {call.request.headers["Idempotency-Key"] for call in route.calls}
        assert route.call_count == 2
        assert len(keys) == 1

    def test_non_retryable_error_is_not_retried(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        route = mock.post(f"{API}/v1/sandboxes/sbx_1/files:mkdir").respond(
            400, json={"code": "invalid_request", "message": "bad path", "retryable": False}
        )
        with pytest.raises(SpaceError) as info:
            client.mkdir("sbx_1", "bad\x00")
        assert info.value.code == "invalid_request"
        assert route.call_count == 1


class TestFiles:
    def test_write_file_chunks_into_write_then_appends(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        route = mock.post(f"{API}/v1/sandboxes/sbx_1/files:write").respond(
            json={"bytes_written": 4}
        )
        client.write_file("sbx_1", "/f", b"abcdefghij")  # chunk_size=4 -> 3 requests

        bodies = [json.loads(call.request.content) for call in route.calls]
        assert [base64.b64decode(b["data_base64"]) for b in bodies] == [b"abcd", b"efgh", b"ij"]
        assert [b["append"] for b in bodies] == [False, True, True]

    def test_write_empty_file_sends_one_request(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        route = mock.post(f"{API}/v1/sandboxes/sbx_1/files:write").respond(json={})
        client.write_file("sbx_1", "/empty", b"")
        assert route.call_count == 1

    def test_read_file_reads_in_chunks_up_to_size(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        mock.post(f"{API}/v1/sandboxes/sbx_1/files:stat").respond(
            json={"entry": {"path": "/f", "file_type": "file", "size_bytes": 6}}
        )
        read = mock.post(f"{API}/v1/sandboxes/sbx_1/files:read")
        read.side_effect = [
            httpx.Response(200, content=b"abcd"),
            httpx.Response(200, content=b"ef"),
        ]

        assert client.read_file("sbx_1", "/f") == b"abcdef"
        assert [json.loads(c.request.content)["offset"] for c in read.calls] == [0, 4]

    def test_read_file_with_known_size_skips_stat(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        stat = mock.post(f"{API}/v1/sandboxes/sbx_1/files:stat")
        mock.post(f"{API}/v1/sandboxes/sbx_1/files:read").respond(200, content=b"ab")
        assert client.read_file("sbx_1", "/f", size=2) == b"ab"
        assert not stat.called

    def test_download_costs_one_stat_and_one_read(self, mock: respx.MockRouter) -> None:
        stat = mock.post(f"{API}/v1/sandboxes/sbx_1/files:stat").respond(
            json={"entry": {"path": "/f", "file_type": "file", "size_bytes": 2}}
        )
        read = mock.post(f"{API}/v1/sandboxes/sbx_1/files:read").respond(200, content=b"ab")
        with HttpSpaceClient(API, DATA_PLANE, "tok") as http:
            [response] = SpaceSandbox(http, "sbx_1").download_files(["/f"])
        assert response.content == b"ab"
        assert (stat.call_count, read.call_count) == (1, 1)


class TestPolling:
    def test_polls_fast_first_then_backs_off_to_poll_interval(
        self, mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr("langchain_space.client.time.sleep", sleeps.append)
        op = mock.get(f"{API}/v1/operations/op_1")
        op.side_effect = [httpx.Response(200, json={"state": "running"})] * 6 + [
            httpx.Response(200, json={"state": "succeeded"})
        ]
        with HttpSpaceClient(API, DATA_PLANE, "tok", poll_interval=0.5) as http:
            http._wait_operation("op_1")
        assert sleeps == [0.05, 0.1, 0.2, 0.4, 0.5, 0.5]


class TestFork:
    def test_fork_uses_ecw_route_and_waits_for_child(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        fork = mock.post(f"{ECW}/sandboxes/sbx_1/fork").respond(
            201, json=[{"sandboxID": "sbx_child", "templateID": "t", "clientID": "c"}]
        )
        mock_running(mock, "sbx_child")

        assert client.fork("sbx_1", lifetime_seconds=600) == "sbx_child"

        request = fork.calls.last.request
        assert request.headers["X-API-Key"] == "ecw-key"
        assert "Authorization" not in request.headers
        assert request.headers["Idempotency-Key"]
        assert json.loads(request.content) == {"timeout": 600, "count": 1}

    def test_fork_target_error_raises(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        mock.post(f"{ECW}/sandboxes/sbx_1/fork").respond(
            201, json=[{"error": {"code": 503, "message": "capture failed"}}]
        )
        with pytest.raises(SpaceError, match="capture failed"):
            client.fork("sbx_1")

    def test_fork_requires_ecw_configuration(self) -> None:
        with HttpSpaceClient(API, DATA_PLANE, "tok") as plain, pytest.raises(SpaceError) as info:
            plain.fork("sbx_1")
        assert info.value.code == "not_configured"


class TestCredentials:
    ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:gh-AbCdEf"

    def test_register_sends_only_the_arn(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        route = mock.post(f"{API}/v1/credential-references").respond(
            201, json={"credential_reference_id": "cref_1"}
        )
        assert client.register_credential_reference(self.ARN) == "cref_1"

        body = json.loads(route.calls.last.request.content)
        assert body["provider"] == "aws_secrets_manager"
        assert body["secret_arn"] == self.ARN
        assert body["request_id"]
        assert set(body) == {"provider", "secret_arn", "request_id"}
        assert route.calls.last.request.headers["Idempotency-Key"]

    def test_set_network_policy_patches_and_waits(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        policy = build_network_policy(
            [
                RegisteredCredential(
                    CredentialRule("api.github.com", "authorization", self.ARN), "cref_1"
                )
            ]
        )
        patch = mock.patch(f"{API}/v1/sandboxes/sbx_1").respond(202, json={"operation_id": "op_9"})
        op = mock.get(f"{API}/v1/operations/op_9").respond(json={"state": "succeeded"})

        client.set_network_policy("sbx_1", policy)

        assert json.loads(patch.calls.last.request.content) == {
            "network_policy": {"mode": "set", "policy": policy}
        }
        assert op.called

    def test_policy_rejected_by_space_surfaces_its_error(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        mock.patch(f"{API}/v1/sandboxes/sbx_1").respond(
            400,
            json={
                "code": "invalid_request",
                "message": "network credentials require tls interception",
                "retryable": False,
            },
        )
        with pytest.raises(SpaceError, match="tls interception"):
            client.set_network_policy("sbx_1", {"version": "v1"})

    def test_create_passes_network_policy(
        self, client: HttpSpaceClient, mock: respx.MockRouter
    ) -> None:
        create = mock.post(f"{API}/v1/sandboxes").respond(
            202, json={"sandbox_id": "sbx_1", "operation_id": "op_1"}
        )
        mock.get(f"{API}/v1/operations/op_1").respond(json={"state": "succeeded"})
        mock_running(mock, "sbx_1")
        policy = build_network_policy(allow_domains=["pypi.org"])

        client.create_sandbox("code-runtime", network_policy=policy)

        assert json.loads(create.calls.last.request.content)["network_policy"] == policy
