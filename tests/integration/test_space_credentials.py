"""Network-layer credential injection on real SPACE (needs AWS; see CLAUDE.md).

Required environment, in addition to the SPACE_* variables:

- SPACE_TEST_SECRET_ARN: AWS Secrets Manager secret tagged `space-org-id=<org>`.
- SPACE_TEST_SECRET_SHA256: hex SHA-256 of that secret's value.
- SPACE_TEST_ECHO_URL: HTTPS endpoint that responds with the hex SHA-256 of the request's
  `Authorization` header (the same contract as SPACE's own credential smoke test).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from urllib.parse import urlsplit

import pytest

from langchain_space.backend import SpaceSandbox
from langchain_space.client import HttpSpaceClient
from langchain_space.credentials import CredentialRule, build_network_policy, register

pytestmark = pytest.mark.integration

ENV = ("SPACE_TEST_SECRET_ARN", "SPACE_TEST_SECRET_SHA256", "SPACE_TEST_ECHO_URL")


@pytest.fixture
def injected_sandbox(
    space_client: HttpSpaceClient, space_template: str
) -> Iterator[tuple[SpaceSandbox, str]]:
    missing = [name for name in ENV if not os.environ.get(name)]
    if missing:
        pytest.skip(f"credential injection not configured (missing {', '.join(missing)})")
    echo_url = os.environ["SPACE_TEST_ECHO_URL"]
    host = urlsplit(echo_url).hostname
    assert host is not None
    rule = CredentialRule(host, "authorization", os.environ["SPACE_TEST_SECRET_ARN"])
    policy = build_network_policy([register(space_client, rule)])
    sandbox = SpaceSandbox.create(space_client, space_template, network_policy=policy)
    try:
        yield sandbox, echo_url
    finally:
        sandbox.terminate()


def test_origin_receives_secret_that_the_guest_never_sees(
    injected_sandbox: tuple[SpaceSandbox, str],
) -> None:
    sandbox, echo_url = injected_sandbox
    expected = os.environ["SPACE_TEST_SECRET_SHA256"]

    # The guest sends a placeholder; the gateway overwrites it with the real secret.
    seen_by_origin = sandbox.execute(
        f"curl -sS --fail -H 'Authorization: placeholder' {echo_url}", timeout=60
    )
    assert seen_by_origin.exit_code == 0, seen_by_origin.output
    assert seen_by_origin.output.strip() == expected

    # Nothing inside the VM holds the secret: not env, not files under home or /etc.
    hunt = sandbox.execute(
        "{ env; grep -rhs . /home /etc /tmp 2>/dev/null; } | "
        "while IFS= read -r line; do printf '%s' \"$line\" | sha256sum; done | "
        f"grep -c {expected} || true",
        timeout=120,
    )
    assert hunt.output.strip() == "0"


def test_exfiltration_to_other_hosts_is_blocked(
    injected_sandbox: tuple[SpaceSandbox, str],
) -> None:
    sandbox, _ = injected_sandbox
    result = sandbox.execute("curl -sS --max-time 10 https://example.com", timeout=30)
    assert result.exit_code != 0
