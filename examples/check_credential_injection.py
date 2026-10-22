"""Check credential injection end to end and say which part of the setup is broken.

    uv run python examples/check_credential_injection.py \\
        --secret-arn arn:aws:secretsmanager:eu-west-1:123456789012:secret:lcs-check-AbCdEf \\
        --expect "Bearer lcs-check-1234"

Use a THROWAWAY secret: the echo service sends the received header back, so the sandbox sees
the value in the response. See docs/credential-injection.md for how to create it.

The sandbox sends `Authorization: lcs-placeholder` to an HTTPS echo service. If SPACE's gateway
injects the secret, the service reports the secret's value instead of the placeholder. Each
way this can go wrong points to a different part of the setup; the script names it. It reads
SPACE_API_URL, SPACE_DATA_PLANE_URL and SPACE_API_TOKEN, and always deletes the sandbox and
retires the credential reference it created.
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib.parse import urlsplit

from langchain_space.backend import SpaceSandbox
from langchain_space.client import HttpSpaceClient
from langchain_space.credentials import CredentialRule, build_network_policy, register, retire

PLACEHOLDER = "lcs-placeholder"
GUIDE = "docs/credential-injection.md"

# curl exit codes worth telling apart (curl(1), EXIT CODES).
DNS_OR_CONNECT = {6: "could not resolve the host", 7: "could not connect", 28: "timed out"}
TLS = {35: "TLS handshake failed", 56: "connection reset", 60: "certificate not trusted"}


def received_header(body: str) -> str | None:
    """The `Authorization` header an echo service (postman-echo, httpbin) says it received."""
    try:
        headers = json.loads(body).get("headers", {})
    except (ValueError, AttributeError):
        return None
    for name, value in headers.items():
        if name.lower() == "authorization":
            return str(value)
    return None


def describe(value: str) -> str:
    """Enough to compare with the expected value, without printing any of it."""
    prefix = "starts" if value.startswith("Bearer ") else "does not start"
    return f"{len(value)} characters, {prefix} with 'Bearer '"


def diagnose(exit_code: int | None, output: str, expected: str) -> tuple[bool, str]:
    """Return (ok, explanation) for one probe request."""
    if exit_code is None:
        return False, f"the probe command did not report an exit code: {output.strip()[:300]}"
    if exit_code in DNS_OR_CONNECT:
        return False, (
            f"the request never reached the echo service ({DNS_OR_CONNECT[exit_code]}). The egress "
            "policy blocks it or DNS fails: the policy must allow the echo host and the public "
            "resolvers (build_network_policy does both)."
        )
    if exit_code in TLS:
        return False, (
            f"TLS failed inside the sandbox ({TLS[exit_code]}). SPACE intercepts TLS to inject "
            "headers, so the template must trust SPACE's interception root CA, and the "
            "credential manager must be able to sign under your Private CA. Check "
            "SPACE_CREDMGR_INTERMEDIATE_CA_ARN and that the seed template (and every template "
            f"built from it) was rebuilt with the root CA. See '{GUIDE}', step 2."
        )
    if exit_code != 0:
        return False, f"curl failed with exit code {exit_code}: {output.strip()[:300]}"
    seen = received_header(output)
    if seen is None:
        return False, f"the echo service did not return JSON headers: {output.strip()[:300]}"
    if seen == PLACEHOLDER:
        return False, (
            "the request went out WITHOUT the secret: TLS interception works, but SPACE did not "
            "inject. Most likely, in order: SPACE_SBXMGR_NETWORK_CREDENTIALS_ENABLED is not true "
            "on sandbox-manager; SPACE_CREDMGR_SECRETS_MANAGER_ROLE_ARN is unset on the "
            "credential-manager authority; the secret is not tagged space-org-id=<your "
            "organization ID>; the reader role's trust or permissions policy differs from the "
            f"guide's; the secret is in another region or account. See '{GUIDE}', steps 3-5."
        )
    if seen == expected:
        return True, "the echo service received the secret; the sandbox only sent a placeholder."
    return False, (
        f"SPACE injected a value, but not the expected one (got {describe(seen)}; expected "
        f"{describe(expected)}). The secret is used verbatim as the header value: store "
        "'Bearer <token>' if the API expects the prefix."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--secret-arn", required=True, help="full ARN of a throwaway secret")
    parser.add_argument("--expect", required=True, help="the throwaway secret's exact value")
    parser.add_argument("--echo-url", default="https://postman-echo.com/headers")
    parser.add_argument("--template", default="code-runtime")
    args = parser.parse_args()

    host = urlsplit(args.echo_url).hostname
    if urlsplit(args.echo_url).scheme != "https" or host is None:
        parser.error("--echo-url must be an https:// URL")
    rule = CredentialRule(host, "authorization", args.secret_arn, methods=("GET",))

    with HttpSpaceClient.from_env() as client:
        credential = register(client, rule)
        print(f"Registered the secret by reference: {credential.credential_reference_id}")
        sandbox = None
        try:
            policy = build_network_policy([credential])
            sandbox = SpaceSandbox.create(client, args.template, network_policy=policy)
            print(f"Created sandbox {sandbox.id} with the injection rule for {host}")
            probe = sandbox.execute(
                f"curl -sS --max-time 20 -H 'Authorization: {PLACEHOLDER}' {args.echo_url}",
                timeout=60,
            )
            ok, why = diagnose(probe.exit_code, probe.output, args.expect)
            if ok:
                env = sandbox.execute("env", timeout=30).output
                if args.expect in env:
                    ok, why = False, "the secret is visible in the sandbox's environment."
            print(f"\n{'OK' if ok else 'FAILED'}: {why}")
            return 0 if ok else 1
        finally:
            if sandbox is not None:
                sandbox.terminate()
            retire(client, credential)
            print("Cleaned up: sandbox deleted, credential reference retired.")


if __name__ == "__main__":
    sys.exit(main())
