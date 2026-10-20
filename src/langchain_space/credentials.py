"""Keep credentials out of the sandbox: SPACE injects them at the network layer.

The sandbox never sees a secret. You register an AWS Secrets Manager secret *by reference*,
and SPACE's per-sandbox gateway adds it as an HTTP header to matching outbound requests
(TLS is intercepted for that). Combined with a deny-by-default egress policy, a
prompt-injected agent has no key to leak and nowhere to send it.

```python
rule = CredentialRule(host="api.github.com", header="authorization",
                      secret_arn="arn:aws:secretsmanager:...:secret:gh-token")
policy = build_network_policy([register(client, rule)], allow_domains=["pypi.org"])
saver = SpaceCheckpointer(..., network_policy=policy,
                          backend_kwargs={"content_check": SecretGuard.from_env("GH_TOKEN")})
```

The secret stored in AWS is used verbatim as the header value: store `Bearer <token>` if
the API expects that prefix. Validation below mirrors SPACE's
(`crates/libs/net/common/src/lib.rs`), so mistakes fail before reaching SPACE.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_space.backend import check_env
from langchain_space.client import SpaceClient

HttpMethod = Literal["GET", "POST", "HEAD"]

SUPPORTED_METHODS: frozenset[str] = frozenset({"GET", "POST", "HEAD"})
"""Methods SPACE injects credentials for; requests with other methods go out without them."""

RESERVED_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "http2-settings",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "space-credential",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
"""Header names SPACE refuses to inject (plus anything starting with `x-space-`)."""

PUBLIC_RESOLVERS: tuple[str, ...] = ("8.8.8.8", "1.1.1.1")
"""Domain egress rules only work if the guest resolves through these, so they are allowed."""

_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9a-z]+$")


class InvalidCredentialRule(ValueError):
    """A credential rule SPACE would reject."""


@dataclass(frozen=True)
class CredentialRule:
    """Inject one secret as one header into requests to one host.

    Args:
        host: Exact host name (no wildcards in SPACE).
        header: Header to set; it overwrites any value the guest sends.
        secret_arn: AWS Secrets Manager ARN; the secret must be tagged `space-org-id=<org>`.
        path_prefix: Only inject for paths under this prefix (whole segments).
        port: Destination port.
        methods: HTTP methods to inject for (subset of GET, POST, HEAD).
        rule_id: Stable id for the rule (defaults to one derived from host and header).
    """

    host: str
    header: str
    secret_arn: str
    path_prefix: str | None = None
    port: int = 443
    methods: tuple[HttpMethod, ...] = ("GET", "POST")
    rule_id: str = field(default="")

    def __post_init__(self) -> None:
        header = self.header.lower()
        object.__setattr__(self, "header", header)
        if not self.rule_id:
            slug = re.sub(r"[^a-z0-9]+", "-", f"{self.host}-{header}".lower()).strip("-")
            object.__setattr__(self, "rule_id", slug)
        if not self.host or "*" in self.host or "/" in self.host:
            raise InvalidCredentialRule(f"host must be an exact host name, got {self.host!r}")
        if not _HEADER_NAME.match(header):
            raise InvalidCredentialRule(f"invalid header name {self.header!r}")
        if header in RESERVED_HEADERS or header.startswith("x-space-"):
            raise InvalidCredentialRule(f"SPACE refuses to inject reserved header {header!r}")
        if not self.methods or not set(self.methods) <= SUPPORTED_METHODS:
            raise InvalidCredentialRule(
                f"methods must be a non-empty subset of {sorted(SUPPORTED_METHODS)}"
            )
        if not self.secret_arn.startswith("arn:"):
            raise InvalidCredentialRule("secret_arn must be an AWS ARN")
        if self.path_prefix is not None and not self.path_prefix.startswith("/"):
            raise InvalidCredentialRule("path_prefix must start with '/'")
        if not 0 < self.port < 65536:
            raise InvalidCredentialRule(f"invalid port {self.port}")


@dataclass(frozen=True)
class RegisteredCredential:
    """A rule whose secret is registered with SPACE."""

    rule: CredentialRule
    credential_reference_id: str


def register(client: SpaceClient, rule: CredentialRule) -> RegisteredCredential:
    """Register the rule's secret with SPACE by reference (the value never leaves AWS)."""
    return RegisteredCredential(rule, client.register_credential_reference(rule.secret_arn))


def build_network_policy(
    credentials: Sequence[RegisteredCredential] = (),
    *,
    allow_domains: Iterable[str] = (),
    allow_ips: Iterable[str] = (),
    deny_by_default: bool = True,
) -> dict[str, Any]:
    """Build a SPACE network policy: egress allowlist plus credential injection rules.

    Hosts that receive credentials are allowed automatically. With `deny_by_default`, any
    other destination is blocked, which is what stops exfiltration.
    """
    allow: list[dict[str, str]] = [{"ip": ip} for ip in (*PUBLIC_RESOLVERS, *allow_ips)]
    domains = dict.fromkeys([*allow_domains, *(c.rule.host for c in credentials)])
    allow += [{"domain": domain} for domain in domains]
    policy: dict[str, Any] = {
        "version": "v1",
        "egress": {"default_action": "deny" if deny_by_default else "allow", "allow": allow},
    }
    if credentials:
        policy["tls_interception"] = "on"  # required by SPACE for credential rules
        policy["network_credential_rules"] = [_rule_json(c) for c in credentials]
    return policy


def _rule_json(credential: RegisteredCredential) -> dict[str, Any]:
    rule = credential.rule
    destination: dict[str, Any] = {"exact_host": rule.host, "port": rule.port}
    if rule.path_prefix is not None:
        destination["path_prefix"] = rule.path_prefix
    return {
        "rule_id": rule.rule_id,
        "destination": destination,
        "credential_header_bindings": [
            {
                "header_name": rule.header,
                "credential_reference_id": credential.credential_reference_id,
            }
        ],
        "supported_http_methods": sorted(set(rule.methods)),
    }


class SecretLeakError(RuntimeError):
    """A known secret value was about to be sent into a sandbox."""


class SecretGuard:
    """Refuses to send known secret values into a sandbox (commands, files, env).

    A last line of defence for the host process: secrets the *host* holds (e.g. the model
    API key in its environment) must never reach the VM, where a prompt-injected agent
    could read them. Pass it as `SpaceSandbox(content_check=...)`.

    Args:
        secrets: Secret values. Values shorter than `min_length` are ignored to avoid
            false positives on trivial strings.
    """

    def __init__(self, secrets: Iterable[str], *, min_length: int = 8) -> None:
        self._secrets = [s.encode() for s in secrets if s and len(s) >= min_length]

    @classmethod
    def from_env(cls, *names: str, min_length: int = 8) -> SecretGuard:
        """Guard the values of these host environment variables (unset ones are skipped)."""
        return cls((os.environ.get(n, "") for n in names), min_length=min_length)

    def __call__(self, content: bytes, where: str) -> None:
        """Raise `SecretLeakError` if `content` contains a guarded secret."""
        for secret in self._secrets:
            if secret in content:
                raise SecretLeakError(f"refusing to send a secret into the sandbox ({where})")

    def check_mapping(self, values: Mapping[str, str], where: str) -> None:
        """Check every key and value of a mapping (e.g. sandbox env)."""
        check_env(self, values, where)
