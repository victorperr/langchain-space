# CLAUDE.md — langchain-space

## What this project is
`langchain-space` is a LangChain Deep Agents sandbox backend for SPACE (Perplexity's open-source
Firecracker microVM platform, Apache 2.0). The SPACE source is vendored read-only in [SPACE/](SPACE/);
it is reference material, not ours to edit.

1. **Adapter**: `SpaceSandbox(BaseSandbox)` implements `execute`, `upload_files`, `download_files`.
2. **Snapshot-consistent time travel (headline)**: every LangGraph checkpoint is bound to an immutable
   SPACE snapshot (an "anchor"), so rewinding or forking a thread restores the sandbox filesystem too.
3. **Fork-and-select**: at a risky step, fork the sandbox N ways, run branches concurrently, keep the
   branch whose tests pass.
4. **Credentials stay out of the VM**: tool auth uses SPACE network-layer credential injection.

Core claim to prove: *with any other backend, rewinding a LangGraph thread leaves the sandbox in the
wrong state; with ours, graph state and sandbox filesystem stay consistent.*

Scope note: we restore the **filesystem**, not guest memory. SPACE has no public "new sandbox from a
full snapshot" route (see API notes). Deep Agents runs stateless shell commands, so the filesystem is
the state that matters.

## Core design: fork-anchored snapshot binding
- **Anchor** = a suspended sandbox forked from the live sandbox. Never run again, only forked from.
  Each checkpoint stores `metadata["space"] = {template, anchor_id}` (`SnapshotBinding`);
  `anchor_id: None` means "pristine template", an explicit state, not a missing snapshot.
- **Dirty tracking**: `SpaceSandbox.generation` is bumped by `execute` and `upload_files`, but not by
  the read-only `BaseSandbox` helpers (`ls`, `read`, `grep`, `glob`), which run under a context flag;
  `ThreadRoutedSandbox` forwards those helpers whole to the thread's sandbox. A new anchor
  is taken only if it changed since the last one (one snapshot per state change, not per superstep).
- **Position comes from lineage, never from loads** (`SandboxRegistry` in `snapshots.py`). LangGraph
  calls `get_tuple` for pure inspection too (e.g. `checkpoint_superseded` probes the latest), so loads
  move nothing. Instead: `put(parent=P, X)` binds X to the current sandbox state if the thread is at P,
  otherwise the thread jumped (rewind/replay/update_state/fork) and X binds to P's state. Replays in
  LangGraph 1.2 write a `source="fork"` checkpoint first, so they go through the same path.
- **Async durability**: with LangGraph's default `durability="async"`, checkpoint X is saved in the
  background while tasks based on X already run (verified). So every backend call first waits until
  its base checkpoint (`configurable["checkpoint_map"][""]`) is bound. `durability="exit"` is rejected.
- **Routing**: Deep Agents 0.7 takes a backend *instance*, not a factory. `SpaceCheckpointer.backend` is
  a `ThreadRoutedSandbox` that reads `thread_id`/`checkpoint_map` from `langgraph.config.get_config()`
  per call. It lives in `checkpointer.py`, so `backend.py` stays free of LangGraph.
- **Restore** = fork a new sandbox from the anchor, lazily on first use. The previous live sandbox is
  retired (never mutated); `close()` / `delete_retired()` delete retired ones,
  `delete_unreferenced_anchors()` deletes anchors no stored checkpoint references.
- **Known limitation**: a step that fails mid-way leaves its partial sandbox writes for its retry
  (keeping them is right for parallel peers that succeeded; exact fidelity needs per-task snapshots).
- **Fork-and-select**: anchor the current state, issue N concurrent fork requests, run each branch, run
  the test command in each, keep the lowest-index passing branch, delete the rest in `finally`.

## Repo layout
```
src/langchain_space/
  client.py        # SpaceClient Protocol + HttpSpaceClient (the ONLY module that knows SPACE's API)
  backend.py       # SpaceSandbox(BaseSandbox); tracks a mutation `generation` counter
  snapshots.py     # SnapshotBinding, anchors (fork-to-suspended), lineage-driven SandboxRegistry
  checkpointer.py  # SpaceCheckpointer (wraps any saver) + ThreadRoutedSandbox (per-thread backend)
  fork.py          # fork_and_select
  credentials.py   # CredentialRule -> network_policy JSON; secret-leak guard
  testing/fake.py  # FakeSpaceClient (in-memory FS per sandbox, fork = deep copy, pluggable exec handler)
  testing/graphs.py # deterministic one-node edit graph + time-travel scenarios (consistency harness)
tests/
  unit/            # fake client only; must pass on Windows
  consistency/     # rewind/fork correctness tests (the core claim)
  integration/     # real SPACE, @pytest.mark.integration; test_standard.py = langchain-tests suite
benchmarks/        # fork latency, anchor latency, restore correctness after N branches
examples/          # buggy_repo/ + fork_and_select_demo.py
SPACE/             # vendored SPACE source (own .git); excluded from git, ruff, mypy
```

## Commands
- Install: `uv sync`
- Lint and format: `uv run ruff check . && uv run ruff format .`
- Types: `uv run mypy src`
- Unit + consistency tests: `uv run pytest tests/unit tests/consistency`
- Integration tests (needs SPACE): set `SPACE_API_URL`, `SPACE_DATA_PLANE_URL`, `SPACE_API_TOKEN`, then
  `uv run pytest -m integration`
- Standard LangChain sandbox suite: `uv run pytest tests/integration/test_standard.py`
  (subclasses `langchain_tests.integration_tests.SandboxIntegrationTests`; mirror the Daytona partner
  package in the deepagents repo)
- Benchmarks: `uv run python benchmarks/run.py`

Run lint, types, unit and consistency tests before saying any change is done.

## Environment
The dev machine is Windows. `uv` was installed with `pip install --user uv`; if `uv` is not found, add
`%APPDATA%\Python\Python311\Scripts` to PATH. Everything except integration tests and benchmarks runs against
`FakeSpaceClient`; don't depend on `sh` being present for unit tests (use the fake's exec handler).
SPACE itself needs Linux x86_64 with /dev/kvm, btrfs, loop devices and privileged Docker
([SPACE/docs/installation.md](SPACE/docs/installation.md)); macOS/Windows are unsupported upstream.
Local endpoints: API `http://127.0.0.1:8080`, data plane `http://127.0.0.1:8090`.
Session token: `POST /auth/v1/sessions {"email":"root@local","password":"password"}` -> `.token`.

## SPACE API notes
Verified from the SPACE source. Cite the file when adding more. Reference clients to copy patterns from:
[SPACE/space_smoke/native.py](SPACE/space_smoke/native.py), [SPACE/examples/first_sandbox.py](SPACE/examples/first_sandbox.py).
There is no SPACE Python SDK; we call the HTTP API with `httpx` + `httpx-sse`.

- **Auth / conventions**: `Authorization: Bearer <token>`. Every mutation needs a unique
  `Idempotency-Key` header. IDs are prefix + UUIDv7: `sbx_`, `op_`, `proc_`, `snap_`. Errors are
  `{code, message, request_id, retryable, details[{key,value}]}`; retry only if `retryable`, honour
  `Retry-After`. Route list: `SPACE/crates/libs/public/api/src/routes.rs`; types: `.../public/api/src/lib.rs`.
- **Create sandbox from template**: `POST /v1/sandboxes {template, env?, resources?{vcpu,memory_mib,disk_mib},
  metadata?, network_policy?, idle_timeout?, idle_action?}` -> 202 `{sandbox_id, operation_id}`.
  Poll `GET /v1/operations/{op}` until `succeeded`, then `GET /v1/sandboxes/{id}` until
  `state == "running"`, then probe `GET /v1/sandboxes/{id}/processes?limit=1`.
  Templates: `seed` (minimal Ubuntu, may lack python/git) is the only one published by the local
  install; use `code-runtime` (python, git, uv, user `user`, `/home/user/workspace`; pytest must be
  installed), built with `space-admin-cli template build code-runtime`.
- **Delete / lifecycle**: `DELETE /v1/sandboxes/{id}` -> 202 + operation; 404 = already gone; retry with a
  new key on `conflict`. `POST /v1/sandboxes/{id}:suspend|:pause|:wake` -> operation to poll.
- **Exec command / stream output**: `POST /v1/sandboxes/{id}/commands {command: argv[], cwd?, env?,
  timeout?: "30s"}` -> 201 with `attach.{events_url, token}`. argv is NOT run through a shell (use
  `["sh","-c",cmd]`); no timeout means no deadline. The process starts only when you open the SSE stream
  `GET {data_plane}/v1/sessions/{proc}/events` with `Authorization: Bearer <attach.token>` (one attach
  only, no replay). Events: `stdout`/`stderr` (`data_base64`), `process_exited` (`exit_code`; 124 =
  timeout, 128+n = signal), `closed`, `error`, `resync_required`. Only `process_exited` means done.
- **File upload / download** (incubating): `POST /v1/sandboxes/{id}/files:write {path, data_base64,
  append?}` (≤16 MiB; parent dir must exist), `files:write_batch {entries[]}` (≤256 entries, ≤3 MiB),
  `files:read {path, offset, length (1..16 MiB)}` -> raw bytes, `files:stat`, `files:list`,
  `files:mkdir {path, parents}`, `files:remove`, `files:rename`. Missing file = 404 with detail
  `resource=path` (a 404 without it = sandbox not found). Relative paths resolve under `/home/user`.
- **Snapshot**: full (memory + disk) `POST /v1/sandboxes/{id}:checkpoint` and `:rollback` are in place
  only, pause the VM, write a full RAM file (no diffs), and `:checkpoint` does not return the snapshot
  name. **We do not use them.** Rolling snapshots exist (opt-in, `:enable_snapshots`) but are node-local
  and not addressable from the public API.
- **Fork / resume from snapshot**: E2B-compatible route (not in OpenAPI): `POST /sandboxes/{id}/fork
  {timeout?, count: 1}` + `Idempotency-Key` -> new sandbox id. Disk-only (child cold-boots), source may
  be running, paused or suspended, `count` must be 1, so N-way = N concurrent requests (they share one
  capture). Hard deadline 240 s. Not supported on the direct (non-Kubernetes) substrate.
  Code: `SPACE/crates/libs/api/src/routes/e2b/router.rs`, `.../api/src/fork_operations.rs`.
- **Credential injection / egress**: `POST /v1/credential-references {provider: "aws_secrets_manager",
  request_id, secret_arn}` -> `{credential_reference_id}`. Bind via `network_policy` (on create or
  `PATCH /v1/sandboxes/{id} {network_policy: {mode: "set", policy}}`) with `tls_interception: "on"` and
  `network_credential_rules[{rule_id, destination{exact_host, port, path_prefix},
  credential_header_bindings[{header_name, credential_reference_id}], supported_http_methods}]`
  (only GET/POST/HEAD). The gateway overwrites the header; the secret is used verbatim (store
  `Bearer xyz` if you need the prefix); the guest never sees it. Requires AWS Secrets Manager + AWS
  Private CA; not configured in a stock local install. Egress: `egress{default_action, allow[], deny[]}`
  with `{ip|cidr|domain|wildcard_domain}` entries; domain rules need DNS via 8.8.8.8/1.1.1.1 allowed.
  Example: `SPACE/space_smoke/scenarios/native_aws_secret_network_credential_injection.py`.

### Open questions (resolve on a real SPACE host, then move into the notes above)
- Is the E2B fork route reachable on a local Kind install without ECW wildcard hosts? Which auth header?
- Fork latency from a suspended anchor; does the child come up `running` or need `:wake`?
  (`HttpSpaceClient.fork` currently waits for `running`.)
- The fork `timeout` is the child's lifetime (u32 seconds, default 300; only `> 0` is validated). Does it
  still expire a *suspended* anchor? Is there a server-side cap? We request 7 days by default.
- Does a fork inherit the source's `network_policy`? (`Snapshotter.restore` re-applies it to be safe.)
- Does `seed` include python3? Does `files:read` with `length` past EOF return up to EOF?
- Confirm the exact `BaseSandbox` / `ExecuteResponse` / `FileUploadResponse` signatures in the installed
  `deepagents` version before writing `backend.py`.

## Design rules
- `client.py` is the only module that knows SPACE URLs and JSON. Everything else depends on the
  `SpaceClient` Protocol, so unit tests use `FakeSpaceClient`.
- `backend.py` must not import LangGraph, so it stays usable from a plain script.
- **Write-before-commit**: create the anchor first, then save the checkpoint with it in
  `metadata["space"]`. If the fork fails, no checkpoint is written. A checkpoint without an anchor is an
  error (`MissingSnapshotError`), never a silent fallback. Only skip a new anchor when `generation` is
  unchanged.
- Restoring an old checkpoint **forks a new sandbox** from its anchor. Never mutate a sandbox or anchor
  that another branch may still be using.
- Never put real credentials in sandbox env vars, files or commands. Tests assert this via the fake's
  request log.
- Always clean up sandboxes in tests (fixtures with teardown), including on failure.

## Roadmap
Full plan: `.claude/plans/plan-the-project-describe-keen-naur.md` (local only; gitignored).
M0 scaffold (git init, pyproject, ruff/mypy/pytest, CI) → M1 client + fake (respx tests) → M2 backend +
standard suite wiring → M3 consistency tests, failing first → M4 snapshots + checkpointer (M3 green) →
M5 fork-and-select + demo → M6 Linux SPACE host + integration → M7 benchmarks → M8 credentials on real
AWS + exfiltration demo → M9 README.

Status: everything that runs on the fake/mocked HTTP is done: M0–M5, M7 (harness; simulated numbers),
M8 (wiring, mocked HTTP tests, simulated exfiltration demo) and M9 (README). Remaining work needs real
infrastructure:
- M6: bring up SPACE on Linux/KVM, build `code-runtime`, run `uv run pytest -m integration`, resolve the
  open questions above.
- M7: `uv run python benchmarks/run.py --space` and replace the simulated table in README.md.
- M8: Secrets Manager + Private CA + an echo endpoint, then `tests/integration/test_space_credentials.py`.
- README: record the demo GIF (placeholder comment in README.md).

## Testing philosophy
- For every feature, write the failing test first. The consistency tests must show the bug with a plain
  backend before showing the fix with ours.
- Integration tests are slow and need SPACE; keep them few and meaningful.
- Benchmarks report median and p90, number of runs, and the machine they ran on.

## Code style
- Python >= 3.11, full type hints (mypy strict), ruff defaults, docstrings on public classes and functions.
- Small, focused commits with clear messages (this repo is a portfolio piece; history matters).
- No new dependencies without a reason noted in the PR or commit message. Planned: `deepagents`,
  `langgraph`, `langchain-core`, `httpx`, `httpx-sse` (SPACE streams exec output over SSE); dev:
  `pytest`, `pytest-asyncio`, `respx`, `ruff`, `mypy`, `langchain-tests`.

## README goals (keep in mind while building)
Problem in two sentences, one architecture diagram, a short demo GIF, the benchmark table, and a
"Design decisions" section: fork-anchored vs in-place rollback, filesystem-only restore, dirty tracking.
