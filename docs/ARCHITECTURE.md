# Architecture

How `langchain-space` keeps a LangGraph thread and its sandbox filesystem consistent, and the
SPACE and LangSmith API details it relies on. The README covers usage; this file covers the why.

Scope: we restore the **filesystem**, not guest memory. SPACE has no public "new sandbox from a
full snapshot" route (see below), and Deep Agents runs stateless shell commands, so the
filesystem is the state that matters.

## Fork-anchored snapshot binding

### Anchors
An **anchor** is an immutable copy of the live sandbox's filesystem, made by the client's
`create_anchor` / `restore_anchor` / `delete_anchor` (on SPACE a suspended fork, on LangSmith a
snapshot). It is never run, only restored from.

`HttpSpaceClient` returns an anchor as soon as its fork exists and suspends it in the background
(`suspend_anchors_in_background`, on by default). Restoring or deleting the anchor waits for that
suspend first. If the background suspend fails, the anchor stays running.

Each checkpoint stores `metadata["space"] = {template, anchor_id, provider}` (`SnapshotBinding`).
`anchor_id: None` means "pristine template": an explicit state, not a missing snapshot.
`provider` is the client's (`"space"`, `"langsmith"`); `get_tuple` raises `ProviderMismatchError`
for a checkpoint saved by another provider (`None` = older checkpoint, not checked).

### Dirty tracking
`SpaceSandbox.generation` is bumped by `execute` and `upload_files`, but not by the read-only
`BaseSandbox` helpers (`ls`, `read`, `grep`, `glob`), which run under a context flag.
`ThreadRoutedSandbox` forwards those helpers whole to the thread's sandbox. A new anchor is taken
only if `generation` changed since the last one: one snapshot per state change, not per superstep.

### Position comes from lineage, never from loads
LangGraph calls `get_tuple` for pure inspection too (e.g. `checkpoint_superseded` probes the
latest checkpoint), so loads cannot move a thread. `SandboxRegistry` (`snapshots.py`) uses
lineage instead: `put(parent=P, X)` binds X to the current sandbox state if the thread is at P;
otherwise the thread jumped (rewind, replay, `update_state`, fork) and X binds to P's state.
Replays in LangGraph 1.2 write a `source="fork"` checkpoint first, so they take the same path.

### Async durability
With LangGraph's default `durability="async"`, checkpoint X is saved in the background while
tasks based on X already run. So every backend call first waits until its base checkpoint
(`configurable["checkpoint_map"][""]`) is bound. `durability="exit"` never saves intermediate
checkpoints, so it is rejected.

### Routing
Deep Agents 0.7 takes a backend *instance*, not a factory. `SpaceCheckpointer.backend` is a
`ThreadRoutedSandbox` that reads `thread_id` / `checkpoint_map` from
`langgraph.config.get_config()` on each call. It lives in `checkpointer.py`, so `backend.py`
stays free of LangGraph.

### Restore
Restoring forks a new sandbox from the anchor, lazily on first use. The previous live sandbox is
retired, never mutated. `close()` / `delete_retired()` delete retired sandboxes;
`delete_unreferenced_anchors()` deletes anchors no stored checkpoint references.

### Retention
`SnapshotRetention(keep_last, keep)`, opt-in via `retention=` or `prune_snapshots()`. Per thread,
keep the newest `keep_last` distinct anchors plus those of checkpoints `keep` accepts. An anchor
is deleted only if no kept checkpoint of *any* thread uses it (copied threads share anchors).
Checkpoints stay; rewinding to one whose anchor is gone raises `SnapshotGoneError` (also raised
when a restore finds the anchor `not_found`). `claim_for_pruning` marks anchors before it checks
`anchors_in_use`, so a concurrent rewind either keeps the anchor or sees the mark.

### Failed background saves
LangGraph saves a run's checkpoints one after another and stops at the first failure, so tasks
would wait forever for a checkpoint that never comes. A failed root `bind` marks the thread
`broken`, and waiters raise that error. The mark is cleared by the next successful bind and by
any `get_tuple` (every run loads a checkpoint before saving one).

### Known limitation
A step that fails midway leaves its partial sandbox writes for its retry. Keeping them is right
for parallel peers that succeeded; exact fidelity would need per-task snapshots.

### Fork-and-select
Anchor the current state, issue N concurrent fork requests, run each branch, run the test
command in each, keep the lowest-index passing branch, delete the rest in `finally`. Branches use
`SpaceSandbox.derive` (same `content_check`, limits and network policy; the policy is re-applied
to each fork), and `adopt` rewraps the winner with the registry's `backend_kwargs`.

### Rules the code follows
- `client.py` is the only module that knows SPACE URLs and JSON; everything else depends on the
  `SpaceClient` protocol, so unit tests run on `FakeSpaceClient`.
- `backend.py` does not import LangGraph, so it works from a plain script.
- **Write before commit**: create the anchor, then save the checkpoint with it. If the fork
  fails, no checkpoint is written. A checkpoint without an anchor is an error
  (`MissingSnapshotError`), never a silent fallback.
- Never mutate a sandbox or anchor that another branch may still use.
- Never put real credentials in sandbox env vars, files or commands. Tests assert this through
  the fake's request log.

## SPACE API notes
Verified from the SPACE source ([perplexityai/SPACE](https://github.com/perplexityai/SPACE)).
There is no SPACE Python SDK; the client calls the HTTP API with `httpx` + `httpx-sse`. Reference
clients in SPACE: `space_smoke/native.py`, `examples/first_sandbox.py`.

- **Auth and conventions**: `Authorization: Bearer <token>`. Every mutation needs a unique
  `Idempotency-Key`. IDs are prefix + UUIDv7: `sbx_`, `op_`, `proc_`, `snap_`. Errors are
  `{code, message, request_id, retryable, details[{key,value}]}`; retry only if `retryable`,
  honour `Retry-After`. Routes: `crates/libs/public/api/src/routes.rs`; types:
  `crates/libs/public/api/src/lib.rs`.
- **Create from template**: `POST /v1/sandboxes {template, env?, resources?{vcpu,memory_mib,
  disk_mib}, metadata?, network_policy?, idle_timeout?, idle_action?}` -> 202 `{sandbox_id,
  operation_id}`. Poll `GET /v1/operations/{op}` until `succeeded`, then `GET /v1/sandboxes/{id}`
  until `state == "running"`, then probe `GET /v1/sandboxes/{id}/processes?limit=1`.
  The local install publishes only `seed` (minimal Ubuntu, has `python3` but no `python`); use a
  template built from SPACE's `code-runtime` recipe (python, git, uv, user `user`,
  `/home/user/workspace`; install pytest yourself).
- **Delete and lifecycle**: `DELETE /v1/sandboxes/{id}` -> 202 + operation; 404 = already gone;
  retry with a new key on `conflict`. `POST /v1/sandboxes/{id}:suspend|:pause|:wake` -> operation.
- **Commands**: `POST /v1/sandboxes/{id}/commands {command: argv[], cwd?, env?, timeout?: "30s"}`
  -> 201 with `attach.{events_url, token}`. argv is not run through a shell (use
  `["sh","-c",cmd]`); no timeout means no deadline. The process starts only when the SSE stream
  `GET {data_plane}/v1/sessions/{proc}/events` is opened with `Authorization: Bearer
  <attach.token>` (one attach, no replay). Events: `stdout`/`stderr` (`data_base64`),
  `process_exited` (`exit_code`; 124 = timeout, 128+n = signal), `closed`, `error`,
  `resync_required`. Only `process_exited` means done.
- **Files** (incubating): `files:write {path, data_base64, append?, parents?}` (≤16 MiB; without
  `parents: true` the parent directory must exist). `files:write_batch {entries[{path,
  data_base64, append?}], parents?}` -> `{files_written}` (≤256 entries, ≤3 MiB of data + path
  bytes; `parents` cannot be combined with `append`). The guest writes entries in order and stops
  at the first failure without naming it; earlier entries stay written
  (`api/src/routes/public/files.rs`, `spaced/guest/execution/src/filesystem/write.rs`), so
  `HttpSpaceClient.write_files` rewrites a failed batch one file at a time.
  There is no batch read route: `read_files` reads several files with one `sh` command printing
  base64 records (`_READ_FILES_SCRIPT`, needs `wc`, `tr`, `base64`), sends files over ~3/4 of
  `max_output_bytes` to `files:read`, and reads anything it could not parse one by one.
  `files:read {path, offset, length (1..16 MiB)}` -> raw bytes; also `files:stat`, `files:list`,
  `files:mkdir {path, parents}`, `files:remove`, `files:rename`. Missing file = 404 with detail
  `resource=path` (a 404 without it = sandbox not found). Relative paths resolve under
  `/home/user`.
- **Full snapshots, not used**: `:checkpoint` and `:rollback` work in place only, pause the VM,
  write a full RAM file, and `:checkpoint` does not return the snapshot name. Rolling snapshots
  (`:enable_snapshots`) are node-local and not addressable from the public API.
- **Fork**: E2B-compatible route (not in OpenAPI): `POST /sandboxes/{id}/fork {timeout?, count: 1}`
  + `Idempotency-Key` -> new sandbox id. Disk-only (the child cold-boots); the source may be
  running, paused or suspended; `count` must be 1, so N-way means N concurrent requests (they
  share one capture). Hard deadline 240 s. Not supported on the direct (non-Kubernetes)
  substrate. Code: `crates/libs/api/src/routes/e2b/router.rs`, `.../api/src/fork_operations.rs`.
  IDs differ between APIs: native = `sbx_<uuid>`, E2B = bare `<uuid>`; `HttpSpaceClient.fork`
  converts. The E2B API key must belong to the same organization as the native token.
  API keys may not publish templates named `seed` or `code-runtime`, so the integration workflow
  builds the recipe as `lcs-code-runtime`.
- **Credential injection and egress**: `POST /v1/credential-references {provider:
  "aws_secrets_manager", request_id, secret_arn}` -> `{credential_reference_id}`. Bind it through
  `network_policy` (on create, or `PATCH /v1/sandboxes/{id} {network_policy: {mode: "set",
  policy}}`) with `tls_interception: "on"` and `network_credential_rules[{rule_id,
  destination{exact_host, port, path_prefix}, credential_header_bindings[{header_name,
  credential_reference_id}], supported_http_methods}]` (GET/POST/HEAD only). The gateway
  overwrites the header with the secret verbatim; the guest never sees it. Registering only
  stores the ARN, so a broken AWS setup shows up only as requests leaving without the header.
  Egress: `egress{default_action, allow[], deny[]}` with `{ip|cidr|domain|wildcard_domain}`
  entries; domain rules need DNS via 8.8.8.8/1.1.1.1 allowed. Setup:
  [`credential-injection.md`](credential-injection.md).

### Open questions
To resolve on a real SPACE host:
- Fork latency from a suspended anchor; does the child come up `running` or need `:wake`?
  (`HttpSpaceClient.fork` currently waits for `running`.)
- The fork `timeout` is the child's lifetime (u32 seconds, default 300). Does it still expire a
  *suspended* anchor? Is there a server-side cap? We request 7 days by default.
- With the opt-in `idle_timeout`, does a command wake a suspended sandbox, or must we `:wake` it?
  Forks get no idle policy (the E2B fork body has no such field).
- If the opt-in `live_lifetime` is a hard kill, a thread unused for longer loses its live
  sandbox (the registry does not recover).
- Does a fork inherit the source's `network_policy`? (`Snapshotter.restore` re-applies it.)
- Does `files:read` with `length` past EOF return up to EOF?
- Standard suite: `langchain-tests` 1.1.9 still expects relative glob paths, directories in glob
  and `write` refusing existing files; deepagents 0.7 changed all three. Those 7 tests are
  `xfail(strict=True)` in `tests/integration/test_standard.py`.

## LangSmith notes
Verified from the `langsmith` 0.14.11 SDK (`langsmith/sandbox/_client.py`, `_sandbox.py`) and
deepagents' `backends/langsmith.py`. Only `langchain_space/langsmith/client.py` depends on them.
- Sandboxes are addressed by name. `create_sandbox(snapshot=name[:tag] | snapshot_id=uuid,
  run_config={env_vars, work_dir, user}, idle_ttl_seconds (multiple of 60, default 600, 0 =
  never), timeout)`; `stop_sandbox` / `start_sandbox`; `delete_sandbox`.
- `capture_snapshot(sandbox_name, name, run_config=...)` captures the live root filesystem and
  waits until `ready`. The sandbox's own env vars are not carried, so ours go into `run_config`.
  `delete_snapshot(id)`. Snapshots persist until deleted.
- `Sandbox.run(command, timeout=, run_config=)` runs through `/bin/bash`; timeout `0` = none;
  separate stdout/stderr; `CommandTimeoutError` on timeout. `read` raises
  `ResourceNotFoundError(resource_type="file")`; `stat` is HEAD on files only, so our `stat` and
  `mkdir` are small `sh` scripts.
- Open: can a snapshot be deleted while a sandbox booted from it runs? (`close()` retries refused
  deletes.) Snapshot capture latency. Not yet run against real LangSmith.
