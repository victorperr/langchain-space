# CLAUDE.md — langchain-space

A LangChain Deep Agents sandbox backend for SPACE (Perplexity's open-source Firecracker microVM
platform, Apache 2.0) where rewinding or forking a LangGraph thread also restores the sandbox
filesystem. The SPACE source may be vendored read-only in `SPACE/` (gitignored); it is reference
material, not ours to edit.

**Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing `snapshots.py`,
`checkpointer.py` or `client.py`.** It holds the design (anchors, lineage-based binding, async
durability, retention), the SPACE and LangSmith API notes, and the open questions. Keep it up to
date when you verify a new API fact (cite the SPACE source file).

## Layout
```
src/langchain_space/
  client.py        # SpaceClient protocol + HttpSpaceClient (the only module that knows SPACE's API)
  backend.py       # SpaceSandbox(BaseSandbox); no LangGraph imports
  snapshots.py     # SnapshotBinding, anchors, lineage-driven SandboxRegistry
  checkpointer.py  # SpaceCheckpointer + ThreadRoutedSandbox
  fork.py          # fork_and_select
  credentials.py   # CredentialRule -> network_policy JSON; SecretGuard
  testing/         # FakeSpaceClient, consistency harness graph
  langsmith/       # optional extra: LangSmithClient; the core never imports it
tests/unit, tests/consistency, tests/langsmith   # fake/stub only; must pass on Windows
tests/integration                                 # real SPACE, -m integration
benchmarks/, examples/
```

## Commands
- Install: `uv sync` (if `uv` is missing, add `%APPDATA%\Python\Python311\Scripts` to PATH)
- Lint and format: `uv run ruff check . && uv run ruff format .`
- Types: `uv run mypy src`
- Tests: `uv run pytest tests/unit tests/consistency tests/langsmith`
- Real SPACE: set `SPACE_API_URL`, `SPACE_DATA_PLANE_URL`, `SPACE_API_TOKEN`, then
  `uv run pytest -m integration`; or run the manual workflow `gh workflow run integration.yml`
  (SPACE in Kind on a GitHub runner).
- Real LangSmith: set `LANGSMITH_API_KEY`, then `uv run pytest -m langsmith`
- Benchmarks: `uv run python benchmarks/run.py [--space]`

Run lint, types and tests before calling a change done.

## Rules
- The dev machine is Windows: unit tests must not depend on `sh` (use the fake's exec handler).
- Write the failing test first; consistency tests show the bug with a plain backend, then the fix.
- Write before commit: anchor first, then save the checkpoint. Never mutate a sandbox or anchor
  another branch may use. Never put credentials in sandbox env, files or commands.
- Clean up sandboxes in tests (fixtures with teardown), including on failure.
- Python >= 3.11, mypy strict, ruff defaults, docstrings on public API only.
- Small, focused commits with consistent conventional-commit messages. No new dependencies
  without a reason in the commit message.

## Status
Done: adapter, time travel, fork-and-select, LangSmith provider, integration and benchmarks on
real SPACE (Kind). Left: the open questions in docs/ARCHITECTURE.md, and credential injection on
SPACE deployed on AWS (`docs/credential-injection.md`, `examples/check_credential_injection.py`).
