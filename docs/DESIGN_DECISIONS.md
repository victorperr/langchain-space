# Design decisions

**Fork-anchored snapshots, not in-place rollback.** SPACE has full memory+disk checkpoints, but
only as in-place `:checkpoint`/`:rollback` on the same sandbox, and `:checkpoint` does not
return the snapshot's id. Rolling back in place would mutate a sandbox that another branch may
still be using. A fork that is immediately suspended gives an immutable, addressable snapshot,
and any number of branches can fork from it concurrently.

**Filesystem only, not memory.** SPACE forks copy the disk, and the child cold-boots. Deep
Agents runs one-off shell commands, so the filesystem is the state that matters. Long-running
processes do not survive a rewind.

**Position from lineage, not from loads.** LangGraph also calls `get_tuple` just to inspect,
for example to check whether an addressed checkpoint has been superseded. So loading a
checkpoint moves nothing. Only `put(parent, child)` and the base checkpoint of a running task
decide where a thread is.

**Wait for the base checkpoint.** With LangGraph's default `durability="async"`, checkpoint N is
saved in the background while step N+1 is already running (verified on LangGraph 1.2). A
snapshot taken inside `put` could therefore capture step N+1's partial writes. Every sandbox
call first waits until the checkpoint its task is based on has been bound.
`durability="exit"` never saves intermediate checkpoints, so it is rejected.

**One snapshot per change, not per step.** `SpaceSandbox.generation` is bumped by every call
that may write. Steps that don't touch the sandbox, such as most LLM turns, reuse the previous
anchor.

**Write-before-commit.** The snapshot is taken before the checkpoint is saved. A checkpoint
without a binding raises `MissingSnapshotError` and never falls back silently.

**Known limitation.** A step that fails partway leaves its partial writes for its retry.
Keeping them is correct for parallel tasks in that step that succeeded, because LangGraph
won't rerun them. Exact fidelity would need a snapshot per task.
