# langchain-space on LangSmith sandboxes

Rewind a Deep Agents thread and its sandbox files go back too, on
[LangSmith sandboxes](https://docs.langchain.com/oss/python/integrations/sandboxes/langsmith)
instead of SPACE. No SPACE deployment needed.

## Why deepagents' `LangSmithSandbox` alone is not enough

`LangSmithSandbox` is a plain backend: LangGraph checkpoints save the messages, not the sandbox.
Rewind a thread and the agent resumes on files written by steps that, from its point of view,
never happened. `SpaceCheckpointer` binds every checkpoint to a snapshot of the sandbox, so a
rewind restores the files as well. This folder makes it work with LangSmith.

## Install and use

```bash
uv add "langchain-space[langsmith]"
export LANGSMITH_API_KEY=...
```

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_space import SpaceCheckpointer
from langchain_space.langsmith import LangSmithClient

client = LangSmithClient()
saver = SpaceCheckpointer(InMemorySaver(), client, template="")  # "" = LangSmith's default image
agent = create_deep_agent(model=..., backend=saver.backend, checkpointer=saver)

try:
    ...  # invoke, rewind with get_state_history / update_state, resume
finally:
    saver.close()  # delete the live sandboxes
    for anchor_id in saver.registry.anchors:  # and the snapshots, once you no longer need them
        client.delete_anchor(anchor_id)
    client.close()
```

`template` is a LangSmith snapshot (`name`, `name:tag` or id) to boot from, or `""` for the
default image. The image needs `sh`, `python3` (deepagents' file tools use it) and coreutils.
`fork_and_select` works too.

## How it maps

| langchain-space | SPACE | LangSmith |
| --- | --- | --- |
| anchor (one per state change) | fork, then suspend | `capture_snapshot` |
| restore a checkpoint | fork the anchor | `create_sandbox(snapshot_id=...)` |
| delete an anchor | delete the sandbox | `delete_snapshot` |
| fork (fork-and-select) | fork | temporary snapshot, new sandbox, delete the snapshot |

`LangSmithClient` implements the same `SpaceClient` protocol as `HttpSpaceClient`, so the
checkpointer, registry and fork-and-select code is shared, not copied.

## Limits

- **Snapshot cost.** A snapshot copies the sandbox's whole root filesystem, which is likely slower
  than a SPACE fork. One is taken only when the sandbox changed since the last one.
- **Snapshots last until deleted.** They are resources in your LangSmith account: delete them
  (`delete_anchor`, or `saver.delete_unreferenced_anchors()`) when their checkpoints are gone.
- **Idle stop.** LangSmith stops idle sandboxes; `LangSmithClient(idle_ttl_seconds=...)` defaults
  to one hour instead of LangSmith's ten minutes.
- **SPACE-only features.** Network policies and credential references raise
  `SpaceError(code="not_supported")`. For credentials, use LangSmith's own auth proxy
  (`proxy_config`).
- **One provider per thread.** You can use SPACE and LangSmith in the same program, with one
  `SpaceCheckpointer` each, but a thread stays on the provider it started on. Each checkpoint
  records its provider; loading it with the other provider's client raises
  `ProviderMismatchError`.
- **Output order.** A command's stdout comes first, then stderr (SPACE interleaves them).

## Tests

```bash
uv run pytest tests/langsmith                             # stub LangSmith, no network
LANGSMITH_API_KEY=... uv run pytest -m langsmith          # real LangSmith
```

The stub (`langchain_space.langsmith.testing.StubSandboxClient`) runs the same rewind scenarios
as `tests/consistency`.
