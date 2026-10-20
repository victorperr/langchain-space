"""`SpaceCheckpointer`: a LangGraph checkpointer that keeps the sandbox in step with the graph.

```python
saver = SpaceCheckpointer(InMemorySaver(), HttpSpaceClient.from_env(), template="code-runtime")
agent = create_deep_agent(model=..., backend=saver.backend, checkpointer=saver)
```

`saver.backend` is one Deep Agents backend for every thread: each call is routed to the
current thread's sandbox. Rewinding, replaying or forking a thread (any run started from an
older `checkpoint_id`) transparently continues on a sandbox forked from that checkpoint's
snapshot. See `langchain_space.snapshots` for how checkpoints are bound to snapshots.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
)
from langgraph.config import get_config

from langchain_space.backend import SpaceSandbox
from langchain_space.client import SpaceClient
from langchain_space.snapshots import (
    ROOT_NS,
    SPACE_METADATA_KEY,
    SandboxRegistry,
    SnapshotBinding,
    Snapshotter,
)

_DURABILITY_KEY = "__pregel_durability"
"""LangGraph's run-config key for the durability mode; absent means the default, 'async'."""


class SpaceCheckpointer(BaseCheckpointSaver[Any]):
    """Wraps any LangGraph checkpointer and binds every checkpoint to a SPACE snapshot.

    Storage is delegated to `inner` (e.g. `InMemorySaver`, `PostgresSaver`); this class only
    adds `metadata["space"]` on save and repositions the thread's sandbox on load.

    Args:
        inner: The checkpointer that actually stores checkpoints.
        client: The SPACE client.
        template: SPACE template for a thread's first sandbox.
        env: Environment for new sandboxes. Never put credentials here; see
            `langchain_space.credentials`.
        network_policy: SPACE network policy for every sandbox of every thread.
        anchor_lifetime_seconds: Lifetime requested for snapshot anchors.
        backend_kwargs: Extra arguments for each `SpaceSandbox` (e.g. `default_timeout`).
    """

    def __init__(
        self,
        inner: BaseCheckpointSaver[Any],
        client: SpaceClient,
        *,
        template: str,
        env: Mapping[str, str] | None = None,
        network_policy: Mapping[str, Any] | None = None,
        anchor_lifetime_seconds: int | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(serde=inner.serde)
        content_check = (backend_kwargs or {}).get("content_check")
        if content_check is not None:
            for key, value in (env or {}).items():
                content_check(f"{key}={value}".encode(), "sandbox env")
        self.inner = inner
        self.client = client
        snapshotter = Snapshotter(
            client,
            anchor_lifetime_seconds=anchor_lifetime_seconds,
            env=env,
            network_policy=network_policy,
        )
        self.registry = SandboxRegistry(
            snapshotter, template, load_binding=self._load_binding, backend_kwargs=backend_kwargs
        )
        self.backend = ThreadRoutedSandbox(self.registry)
        """The Deep Agents backend to pass to `create_deep_agent(backend=...)`."""

    # ------------------------------------------------------------------ binding

    def _bind(self, config: RunnableConfig, checkpoint: Checkpoint) -> dict[str, Any]:
        configurable = config["configurable"]
        binding = self.registry.bind(
            configurable["thread_id"],
            configurable.get("checkpoint_ns", ROOT_NS),
            configurable.get("checkpoint_id"),  # the parent of the checkpoint being saved
            checkpoint["id"],
        )
        return {SPACE_METADATA_KEY: binding.to_metadata()}

    def _load_binding(self, thread_id: str, checkpoint_id: str) -> SnapshotBinding | None:
        stored = self.inner.get_tuple(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ROOT_NS,
                    "checkpoint_id": checkpoint_id,
                }
            }
        )
        return SnapshotBinding.from_metadata(stored.metadata) if stored is not None else None

    @staticmethod
    def _validate(checkpoint_tuple: CheckpointTuple | None) -> None:
        if checkpoint_tuple is None:
            return
        if checkpoint_tuple.config["configurable"].get("checkpoint_ns", ROOT_NS) == ROOT_NS:
            SnapshotBinding.from_metadata(checkpoint_tuple.metadata)  # raises if missing

    def sandbox(self, thread_id: str) -> SpaceSandbox:
        """The sandbox for `thread_id` at its current position (e.g. to inspect files)."""
        return self.registry.sandbox(thread_id)

    def delete_unreferenced_anchors(self) -> list[str]:
        """Delete anchors created by this process that no stored checkpoint references."""
        referenced = {
            SnapshotBinding.from_metadata(t.metadata).anchor_id for t in self.inner.list(None)
        }
        deleted = [a for a in self.registry.anchors if a not in referenced]
        for anchor_id in deleted:
            self.client.delete_sandbox(anchor_id)
        self.registry.anchors = [a for a in self.registry.anchors if a in referenced]
        return deleted

    def close(self) -> None:
        """Delete every live sandbox. Anchors stay, so stored checkpoints remain restorable."""
        self.registry.close()

    # ------------------------------------------------------------------ sync API

    @property
    def config_specs(self) -> list[Any]:
        """Delegated to `inner`."""
        return self.inner.config_specs

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Load a checkpoint, refusing root checkpoints that have no snapshot binding.

        Loading does not move the sandbox: LangGraph also loads checkpoints just to inspect
        them. The thread moves when a run saves or works on top of another checkpoint.
        """
        checkpoint_tuple = self.inner.get_tuple(config)
        self._validate(checkpoint_tuple)
        return checkpoint_tuple

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Delegated to `inner`; listing history does not move any sandbox."""
        return self.inner.list(config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Snapshot first, then save the checkpoint with its snapshot (write-before-commit)."""
        bound = {**metadata, **self._bind(config, checkpoint)}
        return self.inner.put(config, checkpoint, bound, new_versions)  # type: ignore[arg-type]

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Delegated to `inner`."""
        self.inner.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        """Delete the thread's checkpoints and retire its sandbox."""
        self.inner.delete_thread(thread_id)
        self.registry.forget(thread_id)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        """Delegated to `inner`."""
        self.inner.delete_for_runs(run_ids)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Delegated to `inner`; the copy restores from the same snapshots on first use."""
        self.inner.copy_thread(source_thread_id, target_thread_id)

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        """Delegated to `inner`."""
        self.inner.prune(thread_ids, strategy=strategy)

    def get_next_version(self, current: Any, channel: None) -> Any:
        """Delegated to `inner` (version formats differ between savers)."""
        return self.inner.get_next_version(current, channel)

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Delegated to `inner`, so walking ancestors does not move the sandbox."""
        return self.inner.get_delta_channel_history(config=config, channels=channels)

    # ------------------------------------------------------------------ async API

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async `get_tuple`."""
        checkpoint_tuple = await self.inner.aget_tuple(config)
        self._validate(checkpoint_tuple)
        return checkpoint_tuple

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async `list`."""
        async for item in self.inner.alist(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async `put`. The snapshot (blocking SPACE calls) runs in a worker thread."""
        binding = await asyncio.to_thread(self._bind, config, checkpoint)
        bound = {**metadata, **binding}
        return await self.inner.aput(config, checkpoint, bound, new_versions)  # type: ignore[arg-type]

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async `put_writes`."""
        await self.inner.aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        """Async `delete_thread`."""
        await self.inner.adelete_thread(thread_id)
        self.registry.forget(thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        """Async `delete_for_runs`."""
        await self.inner.adelete_for_runs(run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Async `copy_thread`."""
        await self.inner.acopy_thread(source_thread_id, target_thread_id)

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        """Async `prune`."""
        await self.inner.aprune(thread_ids, strategy=strategy)

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Async `get_delta_channel_history`."""
        return await self.inner.aget_delta_channel_history(config=config, channels=channels)


class ThreadRoutedSandbox(BaseSandbox):
    """A Deep Agents backend that forwards each call to the current thread's sandbox.

    The thread and the checkpoint the running task is based on come from LangGraph's run
    config, so this backend only works inside a graph run. Each call first makes sure the
    thread's sandbox is in that checkpoint's state (see `langchain_space.snapshots`).
    """

    def __init__(self, registry: SandboxRegistry) -> None:
        self._registry = registry

    @staticmethod
    def _configurable() -> dict[str, Any]:
        try:
            configurable: dict[str, Any] = get_config()["configurable"]
        except RuntimeError as error:
            raise RuntimeError(
                "SpaceCheckpointer.backend is only usable inside a LangGraph run; outside one, "
                "use SpaceCheckpointer.sandbox(thread_id)"
            ) from error
        if "thread_id" not in configurable:
            raise RuntimeError("SpaceCheckpointer.backend needs a thread_id in the run config")
        return configurable

    def _task(self) -> tuple[str, str | None]:
        """(thread_id, base root checkpoint id) of the running task."""
        configurable = self._configurable()
        if configurable.get(_DURABILITY_KEY) == "exit":
            raise ValueError(
                "SpaceCheckpointer needs durability='sync' or 'async': with 'exit', "
                "intermediate checkpoints are never saved, so they cannot be snapshotted"
            )
        base_checkpoints: dict[str, str] = configurable.get("checkpoint_map") or {}
        return configurable["thread_id"], base_checkpoints.get(ROOT_NS)

    def _target(self) -> SpaceSandbox:
        return self._registry.acquire(*self._task())

    def current(self) -> SpaceSandbox:
        """The running task's sandbox, e.g. to pass to `fork_and_select`."""
        return self._target()

    def adopt(self, sandbox: SpaceSandbox) -> None:
        """Continue the running task's thread in `sandbox` (e.g. a `fork_and_select` winner).

        The previous sandbox is retired, not mutated; the next checkpoint snapshots `sandbox`.
        """
        thread_id, base = self._task()
        self._registry.adopt(thread_id, base, sandbox)

    @property
    def id(self) -> str:
        """`space-thread:<thread_id>` inside a run, `space-routed` outside."""
        try:
            return f"space-thread:{self._configurable()['thread_id']}"
        except RuntimeError:
            return "space-routed"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run `command` in the current thread's sandbox."""
        return self._target().execute(command, timeout=timeout)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload to the current thread's sandbox."""
        return self._target().upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download from the current thread's sandbox."""
        return self._target().download_files(paths)
