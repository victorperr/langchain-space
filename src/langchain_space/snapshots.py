"""Binding LangGraph checkpoints to SPACE snapshots ("anchors").

An *anchor* is a suspended sandbox forked from a live sandbox. It is never run again, only
forked from, so it is an immutable snapshot of a filesystem. Every checkpoint records the
anchor holding the sandbox state it corresponds to (`SnapshotBinding`). Moving a thread to an
older checkpoint forks a **new** sandbox from that checkpoint's anchor; nothing another
branch may still use is ever mutated.

How the registry knows where a thread is: from checkpoint *lineage*, not from loads.
LangGraph passes the parent checkpoint id to every `put`, and every task knows the
checkpoint it runs on top of (its base). So:

- `put(parent=P, X)`: if the live sandbox is positioned at P, X binds to the current
  sandbox state (snapshotting only if it changed). Otherwise the thread jumped (rewind,
  replay, `update_state`, fork) and X binds to P's state, which the next use restores.
- A sandbox call from a task based on X first waits until X's binding is known. With
  `durability="async"` (LangGraph's default) checkpoint X is saved in the background while
  tasks on top of it already run; waiting guarantees X is snapshotted before they write.
  With `durability="exit"` intermediate checkpoints are never saved, so it is rejected.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_space.backend import SpaceSandbox
from langchain_space.client import SpaceClient

SPACE_METADATA_KEY = "space"
"""Key under which a checkpoint's metadata stores its `SnapshotBinding`."""

ROOT_NS = ""

BindingLoader = Callable[[str, str], "SnapshotBinding | None"]
"""(thread_id, checkpoint_id) -> binding of a stored root checkpoint, or None if not stored."""


class MissingSnapshotError(LookupError):
    """A checkpoint has no SPACE snapshot binding, so its sandbox state cannot be restored."""


@dataclass(frozen=True)
class SnapshotBinding:
    """The sandbox state a checkpoint corresponds to.

    `anchor_id=None` means "pristine": no sandbox had been used yet, so the state is a fresh
    sandbox from `template`. That is an explicit state, not a missing snapshot.
    """

    template: str
    anchor_id: str | None = None

    @property
    def pristine(self) -> bool:
        """True when the state is a fresh sandbox from `template`."""
        return self.anchor_id is None

    def to_metadata(self) -> dict[str, str | None]:
        """Serialize for `CheckpointMetadata`."""
        return {"template": self.template, "anchor_id": self.anchor_id}

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> SnapshotBinding:
        """Read the binding stored in checkpoint metadata.

        Raises:
            MissingSnapshotError: the checkpoint was not saved by `SpaceCheckpointer`.
        """
        raw = (metadata or {}).get(SPACE_METADATA_KEY)
        if not isinstance(raw, Mapping) or not isinstance(raw.get("template"), str):
            raise MissingSnapshotError(
                "checkpoint has no SPACE snapshot binding; it was not saved by "
                "SpaceCheckpointer, so its sandbox state cannot be restored"
            )
        anchor_id = raw.get("anchor_id")
        return cls(raw["template"], anchor_id if isinstance(anchor_id, str) else None)


class Snapshotter:
    """Creates anchors and restores sandboxes from them.

    Args:
        client: The SPACE client.
        anchor_lifetime_seconds: Lifetime requested for anchors (SPACE fork `timeout`).
        env: Environment for pristine sandboxes.
        network_policy: Network policy for pristine sandboxes, re-applied to restored ones
            (SPACE does not document whether forks inherit it).
    """

    def __init__(
        self,
        client: SpaceClient,
        *,
        anchor_lifetime_seconds: int | None = None,
        env: Mapping[str, str] | None = None,
        network_policy: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self._anchor_lifetime = anchor_lifetime_seconds
        self._env = dict(env or {})
        self._network_policy = dict(network_policy) if network_policy is not None else None

    @property
    def network_policy(self) -> dict[str, Any] | None:
        """The network policy every restored sandbox gets."""
        return self._network_policy

    def anchor(self, sandbox_id: str) -> str:
        """Fork `sandbox_id` into a suspended, never-again-run anchor; return its id."""
        anchor_id = self.client.fork(sandbox_id, lifetime_seconds=self._anchor_lifetime)
        try:
            self.client.suspend(anchor_id)
        except BaseException:
            with contextlib.suppress(Exception):
                self.client.delete_sandbox(anchor_id)
            raise
        return anchor_id

    def restore(self, binding: SnapshotBinding) -> str:
        """Create a new running sandbox in the state `binding` describes; return its id."""
        if binding.anchor_id is None:
            return self.client.create_sandbox(
                binding.template, env=self._env, network_policy=self._network_policy
            )
        sandbox_id = self.client.fork(binding.anchor_id)
        if self._network_policy is not None:
            self.client.set_network_policy(sandbox_id, self._network_policy)
        return sandbox_id


@dataclass
class _Live:
    backend: SpaceSandbox
    anchor_id: str | None
    """Anchor holding this sandbox's state as of `anchored_generation`."""
    anchored_generation: int

    @property
    def clean(self) -> bool:
        return self.backend.generation == self.anchored_generation


@dataclass
class _Thread:
    position: str | None = None
    """Root checkpoint the thread's filesystem is at (plus writes by tasks based on it)."""
    position_binding: SnapshotBinding | None = None
    live: _Live | None = None
    bindings: dict[str, SnapshotBinding] = field(default_factory=dict)
    """Bindings of root checkpoints saved by this process."""
    changed: threading.Condition = field(default_factory=lambda: threading.Condition())


class SandboxRegistry:
    """Tracks, per LangGraph thread, where its filesystem is and which sandbox holds it.

    Args:
        snapshotter: Creates anchors and restores sandboxes.
        template: SPACE template for threads that never had a sandbox.
        load_binding: Reads the binding of a root checkpoint from storage (for checkpoints
            saved by another process). Returns None if the checkpoint is not stored.
        wait_timeout: Seconds a sandbox call waits for its base checkpoint to be saved.
        backend_kwargs: Extra arguments for each `SpaceSandbox` (e.g. `default_timeout`).
    """

    def __init__(
        self,
        snapshotter: Snapshotter,
        template: str,
        *,
        load_binding: BindingLoader,
        wait_timeout: float = 600,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.snapshotter = snapshotter
        self.template = template
        self._load_binding = load_binding
        self._wait_timeout = wait_timeout
        self._backend_kwargs = dict(backend_kwargs or {})
        self._threads: dict[str, _Thread] = {}
        self._lock = threading.Lock()
        self.anchors: list[str] = []
        """Every anchor this registry created, oldest first."""
        self.retired: list[str] = []
        """Live sandboxes no thread uses anymore (safe to delete; anchors are separate)."""

    def _thread(self, thread_id: str) -> _Thread:
        with self._lock:
            return self._threads.setdefault(thread_id, _Thread())

    # ------------------------------------------------------------------ checkpoint events

    def bind(
        self, thread_id: str, checkpoint_ns: str, parent_id: str | None, checkpoint_id: str
    ) -> SnapshotBinding:
        """Decide the binding of a checkpoint about to be saved (call before saving it).

        Raises whatever snapshotting raises; nothing is recorded in that case, so the
        caller must not save the checkpoint (write-before-commit).
        """
        slot = self._thread(thread_id)
        with slot.changed:
            if checkpoint_ns != ROOT_NS:
                # Subgraph checkpoints share the thread's sandbox, which follows the root
                # graph's lineage; record the current state without moving the thread.
                return self._current_binding(slot)
            if slot.position != parent_id:
                self._move_to(slot, thread_id, parent_id)
            binding = self._current_binding(slot)
            slot.bindings[checkpoint_id] = binding
            slot.position, slot.position_binding = checkpoint_id, binding
            slot.changed.notify_all()
            return binding

    def acquire(self, thread_id: str, base_checkpoint_id: str | None) -> SpaceSandbox:
        """Return the sandbox for a task based on `base_checkpoint_id`, in that state.

        Waits until the base checkpoint's binding is known (it may still be saving in the
        background), then restores the sandbox if the thread is elsewhere.
        """
        slot = self._thread(thread_id)
        with slot.changed:
            return self._materialize_at(slot, thread_id, base_checkpoint_id)

    def adopt(self, thread_id: str, base_checkpoint_id: str | None, sandbox: SpaceSandbox) -> None:
        """Make `sandbox` the live sandbox of a task based on `base_checkpoint_id`.

        Used to continue in the winner of `fork_and_select`. The winner descends from the
        base state plus the branch's writes, exactly like writes made in place, so the
        thread keeps its position and the next checkpoint snapshots the winner.

        Only `sandbox.id` is taken: the thread drives it through a backend built with this
        registry's settings, so `backend_kwargs` (e.g. `content_check`) still apply.
        """
        slot = self._thread(thread_id)
        with slot.changed:
            self._materialize_at(slot, thread_id, base_checkpoint_id)
            if slot.live is not None and slot.live.backend.id != sandbox.id:
                self.retired.append(slot.live.backend.id)
            # anchored_generation=-1: never "clean", so the next checkpoint snapshots it.
            slot.live = _Live(self._wrap(sandbox.id), anchor_id=None, anchored_generation=-1)

    # ------------------------------------------------------------------ internals

    def _materialize_at(self, slot: _Thread, thread_id: str, base: str | None) -> SpaceSandbox:
        if base is not None and slot.position != base:
            self._wait_known(slot, thread_id, base)
            if slot.position != base:
                self._move_to(slot, thread_id, base)
        return self._materialize(slot)

    def _lookup(self, slot: _Thread, thread_id: str, checkpoint_id: str) -> SnapshotBinding | None:
        return slot.bindings.get(checkpoint_id) or self._load_binding(thread_id, checkpoint_id)

    def _wait_known(self, slot: _Thread, thread_id: str, checkpoint_id: str) -> None:
        deadline = time.monotonic() + self._wait_timeout
        while self._lookup(slot, thread_id, checkpoint_id) is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"checkpoint {checkpoint_id} of thread {thread_id} was never saved; "
                    "SpaceCheckpointer needs durability='sync' or 'async'"
                )
            slot.changed.wait(min(remaining, 0.5))

    def _move_to(self, slot: _Thread, thread_id: str, checkpoint_id: str | None) -> None:
        """Position the thread at `checkpoint_id` (None = before its first checkpoint)."""
        if checkpoint_id is None:
            binding = SnapshotBinding(self.template)
        else:
            found = self._lookup(slot, thread_id, checkpoint_id)
            if found is None:
                raise MissingSnapshotError(
                    f"checkpoint {checkpoint_id} of thread {thread_id} has no snapshot binding"
                )
            binding = found
        live = slot.live
        if live is not None and not (live.clean and live.anchor_id == binding.anchor_id):
            # The live sandbox has moved on from that state. Leave it untouched (it is
            # retired, not mutated) and restore from the anchor on next use.
            self.retired.append(live.backend.id)
            slot.live = None
        slot.position, slot.position_binding = checkpoint_id, binding

    def _current_binding(self, slot: _Thread) -> SnapshotBinding:
        live = slot.live
        if live is None:
            return slot.position_binding or SnapshotBinding(self.template)
        if not live.clean:
            generation = live.backend.generation
            live.anchor_id = self.snapshotter.anchor(live.backend.id)
            live.anchored_generation = generation
            self.anchors.append(live.anchor_id)
        return SnapshotBinding(self.template, live.anchor_id)

    def _materialize(self, slot: _Thread) -> SpaceSandbox:
        if slot.live is None:
            binding = slot.position_binding or SnapshotBinding(self.template)
            backend = self._wrap(self.snapshotter.restore(binding))
            slot.live = _Live(backend, binding.anchor_id, backend.generation)
        return slot.live.backend

    def _wrap(self, sandbox_id: str) -> SpaceSandbox:
        """The backend for a thread's live sandbox, with this registry's settings."""
        return SpaceSandbox(
            self.snapshotter.client,
            sandbox_id,
            owns_sandbox=True,
            network_policy=self.snapshotter.network_policy,
            **self._backend_kwargs,
        )

    # ------------------------------------------------------------------ inspection / cleanup

    def sandbox(self, thread_id: str) -> SpaceSandbox:
        """The thread's sandbox at its current position (restored if needed)."""
        slot = self._thread(thread_id)
        with slot.changed:
            return self._materialize(slot)

    def anchors_in_use(self) -> set[str]:
        """Anchors a thread is positioned on or its live sandbox descends from.

        This covers the anchor of a checkpoint that is being saved right now: `bind`
        snapshots first and the checkpoint is stored after, so storage alone would not
        show it yet.
        """
        with self._lock:
            slots = list(self._threads.values())
        used: set[str] = set()
        for slot in slots:
            with slot.changed:
                if slot.live is not None and slot.live.anchor_id is not None:
                    used.add(slot.live.anchor_id)
                if slot.position_binding is not None and slot.position_binding.anchor_id:
                    used.add(slot.position_binding.anchor_id)
        return used

    def live_sandbox_id(self, thread_id: str) -> str | None:
        """The thread's materialized sandbox id, if any."""
        live = self._thread(thread_id).live
        return live.backend.id if live is not None else None

    def forget(self, thread_id: str) -> None:
        """Drop a thread (after its checkpoints were deleted); its sandbox is retired."""
        with self._lock:
            slot = self._threads.pop(thread_id, None)
        if slot is not None and slot.live is not None:
            self.retired.append(slot.live.backend.id)

    def delete_retired(self) -> None:
        """Delete live sandboxes no thread uses anymore. Anchors are kept."""
        while self.retired:
            self.snapshotter.client.delete_sandbox(self.retired.pop())

    def close(self) -> None:
        """Delete every live and retired sandbox. Anchors are kept, so stored checkpoints
        stay restorable."""
        with self._lock:
            threads, self._threads = list(self._threads.values()), {}
        for slot in threads:
            if slot.live is not None:
                self.retired.append(slot.live.backend.id)
        self.delete_retired()
