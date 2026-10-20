"""A tiny deterministic LangGraph graph whose only effect is editing one file in a sandbox.

Each run of the `edit` node records the file content it *observed* before writing the new
`target`. If graph state and sandbox state are consistent, a node resumed from checkpoint C
must observe the content that was in the file when C was saved. That makes rewind bugs
visible as plain string mismatches, with no LLM involved.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Durability, StateSnapshot

EDIT_FILE = "/home/user/work/app.txt"
MISSING = "<missing>"


class EditState(TypedDict, total=False):
    """Graph state: the content to write, and what each `edit` run saw beforehand."""

    target: str
    observed: Annotated[list[str], operator.add]


def build_edit_graph(
    backend: SandboxBackendProtocol, checkpointer: BaseCheckpointSaver[Any]
) -> CompiledStateGraph[EditState, Any, EditState, EditState]:
    """Compile `START -> edit -> END`, where `edit` reads then overwrites `EDIT_FILE`."""

    def edit(state: EditState) -> EditState:
        [found] = backend.download_files([EDIT_FILE])
        seen = found.content.decode() if found.content is not None else MISSING
        [written] = backend.upload_files([(EDIT_FILE, state["target"].encode())])
        if written.error is not None:
            raise RuntimeError(f"upload failed: {written.error}")
        return {"observed": [seen]}

    builder = StateGraph(EditState)
    builder.add_node("edit", edit)
    builder.add_edge(START, "edit")
    builder.add_edge("edit", END)
    return builder.compile(checkpointer=checkpointer)


@dataclass(frozen=True)
class Observation:
    """What a resumed `edit` run saw, versus what a consistent sandbox would have shown."""

    scenario: str
    observed: str
    expected: str

    @property
    def consistent(self) -> bool:
        """True when the sandbox matched graph state."""
        return self.observed == self.expected


def thread(thread_id: str) -> RunnableConfig:
    """Config for a thread."""
    return {"configurable": {"thread_id": thread_id}}


def find_state(
    graph: CompiledStateGraph[EditState, Any, EditState, EditState],
    thread_id: str,
    *,
    target: str,
    finished: bool,
) -> StateSnapshot:
    """Find the checkpoint with `target` that is after (`finished`) or before its `edit` run.

    "Before" means `next == ("edit",)`; input checkpoints (`next == ("__start__",)`) still
    hold the *previous* run's target, so they never match.
    """
    wanted_next: tuple[str, ...] = () if finished else ("edit",)
    for snapshot in graph.get_state_history(thread(thread_id)):
        if snapshot.values.get("target") == target and snapshot.next == wanted_next:
            return snapshot
    raise LookupError(f"no checkpoint with target={target!r} finished={finished}")


def last_observed(result: dict[str, Any]) -> str:
    """The observation recorded by the most recent `edit` run."""
    observed: list[str] = result["observed"]
    return observed[-1]


def run_scenarios(
    graph: CompiledStateGraph[EditState, Any, EditState, EditState],
    *,
    durability: Durability = "async",
) -> list[Observation]:
    """Run every time-travel scenario on fresh threads and report what each resumed run saw.

    `durability="async"` (LangGraph's default) saves checkpoints in the background while
    the next step already runs, which is the hardest case for snapshot binding.
    """
    observations: list[Observation] = []
    prefix = f"{durability}-"

    def invoke(value: EditState | None, config: RunnableConfig) -> dict[str, Any]:
        return graph.invoke(value, config, durability=durability)

    # 1. Rewind and continue: go back to the checkpoint after v1, then give new input.
    tid = prefix + "rewind"
    for target in ("v1", "v2", "v3"):
        invoke({"target": target}, thread(tid))
    after_v1 = find_state(graph, tid, target="v1", finished=True)
    result = invoke({"target": "v4"}, after_v1.config)
    observations.append(Observation("rewind", last_observed(result), "v1"))

    # 2. Replay: re-execute run 2 from its input checkpoint (taken before `edit` ran).
    tid = prefix + "replay"
    for target in ("v1", "v2", "v3"):
        invoke({"target": target}, thread(tid))
    before_v2 = find_state(graph, tid, target="v2", finished=False)
    result = invoke(None, before_v2.config)
    observations.append(Observation("replay", last_observed(result), "v1"))

    # 3. Branch isolation: two branches from one checkpoint must not see each other's files.
    tid = prefix + "branches"
    invoke({"target": "v1"}, thread(tid))
    after_v1 = find_state(graph, tid, target="v1", finished=True)
    result_a = invoke({"target": "a"}, after_v1.config)
    head_a = graph.get_state(thread(tid))
    observations.append(Observation("branch-a", last_observed(result_a), "v1"))
    result_b = invoke({"target": "b"}, after_v1.config)
    observations.append(Observation("branch-b", last_observed(result_b), "v1"))
    result_a2 = invoke({"target": "a2"}, head_a.config)
    observations.append(Observation("branch-a-resumed", last_observed(result_a2), "a"))

    return observations
