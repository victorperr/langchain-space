"""Look inside: which snapshot each checkpoint is bound to, and what each sandbox holds.

    uv run python examples/inspect_anchors.py       # simulated sandboxes, runs anywhere

Runs the rewind demo's graph (write v1, v2, v3, rewind to after v1, write v4) on an in-memory
fake of SPACE, then prints the thread's checkpoints with the anchor each is bound to, and
every sandbox the run created: live ones, anchors (suspended forks, never run again) and
restores (forked from an anchor).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.snapshots import SnapshotBinding
from langchain_space.testing import FakeSpaceClient
from langchain_space.testing.graphs import EDIT_FILE, build_edit_graph, find_state, thread

THREAD = "inspect"


def main() -> None:
    fake = FakeSpaceClient()
    saver = SpaceCheckpointer(InMemorySaver(), fake, template="code-runtime")
    graph = build_edit_graph(saver.backend, saver)

    for target in ("v1", "v2", "v3"):
        graph.invoke({"target": target}, thread(THREAD))
    after_v1 = find_state(graph, THREAD, target="v1", finished=True)
    print(f"Rewinding to checkpoint {after_v1.config['configurable']['checkpoint_id'][-8:]}")
    graph.invoke({"target": "v4"}, after_v1.config)

    print("\nCheckpoints, newest first (target = what the next edit writes):")
    print(f"  {'checkpoint':10} {'target':6} {'next':9} bound to")
    for snapshot in graph.get_state_history(thread(THREAD)):
        binding = SnapshotBinding.from_metadata(snapshot.metadata)
        bound = "pristine template" if binding.pristine else f"anchor {binding.anchor_id}"
        next_node = ",".join(snapshot.next) or "-"
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"][-8:]
        target = snapshot.values.get("target", "-")
        print(f"  {checkpoint_id:10} {target:6} {next_node:9} {bound}")

    live = saver.registry.live_sandbox_id(THREAD)
    anchors = set(saver.registry.anchors)
    print(f"\nSandboxes ({len(fake.sandboxes)}); {EDIT_FILE} in each:")
    for sandbox in fake.sandboxes.values():
        role = "anchor" if sandbox.id in anchors else "LIVE" if sandbox.id == live else "retired"
        origin = f"forked from {sandbox.forked_from}" if sandbox.forked_from else "from template"
        content = sandbox.files.get(EDIT_FILE, b"<missing>").decode()
        print(f"  {sandbox.id:12} {role:8} {sandbox.state:10} {content:10} {origin}")
    saver.close()


if __name__ == "__main__":
    main()
