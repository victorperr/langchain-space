"""Rewind demo: rewind a LangGraph thread and see what the next step finds in the sandbox.

    uv run python examples/rewind_demo.py           # simulated sandboxes, runs anywhere
    uv run python examples/rewind_demo.py --space   # real SPACE (SPACE_* and SPACE_ECW_* env vars)

A one-node graph writes v1, v2, then v3 to a file. We rewind to the checkpoint taken after v1
and run one more step. With a plain sandbox backend that step still finds v3: the graph went
back, the files did not. With langchain-space it finds v1, because every checkpoint is bound
to a SPACE snapshot and rewinding forks a sandbox from it.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from collections.abc import Callable, Iterator
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from langchain_space.backend import SpaceSandbox
from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.client import HttpSpaceClient, SpaceClient
from langchain_space.testing import FakeSpaceClient
from langchain_space.testing.graphs import (
    EDIT_FILE,
    build_edit_graph,
    find_state,
    last_observed,
    thread,
)

GREEN, RED, BOLD, DIM, RESET = "\033[32m", "\033[31m", "\033[1m", "\033[2m", "\033[0m"
THREAD = "demo"


def rewind_and_step(graph: Any) -> tuple[str, float]:
    """Write v1, v2, v3; rewind to after v1; run one step; return what it saw and its time."""
    for target in ("v1", "v2", "v3"):
        started = time.perf_counter()
        graph.invoke({"target": target}, thread(THREAD))
        print(f"  step writes {target}  {DIM}({time.perf_counter() - started:.1f}s){RESET}")
    after_v1 = find_state(graph, THREAD, target="v1", finished=True)
    print(f"  {BOLD}rewind to the checkpoint after v1{RESET}, run one more step")
    started = time.perf_counter()
    seen = last_observed(graph.invoke({"target": "v4"}, after_v1.config))
    return seen, time.perf_counter() - started


def verdict(seen: str, seconds: float) -> str:
    took = f"{DIM}({seconds:.1f}s){RESET}"
    if seen == "v1":
        return f'{GREEN}{BOLD}it sees "{seen}": OK, the sandbox rewound too{RESET}  {took}'
    return f'{RED}{BOLD}it sees "{seen}": WRONG, the sandbox stayed in the future{RESET}  {took}'


@contextlib.contextmanager
def plain_backend(client: SpaceClient, template: str) -> Iterator[Any]:
    backend = SpaceSandbox.create(client, template)
    try:
        yield build_edit_graph(backend, InMemorySaver())
    finally:
        backend.terminate()


@contextlib.contextmanager
def langchain_space(client: SpaceClient, template: str) -> Iterator[Any]:
    saver = SpaceCheckpointer(InMemorySaver(), client, template=template)
    try:
        yield build_edit_graph(saver.backend, saver)
    finally:
        saver.close()
        for anchor_id in saver.registry.anchors:  # anchors are ours to clean up in a demo
            with contextlib.suppress(Exception):
                client.delete_sandbox(anchor_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", action="store_true", help="use a real SPACE deployment")
    parser.add_argument("--template", default="code-runtime")
    args = parser.parse_args()

    make_client: Callable[[], SpaceClient] = (
        HttpSpaceClient.from_env if args.space else FakeSpaceClient
    )
    where = "real SPACE microVMs" if args.space else "simulated sandboxes"
    print(f"{BOLD}Rewinding a LangGraph agent: does the sandbox rewind too?{RESET}")
    print(f"{DIM}{where}; the graph edits {EDIT_FILE}{RESET}\n")

    for title, setup in (
        ("Plain sandbox backend + LangGraph checkpointer", plain_backend),
        ("langchain-space: each checkpoint bound to a SPACE snapshot", langchain_space),
    ):
        print(f"{BOLD}{title}{RESET}")
        with setup(make_client(), args.template) as graph:
            print(f"  {verdict(*rewind_and_step(graph))}\n")


if __name__ == "__main__":
    main()
