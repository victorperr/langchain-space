"""Rewind demo: rewind a LangGraph thread and see what the next step finds in the sandbox.

    uv run python examples/rewind_demo.py           # simulated sandboxes, runs anywhere
    uv run python examples/rewind_demo.py --space   # real SPACE (SPACE_* and SPACE_ECW_* env vars)

The graph has one step, `edit` (`langchain_space.testing.graphs`): a LangGraph node, no LLM,
that reads a file in the sandbox and overwrites it, through the same backend as Deep Agents'
file tools. Steps 1-3 write v1, v2, v3. We rewind to the checkpoint saved after step 1 and run
step 4. With a plain sandbox backend, step 4 reads v3: the graph went back, the file did not.
With langchain-space it reads v1, because every checkpoint is bound to a SPACE snapshot and
rewinding forks a sandbox from it.
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


def took(started: float) -> str:
    return f"{DIM}({time.perf_counter() - started:.1f}s){RESET}"


def rewind_and_step(graph: Any) -> str:
    """Steps 1-3 write v1, v2, v3; rewind to after step 1; step 4 reports what it read."""
    for number, target in enumerate(("v1", "v2", "v3"), start=1):
        started = time.perf_counter()
        graph.invoke({"target": target}, thread(THREAD))
        print(f'  step {number} writes app.txt = "{target}"  {took(started)}')
    after_step_1 = find_state(graph, THREAD, target="v1", finished=True)
    print(f"  {BOLD}rewind to the checkpoint saved after step 1{RESET}")
    started = time.perf_counter()
    seen = last_observed(graph.invoke({"target": "v4"}, after_step_1.config))
    print(f'  step 4 reads app.txt before writing: "{seen}"  {took(started)}')
    return seen


def verdict(seen: str) -> str:
    if seen == "v1":
        return f"{GREEN}{BOLD}OK: the agent and its file are both back after step 1{RESET}"
    return f"{RED}{BOLD}WRONG: the agent is back after step 1, its file is not{RESET}"


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
    print(f"{DIM}{where}{RESET}")
    print("The graph has one step, `edit` (a LangGraph node, no LLM). Each time it runs, it")
    print(f"reads {EDIT_FILE} in the sandbox, then overwrites it with a new value.\n")

    for title, setup in (
        ("Plain sandbox backend + LangGraph checkpointer", plain_backend),
        ("langchain-space: each checkpoint bound to a SPACE snapshot", langchain_space),
    ):
        print(f"{BOLD}{title}{RESET}")
        with setup(make_client(), args.template) as graph:
            print(f"  {verdict(rewind_and_step(graph))}\n")


if __name__ == "__main__":
    main()
