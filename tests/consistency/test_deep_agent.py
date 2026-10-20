"""End to end through `create_deep_agent`: the agent's own `execute` tool, then a rewind.

The model is scripted (no LLM): each turn either calls the `execute` tool or answers. The
fake sandbox simulates the two shell commands used.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_space.backend import SpaceSandbox
from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.testing import FakeSandbox, FakeSpaceClient

pytestmark = pytest.mark.consistency

NOTES = "/home/user/notes.txt"


def shell(sandbox: FakeSandbox, argv: Sequence[str]) -> tuple[str, int]:
    command = argv[2]
    if match := re.fullmatch(r"printf (\S+) > (\S+)", command):
        sandbox.write_text(match[2], match[1])
        return "", 0
    if match := re.fullmatch(r"cat (\S+)", command):
        path = sandbox.normalize(match[1])
        if path not in sandbox.files:
            return f"cat: {match[1]}: No such file or directory", 1
        return sandbox.read_text(path), 0
    return f"unsupported: {command}", 127


class ScriptedModel(GenericFakeChatModel):
    """Replays fixed AI messages; tools are accepted but ignored."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedModel:
        return self


def run(command: str, call_id: str) -> list[AIMessage]:
    """One agent turn: call `execute(command)`, then answer."""
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "execute", "args": {"command": command}, "id": call_id}],
        ),
        AIMessage(content="done"),
    ]


def script() -> Iterator[BaseMessage]:
    yield from run(f"printf v1 > {NOTES}", "c1")
    yield from run(f"printf v2 > {NOTES}", "c2")
    yield from run(f"cat {NOTES}", "c3")  # issued after rewinding to the end of run 1


def last_tool_output(state: dict[str, Any]) -> str:
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    return str(tool_messages[-1].content)


def rewind_and_read(backend: Any, checkpointer: Any) -> str:
    agent = create_deep_agent(
        model=ScriptedModel(messages=script()), backend=backend, checkpointer=checkpointer
    )
    config = {"configurable": {"thread_id": "agent"}}
    agent.invoke({"messages": [{"role": "user", "content": "write v1"}]}, config)
    agent.invoke({"messages": [{"role": "user", "content": "write v2"}]}, config)

    end_of_run_1 = next(
        s
        for s in agent.get_state_history(config)
        if s.next == () and len([m for m in s.values["messages"] if m.type == "human"]) == 1
    )
    state = agent.invoke(
        {"messages": [{"role": "user", "content": "what is in the notes?"}]}, end_of_run_1.config
    )
    return last_tool_output(state)


def test_plain_backend_agent_reads_a_file_from_the_future() -> None:
    backend = SpaceSandbox.create(FakeSpaceClient(shell), "code-runtime")
    assert "v2" in rewind_and_read(backend, InMemorySaver())


def test_space_checkpointer_agent_reads_the_rewound_file() -> None:
    saver = SpaceCheckpointer(InMemorySaver(), FakeSpaceClient(shell), template="code-runtime")
    output = rewind_and_read(saver.backend, saver)
    assert "v1" in output
    assert "v2" not in output
