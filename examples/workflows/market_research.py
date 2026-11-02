"""Rewind a market research agent: its files go back with it.

    uv run --with tavily-python python examples/workflows/market_research.py              # SPACE
    uv run --with tavily-python python examples/workflows/market_research.py --langsmith

1. Research: an analyst sub-agent writes notes on Mistral AI, Anthropic and OpenAI, then the
   lead agent writes a report.
2. Rewind to just before the OpenAI analysis and give it a new focus. The sandbox rewinds too:
   notes/openai.md and report.md are gone.
3. Resume: the agent redoes the OpenAI notes (new focus) and the report.

No LLM: the agents follow a fixed script, so every run is the same. Web search is real if
TAVILY_API_KEY is set. SPACE needs the SPACE_* variables, LangSmith needs LANGSMITH_API_KEY.
An optional last argument picks the template (SPACE: `code-runtime`; LangSmith: default image).
"""

import os
import sys
import uuid
from collections.abc import Callable
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_space import HttpSpaceClient, SpaceCheckpointer

NOTES = "/home/user/research"
COMPANIES = ["Mistral AI", "Anthropic", "OpenAI"]
CONFIG = {"configurable": {"thread_id": "market-research"}}


# --------------------------------------------------------------------------- the tool


@tool
def web_search(query: str) -> str:
    """Search the web."""
    if not os.environ.get("TAVILY_API_KEY"):
        return f"(no TAVILY_API_KEY set, so no web results for '{query}')"
    from tavily import TavilyClient

    results = TavilyClient().search(query, max_results=3)["results"]
    return "\n".join(f"- {r['title']} ({r['url']})" for r in results)


# --------------------------------------------------------------------------- the agents' scripts
# Stand-ins for an LLM. To use a real one, pass e.g. model="anthropic:claude-sonnet-5-5"
# to create_deep_agent and drop the scripts.


class ScriptedModel(BaseChatModel):
    """A chat model that answers with `script(messages)` instead of calling an LLM."""

    script: Callable[[list[BaseMessage]], AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script(messages))])


def call(tool_name: str, message_id: str | None = None, **args: Any) -> AIMessage:
    """An AI message asking to run one tool."""
    tool_call = {"name": tool_name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}
    return AIMessage(content="", tool_calls=[tool_call], id=message_id)


def ask_analyst(company: str, focus: str = "pricing", message_id: str | None = None) -> AIMessage:
    """The lead asks the analyst sub-agent to study one company."""
    notes_file = f"{NOTES}/notes/{company.split()[0].lower()}.md"
    task = f"Analyze {company}, focus: {focus}. Notes: {notes_file}"
    return call("task", message_id, subagent_type="analyst", description=task)


def lead_script(messages: list[BaseMessage]) -> AIMessage:
    """Ask the analyst about each company, then write the report from its answers."""
    answers = [m.text for m in messages if m.type == "tool"]
    if len(answers) < len(COMPANIES):
        return ask_analyst(COMPANIES[len(answers)])
    if len(answers) == len(COMPANIES):
        report = "# LLM API market\n\n" + "".join(f"- {answer}\n" for answer in answers)
        return call("write_file", file_path=f"{NOTES}/report.md", content=report)
    return AIMessage(content=f"Report written to {NOTES}/report.md")


def analyst_script(messages: list[BaseMessage]) -> AIMessage:
    """Search the web, write the results to the notes file, answer the lead."""
    task = next(m.text for m in messages if m.type == "human")
    topic, _, notes_file = task.partition(". Notes: ")
    results = [m.text for m in messages if m.type == "tool"]
    if not results:
        return call("web_search", query=topic)
    if len(results) == 1:
        return call("write_file", file_path=notes_file, content=f"# {topic}\n\n{results[0]}\n")
    return AIMessage(content=f"{topic}: see {notes_file}")


# --------------------------------------------------------------------------- the agent


def build_agent(saver: SpaceCheckpointer) -> Any:
    """A Deep Agent whose checkpoints are bound to snapshots of its sandbox."""
    analyst = {
        "name": "analyst",
        "description": "Analyzes one company and writes notes to a file",
        "system_prompt": "You analyze a company's positioning and pricing. Cite sources.",
        "tools": [web_search],
        "model": ScriptedModel(script=analyst_script),
    }
    return create_deep_agent(
        model=ScriptedModel(script=lead_script),
        system_prompt="You lead market research: delegate to the analyst, then write a report.",
        subagents=[analyst],
        backend=saver.backend,  # the sandbox, as seen from the current checkpoint
        checkpointer=saver,  # binds each checkpoint to a snapshot of the sandbox
    )


def make_client() -> tuple[Any, str]:
    """SPACE, or LangSmith with --langsmith. An optional argument picks the template."""
    args = [arg for arg in sys.argv[1:] if arg != "--langsmith"]
    if "--langsmith" in sys.argv:
        from langchain_space.langsmith import LangSmithClient

        return LangSmithClient(), args[0] if args else ""
    return HttpSpaceClient.from_env(), args[0] if args else "code-runtime"


def asks_about_openai(state: Any) -> bool:
    """True for the checkpoint where the lead is about to ask the analyst about OpenAI."""
    if state.next != ("tools",):
        return False
    [tool_call] = state.values["messages"][-1].tool_calls
    return str(tool_call["args"].get("description", "")).startswith("Analyze OpenAI")


def show_files(saver: SpaceCheckpointer) -> None:
    """Print the files in the thread's sandbox."""
    found = saver.sandbox("market-research").glob("**/*.md", path=NOTES).matches or []
    for path in sorted(f["path"].removeprefix(NOTES + "/") for f in found):
        print(f"  {path}")


# --------------------------------------------------------------------------- the demo


def main() -> None:
    client, template = make_client()
    saver = SpaceCheckpointer(InMemorySaver(), client, template=template)
    agent = build_agent(saver)
    try:
        print("1. Research")
        task = "Compare Mistral AI, Anthropic and OpenAI on the LLM API market."
        agent.invoke({"messages": [{"role": "user", "content": task}]}, CONFIG)
        show_files(saver)

        print("\n2. Rewind to just before the OpenAI analysis, with a new focus")
        before_openai = next(s for s in agent.get_state_history(CONFIG) if asks_about_openai(s))
        old_request = before_openai.values["messages"][-1]
        new_request = ask_analyst("OpenAI", "EU data residency", message_id=old_request.id)
        rewound = agent.update_state(before_openai.config, {"messages": [new_request]})
        show_files(saver)  # notes/openai.md and report.md are gone

        print("\n3. Resume")
        agent.invoke(None, rewound)
        show_files(saver)
        [notes] = saver.sandbox("market-research").download_files([f"{NOTES}/notes/openai.md"])
        first_line = (notes.content or b"").decode().splitlines()[0]
        print(f"\n  notes/openai.md now starts with: {first_line}")
    finally:
        saver.close()  # delete the sandboxes
        for anchor_id in saver.registry.anchors:  # and the snapshots, since we are done
            client.delete_anchor(anchor_id)


if __name__ == "__main__":
    main()
