# Workflows

Deep Agents workflows that show what langchain-space changes.

## 1. Market research: rewind the agent, and its files

Code: [market_research.py](market_research.py)

A lead agent asks an analyst sub-agent to study Mistral AI, Anthropic and OpenAI. The analyst
searches the web and writes one notes file per company, then the lead writes a report:

```
/home/user/research/
  notes/mistral.md
  notes/anthropic.md
  notes/openai.md
  report.md
```

Then we rewind to just before the OpenAI analysis, to redo it with a different focus.

### The problem

With a normal sandbox, a rewind only moves the conversation back. The files stay as they are:
`notes/openai.md` and `report.md` are still there, written by steps that, for the agent, never
happened.

### The fix: two lines

```python
saver = SpaceCheckpointer(InMemorySaver(), client, template="code-runtime")

agent = create_deep_agent(
    ...,
    backend=saver.backend,  # instead of your sandbox
    checkpointer=saver,  # instead of InMemorySaver()
)
```

Each checkpoint is now bound to a snapshot of the sandbox. Rewinding the thread restores the files
too.

### What happens

| Step | Files in the sandbox |
| --- | --- |
| 1. Research | notes on Mistral AI, Anthropic, OpenAI + report |
| 2. Rewind to before the OpenAI analysis | notes on Mistral AI, Anthropic |
| 3. Resume with the new focus | notes on Mistral AI, Anthropic, OpenAI (new focus) + new report |

The rewind itself is plain LangGraph:

```python
before_openai = next(s for s in agent.get_state_history(config) if asks_about_openai(s))
rewound = agent.update_state(before_openai.config, {"messages": [new_request]})
agent.invoke(None, rewound)  # resume
```

### Run it

On SPACE (`SPACE_API_URL`, `SPACE_DATA_PLANE_URL`, `SPACE_API_TOKEN`, `SPACE_ECW_API_URL`,
`SPACE_ECW_API_KEY`):

```sh
uv run --with tavily-python python examples/workflows/market_research.py
```

On LangSmith sandboxes (`LANGSMITH_API_KEY`), no SPACE needed:

```sh
uv run --with tavily-python python examples/workflows/market_research.py --langsmith
```

No LLM is needed: the agents follow a fixed script, so every run is the same. Web search is real
if `TAVILY_API_KEY` is set. To use a real LLM, pass a model such as
`"anthropic:claude-sonnet-5-5"` to `create_deep_agent` instead of the scripted one.
