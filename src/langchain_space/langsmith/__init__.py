"""Run langchain-space on LangSmith sandboxes instead of SPACE.

    from langchain_space import SpaceCheckpointer
    from langchain_space.langsmith import LangSmithClient

    saver = SpaceCheckpointer(InMemorySaver(), LangSmithClient(), template="")

Kept out of the top-level package, so `import langchain_space` never needs `langsmith`.
"""

from langchain_space.langsmith.client import DEFAULT_TEMPLATE, LangSmithClient

__all__ = ["DEFAULT_TEMPLATE", "LangSmithClient"]
