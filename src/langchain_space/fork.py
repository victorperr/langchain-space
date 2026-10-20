"""Fork-and-select: try N alternatives in parallel sandboxes, keep the one whose tests pass.

SPACE forks are cheap copies of a sandbox's disk, so exploring several fixes costs N forks
instead of N full environment setups:

```python
result = fork_and_select(sandbox, [apply_fix_a, apply_fix_b, apply_fix_c],
                         test_command="cd /home/user/work && python -m pytest -q")
if result.sandbox is not None:
    ...  # continue in the winning sandbox
```

Inside a graph run with `SpaceCheckpointer`, pass `saver.backend.current()` and hand the
winner to `saver.backend.adopt(result.sandbox)` so the thread continues in it (and the next
checkpoint snapshots it).
"""

from __future__ import annotations

import contextlib
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

from langchain_space.backend import SpaceSandbox

T = TypeVar("T")

Branch = Callable[[SpaceSandbox], T]
"""Applies one alternative to its own forked sandbox; its return value is kept."""


@dataclass(frozen=True)
class BranchResult(Generic[T]):
    """What happened in one branch."""

    index: int
    sandbox_id: str | None
    """The branch's sandbox (deleted afterwards unless it won or `keep_losers=True`)."""
    passed: bool
    value: T | None = None
    test_output: str = ""
    exit_code: int | None = None
    error: str | None = None
    """Set when forking, the branch itself, or the test command raised."""


@dataclass(frozen=True)
class ForkResult(Generic[T]):
    """All branch outcomes and the winner (lowest index whose tests passed), if any."""

    branches: list[BranchResult[T]]
    winner: BranchResult[T] | None
    sandbox: SpaceSandbox | None
    """The winning branch's sandbox, still running; the caller owns it."""


def fork_and_select(
    sandbox: SpaceSandbox,
    branches: Sequence[Branch[T]],
    *,
    test_command: str,
    test_timeout: int | None = None,
    max_workers: int | None = None,
    keep_losers: bool = False,
) -> ForkResult[T]:
    """Fork `sandbox` once per branch, run branches concurrently, keep the first that passes.

    The source sandbox is never modified. Each branch gets its own fork, applies its change,
    then `test_command` runs in that fork; exit code 0 means pass. The winner is the
    passing branch with the lowest index, so the result does not depend on timing.

    Args:
        sandbox: The state every branch starts from.
        branches: One callable per alternative.
        test_command: Shell command deciding pass (exit 0) or fail.
        test_timeout: Seconds for the test command (`None`: the client's default).
        max_workers: Concurrency cap (default: all branches at once).
        keep_losers: Keep failed branches' sandboxes for inspection instead of deleting them.
    """
    if not branches:
        raise ValueError("fork_and_select needs at least one branch")
    client = sandbox.client

    def run(index: int, branch: Branch[T]) -> BranchResult[T]:
        try:
            child_id = client.fork(sandbox.id)
        except Exception:
            return BranchResult(index, None, passed=False, error=traceback.format_exc())
        child = SpaceSandbox(client, child_id, owns_sandbox=True)
        try:
            value = branch(child)
            test = child.execute(test_command, timeout=test_timeout)
        except Exception:
            return BranchResult(index, child_id, passed=False, error=traceback.format_exc())
        return BranchResult(
            index,
            child_id,
            passed=test.exit_code == 0,
            value=value,
            test_output=test.output,
            exit_code=test.exit_code,
        )

    with ThreadPoolExecutor(max_workers=max_workers or len(branches)) as pool:
        results = list(pool.map(run, range(len(branches)), branches))

    winner = next((r for r in results if r.passed), None)
    if not keep_losers:
        for result in results:
            if result is not winner and result.sandbox_id is not None:
                # A failed cleanup must not hide the outcome.
                with contextlib.suppress(Exception):
                    client.delete_sandbox(result.sandbox_id)
    winning_sandbox = (
        SpaceSandbox(client, winner.sandbox_id, owns_sandbox=True)
        if winner is not None and winner.sandbox_id is not None
        else None
    )
    return ForkResult(results, winner, winning_sandbox)
