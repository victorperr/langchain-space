"""Fork-and-select demo: try three fixes for a bug in parallel sandboxes, keep the one that passes.

    uv run python examples/fork_and_select_demo.py           # simulated sandboxes, runs anywhere
    uv run python examples/fork_and_select_demo.py --space   # real SPACE (SPACE_* env vars)

In simulated mode, sandboxes are `FakeSpaceClient` filesystems and commands run with the
local Python in a temporary copy of the sandbox's work directory.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from langchain_space.backend import SpaceSandbox
from langchain_space.client import HttpSpaceClient, SpaceClient
from langchain_space.fork import fork_and_select
from langchain_space.testing import FakeSandbox, FakeSpaceClient

REPO = Path(__file__).parent / "buggy_repo"
WORK = "/home/user/work"
TEST_COMMAND = f"cd {WORK} && python3 test_stats.py"

CANDIDATE_FIXES = {
    "divide by len": '''def mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list; raises ValueError on an empty one."""
    return sum(values) / len(values)
''',
    "guard empty only": '''def mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list; raises ValueError on an empty one."""
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / (len(values) - 1)
''',
    "guard + divide by len": '''def mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list; raises ValueError on an empty one."""
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)
''',
}


def local_exec(sandbox: FakeSandbox, argv: Sequence[str]) -> tuple[str, int]:
    """Run `sh -c` commands against a temp copy of the fake sandbox's work directory."""
    with tempfile.TemporaryDirectory() as tmp:
        for path, content in sandbox.files.items():
            if path.startswith(WORK + "/"):
                target = Path(tmp, path[len(WORK) + 1 :])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        command = argv[2].replace(WORK, tmp)
        command = re.sub(r"\bpython3\b", lambda _: f'"{sys.executable}"', command)
        done = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
        return done.stdout + done.stderr, done.returncode


def apply_fix(source: str) -> Callable[[SpaceSandbox], str]:
    """A branch that replaces stats.py with `source`."""

    def branch(sandbox: SpaceSandbox) -> str:
        [result] = sandbox.upload_files([(f"{WORK}/stats.py", source.encode())])
        if result.error:
            raise RuntimeError(result.error)
        return source

    return branch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", action="store_true", help="use a real SPACE deployment")
    parser.add_argument("--template", default="code-runtime")
    args = parser.parse_args()

    client: SpaceClient = HttpSpaceClient.from_env() if args.space else FakeSpaceClient(local_exec)
    print(f"Creating a sandbox ({'SPACE' if args.space else 'simulated'})...")
    sandbox = SpaceSandbox.create(client, args.template)
    winner = None
    try:
        sandbox.upload_files([(f"{WORK}/{p.name}", p.read_bytes()) for p in REPO.glob("*.py")])
        before = sandbox.execute(TEST_COMMAND)
        print(f"\nBefore: exit {before.exit_code}\n{before.output.strip()}\n")

        names = list(CANDIDATE_FIXES)
        started = time.perf_counter()
        result = fork_and_select(
            sandbox, [apply_fix(CANDIDATE_FIXES[n]) for n in names], test_command=TEST_COMMAND
        )
        elapsed = time.perf_counter() - started

        print(f"Tried {len(names)} fixes in parallel forks ({elapsed:.2f}s):")
        for branch in result.branches:
            verdict = "PASS" if branch.passed else "fail"
            summary = (branch.error or branch.test_output).strip().splitlines()[-1]
            print(f"  [{verdict}] {names[branch.index]:<24} {summary}")
        winner = result.sandbox
        if result.winner is None:
            print("\nNo fix passed.")
        else:
            print(f"\nKept '{names[result.winner.index]}' in sandbox {result.winner.sandbox_id}.")
            after = sandbox.execute(TEST_COMMAND)
            print(f"The original sandbox is untouched: still exit {after.exit_code}.")
    finally:
        sandbox.terminate()
        if winner is not None:
            winner.terminate()


if __name__ == "__main__":
    main()
