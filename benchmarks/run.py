"""Benchmarks: snapshot/fork latency and restore correctness after N branches.

    uv run python benchmarks/run.py                 # simulated sandboxes (checks the harness)
    uv run python benchmarks/run.py --space         # real SPACE (SPACE_* + SPACE_ECW_* env vars)
    uv run python benchmarks/run.py --space --runs 30 --branches 20

Latency results report median and p90 over `--runs` samples. Correctness results report how
many of N branches restored exactly. Simulated runs use the in-memory fake client: their
latencies only measure this library's overhead, not SPACE. Results are printed as a
Markdown table and written as JSON to benchmarks/results/.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from langgraph.checkpoint.memory import InMemorySaver

from langchain_space.backend import SpaceSandbox
from langchain_space.checkpointer import SpaceCheckpointer
from langchain_space.client import HttpSpaceClient, SpaceClient
from langchain_space.snapshots import SnapshotBinding, Snapshotter
from langchain_space.testing import FakeSpaceClient
from langchain_space.testing.graphs import build_edit_graph, find_state, thread

RESULTS_DIR = Path(__file__).parent / "results"
WORK = "/home/user/work"


@dataclass
class Latency:
    """Timing samples for one operation, in milliseconds."""

    name: str
    samples_ms: list[float] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        if len(self.samples_ms) < 2:
            return self.samples_ms[0]
        return statistics.quantiles(self.samples_ms, n=10, method="inclusive")[8]


@dataclass
class Correctness:
    """How many of N restores matched the expected state."""

    name: str
    branches: int
    correct: int
    failures: list[str] = field(default_factory=list)


@dataclass
class Report:
    """Everything one benchmark run produced, plus where it ran."""

    backend: str
    machine: dict[str, str]
    started_at: str
    latencies: list[Latency] = field(default_factory=list)
    correctness: list[Correctness] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"Backend: {self.backend}  ",
            f"Machine: {self.machine['platform']}, {self.machine['cpus']} CPUs, "
            f"Python {self.machine['python']}  ",
            f"Run at: {self.started_at}",
            "",
            "| Operation | Runs | Median (ms) | p90 (ms) |",
            "|---|---:|---:|---:|",
        ]
        lines += [
            f"| {lat.name} | {len(lat.samples_ms)} | {lat.median_ms:.2f} | {lat.p90_ms:.2f} |"
            for lat in self.latencies
        ]
        lines += ["", "| Restore check | Branches | Correct |", "|---|---:|---:|"]
        lines += [
            f"| {c.name} | {c.branches} | {c.correct}/{c.branches} |" for c in self.correctness
        ]
        return "\n".join(lines)


def machine_info() -> dict[str, str]:
    """Where the benchmark client ran (SPACE itself may run elsewhere)."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpus": str(os.cpu_count()),
        "python": platform.python_version(),
    }


@contextlib.contextmanager
def timed(latency: Latency) -> Iterator[None]:
    """Append the block's duration to `latency`."""
    started = time.perf_counter()
    yield
    latency.samples_ms.append((time.perf_counter() - started) * 1000)


def digest(sandbox: SpaceSandbox, paths: list[str]) -> str:
    """SHA-256 over the contents of `paths` (missing files hash as a marker)."""
    hasher = hashlib.sha256()
    for response in sandbox.download_files(paths):
        hasher.update(response.path.encode())
        hasher.update(response.content if response.content is not None else b"<missing>")
    return hasher.hexdigest()


def seed_files(sandbox: SpaceSandbox, count: int) -> list[str]:
    """Write `count` small files with distinct contents; return their paths."""
    files = [(f"{WORK}/file_{i:03d}.txt", f"seed {i}\n".encode() * 64) for i in range(count)]
    errors = [r.error for r in sandbox.upload_files(files) if r.error]
    if errors:
        raise RuntimeError(f"seeding failed: {errors[0]}")
    return [path for path, _ in files]


def release(client: SpaceClient, created: list[str], *sandbox_ids: str) -> None:
    """Delete sandboxes a benchmark no longer needs, so samples don't pile up running VMs.

    An id whose delete fails stays in `created` for the final cleanup.
    """
    for sandbox_id in sandbox_ids:
        with contextlib.suppress(Exception):
            client.delete_sandbox(sandbox_id)
            created.remove(sandbox_id)


def bench_snapshots(
    client: SpaceClient, template: str, runs: int, files: int, created: list[str]
) -> list[Latency]:
    """Fork, anchor (fork + suspend) and restore (fork from anchor) latency.

    Every sandbox id goes into `created` as soon as it exists, so the caller can clean up
    after a failure; each sample's sandboxes are deleted before the next sample.
    """
    fork, anchor, restore = Latency("fork live sandbox"), Latency("anchor"), Latency("restore")
    snapshotter = Snapshotter(client)
    source = SpaceSandbox.create(client, template)
    created.append(source.id)
    seed_files(source, files)
    for _ in range(runs):
        with timed(fork):
            fork_id = client.fork(source.id)
        created.append(fork_id)
        with timed(anchor):
            anchor_id = snapshotter.anchor(source.id)
        created.append(anchor_id)
        with timed(restore):
            restored_id = snapshotter.restore(SnapshotBinding(template, anchor_id))
        created.append(restored_id)
        release(client, created, fork_id, restored_id, anchor_id)
    release(client, created, source.id)
    return [fork, anchor, restore]


def bench_branch_restores(
    client: SpaceClient, template: str, branches: int, files: int, created: list[str]
) -> Correctness:
    """N forks of one anchor each rewrite every file; a restore must still match the anchor."""
    result = Correctness("anchor restore after N divergent branches", branches, 0)
    snapshotter = Snapshotter(client)
    source = SpaceSandbox.create(client, template)
    created.append(source.id)
    paths = seed_files(source, files)
    expected = digest(source, paths)
    anchor_id = snapshotter.anchor(source.id)
    created.append(anchor_id)

    for i in range(branches):
        branch = SpaceSandbox(client, snapshotter.restore(SnapshotBinding(template, anchor_id)))
        created.append(branch.id)
        branch.upload_files([(p, f"branch {i}\n".encode()) for p in paths])
        restored = SpaceSandbox(client, snapshotter.restore(SnapshotBinding(template, anchor_id)))
        created.append(restored.id)
        got = digest(restored, paths)
        branch_isolated = digest(branch, paths) != expected
        if got == expected and branch_isolated:
            result.correct += 1
        else:
            result.failures.append(f"branch {i}: restored={got[:12]} isolated={branch_isolated}")
        release(client, created, branch.id, restored.id)
    release(client, created, source.id, anchor_id)
    return result


def bench_checkpoint_branches(
    saver: SpaceCheckpointer, branches: int
) -> tuple[Correctness, Latency]:
    """N branches from one LangGraph checkpoint: each must observe that checkpoint's file."""
    result = Correctness("LangGraph branches from one checkpoint", branches, 0)
    per_branch = Latency("branch run from old checkpoint (restore + 1 step)")
    graph = build_edit_graph(saver.backend, saver)
    for target in ("base", "later"):
        graph.invoke({"target": target}, thread("bench"))
    base = find_state(graph, "bench", target="base", finished=True)
    for i in range(branches):
        with timed(per_branch):
            observed = graph.invoke({"target": f"branch-{i}"}, base.config)["observed"][-1]
        if observed == "base":
            result.correct += 1
        else:
            result.failures.append(f"branch {i} observed {observed!r}")
        saver.registry.delete_retired()  # the previous branch's sandbox is never used again
    return result, per_branch


def run_benchmarks(
    client: SpaceClient, *, backend: str, template: str, runs: int, branches: int, files: int
) -> Report:
    """Run every benchmark, cleaning up all sandboxes it created even on failure."""
    report = Report(backend, machine_info(), datetime.now(UTC).isoformat(timespec="seconds"))
    created: list[str] = []
    saver = SpaceCheckpointer(InMemorySaver(), client, template=template)
    try:
        report.latencies += bench_snapshots(client, template, runs, files, created)
        report.correctness.append(bench_branch_restores(client, template, branches, files, created))
        checkpoint_result, per_branch = bench_checkpoint_branches(saver, branches)
        report.correctness.append(checkpoint_result)
        report.latencies.append(per_branch)
    finally:
        with contextlib.suppress(Exception):
            saver.close()
        created += saver.registry.retired + saver.registry.anchors
        for sandbox_id in created:
            with contextlib.suppress(Exception):
                client.delete_sandbox(sandbox_id)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--space", action="store_true", help="benchmark a real SPACE deployment")
    parser.add_argument("--template", default="code-runtime")
    parser.add_argument("--runs", type=int, default=20, help="samples per latency benchmark")
    parser.add_argument("--branches", type=int, default=10, help="branches per correctness check")
    parser.add_argument("--files", type=int, default=50, help="files seeded in each sandbox")
    args = parser.parse_args()

    client: SpaceClient
    if args.space:
        client = HttpSpaceClient.from_env()
        backend = f"SPACE at {urlsplit(os.environ['SPACE_API_URL']).netloc}"
    else:
        client = FakeSpaceClient()
        backend = "simulated (FakeSpaceClient; latencies are library overhead, not SPACE)"

    report = run_benchmarks(
        client,
        backend=backend,
        template=args.template,
        runs=args.runs,
        branches=args.branches,
        files=args.files,
    )
    print(report.to_markdown())
    for check in report.correctness:
        for failure in check.failures:
            print(f"  FAILED {check.name}: {failure}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = report.started_at.replace(":", "").replace("-", "")
    out = RESULTS_DIR / f"{'space' if args.space else 'simulated'}-{stamp}.json"
    payload = asdict(report) | {
        "summary": {
            lat.name: {"median_ms": lat.median_ms, "p90_ms": lat.p90_ms} for lat in report.latencies
        }
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
