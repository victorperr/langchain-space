"""The benchmark harness: it runs, cleans up, and actually detects broken restores."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from langchain_space.testing import FakeSpaceClient

RUN_PY = Path(__file__).parents[2] / "benchmarks" / "run.py"


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmarks_run", RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_reports_full_correctness_and_cleans_up(bench: ModuleType) -> None:
    fake = FakeSpaceClient()
    report = bench.run_benchmarks(fake, backend="fake", template="t", runs=3, branches=4, files=5)

    assert {lat.name for lat in report.latencies} >= {"fork live sandbox", "anchor", "restore"}
    assert all(len(lat.samples_ms) > 0 for lat in report.latencies)
    assert [(c.correct, c.branches) for c in report.correctness] == [(4, 4), (4, 4)]
    assert fake.live_sandboxes == []
    assert "| Operation | Runs | Median (ms) | p90 (ms) |" in report.to_markdown()


class SharedStateFork(FakeSpaceClient):
    """A broken SPACE: forks share the source's filesystem instead of copying it."""

    def fork(self, sandbox_id: str, *, lifetime_seconds: int | None = None) -> str:
        child_id = super().fork(sandbox_id, lifetime_seconds=lifetime_seconds)
        source, child = self.sandbox(sandbox_id), self.sandbox(child_id)
        child.files = source.files
        return child_id


def test_detects_restores_that_do_not_isolate_branches(bench: ModuleType) -> None:
    report = bench.run_benchmarks(
        SharedStateFork(), backend="broken", template="t", runs=1, branches=3, files=2
    )
    anchor_check = report.correctness[0]
    assert anchor_check.correct == 0
    assert anchor_check.failures
