"""fork_and_select on real SPACE sandboxes."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from langchain_space.backend import SpaceSandbox
from langchain_space.client import HttpSpaceClient
from langchain_space.fork import fork_and_select

pytestmark = pytest.mark.integration

FIX = "/home/user/work/fix.txt"


@pytest.fixture
def source(forking_client: HttpSpaceClient, space_template: str) -> Iterator[SpaceSandbox]:
    sandbox = SpaceSandbox.create(forking_client, space_template)
    try:
        sandbox.upload_files([(FIX, b"original")])
        yield sandbox
    finally:
        sandbox.terminate()


def write(content: bytes) -> Callable[[SpaceSandbox], None]:
    def branch(sandbox: SpaceSandbox) -> None:
        [result] = sandbox.upload_files([(FIX, content)])
        assert result.error is None, result.error

    return branch


def test_parallel_branches_are_isolated_and_winner_is_kept(source: SpaceSandbox) -> None:
    result = fork_and_select(
        source,
        [write(b"bad"), write(b"good"), write(b"worse")],
        test_command=f'test "$(cat {FIX})" = good',
    )
    try:
        assert [b.passed for b in result.branches] == [False, True, False]
        assert result.sandbox is not None
        assert result.sandbox.download_files([FIX])[0].content == b"good"
        assert source.download_files([FIX])[0].content == b"original"
    finally:
        if result.sandbox is not None:
            result.sandbox.terminate()
