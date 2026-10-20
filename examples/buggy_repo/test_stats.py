"""Tests for stats.mean. Runs without pytest: `python3 test_stats.py` exits non-zero on failure."""

import sys

from stats import mean


def test_mean_of_two() -> None:
    assert mean([2, 4]) == 3


def test_mean_of_one() -> None:
    assert mean([5]) == 5


def test_empty_raises_value_error() -> None:
    try:
        mean([])
    except ValueError:
        return
    except Exception as error:
        raise AssertionError(f"expected ValueError, got {type(error).__name__}") from error
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [test_mean_of_two, test_mean_of_one, test_empty_raises_value_error]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as error:
            failed += 1
            print(f"FAIL {test.__name__}: {error!r}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
