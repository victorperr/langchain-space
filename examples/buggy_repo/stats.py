"""A tiny module with a bug, for the fork-and-select demo."""


def mean(values: list[float]) -> float:
    """Arithmetic mean of a non-empty list; raises ValueError on an empty one."""
    return sum(values) / (len(values) - 1)
