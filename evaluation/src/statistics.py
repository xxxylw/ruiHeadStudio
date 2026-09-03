from __future__ import annotations

import math
from collections.abc import Sequence


def _finite_values(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("values must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("values must all be finite")
    return result


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    result = _finite_values(values)
    count = len(result)
    mean = sum(result) / count
    variance = (
        sum((value - mean) ** 2 for value in result) / (count - 1)
        if count > 1
        else 0.0
    )
    std = math.sqrt(variance)
    half_width = 1.96 * std / math.sqrt(count)
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def summarize_retrieval(ranks: Sequence[int]) -> dict[str, float | int]:
    if not ranks or any(not isinstance(rank, int) or rank < 1 for rank in ranks):
        raise ValueError("ranks must be a non-empty sequence of integers >= 1")
    count = len(ranks)
    return {
        "count": count,
        "recall_at_1": sum(rank == 1 for rank in ranks) / count,
        "mrr": sum(1 / rank for rank in ranks) / count,
        "mean_rank": sum(ranks) / count,
    }
