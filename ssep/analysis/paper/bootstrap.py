"""Group-respecting nonparametric uncertainty intervals for paper metrics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


def cluster_bootstrap_indices(
    groups: Sequence[object], *, replicates: int, seed: int
) -> Iterator[np.ndarray]:
    """Yield row indices after resampling source groups with replacement."""
    values = np.asarray(groups)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("groups must be a nonempty vector")
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    unique, inverse = np.unique(values.astype(str), return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled = rng.integers(0, len(unique), size=len(unique))
        yield np.concatenate([members[index] for index in sampled])


def percentile_interval(values: Sequence[float]) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 50:
        return np.nan, np.nan, int(len(finite))
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(len(finite))
