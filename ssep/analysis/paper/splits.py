"""Stable source-group folds and leakage checks for paper analyses."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np

CANONICAL_SPLITS = ("discovery", "train", "test")


def stable_group_fold(group: str, *, n_folds: int, salt: str) -> int:
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    payload = f"{salt}\0{group}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % n_folds


def crossfit_folds(groups: Sequence[object], *, n_folds: int, salt: str) -> np.ndarray:
    """Assign identical source groups to identical deterministic folds."""
    normalized = [str(group) for group in groups]
    if any(not group for group in normalized):
        raise ValueError("crossfit groups must be nonempty")
    return np.asarray(
        [stable_group_fold(group, n_folds=n_folds, salt=salt) for group in normalized],
        dtype=np.int16,
    )


def validate_split_groups(splits: Sequence[object], groups: Sequence[object]) -> dict[str, int]:
    """Reject source-question leakage or incomplete canonical split labels."""
    if len(splits) != len(groups):
        raise ValueError("splits and groups must have equal length")
    split_names = [str(value) for value in splits]
    unknown = sorted(set(split_names) - set(CANONICAL_SPLITS))
    if unknown:
        raise ValueError(f"unknown split labels: {unknown}")
    seen: dict[str, str] = {}
    counts = {name: 0 for name in CANONICAL_SPLITS}
    for split, group_value in zip(split_names, groups, strict=True):
        group = str(group_value)
        if not group:
            raise ValueError("source groups must be nonempty")
        previous = seen.setdefault(group, split)
        if previous != split:
            raise ValueError(
                f"source group {group!r} crosses canonical splits {previous!r}/{split!r}"
            )
        counts[split] += 1
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"canonical splits have no rows: {missing}")
    return counts


def validate_oof_predictions(
    predictions: Sequence[float],
    predicted_folds: Sequence[int],
    expected_folds: Sequence[int],
) -> None:
    """Ensure downstream policy scores came from each row's held-out fold."""
    p = np.asarray(predictions, dtype=np.float64)
    got = np.asarray(predicted_folds)
    expected = np.asarray(expected_folds)
    if not (p.ndim == got.ndim == expected.ndim == 1 and len(p) == len(got) == len(expected)):
        raise ValueError("OOF predictions/folds must be aligned vectors")
    if not np.isfinite(p).all():
        raise ValueError("OOF predictions must be finite")
    if not np.array_equal(got, expected):
        bad = int(np.flatnonzero(got != expected)[0])
        raise ValueError(f"prediction at row {bad} was not produced by its held-out fold")


def groups_are_disjoint(*group_sets: Iterable[object]) -> bool:
    normalized = [{str(value) for value in values} for values in group_sets]
    return all(
        normalized[i].isdisjoint(normalized[j])
        for i in range(len(normalized))
        for j in range(i + 1, len(normalized))
    )
