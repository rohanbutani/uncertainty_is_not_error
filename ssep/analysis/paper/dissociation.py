"""Study-A summary statistics independent of a particular probe implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def feature_set_statistics(
    se_features: Sequence[int],
    correctness_features: Sequence[int],
    *,
    dictionary_size: int,
) -> dict[str, float | int]:
    """Overlap plus a hypergeometric enrichment test for two frozen sets."""
    se, correctness = set(map(int, se_features)), set(map(int, correctness_features))
    if dictionary_size < 1 or any(
        value < 0 or value >= dictionary_size for value in se | correctness
    ):
        raise ValueError("feature IDs must be inside the SAE dictionary")
    if not se or not correctness:
        raise ValueError("feature sets must be nonempty")
    overlap = len(se & correctness)
    union = len(se | correctness)
    expected = len(se) * len(correctness) / dictionary_size

    # P[X >= overlap], X~Hypergeom(N, |SE|, |correctness|).  lgamma avoids
    # enormous integer combinations at 131k-feature widths.
    def log_choose(n: int, k: int) -> float:
        if k < 0 or k > n:
            return -math.inf
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    upper = min(len(se), len(correctness))
    log_denom = log_choose(dictionary_size, len(correctness))
    probabilities = [
        math.exp(
            log_choose(len(se), value)
            + log_choose(dictionary_size - len(se), len(correctness) - value)
            - log_denom
        )
        for value in range(overlap, upper + 1)
    ]
    return {
        "se_features": len(se),
        "correctness_features": len(correctness),
        "overlap": overlap,
        "union": union,
        "jaccard": overlap / union,
        "overlap_coefficient": overlap / min(len(se), len(correctness)),
        "expected_overlap_null": expected,
        "fold_enrichment": overlap / expected if expected else math.nan,
        "hypergeom_p_greater": min(1.0, math.fsum(probabilities)),
    }


def feature_stability(feature_sets: Mapping[str, Sequence[int]]) -> dict[str, object]:
    """Pairwise Jaccard stability without pretending split seeds are replications."""
    if len(feature_sets) < 2:
        raise ValueError("at least two feature sets are needed for stability")
    normalized = {name: set(map(int, values)) for name, values in feature_sets.items()}
    if any(not values for values in normalized.values()):
        raise ValueError("feature sets must be nonempty")
    names = list(normalized)
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a, b = normalized[left], normalized[right]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "jaccard": len(a & b) / len(a | b),
                    "overlap": len(a & b),
                }
            )
    return {
        "pairwise": rows,
        "mean_pairwise_jaccard": float(np.mean([row["jaccard"] for row in rows])),
        "intersection_all": sorted(set.intersection(*normalized.values())),
        "union_size": len(set.union(*normalized.values())),
    }


def paired_metric_increment(
    baseline_draws: Sequence[float], combined_draws: Sequence[float]
) -> dict[str, float | int]:
    """Summarize aligned prompt-bootstrap metric draws for a conditional increment."""
    baseline = np.asarray(baseline_draws, dtype=np.float64)
    combined = np.asarray(combined_draws, dtype=np.float64)
    if baseline.ndim != 1 or combined.ndim != 1 or len(baseline) != len(combined):
        raise ValueError("bootstrap draws must be aligned vectors")
    if len(baseline) < 100 or not np.isfinite(baseline).all() or not np.isfinite(combined).all():
        raise ValueError("at least 100 finite paired bootstrap draws are required")
    delta = combined - baseline
    low, high = np.quantile(delta, [0.025, 0.975])
    return {
        "replicates": len(delta),
        "delta_mean": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_positive": float((delta > 0).mean()),
    }
