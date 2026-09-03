"""Leakage-safe offline fix-rate and routing policy evaluation (Studies B/C)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    budget: int
    n: int
    emitted: int
    abstained: int
    extra_generations: int
    total_generations: int
    expected_correct_answers: float
    expected_accuracy: float
    coverage: float
    selective_accuracy: float
    accuracy_per_extra_generation: float
    correct_answers_per_generation: float


def _aligned_inputs(
    greedy_correct: np.ndarray,
    sample_correct: np.ndarray,
    predicted_correctness: np.ndarray,
    predicted_se: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    greedy = np.asarray(greedy_correct, dtype=bool)
    samples = np.asarray(sample_correct, dtype=bool)
    p_correct = np.asarray(predicted_correctness, dtype=np.float64)
    p_se = np.asarray(predicted_se, dtype=np.float64)
    if greedy.ndim != 1 or samples.ndim != 2 or p_correct.ndim != 1 or p_se.ndim != 1:
        raise ValueError("policy inputs must be vectors plus a (rows, samples) matrix")
    if not (len(greedy) == len(samples) == len(p_correct) == len(p_se)):
        raise ValueError("policy inputs are not row-aligned")
    if not np.isfinite(p_correct).all() or not np.isfinite(p_se).all():
        raise ValueError("policy predictions must be finite")
    return greedy, samples, p_correct, p_se


def fix_rate_table(
    greedy_correct: np.ndarray,
    sample_correct: np.ndarray,
    predicted_correctness: np.ndarray,
    predicted_se: np.ndarray,
    *,
    correctness_threshold: float,
    se_threshold: float,
    budgets: tuple[int, ...] = (1, 2, 5, 10),
) -> list[dict[str, float | int | str]]:
    """Study B table using frozen, preferably out-of-fold, probe scores.

    Fix@k means at least one correct answer among the first k banked resamples.
    Random-resample accuracy is the mean correctness of those k samples; it is
    the expected accuracy when one is selected uniformly without an oracle.
    """
    greedy, samples, p_correct, p_se = _aligned_inputs(
        greedy_correct, sample_correct, predicted_correctness, predicted_se
    )
    predicted_wrong = p_correct < correctness_threshold
    high_se = p_se >= se_threshold
    quadrants = {
        "predicted_high_se_error": predicted_wrong & high_se,
        "predicted_low_se_error": predicted_wrong & ~high_se,
        "predicted_correct_high_se": ~predicted_wrong & high_se,
        "predicted_correct_low_se": ~predicted_wrong & ~high_se,
    }
    rows: list[dict[str, float | int | str]] = []
    for budget in budgets:
        if not 1 <= budget <= samples.shape[1]:
            raise ValueError(f"resample budget {budget} outside [1, {samples.shape[1]}]")
        local_samples = samples[:, :budget]
        for quadrant, mask in quadrants.items():
            n = int(mask.sum())
            rows.append(
                {
                    "quadrant": quadrant,
                    "budget": budget,
                    "n": n,
                    "greedy_accuracy": float(greedy[mask].mean()) if n else np.nan,
                    "random_resample_accuracy": float(local_samples[mask].mean()) if n else np.nan,
                    "fix_at_k": float(local_samples[mask].any(axis=1).mean()) if n else np.nan,
                    "zero_correct_at_k": float((~local_samples[mask].any(axis=1)).mean())
                    if n
                    else np.nan,
                }
            )
    return rows


def evaluate_policy(
    policy: str,
    greedy_correct: np.ndarray,
    sample_correct: np.ndarray,
    predicted_correctness: np.ndarray,
    predicted_se: np.ndarray,
    *,
    budget: int,
    correctness_threshold: float,
    se_threshold: float,
) -> PolicyResult:
    """Study C evaluation without oracle sample selection.

    A resampling policy uses the expected correctness of a uniformly selected
    banked sample.  ``joint_low_se_error_abstain`` abstains instead of assigning
    unearned correctness to a retrieval system that has not been implemented.
    """
    greedy, samples, p_correct, p_se = _aligned_inputs(
        greedy_correct, sample_correct, predicted_correctness, predicted_se
    )
    if not 1 <= budget <= samples.shape[1]:
        raise ValueError(f"budget must be in [1, {samples.shape[1]}]")
    predicted_wrong = p_correct < correctness_threshold
    high_se = p_se >= se_threshold
    if policy == "never_resample":
        resample = np.zeros(len(greedy), dtype=bool)
        abstain = np.zeros(len(greedy), dtype=bool)
    elif policy == "always_resample":
        resample = np.ones(len(greedy), dtype=bool)
        abstain = np.zeros(len(greedy), dtype=bool)
    elif policy == "correctness_trigger":
        resample = predicted_wrong
        abstain = np.zeros(len(greedy), dtype=bool)
    elif policy == "token_entropy_trigger":
        resample = high_se
        abstain = np.zeros(len(greedy), dtype=bool)
    elif policy == "joint_high_se_error":
        resample = predicted_wrong & high_se
        abstain = np.zeros(len(greedy), dtype=bool)
    elif policy == "joint_low_se_error_abstain":
        resample = predicted_wrong & high_se
        abstain = predicted_wrong & ~high_se
    else:
        raise ValueError(f"unknown routing policy {policy!r}")
    outcome = greedy.astype(np.float64)
    outcome[resample] = samples[resample, :budget].mean(axis=1)
    emitted = ~abstain
    selective = float(outcome[emitted].mean()) if emitted.any() else np.nan
    extra = int(resample.sum()) * budget
    baseline_correct = float(greedy[emitted].sum())
    gain = float(outcome[emitted].sum()) - baseline_correct
    expected_correct = float(outcome[emitted].sum())
    total_generations = len(greedy) + extra
    return PolicyResult(
        policy=policy,
        budget=budget,
        n=len(greedy),
        emitted=int(emitted.sum()),
        abstained=int(abstain.sum()),
        extra_generations=extra,
        total_generations=total_generations,
        expected_correct_answers=expected_correct,
        expected_accuracy=float(expected_correct / len(greedy)),
        coverage=float(emitted.mean()),
        selective_accuracy=selective,
        accuracy_per_extra_generation=(gain / extra if extra else 0.0),
        correct_answers_per_generation=expected_correct / total_generations,
    )
