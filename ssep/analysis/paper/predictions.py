"""Prediction provenance contract for downstream Study B/C evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ssep.analysis.paper.splits import validate_oof_predictions

PREDICTION_COLUMNS = {
    "prompt_id",
    "se_score",
    "correctness_score",
    "prediction_role",
    "predicted_fold",
}


def validate_prediction_roles(
    *,
    splits: Sequence[object],
    roles: Sequence[object],
    predicted_folds: Sequence[int],
    expected_folds: Sequence[int],
    se_scores: Sequence[float],
    correctness_scores: Sequence[float],
) -> None:
    """Enforce OOF development scores and wholly held-out test scores.

    Canonical train rows are used to tune downstream thresholds and therefore
    require OOF scores. Canonical test rows must be scored by a model fitted on
    non-test data and use role ``external_test``/fold ``-1``. Cross-fitting
    *within test* is explicitly rejected because it still uses other test
    labels during evaluation.
    """
    split_values = np.asarray([str(value) for value in splits])
    role_values = np.asarray([str(value) for value in roles])
    folds = np.asarray(predicted_folds)
    expected = np.asarray(expected_folds)
    se = np.asarray(se_scores, dtype=np.float64)
    correctness = np.asarray(correctness_scores, dtype=np.float64)
    n = len(split_values)
    if any(len(value) != n for value in (role_values, folds, expected, se, correctness)):
        raise ValueError("prediction provenance columns must be row-aligned")
    if not np.isfinite(se).all() or not np.isfinite(correctness).all():
        raise ValueError("prediction scores must be finite")
    train = split_values == "train"
    test = split_values == "test"
    discovery = split_values == "discovery"
    if not train.any() or not test.any() or not discovery.any():
        raise ValueError("prediction contract requires all canonical splits")
    if not np.all(role_values[train] == "oof_train"):
        raise ValueError("every train score must have prediction_role='oof_train'")
    validate_oof_predictions(se[train], folds[train], expected[train])
    validate_oof_predictions(correctness[train], folds[train], expected[train])
    if not np.all(role_values[test] == "external_test") or not np.all(folds[test] == -1):
        raise ValueError(
            "test scores must be external_test/fold=-1; cross-fitting within test is leakage"
        )
    allowed_discovery = np.isin(role_values[discovery], ["unused", "discovery_fit"])
    if not allowed_discovery.all():
        raise ValueError("discovery rows must be marked unused or discovery_fit")
