"""Freeze Study B/C routing thresholds using train-OOF predictions only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from ssep.analysis.paper.artifacts import atomic_json, sha256_file
from ssep.analysis.paper.data_contract import load_run_frame
from ssep.analysis.paper.offline import _load_predictions
from ssep.analysis.paper.predictions import validate_prediction_roles
from ssep.analysis.paper.registry import load_registry
from ssep.analysis.paper.splits import crossfit_folds


def balanced_accuracy_threshold(target: np.ndarray, score: np.ndarray) -> float:
    """Return the deterministic Youden-J threshold for a binary target."""
    y = np.asarray(target, dtype=np.int8)
    values = np.asarray(score, dtype=np.float64)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(values).all():
        raise ValueError("threshold calibration needs finite scores and both target classes")
    fpr, tpr, thresholds = roc_curve(y, values)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("no finite threshold candidate")
    objective = tpr - fpr
    objective[~finite] = -np.inf
    best = np.flatnonzero(objective == np.max(objective))
    # Stable tie-break: the largest threshold predicts fewer rows as wrong.
    return float(np.max(thresholds[best]))


def freeze_thresholds(
    *,
    config_path: Path,
    run_alias: str,
    predictions_path: Path,
    output_path: Path,
    correctness_target: str,
    se_high_quantile: float,
) -> Path:
    if not 0 < se_high_quantile < 1:
        raise ValueError("se_high_quantile must be strictly between zero and one")
    registry = load_registry(config_path)
    frame, _ = load_run_frame(registry, run_alias)
    predictions = _load_predictions(predictions_path)
    joined = frame.merge(predictions, on="prompt_id", how="inner", validate="one_to_one")
    if len(joined) != len(predictions):
        raise ValueError("prediction prompt IDs are missing from the registered run")
    expected_folds = crossfit_folds(
        joined.source_id,
        n_folds=int(registry.evaluation["crossfit_folds"]),
        salt=f"{registry.paper_id}:{run_alias}",
    )
    validate_prediction_roles(
        splits=joined.split,
        roles=joined.prediction_role,
        predicted_folds=joined.predicted_fold,
        expected_folds=expected_folds,
        se_scores=joined.se_score,
        correctness_scores=joined.correctness_score,
    )
    train = joined.split.to_numpy() == "train"
    if correctness_target == "judge_binary":
        if "judge_binary_label" not in joined:
            raise ValueError("judge_binary_label is unavailable")
        correct = joined.judge_binary_label.to_numpy(dtype=np.int8)
    elif correctness_target == "f1_50":
        correct = (joined.f1_squad.to_numpy(dtype=float) >= 50).astype(np.int8)
    else:
        raise ValueError("correctness target must be judge_binary or f1_50")
    correctness_threshold = balanced_accuracy_threshold(
        correct[train], joined.correctness_score.to_numpy(dtype=float)[train]
    )
    se_threshold = float(
        np.quantile(joined.se_score.to_numpy(dtype=float)[train], se_high_quantile)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    atomic_json(
        output_path,
        {
            "schema_version": 1,
            "run_alias": run_alias,
            "predictions_path": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "fit_split": "train",
            "prediction_role": "oof_train",
            "n_fit": int(train.sum()),
            "correctness_target": correctness_target,
            "correctness_strategy": "maximize_train_oof_balanced_accuracy",
            "correctness_threshold": correctness_threshold,
            "se_strategy": "train_oof_score_quantile",
            "se_high_quantile": se_high_quantile,
            "se_threshold": se_threshold,
        },
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/three_run_working_paper.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness-target", choices=("judge_binary", "f1_50"), default="f1_50")
    parser.add_argument("--se-high-quantile", type=float, default=0.5)
    args = parser.parse_args(argv)
    output = freeze_thresholds(
        config_path=Path(args.config),
        run_alias=args.run,
        predictions_path=args.predictions,
        output_path=args.output,
        correctness_target=args.correctness_target,
        se_high_quantile=args.se_high_quantile,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
