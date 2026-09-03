"""Run leakage-checked offline Study B/C evaluation from frozen OOF probe scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ssep.analysis.paper.artifacts import (
    atomic_json,
    create_experiment_manifest,
    mark_test_read,
    sha256_file,
)
from ssep.analysis.paper.bootstrap import cluster_bootstrap_indices, percentile_interval
from ssep.analysis.paper.data_contract import load_run_frame, sample_correctness
from ssep.analysis.paper.policies import evaluate_policy, fix_rate_table
from ssep.analysis.paper.predictions import PREDICTION_COLUMNS, validate_prediction_roles
from ssep.analysis.paper.registry import load_registry
from ssep.analysis.paper.splits import crossfit_folds


def _load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("prediction file must be .parquet or .csv")
    missing = PREDICTION_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"prediction file is missing {sorted(missing)}")
    if frame.prompt_id.isna().any() or frame.prompt_id.duplicated().any():
        raise ValueError("prediction prompt_id must be complete and unique")
    return frame


def _greedy_correct(frame: pd.DataFrame, target: str) -> np.ndarray:
    if target == "judge_binary":
        if "judge_binary_label" not in frame:
            raise ValueError("judge_binary_label is unavailable")
        return frame.judge_binary_label.to_numpy(dtype=bool)
    if target == "f1_50":
        return frame.f1_squad.to_numpy(dtype=float) >= 50.0
    raise ValueError("correctness target must be judge_binary or f1_50")


def _load_thresholds(path: Path, *, predictions_path: Path, run_alias: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "run_alias",
        "predictions_sha256",
        "fit_split",
        "prediction_role",
        "correctness_target",
        "correctness_threshold",
        "se_threshold",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"threshold artifact is missing {sorted(missing)}")
    if payload["schema_version"] != 1 or payload["run_alias"] != run_alias:
        raise ValueError("threshold artifact schema/run does not match this evaluation")
    if payload["fit_split"] != "train" or payload["prediction_role"] != "oof_train":
        raise ValueError("thresholds must be fit on train-OOF predictions")
    if payload["predictions_sha256"] != sha256_file(predictions_path):
        raise ValueError("threshold artifact was frozen for a different prediction file")
    return payload


def _attach_intervals(
    point_rows: list[dict],
    bootstrap_rows: list[list[dict]],
    *,
    key_columns: tuple[str, ...],
    metric_columns: tuple[str, ...],
) -> list[dict]:
    """Attach percentile intervals from aligned bootstrap tables."""
    draws: dict[tuple, dict[str, list[float]]] = {}
    for table in bootstrap_rows:
        for row in table:
            key = tuple(row[column] for column in key_columns)
            bucket = draws.setdefault(key, {metric: [] for metric in metric_columns})
            for metric in metric_columns:
                bucket[metric].append(float(row[metric]))
    output = []
    for original in point_rows:
        row = dict(original)
        key = tuple(row[column] for column in key_columns)
        for metric in metric_columns:
            low, high, valid = percentile_interval(draws[key][metric])
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_bootstrap_valid"] = valid
        output.append(row)
    return output


def run_offline(
    *,
    config_path: Path,
    run_alias: str,
    predictions_path: Path,
    thresholds_path: Path,
    output_dir: Path,
) -> Path:
    registry = load_registry(config_path)
    predictions = _load_predictions(predictions_path)
    thresholds = _load_thresholds(
        thresholds_path, predictions_path=predictions_path, run_alias=run_alias
    )
    correctness_target = str(thresholds["correctness_target"])
    correctness_threshold = float(thresholds["correctness_threshold"])
    se_threshold = float(thresholds["se_threshold"])
    n_folds = int(registry.evaluation["crossfit_folds"])
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = create_experiment_manifest(
        experiment_id=output_dir.name,
        study="B",
        config_path=config_path,
        input_paths=[
            predictions_path,
            thresholds_path,
            registry.metadata_path(run_alias),
            registry.tables_dir(run_alias) / "prompts.parquet",
            registry.tables_dir(run_alias) / "labels.parquet",
        ],
        split_protocol={
            "canonical_evaluation_split": "test",
            "group_column": "source_id",
            "crossfit_folds": n_folds,
            "prediction_fold_salt": f"{registry.paper_id}:{run_alias}",
            "test_prediction_role": "external_test",
            "correctness_threshold": correctness_threshold,
            "se_threshold": se_threshold,
            "eligible_rows": len(predictions),
        },
    )
    atomic_json(output_dir / "manifest.pre_test.json", manifest)

    frame, _ = load_run_frame(registry, run_alias)
    joined = frame.merge(predictions, on="prompt_id", how="inner", validate="one_to_one")
    if (
        len(joined) != len(predictions)
        or joined[list(PREDICTION_COLUMNS - {"prompt_id"})].isna().any().any()
    ):
        raise ValueError("prediction prompt IDs are missing from the registered run")
    if not set(predictions.prompt_id).issubset(set(frame.prompt_id)):
        raise ValueError("prediction file contains prompt IDs outside the registered run")
    expected_folds = crossfit_folds(
        joined.source_id,
        n_folds=n_folds,
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
    test = joined.split.to_numpy() == "test"
    if not test.any():
        raise ValueError("run has no canonical test rows")
    samples, sample_channel = sample_correctness(joined)
    greedy = _greedy_correct(joined, correctness_target)
    budgets = tuple(
        int(value) for value in registry.raw["studies"]["b_fix_rate"]["resample_budgets"]
    )

    test_greedy = greedy[test]
    test_samples = samples[test]
    test_correctness_score = joined.correctness_score.to_numpy()[test]
    test_se_score = joined.se_score.to_numpy()[test]
    rows = fix_rate_table(
        test_greedy,
        test_samples,
        test_correctness_score,
        test_se_score,
        correctness_threshold=correctness_threshold,
        se_threshold=se_threshold,
        budgets=budgets,
    )
    policies = registry.raw["studies"]["c_routing"]["policies"]
    policy_rows = []
    for budget in registry.raw["studies"]["c_routing"]["resample_budgets"]:
        for policy in policies:
            result = evaluate_policy(
                policy,
                test_greedy,
                test_samples,
                test_correctness_score,
                test_se_score,
                budget=int(budget),
                correctness_threshold=correctness_threshold,
                se_threshold=se_threshold,
            )
            policy_rows.append(asdict(result))

    replicates = int(registry.evaluation["bootstrap_replicates"])
    bootstrap_fix = []
    bootstrap_policy = []
    for indices in cluster_bootstrap_indices(
        joined.source_id.to_numpy()[test],
        replicates=replicates,
        seed=int(registry.evaluation["bootstrap_seed"]),
    ):
        bootstrap_fix.append(
            fix_rate_table(
                test_greedy[indices],
                test_samples[indices],
                test_correctness_score[indices],
                test_se_score[indices],
                correctness_threshold=correctness_threshold,
                se_threshold=se_threshold,
                budgets=budgets,
            )
        )
        replicate_policies = []
        for budget in registry.raw["studies"]["c_routing"]["resample_budgets"]:
            for policy in policies:
                replicate_policies.append(
                    asdict(
                        evaluate_policy(
                            policy,
                            test_greedy[indices],
                            test_samples[indices],
                            test_correctness_score[indices],
                            test_se_score[indices],
                            budget=int(budget),
                            correctness_threshold=correctness_threshold,
                            se_threshold=se_threshold,
                        )
                    )
                )
        bootstrap_policy.append(replicate_policies)
    rows = _attach_intervals(
        rows,
        bootstrap_fix,
        key_columns=("quadrant", "budget"),
        metric_columns=("greedy_accuracy", "random_resample_accuracy", "fix_at_k"),
    )
    policy_rows = _attach_intervals(
        policy_rows,
        bootstrap_policy,
        key_columns=("policy", "budget"),
        metric_columns=(
            "expected_accuracy",
            "coverage",
            "selective_accuracy",
            "accuracy_per_extra_generation",
            "correct_answers_per_generation",
        ),
    )
    pd.DataFrame(rows).to_csv(output_dir / "fix_rate.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(output_dir / "routing_policies.csv", index=False)
    atomic_json(
        output_dir / "run_summary.json",
        {
            "run_alias": run_alias,
            "correctness_target": correctness_target,
            "sample_correctness_channel": sample_channel,
            "test_rows": int(test.sum()),
            "bootstrap_replicates": replicates,
            "bootstrap_unit": "source_id",
            "important_limit": (
                "banked-sample policy simulation; not a live adaptive intervention and "
                "not causal evidence about SAE features"
            ),
        },
    )
    atomic_json(output_dir / "manifest.json", mark_test_read(manifest))
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/three_run_working_paper.yaml")
    parser.add_argument("--run", required=True, help="registry alias")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    path = run_offline(
        config_path=Path(args.config),
        run_alias=args.run,
        predictions_path=args.predictions,
        thresholds_path=args.thresholds,
        output_dir=args.output,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
