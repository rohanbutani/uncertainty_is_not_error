"""Nested, group-safe probe runner for aligned dense or SAE feature matrices."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ssep.analysis.paper.artifacts import (
    atomic_json,
    create_experiment_manifest,
    mark_test_read,
)
from ssep.analysis.paper.bootstrap import cluster_bootstrap_indices, percentile_interval
from ssep.analysis.paper.data_contract import load_run_frame
from ssep.analysis.paper.dissociation import feature_set_statistics
from ssep.analysis.paper.registry import load_registry
from ssep.analysis.paper.splits import crossfit_folds, validate_split_groups
from ssep.feature_selection import select_sae_features

TargetKind = Literal["binary", "continuous"]


@dataclass(frozen=True)
class Candidate:
    selector: str
    k: int
    regularization: float
    fold_scores: tuple[float, ...]
    mean_score: float


def _metric(kind: TargetKind, target: np.ndarray, score: np.ndarray) -> float:
    if kind == "binary":
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(target, score))
    from sklearn.metrics import r2_score

    return float(r2_score(target, score))


def _metric_bundle(kind: TargetKind, target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if kind == "binary":
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )

        clipped = np.clip(score, 1e-7, 1 - 1e-7)
        return {
            "auroc": float(roc_auc_score(target, score)),
            "auprc": float(average_precision_score(target, score)),
            "brier": float(brier_score_loss(target, score)),
            "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        }
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    return {
        "spearman": float(spearmanr(target, score).statistic),
        "r2": float(r2_score(target, score)),
        "mae": float(mean_absolute_error(target, score)),
    }


def _bootstrap_probe_metrics(
    *,
    se_target: np.ndarray,
    correctness_target: np.ndarray,
    scores: dict[str, np.ndarray],
    groups: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    kinds: dict[str, TargetKind] = {
        "se_direct": "continuous",
        "se_from_correctness_features": "continuous",
        "correctness_direct": "binary",
        "correctness_from_se_features": "binary",
    }
    targets = {"continuous": se_target, "binary": correctness_target}
    draws: dict[str, dict[str, list[float]]] = {name: {} for name in scores}
    for indices in cluster_bootstrap_indices(groups, replicates=replicates, seed=seed):
        if np.unique(correctness_target[indices]).size < 2:
            continue
        for name, values in scores.items():
            metrics = _metric_bundle(kinds[name], targets[kinds[name]][indices], values[indices])
            for metric, value in metrics.items():
                draws[name].setdefault(metric, []).append(value)
    summary: dict[str, dict] = {}
    for name, metric_draws in draws.items():
        summary[name] = {}
        point = _metric_bundle(kinds[name], targets[kinds[name]], scores[name])
        for metric, values in metric_draws.items():
            low, high, valid = percentile_interval(values)
            summary[name][metric] = {
                "point": point[metric],
                "ci_low": low,
                "ci_high": high,
                "valid_replicates": valid,
            }
    for delta_name, direct, cross, metric in (
        ("se_cross_minus_direct_r2", "se_direct", "se_from_correctness_features", "r2"),
        (
            "correctness_cross_minus_direct_auroc",
            "correctness_direct",
            "correctness_from_se_features",
            "auroc",
        ),
    ):
        values = np.asarray(draws[cross][metric]) - np.asarray(draws[direct][metric])
        low, high, valid = percentile_interval(values)
        summary[delta_name] = {
            "point": summary[cross][metric]["point"] - summary[direct][metric]["point"],
            "ci_low": low,
            "ci_high": high,
            "valid_replicates": valid,
        }
    return summary


def _fit(kind: TargetKind, x: np.ndarray, target: np.ndarray, regularization: float, seed: int):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if regularization <= 0:
        raise ValueError("regularization values must be positive")
    if kind == "binary":
        estimator = LogisticRegression(
            C=regularization,
            max_iter=5000,
            random_state=seed,
            solver="liblinear",
        )
    else:
        estimator = Ridge(alpha=regularization)
    return make_pipeline(StandardScaler(), estimator).fit(x, target)


def _predict(kind: TargetKind, model, x: np.ndarray) -> np.ndarray:
    if kind == "binary":
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def _rankings(
    x_discovery: np.ndarray,
    target_discovery: np.ndarray,
    *,
    kind: TargetKind,
    selectors: tuple[str, ...],
    max_k: int,
    seed: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for selector in selectors:
        if selector == "identity":
            if max_k != x_discovery.shape[1]:
                raise ValueError("identity selector requires k equal to the full feature width")
            out[selector] = np.arange(x_discovery.shape[1])
        elif selector.startswith("forced_first_"):
            base_selector = selector.removeprefix("forced_first_")
            if x_discovery.shape[1] < 2 or max_k < 2:
                raise ValueError("forced_first selectors need at least two features and k >= 2")
            ranked = select_sae_features(
                x_discovery[:, 1:],
                target_discovery,
                max_k - 1,
                method=base_selector,  # type: ignore[arg-type]
                target_kind=kind,
                random_state=seed,
            ).feature_ids
            out[selector] = np.concatenate(([0], ranked + 1))
        else:
            out[selector] = select_sae_features(
                x_discovery,
                target_discovery,
                max_k,
                method=selector,  # type: ignore[arg-type]
                target_kind=kind,
                random_state=seed,
            ).feature_ids
    return out


def _select_candidate(
    x: np.ndarray,
    y: np.ndarray,
    eligible: np.ndarray,
    folds: np.ndarray,
    rankings: dict[str, np.ndarray],
    *,
    kind: TargetKind,
    k_values: tuple[int, ...],
    regularization_values: tuple[float, ...],
    seed: int,
) -> tuple[Candidate, list[Candidate]]:
    """Choose a candidate from held-out folds using only eligible rows."""
    candidates: list[Candidate] = []
    validation_folds = sorted(set(folds[eligible].tolist()))
    if len(validation_folds) < 2:
        raise ValueError("candidate selection needs at least two nonempty folds")
    for selector, ranking in rankings.items():
        local_ks = (x.shape[1],) if selector == "identity" else k_values
        for k in local_ks:
            feature_ids = ranking[:k]
            for regularization in regularization_values:
                fold_scores = []
                for fold in validation_folds:
                    fit_mask = eligible & (folds != fold)
                    val_mask = eligible & (folds == fold)
                    if not fit_mask.any() or not val_mask.any():
                        raise ValueError(f"empty candidate-selection fold {fold}")
                    if kind == "binary" and (
                        np.unique(y[fit_mask]).size < 2 or np.unique(y[val_mask]).size < 2
                    ):
                        raise ValueError(f"binary class missing from fold {fold}")
                    model = _fit(
                        kind,
                        x[fit_mask][:, feature_ids],
                        y[fit_mask],
                        regularization,
                        seed + int(fold),
                    )
                    fold_scores.append(
                        _metric(
                            kind,
                            y[val_mask],
                            _predict(kind, model, x[val_mask][:, feature_ids]),
                        )
                    )
                candidates.append(
                    Candidate(
                        selector,
                        k,
                        float(regularization),
                        tuple(fold_scores),
                        float(np.mean(fold_scores)),
                    )
                )
    winner = max(candidates, key=lambda item: (item.mean_score, -item.k))
    return winner, candidates


def nested_probe_predictions(
    features: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    groups: np.ndarray,
    *,
    kind: TargetKind,
    selectors: tuple[str, ...],
    k_values: tuple[int, ...],
    regularization_values: tuple[float, ...],
    n_folds: int,
    fold_salt: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Candidate, np.ndarray, list[Candidate]]:
    """Select on discovery/train CV; return train-OOF and external-test scores."""
    x = np.asarray(features)
    y = np.asarray(target)
    split_values = np.asarray(splits)
    group_values = np.asarray(groups)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features/target must have shapes (rows, dims)/(rows,)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("probe inputs must be finite")
    validate_split_groups(split_values, group_values)
    discovery, train, test = (
        split_values == "discovery",
        split_values == "train",
        split_values == "test",
    )
    if kind == "binary" and set(np.unique(y[train])) != {0, 1}:
        raise ValueError("binary training target must contain both classes")
    ks = tuple(sorted(set(map(int, k_values))))
    if not ks or ks[0] < 1 or ks[-1] > x.shape[1]:
        raise ValueError(f"k values must be inside [1, {x.shape[1]}]")
    rankings = _rankings(
        x[discovery],
        y[discovery],
        kind=kind,
        selectors=selectors,
        max_k=ks[-1],
        seed=seed,
    )
    all_folds = crossfit_folds(group_values, n_folds=n_folds, salt=fold_salt)
    winner, candidates = _select_candidate(
        x,
        y,
        train,
        all_folds,
        rankings,
        kind=kind,
        k_values=ks,
        regularization_values=regularization_values,
        seed=seed,
    )
    selected = rankings[winner.selector][: winner.k]
    scores = np.zeros(len(x), dtype=np.float64)
    predicted_folds = np.full(len(x), -1, dtype=np.int16)
    for fold in range(n_folds):
        outer_fit = train & (all_folds != fold)
        val_mask = train & (all_folds == fold)
        # A global winner is valid for the external test model, but using it for
        # train-OOF scores lets validation labels influence their own model
        # choice. Re-select inside every outer fold.
        outer_winner, _ = _select_candidate(
            x,
            y,
            outer_fit,
            all_folds,
            rankings,
            kind=kind,
            k_values=ks,
            regularization_values=regularization_values,
            seed=seed + 1000 + fold * 100,
        )
        outer_selected = rankings[outer_winner.selector][: outer_winner.k]
        model = _fit(
            kind,
            x[outer_fit][:, outer_selected],
            y[outer_fit],
            outer_winner.regularization,
            seed + fold,
        )
        scores[val_mask] = _predict(kind, model, x[val_mask][:, outer_selected])
        predicted_folds[val_mask] = fold
    final_model = _fit(
        kind,
        x[train][:, selected],
        y[train],
        winner.regularization,
        seed,
    )
    scores[test] = _predict(kind, final_model, x[test][:, selected])
    # Discovery predictions are intentionally unused downstream.
    scores[discovery] = 0.0
    return scores, predicted_folds, winner, selected, candidates


def _load_feature_rows(features_path: Path, rows_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    features = np.load(features_path, mmap_mode="r")
    rows = pd.read_parquet(rows_path)
    if "prompt_id" not in rows or rows.prompt_id.isna().any() or rows.prompt_id.duplicated().any():
        raise ValueError("feature rows need unique, complete prompt_id")
    if features.ndim != 2 or len(features) != len(rows):
        raise ValueError("feature matrix and row map are not aligned")
    return features, rows[["prompt_id"]].copy()


def _correctness_target(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "judge_binary":
        if "judge_binary_label" not in frame:
            raise ValueError("judge_binary_label is unavailable in this run")
        return frame.judge_binary_label.to_numpy(dtype=np.int8)
    if name == "f1_50":
        return (frame.f1_squad.to_numpy(dtype=float) >= 50).astype(np.int8)
    raise ValueError("correctness target must be judge_binary or f1_50")


def _candidate_rows(target: str, candidates: list[Candidate]) -> list[dict]:
    return [{"target": target, **asdict(candidate)} for candidate in candidates]


def run_probe(
    *,
    config_path: Path,
    run_alias: str,
    features_path: Path,
    rows_path: Path,
    output_dir: Path,
    representation: str,
    layer: int,
    position: str,
    correctness_target_name: str,
    selectors: tuple[str, ...],
    k_values: tuple[int, ...],
    regularization_values: tuple[float, ...],
    seed: int,
) -> Path:
    registry = load_registry(config_path)
    n_folds = int(registry.evaluation["crossfit_folds"])
    fold_salt = f"{registry.paper_id}:{run_alias}"
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = create_experiment_manifest(
        experiment_id=output_dir.name,
        study="A",
        config_path=config_path,
        input_paths=[
            features_path,
            rows_path,
            registry.metadata_path(run_alias),
            registry.tables_dir(run_alias) / "prompts.parquet",
            registry.tables_dir(run_alias) / "labels.parquet",
        ],
        split_protocol={
            "feature_selection": "discovery only",
            "candidate_selection": "nested grouped cross-validation within canonical train",
            "test_scoring": "final model fit on canonical train only",
            "group_column": "source_id",
            "fold_salt": fold_salt,
            "n_folds": n_folds,
        },
    )
    atomic_json(output_dir / "manifest.pre_test.json", manifest)

    frame, _ = load_run_frame(registry, run_alias)
    features, feature_rows = _load_feature_rows(features_path, rows_path)
    frame = feature_rows.merge(frame, on="prompt_id", how="left", validate="one_to_one")
    if frame.source_id.isna().any():
        raise ValueError("feature row map contains prompt IDs absent from run tables")
    splits = frame.split.to_numpy()
    groups = frame.source_id.to_numpy()
    se_target = frame.se_discrete.to_numpy(dtype=np.float64)
    correctness_target = _correctness_target(frame, correctness_target_name)

    common = dict(
        features=features,
        splits=splits,
        groups=groups,
        selectors=selectors,
        k_values=k_values,
        regularization_values=regularization_values,
        n_folds=n_folds,
        fold_salt=fold_salt,
        seed=seed,
    )
    se_score, se_folds, se_winner, se_features, se_grid = nested_probe_predictions(
        target=se_target,
        kind="continuous",
        **common,
    )
    corr_score, corr_folds, corr_winner, corr_features, corr_grid = nested_probe_predictions(
        target=correctness_target,
        kind="binary",
        **common,
    )
    if not np.array_equal(se_folds, corr_folds):
        raise AssertionError("SE and correctness prediction folds differ")
    correctness_from_se, cross_corr_folds, _, _, cross_corr_grid = nested_probe_predictions(
        features[:, se_features],
        correctness_target,
        splits,
        groups,
        kind="binary",
        selectors=("identity",),
        k_values=(len(se_features),),
        regularization_values=regularization_values,
        n_folds=n_folds,
        fold_salt=fold_salt,
        seed=seed,
    )
    se_from_correctness, cross_se_folds, _, _, cross_se_grid = nested_probe_predictions(
        features[:, corr_features],
        se_target,
        splits,
        groups,
        kind="continuous",
        selectors=("identity",),
        k_values=(len(corr_features),),
        regularization_values=regularization_values,
        n_folds=n_folds,
        fold_salt=fold_salt,
        seed=seed,
    )
    if not (
        np.array_equal(se_folds, cross_corr_folds) and np.array_equal(se_folds, cross_se_folds)
    ):
        raise AssertionError("cross-prediction folds differ from direct-probe folds")
    roles = np.full(len(frame), "unused", dtype=object)
    roles[splits == "train"] = "oof_train"
    roles[splits == "test"] = "external_test"
    prediction_frame = pd.DataFrame(
        {
            "prompt_id": frame.prompt_id,
            "se_score": se_score,
            "correctness_score": corr_score,
            "correctness_from_se_features_score": correctness_from_se,
            "se_from_correctness_features_score": se_from_correctness,
            "prediction_role": roles,
            "predicted_fold": se_folds,
        }
    )
    prediction_frame.to_parquet(output_dir / "predictions.parquet", index=False)
    pd.DataFrame(
        _candidate_rows("se_discrete", se_grid)
        + _candidate_rows(correctness_target_name, corr_grid)
        + _candidate_rows(f"{correctness_target_name}_from_se_features", cross_corr_grid)
        + _candidate_rows("se_discrete_from_correctness_features", cross_se_grid)
    ).to_json(output_dir / "validation_grid.jsonl", orient="records", lines=True)
    atomic_json(
        output_dir / "selected_features.json",
        {
            "representation": representation,
            "se": {"winner": asdict(se_winner), "feature_ids": se_features.tolist()},
            "correctness": {
                "target": correctness_target_name,
                "winner": asdict(corr_winner),
                "feature_ids": corr_features.tolist(),
            },
        },
    )
    test = splits == "test"
    study_a = {
        "run_alias": run_alias,
        "representation": representation,
        "layer": layer,
        "position": position,
        "seed": seed,
        "correctness_target": correctness_target_name,
        "feature_overlap": feature_set_statistics(
            se_features, corr_features, dictionary_size=features.shape[1]
        ),
        "important_limit": (
            "hypergeometric overlap is descriptive because learned SAE features and "
            "selection statistics are not independent draws from the dictionary"
        ),
    }
    atomic_json(output_dir / "study_a.json", study_a)
    test_scores = {
        "se_direct": se_score[test],
        "se_from_correctness_features": se_from_correctness[test],
        "correctness_direct": corr_score[test],
        "correctness_from_se_features": correctness_from_se[test],
    }
    metric_bundles = {
        "se_direct": _metric_bundle("continuous", se_target[test], se_score[test]),
        "se_from_correctness_features": _metric_bundle(
            "continuous", se_target[test], se_from_correctness[test]
        ),
        "correctness_direct": _metric_bundle("binary", correctness_target[test], corr_score[test]),
        "correctness_from_se_features": _metric_bundle(
            "binary", correctness_target[test], correctness_from_se[test]
        ),
    }
    atomic_json(
        output_dir / "test_metrics.json",
        {
            "metrics": metric_bundles,
            "se_r2": _metric("continuous", se_target[test], se_score[test]),
            "se_from_correctness_features_r2": _metric(
                "continuous", se_target[test], se_from_correctness[test]
            ),
            "correctness_auroc": _metric("binary", correctness_target[test], corr_score[test]),
            "correctness_from_se_features_auroc": _metric(
                "binary", correctness_target[test], correctness_from_se[test]
            ),
            "n_test": int(test.sum()),
        },
    )
    atomic_json(
        output_dir / "test_bootstrap.json",
        _bootstrap_probe_metrics(
            se_target=se_target[test],
            correctness_target=correctness_target[test],
            scores=test_scores,
            groups=groups[test],
            replicates=int(registry.evaluation["bootstrap_replicates"]),
            seed=int(registry.evaluation["bootstrap_seed"]),
        ),
    )
    atomic_json(output_dir / "manifest.json", mark_test_read(manifest))
    return output_dir


def _csv_tuple(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/three_run_working_paper.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--position", required=True)
    parser.add_argument("--correctness-target", choices=("judge_binary", "f1_50"), default="f1_50")
    parser.add_argument("--selectors", default="spearman,mutual_information")
    parser.add_argument("--k", default="16,32,64,128")
    parser.add_argument("--regularization", default="0.0001,0.001,0.01,0.1,1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = run_probe(
        config_path=Path(args.config),
        run_alias=args.run,
        features_path=args.features,
        rows_path=args.rows,
        output_dir=args.output,
        representation=args.representation,
        layer=args.layer,
        position=args.position,
        correctness_target_name=args.correctness_target,
        selectors=_csv_tuple(args.selectors, str),
        k_values=_csv_tuple(args.k, int),
        regularization_values=_csv_tuple(args.regularization, float),
        seed=args.seed,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
