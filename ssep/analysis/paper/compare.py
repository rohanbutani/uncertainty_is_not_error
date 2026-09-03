"""Paired held-out comparison of probe representations for one registered run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

from ssep.analysis.paper.artifacts import atomic_json
from ssep.analysis.paper.bootstrap import cluster_bootstrap_indices, percentile_interval
from ssep.analysis.paper.data_contract import load_run_frame
from ssep.analysis.paper.registry import load_registry


def _load_bundle(label: str, path: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("status") != "test_evaluated" or manifest.get("test_read_count") != 1:
        raise ValueError(f"probe {label!r} is not a frozen single-read test bundle")
    study = json.loads((path / "study_a.json").read_text())
    predictions = pd.read_parquet(path / "predictions.parquet")
    return study, predictions


def compare_representations(
    *,
    config_path: Path,
    run_alias: str,
    probes: list[tuple[str, Path]],
    output_dir: Path,
) -> Path:
    if len(probes) < 2 or len({name for name, _ in probes}) != len(probes):
        raise ValueError("provide at least two uniquely named probe bundles")
    registry = load_registry(config_path)
    frame, _ = load_run_frame(registry, run_alias)
    loaded: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict] = {}
    for name, path in probes:
        study, predictions = _load_bundle(name, path)
        if study["run_alias"] != run_alias:
            raise ValueError(f"probe {name!r} belongs to {study['run_alias']!r}")
        loaded[name] = predictions
        metadata[name] = {"path": str(path), "representation": study["representation"]}
    names = [name for name, _ in probes]
    common_ids = loaded[names[0]].prompt_id.tolist()
    if any(table.prompt_id.tolist() != common_ids for table in loaded.values()):
        raise ValueError("probe prediction rows are not identically ordered")
    joined = loaded[names[0]][["prompt_id", "prediction_role"]].merge(
        frame, on="prompt_id", how="left", validate="one_to_one"
    )
    test = joined.prediction_role.to_numpy() == "external_test"
    if not test.any() or not np.all(joined.split.to_numpy()[test] == "test"):
        raise ValueError("external-test prediction roles do not match canonical test")
    se = joined.se_discrete.to_numpy(dtype=float)[test]
    correct = (joined.f1_squad.to_numpy(dtype=float)[test] >= 50).astype(np.int8)
    groups = joined.source_id.to_numpy()[test]

    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metric_rows = []
    for name in names:
        table = loaded[name]
        se_score = table.se_score.to_numpy(dtype=float)[test]
        corr_score = table.correctness_score.to_numpy(dtype=float)[test]
        scores[name] = se_score, corr_score
        metric_rows.append(
            {
                "probe": name,
                "representation": metadata[name]["representation"],
                "se_r2": float(r2_score(se, se_score)),
                "correctness_auroc": float(roc_auc_score(correct, corr_score)),
                "n_test": int(test.sum()),
            }
        )

    reference = names[0]
    draws = {name: {"se_r2": [], "correctness_auroc": []} for name in names[1:]}
    replicates = int(registry.evaluation["bootstrap_replicates"])
    for indices in cluster_bootstrap_indices(
        groups, replicates=replicates, seed=int(registry.evaluation["bootstrap_seed"])
    ):
        if np.unique(correct[indices]).size < 2:
            continue
        ref_se, ref_corr = scores[reference]
        ref_r2 = r2_score(se[indices], ref_se[indices])
        ref_auc = roc_auc_score(correct[indices], ref_corr[indices])
        for name in names[1:]:
            candidate_se, candidate_corr = scores[name]
            draws[name]["se_r2"].append(r2_score(se[indices], candidate_se[indices]) - ref_r2)
            draws[name]["correctness_auroc"].append(
                roc_auc_score(correct[indices], candidate_corr[indices]) - ref_auc
            )
    points = {row["probe"]: row for row in metric_rows}
    delta_rows = []
    for name in names[1:]:
        row = {"reference": reference, "candidate": name}
        for metric in ("se_r2", "correctness_auroc"):
            values = draws[name][metric]
            low, high, valid = percentile_interval(values)
            row[f"delta_{metric}"] = points[name][metric] - points[reference][metric]
            row[f"delta_{metric}_ci_low"] = low
            row[f"delta_{metric}_ci_high"] = high
            row[f"delta_{metric}_bootstrap_valid"] = valid
        delta_rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(output_dir / "paired_deltas.csv", index=False)
    atomic_json(
        output_dir / "summary.json",
        {
            "run_alias": run_alias,
            "reference": reference,
            "probes": metadata,
            "bootstrap_unit": "source_id",
            "bootstrap_replicates": replicates,
        },
    )
    return output_dir


def _probe_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("probe must be NAME=PATH")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper/three_run_working_paper.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--probe", action="append", type=_probe_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        compare_representations(
            config_path=Path(args.config),
            run_alias=args.run,
            probes=args.probe,
            output_dir=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
