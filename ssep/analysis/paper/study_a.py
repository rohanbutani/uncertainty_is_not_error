"""Aggregate frozen probe bundles into Study A dissociation tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ssep.analysis.paper.artifacts import atomic_json, sha256_file
from ssep.analysis.paper.dissociation import feature_stability


def aggregate_study_a(probe_dirs: list[Path], output_dir: Path) -> Path:
    if not probe_dirs:
        raise ValueError("at least one probe directory is required")
    rows = []
    inputs = []
    selected_by_site: dict[tuple, dict[str, dict[str, list[int]]]] = {}
    for directory in probe_dirs:
        study_path = directory / "study_a.json"
        metrics_path = directory / "test_metrics.json"
        bootstrap_path = directory / "test_bootstrap.json"
        manifest_path = directory / "manifest.json"
        selected_path = directory / "selected_features.json"
        for path in (study_path, metrics_path, bootstrap_path, manifest_path, selected_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            inputs.append({"path": str(path), "sha256": sha256_file(path)})
        study = json.loads(study_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.pop("metrics", None)
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "test_evaluated" or manifest.get("test_read_count") != 1:
            raise ValueError(f"probe bundle is not a frozen one-read result: {directory}")
        overlap = study.pop("feature_overlap")
        study.pop("important_limit", None)
        row = {
            "probe_dir": str(directory),
            **study,
            **{f"overlap_{key}": value for key, value in overlap.items()},
            **metrics,
        }
        row["se_cross_minus_direct_r2"] = row["se_from_correctness_features_r2"] - row["se_r2"]
        row["correctness_cross_minus_direct_auroc"] = (
            row["correctness_from_se_features_auroc"] - row["correctness_auroc"]
        )
        for delta in (
            "se_cross_minus_direct_r2",
            "correctness_cross_minus_direct_auroc",
        ):
            row[f"{delta}_ci_low"] = bootstrap[delta]["ci_low"]
            row[f"{delta}_ci_high"] = bootstrap[delta]["ci_high"]
        rows.append(row)
        site = (
            study["run_alias"],
            study["position"],
            study["layer"],
            study["representation"],
        )
        seed_name = f"seed{study['seed']}"
        bucket = selected_by_site.setdefault(site, {"se": {}, "correctness": {}})
        bucket["se"][seed_name] = selected["se"]["feature_ids"]
        bucket["correctness"][seed_name] = selected["correctness"]["feature_ids"]
    frame = pd.DataFrame(rows).sort_values(
        ["run_alias", "position", "layer", "representation", "seed"]
    )
    identity = ["run_alias", "position", "layer", "representation", "seed"]
    if frame.duplicated(identity).any():
        raise ValueError(f"duplicate Study A bundle identity: {identity}")
    output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output_dir / "dissociation_by_site.csv", index=False)
    frame[
        identity
        + [
            "correctness_target",
            "se_r2",
            "se_from_correctness_features_r2",
            "se_cross_minus_direct_r2",
            "se_cross_minus_direct_r2_ci_low",
            "se_cross_minus_direct_r2_ci_high",
            "correctness_auroc",
            "correctness_from_se_features_auroc",
            "correctness_cross_minus_direct_auroc",
            "correctness_cross_minus_direct_auroc_ci_low",
            "correctness_cross_minus_direct_auroc_ci_high",
        ]
    ].to_csv(output_dir / "cross_prediction.csv", index=False)
    frame[identity + [column for column in frame if column.startswith("overlap_")]].to_csv(
        output_dir / "feature_overlap.csv", index=False
    )
    stability_payload = []
    stability_rows = []
    for site, targets in selected_by_site.items():
        if len(targets["se"]) < 2:
            continue
        for target, feature_sets in targets.items():
            summary = feature_stability(feature_sets)
            item = {
                "run_alias": site[0],
                "position": site[1],
                "layer": site[2],
                "representation": site[3],
                "target": target,
                "n_seeds": len(feature_sets),
                **summary,
            }
            stability_payload.append(item)
            stability_rows.append(
                {
                    **{key: item[key] for key in (*identity[:-1], "target", "n_seeds")},
                    "mean_pairwise_jaccard": item["mean_pairwise_jaccard"],
                    "intersection_all_size": len(item["intersection_all"]),
                    "union_size": item["union_size"],
                }
            )
    pd.DataFrame(stability_rows).to_csv(output_dir / "feature_stability.csv", index=False)
    atomic_json(output_dir / "feature_stability.json", stability_payload)
    atomic_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "study": "A",
            "status": "aggregate_of_frozen_probe_bundles",
            "inputs": inputs,
            "n_bundles": len(frame),
            "important_limit": (
                "cross-prediction and overlap establish dissociation only if direct probes "
                "are predictive and results replicate across runs/seeds/sites; they do not "
                "by themselves establish interpretability or causality"
            ),
        },
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(aggregate_study_a(args.probe_dir, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
