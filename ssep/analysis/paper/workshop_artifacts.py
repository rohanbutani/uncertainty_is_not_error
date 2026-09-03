"""Assemble the frozen three-run result into workshop-ready tables and figures."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssep.analysis.paper.artifacts import atomic_json, sha256_file

RUNS = {
    "triviaqa_brief": "TriviaQA brief",
    "popqa_brief": "PopQA brief",
    "triviaqa_long": "TriviaQA long",
}
SLUGS = {
    "triviaqa_brief": "triviaqa-brief",
    "popqa_brief": "popqa-brief",
    "triviaqa_long": "triviaqa-long",
}


def _read_csv(path: Path, inputs: list[dict]) -> pd.DataFrame:
    inputs.append({"path": str(path), "sha256": sha256_file(path)})
    return pd.read_csv(path)


def _read_json(path: Path, inputs: list[dict]) -> dict:
    inputs.append({"path": str(path), "sha256": sha256_file(path)})
    return json.loads(path.read_text())


def assemble(*, results_root: Path, data_root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs: list[dict] = []
    performance = []
    comparisons = []
    dissociation = []
    fix_rows = []
    routing_rows = []
    selected: dict[str, dict[str, set[int]]] = {}

    representation_paths = {
        "token_entropy": "token-entropy-v1",
        "dense_128": "dense-coordinate-l20-tbg-v1",
        "sae_128": "sae-l20-tbg-v1",
        "token_entropy+dense": "token-entropy-plus-dense-l20-tbg-v1",
        "token_entropy+sae": "token-entropy-plus-sae-l20-tbg-v1",
    }
    for run, display in RUNS.items():
        slug = SLUGS[run]
        for representation, suffix in representation_paths.items():
            metrics = _read_json(
                results_root / "probes" / f"{slug}-{suffix}" / "test_metrics.json", inputs
            )["metrics"]
            performance.append(
                {
                    "run": run,
                    "dataset": display,
                    "representation": representation,
                    "se_r2": metrics["se_direct"]["r2"],
                    "se_spearman": metrics["se_direct"]["spearman"],
                    "correctness_auroc": metrics["correctness_direct"]["auroc"],
                    "correctness_auprc": metrics["correctness_direct"]["auprc"],
                }
            )
        for family in (
            "representation_comparison",
            "conditional_increment",
            "conditional_representation_comparison",
        ):
            frame = _read_csv(results_root / family / f"{run}-l20-tbg-v1" / "paired_deltas.csv", inputs)
            frame.insert(0, "comparison_family", family)
            frame.insert(0, "run", run)
            comparisons.append(frame)

        probe = results_root / "probes" / f"{slug}-sae-l20-tbg-v1"
        study = _read_json(probe / "study_a.json", inputs)
        metrics = _read_json(probe / "test_metrics.json", inputs)
        bootstrap = _read_json(probe / "test_bootstrap.json", inputs)
        overlap = study["feature_overlap"]
        dissociation.append(
            {
                "run": run,
                "dataset": display,
                "se_features": overlap["se_features"],
                "correctness_features": overlap["correctness_features"],
                "overlap": overlap["overlap"],
                "jaccard": overlap["jaccard"],
                "overlap_coefficient": overlap["overlap_coefficient"],
                "se_cross_minus_direct_r2": (
                    metrics["se_from_correctness_features_r2"] - metrics["se_r2"]
                ),
                "se_cross_ci_low": bootstrap["se_cross_minus_direct_r2"]["ci_low"],
                "se_cross_ci_high": bootstrap["se_cross_minus_direct_r2"]["ci_high"],
                "correctness_cross_minus_direct_auroc": (
                    metrics["correctness_from_se_features_auroc"]
                    - metrics["correctness_auroc"]
                ),
                "correctness_cross_ci_low": bootstrap[
                    "correctness_cross_minus_direct_auroc"
                ]["ci_low"],
                "correctness_cross_ci_high": bootstrap[
                    "correctness_cross_minus_direct_auroc"
                ]["ci_high"],
            }
        )
        feature_sets = _read_json(probe / "selected_features.json", inputs)
        selected[run] = {
            "se": set(feature_sets["se"]["feature_ids"]),
            "correctness": set(feature_sets["correctness"]["feature_ids"]),
        }

        fix = _read_csv(results_root / "study_bc" / f"{run}-v1" / "fix_rate.csv", inputs)
        fix = fix[(fix.budget == 2) & fix.quadrant.isin(
            ["predicted_high_se_error", "predicted_low_se_error"]
        )].copy()
        fix.insert(0, "run", run)
        fix_rows.append(fix)
        routing = _read_csv(
            results_root / "study_bc" / f"{run}-v1" / "routing_policies.csv", inputs
        )
        routing = routing[(routing.budget == 2) & routing.policy.isin(
            ["never_resample", "always_resample", "correctness_trigger", "joint_high_se_error"]
        )].copy()
        routing.insert(0, "run", run)
        routing_rows.append(routing)

    performance_frame = pd.DataFrame(performance)
    comparison_frame = pd.concat(comparisons, ignore_index=True)
    dissociation_frame = pd.DataFrame(dissociation)
    fix_frame = pd.concat(fix_rows, ignore_index=True)
    routing_frame = pd.concat(routing_rows, ignore_index=True)
    performance_frame.to_csv(output_dir / "table_1_performance.csv", index=False)
    comparison_frame.to_csv(output_dir / "table_2_paired_comparisons.csv", index=False)
    dissociation_frame.to_csv(output_dir / "table_3_dissociation.csv", index=False)
    fix_frame.to_csv(output_dir / "table_4_fix_rate_budget2.csv", index=False)
    routing_frame.to_csv(output_dir / "table_5_routing_budget2.csv", index=False)

    stability_path = (
        results_root / "study_a" / "three-run-l20-tbg-five-seeds-v1" / "feature_stability.csv"
    )
    _read_csv(stability_path, inputs).to_csv(output_dir / "table_s1_seed_stability.csv", index=False)
    cross_run = []
    for target in ("se", "correctness"):
        for left, right in combinations(RUNS, 2):
            a, b = selected[left][target], selected[right][target]
            cross_run.append(
                {
                    "target": target,
                    "left": left,
                    "right": right,
                    "left_features": len(a),
                    "right_features": len(b),
                    "overlap": len(a & b),
                    "jaccard": len(a & b) / len(a | b),
                    "overlap_coefficient": len(a & b) / min(len(a), len(b)),
                }
            )
    pd.DataFrame(cross_run).to_csv(output_dir / "table_s2_cross_run_features.csv", index=False)

    colors = {
        "token_entropy": "#6B7280",
        "dense_128": "#2563EB",
        "sae_128": "#D97706",
        "token_entropy+dense": "#60A5FA",
        "token_entropy+sae": "#F59E0B",
    }
    reps = list(representation_paths)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(RUNS))
    width = 0.16
    for offset, rep in enumerate(reps):
        subset = performance_frame.set_index(["run", "representation"])
        axes[0].bar(
            x + (offset - 2) * width,
            [subset.loc[(run, rep), "se_r2"] for run in RUNS],
            width,
            label=rep,
            color=colors[rep],
        )
        axes[1].bar(
            x + (offset - 2) * width,
            [subset.loc[(run, rep), "correctness_auroc"] for run in RUNS],
            width,
            color=colors[rep],
        )
    for axis, ylabel in zip(axes, ("Semantic entropy R²", "Correctness AUROC"), strict=True):
        axis.set_xticks(x, RUNS.values(), rotation=15, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output_dir / "figure_1_representation_performance.png", dpi=220)
    fig.savefig(output_dir / "figure_1_representation_performance.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    high = fix_frame[fix_frame.quadrant == "predicted_high_se_error"].set_index("run")
    low = fix_frame[fix_frame.quadrant == "predicted_low_se_error"].set_index("run")
    axes[0].bar(x - 0.18, [high.loc[run, "fix_at_k"] for run in RUNS], 0.36, label="high SE")
    axes[0].bar(x + 0.18, [low.loc[run, "fix_at_k"] for run in RUNS], 0.36, label="low SE")
    axes[0].set_ylabel("Fix@2 (oracle diagnostic)")
    axes[0].legend(frameon=False)
    route = routing_frame.set_index(["run", "policy"])
    for offset, policy in enumerate(("never_resample", "joint_high_se_error")):
        axes[1].bar(
            x + (offset - 0.5) * 0.36,
            [route.loc[(run, policy), "expected_accuracy"] for run in RUNS],
            0.36,
            label=policy,
        )
    axes[1].set_ylabel("Expected accuracy, budget 2")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x, RUNS.values(), rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "figure_2_resampling_failure.png", dpi=220)
    fig.savefig(output_dir / "figure_2_resampling_failure.pdf")
    plt.close(fig)

    summaries = {
        "schema_version": 1,
        "scope": "development evidence; layer 20 TBG; Gemma-2-9B; F1>=50 correctness",
        "inputs": inputs,
        "important_limits": [
            "No causal SAE intervention was run.",
            "The three runs are development data, not untouched confirmation.",
            "Long-form correctness was deterministically derived from banked text and gold answers.",
            "Dense comparison is a matched 128-coordinate budget, not an unrestricted full-state probe.",
            "Fix@2 is an oracle diagnostic; policy accuracy uses mean banked-sample correctness.",
        ],
    }
    atomic_json(output_dir / "manifest.json", summaries)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/paper"))
    parser.add_argument("--data-root", type=Path, default=Path("data/analysis/paper"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(assemble(results_root=args.results_root, data_root=args.data_root, output_dir=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
