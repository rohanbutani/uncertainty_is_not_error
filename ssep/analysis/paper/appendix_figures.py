"""Generate reviewer-facing appendix figures from frozen paper artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = ["triviaqa_brief", "popqa_brief", "triviaqa_long"]
DISPLAY = {
    "triviaqa_brief": "TriviaQA\nbrief",
    "popqa_brief": "PopQA\nbrief",
    "triviaqa_long": "TriviaQA\nlong",
}
SLUG = {
    "triviaqa_brief": "triviaqa-brief",
    "popqa_brief": "popqa-brief",
    "triviaqa_long": "triviaqa-long",
}


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate(*, results_root: Path, analysis_root: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    # A1: SAE quality and sparsity, taken from materialization summaries.
    quality = []
    for run in RUNS:
        summary = json.loads((analysis_root / run / "l20_tbg" / "summary.json").read_text())
        sae = summary["sae"]
        quality.append(
            {
                "run": run,
                "l0": sae["realized_l0_mean"],
                "cosine": sae["cos_sim_mean"],
                "error": sae["recon_rel_err"],
            }
        )
    q = pd.DataFrame(quality)
    q.to_csv(out / "table_a1_sae_quality.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), constrained_layout=True)
    x = np.arange(len(RUNS))
    for ax, column, label in zip(
        axes,
        ["l0", "cosine", "error"],
        ["Mean realized L0", "Reconstruction cosine", "Relative reconstruction error"],
        strict=True,
    ):
        ax.bar(x, q.set_index("run").loc[RUNS, column], color=["#D97706", "#2563EB", "#059669"])
        ax.set_xticks(x, [DISPLAY[r] for r in RUNS])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    _save(fig, out, "figure_a1_sae_quality")

    # A2: paired representation contrasts, with source-bootstrap intervals.
    rows = []
    for run in RUNS:
        for family in [
            "representation_comparison",
            "conditional_increment",
            "conditional_representation_comparison",
        ]:
            frame = pd.read_csv(results_root / family / f"{run}-l20-tbg-v1" / "paired_deltas.csv")
            for _, row in frame.iterrows():
                rows.append({"run": run, "family": family, **row.to_dict()})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "table_a2_paired_contrasts.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for ax, metric, label in zip(
        axes,
        ["delta_se_r2", "delta_correctness_auroc"],
        ["Δ SE R²", "Δ correctness AUROC"],
        strict=True,
    ):
        subset = contrasts[
            (contrasts.reference == "dense") & (contrasts.candidate == "sae")
        ].set_index("run")
        vals = [subset.loc[r, metric] for r in RUNS]
        lows = [subset.loc[r, f"{metric}_ci_low"] for r in RUNS]
        highs = [subset.loc[r, f"{metric}_ci_high"] for r in RUNS]
        ax.errorbar(
            x,
            vals,
            yerr=[np.array(vals) - lows, np.array(highs) - vals],
            fmt="o",
            capsize=4,
            color="#B45309",
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, [DISPLAY[r] for r in RUNS])
        ax.set_ylabel(f"SAE − dense ({label})")
        ax.grid(axis="y", alpha=0.25)
    _save(fig, out, "figure_a2_sae_vs_dense_paired")

    # A3: cross-prediction and overlap.
    diss = pd.read_csv(results_root / "workshop_submission_v1" / "table_3_dissociation.csv")
    diss.to_csv(out / "table_a3_dissociation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    axes[0].bar(
        x - 0.18, diss.overlap_coefficient, 0.36, label="Overlap coefficient", color="#7C3AED"
    )
    axes[0].bar(x + 0.18, diss.jaccard, 0.36, label="Jaccard", color="#A78BFA")
    axes[0].set_xticks(x, [DISPLAY[r] for r in RUNS])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Feature-set overlap")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    for metric, low, high, color, label in [
        (
            "se_cross_minus_direct_r2",
            "se_cross_ci_low",
            "se_cross_ci_high",
            "#2563EB",
            "SE cross − direct R²",
        ),
        (
            "correctness_cross_minus_direct_auroc",
            "correctness_cross_ci_low",
            "correctness_cross_ci_high",
            "#D97706",
            "Correctness cross − direct AUROC",
        ),
    ]:
        vals = diss[metric].to_numpy()
        lo = diss[low].to_numpy()
        hi = diss[high].to_numpy()
        axes[1].errorbar(
            x + (0.04 if metric.startswith("correctness") else -0.04),
            vals,
            yerr=[vals - lo, hi - vals],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x, [DISPLAY[r] for r in RUNS])
    axes[1].set_ylabel("Cross-prediction difference")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    _save(fig, out, "figure_a3_dissociation_cross_prediction")

    # A4: five-seed stability and cross-run identity overlap.
    stability = pd.read_csv(results_root / "workshop_submission_v1" / "table_s1_seed_stability.csv")
    stability.to_csv(out / "table_a4_seed_stability.csv", index=False)
    cross = pd.read_csv(results_root / "workshop_submission_v1" / "table_s2_cross_run_features.csv")
    cross.to_csv(out / "table_a5_cross_run_feature_overlap.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.6, 3.4), constrained_layout=True)
    labels = [f"{r}\n{t}" for r, t in zip(stability.run_alias, stability.target, strict=True)]
    ax.bar(np.arange(len(stability)), stability.mean_pairwise_jaccard, color="#0F766E")
    ax.set_xticks(np.arange(len(stability)), labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean pairwise Jaccard across seeds")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out, "figure_a4_seed_stability")

    # A5: behavioral failure with uncertainty bars.
    fix = pd.read_csv(results_root / "workshop_submission_v1" / "table_4_fix_rate_budget2.csv")
    fix.to_csv(out / "table_a6_fix_rate_budget2.csv", index=False)
    route = pd.read_csv(results_root / "workshop_submission_v1" / "table_5_routing_budget2.csv")
    route.to_csv(out / "table_a7_routing_budget2.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for offset, quadrant, color, label in [
        (-0.18, "predicted_high_se_error", "#2563EB", "Predicted high SE"),
        (0.18, "predicted_low_se_error", "#D97706", "Predicted low SE"),
    ]:
        d = fix[fix.quadrant == quadrant].set_index("run")
        vals = d.fix_at_k.loc[RUNS].to_numpy()
        lo = d.fix_at_k_ci_low.loc[RUNS].to_numpy()
        hi = d.fix_at_k_ci_high.loc[RUNS].to_numpy()
        axes[0].errorbar(
            x + offset,
            vals,
            yerr=[vals - lo, hi - vals],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
    axes[0].set_xticks(x, [DISPLAY[r] for r in RUNS])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fix@2 (oracle diagnostic)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    for offset, policy, color, label in [
        (-0.18, "never_resample", "#2563EB", "Never resample"),
        (0.18, "joint_high_se_error", "#D97706", "Joint routing"),
    ]:
        d = route[route.policy == policy].set_index("run")
        vals = d.expected_accuracy.loc[RUNS].to_numpy()
        lo = d.expected_accuracy_ci_low.loc[RUNS].to_numpy()
        hi = d.expected_accuracy_ci_high.loc[RUNS].to_numpy()
        axes[1].errorbar(
            x + offset,
            vals,
            yerr=[vals - lo, hi - vals],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
    axes[1].set_xticks(x, [DISPLAY[r] for r in RUNS])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Expected accuracy (budget 2)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    _save(fig, out, "figure_a5_behavioral_failure")

    # A6: routing policy sensitivity over all banked budgets.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True, constrained_layout=True)
    all_route = []
    for i, run in enumerate(RUNS):
        d = pd.read_csv(results_root / "study_bc" / f"{run}-v1" / "routing_policies.csv")
        d = d[d.policy.isin(["never_resample", "always_resample", "joint_high_se_error"])]
        all_route.append(d.assign(run=run))
        for policy, color in [
            ("never_resample", "#2563EB"),
            ("always_resample", "#6B7280"),
            ("joint_high_se_error", "#D97706"),
        ]:
            subset = d[d.policy == policy].sort_values("budget")
            axes[i].plot(
                subset.budget, subset.expected_accuracy, marker="o", label=policy, color=color
            )
        axes[i].set_title(DISPLAY[run].replace("\n", " "))
        axes[i].set_xlabel("Resampling budget")
        axes[i].set_xticks(sorted(d.budget.unique()))
        axes[i].grid(alpha=0.25)
    axes[0].set_ylabel("Expected accuracy")
    axes[-1].legend(frameon=False, fontsize=7)
    pd.concat(all_route, ignore_index=True).to_csv(
        out / "table_a8_routing_all_budgets.csv", index=False
    )
    _save(fig, out, "figure_a6_routing_budget_sensitivity")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/paper"))
    parser.add_argument("--analysis-root", type=Path, default=Path("data/analysis/paper"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        generate(results_root=args.results_root, analysis_root=args.analysis_root, out=args.output)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
