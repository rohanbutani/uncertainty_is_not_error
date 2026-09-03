"""Generate submission figures from the frozen result ledgers.

The source numbers are recorded in notebooks/RESULTS.md,
notebooks/RETEST_DISSOCIATION_RESULTS.md, and
notebooks/STEERING_OFFLINE_RESULTS.md.  This script deliberately contains no
statistical recomputation: it makes the provenance of every plotted value
inspectable while the raw retest JSON bundle is restored for release.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


OUT = Path("results/paper/final_submission")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#2C6EBA",
    "orange": "#E07A1F",
    "green": "#2E8B57",
    "red": "#C44E52",
    "gray": "#777777",
    "light": "#E9EEF4",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=10)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def validation_ladder() -> None:
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.0), constrained_layout=True)

    # RETEST R1: held-out values at each target's selected site (seed 42).
    ax = axs[0]
    x = np.arange(2)
    se = [0.891, 0.958]
    corr = [0.898, 0.926]
    w = 0.34
    ax.bar(x - w / 2, se, w, color=COLORS["blue"], label="SE reader")
    ax.bar(x + w / 2, corr, w, color=COLORS["orange"], label="Correctness reader")
    ax.set_xticks(x, ["TriviaQA\nL24/SLT", "PopQA\nL24/SLT"])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("Matched readers at L24/SLT")
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "A")

    # RETEST R2/R3, judge-argmax site, seed 42. Nulls are the whitened
    # label-coupling null intervals, plotted as error bars.
    ax = axs[1]
    raw = np.array([0.966, 0.968])
    white = np.array([0.306, 0.130])
    null_mid = np.array([(0.328 + 0.436) / 2, (0.242 + 0.357) / 2])
    null_err = np.array([(0.436 - 0.328) / 2, (0.357 - 0.242) / 2])
    xx = np.arange(2)
    ax.scatter(xx - 0.16, raw, s=32, color=COLORS["gray"], label="Raw")
    ax.scatter(xx, white, s=32, color=COLORS["blue"], label="Whitened")
    ax.errorbar(
        xx + 0.16,
        null_mid,
        yerr=null_err,
        fmt="o",
        capsize=3,
        color=COLORS["red"],
        label="Coupling null",
    )
    ax.set_xticks(xx, ["TriviaQA", "PopQA"])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel(r"$|\cos(\mathrm{SE},\mathrm{correct})|$")
    ax.set_title("Collinearity was misleading")
    ax.legend(loc="center right", frameon=False)
    panel_label(ax, "B")

    # RETEST R5: state-score increment beyond increasingly strict scalar bases.
    ax = axs[2]
    labels = [r"$U$", r"$E^+$", r"$U^+$"]
    tq = np.array([0.0192, 0.0169, 0.0035])
    pq = np.array([0.0326, 0.0356, 0.0207])
    tq_err = np.array([[0.0192 - 0.011, 0.0169 - 0.009, 0.0035 - (-0.003)],
                       [0.027 - 0.0192, 0.025 - 0.0169, 0.010 - 0.0035]])
    pq_err = np.array([[0.0326 - 0.027, 0.0356 - 0.030, 0.0207 - 0.016],
                       [0.038 - 0.0326, 0.041 - 0.0356, 0.025 - 0.0207]])
    xx = np.arange(3)
    ax.axhline(0, color="#BBBBBB", lw=0.8)
    ax.errorbar(xx - 0.08, tq, yerr=tq_err, fmt="o", capsize=2.5,
                color=COLORS["blue"], label="TriviaQA")
    ax.errorbar(xx + 0.08, pq, yerr=pq_err, fmt="o", capsize=2.5,
                color=COLORS["orange"], label="PopQA")
    ax.set_xticks(xx, labels)
    ax.set_ylim(-0.01, 0.045)
    ax.set_ylabel(r"Incremental $\Delta$AUROC")
    ax.set_title("Correctness beyond scalars")
    ax.legend(loc="upper left", frameon=False)
    panel_label(ax, "C")

    # RETEST R4/R6: verified LEACE erasure and full-pool SAE partition.
    ax = axs[3]
    post = [0.711, 0.678]  # mean of seeds 42/43 at judge sites
    pure_corr = [21, 42]
    ax.bar([0, 1], post, 0.52, color=[COLORS["blue"], COLORS["orange"]])
    ax.axhspan(0.50, 0.54, color=COLORS["light"], zorder=-2, label="SE after erasure")
    ax.set_xticks([0, 1], ["TriviaQA\n21 features", "PopQA\n42 features"])
    ax.set_ylim(0.48, 0.76)
    ax.set_ylabel("Correctness AUROC")
    ax.set_title("After linear SE erasure")
    for i, (v, n) in enumerate(zip(post, pure_corr, strict=True)):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
        del n
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "D")

    save(fig, "figure_1_validation_ladder")


def limits_and_actionability() -> None:
    fig, axs = plt.subplots(
        1, 4, figsize=(11.4, 2.05), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.55]},
    )

    # RESULTS §§2,12: test AUROC after fair regularization tuning.
    ax = axs[0]
    labels = ["Dense\n(all 3,584)", "SAE\n(top-K)", "Token\nentropy"]
    tq = [0.8909, 0.8890, 0.9334]
    pq = [0.9531, 0.9428, 0.9758]
    x = np.arange(3)
    w = 0.34
    ax.bar(x - w / 2, tq, w, color=COLORS["blue"], label="TriviaQA")
    ax.bar(x + w / 2, pq, w, color=COLORS["orange"], label="PopQA")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("SE AUROC")
    ax.set_title("No SAE accuracy advantage")
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "A")

    # STEERING E1/E2: held-out tail quadrants, at least one correct in ten samples.
    ax = axs[1]
    hi = [0.508, 0.123]
    lo = [0.619, 0.310]
    x = np.arange(2)
    ax.bar(x - w / 2, hi, w, color=COLORS["red"], label="High-SE error")
    ax.bar(x + w / 2, lo, w, color=COLORS["green"], label="Low-SE error")
    ax.set_xticks(x, ["TriviaQA\n883 / 21", "PopQA\n3377 / 551"])
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Any correct sample in 10")
    ax.set_title("Correct samples in bank")
    ax.legend(loc="upper right", frameon=False)
    panel_label(ax, "B")

    # PAPER Study C, budget 2: expected accuracy for never vs joint routing.
    ax = axs[2]
    never = [0.775, 0.336, 0.787]
    joint = [0.731, 0.318, 0.738]
    x = np.arange(3)
    ax.bar(x - w / 2, never, w, color=COLORS["gray"], label="Never resample")
    ax.bar(x + w / 2, joint, w, color=COLORS["red"], label="Joint routing")
    ax.set_xticks(x, ["TriviaQA\nbrief", "PopQA\nbrief", "TriviaQA\nlong"])
    ax.set_ylim(0.25, 0.84)
    ax.set_ylabel("Expected accuracy")
    ax.set_title("Routing hurts at budget 2")
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "C")

    # Focused held-out validation: PopQA L24/SLT, mean over seeds 42/43.
    ax = axs[3]
    validation = json.loads(Path("results/paper/final_validation/pregen_and_risk_coverage.json").read_text())
    rows = [r for r in validation if r["dataset"] == "popqa" and r["layer"] == 24 and r["position"] == "slt"]
    cov = np.asarray(rows[0]["risk_coverage"]["basis"]["coverage"])
    basis = np.mean([r["risk_coverage"]["basis"]["selective_accuracy"] for r in rows], axis=0)
    plus = np.mean([r["risk_coverage"]["basis_plus_state"]["selective_accuracy"] for r in rows], axis=0)
    ax.fill_between(cov, basis, plus, where=plus >= basis, color="#BFE3D0", alpha=.65, zorder=0)
    ax.plot(cov, basis, "o--", ms=3.0, lw=1.5, color="#555555", label=r"$U^+$ scalars")
    ax.plot(cov, plus, "o-", ms=3.2, lw=1.8, color=COLORS["green"], label=r"$U^+$ + L24/SLT state")
    ax.set_xlim(.08, 1.02)
    ax.set_ylim(.28, 1.0)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective accuracy")
    ax.set_title("PopQA risk--coverage")
    ax.text(.98, .60, r"$\Delta$AURC $=-.0149/-.0129$", ha="right", va="center", fontsize=7,
            color=COLORS["green"], transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1.5})
    ax.legend(loc="upper right", frameon=False, handlelength=2.4)
    panel_label(ax, "D")

    save(fig, "figure_2_limits_actionability")


if __name__ == "__main__":
    validation_ladder()
    limits_and_actionability()
    print(f"Wrote submission figures to {OUT}")
