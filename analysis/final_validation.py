"""Focused confirmatory controls for the workshop submission.

This is intentionally narrower than the exploratory grid.  It (1) tests whether
the answer-side correctness increment is already present before generation, (2)
evaluates selective prediction from the same held-out predictions, and (3)
independently re-tests discovery-set SAE feature identities on a disjoint split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent / "retest"))
import rt  # noqa: E402
from r5_residual import basis_cols  # noqa: E402

OUT = Path("results/paper/final_validation")
C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def load_df(dataset):
    df = rt.common.load_df(dataset)
    target = "judge_binary"
    if "judge_binary_label" not in df:
        # The local TriviaQA table predates persistence of the LLM-judge channel.
        # Use the paper's prespecified deterministic robustness label, explicitly.
        df["judge_binary_label"] = (df.f1_squad.to_numpy() >= 50).astype(int)
        target = "f1_50"
    return df, target


def fit_lr(X, y, fit, val, seed):
    best = None
    for c in C_GRID:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, random_state=seed))
        m.fit(X[fit], y[fit])
        a = roc_auc_score(y[val], m.predict_proba(X[val])[:, 1])
        if best is None or a > best[0]:
            best = (a, c)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=best[1], max_iter=5000, random_state=seed))
    m.fit(X[fit | val], y[fit | val])
    return m, best[1]


def paired_boot(y, base, aug, seed, n_boot=1000):
    point = roc_auc_score(y, aug) - roc_auc_score(y, base)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size == 2:
            draws.append(roc_auc_score(y[ix], aug[ix]) - roc_auc_score(y[ix], base[ix]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta_auroc": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def risk_curve(y, score):
    order = np.argsort(-score, kind="stable")
    coverages = np.arange(0.1, 1.01, 0.1)
    acc = []
    for cov in coverages:
        k = max(1, int(np.floor(cov * len(y))))
        acc.append(float(y[order[:k]].mean()))
    risks = 1.0 - np.asarray(acc)
    aurc = float(np.trapezoid(risks, coverages) / (coverages[-1] - coverages[0]))
    return {"coverage": coverages.tolist(), "selective_accuracy": acc, "aurc": aurc}


def paired_risk_boot(y, base, aug, seed, n_boot=1000):
    point = risk_curve(y, aug)["aurc"] - risk_curve(y, base)["aurc"]
    rng, draws = np.random.default_rng(seed + 7103), []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), len(y))
        draws.append(risk_curve(y[ix], aug[ix])["aurc"] - risk_curve(y[ix], base[ix])["aurc"])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta_aurc": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def readout(dataset, layer, pos, seed):
    df, target = load_df(dataset)
    sp = rt.common.splits(len(df), seed)
    labs = rt.labels(df, sp)
    X = rt.load_site(df, dataset, layer, pos, sp)
    y = labs["judge"]
    cols = basis_cols(df, labs, pos)["judge~U+"][1]
    B = np.column_stack(cols)
    state, state_c = fit_lr(X, y, sp["fit"], sp["val"], seed)
    state_score = state.predict_proba(X)[:, 1]
    base, base_c = fit_lr(B, y, sp["fit"], sp["val"], seed)
    aug, aug_c = fit_lr(np.column_stack([B, state_score]), y, sp["fit"], sp["val"], seed)
    te = sp["test"]
    pb = base.predict_proba(B[te])[:, 1]
    pa = aug.predict_proba(np.column_stack([B, state_score])[te])[:, 1]
    return {
        "dataset": dataset, "correctness_target": target, "layer": layer, "position": pos, "seed": seed,
        "n_test": int(te.sum()), "C": {"state": state_c, "base": base_c, "aug": aug_c},
        "base_auroc": float(roc_auc_score(y[te], pb)),
        "aug_auroc": float(roc_auc_score(y[te], pa)),
        **paired_boot(y[te], pb, pa, seed),
        "risk_coverage": {"basis": risk_curve(y[te], pb), "basis_plus_state": risk_curve(y[te], pa)},
        "risk_coverage_paired": paired_risk_boot(y[te], pb, pa, seed),
    }


def _p_and_effect(S, a, b):
    p = mannwhitneyu(S[a], S[b], axis=0, method="asymptotic").pvalue
    return np.nan_to_num(p, nan=1.0), S[a].mean(0) - S[b].mean(0)


def confirm_features(dataset, seed=2026):
    """Discover on one half and confirm conditional correctness effects on the other.

    Confirmation establishes replicated correctness association in both SE strata.
    It does not turn failure to detect an SE effect into evidence of no SE effect.
    """
    df, target = load_df(dataset)
    rng = np.random.default_rng(seed)
    disc = np.zeros(len(df), bool)
    # Stratify the split by the two labels so all four cells retain power.
    sp0 = rt.common.splits(len(df), 42)
    labs = rt.labels(df, sp0)
    y, u = labs["judge"], labs["se_bin"]
    for cell in range(4):
        ids = np.flatnonzero((2 * u + y) == cell)
        disc[rng.permutation(ids)[: len(ids) // 2]] = True
    conf = ~disc
    X, _ = rt.common.rms_normalize(rt.common.load_slice(df, dataset, 24, "slt"), disc)
    sae = rt.common.load_sae(24)
    S = rt.common.sae_encode(sae, X)

    def effects(mask):
        p_hi, d_hi = _p_and_effect(S, mask & (u == 1) & (y == 1), mask & (u == 1) & (y == 0))
        p_lo, d_lo = _p_and_effect(S, mask & (u == 0) & (y == 1), mask & (u == 0) & (y == 0))
        p_se_c, _ = _p_and_effect(S, mask & (y == 1) & (u == 1), mask & (y == 1) & (u == 0))
        p_se_w, _ = _p_and_effect(S, mask & (y == 0) & (u == 1), mask & (y == 0) & (u == 0))
        return p_hi, p_lo, d_hi, d_lo, p_se_c, p_se_w

    d = effects(disc)
    sig = [multipletests(p, 0.05, method="fdr_bh")[0] for p in (d[0], d[1], d[4], d[5])]
    candidates = np.flatnonzero(sig[0] & sig[1] & ~sig[2] & ~sig[3] & (d[2] * d[3] > 0))
    c = effects(conf)
    if len(candidates):
        # Confirm both conditional contrasts, correcting over candidate x stratum tests.
        joint_p = np.concatenate([c[0][candidates], c[1][candidates]])
        joint_sig = multipletests(joint_p, 0.05, method="fdr_bh")[0]
        ok = joint_sig[:len(candidates)] & joint_sig[len(candidates):]
        ok &= (c[2][candidates] * d[2][candidates] > 0) & (c[3][candidates] * d[3][candidates] > 0)
        confirmed = candidates[ok]
    else:
        confirmed = candidates
    return {
        "dataset": dataset, "correctness_target": target, "seed": seed, "site": "L24/slt", "n_discovery": int(disc.sum()),
        "n_confirmation": int(conf.sum()), "n_discovered_candidates": int(len(candidates)),
        "n_confirmed_conditional_correctness": int(len(confirmed)),
        "discovered_feature_ids": candidates.tolist(), "confirmed_feature_ids": confirmed.tolist(),
        "scope_note": "Replicates correctness association within both SE strata; does not prove equivalence to zero for SE effects.",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Same-layer position control plus the pre-specified SE-TBG site for TriviaQA.
    cells = [("trivia_qa", 24, "slt"), ("trivia_qa", 24, "tbg"),
             ("trivia_qa", 40, "tbg"), ("popqa", 24, "slt"), ("popqa", 24, "tbg")]
    readouts = [readout(ds, layer, pos, seed) for ds, layer, pos in cells for seed in rt.common.SEEDS]
    (OUT / "pregen_and_risk_coverage.json").write_text(json.dumps(readouts, indent=2))
    confirmations = [confirm_features(ds) for ds in ("trivia_qa", "popqa")]
    (OUT / "heldout_feature_confirmation.json").write_text(json.dumps(confirmations, indent=2))
    print(json.dumps({"readouts": readouts, "feature_confirmation": confirmations}, indent=2))


if __name__ == "__main__":
    main()
