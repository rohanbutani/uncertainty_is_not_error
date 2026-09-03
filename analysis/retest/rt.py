"""Shared glue for the dissociation re-test (v2) — RETEST_DISSOCIATION_RESULTS.md.

Reuses analysis/steering's loaders, splits, seeds, and rms handling wholesale.
Adds only what this task needs: the §3 grid coordinates, the three label axes
(SE / judge / token-entropy, plus the f1 robustness channel), a vectorized
prompt-level bootstrap AUROC, and Ledoit–Wolf whitening for the Fisher/LDA
variants of the diff-in-means directions (Marks & Tegmark 2310.06824).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steering"))
import common  # noqa: E402
import directions as D  # noqa: E402, F401  (re-exported: tuned_lr, C_GRID)

RESULTS = common.REPO / "results" / "retest"
GRID_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32, 36, 38, 39, 40, 41)
GRID_POS = ("tbg", "ans0", "slt", "eoa", "genlast", "kw", "nl", "qlast")

# SE-argmax sites (RESULTS.md §3, carried over unchanged); judge-/tokH-argmax
# sites are appended per dataset by Phase R1 into results/retest/r1_sites.json.
SE_SITES = {"trivia_qa": {"se_tbg": (40, "tbg"), "se_slt": (28, "slt")},
            "popqa": {"se_tbg": (24, "tbg"), "se_slt": (24, "slt")}}


def sites(dataset: str) -> dict[str, tuple[int, str]]:
    """The R2+ site set: SE-argmax TBG/SLT + R1's judge- and tokH-argmax."""
    out = dict(SE_SITES[dataset])
    r1 = json.loads((RESULTS / "r1_sites.json").read_text())[dataset]
    for k in ("judge_argmax", "tokH_argmax"):
        out[k] = tuple(r1[k])
    return out


def labels(df, sp) -> dict:
    """Every label axis used anywhere in R1–R8, one place.
    tokH = greedy_entropy_full[0] (next-token entropy at TBG), train-median split."""
    y_se, se_thr = common.se_label(df, sp["train"])
    ent0 = np.array([e[0] for e in df.greedy_entropy_full], dtype=np.float64)
    tokH_thr = float(np.median(ent0[sp["train"]]))
    return {
        "se_bin": y_se, "se_cont": df.se_discrete.to_numpy().astype(np.float64),
        "se_thr": se_thr,
        "judge": df.judge_binary_label.to_numpy().astype(int),
        "f1_50": (df.f1_squad.to_numpy() >= 50).astype(int),
        "tokH_cont": ent0, "tokH_bin": (ent0 > tokH_thr).astype(int),
        "tokH_thr": tokH_thr,
    }


def load_site(df, dataset: str, layer: int, pos: str, sp) -> np.ndarray:
    """rms-normalized states at (layer, pos) — the §3 harness space, used at
    every site in this task (final layer included, as in the §3 grid)."""
    X = common.load_slice(df, dataset, layer, pos)
    Xn, _ = common.rms_normalize(X, sp["train"])
    return Xn


# ------------------------------------------------------------- statistics ----
def _rank_auroc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata

    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


def boot_auroc(y, s, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC + prompt-level bootstrap 95% CI (percentile, n_boot resamples)."""
    from scipy.stats import rankdata

    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    point = _rank_auroc(y, s)
    n = len(y)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    yb = y[idx]
    r = rankdata(s[idx], axis=1)
    npos = yb.sum(1)
    ok = (npos > 0) & (npos < n)
    a = ((r * yb).sum(1) - npos * (npos + 1) / 2) / np.maximum(npos * (n - npos), 1)
    lo, hi = np.percentile(a[ok], [2.5, 97.5])
    return {"auroc": point, "lo": float(lo), "hi": float(hi), "n": n}


def lw_chol(X_train: np.ndarray) -> np.ndarray:
    """Cholesky factor L (Σ = L Lᵀ) of the Ledoit–Wolf shrinkage covariance."""
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(X_train.astype(np.float64))
    return np.linalg.cholesky(lw.covariance_)


def whiten_dir(L: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Σ⁻¹ v via the Cholesky factor — the Fisher/LDA direction for a diffmean v."""
    from scipy.linalg import solve_triangular

    z = solve_triangular(L, v.astype(np.float64), lower=True)
    w = solve_triangular(L.T, z, lower=False)
    return w / np.linalg.norm(w)


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float64)


def diffmean(X, m_pos, m_neg) -> np.ndarray:
    return X[m_pos].mean(0).astype(np.float64) - X[m_neg].mean(0).astype(np.float64)


def cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------- outputs ----
def save_result(name: str, payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")

    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1, default=default))
    print(f"saved results/retest/{name}.json")


def load_result(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())
