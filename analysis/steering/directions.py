"""Direction builders (Phase 1). Every direction is built on training rows only
and stored as (unit vector, raw norm of the un-normalized construction).

Literature anchors are in the task spec / STEERING_OFFLINE_RESULTS.md; nothing
beyond that list is implemented here.
"""

from __future__ import annotations

import common
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # RESULTS.md §12's tuning grid
N_PCA_PAIRS = 4096
N_RANDOM = 5


def _unit(v: np.ndarray) -> tuple[np.ndarray, float]:
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32), n


def _diffmean(X, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return X[a].mean(0).astype(np.float64) - X[b].mean(0).astype(np.float64)


def tuned_lr(Z, y, sp, seed: int) -> tuple[LogisticRegression, float, float]:
    """§12 protocol: C chosen on val (fit on fit rows), refit on the full train
    draw. Returns (model, chosen C, val AUROC)."""
    best = None
    for c in C_GRID:
        m = LogisticRegression(C=c, max_iter=5_000, random_state=seed)
        m.fit(Z[sp["fit"]], y[sp["fit"]])
        a = roc_auc_score(y[sp["val"]], m.decision_function(Z[sp["val"]]))
        if best is None or a > best[1]:
            best = (c, a)
    c, a = best
    final = LogisticRegression(C=c, max_iter=5_000, random_state=seed)
    final.fit(Z[sp["train"]], y[sp["train"]])
    return final, c, a


def sae_feature_probe(S, df, dataset, site_pos: str, y_se, sp, seed: int):
    """The frozen §1 sparse probe at this site: (feature_ids, coefficients)."""
    from ssep.feature_selection import select_sae_features

    cfg = common.DATASETS[dataset]
    if cfg["guide"] == "entropy":
        guide, kind = common.entropy_at(df, site_pos), "continuous"
    else:
        guide, kind = y_se.astype(np.int8), "binary"
    ids = select_sae_features(
        S[sp["train"]], guide[sp["train"]], cfg["k"],
        method=cfg["select"], target_kind=kind, random_state=seed,
    ).feature_ids
    probe = LogisticRegression(max_iter=5_000, random_state=seed)  # §1: C=1.0
    probe.fit(S[sp["train"]][:, ids], y_se[sp["train"]])
    return np.asarray(ids), probe.coef_[0].astype(np.float64)


def build(dataset: str, df, site_key: str, seed: int, X: np.ndarray,
          S: np.ndarray | None = None, W_dec: np.ndarray | None = None) -> dict:
    """All directions at one (dataset, site, seed). X = the site's states in the
    space the site uses (rms-normalized at SAE sites, raw at final sites).
    S/W_dec: SAE codes + decoder rows; None at sites without an SAE readout."""
    sp = common.splits(len(df), seed)
    y_se, thr = common.se_label(df, sp["train"])
    y_j = df.judge_binary_label.to_numpy()
    ent0 = np.array([e[0] for e in df.greedy_entropy_full], dtype=np.float64)
    tokH_hi = ent0 > float(np.median(ent0[sp["train"]]))
    tr = sp["train"]

    def dm(a, b):
        return _diffmean(X, a & tr, b & tr)

    hi, lo = y_se == 1, y_se == 0
    cor, wrong = y_j == 1, y_j == 0
    out: dict[str, tuple[np.ndarray, float]] = {
        "diffmean_se": _unit(dm(hi, lo)),
        "diffmean_se_strat": _unit(0.5 * (dm(hi & cor, lo & cor) + dm(hi & wrong, lo & wrong))),
        "diffmean_corr": _unit(dm(cor, wrong)),
        "diffmean_corr_strat": _unit(0.5 * (dm(cor & hi, wrong & hi) + dm(cor & lo, wrong & lo))),
        "diffmean_tokH": _unit(dm(tokH_hi, ~tokH_hi)),
    }

    lr, c, val_a = tuned_lr(X, y_se, sp, seed)
    out["probe_lr"] = _unit(lr.coef_[0].astype(np.float64))

    rng = np.random.default_rng(seed)
    hi_idx, lo_idx = np.flatnonzero(hi & tr), np.flatnonzero(lo & tr)
    pairs = X[rng.choice(hi_idx, N_PCA_PAIRS)] - X[rng.choice(lo_idx, N_PCA_PAIRS)]
    pairs = pairs - pairs.mean(0)
    _, _, vt = np.linalg.svd(pairs, full_matrices=False)
    pc1 = vt[0].astype(np.float64)
    if pc1 @ out["diffmean_se"][0] < 0:
        pc1 = -pc1
    out["pca_contrast"] = _unit(pc1)

    meta = {"se_thr": thr, "probe_lr_C": c, "probe_lr_val_auroc": val_a}
    if S is not None:
        ids, w = sae_feature_probe(S, df, dataset, common.DATASETS[dataset]["sites"][site_key][1],
                                   y_se, sp, seed)
        out["sae_topk"] = _unit(w @ W_dec[ids])
        out["sae_top1"] = _unit(W_dec[ids[np.argmax(np.abs(w))]].astype(np.float64))
        meta |= {"sae_ids": ids, "sae_coefs": w}

    import zlib

    rrng = np.random.default_rng(zlib.crc32(f"{dataset}/{site_key}/{seed}".encode()))
    for i in range(N_RANDOM):
        out[f"random{i}"] = _unit(rrng.standard_normal(X.shape[1]))

    return {"dirs": out, "meta": meta, "splits": sp, "y_se": y_se, "y_judge": y_j}


def han_clamp_values(X, y_se, is_train: np.ndarray) -> dict:
    """Han et al. Eq. 1 clamp magnitudes at the two coordinates, train rows only:
    high->low uses min |h_i| over the LOW-SE group, low->high max |h_i| over HIGH."""
    out = {}
    for coord in common.HAN_COORDS:
        col = X[is_train, coord]
        out[coord] = {
            "min_abs_low": float(np.abs(col[y_se[is_train] == 0]).min()),
            "max_abs_high": float(np.abs(col[y_se[is_train] == 1]).max()),
        }
    return out
