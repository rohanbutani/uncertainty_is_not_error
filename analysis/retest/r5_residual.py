"""Phase R5 — residualized-target probing (the decisive test).

Axes-swapped RESULTS.md §9. On training rows, OLS `target ~ basis`; frozen
coefficients applied everywhere; residual binarized at the training median.
Probes: dense tuned-C LR, and the SAE top-K arm (selection on fit rows guided
by the continuous residual — §9's rule — dataset's frozen §1 selector, K grid
chosen on validation, probe C=1.0 refit on the train draw). Controls exactly
as §9: the basis's own AUROC against the residual label (ctl floor), a
shuffled-residual dense probe, a random direction.

Residual specs (name: target ~ basis):
  judge~U    judge_binary ~ {se_cont, se_lnrao, n_clusters}
  judge~U+   judge_binary ~ U ∪ {E(site), E², mean greedy logprob, p_true, logit_gap[0]}
  judge~E+   judge_binary ~ {E(site), E², mean logprob, p_true, logit_gap[0]}   (R7)
  se~CF      se_cont      ~ {judge_binary, f1_squad}                    (the mirror)
  tokH~CF    tokH_cont    ~ {judge_binary, f1_squad}                    (R7 mirror)
"""

import sys
import zlib

import numpy as np
import rt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

K_GRID = (16, 32, 64, 128)


def basis_cols(df, labs, pos):
    e = rt.common.entropy_at(df, pos)
    gtl = df.greedy_token_logprobs.to_numpy()
    ns = df.greedy_sliced_token_count.to_numpy()
    mean_lp = np.array([float(np.mean(g[: max(1, min(len(g), n))]))
                        for g, n in zip(gtl, ns, strict=True)])
    gap0 = np.array([float(g[0]) for g in df.greedy_logit_gap])
    out_scalars = [e, e**2, mean_lp, df.p_true.to_numpy().astype(float), gap0]
    return {
        "judge~U": (labs["judge"], [labs["se_cont"], df.se_lnrao.to_numpy().astype(float),
                                    df.n_clusters.to_numpy().astype(float)]),
        "judge~U+": (labs["judge"], [labs["se_cont"], df.se_lnrao.to_numpy().astype(float),
                                     df.n_clusters.to_numpy().astype(float)] + out_scalars),
        "judge~E+": (labs["judge"], out_scalars),
        "se~CF": (labs["se_cont"], [labs["judge"].astype(float),
                                    df.f1_squad.to_numpy().astype(float)]),
        "tokH~CF": (labs["tokH_cont"], [labs["judge"].astype(float),
                                        df.f1_squad.to_numpy().astype(float)]),
    }


def residualize(y, cols, tr):
    A = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(A[tr], np.asarray(y, float)[tr], rcond=None)
    pred = A @ beta
    resid = np.asarray(y, float) - pred
    thr = float(np.median(resid[tr]))
    r2 = 1.0 - resid[tr].var() / max(np.asarray(y, float)[tr].var(), 1e-12)
    return resid, (resid > thr).astype(int), pred, float(r2)


def sparse_arm(S, resid, rbin, sp, dataset, seed):
    """§9 rule: rank on fit rows vs the continuous residual; K on validation;
    C=1.0 probe refit on the train draw; test once."""
    from ssep.feature_selection import select_sae_features

    sel = select_sae_features(S[sp["fit"]], resid[sp["fit"]], max(K_GRID),
                              method=rt.common.DATASETS[dataset]["select"],
                              target_kind="continuous", random_state=seed)
    best = None
    for k in K_GRID:
        ids = sel.feature_ids[:k]
        m = LogisticRegression(max_iter=5_000, random_state=seed)
        m.fit(S[sp["fit"]][:, ids], rbin[sp["fit"]])
        a = roc_auc_score(rbin[sp["val"]], m.decision_function(S[sp["val"]][:, ids]))
        if best is None or a > best[1]:
            best = (k, a)
    k = best[0]
    ids = sel.feature_ids[:k]
    m = LogisticRegression(max_iter=5_000, random_state=seed)
    m.fit(S[sp["train"]][:, ids], rbin[sp["train"]])
    te = sp["test"]
    return {"K": k, "val_auroc": best[1],
            **rt.boot_auroc(rbin[te], m.decision_function(S[te][:, ids]), seed=seed)}


def run_site(df, dataset, site_name, layer, pos, seed, sae):
    sp = rt.common.splits(len(df), seed)
    labs = rt.labels(df, sp)
    Xn = rt.load_site(df, dataset, layer, pos, sp)
    S = rt.common.sae_encode(sae, Xn) if sae is not None else None
    tr, te = sp["train"], sp["test"]
    rng = np.random.default_rng(zlib.crc32(f"{dataset}/{site_name}/{seed}/r5".encode()))
    v_rand = rng.standard_normal(Xn.shape[1])
    out = {}
    for spec, (y, cols) in basis_cols(df, labs, pos).items():
        resid, rbin, pred, r2 = residualize(y, cols, tr)
        m, c, _ = rt.D.tuned_lr(Xn, rbin, sp, seed)
        dense = rt.boot_auroc(rbin[te], m.decision_function(Xn[te]), seed=seed)
        y_sh = rbin.copy()
        y_sh[tr] = np.random.default_rng(seed).permutation(rbin[tr])
        ms, _, _ = rt.D.tuned_lr(Xn, y_sh, sp, seed)
        row = {
            "basis_R2": r2, "dense_C": c, "dense": dense,
            "ctl_floor": rt.boot_auroc(rbin[te], pred[te], seed=seed),
            "shuffled": float(rt._rank_auroc(rbin[te], ms.decision_function(Xn[te]))),
            "random_dir": float(rt._rank_auroc(rbin[te], Xn[te] @ v_rand)),
        }
        if S is not None:
            row["sparse"] = sparse_arm(S, resid, rbin, sp, dataset, seed)
        out[spec] = row
        print(f"  {dataset}/{site_name}/s{seed} {spec}: R²={r2:.3f} "
              f"dense {dense['auroc']:.4f} [{dense['lo']:.4f},{dense['hi']:.4f}] "
              f"ctl {row['ctl_floor']['auroc']:.4f} "
              f"sparse {row.get('sparse', {}).get('auroc', float('nan')):.4f}", flush=True)
    return out


def run_dataset(dataset):
    df = rt.common.load_df(dataset)
    site_map = rt.sites(dataset)
    payload = {"sites": {k: list(v) for k, v in site_map.items()}}
    done, sae_cache = {}, {}
    for site_name, (layer, pos) in site_map.items():
        if layer not in sae_cache:
            try:
                sae_cache[layer] = rt.common.load_sae(layer)
            except Exception as err:  # no canonical 16k SAE at this layer
                print(f"NOTE: no SAE at L{layer} ({err}); sparse arm skipped there")
                sae_cache[layer] = None
        for seed in rt.common.SEEDS:
            key = f"{site_name}_seed{seed}"
            if (layer, pos, seed) in done:
                payload[key] = {"alias_of": done[(layer, pos, seed)]}
                continue
            payload[key] = run_site(df, dataset, site_name, layer, pos, seed, sae_cache[layer])
            done[(layer, pos, seed)] = site_name
    return payload


if __name__ == "__main__":
    for dataset in (sys.argv[1:] or list(rt.common.DATASETS)):
        rt.save_result(f"r5_{dataset}", run_dataset(dataset))
