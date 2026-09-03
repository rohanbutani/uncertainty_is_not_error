"""Shared config + loaders for the offline steering-direction experiments.

Stored data only: the two banked runs' tables and `resid_post` slices, the
Gemma Scope SAEs, and the Gemma-2-9B unembedding. No generation, no GPU needed.

Conventions (pinned here, recorded in STEERING_OFFLINE_RESULTS.md):
- Row order: the FULL prompts.parquet order (n=7402 / 13067). Analysis frames
  drop the p_true few-shot exemplars (-> 7382 / 13047) and carry `_full_row` to
  index into cached slices. Matches the existing `data/cache/longshort` files.
- Slice cache: `data/cache/steering/{dataset}_L{layer}_{pos}.npy`, uint16 bf16
  bitcast, full-table row order. `data/cache/longshort` is consulted first.
- Layer numbering is the notebooks' (= Gemma Scope's): storage index = L + 1.
  L=41 is the last pre-norm residual (verified by the Phase 0 parity gate).
- Splits: notebook convention — rng(seed).permutation, first N_TRAIN=2048 are
  train, rest test; train re-split fit/val with rng(seed+9973), VAL_FRAC=0.25.
- `se_hi` label: se_discrete > sep_best_split(train rows). Quadrants: tail
  quartiles of se_discrete (thresholds from the train draw) x judge_binary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / "data" / "cache" / "tables"
SLICES = REPO / "data" / "cache" / "steering"
LEGACY_SLICES = REPO / "data" / "cache" / "longshort"
RESULTS = REPO / "results" / "steering"

MODEL_SAFE = "google--gemma-2-9b"
D_MODEL = 3584
FINAL_LAYER = 41  # notebook numbering; storage index 42 — gate-1 verified

SEEDS = (42, 43)
N_TRAIN = 2048
VAL_FRAC = 0.25

# Frozen per-dataset knobs. `select/guide/K` are RESULTS.md §1's frozen configs,
# applied at every SAE site of that dataset (no §1 config exists off-TBG; the
# realized choice is recorded in the write-up).
DATASETS = {
    "trivia_qa": {
        "run_group": "fp1-triviaqa-20260812",
        "sites": {"tbg": (40, "tbg"), "slt": (28, "slt"), "han": (30, "slt"),
                  "final_tbg": (FINAL_LAYER, "tbg"), "final_slt": (FINAL_LAYER, "slt")},
        "select": "mutual_information", "guide": "entropy", "k": 64,
    },
    "popqa": {
        "run_group": "fp1-popqa-20260814",
        "sites": {"tbg": (24, "tbg"), "slt": (24, "slt"), "han": (30, "slt"),
                  "final_tbg": (FINAL_LAYER, "tbg"), "final_slt": (FINAL_LAYER, "slt")},
        "select": "spearman", "guide": "label", "k": 32,
    },
}
HAN_COORDS = (1279, 2558)  # Han et al. (OpenReview BbZKxrZCNn), L30 residual

# §3 grid layers for A4's depth-rotation row: cos(v_l, v_{l+4})
A4_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32, 36, 40)


# ---------------------------------------------------------------- tables ----
def load_df(dataset: str) -> pd.DataFrame:
    """Joined prompts+labels, p_true exemplars dropped, `_full_row` kept."""
    prompt_path = TABLES / f"{dataset}_prompts.parquet"
    label_path = TABLES / f"{dataset}_labels.parquet"
    meta_path = TABLES / f"{dataset}_run_metadata.json"
    if not prompt_path.exists():
        table_root = REPO / "data" / "azure" / "tables" / "v1" / MODEL_SAFE / dataset
        prompt_path, label_path = table_root / "prompts.parquet", table_root / "labels.parquet"
        run_group = DATASETS[dataset]["run_group"]
        meta_path = REPO / "data" / "azure" / "runs" / f"{MODEL_SAFE}--{run_group}" / "run_metadata.json"
    prompts = pd.read_parquet(prompt_path)
    labels = pd.read_parquet(label_path)
    meta = json.loads(meta_path.read_text())
    df = prompts.merge(labels, on="prompt_id", suffixes=("", "_lbl"))
    if len(df) != len(prompts):
        raise ValueError(f"{dataset}: join changed row count {len(prompts)}->{len(df)}")
    df["_full_row"] = np.arange(len(df))
    ds_meta = next(d for d in meta["datasets"] if d["name"] == dataset)
    df = df[~df.prompt_id.isin(set(ds_meta["ptrue_fewshot_ids"]))].reset_index(drop=True)
    return df


def positions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Token index of each named position inside the stored window — the full
    §3 grid map (notebook cell 4). `tbg`/`slt` are SEP's two (rule F6)."""
    tbg = df.tbg_index_rel.to_numpy()
    a0 = df.answer_start_token_rel.to_numpy()
    ns = df.greedy_sliced_token_count.to_numpy()
    nf = df.greedy_token_count.to_numpy()
    return {"tbg": tbg, "ans0": a0, "slt": a0 + ns - 1, "eoa": a0 + ns,
            "genlast": a0 + nf - 1, "kw": tbg - 1, "nl": tbg - 2, "qlast": tbg - 3}


def splits(n: int, seed: int, n_train: int = N_TRAIN) -> dict[str, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    is_train = np.zeros(n, dtype=bool)
    is_train[perm[:n_train]] = True
    idx = np.flatnonzero(is_train)
    p = np.random.default_rng(seed + 9973).permutation(len(idx))
    nv = int(round(VAL_FRAC * len(idx)))
    is_val = np.zeros(n, dtype=bool)
    is_val[idx[p[:nv]]] = True
    is_fit = is_train & ~is_val
    return {"train": is_train, "test": ~is_train, "fit": is_fit, "val": is_val}


def se_label(df: pd.DataFrame, is_train: np.ndarray) -> tuple[np.ndarray, float]:
    """se_discrete binarized by sep_best_split fit on the training draw."""
    from ssep.uncertainty.binarize import sep_best_split

    se = df.se_discrete.to_numpy()
    thr = sep_best_split(se[is_train])
    return (se > thr).astype(int), thr


def quadrant_masks(df: pd.DataFrame, is_train: np.ndarray) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
    """Tail-quartile SE x judge_binary. Middle-SE rows fall in no quadrant."""
    from ssep.uncertainty.binarize import tail_quartile_thresholds

    se = df.se_discrete.to_numpy()
    lo, hi = tail_quartile_thresholds(se[is_train])
    j = df.judge_binary_label.to_numpy()
    return {
        "hiSE_correct": (se >= hi) & (j == 1), "hiSE_wrong": (se >= hi) & (j == 0),
        "loSE_correct": (se <= lo) & (j == 1), "loSE_wrong": (se <= lo) & (j == 0),
    }, (lo, hi)


def entropy_at(df: pd.DataFrame, pos_name: str) -> np.ndarray:
    """Next-token entropy the state at this position emits (§9 indexing:
    greedy_entropy_full[p - answer_start + 1]; tbg -> 0, slt -> sliced_count)."""
    ent = df.greedy_entropy_full.to_numpy()
    j = np.clip(positions(df)[pos_name] - df.answer_start_token_rel.to_numpy() + 1, 0, None)
    return np.array([e[min(int(i), len(e) - 1)] for e, i in zip(ent, j, strict=True)], dtype=np.float64)


# ---------------------------------------------------------------- slices ----
def _remote_array(dataset: str):
    import zarr
    local = REPO / "data" / "azure" / "acts" / "v1" / MODEL_SAFE / dataset / "resid_post"
    if local.exists():
        return zarr.open_array(local, mode="r")
    from dotenv import load_dotenv
    from obstore.store import AzureStore
    from zarr.storage import ObjectStore

    load_dotenv(REPO / ".env")
    azure = AzureStore(
        os.environ["AZURE_STORAGE_CONTAINER"],
        account_name=os.environ["AZURE_STORAGE_ACCOUNT"],
        sas_key=os.environ["AZURE_STORAGE_SAS_TOKEN"].lstrip("?"),
        credential_type="sas_token",
    )
    return zarr.open_array(
        store=ObjectStore(azure, read_only=True),
        path=f"acts/v1/{MODEL_SAFE}/{dataset}/resid_post", mode="r",
    )


def fetch_layer(dataset: str, layer: int, pos_names: list[str]) -> None:
    """One remote pass over a layer, extracting every requested position at once.

    Slices are cached in FULL-table row order so they are split-independent.
    """
    missing = [p for p in pos_names if _slice_path(dataset, layer, p) is None]
    if not missing:
        return
    prompt_path = TABLES / f"{dataset}_prompts.parquet"
    if not prompt_path.exists():
        prompt_path = REPO / "data" / "azure" / "tables" / "v1" / MODEL_SAFE / dataset / "prompts.parquet"
    prompts = pd.read_parquet(prompt_path)
    pos = positions(prompts)
    arr = _remote_array(dataset)
    n, t = arr.shape[0], arr.shape[2]
    out = {p: np.empty((n, D_MODEL), dtype=np.uint16) for p in missing}
    for s0 in range(0, n, 1024):
        s1 = min(s0 + 1024, n)
        block = arr[s0:s1, layer + 1]  # (rows, T, D) uint16
        for p in missing:
            idx = np.clip(pos[p][s0:s1], 0, t - 1)
            out[p][s0:s1] = block[np.arange(s1 - s0), idx]
    SLICES.mkdir(parents=True, exist_ok=True)
    for p in missing:
        tmp = SLICES / f".{dataset}_L{layer}_{p}.tmp.npy"
        np.save(tmp, out[p])
        tmp.rename(SLICES / f"{dataset}_L{layer}_{p}.npy")
    print(f"fetched {dataset} L{layer} {missing}")


def _slice_path(dataset: str, layer: int, pos_name: str) -> Path | None:
    for root in (SLICES, LEGACY_SLICES):
        f = root / f"{dataset}_L{layer}_{pos_name}.npy"
        if f.exists():
            return f
    return None


def load_slice(df: pd.DataFrame, dataset: str, layer: int, pos_name: str) -> np.ndarray:
    """(n_df, d_model) float32 residual at (layer, position), df row order."""
    f = _slice_path(dataset, layer, pos_name)
    if f is None:
        fetch_layer(dataset, layer, [pos_name])
        f = _slice_path(dataset, layer, pos_name)
    import ml_dtypes

    raw = np.load(f, mmap_mode="r")
    return raw[df._full_row.to_numpy()].view(ml_dtypes.bfloat16).astype(np.float32)


def drop_slices(dataset: str, layers: list[int]) -> None:
    """Cache hygiene: remove per-layer slices no longer needed (steering dir only)."""
    for f in SLICES.glob(f"{dataset}_L*.npy"):
        if int(f.stem.split("_L")[1].split("_")[0]) in layers:
            f.unlink()
            print(f"dropped {f.name}")


def rms_normalize(X: np.ndarray, is_train: np.ndarray) -> tuple[np.ndarray, float]:
    """Rescale rows to the train-mean RMS (the notebooks' SAE_NORM='rms')."""
    r = np.sqrt((X.astype(np.float64) ** 2).mean(axis=1))
    scale = float(r[is_train].mean())
    return (X * (scale / np.maximum(r, 1e-8))[:, None]).astype(np.float32), scale


# -------------------------------------------------------------- SAE / W_U ----
def load_sae(layer: int, device: str = "cpu"):
    from ssep.saes import SaeLensAdapter

    return SaeLensAdapter(
        "gemma-scope-9b-pt-res-canonical", f"layer_{layer}/width_16k/canonical",
        device=device,
    )


def sae_encode(sae, X: np.ndarray, batch: int = 256) -> np.ndarray:
    import torch

    out = np.empty((len(X), sae.d_sae), dtype=np.float32)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.ascontiguousarray(X[i : i + batch]))
        out[i : i + batch] = sae.encode(xb).float().cpu().numpy()
    return out


def load_unembed():
    """(W bf16 torch (vocab, d), norm_weight f32, eps, softcap) from the HF cache."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cfg = json.loads(Path(hf_hub_download("google/gemma-2-9b", "config.json")).read_text())
    tensors = {}
    for shard, names in {
        "model-00001-of-00008.safetensors": ["model.embed_tokens.weight"],
        "model-00008-of-00008.safetensors": ["model.norm.weight"],
    }.items():
        with safe_open(hf_hub_download("google/gemma-2-9b", shard), framework="pt") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)

    return (
        tensors["model.embed_tokens.weight"].to(torch.bfloat16),  # tied lm_head; capture ran bf16
        tensors["model.norm.weight"].float(),
        float(cfg["rms_norm_eps"]),
        float(cfg["final_logit_softcapping"]),
    )


def logprobs_from_resid(h: np.ndarray | object, W, norm_w, eps: float, softcap: float,
                        device: str = "cpu"):
    """Mirror of ssep.models.logits_head: RMSNorm (f32) -> bf16 lm_head -> softcap
    -> f32 log_softmax. h: (n, d) float32 -> (n, vocab) f32 torch logprobs."""
    import torch

    if not torch.is_tensor(h):
        h = torch.from_numpy(np.ascontiguousarray(h))
    h = h.to(device, torch.float32)
    x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    x = (x * (1.0 + norm_w.to(device))).to(torch.bfloat16)
    logits = x @ W.to(device).T
    logits = softcap * torch.tanh(logits / softcap)
    return torch.log_softmax(logits.float(), dim=-1)


# ---------------------------------------------------------------- report ----
def md_table(headers: list[str], rows: list[list]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def save_result(name: str, payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")

    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1, default=default))
    print(f"saved results/steering/{name}.json")
