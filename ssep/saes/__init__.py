"""SAE loading + sanity validation (schema §11: encode-on-the-fly; §G.3 / parity
test (d) — "the silent-killer catch").

Suites: Gemma Scope / Llama Scope through SAELens (verified release ids in
docs/lit/18-resources.md, e.g. release="gemma-scope-9b-pt-res-canonical",
sae_id="layer_20/width_16k/canonical"); Qwen-Scope through the custom loader in
`ssep.saes.qwen_scope` (raw 4-tensor dicts, not SAELens-native). sae_lens is
imported lazily — it pulls in transformer_lens, heavy and unneeded elsewhere.

Validation: realized L0 and reconstruction quality on a residual slab must land
in the release's reported range; for Qwen-Scope TopK, realized L0 == k validates
the loader/BOS/hook chain end-to-end and orientation errors are catastrophic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


class SaeLensAdapter:
    def __init__(self, release: str, sae_id: str, *, device: str = "cpu") -> None:
        from sae_lens import SAE  # lazy: heavy import chain

        loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        # from_pretrained returned (sae, cfg_dict, sparsity) in sae_lens<=5;
        # newer versions may return the SAE alone — accept both.
        self.sae = loaded[0] if isinstance(loaded, tuple) else loaded
        self.metadata: dict[str, Any] = {
            "suite": "sae_lens",
            "release": release,
            "sae_id": sae_id,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
        }

    @property
    def d_in(self) -> int:
        return int(self.sae.cfg.d_in)

    @property
    def d_sae(self) -> int:
        return int(self.sae.cfg.d_sae)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x.to(dtype=self.sae.dtype))

    @torch.no_grad()
    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.sae.decode(feats.to(dtype=self.sae.dtype))


def load_sae(
    suite: Literal["sae_lens", "qwen_scope"],
    *,
    # sae_lens suite (Gemma Scope / Llama Scope):
    release: str | None = None,
    sae_id: str | None = None,
    # qwen_scope suite:
    repo_id: str | None = None,
    layer: int | None = None,
    revision: str | None = None,
    k: int | None = None,
    device: str = "cpu",
):
    if suite == "sae_lens":
        if release is None or sae_id is None:
            raise ValueError("sae_lens suite requires release and sae_id")
        return SaeLensAdapter(release, sae_id, device=device)
    if suite == "qwen_scope":
        if repo_id is None or layer is None:
            raise ValueError("qwen_scope suite requires repo_id and layer")
        from ssep.saes.qwen_scope import QwenScopeSAE

        return QwenScopeSAE.from_hf(repo_id, layer, revision=revision, k=k).to(device)
    raise ValueError(f"unknown SAE suite: {suite!r}")


@dataclass(frozen=True)
class SaeValidationReport:
    d_in: int
    d_sae: int
    n_vectors: int
    realized_l0_mean: float
    realized_l0_median: float
    recon_rel_err: float  # ||x - x_hat|| / ||x||, mean over vectors
    cos_sim_mean: float


@torch.no_grad()
def validate_sae(adapter, x: torch.Tensor) -> SaeValidationReport:
    """x: (n, d_in) residual vectors (any float dtype; upcast to f32 here)."""
    if x.ndim != 2:
        raise ValueError(f"expected (n, d_in), got shape {tuple(x.shape)}")
    x = x.float()
    feats = adapter.encode(x).float()
    x_hat = adapter.decode(feats).float()

    l0 = (feats != 0).sum(dim=-1).float()
    rel_err = (x - x_hat).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-12)
    cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)

    return SaeValidationReport(
        d_in=int(x.shape[1]),
        d_sae=int(feats.shape[1]),
        n_vectors=int(x.shape[0]),
        realized_l0_mean=float(l0.mean()),
        realized_l0_median=float(l0.median()),
        recon_rel_err=float(rel_err.mean()),
        cos_sim_mean=float(cos.mean()),
    )
