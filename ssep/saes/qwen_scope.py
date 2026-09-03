"""Custom loader for the official Qwen-Scope SAE release (arXiv 2605.11887).

Checkpoint contract, verified against the HF model card of
`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50` (fetched 2026-08-12):

  - one file per layer, `layer{n}.sae.pt` — a torch dict of exactly four tensors:
        W_enc (d_sae, d_in)   b_enc (d_sae,)
        W_dec (d_in, d_sae)   b_dec (d_in,)
    (Qwen3-8B-Base: d_in=4096, d_sae=65536, layers 0-35, hook = resid_post.)
  - encode, card-verbatim (NO pre-encode b_dec subtraction, NO ReLU):
        pre = x @ W_enc.T + b_enc
        vals, idx = pre.topk(k, dim=-1); acts = zeros.scatter(idx, vals)
    TopK keeps the k largest raw pre-activations, so realized L0 (nonzeros per
    row) == k almost surely — the §G.3 validation signature.
  - decode: acts @ W_dec.T + b_dec (the only shape-consistent orientation).

The loader introspects shapes rather than trusting orientation: b_enc/b_dec
lengths define (d_sae, d_in) and each W is transposed into card-native
orientation if it arrives flipped, with a hard error when d_sae == d_in makes
that ambiguous. Everything upcasts to float32 (card demo runs float32).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

QWEN_SCOPE_KEYS = frozenset({"W_enc", "W_dec", "b_enc", "b_dec"})
CONVENTION = "qwen-scope-modelcard-2026-08 (pre=x@W_enc.T+b_enc; topk scatter; no pre-bias)"


class QwenScopeFormatError(RuntimeError):
    pass


def parse_k_from_repo_id(repo_id: str) -> int | None:
    m = re.search(r"L0[_-](\d+)", repo_id)
    return int(m.group(1)) if m else None


def _orient(name: str, w: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Return `w` shaped (dim0, dim1), transposing if it arrived flipped."""
    if w.ndim != 2:
        raise QwenScopeFormatError(f"{name}: expected 2-D tensor, got shape {tuple(w.shape)}")
    if tuple(w.shape) == (dim0, dim1):
        if dim0 == dim1:
            raise QwenScopeFormatError(
                f"{name}: d_sae == d_in == {dim0}; orientation is ambiguous — refusing to guess"
            )
        return w
    if tuple(w.shape) == (dim1, dim0):
        return w.T.contiguous()
    raise QwenScopeFormatError(
        f"{name}: shape {tuple(w.shape)} matches neither ({dim0},{dim1}) nor ({dim1},{dim0})"
    )


@dataclass
class QwenScopeSAE:
    W_enc: torch.Tensor  # (d_sae, d_in), card-native
    W_dec: torch.Tensor  # (d_in, d_sae)
    b_enc: torch.Tensor  # (d_sae,)
    b_dec: torch.Tensor  # (d_in,)
    k: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def d_in(self) -> int:
        return int(self.b_dec.shape[0])

    @property
    def d_sae(self) -> int:
        return int(self.b_enc.shape[0])

    @classmethod
    def from_state_dict(
        cls, sd: dict[str, torch.Tensor], *, k: int, source: str = "<memory>"
    ) -> QwenScopeSAE:
        keys = set(sd.keys())
        if keys != QWEN_SCOPE_KEYS:
            missing, extra = QWEN_SCOPE_KEYS - keys, keys - QWEN_SCOPE_KEYS
            raise QwenScopeFormatError(
                f"{source}: checkpoint keys mismatch — missing {sorted(missing)}, "
                f"unexpected {sorted(extra)}"
            )
        b_enc, b_dec = sd["b_enc"], sd["b_dec"]
        if b_enc.ndim != 1 or b_dec.ndim != 1:
            raise QwenScopeFormatError(
                f"{source}: b_enc/b_dec must be 1-D, got {tuple(b_enc.shape)}/{tuple(b_dec.shape)}"
            )
        d_sae, d_in = int(b_enc.shape[0]), int(b_dec.shape[0])
        if not 1 <= k <= d_sae:
            raise QwenScopeFormatError(f"{source}: k={k} outside [1, d_sae={d_sae}]")
        orig_dtype = str(sd["W_enc"].dtype)
        sae = cls(
            W_enc=_orient("W_enc", sd["W_enc"], d_sae, d_in).float(),
            W_dec=_orient("W_dec", sd["W_dec"], d_in, d_sae).float(),
            b_enc=b_enc.float(),
            b_dec=b_dec.float(),
            k=k,
            metadata={
                "source": source,
                "convention": CONVENTION,
                "checkpoint_dtype": orig_dtype,
                "d_in": d_in,
                "d_sae": d_sae,
                "k": k,
            },
        )
        for name, t in (("W_enc", sae.W_enc), ("W_dec", sae.W_dec),
                        ("b_enc", sae.b_enc), ("b_dec", sae.b_dec)):
            if not torch.isfinite(t).all():
                raise QwenScopeFormatError(f"{source}: non-finite values in {name}")
        return sae

    @classmethod
    def from_file(cls, path: str | Path, *, k: int) -> QwenScopeSAE:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict):
            raise QwenScopeFormatError(f"{path}: expected a dict checkpoint, got {type(sd)}")
        return cls.from_state_dict(sd, k=k, source=str(path))

    @classmethod
    def from_hf(
        cls, repo_id: str, layer: int, *, revision: str | None = None, k: int | None = None
    ) -> QwenScopeSAE:
        from huggingface_hub import hf_hub_download

        parsed = parse_k_from_repo_id(repo_id)
        if k is None:
            if parsed is None:
                raise QwenScopeFormatError(
                    f"{repo_id}: cannot parse k (no 'L0_<k>' in repo id); pass k explicitly"
                )
            k = parsed
        elif parsed is not None and parsed != k:
            raise QwenScopeFormatError(f"{repo_id}: explicit k={k} contradicts repo id L0_{parsed}")
        path = hf_hub_download(repo_id, f"layer{layer}.sae.pt", revision=revision)
        sae = cls.from_file(path, k=k)
        sae.metadata.update({"repo_id": repo_id, "layer": layer, "revision": revision})
        return sae

    def to(self, device: str | torch.device) -> QwenScopeSAE:
        for name in ("W_enc", "W_dec", "b_enc", "b_dec"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(..., d_in) -> (..., d_sae) with exactly k nonzeros per row (card-verbatim)."""
        if x.shape[-1] != self.d_in:
            raise QwenScopeFormatError(f"encode: last dim {x.shape[-1]} != d_in {self.d_in}")
        pre_acts = x.float() @ self.W_enc.T + self.b_enc
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts

    @torch.no_grad()
    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.shape[-1] != self.d_sae:
            raise QwenScopeFormatError(f"decode: last dim {feats.shape[-1]} != d_sae {self.d_sae}")
        return feats.float() @ self.W_dec.T + self.b_dec
