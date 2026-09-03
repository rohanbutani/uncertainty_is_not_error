# Uncertainty Is Not Error — Reproduction Package

This anonymous package contains the smallest path used to regenerate the reported paper
figures and appendix tables from the frozen analysis outputs. It includes no raw activation
store: the original Azure activation data are external inputs and are not redistributed.

## Environment

Use Python 3.12 and the pinned `uv.lock`:

```bash
uv sync --frozen
```

## Execution order

From the package root:

```bash
uv run python analysis/paper_final_figures.py
uv run python -m ssep.analysis.paper.appendix_figures \
  --results-root results/paper \
  --analysis-root data/analysis/paper \
  --output regenerated/appendix
```

The first command regenerates the two main-text figure PDFs and PNGs. The second regenerates
the appendix figures and tables from the included frozen result ledgers and CSV/JSON outputs.
The included `analysis/final_validation.py` is the confirmatory recomputation path; running it
requires the external row-level activation/data stores referenced by its manifests and the
Gemma-2-9B and Gemma Scope checkpoints. Those inputs are intentionally not bundled.

All paths are relative to this package. No credentials, absolute paths, git metadata, caches,
or author-identifying files are included.
