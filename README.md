# Are LLM Uncertainty and Correctness Encoded by the Same Features?

This repository contains the code and frozen analysis artifacts for **“Are LLM Uncertainty and Correctness Encoded by the Same Features? A Functional Dissociation via Sparse Autoencoders.”** The project studies whether uncertainty and correctness are represented by the same features in large language models, using sparse autoencoder (SAE) representations, probing, feature selection, erasure, and controlled readouts.

The experiments use Gemma-2-9B activations on TriviaQA and PopQA. The analyses measure semantic-entropy prediction, correctness prediction, cross-dataset feature transfer, representation comparisons, dissociation tests, and risk-coverage behavior. This repository includes reported result tables, JSON ledgers, manifests, figures, and scripts used to regenerate the paper artifacts.

## Main findings

- SAE features contain robust signal about semantic uncertainty.
- Top-k feature selection improves probe performance substantially over an unselected dense baseline.
- In the primary within-dataset comparison, the SAE representation contributes little additional accuracy beyond feature selection; its principal value is interpretability.
- Correctness and uncertainty are not exhausted by a single representation: the dissociation analyses test residual signal, alternative uncertainty measures, and held-out feature behavior.

These conclusions are protocol- and dataset-dependent. See [`paper_inputs/RESULTS.md`](paper_inputs/RESULTS.md) and [`paper_inputs/RETEST_DISSOCIATION_RESULTS.md`](paper_inputs/RETEST_DISSOCIATION_RESULTS.md) for exact configurations, caveats, and confidence intervals.

## Repository layout

| Path | Contents |
|---|---|
| `ssep/` | SSEP package and paper-analysis modules |
| `analysis/` | Entry points for final validation and paper figures |
| `results/paper/` | Frozen probe, study, comparison, and figure artifacts |
| `data/analysis/paper/` | Frozen analysis summaries used by figure scripts |
| `paper_inputs/` | Result ledgers and detailed experimental notes |
| `configs/` | Experiment configurations |
| `MANIFEST.md` | Mapping from paper artifacts to reproduction code and inputs |

## Reproducing the included artifacts

The project requires Python 3.12. Install the locked environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
```

Regenerate the main paper figures:

```bash
uv run python analysis/paper_final_figures.py
```

Regenerate appendix figures and tables:

```bash
uv run python -m ssep.analysis.paper.appendix_figures \
  --results-root results/paper \
  --analysis-root data/analysis/paper \
  --output regenerated/appendix
```

The confirmatory validation path is:

```bash
uv run python analysis/final_validation.py
```

That path requires the external row-level activation stores and model/SAE checkpoints referenced by its manifests. Those large inputs are not redistributed here. The included frozen outputs are sufficient for the offline figure and table regeneration commands above.

## Data and reproducibility

Raw activations, model checkpoints, credentials, caches, and absolute local paths are not included, but are available upon request. Result manifests record the source runs and expected inputs. The repository is intended to make the analysis transparent and auditable while keeping large or access-controlled data external.

