"""Config-driven infrastructure for the working-paper analyses.

This package is deliberately separate from exploratory notebooks.  It owns
run discovery, artifact provenance, frozen group splits, out-of-fold contracts,
and downstream Study B/C evaluation.  Existing feature-selection and model
capture modules remain the computational backend.
"""

from ssep.analysis.paper.registry import PaperRegistry, RunSpec, load_registry

__all__ = ["PaperRegistry", "RunSpec", "load_registry"]
