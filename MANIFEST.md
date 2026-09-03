# Result-to-code manifest

| Paper artifact | Reproduction code | Included inputs |
|---|---|---|
| Figure 1A–D | `analysis/paper_final_figures.py` (`validation_ladder`) | `paper_inputs/` ledgers |
| Figure 2A–D | `analysis/paper_final_figures.py` (`limits_and_actionability`) | `results/paper/final_validation/`, ledgers |
| Appendix Figure A1 | `ssep/analysis/paper/appendix_figures.py` | `results/paper/appendix_figures_v1/`, paper ledgers |
| Appendix Figure A2 | `ssep/analysis/paper/appendix_figures.py` | same |
| Appendix Figure A3 | `ssep/analysis/paper/appendix_figures.py` | same |
| Appendix Figure A4 | `ssep/analysis/paper/appendix_figures.py` | same |
| Appendix Figure A5 | `ssep/analysis/paper/appendix_figures.py` | same |
| Appendix Figure A6 | `ssep/analysis/paper/appendix_figures.py` | same |
| Appendix Tables A1–A8 | `ssep/analysis/paper/appendix_figures.py` | `results/paper/appendix_figures_v1/*.csv` |
| TBG/SLT increments and risk coverage | `analysis/final_validation.py` (`readout`) | `results/paper/final_validation/`, manifests |
| Held-out SAE confirmation | `analysis/final_validation.py` (`confirm_features`) | SAE/data paths in manifests |
| Conditional probe and dissociation ledgers | `ssep/analysis/paper/{probe,study_a,compare}.py` | `results/paper/conditional_*`, probe outputs |

The final PDF itself is generated outside this analysis package from the paper source; all
quantitative values and figure inputs used by that source are included here.
