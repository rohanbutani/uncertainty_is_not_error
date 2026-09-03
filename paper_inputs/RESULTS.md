# Results — single-layer, single-token SE probes

Every result table from both probe notebooks lives here, so the notebooks stay code.

| notebook | model / dataset | run group |
|---|---|---|
| [`sae_topk_probe_gemma_triviaqa.ipynb`](sae_topk_probe_gemma_triviaqa.ipynb) | Gemma-2-9B / TriviaQA | `fp1-triviaqa-20260812` |
| [`sae_topk_probe_gemma_popqa.ipynb`](sae_topk_probe_gemma_popqa.ipynb) | Gemma-2-9B / PopQA | `fp1-popqa-20260814` |
| [`transfer_probe_gemma.ipynb`](transfer_probe_gemma.ipynb) | both, cross-trained (§10) | both |
| [`entropy_feature_probe_gemma.ipynb`](entropy_feature_probe_gemma.ipynb) | both × {TBG, SLT}, entropy as a feature (§13) | both |
| [`correctness_probe_gemma.ipynb`](correctness_probe_gemma.ipynb) | Gemma-2-9B / TriviaQA, direct correctness target (§14) | `fp1-triviaqa-20260812` |

Both per-dataset notebooks run the identical protocol: one hidden state per prompt at a chosen (layer, token),
target = semantic entropy binarized with SEP's `best_split` (threshold fit on train
only), probe = `LogisticRegression` at sklearn's default **C = 1.0** for **every** arm
(`max_iter` raised only so the solver converges at d = 3584 — convergence, not
regularisation, and it strengthens the baseline). A fresh random draw of `N_TRAIN`
rows is the training set and everything else is test; the stored `split` column is
ignored. The training draw is split again into `is_fit` / `is_val`, **every knob is
chosen on `is_val`**, and test is read once at the end.

Numbers are **validation** AUROC unless marked TEST. **Every table below carries both
datasets, measured by the same harness on the same day** — where a TriviaQA number
differs from the older log in `docs/PROGRESS.md`, the harness value is what is reported
here and the difference is named at the table.

---

## 0. The two datasets are not the same problem

| | TriviaQA | PopQA |
|---|---:|---:|
| prompts after the p_true-exemplar drop | 7,382 | **13,047** |
| greedy answer correct (`judge_binary`) | 76.9% | **30.7%** |
| `f1_squad ≥ 50` rate | 77.8% | 34.1% |
| mean `f1_squad` | 76.0 | 32.4 |
| exact match | 72.1% | 27.7% |
| mean `n_clusters` over 10 samples | 3.45 | **7.08** |
| mean `se_discrete` | 0.786 | **1.692** |
| `se_discrete` best_split threshold | 0.9536 | 1.3257 |
| **high-SE positives after binarisation** | **34.5%** | **75.0%** |
| 10-sample label reliability (Spearman-Brown) | r = .858 | **r = .903** |
| mean sliced greedy answer length | 2.28 tokens | 2.47 tokens |
| stop reason | 7,379 stop_string / 3 max_tokens | **100% stop_string** |

The string channel and the judge agree closely on both datasets (77.8 vs 76.9 on
TriviaQA, 34.1 vs 30.7 on PopQA), which is the cheap sanity check that Script 2's two
independent accuracy channels are measuring the same thing.

PopQA is the confabulation engine of the roster and it shows: three quarters of prompts
are high-entropy, the model is wrong on 69% of them (versus 23% on TriviaQA), it produces
**twice** as many semantic clusters per prompt, and the SE label is measurably cleaner.
Every probe number lands 6–9 AUROC points higher than on TriviaQA. **That is the dataset,
not the method** — read the two studies as a contrast, never as a leaderboard.

One structural quirk worth knowing: on PopQA every greedy generation stops on a
stop-string with exactly one token after the sliced answer, so the `eoa` and `genlast`
positions are **the same token for all 13,047 rows** and their columns are identical
everywhere below. TriviaQA has 3 rows that hit `max_tokens`, so the two positions differ
there.

---

## 1. THE HEADLINE, AND THE CAVEAT THAT GOES WITH IT

Three arms, identical shared knobs, **K re-chosen on validation per seed**, 5 seeds,
`se_discrete`, test read once per seed:

| arm | TriviaQA (L40/TBG, n=5,334) | PopQA (L24/TBG, n=10,999) |
|---|---:|---:|
| **A** SEP dense, all 3,584 dims | .8497 ± .0065 | .9282 ± .0040 |
| **B** dense + top-K over the *same* 3,584 dims — **no SAE** | .8763 ± .0022 | .9441 ± .0020 |
| **C** ours: SAE + top-K | **.8847 ± .0026** | **.9444 ± .0021** |
| selection alone (B − A) | **+.0266** | **+.0159** |
| the SAE (C − B) | **+.0085** | **+.0003** |
| total (C − A) | +.0350 | +.0162 |

**On PopQA the SAE contributes nothing measurable to accuracy — +.0003, an order of
magnitude below the seed spread.** The entire gap over SEP is top-K acting as a
regulariser on any representation. TriviaQA already said 76% of the gap was selection;
PopQA says 98%. Two datasets, the same verdict, stated as strongly as the data allows:

> **Sparsity here buys interpretability, not accuracy. Never quote the gap without this.**
> (Three qualifications, all measured: §10 — out of domain the SAE arm earns ~+.021; §11 —
> the "SAE" being tested here is the *sparse code*, and dropping the JumpReLU mask while
> keeping the same features lifts the SAE's share to +.0135 / +.0046; §12 — see below.)

> **⚠ AND READ §13 BEFORE QUOTING ANY OF IT AS A CEILING.** Every arm above reads the
> hidden state and nothing else. Add one free column — the model's own next-token entropy
> at the probed position, from the same forward pass — and arm **C** goes .8893 → **.9423**
> (TriviaQA) and .9476 → **.9756** (PopQA), 5/5 seeds. The +.053 / +.028 that buys is larger
> than every representation effect in this document combined.

> **⚠ READ §12 BEFORE QUOTING THIS TABLE.** Every arm above is pinned at SEP's published
> **C = 1.0**, which is what SEP-comparability means and is applied equally to all arms. But
> C = 1.0 is far from optimal for a 3,584-dim probe on 2,048 rows, and it is the baseline it
> penalises most. Choose C on validation **for both arms** and the gap does not shrink — it
> **reverses**: TriviaQA .8909 (SEP dense) vs .8901 (ours), PopQA .9531 vs .9473. The
> +.0350 / +.0162 above is a fact about C = 1.0, not about representations.

Per seed:

| seed | TriviaQA A | B | C | K(B) | K(C) | | PopQA A | B | C | K(B) | K(C) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 42 | .8517 | .8774 | .8857 | 128 | 128 | | .9277 | .9415 | .9459 | 32 | 32 |
| 43 | .8571 | .8785 | .8874 | 32 | 64 | | .9237 | .9453 | .9439 | 64 | 32 |
| 44 | .8538 | .8734 | .8845 | 64 | 32 | | .9267 | .9467 | .9466 | 32 | 32 |
| 45 | .8428 | .8775 | .8857 | 64 | 32 | | .9346 | .9433 | .9443 | 32 | 64 |
| 46 | .8430 | .8745 | .8804 | 32 | 16 | | .9282 | .9437 | .9412 | 32 | 32 |

Frozen configs, all chosen on validation before any test read:

| | TriviaQA | PopQA |
|---|---|---|
| layer / token | **40** / TBG | **24** / TBG |
| target / binarisation | `se_discrete` / `sep_best_split` | same |
| N_TRAIN / C / `SAE_NORM` | 2048 / 1.0 / `rms` | same |
| SAE (ours only) | res / 16k / canonical | same |
| `SELECT` / `SELECT_GUIDE` / K | `mutual_information` / entropy / 64 | `spearman` / label / 32 |
| validation argmax | .8855 | .9538 |

---

## 2. Label diagnostics and the noise ceiling (no probe involved)

AUROC of single **stored scalars** against the binarised `se_discrete` label:

| scalar | TriviaQA | PopQA |
|---|---:|---:|
| `judge_binary` (negated) | .7535 | .8263 |
| `f1_squad` (negated) | .7682 | .8327 |
| `greedy_logit_gap[0]` (negated) | .8322 | .8929 |
| `p_true` (negated) | .8757 | .9208 |
| mean greedy-token logprob (negated) | .9235 | .9563 |
| **`greedy_entropy_full[0]` — next-token entropy at TBG** | **.9334** | **.9758** |
| `pe_naive` | .9386 | .9598 |
| LR on 4 of those scalars jointly (TEST) | .9608 | .9836 |

Parametric bootstrap over the stored `semantic_cluster_ids` (verified: recomputing
`se_discrete` from them matches the stored column to 6e-8; index 0 is the greedy answer
per rule F4, so the samples are `[1:11]`). Treat each prompt's observed cluster
proportions as latent, redraw multinomial(10, p), 25 replicates:

| predictor | TriviaQA | PopQA |
|---|---:|---:|
| **a PERFECT predictor that knows the latent SE** | **.9813** | **.9865** ← hard ceiling |
| an independent **10-sample re-run** of the whole pipeline | .9572 | .9707 |
| `pe_naive` (computed *from the same 10 samples*) | .9386 | .9598 |
| the model's own next-token entropy at TBG | .9334 | .9758 |
| **this probe (test)** | **.8847** | **.9444** |

Neither dataset is label-limited. PopQA's probe sits .042 under its ceiling; TriviaQA's
sits .097 under. The bottleneck is the same on both: SE at TBG is essentially
`H(softmax(W_U·norm(h)))`, and decoding that scalar precisely enough from `h` to preserve
its *ranking* is a nonlinear problem that neither prompt count pins down.

> **§13 acts on exactly that sentence** by handing the scalar to the probe instead of making
> it decode one, and moves the last row to **.9530 / .9807** — above the independent
> 10-sample re-run on PopQA. Read this table's ordering with §13 next to it.

*(Two TriviaQA cells moved when this column was re-measured under the shared harness:
mean greedy-token logprob .9084 → .9235 and reliability r = .866 → .858, both because
`docs/PROGRESS.md` used a different token span and a different split-half pairing. The
harness numbers are the ones that are comparable across the two datasets.)*

---

## 3. Where the signal lives — dense arm, validation, N_TRAIN = 2048

13 layers × 8 positions, both datasets, dense probe on the rms-normalised residual.

### TriviaQA

| layer | tbg | ans0 | slt | eoa | genlast | kw | nl | qlast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | .6927 | .6486 | .6499 | .6943 | .6869 | .6949 | .6862 | .6552 |
| 8 | .7315 | .6814 | .7123 | .7059 | .7008 | .7067 | .7052 | .6912 |
| 12 | .7548 | .7155 | .7727 | .7962 | .7961 | .7283 | .7280 | .7254 |
| 16 | .7879 | .7753 | .7847 | .8119 | .8142 | .7492 | .7594 | .7294 |
| 20 | .8005 | .8031 | .7953 | .8313 | .8324 | .7631 | .7780 | .7706 |
| 24 | .7968 | .8001 | .8099 | **.8387** | .8333 | .7886 | .7965 | .7438 |
| 28 | .8138 | .8182 | .8392 | .8277 | .8295 | .7993 | .7843 | .7724 |
| 32 | .8252 | .8123 | .8106 | .8104 | .8157 | .7871 | .7812 | .7542 |
| 36 | .8501 | .8116 | .8078 | .8207 | .8239 | .8034 | .7629 | .7597 |
| 38 | .8508 | .8030 | .8128 | .8302 | .8305 | .8051 | .7366 | .7537 |
| 39 | .8531 | .8238 | .8284 | .8204 | .8208 | .7882 | .7736 | .7634 |
| 40 | **.8592** | .8158 | .8150 | .8122 | .8096 | .8004 | .7730 | .7596 |
| 41 | .8351 | .7932 | .7883 | .8020 | .7980 | .7992 | .7722 | .7518 |

### PopQA

| layer | tbg | ans0 | slt | eoa | genlast | kw | nl | qlast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | .7999 | .6896 | .6981 | .7972 | .7972 | .7878 | .7876 | .7781 |
| 8 | .8850 | .7774 | .7497 | .8295 | .8295 | .8357 | .8656 | .8618 |
| 12 | .9007 | .8594 | .8640 | .8845 | .8845 | .8814 | .8974 | .8767 |
| 16 | .8915 | .8667 | .8672 | .8814 | .8814 | .8804 | .8768 | .8874 |
| 20 | .9141 | .9108 | .9039 | .9033 | .9033 | .8861 | .8888 | .8933 |
| 24 | **.9194** | .8970 | .9062 | .9003 | .9003 | .8910 | .8869 | .8932 |
| 28 | .8960 | .8842 | .8759 | .9082 | .9082 | .8888 | .8744 | .8903 |
| 32 | .9134 | .8377 | .8536 | .8988 | .8988 | .8838 | .8623 | .8908 |
| 36 | .8979 | .8665 | .8795 | .8791 | .8791 | .8840 | .8539 | .8590 |
| 38 | .9144 | .8423 | .8609 | .8739 | .8739 | .9009 | .8730 | .8740 |
| 39 | .9169 | .8732 | .8638 | .8652 | .8652 | .9057 | .8991 | .8938 |
| 40 | .9133 | .8753 | .8650 | .8451 | .8451 | .9085 | .8861 | .8807 |
| 41 | .9148 | .8698 | .8597 | .8494 | .8494 | .9021 | .8870 | .8818 |

**TriviaQA peaks late (L40) and sharply; PopQA peaks early (L24) and flatly** — every
PopQA layer in {20, 24, 32, 38, 39, 40, 41} lands in .913–.919, a spread of .006 on a
512-row validation slice, so layer choice barely matters there. TriviaQA's L40 transfers
to PopQA at .9133, only .006 behind PopQA's own argmax; PopQA's L24 costs TriviaQA .062.
On both datasets **TBG wins the top layers**, but the answer-side ordering differs: on
TriviaQA `eoa` beats SEP's `slt` at every layer up to 24 and they trade above it, while
on PopQA `slt` is the better answer-side site almost everywhere.

*(This grid reads the dense probe after the shared rms transform. `docs/PROGRESS.md`'s
TriviaQA grid was pre-normalisation, which is the whole reason L40/TBG is .8592 here and
.8549 there — the .8549 cell reappears below as the `SAE_NORM="none"` row of §5.)*

**Independently of any probe**, the model's own entropy at each single position:

| position | TriviaQA | PopQA |
|---|---:|---:|
| tbg | **.9334** | **.9758** |
| ans0 | .7269 | .8841 |
| slt | .6157 | .6812 |
| eoa | .6750 | .6040 |
| genlast | .6769 | .6040 |

TBG is fixed on mechanism, not on a probe leaderboard, on both datasets. (This table uses
the notebooks' ceiling-cell indexing, where `slt` reads `greedy_entropy_full[n_sliced−1]`.
§9 needs the distribution each hidden state itself *emits* and therefore indexes one step
later; the two conventions are noted where they are used and are not interchangeable.)

**The token axis** — next-token entropy aggregated over the answer's tokens instead of
just the first:

| aggregate | TriviaQA | PopQA |
|---|---:|---:|
| first token only (= TBG) | .9334 | .9758 |
| **max over the sliced answer** | **.9461** | **.9773** |
| mean over the sliced answer | .9046 | .9315 |

+.013 on TriviaQA, +.0015 on PopQA. PopQA's answers average 2.47 tokens, so there is
almost no token axis left to exploit — **the increment TriviaQA points at barely exists
on PopQA.**

---

## 4. Selector × guide × K (validation, mean of seeds 42/43/44)

### TriviaQA, L40/TBG, res/16k

| selector | guide | K=16 | K=32 | K=64 | K=128 | K=256 |
|---|---|---:|---:|---:|---:|---:|
| `mean_difference` | label | .8788 | .8767 | .8796 | .8694 | .8408 |
| `mean_difference` | entropy | .8798 | .8776 | .8760 | .8731 | .8514 |
| `mann_whitney` | label | .8638 | .8717 | .8800 | .8632 | .8155 |
| `mann_whitney` | entropy | .8714 | .8815 | .8840 | .8750 | .8403 |
| `pearson` | label | .8638 | .8717 | .8800 | .8632 | .8188 |
| `pearson` | entropy | .8632 | .8767 | .8815 | .8691 | .8325 |
| `spearman` | label | .8633 | .8736 | .8776 | .8655 | .8226 |
| `spearman` | entropy | .8763 | .8813 | .8841 | .8753 | .8385 |
| `l1_logistic` | label | .6999 | .8045 | .8088 | .8020 | .8004 |
| `l1_logistic` | entropy | .8569 | .8753 | .8702 | .8600 | .8195 |
| `elastic_net` | label | .6018 | .6878 | .7854 | .7960 | .7629 |
| `elastic_net` | entropy | .8001 | .8095 | .8116 | .8011 | .8193 |
| `mutual_information` | label | .8659 | .8748 | .8752 | .8600 | .8289 |
| `mutual_information` | entropy | .8777 | .8829 | **.8855** | .8790 | .8457 |

Validation argmax **`mutual_information` / entropy / K=64 = .8855**. At 131k the argmax
is `mean_difference` / label / K=32 = **.8830**, so **16k wins**.

### PopQA, L24/TBG, res/16k

| selector | guide | K=16 | K=32 | K=64 | K=128 | K=256 |
|---|---|---:|---:|---:|---:|---:|
| `mean_difference` | label | .9465 | .9524 | .9493 | .9458 | .9177 |
| `mean_difference` | entropy | .9470 | .9489 | .9489 | .9396 | .9193 |
| `mann_whitney` | label | .9452 | .9513 | .9494 | .9392 | .9239 |
| `mann_whitney` | entropy | .9450 | .9470 | .9505 | .9433 | .9119 |
| `pearson` | label | .9452 | .9513 | .9494 | .9392 | .9237 |
| `pearson` | entropy | .9411 | .9471 | .9499 | .9428 | .9233 |
| `spearman` | label | .9438 | **.9538** | .9488 | .9435 | .9203 |
| `spearman` | entropy | .9449 | .9515 | .9537 | .9426 | .9161 |
| `l1_logistic` | label | .6454 | .7916 | .8862 | .9034 | .9153 |
| `l1_logistic` | entropy | .9374 | .9495 | .9475 | .9466 | .9478 |
| `elastic_net` | label | .9255 | .9267 | .9351 | .9271 | .9233 |
| `elastic_net` | entropy | .9028 | .9243 | .9346 | .9261 | .8974 |
| `mutual_information` | label | .9453 | .9529 | .9509 | .9509 | .9485 |
| `mutual_information` | entropy | .9484 | .9496 | .9476 | .9428 | .9166 |

Validation argmax **`spearman` / label / K=32 = .9538**. At 131k the argmax is
`mean_difference` / label / K=32 = **.9515**, so **16k wins here too** — same verdict on
both datasets, and both 131k argmaxes land on `mean_difference` / label.

`mann_whitney` (FDR + |Cohen's d|) and `pearson` give identical rankings on a binary
target, as expected: point-biserial *r* is monotone in Cohen's d at fixed group sizes.

### The entropy guide is a ROBUSTNESS knob — confirmed on both datasets

Spread across the five ranking selectors at K=32:

| dataset / site | guide | min | max | **spread** | `l1_logistic` |
|---|---|---:|---:|---:|---:|
| TriviaQA L40/TBG 16k | label | .8045 | .8767 | .0722 | .8045 |
| TriviaQA L40/TBG 16k | **entropy** | .8753 | .8815 | **.0062** | .8753 |
| TriviaQA L40/TBG 131k | label | .7788 | .8830 | **.1042** | .7788 |
| TriviaQA L40/TBG 131k | **entropy** | .8796 | .8827 | **.0031** | .8827 |
| TriviaQA L28/SLT 16k | label | .6161 | .8318 | **.2157** | .6161 |
| TriviaQA L28/SLT 16k | **entropy** | .8097 | .8292 | **.0195** | .8201 |
| PopQA L24/TBG 16k | label | .7916 | .9538 | **.1622** | .7916 |
| PopQA L24/TBG 16k | **entropy** | .9470 | .9515 | **.0045** | .9495 |
| PopQA L24/TBG 131k | label | .6762 | .9515 | **.2753** | .6762 |
| PopQA L24/TBG 131k | **entropy** | .9383 | .9426 | **.0043** | .9383 |
| PopQA L24/SLT 16k | label | .8933 | .9376 | .0443 | .8933 |
| PopQA L24/SLT 16k | **entropy** | .9208 | .9339 | **.0131** | .9316 |

Ranking against `greedy_entropy_full[0]` — continuous, exactly measured, already stored,
and the physical *cause* of SE — collapses the spread in **every one of the six cells**,
and rescues `l1_logistic` by up to +.26. What it does **not** do is raise the ceiling:

- **PopQA**: at the optimum it is a small **loss** — `spearman` .9538 → .9537,
  `mutual_information` .9529 → .9496.
- **TriviaQA**: a tie for `mean_difference`, **+.011 for `mutual_information`**
  (the selector that most needs a rich target gains most: MI against a binary target is
  capped at H(y) ≈ 0.65 nats, so a one-bit guide throttles the very quantity MI
  estimates).

> **Use the entropy guide when you cannot afford to pick `SELECT` correctly.** It is
> insurance, not accuracy. Both datasets agree; TriviaQA's `mutual_information` gain is
> the exception, not the rule, and it does not transfer.

It is fair on the two counts that matter: it touches no test row, and it needs nothing a
deployed probe lacks — the entropy comes from the same forward pass as the hidden state.

### Selection oracle — saturated on both

Let selection cheat and rank against **every** row's label (or entropy), including test:

| ranking sees | TriviaQA (L40/TBG) | PopQA (L24/TBG) |
|---|---:|---:|
| honest (fit rows, the frozen guide) | **.8858** | **.9517** |
| oracle: every row's label | .8793 | .9513 |
| oracle: every row's entropy | .8825 | .9522 |

A cheating ranking buys **nothing** on either dataset (−.0033 / −.0004 at the argmax K).
The honest ranking is already at its estimation ceiling.

---

## 5. `SAE_NORM` — and the correction the second dataset makes mechanistic

Validation, at each dataset's frozen cell:

| `SAE_NORM` | TriviaQA dense | ours | realised L0 | | PopQA dense | ours | realised L0 |
|---|---:|---:|---:|---|---:|---:|---:|
| `none` | .8549 | .8794 | 200.9 | | .9210 | .9509 | 71.7 |
| **`rms`** (reported) | .8592 | **.8858** | 199.5 | | .9194 | **.9517** | 72.7 |
| `unit` | .8638 | .7767 | **10.6** | | .9292 | .9280 | **7.2** |

RMS-normalising to the train-set mean RMS is worth **+.006 on TriviaQA** and **+.0008 on
PopQA** — a small effect, close to a no-op on PopQA.

> ⚠️ **The unit-norm correction, now with its mechanism, on both datasets.** An early
> TriviaQA note claimed row-normalising to *unit* norm was worth +.035 and credited it to
> the RMSNorm argument. That was wrong — it is back-door regularisation, and the machinery
> is visible in the L0 column: unit-norming drives realised L0 from **199.5 → 10.6**
> (TriviaQA) and **72.7 → 7.2** (PopQA), because shrinking the vector to O(1/√d) pushes
> almost every feature under its JumpReLU threshold. The SAE arm then loses **.109** on
> TriviaQA and **.024** on PopQA, while the *dense* arm — which has no thresholds to fall
> under, only a fixed C — gains on both. Only the scale-preserving version is used
> anywhere.

---

## 6. Knobs swept and closed

**SAE sparsity ladder** (each dataset's frozen cell, res/16k, val, K chosen per rung):

| TriviaQA `average_l0` | 10 | 18 | 32 | 61 | **125 (=canonical)** | 292 |
|---|---:|---:|---:|---:|---:|---:|
| realised L0 at TBG | 13.9 | 23.7 | 40.1 | 82.4 | **199.5** | 535.6 |
| best val AUROC | .8558 | .8711 | .8736 | .8758 | **.8858** | .8720 |

| PopQA `average_l0` | 10 | 19 | 34 | 61 | **114 (=canonical)** | 234 |
|---|---:|---:|---:|---:|---:|---:|
| realised L0 at TBG | 6.2 | 11.8 | 22.2 | 38.8 | **72.7** | 149.8 |
| best val AUROC | .8993 | .9380 | .9401 | .9380 | **.9517** | .9523 |

AUROC rises with SAE density and **flattens at the canonical rung on both** — PopQA gains
+.0006 beyond it, TriviaQA loses .014. Read alongside the decomposition, both say the same
thing: the closer the sparse code gets to the dense residual, the better. *(This corrects
`docs/PROGRESS.md`'s "rises monotonically to L0 535" for TriviaQA — under the shared
harness the L0-535 rung is worse than canonical, and the shape matches PopQA's.)*

**SAE site and width** (frozen cell, val):

| suite | width | TriviaQA | PopQA |
|---|---|---:|---:|
| `res` | 16k | **.8858** | .9517 |
| `res` | 131k | .8830 | .9515 |
| `mlp` | 16k | .8801 | **.9555** |
| `mlp` | 131k | .8824 | .9505 |

> ⚠️ **Flagged, not buried:** the `mlp` suite beats `res` on PopQA by **+.0038** and loses
> to it on TriviaQA by **−.0057**. The reported arm stays on `res` for both — it is the
> residual stream the first-pass store holds, the site the SEP-comparable literature
> probes, and the one both notebooks share. Switching to `mlp` would buy ~+.004 on one
> dataset, cost ~.006 on the other, and cost cross-dataset comparability everywhere.
> Gemma Scope's `att` SAEs remain unusable here: they read the 4,096-dim concatenated
> per-head `z`, which the store does not hold.

**Target scalar** (frozen cell, val):

| `SE_SCALAR` | TriviaQA pos. | dense | ours | | PopQA pos. | dense | ours |
|---|---:|---:|---:|---|---:|---:|---:|
| **`se_discrete`** (SEP's target) | 34.5% | .8592 | **.8858** | | 75.0% | .9194 | **.9517** |
| `se_lnrao` (Nature headline) | 28.2% | .8481 | .8639 | | 67.6% | .8924 | .9296 |
| `se_mc` | 62.6% | .8117 | .8556 | | 83.2% | .9271 | .9524 |
| `pe_naive` | 33.2% | .8677 | .8977 | | 66.2% | .9433 | .9541 |

`se_lnrao` is worse for every arm on **both** datasets. (`pe_naive` scores higher, but it
is a *different, easier target*, not a better probe: it is token-level entropy aggregated
over samples, not a semantic quantity.)

**N_TRAIN learning curve** (frozen cell, res/16k, val):

| N_TRAIN | 512 | 1024 | 2048 | 3072 | 4096 | 5120 | 6000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TriviaQA SEP dense | .8441 | .8987 | .8592 | .8323 | .8397 | .8156 | .8167 |
| TriviaQA ours | .8962 | .9120 | **.8858** | .8833 | .8871 | .8822 | .8860 |
| PopQA SEP dense | .9496 | .9079 | .9194 | .9190 | .9060 | .9146 | .9051 |
| PopQA ours | .9593 | .9248 | **.9517** | .9473 | .9429 | .9440 | .9416 |

Flat-to-declining for both arms past ~2k rows on both datasets; the dense arm degrades
faster (at C=1.0 with d=3,584 and residual norms ~10³ it interpolates). The small-N cells
are noisy because the validation slice shrinks with the draw. **Not data-limited.**

**Supervision per row:** multinomial LR on Q SE quantile bins, collapsed back to
P(high SE), versus the binary probe:

| Q | 2 | 4 | 6 | 10 | binary probe |
|---|---:|---:|---:|---:|---:|
| TriviaQA | .8878 | .8782 | .8820 | .8840 | .8858 |
| PopQA | .9435 | .9485 | .9476 | .9474 | .9517 |

**Flat on both — the probe is not supervision-limited.**

**Train-side label denoising** (drop the middle band of the training draw's SE
distribution; evaluation untouched):

| dropped | TriviaQA dense | ours | | PopQA dense | ours |
|---|---:|---:|---|---:|---:|
| 0% | .8592 | **.8858** | | .9194 | **.9517** |
| 20% | .8701 | .8870 | | .9204 | .9509 |
| 40% | .8912 | .8863 | | .9298 | .9485 |

Helps the dense arm on both (TriviaQA +.032, and at 40% it overtakes ours), never helps
ours. Not our route — it is a baseline improvement, and it is reported as one.

---

## 7. It is not the probe class, and not the fixed-C rule

Six learner families, richest representations, each dataset's frozen cell, 4,500 fit rows
(N_TRAIN = 6000), val:

| learner | representation | TriviaQA | PopQA |
|---|---|---:|---:|
| RandomForest(500) | top-2048 SAE features | **.8863** | **.9468** |
| HistGradientBoosting(400) | full residual (3,584 dims) | .8860 | .9439 |
| HistGradientBoosting(400) | top-2048 SAE features | .8838 | .9447 |
| MLP(512) | top-2048 SAE features | .8623 | .9234 |
| MLP(256,128) | standardised residual | .8604 | .9265 |
| kNN(k=100, cosine) | residual | .8161 | .9258 |
| **our logistic probe** | **K SAE features** | **.8860** | **.9416** |
| SEP dense logistic | residual | .8172 | .9049 |

**Nothing beats .8863 / .9468**, and the logistic probe on 64 (TriviaQA) / 32 (PopQA) SAE
features lands within **.0003** and **.005** of the best of them.

C sweep (forbidden as a method; run once as a diagnostic), N_TRAIN = 6000, every arm
equally:

| arm | C=1e-5 | 1e-4 | 1e-3 | 1e-2 | 0.1 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| TriviaQA SEP dense (d=3584) | .8857 | .8561 | .8315 | .8223 | .8181 | .8172 |
| TriviaQA ours 16k K=32 | .8798 | .8819 | .8813 | .8809 | .8809 | .8808 |
| TriviaQA ours 16k K=512 | **.8862** | .8845 | .8762 | .8719 | .8707 | .8708 |
| TriviaQA ours 16k K=8192 | .8841 | .8710 | .8481 | .8379 | .8335 | .8306 |
| PopQA SEP dense (d=3584) | .9441 | **.9497** | .9483 | .9366 | .9161 | .9049 |
| PopQA ours 16k K=32 | .9264 | .9336 | .9399 | .9415 | .9416 | .9416 |
| PopQA ours 16k K=512 | .9326 | .9426 | .9474 | .9454 | .9379 | .9226 |
| PopQA ours 16k K=8192 | .9355 | .9442 | .9476 | .9408 | .9179 | .8935 |

Relaxing the no-tuning rule moves **the baseline** most: SEP dense climbs .8172 → .8857
on TriviaQA and .9049 → .9497 on PopQA, while our arm barely moves. The best cell anywhere
is a tie on TriviaQA (ours K=512 .8862 vs dense .8857) and the **dense** arm on PopQA
(.9497). Either way it *shrinks* the gap rather than opening one, and no cell clears the
frontier the rest of this document establishes. The "SAE as a large nonlinear
random-feature basis" hypothesis stays dead.

---

## 8. Binarisation rule (schema F9)

**Test split**, each dataset at its own frozen cell, knobs already frozen:

| rule | TriviaQA n | dense | ours | | PopQA n | dense | ours |
|---|---:|---:|---:|---|---:|---:|---:|
| F9b `sep_best_split` (SEP-comparable) | 5,334 | .8517 | **.8848** | | 10,999 | .9277 | **.9459** |
| F9a `tail_quartile` (middle dropped) | 2,928 | **.9709** | .9635 | | 6,228 | .9904 | **.9911** |

`tail_quartile` clears .96/.99 on both — and on TriviaQA it **inverts** the result, handing
the baseline .9709 against our .9635. On PopQA it does not invert (.9911 vs .9904), but
both arms sit at .99 because dropping the ambiguous middle half makes the task nearly
trivial. `sep_best_split` remains the reported rule on both datasets: it is what
SEP-protocol comparability requires and it does not manufacture the number.

---

## 9. Does the probe know anything token entropy does not already say?

**The objection this experiment exists to kill.** *"The model's own next-token entropy
predicts SE at .93 (TriviaQA) / .98 (PopQA) for free. Your SAE features are just an
expensive, worse way of reading out predictive entropy — you have built a
symbol-spread detector wearing a meaning-detector's clothes."* If true, that is fatal to
the project's premise.

**The test.** Partial token entropy out of the target and re-run the probe on what is
left. Fit `SE ~ basis` by OLS on the **training rows only**, apply those frozen
coefficients everywhere, binarise the residual at the **training median**, rank features
against the continuous residual on fit rows, probe, read test once. Two control bases:

- **E** — the single-position next-token entropy **at this site** (the plain objection).
- **E+** — `E`, `E²`, `p_true`, `greedy_logit_gap` at this site: the whole
  output-distribution summary the store holds. This is the version worth defending.

Entropy indexing matters: `greedy_entropy_full[j]` is the distribution that *emitted*
generated token j, so the readout of the hidden state at position p is index
`p − answer_start + 1` (TBG → 0, SLT → `sliced_count`). The control has to be the
distribution *this* hidden state produces.

`entropy-ctl` is the sanity check — token entropy's own AUROC against the residual label.

Run on the **top-2 validation configs at each site, per dataset** (8 configs; the site's
best layer comes from §3, the selector/K from that site's own grid).

### TriviaQA

| site | layer | selector / K | val | raw SE (ours) | basis | **R²** | dense | **ours** | entropy-ctl |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| TBG | 40 | `mutual_information` K64 | .8855 | .8815 | E | .709 | .6105 | .6360 | .6148 |
| | | | | | **E+** | **.765** | .5624 | **.6015** | .5561 |
| TBG | 40 | `spearman` K64 | .8841 | .8859 | E | .709 | .6105 | .6561 | .6148 |
| | | | | | **E+** | **.765** | .5624 | **.5853** | .5561 |
| SLT | 28 | `mean_difference` K64 | .8470 | .8459 | E | .070 | .7868 | .8085 | .6045 |
| | | | | | **E+** | **.511** | .6707 | **.7063** | .5788 |
| SLT | 28 | `spearman` K64 | .8391 | .8438 | E | .070 | .7868 | .8057 | .6045 |
| | | | | | **E+** | **.511** | .6707 | **.7038** | .5788 |

### PopQA

| site | layer | selector / K | val | raw SE (ours) | basis | **R²** | dense | **ours** | entropy-ctl |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| TBG | 24 | `spearman` K32 | .9538 | .9459 | E | .824 | .6943 | .7219 | .5687 |
| | | | | | **E+** | **.858** | .6023 | **.6575** | **.5039** |
| TBG | 24 | `mutual_information` K32 | .9529 | .9453 | E | .824 | .6943 | .7247 | .5687 |
| | | | | | **E+** | **.858** | .6023 | **.6587** | **.5039** |
| SLT | 24 | `mean_difference` K64 | .9378 | .9367 | E | .008 | .8983 | .9216 | .6178 |
| | | | | | **E+** | **.520** | .6868 | **.7305** | .5319 |
| SLT | 24 | `pearson` K64 | .9372 | .9350 | E | .008 | .8983 | .9182 | .6178 |
| | | | | | **E+** | **.520** | .6868 | **.7134** | .5319 |

### What it says

**1. The site story is exactly as theorised, and it is the strongest part of the result.**
The basis R² is the whole argument in one column. At **TBG** the model has generated
nothing, so "how much will my samples disagree in meaning" is very nearly "how spread is
my next-token distribution": token entropy alone explains **71% (TriviaQA) / 82% (PopQA)**
of SE's variance. At **SLT** the model has committed to a full answer and the two
entropies come apart: the same single scalar explains **7% (TriviaQA) / 0.8% (PopQA)**.
That is a 10–100× collapse, in the direction theory predicts, on both datasets
independently. Confident-sounding confabulation — every token locally unsurprising,
samples landing on incompatible assertions — is invisible to token entropy and is
precisely what lives at SLT.

**2. The answer to the reviewer is yes, on both datasets and at both sites.** Against the
full four-scalar basis the SAE probe reads **.5853–.6587 at TBG** and **.7038–.7305 at
SLT**, versus entropy controls of **.5039–.5788**. The margin over the control is
+.03…+.15 at TBG and +.13…+.20 at SLT. The features carry semantic-uncertainty signal
that the model's whole output-distribution summary does not contain.

**3. But TBG is not empty, and that was not the prediction.** The theory said TBG should
*structurally fail* this test. It does not — PopQA's TBG probe reads .6575 against a
control of .5039, which is as close to a clean chance baseline as this design gets. So the
honest claim is a **graded dissociation, not an on/off one**: residual signal exists at
both sites and is consistently **~+.07 larger at SLT**, where meaning-dispersion and
symbol-dispersion have had a chance to diverge.

**4. It is not SAE-specific — but the SAE arm wins it.** The dense residual also predicts
the residual target (.60 at TBG, .69 at SLT on PopQA), so the headline is really "the
hidden state knows more than the output distribution". The SAE arm beats dense in
**8 of 8** configs, by +.04…+.06. That is a bigger and more consistent SAE margin than
anything in the raw-SE decomposition of §1, where the SAE bought +.0003 on PopQA.

**5. Read it as a dissociation, not a leaderboard.** These AUROCs are modest by
construction — the dominant signal has been scrubbed out. The result is *"reliably above
a matched control on a target token entropy has been removed from, and more so at SLT"*,
not *"high AUROC"*.

### Caveats, named

- **Partialling out a small linear basis is a weak control.** `E+` removes the linear
  dependence on four output-distribution scalars (with one quadratic term). Nonlinear
  dependence beyond that leaks into the residual and inflates the result. The
  `entropy-ctl` column bounds how much: it is .50–.58, so the leak is small but not zero.
- **A multi-token entropy control would be stricter.** `max` next-token entropy over the
  answer's tokens scores .9461 (TriviaQA) / .9773 (PopQA) against raw SE — a stronger
  predictor than the single-position `E` used here. The single-token rule forbids it as a
  *probe input*, and it was not used as a *control* either; that is the conservative bound
  this experiment did not claim.
- **Binarising at the training median** makes the residual target balanced and comparable
  across configs, but it is a different rule from `sep_best_split` (which assumes a
  non-negative entropy scale and does not apply to a signed residual).
- The two configs per (dataset, site) are the top-2 **distinct selectors** by validation
  AUROC; within a site they differ by ≤ .002 and give near-identical residual numbers, so
  the result is not selector-sensitive.

---

## 10. Cross-dataset transfer: one probe, two datasets

Train on PopQA, on TriviaQA, or on both; test on both. Every cell carries all three arms
of §1 (**A** SEP dense · **B** dense + top-K · **C** ours), so a transfer gap can never be
read as "ours beats SEP" when it is really selection.

**One frozen config for the whole study**, because a cross-trained probe needs one feature
space: TBG · res/16k/canonical · rms · `spearman` guided by the model's own next-token
entropy · **K = 64**. That config costs each dataset ≤ .0014 against its own validation
argmax (PopQA .9537 vs .9538; TriviaQA .8841 vs .8855), so neither is handicapped by it.
The matrix runs at **both** candidate layers, 24 (PopQA's argmax) and 40 (TriviaQA's).

Held fixed so that only the probe's training source varies: the label (each dataset's rows
are binarised with `sep_best_split` fit on *that* dataset's own training draw — a
threshold is a per-dataset scale calibration, not part of the probe); the evaluation rows
(always the target dataset's test split); and the RMS scale (one scalar per dataset,
fitted once on unlabelled training rows). 3 seeds, mean test AUROC.

### Layer 40

| train on | → PopQA A | B | **C** | | → TriviaQA A | B | **C** |
|---|---:|---:|---:|---|---:|---:|---:|
| PopQA (2048) | .9299 | .9435 | **.9447** | | .7880 | .7886 | **.8401** |
| TriviaQA (2048) | .8590 | .9239 | **.8989** | | .8541 | .8758 | **.8866** |
| **both (2048+2048)** | .9174 | .9445 | **.9446** | | .8359 | .8763 | **.8839** |
| both, size-matched (1024+1024) | .9279 | .9418 | **.9401** | | .8492 | .8673 | **.8780** |

### Layer 24

| train on | → PopQA A | B | **C** | | → TriviaQA A | B | **C** |
|---|---:|---:|---:|---|---:|---:|---:|
| PopQA (2048) | .9261 | .9448 | **.9451** | | .7021 | .7356 | **.7829** |
| TriviaQA (2048) | .8846 | .9053 | **.9160** | | .8173 | .8690 | **.8676** |
| **both (2048+2048)** | .9176 | .9459 | **.9411** | | .7978 | .8552 | **.8656** |
| both, size-matched (1024+1024) | .9231 | .9425 | **.9383** | | .8110 | .8497 | **.8596** |

### What it says

**1. One pooled probe serves both datasets essentially for free.** At L40, training on
both and testing in-domain gives **.9446 on PopQA (vs .9447 trained on PopQA alone)** and
**.8839 on TriviaQA (vs .8866)**. The cost of merging is .0001 and .0027. The
size-matched pooled arm (1024 rows from each, so the *same* 2048 total as every other
row) still reaches .9401 / .8780 — so pooling wins on generality, not on row count.

**2. The probe transfers, but it is not free.** A PopQA-trained probe loses **.047** on
TriviaQA (.8866 → .8401) and a TriviaQA-trained probe loses **.046** on PopQA
(.9447 → .8989) — a symmetric ~.046 penalty at L40, and both remain far above chance.
At L24 the penalty is lopsided (−.085 popqa→trivia, −.029 trivia→popqa) because L24 is a
bad layer for TriviaQA in the first place (§3).

**3. Out of domain is the one place the SAE earns its keep on raw AUROC.** Averaged over
the four OOD cells, **C − B = +.021**, versus **+.003** in-domain — an order of magnitude
more than the SAE buys anywhere in §1. It is not uniform: the SAE wins the hardest
direction by a lot (popqa → trivia: +.052 at L40, +.047 at L24) and *loses* one cell
(trivia → popqa at L40, −.025). The defensible statement is that **the sparse code
degrades more gracefully under domain shift than the dense residual does**, not that it
always wins.

**4. The dense baseline transfers worst of the three.** At L40, arm **A** loses .071
(popqa-ward) and .066 (trivia-ward); **B** is erratic (−.020 one way, −.087 the other);
**C** is the only arm whose loss is bounded and symmetric at ~.046. And **C beats A in all
16 cells** of the two tables.

**5. L40 is the better shared layer.** It costs PopQA .0004 (.9447 vs .9451 at L24) and
gains TriviaQA .019 (.8866 vs .8676) — a direct consequence of §3's asymmetry, where
PopQA's layer profile is flat and TriviaQA's is not. If one cell has to serve both
datasets, it is **L40 / TBG**.

---

## 11. Sparse codes vs SAE-basis pre-activations

JumpReLU zeroes every unit whose pre-activation sits under its threshold. That mask is
what makes the code sparse and readable — and it is also, potentially, signal being thrown
away. So probe the **same learned directions with the mask off**: `pre = W_enc·x̃ + b_enc`
instead of `post = pre · 1[pre > θ]`. The basis is identical, the interpretation of a
column is identical; only the sparsification is gone.

Both datasets at their own frozen config from §1 (TriviaQA L40/TBG/`mutual_information`+entropy,
PopQA L24/TBG/`spearman`+label; res/16k/canonical, rms, C=1.0), 5 seeds, K re-chosen on
validation per arm per seed, test read once. `post` is recomputed from `pre` in the same
forward pass and asserted equal to SAELens's own `encode()` (observed max |Δ| = 0), so the
two arms differ by the mask and nothing else.

| arm | TriviaQA (n=5,334) | PopQA (n=10,999) |
|---|---:|---:|
| **A** SEP dense, all 3,584 dims | .8497 ± .0065 | .9282 ± .0040 |
| **B** dense + top-K, no SAE | .8763 ± .0022 | .9441 ± .0020 |
| **C** ours: **sparse codes** + top-K | .8847 ± .0026 | .9444 ± .0021 |
| **D** **pre-activations** + top-K, ranked on pre-acts | **.8897 ± .0018** | .9456 ± .0008 |
| **D′** **pre-activations** at C's features and K | **.8898 ± .0035** | **.9487 ± .0012** |
| C′ sparse codes at D's features and K | .8733 ± .0135 | .9308 ± .0008 |
| **E** all 16,384 sparse codes, **no selection** | .8539 ± .0035 | .9169 ± .0037 |
| **F** all 16,384 pre-activations, **no selection** | .8491 ± .0048 | .9244 ± .0040 |

D′ and C′ are the off-diagonal of a 2×2 over *ranking basis* × *value basis*, which is the
only way to tell "the mask destroyed information" apart from "pre-activations rank features
differently". D′ is the controlled contrast: same directions, same K, mask off — and it is
handicapped, since its K was tuned for C.

| | TriviaQA | PopQA |
|---|---:|---:|
| **whole swap (D − C)** | **+.0049** | **+.0012** |
| ├ mask off, same features (D′ − C) | **+.0050** | **+.0043** |
| └ re-ranking on pre-acts (D − D′) | −.0001 | −.0032 |
| same contrast, z-scored on train (Dz − Cz) | +.0048 | +.0013 |
| the SAE's share of the gap, post-threshold (C − B) | +.0085 | +.0003 |
| **the SAE's share, pre-activation (D′ − B)** | **+.0135** | **+.0046** |

The z-scored row matters because `pre` and `post` live on different scales and a fixed L2
penalty is not scale-free; standardising both on train rows leaves the difference intact
(TriviaQA .8851 → .8900, PopQA .9446 → .9458), so it is information, not regularisation.

Per seed, the mask-only contrast (D′ − C) is positive **10 times out of 10**:

| seed | TriviaQA C | D′ | | PopQA C | D′ |
|---|---:|---:|---|---:|---:|
| 42 | .8857 | .8864 | | .9459 | .9492 |
| 43 | .8874 | .8922 | | .9439 | .9495 |
| 44 | .8845 | .8912 | | .9466 | .9500 |
| 45 | .8857 | .8933 | | .9443 | .9476 |
| 46 | .8804 | .8857 | | .9412 | .9473 |

Why the effect is as small as it is — the selected features are barely sparse:

| | TriviaQA | PopQA |
|---|---:|---:|
| mean realised L0 (of 16,384) | 202.4 (1.2%) | 72.1 (0.44%) |
| fraction of rows where **C's** chosen features are non-zero | **.400** | **.352** |
| fraction of rows where **D's** chosen features are non-zero | .103 | .071 |
| overlap of C's and D's feature sets | .33 | .21 |

### What it says

**1. Pre-activations win, but they do not win big — +.005 and +.001.** The hypothesis is
right in sign and right about the mechanism; the "win big" branch does not happen. On
TriviaQA the swap is worth **+.0049** over ours and **+.0400** over SEP dense; on PopQA
**+.0012** and **+.0174**. Those are a few thousandths — but paired across seeds the sign
never flips, so they are small rather than noisy. To be exact about it: **D′ is the
highest in-domain number anywhere in this document, on both datasets** — .8898 edges past
the best of §7's six learner families (.8863) and .9487 past theirs (.9468). It edges past
them by .0035 and .0019, against noise ceilings of .9813 and .9865, so the ~.89 / ~.95
frontier statements stand unchanged. The finding is real and it is small.

**2. The mask does destroy usable signal — and re-ranking on pre-acts then spends it.**
On the *same* directions with the *same* K, unmasking is worth **+.0050 / +.0043**, in
10/10 seeds. Ranking on pre-activations instead of on the codes hands most of that back
(−.0001 TriviaQA, −.0032 PopQA), because pre-activation rankings prefer features that are
almost never above threshold (active on 7–10% of rows against 35–40% for the sparse-code
ranking), and a feature that only ever fires a little is worth little.
So the best pipeline is a hybrid: **select with the sparse code, read the pre-activation.**
That is the D′ arm, and it is the best number in the table on both datasets.

**3. The clearest consequence is for §1's decomposition, not for the leaderboard.** The
SAE's own contribution over dense + top-K rises from **+.0085 → +.0135** on TriviaQA and
from **+.0003 → +.0046** on PopQA — a 15× increase on the dataset where §1 concluded the
SAE "contributes nothing measurable to accuracy". That conclusion needs the qualifier it
did not have: *the sparse code* contributes nothing measurable on PopQA; **the SAE basis
does**, once you stop throwing away the sub-threshold values. Selection is still the
dominant term (+.0159 there against +.0046).

**4. The basis alone is worth nothing without selection.** Arms E and F are the SAE-basis
analogues of A — 16,384 dims, no top-K — and they land at or below SEP's own 3,584-dim
dense probe in three of four cells (E: +.0042 TriviaQA, **−.0113** PopQA; F: −.0006,
−.0038), and sit .024–.041 beneath the selected arms. Un-thresholded does not rescue them
either: F beats E on PopQA (+.0075) and loses on TriviaQA (−.0048). Everything in §11
lives inside the top-K regime, which is where §1 already said the win comes from.

**5. Where the signal lives, stated plainly.** ~1.2% / 0.44% of features fire on a given
row, but the features the probe selects fire on **35–40%** of rows — 30–80× the average
density. The probe is not reading rare, crisply-firing latents; it is reading a handful of
unusually promiscuous ones, and those are the features the JumpReLU mask barely touches.
That is why sub-threshold structure is worth only ~.005 here, and it is a more interesting
fact about SAE probes than the AUROC is.

**6. What is not tested.** Only these two frozen configs, this width (16k), this site
(res), this position (TBG), and only in-domain. §10 found the SAE arm's one clear win out
of domain — whether unmasking helps or hurts *there* is open, and is the version of this
experiment most likely to move a number.

## 12. Training on the continuous SE target

SEP only ever classified: it binarises SE with `best_split` and discards the graded value
before the probe sees it. But `se_discrete` is a real number per prompt. So **fit a
regressor against raw SE and score the prediction against the same binary label** — the
eval protocol is byte-identical to SEP's, the training signal is simply not thrown away.

Two regressors, picked from §7's learner sweep as the only families with a real chance:
**Ridge**, the closed-form linear counterpart of the logistic probe (same features, same
rows, same eval — so any difference *is* the graded-target effect), and
**HistGradientBoostingRegressor**, §7's strongest non-linear family. Each is paired with a
classifier of the same family on the same splits. Two logistic controls, and the
distinction matters: `pinned` is SEP's published C = 1.0, `tuned` picks C on validation over
a grid the same size Ridge gets for alpha (α ∈ {1e-1 … 1e7}). **Only the tuned one is a
fair pair** — against the pinned one, "regression wins" would just be §12's finding again.

### TriviaQA (L40/TBG, n=5,334, 5 seeds)

| feature space | LogReg pinned | LogReg tuned | HGB clf | **Ridge on `se_discrete`** | Ridge on `se_lnrao` | HGB reg |
|---|---:|---:|---:|---:|---:|---:|
| dense 3,584 | .8497 | .8909 | .8858 | **.8992 ± .0027** | .8977 | .8821 |
| SAE codes + top-K | .8855 | .8890 | .8810 | **.8912 ± .0026** | .8888 | .8846 |
| pre-acts at those ids | .8921 | .8943 | .8874 | **.8976 ± .0022** | .8953 | .8909 |
| codes + top-K, SE-guided | .8764 | .8827 | .8779 | .8839 | .8807 | .8823 |
| all 16,384 codes | .8539 | .8866 | — | .8893 | — | — |

### PopQA (L24/TBG, n=10,999, 5 seeds)

| feature space | LogReg pinned | LogReg tuned | HGB clf | **Ridge on `se_discrete`** | Ridge on `se_lnrao` | HGB reg |
|---|---:|---:|---:|---:|---:|---:|
| dense 3,584 | .9282 | .9531 | .9488 | **.9546 ± .0017** | .9534 | .9484 |
| SAE codes + top-K | .9446 | .9428 | .9421 | .9436 ± .0031 | .9428 | .9431 |
| pre-acts at those ids | .9489 | .9492 | .9448 | **.9497 ± .0007** | .9494 | .9456 |
| codes + top-K, SE-guided | .9439 | .9448 | .9421 | .9449 | .9447 | .9429 |
| all 16,384 codes | .9169 | .9493 | — | .9516 | — | — |

### What it says

**1. Not discarding the graded target is worth a small, consistent gain — and it is the
best arm in this document.** Against its *fair* pair (the tuned logistic), Ridge on
`se_discrete` wins **10/10 space × dataset cells** — +.0012 … +.0083 on TriviaQA, +.0001 …
+.0023 on PopQA. (Against the *pinned* control it loses one cell, PopQA's codes + top-K,
.9436 vs .9446 — which is §12's finding, not regression's.)
The best cell on each dataset is **Ridge on raw SE over SEP's own 3,584-dim residual:
.8992 (TriviaQA) and .9546 (PopQA)**, which are the highest numbers anywhere in this
document. Both are on the plain residual — **no SAE, no selection, and no classification.**

**2. Regression is the smaller of the two effects, and the honest ordering matters.** Going
from SEP's published probe to the best arm here is +.0495 (TriviaQA) and +.0264 (PopQA), and
it decomposes as **tuning the penalty +.0412 / +.0249** (§12) plus **using the graded target
+.0083 / +.0015**. The graded target is real and it is free; it is not the main term.

**3. `se_discrete` is the right raw target, not `se_lnrao`.** The Nature headline estimator
loses in 8/8 space × dataset cells, by .0002–.0032. Consistent with §6, and it settles the
ambiguity in "train on raw SE": train on the same scalar whose binarisation is the label.

**4. Boosting loses in both roles.** HGB regression is below Ridge in 8/8 cells and below
its own classifier on the dense residual (.8821 vs .8858 / .9484 vs .9488). Whatever the
graded target adds, a linear model extracts it — the ordering is smooth, not something a
tree needs to carve.

**5. Re-guiding *selection* by raw SE changes nothing.** `codes_topK_se` ranks features
against the continuous SE instead of the frozen guide — closing the other place label
information was being discarded — and it lands at .8839 / .9449 against .8912 / .9436.
Feature overlap with the frozen ranking is .47 (TriviaQA) / .73 (PopQA): a substantially
different feature set, the same accuracy. The graded target helps at *fitting* time, not at
*selection* time.

**6. What is not tested.** In-domain only, one layer/token per dataset, and no ordinal
model (`se_discrete` has ~40 distinct levels, closer to continuous than to few-level
ordinal; `mord` is not in the environment). Out-of-domain regression — §10's setting — is
the open question most likely to matter.

---

## 13. Handing the probe the scalar: token entropy as a probe FEATURE

**The lever — decompose the target instead of the model.** SEP asks one linear map on `h`
to predict total SE. But at TBG, SE ≈ `H(softmax(W_U·norm(h)))`: a strongly nonlinear
function of `h`. A linear probe structurally cannot reconstruct that, it can only
approximate it — which is exactly why every probe in this document loses to the raw scalar
*at TBG* (§2: token entropy **.9334 / .9758** against a probe frontier of ~**.89 / .95**).
§7 already closed the escape of enlarging the model class: six learner families, none past
.8863 / .9468. So take the other route — **stop making the linear probe relearn a scalar
you already have exactly.** Hand it over as a feature and let the K sparse features carry
only the part it cannot explain.

This stays inside every *cost* constraint the study has kept — one layer, one token, a
linear probe, no extra generations — and gives up exactly one thing: the purity of reading
`h` and nothing else. That trade is the subject of the first caveat at the end.

The probe therefore reads **two things**: the **top-K selected sparse SAE features**
(`res` / 16k / canonical — the general-sweep winner of §6) and **one extra column**, the
model's own next-token entropy at this site. Nothing else changes.

§9 predicts where this should pay and where it should not: the entropy basis explains
**71% / 82%** of SE's variance at TBG and **7% / 0.8%** at SLT. Both sites are run on both
datasets — at each site's own validation-argmax layer from §3 (TriviaQA TBG L40 / SLT L28,
PopQA L24 for both), which is the same four cells §9 froze.

Notebook: [`entropy_feature_probe_gemma.ipynb`](entropy_feature_probe_gemma.ipynb).

### Arms

| arm | features | free at inference? |
|---|---|---|
| `E` | token entropy at this site — one column | yes, same forward pass |
| `E2` | `E`, `E²` | yes |
| `Ef` | `E`, `E²`, the logit gap at this site | yes |
| `E+` | §9's basis: `Ef` **+ `p_true`** | **no** — `p_true` is a second forward pass |
| `A` | SEP dense, all 3,584 residual dims | — |
| `B` | dense + top-K over the same 3,584 dims — no SAE | — |
| `C` | ours: SAE codes + top-K | — |
| `A+E`, `B+E` | the same, with the scalar appended | yes |
| **`C+E`** | **K sparse features ⊕ token entropy** | yes |
| `C+E2`, `C+Ef`, `C+E+` | `C` ⊕ the wider scalar blocks | `C+E+` no, rest yes |
| `C+Eiso` | `C` ⊕ the best **monotone** prediction of raw SE from `E` (isotonic, train rows only) | yes |

`A+E` and `B+E` are what stop "ours plus entropy beats SEP" from being read as a win for
sparsity: **every arm gets the same scalar.**

The entropy column is `greedy_entropy_full[p − answer_start + 1]` — the distribution *this*
hidden state emits, §9's indexing rule, not §3's ceiling-cell convention. That is what makes
it free: it comes out of the same forward pass as the hidden state, at the same position.

### Protocol

As everywhere else — `se_discrete` binarised by `sep_best_split` on the training draw,
`N_TRAIN` = 2048 with everything else test, the draw re-split into `is_fit` / `is_val`,
**every knob chosen on `is_val`**, test read once, 5 seeds. Three things are specific here:

- **Features are z-scored** (mean/sd from the fitting rows). Mixing a ~1-nat scalar with SAE
  codes under one fixed L2 penalty is not scale-free. The raw-scale control below shows the
  conclusions do not depend on it.
- **Three C rules are reported side by side.** `pinned` is SEP's published C = 1.0 applied
  to every arm; `tuned` picks C on validation; `Ridge` is §12's protocol — regress on raw
  `se_discrete`, score against the same binary label, α on validation. §12 showed that
  quoting only the pinned rule is how the §1 headline went wrong.
- **The selection guide is an axis**, chosen on validation from `label` / `entropy` /
  `residual`, where `residual` ranks features against `SE − OLS(SE ~ E, E²)` — the same
  decomposition idea applied to selection. It has to be an axis, because guiding selection
  *by* entropy while entropy is *already a feature* selects features that duplicate it.

`SELECT` is `spearman` throughout — §10's shared frozen selector, ≤ .0014 off each dataset's
own argmax. Two other selectors are run as controls below.

### TBG — test AUROC, mean of 5 seeds

| arm | TriviaQA pinned | tuned | Ridge | PopQA pinned | tuned | Ridge |
|---|---:|---:|---:|---:|---:|---:|
| `E` | .9336 | .9336 | .9336 | .9754 | .9754 | .9754 |
| `E2` | .9336 | .9336 | .9336 | .9754 | .9754 | .9754 |
| `Ef` | .9336 | .9332 | .9335 | .9768 | .9768 | .9769 |
| `E+` | .9484 | .9484 | .9479 | .9805 | .9804 | .9794 |
| `A` SEP dense | .8475 | .8893 | .8976 | .9248 | .9536 | .9556 |
| `B` dense + top-K | .8752 | .8807 | — | .9430 | .9485 | — |
| `C` ours | .8858 | .8893 | .8926 | .9454 | .9476 | .9494 |
| `A+E` | .8616 | .8945 | — | .9398 | .9551 | — |
| `B+E` | .9399 | .9399 | — | .9761 | .9758 | — |
| **`C+E`** | **.9423** | **.9423** | **.9429** | **.9756** | **.9756** | **.9776** |
| `C+E2` | .9426 | .9426 | .9436 | .9758 | .9762 | .9776 |
| `C+Eiso` | .9421 | .9418 | **.9440** | .9739 | .9743 | .9770 |
| `C+Ef` | .9423 | .9408 | .9437 | .9766 | .9776 | .9788 |
| `C+E+` | .9515 | .9502 | **.9530** | .9800 | .9803 | **.9807** |

### SLT — test AUROC, mean of 5 seeds

| arm | TriviaQA pinned | tuned | Ridge | PopQA pinned | tuned | Ridge |
|---|---:|---:|---:|---:|---:|---:|
| `E` | .6740 | .6740 | .6740 | .6047 | .6047 | .6047 |
| `E2` | .6739 | .6740 | .6739 | .6047 | .6047 | .6047 |
| `Ef` | .6263 | .6840 | .6828 | .6606 | .6626 | .6631 |
| `E+` | .8759 | .8758 | .8758 | .9216 | .9211 | .9186 |
| `A` SEP dense | .8320 | .8765 | .8811 | .9324 | .9554 | .9586 |
| `B` dense + top-K | .8298 | .8365 | — | .9277 | .9386 | — |
| `C` ours | .8402 | .8436 | .8453 | .9313 | .9377 | .9394 |
| `A+E` | .8325 | .8766 | — | .9323 | .9554 | — |
| `B+E` | .8310 | .8359 | — | .9293 | .9390 | — |
| **`C+E`** | **.8414** | **.8448** | **.8459** | **.9329** | **.9379** | **.9385** |
| `C+E2` | .8414 | .8448 | .8463 | .9329 | .9379 | .9385 |
| `C+Eiso` | .8420 | .8457 | .8470 | .9319 | .9384 | .9384 |
| `C+Ef` | .8417 | .8449 | .8466 | .9329 | .9380 | .9387 |
| `C+E+` | .8983 | .8981 | **.9045** | .9550 | .9558 | **.9557** |

Seed spreads for the arms quoted below (± sd over the 5 seeds): `C+E` tuned
.9423 ± .0028 / .9756 ± .0019 at TBG and .8448 ± .0046 / .9379 ± .0037 at SLT;
`C+E+` Ridge .9530 ± .0024 / .9807 ± .0009 at TBG and .9045 ± .0032 / .9557 ± .0018 at SLT.

### What each piece actually buys — paired per seed (`tuned`, wins/5 in brackets)

| contrast | what it isolates | TriviaQA TBG | PopQA TBG | TriviaQA SLT | PopQA SLT |
|---|---|---:|---:|---:|---:|
| `C+E` − `C` | the free scalar, to the probe | **+.0530** (5/5) | **+.0280** (5/5) | +.0012 (5/5) | +.0002 (4/5) |
| `C+E` − `E` | the probe, to the free scalar | **+.0087** (5/5) | +.0002 (3/5) | **+.1708** (5/5) | **+.3331** (5/5) |
| `C+E` − `B+E` | the SAE, once the scalar is in | +.0024 (4/5) | −.0002 (3/5) | +.0089 (5/5) | −.0012 (1/5) |
| `C+E` − `A+E` | selection, once the scalar is in | **+.0478** (5/5) | **+.0205** (5/5) | −.0318 (0/5) | −.0175 (0/5) |
| `A+E` − `E` | appending 3,584 dims to the scalar | **−.0391** (0/5) | **−.0203** (0/5) | +.2026 (5/5) | +.3507 (5/5) |
| `A+E` − `A` | the scalar inside a 3,584-dim probe | +.0053 (5/5) | +.0015 (5/5) | +.0001 (5/5) | +.0000 (5/5) |
| `C+E2` − `C+E` | the quadratic term | +.0003 (4/5) | +.0006 (5/5) | −.0000 (4/5) | +.0000 (3/5) |
| `C+Eiso` − `C+E` | a monotone `g()` instead of a linear one | −.0004 (2/5) | −.0013 (0/5) | +.0009 (5/5) | +.0005 (5/5) |
| `C+Ef` − `Ef` | **the probe, on top of the strictly free scalars** | **+.0076** (5/5) | +.0008 (4/5) | **+.1608** (5/5) | **+.2754** (5/5) |
| `C+E+` − `E+` | **the probe, on top of the full basis** | +.0018 (4/5) | −.0001 (3/5) | **+.0222** (5/5) | **+.0346** (5/5) |
| `E+` − `Ef` | `p_true` (a second forward pass) | +.0152 (5/5) | +.0036 (5/5) | **+.1918** (5/5) | **+.2585** (5/5) |

Under `Ridge` the same contrasts read: `C+E` − `E` **+.0093 / +.0022** (5/5 both) at TBG;
`C+E+` − `E+` **+.0051 / +.0014** at TBG against **+.0287 / +.0371** at SLT, 5/5 everywhere.

### What it says

**1. The scalar is worth more than every method change in this document put together.**
Handing one free column to the sparse probe moves it **+.0530 (TriviaQA) / +.0280 (PopQA)**
at TBG, in 5/5 seeds. For scale: the entire §1 headline over SEP was +.0350 / +.0162 and
§12 showed most of that was the pinned penalty; §11's mask removal was +.0050 / +.0043;
§12's graded target was +.0083 / +.0015. **The ~.89 / ~.95 frontier was never a fact about
representations. It was the price of asking a linear map on `h` to reconstruct
`H(softmax(W_U·norm(h)))`, and it is refundable.**

**2. Read the decomposition before quoting that.** At TBG almost all of the level is `E`
itself (.9336 / .9754 on its own). What the sparse features add *on top of* the scalar is
**+.0087 (5/5) on TriviaQA and +.0002 (3/5) on PopQA** — the same shape as §1, where the
SAE bought +.0085 / +.0003. Two datasets, two experiments, the same verdict: **on PopQA at
TBG the hidden state adds essentially nothing to what the output distribution already
says.** The honest headline for TBG is *"the scalar, with a small probe attached"*, not
*"a much better probe"*.

**3. `A+E` is the sharpest result here, and it is the one place selection stops being
cosmetic.** Appending the scalar to SEP's full 3,584-dim residual is **worse than the
scalar alone** — −.0391 / −.0203, 0/5 seeds — and worth almost nothing to the dense probe
(+.0053 / +.0015). Under one shared penalty a single informative column among 3,585 is
drowned: it draws **0.5%** of the fitted coefficient magnitude, against **~50% / ~42%** in
`C+E`.

| share of fitted coefficient magnitude on the scalar column(s), `tuned` | TriviaQA TBG | PopQA TBG | TriviaQA SLT | PopQA SLT |
|---|---:|---:|---:|---:|
| `A+E` | .005 | .005 | .001 | .000 |
| `B+E` | .575 | .375 | .011 | .011 |
| `C+E` | .495 | .419 | .023 | .016 |
| `C+E2` | .607 | .456 | .027 | .023 |

**The scalar needs a small probe beside it**, and both routes to a small probe work —
`B+E` (dense + top-K) reaches .9399 / .9761 and `C+E` .9423 / .9756. The SAE's own share
once the scalar is in is +.0024 (4/5) on TriviaQA and −.0002 on PopQA: unchanged from §1,
which is the point. **Selection is what buys the room; the SAE is what makes the room
readable.**

**4. SLT is the mirror image, exactly where §9 said it would be.** There the free scalars
are weak — `E` alone is .6740 / .6047 — and the probe carries the result: `C+E` − `E` is
**+.171 / +.333**. Measured the other way round, against the *whole* output-distribution
basis, the sparse features are worth **+.029 / +.037 at SLT** and only **+.005 / +.001 at
TBG** (Ridge, 5/5 seeds everywhere) — a **6× (TriviaQA) / 26× (PopQA) dissociation, in the
direction §9 predicted**, now stated in deployable AUROC rather than in residual AUROC.
Confident-sounding confabulation — every token locally unsurprising, samples landing on
incompatible assertions — is what lives at SLT, and it is the part no output-distribution
summary reaches.

**5. The right selection guide flips depending on whether the scalar is a feature.**
Validation AUROC by guide, mean of 5 seeds:

| arm | guide | TriviaQA TBG | PopQA TBG | TriviaQA SLT | PopQA SLT |
|---|---|---:|---:|---:|---:|
| `C` | `label` | .8873 | .9582 | .8457 | .9466 |
| `C` | **`entropy`** | **.8898** | **.9600** | .8248 | .9269 |
| `C` | `residual` | .8670 | .9488 | **.8474** | **.9478** |
| `C+E` | `label` | .9385 | .9827 | .8458 | .9469 |
| `C+E` | `entropy` | .9393 | .9831 | .8250 | .9273 |
| `C+E` | **`residual`** | **.9423** | **.9844** | **.8476** | **.9484** |

Without the scalar, ranking features *by* entropy is the best guide at TBG and the
`residual` guide is the worst (−.023 / −.011). With the scalar in the model the ordering
inverts and `residual` wins all four `C+E` cells — and all four again for `C+E2`.
Selecting features that duplicate a column you already have is waste, and the grid measures
the waste. This is §4's guide question re-asked in the one setting where it has a mechanical
answer.

**6. K collapses to 16.** With the scalar present, validation picks the *smallest* K in the
grid at TBG on both datasets (`C+E`: K = 16, `residual` guide) and the *largest* at SLT
(K = 128). The probe needs 16 sparse features next to the scalar where the scalar is
informative, and 128 where it is not — a compression result that falls straight out of the
same dissociation.

**7. Where it leaves the numbers.** The best arms this document has produced under the
split protocol, both at TBG with Ridge on the graded target:

| | TriviaQA | PopQA |
|---|---:|---:|
| best before §13 (§12, Ridge on dense) | .8992 | .9546 |
| **`C+E+`** (needs `p_true`'s extra forward pass) | **.9530 ± .0024** | **.9807 ± .0009** |
| **`C+Ef`** — strictly one forward pass | **.9437 ± .0018** | **.9788 ± .0004** |
| `C+E` — the two-feature arm exactly as specified | .9429 ± .0015 | .9776 ± .0007 |
| `C+E2` — the same plus `E²` | .9436 ± .0014 | .9776 ± .0006 |
| §2's noise ceiling | .9813 | .9865 |
| §2's independent 10-sample re-run of the pipeline | .9572 | .9707 |

**On PopQA the arm now beats a second run of the SE pipeline itself** (.9807 vs .9707) and
sits .006 under the hard ceiling. On TriviaQA it lands .004 under the re-run and .028 under
the ceiling. The remaining headroom on PopQA is label noise, not method.

### Controls

| control | TriviaQA TBG | PopQA TBG | TriviaQA SLT | PopQA SLT |
|---|---:|---:|---:|---:|
| `C+E` tuned, as reported (`spearman`) | .9423 | .9756 | .8448 | .9379 |
| same, `mean_difference` | .9387 | .9760 | .8461 | .9373 |
| same, `mutual_information` (3 seeds, TBG only) | .9382 | — | — | — |
| same, **raw scale, no z-scoring** | .9422 | .9751 | .8468 | .9389 |

Each row is the notebook re-run with one knob changed (`SELECT`, `ZSCORE`); the
`mutual_information` row is 3 seeds because that selector costs ~1 min per call at 16k.
The raw-scale run doubles as a harness check: it reproduces §12's published dense numbers
to the fourth decimal (`A` pinned .8513 / .9284 against §1's .8497 / .9282; `A` Ridge
**.8992 / .9546**, §12's values exactly), which is why the z-scored primary can be read
against the rest of this document.

### Caveats, named

- **This arm reads the output distribution, not only the hidden state.** That is a real
  change to what the artefact *is* — SEP's claim is about `h` alone. It costs nothing at
  TBG (same forward pass, same position) and it is stated, not hidden, which is why `E`,
  `A+E` and `B+E` are in every table.
- **`p_true` is not free.** Every `E+` arm needs a second forward pass with a purpose-built
  prompt, and the tables show it earns +.015 / +.004 at TBG and +.19 / +.26 at SLT on its
  own. The strictly-free ceiling is the `C+Ef` row.
- **Interpretability takes a real hit at TBG.** Half the fitted weight sits on one column
  that is not an SAE feature. The sparse features are still 16 of them and still readable,
  but "the probe is a sparse readout of meaning" is now a claim about the *other* half, and
  at TBG that half is worth +.009 / +.000. At SLT the scalar carries 2% of the weight and
  the claim is intact.
- **The single-token rule still binds, and it is now the binding constraint.** §3's `max`
  next-token entropy over the answer's tokens scores .9461 / .9773 against raw SE, better
  than the single-position `E` used here. A multi-token scalar is the obvious next feature
  and is not tested.
- **§2's four-scalar diagnostic row (.9608 / .9836) is higher than anything here**, but its
  composition is not pinned down in that section and it is a stored-scalar diagnostic, not
  an arm fitted under this split protocol. Reconciling the two is open.
- **The `E` index convention matters at SLT and is not the only defensible one.** §9's rule
  (the distribution this state emits, index `sliced_count`) is used; §3's ceiling-cell
  convention (index `sliced_count − 1`) scores higher against raw SE (.6157 / .6812 vs
  .6740 / .6047 here). Since the scalar adds ~.001 at SLT either way, this does not move
  the dissociation, but the SLT `E` column is not the strongest single scalar available.
- **In domain only**, one layer per (dataset, site), 16k `res` canonical, no transfer.
  §10's setting — where the SAE has its one clear raw-AUROC win — is untested here.

---

## 14. Direct correctness probes

This is the first experiment here that trains the probe **directly on correctness**, rather
than training on SE and checking correctness afterward. It uses the cached layer-20 TBG
residual and 16k SAE matrix, the canonical stored splits (`discovery` 1,480 / `train` 3,681 /
`test` 2,221), correctness-guided feature selection on discovery, selector/K choice on an
inner train validation split, and 1,000 paired test bootstraps. The binary judge is primary;
`f1_squad >= 50` is the predeclared sensitivity channel.

| model | judge correctness AUROC (95% CI) | F1>=50 AUROC (95% CI) |
|---|---:|---:|
| token entropy | **.8866** (.8690, .9037) | **.8841** (.8663, .9008) |
| semantic entropy | .8928 (.8775, .9071) | .8812 (.8643, .8974) |
| token + semantic entropy | **.9056** (.8902, .9193) | **.8981** (.8829, .9122) |
| dense correctness probe | .7788 (.7537, .8018) | .7488 (.7225, .7727) |
| direct SAE correctness probe | .8233 (.8035, .8425) | .7948 (.7734, .8160) |
| SAE + token entropy | .8787 (.8611, .8963) | .8569 (.8379, .8746) |

The judge-target SAE winner was `mean_difference`, K=128 (validation AUROC .8333).
It compresses the 3,584-dimensional dense probe and improves its test AUROC by +.0445,
but it does **not** beat the cheap scalar: SAE minus token entropy is -.0632 with paired
95% CI (-.0825, -.0428). Adding the SAE features to token entropy is -.0079 (-.0189,
+.0030), so there is no positive conditional result; the F1 sensitivity analysis is
stronger against it at -.0272 (-.0406, -.0149).

Within the low-SE test stratum, the direct SAE probe remains above chance at .654 AUROC,
but token entropy is .712. This slice has only 37 incorrect answers among 1,180 examples,
so it is evidence of weak residual decodability, not a stable advantage. The constructive
result is that token entropy and sampled semantic entropy are complementary for correctness
(.9056 together), whereas this layer-20 TBG SAE representation does not recover that gain.
Next tests should sweep layers/sites on validation—especially answer-side SLT—and test
transfer, where the earlier SE experiments found the strongest rationale for sparse features.

Artifacts: `results/correctness_probe/20260822T185230Z_judge_binary/` and
`results/correctness_probe/20260822T185349Z_f1_50/`.

---

## VERDICT

> **⚠ §13 changed the top-level answer, and this section is written from *inside* the
> hidden-state-only constraint.** Everything below is true of probes that read `h` and
> nothing else. §13 lifts that constraint by one free column — the model's own next-token
> entropy at the probed position, from the same forward pass — and the frontier moves from
> ~.89 / ~.95 to **.9437 / .9788** (strictly free) or **.9530 / .9807** (with `p_true`'s
> extra forward pass). Read the table below as *"what limits a probe on `h` alone"*, which
> is the question SEP asks, and §13 as the answer to *"what limits a cheap SE predictor"*,
> which is the question a deployment asks.

**On raw SE, neither dataset's probe is limited by anything the method can fix.** Every
escape route is closed by a measurement, on both:

| candidate explanation | measurement that rules it out |
|---|---|
| the SE label is too noisy | **no** — ceilings are .9813 (TriviaQA) / .9865 (PopQA) |
| the TBG site lacks the information | **no** — entropy at TBG scores .9334 / .9758 |
| wrong layer / token | 13 layers × 8 positions swept on both, every arm |
| wrong selector | 7 selectors × 2 guides × 5 K swept at both widths; a cheating oracle ties the honest ranking on both |
| wrong SAE width / sparsity / site / normalisation | all swept on both; `mlp` beats `res` on PopQA by +.004 and loses on TriviaQA by .006, flagged in §6 |
| not enough training data | learning curves flat past ~1–2k rows on both |
| not enough supervision per row | multinomial LR on SE levels is flat on both |
| the probe is too weak (linear) | six learner families ≤ .8863 / ≤ .9468; §11's pre-activation arm .8898 / .9487; §12's Ridge-on-raw-SE **.8992 / .9546** — the ceiling moved, but only to ~.90 / ~.955 against noise ceilings of .9813 / .9865 |
| **the probe was asked to reconstruct a scalar it already has** | **this one was the big one — §13.** Give the probe the model's own next-token entropy as a feature and the same sparse arm goes **.8893 → .9423 (TriviaQA) / .9476 → .9756 (PopQA)** at TBG, 5/5 seeds. Nothing in the escape routes above is worth a tenth of it |
| the fixed C=1.0 rule | **this one was real** — §12 measures it on test: tuning C lifts SEP dense +.0412 / +.0249 and *reverses* the §1 gap. It does not lift anyone past the frontier, but it does erase the method's margin |
| the target scalar | `se_lnrao` worse than `se_discrete` for every arm on both |

**PopQA clears .95 on validation (.9538) where TriviaQA capped at ~.89 — but that is the
dataset, not the method.** PopQA is 75% high-SE with a cleaner label (r = .903 vs .858)
and a far more informative TBG site (.9758 vs .9334). The *gap over SEP* went the other
way: +.016 on PopQA versus +.035 on TriviaQA, and the SAE's share of it fell from +.009
to +.000.

**Where this leaves the method — revised by §12.** The sparse probe matches the dense probe
from 32–64 features instead of 3,584, and no learner family beats it by more than .005.
That is **interpretability at no accuracy cost**, and it is now the *whole* in-domain claim:
§12 shows the apparent accuracy advantage over SEP was the pinned C = 1.0, and it reverses
once both arms tune the penalty (.8909 vs .8901, .9531 vs .9473). Quote the compression,
never the gap. Five results point past it:

- **§9 is the positive scientific result:** the features carry semantic-uncertainty signal
  beyond the model's own predictive distribution, and measurably more of it at SLT than at
  TBG, exactly where meaning-dispersion should first become distinguishable from
  symbol-dispersion.
- **§10 is the first place sparsity pays in raw AUROC:** out of domain the SAE arm earns
  ~+.021 over dense + top-K, against +.003 in domain, and a single pooled probe matches
  both in-domain probes at once.
- **§11 says the sparsification, not the basis, is what costs accuracy** — and costs very
  little: +.0050 / +.0043 from dropping the JumpReLU mask on the same features, in 10/10
  seeds. Best in-domain arm on both datasets is the hybrid **select with the sparse code,
  read the pre-activation** (.8898 / .9487). Because the selected features already fire on
  35–40% of rows, there was never much sub-threshold structure to recover.
- **§12 is the one free win inside the constraint: stop discarding the graded target.**
  Ridge on raw `se_discrete`, scored against the same binary label, beats its tuned logistic
  pair in **10/10** cells and gives the best hidden-state-only numbers in this document —
  **.8992 / .9546**, on SEP's own dense residual, with no SAE and no selection.
- **§13 leaves the constraint, and it is the largest effect in the document.** Handing the
  probe the model's own next-token entropy as a *feature* — one free column from the same
  forward pass — is worth **+.0530 / +.0280** to the sparse arm at TBG, against +.0083 /
  +.0015 for §12's graded target and +.0050 / +.0043 for §11's mask. It also makes the two
  sites say opposite things, which is the scientific content: on top of the full
  output-distribution basis the sparse features are worth **+.029 / +.037 at SLT** and only
  **+.005 / +.001 at TBG** — §9's dissociation, restated in deployable AUROC.

**The best-known arm, end to end — revised by §13.** Within the hidden-state-only
constraint it is still §12's **Ridge on raw `se_discrete` over the 3,584-dim residual**
(.8992 / .9546). Without that constraint it is **§13's `C+E+` at TBG — K = 16 SAE features
alongside the output-distribution scalars, Ridge on the graded target — .9530 (TriviaQA) and
.9807 (PopQA)**, or **.9437 / .9788** if `p_true`'s extra forward pass is disallowed, against
noise ceilings of .9813 / .9865. The sparse probe's case is compression and
interpretability, plus §9's and §13's dissociation and §10's robustness under shift; at TBG
it is not raw in-domain AUROC, and at SLT — where the free scalars go quiet — it is.

**Do not re-run the closed directions.** Anything new should attack the token axis (open on
TriviaQA, nearly closed on PopQA, and §13 makes it the binding constraint: `max` entropy
over the answer scores .9461 / .9773 against the single-position .9334 / .9758 used there),
the prompt count, or extend §10–§13 — a third dataset, the same protocols on the remaining
roster models, pre-activations *out of domain* (where §10 says the sparse code has its one
clear win), **regression out of domain** (§12 leaves it untested), and **§13's arms out of
domain**, which is where a scalar the probe did not have to learn should transfer best.
