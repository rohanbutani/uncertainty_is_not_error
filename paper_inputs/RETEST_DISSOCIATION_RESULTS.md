# Dissociation re-test, offline (v2) — correctness vs uncertainty with matched instrumentation

Gemma-2-9B, TriviaQA (`fp1-triviaqa-20260812`) + PopQA (`fp1-popqa-20260814`), stored
activations only. **Nothing in this document measures interventions** — every number is a
read-out (probe, projection, statistical test) of unmodified stored states.

**Why this document exists.** `STEERING_OFFLINE_RESULTS.md` concluded that the SE and
correctness mean-difference directions are collinear (cos −.96…−.997, A1) and that
correctness-beyond-SE is not readable (D1–D3). That conclusion was under-instrumented:
correctness was only ever read through mean differences (no tuned probe on `judge_binary`
existed anywhere), every site was chosen by SE-probe argmax, the cosine was never compared
to the null value that label coupling alone predicts, the orthogonal component of the
correctness direction was never evaluated, no erasure or residualized-target test was run,
and the feature partition ran at 2,048 rows with a 77-row stratum inside a both-strata FDR
requirement. cos = −.97 leaves room for an orthogonal correctness component of up to ~25%
of the vector's magnitude. Everything below is instrumented to detect or exclude exactly
that component, and adds the token-entropy × correctness replication of Patel et al.
(2604.19974). Run order was R1 → R2 → R3 → R5 → R4 → R6 → R7 → R8 (R5 before R4 because
R5 is the decisive test and its outcome sets how much interpretation weight R4 carries).

## Protocol, pinned (realized values; deviations from the spec named where they occur)

| knob | value |
|---|---|
| runs | the two banked runs only (never `trivia_qa_long`) |
| splits | `rng(seed).permutation`, first **N_TRAIN = 2048** train, rest test; fit/val re-split `rng(seed+9973)`, VAL_FRAC = .25; seeds **42, 43** |
| state space | **rms-normalized residual at every site** (the §3 harness space), train-mean RMS per seed — including any final-layer site R1 selects |
| correctness labels | `judge_binary` (primary); `f1_squad ≥ 50` (robustness channel, R8) |
| uncertainty labels | `se_discrete` binarized by `sep_best_split` on the train draw; continuous `se_discrete`; tokH = `greedy_entropy_full[0]` binarized at the train median (R7) |
| tuned probes | dense logistic, C tuned on validation over §12's grid {1e-5…1}, refit on the train draw (grid cells in R1: best-C validation AUROC, no refit — selection metric only) |
| whitening | Ledoit–Wolf shrinkage covariance on training rows; whitened diffmean = Σ⁻¹(μ₊−μ₋) (Fisher/LDA; Marks & Tegmark 2310.06824) |
| CIs | prompt-level bootstrap, **1,000 resamples**, percentile 95%; any stratum with n < 100 flagged in the cell |
| INLP | erasure predictor LR **C = 1.0 pinned** (Ravfogel et al. 2004.07667 leaves it free); r = 1…10; probe after each r = fresh tuned-C LR; control = same procedure on train-shuffled erase-labels |
| SAE arm (R5) | dataset's frozen §1 selector (`mutual_information` / `spearman`), ranking on **fit rows** against the continuous residual (§9's rule), K ∈ {16, 32, 64, 128} chosen on validation, probe C = 1.0 refit on the train draw |
| R6a pool | matching on the training pool (the "full non-test pool" of the ground rules); held-out readout on test rows; within-pair test on pairs matched **among test rows** by the identical procedure |
| R6b pool | the full pool (all 7,382 / 13,047 rows) — discovery-only, nothing downstream is trained on it; strata thresholds from each seed's train draw |
| §9 E-indexing | `greedy_entropy_full[p − answer_start + 1]`, clipped ≥ 0 — prompt-side positions (`kw`/`nl`/`qlast`) therefore share TBG's index 0 |
| LEACE | via `concept-erasure` when importable; otherwise noted and skipped |

Sites: SE-argmax sites carried from §3 (TriviaQA L40/tbg, L28/slt; PopQA L24/tbg,
L24/slt); judge-argmax and tokH-argmax from Phase R1. Directions are rebuilt per seed on
training rows with the steering pass's exact recipe; where a site coincides with a banked
steering site the rebuilt diffmeans were verified against the banked vectors (cos > .999).

## R1. Site sweep for the correctness target

The dissociation had only ever been tested at SE-argmax sites. Same grid and harness as
`RESULTS.md` §3 (13 layers × 8 positions, dense probe on the rms-normalized residual,
N_TRAIN = 2048, validation AUROC), target `judge_binary`, C tuned on validation over §12's
grid, seeds 42/43 (seed-mean shown; per-seed spread ≤ .01 in the argmax region). Precedent
for answer-side truth positions: SAPLMA (Azaria & Mitchell 2023); SEP (2406.15927)
accuracy probes at TBG/SLT. These are selection metrics (best-C validation AUROC), the
same mild optimism convention as §3's grid — held-out numbers live in R2b.

### TriviaQA — target `judge_binary`

| layer | tbg | ans0 | slt | eoa | genlast | kw | nl | qlast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | .6809 | .6894 | .6750 | .6980 | .6986 | .6847 | .6769 | .6842 |
| 8 | .7242 | .7430 | .7281 | .7548 | .7576 | .7118 | .7383 | .7499 |
| 12 | .7654 | .7763 | .8315 | .8109 | .8121 | .7359 | .7505 | .7721 |
| 16 | .8114 | .8057 | .8713 | .8314 | .8318 | .7763 | .7696 | .7960 |
| 20 | .8431 | .8275 | .8815 | .8536 | .8555 | .8021 | .7899 | .8094 |
| 24 | .8613 | .8479 | **.8862** | .8555 | .8569 | .8333 | .8126 | .8253 |
| 28 | .8641 | .8406 | .8730 | .8508 | .8537 | .8361 | .8036 | .8057 |
| 32 | .8599 | .8156 | .8525 | .8483 | .8508 | .8288 | .7951 | .7878 |
| 36 | .8602 | .8141 | .8481 | .8431 | .8464 | .8274 | .7948 | .7876 |
| 38 | .8635 | .8162 | .8458 | .8478 | .8496 | .8306 | .7954 | .7923 |
| 39 | .8632 | .8339 | .8434 | .8406 | .8422 | .8332 | .7964 | .7979 |
| 40 | .8620 | .8202 | .8332 | .8278 | .8313 | .8321 | .7921 | .7972 |
| 41 | .8479 | .7931 | .8229 | .8141 | .8206 | .8226 | .7891 | .7821 |

### PopQA — target `judge_binary`

| layer | tbg | ans0 | slt | eoa | genlast | kw | nl | qlast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | .7979 | .8427 | .8433 | .8532 | .8532 | .7924 | .7942 | .7992 |
| 8 | .8681 | .8847 | .8721 | .8862 | .8862 | .8722 | .8819 | .8792 |
| 12 | .8903 | .9035 | .9097 | .9021 | .9021 | .8818 | .8858 | .8917 |
| 16 | .9044 | .9184 | .9284 | .9124 | .9124 | .8911 | .8940 | .9005 |
| 20 | .9205 | .9230 | .9339 | .9196 | .9196 | .9004 | .8966 | .9088 |
| 24 | .9232 | .9305 | **.9350** | .9200 | .9200 | .9059 | .9006 | .9146 |
| 28 | .9260 | .9169 | .9236 | .9104 | .9104 | .9069 | .9049 | .9141 |
| 32 | .9258 | .9166 | .9170 | .9060 | .9060 | .9074 | .8977 | .9060 |
| 36 | .9243 | .9101 | .9153 | .8984 | .8984 | .9089 | .8922 | .8954 |
| 38 | .9235 | .9109 | .9151 | .8905 | .8905 | .9102 | .8940 | .8945 |
| 39 | .9255 | .9158 | .9144 | .8909 | .8909 | .9104 | .9009 | .8982 |
| 40 | .9262 | .9131 | .9068 | .8735 | .8735 | .9113 | .8992 | .8971 |
| 41 | .9245 | .9089 | .8954 | .8650 | .8650 | .9086 | .8965 | .8965 |

### Binarized token-entropy target (tokH = `greedy_entropy_full[0]` > train median)

Peaks, for the record (full grids in `results/retest/r1_*.json`): TriviaQA
**L39/tbg .9065** (tbg column .87–.91 from L28 up; slt peaks .8881 at L24); PopQA
**L40/tbg .9832** (tbg ≥ .977 from L24 up). The probe is reading the model's own
next-token entropy back out of the late-TBG state — the state that computes that
distribution — consistent with §13's mechanism story.

### Carried-forward sites, and where the argmaxes differ

| site | TriviaQA | PopQA |
|---|---|---|
| SE-argmax TBG (§3, unchanged) | L40/tbg | L24/tbg |
| SE-argmax SLT (§3, unchanged) | L28/slt | L24/slt |
| **judge-argmax (new)** | **L24/slt** (.8862) | **L24/slt** (.9350) |
| **tokH-argmax (new)** | **L39/tbg** (.9065) | **L40/tbg** (.9832) |

**What it says.** The suspicion behind this task is confirmed at the first gate: the
correctness target has its own geography, and it is not the SE geography. On **both
datasets independently** `judge_binary` peaks at **L24/slt** — answer-side, mid-depth —
exactly the SAPLMA statement-end pattern, while SE peaks at TBG (L40 / L24, §3). On
TriviaQA the SE-chosen site costs the correctness reader .024 (.8862 → .8620 at L40/tbg,
and the previous document's only correctness readers were diffmeans at SE sites); on PopQA
it costs .012 against the tbg column (.9350 vs .9262 at L40/tbg). The judge grid is also
answer-side-shaped in a way the SE grid is not: on TriviaQA `slt` beats `tbg` at every
layer 12–24 by .04–.07, whereas §3's SE grid has `tbg` winning the top layers. PopQA's
judge-argmax coincides with its SE-argmax-SLT site (both L24/slt) — that cell is one site
wearing two names below. Note `judge_binary` is at least as linearly readable at its own
site (.886/.935 val) as `se_discrete` is at its own (.859/.919, §3) — a first strike
against "correctness is not readable": it was simply never given its own reader or site.

## R2. Matched readers and whitened geometry

Sites: {SE-argmax TBG, SE-argmax SLT, judge-argmax, tokH-argmax} × both datasets (PopQA's
judge-argmax = its SE-argmax SLT site, one site with two names). Readers built per seed on
training rows: tuned-C dense probes on each target (`probe_lr_judge` is the reader the
previous document never built), raw diff-in-means, and Ledoit–Wolf-whitened diff-in-means
Σ⁻¹(μ₊−μ₋) (the Fisher/LDA direction; Marks & Tegmark 2310.06824). Where sites coincide
with banked steering sites the rebuilt diffmeans match the banked vectors at cos = 1.0000.

### R2a — the cosine panels (seed 42; seed 43 in the note)

Each panel answers a different question: do the **mean displacements** align (raw), and do
the **discriminative axes** align (whitened / probe weights).

| cosine | T-se-tbg | T-se-slt | T-jdg L24/slt | T-tokH L39/tbg | P-se-tbg | P-slt (=jdg) | P-tokH L40/tbg |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw (`se`,`corr`) | **−.971** | −.963 | −.966 | −.967 | **−.986** | −.968 | −.983 |
| raw (`se`,`tokH`) | .949 | .965 | .967 | .956 | .878 | .762 | .799 |
| raw (`corr`,`tokH`) | −.910 | −.924 | −.926 | −.918 | −.903 | −.789 | −.821 |
| whitened (`se`,`corr`) | **−.290** | −.322 | −.306 | −.299 | **−.116** | −.130 | −.066 |
| whitened (`se`,`tokH`) | .318 | .392 | .388 | .320 | .108 | .163 | .075 |
| whitened (`corr`,`tokH`) | −.186 | −.247 | −.230 | −.193 | −.078 | −.104 | −.051 |
| probes (`se`,`judge`) | −.459 | −.608 | −.609 | −.505 | −.373 | −.634 | −.497 |
| probes (`se`,`tokH`) | .539 | .640 | .633 | .588 | .519 | .502 | .463 |
| probes (`judge`,`tokH`) | −.380 | −.507 | −.499 | −.422 | −.244 | −.407 | −.350 |

Seed 43 agrees within ±.05 everywhere except PopQA-tbg's probe (`se`,`judge`) cell
(−.708 vs −.373 — probe weights at tiny C are draw-sensitive there); the whitened panel
moves ≤ .04 across seeds.

**What it says.** The −.97 collinearity is a property of **mean displacements only**. The
same two labels, at the same sites, on the same rows, give discriminative axes at
cos −.07…−.32 (whitened) and −.37…−.63 (probe weights). A1's "same axis with the sign
flipped" was an artifact of reading everything through raw diffmeans in a state space
whose dominant variance directions are shared: Σ⁻¹ strips the common covariance and most
of the alignment goes with it. PopQA — the dataset the old document called most
confounded — has the **least** aligned discriminative axes (whitened −.07…−.15).

### R2b — every reader on every target, held-out, with CIs

At the judge-argmax site (the full matrix; other sites in `results/retest/r2_*.json`).
Signed projections, as oriented — a value near 0 reads the label's other end. Seed 42;
seed 43 within ±.01 unless noted. n = 5,334 (T) / 10,999 (P) test rows.

**TriviaQA, L24/slt:**

| reader | `se_discrete` | `judge_binary` | `f1_50` | tokH |
|---|---:|---:|---:|---:|
| `diffmean_se` | .836 [.825,.847] | .203 [.189,.217] | .219 [.204,.233] | .829 [.818,.839] |
| `diffmean_corr` | .160 [.149,.171] | .814 [.801,.827] | .799 [.786,.813] | .168 [.158,.178] |
| `diffmean_tokH` | .828 [.817,.839] | .216 [.201,.229] | .230 [.215,.244] | .831 [.820,.841] |
| `diffmean_se_w` | .804 [.791,.816] | .200 [.187,.213] | .217 [.203,.231] | .779 [.767,.792] |
| `diffmean_corr_w` | .234 [.219,.247] | .813 [.799,.827] | .799 [.784,.815] | .272 [.258,.286] |
| `diffmean_tokH_w` | .787 [.775,.799] | .226 [.211,.241] | .237 [.221,.251] | .813 [.800,.824] |
| `probe_lr_se` | **.891 [.882,.899]** | .124 [.114,.134] | .141 [.129,.151] | .871 [.862,.880] |
| `probe_lr_judge` | .126 [.116,.136] | **.898 [.888,.907]** | .880 [.869,.890] | .154 [.143,.163] |
| `probe_lr_tokH` | .880 [.870,.889] | .137 [.126,.148] | .152 [.140,.163] | **.897 [.888,.905]** |

**PopQA, L24/slt (= judge-argmax = SE-argmax SLT):**

| reader | `se_discrete` | `judge_binary` | `f1_50` | tokH |
|---|---:|---:|---:|---:|
| `diffmean_se` | .828 [.820,.836] | .202 [.194,.211] | .221 [.212,.229] | .863 [.856,.869] |
| `diffmean_corr` | .178 [.170,.186] | .803 [.795,.811] | .785 [.776,.793] | .134 [.128,.141] |
| `diffmean_tokH` | .753 [.743,.763] | .259 [.249,.268] | .270 [.261,.280] | .858 [.851,.864] |
| `diffmean_se_w` | .887 [.880,.896] | .185 [.175,.193] | .212 [.202,.222] | .790 [.782,.799] |
| `diffmean_corr_w` | .158 [.148,.166] | .850 [.842,.858] | .817 [.808,.825] | .216 [.207,.224] |
| `diffmean_tokH_w` | .842 [.834,.850] | .187 [.179,.195] | .205 [.196,.213] | .926 [.921,.930] |
| `probe_lr_se` | **.958 [.954,.961]** | .094 [.088,.100] | .127 [.119,.133] | .923 [.918,.928] |
| `probe_lr_judge` | .062 [.058,.067] | **.926 [.921,.931]** | .893 [.887,.899] | .103 [.097,.109] |
| `probe_lr_tokH` | .917 [.911,.922] | .122 [.116,.128] | .145 [.138,.153] | **.970 [.968,.973]** |

The site spread of the correctness reader (`probe_lr_judge` → `judge_binary`, s42/s43):

| site | TriviaQA | PopQA |
|---|---|---|
| SE-argmax TBG | .8673/.8710 [.855,.879] | .9117/.9089 [.906,.917] |
| SE-argmax SLT | .8803/.8872 [.870,.891] | .9263/.9255 [.921,.931] |
| judge-argmax | **.8975/.9028** [.888,.907] | = SLT cell |
| tokH-argmax | .8700/.8752 [.858,.881] | .9166/.9138 [.911,.922] |

**What it says.** This is the matched-instrumentation replacement for the old A2/D1, and
it moves the correctness numbers a long way. The old document's best correctness readout
was `diffmean_corr` at an SE site: **.830 (T) / .846 (P)**. The tuned probe at the
correctness target's own site reads **.898 (T) / .926 (P)** — +.07/+.08, CIs nowhere near
overlapping. Correctness is *more* linearly readable than `se_discrete` on TriviaQA
(.898 vs .891) and nearly as readable on PopQA (.926 vs .958). Whitening also fixes the
diffmean correctness reader specifically on PopQA (.803 → .850 at L24/slt) while *hurting*
every SE diffmean on TriviaQA — the covariance correction moves the two axes differently,
which is what "not the same axis" looks like in readout space. The `f1_50` channel tracks
`judge_binary` within .01–.03 for every reader (channel-robust). Cross-readings stay high
(`probe_lr_judge` reads SE flipped at .87/.94) — the labels are correlated and nothing
here denies it; the question dissociation asks is what remains *beyond* that, which is
R3–R6's job.

## R3. Null calibration of the cosine — what does label coupling alone predict?

Fit `judge ~ continuous se_discrete` (logistic, unregularized) on training rows; draw
K = 200 synthetic `judge*` vectors from the fitted probabilities (correctness-beyond-SE
exactly zero by construction); recompute `diffmean_corr*` per draw and its cosine to
`diffmean_se`, raw and whitened. Variant basis adds `greedy_entropy_full[0]`; the
tokH-only basis is Phase R7's row. Bands are the null's [2.5, 97.5] percentiles. Seed 42
shown; seed-by-seed tallies below count all 14 non-alias (site × seed) cells.

| | T-se-tbg | T-se-slt | T-jdg | T-tokH | P-se-tbg | P-slt (=jdg) | P-tokH |
|---|---:|---:|---:|---:|---:|---:|---:|
| observed raw | −.971 | −.963 | −.966 | −.967 | −.986 | −.968 | −.983 |
| null raw (se) | [−.978,−.968] | [−.983,−.967] | [−.985,−.971] | [−.976,−.966] | [−.996,−.976] | [−.988,−.952] | [−.991,−.960] |
| observed whitened | **−.290** | −.322 | −.306 | −.299 | **−.116** | −.130 | −.066 |
| null whit (se) | [−.412,−.317] | [−.442,−.332] | [−.436,−.328] | [−.417,−.326] | [−.341,−.230] | [−.357,−.242] | [−.308,−.180] |
| null whit (se+tokH) | [−.356,−.249] | [−.386,−.293] | [−.384,−.292] | [−.361,−.262] | [−.237,−.128] | [−.266,−.146] | [−.201,−.071] |

**Raw panel: the −.97 is exactly what label coupling predicts.** The observed raw cosine
sits inside the se-basis null band in 9 of 14 cells and *above* it (|cos| **below** null)
in the other 5 (all four TriviaQA sites at seed 43, PopQA-slt at 42). Nowhere is the
observed raw collinearity stronger than the null. So A1's headline number carried **zero
evidence about representation** — a state that encoded correctness and SE in perfectly
separated subspaces would have produced the same −.97, given these labels' correlation.

**Whitened panel: the observed coupling is reliably below the null.** Against the se-only
basis the observed whitened |cos| falls **below the null band in 14 of 14 cells** — label
coupling alone predicts whitened cos ≈ −.36 (T) / −.29 (P) and the data shows −.26…−.32
(T) / −.07…−.15 (P). Against the stricter se+tokH basis: below the band in 6/14, inside
in 8/14, above in none. Per the pinned interpretation, **a correctness-specific mean
component exists already at this level** — the correctness contrast is *less* aligned
with the SE axis than a correctness signal made purely of SE would be, and PopQA (whose
raw cosine was the most extreme, −.986) shows the largest gap.

**tokH basis (R7 row).** Observed (tokH, corr) cosines sit inside the tokH-basis null in
most cells, but on TriviaQA 4 of 8 whitened cells are slightly *more* negative than the
band (e.g. −.193 vs [−.188, −.097]) — the pre-registered "above the band" case: judge and
tokH share structure beyond their scalar coupling there (the shared piece is SE-shaped,
which the tokH-only simulation omits). On PopQA all 6 cells sit inside the band.

## R5. Residualized-target probing — run as pinned, then corrected mid-flight

The axes-swapped §9. As pinned: OLS `target ~ basis` on training rows, frozen coefficients
everywhere, residual binarized at the training median; dense tuned-C probe + the SAE arm
(§9 selection rule, K on validation); controls = basis-prediction floor, shuffled-residual
probe, random direction. Bases: `U` = {`se_discrete`, `se_lnrao`, `n_clusters`};
`U+` = `U` ∪ {`E`(site), `E²`, mean greedy token logprob, `p_true`, `greedy_logit_gap[0]`};
`E+` = the output-scalar set alone (Phase R7's basis). Mirrors: `se~CF` and `tokH~CF`
residualize the continuous uncertainty scalars on {`judge_binary`, `f1_squad`}.

**The design correction, found by running the control properly.** §9's residual trick is
sound for a *continuous* target. For a **binary** target it degenerates: with base rate
.77 (T judge), `resid = y − ŷ` puts every wrong row below every correct row, the training
median lands inside the correct cluster, and the binarized residual becomes the
deterministic rule *"correct ∧ ŷ below a threshold"*. A basis-only reader that knows that
rule (the plug-in P(resid-bin | basis), still using no labels) reads the "residual" label
at **.92–.95 (T) / .92–.93 (P)** — far above any state probe below. The spec's linear-ŷ
ctl floor (.34–.59 as pinned) understates what the basis knows by ~.4, so
probe-beats-that-floor is not evidence of anything. The judge-direction cells below are
therefore reported as run but **carry no dissociation weight**; the corrected instrument
(incremental AUROC on `judge_binary` itself — state score added to the basis features,
both models fit on train, read on test) follows. The mirrors are unaffected: their
corrected basis-only floors (a classifier of the residual label on the basis features)
sit at .51–.59, right where the pinned ctl floor sits.

### As pinned (dense tuned-C / SAE top-K, held-out; seed 42, seed 43 within ±.02)

TriviaQA (judge-residual rows: no dissociation weight, see above):

| site | spec | basis R² | dense | sparse (K) | basis-only floor | shuffled | random |
|---|---|---:|---:|---:|---:|---:|---:|
| jdg L24/slt | judge~U | .464 | .684 [.669,.699] | .644 (64) | **.938** | .498 | .504 |
| jdg L24/slt | judge~U+ | .495 | .706 [.691,.719] | .686 (128) | **.945** | .469 | .502 |
| jdg L24/slt | judge~E+ | .424 | .726 [.713,.740] | .701 (128) | **.935** | .471 | .507 |
| se-tbg L40 | judge~U+ | .512 | .668 [.653,.681] | .671 (64) | **.947** | .499 | .484 |
| jdg L24/slt | **se~CF** | .434 | **.800 [.789,.812]** | .773 (32) | .588 | .533 | .493 |
| jdg L24/slt | **tokH~CF** | .448 | **.798 [.787,.810]** | .764 (128) | .529 | .489 | .491 |

PopQA:

| site | spec | basis R² | dense | sparse (K) | basis-only floor | shuffled | random |
|---|---|---:|---:|---:|---:|---:|---:|
| slt L24 (=jdg) | judge~U | .431 | .755 [.746,.763] | .738 (128) | **.921** | .502 | .419 |
| slt L24 (=jdg) | judge~U+ | .463 | .827 [.819,.834] | .763 (128) | **.926** | .502 | .438 |
| slt L24 (=jdg) | judge~E+ | .414 | .807 [.799,.816] | .744 (128) | **.923** | .469 | .452 |
| slt L24 (=jdg) | **se~CF** | .430 | **.829 [.820,.836]** | .804 (128) | .543 | .456 | .416 |
| slt L24 (=jdg) | **tokH~CF** | .474 | **.908 [.903,.913]** | .889 (128) | .578 | .455 | .396 |

(Other sites and seeds in `results/retest/r5_*.json` + `r5_ctl_opt.json` +
`r5_corrected.json`; the mirror numbers move ≤ .03 across sites.)

**The mirror stands, decisively.** SE-beyond-correctness reads **.79–.80 (T) / .81–.83
(P)** and tokH-beyond-correctness **.79–.83 (T) / .90–.92 (P)** against corrected
basis-only floors of .51–.59, shuffled probes at .5, on both datasets, both seeds, every
site. The uncertainty axes carry large linear signal that correctness labels cannot
account for — the expected direction, now properly controlled.

### Corrected judge-direction instrument: incremental AUROC on `judge_binary`

Basis-only logistic vs basis + the site's `probe_lr_judge` score, both fit on training
rows, both read once on test; Δ = augmented − basis, paired bootstrap 95% CI. s42 / s43.

| dataset, site | vs `U` | vs `E+` | vs **`U+`** (the full basis) |
|---|---:|---:|---:|
| T, judge-argmax L24/slt | **+.0192** [.011,.027] / **+.0162** [.009,.023] | **+.0169** [.009,.025] / **+.0141** [.006,.022] | +.0035 [−.003,.010] / +.0022 [−.004,.008] |
| T, SE-argmax L40/tbg | −.0201 / −.0261 | −.0437 / −.0508 | −.0450 / −.0520 |
| P, L24/slt (=judge-argmax) | **+.0326** [.027,.038] / **+.0358** [.031,.041] | **+.0356** [.030,.041] / **+.0340** [.029,.040] | **+.0207** [.016,.025] / **+.0214** [.017,.026] |
| P, SE-argmax L24/tbg | +.0182 / +.0258 | −.0017 / +.0073 | −.0031 [−.008,.002] / +.0070 [.004,.010] |

**What it says — the decisive result of the document.**

1. **PopQA: correctness is represented beyond everything the store knows about
   uncertainty.** At L24/slt the state score adds **+.021 AUROC on top of the full `U+`
   basis** — semantic-uncertainty scalars *and* every stored output-distribution scalar —
   both seeds, CIs clear of zero (.9076 → .9283 at seed 42). Dissociation exists, and it
   is not small: the increment is ~2× the SAE arm's best raw-SE increment anywhere in
   `RESULTS.md`.
2. **TriviaQA: graded and site-dependent.** At its judge site the state adds beyond the
   semantic-uncertainty basis `U` (+.016–.019, CIs clear) and beyond the output-scalar
   basis `E+` (+.014–.017) — but **not beyond their union `U+`** (+.002–.004, n.s.). What
   the TriviaQA state knows about correctness beyond SE is (linearly) the same
   information carried by `p_true`, the logit gap, mean logprob and `E`; PopQA's state
   knows strictly more than both families combined.
3. **The site matters exactly as R1 predicted.** At SE-chosen sites the increments shrink
   or go negative (a stacked score from the wrong site *hurts* held-out calibration on
   TriviaQA). Testing dissociation only at SE-argmax sites — the previous document's
   design — is the difference between finding it and not.

## R4. Orthogonal components and erasure

### R4a — the correctness direction with its SE projection removed

`v_corr⊥` = `diffmean_corr` minus its projection onto span{`diffmean_se`} (stricter:
span{`diffmean_se`, `diffmean_tokH`}), in raw and whitened space, plus `probe_lr_judge`
orthogonalized to `probe_lr_se`; control = a random direction through the identical
orthogonalization. Read against `judge_binary` on test rows, overall and within SE
strata (all strata n > 700 — no n < 100 flags). "frac" = ‖v⊥‖ of the unit vector, i.e.
how much of the direction survives; the cosine said this could be up to ~25%, and it is.
Seed 42 at the judge-argmax sites (all sites/seeds in `results/retest/r4_*.json`):

| direction (judge-argmax site) | frac | judge all | within hiSE | within loSE |
|---|---:|---:|---:|---:|
| T `corr⊥se` (raw) | .26 | .600 [.581,.619] | .661 [.635,.686] | .630 [.586,.672] |
| T `corr⊥{se,tokH}` (raw) | .26 | .602 [.583,.621] | .653 [.628,.678] | .637 [.594,.682] |
| T `corr⊥se` (whitened) | .95 | .761 [.745,.778] | .711 [.688,.736] | .679 [.638,.720] |
| T `probe_judge⊥probe_se` | .79 | **.849 [.836,.862]** | **.799 [.780,.820]** | **.759 [.722,.793]** |
| T random-⊥ control | 1.0 | .468 | .442 | .465 |
| T `se_discrete` scalar floor | — | .106 (fl .894) | .265 (fl .735) | .237 (fl .763) |
| T ref `probe_lr_judge` (unorth.) | — | .898 | .811 | .807 |
| P `corr⊥se` (raw) | .25 | .624 [.612,.635] | .734 [.718,.748] | .544 [.516,.571] |
| P `corr⊥se` (whitened) | .99 | .832 [.823,.840] | .779 [.763,.794] | .709 [.686,.732] |
| P `probe_judge⊥probe_se` | .77 | **.823 [.814,.831]** | **.833 [.820,.845]** | .688 [.664,.711] |
| P random-⊥ control | 1.0 | .497 | .466 | .571 |
| P `se_discrete` scalar floor | — | .109 (fl .891) | .211 (fl .789) | .316 (fl .684) |
| P ref `probe_lr_judge` (unorth.) | — | .926 | .888 | .779 |

**What it says.** The component the cosine could not see is real and it reads correctness.
The raw orthogonal remnant (24–36% of the diffmean) reads judge within-hiSE at .65–.73,
right-side-up, against a random-⊥ control of .44–.51. The probe-weight version is the
strong result: remove everything `probe_lr_se` spans from `probe_lr_judge` and **79–89%
of the weight vector survives and still reads judge at .82–.85 overall and .76–.83 within
strata** — above the SE-scalar floor's within-stratum magnitude at the judge sites
(T hiSE: .799 vs floor .735-flipped; P hiSE: .833 vs .789-flipped), in the *correct*
orientation (the floor reads judge only through anti-SE). D1's conclusion — "within-
stratum correctness ≈ what the SE label itself carries" — fails exactly where the ≤25%
component was predicted to live. (One instability, named: at TriviaQA/PopQA TBG sites at
tiny C the probe-⊥ readout swings across seeds (P-tbg s42 .354 vs s43 .800); the
answer-side sites are stable.)

### R4b — INLP, and the erasure that actually worked (LEACE)

INLP (Ravfogel et al. 2004.07667), r = 1…10 removals, erasure predictor C = 1.0, fresh
tuned-C probe per r; shuffled-label erasure as control. Judge-argmax sites, seed 42
(pattern identical everywhere):

| role | probe r=0 → r=10 | erase-label AUROC r=0 → r=10 | shuffled ctl (probe / erase) |
|---|---|---|---|
| T erase SE → probe judge | .8975 → .8947 [.885,.904] | .845 → .811 | .8975 → .8975 / .49→.51 |
| T erase judge → probe SE | .8905 → .8723 [.863,.881] | .857 → .808 | flat / .50 |
| P erase SE → probe judge | .9263 → .9243 [.919,.929] | .925 → .906 | flat / .45→.47 |
| P erase judge → probe SE | .9576 → .9565 [.953,.960] | .889 → .867 | flat / .47 |
| T erase tokH → probe judge | .8975 → .8952 | .859 → .818 | flat |
| P erase tokH → probe judge | .9263 → .9233 | .961 → .930 | flat |

**INLP is reported honestly as too weak to decide anything here.** Ten removals leave the
erase-label itself readable at .64–.94 — in this 3,584-dim space with signal spread over
many directions, removing the top-10 LR directions is a scratch, and "the probe survived"
is uninformative when the erased concept also survived. (At PopQA-TBG the SE erase AUROC
*rises* under erasure — each removal exposes the next of many SE directions.)

**LEACE (Belrose et al. 2306.03819), the closed-form version, settles it.** Concept =
SE quartile bins (one-hot); after erasure a fresh C = 1.0 probe on the SE label reads
**.50–.54 — the erasure is verified complete** — and a fresh tuned-C probe on
`judge_binary` still reads:

| site | TriviaQA s42/s43 | PopQA s42/s43 |
|---|---|---|
| SE-argmax TBG | .654 / .686 | .679 / .670 |
| SE-argmax SLT | .667 / .677 | .691 / .665 |
| judge-argmax | **.706 [.689,.724] / .715 [.699,.729]** | = SLT cell |
| tokH-argmax | .659 / .693 | .713 / .680 |

Per the pinned interpretation: **linearly readable correctness-beyond-SE exists** — judge
AUROC .65–.72 on a space from which every linear trace of the (binned) SE concept has
been removed, on both datasets, both seeds, every site, highest at the judge-argmax
sites. The previous document's conclusion is overturned by its own missing experiment.
The symmetric statement also holds and is larger where expected (the mirror was never in
doubt). Named caveat: LEACE's guarantee is linear-w.r.t.-the-bins; continuous-SE
nonlinear residue can survive it, which is why R5's incremental test (which controls the
continuous scalars directly) is the primary evidence and this is the converging second
line.

## R6. Matched pairs, and the feature partition at full power

### R6a — the correctness-at-fixed-SE direction from matched pairs

Greedy NN matching without replacement, correct↔wrong within |Δ continuous
`se_discrete`| ≤ .05 (variant: additionally |ΔE| ≤ .1 nat), on the training pool;
direction = mean(h_correct − h_wrong) over pairs; held-out readout on test rows +
within-pair ordering on pairs matched among test rows by the identical procedure.
Balance: mean |ΔSE| ≤ .003 in every cell (the caliper binds at ~1/50 of the SE scale).

| cell | n pairs (tr/te) | mean ΔE | judge AUROC s42/s43 | within-pair s42/s43 | cos vs `corr⊥se` |
|---|---|---:|---:|---:|---:|
| T judge-argmax, SE caliper | 286/777 | −.47 | .746 [.729,.761] / .644 | .713 / .694 | .84/.93 |
| T judge-argmax, SE+E caliper | 201/628 | .00 | .673 [.656,.691] / .555 | .696 / .660 | .76/.85 |
| T se-tbg, SE caliper | 286/777 | −.47 | .810 / .690 | .669 / .612 | .75/.82 |
| T se-tbg, SE+E caliper | 201/628 | .00 | .632 / .485 | .599 / .572 | .61/.69 |
| P L24/slt (=jdg), SE caliper | 324/1748 | −.54 | .718 [.708,.728] / .624 | .650 / .649 | .55/.52 |
| P L24/slt, SE+E caliper | 249/1540 | .00 | .565 / .552 | .619 / .620 | .49/.49 |
| P tokH-argmax, SE caliper | 324/1748 | −.54 | .799 [.791,.809] / .762 | .652 / .626 | .33/.29 |
| P tokH-argmax, SE+E caliper | 249/1540 | .00 | **.681 [.669,.691] / .725** | .612 / .610 | .42/.36 |

**What it says.** Three things. (1) With SE matched to ≤ .003, correct and wrong rows are
still separable: the pair direction reads judge at .62–.81 held-out and orders 60–71% of
*held-out* pairs — the properly-powered replacement for the 77-row-stratum construction,
and it correlates with R4a's orthogonal component (cos .3–.9), i.e. the two phases find
the same object. (2) **The E caliper is the honest asterisk**: SE-matched pairs still
differ in token entropy (mean ΔE ≈ −.5 nat — correct answers are lower-entropy), and
matching E away drops the readout to .55–.68 (T) / .55–.73 (P). A large share of
"correctness at fixed SE" is carried by the token-entropy axis; what survives *both*
calipers is solid on PopQA at L40/tbg (.68/.73, CIs clear of .5; within-pair .61) and
seed-fragile on TriviaQA (.49–.67 — 200 train pairs is still thin there). (3) Within-pair
ordering survives the E caliper everywhere (.57–.70) even where the global AUROC drops —
the direction ranks *matched opponents* better than it ranks the unmatched population.

### R6b — the Patel-style partition on the full pool

Mann–Whitney per feature, both-strata requirement, BH-FDR .05, on **all rows**
(discovery-only; per-cell n below), with the lenient p < .001 row making power visible.
Strata thresholds from the seed-42 train draw; seed 43 shifts every count by ≤ 2.
Old D2 (n = 2,048, SE sites) found pure-corr **0–5**; the starved cells were 77 (T) /
112 (P) rows — now 269 / 663.

| site (pool cells hi∧cor/lo∧cor/hi∧wr/lo∧wr) | rule | pure-SE | **pure-corr** | confounded |
|---|---|---:|---:|---:|
| T se-tbg (1115/4563/1435/269) | FDR .05 | 33 | 2 | 21 |
| T judge-argmax L24/slt | FDR .05 | 50 | **21** | 20 |
| T judge-argmax | p < .001 | 56 | 25 | 22 |
| P se-tbg (1410/2597/8377/663) | FDR .05 | 64 | **17** | 93 |
| P L24/slt (=jdg) | FDR .05 | 132 | **42** | 107 |
| P L24/slt | p < .001 | 109 | 46 | 92 |

tokH-axis classes (R7): T judge-argmax pure-tokH 58 / pure-corr **38**; P tokH-argmax
pure-tokH 83 / pure-corr **70**. Jaccard(pure-SE, pure-tokH) = **.01–.22** — the two
uncertainty axes' pure feature sets are nearly disjoint. Where the frozen §1 probe's K
features land (rest in "neither"): at the TBG sites 20–28 in **confounded** and 0 in
pure-corr; at T-slt 8 pure-SE / 3 confounded / 1 pure-corr — the production probe reads
the entangled axis, never more than 1 feature of the pure-correctness vocabulary.

**What it says.** D2's near-empty pure-correctness class was **power censoring, not
absence**: at full pool size the class exists on both datasets (21–22 features at
TriviaQA's judge site, 42 at PopQA's), and the lenient row shows the FDR discipline is no
longer what's binding. The site matters at feature granularity too — at SE-chosen sites
pure-corr stays at 2 (T), which is exactly the cell the old document reported. The
SE-pure and tokH-pure vocabularies barely overlap, foreshadowing R7's construct
distinction; and the probe living in "confounded" explains why probe-based readouts never
separated the axes.

## R7. The Patel replication — token entropy × correctness

Uncertainty label: next-token entropy at TBG (`greedy_entropy_full[0]`, train-median
binarized) — the closest open-ended analog of Patel et al. 2604.19974's MCQ answer-token
entropy. The corrected battery ran with tokH in place of SE at the judge- and
tokH-argmax sites inside R2–R6; this section collects those rows.

| test | TriviaQA | PopQA |
|---|---|---|
| R2a raw cos(tokH, corr) | −.918…−.926 | −.789…−.821 |
| R2a whitened cos(tokH, corr) | −.19…−.23 | −.05…−.10 |
| R2a probe cos(judge, tokH) | −.42…−.50 | −.35…−.41 |
| R3 raw cos vs tokH-coupling null | inside band (5/8), *more* negative (3/8) | inside band (6/6) |
| R3 whitened vs null | more negative than band 4/8 — judge–tokH share extra (SE-shaped) structure | inside band 6/6 |
| R4a `corr⊥tokH` (raw, frac .38–.62) → judge | .66–.69 [rand-⊥ ctl .44–.53] | .68–.72 [rand-⊥ ctl .45–.56] |
| R4a `corr⊥tokH` (whitened) → judge | .75–.80 | .78–.84 |
| R4b INLP erase tokH → probe judge | .898 → .895 (erase .86→.82; too-weak caveat as in R4b) | .926 → .923 (erase .96→.93) |
| R5 incremental AUROC vs `E+` basis | **+.0169/+.0141** (CI-clear, judge site) | **+.0356/+.0340** (CI-clear) |
| R5 mirror `tokH~CF` (residual probe vs floor) | .80 vs .53 | .91 vs .58 |
| R6b tokH-axis partition (FDR .05) | pure-tokH 58, **pure-corr 38** (judge site) | pure-tokH 83, **pure-corr 70** (tokH site) |
| R6b Jaccard(pure-SE, pure-tokH) | .06–.22 | .01–.06 |

**Pinned three-way readout: dissociation appears under BOTH SE and tokH conditioning.**
Correctness separates from symbol-level uncertainty (incremental AUROC beyond the full
output-scalar basis `E+`, CIs clear of zero on both datasets; 38–70 pure-correctness
features on the tokH axis) and from meaning-level uncertainty (R4/R5/R6 above). The
earlier no-dissociation result was the design artifact, and **Patel et al.'s
MCQ dissociation structure transfers to short-form open-ended generation** — with one
dataset-shaped qualifier from R5: on TriviaQA correctness separates from each
uncertainty family *alone* but not from their union `U+` (the beyond-both increment is
+.002–.004, n.s.), while on PopQA it clears even the union (+.021). A second
Patel-consistent detail: the pure-SE and pure-tokH feature vocabularies are nearly
disjoint (Jaccard ≤ .22) — "semantic uncertainty" and "token uncertainty" are different
feature sets in these SAEs, not one set read twice.

## R8. Label-noise robustness — the headline cells under `f1_squad ≥ 50`

The headline cells of R4b, R5, and R6a, rerun at the judge-argmax sites with the second
correctness channel. s42 / s43.

| cell | `judge_binary` (from above) | `f1_50` |
|---|---|---|
| R4b headline: probe after LEACE (SE verified erased ≤ .54) | T .706/.715, P .691/.665 | T **.696/.681**, P **.690/.669** |
| R5 headline: incremental AUROC vs `U+` | T +.004/+.002 n.s., P **+.021/+.021** | T +.000/−.006 n.s., P **+.020 [.015,.025] /+.019 [.014,.024]** |
| R6a: matched-pair dir → corr label (SE caliper) | T .746/.644, P .718/.624 | T .674/.607, P .736/.645 |
| R6a: same, SE+E caliper | T .673/.555, P .565/.552 | T .582/.528, P .558/.570 |
| R6a: within-pair ordering (SE caliper) | T .713/.694, P .650/.649 | T .680/.632, P .664/.656 |
| (INLP se→f1, same too-weak caveat as R4b) | — | flat, r10 ≈ r0 ≈ shuffled |

**Nothing is channel-dependent.** Every positive stays positive (LEACE survival .67–.70;
PopQA's beyond-`U+` increment +.019/+.020 with CIs clear), and TriviaQA's beyond-`U+`
null stays null in the same cells. The dissociation claims below are therefore reported
as findings, not artifacts of one grading channel.

---

## Verdict table — pre-registered interpretations vs observed outcomes

| phase, pinned interpretation | observed | verdict |
|---|---|---|
| R3-raw: obs \|cos\| inside null band ⇒ mean-level collinearity fully explained by label coupling | inside/above band 14/14 cells | **confirmed** — A1's −.97 carried no representational evidence |
| R3-whitened: obs \|cos\| below band ⇒ correctness-specific mean component exists | below band **14/14** (se basis) | **confirmed** |
| R5: probe > ctl floor ⇒ correctness represented beyond the uncertainty axis | as-pinned design degenerate for a binary target (honest basis-only floor .92–.95 > any probe); corrected incremental-AUROC instrument: P **+.021 beyond full `U+`** (CI-clear, both seeds, both channels); T +.016–.019 beyond `U` alone, ~0 beyond `U+` | **dissociation established on PopQA beyond everything stored; on TriviaQA beyond each uncertainty family but not their union** |
| R4: judge survives SE erasure above shuffled ctl and .5 ⇒ previous conclusion overturned | INLP: erasure too weak to decide (erase-label survives at .64–.93). LEACE: SE verified at .50–.54, judge still **.65–.72**, all sites, both datasets, both channels | **confirmed — overturned** (with the linear-w.r.t.-bins caveat; R5 is the primary evidence, this the converging line) |
| R4/R5 all-at-floor ⇒ entanglement confirmed with stronger instrumentation | did not occur | — |
| R7 three-way: dissociation under both SE and tokH ⇒ earlier result was the design artifact; Patel transfers to open-ended generation | both (R5-incremental beyond `E+` CI-clear both datasets; 21–70 pure-corr features on either axis at the right sites) | **confirmed — "both" branch**, with TriviaQA's `U+`-union qualifier |

## Against the superseded conclusions of STEERING_OFFLINE_RESULTS.md

| old claim (A1/D1–D3) | status after this document |
|---|---|
| A1: `diffmean_se` ≈ −`diffmean_corr` (cos −.96…−.997), "the same axis with the sign flipped" | **number replicates, interpretation overturned**: R3 shows −.97 is exactly the label-coupling null; the discriminative axes sit at cos −.07…−.32 (whitened) / −.37…−.63 (probes) |
| D1: within-stratum correctness readout of every direction ≈ the SE-label floor; "dissociation bounded by the label floor, not demonstrated beyond it" | **overturned**: the floor comparison was run only on SE-built directions at SE-chosen sites; `probe_lr_judge⊥probe_lr_se` reads .76–.83 within strata (right-side-up, above the floor's magnitude at judge sites), and LEACE-erased states still read judge at .65–.72 |
| D2: pure-correctness features ≈ 0 everywhere | **overturned — power censoring**: 21–42 pure-corr features at full pool at the judge sites (38–70 on the tokH axis); still ~2 at the SE-TBG sites the old test used |
| D3: "correctness-beyond-SE is not recoverable from these sites" | **overturned** at the sites R1 selects, by three independent designs (R5 incremental, R4 LEACE, R6a matched pairs) |
| D1/D3's asymmetry: SE-beyond-correctness is the larger object | **survives**: the mirror reads .79–.92 vs floors ~.55 while the judge increments are +.02 — the axes dissociate, but unequally |
| A6/D4: the frozen sparse probe reads the entangled axis | **survives**: 20–28 of its K features land in "confounded", 0–1 in pure-corr |
| Phase 2/3/5 (steering ops, C2 temperature result, E2 fix-rate ceiling) | untouched — those are intervention results; this document measured none |

## Not established here

- **No interventions were measured**; every claim is correlational readout of stored
  states. Whether the correctness-beyond-SE component is causally *used* by the model —
  or steerable — is exactly the open question for a generation run.
- Nonlinear correctness-beyond-SE: all probes and erasures are linear; LEACE's guarantee
  is linear-w.r.t.-quartile-bins, and continuous/nonlinear SE residue can survive it
  (which is why R5's incremental test, which controls the continuous scalars directly,
  carries the primary weight).
- The as-pinned R5 judge-residual cells establish nothing either way (binarized binary
  residual ≈ basis-determined); by extension, §9's residual design should not be
  axis-swapped onto binary targets without the plug-in floor computed here.
- tokH conditioning uses the single TBG scalar; multi-token entropy aggregates (§3's
  `max` variant) were not partialled — same conservative bound §9 named.
- `judge_binary` and `f1_50` agree, but both grade the same greedy string; a
  human-verified correctness channel was not available.
- The whitened geometry is Ledoit–Wolf-estimator-specific (2,048 train rows for a
  3,584-dim covariance); other shrinkage choices will move the whitened cosines, though
  not plausibly across the R3 null bands, which use the same estimator.
- INLP as run (10 removals, C = 1.0) is reported as inconclusive, not as evidence against
  erasure-based dissociation.
- TriviaQA's R6a SE+E-caliper cells are seed-fragile (~200 train pairs); PopQA's are not.
- Han coordinates and the final-layer readout were not revisited.

## Config appendix (realized)

Sites: T se-tbg L40/tbg, se-slt L28/slt, judge L24/slt, tokH L39/tbg; P se-tbg L24/tbg,
se-slt = judge L24/slt, tokH L40/tbg. n = 7,382 / 13,047 rows (test 5,334 / 10,999);
judge base rate .769 / .308. Tuned C landed at 1e-5…1e-3 everywhere; R5 sparse K landed
at 64/128 in 29 of 40 cells (selector `mutual_information` T / `spearman` P, guide =
continuous residual on fit rows, probe C = 1.0). Whitening: Ledoit–Wolf on the 2,048
train rows, per seed. LEACE: `concept-erasure` (ephemeral install), concept = one-hot SE
quartile bins from the train draw. Some C = 1e6 / C = 1.0 lbfgs fits hit max_iter (2,000
–5,000; convergence warnings recorded) — post-LEACE SE checks are additionally covered by
LEACE's linear-optimality guarantee. Bootstrap: 1,000 prompt-level resamples throughout.
Grid fetch: 26 remote layer passes (~30 GB egress); grid-only slices deleted after R1,
cache kept at 750 MB (`data/cache/steering`). Raw outputs: `results/retest/*.json`,
directions `results/retest/dirs_*.npz`; code `analysis/retest/` (rt.py + r1–r8 scripts;
`analysis/steering/common.py::positions()` extended to the full 8-position §3 map).
