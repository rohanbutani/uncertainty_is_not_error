# Steering directions, offline — Gemma-2-9B / TriviaQA + PopQA

Rough first pass over **stored activations only** — no generation, no GPU pass. Code:
[`analysis/steering/`](../analysis/steering/); numbers bank to `results/steering/*.json`.
Runs: `fp1-triviaqa-20260812` and `fp1-popqa-20260814` (the `trivia_qa_long` arm is not used).

**What this can and cannot establish, once:** without running generation under a hook,
nothing here measures the effect of an intervention on re-sampled semantic entropy. What it
does measure is (1) the geometry of candidate steering directions, (2) whether an SE
direction is separable from a correctness direction at the direction and feature level,
(3) what each steering operation does to the hidden state as read through the SAE, and to
the next-token distribution at the one site where that readout is exact, and (4) the
offline part of Study B, the resampling fix-rate. The output is a ranked shortlist for a
later generation run, plus the dissociation results, which stand on their own.

## Protocol, pinned (fail-fast; realized values recorded here)

| knob | realized value |
|---|---|
| rows | joined tables minus p_true exemplars: **7,382** (TriviaQA) / **13,047** (PopQA) |
| split | `RESULTS.md` convention: `rng(seed)` draw, `N_TRAIN=2048`, rest held-out; fit/val re-split `rng(seed+9973)`, `VAL_FRAC=.25` |
| seeds | **42, 43** — every direction built on that seed's training rows; every evaluation on its held-out rows |
| `se_discrete` label | binarized by `sep_best_split` on the training draw (thr .9536 / 1.3257 at seed 42) |
| `judge_binary` | the stored ancestor-verbatim channel; never derived from strings |
| quadrants | **tail quartiles** of `se_discrete`, thresholds from the training draw, × `judge_binary` (mid-SE rows fall in no quadrant) |
| sites | TriviaQA **L40/tbg**, **L28/slt**, **L30/slt** (Han), **final/tbg+slt**; PopQA **L24/tbg**, **L24/slt**, **L30/slt**, **final/tbg+slt** |
| "final" | the last **pre-norm** residual = storage index 42 (notebook layer 41) — fixed by gate 1, not by arithmetic |
| state space | SAE sites: rows rescaled to the train-mean RMS (the notebooks' `SAE_NORM="rms"`) — the space the SAE reads and every Phase 2 op acts in. Final sites: **raw** residual (the unembedding needs it) |
| SAE | `gemma-scope-9b-pt-res-canonical`, width 16k, per-site layer |
| sparse probe | §1 frozen per dataset, applied at every SAE site: TriviaQA `mutual_information`/entropy/K=64, PopQA `spearman`/label/K=32; probe C=1.0. The entropy guide at answer-side sites uses §9's indexing (the entropy this state emits). No §1 config exists off-TBG; this is the recorded choice |
| `probe_lr` | dense logistic, C tuned on validation over §12's grid {1e-5…1} |
| α units | multiples of ‖`diffmean_se`‖ at that site (per seed), for **every** direction |

Not stored / not available: `judge_4way` and `sample_judge_binary_labels` were never
banked for these two runs — Phase 5's judge-channel check is unavailable and the string
channel (`sample_f1 ≥ 50`) carries it alone. SAE codes are encoded on the fly from
`resid_post` (schema §11).

---

## Phase 0 — gates

### Gate 1 — unembedding parity (the gate that makes Phase 3 exact)

`softmax(W_U · RMSNorm(h))` from stored `resid_post` at the TBG position, mirrored from
`ssep.models.logits_head` (RMSNorm in f32 → bf16 tied lm_head → softcap 30 → f32
log-softmax), against the stored `topk_logprobs[0]` / `greedy_entropy_full[0]`, 200
prompts. Candidate final index 42 (layer "41") vs the negative control 41 (layer "40"):

| | TriviaQA L41 | TriviaQA L40 (control) | PopQA L41 | PopQA L40 (control) |
|---|---:|---:|---:|---:|
| median \|Δ entropy\| (nats) | **.0026** | .459 | **.0041** | .844 |
| max \|Δ entropy\| | .099 | 3.39 | .144 | 3.36 |
| top-5 id agreement | .898 | .308 | .814 | .377 |
| max \|Δ logprob\| at stored top-5 | .195 | 12.4 | .126 | 7.59 |

**What it says.** Storage index 42 is the final pre-norm residual; index 41 is decisively
not. The residual mismatch at L41 is bf16-pipeline noise: post-softcap logits live in
bf16, whose ulp is .06–.25 at magnitudes 8–30, exactly the observed ceiling; top-5
disagreements are near-ties inside one ulp (PopQA's flatter distributions have more of
them). The realized tolerance — median .003–.004 nats — is two orders below any Δ quoted
in Phase 3, and every Phase 3 delta subtracts a baseline recomputed through the same code
path, so the systematic part cancels. Gate PASSED at bf16 tolerance; "exact" below means
exact up to this floor.

### Gate 2 — SAE parity

Adapter codes vs manual JumpReLU (`W_enc`, `b_enc`, `threshold`) on 200 stored SLT states
(rerun of the §14(d) test):

| | TriviaQA L28/slt | PopQA L24/slt |
|---|---:|---:|
| max \|code difference\| | **0.0** | **0.0** |
| realized L0 (of 16,384) | 123.3 | 137.7 |
| reconstruction cosine | .941 | .943 |

PASSED — bit-exact; L0 in the canonical release's range.

### Gate 3 — quadrant occupancy (seed 42; held-out counts in brackets)

| quadrant | TriviaQA n | (test) | `greedy_in_majority` | PopQA n | (test) | `greedy_in_majority` |
|---|---:|---:|---:|---:|---:|---:|
| high-SE, correct | 666 | (504) | .757 | 124 | (99) | .573 |
| high-SE, wrong | 1,246 | (883) | .541 | 4,019 | (3,377) | .208 |
| low-SE, correct | 2,093 | (1,520) | 1.000 | 2,600 | (2,201) | .961 |
| low-SE, wrong | **28** | (21) | 1.000 | 664 | (551) | .920 |
| tail thresholds (lo/hi) | 0.0 / 1.359 | | | 1.352 / 2.303 | | |

**What it says.** Every cell is populated, so Experiments D and E can run — but two cells
are thin and every later number conditioned on them inherits the width: TriviaQA's
low-SE-incorrect cell (n=28; the model is rarely confidently wrong on TriviaQA) and
PopQA's high-SE-correct cell (n=124; when PopQA answers are right the model is rarely
uncertain). Seed 43 moves no count by more than a handful of rows (PopQA's low-SE cells
shift by ~90 because its tail threshold sits on a dense part of the SE distribution).
`greedy_in_majority` orders exactly as it must if the label and cluster structure are
coherent: ~1.0 in the low-SE cells, .21–.76 in the high-SE cells, and within high-SE it is
higher for correct rows than wrong ones on both datasets.

---

## Phase 1 — Experiment A: the directions and their geometry

All directions built on training rows, per seed; seed 42 shown, seed 43 in the note when it
moves a conclusion. Sites are abbreviated T-tbg = TriviaQA L40/tbg, T-slt = L28/slt,
T-han = L30/slt, P-tbg = PopQA L24/tbg, etc.; `fin-tbg`/`fin-slt` are the final pre-norm
residual sites.

### A1 — pairwise cosines, the named cells

| cosine | T-tbg | T-slt | T-han | T-fin-tbg | T-fin-slt | P-tbg | P-slt | P-han | P-fin-tbg | P-fin-slt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (`diffmean_se`, `diffmean_se_strat`) | .957 | .945 | .943 | .975 | .941 | .984 | .960 | .949 | .996 | .951 |
| (`diffmean_se`, `diffmean_corr`) | **−.971** | −.963 | −.963 | −.981 | −.977 | **−.986** | −.968 | −.957 | **−.997** | −.795 |
| (`diffmean_se_strat`, `diffmean_corr_strat`) | **−.620** | −.518 | −.512 | −.700 | −.645 | **−.808** | −.614 | −.507 | −.953 | −.161 |
| (`diffmean_se`, `diffmean_tokH`) | .949 | .965 | .963 | .956 | .976 | .878 | .762 | .691 | .895 | −.059 |
| (`diffmean_se`, `probe_lr`) | .552 | .484 | .442 | .395 | .332 | .751 | .628 | .482 | .254 | .187 |
| (`diffmean_se`, `sae_topk`) | .339 | .176 | .231 | — | — | .450 | .326 | .242 | — | — |
| (`diffmean_se`, `pca_contrast`) | .453 | .297 | .511 | .577 | .700 | .448 | .499 | .487 | .856 | .706 |
| max \|cos\| vs the 5 randoms (floor) | .037 | .038 | .033 | .028 | .028 | .021 | .025 | .021 | .022 | .009 |

**What it says.** Three facts carry the rest of the document. **(1) The unstratified SE and
correctness mean-difference directions are the same axis with the sign flipped** —
cos = −.96…−.997 at every site on both datasets. `diffmean_se` as usually built (Marks &
Tegmark / Rimsky recipe on the raw label) *is* an incorrectness direction. **(2)
Stratifying à la Patel et al. genuinely separates them, partially**: the stratified pair
drops to −.51…−.62 at the answer-side sites — far from orthogonal, but no longer the same
vector — while staying −.81/−.95 at PopQA's TBG/final sites, where SE and error are most
confounded in the data itself. **(3) `diffmean_se` ≈ `diffmean_tokH`** (cos .88–.98
everywhere except PopQA answer-side, .69–.76): the SE mean-difference is mostly the
token-entropy mean-difference. The learned directions (`probe_lr`, `sae_topk`) sit at
cos .18–.75 from the diffmean family — same half-space, different vector. Seed 43 agrees
everywhere within ±.03 except the stratified pair at T-tbg/T-slt (−.67/−.54), which is
train-draw-sensitive because the wrong-stratum is small.

### A2 — projection AUROC on held-out rows (both targets; direction as oriented)

Signed projections; a value near 0 means the direction reads the *other* end of that
label. seed 42 / seed 43.

| direction | target | T-tbg | T-slt | T-han | T-fin-tbg | P-tbg | P-slt | P-han | P-fin-tbg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `diffmean_se` | `se_discrete` | .853/.851 | .801/.797 | **.787/.785** | .765/.751 | .884/.879 | .828/.823 | **.778/.776** | .842/.838 |
| `diffmean_se_strat` | `se_discrete` | .847/.833 | .795/.784 | .789/.780 | .758/.746 | .893/.881 | .826/.823 | .772/.770 | .841/.836 |
| `diffmean_tokH` | `se_discrete` | .836/.828 | .797/.790 | .783/.776 | .719/.702 | .810/.808 | .753/.748 | .672/.677 | .872/.870 |
| `probe_lr` | `se_discrete` | **.892/.893** | **.879/.879** | **.871/.868** | **.881/.882** | **.954/.953** | **.958/.953** | **.947/.941** | **.951/.948** |
| `sae_topk` | `se_discrete` | .865/.825 | .775/.686 | .822/.712 | — | .935/.917 | .926/.917 | .837/.836 | — |
| `pca_contrast` | `se_discrete` | .647/.600 | .563/.542 | .617/.609 | .614/.606 | .640/.647 | .608/.604 | .580/.575 | .796/.794 |
| `diffmean_corr` | `judge_binary` | .830/.847 | .765/.780 | .750/.763 | .754/.772 | .846/.846 | .803/.801 | .761/.763 | .801/.800 |
| `diffmean_corr_strat` | `judge_binary` | .836/.842 | .759/.755 | .735/.732 | .783/.827 | .845/.853 | .813/.813 | .772/.782 | .799/.802 |
| random (range over 5, vs `se_discrete`) | | .35–.54 | .40–.64 | .36–.60 | .45–.64 | .29–.50 | .45–.51 | .42–.55 | .30–.52 |

**What it says.** The tuned dense probe direction is the best single-axis readout of its
own target at every site, by .04–.12 over any diffmean. `diffmean_se` stays above the .80
flag line at the SLT sites but **falls below it at both Han sites (L30/slt: .787 / .778)**
— every Han-site number later carries that flag. The final-TBG site reads .77–.84, and
final-SLT collapses (SE .69/.67, in the JSON) — consistent with §3's answer-side layer
profile. Two controls matter: `pca_contrast` (RepE's LAT vector, unsupervised) reads only
.54–.65 at the SAE sites, so the supervised contrast is doing real work; and **the random
"floor" is not .5** — random projections read .29–.64 against `se_discrete` because these
states are extremely anisotropic and the dominant variance directions themselves correlate
with the label. Any claimed direction has to beat *that* floor, not .5. `sae_topk` is the
seed-fragile row (T-slt .775 vs .686 across seeds): 64 features chosen on 1.5k fit rows at
an answer-side site is an unstable object. (`sae_top1`'s orientation follows the raw
decoder row; on PopQA the top-|coefficient| is negative, so its projection reads anti-SE —
flipped it is .84/.80 tbg/slt. It is a *feature*, not a calibrated direction.)

### A3 — the Han two-coordinate replication, at L30/slt

| | TriviaQA | PopQA |
|---|---:|---:|
| logistic probe on coords {1279, 2558} only, vs `se_discrete` | .7365 / .7402 | .6374 / .6415 |
| full dense probe at L30/slt (same protocol) | .8706 / .8676 | .9473 / .9406 |
| variance rank of coord 1279 / 2558 (of 3,584) | 7 / 28 | 18 / 35 |
| mean \|activation\| at 1279 / 2558 (vs all-coord mean) | 40.6 / 37.5 (vs 2.99) | 43.8 / 56.6 (vs 3.08) |
| share of `diffmean_se`'s squared norm on the two coords | .061 / .053 | .0055 / .0039 |

Han clamp values (Eq. 1, training rows, rms-normalized space, seed 42): TriviaQA
high→low magnitudes .52 / 3.56, low→high 68.5 / 79.1; PopQA in `phase1_popqa.json`.

**What it says.** Directionally replicated, quantitatively partial. Two coordinates out of
3,584 do carry a startling amount of SE signal (.74 on TriviaQA — more than `pca_contrast`
gets from the full space), but they are **.13–.31 below the dense probe at the same site**,
so they are not "the" SE representation. And the Sun et al. reading fits: both coordinates
are massive-activation dimensions — 13–18× the mean coordinate magnitude, variance ranks
in the top 1% — i.e., the kind of coordinate that encodes *lots* of global state, of which
SE is one correlate. The SE mean-difference direction puts 5–6% (TriviaQA) but only ~0.5%
(PopQA) of its energy there, against a 0.056% uniform share: concentrated, yes;
coordinate-aligned sparse, no.

### A4 — stability across depth and datasets

Depth rotation of `diffmean_se`, cos(layer l, layer l+4), §3 grid, seed 42 (seed 43 within
±.03):

| position | L4→8 | L8→12 | L12→16 | L16→20 | L20→24 | L24→28 | L28→32 | L32→36 | L36→40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TriviaQA tbg | .245 | .453 | .565 | .642 | .647 | .695 | .802 | .834 | **.599** |
| TriviaQA slt | .607 | .727 | .734 | .752 | .737 | .706 | .852 | .884 | **.510** |
| PopQA tbg | .203 | .387 | .545 | .614 | .649 | .734 | .813 | .889 | .698 |
| PopQA slt | .674 | .727 | .638 | .761 | .732 | .715 | .846 | .904 | .679 |

Shared cells, seed 42 (seed 43 in brackets):

| | cos(trivia, popqa) | trivia dir → PopQA test | popqa dir → TriviaQA test | in-domain (T / P) |
|---|---:|---:|---:|---:|
| L40 / tbg | .395 (.387) | .8395 (.8341) | .8047 (.8024) | .853 / .917 |
| L24 / slt | .613 (.572) | .8040 (.7892) | .8042 (.7942) | .836 / .828 |

(All AUROCs vs the target dataset's own `se_discrete` label.)

**What it says.** The direction **rotates continuously with depth** — adjacent-grid cosine
climbs from ~.2 in early layers to ~.9 by L32→36, then drops at L36→40 on TriviaQA (.51–.60),
so "the" SE direction is depth-local, and a steering vector moved even 4 layers is a
substantially different vector exactly where the probes work best. Across datasets the
directions share only cos .39–.61 — yet the *projection* transfers at .79–.84 AUROC against
an in-domain .83–.92, the Marks & Tegmark pattern: the readable subspace is shared even
when the vectors are not. Same story as `RESULTS.md` §10, at direction rather than probe
level.

### A5 — per-prompt projection spread at the SLT site (`diffmean_se`, held-out)

| | TriviaQA | PopQA |
|---|---:|---:|
| high-SE rows projecting below the low-SE median | .112 / .125 | .112 / .120 |
| low-SE rows projecting above the high-SE median | .122 / .118 | .073 / .063 |

Reported, per Tan et al., as the descriptive precursor of per-input steering variability:
~10% of rows sit on the wrong side of the other class's median before any intervention.
Histograms are banked in `phase1_*.json`.

### A6 — the dense directions, decomposed in the SAE basis

Decoder-cosine decomposition (working reading), top-20 span, and the encoder route
(Mayne et al.'s caveat), seed 42:

| direction | site | top-20 span energy | encoder recon cos | site | top-20 span | encoder cos |
|---|---|---:|---:|---|---:|---:|
| | **TriviaQA tbg** | | | **TriviaQA slt** | | |
| `diffmean_se` | | .402 | .308 | | .354 | .239 |
| `diffmean_se_strat` | | .441 | .162 | | .326 | .188 |
| `diffmean_corr_strat` | | .320 | .068 | | .313 | −.069 |
| `diffmean_tokH` | | .412 | .322 | | .332 | .201 |
| | **PopQA tbg** | | | **PopQA slt** | | |
| `diffmean_se` | | .499 | .174 | | .415 | .174 |
| `diffmean_se_strat` | | .517 | .042 | | .405 | .094 |
| `diffmean_corr_strat` | | .435 | .054 | | .350 | .076 |
| `diffmean_tokH` | | .516 | .372 | | .395 | .299 |

Feature-list overlaps (Jaccard), seed 42:

| overlap | T-tbg 20 | T-tbg 100 | T-slt 20 | T-slt 100 | P-tbg 20 | P-tbg 100 | P-slt 20 | P-slt 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SE vs correctness direction | .60 | .63 | .67 | .52 | .67 | .65 | .38 | .46 |
| SE-strat vs corr-strat | .54 | .55 | .38 | .28 | .60 | .56 | .33 | .29 |
| SE vs token-entropy direction | .67 | .68 | .67 | .75 | .48 | .53 | .14 | .31 |
| SE vs the probe's K features | .12 | .24 | .11 | .11 | .21 | .16 | .21 | .16 |

Top decoder-cosine features of `diffmean_se` (Neuronpedia auto-interp, top 5 of the
banked top-20): TriviaQA tbg — 3378 "scientific observations/explanations" (−.33), 6271
"medical conditions and associations" (−.32), 13512 "words for conditions and phenomena"
(+.31), 2646 "programming syntax" (+.27), 1113 "specific nouns + associated verbs" (−.25).
TriviaQA slt — 2496 "proper nouns, names and brands" (+.31), 13523 (−.27), 5180
"problem-solving/evaluating complex situations" (−.27), 3398 "sentence-ending punctuation
indicating uncertainty" (−.26), 13947 (−.26). PopQA tbg — 13123 "proper nouns of notable
individuals/events" (−.44), 6838 "dates, particularly historical" (−.32), 6787 (+.31),
6704 (+.30), 15280 (−.29). Full lists with labels: `results/steering/phase1_*.json` +
`neuronpedia_cache.json`.

**What it says.** (1) **No dense SE direction is sparse in the SAE basis**: the best 20 of
16,384 decoder rows span only 31–52% of its energy. (2) The **encoder route fails exactly
as Mayne et al. predict** (recon cosine −.07…+.37 on a vector far off the activation
distribution); the decoder-cosine reading is the usable decomposition, and everything
below uses it. (3) The SE and correctness directions **share 38–67% of their top feature
lists, with signs flipped feature-by-feature** — e.g. TriviaQA-tbg 3378 is −.33 in the SE
list and +.25 in the correctness list; PopQA-tbg 13123 is −.44 vs +.38 — which is A1's
anti-alignment made concrete. Stratification lowers the overlap (to .28–.60) but does not
produce disjoint feature vocabularies. (4) The features the *probe* selected overlap the
diffmean's decoder ranking at only .11–.24 — the probe reads a different, more
distributed slice of the code than the mean-difference geometry suggests, consistent with
`RESULTS.md` §11's "promiscuous features" finding. Auto-interp labels are
knowledge-domain-ish (people, dates, conditions) rather than "uncertainty" concepts, with
one exception worth naming: TriviaQA-slt 3398, "punctuation indicating uncertainty".

---

## Phase 2 — Experiment B: operations × directions, read through the SAE

All ops act on the rms-normalized state; readout = the JumpReLU SAE at the same layer,
plus the frozen probes. 300 high-SE + 300 low-SE held-out rows; α in units of
‖`diffmean_se`‖ at the site (TriviaQA tbg 129.8, slt 38.0; PopQA tbg 50.3, slt 48.9 —
seed 42). Columns: Δse = change in projection onto v̂_SE for high-SE rows (α·unit·cos(v,
v̂_SE) for `add`, so it doubles as the cosine check); NN = distance of h' to the nearest
real held-out activation / the typical real NN distance (off-distribution check); on/off =
features (of 16,384) switched on/off; conc = share of total mean-|Δactivation| on the
top-10 features; Δsp = frozen sparse-probe score change (hi rows / lo rows); ΔK = probe
features activated + silenced.

### TriviaQA L40/tbg (baseline L0 = 209)

| condition | ‖h'‖/‖h‖ | Δse (hi) | NN | on / off | conc | Δsp hi / lo | ΔK on+off |
|---|---:|---:|---:|---:|---:|---:|---:|
| `diffmean_se` add +1 | .99 | +130 | .26 | 24 / 28 | .145 | +3.0 / +2.7 | 3.6+4.3 |
| `diffmean_se` add −1 | 1.03 | −130 | .26 | 46 / 17 | .141 | −3.1 / −3.4 | 5.7+2.1 |
| `diffmean_se` add +2 | 1.00 | +260 | .53 | 60 / 45 | .134 | +7.4 / +5.6 | 8.8+7.1 |
| `diffmean_se` ablate | .99 | +54 (hi), +176 (lo) | .18/.36 | 20 / 29 | .164 | +0.9 / **+3.6** | 4.1+5.0 |
| `diffmean_se` ablate_add_hi | .99 | +7 (hi), +130 (lo) | **.16/.27** | 15 / 24 | .168 | −0.1 / +2.8 | 3.2+4.0 |
| `diffmean_se` ablate_add_lo | 1.00 | −123 (hi), 0 (lo) | .25/**.13** | 23 / 19 | .156 | −3.1 / +0.2 | 3.6+2.6 |
| `diffmean_corr_strat` add +1 | 1.02 | −104 | .26 | 49 / 18 | .107 | −2.5 / −2.8 | 4.9+1.7 |
| `probe_lr` add +1 | 1.01 | +72 | .26 | 26 / 33 | .073 | +2.2 / +2.3 | 1.3+2.8 |
| `sae_topk` add +1 | 1.00 | +44 | .26 | 7 / 19 | **.255** | **+30.9** / +26.5 | 2.0+3.2 |
| random add ±1 (2 vecs) | 1.01 | ±0.6 | .26 | ~38 / ~22 | **.029–.033** | ±0.1 | 0.6+0.8 |
| `clamp_sae_top1` + (to hi-max) | 1.00 | +18 | **.18** | 6 / 9 | **.310** | **+19.9** / +15.9 | 1.5+1.1 |
| `clamp_sae_top1` − (to 0) | 1.00 | −0.3 | .00 | 0 / 0 | .446 | −0.5 / 0.0 | 0+0 |

### PopQA L24/tbg (baseline L0 = 74)

| condition | ‖h'‖/‖h‖ | Δse (hi) | NN | on / off | conc | Δsp hi / lo | ΔK on+off |
|---|---:|---:|---:|---:|---:|---:|---:|
| `diffmean_se` add +1 | 1.04 | +50 | **1.29** | 27 / 10 | .217 | +3.2 / +4.2 | 3.9+2.5 |
| `diffmean_se` add −1 | 1.00 | −50 | **1.29** | 18 / 13 | .229 | −4.4 / −5.9 | 4.0+3.1 |
| `diffmean_se` ablate | .98 | −47 (hi), +6 (lo) | 1.23 | 7 / 13 | .262 | −4.3 / +0.7 | 2.9+3.5 |
| `diffmean_se` ablate_add_hi | 1.00 | −2 (hi), +52 (lo) | **.58** | 11 / 11 | .259 | −0.2 / +4.5 | 3.3+2.7 |
| `diffmean_se` ablate_add_lo | .98 | −52 (hi), +1 (lo) | 1.33 | 8 / 14 | .259 | −4.8 / +0.2 | 3.0+3.6 |
| random add +1 (2 vecs) | 1.02 | −1.1 | 1.29 | 14 / 8 | .112 | +0.2 | 0.6+0.8 |
| `sae_topk` add +1 | 1.02 | +23 | 1.29 | 12 / 11 | .253 | +4.5 / +7.0 | 2.8+2.8 |
| `clamp_sae_top1` + (to hi-max) | 1.00 | −2.4 | **.29** | 4 / 2 | .265 | −1.7 / −2.4 | 1.2+0.1 |

At the SLT sites the same table (in `phase2_*.json`) reads similarly with two shifts: the
NN column is milder (TriviaQA .29, PopQA .66 at α=±1 — PopQA's **tbg** is the
off-distribution site, 1.29), and concentration is lower (conc .12–.16 vs random .06–.10).

**B4, in one sentence:** `diffmean_se_strat` behaves identically to `diffmean_se` under
every op (add +1 at T-tbg: conc .216 vs .217, Δsp +3.2 vs +3.2, same NN) — the
stratification changes the *label geometry* (A1), not the mechanical footprint.

**Han coordinate clamp at its own site (L30/slt), both directions:**

| | ‖h'‖/‖h‖ | Δproj_se hi / lo | NN | on / off | Δsp hi / lo |
|---|---:|---:|---:|---:|---:|
| TriviaQA high→low | .99 | +4.4 / +6.7 | .35 | 24 / 5 | −1.1 / −0.7 |
| TriviaQA low→high | 1.03 | −3.1 / −1.6 | .34 | 5 / 13 | +1.0 / +1.3 |
| PopQA high→low | .98 | +0.2 / +0.6 | .58 | 21 / 4 | +0.0 / −0.4 |
| PopQA low→high | 1.04 | +1.4 / +1.7 | .58 | 5 / 12 | −0.2 / +0.1 |

**What it says.** (1) **Norm-matched random is the control that matters, and the real
directions clear it**: at the same α a random vector switches just as many features
on/off (~38 on at T-tbg) but its Δ is spread across the dictionary (conc .03 vs .15) and
moves *no* readout (Δsp ±0.1, ΔK ~1). The SE direction's footprint is 2–5× more
concentrated and lands on interpretable features — the top-|ΔF| list under add+1 at T-tbg
is 1448 "retail entities", 4843 "medical research", 9671 "retail brands", 9309 "legal
roles/titles", 6271 "medical conditions" — the same knowledge-domain vocabulary as A6, not
an "uncertainty" module. (2) **The conditional ops are the mechanically clean ones.**
`ablate_add_hi/lo` move only the rows that are not already at the target-class mean
(Δsp hi −0.1 vs lo +2.8 for induce-high), and they are the most on-distribution write ops
in the table (NN .13–.58 vs 1.23–1.33 for plain add/ablate at P-tbg). Plain **ablate is
not a "remove the concept" op here**: baseline projections are far from zero, so zeroing
them teleports *both* classes (at T-tbg it moves low-SE rows +176 — reading to the probes
as *more* high-SE). (3) **PopQA's TBG site rejects norm-scale additions**: α=±1 lands 1.29
typical-NN-distances from the nearest real activation — any generation-time result there
would be an off-distribution artifact, exactly the failure mode Tan et al. and Braun et
al. describe. (4) **The single-feature clamp is the surgical option on TriviaQA**:
`clamp_sae_top1`+ moves the sparse probe +19.9 (more than add α=+2) while touching ~6
features at NN .18. It does not transfer as a recipe: PopQA's top-|w| feature has a
negative coefficient and its clamp moves the probe the other way (−1.7); and per-feature
clamps over the full probe-K set average out to nearly nothing on both datasets
(mean Δsp +0.14 / −0.16 — one feature carries the trivia effect, max +40). (5) **The Han
coordinate clamp does approximately nothing at the state level**: Δ projections ≤ 4.4
(class separation ~38–59), sparse-probe Δ ≤ 1.3, and on PopQA the high→low and low→high
clamps move the (near-zero) readouts in inconsistent directions. Whatever Han et al.'s
Eq. 2 forward-pass result reflects, it is not visible as a state-geometry change at the
site the clamp is defined on. (Flag from A2 carries: the Han site's own `diffmean_se`
reads only .78.)

---

## Phase 3 — Experiment C: exact next-token readout at the final layer

Final pre-norm residual, TBG position, 2,000 held-out rows per dataset per seed;
directions rebuilt at this site on raw states; α in units of ‖`diffmean_se`‖ here
(TriviaQA 182.4, PopQA 440.1). Readout is exact per gate 1 (bf16 floor, and every Δ
subtracts a baseline computed by the same code path). **Single-position, final-layer
result** — nothing here reads layers 24–40, where the probes live.

Baselines (all rows): TriviaQA H = 1.66 nats, p(greedy tok) = e^−0.63, mass Y_same/other/
rest = .67/.11/.22; PopQA H = 3.70, .28/.16/.56. Multi-cluster rows: 70% / 92%.

### C1 — dose-response of `diffmean_se` (group = all; seed 42, seed 43 within .01–.11)

| condition | TQ ΔH | Δlog p(greedy) | ΔY_same | ΔY_other | ΔY_rest | PQ ΔH | Δlog p(greedy) | ΔY_same | ΔY_other | ΔY_rest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| add −1 | −.28 | +.09 | +.037 | +.004 | −.041 | −.20 | +.19 | +.022 | +.013 | −.035 |
| add −0.5 | −.18 | +.07 | +.026 | +.002 | −.028 | −.42* | +.34* | +.037* | +.025* | −.061* |
| add +0.5 | +.26 | −.11 | −.041 | −.005 | +.045 | +.91 | −.72 | −.064 | −.047 | +.112 |
| add +1 | +.63 | −.27 | −.096 | −.012 | +.108 | +2.04 | −1.61 | −.131 | −.089 | +.220 |
| add +2 | +1.69 | −.75 | −.247 | −.034 | +.281 | +3.63 | −3.06 | −.209 | −.127 | +.336 |
| ablate | +3.42 | −1.55 | −.480 | −.052 | +.531 | +3.34 | −2.78 | −.247 | −.117 | +.364 |
| ablate_add_hi | +.30 | −.11 | −.069 | +.006 | +.063 | +.09 | +.00 | −.070 | +.019 | +.050 |
| ablate_add_lo | −.19 | +.01 | +.009 | +.006 | −.015 | −.80 | +.02 | +.050 | −.000 | −.050 |
| T = 1.5 | +2.70 | −.73 | −.275 | −.040 | +.316 | +2.24 | −.99 | −.148 | −.073 | +.221 |
| T = 2.0 | +6.56 | −1.98 | −.549 | −.083 | +.632 | +4.48 | −2.07 | −.244 | −.118 | +.363 |

*PopQA's negative branch is non-monotone: −0.5 reduces entropy more than −1.0 (the −1.0
row is already overshooting on low-SE rows, ΔH +.83 there). TriviaQA is monotone to −1
and overshoots at −2.

By starting group at add +1 (seed 42): TriviaQA hiSE ΔH +.79 / loSE +.54; PopQA hiSE
+2.00 / loSE +2.17. Single-cluster rows move like multi-cluster rows (TQ +.42 vs +.72; PQ
+1.33 vs +2.10) — the op does not know whether alternatives exist. SE-reduction on the
cells that matter (add −1 / the conditional op, `hiSE_wrong` rows): TriviaQA ΔH −.52 /
−.77 with Δlog p(greedy) +.22 / +.02 and **ΔY_other +.030 in both**; PopQA −.82 / −1.15
with +.47 / −.31 and ΔY_other +.052 / −.017. The conditional `ablate_add_lo` leaves
low-SE rows untouched by construction (TQ ΔH +.01, PQ −.12 there) — the only op in the
table with that property.

### C2 — Han's relevance test: direction vs temperature at matched ΔH (multi-cluster rows)

`gain_other` / `gain_rest` = total mass on Y_other / Y_rest as a multiple of baseline;
"temp @ΔH" interpolates the fine temperature curve (T ∈ 1.05…3.0) at the direction's own
ΔH. R = gain_other / gain_rest. Seed 42 (seed 43 agrees, shown in `phase3_*` json):

| condition | dataset | ΔH | gain_other | temp @ΔH | R(dir) | R(temp) |
|---|---|---:|---:|---:|---:|---:|
| `diffmean_se` +0.5 | TriviaQA | +0.31 | .960 | .963 | .816 | .842 |
| `diffmean_se` +1 | TriviaQA | +0.72 | .897 | .911 | .637 | .688 |
| `diffmean_se` +2 | TriviaQA | +1.82 | .695 | .761 | .350 | .434 |
| `diffmean_se_strat` +1 | TriviaQA | +0.61 | .914 | .925 | .675 | .724 |
| `diffmean_se` +0.5 | PopQA | +0.95 | .703 | .774 | .590 | .663 |
| `diffmean_se` +1 | PopQA | +2.10 | .437 | .546 | .321 | .411 |
| `diffmean_se` +2 | PopQA | +3.64 | .199 | .321 | .131 | .219 |
| `diffmean_se_strat` +1 | PopQA | +2.16 | .428 | .536 | .313 | .402 |

**What it says (C2 is the verdict of this phase).** In 16 of 16 cells (2 datasets × 2
seeds × 4 conditions) the SE direction moves **less** mass to the other semantic clusters
than temperature does at the same entropy increase, and correspondingly more to Y_rest.
At this site, +α on an SE direction is **not** a meaning-aware spread knob — it is a
temperature knob with a small *anti*-relevance bias. Han et al.'s Eq. 2 effect, whatever
it is, is not reproduced by additive steering read exactly at the final layer; their
intervention ran at layer 30 under a forward pass, which is precisely what stored data
cannot reach (see "not established here").

### C3 — direction specificity: there is none at this site

At matched α = ±1, `diffmean_corr_strat` (flipped) and `diffmean_tokH` produce the same
signature as `diffmean_se` — TriviaQA ΔH +.60 / +.66 vs +.63; PopQA +1.85 / +1.82 vs
+2.04; relevance R .64 / .64 vs .64 (TQ), .35 / .33 vs .32 (PQ). The prediction that the
correctness direction would move p(greedy) *without* the entropy/Y_other signature
**fails** — as it had to, given A1's cos(se, corr) = −.98 and cos(se, tokH) = .96 at this
site. The controls behave: random directions at the same α produce ΔH ≈ 0 on TriviaQA
(+.01) and modest, unconcentrated spread on PopQA (+.53, from its larger α unit of 440);
ablate of a random direction is a no-op (ΔH +.006).

### C4 — the quadrant readout

Baseline states differ exactly as the labels say they should (seed 42): TriviaQA
hiSE-wrong H = 3.74, p(greedy) = e^−1.65, Y_same .21 / Y_other .25; loSE-correct H = .56,
Y_same .93 / Y_other .000. PopQA hiSE-wrong H = 5.23, Y_same .016 / Y_other .17;
loSE-correct H = 1.11, Y_same .79. Under add +1 the four quadrants move together
(TriviaQA ΔH +.79 in both hiSE cells, +.4–.6 in loSE cells); the only quadrant-selective
op is `ablate_add_lo` (above). PopQA's hiSE-correct cell is n = 24 in the eval subsample —
its rows are indicative only.

---

## Phase 4 — Experiment D: reading the dissociation

Strata use the binarized labels (`se_hi` by `sep_best_split` on the training draw,
`judge_binary` as stored); all AUROCs on held-out rows. Held-out stratum sizes at
seed 42: TriviaQA j1/j0 = 4,122/1,212, hiSE/loSE = 1,845/3,489; PopQA j1/j0 =
3,393/7,606, hiSE/loSE = 8,248/2,751.

### D1 — within-stratum projection AUROC (seed 42; seed 43 within ±.02 unless noted)

Values are the signed projection's AUROC; a cell near 0 reads the label's other end.
The last row is the control that sets the floor: **the `se_discrete` scalar itself**
still reads `judge_binary` inside the binarized strata, because binarization does not
remove the SE–correctness correlation.

**TriviaQA** (tbg | slt):

| direction | se all | se \| j=1 | se \| j=0 | judge all | judge \| hiSE | judge \| loSE |
|---|---:|---:|---:|---:|---:|---:|
| `diffmean_se` | .853 \| .801 | .810 \| .779 | .783 \| .723 | .178 \| .244 | .323 \| .400 | .298 \| .338 |
| `diffmean_se_strat` | .847 \| .795 | .806 \| .773 | .776 \| .741 | .185 \| .258 | .332 \| .409 | .304 \| .369 |
| `diffmean_corr` | .144 \| .201 | .192 \| .229 | .208 \| .292 | .830 \| .765 | .693 \| .614 | .704 \| .680 |
| `diffmean_corr_strat` | .149 \| .231 | .207 \| .270 | .204 \| .344 | .836 \| .759 | .712 \| .627 | .702 \| .695 |
| `probe_lr` | .892 \| .879 | .851 \| .837 | .794 \| .791 | .130 \| .144 | .267 \| .285 | .231 \| .235 |
| `sae_topk` | .865 \| .775 | .827 \| .714 | .763 \| .751 | .162 \| .240 | .305 \| .343 | .269 \| .402 |
| random0 | .538 \| .518 | .531 \| .494 | .517 \| .540 | .461 \| .467 | .484 \| .456 | .469 \| .501 |
| **`se_discrete` scalar (floor)** | 1.0 \| — | — | — | **.106** | **.265** | **.237** |

**PopQA** (tbg | slt):

| direction | se all | se \| j=1 | se \| j=0 | judge all | judge \| hiSE | judge \| loSE |
|---|---:|---:|---:|---:|---:|---:|
| `diffmean_se` | .884 \| .828 | .784 \| .714 | .842 \| .798 | .148 \| .202 | .215 \| .270 | .328 \| .374 |
| `diffmean_se_strat` | .893 \| .826 | .799 \| .723 | .855 \| .801 | .145 \| .215 | .215 \| .293 | .335 \| .390 |
| `diffmean_corr` | .129 \| .178 | .238 \| .303 | .173 \| .211 | .846 \| .803 | .781 \| .746 | .674 \| .634 |
| `diffmean_corr_strat` | .136 \| .178 | .250 \| .304 | .187 \| .233 | .845 \| .813 | .781 \| .756 | .698 \| .677 |
| `probe_lr` | .954 \| .958 | .899 \| .904 | .930 \| .934 | .096 \| .094 | .160 \| .157 | .269 \| .281 |
| `sae_topk` | .934 \| .926 | .859 \| .829 | .905 \| .903 | .108 \| .112 | .171 \| .171 | .296 \| .323 |
| random0 | .294 \| .487 | .386 \| .489 | .338 \| .451 | .698 \| .495 | .649 \| .489 | .602 \| .460 |
| **`se_discrete` scalar (floor)** | 1.0 \| — | — | — | **.109** | **.211** | **.316** |

**What it says.** Read against .5, every SE direction looks confounded: all of them still
read `judge_binary` inside both SE strata at .60–.79 (flipped). Read against the **label
floor**, the picture inverts: the `se_discrete` scalar itself reads judge-within-hiSE at
.735 / .789 (flipped), and `diffmean_se` (.677 / .785), `probe_lr` (.733 / .840) and
`sae_topk` (.695 / .829) sit **at or below** the floor on TriviaQA and within .05 of it on
PopQA. So the within-stratum correctness signal these directions carry is roughly what the
SE label itself carries — there is no evidence any of them reads correctness *beyond* SE.
The mirror statement holds for the correctness directions (judge-within-strata .61–.78,
around their own analogous floor). Two soberer notes: stratified construction barely
changes the projections (`_strat` rows ≈ unstratified rows, ±.01 — A1 said the vectors
moved, D1 says the *readouts* barely did); and on-target within-stratum AUROC does drop as
expected (e.g. `diffmean_se` .853 → .810/.783 on TriviaQA), the part of the prediction
that did come true. The clean "high–high–.5–.5" pure pattern exists nowhere; the honest
summary is **dissociation bounded by the labels' own residual correlation, not
demonstrated beyond it** at the direction level.

### D2 — feature-level partition (all 16,384 features, training rows, BH-FDR .05)

Per-contrast significant counts, and the Patel-style classes (significant on SE in *both*
correctness strata / on correctness in *both* SE strata):

| | T-tbg s42 | T-tbg s43 | T-slt s42 | T-slt s43 | P-tbg s42 | P-tbg s43 | P-slt s42 | P-slt s43 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sig. SE within j=1 | 255 | 349 | 115 | 181 | 120 | 110 | 127 | 148 |
| sig. SE within j=0 | 34 | 23 | 3 | 7 | 314 | 337 | 452 | 461 |
| sig. judge within hiSE | 40 | 26 | 9 | 0 | 215 | 227 | 389 | 427 |
| sig. judge within loSE | 199 | 231 | 224 | 188 | 6 | 19 | 26 | 17 |
| **pure-SE** | 4 | 8 | 1 | 6 | **33** | 28 | **25** | 32 |
| **pure-correctness** | 1 | 0 | 0 | 0 | 0 | 2 | 5 | 4 |
| **confounded** | 2 | 3 | 0 | 0 | 1 | 8 | 2 | 3 |
| neither | 16,377 | 16,373 | 16,383 | 16,378 | 16,350 | 16,346 | 16,352 | 16,345 |

Training-stratum sizes explain the shape (seed 42): TriviaQA hi∧cor/lo∧cor/hi∧wrong/lo∧wrong
= 290/1,266/415/**77**; PopQA = 217/397/1,322/**112**. Each dataset has one starved
stratum, and it caps whichever "both strata" requirement runs through it.

Top pure-SE features (mean |d| across strata; Neuronpedia auto-interp): PopQA-tbg —
15280 "banking/financial terms" (d=.98), 3389 "locations and administrative divisions"
(.81), 10072 "statistical performance in sports" (.75), 15794 "legal text, court cases"
(.73), 5190 "medical research / genetic analysis" (.72). PopQA-slt — 12096 "U.S. states
and abbreviations" (.65), 13381 "geographical locations" (.62), 3389 (.61), 6636
"country names" (.55). TriviaQA-tbg (all 4): 14434 "cookie consent / website UI" (.60),
2411 "mathematical/statistical expressions" (.59), 8082 "programming methods" (.53),
15575 "scientific terminology" (.43). Pure-correctness is essentially empty everywhere
(0–5 features).

**What it says.** With FDR discipline and the both-strata requirement, the partition is
**power-limited on TriviaQA** (4–8 pure-SE of 16,384) and merely **thin on PopQA** (25–33
pure-SE; ~0 pure-correctness on both). What survives on PopQA is knowledge-domain
features — geography, dates-adjacent, finance — not "uncertainty" features: pure-SE
features look like *topic* indicators for the domains where the model happens to be
uncertain, which is what Patel et al.'s partition finds when uncertainty is
domain-confounded. The near-empty pure-correctness class on both datasets is itself the
headline: **at feature granularity there is no correctness vocabulary separate from the SE
vocabulary in these SAEs** at these sites, under this test's power.

### D3 — cross-prediction probes on the classes (pinned C=1.0; seed 42 / 43)

| feature set (n s42/s43) | vs `se_discrete` | vs `judge_binary` | dataset/site |
|---|---:|---:|---|
| pure-SE (4/8) | .732 / .764 | .713 / .744 | TriviaQA tbg |
| confounded (2/3) | .797 / .800 | .809 / .808 | TriviaQA tbg |
| pure-SE (33/28) | .873 / .873 | .806 / .792 | PopQA tbg |
| pure-SE (25/32) | .816 / .844 | .739 / .780 | PopQA slt |
| pure-correctness (5/4) | .549 / .590 | .552 / .580 | PopQA slt |
| confounded (1/8) | .880 / .929 | .800 / .878 | PopQA tbg |

**What it says.** The "pure-SE" sets still predict `judge_binary` at .71–.81 — but that is
again ≈ the D1 label floor (the SE scalar itself reads judge at .74–.79 flipped), so the
cross-prediction is what the label correlation forces, not evidence the features encode
correctness per se. Nothing here isolates a correctness-only readout (the 4–5
pure-correctness features predict either target at .55–.59, barely above chance in either
direction). Feature-level and direction-level agree: **the separable object in this data
is SE-beyond-correctness; correctness-beyond-SE is not recoverable** from these sites.

### D4 — feature classes vs the direction decompositions

Overlap between the pure-SE class and `diffmean_se_strat`'s top-20 decoder-cosine list:
TriviaQA 0/4 and 0/8 (tbg, two seeds), 0/1 and 0/6 (slt); PopQA 4/33 and 2/28 (tbg), 5/25
and 5/32 (slt). Pure-correctness vs `diffmean_corr_strat` top-20: 0 everywhere.
Confounded vs unstratified `diffmean_se` top-20: 1–4 of 1–8. Where the sparse probe's K
features land in the partition: TriviaQA 0–7 pure-SE, 0 pure-corr, 0–3 confounded, rest
"neither"; PopQA 6–11 pure-SE of K=32, 0 pure-corr, 1–7 confounded.

**What it says.** The statistically-pure features and the geometrically-top features are
**mostly different features** (overlaps 0–5 of 20), and the probe's working set is drawn
overwhelmingly from the "neither" class. Consistent with A6: mean-difference geometry,
statistical purity, and probe utility each pick a different slice of the SAE dictionary.

---

## Phase 5 — Experiment E: the offline part of Study B (resampling fix-rate)

Sample correctness channel: `sample_f1 ≥ 50` per sample (the per-sample judge channel was
never banked for these runs — stated limitation, no judge cross-check possible). Held-out
rows, tail-quartile quadrants, seed 42 (seed 43 within ±.01 everywhere):

### E1 — per-quadrant sample-correct rates

| quadrant | TriviaQA n | mean rate | ≥1 correct | PopQA n | mean rate | ≥1 correct |
|---|---:|---:|---:|---:|---:|---:|
| high-SE, greedy wrong | 883 | .129 | **.508** | 3,377 | .021 | **.123** |
| low-SE, greedy wrong | 21 | .557 | .619 | 551 | .126 | .310 |
| high-SE, greedy correct | 504 | .422 | .966 | 99 | .153 | .778 |
| low-SE, greedy correct | 1,520 | .977 | .999 | 2,201 | .869 | .997 |

### E2 — the ceiling on any uncertainty-axis intervention

Among **high-SE incorrect** prompts, the fraction with **zero** correct samples in 10:
**TriviaQA .492 / .502, PopQA .877 / .881** (seeds 42/43).

**What it says.** Farquhar et al.'s direction is confirmed — resampling helps where SE is
high and the greedy answer is wrong (a correct sample exists for 51% of such TriviaQA
prompts) — but the ceiling is the operative number: **half of TriviaQA's and 88% of
PopQA's high-SE errors have no correct answer anywhere in 10 samples.** No intervention
that only moves the model along an uncertainty axis (resample, steer-then-resample,
temperature) can fix those; the model does not know the answer. On PopQA the maximum
achievable gain from perfect uncertainty-gated resampling is ~12% of its errors — CLAP's
observation that binary-gated resampling can *hurt* is compatible with this: the low-SE
error cell is small (21 / 551 held-out rows) and its samples are correct only .56/.13 of
the time, so a gate that resamples everything uncertain trades a large no-win pool against
a small win pool. One channel caveat: TriviaQA's low-SE-wrong cell has *string-correct*
samples on 56% of rows — with `judge_binary` calling the greedy wrong while `f1≥50` calls
samples right, some of that cell is judge/string disagreement, and n=21 anyway.

---

## The shortlist for a generation run — ranked on B1–B3 concentration, C2 relevance, and D1 purity only

1. **`ablate_add_lo` on `diffmean_se_strat`, TBG site (TriviaQA L40 / PopQA L24).** The
   only conditional op: moves high-SE rows toward the low-SE mean and provably leaves
   low-SE rows alone (C1: ΔH +.01/−.12 on loSE; B3: Δsp ≈ 0 on the already-low class);
   among write ops it is the most on-distribution at the site where adds are safe
   (B1 NN .13–.25 TriviaQA) — with the PopQA-TBG off-distribution caveat (NN 1.33) arguing
   for running its PopQA arm at SLT instead. `_strat` over plain: same footprint (B4),
   less correctness-aligned by construction (D1/A1).
2. **`clamp_sae_top1` (+ to high-SE max), TriviaQA L40/tbg.** Most concentrated
   condition in Phase 2 (conc .31, ~6 features, NN .18) with the largest sparse-probe
   response (+19.9). TriviaQA-specific: PopQA's top-|w| feature inverts the sign. The
   right generation-time test of "a feature, not a direction".
3. **`add` ±0.5…1 on `diffmean_se_strat`, SLT site (T-L28 / P-L24).** Concentration
   beats random 2× and probes respond (B2/B3); C-level relevance is *untestable offline*
   at SLT (final-layer only), and C2's final-layer result predicts it will behave like
   temperature unless the answer-side site differs — that is the hypothesis a generation
   run should target, since SLT is where SE dissociates from token entropy
   (`RESULTS.md` §9/§13).
4. **`add` −0.5 (not −1) on `diffmean_se`, PopQA, SLT site.** The SE-reduction branch is
   non-monotone on PopQA (C1); −0.5 is the empirical knee. Gated on quadrant: E2 caps any
   accuracy claim at 12% of PopQA errors.

Ruled out by the same criteria: plain `ablate` anywhere (C1: +3.3–3.4 nats, B1
off-manifold, reads to probes as *more* high-SE); any α ≥ 1 addition at PopQA TBG
(B1 NN 1.29); `clamp_coord` at L30 (B: no state-level effect, A2: site flagged at .78);
`sae_topk`-as-addition for relevance claims (it moves its own probe 10× more than the
projection — a readout echo, not evidence of steering); `diffmean_corr_strat` as a
"correctness" knob (C3: indistinguishable from the SE direction).

## Not established here

Nothing in this document measures the effect of any intervention on **re-sampled semantic
entropy under generation**: every number is a property of stored states, of directions
built from them, and of the one output distribution that is an exact function of the
stored final-layer state. In particular, C2's negative verdict is a final-layer statement
and does not test steering at L24–L40 under a forward pass, which is where Han et al.
operated and where the shortlist above would have to be run.

## Config appendix

Realized values not already in the protocol table: seed-42 `sep_best_split` thresholds
.9536 (TQ) / 1.3257 (PQ), seed 43 .9536 / 1.3257; rms scales (seed 42) TQ tbg 15.53 / slt
5.29 / han 5.77, PQ tbg 3.84 / slt 5.01 / han 6.02 (per-seed values in the dirs npz);
`probe_lr` tuned C = 1e-5–1e-2 by site (recorded per site in `phase1_*.json`); PCA pairs
4,096; random vectors 5 per site, crc32-seeded; Phase 2 rows 300+300, Phase 3 rows 2,000,
fine temperatures {1.05, 1.1, 1.15, 1.2, 1.35, 1.5, 2.0}; B/C/D/E all reported at seeds
{42, 43}. Code: `analysis/steering/`; raw numbers: `results/steering/*.json`.
