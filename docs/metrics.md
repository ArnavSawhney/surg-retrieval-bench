# Retrieval metrics: what this benchmark reports, and why

Decided 25 Sep 2026. Implemented in `src/srb/metrics.py`, tested in
`tests/test_metrics.py`, demonstrated by `docs/metric_rationale.py`.

## The metric set

| Role | Metric | Definition |
|---|---|---|
| **Primary** | **AP per query → mAP** (macro over queries) | Full-ranking average precision: the mean of precision@*i* taken at every rank *i* that holds a relevant frame. |
| **Primary baseline** | **Random-ranker AP ≈ prevalence `R/N`** | Reported next to every AP, together with the lift `AP / prevalence`. |
| **Secondary** | **R-Precision** | Precision in the top `R`, where `R` is the number of relevant frames for that query. |
| **Descriptive only** | **P@10, P@50** | "What a user sees first." Reported, never used to rank models. |
| **Sanity check only** | **Hit@k** | Kept in the code and tests for debugging a ranking that looks broken. Not in the main tables. |
| **Removed** | Recall@k, nDCG@10 | See below. |

Zero-shot classification, probe and temporal metrics are unaffected: accuracy and
macro-F1 for phases, mAP for tools (multi-label), and Jaccard / edit score for
temporal phase segmentation.

## Why Recall@k and nDCG@10 were removed

Relevance here is **dense**. A frame counts as relevant if its ground-truth phase or
tool label matches the query, so a single query like *"clipping the cystic duct"* has
thousands of relevant frames. Using EndoNet's mean phase durations across the 40 test
videos at 1 fps, the test split is roughly 98,500 frames and
CalotTriangleDissection alone accounts for about 39% of them.

Under those conditions:

1. **Recall@k measures `R`, not the model.** Recall@k ≤ `k/R`. With
   `R ≈ 38,000`, even a *perfect* ranker scores Recall@10 ≤ 0.00026. Differences
   between models are swamped by differences in class prevalence. Recall@k is the
   right metric when there is one correct item to find — for example SurgVLP's paired
   video–text retrieval test — and the wrong one here.

2. **Hit@k cannot discriminate.** For a *uniformly random* ranker,
   Hit@10 = 1 − (1 − p)^10. At p = 0.387 that is **0.993**. A random baseline that
   scores 0.99 leaves no room to measure anything.

3. **P@10 and nDCG@10 saturate.** In the simulation, a weak model, a strong model and
   a degenerate "one-clip shortcut" ranker — one that surfaces 50 frames from a single
   clip and is random everywhere else — *all* score P@10 = 1.00 and nDCG@10 = 1.00.
   AP separates them: the shortcut ranker's AP sits at the random baseline
   (0.39 against a prevalence of 0.387), while a model with real signal reaches 0.67–0.89.

4. **nDCG is for graded relevance.** Labels here are binary (the phase matches or it
   does not), so nDCG adds nothing over P@k while being harder to interpret.

Run `python docs/metric_rationale.py` to reproduce the table. Those numbers are a
**simulation** with synthetic scores, used to choose the metrics — they are not model
results.

## Why AP and R-Precision instead

* **AP uses the whole ranking.** Every relevant frame contributes a precision term, so
  there is no cutoff to game and no saturation: getting one clip right cannot buy a
  good AP.
* **AP has an interpretable floor.** The expected AP of a query-blind random ranker is
  the prevalence `R/N`, so `lift = AP / prevalence` says how much the model actually
  contributes. An AP at or below prevalence means no signal for that query — and
  `random_baseline_ap()` exists so that floor is printed next to every result rather
  than left to the reader.
  (Strictly, prevalence is the large-`N` limit; the exact expectation is marginally
  higher — the simulation shows 0.3880 against a prevalence of 0.3873 at N ≈ 98k. It
  is a reference line, not a significance test.)
* **R-Precision is self-normalising.** Comparing the top `R` to the `R` that exist
  makes a long phase and a short one directly comparable: a perfect ranker scores 1.0
  in both cases.
* **mAP is macro-averaged over queries**, so CalotTriangleDissection (39% of frames)
  does not dominate GallbladderRetraction (3%).

## Conventions

### Ties: every metric is a tie-neutral expectation

*(Changed 26 Sep 2026; closes open decision #5 in `LOG.md`.)*

Frames with exactly equal scores form a **tied group**. Every metric returns its
**expected value under a uniformly random ordering within each tied group**
(McSherry & Najork, ECIR 2008). Groups are processed in descending score order. For a
group of `g` frames, `r` of them relevant, preceded by `n_b` frames of which `r_b` are
relevant:

* **AP.** The group contributes
  `Σ_{j=1..g} (r/g) · (r_b + 1 + (j−1)(r−1)/(g−1)) / (n_b + j)`, with
  `(r−1)/(g−1) := 0` when `g = 1`; the total is divided by `R`. This is exact by
  linearity of expectation: `r/g` is the probability that slot `j` holds a relevant
  frame, and given that, the expected number of relevant frames in slots `1..j` is
  `r_b + 1 + (j−1)(r−1)/(g−1)`.
* **P@k and R-Precision.** A group straddling the cutoff with `t` of its `g` slots
  inside the top `k` contributes `t · r / g` relevant frames.
* **Hit@k.** If no relevant frame ranks above the straddling group,
  `P(hit) = 1 − C(g−r, t) / C(g, t)`.

Example: four equal scores with relevance `[1,1,0,0]` give AP = **49/72** ≈ 0.681, and
so does `[0,0,1,1]`. Under the Session 1 convention (stable sort, ties broken by
ascending frame index) the same inputs gave 1.0 and 5/12, so frame index — which tracks
video order — leaked into the score. The tests check every metric against brute-force
enumeration of all within-group orderings in exact rational arithmetic, check that
shuffling frame indices changes nothing, and check that tie-free scores give the plain
sorted definition (and sklearn's AP, to 1e-12).

**Why ties are not rare here.** Embeddings are stored in fp16. In a simulation
(98,520 random unit vectors, 1152-d, stored in fp16, one random unit query), computing
the similarities *in fp16* left 95,257 frames (96.7%) in 11,331 tied groups; computing
them in float32 after upcasting left 156 tied frames (0.16%). Real data adds true
duplicates (black and out-of-body frames). So:

* every metric **raises on a float16 score array**; similarities are computed in
  float32 or higher after upcasting the stored embeddings;
* metrics cast scores to float64 before grouping;
* `count_ties(scores) -> (n_tied_frames, n_groups)` is reported per query in every
  retrieval output.

**Why not sklearn's convention.** `sklearn.metrics.average_precision_score` is also
order-invariant, but it interpolates at score thresholds: every relevant frame in a
tied group gets the precision at the *end* of the group. For `[1,1,0,0]` all tied that
is 0.5, not the expectation 49/72. The divergence is asserted in the tests.

`rank_order` still returns a deterministic stable-sort ranking for *display*; no metric
depends on its tie-break.

A document-name tie rule has been shown to bias TREC results (Cabanac, Hubert,
Boughanem and Chrisment, "Tie-breaking bias", CLEF 2010), which is why the index tie-break was dropped rather than kept for
reproducibility: the expectation is just as reproducible.

### Undefined values

A query with **zero** relevant frames has no defined AP or R-Precision.
`average_precision`, `r_precision` and `random_baseline_ap` return `nan` and emit a
`RuntimeWarning`. They never return 0.0, which would read as a real, terrible score in
a results table. `mean_ap` drops `nan` queries and warns, naming how many it dropped.

**Negative and control queries** ("a car on a road") have `R = 0` by construction.
They are **excluded from mAP** and evaluated separately as a score-calibration check.

### Other edge cases

`k > N` clamps to `N` with a warning (the denominator is then `N`, so the value is not
comparable to a P@k measured on a larger pool). `k < 1` and non-integer `k` raise.
`R = N` (all relevant) gives AP = 1.0 and R-Precision = 1.0. Non-finite scores raise
rather than being ranked — a `nan` similarity means the encoder is broken, and ranking
it would hide that. Graded (non 0/1) relevance raises, because relevance here is binary.

## Sources

* Manning, Raghavan and Schütze, *Introduction to Information Retrieval*, Ch. 8.4
  (AP, R-Precision, and the limits of cutoff metrics).
* Musgrave, Belongie and Lim, *A Metric Learning Reality Check*, ECCV 2020
  (cutoff metrics and unfair baselines in retrieval evaluation).
* Funke, Rivoir and Speidel, *Metrics Matter in Surgical Phase Recognition*, 2023
  (metric choice in this specific domain).
* Twinanda et al., *EndoNet*, IEEE TMI 2017 (Cholec80; the mean phase durations used
  for the prevalence figures above).
* McSherry and Najork, *Computing information retrieval performance measures
  efficiently in the presence of tied scores*, ECIR 2008, pp. 414–421 (the
  tie-neutral formulas).
* Cabanac, Hubert, Boughanem and Chrisment, *Tie-breaking bias: effect of an
  uncontrolled parameter on information retrieval evaluation*, CLEF 2010, LNCS 6360,
  pp. 112–123 (tie-breaking by document name biases TREC scores).
