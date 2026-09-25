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

### Ties

Frames are ranked by `np.argsort(-scores, kind="stable")`: **descending score, ties
broken by ascending frame index.** This is deterministic — the same scores always give
the same ranking, so a run is bit-reproducible.

⚠️ **It is deterministic but not tie-neutral.** With four equal scores and relevance
`[1,1,0,0]` the AP is 1.0; with the same scores and relevance `[0,0,1,1]` it is 5/12.
If a model ever produced large blocks of exactly-equal similarities, frame index
(which tracks video order) would leak into the score. `sklearn`'s
`average_precision_score` instead groups tied scores at one threshold, which averages
over the tie — this is why `average_precision` only agrees with sklearn on tie-free
scores, which is asserted in the tests, along with the divergence itself. Whether to
switch to the tie-neutral convention is an **open decision** in `LOG.md`; cosine
similarities between float32 embeddings tie only rarely, and the check for that
belongs in the evaluation script.

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
