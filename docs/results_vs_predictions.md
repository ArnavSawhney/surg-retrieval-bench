# Results against the registered predictions

Each registered prediction (`docs/preregistration.md`, section 5, tag `prereg-v1` =
`ff2b6e5`) next to the test-set number that decides it. Numbers come from the single
registered test run on Cholec80 videos 41–80 (`results/test/`, run at `1759cb4` on
13 October 2026, seed 20261011, B = 10,000 for the confirmatory family, the first 2,000
resamples for descriptive intervals). Every number below was recomputed from the
per-query and per-target CSVs in `results/test/` and agrees with `summary.csv` and
`pairs.csv`.

Verdicts: **supported** (a prediction with a significance condition that is met),
**holds** (a point-estimate prediction that is met), **falsified**, **pending** (not run
yet). Only RQ1a is confirmatory (section 4.6); every other verdict is descriptive and
rests on point estimates, as section 5 says.

Written on 13 October 2026, after the RQ1/RQ2 test run and before any RQ3–RQ5
experiment on the test split. RQ3 added on 13 October 2026, after the single linear-probe
test run (`6918ba5`, run at `d57bac8`; probes selected on videos 33–40 at `7077afb`).
Its numbers were recomputed from `results/test/probe_summary.csv`, the registered
zero-shot tables `results/test/*_classification.csv`, and the saved test-set phase
posteriors scored against the labels; all three agree. RQ4 added on 15 October 2026, after
the single temporal-model test run (`4f489a7`, run at `3bbe61f`; window, HMM and TCN
selected or fitted on videos 1–40 at `c423454`). Its numbers were recomputed from the
saved test-set phase posteriors, the saved HMM parameters and TCN weights, and the
selected windows, scored against the labels on the same 2,000 resamples, and agree with
`results/test/temporal_summary.csv` and `temporal_backbone_mean_gain.csv`. RQ5 added on
19 October 2026, after the single per-tag run (`b888322`, run at `854398d`) on the 816
tagged pool frames, with the tags frozen beforehand (`LOG.md`, 19 October) and RQ5c
clarified before any per-tag number (Deviations, 2026-10-19, `d85eb56`). Its probe
numbers were recomputed from the frozen corrected tags and the saved test posteriors on
the same resamples, and agree with `results/test/rq5_*.csv`.

## Test-set numbers used

Overall Cholec80 mAP = macro mean of the 14 phase and tool targets, each target's
headline being the mean AP over its five rephrasings (section 4.1). Interval: 95%
percentile, paired video-level bootstrap, B = 10,000.

| model | overall mAP [95% CI] | phase mAP | tool mAP | ensemble overall mAP (secondary) |
|---|---|---|---|---|
| PeskaVLP | 0.2775 [0.2684, 0.2885] | 0.2689 | 0.2861 | 0.3111 |
| SigLIP so400m-384 | 0.2636 [0.2541, 0.2740] | 0.2167 | 0.3105 | 0.2839 |
| MedSigLIP-448 | 0.2157 [0.2094, 0.2245] | 0.1822 | 0.2492 | 0.2242 |
| CLIP ViT-L/14 | 0.2026 [0.1966, 0.2117] | 0.1757 | 0.2296 | 0.2160 |

Random-ranker level (mean prevalence over the 14 targets): **0.1585**. Section 5 quoted
"about 0.175" from the label prevalences of videos 1–40; the test videos have a lower
one. This does not enter any verdict.

Confirmatory pairwise differences in overall mAP (B = 10,000; a p of 0 means no
resample crossed zero, so p < 0.0002):

| pair | difference [95% CI] | p | Holm p |
|---|---|---|---|
| PeskaVLP − SigLIP | +0.0139 [+0.0057, +0.0235] | 0.0010 | 0.0020 |
| PeskaVLP − MedSigLIP | +0.0618 [+0.0530, +0.0712] | 0 | 0 |
| PeskaVLP − CLIP | +0.0749 [+0.0672, +0.0820] | 0 | 0 |
| SigLIP − MedSigLIP | +0.0479 [+0.0396, +0.0558] | 0 | 0 |
| SigLIP − CLIP | +0.0610 [+0.0524, +0.0680] | 0 | 0 |
| MedSigLIP − CLIP | +0.0131 [+0.0045, +0.0205] | 0.0024 | 0.0024 |

All six reject at family-wise α = 0.05 (Holm).

RQ2, per model (section 4.1): mean over the 14 targets of the SD (ddof = 1) of AP across
the five rephrasings, and mean AP by rephrasing role (mean over the 14 targets).

| model | mean SD | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|---|
| CLIP ViT-L/14 | 0.0397 | 0.2074 | 0.1894 | 0.2084 | 0.2185 | 0.1894 |
| SigLIP so400m-384 | 0.0379 | 0.2773 | 0.2672 | 0.2777 | 0.2666 | 0.2291 |
| MedSigLIP-448 | 0.0229 | 0.2296 | 0.2202 | 0.2241 | 0.2049 | 0.1995 |
| PeskaVLP | 0.0770 | 0.2809 | 0.2864 | 0.2481 | 0.2626 | 0.3094 |
| **mean over models** | **0.0444** | **0.2488** | 0.2408 | 0.2396 | 0.2382 | **0.2319** |

RQ3, linear probe (section 4.3): phase accuracy of the probe selected on videos 33–40,
zero-shot phase accuracy (headline: mean over r1–r5, section 4.2) and the gain. Intervals:
95% percentile, paired video-level bootstrap, the first 2,000 resamples (descriptive).

| model | probe phase accuracy [95% CI] | zero-shot phase accuracy | gain [95% CI] |
|---|---|---|---|
| MedSigLIP-448 | 0.7383 [0.7125, 0.7651] | 0.1706 | +0.5678 [+0.5393, +0.5966] |
| SigLIP so400m-384 | 0.7341 [0.7031, 0.7643] | 0.1440 | +0.5901 [+0.5553, +0.6228] |
| CLIP ViT-L/14 | 0.6839 [0.6523, 0.7156] | 0.1314 | +0.5526 [+0.5202, +0.5825] |
| PeskaVLP | 0.6706 [0.6438, 0.6974] | 0.3459 | +0.3247 [+0.3027, +0.3452] |

Best-to-worst spread across the four models: zero-shot 0.2145 [0.1976, 0.2298] (PeskaVLP
− CLIP); probe 0.0678 [0.0560, 0.0848] (MedSigLIP − PeskaVLP).

RQ4, causal temporal models (section 4.4 and its 2026-10-13 clarification): phase
accuracy of each causal method on the selected probe's posteriors (TCN on the
embeddings), and its gain over the single-frame probe in parentheses. The TCN is the mean
over seeds 0, 1, 2. Intervals: 95% percentile, paired video-level bootstrap, the first
2,000 resamples (descriptive).

| model | single frame | moving average, causal (w) | HMM, forward | TCN, causal (mean of 3 seeds ± SD) |
|---|---|---|---|---|
| CLIP ViT-L/14 | 0.6839 | 0.7638 (+0.0799), w = 30 | 0.7394 (+0.0555) | 0.6964 ± 0.0029 (+0.0124) |
| SigLIP so400m-384 | 0.7341 | 0.8008 (+0.0667), w = 20 | 0.7842 (+0.0502) | 0.7354 ± 0.0173 (+0.0014) |
| MedSigLIP-448 | 0.7383 | 0.8020 (+0.0636), w = 20 | 0.7654 (+0.0271) | 0.7536 ± 0.0121 (+0.0153) |
| PeskaVLP | 0.6706 | 0.7561 (+0.0855), w = 30 | 0.7324 (+0.0618) | 0.7317 ± 0.0063 (+0.0611) |
| **mean gain over models [95% CI]** | — | **+0.0739 [+0.0661, +0.0812]** | +0.0487 [+0.0319, +0.0653] | +0.0225 [+0.0090, +0.0368] (± 0.0039 over seeds) |

Paired differences of the mean gain over models (same resamples): moving average − HMM
+0.0253 [+0.0042, +0.0466]; HMM − TCN +0.0261 [+0.0103, +0.0419]; moving-average gain −
single-frame probe spread +0.0062 [−0.0101, +0.0172], with the gain at least as large as
the spread on 70% of the resamples. Macro-F1, Jaccard and edit score (no prediction,
section 5) are in `results/test/temporal_summary.csv`.

RQ5, failure modes (section 4.5; Deviations 2026-10-18 and 2026-10-19): phase accuracy on
the 816 tagged pool frames (40 test videos), by tag, using the corrected tags
(out-of-body frames count as clean, or as blur only). A frame with several tags counts
in each. **Drop** = accuracy on clean frames − accuracy on the tag's frames (positive =
worse on the tag). Intervals: 95% percentile, paired video-level bootstrap over the 40
videos the tagged frames come from, the first 2,000 resamples (descriptive).

Linear-probe phase accuracy on clean frames and drop per artefact (frames; videos):

| model | clean (438; 40) | smoke (50; 25) | blur (211; 38) | glare (69; 29) | blood (66; 15) | occlusion (117; 37) | mean drop, 5 artefacts [95% CI] |
|---|---|---|---|---|---|---|---|
| CLIP ViT-L/14 | 0.7694 | +0.1894 | +0.1438 | +0.0593 | +0.0724 | −0.0255 | +0.0879 [+0.0209, +0.1571] |
| SigLIP so400m-384 | 0.7877 | +0.0477 | +0.1242 | −0.0239 | +0.1210 | −0.0157 | +0.0506 [−0.0292, +0.1299] |
| MedSigLIP-448 | 0.8105 | +0.0705 | +0.1423 | +0.0279 | +0.0832 | +0.0071 | +0.0662 [−0.0013, +0.1377] |
| PeskaVLP | 0.6781 | +0.0581 | +0.0762 | +0.0549 | +0.1781 | −0.1082 | +0.0518 [−0.0158, +0.1199] |
| **mean over models [95% CI]** | — | +0.0914 [−0.0308, +0.2143] | **+0.1216 [+0.0575, +0.1854]** | +0.0295 [−0.0322, +0.0996] | +0.1137 [−0.0096, +0.2431] | −0.0356 [−0.0982, +0.0394] | — |

Paired differences (same resamples): blur − blood +0.0079 [−0.1097, +0.1301]; blur −
smoke +0.0302 [−0.1070, +0.1683]; blur − glare +0.0921 [+0.0246, +0.1523]; blur −
occlusion +0.1572 [+0.0856, +0.2147] (mean over models). PeskaVLP's mean drop − CLIP's
−0.0361 [−0.1144, +0.0464]; − SigLIP's +0.0012 [−0.0932, +0.0917].

Zero-shot phase accuracy (headline over r1–r5), descriptive only (Deviations
2026-10-19): clean accuracy and mean drop over the five artefacts.

| model | clean | mean drop [95% CI] |
|---|---|---|
| CLIP ViT-L/14 | 0.1338 | +0.0074 [−0.0157, +0.0289] |
| SigLIP so400m-384 | 0.1589 | +0.0398 [+0.0131, +0.0638] |
| MedSigLIP-448 | 0.1863 | +0.0520 [+0.0226, +0.0793] |
| PeskaVLP | 0.3936 | +0.0741 [+0.0392, +0.1092] |

Per-artefact zero-shot drops, every per-tag accuracy with its interval, and the secondary
analysis are in `results/test/rq5_per_tag.csv`, `rq5_backbone_mean.csv`,
`rq5_mean_drop.csv` and `rq5_pairs.csv`.

## Verdicts

| ID | Prediction (section 5, verbatim) | Test-set number | Verdict | Deciding rule (section 5) |
|---|---|---|---|---|
| RQ1a | PeskaVLP has the highest overall Cholec80 mAP. *Falsified if* any other model's overall mAP is higher. Counted as **supported** only if PeskaVLP's difference from each of the other three models is significant after Holm–Bonferroni (section 4.6). | PeskaVLP 0.2775, highest; differences +0.0139 (SigLIP), +0.0618 (MedSigLIP), +0.0749 (CLIP), Holm p 0.0020, < 0.0002, < 0.0002 | **supported** | highest point estimate, and all three differences significant at Holm α = 0.05 |
| RQ1b | CLIP has the lowest overall Cholec80 mAP. *Falsified if* any other model's is lower. | CLIP 0.2026, lowest (next: MedSigLIP 0.2157) | **holds** | no other model lower |
| RQ1c | the full order is **PeskaVLP > MedSigLIP > SigLIP > CLIP**. *Falsified if* the point estimates give any other order. | PeskaVLP 0.2775 > **SigLIP 0.2636 > MedSigLIP 0.2157** > CLIP 0.2026 | **falsified** | point-estimate order differs: SigLIP is above MedSigLIP |
| RQ1d | PeskaVLP's overall Cholec80 mAP, used zero-shot exactly as released, is **between 0.30 and 0.40** (random-ranker level about 0.175). *Falsified if* it is outside that range. | 0.2775 [0.2684, 0.2885] | **falsified** | outside 0.30–0.40; the whole interval is below 0.30 |
| RQ2a | rewording matters moderately. The mean over targets of the SD of AP across the five rephrasings, averaged over the models, is **between 0.02 and 0.05**. *Falsified if* it is below 0.02 or above 0.05. | 0.0444 (per model 0.0229–0.0770) | **holds** | inside 0.02–0.05 |
| RQ2b | averaged over models and targets, **r3 (what is visible) has the highest mean AP**. *Falsified if* another role is higher. | r1 0.2488 is highest; r3 0.2396 is third | **falsified** | another role (r1) is higher |
| RQ2c | **r2 (the action, surgeon or instrument as subject) has the lowest mean AP**. *Falsified if* another role is lower. | r5 0.2319 is lowest; r2 0.2408 is second | **falsified** | another role (r5) is lower |
| RQ3a | PeskaVLP's linear-probe phase accuracy is **between 60% and 75%**. *Falsified if* it is outside that range. | 0.6706 (67.1%) | **holds** | inside 60–75% |
| RQ3b | the probe adds **20 to 35 points** over PeskaVLP's zero-shot phase accuracy (headline over r1–r5, section 4.2). The published gain is about 25 points with clinician-written prompts; the headline here averages five untuned rephrasings, so zero-shot accuracy is expected to be lower and the gain at least as large. *Falsified if* the gain is outside 20–35 points. | 0.6706 − 0.3459 = +0.3247 (32.5 points) | **holds** | inside 20–35 points |
| RQ3c | the probe **narrows** the gap between backbones but PeskaVLP stays on top. The spread between the best and worst backbone in probe phase accuracy is **smaller** than the spread in zero-shot phase accuracy, and PeskaVLP has the highest probe accuracy. *Falsified if* the spread does not shrink, or another backbone has a higher probe accuracy. | spread 0.2145 (zero-shot) → 0.0678 (probe); PeskaVLP's probe accuracy 0.6706 is the **lowest** (MedSigLIP 0.7383, SigLIP 0.7341, CLIP 0.6839) | **falsified** | the spread shrinks, but three other backbones have a higher probe accuracy |
| RQ4a | the best causal temporal method adds **3 to 8 accuracy points** over the single-frame probe, averaged over the backbones. *Falsified if* the gain is below 3 or above 8 points. | best causal method: moving average, +0.0739 (7.4 points) [+0.0661, +0.0812] | **holds** | inside 3–8 points |
| RQ4b | the **HMM (forward filtering)** gives the largest gain among the causal methods (moving average, HMM, TCN). *Falsified if* the causal moving average or the TCN gives a larger average gain. | moving average +0.0739 > HMM +0.0487 > TCN +0.0225 | **falsified** | the causal moving average gives a larger average gain |
| RQ4c | choosing the backbone matters more than temporal context. The temporal gain in RQ4a is **smaller** than the spread in single-frame probe accuracy between the best and worst backbone. *Falsified if* it is equal or larger. | gain 0.0739 vs spread 0.0678 (MedSigLIP − PeskaVLP) | **falsified** | the gain is larger than the spread (point estimates) |
| RQ5a | **motion blur** gives the largest drop in phase accuracy relative to clean frames, measured on linear-probe phase accuracy averaged over the backbones. *Falsified if* another artefact type gives a larger drop. | blur +0.1216 > blood +0.1137 > smoke +0.0914 > glare +0.0295 > occlusion −0.0356 (211 blurred frames) | **holds** | no other artefact has a larger mean drop |
| RQ5b | that drop is **small, under 5 points**. *Falsified if* it is 5 points or more. | blur drop +0.1216 (12.2 points) [+0.0575, +0.1854] | **falsified** | 5 points or more |
| RQ5c | PeskaVLP is **more robust** than the general models: its drop from clean frames, averaged over artefact types, is smaller than both CLIP's and SigLIP's. *Falsified if* it is equal to or larger than either. | mean probe drop: PeskaVLP +0.0518, CLIP +0.0879, **SigLIP +0.0506** | **falsified** | PeskaVLP's drop is larger than SigLIP's (by 0.0012) |

Notes on the verdicts:

* **RQ1d and the prompt ensemble.** PeskaVLP's prompt-ensemble overall mAP is 0.3111
  (descriptive interval [0.2999, 0.3245], B = 2,000), inside 0.30–0.40. The ensemble is a
  secondary system (section 4.1: "reported as a separate system, not the headline"), and
  RQ1d is about the headline, so it does not change the verdict.
* **RQ2a** is about the average over models. Taken one by one, PeskaVLP (0.0770) is
  above the range and MedSigLIP (0.0229) is near its lower end.
* **RQ2b/RQ2c.** The role means span only 0.2319–0.2488 when averaged over the models,
  and r2, r3 and r4 are within 0.003 of each other, so the order of the middle roles is
  not stable. The verdicts follow the point-estimate rule as registered. The models
  disagree on which role is best: r5 is PeskaVLP's best role and the worst role of each
  of the other three.

* **RQ3 and the sensitivity analysis.** The tool-head C-grid sensitivity analysis
  (Deviations, 2026-10-13; `results/test/probe_sensitivity_*.csv`) is not registered
  and is not part of any verdict. It changes only tool heads, and section 5 makes no
  prediction for tool mAP; every RQ3 verdict uses the primary, registered analysis.
* **RQ3c** fails on its second condition only. The first condition holds: the spread
  falls from 0.2145 to 0.0678. PeskaVLP is last on the point estimates, which is the
  rule. The interval of its gap to CLIP (−0.0134 [−0.0366, +0.0106]) includes zero; those
  of its gaps to SigLIP (−0.0635 [−0.0796, −0.0455]) and MedSigLIP (−0.0678
  [−0.0840, −0.0521]) do not (descriptive intervals, no significance claim, section 4.6).
  PeskaVLP's zero-shot lead (0.3459, the highest) does not survive the probe.
* **RQ3a and RQ3b** also hold over their whole descriptive intervals (accuracy
  [0.6438, 0.6974]; gain [+0.3027, +0.3452]). PeskaVLP's gain is the smallest of the
  four (the others gain +0.55 to +0.59): it starts from the highest zero-shot accuracy
  and ends at the lowest probe accuracy.
* **RQ4 and the TCN sensitivity analysis.** The TCN re-selected over depths 2–8 and up
  to 100 epochs (Deviations, 2026-10-13; `setting = tcn-causal-sensitivity-not-registered`,
  `results/test/temporal_tcn_sensitivity.csv`) is not registered and is not part of any
  verdict. Every RQ4 verdict uses the registered TCN (depths 2–4, at most 50 epochs).
  The offline methods (centred moving average, Viterbi) are registered as secondary and
  are not causal, so they do not enter RQ4a–c either.
* **RQ4a.** "Best" is the causal method with the largest mean accuracy gain over the
  four backbones; that is the moving average, which also has the largest gain on each
  backbone. The point estimate is inside 3–8 points; the upper end of its descriptive
  interval (8.1 points) is just above 8.
* **RQ4b.** The moving average beats the HMM on all four backbones, and the interval of
  their paired difference excludes zero. The TCN is last on average; on PeskaVLP the HMM
  and TCN gains are almost equal (+0.0618 vs +0.0611).
* **RQ4c** fails on the point estimates, which is the rule, but only narrowly: 0.0739
  against 0.0678. The descriptive intervals overlap heavily (gain [0.0661, 0.0812],
  spread [0.0560, 0.0848]), and the interval of the paired difference, +0.0062
  [−0.0101, +0.0172], includes zero. The best causal temporal gain and the choice of
  backbone are about equal in size. (The spread is best minus worst on each resample,
  as in RQ3c.)
* **RQ5 and the out-of-body secondary analysis.** The analysis with the 11 out-of-body
  frames excluded (Deviations, 2026-10-18 (iii); `setting =
  rq5-oob-excluded-not-registered`) is not registered and is not part of any verdict.
  Excluding them raises clean probe accuracy by 0.012–0.014 on every backbone, and every
  probe drop by 0.006–0.014. Zero-shot drops are descriptive only and enter no verdict
  (Deviations, 2026-10-19).
* **RQ5a** holds on the point estimates, but narrowly: blur's lead over blood (+0.0079)
  and smoke (+0.0302) is small, and the intervals of both paired differences include
  zero; only glare and occlusion are clearly below blur. Blood frames come from only 15
  of the 40 videos. Backbone by backbone, blur has the largest drop for SigLIP and
  MedSigLIP only; smoke is largest for CLIP and blood for PeskaVLP.
* **RQ5b** fails clearly: the whole interval of blur's drop, [+0.0575, +0.1854], is above
  5 points.
* **RQ5c** fails on its second condition only, and by 0.0012 on the point estimates; the
  interval of PeskaVLP − SigLIP is [−0.0932, +0.0917], so the two are not distinguishable.
  PeskaVLP's drop is smaller than CLIP's (−0.0361 [−0.1144, +0.0464]). Its mean includes
  a negative occlusion drop (−0.1082): on occluded frames its probe is more accurate
  than on clean frames, as is the case, less strongly, for CLIP and SigLIP. In the
  descriptive zero-shot numbers, PeskaVLP has the largest mean drop of the four.

## Other registered checks

**Calibration (section 4.1):** a negative query should score below every positive target's
mean cosine. Count of the 7 negatives that do, per model:

| model | negatives below every positive target mean |
|---|---|
| CLIP ViT-L/14 | 7 of 7 |
| SigLIP so400m-384 | 7 of 7 |
| MedSigLIP-448 | 7 of 7 |
| PeskaVLP | **2 of 7** (the X-ray and MRI negatives) |

PeskaVLP's lowest positive target mean is −0.026 (specimen bag); five negatives score
above it (means +0.005 to +0.099). Section 4.1 says this check is descriptive and that "a
failure is reported, not fixed". Details: `results/test/*_calibration.csv`.
