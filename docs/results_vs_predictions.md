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
experiment on the test split.

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
| RQ3a | PeskaVLP's linear-probe phase accuracy is between 60% and 75%. | — | pending | — |
| RQ3b | the probe adds 20 to 35 points over PeskaVLP's zero-shot phase accuracy. | — | pending | — |
| RQ3c | the probe narrows the gap between backbones but PeskaVLP stays on top. | — | pending | — |
| RQ4a–c | temporal context (gain, best causal method, gain vs backbone spread). | — | pending | — |
| RQ5a–c | failure modes (largest artefact drop, its size, PeskaVLP robustness). | — | pending | — |

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
