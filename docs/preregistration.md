# Pre-registration

**Status: fixed on 12 October 2026 and tagged `prereg-v1`.** Later changes go only in
"Deviations" at the end.

This document fixes the protocol of the benchmark **before any number exists on the
Cholec80 test videos (41–80) or the Endoscapes2023 test split**. It is committed and
tagged `prereg-v1`; the GitHub timestamp of that tag is the record of when it was
fixed. After that commit the text above the "Deviations" section is never edited.
Every later change is added to "Deviations", dated, with the reason (section 6).

State of the data when this was fixed (12 October 2026): frames and labels exist for
Cholec80 training and validation videos 1–40 (86,304 frames); embeddings exist only for
videos 1–5 (CLIP, SigLIP and PeskaVLP; none for MedSigLIP). No test video has been
fetched, extracted, embedded or scored. The only model numbers computed so far are
pipeline checks on videos 1–5 with four development queries; they are not results and
are not reported as such. Label prevalences of videos 1–40 were also computed, as a
chance-level reference for the predictions.

---

## 1. Data and splits

### 1.1 Cholec80

* 80 laparoscopic cholecystectomy videos (Twinanda et al., IEEE TMI 2017).
* **Videos 1–32: train. Videos 33–40: validation. Videos 41–80: test.**
* Frames: 1 fps by frame index (25 fps indices 0, 25, 50, …), 448 px short side,
  JPEG q90. A frame's phase is its 25 fps phase label at that index; its tools are the
  1 fps tool labels at that index. The one trailing sample per video that has a phase
  but no tool row is dropped and counted (`docs/datasets.md`). No other frame is
  excluded: dark, blurred and out-of-body frames stay in.
* Nothing is tuned on 41–80: no query wording, prompt, hyperparameter, window,
  threshold or model choice.

### 1.2 Endoscapes2023

* Official train/val/test splits (Murali et al.). **Used as a test set only**, for the
  CVS-criteria queries and cross-dataset generalisation. Nothing is trained or tuned on
  any Endoscapes split.
* Evaluated frames: the CVS-annotated frames of the official test split.
* A frame is relevant to criterion C*k* if at least 2 of the 3 annotators marked C*k*
  as achieved (mean annotator score ≥ 0.5). If the released label format turns out not
  to allow this rule, the rule used instead is recorded as a deviation before any
  Endoscapes score is computed.

### 1.3 Contamination policy

* Dataset overlap with Cholec80 test videos is mapped by video identity and enforced in
  code: `src/srb/datasets/overlap.py` (`forbidden_for_training`,
  `assert_no_test_leakage`), using CAMMA's `camma_dataset_overlaps` at commit
  `8347b9f4cb02ebe739747903e6eada272ee9d25e`. Every probe, tuning or cross-dataset
  training set is checked with `assert_no_test_leakage`. Endoscapes2023 train/val
  videos 1, 2, 3, 4, 7 and 121 are Cholec80 test videos and are never trained on.
* Backbone pretraining data is audited per model in `docs/datasets.md`, from sources
  actually read; anything not verified is marked "unverified".
* If a backbone's pretraining data is found to include Cholec80 test videos (or
  frames from them), its results are **still reported, with a contamination flag** in
  every table and in the text. Where the overlapping videos can be identified, its
  metrics are also reported on the test videos that do not overlap, as a secondary
  analysis. A result is never dropped silently. "Unverified" is reported as
  "unverified", not as "clean".

## 2. Models

| Name in code | Model | Source and pinned revision | Role |
|---|---|---|---|
| `clip-vit-l14` | CLIP ViT-L/14 | `openai/clip-vit-large-patch14` @ `32bd64288804d66eefd0ccbe215aa642df71cc41` | general baseline |
| `siglip-so400m-384` | SigLIP so400m, 384 px | `google/siglip-so400m-patch14-384` @ `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` | strong general baseline |
| `medsiglip-448` | MedSigLIP, 448 px | `google/medsiglip-448` @ `9cea28a1a1195f665105faa6e8544c112fd960a4` | medical |
| `peskavlp` | PeskaVLP, 224 px | code `CAMMA-public/SurgVLP` @ `85e858998eec47614614b89f6af1363e0ad3f47b`; checkpoint `PeskaVLP.pth` SHA-256 `d394be976dde11f7cf6014b903a29980099a802e7be8530906b07704c42149c8` (from the zip with SHA-256 `d82e27108ae4782652e663bfcfb0832f9af97201333b81401bb5a1f2dab712f0`; the download URL is unversioned, so the hash is the pin); text tower `emilyalsentzer/Bio_ClinicalBERT` @ `d5892b39a4adaed74b92212a44081509db72f87b`. Preprocessing and tokenisation as in SurgVLP's own `load`/`tokenize` at that commit (resize 360×640, centre-crop 224, ImageNet normalisation; padding to 77 tokens) | surgical |

* Image and text embeddings are L2-normalised; similarity is cosine, computed in
  float32 or higher from stored embeddings. SigLIP-family text is padded to 64 tokens
  (`padding="max_length"`), as the models were trained.
* Text is the query string exactly as in section 3, with no added template.
* **A model that cannot be run** (install failure, gated access lost, runtime failure)
  is reported as **excluded, with the reason**, recorded as a deviation. The remaining
  models are evaluated unchanged. The confirmatory comparison set (section 4.6)
  shrinks to the pairs among the models that ran.
* **Optional image-only encoder** (EndoFM / SurgeNet-style): if run, it is used for the
  probe and temporal experiments (RQ3, RQ4) only, never for text retrieval, with the
  same protocol as the other backbones; its revision is pinned as an addendum,
  before any test-set number exists for any model.
* A LoRA/adapter on the best backbone (plan: Week 7, first item cut if time runs
  short) is **exploratory** if it is run: it is not part of the confirmatory set.

## 3. Queries

Final wording. The machine-readable copy is `src/srb/queries.py`;
`tests/test_queries.py` checks that every string below appears there verbatim and vice
versa.

Written from the Cholec80 phase and tool definitions (Twinanda et al. 2017), the
Strasberg critical-view-of-safety criteria as annotated in Endoscapes2023, and
standard descriptions of the steps of laparoscopic cholecystectomy. **No wording was
chosen or changed on the basis of any score.** Three strings were scored during
development as pipeline checks on training videos 1–5 before this document existed
and are kept unchanged: "dissection of Calot's triangle in a laparoscopic
cholecystectomy", "clipping and cutting the cystic duct and artery", "a hook electrode
in the image", plus the negative control "a car on a road". Their development scores
did not influence any wording here.

**No query is truncated.** Text encoders cut long inputs silently, which would score a
different wording than the one registered. Every string below was tokenised with each
model's pinned tokenizer (CLIP 77 tokens; SigLIP and MedSigLIP 64; PeskaVLP's
Bio_ClinicalBERT tokenizer as called by SurgVLP, 77): the longest is 27 tokens for the
SigLIP-family tokenizers and 28 for Bio_ClinicalBERT, including special tokens.
`tests/test_queries.py` re-checks this against every tokenizer.

**A known ambiguity, kept on purpose.** The Cholec80 label *GallbladderRetraction*
means taking the bagged gallbladder out of the abdomen, but in surgical usage
"retraction" usually means pulling the gallbladder up to expose the field, which happens
during Preparation and Calot's triangle dissection. Its r5 wording, "gallbladder
retraction", is the label name a naive user would type and is kept; a low r5 score for
that target may reflect the ambiguity of the word rather than a failure of the model.

Each target has five rephrasings with fixed roles, the same for every target:
**r1** technical description, **r2** the action (surgeon or instrument as subject),
**r3** what is visible, **r4** lay wording without jargon, **r5** short label-like
phrase.

### 3.1 Cholec80 phases (7 × 5)

**Preparation**

* r1: preparation phase of a laparoscopic cholecystectomy: instruments are introduced and the gallbladder is exposed
* r2: the surgeon grasps the gallbladder fundus and retracts it to expose the operative field
* r3: an intact gallbladder under the liver, not yet dissected
* r4: the beginning of keyhole gallbladder surgery, before any cutting
* r5: surgical preparation and exposure

**CalotTriangleDissection**

* r1: dissection of Calot's triangle in a laparoscopic cholecystectomy
* r2: the surgeon dissects fat and peritoneum to expose the cystic duct and cystic artery
* r3: the cystic duct and cystic artery being exposed at the neck of the gallbladder
* r4: carefully separating tissue to find the tube and the blood vessel leading to the gallbladder
* r5: Calot's triangle dissection

**ClippingCutting**

* r1: clipping and cutting the cystic duct and artery
* r2: surgeon applies clips before cutting
* r3: clip applier on the cystic duct
* r4: sealing the gallbladder's tube and blood vessel with small metal clips, then cutting them
* r5: laparoscopic clipping step of a cholecystectomy

**GallbladderDissection**

* r1: dissection of the gallbladder from the liver bed in a laparoscopic cholecystectomy
* r2: the surgeon separates the gallbladder from the liver using electrocautery
* r3: a gallbladder partly detached from the liver bed
* r4: peeling the gallbladder away from the liver
* r5: gallbladder dissection

**GallbladderPackaging**

* r1: packaging the detached gallbladder into a specimen retrieval bag
* r2: the surgeon places the gallbladder inside a specimen bag
* r3: a fully detached gallbladder next to an open specimen bag
* r4: putting the removed gallbladder into a plastic pouch inside the abdomen
* r5: gallbladder packaging

**CleaningCoagulation**

* r1: cleaning and coagulation of the liver bed after the gallbladder has been detached
* r2: the surgeon irrigates, suctions and coagulates bleeding points on the liver bed
* r3: the empty liver bed where the gallbladder used to be
* r4: washing the area and stopping small bleeds after the gallbladder is out
* r5: cleaning and coagulation

**GallbladderRetraction**

* r1: extraction of the gallbladder in its specimen bag through the trocar port
* r2: the surgeon pulls the specimen bag with the gallbladder out of the abdomen
* r3: a specimen bag being drawn towards the port site
* r4: taking the bagged gallbladder out of the body
* r5: gallbladder retraction

### 3.2 Cholec80 tools (7 × 5)

**Grasper**

* r1: a laparoscopic grasper in the image
* r2: a grasper holding tissue
* r3: atraumatic forceps with small jaws at the tip of a long shaft
* r4: a long thin tool with small jaws that holds tissue
* r5: grasper

**Bipolar**

* r1: bipolar coagulation forceps in the laparoscopic image
* r2: bipolar forceps coagulating a bleeding vessel
* r3: forceps with two fine insulated tips for electrosurgical coagulation
* r4: a tool that stops bleeding by sealing tissue with electric current
* r5: bipolar

**Hook**

* r1: a hook electrode in the image
* r2: an electrocautery hook dissecting tissue
* r3: an L-shaped metal hook at the tip of a laparoscopic instrument
* r4: a hooked tool that burns through tissue
* r5: hook

**Scissors**

* r1: laparoscopic scissors in the image
* r2: scissors cutting tissue
* r3: curved scissor blades at the tip of a laparoscopic instrument
* r4: a long pair of surgical scissors inside the abdomen
* r5: scissors

**Clipper**

* r1: a clip applier in the laparoscopic image
* r2: a clip applier placing a metal clip
* r3: an instrument holding a metal clip between its jaws
* r4: a tool that closes off tubes with small metal clips
* r5: clipper

**Irrigator**

* r1: a suction-irrigation cannula in the laparoscopic image
* r2: an irrigator rinsing the operative field with fluid
* r3: a straight hollow tube suctioning fluid and blood
* r4: a tube that washes the area with water and sucks it away
* r5: irrigator

**SpecimenBag**

* r1: a specimen retrieval bag in the laparoscopic image
* r2: a specimen bag being opened inside the abdomen
* r3: a plastic retrieval pouch
* r4: a plastic bag used to remove tissue from the body
* r5: specimen bag

### 3.3 Endoscapes2023 CVS criteria (3 × 5)

**C1 (two structures)**

* r1: two tubular structures entering the gallbladder
* r2: the surgeon has exposed only two structures, the cystic duct and the cystic artery, entering the gallbladder
* r3: the cystic duct and cystic artery clearly seen connecting to the gallbladder
* r4: two thin tubes clearly visible going into the gallbladder
* r5: two structures connected to the gallbladder

**C2 (hepatocystic triangle cleared)**

* r1: hepatocystic triangle cleared of fat
* r2: the surgeon has cleared all fat and fibrous tissue from the hepatocystic triangle
* r3: a clean, fat-free triangle between the cystic duct, the liver and the common hepatic duct
* r4: the area at the neck of the gallbladder fully cleaned of fat
* r5: cleared hepatocystic triangle

**C3 (cystic plate)**

* r1: lower gallbladder separated from the liver bed
* r2: the surgeon has separated the lower third of the gallbladder from the liver to expose the cystic plate
* r3: the cystic plate visible beneath the lifted lower part of the gallbladder
* r4: the bottom of the gallbladder lifted off the liver
* r5: cystic plate exposed

### 3.4 Negative / control queries (7)

Five non-medical scenes and two non-surgical medical images (a calibration check that matters most for the medical model).

* neg1: a car on a road
* neg2: a dog playing in a park
* neg3: a plate of food on a table
* neg4: a city skyline at night
* neg5: a page of printed text
* neg6: a chest X-ray
* neg7: an MRI scan of the brain

### 3.5 Relevance rules

* **Phase query** for phase *P*: a frame is relevant iff its phase label is *P*.
* **Tool query** for tool *T*: a frame is relevant iff tool *T* is present (= 1) at
  that frame.
* **CVS query** for criterion C*k*: the Endoscapes rule in section 1.2.
* **Negative query**: no frame is relevant (R = 0). Negatives are excluded from every
  mAP and used only for the calibration check (section 4.1).

## 4. Metrics and analysis

### 4.1 Retrieval (RQ1, RQ2)

As in `docs/metrics.md`, with every metric tie-neutral (expected value over random
orderings within tied groups).

* **Ranking pool:** all frames of the evaluated split pooled across videos (Cholec80:
  all 40 test videos; Endoscapes: the CVS-annotated test frames).
* **Primary:** AP per query string over the full ranking, always reported next to its
  prevalence R/N and lift = AP/prevalence.
* **RQ1 headline per target** (phase, tool or criterion): the **mean AP over its five
  rephrasings**. **mAP** = macro mean of the target headlines within a family: phase
  mAP (7 targets), tool mAP (7), CVS mAP (3). **Overall Cholec80 mAP** = macro mean of
  the 14 phase and tool targets; it is the quantity of the confirmatory comparison.
* **Secondary:**
  * R-Precision (same aggregation);
  * **per-video AP**: AP within each test video in which the target has R > 0, averaged
    over those videos, then over rephrasings and targets; the number of videos entering
    each target is reported;
  * **prompt ensemble**: one text embedding per target = the normalised mean of its five
    rephrasing embeddings, ranked once; reported as a separate system, not the headline.
* **Descriptive only:** P@10, P@50, and **distinct videos in the top 50** per query.
* **Calibration check (negatives):** per model, the mean and maximum cosine of each
  negative query over the test frames, next to the same for the positive queries.
  Expected: every negative scores below every positive target's mean. Descriptive; a
  failure is reported, not fixed.
* **RQ2 (prompt sensitivity):** per target, the standard deviation (ddof = 1) and the
  range of AP over its five rephrasings; per model, the mean SD over targets; and the
  mean AP by rephrasing role (r1–r5). Also, for each pair of models, the fraction of
  targets for which the two models' order changes between rephrasings.

### 4.2 Zero-shot classification

* **Phases:** for rephrasing role *r*, the 7 phase strings of role *r* are the class
  prompts; each frame is assigned the argmax cosine. Frame-level **accuracy** and
  **macro-F1** over the 7 phases, pooled over all test frames. **Headline: the mean
  over r = 1…5**; the prompt-ensemble classifier is secondary. The mean over test
  videos of per-video accuracy is reported as a secondary figure for comparability
  with the phase-recognition literature.
* **Tools:** multi-label; per-tool AP of the tool query scores, macro-averaged
  (tool mAP). This is the same quantity as tool retrieval AP in 4.1 and is reported
  once.

### 4.3 Linear probe (RQ3)

* Frozen image embeddings (as stored in the index, upcast to float32), no further
  normalisation.
* **Phases:** multinomial logistic regression (scikit-learn, L2 penalty, `lbfgs`,
  `max_iter = 5000`). **C ∈ {10⁻³, 10⁻², 10⁻¹, 1, 10, 100}**, trained on videos 1–32 and
  selected by macro-F1 on 33–40. The model at the chosen C, trained on 1–32 only, is
  evaluated once on 41–80. Accuracy and macro-F1 (as in 4.2).
* **Tools:** one binary logistic regression per tool, same C grid, C selected per tool
  by AP on 33–40; tool mAP on 41–80.
* No class weighting. The training manifest passes `assert_no_test_leakage`.

### 4.4 Temporal models (RQ4)

Inputs are per-frame phase posteriors from the selected linear probe of each backbone
(4.3), in video order at 1 fps; the TCN takes the frozen embeddings. All tuning on
33–40, all training on 1–32.

* **Single frame:** the probe's argmax (the RQ3 number).
* **Moving average (causal):** mean of the posteriors over the trailing window of *w*
  frames, then argmax; **w ∈ {5, 10, 20, 30, 60, 120}** s, chosen on 33–40 by macro-F1.
  A centred window with the same grid is reported as secondary (offline).
* **HMM:** 7 states; transition matrix and initial distribution estimated from the
  1 fps phase sequences of videos 1–32 with add-one smoothing; emission scores = probe
  posterior / training class prior. **Primary: forward filtering (causal).** Viterbi
  decoding (offline) secondary. No tuned parameters.
* **Causal TCN:** dilated causal 1-D convolutions over the embedding sequence, kernel 3,
  64 channels, dilations 1, 2, 4, …; **depth ∈ {2, 3, 4}** chosen on 33–40; Adam,
  learning rate 10⁻³, cross-entropy, at most 50 epochs with the epoch chosen by
  validation macro-F1; one video per batch. **Seeds 0, 1, 2**: each seed is reported,
  and the headline is their mean ± SD.
* **Metrics:** accuracy and macro-F1 (as in 4.2); **Jaccard** per phase computed per
  test video, averaged over the videos in which the phase occurs in the ground truth,
  then over phases; **segmental edit score** (Lea et al., 2016: 100 × (1 − normalised
  Levenshtein distance between segment label sequences)) per video, averaged over
  videos.

### 4.5 Failure modes (RQ5)

* **Sample, fixed now:** a pool of 2,000 Cholec80 test frames, 50 per test video, drawn
  uniformly without replacement within each video with NumPy
  `default_rng(20261008)`; the pool is then put in a random tagging order with
  `default_rng(20261009)`. Both are generated by a committed script, and the pool's
  frame list is hashed and recorded before tagging starts.
* **Tagging:** frames are tagged in that order, each with every artefact that applies:
  smoke, blur, glare (specular), blood, instrument occlusion, or **clean** (none).
  Tagging stops once every artefact type (smoke, blur, glare, blood, occlusion; **"clean"
  does not count**) has at least 50 frames, or after 1,000 frames, whichever comes first
  (at least 300 frames are tagged in any case). Up to 1,000 frames is several hours of
  manual tagging by the author; that cost is accepted. The tagger sees
  only the image, never a model score, rank or prediction. Tags are frozen before any
  per-tag metric is computed.
* **Metrics per tag:** zero-shot phase accuracy (headline over r = 1…5) and linear-probe
  phase accuracy on frames with that tag, with the difference to clean frames.
  Artefact types with fewer than 20 tagged frames are reported as counts only.
* **Optional expert check:** if a surgeon or resident is available, 50 tagged frames
  drawn with `default_rng(20261010)` are re-tagged blind; Cohen's κ per artefact type
  is reported.

### 4.6 Uncertainty and multiple comparisons

* **Paired video-level bootstrap.** Frames within a video are not independent, so
  resampling is over **videos**: 40 Cholec80 test videos (or the Endoscapes test
  videos), drawn with replacement, NumPy `default_rng(20261011)`. **B = 10,000 resamples for
  the confirmatory family** (below: the models' overall Cholec80 mAP and their pairwise
  differences) and **B = 2,000 for every descriptive interval**, drawn as the first
  2,000 resamples of the same stream. Reason, measured before this was fixed: one
  resample of pooled AP over 70 query strings takes ≈ 1.0 s per model on the 86,304
  frames of training videos 1–40 (≈ 15 ms per AP, one M1 core), so 10,000 resamples
  cost ≈ 3 CPU-hours per model. Every model is evaluated on the same resamples, and each
  metric is recomputed from scratch on the pooled frames of each resample (a video
  drawn twice contributes its frames twice; the tie-neutral metrics handle the exact
  duplicates). 95% percentile intervals are reported for every model's headline metrics
  and for every pairwise difference. Applies to retrieval, classification, probe,
  temporal and per-tag metrics alike (for RQ5, over the videos the tagged frames come
  from). With only 40 clusters, percentile intervals may under-cover slightly; this is
  stated, not corrected.
* **Confirmatory set:** the **6 pairwise differences in overall Cholec80 mAP** among
  the 4 text-capable models (fewer if a model is excluded). Two-sided bootstrap
  p-value = 2 × min(share of resamples with difference ≤ 0, share with difference ≥ 0);
  **Holm–Bonferroni** at family-wise α = 0.05.
* **Everything else is descriptive:** per-query, per-family, per-rephrasing,
  classification, probe, temporal, Endoscapes and per-tag results are reported with
  unadjusted 95% intervals, labelled as such, and no significance claim is made from
  them.

### 4.7 Reporting

* Every model and setting that is run is reported, including the losing ones and the
  excluded ones.
* One row per (model, setting, split, metric) in `results/results.csv` with seed, git
  hash and date. Development numbers on training/validation videos are never written
  there.

## 5. Predictions

Written by the author on 12 October 2026, before any test number exists. Each
prediction states what would count as falsifying it. Numbers refer to the Cholec80
test videos 41–80 and the metrics defined in section 4. Only the RQ1 model comparison
is confirmatory (section 4.6); everything else is judged on point estimates and is
descriptive.

**Prior evidence read before predicting.** On 12 October 2026, after a first draft
and before committing, the author read two published Cholec80 results and revised the
RQ3 predictions where the draft contradicted them. Both papers use their own splits and
protocols, so the numbers are a guide, not a target:

* PeskaVLP paper (Yuan et al., 2024, arXiv:2410.00263), frame-wise phase recognition
  on Cholec80: zero-shot accuracy CLIP 30.8%, PeskaVLP 45.1% (clinician-written
  prompts); linear probe CLIP 64.8%, PeskaVLP 69.9%.
* TeCNO (Czempiel et al., 2020, arXiv:2003.10751): a fine-tuned single-frame ResNet-50
  at 82.2% accuracy rises to 88.4% with a single-stage causal TCN.

### RQ1: zero-shot retrieval

* **RQ1a:** PeskaVLP has the highest overall Cholec80 mAP. *Falsified if* any other
  model's overall mAP is higher. Counted as **supported** only if PeskaVLP's
  difference from each of the other three models is significant after
  Holm–Bonferroni (section 4.6).
* **RQ1b:** CLIP has the lowest overall Cholec80 mAP. *Falsified if* any other
  model's is lower.
* **RQ1c:** the full order is **PeskaVLP > MedSigLIP > SigLIP > CLIP**. *Falsified
  if* the point estimates give any other order.
* **RQ1d:** PeskaVLP's overall Cholec80 mAP, used zero-shot exactly as released, is
  **between 0.30 and 0.40** (random-ranker level about 0.175). *Falsified if* it is
  outside that range.

### RQ2: prompt sensitivity

* **RQ2a:** rewording matters moderately. The mean over targets of the SD of AP across
  the five rephrasings, averaged over the models, is **between 0.02 and 0.05**.
  *Falsified if* it is below 0.02 or above 0.05.
* **RQ2b:** averaged over models and targets, **r3 (what is visible) has the highest
  mean AP**. *Falsified if* another role is higher.
* **RQ2c:** **r2 (the action, surgeon or instrument as subject) has the lowest mean
  AP**. *Falsified if* another role is lower.

### RQ3: linear probe

* **RQ3a:** PeskaVLP's linear-probe phase accuracy is **between 60% and 75%**.
  *Falsified if* it is outside that range.
* **RQ3b:** the probe adds **20 to 35 points** over PeskaVLP's zero-shot phase accuracy
  (headline over r1–r5, section 4.2). The published gain is about 25 points with
  clinician-written prompts; the headline here averages five untuned rephrasings, so
  zero-shot accuracy is expected to be lower and the gain at least as large.
  *Falsified if* the gain is outside 20–35 points.
* **RQ3c:** the probe **narrows** the gap between backbones but PeskaVLP stays on top.
  The spread between the best and worst backbone in probe phase accuracy is
  **smaller** than the spread in zero-shot phase accuracy, and PeskaVLP has the
  highest probe accuracy. *Falsified if* the spread does not shrink, or another
  backbone has a higher probe accuracy.
* No numerical prediction is made for probe macro-F1 or tool mAP.

### RQ4: temporal context

* **RQ4a:** the best causal temporal method adds **3 to 8 accuracy points** over the
  single-frame probe, averaged over the backbones. *Falsified if* the gain is below 3
  or above 8 points.
* **RQ4b:** the **HMM (forward filtering)** gives the largest gain among the causal
  methods (moving average, HMM, TCN). *Falsified if* the causal moving average or the
  TCN gives a larger average gain.
* **RQ4c:** choosing the backbone matters more than temporal context. The temporal
  gain in RQ4a is **smaller** than the spread in single-frame probe accuracy between
  the best and worst backbone. *Falsified if* it is equal or larger.
* No numerical prediction is made for macro-F1, Jaccard or edit score.

### RQ5: failure modes

* **RQ5a:** **motion blur** gives the largest drop in phase accuracy relative to clean
  frames, measured on linear-probe phase accuracy averaged over the backbones.
  *Falsified if* another artefact type gives a larger drop.
* **RQ5b:** that drop is **small, under 5 points**. *Falsified if* it is 5 points or
  more.
* **RQ5c:** PeskaVLP is **more robust** than the general models: its drop from clean
  frames, averaged over artefact types, is smaller than both CLIP's and SigLIP's.
  *Falsified if* it is equal to or larger than either.
* If fewer than 20 blurred frames are tagged (section 4.5), RQ5a and RQ5b are reported
  as **untestable**, not as supported.

## 6. Deviations policy

After the `prereg-v1` commit, the text above this section is not edited. Any change of
protocol (a query wording, a metric, a split, a hyperparameter grid, an excluded model,
a different Endoscapes relevance rule, a bug fix that changes numbers) is appended
below as a dated entry with: what changed, why, whether any test-set number had been
seen at the time, and which results it affects. Pinning a revision that section 2
leaves open (the optional encoder) is recorded the same way, as an addendum,
and is only valid if dated before the first test-set number of any model.

## Deviations

### 2026-10-13 — clarification of section 2 (no protocol change)

Section 2 pins model revisions, not the compute precision of the image encoders. Frame
embeddings for videos 1–80 were computed in float16 on an NVIDIA L4 for CLIP, SigLIP and
MedSigLIP, and in float32 on Apple MPS for PeskaVLP; all four are stored in float16 as
before. The float16 choice was made on 12 October 2026, before any test-set number
existed (the first came on 13 October at `1759cb4`). Check against float32 on the same
512 frames: min cosine 0.99994 (CLIP), 0.99982 (SigLIP), 0.99997 (MedSigLIP)
(`PROGRESS.md` §1b). Cross-device check, CLIP video04, L4 float16 vs Mac float32: min
cosine 0.99977, run before both the 1–40 and the 41–80 runs. Similarities are computed
in float32 from the stored embeddings, as section 2 requires, and the probe (4.3)
upcasts them to float32. Each index's `meta.json` records its compute dtype. Test-set
numbers existed when this entry was written; the choice was made before them. No
reported number changes, because every test-set number was computed from these
embeddings.

### 2026-10-13 — secondary sensitivity analysis: probe tool heads over an extended C grid (not registered)

**What is added.** A secondary RQ3 analysis, labelled "sensitivity, not registered".
Each tool head of the linear probe (section 4.3) is re-selected by AP on videos 33–40
over the registered C grid extended with {1e3, 1e4}, giving {1e-3, 1e-2, 1e-1, 1, 10,
100, 1e3, 1e4}. Everything else follows 4.3: training on videos 1–32, lbfgs, L2,
max_iter 5000, no class weighting, and ties going to the smaller C, as in the
registered implementation (section 4.3 does not specify a tie rule). Phase heads stay
on the registered grid only. On videos 41–80, this analysis's tool mAP is reported next
to the primary tool mAP. Both use the same 2,000 paired resamples of
`default_rng(20261011)`, and the paired difference is reported too.

**Why.** In the validation run (videos 33–40, 13 October 2026, `7077afb`), 13 of the 28
tool heads (4 models × 7 tools) chose C = 100, the largest value in the grid. The best
C for those heads may lie above the grid.

**Test-set numbers seen.** The RQ1/RQ2 test-set numbers existed (13 October,
`1759cb4`). No RQ3 test-set number existed: the probe had not been scored on any test
video. The extended-grid selection on 33–40 was run before this entry was committed
(dev, `fa89df2`) and used no test video.

**Affects.** Nothing registered. The primary RQ3 analysis, its grid and its tool mAP are
unchanged, and only the primary analysis is compared with the RQ3 predictions. The
sensitivity numbers are added rows (`setting = linear-probe-sensitivity-not-registered`)
and files (`results/test/probe_sensitivity_*.csv`).

### 2026-10-13 — clarification of section 4.4 (no protocol change)

Section 4.4 leaves some implementation choices open. They are fixed here, before any
RQ4 number exists on any split: no temporal model had been fitted, selected or scored on
videos 33–40 or 41–80 when this entry was written. The inputs are the phase posteriors
of the selected probes (4.3) exactly as saved, and the probes are not refitted.

* **Moving average.** At the start of a video the trailing window is shortened to the
  frames available (the mean over frames max(0, t−w+1) … t). The centred window
  (secondary) covers frames t−⌊w/2⌋ … t+⌈w/2⌉−1, which is exactly w frames, shortened the
  same way at both ends of the video. Ties in the choice of w go to the smaller w.
* **HMM.** The class prior is the share of each phase among the 1 fps frames of videos
  1–32 (the probe's training frames). Posteriors are floored at 10⁻¹² before they are
  divided by the prior; the number of floored values is reported (none on videos 1–40).
  Forward filtering: α₀ ∝ π ⊙ e₀, then αₜ ∝ (αₜ₋₁ A) ⊙ eₜ, normalised at every step;
  the prediction at t is argmax αₜ. Viterbi uses the same π, A and emissions, in log
  space. Add-one smoothing applies to every cell of A (counts of consecutive 1 fps
  phase pairs within each video of 1–32) and to every entry of π (counts of each video's
  first phase).
* **Causal TCN.** Layer 1 is a causal convolution from the embedding dimension to 64
  channels (kernel 3, dilation 1); layers 2…D map 64 to 64 with dilations 2, 4, 8; each
  convolution is followed by a ReLU; a 1×1 convolution maps 64 channels to the 7 phases.
  Causality comes from left-only zero padding of (k−1)·d frames per layer. There is no
  normalisation, dropout or residual connection, and PyTorch's default initialisation is
  used. Receptive field: 7, 15 and 31 frames (seconds) for D = 2, 3, 4. The input is the
  frozen embeddings as stored in the index, upcast to float32, with no further
  normalisation (as in 4.3). The loss is the unweighted mean cross-entropy over the
  frames of one video; the order of the 32 training videos is reshuffled every epoch by a
  generator seeded with the run's seed.
* **TCN selection.** Each seed (0, 1, 2) is a full replicate: for every depth all 50
  epochs are trained (no early stopping), and the (depth, epoch) pair with the best
  macro-F1 on videos 33–40 is selected jointly, so the depth can differ between seeds.
  Ties go to the smaller depth, then the earlier epoch. Training runs on the CPU with
  `torch.use_deterministic_algorithms(True)` and fixed seeds.
* **Edit score and Jaccard.** No segment is filtered: short segments in the predictions
  are kept as they are. The edit score of a video is 100 × (1 − Levenshtein distance
  between the predicted and true segment label sequences / the longer of the two
  sequence lengths) (Lea et al., 2016). Jaccard follows 4.4 literally: for each video
  and each phase in its ground truth, |P ∩ G| / |P ∪ G| over frames, averaged over those
  videos, then over the 7 phases; a phase predicted in a video where it does not occur
  adds no term for that phase.
* **Intervals (4.6).** On each of the first 2,000 resamples of `default_rng(20261011)`:
  accuracy and macro-F1 from multiplicity-weighted confusion counts; Jaccard and edit
  score as multiplicity-weighted means of the per-video values (for Jaccard, each phase
  over the resampled videos in which it occurs). The TCN headline on each resample is the
  mean over the three seeds; its interval is the interval of that mean, and the ± is the
  SD (ddof = 1) over seeds on the full set. The gain over single frame is the paired
  difference on the same resamples, reported for all four metrics. All descriptive.

**Affects.** No registered choice changes; these fix details that 4.4 leaves open.

### 2026-10-13 — secondary sensitivity analysis: causal TCN over an extended depth and epoch grid (not registered)

**What is added.** A secondary RQ4 analysis, labelled "sensitivity, not registered". The
causal TCN of section 4.4 is re-selected on videos 33–40 over **depths {2, 3, 4, 5, 6, 7,
8}** and **at most 100 epochs**. Everything else is as registered and as in the
clarification of section 4.4 above: the same architecture rule (dilations 1, 2, 4, …,
2^(D−1)), the same seeds (0, 1, 2), training on videos 1–32 only, the same joint
(depth, epoch) selection by macro-F1 on 33–40 per seed with the same tie rule, and the
same CPU threads with deterministic algorithms. Because training does not depend on the
epoch cap, the first 50 epochs of depths 2–4 reproduce the registered validation curves
exactly (checked by the script, which stops otherwise). On videos 41–80 this analysis is
reported next to the registered TCN, per seed and as the mean over seeds, on the same
2,000 paired resamples of `default_rng(20261011)`, with the paired difference
(sensitivity − registered).

**Why.** In the validation run (videos 33–40, 13 October 2026, `c423454`), 10 of the 12
(backbone, seed) TCN selections chose depth 4, the largest registered depth, and 2 chose
epoch 50, the last registered epoch. With kernel 3 and dilations 1, 2, 4, 8, a depth-4
causal TCN has a receptive field of exactly 31 frames (1 + 2·(1 + 2 + 4 + 8)), i.e. 31 s
at 1 fps; depths 5–8 give 63, 127, 255 and 511 s. Cholec80 phases last minutes: in
training videos 1–32 the median phase segment lasts 152 s (from 74 s for
GallbladderRetraction to 666 s for CalotTriangleDissection). The best TCN may lie
outside the registered grid.

**Test-set numbers seen.** The RQ1/RQ2 (13 October, `1759cb4`) and RQ3 (13 October,
`d57bac8`) test-set numbers existed. No RQ4 test-set number existed: no temporal model
had been scored on any test video. The registered RQ4 validation run (`c423454`) and the
extended selection on 33–40 (dev, `d63c5ec`) were run before this entry was committed;
neither used a test video.

**Affects.** Nothing registered. The registered TCN (depths 2–4, at most 50 epochs), its
selection and every verdict are unchanged, and only the registered TCN is compared with
the RQ4 predictions. The sensitivity numbers are added rows
(`setting = tcn-causal-sensitivity-not-registered`) and a file
(`results/test/temporal_tcn_sensitivity.csv`).

### 2026-10-18 — RQ5: out-of-body frames (clarification of section 4.5, review-pass correction, and a secondary analysis, not registered)

**(i) Clarification of section 4.5 (no protocol change).** A frame in which the camera is
outside the patient (operating room, hands or gloves, drapes, the trocar from outside, a
black or blank screen) shows none of the five artefacts as section 4.5 means them, so in
the registered RQ5 analysis it is **clean** ("clean (none)"). If it is also blurred it is
tagged blur, and it is then not clean. "Instrument occlusion" means an instrument, the
specimen bag or gauze hiding the anatomy **inside the abdomen**. Hands or the operating
room filling the screen are not occlusion, and smoke, glare and blood are tagged only
inside the abdomen. Out of body is not an artefact type: it counts toward no artefact's
50 frames in the stopping rule. The tagging tool records it as clean (or blur) in the
tags file plus a flag in a separate file (`data/rq5/out_of_body.csv`).

**(ii) Review-pass correction of frames 0–435.** Frames 0–435 of the pool were tagged
between 15 October (first tag 23:45 IST) and 18 October 2026, before the tool had an
out-of-body key. During that tagging, out-of-body frames were tagged occlusion. The raw
tags are kept unchanged (`data/rq5/tags.csv`, 436 frames, SHA-256
`750483857807963d270812ab1696242186d6e105004772d83ec5070f50d732a6`, with a backup copy).
One review pass over frames 0–435, in tag order, shows only the image (no existing tags,
no labels, no model output) and records only "out of body: yes/no". For frames answered
yes, smoke, glare, blood and occlusion are set to 0, blur is kept, and clean is set to 1
if blur is not set (`scripts/rq5_apply_corrections.py`, which writes
`data/rq5/tags_corrected.csv` and never touches the raw tags). The review is done before
any per-tag metric is computed. Frames from 436 on are tagged with the out-of-body key
directly. The stopping rule is counted on the corrected tags; tagging resumes only after
the review is complete. The tagging guide (`docs/rq5_tagging_guide.md`) was written
before the first tag (15 October, 23:38 IST) but committed only after 436 frames
(`091a67a`); the out-of-body rule and this correction are in its change log.

**(iii) Secondary analysis (sensitivity, not registered).** Every per-tag metric of
section 4.5 (zero-shot phase accuracy, headline over r = 1…5; linear-probe phase
accuracy; their difference to clean frames; counts only below 20 frames) is repeated
with out-of-body frames **excluded entirely**: from clean, and from blur if they were
blurred. It is reported next to the registered analysis, labelled "out of body excluded,
not registered", and is part of no verdict.

**Test-set numbers seen.** The RQ1/RQ2 (`1759cb4`), RQ3 (`d57bac8`) and RQ4 (`3bbe61f`)
test-set numbers existed. No RQ5 per-tag metric existed: no model score, prediction or
accuracy had been computed on or looked at for any tagged pool frame, and none is
computed before the tags are frozen. The issue was found while tagging, from the images
only.

**Affects.** RQ5 only. The registered RQ5 analysis and the RQ5a–c verdicts use the
corrected tags (out-of-body frames as clean, or as blur only). Section 4.5's pool,
tagging order, stopping thresholds and metrics are unchanged. The secondary analysis
adds rows and files labelled not registered.

### 2026-10-19 — clarification of RQ5c (no protocol change)

**What is clarified.** RQ5c is judged on the same metric as RQ5a: **linear-probe phase
accuracy**. A "drop" is the accuracy on clean frames minus the accuracy on frames with
the tag (positive = worse on the tag). "Averaged over artefact types" is the unweighted
mean of the drops for smoke, blur, glare, blood and occlusion. Zero-shot drops (headline
over r = 1…5, section 4.2) are reported as descriptive only. For RQ5a and RQ5b the same
drop is used, averaged without weights over the four backbones; RQ5b's "that drop" is
blur's drop, the artefact RQ5a names. A frame with several tags counts in each of them,
and clean frames are the frames tagged clean in the corrected tags (Deviations entry on
out-of-body frames, 2026-10-18).

**Test-set numbers seen.** Written before any per-tag metric was computed. The RQ5 tags
were complete and frozen (816 frames; SHA-256 of the three tag files in `LOG.md`,
19 October 2026). RQ1–RQ4 test-set numbers existed.

**Affects.** No registered choice changes; this fixes how RQ5c, and the averaging in
RQ5a and RQ5b, are computed.

### 2026-10-20 — clarification of sections 1.2, 3.3 and 4.1 for Endoscapes2023 (no protocol change), and a triggered secondary analysis (not registered)

**Source files.** Read from the official archive by HTTP range request (CRC-32 checked):
`all_metadata.csv`, `train_vids.txt`, `val_vids.txt`, `test_vids.txt`, `README.md` and
`test/annotation_{coco,coco_vid,ds_coco}.json`.

**Evaluated frames (1.2).** The CVS-annotated frames of the official test split are the
rows of `all_metadata.csv` with `is_ds_keyframe == True` and `vid` in `test_vids.txt`
(40 videos, public IDs 162–201): **1,799 frames**, 7–178 per video, 5 s apart. They are
exactly the images of `test/annotation_ds_coco.json`. The other 47,723 frames of the
archive also carry `C1`–`C3` values, but these have no per-annotator columns and are
derived, not annotated; they are not used. The 228 frames that are in `test/` but not in
`all/` (videos 190 and 194) include no keyframe.

**Relevance (1.2, 3.5).** The release gives each annotator's binary `[C1, C2, C3]`
(`cvs_annotator_1..3`), and every one of the 11,090 keyframes has all three. A frame is
relevant to criterion C*k* iff **at least 2 of the 3 annotators** marked it achieved. The
code asserts that this equals `Ck ≥ 0.5` (the released mean) on every frame; it does on
all 11,090. The rule is therefore applied literally, and it is the authors' own ground
truth ("majority vote for each criterion", Murali et al., arXiv:2312.12429). On the
1,799 test frames, R (prevalence) = **C1 423 (0.2351), C2 302 (0.1679), C3 501
(0.2785)**.

**Unanimity secondary, rule fixed before any image was fetched.** The authors report low
inter-rater agreement (Cohen's κ averaged over the three annotator pairs: C1 0.33,
C2 0.53, C3 0.44; arXiv:2312.12429). Rule: if, for any criterion, the 2–1 splits are
≥ 20% of the positive test keyframes, a unanimity analysis (relevant iff 3 of 3) is added
for all three criteria, labelled "unanimity, not registered". Counts (labels only):

| criterion | positives (≥ 2 of 3) | 2–1 | 3–0 | share 2–1 |
|---|---|---|---|---|
| C1 | 423 | 346 | 77 | 0.818 |
| C2 | 302 | 164 | 138 | 0.543 |
| C3 | 501 | 279 | 222 | 0.557 |

**Outcome: triggered** (all three ≥ 20%). Unanimity R (prevalence): C1 77 (0.0428),
C2 138 (0.0767), C3 222 (0.1234). The registered analysis and every headline remain
majority vote; the unanimity numbers enter no headline and no `results.csv` row without
the "not registered" label.

**Preprocessing.** As for Cholec80 (1.1): each released JPEG is resized to a 448 px short
side (bicubic) and re-encoded at JPEG q90, then passed to each model's own processor. This
is a second JPEG encode. The image size is read from every fetched JPEG (not from the COCO
`width`/`height`, which are swapped in 1,487 of the 1,799 test entries); all must be equal
and 854×480 (or 480×854), or the run stops.

**Device and dtype.** All four Endoscapes indexes are computed in **float32 on Apple MPS**
and stored in float16. The Cholec80 indexes of CLIP, SigLIP and MedSigLIP were computed in
float16 on an L4 (Deviations, 2026-10-13); float16 vs float32 on the same 512 frames gave
min cosine 0.99994 / 0.99982 / 0.99997 (all ≥ 0.9998). PeskaVLP is float32 on MPS for both
datasets. `meta.json` records the dtype.

**Uncertainty (4.6).** Paired video-level bootstrap over the 40 Endoscapes test videos,
first 2,000 resamples of `default_rng(20261011)`, descriptive only. Because n = 40 as for
Cholec80, the resample index matrix is **identical** to Cholec80's; here it is mapped to
the Endoscapes test IDs sorted ascending (162–201), and that order is recorded in the run's
`run.json`. No inference spans the two datasets.

**Metrics (4.1), on the 1,799-frame pool.** Headline: per criterion, the mean AP over the
five rephrasings, with prevalence and lift; **CVS mAP** = mean over C1–C3. Secondary:
R-Precision; per-video AP over the videos where the criterion has R > 0 (C1 25, C2 20,
C3 23 of 40; the count is reported); the prompt ensemble as a separate system. Descriptive:
P@10, P@50, distinct videos in the top 50; the calibration check (the 7 negatives against
the three CVS target means, over the 1,799 frames); and the RQ2 quantities (SD and range
over rephrasings per criterion, mean AP by role, and the pairwise order-change fraction,
which is coarse with 3 targets). Zero-shot classification (4.2) and tool mAP do not apply:
Endoscapes has no phase or tool labels, and the criteria are not exclusive classes.

**"Cross-dataset generalisation" (1.2)** means each zero-shot model's CVS mAP on the
Endoscapes test frames, reported descriptively. Section 5 makes no CVS prediction, so no
verdict is given.

**Overlap.** No Endoscapes test video is a Cholec80 or CholecT50 video
(`camma_dataset_overlaps` @ `8347b9f`, README summary; `overlap.py`).

**Test-set numbers seen.** Written before any Endoscapes image was fetched and before any
Endoscapes score on any split existed. Only the label and split files above had been read.
Cholec80 RQ1–RQ5 test numbers existed.

**Affects.** No registered choice changes. Fixes the Endoscapes frame set, relevance
source, preprocessing, device, resample mapping and metric pool; adds one secondary,
not-registered unanimity analysis.
