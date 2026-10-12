# Log

One line per session: date, what was done, what's next. Plus running lists of open
decisions and parked ideas. `PROGRESS.md` holds the detailed current state (environment
facts, what is verified, what is next); this file is the chronological record.

## Sessions

### 2026-09-25 — Session 1: repo scaffolding, retrieval metrics, CLIP registry

Repo initialised (MIT, public) with the Section 7 layout; retrieval metric set revised
to AP/mAP + R-Precision with a random-ranker baseline (`src/srb/metrics.py`,
`docs/metrics.md`, 59 tests); CLIP ViT-L/14 wired into `src/srb/models/registry.py`
behind a uniform `encode_image`/`encode_text` interface and verified end to end on MPS
(`scripts/smoke_clip.py`: 10 synthetic images and 3 texts → `(10, 768)` and `(3, 768)`
float32 unit-norm embeddings, bit-identical on re-encode); resumable download scripts
for Cholec80 and Endoscapes2023 with a live archive-size disk check. 82 tests pass.
**No dataset downloaded, no model evaluated, no result produced.**

**Next (Week 1):** start the dataset downloads on a machine with disk, write
`scripts/extract_frames.py` (1 fps, 448 px short side, JPEG q90) and the Cholec80
label reader, extract videos 1–5, then add SigLIP to the registry and build a toy
retrieval index over those 5 videos.

### 2026-09-26 to 09-29 — Week 1: metrics, overlap guard, SigLIP, frames, index

Tie-neutral metrics (`f831818`, closes decision #5); cross-dataset overlap guard
(`1c92ee9`, `77df245`); SigLIP so400m in the registry (`88d35b1`); Cholec80 frame
extraction with frame-index label alignment for videos 1–5, 14,266 frames, 1.11 GB
(`ed2e57c`, closes decision #3); frame-embedding index + dev retrieval eval, with a
Colab path because SigLIP swaps on the 8 GB Mac (`dd065b1`). CLIP and SigLIP indexes
for videos 1–5 built on a T4 after a Mac-vs-GPU check (min cosine 1.00000). Dev eval
on training videos 1–5: every AP within 0.84–1.33× its prevalence, negative control
excluded and scored lowest. 490 tests pass. **Dev checks only; no result produced.**

**Next (Week 2):** write and commit `docs/preregistration.md` first (decides open
decision #1, near-duplicate frames); then more backbones as time allows (#4).

### 2026-10-08 — Week 2 starts late: slip logged, test scoring locked

**Slip:** nothing happened between the Week 1 commit (29 Sep) and 8 Oct, so Week 2
(scheduled 3–9 Oct) starts about a week late. Buffer rule: cut scope from the bottom
(LoRA, then CholecT50, then the demo); the paper date does not move. Within Week 2 the
pre-registration comes first and model count is what gets cut (decision #4).
Test-split scoring now has its own lock, independent of the fetch guard (`66faa00`).

### 2026-10-09 — Videos 6–40 extracted; PeskaVLP runs

**Extraction:** `scripts/fetch_extract.sh 6 40 32` ran overnight (21:53–08:48), one
video at a time, every fetch on its first attempt despite connection resets. The
manifest now holds videos 1–40: 86,304 frames (71,000 train, 15,304 val), all
854×480, frame files = manifest rows for every video, `tests/test_manifest.py` passes.
Videos 15 and 37 have one more phase row than video frames (−1); the 1 fps sample at
that end is the trailing one already dropped for lacking a tool row, so no used label
points outside the video.

**PeskaVLP** works, inside the 1-day timebox. Separate `.venv-surgvlp` (Python 3.11,
torch 2.5.1, transformers 4.30.2, mmengine 0.7.0; `envs/surgvlp-requirements.txt`)
behind `scripts/peskavlp_encode.py`; the main env reaches its text encoder through
`get_backbone("peskavlp")` (a subprocess), and its frame index is written in the same
format as `build_index.py`. Pins: SurgVLP @ `85e85899`, checkpoint by SHA-256 (the
Seafile URL is unversioned), Bio_ClinicalBERT @ `d5892b39`. Smoke on MPS: 768-d,
unit norm, bit-identical re-encode, tokens identical to `surgvlp.tokenize`, checkpoint
loads with no missing or unexpected keys (`weights_only=True`). Index of videos 1–5:
35.3 frames/s on MPS. Dev check on 1–5 with the 4 dev queries ran (results/dev/, not a
result). Contamination row filled from both papers: **unverified**, since SVL is mostly
WebSurg (IRCAD Strasbourg) lectures and Cholec80 was recorded in Strasbourg.

### 2026-10-12 — Pre-registration committed: `prereg-v1` = `ff2b6e5`

`docs/preregistration.md`, `src/srb/queries.py` and `tests/test_queries.py` committed
as `ff2b6e5c553252749792ce72c05025b10de8bda9`, annotated tag `prereg-v1`. The query set
is 85 target strings (7 phases, 7 tools, 3 CVS criteria, 5 rephrasings each) plus 7
negative controls. Predictions for RQ1–RQ5 are Arnav's; the RQ3 gap and gain
predictions were revised before the commit after reading the PeskaVLP (arXiv:2410.00263)
and TeCNO (arXiv:2003.10751) Cholec80 results, as section 5 records. At the commit no
test video was fetched, embedded or scored; embeddings exist only for videos 1–5.
Closes decision #1 (all-frame AP primary; per-video AP secondary; distinct videos in the
top 50 descriptive).

**Next (Week 3, ~4 days late):** rebuild the 6–40 Colab bundle on the clean hash; index
videos 6–40 with CLIP, SigLIP and MedSigLIP on a GPU and PeskaVLP on the Mac; lift the
fetch/extract/index guard for 41–80 in its own commit citing `ff2b6e5`.

### 2026-10-12 — Week 3, Task 0: `prereg-v1` on GitHub

Pushed `main` and `prereg-v1`. On GitHub the annotated tag (object `3be99b5`) peels to
`ff2b6e5`. Timestamps from the GitHub API: tagger date 2026-10-12T13:58:57Z (set by the
client); the push that carried `ff2b6e5` and the tag was received at
**2026-10-12T14:02:13Z** (repository activity API). `pytest`: 516 passed, none skipped,
including `test_no_query_is_truncated_by_any_text_encoder` for all four tokenizers (its
first run; no query exceeds any limit, so no deviation). Private-file check empty.

### 2026-10-12 — Week 3, Tasks 1–4: bundles, 1–40 indexes, 41–80 unlocked, registered eval

**Task 1.** Bundles rebuilt on a clean tree (`db5ed24`): 1–5 (1.0 GB) and 6–40 (4.6 GB);
the old 29 Sep `srb_bundle.tar` deleted with Arnav's approval. README hypothesis line
made neutral (`db5ed24`; the registered predictions are in the pre-registration).

**Task 2.** All four models indexed over videos 1–40 (86,304 frames). CLIP, SigLIP and
MedSigLIP on a Jarvis Labs L4 in **float16** (Arnav's choice after a bench: float32 on
the L4 was only 2.3× a T4; float16 3.3–3.6× faster at min cosine ≥ 0.9998 vs float32 on
the same frames). The prereg pins revisions, not the compute dtype, so not a deviation.
Cross-device check CLIP video04 L4 float16 vs Mac float32: min cosine 0.99977. Full runs:
101.75 / 34.98 / 25.58 frames/s; tars SHA-256-checked, reported cached locally. L4 time
2 h 24 min, **₹101.40**, instance destroyed. PeskaVLP 6–40 on the Mac (34.0 frames/s),
merged with 1–5 by `scripts/merge_index.py` (`6fde54d`). Procedure: `scripts/gpu_index.sh`.

**Task 3.** Fetch/extract/index guard on 41–80 lifted (`b09781b`, cites `ff2b6e5`); the
scoring lock stays. The first run stopped at video 41: the label files of 1–40 had been
fetched by hand, so `fetch_cholec80_videos.py` now fetches them too (`509e346`).
Videos 41–61 fetched and extracted by 22:20 IST, every one CRC OK, 854×480,
`video_minus_phase_rows = 0`, frames = labelled samples − 1 trailing; still running.

**Task 4.** Registered evaluation (`60a352f`, parallel `369ce40`): 70 target strings +
7 negatives, headline = mean AP over rephrasings, R-Precision, per-video AP, ensemble,
P@10/P@50, distinct videos in the top 50, calibration, zero-shot phase classification,
paired video bootstrap (seed 20261011, B = 10,000, first 2,000 descriptive, Holm). The
fast bootstrap equals the materialised resample exactly in tests. **Validation run on
33–40, all four models: dev, not a result** (`results/dev/registered_val_20261012-220609`,
~18 min per model, four in parallel). Nothing looks broken: no NaN AP, every target in
6–8 of the 8 videos, ties present in every query and handled. Two behaviours, reported
not fixed: CLIP, SigLIP and MedSigLIP's zero-shot phase classifier puts most frames on
one prompt per role (up to 99%); PeskaVLP's negatives are not all below its lowest
positive target mean (2/7 are; its cosines sit on a narrow scale). No protocol change.

**Next:** finish 41–80 (manifest test on 1–80), bundle 41–80, PeskaVLP 41–80 on the
Mac, CLIP/SigLIP/MedSigLIP 41–80 on an L4 (~₹85); Task 5 only after Arnav says go.

### 2026-10-13 — Week 3, Task 3 done: 41–80 indexed by all four models

All 80 videos fetched and extracted (184,498 frames; `docs/datasets.md`, `69ffa40`).
`test_manifest.py`'s pre-prereg ban on test videos replaced by a split-label check
(`f292f62`). PeskaVLP 41–80 on the Mac (28.94 frames/s). CLIP / SigLIP / MedSigLIP on a
second L4 in float16: 90.07 / 34.13 / 24.89 frames/s, cross-check 0.99977, tars
SHA-256-checked and reported cached. First launch stopped at the cross-check (no
`frames/video04` in the 41–80 bundle); relaunched after copying it. The ~₹85 estimate
was low: the upload (6.1 GB, one drop) and the failed launch added about an hour.
**₹128.67**; Week 3 GPU total ₹230.07. Instance destroyed.

### 2026-10-13 — Week 3, Task 5: the registered test run (RESULT)

Arnav said go. Scoring lock lifted in its own commit (`1759cb4`, cites `ff2b6e5`);
`eval_retrieval.py` still refuses test videos with the dev queries. `eval_registered.py
--split test` run **once**, clean tree at `1759cb4`, indexes from `4aed6ba`, seed
20261011, B = 10,000 (first 2,000 for descriptive intervals), four models in parallel,
~63 min each. Outputs: `results/results.csv` (190 rows, `split=test`) and the full tables
in `results/test/` (`run.json` has the hashes). No NaN anywhere; 759 tests pass.

| model | overall mAP [95% CI] | lift | phase mAP | tool mAP |
|---|---|---|---|---|
| CLIP ViT-L/14 | 0.2026 [0.1966, 0.2117] | 1.28 | 0.1757 | 0.2296 |
| SigLIP so400m-384 | 0.2636 [0.2541, 0.2740] | 1.66 | 0.2167 | 0.3105 |
| MedSigLIP-448 | 0.2157 [0.2094, 0.2245] | 1.36 | 0.1822 | 0.2492 |
| PeskaVLP | 0.2775 [0.2684, 0.2885] | 1.75 | 0.2689 | 0.2861 |

Overall prevalence 0.1585. All six confirmatory pairwise differences in overall mAP
reject at Holm 0.05 (largest Holm-adjusted p 0.0024). Intervals, the pairs table and the
RQ2 order-change fractions are in `results/results.csv` and `results/test/`. Not compared
with the registered predictions (not asked for yet). No rerun; no deviation.

## Open decisions

Recorded, deliberately **not** implemented yet.

1. ~~**Near-duplicate frames at 1 fps.**~~ **Closed 2026-10-12 by `prereg-v1`** (section 4.1:
   all-frame AP primary, per-video AP secondary, distinct videos in the top 50
   descriptive). Consecutive 1 fps frames from one video are
   near-identical, so the top-k of a query can be one moment repeated. Add a per-video
   diversity metric (distinct videos in the top k) or cap frames per video in the
   ranking? Either changes what AP means, so **decide before pre-registration (Week 2)**.
2. **Pretraining-data contamination per backbone.** SurgVLP-family and
   EndoFM/SurgeNet-style encoders may have pretrained on video overlapping Cholec80
   test videos 41–80 (and CholecT50 overlaps Cholec80 by construction). Audit each
   backbone's pretraining corpus and write it up in `docs/datasets.md` before
   interpreting any zero-shot win.
   *Half closed 2026-09-26:* dataset-level overlap is mapped and enforced
   (`src/srb/datasets/overlap.py`, CAMMA `camma_dataset_overlaps` @ `8347b9f`).
   Still open: per-backbone pretraining corpora (every row in `docs/datasets.md` is
   currently "unverified"), CholecSeg8k, Cholec80-CVS, CholecT45.

3. ~~**Label frame-rate alignment.**~~ **Closed 2026-09-27.** Labels are joined on the
   25 fps frame index; frames are selected by index (0, 25, 50, …), not timestamp. On
   videos 1–5 every tool index is in the phase labels, each video drops exactly one
   trailing sample (phase but no tool row, counted), and `video_frames − phase_rows = 0`
   for all five. `tests/test_cholec80.py` covers the rules; `tests/test_manifest.py`
   passes 7/7 on the 14,266-row manifest. Details in `docs/datasets.md` ("Label frame
   rates and alignment").
4. **Week 2 is overloaded** (4 models + full extraction + pre-registration).
   **Pre-registration has priority**; model count is the thing that gets cut.
5. ~~**Tie-breaking convention in the ranking.**~~ **Closed 2026-09-26.** Every metric
   is now the expected value under a uniformly random order within each tied group
   (McSherry & Najork, ECIR 2008); float16 score arrays raise; `count_ties` is reported
   per query. The index tie-break was not tie-neutral, and ties are not rare: in a
   simulation, fp16-computed similarities left 96.7% of 98.5k frames tied. Details in
   `docs/metrics.md` ("Ties").

## Parked

Outside Session 1's scope. Not started.

* `scripts/extract_frames.py`, `scripts/build_index.py`, `scripts/eval_retrieval.py`,
  `scripts/eval_probe.py`, `scripts/eval_temporal.py` — Week 1 onwards.
* `src/srb/datasets/cholec80.py`, `src/srb/datasets/endoscapes.py` — Week 1.
* `src/srb/temporal.py` (moving average, HMM, causal TCN) — Weeks 6–7.
* Classification, probe and temporal metrics (accuracy, macro-F1, tool mAP, Jaccard,
  edit score) — added to `metrics.py` when the experiments that need them arrive.
* Per-experiment YAMLs in `configs/` — when there is an experiment to configure.
* CholecT50 action-triplet queries; Streamlit demo (Week 9); LoRA/adapter (Week 7 if
  time allows).
