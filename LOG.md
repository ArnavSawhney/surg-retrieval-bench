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

### 2026-10-13 — Week 4 (started early), Task 0: results vs predictions

`docs/results_vs_predictions.md`: each RQ1/RQ2 prediction against the test run of
13 Oct, every number recomputed from `results/test/` (Arnav asked for the comparison).
RQ1a **supported** (PeskaVLP 0.2775, highest; all three differences Holm-significant),
RQ1b **holds** (CLIP lowest), RQ1c **falsified** (SigLIP 0.2636 above MedSigLIP 0.2157),
RQ1d **falsified** (PeskaVLP 0.2775 [0.2684, 0.2885], below 0.30; its prompt ensemble,
0.3111, is secondary and does not count), RQ2a **holds** (mean SD 0.0444), RQ2b and
RQ2c **falsified** (r1 best at 0.2488, r5 worst at 0.2319). RQ3–RQ5 pending. Calibration:
PeskaVLP 2/7 negatives below every positive target mean, the others 7/7 (reported, not
fixed). The optional image-only encoder can no longer be pre-registered (Parked).

### 2026-10-13 — Week 4, Tasks 1–3: RQ2 figures, linear probe, probe validation (dev)

**Task 1** (`2c94d07`): `scripts/make_rq2_figures.py` builds Fig. 1a/1b (AP per rephrasing,
target and model, with prevalence), Fig. 2 (mean AP by role) and `figures/rq2_table.md`
from the existing `results/test/` CSVs only (no rerun); every recomputed SD and range is
checked against the per-target CSVs. One CVD-checked colour + marker per model
(`src/srb/plotting.py`). Pins matplotlib 3.11.2.
**Task 2** (`0f78684`, `7077afb`): `src/srb/probe.py`, `scripts/eval_probe.py`, 15 tests;
section 4.3 as registered (lbfgs, L2, `max_iter` 5000, C grid, macro-F1 / AP on 33–40,
no class weights, `assert_no_test_leakage` on every fit). 774 tests pass.
**Task 3, dev, not a result** (`results/dev/probe_val_20261013-105258`, clean tree at
`7077afb`): all 48 fits per model converged (max 573 iterations). Chosen phase C: 10
(CLIP, SigLIP, MedSigLIP), 1 (PeskaVLP); 13 of the 28 tool heads chose the grid edge
C = 100 (clipper and irrigator for every model). Validation phase accuracy / macro-F1 /
tool mAP: CLIP 0.698 / 0.592 / 0.675, SigLIP 0.800 / 0.690 / 0.788, MedSigLIP
0.797 / 0.699 / 0.786, PeskaVLP 0.692 / 0.606 / 0.697. 8.8 min wall, two models at a
time, peak RSS 0.69–0.88 GB per model. Weights bit-identical on a rerun. Probes and
phase posteriors of 1–40 in `data/probe/` (1–32 in-sample). Test run waits for Arnav.

### 2026-10-13 — Week 4, Task 4: RQ3 linear probe on 41–80 (RESULT, run once)

Deviations committed after Arnav's OK: float16 clarification (`0365b2d`) and a
secondary, **not registered** tool-head C-grid sensitivity analysis (`d57bac8`; grid
plus {1e3, 1e4}; code `fa89df2`). Extended-grid selection on 33–40 (dev,
`results/dev/probe_sensitivity_val_20261013-185659`) reproduced every registered fit
exactly; 6 of 28 tool heads changed C (2 of the 13 at C = 100). The test code path was
dry-run on 33–40 in a scratch root first. Pushed `c1ed8ed..d57bac8`.
Test run (`6918ba5`; clean tree at `d57bac8`, once; 320 s, peak RSS 0.71–1.22 GB per
model): `results/test/probe_*.csv`, 48 rows appended to `results.csv` (40
`linear-probe`, 8 `linear-probe-sensitivity-not-registered`). Recomputed r1–r5 zero-shot
accuracies equal the registered table (|Δ| ≤ 1e-12). Phase accuracy: CLIP 0.684,
SigLIP 0.734, MedSigLIP 0.738, PeskaVLP 0.671; tool mAP 0.675 / 0.757 / 0.742 / 0.657;
sensitivity tool mAP 0.678 / 0.774 / 0.749 / 0.657. Not compared with the predictions.
`eval_probe.py` re-wrote `results.csv` through pandas, which re-formatted the 190
registered rows' floats; restored byte for byte before committing, and the script now
appends (`a34ed6d`).

### 2026-10-13 — Week 4, Task 5: RQ5 frame pool drawn (before any tagging)

`scripts/make_rq5_pool.py` (committed `0c16ca7`, run on a clean tree there) →
`docs/rq5_frame_pool.csv`: 2,000 frames, 50 per test video, `default_rng(20261008)`,
tagging order `default_rng(20261009)`, numpy 2.4.6. Source: the 98,194 test frames,
manifest SHA-256 `885f81a0…84df0ca` (the same as all four 41–80 indexes). No labels,
scores or predictions in the file.
**Pool SHA-256: `0655cf3ab2ff7bc4d4149a87e976e63fd8705e03c2f8f5c52dd7ca32ec73c291`.**
No frame has been tagged; the tagging UI is Week 8.

### 2026-10-13 — Week 4 close: RQ3 against the predictions

Pushed `d57bac8..dbe09a0`. At Arnav's request, RQ3 added to
`docs/results_vs_predictions.md`, recomputed from `results/test/` (summary, registered
zero-shot tables, saved test posteriors vs labels; all agree): **RQ3a holds**
(PeskaVLP probe accuracy 0.6706), **RQ3b holds** (gain +0.3247), **RQ3c falsified**
(spread shrinks 0.2145 → 0.0678, but PeskaVLP has the lowest probe accuracy). The C-grid
sensitivity analysis is in no verdict. Week 4 done; Week 5 gets its own prompt.

### 2026-10-13 — Week 5 (RQ4, pulled forward from plan Weeks 6–7), Tasks 1–3 and 5

**Task 1** (`eebebfe`): clarification of section 4.4 under Deviations, Arnav's answers,
committed before any RQ4 number on any split (MA window ends, HMM prior/floor/forward
pass, TCN architecture, per-seed (depth, epoch) selection, CPU determinism, edit score,
Jaccard, intervals). No protocol change.
**Task 2** (`c423454`): `src/srb/temporal.py`, `scripts/eval_temporal.py`, 27 tests
(hand-checked 3-state HMM, causality incl. the TCN receptive field, edit/Jaccard hand
cases, leakage guards, end-to-end run on a synthetic root). 807 tests pass.
**Task 3, dev, not a result** (`results/dev/temporal_val_20261013-203339`, clean tree at
`c423454`): single frame reproduces the probe's validation accuracy
exactly for all four models; HMM floored 0 posteriors. Chosen w (causal / centred):
CLIP 30 / 60, SigLIP 20 / 30, MedSigLIP 20 / 60, PeskaVLP 30 / 60 s. TCN (depth, epoch)
per seed 0/1/2: CLIP (3,48) (4,49) (4,43); SigLIP (4,50) (4,50) (3,42); MedSigLIP
(4,46) (4,46) (4,45); PeskaVLP (4,44) (4,36) (4,30); 10 of 12 at the largest depth, 2
at the last epoch. Validation accuracy gain over single frame, mean over backbones:
MA causal +0.085, HMM forward +0.044, TCN +0.034 (MA centred +0.124, Viterbi +0.060).
54 min wall (two models at a time, 2 threads each), peak RSS ≤ 0.87 GB; a retrained
TCN (CLIP, seed 0) is bit-identical to the saved one. Test run waits for Arnav.
**Task 5** (`c75e985`): `scripts/tag_rq5.py`, a local image-only tagging page in pool
order with per-frame saving and stopping-rule progress; shown on 5 training-video frames
(`--demo`). No pool frame tagged.

### 2026-10-13 — Week 5, Task 4: RQ4 on 41–80 (RESULT, run once) + TCN sensitivity

At Arnav's request, a secondary TCN sensitivity analysis, **not registered** (Deviations,
`3bbe61f`; code `d63c5ec`): depths 2–8, up to 100 epochs, all else as registered.
Extended selection on 33–40 (dev, `results/dev/tcn_sensitivity_val_20261013-213707`, at
`d63c5ec`, 87 min): all 1,800 registered curve points (depths 2–4, epochs 1–50)
reproduced exactly; chosen depths 5–7 (receptive fields 63–255 s). Commits
`eebebfe`..`3bbe61f` pushed before the test run (last push received 2026-10-13T17:34:26Z).
Test run once on a clean tree at `3bbe61f` (25 s): single frame reproduces the RQ3 probe
accuracies exactly; HMM floored 0 posteriors; 472 rows appended to `results.csv`
(prefix byte-identical), tables in `results/test/temporal_*.csv`. Accuracy gain over
single frame, mean over the four backbones: MA causal +0.074 [0.066, 0.081], HMM forward
+0.049 [0.032, 0.065], TCN +0.023 [0.009, 0.037] (sensitivity TCN +0.041 [0.025, 0.058];
MA centred +0.102, Viterbi +0.061). Not compared with the predictions.

### 2026-10-15 — Week 5 close: RQ4 against the predictions

Pushed `3bbe61f..4f489a7` (RQ4 test results). At Arnav's request, RQ4 added to
`docs/results_vs_predictions.md` (`9303cff`), recomputed from the saved test posteriors,
HMM parameters, TCN weights and selected windows on the same 2,000 resamples (all agree
with `results/test/temporal_*.csv`). Registered methods only: **RQ4a holds** (best causal
method = causal moving average, mean gain +0.0739 [0.0661, 0.0812], inside 3–8 points),
**RQ4b falsified** (moving average +0.0739 > HMM forward +0.0487 > TCN +0.0225),
**RQ4c falsified** on point estimates (gain 0.0739 ≥ single-frame spread 0.0678), but
narrowly: paired gain − spread +0.0062 [−0.0101, +0.0172], so the two are about equal.
The TCN sensitivity analysis and the offline methods are in no verdict. Week 5 done.

### 2026-10-18 — RQ5 tagging: guide committed after 436 frames; out-of-body issue

Frames 0–435 of the RQ5 pool (`docs/rq5_frame_pool.csv`) were tagged by Arnav between
15 Oct 23:45 IST (first tag) and 18 Oct 20:00 IST. Raw tags `data/rq5/tags.csv` (436
frames, not tracked) and the backup `data/rq5/tags_raw_backup_20261018_436frames.csv`
both have SHA-256 `750483857807963d270812ab1696242186d6e105004772d83ec5070f50d732a6`;
neither file is ever modified.

The tagging guide `docs/rq5_tagging_guide.md` was written on 15 Oct (file saved 23:38
IST), before the first tag, but **committed only on 18 Oct, after 436 frames**
(`091a67a`). The committed version adds the out-of-body rule and its change log.

**Out-of-body issue.** Frames with the camera outside the patient (operating room, hands,
drapes, black screen) had been tagged **occlusion**, although the pre-registration's
occlusion is instrument occlusion inside the abdomen. Handled by an image-only review
pass over frames 0–435 that records only "out of body: yes/no" in a separate file
(`data/rq5/out_of_body.csv`); corrected tags are derived from raw tags + review into
`data/rq5/tags_corrected.csv`. The raw tags are untouched. No per-tag metric had been
computed and no model output had been seen. Deviations entry to follow.

### 2026-10-18 — RQ5: out-of-body review finished; corrected tags built

Deviations entry committed (`84b7ec7`, approved as written); guide change log updated
(`f33a351`: the correction also zeroes smoke, glare and blood, superseding the earlier
wording). Tool changes: `ece1f22`.

Review pass (`tag_rq5.py --review-out-of-body`, image only): 436 of 436 frames
answered, 18 Oct 20:19–23:04 IST, **finished before tagging resumed** (last raw tag
20:00 IST; the tool refuses to tag until the review is complete). **7 out of body**,
tag_order (video, 25 fps frame index): 12 (51, 3575), 225 (50, 26875), 235 (43, 10575),
272 (78, 15725), 356 (77, 1250), 405 (57, 59300), 423 (78, 4550). Raw tags of these:
occlusion 6, blur 2, no smoke, glare or blood.

`scripts/rq5_apply_corrections.py` (not tracked: `data/rq5/`):

* `out_of_body.csv` SHA-256 `f5e96207afc1cbafebe3920f868b00313a0cc07ad7397bf7606660d16276233f`
* `tags_corrected.csv` SHA-256 `3bb263882a784aab4d7706d942b4637f05ac3db98370bc6f23fb3c2c278d2a33`
* raw `tags.csv` unchanged (`750483…d732a6`).

Tag counts, raw → corrected (436 frames): smoke 25 → 25, blur 111 → 111, glare 35 → 35,
blood 35 → 35, occlusion 69 → 63, clean 235 → 240. Stopping rule on the corrected tags:
not met (smoke 25, glare 35 and blood 35 are below 50). Counts only; no per-tag metric
computed.

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
* Classification, probe and temporal metrics (accuracy, macro-F1, tool mAP, Jaccard,
  edit score) — added to `metrics.py` when the experiments that need them arrive.
* Per-experiment YAMLs in `configs/` — when there is an experiment to configure.
* CholecT50 action-triplet queries; Streamlit demo (Week 9); LoRA/adapter (Week 7 if
  time allows).
* **Optional image-only encoder (EndoFM / SurgeNet-style), 2026-10-13.** Section 6 of the
  pre-registration says its pin is valid only if dated before the first test-set number
  of any model; RQ1 test numbers exist since 13 Oct (`c1ed8ed`), so it can no longer be
  pre-registered. If it is ever run, it is **exploratory** and labelled so in every
  table, figure and the text. Not run in Week 4.
