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

## Open decisions

Recorded, deliberately **not** implemented yet.

1. **Near-duplicate frames at 1 fps.** Consecutive 1 fps frames from one video are
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
