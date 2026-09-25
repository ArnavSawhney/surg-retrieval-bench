# Log

One line per session: date, what was done, what's next. Plus running lists of open
decisions and parked ideas.

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
3. **Label frame-rate alignment.** Phases are annotated at 25 fps, tools at 1 fps.
   Needs an explicit frame-index alignment test in the manifest-integrity suite before
   any result is computed.
4. **Week 2 is overloaded** (4 models + full extraction + pre-registration).
   **Pre-registration has priority**; model count is the thing that gets cut.
5. **Tie-breaking convention in the ranking** *(found while writing `metrics.py`)*.
   Ties are currently broken by ascending frame index — deterministic and
   reproducible, but not tie-neutral: with equal scores, AP depends on where the
   relevant frames sit in the index, which tracks video order. sklearn's
   `average_precision_score` instead averages over tied groups (the divergence is
   asserted in `tests/test_metrics.py`). Float32 cosine similarities tie rarely, so
   the cheap first step is for `eval_retrieval.py` to **count exact ties** and report
   them; switch conventions only if the count is non-trivial. Decide before
   pre-registration.

## Parked

Outside Session 1's scope. Not started.

* `scripts/extract_frames.py`, `scripts/build_index.py`, `scripts/eval_retrieval.py`,
  `scripts/eval_probe.py`, `scripts/eval_temporal.py` — Week 1 onwards.
* `src/srb/datasets/cholec80.py`, `src/srb/datasets/endoscapes.py` — Week 1.
* `src/srb/temporal.py` (moving average, HMM, causal TCN) — Weeks 6–7.
* SigLIP and MedSigLIP registry entries — Week 2. Specs are stubbed as comments in
  `registry.py`; MedSigLIP is gated and needs the HAI-DEF terms accepted plus
  `HF_TOKEN`.
* PeskaVLP / SurgVLP — Week 2, in a **separate environment** (mmengine and OpenAI-CLIP
  pins conflict with `transformers`). Timebox the install to one day.
* Classification, probe and temporal metrics (accuracy, macro-F1, tool mAP, Jaccard,
  edit score) — added to `metrics.py` when the experiments that need them arrive.
* Per-experiment YAMLs in `configs/` — when there is an experiment to configure.
* CholecT50 action-triplet queries; Streamlit demo (Week 9); LoRA/adapter (Week 7 if
  time allows).
