# Progress and session handoff

Current state of the project, written so a new session can resume without re-deriving
anything. `LOG.md` stays the one-line-per-session record required by the plan; this
file holds the detail: environment facts, what is verified, what is next, and the
traps already hit.

**Last updated:** 19 Oct 2026 (RQ5 tagging complete; RQ5 analysis not run).
**`prereg-v1` = `ff2b6e5`**, on GitHub since 2026-10-12T14:02:13Z.
**Status:** RQ1/RQ2 (`c1ed8ed`), RQ3 (`6918ba5`) and RQ4 (`4f489a7`) run once on the test split
and compared with the predictions in `docs/results_vs_predictions.md`. Week 5: section 4.4 clarified
under Deviations (`eebebfe`), temporal models implemented (`c423454`), **validation run
done (dev)**; TCN sensitivity analysis (not registered, `3bbe61f`); **RQ4 test run done
once** at `3bbe61f`; RQ4 verdicts in `9303cff` (RQ4a holds, RQ4b and RQ4c falsified). RQ5 tagging complete: 816 frames,
stopping rule met, tags frozen (LOG.md, 19 Oct; out-of-body Deviation `84b7ec7`). No GPU spend since Week 3.

Start a new session by reading the local, git-excluded planning notes (see `.git/info/exclude`)
— they are the source of truth for scope and hard rules, and they are private: never
commit them or copy their content into a tracked file.

---

## 1. Done and verified (Session 1, 25 Sep 2026)

Commit `d5b9283`, pushed to <https://github.com/ArnavSawhney/surg-retrieval-bench>
(public, 24 tracked files).

| Deliverable | State | Evidence |
|---|---|---|
| Repo, MIT licence, Section 7 layout | done | 24 files tracked; `data/` gitignored |
| Privacy: private notes out of git | verified | grepped `git ls-files` for the gitignored planning notes: empty, locally **and** on the GitHub tree via the API |
| Commit identity | verified | `ArnavSawhney <arnav.sawhney.ug23@plaksha.edu.in>`, set per-repo |
| `README.md` (aims, RQ1–RQ5, licences, no-results notice) | done | research content only |
| `docs/metrics.md`, `docs/datasets.md` | done | public, with citations |
| `docs/preregistration.md` | **placeholder only** | states outright that it is unwritten |
| `src/srb/metrics.py` + tests | done | 59 tests |
| `src/srb/models/registry.py` (CLIP only) | done | 23 tests |
| `scripts/smoke_clip.py` on MPS | verified | `(10, 768)` / `(3, 768)` float32, unit norm, `max|delta| = 0.00e+00` on re-encode |
| Download scripts | written, **not run** | disk check verified on both the abort and pass paths |
| `pytest` | **82 passed in 3.72s** | run with no `PYTHONPATH` after editable install |

### Metric set (decided 25 Sep 2026, supersedes the plan's Section 6)

Primary **AP per query → mAP** (macro), always printed next to the random-ranker
baseline (prevalence `R/N`) and the lift. Secondary **R-Precision**. **P@10/P@50
descriptive only.** `hit_at_k` kept for debugging. **Recall@k and nDCG@10 removed** —
with thousands of relevant frames per query, Recall@10 is capped near 0.0003, random
Hit@10 is ≈ 0.99, and P@10/nDCG@10 saturate at 1.0 for weak, strong and degenerate
rankers alike. Reasoning and citations: `docs/metrics.md`. Runnable evidence:
`python docs/metric_rationale.py`.

API: `average_precision`, `r_precision`, `precision_at_k`, `hit_at_k`,
`random_baseline_ap`, `mean_ap`, `rank_order`. Pure NumPy, no I/O. `R = 0` returns
`nan` + `RuntimeWarning`, never 0.0. Negative/control queries are excluded from mAP.

---

## 1a. Week 1 (26 Sep 2026): done so far

| Task | State | Commit / evidence |
|---|---|---|
| 0. Housekeeping | done | `8f101f7`: `results/dev/` and `configs/local.yaml` ignored; private-file patterns moved from `.gitignore` to `.git/info/exclude` (they remain in the *history* of `.gitignore`) |
| 1. Tie-neutral metrics (closes decision #5) | done | `f831818`: expectation over within-group orderings (McSherry & Najork 2008); fp16 scores raise; `count_ties`; AP = 49/72 hand case; exact brute-force check on 300 cases; 98.5k frames in ~20 ms |
| 2. Overlap guard (half of decision #2) | done | `1c92ee9`: `src/srb/datasets/overlap.py`, CAMMA `camma_dataset_overlaps` @ `8347b9f`, re-run of their script matches the README; CholecT50 covers 22/40 of our test videos; Endoscapes 1,2,3,4,7,121 forbidden; backbone table all "unverified" |
| 4. SigLIP | done | `88d35b1`: revision `9fdffc58…`, `padding="max_length"`, `max_length=64`; smoke on MPS `(10,1152)`/`(3,1152)`, unit norm, bit-identical re-encode; adds `sentencepiece`, `protobuf` pins |
| 3. Frames + labels + manifest | **code done, uncommitted**; verified on video04 | `zipsource.py`, `cholec80.py`, `frames.py`, `check_zip_integrity.py`, `fetch_cholec80_videos.py`, `extract_frames.py` + tests |
| 5. build_index + eval_retrieval | **code done, uncommitted**; verified on video04 (CLIP only) | `index.py`, `retrieval.py`, `build_index.py`, `eval_retrieval.py` + tests with a fake backbone |

`pytest`: **482 passed, 7 skipped** with no manifest; with the video04 manifest
present, `tests/test_manifest.py` runs and passes **7/7**.

**video04 end-to-end check (26 Sep, 23:08–23:28):**
* `extract_frames.py --videos 4 --keep-video`: 1,522 frames written (= the labelled
  count above), 1 trailing sample dropped, `video_minus_phase_rows = 0`, 854×480
  source, 87.6 MB of JPEGs, 31.5 s. The mp4 was kept (`--keep-video`) in case of a
  re-run; it is still in `data/tmp/`. The manifest currently holds **video 4 only**.
* `build_index.py --model clip-vit-l14 --videos 4 --bench` →
  `data/index/clip-vit-l14/videos_4-4/`: 1522 × 768, stored float16, computed
  float32 on MPS, batch 8, **3.33 frames/s** (457 s); bench 3.5 / 3.08 / 2.78 / 2.63
  frames/s for batch 8 / 16 / 32 / 64. See trap 11.
* `eval_retrieval.py --model clip-vit-l14 --videos 4` →
  `results/dev/clip-vit-l14_videos_4-4_20260926-232823.csv` (gitignored, confirmed).
  **Dev, video04 only, not a result:** phase_calot AP 0.507 vs prevalence 0.399
  (lift 1.27); phase_clipping 0.055 vs 0.043 (1.26); tool_hook 0.435 vs 0.482
  (0.90, below random); neg_car has R = 0, is excluded from mAP, and scores far below
  the real queries (mean 0.04 vs 0.18–0.27). Ties: 2 frames in 1 group
  (phase_calot), none elsewhere. Nothing looks broken.
* Cosmetic: the eval banner always says "videos 1-5" even when run on one video.
  Fix before committing Task 5.
* CPU vs MPS on 64 video04 frames, batch 8: CPU (4 threads) 2.87 frames/s, MPS
  3.64 frames/s. Both match the saved index to max |Δ| 1.2e-4, min cosine
  0.99991. The Δ comes from the index's float16 storage, not from the device.

**Extraction, videos 1, 2, 4, 5 (26 Sep, 23:45):** `extract_frames.py --videos 1-5`
(video03 skipped, not fetched yet). 1,733 / 2,839 / 1,522 / 2,344 frames written, each
equal to the labelled count; `video_minus_phase_rows = 0` for all four; 1 trailing
sample dropped per video; all 854×480; 29–58 s per video; 541 MB of JPEGs in total. The
mp4s are deleted (including video04's). Manifest has 8,438 rows for videos [1, 2, 4, 5];
`tests/test_manifest.py` passes 7/7. When video03 lands, run
`extract_frames.py --videos 3`: it adds video03 and rebuilds the full manifest.

**Speed with VS Code, Chrome and Spotify closed (26–27 Sep, ~23:50–00:20):**
* CLIP re-check, `build_index.py --model clip-vit-l14 --videos 4 --bench --force`:
  bench 3.61 / 3.81 / 3.71 / 3.57 frames/s for batch 8 / 16 / 32 / 64; chose 16;
  full video04 index at **5.28 frames/s** (288 s), vs 3.33 with the apps open. That
  makes ~45 min for videos 1–5. The video04 CLIP index was rewritten by this run.
* SigLIP so400m-384 on video04 (bench): **1.50 / 1.41 / 0.79 frames/s** for batch 8 /
  16 / 32, with free memory at 17% / 9% / 22%. Batch 32 was already swapping, so I
  stopped the run before batch 64. No SigLIP index was written. At batch 8, videos 1–5
  would take ~2.6 h locally. After SIGKILL the process sat in the exiting state (`E`)
  for minutes while it released MPS memory; do not start another model until it is
  gone.

**Colab path for SigLIP (27 Sep, written, not yet run):** `scripts/make_colab_bundle.sh`
packs code + frames + manifest + a `GIT_HASH` file + the Mac CLIP video04 index into
`data/colab/srb_bundle.tar` (~530 MB for 4 videos; frames are licence-restricted, so keep
it in your own Drive). `scripts/colab_index.sh <model> <videos>` runs on a Colab GPU. It
installs the exact pins without `pip install -e` (Colab's Python is not 3.11), rebuilds
CLIP on video04 and requires min cosine ≥ 0.999 against the Mac index, builds the
requested index with `--dtype float32` (the registry would default to fp16 on cuda), checks
the norms, and writes `index_<model>_<tag>.tar` to Drive. Untar that at the repo root and it
is picked up as cached. Supporting changes: `build_index.py --dtype`, meta now records
python/torch/gpu, and `git_hash` falls back to `GIT_HASH` outside git (+1 test). `pytest`:
**490 passed**. Re-make the bundle after video03 is extracted.

**video03 + full manifest (27 Sep):** video03 finished at 02:13 (CRC OK, 10,388 s).
`extract_frames.py --videos 3`: 5,828 frames, `video_minus_phase_rows = 0`, 170.5 s,
567 MB. Manifest: **14,266 rows, videos 1–5**; `tests/test_manifest.py` 7/7.

**Indexes for videos 1–5, built on Colab (29 Sep)** with `colab_index.sh`, Tesla T4, clean
uv venv (Python 3.11.16, torch 2.14.0+cu130, transformers 5.17.0), `--dtype float32`:
* Cross-device check (run before each model): CLIP video04 Colab vs Mac, max |Δ| 2.44e-4,
  **min cosine 1.00000**.
* `clip-vit-l14/videos_1-5`: 14,266 × 768, 17.96 frames/s (bench 16.9–18.3, batch 64).
* `siglip-so400m-384/videos_1-5`: 14,266 × 1152, **4.7 frames/s** (bench 4.5–4.9,
  batch 8), 3,032 s. The T4 speed barely changes with batch size, for both models.
* Downloaded from Drive and untarred at the repo root; `build_index.py` reports both as
  **cached** (same revision and manifest hash), so the Colab manifest = the Mac one.

**Dev eval, videos 1–5 (29 Sep, 18:05) — dev, not a result.** Text is encoded locally on
MPS in fp32; the image index comes from the T4 in fp32.

| query | R | prevalence | CLIP AP (lift) | SigLIP AP (lift) | ties CLIP / SigLIP (frames, groups) |
|---|---|---|---|---|---|
| phase_calot | 7,391 | 0.518 | 0.604 (1.17) | 0.628 (1.21) | 128/63, 54/27 |
| phase_clipping | 1,027 | 0.072 | 0.076 (1.05) | 0.096 (1.33) | 140/70, 70/35 |
| tool_hook | 7,372 | 0.517 | 0.433 (**0.84**, below random) | 0.539 (1.04) | 74/37, 30/15 |
| neg_car | 0 | — | excluded | excluded | 26/13, 26/13 |

Nothing looks broken: every AP is near its prevalence, neg_car is excluded (R = 0) and
scores far below the real queries for both models (mean 0.05 vs 0.18–0.27 for CLIP; −0.05
vs 0.09–0.17 for SigLIP). The near-equal R of phase_calot and tool_hook was checked: they
overlap on 5,243 frames, hook also covers 2,128 GallbladderDissection frames, so it is a
coincidence, not a label bug. The ties are all pairs (groups of 2), consistent with
near-duplicate 1 fps frames giving identical fp16 embeddings (open decision #1); the
tie-neutral metrics handle them. CSVs: `results/dev/*_videos_1-5_20260929-18*.csv`.
The eval banner now names the actual videos and split (`dev_label`), e.g.
"DEV, video 4 (train split), NOT A RESULT".

**Zip integrity (read from the server, central directory only):**
* Cholec80: 74,916,858,781 bytes = expected; 80 videos, 80 phase, 80 tool, 80
  timestamp files + README.txt. Videos 1–5 = 0.55–4.09 GB each, 7.33 GB total.
* Endoscapes: 6,285,499,749 bytes = expected; 159,985 members; train/val/test JPEGs
  36,694 + 12,372 + 9,747 = 58,813 (matches the paper). `all/` holds 58,585 (228
  fewer; not investigated, Week 8).
* The copies in `~/Downloads` are **truncated** (7.6 MB and 7.0 MB) and fail the check.
  `configs/local.yaml` (gitignored) now points both datasets at the official URLs.

**Labels, videos 1–5 (verified):** tool indices are all in the phase labels; each
video has exactly one trailing 1 fps sample with a phase but no tool row (dropped
and counted). Labelled 1 fps samples: 1,733 / 2,839 / 5,828 / 1,522 / 2,344 = 14,266.

**Video fetch:** `caffeinate -i .venv/bin/python scripts/fetch_cholec80_videos.py
--videos 1-5 --connections 16` (started with `nohup` at 23:07, runs detached from any
terminal), log in `data/fetch_videos.log`, files in `data/tmp/`. Smallest first: 04,
01, 05, 02, 03. With 8 connections video04 took 45 min (~0.2 MB/s). After the
switch to 16 it was **~3–4 MB/s**: video01 182 s, video05 226 s, video02 364 s, all
CRC-verified. video03 (4.09 GB) was at 2.25 GB at 23:40, but by then the speed had
**fallen back to ~170 KB/s** (9 of 16 parts still open, ~19 KB/s each), which puts the
finish ~3 h later, around 02:45 on 27 Sep. The fast spell seems to have been the
server, not only the connection count. Already-fetched videos are
skipped on a re-run. To resume, re-run the same command **with the same
`--connections`** (see trap 10).

---

## 1b. Week 3 (12 Oct 2026)

| Task | State | Commit / evidence |
|---|---|---|
| 0. Push + verify | done | `prereg-v1` tag object `3be99b5` peels to `ff2b6e5` on GitHub; push received 2026-10-12T14:02:13Z; `pytest` 516 passed, none skipped, truncation test passes for all 4 tokenizers |
| 1. Bundles on a clean hash | done | `srb_bundle_videos_1-5.tar` (1.0 GB) and `_6-40.tar` (4.6 GB) at `db5ed24`; SHA-256 in `data/colab/SHA256SUMS`; old `srb_bundle.tar` deleted (Arnav approved) |
| 2. Index 1–40 | **done**, all four models, 86,304 frames each | see below |
| 3. 41–80 | **done**: guard lifted (`b09781b`), fetched, extracted, all four models indexed (98,194 frames each) | `data/fetch_extract_41-80.log`, see below |
| 4. Registered evaluation | code done (`60a352f`; parallel models `369ce40`); **val run done** (dev, `results/dev/registered_val_20261012-220609`, ~18 min per model in parallel), see LOG.md | `tests/test_resample.py`, `tests/test_evaluation.py` |
| 5. Test run (RESULT) | **done once**, 13 Oct: lock lifted `1759cb4` (cites `ff2b6e5`), `eval_registered.py --split test` at `1759cb4`, B = 10,000, ~63 min per model in parallel | `results/results.csv`, `results/test/`, LOG.md |

**Index ranges.** An index is one folder per exact video range, and `eval_retrieval.py`
/ `eval_registered.py` load one folder. `scripts/merge_index.py` (`6fde54d`) joins
consecutive ranges into one (PeskaVLP: `videos_1-5` + `videos_6-40` → `videos_1-40`,
86,304 × 768); it refuses unless the parts share model, revision and dim and tile the
manifest row for row, so the merged index has the same hash as a direct build. CLIP,
SigLIP and MedSigLIP were instead **rebuilt over 1–40 in one go** on the L4, so each
of their indexes has one compute dtype.

**GPU: Jarvis Labs L4 (24 GB), on-demand ₹41.31/hr**, instance 535612, created
2026-10-12T14:11Z. Bundles uploaded from the Mac over ssh (~4 MB/s; both rsyncs died
at 14:20 with an ssh timeout and 6–40 was resumed by appending the missing bytes; both
verified by SHA-256). Python 3.11 venv with the exact pins, weights pre-downloaded at
the pinned revisions. Procedure: `scripts/gpu_index.sh`.

Throughput on the L4 (256-frame bench, then the full run):

| model | float32 | float16 | float16 vs float32, same 512 frames |
|---|---|---|---|
| CLIP ViT-L/14 | 36.6 frames/s | 97.7 (full 1–40 run: **101.75**) | min cosine 0.99994 |
| SigLIP so400m-384 | 10.3–11.0 | 34.3 | 0.99982 |
| MedSigLIP-448 | 7.3–7.8 | 26.5 | 0.99997 |

float32 on the L4 was only 2.3× the T4, below the 3× bar agreed before the paid run,
so Arnav chose **float16** (12 Oct). The prereg pins revisions, not the encoder compute
dtype, so this is not a deviation; `meta.json` records `compute_dtype`. Checks:
CLIP video04 L4 float16 vs Mac float32, min cosine **0.99977** (bar 0.999); CLIP
videos 1–5, new L4 float16 vs old T4 float32 index, min cosine 0.99977, mean 0.999994.
The old float32 `videos_1-5` indexes of CLIP and SigLIP stay on disk but are not used.

Full 1–40 runs on the L4, float16, batch 32, git `db5ed24` (from the bundle):

| model | frames/s | wall time (UTC) | norms | index tar SHA-256 (first 12) |
|---|---|---|---|---|
| CLIP ViT-L/14 | 101.75 | 14:40–14:55 | 0.9998–1.0003 | `90e64ca0c836` |
| SigLIP so400m-384 | 34.98 | 14:55–15:36 | 0.9999–1.0001 | `be30c725600b` |
| MedSigLIP-448 | 25.58 | 15:36–16:33 | 0.9999–1.0001 | `394ea53fc6a6` |

Each tar was checked by SHA-256 on both ends, untarred, and reported **cached** by
`build_index.py --videos 1-40` (same revision and manifest hash as the Mac manifest).
Instance destroyed 16:35Z after 2 h 24 min; **₹101.40** charged (balance before and
after). Remote run log: `data/jarvis_out/run_1-40.log`.

PeskaVLP 6–40 on the Mac (MPS, float32): 72,038 frames at 34.0 frames/s (2,125 s).

**Videos 41–80.** All 40 CRC OK, frames = labelled samples − 1 trailing, totals in
`docs/datasets.md`; `tests/test_manifest.py` passes on 1–80. Bundle
`srb_bundle_videos_41-80.tar` (6.1 GB, SHA-256 `8a8dce12e8a9…`, git `4aed6ba`).
PeskaVLP 41–80 on the Mac: 98,194 frames at 28.94 frames/s. CLIP, SigLIP and MedSigLIP
on a second L4 (instance 535763, same rate, same procedure), float16, batch 32; the
upload dropped once at 4.83 GB and was resumed and SHA-256-checked; the first launch
stopped because the cross-device check reads `frames/video04`, which is not in the 41–80
bundle (copied over, relaunched; note added to `gpu_index.sh`). Cross-check min cosine
0.99977 again.

| model | frames/s | wall time (UTC) | norms | index tar SHA-256 (first 12) |
|---|---|---|---|---|
| CLIP ViT-L/14 | 90.07 | 18:40–18:59 | 0.9998–1.0002 | `7ed24af8eb42` |
| SigLIP so400m-384 | 34.13 | 18:59–19:47 | 0.9999–1.0001 | `0b9b57f02e5a` |
| MedSigLIP-448 | 24.89 | 19:47–20:53 | 0.9999–1.0002 | `6d18c8796145` |

Each tar SHA-256-checked on both ends, untarred, reported **cached** by `build_index.py
--videos 41-80`. Instance destroyed 20:56Z after about 3 h 07 min; **₹128.67** charged.
L4 total for Week 3: **₹230.07** over two sessions. Remote run log:
`data/jarvis_out/run_41-80.log`.

## 2. Environment (reproduce this exactly)

* **Python 3.11.15** from `/opt/homebrew/bin/python3.11`. Venv at `.venv`.
  `uv` is **not installed** on this machine — the plan's uv path was not used.
* Install: `python3.11 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"`.
  After that `import srb` works with no `PYTHONPATH`.
* All deps pinned in `pyproject.toml`: numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1,
  scikit-learn 1.9.1, pillow 12.3.0, opencv-python 5.0.0.93, torch 2.14.0,
  **transformers 5.17.0**, pyyaml 6.0.3, sentencepiece 0.2.2, protobuf 7.36.2
  (the last two for SigLIP's tokenizer), pytest 9.1.1.
* ffmpeg/ffprobe 8.1.2 at `/opt/homebrew/bin` (used for frame extraction).
* `pytest` config lives in `pyproject.toml` and sets
  `filterwarnings = ["error::RuntimeWarning"]`, so any unexpected undefined-value
  warning fails the suite. Keep it.
* CLIP ViT-L/14 weights (~1.7 GB) and SigLIP so400m (~3.5 GB) are in the local HF
  cache, pinned to `32bd6428…` and `9fdffc58…`.
* `gh` is authenticated as `ArnavSawhney`; `origin` is set and `main` tracks it.
* Device: MPS works. fp16 is **rejected** off cuda by design.
* **PeskaVLP setup** (separate env; never install SurgVLP into `.venv`):
  `git clone https://github.com/CAMMA-public/SurgVLP third_party/SurgVLP` and check out
  `85e858998eec47614614b89f6af1363e0ad3f47b`;
  `python3.11 -m venv .venv-surgvlp && .venv-surgvlp/bin/pip install -r envs/surgvlp-requirements.txt`;
  download `PeskaVLP.zip` (URL in `src/srb/models/peskavlp_pins.py`) into
  `data/checkpoints/peskavlp/` with parallel range requests and unzip it. The script
  checks the commit and both SHA-256 pins before loading. Both `third_party/` and
  `.venv-surgvlp/` are gitignored. Smoke: `.venv-surgvlp/bin/python scripts/peskavlp_encode.py smoke`.

---

## 3. Traps already hit — do not rediscover these

1. **transformers 5.x broke the CLIP feature API.** `get_image_features` /
   `get_text_features` now return a `BaseModelOutputWithPooling` with the projected
   vector in `pooler_output`, not a bare tensor. This crashed the first smoke run.
   `_projected_features()` in `registry.py` handles v4 and v5 and **raises** on
   anything else rather than embedding the wrong tensor. Do not unpin `transformers`
   without re-running `scripts/smoke_clip.py`.
2. **Cholec80 is 69.8 GB** (read from the server's `Content-Length`), so ≈ 159 GB with
   extraction. This Mac had 46 GB free, and `download_cholec80.sh` correctly refuses.
   **Run it on Colab/Kaggle.** Endoscapes is 5.9 GB (≈ 18 GB needed) and does fit
   locally. The scripts compute the requirement at runtime; nothing is hardcoded.
3. **Never pipe a verification command through `tail`/`grep`** and read `$?` — it
   reports the pager's status, which hid a smoke-test traceback behind `exit=0`.
   Redirect to a file instead.
4. ~~Frame-index tie-breaking~~ — fixed in Week 1 (tie-neutral metrics).
5. **The CAMMA server is throttled to ~25 KB/s per connection** (our link does 5 MB/s
   to Hugging Face) and drops connections / throws HTTP/2 PROTOCOL_ERRORs. Speed
   scales with parallel range connections. Use HTTP/1.1 and short requests.
6. **Never let curl `--retry` write to an append-mode file.** On retry it restarts the
   range and duplicates bytes. This corrupted the first video04 attempt (parts
   74–78 MB instead of 69.1 MB); it was caught by part size and discarded. The resume
   loop in `_curl_range` now handles retries itself and raises on an oversized part.
7. **One huge range request dies on this server.** `zipfile` reads the Endoscapes
   central directory (17 MB) in one call; `HTTPRangeFile` now caps each request at
   1 MiB.
8. **Colour range in synthetic test videos.** ffmpeg maps limited-range luma to RGB
   as (Y−16)·255/219; a test that assumed grey = Y failed for that reason, not
   because of frame selection.
9. SigLIP needs `sentencepiece` **and** `protobuf`; its "bos/eos token id" config
   warning is harmless.
10. **Part boundaries depend on `--connections`.** A partial `<video>.mp4.parts/`
    directory from a run with a different connection count cannot be resumed (it will
    trip the oversized-part check). Delete that one directory before changing the
    count. Also: `pkill -f fetch_cholec80` matches the shell running the pkill if the
    string is in its own command line.
11. **This Mac is a base M1 with 8 GB of unified memory**, and it is the bottleneck
    for embedding. CLIP ViT-L/14 at 3.3 frames/s had ~4.6 GB wired, ~16% free,
    5.8 M swap-outs, and the process was mostly swapped out waiting on the GPU.
    Larger batches are *slower*. CPU is no rescue (2.87 frames/s). Estimated CLIP
    time for videos 1–5 (14,266 frames) is ~70 min. SigLIP so400m (3.5 GB weights)
    may not fit at all. Close VS Code and Chrome before indexing, and run from the
    plain Terminal app.

12. **Colab's preinstalled torch ecosystem breaks our pinned torch.** Colab (Python
    3.13) ships torchvision and torchaudio built for its own torch/CUDA 12.8; after
    `torch==2.14.0` (cu130) is installed, transformers imports them and dies
    (`operator torchvision::nms does not exist`, then "PyTorch and TorchAudio were
    compiled with different CUDA versions", both surfacing as `Could not import module
    'AutoProcessor'`). Uninstalling them one at a time is whack-a-mole, so
    `colab_index.sh` now builds a clean **Python 3.11 venv with uv** at
    `/content/srb_venv` and `pip install -e .` into it, like the Mac. Resolver warnings
    about cudf/numba/google-colab came from Colab's system env and no longer apply.
13. **The PeskaVLP checkpoint host (seafile.unistra.fr) is throttled to ~22 KB/s per
    connection** (~6 h for the 495 MB zip) but honours HTTP ranges; 32 range requests
    took ~15 min. SurgVLP's `setup.py` fails as an editable install (pkg_resources), so
    it is imported from `third_party/SurgVLP` instead; it also needs `fvcore`,
    `torchmetrics`, `pycm`, `scikit-learn` at import time. Its config file holds a
    top-level `config` dict (`Config.fromfile(...).config.model_config`). SurgVLP calls
    `from_pretrained` without a revision, so Bio_ClinicalBERT is passed as a pinned
    local snapshot path, and the ResNet is built with `pretrained="random"` (every
    weight comes from the checkpoint; this avoids an unpinned torchvision download).

14. **macOS rsync is openrsync**: it rejects `--info=progress2` (exit 1, nothing sent).
    Long ssh uploads to the Jarvis instance can drop with "Operation timed out"; use
    `-o ServerAliveInterval=15` and resume with `tail -c +$((REMOTE_SIZE+1)) file | ssh
    host 'cat >> file'`, then compare SHA-256 on both ends.
15. **The L4 is slow in float32** for the 400M-parameter SigLIP towers (2.3× a T4);
    float16 is 3.3–3.6× faster at min cosine ≥ 0.9998 (§1b). Batch size barely matters
    on either GPU.
16. **Never round-trip `results/results.csv` through pandas.** `read_csv` + `to_csv`
    re-formats floats (`0.17565920732316115` → `0.1756592073231611`) and so rewrites
    registered rows. Append with `to_csv(mode="a", header=False)`.

---

## 4. Next up

Week 5 (LOG.md, 13 Oct):

1. **Done** (`eebebfe`): Deviations, clarification of section 4.4 (no protocol change),
   committed before any RQ4 number.
2. **Done** (`c423454`): `src/srb/temporal.py`, `scripts/eval_temporal.py`,
   `tests/test_temporal.py`, `tests/test_eval_temporal.py`.
3. **Done, dev**: `results/dev/temporal_val_20261013-203339` (run at `c423454`).
   Selection (windows, HMM, TCN weights per seed) in `data/temporal/<model>/`; the test
   run only loads it. 54 min wall with `--jobs 2 --threads 2`; per-(seed, depth) TCN
   training took 45–1,320 s (the long ones all in the second model pair; cause not
   established; weights are deterministic, so it does not affect the numbers).
4. **Done, RESULT**: `eval_temporal.py --split test`, once, clean tree at `3bbe61f`;
   472 rows appended (`single-frame`, `ma-causal`, `ma-centred`, `hmm-forward`,
   `hmm-viterbi`, `tcn-causal` with `seed` = 0, 1, 2 and `mean(0,1,2)`, and
   `tcn-causal-sensitivity-not-registered`); tables `results/test/temporal_*.csv`.
   Sensitivity: `scripts/select_tcn_sensitivity.py` (dev,
   `results/dev/tcn_sensitivity_val_20261013-213707`). Pushed (`4f489a7`).
6. **Done** (`9303cff`, 15 Oct): RQ4 in `docs/results_vs_predictions.md`, registered
   methods only (the TCN sensitivity and the offline methods are in no verdict).
   RQ4a holds (+0.0739, inside 3–8 points); RQ4b falsified (moving average > HMM > TCN);
   RQ4c falsified on point estimates (0.0739 ≥ spread 0.0678; paired difference
   +0.0062 [−0.0101, +0.0172], about equal).
5. **Done** (`c75e985`): RQ5 tagging tool `scripts/tag_rq5.py` (image only, pool order,
   saves every frame). Out-of-body key, image-only review of frames 0–435 and
   `scripts/rq5_apply_corrections.py` added (`ece1f22`, Deviations `84b7ec7`).
   **Tagging done** (19 Oct): 816 frames, stopping rule met; 11 out of body. Frozen
   files and SHA-256s in LOG.md: `data/rq5/tags.csv` (raw), `out_of_body.csv`,
   `tags_corrected.csv` (what the analysis reads).
7. **Next, ask Arnav first:** the RQ5 analysis (per-tag metrics on `tags_corrected.csv`,
   registered, plus the out-of-body-excluded secondary analysis, not registered).

Rules that still bind: the pre-registration is frozen (changes only under `## Deviations`,
shown to Arnav first); **no rerun of the RQ1/RQ2 test split**; no RQ5 verdict
until Arnav asks; the optional image-only encoder can only be exploratory now (LOG.md,
Parked).

**Test scoring is unlocked** (`1759cb4`, after `ff2b6e5`): `evaluate()` and
`evaluate_model()` accept videos 41–80. `eval_retrieval.py` (dev queries) still refuses
them; the registered test run is `eval_registered.py --split test`, which needs a clean
tree and B = 10,000.

## 5. Open decisions and parked work

Both lists live in `LOG.md` and are the single source of truth. Open: #2 per-backbone
pretraining contamination (dataset overlap half is closed); #4 overload (pre-registration
done; model count no longer at risk). Closed: #1 (by `prereg-v1`), #3, #5.

## 6. Manual actions still outstanding for Arnav

* GPU spend so far is in §1b (two L4 sessions, Week 3, ₹230.07); no instance is running,
  and none is needed in Weeks 4–5.
* Approve running the RQ5 analysis (tags are frozen).
* Close VS Code and Chrome before any Mac indexing (trap 11).
* Delete or replace the truncated `~/Downloads/cholec80.zip` / `endoscapes.zip`.
  Endoscapes waits until Week 8 (decided 12 Oct).
* Optional: add a description and topics to the GitHub repo; it was created bare.
