# Progress and session handoff

Current state of the project, written so a new session can resume without re-deriving
anything. `LOG.md` stays the one-line-per-session record required by the plan; this
file holds the detail: environment facts, what is verified, what is next, and the
traps already hit.

**Last updated:** 8 Oct 2026. Week 1 complete (29 Sep; end-of-week report delivered
8 Oct). Week 2 (scheduled 3–9 Oct) started 8 Oct, about a week late (see `LOG.md`).
Test-split scoring now has its own lock (`66faa00`).
**Status:** metrics tie-neutral, overlap guard in place, CLIP + SigLIP in the registry.
Videos 1–5 fetched, extracted (14,266 frames) and indexed with **both** CLIP and SigLIP
(built on a Colab T4; see §1a); dev eval run on both. **No result produced.**
`results/results.csv` must stay empty until `docs/preregistration.md` is written; dev
numbers go to `results/dev/` (gitignored).

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

---

## 4. Next up

Week 1 is done (§1a). **Week 2, in priority order** (pre-registration beats model count;
if time runs out, cut models, never the pre-registration):

1. **Pre-registration** (`docs/preregistration.md`): Arnav decides open decision #1
   (near-duplicate frames), the RQ1 headline per query, the uncertainty method and the
   RQ5 sampling procedure; queries drafted from label definitions and the literature
   and shown to Arnav before committing; predictions are Arnav's own. Then commit,
   `git tag prereg-v1`, push, and record the hash in `LOG.md`.
2. **MedSigLIP-448** in the registry (gated: HAI-DEF terms + `HF_TOKEN`), smoke test on
   MPS with tiny batches.
3. **Full Cholec80 extraction:** videos 6–40 now (measure the CAMMA server speed
   first); videos 41–80 only after `prereg-v1` is on GitHub, with the fetch/extract/
   index guard lifted in its own commit citing the hash.
4. **PeskaVLP:** separate `.venv-surgvlp`, 1-day timebox, then document and move on.

**Test scoring stays locked all of Week 2.** `srb.retrieval.TEST_SCORING_UNLOCKED` is
`False`; `evaluate()` and `eval_retrieval.py` refuse any Cholec80 video in 41–80. This
lock is separate from the fetch/extract/index guard and is lifted only in Week 3.

## 5. Open decisions and parked work

Both lists live in `LOG.md` and are the single source of truth. Open: #1
near-duplicate 1 fps frames; #2 per-backbone pretraining contamination (dataset
overlap half is closed); #3 label alignment (code + tests done; the video04 manifest passes `tests/test_manifest.py`, closes once the 5-video
manifest passes `tests/test_manifest.py`); #4 Week 2 overload. Closed: #5 ties.

## 6. Manual actions still outstanding for Arnav

* **Overnight 8–9 Oct:** keep the Mac plugged in, lid open, online. Videos 6–40 are
  being fetched and extracted by `scripts/fetch_extract.sh 6 40 32` under
  `caffeinate -is` (log: `data/fetch_extract_6-40.log`, ~0.87 MB/s, ETA ~08:00 9 Oct).
  If it stops, re-run the same command: extracted videos are skipped, fetches resume.
* **Before the MedSigLIP Colab run:** sign in to Colab with the Google account that
  holds the bundle (~30 GB free), and add `HF_TOKEN` to *that* account's Colab Secrets
  (key icon, notebook access on). See the header of `scripts/colab_index.sh`.
* Colab bundles go to Drive one chunk at a time (`make_colab_bundle.sh 6-40`, later
  `41-80`): delete the previous bundle from `MyDrive/srb/` before uploading the next.
* Close VS Code and Chrome before indexing, and start Claude Code from Terminal.app
  (trap 11).
* Delete or replace the truncated `~/Downloads/cholec80.zip` / `endoscapes.zip`.
  A full local Endoscapes copy is needed by Week 8; the full Cholec80 zip does not
  fit on this Mac and is not needed (videos are fetched individually).
* Optional: add a description and topics to the GitHub repo; it was created bare.
