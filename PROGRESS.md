# Progress and session handoff

Current state of the project, written so a new session can resume without re-deriving
anything. `LOG.md` stays the one-line-per-session record required by the plan; this
file holds the detail: environment facts, what is verified, what is next, and the
traps already hit.

**Last updated:** 26 Sep 2026 (end of Session 1)
**Status:** scaffolding and metrics done. **No dataset downloaded. No model evaluated.
No result produced.** `results/` is empty and must stay empty until
`docs/preregistration.md` is written.

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

## 2. Environment (reproduce this exactly)

* **Python 3.11.15** from `/opt/homebrew/bin/python3.11`. Venv at `.venv`.
  `uv` is **not installed** on this machine — the plan's uv path was not used.
* Install: `python3.11 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"`.
  After that `import srb` works with no `PYTHONPATH`.
* All deps pinned in `pyproject.toml`: numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1,
  scikit-learn 1.9.1, pillow 12.3.0, opencv-python 5.0.0.93, torch 2.14.0,
  **transformers 5.17.0**, pyyaml 6.0.3, pytest 9.1.1.
* `pytest` config lives in `pyproject.toml` and sets
  `filterwarnings = ["error::RuntimeWarning"]`, so any unexpected undefined-value
  warning fails the suite. Keep it.
* CLIP ViT-L/14 weights (~1.7 GB) are already in the local HF cache, pinned to
  revision `32bd64288804d66eefd0ccbe215aa642df71cc41`.
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
4. Frame index tie-breaking is deterministic but **not tie-neutral** — see open
   decision 5 in `LOG.md`.

---

## 4. Next up (Week 1, 26 Sep – 2 Oct 2026)

Target from the plan: *index built for 5 videos; toy retrieval works.*

1. **Start the downloads** (Arnav, manual): accept both licences, run
   `download_endoscapes.sh` locally and `download_cholec80.sh` on Colab/Kaggle.
2. `scripts/extract_frames.py` — 1 fps, short side 448 px, JPEG q90; write a frame
   manifest (`.parquet`, gitignored) with video id, frame index, timestamp, path.
3. `src/srb/datasets/cholec80.py` — phase and tool label reader. **Phases are annotated
   at 25 fps, tools at 1 fps**; the frame-index alignment test comes with it (open
   decision 3), in a new `tests/test_manifest.py`.
4. Extract videos 1–5 and measure the real bytes-per-frame, to replace the 5–10 GB
   estimate in `docs/datasets.md`.
5. `scripts/build_index.py` — backbone → embeddings `.npy` + manifest; cache and never
   recompute.
6. Add **SigLIP** to `_SPECS` in `registry.py` (`text_padding="max_length"`, pin the
   revision hash) and re-run the smoke script against it.
7. Toy retrieval over the 5 videos to exercise `metrics.py` on real embeddings — and
   **count exact score ties** while doing it, which is the cheap input to open
   decision 5.

**Week 2 has priority conflict already flagged:** pre-registration beats model count.
Do not compute anything on videos 41–80 before `docs/preregistration.md` is committed.

---

## 5. Open decisions and parked work

Both lists live in `LOG.md` and are the single source of truth. Summary: five open
decisions (near-duplicate 1 fps frames; per-backbone pretraining contamination vs
videos 41–80; 25 fps vs 1 fps label alignment; Week 2 overload; ranking tie
convention) and a parked list covering the unwritten scripts, the dataset readers,
`temporal.py`, SigLIP/MedSigLIP, PeskaVLP's separate environment, the
classification/probe/temporal metrics, and the per-experiment YAMLs.

Nothing in `configs/`, `results/` or `figures/` yet — `.gitkeep` only.

## 6. Manual actions still outstanding for Arnav

* Accept the **Cholec80** licence (CAMMA) and the **Endoscapes2023** licence, then
  start both downloads.
* Accept the **MedSigLIP** HAI-DEF terms on Hugging Face (needed in Week 2, not yet).
* `export HF_TOKEN` in the shell. Not needed for CLIP — it already ran without one.
* Optional: add a description and topics to the GitHub repo; it was created bare.
