# surg-retrieval-bench

An open, reproducible benchmark of vision-language models for **text-to-frame
retrieval on laparoscopic cholecystectomy video**.

Given a plain-English query such as *"clipping the cystic duct"* or *"a hook electrode
in the image"*, how well can a vision-language model find the right frames in a
surgical video? This repository measures that for general-purpose models (CLIP,
SigLIP), a medical model (MedSigLIP) and surgical-domain models (SurgVLP /
PeskaVLP) on public data, with a fixed test split and pre-registered predictions.

> **Status: no results yet.** This repo is at Session 1 (scaffolding, metrics,
> and the CLIP wrapper). No model has been evaluated on a test split. Every number
> that eventually appears in `results/` will be produced by a script in this repo,
> and no number is reported here until it has actually been run. See `LOG.md` for
> the running record.

## Research questions

| | Question |
|---|---|
| **RQ1** | **Zero-shot retrieval.** How well do general (CLIP, SigLIP), medical (MedSigLIP) and surgical (SurgVLP / PeskaVLP) vision-language models retrieve the correct frames for a plain-English surgical query, with no training? |
| **RQ2** | **Prompt sensitivity.** How much do results move across 5 rephrasings of the same query? Reported as the spread across rephrasings, not just the best wording. |
| **RQ3** | **Cheap adaptation.** How much does a linear probe (and, if time allows, a LoRA/adapter) on frozen embeddings improve phase and tool recognition over zero-shot? |
| **RQ4** | **Temporal context.** How much does temporal modelling over the embedding sequence (moving average, HMM, small causal TCN) improve phase recognition over independent single frames? |
| **RQ5** | **Failure modes.** Where does each model break: smoke, motion blur, specular glare, bleeding, instrument occlusion? |

The hypothesis under test is that surgical-specific models beat general ones
zero-shot, that a linear probe closes most of the gap, and that temporal context
matters more than the choice of backbone. It is stated so it can be **falsified**,
and the predictions are committed in `docs/preregistration.md` before the test split
is touched.

## Datasets

| Dataset | Content | Role here | Licence |
|---|---|---|---|
| **Cholec80** (Twinanda et al., EndoNet, IEEE TMI 2017) | 80 cholecystectomy videos; 7 surgical phases at 25 fps, 7 tools at 1 fps | Main benchmark: retrieval, zero-shot classification, probes, temporal models | CC BY-NC-SA 4.0 |
| **Endoscapes2023** (Murali et al., CAMMA) | 201 videos, 58,813 frames at 1 fps, 11,090 with Critical View of Safety annotations from 3 experts | Second, **test-only** set: CVS-criteria retrieval and cross-dataset generalisation | CC BY-NC-SA 4.0 |
| CholecT50 (Nwoye et al.) | Action triplets (instrument, verb, target) | Stretch goal: fine-grained action queries | CC BY-NC-SA 4.0 |

**Protocol (Cholec80).** Frames sampled at 1 fps. Videos 1–40 train, of which 33–40
are held out for validation; videos **41–80 are the test split and are never used for
tuning**. Relevance for retrieval is defined by a frame's ground-truth phase or tool
label.

No data is redistributed here. `scripts/download_cholec80.sh` and
`scripts/download_endoscapes.sh` fetch the official archives after you accept the
licences yourself; frames, videos and embeddings are gitignored. Details and citations:
`docs/datasets.md`.

## Metrics

Relevance in this benchmark is *dense*: a phase query can match tens of thousands of
test frames. That breaks the retrieval metrics people usually reach for — Recall@10 is
mathematically capped near 0.0003, and Hit@10 is ≈ 0.99 for a **random** ranker. So:

* **Primary: mAP** — full-ranking average precision per query, macro-averaged, always
  reported next to the **random-ranker baseline** (the prevalence `R/N`) and the lift.
* **Secondary: R-Precision** — precision in the top `R`.
* **Descriptive only: P@10, P@50.** Never used to rank models.
* **Removed: Recall@k, nDCG@10.**

The reasoning, the arithmetic and the citations are in **`docs/metrics.md`**, and
`docs/metric_rationale.py` is a runnable simulation that demonstrates the failure of
the discarded metrics (including a degenerate "one clip" ranker that scores
P@10 = 1.00 while its AP sits at the random baseline).

## Repository layout

```
configs/      one YAML per experiment
data/         gitignored: raw archives and extracted frames
docs/         metrics.md, datasets.md, preregistration.md, metric_rationale.py
scripts/      download_*.sh, extract_frames.py, build_index.py, eval_*.py
src/srb/
  metrics.py          retrieval metrics (pure NumPy, no I/O)
  models/registry.py  uniform encode_image / encode_text for every backbone
  datasets/           Cholec80 and Endoscapes readers
  temporal.py         smoothing, HMM, TCN
tests/        metrics against hand-computed cases, manifest integrity
results/      one CSV row per experiment
figures/
```

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # or: uv sync
pytest                       # metrics and registry tests
```

Then check the model plumbing end to end (downloads CLIP ViT-L/14, ~1.7 GB, and needs
no dataset):

```bash
python scripts/smoke_clip.py
```

Gated models (MedSigLIP) read a Hugging Face token from the `HF_TOKEN` environment
variable. Accept the model's terms on its Hugging Face page first. The token is never
written to a file, logged, or passed as a command-line argument.

## Reproducibility

* Every dependency is pinned in `pyproject.toml`, and every model revision is pinned
  to a **commit hash** in `src/srb/models/registry.py`, so an upstream re-upload cannot
  silently change a published number.
* The test split is fixed in advance (Cholec80 videos 41–80) and the query list,
  prompt templates, metrics and per-RQ predictions are committed in
  `docs/preregistration.md` **before** any test-set result is computed.
* Each experiment writes one row to `results/results.csv` with model, setting, split,
  metric, value, seed, git commit and date, plus one figure and one line in `LOG.md`.
* Every model and setting that is run gets reported, including the ones that lose.
  Simulated or illustrative numbers are always labelled as such.
* Ranking ties are broken deterministically (see `docs/metrics.md`), so a run is
  bit-reproducible on the same hardware.

## Citing the underlying work

If you use this benchmark, cite the datasets and models you actually ran, exactly as
their authors ask. The list is in `docs/datasets.md`. Code here is MIT; the data and
the surgical models are CC BY-NC-SA 4.0 (non-commercial research only) — see `LICENSE`.
