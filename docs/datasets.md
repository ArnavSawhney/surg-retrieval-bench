# Datasets: access, licences, protocol, citations

Nothing in this repository redistributes data. Download each dataset yourself from its
official source after accepting its licence.

## Licence summary

| Dataset / model | Licence | What that means here |
|---|---|---|
| Cholec80 | CC BY-NC-SA 4.0 | Non-commercial research only; attribution; share-alike |
| Endoscapes2023 | CC BY-NC-SA 4.0 | Same |
| CholecT50 | CC BY-NC-SA 4.0 | Same |
| SurgVLP / HecVL / PeskaVLP | CC BY-NC-SA 4.0 | Same |
| MedSigLIP | Health AI Developer Foundations terms | Gated on Hugging Face; accept the terms, then use `HF_TOKEN` |
| CLIP, SigLIP | See each model card | General-purpose baselines |
| **Code in this repo** | MIT | See `LICENSE` |

Consequences, enforced by `.gitignore`:

* **Never commit** videos, extracted frames, annotation-derived per-frame files, or
  per-frame embeddings. Commit the scripts that regenerate them.
* The published artefacts are code, configs, aggregate metric rows in
  `results/results.csv`, and figures.

## Cholec80

80 laparoscopic cholecystectomy videos (Twinanda et al., EndoNet, IEEE TMI 2017).
7 surgical phases annotated at 25 fps; 7 tool presence labels annotated at 1 fps.

* Phases: Preparation, CalotTriangleDissection, ClippingCutting,
  GallbladderDissection, GallbladderPackaging, CleaningCoagulation,
  GallbladderRetraction.
* Tools: Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, SpecimenBag.

Access: request/accept the licence via CAMMA (<http://camma.u-strasbg.fr/datasets>).
Archive: `https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip`.

Download with `bash scripts/download_cholec80.sh`. The script prints the licence
notice, reads the archive size from the server, refuses to start if there is not room
for the archive plus its extracted contents, and resumes an interrupted transfer.

### Protocol used in this benchmark

* Sample frames at **1 fps**, resize the short side to 448 px, JPEG quality 90.
* **Videos 1–32: train. Videos 33–40: validation. Videos 41–80: test.**
* Nothing is tuned on 41–80 — not a prompt, not a smoothing window, not a threshold.
* Retrieval relevance: a frame is relevant to a query if its ground-truth phase or
  tool label matches that query's target class.
* Approximate test-split size at 1 fps: ~98,500 frames (from EndoNet mean phase
  durations × 40 videos). Replace with the real count once the annotations are parsed.

### Note on label frame rates

Phases are annotated at 25 fps and tools at 1 fps, so the two label sources index
frames differently. A frame-index alignment test belongs in the manifest-integrity
suite before any result is produced — open item in `LOG.md`.

## Endoscapes2023

201 laparoscopic cholecystectomy videos (Murali et al., CAMMA); 58,813 frames at
1 fps, 11,090 of them with Critical View of Safety (CVS) annotations from 3 experts,
plus official train/val/test splits.

Access: <https://github.com/CAMMA-public/Endoscapes>.
Archive: `https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip`.
Download with `bash scripts/download_endoscapes.sh`.

Role here: a **test-only** second set, for CVS-criteria retrieval and cross-dataset
generalisation. Use the official splits. Do not train or tune on it.

## CholecT50

50 videos with action triplets (instrument, verb, target); Nwoye et al.
<https://github.com/CAMMA-public/cholect50>. Stretch goal only: fine-grained action
queries. 22 of its videos are Cholec80 test videos (see Contamination, below).

## Storage and where to run downloads

At 1 fps and 448 px JPEG, ~80 videos of roughly 38 minutes give ~180k frames,
estimated at 5–10 GB (measure on 5 videos before trusting that). Embeddings are
small: 180k × 1152 in float16 is about 0.4 GB per model.

The raw archives are far larger than the frames, and are **not** meant for a laptop.
Run the full download and frame extraction on a machine with disk to spare
(Colab/Kaggle), keep only the extracted frames, and delete the raw video once frames
exist. Local development uses 5 videos.

## Contamination

Two separate questions: (1) does another *dataset* we might train or tune on contain
our Cholec80 test videos 41–80, and (2) did a *backbone* see them during pretraining?
Both are mapped by **video identity**, never by split name.

### Dataset overlap with our test split (videos 41–80)

Source: CAMMA, `camma_dataset_overlaps`, pinned at commit
`8347b9f4cb02ebe739747903e6eada272ee9d25e`
(<https://github.com/CAMMA-public/camma_dataset_overlaps>, CC BY-NC-SA 4.0), which
accompanies Walimbe, Baby, Srivastav and Padoy, *Adaptation of Multi-modal
Representation Models for Multi-task Surgical Computer Vision*, MICCAI 2025,
arXiv 2507.05020. The ID tables were transcribed from their README and cross-checked
by re-running their `overlap_analysis.py`; the per-video Endoscapes pairing comes
from their mapping files. The Cholec80 archive's own `README.txt` independently
confirms the M2CAI16-tool overlap ("video 61-76, except video 63"). Their Cholec80 split (train 1–40, val 41–48, test 49–80)
differs from ours (train 1–32, val 33–40, **test 41–80**), so their split labels are
ignored and every video is re-mapped by ID.

| Dataset | Videos that are Cholec80 test videos (41–80) | Source |
|---|---|---|
| Cholec80 | 41–80 by definition | our protocol |
| CholecT50 | **22 of our 40 test videos**: 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 60, 62, 65, 66, 68, 70, 73, 74, 75, 78, 79, 80 (same IDs in both datasets) | CAMMA README |
| CholecT45 | a subset of CholecT50; exact overlap **unverified** | — |
| Endoscapes2023 train | Endoscapes 1→67, 2→68, 3→70, 4→71, 7→72 | CAMMA README + mapping files |
| Endoscapes2023 val | Endoscapes 121→66 | CAMMA README + mapping files |
| Endoscapes2023 test | none | CAMMA README |
| M2CAI16-tool train | M2CAI 1–10 ↔ Cholec80 67–76 (set-level; no per-video pairing published) | CAMMA README |
| M2CAI16-tool test | M2CAI 11–15 ↔ Cholec80 61, 62, 64, 65, 66 (set-level) | CAMMA README |
| M2CAI16-workflow (Strasbourg part) | Cholec80 73, 77, 78, 79, 80; M2CAI-side IDs not given, so the guard refuses this dataset | Cholec80 `README.txt` (inside the archive) |
| CholecSeg8k | built from Cholec80 videos; **unverified** — not covered by CAMMA's analysis | — |
| Cholec80-CVS | built on Cholec80; **unverified** — not covered by CAMMA's analysis | — |

Consequences for this benchmark:

* Endoscapes2023 is used as a test set only; its train/val videos 1, 2, 3, 4, 7 and 121
  must never be used to train or tune anything evaluated on Cholec80.
* CAMMA also note that Cholec80 replaced the M2CAI16 Strasbourg videos, so the M2CAI
  Strasbourg subset should not be combined with Cholec80.
* Enforced in code: `src/srb/datasets/overlap.py` provides
  `forbidden_for_training(dataset)` and `assert_no_test_leakage(train_manifest)`.
  Every probe, tuning or cross-dataset script must call the latter on its training
  data. Datasets without a verified mapping raise instead of passing silently.

### Backbone pretraining data

An apparent zero-shot win could be partial memorisation. Rows are filled only from
sources actually read; anything else is **unverified**.

| Model | Pretraining data (as stated by the source) | Overlaps Cholec80 41–80? | Source read |
|---|---|---|---|
| CLIP ViT-L/14 | WIT: 400M web image–text pairs, not released | **unverified** (corpus not public; web scrape could include published surgical frames) | Radford et al., ICML 2021 |
| SigLIP so400m | WebLI web image–text pairs, not released | **unverified** (corpus not public) | Zhai et al., ICCV 2023 |
| MedSigLIP-448 | MIMIC-CXR, Slake-VQA, PAD-UFES-20, SCIN, TCGA, CAMELYON, PMC-OA, Mendeley knee X-ray, MedQA, licensed/internal Google data, plus natural image–text pairs. No surgical or endoscopic video named. | **unverified**: no surgical video listed, but PMC-OA (figures from papers) may contain published Cholec80 frames | HF model card `google/medsiglip-448` |
| SurgVLP / HecVL / PeskaVLP | "(image, text) pairs from surgical video lectures" with ASR transcripts; lecture sources not named in the README | **unverified** (read the SVL paper before Week 3) | CAMMA-public/SurgVLP README |
| EndoFM / SurgeNet-style | — | **unverified** (not read) | — |

## Citations

Cite the datasets and models you actually ran, in the form their authors ask for.

* Twinanda, Shehata, Mutter, Marescaux, de Mathelin, Padoy. *EndoNet: A Deep
  Architecture for Recognition Tasks on Laparoscopic Videos.* IEEE TMI, 2017. (Cholec80)
* Murali et al. *Endoscapes2023: A dataset of laparoscopic cholecystectomy videos with
  critical view of safety annotations.* CAMMA.
* Nwoye et al. *Rendezvous: Attention mechanisms for the recognition of surgical action
  triplets in endoscopic videos.* (CholecT50)
* Yuan et al. *SurgVLP / HecVL / PeskaVLP.* CAMMA. <https://github.com/CAMMA-public/SurgVLP>
* Google Health AI Developer Foundations. *MedSigLIP.*
* Radford et al. *Learning Transferable Visual Models From Natural Language
  Supervision.* ICML 2021. (CLIP)
* Zhai et al. *Sigmoid Loss for Language Image Pre-Training.* ICCV 2023. (SigLIP)
* Walimbe, Baby, Srivastav, Padoy. *Adaptation of Multi-modal Representation Models
  for Multi-task Surgical Computer Vision.* MICCAI 2025, arXiv 2507.05020. Related
  work to read; source of the dataset-overlap tables above (with the CAMMA
  `camma_dataset_overlaps` repository).
* Czempiel et al. *TeCNO: Surgical Phase Recognition with Multi-Stage Temporal
  Convolutional Networks.* MICCAI 2020.

Each reference must be verified against the actual paper before it goes into the
write-up.
