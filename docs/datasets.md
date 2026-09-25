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
queries. Note that CholecT50 videos overlap Cholec80, which matters for contamination
checks (below).

## Storage and where to run downloads

At 1 fps and 448 px JPEG, ~80 videos of roughly 38 minutes give ~180k frames,
estimated at 5–10 GB (measure on 5 videos before trusting that). Embeddings are
small: 180k × 1152 in float16 is about 0.4 GB per model.

The raw archives are far larger than the frames, and are **not** meant for a laptop.
Run the full download and frame extraction on a machine with disk to spare
(Colab/Kaggle), keep only the extracted frames, and delete the raw video once frames
exist. Local development uses 5 videos.

## Pretraining-data contamination

Surgical-domain models are pretrained on surgical video, and some of that video may
overlap the Cholec80 test videos 41–80 (and CholecT50 overlaps Cholec80 by
construction). An apparent zero-shot win could be partial memorisation. A per-backbone
contamination check — what each model's pretraining corpus contains, and whether it
intersects videos 41–80 — must be written up here before the zero-shot results are
interpreted. Open item in `LOG.md`.

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
* Czempiel et al. *TeCNO: Surgical Phase Recognition with Multi-Stage Temporal
  Convolutional Networks.* MICCAI 2020.

Each reference must be verified against the actual paper before it goes into the
write-up.
