# RQ2 prompt sensitivity: RESULT, Cholec80 test 41-80

Registered run at git `1759cb4` (2026-10-13); built by `scripts/make_rq2_figures.py` from `results/test/`. Per target and model: SD (ddof = 1) and range (max − min) of AP over the five rephrasings. Point estimates (descriptive).

| target | CLIP ViT-L/14 SD | CLIP ViT-L/14 range | SigLIP so400m SD | SigLIP so400m range | MedSigLIP-448 SD | MedSigLIP-448 range | PeskaVLP SD | PeskaVLP range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Preparation | 0.016 | 0.037 | 0.020 | 0.054 | 0.026 | 0.062 | 0.152 | 0.358 |
| Calot triangle dissection | 0.035 | 0.093 | 0.021 | 0.045 | 0.018 | 0.042 | 0.059 | 0.153 |
| Clipping and cutting | 0.011 | 0.022 | 0.017 | 0.037 | 0.017 | 0.041 | 0.089 | 0.218 |
| Gallbladder dissection | 0.017 | 0.036 | 0.014 | 0.034 | 0.006 | 0.013 | 0.090 | 0.241 |
| Gallbladder packaging | 0.123 | 0.255 | 0.115 | 0.258 | 0.106 | 0.278 | 0.147 | 0.393 |
| Cleaning and coagulation | 0.007 | 0.017 | 0.015 | 0.033 | 0.008 | 0.017 | 0.042 | 0.111 |
| Gallbladder retraction | 0.005 | 0.011 | 0.007 | 0.018 | 0.006 | 0.016 | 0.085 | 0.218 |
| Grasper | 0.050 | 0.138 | 0.019 | 0.047 | 0.013 | 0.035 | 0.049 | 0.132 |
| Bipolar | 0.014 | 0.037 | 0.049 | 0.107 | 0.020 | 0.049 | 0.029 | 0.068 |
| Hook | 0.092 | 0.211 | 0.113 | 0.313 | 0.052 | 0.133 | 0.153 | 0.399 |
| Scissors | 0.004 | 0.009 | 0.006 | 0.014 | 0.007 | 0.019 | 0.008 | 0.017 |
| Clipper | 0.007 | 0.016 | 0.011 | 0.029 | 0.013 | 0.030 | 0.030 | 0.071 |
| Irrigator | 0.008 | 0.019 | 0.013 | 0.033 | 0.008 | 0.023 | 0.013 | 0.033 |
| Specimen bag | 0.166 | 0.422 | 0.110 | 0.251 | 0.021 | 0.054 | 0.131 | 0.317 |
| **mean over targets** | **0.040** | **0.094** | **0.038** | **0.091** | **0.023** | **0.058** | **0.077** | **0.195** |

Fraction of the 14 targets on which the two models' order changes between rephrasings (section 4.1 of the pre-registration; from `rq2_order_changes.csv`):

| pair | targets | order changes |
|---|---:|---:|
| CLIP ViT-L/14 vs SigLIP so400m | 14 | 9/14 (0.64) |
| CLIP ViT-L/14 vs MedSigLIP-448 | 14 | 7/14 (0.50) |
| CLIP ViT-L/14 vs PeskaVLP | 14 | 9/14 (0.64) |
| SigLIP so400m vs MedSigLIP-448 | 14 | 8/14 (0.57) |
| SigLIP so400m vs PeskaVLP | 14 | 10/14 (0.71) |
| MedSigLIP-448 vs PeskaVLP | 14 | 11/14 (0.79) |
