# Pre-registration

**STATUS: NOT WRITTEN YET. This is a placeholder.**

Scheduled for **Week 2 (3–9 Oct 2026)** and it must be committed **before any
test-set number exists**. If you are reading this and `results/results.csv` already
contains a row measured on Cholec80 videos 41–80, the protocol was broken.

When written, this document fixes, in advance:

1. **The query list** — every phase, tool and CVS query, final wording.
2. **The prompt templates** — including the 5 rephrasings per query used for RQ2, and
   the negative/control queries used for the score-calibration check.
3. **The metrics** — as already decided in `docs/metrics.md` (primary mAP with the
   random-ranker baseline and lift; secondary R-Precision; P@10/P@50 descriptive only).
4. **The splits** — Cholec80 videos 1–32 train, 33–40 validation, 41–80 test;
   Endoscapes2023 test-only with its official splits.
5. **Numerical predictions for RQ1–RQ5**, written down so they can be wrong. Including
   the expected ordering of backbones and the expected size of the temporal gain.
6. **What would falsify the hypothesis** for each research question.

Nothing in the test split may be inspected before this file is committed.
