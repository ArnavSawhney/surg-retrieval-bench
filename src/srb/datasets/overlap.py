"""Cross-dataset video overlap with Cholec80, and the test-leakage guard.

Source (transcribed by hand, then checked by re-running their script):
    CAMMA, ``camma_dataset_overlaps``, commit 8347b9f4cb02ebe739747903e6eada272ee9d25e
    https://github.com/CAMMA-public/camma_dataset_overlaps  (CC BY-NC-SA 4.0)
    Accompanies Walimbe, Baby, Srivastav and Padoy, "Adaptation of Multi-modal
    Representation Models for Multi-task Surgical Computer Vision", MICCAI 2025,
    arXiv 2507.05020.

Their Cholec80 split (train 1-40, val 41-48, test 49-80) is **not** ours (train 1-32,
val 33-40, test 41-80). Everything here is keyed by **video ID**, never by split name:
each mapping says which Cholec80 video a video in another dataset *is*, and the test
set is decided by our own rule, ``CHOLEC80_TEST_IDS``.

ID conventions
    * CholecT50 video ``VIDxx`` with ``xx <= 80`` is Cholec80 video ``xx`` (CAMMA's
      mapping file keys both datasets by the same public ID). CholecT50 IDs above 80
      (92, 96, 103, 110, 111) are not Cholec80 videos.
    * Endoscapes IDs are Endoscapes2023 *public* video IDs. The per-video pairing is
      not printed in CAMMA's README (it lists two sorted lists); it was derived from
      their ``mapping_to_endoscapes.json`` + ``endoscapes_vid_id_map.csv`` and agrees
      with the README's sets.
    * M2CAI16-tool: the README gives the Cholec80 and M2CAI ID *sets* per split but not
      a per-video pairing, so only the sets are recorded. That is enough for the
      guard: every one of those M2CAI videos maps into our test split.

Not covered by CAMMA's analysis, hence **no mapping here and the guard refuses them**:
CholecSeg8k, Cholec80-CVS, CholecT45 (a subset of CholecT50; which of its videos are
in our test set has not been checked), and M2CAI16-workflow (its Strasbourg part is
Cholec80 73 and 77-80 per the Cholec80 README.txt, but the M2CAI-side IDs are not
given).
"""

from __future__ import annotations

from collections.abc import Iterable

CAMMA_OVERLAPS_COMMIT = "8347b9f4cb02ebe739747903e6eada272ee9d25e"

CHOLEC80_TRAIN_IDS = frozenset(range(1, 33))
CHOLEC80_VAL_IDS = frozenset(range(33, 41))
CHOLEC80_TEST_IDS = frozenset(range(41, 81))


def cholec80_split(video_id: int) -> str:
    """Our split rule: 1-32 train, 33-40 val, 41-80 test."""
    if video_id in CHOLEC80_TRAIN_IDS:
        return "train"
    if video_id in CHOLEC80_VAL_IDS:
        return "val"
    if video_id in CHOLEC80_TEST_IDS:
        return "test"
    raise ValueError(f"Cholec80 video IDs are 1-80, got {video_id}")


# CholecT50 videos that are Cholec80 videos (same ID in both), by CholecT50 split.
# README "Summary" table, rows Cholec80-{train,val,test} x CholecT50-{train,val,test}.
CHOLECT50_IN_CHOLEC80: dict[str, frozenset[int]] = {
    "train": frozenset({1, 2, 4, 5, 13, 15, 18, 22, 23, 25, 26, 27, 31, 35, 36, 40,
                        43, 47, 48, 49, 52, 56, 57, 60, 62, 65, 66, 68, 70, 75, 79}),
    "val": frozenset({8, 12, 29, 50, 78}),
    "test": frozenset({6, 10, 14, 32, 42, 51, 73, 74, 80}),
}

# Endoscapes2023 public video ID -> Cholec80 video ID. README "Summary" table rows
# Endoscapes-{train,val} x Cholec80-test; Endoscapes-test has no Cholec80 overlap.
ENDOSCAPES_TO_CHOLEC80: dict[int, int] = {
    1: 67, 2: 68, 3: 70, 4: 71, 7: 72,  # Endoscapes train
    121: 66,                            # Endoscapes val
}

# M2CAI16-tool: README section "2. M2CAI", table "Cholec-80 vs M2CAI-tool".
# (M2CAI video IDs, Cholec80 video IDs) as sets; no per-video pairing is published.
M2CAI16_TOOL_OVERLAP: dict[str, tuple[frozenset[int], frozenset[int]]] = {
    "train": (frozenset(range(1, 11)), frozenset(range(67, 77))),
    "test": (frozenset({11, 12, 13, 14, 15}), frozenset({61, 62, 64, 65, 66})),
}


def _cholect50_forbidden() -> set[int]:
    all_ids = set().union(*CHOLECT50_IN_CHOLEC80.values())
    return all_ids & CHOLEC80_TEST_IDS  # same ID space


def _m2cai_forbidden() -> set[int]:
    out: set[int] = set()
    for m2cai_ids, cholec_ids in M2CAI16_TOOL_OVERLAP.values():
        # Every Cholec80 ID in both rows is in 41-80, so every listed M2CAI video is
        # forbidden. Assert it rather than assume it, since pairing is set-level only.
        if not cholec_ids <= CHOLEC80_TEST_IDS:
            raise AssertionError("M2CAI overlap no longer lies entirely in 41-80")
        out |= m2cai_ids
    return out


_FORBIDDEN = {
    "cholec80": lambda: set(CHOLEC80_TEST_IDS),
    "cholect50": _cholect50_forbidden,
    "endoscapes": lambda: {e for e, c in ENDOSCAPES_TO_CHOLEC80.items()
                           if c in CHOLEC80_TEST_IDS},
    "m2cai16_tool": _m2cai_forbidden,
}

KNOWN_DATASETS = frozenset(_FORBIDDEN)


def forbidden_for_training(dataset: str) -> set[int]:
    """Video IDs, in ``dataset``'s own numbering, that are Cholec80 test videos 41-80.

    Nothing trained, probed or tuned may use these. Raises ``KeyError`` for a dataset
    whose overlap with Cholec80 has not been verified (e.g. CholecSeg8k), rather than
    returning an empty set that would read as "clean".
    """
    key = dataset.lower()
    if key not in _FORBIDDEN:
        raise KeyError(
            f"No verified Cholec80 overlap mapping for {dataset!r}; known: "
            f"{sorted(KNOWN_DATASETS)}. Verify its overlap before using it for training."
        )
    return _FORBIDDEN[key]()


def assert_no_test_leakage(train_manifest) -> None:
    """Raise if any row of a training/tuning manifest is a Cholec80 test video.

    ``train_manifest`` is a pandas DataFrame (or anything with ``dataset`` and
    ``video_id`` columns). **Every probe, tuning or cross-dataset script must call
    this on its training data before fitting anything.** Unknown datasets raise.
    """
    pairs: Iterable[tuple[str, int]] = zip(
        train_manifest["dataset"], train_manifest["video_id"]
    )
    leaks: dict[str, set[int]] = {}
    cache: dict[str, set[int]] = {}
    for ds, vid in pairs:
        ds = str(ds).lower()
        if ds not in cache:
            cache[ds] = forbidden_for_training(ds)
        if int(vid) in cache[ds]:
            leaks.setdefault(ds, set()).add(int(vid))
    if leaks:
        detail = "; ".join(f"{ds}: {sorted(v)}" for ds, v in sorted(leaks.items()))
        raise ValueError(f"Test leakage: training manifest contains Cholec80 test videos "
                         f"41-80 (by video identity) -> {detail}")
