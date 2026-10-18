"""RQ5 frame pool (pre-registration 4.5) on a synthetic test-video manifest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from srb.rq5 import ORDER_SEED, PER_VIDEO, POOL_SEED, draw_pool, pool_csv, pool_sha256


def _manifest(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{"dataset": "cholec80", "video_id": v, "frame_idx_25fps": 25 * i,
             "path": f"frames/video{v}/{25 * i:06d}.jpg"}
            for v in range(41, 81) for i in range(int(rng.integers(60, 200)))]
    return pd.DataFrame(rows)


def test_pool_shape_and_uniqueness():
    m = _manifest()
    pool = draw_pool(m)
    assert len(pool) == 40 * PER_VIDEO
    assert (pool.groupby("video_id").size() == PER_VIDEO).all()
    assert not pool.duplicated(["video_id", "frame_idx_25fps"]).any()
    assert sorted(pool["tag_order"]) == list(range(len(pool)))
    assert sorted(pool["draw_idx"]) == list(range(len(pool)))
    keys = set(zip(m.video_id, m.frame_idx_25fps))
    assert set(zip(pool.video_id, pool.frame_idx_25fps)) <= keys


def test_pool_follows_the_registered_seeds():
    m = _manifest()
    pool = draw_pool(m)
    rng = np.random.default_rng(POOL_SEED)
    by_draw = pool.sort_values("draw_idx")
    for v in range(41, 81):
        fv = m[m.video_id == v].sort_values("frame_idx_25fps")
        pick = rng.choice(len(fv), size=PER_VIDEO, replace=False)
        got = by_draw[by_draw.video_id == v]["frame_idx_25fps"].to_numpy()
        np.testing.assert_array_equal(got, fv["frame_idx_25fps"].to_numpy()[pick])
    order = np.random.default_rng(ORDER_SEED).permutation(len(pool))
    np.testing.assert_array_equal(pool["draw_idx"].to_numpy(), order)


def test_pool_is_deterministic_and_ignores_row_order_and_labels():
    m = _manifest()
    shuffled = m.sample(frac=1, random_state=3).reset_index(drop=True)
    shuffled["phase"] = "Preparation"  # extra columns are never read
    assert pool_sha256(draw_pool(m)) == pool_sha256(draw_pool(shuffled))
    assert pool_csv(draw_pool(m)) == pool_csv(draw_pool(m))


def test_pool_refuses_wrong_videos():
    m = _manifest()
    with pytest.raises(ValueError, match="test videos"):
        draw_pool(m[m.video_id != 80])
    with pytest.raises(ValueError, match="test videos"):
        draw_pool(pd.concat([m, m.head(1).assign(video_id=12)]))


# ---- tagging ------------------------------------------------------------------------

from srb.rq5 import (ARTEFACTS, MAX_TAGGED, MIN_TAGGED, TAG_COLUMNS, TAGS, check_tags,
                     next_frame, stopping_status)


def _tagged(rows):
    df = pd.DataFrame(rows, columns=TAGS) if rows else pd.DataFrame(columns=TAGS)
    df.insert(0, "tag_order", range(len(df)))
    return df


def test_check_tags():
    check_tags({"smoke", "blur"})
    check_tags({"clean"})
    for bad in (set(), {"clean", "blur"}, {"fog"}):
        with pytest.raises(ValueError):
            check_tags(bad)


def test_stopping_rule_needs_50_of_each_artefact_and_300_frames():
    # 250 frames, 50 of each artefact: every type is at 50 but fewer than 300 frames
    rows = [[int(i // 50 == k) for k in range(5)] + [0] for i in range(250)]
    assert not stopping_status(_tagged(rows))["done"]
    rows += [[0] * 5 + [1]] * 50  # 50 clean frames -> 300
    s = stopping_status(_tagged(rows))
    assert s["done"] and s["counts"]["clean"] == 50 and s["n_tagged"] == MIN_TAGGED
    # "clean" does not count towards the rule: 49 blood frames keep it going
    rows[200] = [0] * 5 + [1]
    assert not stopping_status(_tagged(rows))["done"]


def test_stopping_rule_cap():
    rows = [[0] * 5 + [1]] * MAX_TAGGED
    s = stopping_status(_tagged(rows))
    assert s["done"] and "cap" in s["reason"]
    assert stopping_status(_tagged(rows[:-1]))["done"] is False


def test_next_frame_follows_the_pool_order():
    pool = pd.DataFrame({"tag_order": range(10)})
    assert next_frame(pool, _tagged([])) == 0
    assert next_frame(pool, _tagged([[0] * 5 + [1]] * 3)) == 3
    bad = _tagged([[0] * 5 + [1]] * 2)
    bad["tag_order"] = [0, 2]
    with pytest.raises(ValueError, match="prefix"):
        next_frame(pool, bad)
    assert next_frame(pool, _tagged([[0] * 5 + [1]] * 10)) is None


# ---- out of body ---------------------------------------------------------------------

from srb.rq5 import (OUT_OF_BODY_COLUMNS, apply_corrections, out_of_body_flags,  # noqa: E402
                     raw_prefix_sha256, reviewed_prefix, tag_values)


def test_out_of_body_tags():
    assert tag_values(set(), out_of_body=True)["clean"] == 1
    v = tag_values({"blur"}, out_of_body=True)
    assert (v["blur"], v["clean"]) == (1, 0)
    for t in ("smoke", "glare", "blood", "occlusion", "clean"):
        with pytest.raises(ValueError):
            tag_values({t}, out_of_body=True)
    assert tag_values({"occlusion"}) == {**dict.fromkeys(TAGS, 0), "occlusion": 1}


def _oob(rows):
    return pd.DataFrame(rows, columns=OUT_OF_BODY_COLUMNS)


def test_apply_corrections_rule():
    tags = pd.DataFrame([[i, 41, 25 * i, *r, "t"] for i, r in enumerate(
        [[1, 0, 1, 1, 1, 0], [0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0]])],
        columns=TAG_COLUMNS)
    oob = _oob([[0, 1, "review", "t"], [1, 1, "review", "t"], [2, 0, "review", "t"]])
    out = apply_corrections(tags, oob)
    assert out[list(TAGS) + ["out_of_body"]].values.tolist() == [
        [0, 0, 0, 0, 0, 1, 1],   # smoke, glare, blood, occlusion dropped -> clean
        [0, 1, 0, 0, 0, 0, 1],   # blur kept -> not clean
        [0, 0, 0, 0, 1, 0, 0],   # answered no: unchanged
        [0, 0, 0, 0, 1, 0, 0]]   # no answer: unchanged
    assert tags["occlusion"].tolist() == [1, 1, 1, 1]  # input not modified
    with pytest.raises(ValueError, match="untagged"):
        apply_corrections(tags.iloc[:2], oob)
    with pytest.raises(ValueError, match="more than one"):
        out_of_body_flags(_oob([[0, 1, "review", "t"], [0, 0, "tagging", "t"]]))
    with pytest.raises(ValueError, match="source"):
        out_of_body_flags(_oob([[0, 1, "guess", "t"]]))


def test_reviewed_prefix_and_raw_prefix_hash():
    assert reviewed_prefix(_oob([[0, 1, "review", "t"], [1, 0, "review", "t"],
                                 [5, 1, "tagging", "t"]])) == 2
    with pytest.raises(ValueError, match="prefix"):
        reviewed_prefix(_oob([[1, 1, "review", "t"]]))
    raw = b"h\nr0\nr1\nr2\n"
    assert raw_prefix_sha256(raw, 2) == raw_prefix_sha256(raw + b"r3\n", 2)
    assert raw_prefix_sha256(raw, 2) != raw_prefix_sha256(raw, 3)
    with pytest.raises(ValueError, match="fewer"):
        raw_prefix_sha256(raw, 4)
