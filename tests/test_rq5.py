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
