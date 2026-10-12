"""Build, cache and load a frame-embedding index.

Layout (under data/, gitignored -- per-frame embeddings are licence-restricted)::

    data/index/<model>/<tag>/embeddings.npy    (N, dim) float16, row i = manifest row i
    data/index/<model>/<tag>/manifest.parquet  the rows that were embedded, in order
    data/index/<model>/<tag>/meta.json         model revision, manifest hash, git hash,
                                               device, batch size, throughput

Embeddings are **stored** in fp16 (half the disk) and must be **upcast** before any
similarity is computed: ``load_index`` returns float32, and the metrics refuse fp16
scores (see ``srb.metrics``).

Cache rule: an existing index is reused iff its model revision and manifest hash both
match; otherwise it is rebuilt. Never silently mixed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

KEY_COLUMNS = ["dataset", "video_id", "frame_idx_25fps", "path"]


def manifest_hash(manifest: pd.DataFrame) -> str:
    """SHA-256 of the identifying columns, in row order (row order is the index order)."""
    csv = manifest[KEY_COLUMNS].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(csv.encode()).hexdigest()


def git_hash(repo: Path) -> str:
    """HEAD commit, with ``-dirty`` if tracked files have uncommitted changes.

    Outside a git checkout (a Colab bundle, see scripts/make_colab_bundle.sh) the hash
    is read from the ``GIT_HASH`` file written when the bundle was made.
    """
    if not (repo / ".git").exists() and (repo / "GIT_HASH").exists():
        return (repo / "GIT_HASH").read_text().strip()
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                            "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def cache_is_valid(out_dir: Path, revision: str, mhash: str) -> bool:
    meta = out_dir / "meta.json"
    if not (meta.exists() and (out_dir / "embeddings.npy").exists()):
        return False
    m = json.loads(meta.read_text())
    return m.get("revision") == revision and m.get("manifest_sha256") == mhash


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def embed_frames(backbone, paths: list[Path], batch_size: int, chunk: int = 256,
                 log_every: int = 2048) -> tuple[np.ndarray, float]:
    """Encode images in order. Returns (float32 (N, dim), seconds spent)."""
    out, t0, done = [], time.time(), 0
    with ThreadPoolExecutor(4) as pool:  # JPEG decode off the main thread
        for start in range(0, len(paths), chunk):
            images = list(pool.map(_load_rgb, paths[start:start + chunk]))
            out.append(backbone.encode_image(images, batch_size=batch_size))
            done += len(images)
            if log_every and done % log_every < chunk:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(paths)} frames, {rate:.1f} frames/s", flush=True)
    return np.concatenate(out), time.time() - t0


def save_index(out_dir: Path, emb: np.ndarray, manifest: pd.DataFrame, meta: dict) -> None:
    if len(emb) != len(manifest):
        raise ValueError(f"{len(emb)} embeddings for {len(manifest)} manifest rows")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", emb.astype(np.float16))
    manifest.reset_index(drop=True).to_parquet(out_dir / "manifest.parquet", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_index(out_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """(float32 embeddings, manifest, meta). The fp16 -> float32 upcast happens here."""
    emb = np.load(out_dir / "embeddings.npy").astype(np.float32)
    manifest = pd.read_parquet(out_dir / "manifest.parquet")
    meta = json.loads((out_dir / "meta.json").read_text())
    if len(emb) != len(manifest):
        raise ValueError(f"corrupt index {out_dir}: {len(emb)} rows vs {len(manifest)}")
    if meta.get("manifest_sha256") != manifest_hash(manifest):
        raise ValueError(f"corrupt index {out_dir}: manifest hash mismatch")
    return emb, manifest, meta


def merge_indexes(part_dirs: list[Path], expected: pd.DataFrame
                  ) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Concatenate indexes of consecutive video ranges into one (e.g. 1-5 + 6-40 = 1-40).

    Refuses unless every part has the same model, revision and dim, and the concatenated
    rows equal ``expected`` (the manifest selection a direct build of the union would
    embed) row for row, so the merged index is indistinguishable by hash from a direct
    build. Part metas are kept under ``merged_from``. Returns (float32 emb, manifest,
    meta without git_hash); fp16 -> float32 -> fp16 on save is exact.
    """
    parts = [load_index(d) for d in part_dirs]
    metas = [m for _, _, m in parts]
    for key in ("model", "revision", "dim", "storage_dtype"):
        if len({str(m.get(key)) for m in metas}) != 1:
            raise ValueError(f"parts differ in {key}: {[m.get(key) for m in metas]}")
    manifest = pd.concat([m for _, m, _ in parts], ignore_index=True)
    mhash = manifest_hash(manifest)
    if mhash != manifest_hash(expected):
        raise ValueError("parts do not tile the expected manifest (order, gap or overlap)")
    emb = np.concatenate([e for e, _, _ in parts])
    first = metas[0]
    meta = {
        "model": first["model"], "hf_id": first.get("hf_id"), "revision": first["revision"],
        "videos": [v for m in metas for v in m["videos"]], "n_frames": len(emb),
        "dim": first["dim"], "storage_dtype": first["storage_dtype"],
        "manifest_sha256": mhash, "merged_from": metas,
    }
    return emb, manifest, meta
