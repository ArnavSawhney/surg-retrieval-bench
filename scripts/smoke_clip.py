#!/usr/bin/env python
"""End-to-end smoke test for the model registry: no dataset needed.

Encodes 10 synthetic images and 3 text queries through ``get_backbone(...)`` and prints
the device, the embedding shapes, the unit-norm check and the 3x10 cosine-similarity
matrix. This checks the plumbing (device placement, batching, normalisation, dtype),
**not** model quality: the images are procedural noise/gradients, so the similarity
values are meaningless and must never be reported as a result.

Usage:
    python scripts/smoke_clip.py                 # auto device (mps on the Mac)
    python scripts/smoke_clip.py --device cpu --model clip-vit-l14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srb.models.registry import get_backbone, list_backbones  # noqa: E402

QUERIES = [
    "clipping the cystic duct",
    "a hook electrode in the image",
    "a car on a road",  # negative control: should not look like the others
]


def synthetic_images(n: int, size: int, seed: int) -> list[Image.Image]:
    """n procedurally generated RGB images: 5 noise, 5 gradients. Not surgical data."""
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n):
        if i % 2 == 0:
            arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        else:
            ramp = np.linspace(0, 255, size, dtype=np.uint8)
            arr = np.stack([np.tile(ramp, (size, 1)), np.tile(ramp[:, None], (1, size)),
                            np.full((size, size), i * 25 % 256, dtype=np.uint8)], axis=-1)
        images.append(Image.fromarray(arr))
    return images


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="clip-vit-l14", choices=list_backbones())
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--n-images", type=int, default=10)
    p.add_argument("--image-size", type=int, default=448, help="short side of the synthetic images")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"loading backbone {args.model!r} ...")
    model = get_backbone(args.model, device=args.device)
    print(f"  {model!r}")
    print(f"  hf_id={model.spec.hf_id}  revision={model.spec.revision}")

    images = synthetic_images(args.n_images, args.image_size, args.seed)
    img = model.encode_image(images)
    txt = model.encode_text(QUERIES)

    print(f"\nencode_image -> shape {img.shape}  dtype {img.dtype}")
    print(f"encode_text  -> shape {txt.shape}  dtype {txt.dtype}")
    assert img.shape == (args.n_images, model.embed_dim), img.shape
    assert txt.shape == (len(QUERIES), model.embed_dim), txt.shape
    assert img.dtype == np.float32 and txt.dtype == np.float32

    img_norms, txt_norms = np.linalg.norm(img, axis=1), np.linalg.norm(txt, axis=1)
    print(f"L2 norms: images [{img_norms.min():.6f}, {img_norms.max():.6f}]  "
          f"texts [{txt_norms.min():.6f}, {txt_norms.max():.6f}]  (expect 1.0)")
    assert np.allclose(img_norms, 1.0, atol=1e-5) and np.allclose(txt_norms, 1.0, atol=1e-5)

    sim = txt @ img.T
    print(f"\ncosine similarity, {sim.shape[0]} texts x {sim.shape[1]} synthetic images")
    print("(SYNTHETIC noise/gradient images -- these numbers are NOT a result)")
    print("      " + "".join(f"{'img'+str(j):>8}" for j in range(sim.shape[1])))
    for i, q in enumerate(QUERIES):
        print(f"{'t'+str(i):>5} " + "".join(f"{v:>8.4f}" for v in sim[i]) + f"   {q!r}")

    # Determinism: the same inputs must give the same embeddings.
    again = model.encode_text(QUERIES)
    max_delta = float(np.abs(again - txt).max())
    print(f"\nre-encode max |delta| = {max_delta:.2e} (expect ~0)")
    assert max_delta < 1e-5, "encoder is not deterministic"

    print(f"\nOK: {args.model} encoded {args.n_images} images and {len(QUERIES)} texts "
          f"on {model.device.type}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
