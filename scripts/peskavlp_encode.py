#!/usr/bin/env python
"""PeskaVLP encoder, run in its own environment (``.venv-surgvlp``), never imported.

SurgVLP pins transformers 4.30 / mmengine 0.7 / numpy<2, which clash with the main env,
so PeskaVLP sits behind a process boundary:

    .venv-surgvlp/bin/python scripts/peskavlp_encode.py index --videos 1-5
    .venv-surgvlp/bin/python scripts/peskavlp_encode.py text --texts q.json --out q.npy
    .venv-surgvlp/bin/python scripts/peskavlp_encode.py smoke

``index`` writes ``data/index/peskavlp/<tag>/`` in exactly the format of
``scripts/build_index.py`` (same manifest hash, fp16 storage of unit-norm rows, same
meta fields), so ``eval_retrieval.py`` loads it unchanged. ``text`` is what
``srb.models.registry`` calls for ``get_backbone("peskavlp").encode_text``.

Everything is pinned (see ``PESKAVLP`` in ``srb.models.registry``): the SurgVLP code
commit (checked out under ``third_party/SurgVLP``), the checkpoint by SHA-256 (the
download URL is an unversioned Seafile share), and Bio_ClinicalBERT by HF revision,
passed as a local snapshot path because SurgVLP calls ``from_pretrained`` with no
revision. Preprocessing and tokenisation follow SurgVLP's own ``surgvlp.load`` /
``surgvlp.tokenize`` at that commit: resize to 360x640, centre-crop 224, ImageNet
normalisation; Bio_ClinicalBERT, padding to max_length 77.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # srb.index etc. need only numpy/pandas/PIL
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "SurgVLP"))

from build_index import video_tag  # noqa: E402
from fetch_cholec80_videos import parse_videos, refuse_test_videos  # noqa: E402
from srb.index import (cache_is_valid, embed_frames, git_hash,  # noqa: E402
                       manifest_hash, save_index)
from srb.models.peskavlp_pins import PESKAVLP  # noqa: E402

DATA = ROOT / "data" / "cholec80"
SURGVLP = ROOT / "third_party" / "SurgVLP"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_pins() -> tuple[Path, Path]:
    """SurgVLP commit, checkpoint hash and BERT snapshot, or exit with what to fix."""
    head = subprocess.run(["git", "-C", str(SURGVLP), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != PESKAVLP["surgvlp_commit"]:
        sys.exit(f"third_party/SurgVLP is at {head or 'nothing'}, expected "
                 f"{PESKAVLP['surgvlp_commit']}; see PROGRESS.md (PeskaVLP setup)")
    ckpt = ROOT / PESKAVLP["checkpoint_path"]
    if not ckpt.exists():
        sys.exit(f"missing {ckpt.relative_to(ROOT)}; see PROGRESS.md (PeskaVLP setup)")
    got = sha256(ckpt)
    if got != PESKAVLP["checkpoint_sha256"]:
        sys.exit(f"{ckpt.name} sha256 {got} != pinned {PESKAVLP['checkpoint_sha256']}")
    from huggingface_hub import snapshot_download
    bert = Path(snapshot_download(PESKAVLP["bert_id"], revision=PESKAVLP["bert_revision"],
                                  allow_patterns=["*.json", "*.txt", "pytorch_model.bin"]))
    return ckpt, bert


class PeskaVLP:
    """encode_image / encode_text with the same contract as srb.models.registry.Backbone:
    float32, L2-normalised numpy rows."""

    def __init__(self, device: str | None = None):
        import surgvlp  # noqa: F401  (registers the model classes)
        import torchvision.transforms as T
        from mmengine.config import Config
        from surgvlp.codes.models import build_algorithm
        from transformers import AutoTokenizer

        ckpt, bert = check_pins()
        if device is None:
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.dtype = torch.float32
        cfg = Config.fromfile(str(SURGVLP / "tests" / "config_peskavlp.py")).config.model_config
        cfg.backbone_text.text_bert_type = str(bert)  # pinned snapshot, not a floating id
        cfg.backbone_img.pretrained = "random"  # every weight comes from the checkpoint
        model = build_algorithm(cfg)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            sys.exit(f"checkpoint does not match the model: missing {missing[:5]}, "
                     f"unexpected {unexpected[:5]}")
        self.model = model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(bert))
        self.transform = T.Compose([  # surgvlp._transform(224) at the pinned commit
            T.Resize((360, 640)), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.embed_dim = 768

    @staticmethod
    def _unit(x: torch.Tensor) -> np.ndarray:
        x = x.detach().to(torch.float32)
        n = x.norm(dim=-1, keepdim=True)
        if torch.any(n == 0):
            raise RuntimeError("encoder produced a zero vector; cannot L2-normalise")
        return (x / n).cpu().numpy()

    def encode_image(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for s in range(0, len(images), batch_size):
                x = torch.stack([self.transform(im.convert("RGB"))
                                 for im in images[s:s + batch_size]]).to(self.device)
                out.append(self._unit(self.model(x, None, mode="video")["img_emb"]))
        return np.concatenate(out)

    def tokenize(self, texts: list[str]) -> dict:
        """surgvlp.tokenize at the pinned commit, batched (it pads each text to 77)."""
        tok = self.tokenizer(texts, return_tensors="pt", truncation=True,
                             padding="max_length", max_length=PESKAVLP["text_max_length"])
        return {k: tok[k].to(self.device)
                for k in ("input_ids", "attention_mask", "token_type_ids")}

    def encode_text(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for s in range(0, len(texts), batch_size):
                tok = self.tokenize(texts[s:s + batch_size])
                out.append(self._unit(self.model(None, tok, mode="text")["text_emb"]))
        return np.concatenate(out)


def cmd_text(args) -> None:
    texts = json.loads(Path(args.texts).read_text())
    if not (isinstance(texts, list) and texts and all(isinstance(t, str) for t in texts)):
        sys.exit("--texts must be a JSON list of strings")
    emb = PeskaVLP(args.device).encode_text(texts)
    np.save(args.out, emb.astype(np.float32))


def cmd_smoke(args) -> None:
    """Plumbing check like scripts/smoke_clip.py: synthetic images, so the similarity
    values are meaningless. Also checks the batched tokenizer against surgvlp.tokenize."""
    import surgvlp

    sys.path.insert(0, str(ROOT / "scripts"))
    from smoke_clip import QUERIES, synthetic_images

    enc = PeskaVLP(args.device)
    print(f"device {enc.device.type}, dtype float32, dim {enc.embed_dim}")
    images = synthetic_images(10, 256, seed=0)
    img, txt = enc.encode_image(images, batch_size=4), enc.encode_text(QUERIES)
    assert img.shape == (10, enc.embed_dim) and txt.shape == (len(QUERIES), enc.embed_dim)
    assert img.dtype == np.float32 and txt.dtype == np.float32
    n_i, n_t = np.linalg.norm(img, axis=1), np.linalg.norm(txt, axis=1)
    print(f"L2 norms: images [{n_i.min():.6f}, {n_i.max():.6f}]  "
          f"texts [{n_t.min():.6f}, {n_t.max():.6f}]  (expect 1.0)")
    assert np.allclose(n_i, 1.0, atol=1e-5) and np.allclose(n_t, 1.0, atol=1e-5)
    img2, txt2 = enc.encode_image(images, batch_size=4), enc.encode_text(QUERIES)
    print(f"re-encode bit-identical: images {np.array_equal(img, img2)}, "
          f"texts {np.array_equal(txt, txt2)}")
    assert np.array_equal(img, img2) and np.array_equal(txt, txt2)
    ref = surgvlp.tokenize(QUERIES, model_name=str(enc.tokenizer.name_or_path))
    ours = enc.tokenize(QUERIES)
    same = all(torch.equal(ref[k].cpu(), ours[k].cpu())
               for k in ("input_ids", "attention_mask", "token_type_ids"))
    print(f"tokens identical to surgvlp.tokenize: {same}")
    assert same
    print("cosine similarity (synthetic images, meaningless):")
    print(np.array2string(txt @ img.T, precision=3, suppress_small=True))
    print("SMOKE OK")


def cmd_index(args) -> None:
    videos = parse_videos(args.videos)
    refuse_test_videos(videos)
    manifest = pd.read_parquet(DATA / "manifest.parquet")
    missing = sorted(set(videos) - set(manifest.video_id))
    if missing:
        sys.exit(f"videos {missing} are not in the manifest yet; run extract_frames.py")
    manifest = (manifest[manifest.video_id.isin(videos)]
                .sort_values(["video_id", "frame_idx_25fps"]).reset_index(drop=True))
    mhash = manifest_hash(manifest)
    out_dir = ROOT / "data" / "index" / "peskavlp" / video_tag(videos)
    if cache_is_valid(out_dir, PESKAVLP["revision"], mhash) and not args.force:
        print(f"cached: {out_dir.relative_to(ROOT)} (same revision and manifest); skipping")
        return

    enc = PeskaVLP(args.device)
    paths = [DATA / p for p in manifest.path]
    print(f"embedding {len(paths)} frames with peskavlp on {enc.device.type}")
    emb, seconds = embed_frames(enc, paths, batch_size=args.batch_size)
    meta = {
        "model": "peskavlp", "hf_id": PESKAVLP["hf_id"], "revision": PESKAVLP["revision"],
        "checkpoint_sha256": PESKAVLP["checkpoint_sha256"],
        "bert_revision": PESKAVLP["bert_revision"],
        "videos": videos, "n_frames": len(emb), "dim": int(emb.shape[1]),
        "storage_dtype": "float16", "manifest_sha256": mhash,
        "git_hash": git_hash(ROOT), "device": enc.device.type, "compute_dtype": "float32",
        "python": sys.version.split()[0], "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if enc.device.type == "cuda" else None,
        "batch_size": args.batch_size, "bench_frames_per_sec": {},
        "seconds": round(seconds, 1), "frames_per_sec": round(len(emb) / seconds, 2),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    save_index(out_dir, emb, manifest, meta)
    print(f"wrote {out_dir.relative_to(ROOT)}: {len(emb)} x {emb.shape[1]}, "
          f"{meta['frames_per_sec']} frames/s, git {meta['git_hash'][:12]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text", help="encode a JSON list of strings to a .npy")
    t.add_argument("--texts", required=True)
    t.add_argument("--out", required=True)
    i = sub.add_parser("index", help="build data/index/peskavlp/<videos>")
    i.add_argument("--videos", required=True)
    i.add_argument("--batch-size", type=int, default=PESKAVLP["default_batch_size"])
    i.add_argument("--force", action="store_true")
    sub.add_parser("smoke", help="plumbing check on synthetic images")
    args = ap.parse_args()
    {"text": cmd_text, "index": cmd_index, "smoke": cmd_smoke}[args.cmd](args)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.0f}s", file=sys.stderr)
