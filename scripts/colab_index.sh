#!/usr/bin/env bash
# Build a frame-embedding index on a Colab GPU from the bundle made by
# scripts/make_colab_bundle.sh, then copy it back to Drive.
#
# In a Colab notebook with a GPU runtime (Runtime > Change runtime type > T4 GPU):
#
#   from google.colab import drive; drive.mount('/content/drive')
#   !mkdir -p /content/srb && tar -xf /content/drive/MyDrive/srb/srb_bundle.tar -C /content/srb
#   !cd /content/srb && bash scripts/colab_index.sh siglip-so400m-384 1-5
#
# Gated models (medsiglip-448) also need HF_TOKEN. Add it once in Colab's Secrets panel
# (key icon, name HF_TOKEN, notebook access on), then in a cell before the one above:
#   import os; from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
# The token then lives only in the runtime's environment: never paste it into a cell,
# and never write it to the bundle, a log or Drive.
#
# Output: MyDrive/srb/index_<model>_<tag>.tar. On the Mac, from the repo root:
#   tar -xf ~/Downloads/index_<model>_<tag>.tar
# which restores data/index/<model>/<tag>/; build_index.py then reports it as cached
# (same model revision and manifest hash) and eval_retrieval.py can use it.
#
# Runs in its own Python 3.11 venv (/content/srb_venv) with the exact pyproject pins.
# Compute is forced to float32 so the index is computed like the local MPS ones.
set -euo pipefail

MODEL=${1:?usage: colab_index.sh <model> <videos>}
VIDEOS=${2:?usage: colab_index.sh <model> <videos>}
DRIVE=${DRIVE:-/content/drive/MyDrive/srb}
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || { echo "no GPU: switch the Colab runtime to a GPU" >&2; exit 1; }
[ -d "$DRIVE" ] || { echo "$DRIVE not found: mount Drive first" >&2; exit 1; }
if [ "$MODEL" = medsiglip-448 ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "$MODEL is gated: set HF_TOKEN from Colab Secrets first (see the header)" >&2; exit 1
fi

# Colab's own Python (3.13) carries torchvision/torchaudio built for its torch, which
# transformers imports and which break under our torch pin (see PROGRESS.md trap 12).
# So build a clean Python 3.11 venv with uv and install the package into it, like the Mac.
VENV=/content/srb_venv
if [ ! -x "$VENV/bin/python" ]; then
  pip install -q uv
  uv venv -q --python 3.11 "$VENV"
fi
uv pip install -q --python "$VENV/bin/python" -e .
export PATH="$VENV/bin:$PATH"   # so "python" below is the venv's
python -c "import sys, torch, transformers; print('python', sys.version.split()[0], \
'| torch', torch.__version__, '| transformers', transformers.__version__, \
'| cuda', torch.cuda.is_available())"

# 1. Cross-device check: rebuild CLIP on video04 and compare with the Mac index.
REF=data/colab_ref/clip-vit-l14_videos_4-4
if [ -f "$REF/embeddings.npy" ]; then
  python scripts/build_index.py --model clip-vit-l14 --videos 4 --dtype float32 --force
  python - <<EOF
import numpy as np
from pathlib import Path
from srb.index import load_index
a, ma, _ = load_index(Path("$REF"))
b, mb, meta = load_index(Path("data/index/clip-vit-l14/videos_4-4"))
assert ma.equals(mb), "manifests differ"
cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
d = np.abs(a - b).max()
print(f"CLIP video04 Mac vs Colab: max|delta| {d:.2e}, min cosine {cos.min():.5f}")
# The Mac index matched a CPU re-encode to 1.2e-4 / 0.99991 (fp16 storage).
if cos.min() < 0.999:
    raise SystemExit("GPU index disagrees with the Mac index; do not use it")
EOF
else
  echo "no CLIP reference index in the bundle; skipping the cross-device check"
fi

# 2. The requested index.
python scripts/build_index.py --model "$MODEL" --videos "$VIDEOS" --dtype float32 --bench
DIR=$(ls -td data/index/"$MODEL"/videos_* | head -1)
python - <<EOF
import numpy as np, json
from pathlib import Path
from srb.index import load_index
e, m, meta = load_index(Path("$DIR"))
n = np.linalg.norm(e, axis=1)
assert np.isfinite(e).all() and np.allclose(n, 1, atol=1e-2), (n.min(), n.max())
print(f"$DIR: {e.shape}, norms {n.min():.4f}-{n.max():.4f}, "
      f"{meta['frames_per_sec']} frames/s on {meta.get('gpu')}, {meta['compute_dtype']}")
EOF

TAG=$(basename "$DIR")
tar -cf "$DRIVE/index_${MODEL}_${TAG}.tar" "$DIR"
echo "wrote $DRIVE/index_${MODEL}_${TAG}.tar"
