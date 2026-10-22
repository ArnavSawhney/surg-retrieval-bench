#!/usr/bin/env bash
# Build frame-embedding indexes on a rented cuda GPU (used: Jarvis Labs L4, 12 Oct 2026)
# from bundles made by scripts/make_colab_bundle.sh, in float16.
#
# On the Mac (bundles are licence-restricted: copy them only to a machine you control):
#   shasum -a 256 data/colab/srb_bundle_videos_*.tar
#   rsync -a --partial data/colab/srb_bundle_videos_1-5.tar root@<host>:/home/
#   (the cross-device check reads frames/video04: if the 1-5 bundle is not uploaded,
#    tar -cf - data/cholec80/frames/video04 | ssh <host> 'tar -x -C /home/srb')
#   (if an upload drops: tail -c +$((REMOTE_SIZE+1)) <tar> | ssh <host> 'cat >> <tar>')
# On the GPU machine, with HF_TOKEN in the environment only (for medsiglip-448; pipe it
# in over ssh stdin, never as an argument or into a file):
#   bash gpu_index.sh /home 1-40 clip-vit-l14 siglip-so400m-384 medsiglip-448
#   bash gpu_index.sh /home endoscapes clip-vit-l14 siglip-so400m-384 medsiglip-448
#     (Endoscapes2023 test keyframes, from `make_colab_bundle.sh endoscapes`; PeskaVLP
#      stays on the Mac, it runs in its own environment)
# Output: /home/out/index_<model>_videos_<videos>.tar (or _endoscapes_test.tar); check
# its SHA-256, copy it back, and untar at the repo root (build_index.py then reports it
# as cached).
#
# Why float16: on the L4, float32 ran SigLIP at 10.3 frames/s and MedSigLIP at 7.3;
# float16 at 34.3 and 26.5 (3.3-3.6x). On the same 512 frames, float16 vs float32 had
# min cosine 0.99982 (SigLIP), 0.99997 (MedSigLIP), 0.99994 (CLIP); the indexes are
# stored in float16 either way. The cross-device check below still requires min cosine
# >= 0.999 against the Mac float32 CLIP video04 index before anything is built.
set -euo pipefail
BASE=${1:?usage: gpu_index.sh <dir with bundles> <videos> <model>...}
VIDEOS=${2:?usage: gpu_index.sh <dir with bundles> <videos> <model>...}
shift 2
OUT=$BASE/out
mkdir -p "$BASE/srb" "$OUT"
cd "$BASE/srb"
for t in "$BASE"/srb_bundle_*.tar; do tar -xf "$t"; done  # same-hash bundles only
echo "git $(cat GIT_HASH); frame dirs $(ls data/cholec80/frames | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VENV=/content/srb_venv
if [ ! -x "$VENV/bin/python" ]; then
  command -v uv >/dev/null || pip install -q uv
  uv venv -q --python 3.11 "$VENV"
fi
uv pip install -q --python "$VENV/bin/python" -e .
PY=$VENV/bin/python

$PY scripts/build_index.py --model clip-vit-l14 --videos 4 --dtype float16 --batch-size 32 --force
$PY - <<'EOF'
import numpy as np
from pathlib import Path
from srb.index import load_index
a, ma, _ = load_index(Path("data/colab_ref/clip-vit-l14_videos_4-4"))
b, mb, meta = load_index(Path("data/index/clip-vit-l14/videos_4-4"))
assert ma.equals(mb), "manifests differ"
cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
print(f"CROSSCHECK CLIP video04 Mac fp32 vs GPU {meta['compute_dtype']}: "
      f"max|delta| {np.abs(a - b).max():.2e}, min cosine {cos.min():.5f}", flush=True)
if cos.min() < 0.999:
    raise SystemExit("GPU index disagrees with the Mac index; do not use it")
EOF

if [ "$VIDEOS" = endoscapes ]; then
  SEL=(--dataset endoscapes); TAG=endoscapes_test
else
  SEL=(--videos "$VIDEOS"); TAG=videos_$VIDEOS
fi
for MODEL in "$@"; do
  echo "=== $MODEL $VIDEOS start $(date -u +%FT%TZ)"
  $PY scripts/build_index.py --model "$MODEL" "${SEL[@]}" --dtype float16 --batch-size 32
  DIR=data/index/$MODEL/$TAG
  $PY - <<EOF
import numpy as np
from pathlib import Path
from srb.index import load_index
e, m, meta = load_index(Path("$DIR"))
n = np.linalg.norm(e, axis=1)
assert np.isfinite(e).all() and np.allclose(n, 1, atol=1e-2), (n.min(), n.max())
print(f"CHECKED $DIR: {e.shape}, norms {n.min():.4f}-{n.max():.4f}, "
      f"{meta['frames_per_sec']} frames/s on {meta.get('gpu')}, {meta['compute_dtype']}, "
      f"git {meta['git_hash'][:12]}", flush=True)
EOF
  tar -cf "$OUT/index_${MODEL}_${TAG}.tar" "$DIR"
  echo "=== $MODEL done $(date -u +%FT%TZ)"
done
echo ALL_DONE
