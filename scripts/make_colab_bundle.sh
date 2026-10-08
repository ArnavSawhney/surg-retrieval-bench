#!/usr/bin/env bash
# Pack code + extracted frames + manifest into one tar for building an index on Colab.
#
#   bash scripts/make_colab_bundle.sh 6-40      -> data/colab/srb_bundle_videos_6-40.tar
#   bash scripts/make_colab_bundle.sh           -> every video in the manifest
#
# Bundles are made in chunks (6-40 now, 41-80 after prereg-v1) so that only one sits in
# Drive at a time: delete the previous bundle from MyDrive/srb/ before uploading the
# next. Upload to MyDrive/srb/, then follow scripts/colab_index.sh.
# The tar holds Cholec80 frames (CC BY-NC-SA): keep it in your own Drive, never share
# it publicly. It lives under data/, which is gitignored.
#
# Also packs the local CLIP video04 index (if present) as a reference, so the Colab run
# can check that a GPU-built index matches the Mac one before trusting it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SPEC=${1:-}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

[ -f data/cholec80/manifest.parquet ] || { echo "no manifest; run extract_frames.py" >&2; exit 1; }

# Which videos to pack: the requested range, else every video in the manifest.
SEL=$(.venv/bin/python - "$SPEC" <<'EOF'
import sys
import pandas as pd
sys.path.insert(0, "scripts")
from build_index import video_tag
from fetch_cholec80_videos import parse_videos
have = set(pd.read_parquet("data/cholec80/manifest.parquet").video_id)
want = parse_videos(sys.argv[1]) if sys.argv[1] else sorted(have)
missing = sorted(set(want) - have)
if missing:
    sys.exit(f"videos {missing} are not in the manifest yet; run extract_frames.py")
print(video_tag(want), " ".join(f"video{v:02d}" for v in want))
EOF
)
read -r TAG VIDEOS <<< "$SEL"
OUT=data/colab/srb_bundle_${TAG}.tar
echo "packing $TAG: $VIDEOS"
for v in $VIDEOS; do
  [ -d "data/cholec80/frames/$v" ] || { echo "missing frames for $v" >&2; exit 1; }
done

# Code: tracked + untracked-but-not-ignored files under src/ and scripts/, pyproject,
# and README.md (hatchling needs the readme named in pyproject to build) + LICENSE.
# Private planning notes are excluded by .git/info/exclude and never match these paths.
git ls-files --cached --others --exclude-standard -- src scripts pyproject.toml README.md LICENSE \
  | grep -v '__pycache__' > "$STAGE/code.txt"
.venv/bin/python -c "
from pathlib import Path
from srb.index import git_hash
print(git_hash(Path('.')))" > "$STAGE/GIT_HASH"
echo "git: $(cat "$STAGE/GIT_HASH")"

mkdir -p data/colab
tar -cf "$OUT" -T "$STAGE/code.txt" \
  data/cholec80/manifest.parquet \
  $(for v in $VIDEOS; do echo "data/cholec80/frames/$v"; done)
tar -rf "$OUT" -C "$STAGE" GIT_HASH

REF=data/index/clip-vit-l14/videos_4-4
if [ -f "$REF/embeddings.npy" ]; then
  tar -rf "$OUT" -s "|^$REF|data/colab_ref/clip-vit-l14_videos_4-4|" "$REF"
  echo "packed CLIP video04 reference index"
fi

echo "wrote $OUT ($(du -h "$OUT" | cut -f1)); delete the previous bundle from MyDrive/srb/, then upload this one"
