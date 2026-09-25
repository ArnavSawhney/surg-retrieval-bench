#!/usr/bin/env bash
# Download Cholec80 (Twinanda et al., EndoNet, IEEE TMI 2017).
#
# 80 laparoscopic cholecystectomy videos, 7 phases labelled at 25 fps and 7 tools
# labelled at 1 fps. This is the main benchmark set: videos 1-40 train (33-40 held
# out for validation), 41-80 test. Nothing is ever tuned on 41-80.
#
# RUN THIS ON COLAB/KAGGLE, NOT ON THE MACBOOK. The archive is tens of GB and the
# laptop only needs the extracted 1 fps frames of ~5 videos for development.
#
# Usage:
#   bash scripts/download_cholec80.sh              # download to data/raw/
#   DATA_DIR=/mnt/disk bash scripts/download_cholec80.sh
#   MIN_FREE_GB=80 bash scripts/download_cholec80.sh
#   SKIP_LICENCE_PROMPT=1 bash ...                 # for non-interactive runs

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_download_common.sh

NAME="Cholec80"
URL="https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip"
LICENCE_PAGE="http://camma.u-strasbg.fr/datasets"
DEST="${DATA_DIR}/cholec80.zip"

licence_notice "$NAME" "$LICENCE_PAGE"
if [ -z "${SKIP_LICENCE_PROMPT:-}" ]; then
  printf 'Have you accepted the %s licence? [y/N] ' "$NAME"
  read -r reply
  case "$reply" in [yY]*) ;; *) echo "Aborted."; exit 1 ;; esac
fi

check_space "$URL" "$DATA_DIR"
fetch "$URL" "$DEST"
verify_zip "$DEST"

cat <<NEXT

Next:
  unzip -q "$DEST" -d "${DATA_DIR}/cholec80"
  python scripts/extract_frames.py --dataset cholec80 --fps 1   # (Week 1)

Then delete the raw videos: the benchmark only needs the 1 fps frames.
NEXT
