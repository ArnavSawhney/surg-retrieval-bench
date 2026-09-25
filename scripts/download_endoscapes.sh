#!/usr/bin/env bash
# Download Endoscapes2023 (Murali et al., CAMMA).
#
# 201 cholecystectomy videos, 58,813 frames at 1 fps, 11,090 with critical view of
# safety (CVS) annotations from 3 experts, plus official splits. Used here as a
# second test set: CVS-criteria retrieval and cross-dataset generalisation.
#
# RUN THIS ON COLAB/KAGGLE, NOT ON THE MACBOOK. See download_cholec80.sh.
#
# Usage:
#   bash scripts/download_endoscapes.sh
#   DATA_DIR=/mnt/disk MIN_FREE_GB=40 bash scripts/download_endoscapes.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_download_common.sh

NAME="Endoscapes2023"
URL="https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip"
LICENCE_PAGE="https://github.com/CAMMA-public/Endoscapes"
DEST="${DATA_DIR}/endoscapes.zip"

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
  unzip -q "$DEST" -d "${DATA_DIR}/endoscapes"
  # Endoscapes already ships 1 fps frames and official splits: no extraction needed.

Endoscapes is a TEST-ONLY set in this benchmark. Do not train or tune on it.
NEXT
