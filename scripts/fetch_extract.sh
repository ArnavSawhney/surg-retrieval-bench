#!/usr/bin/env bash
# Fetch and extract Cholec80 videos one at a time: fetch video v, extract its frames
# (extract_frames.py deletes the mp4 afterwards), then move on. Disk never holds more
# than one source video. A failed fetch is retried (fetches resume); a failed
# extraction stops the run, since it means a label or frame-count check failed.
#
#   nohup caffeinate -is bash scripts/fetch_extract.sh 6 40 32 > data/fetch_extract_6-40.log 2>&1 &
#
# Keep --connections fixed for a video once its fetch has started (PROGRESS.md trap 10).
set -uo pipefail

FIRST=${1:?usage: fetch_extract.sh <first> <last> [connections]}
LAST=${2:?usage: fetch_extract.sh <first> <last> [connections]}
CONN=${3:-16}
cd "$(dirname "$0")/.."
PY=.venv/bin/python

for v in $(seq "$FIRST" "$LAST"); do
  if [ -f "data/cholec80/manifests/video$(printf %02d "$v").parquet" ]; then
    echo "$(date '+%F %T') video $v: already extracted, skipping"; continue
  fi
  ok=0
  for attempt in 1 2 3; do
    echo "$(date '+%F %T') video $v: fetch attempt $attempt"
    if $PY scripts/fetch_cholec80_videos.py --videos "$v" --connections "$CONN"; then ok=1; break; fi
    sleep 60
  done
  [ "$ok" = 1 ] || { echo "$(date '+%F %T') video $v: fetch failed 3 times, stopping"; exit 1; }
  echo "$(date '+%F %T') video $v: extract"
  $PY scripts/extract_frames.py --videos "$v" || { echo "video $v: extraction failed, stopping"; exit 1; }
done
echo "$(date '+%F %T') done: videos $FIRST-$LAST"
