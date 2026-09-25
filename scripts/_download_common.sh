#!/usr/bin/env bash
# Shared helpers for the dataset download scripts. Source it, don't run it.
#
# WHERE TO RUN THESE: the full datasets do NOT belong on a laptop. Run the full
# download + frame extraction on Colab/Kaggle (or any box with disk to spare) and
# keep only the extracted 1 fps frames for the 5 development videos locally.
# Raw video is deleted once frames exist -- see docs/datasets.md.

set -euo pipefail

# Abort if free space is below (remote size x SPACE_MULTIPLIER) + SPACE_HEADROOM_GB.
# The multiplier covers the zip plus its extracted contents living side by side.
: "${SPACE_MULTIPLIER:=2.2}"
: "${SPACE_HEADROOM_GB:=5}"
: "${MIN_FREE_GB:=}"          # set this to override the computed requirement entirely
: "${DATA_DIR:=data/raw}"

licence_notice() {
  local name="$1" url="$2"
  cat <<NOTICE
------------------------------------------------------------------------------
LICENCE: $name is released under CC BY-NC-SA 4.0 (non-commercial research only).

  * You must accept the licence on the dataset page before downloading:
      $url
  * Do NOT redistribute videos, frames or per-frame embeddings. This repo commits
    only the scripts that regenerate them (see .gitignore).
  * Cite the dataset exactly as its authors ask (docs/datasets.md).

By continuing you confirm you have accepted the licence yourself.
------------------------------------------------------------------------------
NOTICE
}

# existing_dir <dir> -- nearest existing ancestor of <dir> (df needs a real path)
existing_dir() {
  local dir="$1"
  while [ ! -d "$dir" ]; do dir="$(dirname "$dir")"; done
  printf '%s' "$dir"
}

# free_gb <dir> -- available GiB on the filesystem holding <dir> (nearest integer)
free_gb() {
  # -k for portability between GNU and BSD df; field 4 = available 1K-blocks
  df -k "$(existing_dir "$1")" | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

# remote_size_gb <url> -- Content-Length in GiB (2 dp), or "unknown"
remote_size_gb() {
  local url="$1" bytes
  bytes="$(curl -sIL --max-time 30 "$url" \
           | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {v=$2} END{gsub(/\r/,"",v); print v}')"
  if [ -z "${bytes:-}" ] || [ "$bytes" -le 0 ] 2>/dev/null; then
    echo "unknown"
  else
    awk -v b="$bytes" 'BEGIN{printf "%.2f", b/1024/1024/1024}'
  fi
}

# check_space <url> <dir> -- abort unless there is room for the zip + extraction
check_space() {
  local url="$1" dir="$2" size_gb avail required probe mount
  probe="$(existing_dir "$dir")"
  size_gb="$(remote_size_gb "$url")"
  avail="$(free_gb "$probe")"
  mount="$(df -k "$probe" | awk 'NR==2{print $NF}')"

  if [ "$size_gb" = "unknown" ]; then
    required="${MIN_FREE_GB:-50}"
    echo "WARNING: could not read Content-Length from the server."
    echo "         Falling back to a required-free-space figure of ${required} GB."
    echo "         Override with MIN_FREE_GB=<gb> if you know the real size."
  else
    echo "Remote archive size: ${size_gb} GB"
    required="${MIN_FREE_GB:-$(awk -v s="$size_gb" -v m="$SPACE_MULTIPLIER" \
              -v h="$SPACE_HEADROOM_GB" 'BEGIN{printf "%d", s*m + h + 0.999}')}"
    echo "Required free space: ${required} GB (archive x ${SPACE_MULTIPLIER} to hold the zip"
    echo "                     and its extracted contents, + ${SPACE_HEADROOM_GB} GB headroom)"
  fi
  echo "Available on ${mount}: ${avail} GB"

  if [ "$avail" -lt "$required" ]; then
    cat >&2 <<ABORT

ABORTING: only ${avail} GB free, need ${required} GB.

This is the expected outcome on a laptop. Run this on Colab/Kaggle instead, or
free up space, or override with MIN_FREE_GB=<gb> if you know what you are doing.
ABORT
    exit 1
  fi
  echo "Disk check passed."
}

# fetch <url> <dest> -- resumable download; skips the transfer if already complete
fetch() {
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  echo
  echo "Downloading -> $dest"
  echo "(resumable: re-run this script after an interruption and it continues)"
  # -C - resumes; --retry survives flaky links; -f fails loudly on an HTTP error
  curl -fL -C - --retry 5 --retry-delay 10 --retry-connrefused -o "$dest" "$url"
  echo "Done: $dest ($(du -h "$dest" | cut -f1))"
}

# verify_zip <dest> -- integrity check before anyone spends an hour extracting
verify_zip() {
  local dest="$1"
  if command -v unzip >/dev/null 2>&1; then
    echo "Verifying archive integrity (unzip -t) ..."
    if unzip -tq "$dest"; then
      echo "Archive OK."
    else
      echo "ARCHIVE IS CORRUPT: delete $dest and re-run to download it again." >&2
      exit 1
    fi
  else
    echo "NOTE: unzip not found, skipping the integrity check."
  fi
}
