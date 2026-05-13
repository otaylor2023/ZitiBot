#!/usr/bin/env bash
# Download the GG-CNN2 Cornell-pretrained weights from the upstream release.
#
# Source:
#   https://github.com/dougsm/ggcnn/releases/tag/v0.1
#
# After running, this directory will contain:
#   ggcnn2_cornell_statedict.pt   (the file vision/grasp_demo.py loads)
#
# Re-run is idempotent: skips the download if the target file already exists.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$DIR/ggcnn2_cornell_statedict.pt"
URL="https://github.com/dougsm/ggcnn/releases/download/v0.1/ggcnn2_weights_cornell.zip"
ZIP="$DIR/ggcnn2_weights_cornell.zip"
EXTRACT_DIR="$DIR/_ggcnn2_weights_cornell"

if [[ -f "$TARGET" ]]; then
  echo "[ok] $TARGET already exists; nothing to do."
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
  echo "error: unzip is required" >&2
  exit 1
fi

echo "[1/3] downloading $URL"
curl -L --fail --progress-bar -o "$ZIP" "$URL"

echo "[2/3] extracting"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -q -o "$ZIP" -d "$EXTRACT_DIR"

# The zip contains both a full-model file and a state_dict file per epoch.
# Grab the state_dict (smaller, version-agnostic).
SRC="$(find "$EXTRACT_DIR" -type f -name '*statedict*.pt' | sort | tail -n 1)"
if [[ -z "$SRC" ]]; then
  echo "error: no *statedict*.pt found inside the zip" >&2
  exit 1
fi

echo "[3/3] installing $(basename "$SRC") -> $(basename "$TARGET")"
mv "$SRC" "$TARGET"

rm -rf "$EXTRACT_DIR" "$ZIP"
echo "[done] $TARGET"
