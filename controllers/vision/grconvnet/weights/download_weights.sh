#!/usr/bin/env bash
# Download the GR-ConvNet Cornell RGB-D pretrained checkpoint.
#
# Source: https://github.com/skumra/robotic-grasping/tree/master/trained-models
#
# After running, this directory will contain:
#   grconvnet3_cornell_rgbd.pt   (loaded by python_control/vision/grasp_demo.py --model grconvnet)
#
# Re-run is idempotent: skips the download if the target file already exists.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$DIR/grconvnet3_cornell_rgbd.pt"

# The upstream repo checks in a few pretrained dirs; the canonical Cornell
# RGB-D run is `cornell-randsplit-rgbd-grconvnet3-drop1-ch32/`. The exact
# filename in that dir contains the epoch + IoU and changes between forks, so
# we try a couple of well-known candidates.
BASE="https://github.com/skumra/robotic-grasping/raw/master/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32"
CANDIDATES=(
  "epoch_19_iou_0.98"
  "epoch_19_iou_0.98.pt"
  "epoch_30_iou_0.97"
)

if [[ -f "$TARGET" ]]; then
  echo "[ok] $TARGET already exists; nothing to do."
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is required" >&2
  exit 1
fi

for fname in "${CANDIDATES[@]}"; do
  url="$BASE/$fname"
  echo "[try] $url"
  if curl -L --fail --progress-bar -o "$TARGET.partial" "$url"; then
    mv "$TARGET.partial" "$TARGET"
    echo "[done] $TARGET"
    exit 0
  fi
  rm -f "$TARGET.partial"
done

cat >&2 <<'EOF'
error: could not download a GR-ConvNet Cornell RGB-D checkpoint.

Browse the upstream tree:
  https://github.com/skumra/robotic-grasping/tree/master/trained-models

Pick any file from a `cornell-randsplit-rgbd-grconvnet3-*` directory, then:
  curl -L -o python_control/vision/grconvnet/weights/grconvnet3_cornell_rgbd.pt <raw_github_url>
EOF
exit 1
