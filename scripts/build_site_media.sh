#!/usr/bin/env bash
# Transcode selected ZitiBot media into docs/assets/ for GitHub Pages.
# All video output is H.264 MP4 with audio stripped (-an).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VID_SRC="$ROOT/assets/edited_videos"
PHOTO_SRC="$ROOT/assets/Photos"
LOGS_SRC="${ZITIBOT_LOGS:-$ROOT/logs2}"
OUT_VID="$ROOT/docs/assets/video"
OUT_IMG="$ROOT/docs/assets/img"

mkdir -p "$OUT_VID"/{hero,tasks,taste,demo,bloopers,sim} "$OUT_IMG"/{grasps,photos}

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not installed." >&2
  exit 1
fi

# format=yuv420p forces 8-bit output; some sources are 10-bit (yuv420p10le),
# which browsers cannot decode in H.264.

# Landscape: max 1280 wide, no audio
transcode_landscape() {
  local src="$1" dst="$2"
  echo "  -> $dst"
  ffmpeg -y -hide_banner -loglevel error -i "$src" \
    -an \
    -vf "scale='min(1280,iw)':-2,format=yuv420p" \
    -c:v libx264 -crf 26 -preset medium -movflags +faststart \
    "$dst"
}

# Landscape WITH audio: max 1280 wide (used for taste tests)
transcode_landscape_audio() {
  local src="$1" dst="$2"
  echo "  -> $dst"
  ffmpeg -y -hide_banner -loglevel error -i "$src" \
    -vf "scale='min(1280,iw)':-2,format=yuv420p" \
    -c:v libx264 -crf 26 -preset medium \
    -c:a aac -b:a 128k -movflags +faststart \
    "$dst"
}

# Portrait: max 720 wide, no audio
transcode_portrait() {
  local src="$1" dst="$2"
  echo "  -> $dst"
  ffmpeg -y -hide_banner -loglevel error -i "$src" \
    -an \
    -vf "scale='min(720,iw)':-2,format=yuv420p" \
    -c:v libx264 -crf 26 -preset medium -movflags +faststart \
    "$dst"
}

# Hero: 720p landscape, no audio
transcode_hero() {
  local src="$1" dst="$2"
  echo "  -> $dst"
  ffmpeg -y -hide_banner -loglevel error -i "$src" \
    -an \
    -vf "scale='min(1280,iw)':-2,format=yuv420p" \
    -c:v libx264 -crf 24 -preset medium -movflags +faststart \
    "$dst"
}

resize_jpg() {
  local src="$1" dst="$2" max="${3:-1600}"
  echo "  -> $dst"
  ffmpeg -y -hide_banner -loglevel error -i "$src" \
    -vf "scale='min(${max},iw)':-2" -q:v 3 \
    "$dst"
}

# Resize but keep PNG (for diagrams with text)
resize_png() {
  local src="$1" dst="$2" max="${3:-1800}"
  echo "  -> $dst"
  ffmpeg -y -hide_banner -loglevel error -i "$src" \
    -vf "scale='min(${max},iw)':-2" \
    "$dst"
}

copy_grasp_png() {
  local src="$1" dst="$2"
  echo "  -> $dst"
  cp "$src" "$dst"
}

echo "=== Hero ==="
transcode_hero "$VID_SRC/EggBot Sped Up and Muted.mov" "$OUT_VID/hero/demo.mp4"

echo "=== Tasks (carousel) ==="
# Slide 1
transcode_landscape "$VID_SRC/egg in cracker/egg in cracker 1.mov" "$OUT_VID/tasks/01-egg-in-cracker-1.mp4"
transcode_landscape "$VID_SRC/egg in cracker/egg in cracker 4.mov" "$OUT_VID/tasks/01-egg-in-cracker-2.mp4"
transcode_landscape "$VID_SRC/egg in cracker/egg in cracker 3.mov" "$OUT_VID/tasks/01-egg-in-cracker-3.mp4"
transcode_landscape "$VID_SRC/egg crack/egg crack 1.mov" "$OUT_VID/tasks/02-egg-crack-1.mp4"
transcode_landscape "$VID_SRC/egg crack/egg crack 2.mov" "$OUT_VID/tasks/02-egg-crack-2.mp4"
transcode_landscape "$VID_SRC/egg crack/egg crack 3.mov" "$OUT_VID/tasks/02-egg-crack-3.mp4"
transcode_landscape "$VID_SRC/shell drop/shell drop 1.mov" "$OUT_VID/tasks/03-shell-1.mp4"
transcode_landscape "$VID_SRC/shell drop/shell drop 2.mov" "$OUT_VID/tasks/03-shell-2.mp4"
# Slide 2
transcode_landscape "$VID_SRC/whisk/whisk.mov" "$OUT_VID/tasks/04-whisk-1.mp4"
transcode_landscape "$VID_SRC/whisk/whisk 2.mov" "$OUT_VID/tasks/04-whisk-2.mp4"
transcode_landscape "$VID_SRC/pan pour/pour in new pan.mov" "$OUT_VID/tasks/05-pour-1.mp4"
transcode_landscape "$VID_SRC/pan pour/pour into old pan.mov" "$OUT_VID/tasks/05-pour-2.mp4"
transcode_landscape "$VID_SRC/pan move/new pan move 1.mov" "$OUT_VID/tasks/06-pan-move-1.mp4"
transcode_landscape "$VID_SRC/pan move/old pan move 1.mov" "$OUT_VID/tasks/06-pan-move-2.mp4"
# Slide 3
transcode_landscape "$VID_SRC/ladle flip/ladle 1.mov" "$OUT_VID/tasks/07-ladle-1.mp4"
transcode_landscape "$VID_SRC/ladle flip/ladle flip 2.mov" "$OUT_VID/tasks/07-ladle-2.mp4"
transcode_landscape "$VID_SRC/scramble/scramble 1.mov" "$OUT_VID/tasks/08-scramble-1.mp4"
transcode_landscape "$VID_SRC/scramble/scramble old pan.mov" "$OUT_VID/tasks/08-scramble-2.mp4"

echo "=== Taste (with audio) ==="
transcode_landscape_audio "$VID_SRC/taste/olivia_taste_new.MOV" "$OUT_VID/taste/olivia.mp4"
transcode_landscape_audio "$VID_SRC/taste/william taste.MOV" "$OUT_VID/taste/william.mp4"

echo "=== Sim (OpenSai) ==="
transcode_landscape "$VID_SRC/sim video.mov" "$OUT_VID/sim/sim.mp4"

echo "=== Demo (portrait) ==="
transcode_portrait "$VID_SRC/portrait fixed.mov" "$OUT_VID/demo/portrait.mp4"

echo "=== Bloopers ==="
transcode_landscape "$VID_SRC/bloopers/egg crack then swing.mov" "$OUT_VID/bloopers/egg-crack-swing.mp4"
transcode_landscape "$VID_SRC/bloopers/egg drop.mov" "$OUT_VID/bloopers/egg-drop.mp4"

echo "=== Photos ==="
resize_jpg "$PHOTO_SRC/adrian eggs.jpeg" "$OUT_IMG/photos/adrian-eggs.jpg"
resize_jpg "$PHOTO_SRC/presentation pic.jpeg" "$OUT_IMG/photos/presentation.jpg"
resize_jpg "$PHOTO_SRC/decent team photo.jpg" "$OUT_IMG/photos/team-1.jpg"
resize_jpg "$PHOTO_SRC/another decent team photo.jpg" "$OUT_IMG/photos/team-2.jpg"
resize_jpg "$PHOTO_SRC/scrambling.jpeg" "$OUT_IMG/photos/hardware.jpg"
resize_png "$PHOTO_SRC/state machine.png" "$OUT_IMG/photos/state-machine.png"

echo "=== Force-control photos (cropped from stills) ==="
# tongs + egg, cracker squeeze (cropped from provided photos), whisk in bowl (from footage)
ffmpeg -y -hide_banner -loglevel error -i "$ROOT/assets/egg in tongs.png" \
  -vf "crop=309:171:165:115" -q:v 3 "$OUT_IMG/photos/force-tongs.jpg"
ffmpeg -y -hide_banner -loglevel error -i "$ROOT/assets/better egg mid crack.png" \
  -vf "crop=420:290:200:150" -q:v 3 "$OUT_IMG/photos/force-cracker.jpg"
ffmpeg -y -hide_banner -loglevel error -ss 30 -i "$VID_SRC/whisk/whisk.mov" \
  -frames:v 1 -vf "crop=760:560:680:420,scale=720:-2" "$OUT_IMG/photos/force-whisk.jpg"

echo "=== Grasp images ==="
copy_grasp_png "$LOGS_SRC/gemini_response_pan_grasp.png" "$OUT_IMG/grasps/perpendicular.png"
copy_grasp_png "$LOGS_SRC/gemini_response_egg.png" "$OUT_IMG/grasps/parallel.png"
copy_grasp_png "$LOGS_SRC/gemini_response_egg_pour_new_bowl.png" "$OUT_IMG/grasps/bowl.png"
copy_grasp_png "$LOGS_SRC/gemini_response_egg_egg.png" "$OUT_IMG/grasps/egg-tongs.png"

echo ""
echo "Done. Output in docs/assets/"
du -sh "$ROOT/docs/assets"
