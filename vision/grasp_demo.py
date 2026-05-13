"""Live RealSense stream + heatmap grasp predictor on demand.

Two pretrained models are supported, both single-shot heatmap predictors for
parallel-jaw grippers:

  - ``ggcnn2``   GG-CNN2 (Morrison et al., RSS 2018). Depth-only, 300x300 input.
  - ``grconvnet`` GR-ConvNet v3 (Kumra et al., IROS 2020). RGB-D, 224x224 input.

Press SPACE on the preview window to run the selected model on the current
RGB-D frame. The "grasp" window shows a jet-colormapped quality heatmap blended
over the color image, with the top antipodal grasp drawn as a rectangle:
  - red lines  = gripper plates
  - green lines = opening (gripper width)

Keys:
  SPACE  run grasp inference on the current frame
  s      save the latched grasp overlay as grasp_<timestamp>.png
  q/ESC  quit

Usage:
  # First time only: download whichever model's weights you want.
  bash vision/ggcnn/weights/download_weights.sh        # GG-CNN2
  bash vision/grconvnet/weights/download_weights.sh    # GR-ConvNet

  python vision/grasp_demo.py                    # defaults to ggcnn2
  python vision/grasp_demo.py --model grconvnet
  python vision/grasp_demo.py --model grconvnet --crop 400
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

# Make the sibling packages ('ggcnn', 'grconvnet', shared utils) importable when
# running this file directly from the repo root, e.g. `python vision/grasp_demo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grasp_utils import (  # noqa: E402
    Grasp2D,
    draw_grasp_rect,
    heatmap_overlay,
)
from predictors import (  # noqa: E402
    GraspPrediction,
    Predictor,
    available_models,
    default_weights,
    make_predictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense + grasp-heatmap demo.")
    parser.add_argument(
        "--model",
        choices=available_models(),
        default="ggcnn2",
        help="Which grasp model to use.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Override the default weights path for the chosen model.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=None,
        help="Square crop size in source pixels (defaults to the short side).",
    )
    parser.add_argument(
        "--exclude-bottom-frac",
        type=float,
        default=0.15,
        help=(
            "Fraction of the bottom of the source frame to hide from the model "
            "(useful to mask out a wrist-mounted gripper). 0 = disabled. "
            "Default 0.15 (15%% of the height)."
        ),
    )
    parser.add_argument(
        "--exclude-top-frac",
        type=float,
        default=0.0,
        help="Fraction of the top of the source frame to hide from the model.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=10000,
        help="Per-frame wait timeout in ms.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to discard before display (lets auto-exposure settle).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Torch device for inference.",
    )
    return parser.parse_args()


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def build_grasp_view(
    color_bgr: np.ndarray,
    grasp: Grasp2D,
    q_map: np.ndarray,
    crop,
    model_name: str,
) -> np.ndarray:
    view = heatmap_overlay(color_bgr, q_map, crop, alpha=0.45)
    draw_grasp_rect(view, grasp)
    label = (
        f"[{model_name}] q={grasp.quality:.2f} "
        f"theta={math.degrees(grasp.theta):+.0f}deg "
        f"w={grasp.width_px:.0f}px"
    )
    cv2.putText(view, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(view, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return view


def start_pipeline(args: argparse.Namespace) -> tuple[rs.pipeline, rs.align, float]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth scale: {depth_scale:.6f} m/unit")

    print(f"Warming up ({args.warmup_frames} frames)...")
    for _ in range(args.warmup_frames):
        try:
            pipeline.wait_for_frames(args.timeout_ms)
        except RuntimeError:
            pass
    return pipeline, align, depth_scale


def run_inference(
    predictor: Predictor,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    crop_size: int | None,
    bottom_exclude_px: int,
    top_exclude_px: int,
) -> tuple[GraspPrediction, float]:
    t0 = time.perf_counter()
    pred = predictor.predict(
        color_bgr,
        depth_m,
        crop_size=crop_size,
        bottom_exclude_px=bottom_exclude_px,
        top_exclude_px=top_exclude_px,
    )
    return pred, (time.perf_counter() - t0) * 1000.0


def annotate_excluded_bands(
    img: np.ndarray, bottom_exclude_px: int, top_exclude_px: int
) -> np.ndarray:
    """Dim the excluded regions on the live preview so it's obvious what the model sees."""
    if bottom_exclude_px <= 0 and top_exclude_px <= 0:
        return img
    out = img.copy()
    h = out.shape[0]
    if top_exclude_px > 0:
        roi = out[0:top_exclude_px]
        out[0:top_exclude_px] = (roi.astype(np.uint16) * 1 // 3).astype(np.uint8)
    if bottom_exclude_px > 0:
        roi = out[h - bottom_exclude_px:h]
        out[h - bottom_exclude_px:h] = (roi.astype(np.uint16) * 1 // 3).astype(np.uint8)
    return out


def main() -> int:
    args = parse_args()

    device = pick_device(args.device)
    weights = args.weights or default_weights(args.model)
    print(f"Loading {args.model} on {device} (weights: {weights})...")
    predictor = make_predictor(args.model, args.weights, device)

    bottom_exclude_px = max(0, int(round(args.height * args.exclude_bottom_frac)))
    top_exclude_px = max(0, int(round(args.height * args.exclude_top_frac)))
    if bottom_exclude_px or top_exclude_px:
        print(
            f"Masking frame: top {top_exclude_px}px, bottom {bottom_exclude_px}px "
            f"(of {args.height}px height) will be hidden from the model."
        )

    pipeline, align, depth_scale = start_pipeline(args)

    live_win = "RealSense (live)"
    grasp_win = f"RealSense + {args.model} (grasp)"
    cv2.namedWindow(live_win, cv2.WINDOW_AUTOSIZE)

    latched_view: np.ndarray | None = None
    consecutive_misses = 0
    max_consecutive_misses = 10

    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(args.timeout_ms)
            if not ok:
                consecutive_misses += 1
                print(
                    f"Frame didn't arrive within {args.timeout_ms} ms "
                    f"(miss {consecutive_misses}/{max_consecutive_misses})."
                )
                if consecutive_misses >= max_consecutive_misses:
                    print(
                        "Too many consecutive timeouts. Check USB 3 connection, "
                        "try a different port/cable, or lower --fps / --width / --height.",
                        file=sys.stderr,
                    )
                    return 2
                continue
            consecutive_misses = 0

            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            cv2.imshow(
                live_win,
                annotate_excluded_bands(color_bgr, bottom_exclude_px, top_exclude_px),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if key == ord(" "):
                depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
                pred, dt_ms = run_inference(
                    predictor,
                    color_bgr,
                    depth_m,
                    args.crop,
                    bottom_exclude_px,
                    top_exclude_px,
                )
                g = pred.grasp
                print(
                    f"[{args.model}] grasp: u={g.u:.0f} v={g.v:.0f} "
                    f"theta={math.degrees(g.theta):+.1f}deg "
                    f"width={g.width_px:.0f}px q={g.quality:.3f} "
                    f"({dt_ms:.1f} ms)"
                )
                latched_view = build_grasp_view(
                    color_bgr, g, pred.quality_map, pred.crop, args.model
                )
                cv2.imshow(grasp_win, latched_view)

            elif key == ord("s"):
                if latched_view is None:
                    print("Nothing to save yet -- press SPACE first.")
                else:
                    fname = f"grasp_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(fname, latched_view)
                    print(f"Saved {fname}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
