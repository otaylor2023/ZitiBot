"""Live RealSense stream + GG-CNN2 grasp heatmap on demand.

Press SPACE on the preview window to run GG-CNN2 on the current RGB-D frame.
The "grasp" window shows a jet-colormapped quality heatmap blended over the
color image, with the top antipodal grasp drawn as a rectangle:
  - red lines  = gripper plates
  - green lines = opening (gripper width)

Keys:
  SPACE  run grasp inference on the current frame
  s      save the latched grasp overlay as grasp_<timestamp>.png
  q/ESC  quit

Usage:
  # First time only: download the pretrained Cornell weights.
  bash vision/ggcnn/weights/download_weights.sh

  python vision/grasp_demo.py
  python vision/grasp_demo.py --no-warmup --width 640 --height 480 --fps 30
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

# Make the sibling 'ggcnn' package importable when running this file directly
# from the repo root, e.g. `python vision/grasp_demo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ggcnn import GGCNN2  # noqa: E402
from ggcnn.grasp_utils import (  # noqa: E402
    Grasp2D,
    draw_grasp_rect,
    heatmap_overlay,
    postprocess,
    preprocess_depth,
)


DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent / "ggcnn" / "weights" / "ggcnn2_cornell_statedict.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense + GG-CNN2 grasp demo.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to GG-CNN2 state_dict (.pt).",
    )
    parser.add_argument(
        "--crop",
        type=int,
        default=None,
        help="Square crop size in source pixels (defaults to the short side).",
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


def load_model(weights_path: Path, device: torch.device) -> GGCNN2:
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"GG-CNN2 weights not found at {weights_path}. "
            "Run vision/ggcnn/weights/download_weights.sh first."
        )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model = GGCNN2()
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def run_inference(
    model: GGCNN2, depth_m: np.ndarray, device: torch.device, crop_size: int | None
) -> tuple[Grasp2D, np.ndarray, "CropInfo"]:  # noqa: F821 - forward ref
    from ggcnn.grasp_utils import CropInfo  # local import to avoid top-level dep

    net_input, crop = preprocess_depth(depth_m, crop_size=crop_size)
    x = torch.from_numpy(net_input).to(device)
    pos, cos, sin, width = model(x)
    pos = pos.squeeze().cpu().numpy()
    cos = cos.squeeze().cpu().numpy()
    sin = sin.squeeze().cpu().numpy()
    width = width.squeeze().cpu().numpy()
    grasp, q_map = postprocess(pos, cos, sin, width, crop)
    return grasp, q_map, crop


def build_grasp_view(
    color_bgr: np.ndarray,
    grasp: Grasp2D,
    q_map: np.ndarray,
    crop,
) -> np.ndarray:
    view = heatmap_overlay(color_bgr, q_map, crop, alpha=0.45)
    draw_grasp_rect(view, grasp)
    label = (
        f"q={grasp.quality:.2f} "
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


def main() -> int:
    args = parse_args()

    device = pick_device(args.device)
    print(f"Loading GG-CNN2 on {device}...")
    model = load_model(args.weights, device)

    pipeline, align, depth_scale = start_pipeline(args)

    live_win, grasp_win = "RealSense (live)", "RealSense + GG-CNN2 (grasp)"
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
            cv2.imshow(live_win, color_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if key == ord(" "):
                depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
                t0 = time.perf_counter()
                grasp, q_map, crop = run_inference(model, depth_m, device, args.crop)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                print(
                    f"grasp: u={grasp.u:.0f} v={grasp.v:.0f} "
                    f"theta={math.degrees(grasp.theta):+.1f}deg "
                    f"width={grasp.width_px:.0f}px q={grasp.quality:.3f} "
                    f"({dt_ms:.1f} ms)"
                )
                latched_view = build_grasp_view(color_bgr, grasp, q_map, crop)
                cv2.imshow(grasp_win, latched_view)

            elif key == ord("s"):
                if latched_view is None:
                    print("Nothing to save yet -- press SPACE first.")
                else:
                    fname = f"grasp_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(fname, latched_view)
                    print(f"Saved {fname}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
