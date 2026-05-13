"""Stream color and depth frames from an Intel RealSense camera.

Usage:
    python scripts/test_camera.py                # color + depth
    python scripts/test_camera.py --no-depth     # color only
    python scripts/test_camera.py --width 1280 --height 720 --fps 30

Press 'q' or ESC in the preview window to quit.
"""

import argparse
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense streaming test.")
    parser.add_argument("--width", type=int, default=640, help="Frame width.")
    parser.add_argument("--height", type=int, default=480, help="Frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate.")
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth stream (color only).",
    )
    return parser.parse_args()


def list_connected_devices() -> list[str]:
    ctx = rs.context()
    return [
        f"{d.get_info(rs.camera_info.name)} "
        f"(SN: {d.get_info(rs.camera_info.serial_number)})"
        for d in ctx.query_devices()
    ]


def build_display(frames, show_depth: bool) -> np.ndarray | None:
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())

    if not show_depth:
        return color_image

    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )
    return np.hstack((color_image, depth_colormap))


def main() -> int:
    args = parse_args()

    devices = list_connected_devices()
    if not devices:
        print("No RealSense devices found. Is the camera plugged in?", file=sys.stderr)
        return 1
    print(f"Found {len(devices)} RealSense device(s):")
    for d in devices:
        print(f"  - {d}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    if not args.no_depth:
        config.enable_stream(
            rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
        )

    profile = pipeline.start(config)

    # Align depth to color so the two images line up pixel-for-pixel.
    align = rs.align(rs.stream.color) if not args.no_depth else None

    if not args.no_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale:.6f} m/unit")

    window_name = "RealSense"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    show_depth = not args.no_depth
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            display = build_display(frames, show_depth)
            if display is None:
                continue

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
