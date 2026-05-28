"""Intel RealSense: aligned RGB + depth streams (no Redis)."""

from __future__ import annotations

import cv2
import numpy as np


def depth_uint16_to_colormap_bgr(depth_u16: np.ndarray) -> np.ndarray:
    """BGR colormap for raw Z16 depth (same idea as ``test_camera.py``)."""
    return cv2.applyColorMap(
        cv2.convertScaleAbs(depth_u16, alpha=0.03), cv2.COLORMAP_JET
    )


def start_realsense(
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    timeout_ms: int,
):
    """Start color+depth, align depth→color; return pipeline, align, depth_scale, intrinsics.

    ``color_intrinsics`` are read from the RealSense **SDK** for the active color stream
    (fx, fy, cx, cy, distortion model, etc.). No separate intrinsics file is required for
    live capture; deproject in ``gemini_pointing`` uses these values with aligned depth.
    """
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = color_profile.get_intrinsics()
    print(f"Depth scale: {depth_scale:.6f} m/unit")

    print(
        f"Warming up color+depth (need {warmup_frames} aligned frames, "
        f"{timeout_ms} ms try_wait each)..."
    )
    n_ok = 0
    max_tries = max(warmup_frames * 20, warmup_frames + 60)
    report_every = 30
    for attempt in range(max_tries):
        ok, frames = pipeline.try_wait_for_frames(timeout_ms)
        if not ok:
            if attempt > 0 and attempt % report_every == 0:
                print(
                    f"  ... still waiting ({n_ok}/{warmup_frames} ok, "
                    f"attempt {attempt}/{max_tries})"
                )
            continue
        frames = align.process(frames)
        if not frames.get_color_frame() or not frames.get_depth_frame():
            continue
        n_ok += 1
        if n_ok >= warmup_frames:
            break

    if n_ok < warmup_frames:
        pipeline.stop()
        raise SystemExit(
            f"RealSense depth required but warmup only got {n_ok}/{warmup_frames} "
            "aligned RGB-D frames. Check USB3, cable/port, power, and that no other "
            "process holds the camera. Try lower --fps or resolution."
        )
    print(f"Warmup done ({n_ok} aligned frames).")

    return pipeline, align, depth_scale, color_intrinsics


def next_rgbd_frame(
    pipeline,
    align,
    depth_scale: float,
    timeout_ms: int,
    miss_counter: list[int],
    max_misses: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Aligned color BGR, depth in meters, and depth BGR colormap for preview."""
    ok, frames = pipeline.try_wait_for_frames(timeout_ms)
    if not ok:
        miss_counter[0] += 1
        print(
            f"Frame didn't arrive within {timeout_ms} ms "
            f"(miss {miss_counter[0]}/{max_misses})."
        )
        if miss_counter[0] >= max_misses:
            raise TimeoutError(
                "Too many consecutive timeouts. Check USB 3 connection, "
                "try a different port/cable, or lower --fps / --width / --height."
            )
        return None
    miss_counter[0] = 0

    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    # IMPORTANT: np.asanyarray on a librealsense frame returns a view into
    # SDK-owned memory. Once `frames`/`color_frame`/`depth_frame` go out of
    # scope at function return, the SDK reclaims those buffers and the numpy
    # views become dangling pointers -> use-after-free -> glibc heap
    # corruption (`corrupted unsorted chunks`, SIGABRT). Always copy.
    color_bgr = np.array(color_frame.get_data(), copy=True)
    depth_u16 = np.array(depth_frame.get_data(), copy=True)
    depth_m = depth_u16.astype(np.float32) * depth_scale
    depth_vis = depth_uint16_to_colormap_bgr(depth_u16)
    return color_bgr, depth_m, depth_vis
