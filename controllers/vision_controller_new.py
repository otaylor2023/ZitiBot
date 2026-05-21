#!/usr/bin/env python3
"""Live Gemini + RealSense → OpenSai Franka cartesian goals.

Pipeline:

    Gemini pixel
        ↓
    RealSense deprojection
        ↓
    3D point in camera optical frame
        ↓
    T_base_flange @ T_flange_camera
        ↓
    point in robot base/world frame
        ↓
    publish to OpenSai

SPACE:
    Capture frame + Gemini grasp point + latch world target.

ENTER:
    Publish latched world target to OpenSai Redis.

s:
    Save overlay.

q:
    Quit.

Launch (from repo root)::

  ./ZitiBot/launch_zitibot_panda.sh controllers/vision_controller_new.py

Optional controller args after ``--`` (see ``--goal-offset-*``, ``--no-goal-offset``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import redis

_PYTHON_CONTROL = Path(__file__).resolve().parent
if str(_PYTHON_CONTROL) not in sys.path:
    sys.path.insert(0, str(_PYTHON_CONTROL))

from vision import gemini_pointing as gp
from vision import realsense_rgbd as rs_cam

CONFIG_XML = os.environ.get("ZITIBOT_OPENSAI_CONFIG_XML", "zitibot_panda.xml")
CONTROLLER_TO_USE = "cartesian_controller"

# ---------------------------------------------------------------------
# Camera -> flange calibration
#
# RealSense optical frame:
#   +X right
#   +Y down
#   +Z forward
#
# This MUST match the frame used during hand-eye calibration.
# ---------------------------------------------------------------------

T_FLANGE_CAMERA = np.array([
    [0.0, -1.0, 0.0,  0.053],
    [1.0,  0.0, 0.0, -0.009],
    [0.0,  0.0, 1.0,  0.019],
    [0.0,  0.0, 0.0,  1.0],
], dtype=np.float64)

# Brute-force latch tweak in robot base/world frame (m). Z-up: down = negative Z.
# Set ENABLE_GOAL_OFFSET = False to disable. Tune with --goal-offset-* on CLI.
ENABLE_GOAL_OFFSET = True
GOAL_OFFSET_WORLD_M = np.array([0.053, 0.0, -0.10], dtype=np.float64)


@dataclass(frozen=True)
class OpenSaiRedisKeys:
    cartesian_task_goal_position: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_position"
    )

    cartesian_task_goal_orientation: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_orientation"
    )

    cartesian_task_current_position: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position"
    )
    cartesian_task_current_orientation: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation"
    )

    active_controller: str = (
        "opensai::controllers::FrankaRobot::active_controller_name"
    )

    config_file_name: str = "::sai-interfaces-webui::config_file_name"

    # Franka redis_driver: opensai::redis_driver::<robot>::T_end_effector
    endeffector_transform: str = (
        "opensai::redis_driver::FrankaRobot::T_end_effector"
    )


_KEYS = OpenSaiRedisKeys()

# ---------------------------------------------------------------------

_TEXT_SIZE_MULT = 0.75 * 0.75
_TEXT_FONT_SCALE = 0.48 * 3.0 * _TEXT_SIZE_MULT
_TEXT_LINE_STEP = int(22 * 3.0 * _TEXT_SIZE_MULT)
_TEXT_THICKNESS = 2
_TEXT_EMPTY_SKIP = int(18 * 3.0 * _TEXT_SIZE_MULT)

TEXT_BAND_HEIGHT = int((int(120 * 3.0) + 180) * _TEXT_SIZE_MULT) + 80


# ---------------------------------------------------------------------


def _decode_redis_value(raw: bytes | str | None) -> str | None:
    if raw is None:
        return None

    if isinstance(raw, bytes):
        return raw.decode("utf-8")

    return raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live Gemini pointing + OpenSai cartesian goals (Redis)."
    )
    p.add_argument(
        "--object",
        default="bowl",
        help="Object name in the default bowl-rim prompt (near/bottom rim, closest to camera).",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Override the entire prompt (must ask for JSON points).",
    )
    p.add_argument(
        "--ee-from-cam-json",
        type=Path,
        default=None,
        help="4×4 T_ee_cam JSON (camera optical → EE, meters).",
    )
    p.add_argument(
        "--depth-patch-radius",
        type=int,
        default=2,
        help="Half-size in pixels for depth median window.",
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--endeffector-transform-key",
        default=_KEYS.endeffector_transform,
        help="Redis 4×4 JSON: world/base ← flange (from redis_driver T_end_effector).",
    )
    p.add_argument(
        "--no-goal-offset",
        action="store_true",
        help="Disable brute-force world-frame offset on latched goal.",
    )
    p.add_argument(
        "--goal-offset-x",
        type=float,
        default=float(GOAL_OFFSET_WORLD_M[0]),
        help="World +X offset on latched goal (m). Default: +5.3 cm forward.",
    )
    p.add_argument(
        "--goal-offset-y",
        type=float,
        default=float(GOAL_OFFSET_WORLD_M[1]),
        help="World +Y offset on latched goal (m).",
    )
    p.add_argument(
        "--goal-offset-z",
        type=float,
        default=float(GOAL_OFFSET_WORLD_M[2]),
        help="World +Z offset on latched goal (m). Default: −10 cm (down if Z-up).",
    )
    p.add_argument("--model", default=gp.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--timeout-ms", type=int, default=10000)
    return p.parse_args()


# ---------------------------------------------------------------------


def _try_redis(host: str, port: int):
    try:
        r = redis.Redis(host=host, port=port, decode_responses=False)
        r.ping()
        return r

    except Exception as e:
        print(f"Redis connect failed ({e}).", file=sys.stderr)
        return None


def validate_config(redis_client) -> int | None:
    raw = redis_client.get(_KEYS.config_file_name)

    name = _decode_redis_value(raw)

    if name is None:
        print(
            "Warning: config file key missing in Redis.",
            file=sys.stderr,
        )
        return None

    if name != CONFIG_XML:
        print(
            f"Expected {CONFIG_XML!r} but got {name!r}",
            file=sys.stderr,
        )
        return 1

    return None


# ---------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------


def read_T_base_flange(
    redis_client,
    key: str | None = None,
) -> np.ndarray | None:
    """
    Reads 4×4 homogeneous transform **base ← flange** (p_base = T @ p_flange).
    """

    redis_key = key if key is not None else _KEYS.endeffector_transform

    try:
        raw = redis_client.get(redis_key)

        if raw is None:
            return None

        T = np.array(json.loads(raw), dtype=np.float64)

        if T.shape != (4, 4):
            T = T.reshape(4, 4)

        return T

    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def read_current_tcp_world(
    redis_client,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Cartesian task control point in world: (position (3,), orientation (3,3))."""
    try:
        raw_p = redis_client.get(_KEYS.cartesian_task_current_position)
        raw_o = redis_client.get(_KEYS.cartesian_task_current_orientation)
        if raw_p is None or raw_o is None:
            return None
        pos = np.array(json.loads(raw_p), dtype=np.float64).reshape(3)
        ori = np.array(json.loads(raw_o), dtype=np.float64).reshape(3, 3)
        return pos, ori
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def read_current_orientation(redis_client) -> np.ndarray | None:
    try:
        raw = redis_client.get(_KEYS.cartesian_task_current_orientation)

        if raw is None:
            return None

        R = np.array(json.loads(raw), dtype=np.float64).reshape(3, 3)

        return R

    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    ph = np.ones(4, dtype=np.float64)
    ph[:3] = np.asarray(p, dtype=np.float64).reshape(3)
    out = T @ ph
    return out[:3].copy()


def camera_origin_in_base(T_base_flange: np.ndarray) -> np.ndarray:
    """RealSense optical origin in base/world (m), via T_base←flange @ T_flange←camera."""
    return _transform_point(T_base_flange @ T_FLANGE_CAMERA, np.zeros(3))


def flange_origin_in_base(T_base_flange: np.ndarray) -> np.ndarray:
    """Flange origin in base/world (m)."""
    return np.asarray(T_base_flange, dtype=np.float64)[:3, 3].copy()


def camera_point_to_base(
    T_base_flange: np.ndarray,
    point_camera: np.ndarray,
) -> np.ndarray:
    """
    Convert camera optical-frame point into robot base/world frame.
    """

    p_cam_h = np.ones(4, dtype=np.float64)

    p_cam_h[:3] = np.asarray(point_camera, dtype=np.float64).reshape(3)

    p_base_h = T_base_flange @ T_FLANGE_CAMERA @ p_cam_h

    return p_base_h[:3].copy()


def _print_vec3(label: str, v: np.ndarray) -> None:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    print(f"  {label}: [{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}] m")


def _print_delta(label: str, a: np.ndarray, b: np.ndarray) -> None:
    d = np.asarray(a, dtype=np.float64).reshape(3) - np.asarray(b, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    print(
        f"  {label}: d=[{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}] m  |d|={n*1000:.1f} mm"
    )


def goal_offset_world(args: argparse.Namespace) -> np.ndarray:
    if args.no_goal_offset or not ENABLE_GOAL_OFFSET:
        return np.zeros(3, dtype=np.float64)
    return np.array(
        [args.goal_offset_x, args.goal_offset_y, args.goal_offset_z],
        dtype=np.float64,
    )


def log_tcp_camera_goal_diagnostic(
    *,
    point_camera: np.ndarray,
    latched_goal_world: np.ndarray,
    T_base_flange: np.ndarray,
    tcp_pos: np.ndarray | None,
    raw_goal_world: np.ndarray | None = None,
    goal_offset: np.ndarray | None = None,
) -> None:
    """Compare rim latch, TCP, flange, and camera optical origin (all world frame)."""
    cam_origin = camera_origin_in_base(T_base_flange)
    flange_origin = flange_origin_in_base(T_base_flange)

    print("\n--- SPACE diagnostic (world / base frame, m) ---")
    _print_vec3("p_camera (RS optical, rim point)", point_camera)
    if raw_goal_world is not None:
        _print_vec3("Vision goal (before offset)", raw_goal_world)
    if goal_offset is not None and np.linalg.norm(goal_offset) > 0:
        _print_vec3("Brute-force offset (world)", goal_offset)
    _print_vec3("Latched goal (rim → OpenSai goal_position)", latched_goal_world)
    _print_vec3("Flange origin (Redis T_end_effector)", flange_origin)
    _print_vec3("Camera optical origin (T_bf @ T_fc @ 0)", cam_origin)

    if tcp_pos is not None:
        _print_vec3("TCP / cartesian_task current_position", tcp_pos)
        _print_delta("latched_goal − TCP", latched_goal_world, tcp_pos)
        _print_delta("latched_goal − camera_origin", latched_goal_world, cam_origin)
        _print_delta("TCP − camera_origin", tcp_pos, cam_origin)
        _print_delta("flange_origin − TCP", flange_origin, tcp_pos)
        g_cam = float(np.linalg.norm(latched_goal_world - cam_origin))
        g_tcp = float(np.linalg.norm(latched_goal_world - tcp_pos))
        print(
            "  Interpretation: if |goal−camera| ≪ |goal−TCP|, the latch is near the "
            "camera eye, not the gripper TCP."
        )
        print(
            f"    |goal−camera| = {g_cam*1000:.1f} mm   "
            f"|goal−TCP| = {g_tcp*1000:.1f} mm"
        )
    else:
        print("  TCP: (could not read cartesian_task current_position from Redis)")
        _print_delta("latched_goal − camera_origin", latched_goal_world, cam_origin)

    print("--- end diagnostic ---\n")


# ---------------------------------------------------------------------


def publish_opensai_cartesian(
    redis_client,
    goal_world: np.ndarray,
    goal_orientation: np.ndarray | None,
) -> None:

    while (
        _decode_redis_value(redis_client.get(_KEYS.active_controller))
        != CONTROLLER_TO_USE
    ):
        redis_client.set(
            _KEYS.active_controller,
            CONTROLLER_TO_USE,
        )

    redis_client.set(
        _KEYS.cartesian_task_goal_position,
        json.dumps(goal_world.tolist()),
    )

    if goal_orientation is not None:
        redis_client.set(
            _KEYS.cartesian_task_goal_orientation,
            json.dumps(goal_orientation.tolist()),
        )

    print(f"Published world goal (ENTER): {goal_world.tolist()}")


# ---------------------------------------------------------------------


def _fmt_xyz(label: str, v: np.ndarray) -> list[str]:
    v = np.asarray(v, dtype=np.float64).ravel()

    return [
        label,
        f"x={v[0]:+.4f}  y={v[1]:+.4f}  z={v[2]:+.4f}",
    ]


def _render_text_band(width: int, height: int, lines: list[str]) -> np.ndarray:
    band = np.full((height, width, 3), (28, 28, 32), dtype=np.uint8)

    y = int(22 * 3.0 * _TEXT_SIZE_MULT)

    font = cv2.FONT_HERSHEY_SIMPLEX

    approx_char_px = int(8 * 3.0 * _TEXT_SIZE_MULT)

    max_chars = max(12, (width - 24) // max(1, approx_char_px))

    for line in lines:

        if not line:
            y += _TEXT_EMPTY_SKIP
            continue

        disp = line if len(line) <= max_chars else line[: max_chars - 3] + "..."

        cv2.putText(
            band,
            disp,
            (16, y),
            font,
            _TEXT_FONT_SCALE,
            (230, 230, 235),
            _TEXT_THICKNESS,
            cv2.LINE_AA,
        )

        y += _TEXT_LINE_STEP

        if y > height - int(12 * _TEXT_SIZE_MULT):
            break

    return band


def _gemini_placeholder_panel(h: int, w: int) -> np.ndarray:
    panel = np.full((h, w, 3), (48, 48, 52), dtype=np.uint8)

    cv2.putText(
        panel,
        "Press SPACE",
        (max(10, w // 2 - 110), h // 2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (180, 180, 200),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        "Gemini + depth",
        (max(10, w // 2 - 130), h // 2 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (140, 140, 160),
        1,
        cv2.LINE_AA,
    )

    return panel


def _save_overlay(latched: np.ndarray | None) -> None:
    if latched is None:
        print("Nothing to save yet.")
        return

    fname = f"gemini_{time.strftime('%Y%m%d_%H%M%S')}.png"

    cv2.imwrite(fname, latched)

    print(f"Saved {fname}")


# ---------------------------------------------------------------------


def run_live(args: argparse.Namespace, prompt: str, redis_client) -> int:

    client = gp.make_genai_client(gp.resolve_api_key())

    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")

    pipeline = None

    try:
        pipeline, align, depth_scale, color_intrinsics = (
            rs_cam.start_realsense(
                args.width,
                args.height,
                args.fps,
                args.warmup_frames,
                args.timeout_ms,
            )
        )

    except Exception as e:
        print(f"RealSense startup failed: {e}", file=sys.stderr)
        return 1

    win = "Gemini vision"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(win, 1280, 720)

    latched_overlay = None

    latched_goal_world = None

    latched_goal_orientation = None

    miss_counter = [0]

    print(
        "SPACE = Gemini grasp + TCP/camera/goal diagnostic log\n"
        "ENTER = publish goal\n"
        "s = save overlay\n"
        "q = quit"
    )

    try:

        while True:

            triple = rs_cam.next_rgbd_frame(
                pipeline,
                align,
                depth_scale,
                args.timeout_ms,
                miss_counter,
                max_misses=10,
            )

            if triple is None:
                continue

            color_bgr, depth_m, depth_vis = triple

            h, w = color_bgr.shape[:2]

            gemini_panel = (
                latched_overlay.copy()
                if latched_overlay is not None
                else _gemini_placeholder_panel(h, w)
            )

            top_row = np.hstack((
                color_bgr,
                depth_vis,
                gemini_panel,
            ))

            if latched_goal_world is None:
                goal_lines = [
                    "No target latched",
                    "Press SPACE",
                ]
            else:
                goal_lines = _fmt_xyz(
                    "Latched world goal (m)",
                    latched_goal_world,
                )

            bottom_row = _render_text_band(
                top_row.shape[1],
                TEXT_BAND_HEIGHT,
                goal_lines,
            )

            composite = np.vstack((top_row, bottom_row))

            cv2.imshow(win, composite)

            key = cv2.waitKey(1) & 0xFF

            # ---------------------------------------------------------

            if key in (ord("q"), 27):
                break

            # ---------------------------------------------------------

            if key == ord(" "):

                overlay, point_camera = gp.query_color_depth_overlay(
                        client,
                        args.model,
                        prompt,
                        args.temperature,
                        color_bgr,
                        depth_m,
                        color_intrinsics,
                        args.depth_patch_radius,
                    )

                if overlay is not None:
                    latched_overlay = overlay

                if point_camera is not None:

                    T_base_flange = read_T_base_flange(
                        redis_client,
                        args.endeffector_transform_key,
                    )

                    if T_base_flange is None:
                        print(
                            "Could not read flange pose from Redis key "
                            f"{args.endeffector_transform_key!r}."
                        )
                        continue

                    raw_goal_world = camera_point_to_base(
                        T_base_flange,
                        point_camera,
                    )
                    offset = goal_offset_world(args)
                    latched_goal_world = raw_goal_world + offset

                    latched_goal_orientation = read_current_orientation(
                        redis_client
                    )

                    tcp_pose = read_current_tcp_world(redis_client)
                    tcp_pos = tcp_pose[0] if tcp_pose is not None else None
                    log_tcp_camera_goal_diagnostic(
                        point_camera=np.asarray(point_camera, dtype=np.float64),
                        latched_goal_world=latched_goal_world,
                        T_base_flange=T_base_flange,
                        tcp_pos=tcp_pos,
                        raw_goal_world=raw_goal_world,
                        goal_offset=offset,
                    )

            # ---------------------------------------------------------

            elif key in (10, 13):

                if latched_goal_world is None:
                    print("Press SPACE first.")
                    continue

                publish_opensai_cartesian(
                    redis_client,
                    latched_goal_world,
                    latched_goal_orientation,
                )

            # ---------------------------------------------------------

            elif key == ord("s"):
                _save_overlay(latched_overlay)

    finally:

        if pipeline is not None:
            pipeline.stop()

        cv2.destroyAllWindows()

    return 0


# ---------------------------------------------------------------------


def main() -> int:

    args = parse_args()

    redis_client = _try_redis(
        args.redis_host,
        args.redis_port,
    )

    if redis_client is None:
        return 1

    err = validate_config(redis_client)

    if err is not None:
        return err

    prompt = gp.build_prompt(
        args.object,
        args.prompt,
    )

    off = goal_offset_world(args)
    if np.linalg.norm(off) > 0:
        print(
            "Brute-force goal offset enabled (world frame, m): "
            f"[{off[0]:+.4f}, {off[1]:+.4f}, {off[2]:+.4f}]  "
            "(+X forward, +Z up; use --no-goal-offset to disable)"
        )
    else:
        print("Brute-force goal offset: disabled.")

    return run_live(
        args,
        prompt,
        redis_client,
    )


if __name__ == "__main__":
    sys.exit(main())