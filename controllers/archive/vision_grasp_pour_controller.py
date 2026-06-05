#!/usr/bin/env python3
"""Gemini vision → grasp at detected pose → pour + return (OpenSai Franka).

Two run modes, both ENTER-gated:

- **headless** (default): no OpenCV window. ENTER on stdin runs Gemini on
  the first press, then advances one robot phase per subsequent press.
  Type ``q`` + ENTER to quit. Each Gemini detection is saved as
  ``ZitiBot/logs/gemini_response.png`` (RGB-with-overlay ⊕ depth).

- **UI** (``--ui``): live RGB + depth + Gemini overlay window.

  - **SPACE** — Gemini two rim points + depth → latch grasp pose/orientation
  - **ENTER** — advance one robot phase
  - **q** — quit

Robot sequence after latch (both modes):

1. ENTER — move above detected grasp (+ ``--approach-dz``)
2. ENTER — descend to grasp contact pose
3. ENTER — pregrasp + force close
4. ENTER — move to fixed pour pose
5. ENTER — start pour slerp
6. ENTER — return orientation (auto-completes)
7. ENTER — retract from pour
8. ENTER — move back to grasp pose
9. ENTER — open gripper

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
RealSense, and ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.

Usage::

  # Default headless mode
  ./ZitiBot/launch_zitibot_full.sh controllers/vision_grasp_pour_controller.py

  # With UI
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_grasp_pour_controller.py -- \\
    --ui --object bowl --pour-pose 0.52 0.12 0.63 --approach-dz 0.15
"""

from __future__ import annotations

import argparse
import enum
import json
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

from grasp_and_pour_controller import (
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_GRIPPER_FORCE,
    DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    DEFAULT_GRIPPER_PREGRASP_WIDTH,
    DEFAULT_GRIPPER_GRASP_SETTLE_S,
    DEFAULT_GRIPPER_SPEED,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    GRASP_ORIENTATION,
    MotionParams,
    OrientationSlerpState,
    POUR_POSITION,
    _do_grasp_object,
    _do_open_gripper,
    _publish_cartesian,
    _start_orientation_slerp,
    _tick_orientation_slerp,
    pour_orientation_end,
    read_current_ee_world,
    resolve_gripper_open_width,
    validate_config,
    _try_redis,
)
from vision import gemini_pointing as gp
from vision import realsense_rgbd as rs_cam

# Camera → flange calibration (must match hand-eye setup).
T_FLANGE_CAMERA = np.array(
    [
        [0.0, -1.0, 0.0, 0.053],
        [1.0, 0.0, 0.0, -0.009],
        [0.0, 0.0, 1.0, 0.019],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# World-frame tweak applied to the vision contact point (not the approach height).
ENABLE_GRASP_OFFSET = True
DEFAULT_GRASP_OFFSET_M = np.array([0.03, 0.0, 0.09], dtype=np.float64)

ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"

# Default location for the saved Gemini annotated composite (RGB+overlay ⊕ depth).
DEFAULT_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_GEMINI_RESPONSE_PATH = DEFAULT_LOGS_DIR / "gemini_response.png"

# UI rendering
_TEXT_SIZE_MULT = 0.75 * 0.75
_TEXT_FONT_SCALE = 0.48 * 3.0 * _TEXT_SIZE_MULT
_TEXT_LINE_STEP = int(22 * 3.0 * _TEXT_SIZE_MULT)
_TEXT_THICKNESS = 2
_TEXT_EMPTY_SKIP = int(18 * 3.0 * _TEXT_SIZE_MULT)
TEXT_BAND_HEIGHT = int((int(120 * 3.0) + 180) * _TEXT_SIZE_MULT) + 80


class Phase(enum.Enum):
    VISION_READY = "VISION_READY"
    VISION_LATCHED = "VISION_LATCHED"
    ABOVE_GRASP = "ABOVE_GRASP"
    AT_GRASP = "AT_GRASP"
    GRASPED = "GRASPED"
    MOVING_TO_POUR = "MOVING_TO_POUR"
    AT_POUR = "AT_POUR"
    POURING = "POURING"
    POURED = "POURED"
    RETURNING = "RETURNING"
    RETURNED = "RETURNED"
    RETRACTING = "RETRACTING"
    RETRACTED = "RETRACTED"
    MOVING_TO_RELEASE = "MOVING_TO_RELEASE"
    AT_RELEASE = "AT_RELEASE"
    DONE = "DONE"


@dataclass
class LatchedTarget:
    """Vision-derived grasp target (world frame)."""

    grasp_pos: np.ndarray
    above_grasp_pos: np.ndarray
    grasp_ori: np.ndarray
    orientation_source: str
    rim_yaw_deg: float | None
    rim_yaw_applied: bool


def read_T_base_flange(redis_client, key: str) -> np.ndarray | None:
    try:
        raw = redis_client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        T = np.array(json.loads(raw), dtype=np.float64)
        if T.shape != (4, 4):
            T = T.reshape(4, 4)
        return T
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def read_current_orientation(redis_client) -> np.ndarray | None:
    pose = read_current_ee_world(redis_client)
    if pose is None:
        return None
    return pose[1]


def camera_point_to_base(T_base_flange: np.ndarray, point_camera: np.ndarray) -> np.ndarray:
    p_cam_h = np.ones(4, dtype=np.float64)
    p_cam_h[:3] = np.asarray(point_camera, dtype=np.float64).reshape(3)
    p_base_h = T_base_flange @ T_FLANGE_CAMERA @ p_cam_h
    return p_base_h[:3].copy()


def rim_yaw_rotation_rad(p1_base: np.ndarray, p2_base: np.ndarray) -> float | None:
    """Yaw (rad) in world XY from rim tangent p1→p2."""
    p1 = np.asarray(p1_base, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2_base, dtype=np.float64).reshape(3)
    rim_vec = p2 - p1
    if np.linalg.norm(rim_vec) < 1e-6:
        print("Warning: two rim points are too close")
        return None
    rim_xy = rim_vec.copy()
    rim_xy[2] = 0.0
    rim_xy_norm = np.linalg.norm(rim_xy)
    if rim_xy_norm < 1e-6:
        print("Warning: rim is vertical (Z-aligned)")
        return None
    rim_xy /= rim_xy_norm
    return float(np.arctan2(rim_xy[1], rim_xy[0]))


def apply_rim_yaw_to_orientation(
    base_orientation: np.ndarray,
    p1_base: np.ndarray,
    p2_base: np.ndarray,
    *,
    orientation_source: str,
) -> tuple[np.ndarray, float | None, bool]:
    """Apply rim yaw to a base orientation matrix."""
    base = np.asarray(base_orientation, dtype=np.float64).reshape(3, 3).copy()
    yaw = rim_yaw_rotation_rad(p1_base, p2_base)
    if yaw is None:
        return base, None, False

    c, s = np.cos(yaw), np.sin(yaw)
    R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    print(
        f"  Rim yaw: {np.degrees(yaw):+.2f} deg  "
        f"(orientation_source={orientation_source})"
    )

    if orientation_source == "current":
        # Local/tool Z rotation (matches vision_controller_new).
        out = base @ R_z
    else:
        # World Z rotation on fixed tool-down grasp orientation.
        out = R_z @ base

    return out, yaw, True


def latch_grasp_target(
    redis_client,
    valid_points_base: list[np.ndarray],
    *,
    grasp_offset: np.ndarray,
    approach_dz_m: float,
    orientation_source: str,
) -> LatchedTarget | None:
    if not valid_points_base:
        return None

    raw_contact = valid_points_base[0]
    grasp_pos = raw_contact + np.asarray(grasp_offset, dtype=np.float64).reshape(3)
    above_grasp_pos = grasp_pos + np.array([0.0, 0.0, approach_dz_m], dtype=np.float64)

    if orientation_source == "current":
        base_ori = read_current_orientation(redis_client)
        if base_ori is None:
            print("Warning: no current EE orientation; falling back to fixed GRASP_ORIENTATION")
            base_ori = GRASP_ORIENTATION.copy()
            orientation_source = "fixed"
        else:
            base_ori = base_ori.copy()
    else:
        base_ori = GRASP_ORIENTATION.copy()

    rim_yaw_deg: float | None = None
    rim_applied = False
    grasp_ori = base_ori.copy()

    if len(valid_points_base) >= 2:
        grasp_ori, yaw, rim_applied = apply_rim_yaw_to_orientation(
            base_ori,
            valid_points_base[0],
            valid_points_base[1],
            orientation_source=orientation_source,
        )
        if yaw is not None:
            rim_yaw_deg = float(np.degrees(yaw))

    return LatchedTarget(
        grasp_pos=grasp_pos,
        above_grasp_pos=above_grasp_pos,
        grasp_ori=grasp_ori,
        orientation_source=orientation_source,
        rim_yaw_deg=rim_yaw_deg,
        rim_yaw_applied=rim_applied,
    )


def _phase_hint(phase: Phase, *, headless: bool = False) -> str:
    detect_key = "ENTER" if headless else "SPACE"
    hints = {
        Phase.VISION_READY: f"{detect_key} = detect rim with Gemini",
        Phase.VISION_LATCHED: "ENTER = move above detected grasp",
        Phase.ABOVE_GRASP: "ENTER = descend to grasp contact",
        Phase.AT_GRASP: "ENTER = pregrasp + close gripper",
        Phase.GRASPED: "ENTER = move to pour pose",
        Phase.MOVING_TO_POUR: "ENTER when arm settled at pour pose",
        Phase.AT_POUR: "ENTER = start pour (tilt)",
        Phase.POURING: "Pouring… (auto-completes)",
        Phase.POURED: "ENTER = return orientation",
        Phase.RETURNING: "Returning… (auto-completes)",
        Phase.RETURNED: "ENTER = retract from pour",
        Phase.RETRACTING: "ENTER when retracted",
        Phase.RETRACTED: "ENTER = move back to grasp pose",
        Phase.MOVING_TO_RELEASE: "ENTER when at release pose",
        Phase.AT_RELEASE: "ENTER = open gripper",
        Phase.DONE: "Done — q to quit",
    }
    return hints.get(phase, "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision-guided grasp + pour (OpenSai Franka).")
    p.add_argument(
        "--ui",
        action="store_true",
        help="Show the OpenCV window. Default is headless (stdin ENTER).",
    )
    p.add_argument(
        "--gemini-response-path",
        default=str(DEFAULT_GEMINI_RESPONSE_PATH),
        help="Path to save the annotated RGB ⊕ depth composite after each Gemini call.",
    )
    p.add_argument("--object", default="bowl")
    p.add_argument("--prompt", default=None)
    p.add_argument("--depth-patch-radius", type=int, default=2)
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--endeffector-transform-key", default=ENDEFFECTOR_TRANSFORM_KEY)
    p.add_argument("--no-grasp-offset", action="store_true")
    p.add_argument(
        "--grasp-offset-x",
        type=float,
        default=float(DEFAULT_GRASP_OFFSET_M[0]),
        help="World-frame offset on latched grasp contact point (m).",
    )
    p.add_argument("--grasp-offset-y", type=float, default=float(DEFAULT_GRASP_OFFSET_M[1]))
    p.add_argument("--grasp-offset-z", type=float, default=float(DEFAULT_GRASP_OFFSET_M[2]))
    p.add_argument(
        "--orientation-source",
        choices=("fixed", "current"),
        default="fixed",
        help=(
            "Grasp orientation base: fixed=GRASP_ORIENTATION (+ rim yaw in world Z); "
            "current=live EE orientation (+ rim yaw in tool Z, legacy)."
        ),
    )
    p.add_argument("--model", default=gp.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--timeout-ms", type=int, default=10000)
    p.add_argument(
        "--pour-pose",
        nargs=3,
        type=float,
        default=POUR_POSITION.tolist(),
        help="Pour pose XYZ (m).",
    )
    p.add_argument("--approach-dz", type=float, default=DEFAULT_APPROACH_DZ_M)
    p.add_argument("--pour-tilt-deg", type=float, default=DEFAULT_POUR_TILT_DEG)
    p.add_argument("--pour-axis", choices=("x", "y"), default=DEFAULT_POUR_AXIS)
    p.add_argument("--tilt-duration", type=float, default=DEFAULT_TILT_DURATION_S)
    p.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED)
    p.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE)
    p.add_argument(
        "--gripper-pregrasp-width",
        type=float,
        default=DEFAULT_GRIPPER_PREGRASP_WIDTH,
    )
    p.add_argument(
        "--gripper-pregrasp-settle",
        type=float,
        default=DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    )
    p.add_argument(
        "--gripper-grasp-settle",
        type=float,
        default=DEFAULT_GRIPPER_GRASP_SETTLE_S,
    )
    return p.parse_args()


def grasp_offset_world(args: argparse.Namespace) -> np.ndarray:
    if args.no_grasp_offset or not ENABLE_GRASP_OFFSET:
        return np.zeros(3, dtype=np.float64)
    return np.array(
        [args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z],
        dtype=np.float64,
    )


def _fmt_xyz(label: str, v: np.ndarray) -> list[str]:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return [label, f"x={v[0]:+.4f}  y={v[1]:+.4f}  z={v[2]:+.4f}"]


def _status_lines(phase: Phase, target: LatchedTarget | None) -> list[str]:
    if target is None:
        return ["No target latched", "SPACE = detect with Gemini"]
    lines = [
        f"Phase: {phase.value}",
        *_fmt_xyz("Grasp contact", target.grasp_pos),
        *_fmt_xyz("Above grasp", target.above_grasp_pos),
        f"Ori: {target.orientation_source}"
        + (
            f"  rim_yaw={target.rim_yaw_deg:+.1f} deg"
            if target.rim_yaw_applied and target.rim_yaw_deg is not None
            else "  (no rim yaw)"
        ),
    ]
    return lines


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


def _save_gemini_response(
    overlay: np.ndarray | None,
    depth_vis: np.ndarray,
    save_path: Path,
) -> None:
    """Write [overlay | depth_vis] composite to ``save_path`` (BGR PNG)."""
    if overlay is None:
        return
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ov = overlay
        dv = depth_vis
        if ov.shape[:2] != dv.shape[:2]:
            dv = cv2.resize(dv, (ov.shape[1], ov.shape[0]))
        composite = np.hstack((ov, dv))
        cv2.imwrite(str(save_path), composite)
        print(f"  Saved Gemini response: {save_path}")
    except Exception as e:
        print(f"  Failed to save Gemini response to {save_path}: {e}", file=sys.stderr)


def do_gemini_capture(
    triple: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    redis_client,
    args: argparse.Namespace,
    motion: MotionParams,
    gemini_client,
    prompt: str,
    color_intrinsics,
    save_path: Path,
) -> tuple[LatchedTarget | None, np.ndarray | None]:
    """Run Gemini on the captured frame, save the annotated image, and latch.

    Returns ``(latched_target_or_None, overlay_image_or_None)``. The overlay
    is suitable for display in the UI side panel; ``None`` if Gemini failed.
    """
    color_bgr, depth_m, depth_vis = triple

    overlay, points_camera = gp.query_color_depth_overlay(
        gemini_client,
        args.model,
        prompt,
        args.temperature,
        color_bgr,
        depth_m,
        color_intrinsics,
        args.depth_patch_radius,
    )

    _save_gemini_response(overlay, depth_vis, save_path)

    if points_camera is None or len(points_camera) == 0:
        return None, overlay

    T_base_flange = read_T_base_flange(redis_client, args.endeffector_transform_key)
    if T_base_flange is None:
        print(
            "Could not read flange pose from Redis key "
            f"{args.endeffector_transform_key!r}."
        )
        return None, overlay

    valid_points_base = []
    for p_cam in points_camera:
        if p_cam is not None:
            valid_points_base.append(camera_point_to_base(T_base_flange, p_cam))

    if not valid_points_base:
        print("No valid 3D points detected (depth issues)")
        return None, overlay

    offset = grasp_offset_world(args)
    target = latch_grasp_target(
        redis_client,
        valid_points_base,
        grasp_offset=offset,
        approach_dz_m=motion.approach_dz_m,
        orientation_source=args.orientation_source,
    )
    if target is None:
        return None, overlay

    print(f"Latched grasp contact: {target.grasp_pos.tolist()}")
    print(f"Latched above grasp:  {target.above_grasp_pos.tolist()}")
    return target, overlay


def _grab_fresh_frame(
    pipeline,
    align,
    depth_scale: float,
    timeout_ms: int,
    miss_counter: list[int],
    *,
    drain: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pull frames until we get one (or give up after ``drain`` tries)."""
    triple = None
    for _ in range(max(1, drain)):
        triple = rs_cam.next_rgbd_frame(
            pipeline, align, depth_scale, timeout_ms, miss_counter, max_misses=10
        )
        if triple is not None:
            break
    return triple


def _advance_phase(
    phase: Phase,
    target: LatchedTarget,
    slerp: OrientationSlerpState | None,
    poured_ori: np.ndarray | None,
    *,
    redis_client,
    motion: MotionParams,
    args: argparse.Namespace,
) -> tuple[Phase, OrientationSlerpState | None, np.ndarray | None]:
    """Run a single ENTER-press transition. Returns updated state."""
    if phase == Phase.VISION_LATCHED:
        print("[1] Move above detected grasp...")
        _publish_cartesian(redis_client, target.above_grasp_pos, target.grasp_ori)
        return Phase.ABOVE_GRASP, slerp, poured_ori

    if phase == Phase.ABOVE_GRASP:
        print("[2] Descend to grasp contact...")
        _publish_cartesian(redis_client, target.grasp_pos, target.grasp_ori)
        return Phase.AT_GRASP, slerp, poured_ori

    if phase == Phase.AT_GRASP:
        print("[3] Pregrasp + grasp...")
        _do_grasp_object(redis_client, motion)
        return Phase.GRASPED, slerp, poured_ori

    if phase == Phase.GRASPED:
        print("[4] Move to pour pose...")
        pour_pos = np.asarray(args.pour_pose, dtype=np.float64)
        _publish_cartesian(redis_client, pour_pos, target.grasp_ori)
        return Phase.MOVING_TO_POUR, slerp, poured_ori

    if phase == Phase.MOVING_TO_POUR:
        return Phase.AT_POUR, slerp, poured_ori

    if phase == Phase.AT_POUR:
        print("[5] Starting pour...")
        pose = read_current_ee_world(redis_client)
        if pose is None:
            return phase, slerp, poured_ori
        _, cur_ori = pose
        R_end = pour_orientation_end(cur_ori, motion.pour_tilt_deg, motion.pour_axis)
        pour_pos = np.asarray(args.pour_pose, dtype=np.float64)
        slerp = _start_orientation_slerp(
            redis_client,
            pour_pos,
            cur_ori,
            R_end,
            time.perf_counter(),
            label=(
                f"  Slerp {motion.pour_tilt_deg:.0f}° about world "
                f"+{motion.pour_axis.upper()}"
            ),
        )
        return Phase.POURING, slerp, poured_ori

    if phase == Phase.POURING:
        return phase, slerp, poured_ori

    if phase == Phase.POURED:
        print("[6] Returning orientation...")
        pour_pos = np.asarray(args.pour_pose, dtype=np.float64)
        slerp = _start_orientation_slerp(
            redis_client,
            pour_pos,
            poured_ori,
            target.grasp_ori,
            time.perf_counter(),
            label="  Slerp back to grasp orientation",
        )
        return Phase.RETURNING, slerp, poured_ori

    if phase == Phase.RETURNING:
        return phase, slerp, poured_ori

    if phase == Phase.RETURNED:
        print("[7] Retracting from pour...")
        pour_pos = np.asarray(args.pour_pose, dtype=np.float64)
        retract_z = max(pour_pos[2], target.grasp_pos[2] + motion.approach_dz_m)
        retract_pos = pour_pos.copy()
        retract_pos[2] = retract_z
        _publish_cartesian(redis_client, retract_pos, target.grasp_ori)
        return Phase.RETRACTING, slerp, poured_ori

    if phase == Phase.RETRACTING:
        return Phase.RETRACTED, slerp, poured_ori

    if phase == Phase.RETRACTED:
        print("[8] Move back to grasp pose...")
        _publish_cartesian(redis_client, target.grasp_pos, target.grasp_ori)
        return Phase.MOVING_TO_RELEASE, slerp, poured_ori

    if phase == Phase.MOVING_TO_RELEASE:
        return Phase.AT_RELEASE, slerp, poured_ori

    if phase == Phase.AT_RELEASE:
        print("[9] Open gripper...")
        _do_open_gripper(redis_client, motion)
        return Phase.DONE, slerp, poured_ori

    return phase, slerp, poured_ori


def _drive_slerp_to_completion(
    phase: Phase,
    slerp: OrientationSlerpState | None,
    poured_ori: np.ndarray | None,
    *,
    redis_client,
    motion: MotionParams,
) -> tuple[Phase, OrientationSlerpState | None, np.ndarray | None]:
    """In headless mode, drive POURING/RETURNING slerps to completion."""
    while phase in (Phase.POURING, Phase.RETURNING) and slerp is not None:
        now = time.perf_counter()
        if _tick_orientation_slerp(redis_client, slerp, motion, now):
            if phase == Phase.POURING:
                poured_ori = slerp.R_end.copy()
                phase = Phase.POURED
                slerp = None
                print("Pour complete.")
            else:
                phase = Phase.RETURNED
                slerp = None
                print("Return orientation complete.")
        else:
            time.sleep(0.005)
    return phase, slerp, poured_ori


def run_headless(
    args: argparse.Namespace,
    prompt: str,
    redis_client,
    motion: MotionParams,
) -> int:
    """Default mode: stdin-driven, no OpenCV window."""
    gemini_client = gp.make_genai_client(gp.resolve_api_key())
    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")
    print(f"Orientation source: {args.orientation_source}")
    print("Mode: headless (no UI)")

    pipeline = None
    try:
        pipeline, align, depth_scale, color_intrinsics = rs_cam.start_realsense(
            args.width,
            args.height,
            args.fps,
            args.warmup_frames,
            args.timeout_ms,
        )
    except Exception as e:
        print(f"RealSense startup failed: {e}", file=sys.stderr)
        return 1

    save_path = Path(args.gemini_response_path).expanduser().resolve()
    print(f"Gemini response will be saved to: {save_path}")
    print("Press ENTER to advance phases (or type 'q'+ENTER to quit).")
    print()

    phase = Phase.VISION_READY
    target: LatchedTarget | None = None
    slerp: OrientationSlerpState | None = None
    poured_ori: np.ndarray | None = None
    miss_counter = [0]

    try:
        while phase != Phase.DONE:
            print(_phase_hint(phase, headless=True))
            try:
                cmd = input("> ").strip().lower()
            except EOFError:
                break

            if cmd == "q":
                break

            if phase == Phase.VISION_READY:
                triple = _grab_fresh_frame(
                    pipeline, align, depth_scale, args.timeout_ms, miss_counter
                )
                if triple is None:
                    print("Failed to grab RealSense frame; press ENTER to retry.")
                    continue

                target, _overlay = do_gemini_capture(
                    triple,
                    redis_client=redis_client,
                    args=args,
                    motion=motion,
                    gemini_client=gemini_client,
                    prompt=prompt,
                    color_intrinsics=color_intrinsics,
                    save_path=save_path,
                )
                if target is None:
                    print("No latched target; press ENTER to retry detection.")
                    continue
                phase = Phase.VISION_LATCHED
                continue

            if target is None:
                print("No latched target; press ENTER to retry detection.")
                continue

            phase, slerp, poured_ori = _advance_phase(
                phase,
                target,
                slerp,
                poured_ori,
                redis_client=redis_client,
                motion=motion,
                args=args,
            )

            phase, slerp, poured_ori = _drive_slerp_to_completion(
                phase,
                slerp,
                poured_ori,
                redis_client=redis_client,
                motion=motion,
            )

        print("Sequence complete.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if pipeline is not None:
            pipeline.stop()

    return 0


def run_live(
    args: argparse.Namespace,
    prompt: str,
    redis_client,
    motion: MotionParams,
) -> int:
    gemini_client = gp.make_genai_client(gp.resolve_api_key())
    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")
    print(f"Orientation source: {args.orientation_source}")
    print("Mode: UI (OpenCV)")

    pipeline = None
    try:
        pipeline, align, depth_scale, color_intrinsics = rs_cam.start_realsense(
            args.width,
            args.height,
            args.fps,
            args.warmup_frames,
            args.timeout_ms,
        )
    except Exception as e:
        print(f"RealSense startup failed: {e}", file=sys.stderr)
        return 1

    save_path = Path(args.gemini_response_path).expanduser().resolve()
    print(f"Gemini response will be saved to: {save_path}")

    win = "Vision Grasp Pour"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    phase = Phase.VISION_READY
    latched_overlay = None
    target: LatchedTarget | None = None
    miss_counter = [0]
    slerp: OrientationSlerpState | None = None
    poured_ori: np.ndarray | None = None

    print("Keys: SPACE = detect | ENTER = advance | q = quit")
    print(_phase_hint(phase))

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

            top_row = np.hstack((color_bgr, depth_vis, gemini_panel))
            bottom_row = _render_text_band(
                top_row.shape[1],
                TEXT_BAND_HEIGHT,
                _status_lines(phase, target),
            )
            composite = np.vstack((top_row, bottom_row))
            cv2.imshow(win, composite)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if phase in (Phase.POURING, Phase.RETURNING) and slerp is not None:
                now = time.perf_counter()
                if _tick_orientation_slerp(redis_client, slerp, motion, now):
                    if phase == Phase.POURING:
                        poured_ori = slerp.R_end.copy()
                        phase = Phase.POURED
                        slerp = None
                        print("Pour complete.")
                    else:
                        phase = Phase.RETURNED
                        slerp = None
                        print("Return orientation complete.")
                    print(_phase_hint(phase))

            if key == ord(" "):
                triple = (color_bgr, depth_m, depth_vis)
                target, overlay = do_gemini_capture(
                    triple,
                    redis_client=redis_client,
                    args=args,
                    motion=motion,
                    gemini_client=gemini_client,
                    prompt=prompt,
                    color_intrinsics=color_intrinsics,
                    save_path=save_path,
                )
                if overlay is not None:
                    latched_overlay = overlay
                if target is None:
                    continue
                phase = Phase.VISION_LATCHED
                print(_phase_hint(phase))

            elif key in (10, 13):
                if phase == Phase.VISION_READY:
                    print("Press SPACE first to detect with Gemini.")
                    continue

                if target is None:
                    print("No latched target; press SPACE first.")
                    continue

                if phase == Phase.DONE:
                    print("Sequence complete — q to quit")
                    continue

                phase, slerp, poured_ori = _advance_phase(
                    phase,
                    target,
                    slerp,
                    poured_ori,
                    redis_client=redis_client,
                    motion=motion,
                    args=args,
                )
                if phase not in (Phase.POURING, Phase.RETURNING):
                    print(_phase_hint(phase))

    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    redis_client = _try_redis(args.redis_host, args.redis_port)
    if redis_client is None:
        return 1

    err = validate_config(redis_client)
    if err is not None:
        return err

    prompt = gp.build_prompt(args.object, args.prompt)

    off = grasp_offset_world(args)
    if np.linalg.norm(off) > 0:
        print(
            f"Grasp offset on contact point (world, m): "
            f"[{off[0]:+.4f}, {off[1]:+.4f}, {off[2]:+.4f}]"
        )
    else:
        print("Grasp offset: disabled")

    motion = MotionParams(
        approach_dz_m=args.approach_dz,
        pour_tilt_deg=args.pour_tilt_deg,
        pour_axis=args.pour_axis,
        tilt_duration_s=args.tilt_duration,
        gripper_open_width=None,
        gripper_pregrasp_width=args.gripper_pregrasp_width,
        gripper_close_width=0.0,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        gripper_pregrasp_settle_s=args.gripper_pregrasp_settle,
        gripper_grasp_settle_s=args.gripper_grasp_settle,
    )

    open_w = resolve_gripper_open_width(redis_client, None)
    motion = MotionParams(
        approach_dz_m=motion.approach_dz_m,
        pour_tilt_deg=motion.pour_tilt_deg,
        pour_axis=motion.pour_axis,
        tilt_duration_s=motion.tilt_duration_s,
        gripper_open_width=open_w,
        gripper_pregrasp_width=motion.gripper_pregrasp_width,
        gripper_close_width=motion.gripper_close_width,
        gripper_speed=motion.gripper_speed,
        gripper_force=motion.gripper_force,
        gripper_pregrasp_settle_s=motion.gripper_pregrasp_settle_s,
        gripper_grasp_settle_s=motion.gripper_grasp_settle_s,
    )
    print(
        f"Motion: approach_dz={motion.approach_dz_m} m, pour={motion.pour_tilt_deg:.0f}°, "
        f"gripper_open={open_w:.4f} m"
    )

    if args.ui:
        return run_live(args, prompt, redis_client, motion)
    return run_headless(args, prompt, redis_client, motion)


if __name__ == "__main__":
    sys.exit(main())
