#!/usr/bin/env python3
"""
Barebones click-to-move controller.

Pops up an RGB window + a depth window from a RealSense. Click a pixel on the
RGB image; the script deprojects that pixel (using aligned depth) into the
camera frame, transforms it into the Franka base frame via the live
T_end_effector (read from Redis) and a fixed flange->camera extrinsic, then
sends an OpenSai Cartesian goal so the arm moves to that exact point.

This file is intentionally self-contained: it does NOT import any of the
existing ZitiBot controllers / vision / core modules. The transform chain
mirrors the production one:

    p_base = T_base_flange @ T_FLANGE_CAMERA @ p_camera

Controls
--------
* Left-click on the RGB window : select a target pixel (live preview updates).
* g                            : send the arm to the currently selected target.
* c                            : clear the current selection.
* q  or  ESC                   : quit.

Safety
------
Clicking only PREVIEWS the target. The arm does not move until you press ``g``.
Invalid depth (0 / out of [--min-depth, --max-depth]) is rejected. Orientation
is held at the current EE orientation, so only the position changes.

Prerequisites
-------------
* redis-server running.
* Franka OpenSai driver running and publishing T_end_effector, plus the
  cartesian_controller available.
* RealSense connected (no other process holding it).

Usage
-----
::

    cd /home/tidybot01/OpenSai/ZitiBot/controllers

    # Default: uses the built-in hardcoded fallback extrinsic.
    python click_to_move_controller.py

    # Use the calibrated extrinsic from calibrate_hand_eye.py (.json or .npy):
    python click_to_move_controller.py --calib vision/hand_eye_T_flange_camera.json

    # Gemini mode: press 'h' to ask Gemini for a point on the hand, then 'g' to go:
    python click_to_move_controller.py --gemini

Gemini mode
-----------
With --gemini, pressing ``h`` sends the current RGB frame to Gemini Robotics-ER
and asks for a single point on the requested hand part (``--gemini-part``,
default "the center of the palm of the hand"). The returned point is treated exactly
like a click, except the 3D point is latched from the same RGB-D frame and
T_end_effector pose used for the Gemini query. ``g``/ENTER sends that latched
move target.
Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment / a .env file,
and ``pip install google-genai``.

Dependencies: redis, numpy, pyrealsense2, opencv-python, google-genai (for --gemini)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import redis

# --- Redis keys (OpenSai / FrankaRobot) -------------------------------------
EE_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
GOAL_POS_KEY = (
    "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_position"
)
GOAL_ORI_KEY = (
    "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_orientation"
)
CUR_POS_KEY = (
    "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position"
)
CUR_ORI_KEY = (
    "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation"
)
ACTIVE_CTRL_KEY = "opensai::controllers::FrankaRobot::active_controller_name"
CARTESIAN_CONTROLLER = "cartesian_controller"

# Default hand-eye calibration result (from calibrate_hand_eye.py). Used
# automatically when present; pass --no-calib to ignore it.
DEFAULT_CALIB = Path(__file__).resolve().parent / "vision" / "hand_eye_T_flange_camera.json"

# Fallback flange<-camera extrinsic (camera optical -> flange), same placeholder
# the production pipeline ships with. Override with --calib once you have run
# calibrate_hand_eye.py.
T_FLANGE_CAMERA_HARDCODED_OLD = np.array(
    [
        [0.0, -1.0, 0.0, 0.053401],
        [1.0, 0.0, 0.0, -0.009],
        [0.0, 0.0, 1.0, 0.018930],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

OFFSET = np.array([0.053401, -0.009, 0.018930], dtype=np.float64)



def get_z_rotation_matrix(angle: float) -> np.ndarray:
    return np.array(
        [
            [ np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0.0],
            [ np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.0],
            [ 0.0, 0.0, 1.0]
        ],
        dtype=np.float64
    )

ROT_Z = get_z_rotation_matrix(45)
TRANS_ROT_Z = get_z_rotation_matrix(-45)

TRANSLATION_Z = TRANS_ROT_Z @ OFFSET

T_FLANGE_CAMERA_FALLBACK = np.eye(4, dtype=np.float64)
T_FLANGE_CAMERA_FALLBACK[:3, :3] = ROT_Z
T_FLANGE_CAMERA_FALLBACK[:3, 3] = TRANSLATION_Z


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Barebones click-to-move RGB-D controller.")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--ee-key", default=EE_KEY, help="Redis 4x4 JSON: base <- end-effector.")
    p.add_argument(
        "--calib",
        type=Path,
        default=None,
        help=(
            "Path to flange<-camera extrinsic from calibrate_hand_eye.py "
            "(.json with a 'T_FLANGE_CAMERA' field, or a 4x4 .npy). "
            "When omitted, uses the built-in hardcoded fallback extrinsic. "
            f"Calibration file (if you want it): {DEFAULT_CALIB}"
        ),
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cam-timeout-ms", type=int, default=2000)
    p.add_argument(
        "--patch-radius",
        type=int,
        default=4,
        help="Half-size (px) of the depth patch median-sampled around the click.",
    )
    p.add_argument("--min-depth", type=float, default=0.1, help="Reject depths below this (m).")
    p.add_argument("--max-depth", type=float, default=2.0, help="Reject depths above this (m).")
    p.add_argument(
        "--z-offset",
        type=float,
        default=0.0,
        help="Add this to the target Z in base frame (m); 0 = exact point.",
    )
    p.add_argument(
        "--max-reach",
        type=float,
        default=0.8,
        help="Reject targets farther than this from the base origin (m). Safety guard.",
    )
    p.add_argument(
        "--min-reach",
        type=float,
        default=0.2,
        help="Reject targets closer than this to the base origin (m). Safety guard.",
    )
    p.add_argument(
        "--min-z",
        type=float,
        default=0.0,
        help="Reject targets below this base-frame Z (m). Safety guard.",
    )
    p.add_argument(
        "--no-move",
        action="store_true",
        help="Dry run: never publish goals, just print the computed target.",
    )
    p.add_argument(
        "--gemini",
        action="store_true",
        help="Enable Gemini mode: press 'h' to query a point on the hand.",
    )
    p.add_argument(
        "--gemini-model",
        default="gemini-robotics-er-1.6-preview",
        help="Gemini Robotics-ER model name.",
    )
    p.add_argument(
        "--gemini-part",
        default="the center of the palm of the hand",
        help="Which part of the hand Gemini should point at.",
    )
    p.add_argument(
        "--gemini-prompt",
        default=None,
        help="Full custom Gemini prompt (overrides --gemini-part).",
    )
    p.add_argument(
        "--gemini-timeout-s",
        type=float,
        default=8.0,
        help="Per-call Gemini deadline (seconds).",
    )
    return p.parse_args()


def load_extrinsic(calib: Path | None) -> np.ndarray:
    if calib is None:
        print("[extrinsic] using built-in hardcoded fallback T_FLANGE_CAMERA.")
        return T_FLANGE_CAMERA_FALLBACK.copy()
    calib = Path(calib)
    if not calib.exists():
        print(f"[extrinsic] {calib} not found; using built-in fallback T_FLANGE_CAMERA.")
        return T_FLANGE_CAMERA_FALLBACK.copy()
    if calib.suffix.lower() == ".json":
        with open(calib, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "T_FLANGE_CAMERA" not in data:
            raise SystemExit(f"{calib} has no 'T_FLANGE_CAMERA' field.")
        T = np.array(data["T_FLANGE_CAMERA"], dtype=np.float64)
    else:
        T = np.load(calib).astype(np.float64)
    if T.shape != (4, 4):
        raise SystemExit(f"--calib must be a 4x4 matrix, got {T.shape}")
    print(f"[extrinsic] loaded {calib}")
    return T


def connect_redis(host: str, port: int) -> redis.Redis:
    client = redis.Redis(host=host, port=port, decode_responses=True)
    client.ping()
    return client


# --- Gemini (self-contained; only used with --gemini) -----------------------
def resolve_gemini_key() -> str:
    import os

    try:
        from dotenv import load_dotenv

        here = Path(__file__).resolve().parent
        for cand in (here / ".env", here.parent / ".env", here.parent.parent / ".env"):
            if cand.is_file():
                load_dotenv(cand)
                break
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("No Gemini API key found (set GEMINI_API_KEY or GOOGLE_API_KEY).")
    return key


def make_gemini_client():
    from google import genai

    return genai.Client(api_key=resolve_gemini_key())


def build_hand_prompt(part: str, custom: str | None) -> str:
    if custom:
        return custom
    return (
        f"In the image, locate the human hand and point at {part}. "
        f"Return exactly ONE point. Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{part}"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


def gemini_point(
    client,
    model: str,
    color_bgr: np.ndarray,
    prompt: str,
    timeout_s: float,
) -> tuple[int, int] | None:
    """Query Gemini for one normalized [y, x] point; return pixel (u, v) or None."""
    import concurrent.futures as cf
    import re

    from google.genai import types

    ok, png = cv2.imencode(".png", color_bgr)
    if not ok:
        return None

    def _call() -> str:
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=png.tobytes(), mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.5,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return resp.text or ""

    try:
        with cf.ThreadPoolExecutor(max_workers=1) as ex:
            text = ex.submit(_call).result(timeout=timeout_s)
    except cf.TimeoutError:
        print(f"[gemini] timed out after {timeout_s:.1f}s.")
        return None
    except Exception as e:
        print(f"[gemini] call failed: {e}")
        return None

    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    bracketed = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.S)
    if bracketed:
        candidates.append(bracketed.group(0).strip())

    h, w = color_bgr.shape[:2]
    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list) or not data:
            continue
        pt = data[0].get("point") if isinstance(data[0], dict) else None
        if not pt or len(pt) != 2:
            continue
        y_norm, x_norm = float(pt[0]), float(pt[1])
        u = int(round(x_norm / 1000.0 * (w - 1)))
        v = int(round(y_norm / 1000.0 * (h - 1)))
        u = max(0, min(w - 1, u))
        v = max(0, min(h - 1, v))
        return u, v
    print(f"[gemini] could not parse a point from response: {text!r}")
    return None


def read_matrix(client: redis.Redis, key: str, shape: tuple[int, ...]) -> np.ndarray | None:
    try:
        raw = client.get(key)
        if raw is None:
            return None
        arr = np.array(json.loads(raw), dtype=np.float64)
        if arr.size != int(np.prod(shape)):
            return None
        return arr.reshape(shape)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def start_realsense(args: argparse.Namespace):
    """Start aligned color+depth; return (pipeline, align, color_intrinsics)."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = color_profile.get_intrinsics()
    print(f"[realsense] depth scale {depth_scale:.6f} m/unit, warming up...")
    for _ in range(30):
        pipeline.try_wait_for_frames(args.cam_timeout_ms)
    print("[realsense] ready.")
    return pipeline, align, depth_scale, color_intrinsics


def grab_frames(pipeline, align, depth_scale: float, timeout_ms: int):
    """Return (color_bgr, depth_m) copies, or (None, None) if no frame."""
    ok, frames = pipeline.try_wait_for_frames(timeout_ms)
    if not ok:
        return None, None
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    # Copy out of SDK-owned buffers before they are reclaimed.
    color_bgr = np.array(color_frame.get_data(), copy=True)
    depth_u16 = np.array(depth_frame.get_data(), copy=True)
    depth_m = depth_u16.astype(np.float32) * depth_scale
    return color_bgr, depth_m


def sample_depth(depth_m: np.ndarray, u: int, v: int, radius: int,
                 min_d: float, max_d: float) -> float | None:
    """Median of valid depths in a patch around (u, v). None if no valid depth."""
    h, w = depth_m.shape[:2]
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth_m[v0:v1, u0:u1].reshape(-1)
    valid = patch[(patch >= min_d) & (patch <= max_d)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def depth_colormap(depth_m: np.ndarray) -> np.ndarray:
    vis = np.clip(depth_m, 0.0, 3.0)
    vis = (vis / 3.0 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def compute_target_base(
    pixel: tuple[int, int],
    depth_m: np.ndarray,
    color_intrinsics,
    T_base_flange: np.ndarray,
    T_flange_camera: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray | None:
    u, v = pixel
    d = sample_depth(depth_m, u, v, args.patch_radius, args.min_depth, args.max_depth)
    if d is None:
        return None
    xyz_cam = rs.rs2_deproject_pixel_to_point(color_intrinsics, [float(u), float(v)], d)
    p_cam_h = np.array([xyz_cam[0], xyz_cam[1], xyz_cam[2], 1.0], dtype=np.float64)
    p_base = (T_base_flange @ T_flange_camera @ p_cam_h)[:3]
    p_base[2] += args.z_offset
    return p_base


def publish_goal(client: redis.Redis, pos: np.ndarray, ori: np.ndarray) -> None:
    client.set(ACTIVE_CTRL_KEY, CARTESIAN_CONTROLLER)
    client.set(GOAL_POS_KEY, json.dumps([float(x) for x in pos.reshape(3)]))
    client.set(GOAL_ORI_KEY, json.dumps(ori.reshape(3, 3).tolist()))


def main() -> int:
    args = parse_args()

    try:
        T_flange_camera = load_extrinsic(args.calib)
    except Exception as e:
        print(f"[extrinsic] error: {e}", file=sys.stderr)
        return 1

    try:
        client = connect_redis(args.redis_host, args.redis_port)
    except Exception as e:
        print(f"[redis] connect failed: {e}", file=sys.stderr)
        return 1

    if read_matrix(client, args.ee_key, (4, 4)) is None:
        print(f"[redis] WARNING: {args.ee_key} not available yet (start the Franka driver).",
              file=sys.stderr)

    try:
        pipeline, align, depth_scale, color_intrinsics = start_realsense(args)
    except Exception as e:
        print(f"[realsense] start failed: {e}", file=sys.stderr)
        return 1

    gemini_client = None
    gemini_prompt = None
    if args.gemini:
        try:
            gemini_client = make_gemini_client()
            gemini_prompt = build_hand_prompt(args.gemini_part, args.gemini_prompt)
            print(f"[gemini] enabled; will point at: {args.gemini_part!r}")
        except Exception as e:
            print(f"[gemini] disabled ({e}).", file=sys.stderr)

    rgb_win, depth_win = "rgb (click target)", "depth"
    cv2.namedWindow(rgb_win)
    cv2.namedWindow(depth_win)

    state: dict = {"pixel": None, "target_base": None, "source": None}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pixel"] = (int(x), int(y))
            state["target_base"] = None
            state["source"] = "click"

    cv2.setMouseCallback(rgb_win, on_mouse)

    controls = "left-click = select | g/ENTER = go | c = clear | q/ESC = quit"
    if gemini_client is not None:
        controls = "left-click/h(gemini) = select | g/ENTER = go | c = clear | q/ESC = quit"
    print(f"\nControls: {controls}\n")

    last_status = ""
    try:
        while True:
            color_bgr, depth_m = grab_frames(pipeline, align, depth_scale, args.cam_timeout_ms)
            if color_bgr is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            display = color_bgr.copy()
            target_base = None
            T_base_flange = read_matrix(client, args.ee_key, (4, 4))

            if state["pixel"] is not None:
                u, v = state["pixel"]
                cv2.drawMarker(display, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
                if state.get("target_base") is not None:
                    target_base = np.asarray(state["target_base"], dtype=np.float64)
                    last_status = (
                        f"latched {state.get('source', 'target')} = "
                        f"[{target_base[0]:.3f}, {target_base[1]:.3f}, "
                        f"{target_base[2]:.3f}] m  (ENTER/g)"
                    )
                elif T_base_flange is None:
                    last_status = "no T_end_effector from Redis"
                else:
                    target_base = compute_target_base(
                        (u, v), depth_m, color_intrinsics,
                        T_base_flange, T_flange_camera, args,
                    )
                    if target_base is None:
                        last_status = "invalid depth at pixel"
                    else:
                        last_status = (
                            f"target base = [{target_base[0]:.3f}, "
                            f"{target_base[1]:.3f}, {target_base[2]:.3f}] m  (press g)"
                        )

            cv2.putText(display, last_status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(rgb_win, display)
            cv2.imshow(depth_win, depth_colormap(depth_m))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                state["pixel"] = None
                state["target_base"] = None
                state["source"] = None
                last_status = "cleared"
            if key == ord("h"):
                if gemini_client is None:
                    print("[gemini] not enabled; run with --gemini.")
                else:
                    print(f"[gemini] querying for {args.gemini_part!r}...")
                    # Latch the exact frame and robot pose used for the Gemini query.
                    # The hand may move between query and ENTER; this keeps RGB, depth,
                    # and T_end_effector consistent with the image Gemini saw.
                    gemini_color = color_bgr.copy()
                    gemini_depth = depth_m.copy()
                    gemini_T_base_flange = read_matrix(client, args.ee_key, (4, 4))
                    px = gemini_point(
                        gemini_client, args.gemini_model, gemini_color,
                        gemini_prompt, args.gemini_timeout_s,
                    )
                    if px is None:
                        last_status = "gemini: no point"
                    elif gemini_T_base_flange is None:
                        state["pixel"] = px
                        state["target_base"] = None
                        state["source"] = "gemini-unlatched"
                        last_status = "gemini: no T_end_effector; target not latched"
                        print("[gemini] got pixel, but no T_end_effector from Redis.")
                    else:
                        latched = compute_target_base(
                            px, gemini_depth, color_intrinsics,
                            gemini_T_base_flange, T_flange_camera, args,
                        )
                        state["pixel"] = px
                        state["target_base"] = latched
                        state["source"] = "gemini"
                        if latched is None:
                            last_status = "gemini: invalid depth on query frame"
                            print(f"[gemini] pixel {px}, but query-frame depth was invalid.")
                        else:
                            print(f"[gemini] latched pixel {px} -> target {latched.tolist()}")
            if key in (ord("g"), ord("\r"), 13, 10):
                if target_base is None:
                    print("[move] no valid target to send.")
                    continue
                reach = float(np.linalg.norm(target_base))
                if reach > args.max_reach or reach < args.min_reach or target_base[2] < args.min_z:
                    print(
                        f"[move] WARNING: target {target_base.tolist()} looks out of range "
                        f"(reach={reach:.3f} m, z={target_base[2]:.3f} m; "
                        f"allowed reach [{args.min_reach}, {args.max_reach}] m, z>={args.min_z}). "
                        "Sending anyway."
                    )
                ori = read_matrix(client, CUR_ORI_KEY, (3, 3))
                if ori is None:
                    print("[move] no current orientation in Redis; cannot send.")
                    continue
                if args.no_move:
                    print(f"[dry-run] would move to {target_base.tolist()}")
                else:
                    publish_goal(client, target_base, ori)
                    print(f"[move] sent goal {target_base.tolist()}")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
