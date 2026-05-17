#!/usr/bin/env python3
"""Live Gemini + RealSense → OpenSai Franka cartesian goals (Redis only here).

Uses ``vision/gemini_pointing`` and ``vision/realsense_rgbd`` as libraries (no
Redis inside those modules).

- **SPACE** — capture frame, call Gemini; frozen overlay appears in the **third** pane.
- **ENTER** — write ``cartesian_task`` goal position/orientation JSON on OpenSai keys.
  Values are the **pick pose latched at SPACE** (world position + EE orientation at that
  instant), with **no** extra world +Z lift. They do **not** update if the arm moves before ENTER.
- **s** / **q** — save overlay / quit.

One window: **RGB | depth | Gemini latch** on top; **current EE position** under the
first two panes; **pick / ENTER goal** under the Gemini pane.

The lifted point is in the EE frame at capture; at **SPACE** the world **goal position
and orientation** are latched from Redis and stay fixed in the UI until the next **SPACE**.
**ENTER** sends that same latched pose (not recomputed from the current arm pose).

Usage::

  python ZitiBot/controllers/vision_controller.py
  python ZitiBot/controllers/vision_controller.py --object mug
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

_PYTHON_CONTROL = Path(__file__).resolve().parent
if str(_PYTHON_CONTROL) not in sys.path:
    sys.path.insert(0, str(_PYTHON_CONTROL))

import redis
from vision import gemini_pointing as gp
from vision import realsense_rgbd as rs_cam

CONFIG_XML = os.environ.get("ZITIBOT_OPENSAI_CONFIG_XML", "zitibot_panda.xml")
CONTROLLER_TO_USE = "cartesian_controller"


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
    active_controller: str = "opensai::controllers::FrankaRobot::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"


_KEYS = OpenSaiRedisKeys()

# Bottom status strip: font / layout (25% smaller than prior strip; thicker stroke).
_TEXT_SIZE_MULT = 0.75 * 0.75
_TEXT_FONT_SCALE = 0.48 * 3.0 * _TEXT_SIZE_MULT
_TEXT_LINE_STEP = int(22 * 3.0 * _TEXT_SIZE_MULT)
_TEXT_THICKNESS = 2
_TEXT_EMPTY_SKIP = int(18 * 3.0 * _TEXT_SIZE_MULT)
# Bottom status strip height in the single composite window (pixels).
TEXT_BAND_HEIGHT = int((int(120 * 3.0) + 180) * _TEXT_SIZE_MULT) + 80


def _split_T_ee_cam(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``T_ee_cam`` uses ``p_ee = R @ p_vis + t`` (ZitiBot vision → EE task link)."""
    Tm = np.asarray(T, dtype=np.float64)
    return Tm[:3, :3].copy(), Tm[:3, 3].copy()


def camera_origin_in_ee_frame_m(T_ee_cam: np.ndarray) -> np.ndarray:
    """**Vision** optical origin expressed in the EE link frame (m).

    With ``p_ee = R @ p_vis + t``, ``p_vis = 0`` gives ``p_ee = t``.
    """
    _, t = _split_T_ee_cam(T_ee_cam)
    return t.reshape(3)


def ee_origin_in_vision_frame_m(T_ee_cam: np.ndarray) -> np.ndarray:
    """EE link origin expressed in the **ZitiBot vision** frame (m).

    ``p_vis = -R.T @ t`` for ``p_ee = 0``. Vision: +X up image, +Z into scene, +Y per RS remap.
    """
    R, t = _split_T_ee_cam(T_ee_cam)
    return (-(R.T @ t)).reshape(3)


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
        help="Object name in the default bowl-rim prompt (leftmost rim point).",
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
    p.add_argument("--model", default=gp.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--timeout-ms", type=int, default=10000)
    return p.parse_args()


def _try_redis(host: str, port: int):
    try:
        r = redis.Redis(host=host, port=port, decode_responses=False)
        r.ping()
        return r
    except Exception as e:
        print(f"Redis connect failed ({e}).", file=sys.stderr)
        return None


def read_current_ee_world(redis_client) -> tuple[np.ndarray, np.ndarray] | None:
    """Return current ``(position (3,), orientation (3,3))`` from Redis, or ``None``."""
    try:
        raw_p = redis_client.get(_KEYS.cartesian_task_current_position)
        raw_o = redis_client.get(_KEYS.cartesian_task_current_orientation)
        if raw_p is None or raw_o is None:
            return None
        cur_pos = np.array(json.loads(raw_p), dtype=np.float64).reshape(3)
        cur_ori = np.array(json.loads(raw_o), dtype=np.float64).reshape(3, 3)
        return cur_pos, cur_ori
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def planned_pick_and_goal_world(
    redis_client, pos_ee: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Gemini pick in world, ENTER goal (same point), current EE orientation, EE position.

    Returns ``(pick_world, goal_world, goal_ori, cur_pos)`` or ``None``.
    ``goal_world`` is a copy of ``pick_world`` (no world +Z approach offset).
    ``goal_ori`` is the current EE orientation from Redis (same as before mapping).
    """
    if pos_ee is None:
        return None
    pose = read_current_ee_world(redis_client)
    if pose is None:
        return None
    cur_pos, cur_ori = pose
    p_ee = np.asarray(pos_ee, dtype=np.float64).reshape(3)
    pick_world = (cur_ori @ p_ee) + cur_pos
    goal_world = pick_world.copy()
    return pick_world, goal_world, cur_ori.copy(), cur_pos.copy()


def planned_goal_world(
    redis_client, pos_ee: tuple[float, float, float] | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """World-frame goal position + orientation **as sent on ENTER** (pick pose)."""
    out = planned_pick_and_goal_world(redis_client, pos_ee)
    if out is None:
        return None
    _, goal_world, goal_ori, _ = out
    return goal_world, goal_ori


def _fmt_xyz(label: str, v: np.ndarray) -> list[str]:
    v = np.asarray(v, dtype=np.float64).ravel()
    return [
        label,
        f"x={v[0]:+.4f}  y={v[1]:+.4f}  z={v[2]:+.4f}",
    ]


def _render_text_band(width: int, height: int, lines: list[str]) -> np.ndarray:
    """Dark strip with left-aligned lines (truncated if needed)."""
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
    msg = "Press SPACE"
    sub = "Gemini + depth"
    cv2.putText(
        panel,
        msg,
        (max(10, w // 2 - 110), h // 2 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (180, 180, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        sub,
        (max(10, w // 2 - 130), h // 2 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (140, 140, 160),
        1,
        cv2.LINE_AA,
    )
    return panel


def validate_config(redis_client) -> int | None:
    raw = redis_client.get(_KEYS.config_file_name)
    name = _decode_redis_value(raw)
    if name is None:
        print(
            "Warning: ::sai-interfaces-webui::config_file_name not in Redis; "
            "continuing anyway.",
            file=sys.stderr,
        )
        return None
    if name != CONFIG_XML:
        print(
            f"Expected webui config {CONFIG_XML!r} but Redis has {name!r}. "
            "Set ZITIBOT_OPENSAI_CONFIG_XML if needed.",
            file=sys.stderr,
        )
        return 1
    return None


def publish_opensai_cartesian(
    redis_client,
    pos_ee: tuple[float, float, float] | None,
    *,
    latched_goal_world: np.ndarray | None = None,
    latched_goal_ori: np.ndarray | None = None,
) -> None:
    """Write cartesian_task goals to Redis.

    If ``latched_goal_world`` / ``latched_goal_ori`` are set (from SPACE capture), those
    are sent so the goal matches the frozen overlay. Otherwise goals are computed from
    the **current** Redis EE pose and ``pos_ee``.
    """
    if latched_goal_world is not None and latched_goal_ori is not None:
        goal_world = np.asarray(latched_goal_world, dtype=np.float64).reshape(3).copy()
        goal_ori = np.asarray(latched_goal_ori, dtype=np.float64).reshape(3, 3).copy()
    else:
        if pos_ee is None:
            print("No valid 3D target (depth). Not sending OpenSai cartesian goals.")
            return
        pg = planned_pick_and_goal_world(redis_client, pos_ee)
        if pg is None:
            print("Could not read current EE pose from Redis; not sending goals.")
            return
        _pick_world, goal_world, goal_ori, _cur_pos = pg

    while (
        _decode_redis_value(redis_client.get(_KEYS.active_controller))
        != CONTROLLER_TO_USE
    ):
        redis_client.set(_KEYS.active_controller, CONTROLLER_TO_USE)

    redis_client.set(
        _KEYS.cartesian_task_goal_position, json.dumps(goal_world.tolist())
    )
    redis_client.set(
        _KEYS.cartesian_task_goal_orientation, json.dumps(goal_ori.tolist())
    )
    extra = ""
    if pos_ee is not None:
        p_ee = np.asarray(pos_ee, dtype=np.float64).reshape(3)
        extra = f"  EE_delta={p_ee.tolist()}."
    print(
        "OpenSai: set cartesian_task goals "
        f"goal_pos={goal_world.tolist()}  (pick pose world, no +Z offset){extra}"
    )


def _save_overlay(latched: np.ndarray | None) -> None:
    if latched is None:
        print("Nothing to save yet — press SPACE first.")
        return
    fname = f"gemini_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(fname, latched)
    print(f"Saved {fname}")


def run_live(args: argparse.Namespace, prompt: str, redis_client) -> int:
    client = gp.make_genai_client(gp.resolve_api_key())
    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")

    if args.ee_from_cam_json is not None:
        T_ee_cam = gp.load_T_ee_cam(args.ee_from_cam_json)
        print(f"Loaded T_ee_cam from {args.ee_from_cam_json}")
    else:
        T_ee_cam = gp.PLACEHOLDER_T_EE_CAM.copy()
        print(
            "Using built-in T_ee_cam (override with --ee-from-cam-json). "
            "``T_ee_cam``: vision→EE; vision = RS remap (+X up, +Z into scene, +Y per RS remap). "
            f"Placeholder t=({gp.PLACEHOLDER_T_EE_CAM[0, 3]:.4f},{gp.PLACEHOLDER_T_EE_CAM[1, 3]:.4f},"
            f"{gp.PLACEHOLDER_T_EE_CAM[2, 3]:.4f})m EE (ZITIBOT_PLACEHOLDER_T_EE_CAM_X_M / _Z_M). "
            "URDF EE = link7+(0,0,0.14)m +Z (panda_arm_sphere.urdf)."
        )

    pipeline = None
    try:
        pipeline, align, depth_scale, color_intrinsics = rs_cam.start_realsense(
            args.width,
            args.height,
            args.fps,
            args.warmup_frames,
            args.timeout_ms,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"RealSense startup failed: {e}", file=sys.stderr)
        return 1

    win = "Gemini vision (RGB | depth | latch) + EE status"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    latched: np.ndarray | None = None
    miss_counter = [0]
    pending_first_ee: tuple[float, float, float] | None = None
    have_pending_publish = False
    latched_goal_world: np.ndarray | None = None
    latched_goal_ori: np.ndarray | None = None

    print(
        "Keys: SPACE = Gemini + overlay | ENTER = OpenSai cartesian goal (Redis) | "
        "s = save overlay | q = quit\n"
        "Single window: RGB | depth | Gemini latch; text bands show EE pose and "
        "goal latched at SPACE (unchanged if the arm moves until next SPACE)."
    )

    try:
        while True:
            try:
                triple = rs_cam.next_rgbd_frame(
                    pipeline,
                    align,
                    depth_scale,
                    args.timeout_ms,
                    miss_counter,
                    max_misses=10,
                )
            except TimeoutError as e:
                print(e, file=sys.stderr)
                return 2
            if triple is None:
                continue
            color_bgr, depth_m, depth_vis = triple
            h, w = color_bgr.shape[:2]

            if latched is not None and latched.shape[:2] != (h, w):
                latched = cv2.resize(latched, (w, h), interpolation=cv2.INTER_AREA)
            gemini_panel = (
                latched.copy()
                if latched is not None
                else _gemini_placeholder_panel(h, w)
            )
            if gemini_panel.shape[:2] != (h, w):
                gemini_panel = cv2.resize(
                    gemini_panel, (w, h), interpolation=cv2.INTER_AREA
                )
            if depth_vis.shape[:2] != (h, w):
                depth_vis = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_AREA)

            top_row = np.hstack((color_bgr, depth_vis, gemini_panel))

            pose = read_current_ee_world(redis_client)
            if pose is None:
                cur_lines = [
                    "Current EE (world, m)",
                    "(could not read Redis)",
                ]
            else:
                cur_pos, cur_ori = pose
                cur_lines = _fmt_xyz("Current EE position (world / task, m)", cur_pos)
                cur_lines.append("TCP in own EE frame: (0, 0, 0) m (this link)")
                w_o_in_ee = cur_ori.T @ (-cur_pos)
                cur_lines.extend(
                    _fmt_xyz("World origin in EE frame (m)", w_o_in_ee)
                )
                cam_in_ee = camera_origin_in_ee_frame_m(T_ee_cam)
                ee_in_cam = ee_origin_in_vision_frame_m(T_ee_cam)
                cur_lines.append("Vision origin in EE frame (m) [T column t]:")
                cur_lines.append(
                    f"x={cam_in_ee[0]:+.4f}  y={cam_in_ee[1]:+.4f}  z={cam_in_ee[2]:+.4f}"
                )
                cur_lines.append("EE origin in vision frame (m) [-R^T t]:")
                cur_lines.append(
                    f"x={ee_in_cam[0]:+.4f}  y={ee_in_cam[1]:+.4f}  z={ee_in_cam[2]:+.4f}"
                )

            pg = None
            if have_pending_publish and latched_goal_world is None and pending_first_ee is not None:
                pg = planned_pick_and_goal_world(redis_client, pending_first_ee)
            if not have_pending_publish:
                goal_lines = [
                    "Target (world, m) — NOT sent",
                    "Run Gemini (SPACE) first.",
                ]
            elif latched_goal_world is not None:
                goal_lines = _fmt_xyz(
                    "Pick / ENTER goal (world, m) — latched at SPACE",
                    latched_goal_world,
                )
                if pending_first_ee is not None:
                    p_ee = np.asarray(pending_first_ee, dtype=np.float64).ravel()
                    goal_lines.append(
                        f"Gemini EE delta (m): "
                        f"{p_ee[0]:+.4f}, {p_ee[1]:+.4f}, {p_ee[2]:+.4f}"
                    )
            elif pending_first_ee is None:
                goal_lines = [
                    "Target — NOT sent",
                    "No 3D EE point (depth missing at pick pixel).",
                ]
            elif pg is None:
                goal_lines = [
                    "Target — NOT sent",
                    "(could not read Redis pose at SPACE; goal not latched.)",
                ]
            else:
                _pick_w, goal_w, _, _ = pg
                p_ee = np.asarray(pending_first_ee, dtype=np.float64).ravel()
                goal_lines = _fmt_xyz(
                    "Pick / ENTER goal (world, m) — live (re-latch with SPACE)",
                    goal_w,
                )
                goal_lines.append(
                    f"Gemini EE delta (m): "
                    f"{p_ee[0]:+.4f}, {p_ee[1]:+.4f}, {p_ee[2]:+.4f}"
                )

            band_w = top_row.shape[1]
            w_left = (band_w * 2) // 3
            w_right = band_w - w_left
            bottom_row = np.hstack(
                (
                    _render_text_band(w_left, TEXT_BAND_HEIGHT, cur_lines),
                    _render_text_band(w_right, TEXT_BAND_HEIGHT, goal_lines),
                )
            )

            composite = np.vstack((top_row, bottom_row))
            cv2.imshow(win, composite)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                overlay, first_ee = gp.query_color_depth_overlay(
                    client,
                    args.model,
                    prompt,
                    args.temperature,
                    color_bgr,
                    depth_m,
                    color_intrinsics,
                    T_ee_cam,
                    args.depth_patch_radius,
                )
                if overlay is not None:
                    latched = overlay
                    pending_first_ee = first_ee
                    latched_goal_world = None
                    latched_goal_ori = None
                    if first_ee is not None:
                        pose_l = read_current_ee_world(redis_client)
                        if pose_l is not None:
                            cur_lp, cur_lo = pose_l
                            p_ee_l = np.asarray(first_ee, dtype=np.float64).reshape(3)
                            latched_goal_world = (cur_lo @ p_ee_l) + cur_lp
                            latched_goal_ori = cur_lo.copy()
                    have_pending_publish = True
            elif key in (10, 13):
                if not have_pending_publish:
                    print("Press SPACE first, then ENTER to send the goal.")
                else:
                    publish_opensai_cartesian(
                        redis_client,
                        pending_first_ee,
                        latched_goal_world=latched_goal_world,
                        latched_goal_ori=latched_goal_ori,
                    )
            elif key == ord("s"):
                _save_overlay(latched)
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
    return run_live(args, prompt, redis_client)


if __name__ == "__main__":
    sys.exit(main())
