#!/usr/bin/env python3
"""Gemini pick + grasp (open) + lift/pour on OpenSai Franka (Redis + RealSense).

- **SPACE** — Gemini pointing; latch pick pose in world (same as ``vision_controller``).
- **ENTER (1)** — move to latched pick; gripper **open** (no auto-close).
- **ENTER (2)** — close gripper, lift, then pour (90° about world +Y).
- **s** / **q** — save overlay / quit.

No mobile-base motion. Requires OpenSai ``cartesian_controller``, Franka arm driver,
and Franka gripper Redis driver (``opensai::FrankaRobot::gripper::*``).

Usage::

  python ZitiBot/controllers/grasp_and_pour_controller.py
  python ZitiBot/controllers/grasp_and_pour_controller.py --lift-dz 0.12
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

_PYTHON_CONTROL = Path(__file__).resolve().parent
if str(_PYTHON_CONTROL) not in sys.path:
    sys.path.insert(0, str(_PYTHON_CONTROL))

import redis
from vision import gemini_pointing as gp
from vision import realsense_rgbd as rs_cam

import vision_controller as vc

CONFIG_XML = vc.CONFIG_XML
CONTROLLER_TO_USE = vc.CONTROLLER_TO_USE
@dataclass(frozen=True)
class _RedisKeys(vc.OpenSaiRedisKeys):
    gripper_desired_width: str = "opensai::FrankaRobot::gripper::desired_width"
    gripper_desired_speed: str = "opensai::FrankaRobot::gripper::desired_speed"
    gripper_desired_force: str = "opensai::FrankaRobot::gripper::desired_force"
    gripper_max_width: str = "opensai::FrankaRobot::gripper::max_width"
    gripper_current_width: str = "opensai::FrankaRobot::gripper::current_width"


_KEYS = _RedisKeys()
_TEXT_BAND_HEIGHT = vc.TEXT_BAND_HEIGHT

DEFAULT_LIFT_DZ_M = 0.15
DEFAULT_POS_TOL_M = 0.025
DEFAULT_TILT_DURATION_S = 6.0
DEFAULT_GRIPPER_SPEED = 0.1
DEFAULT_GRIPPER_FORCE = 50.0
MACRO_DT_S = 0.1
GRIPPER_CLOSE_SETTLE_S = 0.5


class Phase(enum.Enum):
    IDLE = "IDLE"
    LATCHED = "LATCHED"
    GRASP_SENT = "GRASP_SENT"
    DONE = "DONE"


class MacroStep(enum.Enum):
    NONE = "NONE"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT = "LIFT"
    POUR = "POUR"
    DONE = "DONE"


@dataclass
class MotionParams:
    lift_dz_m: float
    pos_tol_m: float
    tilt_duration_s: float
    gripper_open_width: float | None
    gripper_close_width: float
    gripper_speed: float
    gripper_force: float


@dataclass
class MacroState:
    step: MacroStep = MacroStep.NONE
    pick_world: np.ndarray | None = None
    pick_ori: np.ndarray | None = None
    lift_world: np.ndarray | None = None
    pour_R_start: np.ndarray | None = None
    pour_R_end: np.ndarray | None = None
    pour_t0: float = 0.0
    last_tick: float = 0.0
    gripper_close_t0: float = 0.0


def _decode_redis_value(raw: bytes | str | None) -> str | None:
    return vc._decode_redis_value(raw)


def _ensure_cartesian_controller(redis_client) -> None:
    while (
        _decode_redis_value(redis_client.get(_KEYS.active_controller))
        != CONTROLLER_TO_USE
    ):
        redis_client.set(_KEYS.active_controller, CONTROLLER_TO_USE)


def _publish_cartesian(
    redis_client,
    goal_pos: np.ndarray,
    goal_ori: np.ndarray,
) -> None:
    _ensure_cartesian_controller(redis_client)
    redis_client.set(
        _KEYS.cartesian_task_goal_position,
        json.dumps(np.asarray(goal_pos, dtype=np.float64).reshape(3).tolist()),
    )
    redis_client.set(
        _KEYS.cartesian_task_goal_orientation,
        json.dumps(np.asarray(goal_ori, dtype=np.float64).reshape(3, 3).tolist()),
    )


def read_gripper_max_width(redis_client) -> float | None:
    raw = redis_client.get(_KEYS.gripper_max_width)
    text = _decode_redis_value(raw)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_gripper_current_width(redis_client) -> float | None:
    raw = redis_client.get(_KEYS.gripper_current_width)
    text = _decode_redis_value(raw)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def set_gripper_width(
    redis_client,
    width_m: float,
    *,
    speed: float,
    force: float,
) -> None:
    redis_client.set(_KEYS.gripper_desired_width, str(float(width_m)))
    redis_client.set(_KEYS.gripper_desired_speed, str(float(speed)))
    redis_client.set(_KEYS.gripper_desired_force, str(float(force)))


def resolve_gripper_open_width(redis_client, override: float | None) -> float:
    if override is not None:
        return float(override)
    w = read_gripper_max_width(redis_client)
    if w is not None and w > 0:
        return w
    return 0.08


def pour_orientation_end(R_start: np.ndarray) -> np.ndarray:
    """+90° about world +Y, same as controller.cpp EE_POUR_BOWL."""
    R_y = R.from_euler("y", math.pi / 2.0).as_matrix()
    return R_y @ np.asarray(R_start, dtype=np.float64).reshape(3, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemini pick + grasp open + lift/pour (OpenSai Franka Redis)."
    )
    p.add_argument(
        "--object",
        default="bowl",
        help="Object in default rim prompt (near/bottom rim, closest to camera).",
    )
    p.add_argument("--prompt", default=None, help="Override full Gemini prompt.")
    p.add_argument(
        "--ee-from-cam-json",
        type=Path,
        default=None,
        help="4×4 T_ee_cam JSON (vision → EE, meters).",
    )
    p.add_argument("--depth-patch-radius", type=int, default=2)
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--model", default=gp.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--timeout-ms", type=int, default=10000)
    p.add_argument("--lift-dz", type=float, default=DEFAULT_LIFT_DZ_M)
    p.add_argument("--pos-tol", type=float, default=DEFAULT_POS_TOL_M)
    p.add_argument("--tilt-duration", type=float, default=DEFAULT_TILT_DURATION_S)
    p.add_argument(
        "--gripper-open-width",
        type=float,
        default=None,
        help="Open width (m); default reads gripper::max_width from Redis.",
    )
    p.add_argument("--gripper-close-width", type=float, default=0.0)
    p.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED)
    p.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE)
    return p.parse_args()


def _phase_hint(phase: Phase, macro: MacroState) -> str:
    if macro.step not in (MacroStep.NONE, MacroStep.DONE):
        return f"Macro: {macro.step.value} (wait…)"
    if phase == Phase.IDLE:
        return "Next: SPACE = Gemini + latch pick"
    if phase == Phase.LATCHED:
        return "Next: ENTER = move to pick + open gripper"
    if phase == Phase.GRASP_SENT:
        return "Next: ENTER = close, lift, pour"
    if phase == Phase.DONE:
        return "Done — SPACE to re-pick"
    return ""


def _start_grasp(
    redis_client,
    pick_world: np.ndarray,
    pick_ori: np.ndarray,
    motion: MotionParams,
) -> None:
    _publish_cartesian(redis_client, pick_world, pick_ori)
    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
    set_gripper_width(
        redis_client,
        open_w,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
    )
    print(
        f"Grasp: cartesian → pick {pick_world.tolist()}, "
        f"gripper open width={open_w:.4f} m"
    )


def _start_lift_pour_macro(
    macro: MacroState,
    pick_world: np.ndarray,
    pick_ori: np.ndarray,
    motion: MotionParams,
) -> None:
    macro.step = MacroStep.CLOSE_GRIPPER
    macro.pick_world = np.asarray(pick_world, dtype=np.float64).reshape(3).copy()
    macro.pick_ori = np.asarray(pick_ori, dtype=np.float64).reshape(3, 3).copy()
    macro.lift_world = macro.pick_world + np.array([0.0, 0.0, motion.lift_dz_m])
    macro.pour_R_start = None
    macro.pour_R_end = None
    macro.pour_t0 = 0.0
    macro.last_tick = 0.0
    macro.gripper_close_t0 = 0.0
    print("Lift/pour macro started: close gripper → lift → pour.")


def _tick_macro(
    redis_client,
    macro: MacroState,
    motion: MotionParams,
    now: float,
) -> MacroStep:
    """Advance lift/pour macro at ~10 Hz (called from UI loop)."""
    if macro.step in (MacroStep.NONE, MacroStep.DONE):
        return macro.step

    if macro.last_tick > 0 and (now - macro.last_tick) < MACRO_DT_S:
        return macro.step
    macro.last_tick = now

    if macro.step == MacroStep.CLOSE_GRIPPER:
        set_gripper_width(
            redis_client,
            motion.gripper_close_width,
            speed=motion.gripper_speed,
            force=motion.gripper_force,
        )
        if macro.gripper_close_t0 == 0.0:
            macro.gripper_close_t0 = now
            print(f"Gripper close → {motion.gripper_close_width:.4f} m")
        if now - macro.gripper_close_t0 >= GRIPPER_CLOSE_SETTLE_S:
            macro.step = MacroStep.LIFT
            assert macro.lift_world is not None and macro.pick_ori is not None
            _publish_cartesian(redis_client, macro.lift_world, macro.pick_ori)
            print(f"Lift goal: {macro.lift_world.tolist()}")
        return macro.step

    if macro.step == MacroStep.LIFT:
        assert macro.lift_world is not None and macro.pick_ori is not None
        pose = vc.read_current_ee_world(redis_client)
        if pose is not None:
            cur_pos, cur_ori = pose
            if np.linalg.norm(cur_pos - macro.lift_world) < motion.pos_tol_m:
                macro.step = MacroStep.POUR
                macro.pour_R_start = cur_ori.copy()
                macro.pour_R_end = pour_orientation_end(macro.pour_R_start)
                macro.pour_t0 = now
                _publish_cartesian(redis_client, macro.lift_world, macro.pour_R_start)
                print("Lift reached; starting pour slerp (+90° world Y).")
        return macro.step

    if macro.step == MacroStep.POUR:
        assert (
            macro.lift_world is not None
            and macro.pour_R_start is not None
            and macro.pour_R_end is not None
        )
        alpha = min(1.0, max(0.0, (now - macro.pour_t0) / motion.tilt_duration_s))
        key_rots = R.concatenate(
            [
                R.from_matrix(macro.pour_R_start),
                R.from_matrix(macro.pour_R_end),
            ]
        )
        R_tilt = Slerp([0.0, 1.0], key_rots)([alpha]).as_matrix()[0]
        _publish_cartesian(redis_client, macro.lift_world, R_tilt)
        if alpha >= 1.0 and macro.step != MacroStep.DONE:
            macro.step = MacroStep.DONE
            print("Pour complete (hold).")
        return macro.step

    return macro.step


def _gemini_placeholder_panel(h: int, w: int) -> np.ndarray:
    panel = vc._gemini_placeholder_panel(h, w)
    cv2.putText(
        panel,
        "ENTER: grasp / lift+pour",
        (max(10, w // 2 - 150), h // 2 + 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (140, 140, 160),
        1,
        cv2.LINE_AA,
    )
    return panel


def run_live(
    args: argparse.Namespace,
    prompt: str,
    redis_client,
    motion: MotionParams,
) -> int:
    client = gp.make_genai_client(gp.resolve_api_key())
    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")

    if args.ee_from_cam_json is not None:
        T_ee_cam = gp.load_T_ee_cam(args.ee_from_cam_json)
        print(f"Loaded T_ee_cam from {args.ee_from_cam_json}")
    else:
        T_ee_cam = gp.PLACEHOLDER_T_EE_CAM.copy()
        print("Using built-in T_ee_cam (override with --ee-from-cam-json).")

    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
    motion = MotionParams(
        lift_dz_m=motion.lift_dz_m,
        pos_tol_m=motion.pos_tol_m,
        tilt_duration_s=motion.tilt_duration_s,
        gripper_open_width=open_w,
        gripper_close_width=motion.gripper_close_width,
        gripper_speed=motion.gripper_speed,
        gripper_force=motion.gripper_force,
    )
    print(
        f"Motion: lift_dz={motion.lift_dz_m} m, pos_tol={motion.pos_tol_m}, "
        f"tilt={motion.tilt_duration_s} s, gripper open={motion.gripper_open_width:.4f} m"
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

    win = "Grasp and pour (RGB | depth | latch)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    latched: np.ndarray | None = None
    miss_counter = [0]
    pending_first_ee: tuple[float, float, float] | None = None
    latched_goal_world: np.ndarray | None = None
    latched_goal_ori: np.ndarray | None = None
    phase = Phase.IDLE
    macro = MacroState()

    print(
        "Keys: SPACE = Gemini + latch | ENTER = grasp (open) then lift+pour | "
        "s = save | q = quit"
    )

    try:
        while True:
            now = time.perf_counter()
            if macro.step not in (MacroStep.NONE, MacroStep.DONE):
                prev = macro.step
                _tick_macro(redis_client, macro, motion, now)
                if macro.step == MacroStep.DONE and prev != MacroStep.DONE:
                    phase = Phase.DONE

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

            pose = vc.read_current_ee_world(redis_client)
            if pose is None:
                cur_lines = ["Current EE (world, m)", "(could not read Redis)"]
            else:
                cur_pos, cur_ori = pose
                cur_lines = vc._fmt_xyz("Current EE (world, m)", cur_pos)
                gw = read_gripper_current_width(redis_client)
                if gw is not None:
                    cur_lines.append(f"Gripper width: {gw:.4f} m")

            hint = _phase_hint(phase, macro)
            goal_lines = [f"Phase: {phase.value}", hint]
            if latched_goal_world is not None:
                goal_lines.extend(
                    vc._fmt_xyz("Pick (latched world, m)", latched_goal_world)
                )
            elif phase == Phase.IDLE:
                goal_lines.append("No pick latched yet.")

            band_w = top_row.shape[1]
            w_left = (band_w * 2) // 3
            w_right = band_w - w_left
            bottom_row = np.hstack(
                (
                    vc._render_text_band(w_left, _TEXT_BAND_HEIGHT, cur_lines),
                    vc._render_text_band(w_right, _TEXT_BAND_HEIGHT, goal_lines),
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
                    macro = MacroState()
                    if first_ee is not None:
                        pose_l = vc.read_current_ee_world(redis_client)
                        if pose_l is not None:
                            cur_lp, cur_lo = pose_l
                            p_ee_l = np.asarray(first_ee, dtype=np.float64).reshape(3)
                            latched_goal_world = (cur_lo @ p_ee_l) + cur_lp
                            latched_goal_ori = cur_lo.copy()
                            phase = Phase.LATCHED
                            print("Latched pick at SPACE.")
                        else:
                            phase = Phase.IDLE
                            print("Gemini OK but Redis EE pose missing; not latched.")
                    else:
                        phase = Phase.IDLE
                        print("No 3D point at pick pixel; not latched.")
            elif key in (10, 13):
                if macro.step not in (MacroStep.NONE, MacroStep.DONE):
                    print("Macro running — wait for lift/pour to finish.")
                    continue
                if phase == Phase.LATCHED:
                    if latched_goal_world is None or latched_goal_ori is None:
                        print("No latched pick — press SPACE first.")
                        continue
                    _start_grasp(
                        redis_client,
                        latched_goal_world,
                        latched_goal_ori,
                        motion,
                    )
                    phase = Phase.GRASP_SENT
                elif phase == Phase.GRASP_SENT:
                    if latched_goal_world is None or latched_goal_ori is None:
                        print("No latched pick.")
                        continue
                    _start_lift_pour_macro(
                        macro,
                        latched_goal_world,
                        latched_goal_ori,
                        motion,
                    )
                elif phase == Phase.DONE:
                    print("Sequence done — SPACE to pick again.")
                else:
                    print("Press SPACE to latch a pick, then ENTER.")
            elif key == ord("s"):
                vc._save_overlay(latched)
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    redis_client = vc._try_redis(args.redis_host, args.redis_port)
    if redis_client is None:
        return 1
    err = vc.validate_config(redis_client)
    if err is not None:
        return err

    motion = MotionParams(
        lift_dz_m=args.lift_dz,
        pos_tol_m=args.pos_tol,
        tilt_duration_s=args.tilt_duration,
        gripper_open_width=args.gripper_open_width,
        gripper_close_width=args.gripper_close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
    )
    prompt = gp.build_prompt(args.object, args.prompt)
    return run_live(args, prompt, redis_client, motion)


if __name__ == "__main__":
    sys.exit(main())
