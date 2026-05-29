#!/usr/bin/env python3
"""OptiTrack base + Gemini vision grasp + pour (headless, ENTER-gated).

Drives the mobile base to grasp pose (A), runs Gemini on RealSense to latch
grasp pose/orientation, picks, drives to pour pose (B), pours, returns,
retracts, and releases. No OpenCV UI; each Gemini call saves
``ZitiBot/logs/gemini_response.png`` (RGB overlay ⊕ depth).

Startup (no motion): prints plan and waits.

Then one ENTER per step:

- ENTER (1) — drive base to grasp Opti pose (A).
- ENTER (2) — Gemini detect + latch grasp (saves gemini_response.png).
- ENTER (3) — move arm above detected grasp + open gripper.
- ENTER (4) — descend to grasp contact.
- ENTER (5) — pregrasp + grasp.
- ENTER (6) — lift (+``--approach-dz`` in world Z).
- ENTER (7) — drive base to pour Opti pose (B).
- ENTER (8) — start pour slerp (auto-completes).
- ENTER (9) — start return slerp (auto-completes).
- ENTER (10) — retract up at pour site.
- ENTER (11) — open gripper (release at B).

Requires tidybot base driver, OptiTrack on Redis, OpenSai cartesian + gripper,
RealSense, and ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/grasp_pour_vision_base_controller.py
  ./ZitiBot/launch_zitibot_full.sh --tune-marker-offset \\
    controllers/grasp_pour_vision_base_controller.py -- --object bowl
"""

from __future__ import annotations

import argparse
import enum
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
    MotionParams,
    OrientationSlerpState,
    POUR_TICK_DT_S,
    _STDIN_EOF,
    _do_descend_to_grasp,
    _do_grasp_object,
    _do_move_above_grasp,
    _do_open_gripper,
    _stdin_line_ready,
    _tick_orientation_slerp,
    resolve_gripper_open_width,
    validate_config,
)
from pour_and_move_controller import (
    DEFAULT_BASE_LOG_HZ,
    DEFAULT_BASE_TOLERANCE_IN,
    DEFAULT_BASE_YAW_TOLERANCE_DEG,
    DEFAULT_GRASP_OPTI_X,
    DEFAULT_GRASP_OPTI_Y,
    DEFAULT_GRASP_OPTI_YAW_DEG,
    DEFAULT_POUR_OPTI_X,
    DEFAULT_POUR_OPTI_Y,
    DEFAULT_POUR_OPTI_YAW_DEG,
    INCHES_TO_METERS,
    _do_lift,
    _do_retract_above_pour,
    _drive_base,
    _start_pour_step,
    _start_return_step,
)
from tidybot_base.opti_nav import NavConfig
from tidybot_base.opti_planner import DEFAULT_MARKER_YAW_OFFSET_DEG
from tidybot_base.redis_io import connect_redis
from vision import gemini_pointing as gp
from vision import realsense_rgbd as rs_cam
from vision_grasp_pour_controller import (
    DEFAULT_GEMINI_RESPONSE_PATH,
    LatchedTarget,
    do_gemini_capture,
    _grab_fresh_frame,
)

DEFAULT_GEMINI_RESPONSE_REL = "logs/gemini_response.png"


class Phase(enum.Enum):
    AWAIT_DRIVE_TO_GRASP = "AWAIT_DRIVE_TO_GRASP"
    AWAIT_VISION = "AWAIT_VISION"
    AWAIT_ABOVE_GRASP = "AWAIT_ABOVE_GRASP"
    ABOVE_GRASP = "ABOVE_GRASP"
    AT_GRASP = "AT_GRASP"
    CLOSED = "CLOSED"
    LIFTED = "LIFTED"
    ABOVE_POUR = "ABOVE_POUR"
    POURING = "POURING"
    POURED = "POURED"
    RETURNING = "RETURNING"
    RETURNED = "RETURNED"
    READY_TO_RELEASE = "READY_TO_RELEASE"
    DONE = "DONE"


@dataclass
class VisionContext:
    gemini_client: object
    pipeline: object
    align: object
    depth_scale: float
    color_intrinsics: object
    prompt: str
    save_path: Path
    miss_counter: list[int]


def _phase_hint(phase: Phase) -> str:
    if phase == Phase.AWAIT_DRIVE_TO_GRASP:
        return "Next: ENTER = drive base to grasp pose (A)"
    if phase == Phase.AWAIT_VISION:
        return (
            "Next: ENTER = run Gemini detection "
            f"(saves {DEFAULT_GEMINI_RESPONSE_REL})"
        )
    if phase == Phase.AWAIT_ABOVE_GRASP:
        return "Next: ENTER = move arm above detected grasp + open gripper"
    if phase == Phase.ABOVE_GRASP:
        return "Next: ENTER = descend to grasp contact"
    if phase == Phase.AT_GRASP:
        return "Next: ENTER = pregrasp + close gripper"
    if phase == Phase.CLOSED:
        return "Next: ENTER = lift (+Z)"
    if phase == Phase.LIFTED:
        return "Next: ENTER = drive base to pour pose (B)"
    if phase == Phase.ABOVE_POUR:
        return "Next: ENTER = start pour (world tilt slerp)"
    if phase == Phase.POURING:
        return "Pouring… (auto-completes)"
    if phase == Phase.POURED:
        return "Next: ENTER = return orientation to grasp"
    if phase == Phase.RETURNING:
        return "Returning orientation… (auto-completes)"
    if phase == Phase.RETURNED:
        return "Next: ENTER = retract up at pour site"
    if phase == Phase.READY_TO_RELEASE:
        return "Next: ENTER = open gripper (release at B)"
    if phase == Phase.DONE:
        return "Done — q to quit"
    return ""


def _run_vision_detect(
    redis_client,
    args: argparse.Namespace,
    motion: MotionParams,
    vision: VisionContext,
) -> LatchedTarget | None:
    triple = _grab_fresh_frame(
        vision.pipeline,
        vision.align,
        vision.depth_scale,
        args.timeout_ms,
        vision.miss_counter,
    )
    if triple is None:
        print("Failed to grab RealSense frame; press ENTER to retry.")
        return None

    print("[2] Gemini detect + latch grasp...")
    target, _overlay = do_gemini_capture(
        triple,
        redis_client=redis_client,
        args=args,
        motion=motion,
        gemini_client=vision.gemini_client,
        prompt=vision.prompt,
        color_intrinsics=vision.color_intrinsics,
        save_path=vision.save_path,
    )
    if target is None:
        print(
            "Vision latch failed; press ENTER to retry or q to quit.",
            file=sys.stderr,
        )
    return target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "OptiTrack base + vision grasp + pour, headless ENTER-gated "
            "(OpenSai Franka Redis)."
        )
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)

    # Base / OptiTrack
    p.add_argument("--grasp-opti-x", type=float, default=DEFAULT_GRASP_OPTI_X)
    p.add_argument("--grasp-opti-y", type=float, default=DEFAULT_GRASP_OPTI_Y)
    p.add_argument(
        "--grasp-opti-yaw-deg",
        type=float,
        default=DEFAULT_GRASP_OPTI_YAW_DEG,
        help="Body yaw at grasp pose in Motive lab (deg).",
    )
    p.add_argument("--pour-opti-x", type=float, default=DEFAULT_POUR_OPTI_X)
    p.add_argument("--pour-opti-y", type=float, default=DEFAULT_POUR_OPTI_Y)
    p.add_argument(
        "--pour-opti-yaw-deg",
        type=float,
        default=DEFAULT_POUR_OPTI_YAW_DEG,
        help="Body yaw at pour pose in Motive lab (deg).",
    )
    p.add_argument(
        "--marker-yaw-offset-deg",
        type=float,
        default=DEFAULT_MARKER_YAW_OFFSET_DEG,
        help=f"Marker→body offset (default {DEFAULT_MARKER_YAW_OFFSET_DEG}).",
    )
    p.add_argument(
        "--base-tolerance-in",
        type=float,
        default=DEFAULT_BASE_TOLERANCE_IN,
        help="Base XY success tolerance (inches).",
    )
    p.add_argument(
        "--base-yaw-tolerance-deg",
        type=float,
        default=DEFAULT_BASE_YAW_TOLERANCE_DEG,
    )
    p.add_argument(
        "--base-log-hz",
        type=float,
        default=DEFAULT_BASE_LOG_HZ,
        help="Opti nav log rate during base moves (default 2 Hz).",
    )
    p.add_argument(
        "--base-monitor",
        action="store_true",
        help="Do not command hb1 during base moves (dry-run nav only).",
    )
    p.add_argument(
        "--base-print-plan",
        action="store_true",
        help="Print full opti nav plan at each base move.",
    )

    # Vision / Gemini
    p.add_argument(
        "--gemini-response-path",
        default=str(DEFAULT_GEMINI_RESPONSE_PATH),
        help="Path to save annotated RGB ⊕ depth composite after Gemini.",
    )
    p.add_argument("--object", default="bowl")
    p.add_argument("--prompt", default=None)
    p.add_argument("--depth-patch-radius", type=int, default=2)
    p.add_argument(
        "--endeffector-transform-key",
        default="opensai::redis_driver::FrankaRobot::T_end_effector",
    )
    p.add_argument("--no-grasp-offset", action="store_true")
    p.add_argument("--grasp-offset-x", type=float, default=0.03)
    p.add_argument("--grasp-offset-y", type=float, default=0.0)
    p.add_argument("--grasp-offset-z", type=float, default=0.09)
    p.add_argument(
        "--orientation-source",
        choices=("fixed", "current"),
        default="fixed",
        help=(
            "Grasp orientation base: fixed=GRASP_ORIENTATION (+ rim yaw in world Z); "
            "current=live EE orientation (+ rim yaw in tool Z)."
        ),
    )
    p.add_argument("--model", default=gp.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--timeout-ms", type=int, default=10000)

    # Arm / gripper / pour
    p.add_argument(
        "--approach-dz",
        type=float,
        default=DEFAULT_APPROACH_DZ_M,
        help="Vertical clearance for approach / lift / retract (m).",
    )
    p.add_argument(
        "--pour-tilt-deg",
        type=float,
        default=DEFAULT_POUR_TILT_DEG,
        help="Pour rotation (degrees); default 90.",
    )
    p.add_argument(
        "--pour-axis",
        choices=("x", "y"),
        default=DEFAULT_POUR_AXIS,
        help="World axis for pour tilt (default x).",
    )
    p.add_argument("--tilt-duration", type=float, default=DEFAULT_TILT_DURATION_S)
    p.add_argument(
        "--gripper-open-width",
        type=float,
        default=None,
        help="Open width (m); default reads gripper::max_width from Redis.",
    )
    p.add_argument(
        "--gripper-pregrasp-width",
        type=float,
        default=DEFAULT_GRIPPER_PREGRASP_WIDTH,
    )
    p.add_argument("--gripper-close-width", type=float, default=0.0)
    p.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED)
    p.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE)
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


def run_loop(
    redis_client,
    motion: MotionParams,
    grasp_xy: tuple[float, float],
    grasp_yaw_deg: float,
    pour_xy: tuple[float, float],
    pour_yaw_deg: float,
    base_cfg: NavConfig,
    vision: VisionContext,
    args: argparse.Namespace,
) -> int:
    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
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
        f"Grasp Opti (A): x={grasp_xy[0]:.3f} y={grasp_xy[1]:.3f} "
        f"body_yaw={grasp_yaw_deg:.1f} deg"
    )
    print(
        f"Pour Opti (B):  x={pour_xy[0]:.3f} y={pour_xy[1]:.3f} "
        f"body_yaw={pour_yaw_deg:.1f} deg"
    )
    print(f"Vision object: {args.object!r}  orientation_source={args.orientation_source}")
    print(f"Gemini response path: {vision.save_path}")
    print(f"approach_dz={motion.approach_dz_m} m  pour_tilt={motion.pour_tilt_deg:.0f}°")
    print(
        "Keys (ENTER to submit): "
        "[empty]=advance phase | q=quit"
    )
    print(
        "(No motion at startup — first ENTER drives base to A; vision runs at A.)"
    )

    phase = Phase.AWAIT_DRIVE_TO_GRASP
    target: LatchedTarget | None = None
    pour_world: np.ndarray | None = None
    slerp: OrientationSlerpState | None = None
    poured_ori: np.ndarray | None = None
    stdin_dead = False
    print(_phase_hint(phase))

    try:
        while True:
            now = time.perf_counter()

            if phase in (Phase.POURING, Phase.RETURNING) and slerp is not None:
                if _tick_orientation_slerp(redis_client, slerp, motion, now):
                    if phase == Phase.POURING:
                        poured_ori = slerp.R_end.copy()
                        phase = Phase.POURED
                        slerp = None
                        print("Pour complete (hold).")
                    else:
                        phase = Phase.RETURNED
                        slerp = None
                        print("Return orientation complete (at pour pose).")
                    print(_phase_hint(phase))

            if stdin_dead:
                time.sleep(POUR_TICK_DT_S)
                continue

            line = _stdin_line_ready(POUR_TICK_DT_S)
            if line is None:
                continue
            if line is _STDIN_EOF:
                print(
                    "stdin closed (no terminal attached). State will NOT advance.\n"
                    "Run this controller directly in a terminal. Ctrl+C to quit.",
                    file=sys.stderr,
                )
                stdin_dead = True
                continue
            token = line.strip().lower()
            if token in ("q", "quit", "exit"):
                print("Quit requested.")
                return 0
            if token != "":
                print(
                    f"(unknown input: {token!r}; press ENTER to advance, q to quit)"
                )
                continue

            if phase == Phase.AWAIT_DRIVE_TO_GRASP:
                if _drive_base(
                    redis_client,
                    grasp_xy,
                    grasp_yaw_deg,
                    label="[1] Base → grasp (A)",
                    base_cfg=base_cfg,
                ):
                    phase = Phase.AWAIT_VISION

            elif phase == Phase.AWAIT_VISION:
                latched = _run_vision_detect(
                    redis_client, args, motion, vision
                )
                if latched is not None:
                    target = latched
                    phase = Phase.AWAIT_ABOVE_GRASP

            elif phase == Phase.AWAIT_ABOVE_GRASP:
                assert target is not None
                # _do_move_above_grasp adds approach_dz to grasp contact internally.
                _do_move_above_grasp(
                    redis_client,
                    target.grasp_pos,
                    target.grasp_ori,
                    motion,
                )
                phase = Phase.ABOVE_GRASP

            elif phase == Phase.ABOVE_GRASP:
                assert target is not None
                _do_descend_to_grasp(
                    redis_client,
                    target.grasp_pos,
                    target.grasp_ori,
                    label="[3] Descend to grasp contact",
                )
                phase = Phase.AT_GRASP

            elif phase == Phase.AT_GRASP:
                _do_grasp_object(redis_client, motion)
                phase = Phase.CLOSED

            elif phase == Phase.CLOSED:
                assert target is not None
                pour_world = _do_lift(
                    redis_client, target.grasp_pos, target.grasp_ori, motion
                )
                phase = Phase.LIFTED

            elif phase == Phase.LIFTED:
                if _drive_base(
                    redis_client,
                    pour_xy,
                    pour_yaw_deg,
                    label="[6] Base → pour (B)",
                    base_cfg=base_cfg,
                ):
                    phase = Phase.ABOVE_POUR

            elif phase == Phase.ABOVE_POUR:
                assert pour_world is not None
                slerp = _start_pour_step(redis_client, pour_world, motion, now)
                if slerp is not None:
                    phase = Phase.POURING

            elif phase == Phase.POURING:
                print("Pour in progress — wait for it to finish.")

            elif phase == Phase.POURED:
                assert pour_world is not None and poured_ori is not None
                assert target is not None
                slerp = _start_return_step(
                    redis_client,
                    pour_world,
                    target.grasp_ori,
                    poured_ori,
                    now,
                )
                phase = Phase.RETURNING

            elif phase == Phase.RETURNING:
                print("Return in progress — wait for it to finish.")

            elif phase == Phase.RETURNED:
                assert pour_world is not None and target is not None
                _do_retract_above_pour(
                    redis_client, pour_world, target.grasp_ori, motion
                )
                phase = Phase.READY_TO_RELEASE

            elif phase == Phase.READY_TO_RELEASE:
                _do_open_gripper(redis_client, motion)
                phase = Phase.DONE
                print("[10] Released at pour site (B).")

            elif phase == Phase.DONE:
                print("Sequence done — q to quit.")

            print(_phase_hint(phase))
    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")
        return 0


def main() -> int:
    args = parse_args()
    try:
        redis_client = connect_redis(args.redis_host, args.redis_port)
    except redis.RedisError as exc:
        print(f"Redis connect failed: {exc}", file=sys.stderr)
        return 1

    err = validate_config(redis_client)
    if err is not None:
        return err

    motion = MotionParams(
        approach_dz_m=args.approach_dz,
        pour_tilt_deg=args.pour_tilt_deg,
        pour_axis=args.pour_axis,
        tilt_duration_s=args.tilt_duration,
        gripper_open_width=args.gripper_open_width,
        gripper_pregrasp_width=args.gripper_pregrasp_width,
        gripper_close_width=args.gripper_close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        gripper_pregrasp_settle_s=args.gripper_pregrasp_settle,
        gripper_grasp_settle_s=args.gripper_grasp_settle,
    )

    base_cfg = NavConfig(
        marker_yaw_offset_deg=float(args.marker_yaw_offset_deg),
        tolerance_m=float(args.base_tolerance_in) * INCHES_TO_METERS,
        tolerance_yaw_rad=math.radians(float(args.base_yaw_tolerance_deg)),
        log_hz=float(args.base_log_hz),
        monitor=args.base_monitor,
        print_plan=args.base_print_plan,
        print_log=True,
        stop_on_exit=False,
    )

    grasp_xy = (float(args.grasp_opti_x), float(args.grasp_opti_y))
    pour_xy = (float(args.pour_opti_x), float(args.pour_opti_y))

    prompt = gp.build_prompt(args.object, args.prompt)
    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")

    try:
        gemini_client = gp.make_genai_client(gp.resolve_api_key())
    except Exception as e:
        print(f"Gemini client setup failed: {e}", file=sys.stderr)
        return 1

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
    vision = VisionContext(
        gemini_client=gemini_client,
        pipeline=pipeline,
        align=align,
        depth_scale=depth_scale,
        color_intrinsics=color_intrinsics,
        prompt=prompt,
        save_path=save_path,
        miss_counter=[0],
    )

    try:
        return run_loop(
            redis_client,
            motion,
            grasp_xy,
            float(args.grasp_opti_yaw_deg),
            pour_xy,
            float(args.pour_opti_yaw_deg),
            base_cfg,
            vision,
            args,
        )
    finally:
        if pipeline is not None:
            pipeline.stop()


if __name__ == "__main__":
    sys.exit(main())
