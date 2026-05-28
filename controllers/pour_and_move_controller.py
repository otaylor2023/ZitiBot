#!/usr/bin/env python3
"""OptiTrack base moves + fixed-pose grasp/pour on OpenSai Franka (Redis only).

Drives the mobile base to two Motive lab poses, then runs the same arm sequence
as :mod:`grasp_and_pour_controller` (pick, lift, pour, return, retract, release).
Every transition is **manually gated by ENTER** (pour/return slerps auto-complete).

Startup (no ENTER, no motion): controller prints the plan and waits.

Then one ENTER per step:

- ENTER (1) — drive base to grasp Opti pose (A).
- ENTER (2) — move arm above pick + open gripper to ``--gripper-open-width``.
- ENTER (3) — descend to pick.
- ENTER (4) — pregrasp + grasp.
- ENTER (5) — lift (+``--approach-dz`` in world Z).
- ENTER (6) — drive base to pour Opti pose (B).
- ENTER (7) — start pour slerp.
- ENTER (8) — start return slerp (after pour finishes).
- ENTER (9) — retract up at pour site.
- ENTER (10) — open gripper (release at B).

Requires:

- ``tidybot_base/redis_driver.py`` on the robot mini-PC.
- OptiTrack on Redis (``tidybot01::pos`` / ``ori`` / ``tracking_valid``).
- OpenSai cartesian controller + Franka gripper Redis driver.

Usage::

  python ZitiBot/controllers/pour_and_move_controller.py
  python ZitiBot/controllers/pour_and_move_controller.py --grasp-opti-x -2.52 --grasp-opti-y 2.9
"""

from __future__ import annotations

import argparse
import enum
import math
import sys
import time
from dataclasses import replace

import numpy as np
import redis

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
    PICK_POSITION,
    POUR_TICK_DT_S,
    _STDIN_EOF,
    _do_descend_to_grasp,
    _do_grasp_object,
    _do_move_above_grasp,
    _do_open_gripper,
    _publish_cartesian,
    _start_orientation_slerp,
    _stdin_line_ready,
    _tick_orientation_slerp,
    pour_orientation_end,
    read_current_ee_world,
    resolve_gripper_open_width,
    validate_config,
)
from tidybot_base.opti_nav import NavConfig, navigate_to_opti_pose
from tidybot_base.opti_planner import DEFAULT_MARKER_YAW_OFFSET_DEG
from tidybot_base.redis_io import connect_redis

INCHES_TO_METERS = 0.0254

# Default Motive lab poses (m, body yaw deg).
DEFAULT_GRASP_OPTI_X = -3.34
DEFAULT_GRASP_OPTI_Y = 0.91
DEFAULT_GRASP_OPTI_YAW_DEG = 180.0
DEFAULT_POUR_OPTI_X = -3.29
DEFAULT_POUR_OPTI_Y = 1.51
DEFAULT_POUR_OPTI_YAW_DEG = 180.0

DEFAULT_BASE_TOLERANCE_IN = 1.0
DEFAULT_BASE_YAW_TOLERANCE_DEG = 5.0
DEFAULT_BASE_LOG_HZ = 2.0


class Phase(enum.Enum):
    AWAIT_DRIVE_TO_GRASP = "AWAIT_DRIVE_TO_GRASP"
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


def _phase_hint(phase: Phase) -> str:
    if phase == Phase.AWAIT_DRIVE_TO_GRASP:
        return "Next: ENTER = drive base to grasp pose (A)"
    if phase == Phase.AWAIT_ABOVE_GRASP:
        return "Next: ENTER = move arm above grasp + open gripper"
    if phase == Phase.ABOVE_GRASP:
        return "Next: ENTER = descend to pick"
    if phase == Phase.AT_GRASP:
        return "Next: ENTER = close gripper"
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


def _do_lift(
    redis_client,
    pick_pos: np.ndarray,
    grasp_ori: np.ndarray,
    motion: MotionParams,
) -> np.ndarray:
    """Lift to approach height above pick (reused at pour site after base move)."""
    lift = pick_pos + np.array([0.0, 0.0, motion.approach_dz_m], dtype=np.float64)
    _publish_cartesian(redis_client, lift, grasp_ori)
    print(f"[4] Lift: pos={lift.tolist()}")
    return lift


def _start_pour_step(
    redis_client,
    pour_world: np.ndarray,
    motion: MotionParams,
    now: float,
) -> OrientationSlerpState | None:
    pose = read_current_ee_world(redis_client)
    if pose is None:
        print("Could not read current EE pose from Redis; not starting pour.")
        return None
    _, cur_ori = pose
    R_start = cur_ori.copy()
    R_end = pour_orientation_end(R_start, motion.pour_tilt_deg, motion.pour_axis)
    return _start_orientation_slerp(
        redis_client,
        pour_world,
        R_start,
        R_end,
        now,
        label=(
            f"[6] Pour started: slerp {motion.pour_tilt_deg:.0f}° "
            f"about world +{motion.pour_axis.upper()}."
        ),
    )


def _start_return_step(
    redis_client,
    pour_world: np.ndarray,
    grasp_ori: np.ndarray,
    poured_ori: np.ndarray,
    now: float,
) -> OrientationSlerpState:
    return _start_orientation_slerp(
        redis_client,
        pour_world,
        poured_ori,
        grasp_ori,
        now,
        label="[7] Return started: slerp back to grasp orientation.",
    )


def _do_retract_above_pour(
    redis_client,
    pour_world: np.ndarray,
    grasp_ori: np.ndarray,
    motion: MotionParams,
) -> np.ndarray:
    """Retract vertically at pour site (another approach_dz above hold pose)."""
    retract = np.asarray(pour_world, dtype=np.float64).reshape(3).copy()
    retract[2] = float(retract[2]) + float(motion.approach_dz_m)
    _publish_cartesian(redis_client, retract, grasp_ori)
    print(f"[8] Retract up at pour site: pos={retract.tolist()}")
    return retract


def _drive_base(
    redis_client,
    target_xy: tuple[float, float],
    body_yaw_deg: float,
    *,
    label: str,
    base_cfg: NavConfig,
) -> bool:
    """Block until base reaches Opti pose. Returns False on failure (retry ENTER)."""
    print(label)
    cfg = replace(
        base_cfg,
        stop_on_exit=False,
        print_log=True,
        exit_on_success=True,
    )
    result = navigate_to_opti_pose(
        redis_client,
        target_xy,
        target_body_yaw_deg=body_yaw_deg,
        config=cfg,
        wait_for_tracking=True,
    )
    if result.success:
        xy_err = result.final_xy_error_m
        yaw_err = result.final_body_yaw_error_deg
        err_note = ""
        if xy_err is not None:
            err_note += f" xy_err={xy_err * 1000:.1f} mm"
        if yaw_err is not None:
            err_note += f" yaw_err={yaw_err:.2f} deg"
        print(f"  Base nav OK ({result.elapsed_s:.1f} s).{err_note}")
        return True
    print(
        f"  Base nav failed: {result.reason} "
        f"(elapsed {result.elapsed_s:.1f} s). Press ENTER to retry or q to quit.",
        file=sys.stderr,
    )
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "OptiTrack base + fixed-pose grasp/pour, ENTER-gated "
            "(OpenSai Franka Redis)."
        )
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)

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
    pick_pos = PICK_POSITION.copy()
    grasp_ori = GRASP_ORIENTATION.copy()

    print(
        f"Grasp Opti (A): x={grasp_xy[0]:.3f} y={grasp_xy[1]:.3f} "
        f"body_yaw={grasp_yaw_deg:.1f} deg"
    )
    print(
        f"Pour Opti (B):  x={pour_xy[0]:.3f} y={pour_xy[1]:.3f} "
        f"body_yaw={pour_yaw_deg:.1f} deg"
    )
    print(
        f"Arm pick (world): {pick_pos.tolist()}  "
        f"approach_dz={motion.approach_dz_m} m"
    )
    print(
        "Keys (ENTER to submit): "
        "[empty]=advance phase | q=quit"
    )
    print(
        "(No motion at startup — first ENTER drives the base; arm only "
        "moves once the base is at the grasp pose.)"
    )

    phase = Phase.AWAIT_DRIVE_TO_GRASP
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
                    phase = Phase.AWAIT_ABOVE_GRASP
            elif phase == Phase.AWAIT_ABOVE_GRASP:
                _do_move_above_grasp(redis_client, pick_pos, grasp_ori, motion)
                phase = Phase.ABOVE_GRASP
            elif phase == Phase.ABOVE_GRASP:
                _do_descend_to_grasp(
                    redis_client,
                    pick_pos,
                    grasp_ori,
                    label="[2] Descend to pick",
                )
                phase = Phase.AT_GRASP
            elif phase == Phase.AT_GRASP:
                _do_grasp_object(redis_client, motion)
                phase = Phase.CLOSED
            elif phase == Phase.CLOSED:
                pour_world = _do_lift(redis_client, pick_pos, grasp_ori, motion)
                phase = Phase.LIFTED
            elif phase == Phase.LIFTED:
                if _drive_base(
                    redis_client,
                    pour_xy,
                    pour_yaw_deg,
                    label="[5] Base → pour (B)",
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
                slerp = _start_return_step(
                    redis_client,
                    pour_world,
                    grasp_ori,
                    poured_ori,
                    now,
                )
                phase = Phase.RETURNING
            elif phase == Phase.RETURNING:
                print("Return in progress — wait for it to finish.")
            elif phase == Phase.RETURNED:
                assert pour_world is not None
                _do_retract_above_pour(redis_client, pour_world, grasp_ori, motion)
                phase = Phase.READY_TO_RELEASE
            elif phase == Phase.READY_TO_RELEASE:
                _do_open_gripper(redis_client, motion)
                phase = Phase.DONE
                print("[9] Released at pour site (B).")
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

    return run_loop(
        redis_client,
        motion,
        grasp_xy,
        float(args.grasp_opti_yaw_deg),
        pour_xy,
        float(args.pour_opti_yaw_deg),
        base_cfg,
    )


if __name__ == "__main__":
    sys.exit(main())
