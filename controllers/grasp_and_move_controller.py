#!/usr/bin/env python3
"""Vision-guided pan pick + base transport + place at oven (modular system).

Sequence:
  1. Base -> PAN_STATION
  2. Gemini pan grasp pose -> pick up pan -> lift
  3. Base -> OVEN_DOOR
  4. Place pan at oven target

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/grasp_and_move_controller.py
  ./ZitiBot/launch_zitibot_full.sh controllers/grasp_and_move_controller.py -- --step
  ./ZitiBot/launch_zitibot_full.sh controllers/grasp_and_move_controller.py -- --log

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
TidyBot base redis_driver, RealSense, OptiTrack on Redis, and
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm, base, gripper
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    DEFAULT_POS_TOL_M,
    OBJECT_DEFAULTS,
    OVEN_EE_ORIENTATION,
    OVEN_EE_POSITION,
    OVEN_EE_WAYPOINTS,
    BaseWaypoint,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_core.runner import step_gate
from zitibot_tasks import gemini, grasp

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_RESPONSE_PATH = _CONTROLLERS.parent / "logs" / "gemini_response_pan.png"

# Post-grasp carry lift, folded into grasp.object via lift_dz_m so it is one
# continuous post-grasp motion (matching pour_and_move_controller.py).
CARRY_LIFT_M = 0.15

def pick_up_pan(ctx: TaskContext) -> tuple[np.ndarray, np.ndarray]:
    """Gemini pan grasp pose -> grasp pan -> lift to carry height.

    Returns ``(pick_pos, grip_R)``. After this function the EE is holding the
    pan at ``pick_pos + [0, 0, CARRY_LIFT_M]`` ready for base transport.
    """
    obj = Object.PAN
    pose = gemini.find_grasp_pose(ctx, obj)
    pick_pos = pose.position
    grip_R = pose.orientation
    print(f"Detected pan handle grasp: {pick_pos.tolist()}")
    if pose.rim_yaw_applied and pose.rim_yaw_deg is not None:
        print(f"Detected handle yaw: {pose.rim_yaw_deg:+.2f} deg")
    grasp.object(ctx, obj, pick_pos=pick_pos, ori=grip_R, lift_dz_m=CARRY_LIFT_M)
    return pick_pos, grip_R


def place_pan_in_oven(ctx: TaskContext, place_pos: np.ndarray) -> None:
    """Place the held pan by following the measured oven insertion waypoints."""
    spec = OBJECT_DEFAULTS[Object.PAN]
    place = np.asarray(place_pos, dtype=np.float64).reshape(3).copy()
    place_R = OVEN_EE_ORIENTATION

    for idx, (waypoint_pos, waypoint_R) in enumerate(OVEN_EE_WAYPOINTS, start=1):
        wp = np.asarray(waypoint_pos, dtype=np.float64).reshape(3).copy()
        arm.move_to(
            ctx,
            wp,
            waypoint_R,
            label=f"[place:pan] oven waypoint {idx} {wp.tolist()}",
            tol_m=spec.approach_tol,
        )
    arm.move_to(
        ctx,
        place,
        place_R,
        label=f"[place:pan] slide horizontally to oven target {place.tolist()}",
        tol_m=spec.grasp_tol,
    )
    step_gate(ctx, "[place:pan] release pan in oven (open gripper)")
    gripper.open_gripper(
        ctx.redis,
        spec.open_width,
        speed=spec.speed,
        force=spec.force,
        use_max_mode=True,
    )
    time.sleep(spec.grasp_settle_s)
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[place:pan] return home after release {ARM_HOME_POSITION.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    print(f"[place:pan] released at {place.tolist()} and returned home")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vision pan pick at pan station, place at oven door."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate each motion/grip step inside subtasks.",
    )
    p.add_argument(
        "--log",
        action="store_true",
        help=(
            "Record per-move EE/base position-vs-time plots to "
            "logs/graphs/<controller>_NNNN/."
        ),
    )
    p.add_argument(
        "--place",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Oven target position in the arm/world frame (m). Defaults to measured OVEN_PLACE_POSITION.",
    )
    p.add_argument(
        "--endeffector-transform-key",
        default=DEFAULT_ENDEFFECTOR_TRANSFORM_KEY,
        help="Redis key for the 4x4 base->flange transform.",
    )
    p.add_argument(
        "--gemini-response-path",
        default=str(DEFAULT_GEMINI_RESPONSE_PATH),
        help="Where to save the annotated Gemini RGB+depth response image.",
    )
    return p.parse_args()


def _resolve_place_pos(override: list[float] | None) -> np.ndarray:
    if override is not None:
        return np.array(override, dtype=np.float64)
    place = OBJECT_DEFAULTS[Object.PAN].place_pose
    if place is None:
        return OVEN_EE_POSITION.copy()
    return np.asarray(place, dtype=np.float64).reshape(3).copy()


def run_pan_to_oven_cycle(
    ctx: TaskContext,
    *,
    place_pos: np.ndarray | None = None,
    gemini_response_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vision pan pick at PAN_STATION → place at oven door.

    Returns ``(pick_pos, place_pos)`` for logging.
    """
    # Always rewrite ``ctx.gemini_response_path`` so the pan grasp
    # always lands in the pan-specific file. Without this, a stale
    # path set by an earlier step in the kitchen routine (e.g. the
    # parmesan strip detection in ``grasp_and_pour_jar_controller``)
    # would silently overwrite the prior step's response when the
    # pan grasp ran with no per-call override.
    ctx.gemini_response_path = (
        str(gemini_response_path)
        if gemini_response_path is not None
        else str(DEFAULT_GEMINI_RESPONSE_PATH)
    )
    if place_pos is None:
        place = OBJECT_DEFAULTS[Object.PAN].place_pose
        place_pos = (
            OVEN_EE_POSITION.copy()
            if place is None
            else np.asarray(place, dtype=np.float64).reshape(3).copy()
        )
    else:
        place_pos = np.asarray(place_pos, dtype=np.float64).reshape(3).copy()

    print(f"Oven target: {place_pos.tolist()}")
    print("Oven orientation:")
    for row in OVEN_EE_ORIENTATION:
        print(f"  [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")

    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[arm] move to home {ARM_HOME_POSITION.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    # Holonomic single-phase to PAN_STATION (default motion) —
    # the pan grasp is vision-localized by Gemini at the counter,
    # so it tolerates a few cm of base XY error without needing
    # the three-phase precision landing. Faster on the long
    # cross-room traverse.
    #
    # Three-phase landing (A holonomic approach, B rotate in
    # place, C pure translation) is kept for OVEN_DOOR — the oven
    # insertion follows hand-taught EE waypoints, so the cart has
    # to be squared up to the door for the arm to clear the
    # opening on the way in.
    base.go_to_pose(ctx, BaseWaypoint.PAN_STATION)
    pick_pos, grip_R = pick_up_pan(ctx)
    base.go_to_pose(ctx, BaseWaypoint.OVEN_DOOR, motion="three_phase")
    place_pan_in_oven(ctx, place_pos)
    return pick_pos, place_pos


def main() -> int:
    args = parse_args()
    try:
        place_pos = _resolve_place_pos(args.place)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    print(f"Step mode: {'on' if args.step else 'off'}")
    print(f"Gemini response: {args.gemini_response_path}")

    try:
        pick_pos, resolved_place = run_pan_to_oven_cycle(
            ctx,
            place_pos=place_pos,
            gemini_response_path=args.gemini_response_path,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()

    print(
        f"Grasp-and-move complete: pan moved from {pick_pos.tolist()} "
        f"to {resolved_place.tolist()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
