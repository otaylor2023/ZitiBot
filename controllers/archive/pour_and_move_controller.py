#!/usr/bin/env python3
"""Vision-guided bowl pick + base transport + pour (modular system).

Sequence:
  1. Base → INGREDIENT_STATION
  2. Gemini grasp pose → pick up pasta bowl → lift
  3. Base → MIXING_STATION
  4. Gemini pour target → move arm → pour → return upright (still holding bowl)

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/pour_and_move_controller.py
  ./ZitiBot/launch_zitibot_full.sh controllers/pour_and_move_controller.py -- --step

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
TidyBot base redis_driver, RealSense, OptiTrack on Redis, and
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm, base
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    DEFAULT_POS_TOL_M,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    OBJECT_DEFAULTS,
    BaseWaypoint,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_tasks import gemini, grasp, pour

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_RESPONSE_PATH = _CONTROLLERS.parent / "logs" / "gemini_response.png"

# Additional Z lift applied after grasp.object so the bowl is clear of the
# counter before the base translates. grasp.object already lifts to
# pick + approach_dz (3 cm for PASTA_BOWL); this adds on top of the grasp
# pose, not on top of "above".
CARRY_LIFT_M = 0.25


def pick_up_bowl(ctx: TaskContext) -> tuple[np.ndarray, np.ndarray]:
    """Gemini grasp pose → grasp pasta bowl → lift to carry height.

    Returns ``(pick_pos, grip_R)``. After this function the EE is holding the
    bowl at ``pick_pos + [0, 0, CARRY_LIFT_M]`` ready for base transport.
    The carry lift is folded into ``grasp.object`` via ``lift_dz_m`` so it's
    one continuous post-grasp motion (no extra ENTER prompt in --step mode).
    """
    obj = Object.PASTA_BOWL
    pose = gemini.find_grasp_pose(ctx, obj)
    pick_pos = pose.position
    grip_R = pose.orientation
    print(f"Detected bowl grasp: {pick_pos.tolist()}")
    if pose.rim_yaw_applied and pose.rim_yaw_deg is not None:
        print(f"Detected rim yaw: {pose.rim_yaw_deg:+.2f} deg")
    grasp.object(ctx, obj, pick_pos=pick_pos, ori=grip_R, lift_dz_m=CARRY_LIFT_M)
    return pick_pos, grip_R


def pour_into_mixing_bowl(
    ctx: TaskContext,
    grip_R: np.ndarray,
    *,
    tilt_deg: float = DEFAULT_POUR_TILT_DEG,
    axis: str = DEFAULT_POUR_AXIS,
    duration_s: float = DEFAULT_TILT_DURATION_S,
) -> None:
    """Gemini pour target → move arm → pour → return upright (still holding bowl)."""
    spec = OBJECT_DEFAULTS[Object.PASTA_BOWL]
    pour_target = gemini.find_pour_target(ctx, Object.MIXING_BOWL)
    print(f"Detected pour target: {pour_target.tolist()}")
    arm.move_to(
        ctx,
        pour_target,
        grip_R,
        label=f"[pour] move to pour target {pour_target.tolist()}",
        tol_m=spec.approach_tol,
    )
    R_poured = pour.into(
        ctx,
        pour_target,
        tilt_deg=tilt_deg,
        axis=axis,
        duration_s=duration_s,
    )
    pour.return_upright(
        ctx,
        pour_target,
        R_poured,
        grip_R,
        duration_s=duration_s,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vision bowl pick at ingredient station, pour at mixing station."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate each motion/grip step inside subtasks.",
    )
    p.add_argument("--pour-tilt-deg", type=float, default=DEFAULT_POUR_TILT_DEG)
    p.add_argument("--pour-axis", default=DEFAULT_POUR_AXIS, choices=("x", "y"))
    p.add_argument("--tilt-duration-s", type=float, default=DEFAULT_TILT_DURATION_S)
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


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    print(f"Step mode: {'on' if args.step else 'off'}")
    print(f"Gemini response: {args.gemini_response_path}")

    ee_pose = arm.read_current_ee_world(ctx.redis)
    if ee_pose is None:
        print("[arm] startup pose: <unavailable on Redis>")
    else:
        ee_pos, ee_R = ee_pose
        np.set_printoptions(precision=4, suppress=True)
        print(
            f"[arm] startup EE world position: "
            f"[{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}] m"
        )
        print("[arm] startup EE world orientation (3x3 rot):")
        for row in ee_R:
            print(f"  [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")

    try:
        arm.move_to(
            ctx,
            ARM_HOME_POSITION,
            ARM_HOME_ORIENTATION,
            label=f"[arm] move to home {ARM_HOME_POSITION.tolist()}",
            tol_m=DEFAULT_POS_TOL_M,
        )
        base.go_to_pose(ctx, BaseWaypoint.INGREDIENT_STATION)
        _, grip_R = pick_up_bowl(ctx)
        base.go_to_pose(ctx, BaseWaypoint.MIXING_STATION)
        pour_into_mixing_bowl(
            ctx,
            grip_R,
            tilt_deg=args.pour_tilt_deg,
            axis=args.pour_axis,
            duration_s=args.tilt_duration_s,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()

    print("Pour-and-move complete (bowl still in gripper).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
