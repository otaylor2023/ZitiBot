#!/usr/bin/env python3
"""Basic base-move controller (OpenSai Franka + TidyBot base, modular system).

Moves the arm to ``ARM_HOME_POSITION`` / ``ARM_HOME_ORIENTATION``, then
drives the base to ``BaseWaypoint.OVEN_DOOR`` by default. Raw Opti
``(x, y, yaw)`` overrides are still available via CLI flags.

Usage::

  # Default move (arm home -> base at oven door)
  ./ZitiBot/launch_zitibot_full.sh --no-gripper controllers/move_base_controller.py

  # ENTER-gate each step
  ./ZitiBot/launch_zitibot_full.sh --no-gripper controllers/move_base_controller.py -- --step

  # Override the base target with raw Opti pose
  ./ZitiBot/launch_zitibot_full.sh --no-gripper controllers/move_base_controller.py -- \
      --base-x 1.12 --base-y -2.64 --base-yaw-deg 180

Requires the TidyBot base ``redis_driver.py`` to be running and Motive/NatNet
publishing ``tidybot01::pos`` / ``tidybot01::ori`` / ``tidybot01::tracking_valid``
to Redis.
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
    BASE_WAYPOINTS,
    BaseWaypoint,
)
from zitibot_core.context import make_context

DEFAULT_ARM_POSE = ARM_HOME_POSITION.copy()
DEFAULT_ARM_ORIENTATION = ARM_HOME_ORIENTATION.copy()
_DEFAULT_BASE_WAYPOINT = BaseWaypoint.OVEN_DOOR
_DEFAULT_BASE_TARGET = BASE_WAYPOINTS[_DEFAULT_BASE_WAYPOINT]
DEFAULT_BASE_X = _DEFAULT_BASE_TARGET.x_m
DEFAULT_BASE_Y = _DEFAULT_BASE_TARGET.y_m
DEFAULT_BASE_YAW_DEG = _DEFAULT_BASE_TARGET.yaw_deg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Move the arm to a fixed home pose, then drive the base to an Opti pose."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate the arm move (base move runs straight through).",
    )
    p.add_argument(
        "--arm-pose",
        nargs=3,
        type=float,
        default=DEFAULT_ARM_POSE.tolist(),
        metavar=("X", "Y", "Z"),
        help=f"Arm home pose in world frame (m). Default: {DEFAULT_ARM_POSE.tolist()}",
    )
    p.add_argument("--base-x", type=float, default=DEFAULT_BASE_X)
    p.add_argument("--base-y", type=float, default=DEFAULT_BASE_Y)
    p.add_argument("--base-yaw-deg", type=float, default=DEFAULT_BASE_YAW_DEG)
    p.add_argument(
        "--skip-arm",
        action="store_true",
        help="Only move the base; leave the arm wherever it is.",
    )
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="Only move the arm; do not drive the base.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)

    arm_pose = np.array(args.arm_pose, dtype=np.float64)
    grip_R = DEFAULT_ARM_ORIENTATION.copy()

    # Use the named waypoint when the user hasn't overridden any base flag,
    # so the canonical OVEN_DOOR pose (with its label) is what go_to_pose
    # actually sees. Any raw --base-x/y/yaw override switches to raw mode.
    base_overridden = (
        args.base_x != DEFAULT_BASE_X
        or args.base_y != DEFAULT_BASE_Y
        or args.base_yaw_deg != DEFAULT_BASE_YAW_DEG
    )

    print(f"Arm pose : {arm_pose.tolist()}")
    if base_overridden:
        print(
            f"Base goal: x={args.base_x:.3f} m, y={args.base_y:.3f} m, "
            f"yaw={args.base_yaw_deg:.1f} deg (raw override)"
        )
    else:
        print(
            f"Base goal: {_DEFAULT_BASE_WAYPOINT.name} "
            f"({args.base_x:.3f}, {args.base_y:.3f}, {args.base_yaw_deg:.1f} deg)"
        )

    try:
        if not args.skip_arm:
            arm.move_to(
                ctx,
                arm_pose,
                grip_R,
                label=f"[move-base] arm to home {arm_pose.tolist()}",
            )
        else:
            print("[move-base] skipping arm move (--skip-arm)")

        if not args.skip_base:
            if base_overridden:
                base.go_to_pose(
                    ctx,
                    x_m=args.base_x,
                    y_m=args.base_y,
                    yaw_deg=args.base_yaw_deg,
                    label=(
                        f"[move-base] base to ({args.base_x:.3f}, {args.base_y:.3f}, "
                        f"{args.base_yaw_deg:.1f}°)"
                    ),
                )
            else:
                base.go_to_pose(ctx, _DEFAULT_BASE_WAYPOINT)
        else:
            print("[move-base] skipping base move (--skip-base)")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130

    print("Move-base complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
