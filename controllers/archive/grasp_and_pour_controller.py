#!/usr/bin/env python3
"""Fixed-pose grasp + transport/pour on OpenSai Franka (Redis only).

Thin CLI wrapper using ``zitibot_tasks.grasp`` and ``zitibot_tasks.pour``.

Usage::

  python ZitiBot/controllers/grasp_and_pour_controller.py
  python ZitiBot/controllers/grasp_and_pour_controller.py --step
  python ZitiBot/controllers/grasp_and_pour_controller.py --approach-dz 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm
from zitibot_core.constants import (
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    OBJECT_DEFAULTS,
    Object,
)
from zitibot_core.context import make_context
from zitibot_tasks import grasp, pour


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-pose grasp + pour sequence (OpenSai Franka Redis)."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate each motion step inside subtasks",
    )
    p.add_argument("--approach-dz", type=float, default=DEFAULT_APPROACH_DZ_M)
    p.add_argument("--pour-tilt-deg", type=float, default=DEFAULT_POUR_TILT_DEG)
    p.add_argument("--pour-axis", default=DEFAULT_POUR_AXIS, choices=("x", "y"))
    p.add_argument("--tilt-duration-s", type=float, default=DEFAULT_TILT_DURATION_S)
    p.add_argument("--gripper-open-width", type=float, default=None)
    p.add_argument("--gripper-pregrasp-width", type=float, default=0.05)
    p.add_argument("--gripper-close-width", type=float, default=0.0)
    p.add_argument("--gripper-speed", type=float, default=0.1)
    p.add_argument("--gripper-force", type=float, default=50.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    obj = Object.PASTA_BOWL
    spec = OBJECT_DEFAULTS[obj]
    spec.approach_dz = args.approach_dz
    spec.open_width = args.gripper_open_width
    spec.pregrasp_width = args.gripper_pregrasp_width
    spec.close_width = args.gripper_close_width
    spec.speed = args.gripper_speed
    spec.force = args.gripper_force

    pick_pos = spec.pick_pose
    pour_pos = spec.pour_pose
    if pick_pos is None or pour_pos is None:
        print("Error: PASTA_BOWL missing pick/pour poses in OBJECT_DEFAULTS", file=sys.stderr)
        return 1

    ctx = make_context(args, step=args.step)
    grip_R = spec.grasp_ori.copy()
    print(f"Pick  = {pick_pos.tolist()}")
    print(f"Pour  = {pour_pos.tolist()}")

    try:
        grasp.object(ctx, obj, pick_pos=pick_pos, ori=grip_R)

        arm.move_to(
            ctx,
            pour_pos,
            grip_R,
            label=f"[transport] move to pour {pour_pos.tolist()}",
            tol_m=spec.approach_tol,
        )

        R_poured = pour.into(
            ctx,
            pour_pos,
            tilt_deg=args.pour_tilt_deg,
            axis=args.pour_axis,
            duration_s=args.tilt_duration_s,
        )
        pour.return_upright(
            ctx,
            pour_pos,
            R_poured,
            grip_R,
            duration_s=args.tilt_duration_s,
        )

        retract = pour_pos.copy()
        retract[2] = max(float(pour_pos[2]), float(pick_pos[2] + spec.approach_dz))
        arm.move_to(
            ctx,
            retract,
            grip_R,
            label=f"[return] retract at pour site {retract.tolist()}",
            tol_m=spec.approach_tol,
        )

        arm.move_to(
            ctx,
            pick_pos,
            grip_R,
            label=f"[return] move to original pick {pick_pos.tolist()}",
            tol_m=spec.grasp_tol,
        )

        grasp.place(ctx, obj, place_pos=pick_pos, ori=grip_R)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print("Sequence complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Re-export legacy symbols for controllers not yet migrated off grasp_and_pour_controller.
from zitibot_core.legacy_grasp_pour import (  # noqa: E402,F401
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_GRIPPER_FORCE,
    DEFAULT_GRIPPER_GRASP_SETTLE_S,
    DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    DEFAULT_GRIPPER_PREGRASP_WIDTH,
    DEFAULT_GRIPPER_SPEED,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    GRASP_ORIENTATION,
    GRASP_POSITION,
    GRIPPER_MODE_GRASP,
    GRIPPER_MODE_MOVE,
    GRIPPER_MODE_OPEN_MAX,
    MotionParams,
    OrientationSlerpState,
    PICK_POSITION,
    POUR_POSITION,
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
    _try_redis,
    pour_orientation_end,
    read_current_ee_world,
    read_gripper_current_width,
    resolve_gripper_open_width,
    set_gripper_width,
    validate_config,
)
