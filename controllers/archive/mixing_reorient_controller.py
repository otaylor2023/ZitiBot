#!/usr/bin/env python3
"""Fixed-pose grasp-ladle + reorient + stir-bowl on OpenSai Franka (Redis only).

Like mixing_simple_controller, but SLERPs the ladle into a tilted mixing
orientation above the bowl before descending, then returns upright after.
Reorientation reuses the same SLERP primitive as pour.into/return_upright.

Usage::

  python ZitiBot/controllers/mixing_reorient_controller.py
  python ZitiBot/controllers/mixing_reorient_controller.py --step
  python ZitiBot/controllers/mixing_reorient_controller.py \\
      --bowl-xyz 0.17 0.62 0.50 --tilt-deg 45 --tilt-axis x
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm
from zitibot_core.constants import (
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_GRIPPER_FORCE,
    DEFAULT_GRIPPER_SPEED,
    OBJECT_DEFAULTS,
    Object,
    TICK_DT_S,
)
from zitibot_core.context import make_context
from zitibot_core.runner import step_gate
from zitibot_tasks import grasp, pour

_DEFAULT_TILT_DEG = 45.0
_DEFAULT_TILT_AXIS = "x"
_DEFAULT_REORIENT_DURATION_S = 3.0


def parse_args() -> argparse.Namespace:
    spec_ladle = OBJECT_DEFAULTS[Object.LADLE]
    spec_bowl = OBJECT_DEFAULTS[Object.MIXING_BOWL]
    p = argparse.ArgumentParser(
        description="Grasp ladle, reorient for mixing, stir bowl (OpenSai Franka Redis)."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--step", action="store_true",
                   help="ENTER-gate each motion step inside subtasks")
    p.add_argument("--ladle-xyz", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"),
                   help=f"Ladle pick position in world frame (m). Default: {spec_ladle.pick_pose.tolist()}")
    p.add_argument("--bowl-xyz", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"),
                   help=f"Bowl centre in world frame (m). Default: {spec_bowl.pick_pose.tolist()}")
    p.add_argument("--approach-dz", type=float, default=DEFAULT_APPROACH_DZ_M)
    p.add_argument("--tilt-deg", type=float, default=_DEFAULT_TILT_DEG,
                   help=f"Ladle tilt angle for mixing (degrees, about --tilt-axis). Default: {_DEFAULT_TILT_DEG}")
    p.add_argument("--tilt-axis", default=_DEFAULT_TILT_AXIS, choices=("x", "y"),
                   help=f"World axis to tilt about. Default: {_DEFAULT_TILT_AXIS}")
    p.add_argument("--reorient-duration-s", type=float, default=_DEFAULT_REORIENT_DURATION_S,
                   help=f"SLERP duration for reorient/return steps (s). Default: {_DEFAULT_REORIENT_DURATION_S}")
    p.add_argument("--mix-radius", type=float, default=0.04)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--cycle-duration-s", type=float, default=4.0)
    p.add_argument("--gripper-open-width", type=float, default=None)
    p.add_argument("--gripper-close-width", type=float, default=0.0)
    p.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED)
    p.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    spec_ladle = OBJECT_DEFAULTS[Object.LADLE]
    spec_bowl = OBJECT_DEFAULTS[Object.MIXING_BOWL]

    if args.ladle_xyz is not None:
        spec_ladle.pick_pose = np.array(args.ladle_xyz, dtype=np.float64)
    if args.bowl_xyz is not None:
        spec_bowl.pick_pose = np.array(args.bowl_xyz, dtype=np.float64)

    spec_ladle.approach_dz = args.approach_dz
    spec_ladle.open_width = args.gripper_open_width
    spec_ladle.close_width = args.gripper_close_width
    spec_ladle.speed = args.gripper_speed
    spec_ladle.force = args.gripper_force

    bowl_pos = spec_bowl.pick_pose
    if bowl_pos is None:
        print("Error: MIXING_BOWL missing pick_pose in OBJECT_DEFAULTS", file=sys.stderr)
        return 1

    ctx = make_context(args, step=args.step)
    grip_R = spec_ladle.grasp_ori.copy()
    mix_center = bowl_pos + np.array([0.0, 0.0, 0.02])
    lower = mix_center - np.array([0.0, 0.0, 0.03])

    print(f"Ladle = {spec_ladle.pick_pose.tolist()}")
    print(f"Bowl  = {bowl_pos.tolist()}")
    print(f"Tilt  = {args.tilt_deg}° about world +{args.tilt_axis.upper()}")

    try:
        # Grasp ladle at default tool-down orientation
        grasp.object(ctx, Object.LADLE, pick_pos=spec_ladle.pick_pose, ori=grip_R)

        # Move above bowl center (still tool-down)
        arm.move_to(ctx, mix_center, grip_R,
                    label=f"[mix] move above bowl center {mix_center.tolist()}")

        # Reorient: SLERP from current orientation to mixing tilt, position held fixed.
        # pour.into() reads the current EE orientation from Redis, so whatever the arm
        # actually settled at above the bowl is the SLERP start point.
        R_tilted = pour.into(ctx, mix_center,
                             tilt_deg=args.tilt_deg,
                             axis=args.tilt_axis,
                             duration_s=args.reorient_duration_s)

        # Lower into bowl at the tilted orientation
        arm.move_to(ctx, lower, R_tilted,
                    label=f"[mix] lower into bowl to {lower.tolist()}")

        # Stir: circular motion in XY at fixed Z, holding tilted orientation
        step_gate(ctx, f"[mix] stir {args.cycles} cycle(s) r={args.mix_radius} m")
        t_cycle = args.cycle_duration_s / max(args.cycles, 1)
        for c in range(args.cycles):
            t0 = time.monotonic()
            while True:
                elapsed = time.monotonic() - t0
                if elapsed >= t_cycle:
                    break
                theta = 2.0 * math.pi * (elapsed / t_cycle)
                pos = lower + np.array([
                    args.mix_radius * math.cos(theta),
                    args.mix_radius * math.sin(theta),
                    0.0,
                ])
                arm.publish_goal_cartesian(ctx.redis, pos, R_tilted)
                if ctx.q_pressed():
                    raise KeyboardInterrupt("quit requested")
                time.sleep(TICK_DT_S)
            print(f"[mix] cycle {c + 1}/{args.cycles} complete")

        # Lift out of bowl, still tilted
        arm.move_to(ctx, mix_center, R_tilted,
                    label=f"[mix] lift out to {mix_center.tolist()}")

        # Return to grasp orientation before ending
        pour.return_upright(ctx, mix_center, R_tilted, grip_R,
                            duration_s=args.reorient_duration_s)

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print("Mixing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
