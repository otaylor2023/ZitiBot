#!/usr/bin/env python3
"""Fixed-pose grasp-ladle + stir-bowl on OpenSai Franka (Redis only).

Thin CLI wrapper around ``zitibot_tasks.mix.in_bowl``.

Usage::

  python ZitiBot/controllers/mixing_simple_controller.py
  python ZitiBot/controllers/mixing_simple_controller.py --step
  python ZitiBot/controllers/mixing_simple_controller.py \\
      --bowl-xyz 0.17 0.62 0.50 --ladle-xyz 0.75 0.68 0.508
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core.constants import (
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_GRIPPER_FORCE,
    DEFAULT_GRIPPER_SPEED,
    OBJECT_DEFAULTS,
    Object,
)
from zitibot_core.context import make_context
from zitibot_tasks import mix


def parse_args() -> argparse.Namespace:
    spec_ladle = OBJECT_DEFAULTS[Object.LADLE]
    spec_bowl = OBJECT_DEFAULTS[Object.MIXING_BOWL]
    p = argparse.ArgumentParser(
        description="Grasp ladle, stir bowl, return ladle (OpenSai Franka Redis)."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate each motion step inside subtasks",
    )
    p.add_argument(
        "--ladle-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=f"Ladle pick position in world frame (m). Default: {spec_ladle.pick_pose.tolist()}",
    )
    p.add_argument(
        "--bowl-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=f"Bowl centre in world frame (m). Default: {spec_bowl.pick_pose.tolist()}",
    )
    p.add_argument("--approach-dz", type=float, default=DEFAULT_APPROACH_DZ_M)
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
    print(f"Ladle = {spec_ladle.pick_pose.tolist()}")
    print(f"Bowl  = {bowl_pos.tolist()}")

    try:
        mix.in_bowl(
            ctx,
            bowl_pos,
            radius_m=args.mix_radius,
            cycles=args.cycles,
            cycle_duration_s=args.cycle_duration_s,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    print("Mixing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
