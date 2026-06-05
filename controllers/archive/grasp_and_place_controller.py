#!/usr/bin/env python3
"""Fixed-pose grasp + transport + place on OpenSai Franka (Redis only).

Thin CLI wrapper around ``zitibot_tasks.grasp.pick_and_place``.

Usage::

  python ZitiBot/controllers/grasp_and_place_controller.py
  python ZitiBot/controllers/grasp_and_place_controller.py \\
      --pick 0.4 -0.2 0.35 --place 0.55 0.15 0.35 --lift-dz 0.12
  python ZitiBot/controllers/grasp_and_place_controller.py --step
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core.constants import DEFAULT_APPROACH_DZ_M, OBJECT_DEFAULTS, Object
from zitibot_core.context import make_context
from zitibot_tasks import grasp


def parse_args() -> argparse.Namespace:
    spec = OBJECT_DEFAULTS[Object.PAN]
    p = argparse.ArgumentParser(
        description="Grasp at pick pose, place at place pose (ENTER-gated with --step)."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate each motion step inside the subtask",
    )
    p.add_argument(
        "--object",
        choices=[o.value for o in Object],
        default=Object.PAN.value,
        help="Object preset from OBJECT_DEFAULTS",
    )
    p.add_argument(
        "--pick",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Pick position in world frame (m).",
    )
    p.add_argument(
        "--place",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Place position in world frame (m).",
    )
    p.add_argument("--lift-dz", type=float, default=DEFAULT_APPROACH_DZ_M)
    p.add_argument("--gripper-open-width", type=float, default=None)
    p.add_argument("--gripper-close-width", type=float, default=0.0)
    p.add_argument("--gripper-speed", type=float, default=spec.speed)
    p.add_argument("--gripper-force", type=float, default=spec.force)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    obj = Object(args.object)
    spec = OBJECT_DEFAULTS[obj]
    pick = (
        np.array(args.pick, dtype=np.float64)
        if args.pick is not None
        else spec.pick_pose
    )
    place = (
        np.array(args.place, dtype=np.float64)
        if args.place is not None
        else spec.place_pose
    )
    if pick is None or place is None:
        print("Error: --pick and --place required for this object.", file=sys.stderr)
        return 1

    spec.approach_dz = args.lift_dz
    spec.open_width = args.gripper_open_width
    spec.close_width = args.gripper_close_width
    spec.speed = args.gripper_speed
    spec.force = args.gripper_force

    ctx = make_context(args, step=args.step)
    print(f"Object: {obj.value}")
    print(f"Pick  = {pick.tolist()}")
    print(f"Place = {place.tolist()}")
    try:
        grasp.pick_and_place(ctx, obj, pick_pos=pick, place_pos=place)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
