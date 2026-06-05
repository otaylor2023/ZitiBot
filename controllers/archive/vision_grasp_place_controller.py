tidy
#!/usr/bin/env python3
"""Vision-guided grasp + place using the modular zitibot_tasks helpers.

This is the pick/place sibling of ``vision_grasp_pour_controller_v2.py``:

  ``gemini.find_grasp_pose``  ->  ``grasp.object``  ->  ``arm.move_to``  ->
  ``grasp.place``

Usage::

  # ENTER-gate each step while testing a new object/pose
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_grasp_place_controller.py -- --step

  # Pick a different object and place target
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_grasp_place_controller.py -- \\
      --object pan --place-pose 0.77 -0.08 0.33 --step

  # Use OBJECT_DEFAULTS place pose, no ENTER gating
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_grasp_place_controller.py

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
RealSense, and ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
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
from zitibot_core.constants import DEFAULT_APPROACH_DZ_M, OBJECT_DEFAULTS, Object
from zitibot_core.context import make_context
from zitibot_tasks import gemini, grasp
from vision import gemini_pointing as gp

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_RESPONSE_PATH = _CONTROLLERS.parent / "logs" / "gemini_response.png"


def _object_choices() -> list[str]:
    return sorted(obj.value for obj in Object)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemini vision -> grasp -> place (OpenSai Franka, modular system)."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate every move/grip step.",
    )
    p.add_argument(
        "--object",
        choices=_object_choices(),
        default=Object.PAN.value,
        help="Object enum to detect + grasp (see zitibot_core.constants.Object).",
    )
    p.add_argument(
        "--place-pose",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Place pose XYZ in world frame (m). Defaults to ObjectSpec.place_pose.",
    )
    p.add_argument("--approach-dz", type=float, default=DEFAULT_APPROACH_DZ_M)
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Gemini detection retries on failure.",
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
    p.add_argument(
        "--orientation-source",
        choices=("fixed", "current"),
        default="fixed",
        help=(
            "Base grasp orientation before rim yaw: fixed=object default; "
            "current=live EE orientation."
        ),
    )
    p.add_argument(
        "--model",
        default=gp.DEFAULT_MODEL,
        help=f"Gemini model name (default: {gp.DEFAULT_MODEL}).",
    )
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--timeout-ms", type=int, default=10000)
    p.add_argument("--depth-patch-radius", type=int, default=2)
    return p.parse_args()


def _resolve_place_pose(args: argparse.Namespace, obj: Object) -> np.ndarray | None:
    if args.place_pose is not None:
        return np.array(args.place_pose, dtype=np.float64)
    spec = OBJECT_DEFAULTS[obj]
    if spec.place_pose is None:
        return None
    return np.asarray(spec.place_pose, dtype=np.float64).copy()


def main() -> int:
    args = parse_args()
    obj = Object(args.object)
    spec = OBJECT_DEFAULTS[obj]
    spec.approach_dz = args.approach_dz

    place_pos = _resolve_place_pose(args, obj)
    if place_pos is None:
        print(
            f"Error: no place pose for {obj.value}. "
            f"Pass --place-pose X Y Z or set OBJECT_DEFAULTS[Object.{obj.name}].place_pose.",
            file=sys.stderr,
        )
        return 1

    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    print(f"Object    : {obj.value}")
    print(f"Place pose: {place_pos.tolist()}")
    print(f"Step mode : {'on' if args.step else 'off'}")
    print(f"Gemini response: {args.gemini_response_path}")

    try:
        # 1. Vision: ask Gemini for object points, then compute grasp pose.
        grasp_pose = gemini.find_grasp_pose(
            ctx,
            obj,
            retries=args.retries,
            orientation_source=args.orientation_source,
        )
        grasp_pos = grasp_pose.position
        grip_R = grasp_pose.orientation
        print(f"Detected grasp: {grasp_pos.tolist()}")
        if grasp_pose.rim_yaw_applied and grasp_pose.rim_yaw_deg is not None:
            print(f"Detected rim yaw: {grasp_pose.rim_yaw_deg:+.2f} deg")
        else:
            print("Detected rim yaw: not applied")

        # 2. Pick.
        grasp.object(ctx, obj, pick_pos=grasp_pos, ori=grip_R)

        # 3. Transport above the place pose.
        above_place = place_pos.copy()
        above_place[2] += spec.approach_dz
        arm.move_to(
            ctx,
            above_place,
            grip_R,
            label=f"[transport] move above place {above_place.tolist()}",
            tol_m=spec.approach_tol,
        )

        # 4. Place at the requested/default target.
        grasp.place(ctx, obj, place_pos=place_pos, ori=grip_R)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()

    print("Vision grasp+place complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
