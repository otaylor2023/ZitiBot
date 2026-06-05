#!/usr/bin/env python3
"""Vision-guided grasp + pour using the new zitibot_core/zitibot_tasks modules.

A slim rewrite of ``vision_grasp_pour_controller.py`` built on the modular
state-machine primitives:

  ``gemini.find_grasp_pose``  →  ``grasp.object``  →  ``arm.move_to``  →
  ``pour.into``  →  ``pour.return_upright``  →  ``grasp.place``

The old controller is preserved unchanged for the OpenCV UI.

Usage::

  # ENTER-gate each step (recommended for first run on a new object/pose)
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_grasp_pour_controller_v2.py -- --step

  # Pick a different object and pour target
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_grasp_pour_controller_v2.py -- \\
      --object pasta_bowl --pour-pose 0.52 0.12 0.63 --step

  # Use OBJECT_DEFAULTS pour pose, no ENTER gating
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_grasp_pour_controller_v2.py

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
from zitibot_core.constants import (
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    OBJECT_DEFAULTS,
    Object,
)
from zitibot_core.context import make_context
from zitibot_tasks import gemini, grasp, pour
from vision import gemini_pointing as gp

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_RESPONSE_PATH = _CONTROLLERS.parent / "logs" / "gemini_response.png"


def _object_choices() -> list[str]:
    return sorted(obj.value for obj in Object)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemini vision → grasp → pour (OpenSai Franka, modular system)."
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
        default=Object.PASTA_BOWL.value,
        help="Object enum to detect + grasp (see zitibot_core.constants.Object).",
    )
    p.add_argument(
        "--pour-pose",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Pour pose XYZ in world frame (m). Defaults to ObjectSpec.pour_pose.",
    )
    p.add_argument("--approach-dz", type=float, default=DEFAULT_APPROACH_DZ_M)
    p.add_argument("--pour-tilt-deg", type=float, default=DEFAULT_POUR_TILT_DEG)
    p.add_argument("--pour-axis", default=DEFAULT_POUR_AXIS, choices=("x", "y"))
    p.add_argument("--tilt-duration-s", type=float, default=DEFAULT_TILT_DURATION_S)
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Gemini detection retries on failure.",
    )
    p.add_argument(
        "--endeffector-transform-key",
        default=DEFAULT_ENDEFFECTOR_TRANSFORM_KEY,
        help="Redis key for the 4x4 base→flange transform.",
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


def _resolve_pour_pose(args: argparse.Namespace, obj: Object) -> np.ndarray | None:
    if args.pour_pose is not None:
        return np.array(args.pour_pose, dtype=np.float64)
    spec = OBJECT_DEFAULTS[obj]
    if spec.pour_pose is None:
        return None
    return np.asarray(spec.pour_pose, dtype=np.float64).copy()


def main() -> int:
    args = parse_args()
    obj = Object(args.object)
    spec = OBJECT_DEFAULTS[obj]
    spec.approach_dz = args.approach_dz

    pour_pos = _resolve_pour_pose(args, obj)
    if pour_pos is None:
        print(
            f"Error: no pour pose for {obj.value}. "
            "Pass --pour-pose X Y Z or set OBJECT_DEFAULTS[Object.{obj.name}].pour_pose.",
            file=sys.stderr,
        )
        return 1

    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    print(f"Object   : {obj.value}")
    print(f"Pour pose: {pour_pos.tolist()}")
    print(f"Step mode: {'on' if args.step else 'off'}")
    print(f"Gemini response: {args.gemini_response_path}")

    try:
        # 1. Vision: ask Gemini for two rim points, then compute grasp pose.
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

        # 3. Transport to pour pose.
        arm.move_to(
            ctx,
            pour_pos,
            grip_R,
            label=f"[transport] move to pour {pour_pos.tolist()}",
            tol_m=spec.approach_tol,
        )

        # 4. Pour: tilt → return upright.
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

        # 5. Retract from pour site, then back over the original grasp.
        retract = pour_pos.copy()
        retract[2] = max(float(pour_pos[2]), float(grasp_pos[2] + spec.approach_dz))
        arm.move_to(
            ctx,
            retract,
            grip_R,
            label=f"[return] retract at pour site {retract.tolist()}",
            tol_m=spec.approach_tol,
        )
        arm.move_to(
            ctx,
            grasp_pos,
            grip_R,
            label=f"[return] move back to grasp {grasp_pos.tolist()}",
            tol_m=spec.grasp_tol,
        )

        # 6. Place back at the original grasp position.
        grasp.place(ctx, obj, place_pos=grasp_pos, ori=grip_R)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()

    print("Vision grasp+pour complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
