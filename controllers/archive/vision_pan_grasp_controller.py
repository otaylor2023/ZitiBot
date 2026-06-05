#!/usr/bin/env python3
"""Vision-guided pan grasp test (OpenSai Franka, modular system).

Asks Gemini for one point on the thick part of the pan handle near the rim,
then runs the standard grasp → lift → place sequence using the new
``zitibot_core`` / ``zitibot_tasks`` modules.

Useful for tuning the ``(Object.PAN, "grasp")`` Detection prompt and offset
without running the whole grasp+pour pipeline.

Usage::

  # ENTER-gate each step (recommended for first run)
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_pan_grasp_controller.py -- --step

  # Place back at the original detected grasp instead of OBJECT_DEFAULTS.place_pose
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_pan_grasp_controller.py -- \\
      --step --place-at-pick

  # Run with no ENTER gating
  ./ZitiBot/launch_zitibot_arm.sh controllers/vision_pan_grasp_controller.py

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemini vision → grasp pan handle (OpenSai Franka, modular system)."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate every move/grip step.",
    )
    p.add_argument(
        "--place-at-pick",
        action="store_true",
        help="Place the pan back at the detected pick location instead of "
        "OBJECT_DEFAULTS[Object.PAN].place_pose.",
    )
    p.add_argument(
        "--place-pos",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override place pose XYZ. Defaults to OBJECT_DEFAULTS[PAN].place_pose "
        "(or detected pick when --place-at-pick is set).",
    )
    p.add_argument("--approach-dz", type=float, default=DEFAULT_APPROACH_DZ_M)
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Gemini detection retries on failure.",
    )
    p.add_argument(
        "--orientation-source",
        choices=("fixed", "current"),
        default="fixed",
        help=(
            "Base grasp orientation: fixed=Object.PAN default; "
            "current=live EE orientation."
        ),
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


def _resolve_place_pos(args: argparse.Namespace, pick_pos: np.ndarray) -> np.ndarray:
    if args.place_pos is not None:
        return np.array(args.place_pos, dtype=np.float64)
    if args.place_at_pick:
        return np.asarray(pick_pos, dtype=np.float64).copy()
    place = OBJECT_DEFAULTS[Object.PAN].place_pose
    if place is None:
        raise RuntimeError(
            "Object.PAN has no place_pose in OBJECT_DEFAULTS; "
            "pass --place-pos X Y Z or --place-at-pick."
        )
    return np.asarray(place, dtype=np.float64).copy()


def main() -> int:
    args = parse_args()
    obj = Object.PAN
    spec = OBJECT_DEFAULTS[obj]
    spec.approach_dz = args.approach_dz

    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    print(f"Object   : {obj.value}")
    print(f"Step mode: {'on' if args.step else 'off'}")
    print(f"Gemini response: {args.gemini_response_path}")

    try:
        grasp_pose = gemini.find_grasp_pose(
            ctx,
            obj,
            retries=args.retries,
            orientation_source=args.orientation_source,
        )
        pick_pos = grasp_pose.position
        grip_R = grasp_pose.orientation
        print(f"Detected pan handle grasp: {pick_pos.tolist()}")
        if grasp_pose.rim_yaw_applied and grasp_pose.rim_yaw_deg is not None:
            print(f"Detected handle yaw: {grasp_pose.rim_yaw_deg:+.2f} deg")
        else:
            print("No yaw applied (single-point detection)")

        grasp.object(ctx, obj, pick_pos=pick_pos, ori=grip_R)

        place_pos = _resolve_place_pos(args, pick_pos)
        if not np.allclose(place_pos, pick_pos):
            arm.move_to(
                ctx,
                place_pos,
                grip_R,
                label=f"[transport] move to place {place_pos.tolist()}",
                tol_m=spec.approach_tol,
            )

        grasp.place(ctx, obj, place_pos=place_pos, ori=grip_R)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()

    print("Vision pan grasp complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
