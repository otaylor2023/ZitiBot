#!/usr/bin/env python3
"""Vision black-bowl pick -> pan-center pour -> return bowl to pick spot.

Flow (stationary):
  1. Arm -> home.
  2. Arm -> shared detection waypoint.
  3. Detect pan center.
  4. Detect black bowl left rim, pick + lift.
  5. Extra +Z lift.
  6. Move above detected pan center and pour.
  7. Place bowl back at exact pick pose.

Unlike egg_crack, there is NO throw-away/drop step after pouring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm, gains
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    HOME_POS_TOL_M,
    Object,
    PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
    PRECISE_GRASP_MAX_LINEAR_VELOCITY,
    PRECISE_GRASP_ORIENTATION_KP,
    PRECISE_GRASP_POSITION_KP,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_tasks import gemini, grasp, pour

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_BOWL_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_egg_pour_bowl.png"
)
DEFAULT_GEMINI_PAN_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_egg_pour_pan.png"
)

# Transit carry lift after grasping the bowl.
BOWL_CARRY_LIFT_M = 0.20
# Extra vertical clearance after grasping, before moving toward pan center.
POST_GRASP_EXTRA_LIFT_M = 0.10
# Pour target = detected pan center + this world offset.
# +Y shifts left in this world frame.
PAN_CENTER_POUR_OFFSET_M = np.array([0.0, 0.07, 0.16], dtype=np.float64)
# During the pour tilt, rise by this much in +Z so the pour ends higher.
POUR_UP_DZ_M = 0.10
# Keep the same pivot convention as bowl_pour_controller.
POUR_PIVOT_BELOW_EE_M = 0.127
# Shared camera framing pose for BOTH pan and bowl detections.
# Captured from terminal startup print (2026-06-08):
#   [arm] startup EE world position: [+0.3230, -0.1954, +0.6936] m
DETECTION_EE_POSITION = np.array([0.3230, -0.1954, 0.6936], dtype=np.float64)
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pick black bowl at left rim, pour into detected pan center, "
            "then place bowl back at pick location."
        )
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--step", action="store_true", help="ENTER-gate each action.")
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Gemini retries for both pan and bowl detections.",
    )
    p.add_argument(
        "--detection-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Shared EE world position for BOTH pan-center and black-bowl "
            "Gemini detections. Default: [0.3230, -0.1954, 0.6936]."
        ),
    )
    p.add_argument(
        "--pour-tilt-deg",
        type=float,
        default=DEFAULT_POUR_TILT_DEG,
        help="Pour tilt angle in degrees.",
    )
    p.add_argument(
        "--pour-axis",
        choices=("x", "y"),
        default=DEFAULT_POUR_AXIS,
        help="World axis for pour tilt.",
    )
    p.add_argument(
        "--tilt-duration-s",
        type=float,
        default=DEFAULT_TILT_DURATION_S,
        help="Duration of pour tilt and return-upright slerps.",
    )
    p.add_argument(
        "--pan-pour-offset-x",
        type=float,
        default=float(PAN_CENTER_POUR_OFFSET_M[0]),
        help="World X offset from detected pan center for pour target (m).",
    )
    p.add_argument(
        "--pan-pour-offset-y",
        type=float,
        default=float(PAN_CENTER_POUR_OFFSET_M[1]),
        help="World Y offset from detected pan center for pour target (m).",
    )
    p.add_argument(
        "--pan-pour-offset-z",
        type=float,
        default=float(PAN_CENTER_POUR_OFFSET_M[2]),
        help="World Z offset from detected pan center for pour target (m).",
    )
    p.add_argument(
        "--post-grasp-extra-lift-m",
        type=float,
        default=POST_GRASP_EXTRA_LIFT_M,
        help=(
            "Extra +Z lift after grasp/carry-lift, before moving toward pan center (m)."
        ),
    )
    p.add_argument(
        "--pour-up-dz-m",
        type=float,
        default=POUR_UP_DZ_M,
        help=(
            "Extra +Z rise during pour tilt (m). The tilt target is "
            "pan_center + pan_pour_offset + [0,0,pour_up_dz_m]."
        ),
    )
    p.add_argument(
        "--bowl-gemini-response-path",
        type=Path,
        default=DEFAULT_GEMINI_BOWL_RESPONSE_PATH,
        help="Path to save Gemini debug image for black bowl grasp detection.",
    )
    p.add_argument(
        "--pan-gemini-response-path",
        type=Path,
        default=DEFAULT_GEMINI_PAN_RESPONSE_PATH,
        help="Path to save Gemini debug image for pan-center detection.",
    )
    p.add_argument(
        "--endeffector-transform-key",
        default=DEFAULT_ENDEFFECTOR_TRANSFORM_KEY,
        help="Redis key for the 4x4 base->flange transform.",
    )
    p.add_argument(
        "--return-home",
        action="store_true",
        help="Return arm to home after replacing the bowl.",
    )
    return p.parse_args()


def _detect_pan_center(
    ctx: TaskContext, *, retries: int, gemini_response_path: str | Path | None
) -> np.ndarray:
    """Detect pan center (and rim for robustness), return center in world."""
    prior_path = ctx.gemini_response_path
    try:
        if gemini_response_path is not None:
            ctx.gemini_response_path = str(gemini_response_path)
        center, radius_m = gemini.find_pan_center_radius(
            ctx, Object.PAN, kind="center_rim", retries=retries
        )
    finally:
        ctx.gemini_response_path = prior_path
    if radius_m is not None:
        print(
            f"[egg_pour] detected pan center={center.tolist()} radius={radius_m:.3f} m"
        )
    else:
        print(f"[egg_pour] detected pan center={center.tolist()} (radius unavailable)")
    return center


def run_egg_pour_cycle(
    ctx: TaskContext,
    *,
    retries: int,
    detection_pos: np.ndarray,
    pour_tilt_deg: float,
    pour_axis: str,
    tilt_duration_s: float,
    pan_pour_offset_m: np.ndarray,
    post_grasp_extra_lift_m: float,
    pour_up_dz_m: float,
    bowl_gemini_response_path: str | Path | None,
    pan_gemini_response_path: str | Path | None,
    return_home: bool,
) -> None:
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[egg_pour] move to home {ARM_HOME_POSITION.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )

    arm.move_to(
        ctx,
        detection_pos,
        ARM_HOME_ORIENTATION,
        label=f"[egg_pour] move to shared detection pose {detection_pos.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )

    # Detect pan center before grasping so the held bowl does not occlude view.
    pan_center = _detect_pan_center(
        ctx, retries=retries, gemini_response_path=pan_gemini_response_path
    )

    # Detect and grasp bowl.
    ctx.gemini_response_path = str(bowl_gemini_response_path)
    bowl_pose = gemini.find_grasp_pose(ctx, Object.PASTA_BOWL, retries=retries)
    pick_pos = bowl_pose.position.astype(np.float64, copy=True)
    grip_R = bowl_pose.orientation
    print(f"[egg_pour] black bowl grasp pose: {pick_pos.tolist()}")
    grasp.object(
        ctx,
        Object.PASTA_BOWL,
        pick_pos=pick_pos,
        ori=grip_R,
        lift_dz_m=BOWL_CARRY_LIFT_M,
    )
    cur_pose = arm.read_current_ee_world(ctx.redis)
    if cur_pose is not None:
        extra_lift_pos = np.asarray(cur_pose[0], dtype=np.float64).reshape(3) + np.array(
            [0.0, 0.0, float(post_grasp_extra_lift_m)], dtype=np.float64
        )
        arm.move_to(
            ctx,
            extra_lift_pos,
            grip_R,
            label=(
                f"[egg_pour] extra post-grasp lift "
                f"+{post_grasp_extra_lift_m * 100:.0f} cm {extra_lift_pos.tolist()}"
            ),
            tol_m=HOME_POS_TOL_M,
        )
    else:
        print(
            "[egg_pour] WARNING: EE pose unavailable after grasp; "
            "skipping extra post-grasp lift."
        )

    # Pour above the detected pan center in precise mode for tighter tracking.
    # Start from pour_start_pos and finish the tilt at pour_up_pos (higher Z).
    pour_start_pos = pan_center + pan_pour_offset_m
    pour_up_pos = pour_start_pos + np.array([0.0, 0.0, float(pour_up_dz_m)], dtype=np.float64)
    pivot_offset_local = np.array([0.0, 0.0, POUR_PIVOT_BELOW_EE_M], dtype=np.float64)
    pour_precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="egg_pour",
    )
    try:
        arm.move_to(
            ctx,
            pour_start_pos,
            grip_R,
            label=f"[egg_pour] move above pan center pour start {pour_start_pos.tolist()}",
        )
        R_poured = pour.into(
            ctx,
            pour_up_pos,
            tilt_deg=pour_tilt_deg,
            axis=pour_axis,
            duration_s=tilt_duration_s,
            pivot_offset_local=pivot_offset_local,
        )
        pour.return_upright(
            ctx,
            pour_up_pos,
            R_poured,
            grip_R,
            duration_s=tilt_duration_s,
            pivot_offset_local=pivot_offset_local,
        )
    finally:
        gains.restore_precise_grasp(ctx.redis, pour_precise, label="egg_pour")

    # Place back at the original pick location.
    grasp.place(ctx, Object.PASTA_BOWL, place_pos=pick_pos, ori=grip_R)
    print("[egg_pour] bowl replaced at original pick pose.")

    if return_home:
        arm.move_to(
            ctx,
            ARM_HOME_POSITION,
            ARM_HOME_ORIENTATION,
            label=f"[egg_pour] return to home {ARM_HOME_POSITION.tolist()}",
            tol_m=HOME_POS_TOL_M,
        )


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    pan_pour_offset = np.array(
        [args.pan_pour_offset_x, args.pan_pour_offset_y, args.pan_pour_offset_z],
        dtype=np.float64,
    )
    detection_pos = (
        DETECTION_EE_POSITION.copy()
        if args.detection_xyz is None
        else np.array(args.detection_xyz, dtype=np.float64)
    )

    print(f"Step mode     : {'on' if args.step else 'off'}")
    print("Base motion   : disabled (stationary mode)")
    print(f"Detect pose   : {detection_pos.tolist()}")
    print(f"Pan log       : {args.pan_gemini_response_path}")
    print(f"Bowl log      : {args.bowl_gemini_response_path}")
    print(f"Pan pour offs : {pan_pour_offset.tolist()}")
    print(f"Post-grasp dz : {args.post_grasp_extra_lift_m}")
    print(f"Pour up dz    : {args.pour_up_dz_m}")

    try:
        run_egg_pour_cycle(
            ctx,
            retries=args.retries,
            detection_pos=detection_pos,
            pour_tilt_deg=args.pour_tilt_deg,
            pour_axis=args.pour_axis,
            tilt_duration_s=args.tilt_duration_s,
            pan_pour_offset_m=pan_pour_offset,
            post_grasp_extra_lift_m=args.post_grasp_extra_lift_m,
            pour_up_dz_m=args.pour_up_dz_m,
            bowl_gemini_response_path=args.bowl_gemini_response_path,
            pan_gemini_response_path=args.pan_gemini_response_path,
            return_home=args.return_home,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
