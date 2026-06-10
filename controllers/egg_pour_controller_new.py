#!/usr/bin/env python3
"""Vision black-bowl pick -> pan-center pour -> return bowl to pick spot.

Variant of ``egg_pour_controller_new`` that adds a simultaneous world −Y
translation during the tilt move: the EE moves in the negative-Y direction at
the same time as it tilts and rises. The return-upright move goes back to the
original ``pour_start_pos``, so the Y motion reverses on the way back.

Flow:
  1. Base -> STOVE_STATION.
  2. Arm -> home.
  3. Arm -> shared detection waypoint.
  4. Detect pan center.
  5. Detect black bowl left rim, pick + lift.
  6. Move above detected pan center.
  7. Tilt (arm.move_to tilted_ori + rise + neg-Y shift) in precise mode.
  8. Return upright (arm.move_to straight_down_R, back to pour_start_pos) in precise mode.
  9. Place bowl back at exact pick pose.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm, base, gains
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    HOME_POS_TOL_M,
    OBJECT_DEFAULTS,
    Object,
    PRECISE_GRASP_MAX_LINEAR_VELOCITY,
    PRECISE_GRASP_ORIENTATION_KP,
    PRECISE_GRASP_POSITION_KP,
    BaseWaypoint,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_tasks import gemini, grasp

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_BOWL_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_egg_pour_new_bowl.png"
)
DEFAULT_GEMINI_PAN_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_egg_pour_new_pan.png"
)

BOWL_CARRY_LIFT_M = 0.10
# Pre-grasp hover height above the bowl when picking it up.
BOWL_PICK_APPROACH_DZ_M = 0.10
# Hover height above the place spot before descending to set the bowl back down.
# Higher than the pick approach so the bowl comes in ~10 cm higher before the
# final descent.
BOWL_PLACE_APPROACH_DZ_M = 0.20
PAN_CENTER_POUR_OFFSET_M = np.array([0.0, 0.13, 0.20], dtype=np.float64)
POUR_TILT_RISE_M = 0.10
POUR_TILT_TOL_M = 0.08
# Angular velocity cap during tilt/return (rad/s).
POUR_TILT_MAX_ANGULAR_VEL_RAD_S = math.pi / 12  # ≈ 0.26 rad/s
# How far to shift in world −Y during the tilt move (m).
POUR_TILT_NEG_Y_M = 0.08
# Bowl/pan detection pose: arm home shifted +10 cm up (world +Z) and +10 cm
# left (world +Y, the robot's left while facing +X).
DETECTION_EE_POSITION = ARM_HOME_POSITION + np.array(
    [0.0, -0.30, 0.10], dtype=np.float64
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pick black bowl, pour into pan with simultaneous −Y EE shift during tilt, "
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
            "Gemini detections. Default: DETECTION_EE_POSITION."
        ),
    )
    p.add_argument(
        "--pour-tilt-deg",
        type=float,
        default=DEFAULT_POUR_TILT_DEG,
        help="Pour tilt angle about world +X (degrees).",
    )
    p.add_argument(
        "--pour-axis",
        choices=("x", "y"),
        default=DEFAULT_POUR_AXIS,
        help="World axis for pour tilt.",
    )
    p.add_argument(
        "--pour-tilt-rise-m",
        type=float,
        default=POUR_TILT_RISE_M,
        help="Extra +Z rise applied simultaneously with the tilt (m).",
    )
    p.add_argument(
        "--tilt-neg-y-m",
        type=float,
        default=POUR_TILT_NEG_Y_M,
        help=(
            "World −Y shift applied simultaneously with the tilt (m). "
            f"Default: {POUR_TILT_NEG_Y_M:.3f} m. "
            "The return-upright move reverses this shift."
        ),
    )
    p.add_argument(
        "--tilt-angular-vel-rad-s",
        type=float,
        default=POUR_TILT_MAX_ANGULAR_VEL_RAD_S,
        help=(
            "Max angular velocity during tilt + return-upright moves (rad/s). "
            f"Default: π/12 ≈ {POUR_TILT_MAX_ANGULAR_VEL_RAD_S:.3f}. "
            "Lower = slower pour rotation."
        ),
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
            f"[egg_pour_new] detected pan center={center.tolist()} radius={radius_m:.3f} m"
        )
    else:
        print(f"[egg_pour_new] detected pan center={center.tolist()} (radius unavailable)")
    return center


def _build_straight_down_R() -> np.ndarray:
    """Clean 'tool pointing straight down' orientation (mirrors egg_crack_controller)."""
    tool_x = ARM_HOME_ORIENTATION[:, 0].astype(np.float64).copy()
    tool_x[2] = 0.0
    nx = float(np.linalg.norm(tool_x))
    tool_x = tool_x / nx if nx > 1e-9 else np.array([1.0, 0.0, 0.0])
    tool_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)
    return np.column_stack([tool_x, tool_y, tool_z])


def run_egg_pour_new_cycle(
    ctx: TaskContext,
    *,
    retries: int,
    detection_pos: np.ndarray,
    pour_tilt_deg: float,
    pour_axis: str,
    pan_pour_offset_m: np.ndarray,
    pour_tilt_rise_m: float,
    tilt_neg_y_m: float,
    tilt_angular_vel_rad_s: float,
    bowl_gemini_response_path: str | Path | None,
    pan_gemini_response_path: str | Path | None,
    return_home: bool,
) -> None:
    OBJECT_DEFAULTS[Object.PASTA_BOWL].approach_dz = BOWL_PICK_APPROACH_DZ_M

    base.go_to_pose(ctx, BaseWaypoint.STOVE_STATION)

    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[egg_pour_new] move to home {ARM_HOME_POSITION.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )

    arm.move_to(
        ctx,
        detection_pos,
        ARM_HOME_ORIENTATION,
        label=f"[egg_pour_new] move to shared detection pose {detection_pos.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )

    pan_center = _detect_pan_center(
        ctx, retries=retries, gemini_response_path=pan_gemini_response_path
    )

    ctx.gemini_response_path = str(bowl_gemini_response_path)
    bowl_pose = gemini.find_grasp_pose(ctx, Object.PASTA_BOWL, retries=retries)
    pick_pos = bowl_pose.position.astype(np.float64, copy=True)
    grip_R = bowl_pose.orientation
    print(f"[egg_pour_new] black bowl grasp pose: {pick_pos.tolist()}")
    grasp.object(
        ctx,
        Object.PASTA_BOWL,
        pick_pos=pick_pos,
        ori=grip_R,
        lift_dz_m=BOWL_CARRY_LIFT_M,
    )

    pour_start_pos = pan_center + pan_pour_offset_m
    arm.move_to(
        ctx,
        pour_start_pos,
        grip_R,
        label=f"[egg_pour_new] move above pan center {pour_start_pos.tolist()}",
    )

    straight_down_R = _build_straight_down_R()
    tilted_ori = arm.pour_orientation_end(straight_down_R, pour_tilt_deg, axis=pour_axis)
    # Tilt target: rise in Z and shift in −Y simultaneously with the rotation.
    tilt_pos = pour_start_pos + np.array(
        [0.0, -float(tilt_neg_y_m), float(pour_tilt_rise_m)], dtype=np.float64
    )

    precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=tilt_angular_vel_rad_s,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="egg_pour_new tilt",
    )
    try:
        arm.move_to(
            ctx,
            tilt_pos,
            tilted_ori,
            label=(
                f"[egg_pour_new] tilt {pour_tilt_deg:.0f}° about world "
                f"+{pour_axis.upper()} + rise {pour_tilt_rise_m * 100:.0f} cm "
                f"+ −Y {tilt_neg_y_m * 100:.0f} cm → {tilt_pos.tolist()}"
            ),
            tol_m=POUR_TILT_TOL_M,
            timeout_s=8.0,
        )
        arm.move_to(
            ctx,
            pour_start_pos,
            straight_down_R,
            label=f"[egg_pour_new] return upright to {pour_start_pos.tolist()}",
            tol_m=POUR_TILT_TOL_M,
            timeout_s=8.0,
        )
    finally:
        gains.restore_precise_grasp(ctx.redis, precise, label="egg_pour_new tilt")

    # Come in higher above the place spot than the pick approach before descending.
    OBJECT_DEFAULTS[Object.PASTA_BOWL].approach_dz = BOWL_PLACE_APPROACH_DZ_M
    grasp.place(ctx, Object.PASTA_BOWL, place_pos=pick_pos, ori=grip_R)
    print("[egg_pour_new] bowl replaced at original pick pose.")

    if return_home:
        arm.move_to(
            ctx,
            ARM_HOME_POSITION,
            ARM_HOME_ORIENTATION,
            label=f"[egg_pour_new] return to home {ARM_HOME_POSITION.tolist()}",
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
    print("Base motion   : STOVE_STATION at start")
    print(f"Detect pose   : {detection_pos.tolist()}")
    print(f"Pan log       : {args.pan_gemini_response_path}")
    print(f"Bowl log      : {args.bowl_gemini_response_path}")
    print(f"Pan pour offs : {pan_pour_offset.tolist()}")
    print(f"Tilt          : {args.pour_tilt_deg:.0f}° about world +{args.pour_axis.upper()}")
    print(f"Tilt rise     : {args.pour_tilt_rise_m * 100:.0f} cm")
    print(f"Tilt −Y shift : {args.tilt_neg_y_m * 100:.0f} cm")
    print(f"Tilt ang vel  : {args.tilt_angular_vel_rad_s:.3f} rad/s")

    try:
        run_egg_pour_new_cycle(
            ctx,
            retries=args.retries,
            detection_pos=detection_pos,
            pour_tilt_deg=args.pour_tilt_deg,
            pour_axis=args.pour_axis,
            pan_pour_offset_m=pan_pour_offset,
            pour_tilt_rise_m=args.pour_tilt_rise_m,
            tilt_neg_y_m=args.tilt_neg_y_m,
            tilt_angular_vel_rad_s=args.tilt_angular_vel_rad_s,
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
