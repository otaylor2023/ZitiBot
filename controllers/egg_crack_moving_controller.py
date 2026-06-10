#!/usr/bin/env python3
"""Multi-station egg-cracker flow built on ``egg_crack_stationary_controller``.

End-to-end sequence:

  1. Arm → home pose.
  2. Base → ``EGG_CRACK_STATION``.
  3. Coarse Gemini grasp + optional refine + LK visual servo (same pipeline
     as the stationary controller), then ``grasp.object`` picks up the
     cracker and lifts.
  4. Arm → transit carry pose (``ARM_HOME + +Z * carry_lift_m``).
  5. Base → ``STIRRING_STATION`` (holds the carry pose throughout).
  6. Arm → bowl-detection camera pose; Gemini locates the **black pasta
     bowl** interior center.
  7. Move the held cracker above the black bowl, ``egg_crack.crack``,
     return to above.
  8. Gemini locates the **white plastic bowl** interior center; move
     above it, rotate about world +X to dump the shell, unrotate.
  9. Arm → transit carry pose.
 10. Base → ``EGG_CRACK_STATION``.
 11. Return to the pick spot, release the cracker, lift away.

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/egg_crack_moving_controller.py -- --step

  # Arm-only debug when the cart is already at the egg-crack station:
  ./ZitiBot/launch_zitibot_full.sh controllers/egg_crack_moving_controller.py -- \\
      --skip-base --step

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
TidyBot base ``redis_driver``, RealSense, OptiTrack on Redis, and
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from egg_crack_stationary_controller import (
    ABOVE_STANDOFF_M,
    DEFAULT_ENDEFFECTOR_TRANSFORM_KEY,
    DEFAULT_GEMINI_REFINE_RESPONSE_PATH,
    DEFAULT_GEMINI_RESPONSE_PATH,
    DEFAULT_SERVO_LOG_DIR,
    EGG_CRACKER_CARRY_LIFT_M,
    FINAL_UP_M,
    RELEASE_DROP_M,
    RELEASE_SETTLE_S,
    ROTATE_DEG,
    ROTATE_LIFT_M,
    SERVO_GAIN,
    SERVO_KI,
    SERVO_LATERAL_OFFSET_X,
    SERVO_LATERAL_OFFSET_Y,
    SERVO_MAX_ITERS,
    SERVO_MIN_BOX_H,
    SERVO_MIN_BOX_W,
    SERVO_PX_TOL,
    SERVO_SETTLE_S,
    SERVO_STEP_CLIP_M,
    SERVO_TARGET_U,
    SERVO_TARGET_V,
    _above_pose,
    _expand_box,
    _rotated_about_world_x,
)
from zitibot_core import arm, base, gripper
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    DEFAULT_POS_TOL_M,
    EGG_CRACKER_DETECTION_EE_ORIENTATION,
    EGG_CRACKER_DETECTION_EE_POSITION,
    HOME_POS_TOL_M,
    OBJECT_DEFAULTS,
    BaseWaypoint,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_tasks import egg_crack, gemini, grasp, visual_servo

DEFAULT_GEMINI_BLACK_BOWL_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_black_bowl_center.png"
)
DEFAULT_GEMINI_WHITE_BOWL_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_white_bowl_center.png"
)

# Camera-framing pose at STIRRING_STATION for the two-bowl layout. Same
# convention as the cracker detection pose: home XY, dropped below home in -Z.
STIRRING_BOWL_DETECTION_EE_POSITION = ARM_HOME_POSITION + np.array(
    [0.0, 0.0, -0.10], dtype=np.float64
)

# EE offset from a vision-detected bowl interior center to the cracker
# squeeze pose. Tune at the bench if the egg misses the bowl center.
CRACK_EE_OFFSET_FROM_BOWL_CENTER = np.array([0.0, 0.0, 0.12], dtype=np.float64)
SHELL_EE_OFFSET_FROM_BOWL_CENTER = np.array([0.0, 0.0, 0.12], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Egg-cracker: grasp at EGG_CRACK_STATION, crack at STIRRING_STATION, "
            "return to release."
        )
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate every motion / gripper / base step.",
    )
    p.add_argument(
        "--skip-base",
        action="store_true",
        help=(
            "Do not drive the base (arm-only debug when the cart is already "
            "parked at EGG_CRACK_STATION)."
        ),
    )

    # Cracker detection pose at EGG_CRACK_STATION.
    p.add_argument(
        "--detection-x",
        type=float,
        default=float(EGG_CRACKER_DETECTION_EE_POSITION[0]),
        help="Camera-framing EE X for cracker Gemini detection.",
    )
    p.add_argument(
        "--detection-y",
        type=float,
        default=float(EGG_CRACKER_DETECTION_EE_POSITION[1]),
        help="Camera-framing EE Y for cracker Gemini detection.",
    )
    p.add_argument(
        "--detection-z",
        type=float,
        default=float(EGG_CRACKER_DETECTION_EE_POSITION[2]),
        help="Camera-framing EE Z for cracker Gemini detection.",
    )

    # Bowl detection pose at STIRRING_STATION.
    p.add_argument(
        "--bowl-detection-x",
        type=float,
        default=float(STIRRING_BOWL_DETECTION_EE_POSITION[0]),
        help="Camera-framing EE X for bowl Gemini detection at STIRRING_STATION.",
    )
    p.add_argument(
        "--bowl-detection-y",
        type=float,
        default=float(STIRRING_BOWL_DETECTION_EE_POSITION[1]),
        help="Camera-framing EE Y for bowl Gemini detection at STIRRING_STATION.",
    )
    p.add_argument(
        "--bowl-detection-z",
        type=float,
        default=float(STIRRING_BOWL_DETECTION_EE_POSITION[2]),
        help="Camera-framing EE Z for bowl Gemini detection at STIRRING_STATION.",
    )

    p.add_argument(
        "--carry-lift-m",
        type=float,
        default=EGG_CRACKER_CARRY_LIFT_M,
        help="Post-grasp lift height and transit carry height above home (m).",
    )
    p.add_argument(
        "--rotate-deg",
        type=float,
        default=ROTATE_DEG,
        help="Wrist rotation about world X to dump shell into the white bowl (deg).",
    )
    p.add_argument(
        "--rotate-lift-m",
        type=float,
        default=ROTATE_LIFT_M,
        help="Lift in +Z applied simultaneously with the shell-dump rotate (m).",
    )
    p.add_argument(
        "--release-drop-m",
        type=float,
        default=RELEASE_DROP_M,
        help="Descent below the above-pick pose before releasing the cracker (m).",
    )
    p.add_argument(
        "--final-up-m",
        type=float,
        default=FINAL_UP_M,
        help="Retract distance in +Z after releasing the cracker (m).",
    )
    p.add_argument(
        "--release-settle-s",
        type=float,
        default=RELEASE_SETTLE_S,
        help="Pause after opening the gripper before lifting away (s).",
    )

    p.add_argument("--gripper-lift-force", type=float, default=8.0)
    p.add_argument("--gripper-crack-force", type=float, default=70.0)

    p.add_argument("--retries", type=int, default=1, help="Gemini retries on failure.")
    p.add_argument(
        "--endeffector-transform-key",
        default=DEFAULT_ENDEFFECTOR_TRANSFORM_KEY,
        help="Redis key for the 4x4 base->flange transform.",
    )
    p.add_argument(
        "--gemini-response-path",
        default=str(DEFAULT_GEMINI_RESPONSE_PATH),
        help="Coarse cracker Gemini overlay path.",
    )
    p.add_argument(
        "--gemini-refine-response-path",
        default=str(DEFAULT_GEMINI_REFINE_RESPONSE_PATH),
        help="Refine cracker Gemini overlay path.",
    )
    p.add_argument(
        "--gemini-black-bowl-path",
        default=str(DEFAULT_GEMINI_BLACK_BOWL_PATH),
        help="Black bowl center Gemini overlay path.",
    )
    p.add_argument(
        "--gemini-white-bowl-path",
        default=str(DEFAULT_GEMINI_WHITE_BOWL_PATH),
        help="White bowl center Gemini overlay path.",
    )
    p.add_argument(
        "--orientation-source",
        choices=("fixed", "current"),
        default="fixed",
        help="Base grasp orientation for the perpendicular strip yaw.",
    )

    p.add_argument(
        "--servo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Coarse → refine → LK servo cracker grasp (default on).",
    )
    p.add_argument(
        "--above-standoff-m",
        type=float,
        default=ABOVE_STANDOFF_M,
        help="World +Z above coarse grasp Z for the tool-down refine/servo pose (m).",
    )
    p.add_argument(
        "--servo-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ENTER-gate each servo iteration (default off).",
    )
    p.add_argument("--servo-gain", type=float, default=SERVO_GAIN)
    p.add_argument("--servo-ki", type=float, default=SERVO_KI)
    p.add_argument("--servo-px-tol", type=float, default=SERVO_PX_TOL)
    p.add_argument("--servo-max-iters", type=int, default=SERVO_MAX_ITERS)
    p.add_argument("--servo-step-clip-m", type=float, default=SERVO_STEP_CLIP_M)
    p.add_argument("--servo-settle-s", type=float, default=SERVO_SETTLE_S)
    p.add_argument(
        "--servo-lateral-offset-x",
        type=float,
        default=SERVO_LATERAL_OFFSET_X,
    )
    p.add_argument(
        "--servo-lateral-offset-y",
        type=float,
        default=SERVO_LATERAL_OFFSET_Y,
    )
    p.add_argument(
        "--servo-log-dir",
        default=str(DEFAULT_SERVO_LOG_DIR),
        help="Folder for per-iteration servo frames.",
    )
    p.add_argument("--servo-target-u", type=int, default=SERVO_TARGET_U)
    p.add_argument("--servo-target-v", type=int, default=SERVO_TARGET_V)
    p.add_argument(
        "--servo-probe-mm",
        type=float,
        default=12.0,
        help="Jacobian probe distance (mm) for the visual servo.",
    )
    return p.parse_args()


def _transit_carry_pos(carry_lift_m: float) -> np.ndarray:
    return ARM_HOME_POSITION + np.array([0.0, 0.0, carry_lift_m], dtype=np.float64)


def _move_to_transit_carry(
    ctx: TaskContext,
    grip_R: np.ndarray,
    *,
    carry_lift_m: float,
) -> None:
    transit = _transit_carry_pos(carry_lift_m)
    arm.move_to(
        ctx,
        transit,
        grip_R,
        label=(
            f"[moving] transit carry pose {transit.tolist()} "
            f"(home + {carry_lift_m * 100:.0f} cm Z)"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )


def _drive_base(
    ctx: TaskContext,
    waypoint: BaseWaypoint,
    *,
    skip_base: bool,
) -> None:
    if skip_base:
        print(f"[moving] base drive to {waypoint.name} skipped (--skip-base)")
        return
    # Holonomic single-phase (default motion), same as grasp_and_move_controller
    # PAN_STATION leg: one drive to (x, y, yaw) from BASE_WAYPOINTS. Vision
    # localizes the cracker/bowls at the station, so we do not use the
    # three-phase oven-door landing here.
    base.go_to_pose(ctx, waypoint)


def _frame_cracker_and_pick(
    ctx: TaskContext,
    *,
    detection_xyz: tuple[float, float, float],
    retries: int,
    orientation_source: str,
    servo: bool,
    servo_gate: bool,
    above_standoff_m: float,
    gemini_refine_response_path: str,
    servo_gain: float,
    servo_ki: float,
    servo_px_tol: float,
    servo_max_iters: int,
    servo_step_clip_m: float,
    servo_settle_s: float,
    servo_lateral_offset_x: float,
    servo_lateral_offset_y: float,
    servo_log_dir: str | None,
    servo_target_px: tuple[int, int] | None,
    servo_probe_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detection pose → coarse/refine/servo → pick pose (no initial home move)."""
    detection_pos = np.array(detection_xyz, dtype=np.float64)
    arm.move_to(
        ctx,
        detection_pos,
        EGG_CRACKER_DETECTION_EE_ORIENTATION,
        label=(
            f"[moving] cracker detection pose {detection_pos.tolist()} "
            "(coarse Gemini framing)"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    coarse_path = ctx.gemini_response_path
    coarse = gemini.find_grasp_pose(
        ctx,
        Object.EGG_CRACKER,
        retries=retries,
        orientation_source=orientation_source,
    )
    print(f"[moving] coarse grasp: {coarse.position.tolist()}")
    if coarse.rim_yaw_applied and coarse.rim_yaw_deg is not None:
        print(f"[moving] coarse strip axis yaw: {coarse.rim_yaw_deg:+.2f} deg")

    if not servo:
        return coarse.position.astype(np.float64, copy=True), coarse.orientation

    grip_R = coarse.orientation
    above_pos = coarse.position.astype(np.float64, copy=True)
    above_pos[2] += float(above_standoff_m)
    above_pos[0] += -0.10
    arm.move_to(
        ctx,
        above_pos,
        grip_R,
        label=(
            f"[moving] tool-down above pose {above_pos.tolist()} "
            f"(+{above_standoff_m * 100:.0f} cm standoff for refine/servo)"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    ctx.gemini_response_path = gemini_refine_response_path
    try:
        refine = gemini.find_grasp_pose(
            ctx,
            Object.EGG_CRACKER,
            retries=retries,
            orientation_source=orientation_source,
        )
    finally:
        ctx.gemini_response_path = coarse_path

    print(f"[moving] refine grasp (open-loop): {refine.position.tolist()}")
    grip_R = refine.orientation

    boxes = refine.source_boxes
    if boxes is None or len(boxes) < 2:
        print(
            "[moving] refine returned <2 handle boxes; "
            "skipping servo, using refine open-loop position"
        )
        return refine.position.astype(np.float64, copy=True), grip_R

    boxes = [_expand_box(b, SERVO_MIN_BOX_W, SERVO_MIN_BOX_H) for b in boxes]
    print(f"[moving] template-match servo seed boxes: {boxes}")
    converged_ee, mid_z, final_px = visual_servo.servo_align_to_principal_point(
        ctx,
        boxes,
        fixed_ori=grip_R,
        gain=servo_gain,
        ki=servo_ki,
        px_tol=servo_px_tol,
        max_iters=servo_max_iters,
        step_clip_m=servo_step_clip_m,
        settle_s=servo_settle_s,
        target_px=servo_target_px,
        probe_delta_m=servo_probe_m,
        save_dir=servo_log_dir,
        step=servo_gate,
        grasp_z_nominal=float(refine.position[2]),
    )
    print(f"[moving] servo final pixels: {final_px}")

    pick_pos = np.array(
        [
            converged_ee[0] + servo_lateral_offset_x,
            converged_ee[1] + servo_lateral_offset_y,
            mid_z,
        ],
        dtype=np.float64,
    )
    print(
        f"[moving] servo-corrected pick: {pick_pos.tolist()} "
        f"(lateral offset [{servo_lateral_offset_x:+.4f}, {servo_lateral_offset_y:+.4f}])"
    )
    return pick_pos, grip_R


def _detect_bowl_center(
    ctx: TaskContext,
    obj: Object,
    *,
    detection_xyz: tuple[float, float, float],
    gemini_response_path: str,
    retries: int,
) -> np.ndarray:
    """Frame both bowls, then run Gemini center detection for ``obj``."""
    detection_pos = np.array(detection_xyz, dtype=np.float64)
    arm.move_to(
        ctx,
        detection_pos,
        ARM_HOME_ORIENTATION,
        label=f"[moving] bowl detection pose {detection_pos.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    prior = ctx.gemini_response_path
    ctx.gemini_response_path = gemini_response_path
    try:
        center = gemini.find_center(ctx, obj, retries=retries)
    finally:
        ctx.gemini_response_path = prior
    center = np.asarray(center, dtype=np.float64).reshape(3)
    print(f"[moving] {obj.value} center detected: {center.tolist()}")
    return center


def _ee_above_bowl(
    bowl_center: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    return np.asarray(bowl_center, dtype=np.float64).reshape(3) + np.asarray(
        offset, dtype=np.float64
    ).reshape(3)


def run_moving_cycle(
    ctx: TaskContext,
    *,
    skip_base: bool = False,
    carry_lift_m: float = EGG_CRACKER_CARRY_LIFT_M,
    gripper_lift_force: float = 8.0,
    gripper_crack_force: float = 70.0,
    retries: int = 1,
    orientation_source: str = "fixed",
    detection_xyz: tuple[float, float, float] | None = None,
    bowl_detection_xyz: tuple[float, float, float] | None = None,
    gemini_response_path: str | None = None,
    gemini_refine_response_path: str | None = None,
    gemini_black_bowl_path: str | None = None,
    gemini_white_bowl_path: str | None = None,
    servo: bool = True,
    servo_gate: bool = False,
    above_standoff_m: float = ABOVE_STANDOFF_M,
    servo_gain: float = SERVO_GAIN,
    servo_ki: float = SERVO_KI,
    servo_px_tol: float = SERVO_PX_TOL,
    servo_max_iters: int = SERVO_MAX_ITERS,
    servo_step_clip_m: float = SERVO_STEP_CLIP_M,
    servo_settle_s: float = SERVO_SETTLE_S,
    servo_lateral_offset_x: float = SERVO_LATERAL_OFFSET_X,
    servo_lateral_offset_y: float = SERVO_LATERAL_OFFSET_Y,
    servo_log_dir: str | None = None,
    servo_target_px: tuple[int, int] | None = None,
    servo_probe_m: float = 0.012,
    rotate_deg: float = ROTATE_DEG,
    rotate_lift_m: float = ROTATE_LIFT_M,
    release_drop_m: float = RELEASE_DROP_M,
    final_up_m: float = FINAL_UP_M,
    release_settle_s: float = RELEASE_SETTLE_S,
) -> None:
    """Grasp at EGG_CRACK_STATION, crack at STIRRING_STATION, return to release."""
    if gemini_response_path is not None:
        ctx.gemini_response_path = gemini_response_path
    if ctx.gemini_response_path is None:
        ctx.gemini_response_path = str(DEFAULT_GEMINI_RESPONSE_PATH)
    if gemini_refine_response_path is None:
        gemini_refine_response_path = str(DEFAULT_GEMINI_REFINE_RESPONSE_PATH)
    if gemini_black_bowl_path is None:
        gemini_black_bowl_path = str(DEFAULT_GEMINI_BLACK_BOWL_PATH)
    if gemini_white_bowl_path is None:
        gemini_white_bowl_path = str(DEFAULT_GEMINI_WHITE_BOWL_PATH)
    if detection_xyz is None:
        detection_xyz = tuple(float(v) for v in EGG_CRACKER_DETECTION_EE_POSITION)
    if bowl_detection_xyz is None:
        bowl_detection_xyz = tuple(float(v) for v in STIRRING_BOWL_DETECTION_EE_POSITION)
    if servo_log_dir is None:
        servo_log_dir = str(DEFAULT_SERVO_LOG_DIR)

    # 1. Home, then drive to the egg-crack station.
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[arm] move to home {ARM_HOME_POSITION.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )
    _drive_base(ctx, BaseWaypoint.EGG_CRACK_STATION, skip_base=skip_base)

    # 2. Vision grasp the cracker (stationary pipeline, no re-home).
    pick, grip_R = _frame_cracker_and_pick(
        ctx,
        detection_xyz=detection_xyz,
        retries=retries,
        orientation_source=orientation_source,
        servo=servo,
        servo_gate=servo_gate,
        above_standoff_m=above_standoff_m,
        gemini_refine_response_path=gemini_refine_response_path,
        servo_gain=servo_gain,
        servo_ki=servo_ki,
        servo_px_tol=servo_px_tol,
        servo_max_iters=servo_max_iters,
        servo_step_clip_m=servo_step_clip_m,
        servo_settle_s=servo_settle_s,
        servo_lateral_offset_x=servo_lateral_offset_x,
        servo_lateral_offset_y=servo_lateral_offset_y,
        servo_log_dir=servo_log_dir,
        servo_target_px=servo_target_px,
        servo_probe_m=servo_probe_m,
    )
    grasp.object(
        ctx,
        Object.EGG_CRACKER,
        pick_pos=pick,
        ori=grip_R,
        lift_dz_m=carry_lift_m,
    )
    above_pick = _above_pose(pick, grip_R)

    # 3. Transit carry → STIRRING_STATION.
    _move_to_transit_carry(ctx, grip_R, carry_lift_m=carry_lift_m)
    _drive_base(ctx, BaseWaypoint.STIRRING_STATION, skip_base=skip_base)

    # 4. Detect black bowl, move above it, crack.
    black_center = _detect_bowl_center(
        ctx,
        Object.PASTA_BOWL,
        detection_xyz=bowl_detection_xyz,
        gemini_response_path=gemini_black_bowl_path,
        retries=retries,
    )
    above_black = _ee_above_bowl(black_center, CRACK_EE_OFFSET_FROM_BOWL_CENTER)
    arm.move_to(
        ctx,
        above_black,
        grip_R,
        label=f"[moving] above black bowl {above_black.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    egg_crack.crack(
        ctx,
        crack_force=gripper_crack_force,
        lift_force=gripper_lift_force,
    )
    arm.move_to(
        ctx,
        above_black,
        grip_R,
        label=f"[moving] back above black bowl after crack {above_black.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 5. Detect white bowl, dump shell (rotate + lift, then unrotate).
    white_center = _detect_bowl_center(
        ctx,
        Object.PLASTIC_BOWL_TOP,
        detection_xyz=bowl_detection_xyz,
        gemini_response_path=gemini_white_bowl_path,
        retries=retries,
    )
    above_white = _ee_above_bowl(white_center, SHELL_EE_OFFSET_FROM_BOWL_CENTER)
    arm.move_to(
        ctx,
        above_white,
        grip_R,
        label=f"[moving] above white bowl {above_white.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    R_rot = _rotated_about_world_x(grip_R, rotate_deg)
    rotate_pos = above_white + np.array([0.0, 0.0, rotate_lift_m], dtype=np.float64)
    arm.move_to(
        ctx,
        rotate_pos,
        R_rot,
        label=(
            f"[moving] dump shell: rotate {rotate_deg:.0f} deg about world X "
            f"+ lift {rotate_lift_m * 100:.0f} cm {rotate_pos.tolist()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )
    arm.move_to(
        ctx,
        above_white,
        grip_R,
        label="[moving] unrotate after shell dump",
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 6. Transit carry → EGG_CRACK_STATION, release cracker at pick spot.
    _move_to_transit_carry(ctx, grip_R, carry_lift_m=carry_lift_m)
    _drive_base(ctx, BaseWaypoint.EGG_CRACK_STATION, skip_base=skip_base)

    arm.move_to(
        ctx,
        above_pick,
        grip_R,
        label=f"[moving] return to above pick spot {above_pick.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    release_pos = above_pick - np.array([0.0, 0.0, release_drop_m], dtype=np.float64)
    arm.move_to(
        ctx,
        release_pos,
        grip_R,
        label=(
            f"[moving] lower {release_drop_m * 100:.0f} cm below above pick "
            f"{release_pos.tolist()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    gripper.open_gripper(
        ctx.redis,
        spec.open_width,
        speed=spec.speed,
        force=spec.force,
        use_max_mode=True,
    )
    print(
        f"[moving] gripper opened — cracker released. Waiting {release_settle_s:.1f} s."
    )
    time.sleep(release_settle_s)

    up_pos = release_pos + np.array([0.0, 0.0, final_up_m], dtype=np.float64)
    arm.move_to(
        ctx,
        up_pos,
        grip_R,
        label=f"[moving] lift away {final_up_m * 100:.0f} cm {up_pos.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[arm] return to home {ARM_HOME_POSITION.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key
    ctx.gemini_response_path = args.gemini_response_path

    print(f"Step mode      : {'on' if args.step else 'off'}")
    print(f"Skip base      : {'yes' if args.skip_base else 'no'}")
    print(
        "Base route     : EGG_CRACK_STATION → STIRRING_STATION → EGG_CRACK_STATION"
    )
    print(f"Visual servo   : {'on' if args.servo else 'off (--no-servo)'}")
    print(f"Carry lift     : {args.carry_lift_m * 100:.0f} cm above home (transit + grasp)")
    print(f"Cracker detect : ({args.detection_x:.4f}, {args.detection_y:.4f}, {args.detection_z:.4f})")
    print(
        f"Bowl detect    : ({args.bowl_detection_x:.4f}, "
        f"{args.bowl_detection_y:.4f}, {args.bowl_detection_z:.4f})"
    )
    print(f"Coarse Gemini  : {args.gemini_response_path}")

    try:
        run_moving_cycle(
            ctx,
            skip_base=args.skip_base,
            carry_lift_m=args.carry_lift_m,
            gripper_lift_force=args.gripper_lift_force,
            gripper_crack_force=args.gripper_crack_force,
            retries=args.retries,
            orientation_source=args.orientation_source,
            detection_xyz=(args.detection_x, args.detection_y, args.detection_z),
            bowl_detection_xyz=(
                args.bowl_detection_x,
                args.bowl_detection_y,
                args.bowl_detection_z,
            ),
            gemini_response_path=args.gemini_response_path,
            gemini_refine_response_path=args.gemini_refine_response_path,
            gemini_black_bowl_path=args.gemini_black_bowl_path,
            gemini_white_bowl_path=args.gemini_white_bowl_path,
            servo=args.servo,
            servo_gate=args.servo_gate,
            above_standoff_m=args.above_standoff_m,
            servo_gain=args.servo_gain,
            servo_ki=args.servo_ki,
            servo_px_tol=args.servo_px_tol,
            servo_max_iters=args.servo_max_iters,
            servo_step_clip_m=args.servo_step_clip_m,
            servo_settle_s=args.servo_settle_s,
            servo_lateral_offset_x=args.servo_lateral_offset_x,
            servo_lateral_offset_y=args.servo_lateral_offset_y,
            servo_log_dir=args.servo_log_dir,
            servo_target_px=(args.servo_target_u, args.servo_target_v),
            servo_probe_m=args.servo_probe_mm / 1000.0,
            rotate_deg=args.rotate_deg,
            rotate_lift_m=args.rotate_lift_m,
            release_drop_m=args.release_drop_m,
            final_up_m=args.final_up_m,
            release_settle_s=args.release_settle_s,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
