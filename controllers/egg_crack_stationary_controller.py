#!/usr/bin/env python3
"""Stationary egg-cracker manipulation test (single-station, no base motion).

This is a stripped-down sibling of ``egg_crack_controller.py``. It keeps the
Gemini vision grasp and then runs a fixed in-place manipulation sequence
instead of the multi-station ingredient → mixing → sink flow. The base never
moves; the cart is assumed to already be at the work station.

  1. Arm → home pose.
  2. Arm → detection pose, coarse Gemini grasp, tool-down above pose, refine
     detect, LK visual-servo to principal point, then ``grasp.object`` picks up.
  3. Move the cracker 10 cm left (+Y).
  4. ``egg_crack.crack`` squeezes the cracker.
  5. Move back to the above-pick pose.
  6. Rotate 90° about world X while lifting ~10 cm, then unrotate back down.
  7. Lower to a bit below the above pose.
  8. Release (open gripper), wait 2 s.
  9. Go up.

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/egg_crack_stationary_controller.py -- --step

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
RealSense, OptiTrack on Redis, and ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
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

from zitibot_core import arm, gripper
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    DEFAULT_POS_TOL_M,
    EGG_CRACKER_STATIONARY_DETECTION_EE_ORIENTATION,
    EGG_CRACKER_STATIONARY_DETECTION_EE_POSITION,
    OBJECT_DEFAULTS,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_tasks import egg_crack, gemini, grasp, visual_servo

def _expand_box(
    box: tuple[int, int, int, int],
    min_w: int,
    min_h: int,
) -> tuple[int, int, int, int]:
    """Grow ``(x0, y0, x1, y1)`` around its center to at least ``min_w x min_h``.

    The center is preserved (it is the grasp point), so only the template
    extent changes. Boxes already larger than the minimum are left as-is.
    """
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w = max(x1 - x0, min_w)
    h = max(y1 - y0, min_h)
    nx0 = int(round(cx - w / 2.0))
    ny0 = int(round(cy - h / 2.0))
    nx1 = int(round(cx + w / 2.0))
    ny1 = int(round(cy + h / 2.0))
    return (nx0, ny0, nx1, ny1)


DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_RESPONSE_PATH = _CONTROLLERS.parent / "logs" / "gemini_response_egg_cracker.png"
DEFAULT_GEMINI_REFINE_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_egg_cracker_refine.png"
)
DEFAULT_SERVO_LOG_DIR = _CONTROLLERS.parent / "logs" / "visual_servo"

# Two-stage grasp + visual servo defaults.
ABOVE_STANDOFF_M = 0.12
SERVO_GAIN = 0.5
# Integral gain on accumulated pixel error. Pushes a stalled axis (constant
# residual error, frozen feature) past static friction / a soft limit that the
# clipped proportional step alone can't overcome. 0 disables.
SERVO_KI = 0.05
SERVO_PX_TOL = 16.0
SERVO_MAX_ITERS = 40
SERVO_STEP_CLIP_M = 0.006
SERVO_SETTLE_S = 0.35
SERVO_LATERAL_OFFSET_X = 0.0
SERVO_LATERAL_OFFSET_Y = 0.0
# Servo target pixel (640x480 frame). Offset from the old (384, 288) aim point:
# -64 u (-10% of 640 px width, left), +48 v (+10% of 480 px height, down).
SERVO_TARGET_U = 352
SERVO_TARGET_V = 288
# Minimum servo template size (px). Gemini's refine box_2d is sometimes a tiny
# square around just the blue grasp mark; expand to at least this around the
# box center so the template has enough texture to track reliably.
SERVO_MIN_BOX_W = 70
SERVO_MIN_BOX_H = 70

# Post-grasp lift applied by ``grasp.object`` after closing on the cracker.
EGG_CRACKER_CARRY_LIFT_M = 0.10

# In-place manipulation tunables (the new stationary sequence).
SHIFT_LEFT_M = 0.20      # move the held cracker this far in +Y (left) to crack
ROTATE_DEG = 90.0          # wrist rotation about world X (then undone)
ROTATE_LIFT_M = 0.10       # lift this far in +Z simultaneously with the rotate
ROTATE_X_OFFSET_M = -0.05    # offset the rotate pose in -X to to ensure we're above the bowl
RELEASE_DROP_M = 0.01      # descend this far below the above pose before release
FINAL_UP_M = 0.15          # retract this far in +Z after release
RELEASE_SETTLE_S = 2.0     # pause after opening the gripper, before going up


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stationary egg-cracker grasp + in-place manipulation test."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate every motion / gripper step.",
    )

    # Detection-pose override. Defaults to ``EGG_CRACKER_DETECTION_EE_POSITION``.
    p.add_argument(
        "--detection-x",
        type=float,
        default=float(EGG_CRACKER_STATIONARY_DETECTION_EE_POSITION[0]),
        help="Camera-framing EE X for Gemini detection.",
    )
    p.add_argument(
        "--detection-y",
        type=float,
        default=float(EGG_CRACKER_STATIONARY_DETECTION_EE_POSITION[1]),
        help="Camera-framing EE Y for Gemini detection.",
    )
    p.add_argument(
        "--detection-z",
        type=float,
        default=float(EGG_CRACKER_STATIONARY_DETECTION_EE_POSITION[2]),
        help="Camera-framing EE Z for Gemini detection.",
    )

    # In-place manipulation tunables.
    p.add_argument(
        "--shift-left-m",
        type=float,
        default=SHIFT_LEFT_M,
        help="Distance to move the held cracker in +Y (left) before cracking (m).",
    )
    p.add_argument(
        "--rotate-deg",
        type=float,
        default=ROTATE_DEG,
        help="Wrist rotation about world X applied then undone (deg).",
    )
    p.add_argument(
        "--rotate-lift-m",
        type=float,
        default=ROTATE_LIFT_M,
        help="Lift in +Z applied simultaneously with the rotate (m).",
    )
    p.add_argument(
        "--release-drop-m",
        type=float,
        default=RELEASE_DROP_M,
        help="Descent below the above pose before releasing the cracker (m).",
    )
    p.add_argument(
        "--final-up-m",
        type=float,
        default=FINAL_UP_M,
        help="Retract distance in +Z after releasing (m).",
    )
    p.add_argument(
        "--release-settle-s",
        type=float,
        default=RELEASE_SETTLE_S,
        help="Pause after opening the gripper before going up (s).",
    )
    p.add_argument(
        "--carry-lift-m",
        type=float,
        default=EGG_CRACKER_CARRY_LIFT_M,
        help="Post-grasp lift height applied by grasp.object (m).",
    )

    # Gripper forces.
    p.add_argument("--gripper-lift-force", type=float, default=8.0)
    p.add_argument("--gripper-crack-force", type=float, default=70.0)

    # Gemini / camera plumbing.
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
            "Base grasp orientation for the perpendicular yaw: "
            "fixed=Object.EGG_CRACKER default (tool-down), "
            "current=live EE orientation at detection time."
        ),
    )

    # Two-stage detect + LK visual servo.
    p.add_argument(
        "--servo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Coarse detect -> tool-down above -> refine detect -> LK servo "
            "to camera principal point (default on)."
        ),
    )
    p.add_argument(
        "--above-standoff-m",
        type=float,
        default=ABOVE_STANDOFF_M,
        help="World +Z above coarse grasp Z for the tool-down refine/servo pose (m).",
    )
    p.add_argument(
        "--gemini-refine-response-path",
        default=str(DEFAULT_GEMINI_REFINE_RESPONSE_PATH),
        help="Where to save the second (refine) Gemini overlay image.",
    )
    p.add_argument(
        "--servo-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "ENTER-gate each servo move (default off; servo runs continuously: "
            "publish + settle, no per-move ENTER). Pass --servo-gate to step "
            "through each move with ENTER for debugging."
        ),
    )
    p.add_argument("--servo-gain", type=float, default=SERVO_GAIN)
    p.add_argument(
        "--servo-ki",
        type=float,
        default=SERVO_KI,
        help=(
            "Integral gain on accumulated pixel error; pushes a stalled axis "
            "past a sticking point the proportional step can't. 0 disables."
        ),
    )
    p.add_argument("--servo-px-tol", type=float, default=SERVO_PX_TOL)
    p.add_argument("--servo-max-iters", type=int, default=SERVO_MAX_ITERS)
    p.add_argument("--servo-step-clip-m", type=float, default=SERVO_STEP_CLIP_M)
    p.add_argument("--servo-settle-s", type=float, default=SERVO_SETTLE_S)
    p.add_argument(
        "--servo-lateral-offset-x",
        type=float,
        default=SERVO_LATERAL_OFFSET_X,
        help="World +X offset from servo-converged EE to grasp TCP (camera mount).",
    )
    p.add_argument(
        "--servo-lateral-offset-y",
        type=float,
        default=SERVO_LATERAL_OFFSET_Y,
        help="World +Y offset from servo-converged EE to grasp TCP (camera mount).",
    )
    p.add_argument(
        "--servo-log-dir",
        default=str(DEFAULT_SERVO_LOG_DIR),
        help=(
            "Folder for per-iteration annotated servo frames (servo_NNN.png). "
            "Cleared at the start of each run."
        ),
    )
    p.add_argument(
        "--servo-target-u",
        type=int,
        default=SERVO_TARGET_U,
        help="Servo target pixel U (column) the bolt midpoint is driven to.",
    )
    p.add_argument(
        "--servo-target-v",
        type=int,
        default=SERVO_TARGET_V,
        help="Servo target pixel V (row) the bolt midpoint is driven to.",
    )
    p.add_argument(
        "--servo-probe-mm",
        type=float,
        default=12.0,
        help=(
            "World-frame distance (mm) for the two Jacobian calibration probes "
            "(+X, +Y) the servo runs at startup to learn the image->world map."
        ),
    )
    return p.parse_args()


def _vision_pick_pose(
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
    """Home → detect → [above → refine → servo] → pick pose."""
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[arm] move to home {ARM_HOME_POSITION.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )

    detection_pos = np.array(detection_xyz, dtype=np.float64)
    arm.move_to(
        ctx,
        detection_pos,
        EGG_CRACKER_STATIONARY_DETECTION_EE_ORIENTATION,
        label=(
            f"[arm] move to egg-cracker detection pose "
            f"{detection_pos.tolist()} (coarse Gemini framing)"
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
    print(f"[stationary] coarse grasp: {coarse.position.tolist()}")
    if coarse.rim_yaw_applied and coarse.rim_yaw_deg is not None:
        print(f"[stationary] coarse strip axis yaw: {coarse.rim_yaw_deg:+.2f} deg")

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
            f"[stationary] tool-down above pose {above_pos.tolist()} "
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

    print(f"[stationary] refine grasp (open-loop): {refine.position.tolist()}")
    grip_R = refine.orientation

    boxes = refine.source_boxes
    if boxes is None or len(boxes) < 2:
        print(
            "[stationary] refine returned <2 handle boxes; "
            "skipping servo, using refine open-loop position"
        )
        return refine.position.astype(np.float64, copy=True), grip_R

    # Gemini often boxes only the small blue grasp mark on the close-up refine
    # frame, which makes a tiny, texture-poor template the servo can't track.
    # Expand each box around its center (center == grasp point, unchanged) to a
    # usable minimum size for template matching.
    boxes = [_expand_box(b, SERVO_MIN_BOX_W, SERVO_MIN_BOX_H) for b in boxes]

    print(f"[stationary] template-match servo seed boxes: {boxes}")
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
    print(f"[stationary] servo final pixels: {final_px}")

    pick_pos = np.array(
        [
            converged_ee[0] + servo_lateral_offset_x,
            converged_ee[1] + servo_lateral_offset_y,
            mid_z,
        ],
        dtype=np.float64,
    )
    print(
        f"[stationary] servo-corrected pick: {pick_pos.tolist()} "
        f"(lateral offset [{servo_lateral_offset_x:+.4f}, {servo_lateral_offset_y:+.4f}])"
    )
    return pick_pos, grip_R


def _above_pose(pick: np.ndarray, grip_R: np.ndarray) -> np.ndarray:
    """Pre-grasp "above" pose for the egg cracker (mirrors ``grasp.object``)."""
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    pick = np.asarray(pick, dtype=np.float64).reshape(3)
    dz = float(spec.approach_dz)
    if spec.approach_along_tool_z:
        return pick - np.asarray(grip_R, dtype=np.float64).reshape(3, 3)[:, 2] * dz
    return pick + np.array([0.0, 0.0, dz], dtype=np.float64)


def _rotated_about_world_x(R: np.ndarray, deg: float) -> np.ndarray:
    """Return ``R`` pre-rotated by ``deg`` about the world (base-frame) X axis."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    return Rx @ np.asarray(R, dtype=np.float64).reshape(3, 3)


def run_stationary_cycle(
    ctx: TaskContext,
    *,
    carry_lift_m: float = EGG_CRACKER_CARRY_LIFT_M,
    gripper_lift_force: float = 8.0,
    gripper_crack_force: float = 70.0,
    retries: int = 1,
    orientation_source: str = "fixed",
    detection_xyz: tuple[float, float, float] | None = None,
    gemini_response_path: str | None = None,
    servo: bool = True,
    servo_gate: bool = True,
    above_standoff_m: float = ABOVE_STANDOFF_M,
    gemini_refine_response_path: str | None = None,
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
    shift_left_m: float = SHIFT_LEFT_M,
    rotate_deg: float = ROTATE_DEG,
    rotate_lift_m: float = ROTATE_LIFT_M,
    release_drop_m: float = RELEASE_DROP_M,
    final_up_m: float = FINAL_UP_M,
    release_settle_s: float = RELEASE_SETTLE_S,
) -> None:
    """Vision grasp the cracker, then run the in-place manipulation sequence."""
    if gemini_response_path is not None:
        ctx.gemini_response_path = gemini_response_path
    if ctx.gemini_response_path is None:
        ctx.gemini_response_path = str(DEFAULT_GEMINI_RESPONSE_PATH)
    if detection_xyz is None:
        detection_xyz = tuple(float(v) for v in EGG_CRACKER_STATIONARY_DETECTION_EE_POSITION)

    if gemini_refine_response_path is None:
        gemini_refine_response_path = str(DEFAULT_GEMINI_REFINE_RESPONSE_PATH)
    if servo_log_dir is None:
        servo_log_dir = str(DEFAULT_SERVO_LOG_DIR)

    # 1-2. Home → coarse/refine/servo → pick up + lift.
    pick, grip_R = _vision_pick_pose(
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

    above = _above_pose(pick, grip_R)

    # 4. Move the held cracker 10 cm left (+Y), at the above height.
    left_pos = above + np.array([0.0, shift_left_m, 0.0], dtype=np.float64)
    arm.move_to(
        ctx,
        left_pos,
        grip_R,
        label=f"[stationary] move {shift_left_m * 100:.0f} cm left {left_pos.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 5. Crack (squeeze the held cracker).
    egg_crack.crack(
        ctx,
        crack_force=gripper_crack_force,
        lift_force=gripper_lift_force,
    )

    # 6. Move back to the above-pick pose.
    arm.move_to(
        ctx,
        above,
        grip_R,
        label=f"[stationary] back to above pose {above.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 7. Rotate 90° about world X while simultaneously lifting, then unrotate
    #    back down to the above pose.
    R_rot = _rotated_about_world_x(grip_R, rotate_deg)
    rotate_pos = above + np.array([0.0, 0.0, rotate_lift_m], dtype=np.float64)
    arm.move_to(
        ctx,
        rotate_pos,
        R_rot,
        label=(
            f"[stationary] rotate {rotate_deg:.0f} deg about world X "
            f"+ lift {rotate_lift_m * 100:.0f} cm {rotate_pos.tolist()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )
    arm.move_to(
        ctx,
        above,
        grip_R,
        label="[stationary] unrotate + lower back to grasp orientation",
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 8. Lower to a bit below the above pose.
    release_pos = above - np.array([0.0, 0.0, release_drop_m], dtype=np.float64)
    arm.move_to(
        ctx,
        release_pos,
        grip_R,
        label=(
            f"[stationary] lower {release_drop_m * 100:.0f} cm below above "
            f"{release_pos.tolist()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 9. Release, then wait.
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    gripper.open_gripper(
        ctx.redis,
        spec.open_width,
        speed=spec.speed,
        force=spec.force,
        use_max_mode=True,
    )
    print(f"[stationary] gripper opened — cracker released. Waiting {release_settle_s:.1f} s.")
    time.sleep(release_settle_s)

    # 10. Go up.
    up_pos = release_pos + np.array([0.0, 0.0, final_up_m], dtype=np.float64)
    arm.move_to(
        ctx,
        up_pos,
        grip_R,
        label=f"[stationary] go up {final_up_m * 100:.0f} cm {up_pos.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key
    ctx.gemini_response_path = args.gemini_response_path

    print(f"Step mode      : {'on' if args.step else 'off'}")
    print(f"Base move      : none (stationary)")
    print(f"Visual servo   : {'on' if args.servo else 'off (--no-servo)'}")
    if args.servo:
        print(f"Above standoff : {args.above_standoff_m * 100:.0f} cm")
        print(
            f"Servo          : gain={args.servo_gain} ki={args.servo_ki} "
            f"px_tol={args.servo_px_tol} max_iters={args.servo_max_iters}"
        )
        print(
            f"Lateral offset : x={args.servo_lateral_offset_x:+.4f} m  "
            f"y={args.servo_lateral_offset_y:+.4f} m"
        )
        print(f"Refine Gemini  : {args.gemini_refine_response_path}")
        print(f"Servo target   : px=({args.servo_target_u}, {args.servo_target_v})")
        print(
            f"Servo mode     : seed Jacobian + sign flip (no cal probes) "
            f"({'ENTER-gated discrete' if args.servo_gate else 'continuous chase'})"
        )
        print(f"Servo frames   : {args.servo_log_dir} (cleared each run)")
        print(
            f"Servo gating   : "
            f"{'ENTER per move' if args.servo_gate else 'off (--no-servo-gate)'}"
        )
    print(f"Shift left     : {args.shift_left_m * 100:.0f} cm (+Y)")
    print(f"Rotate         : {args.rotate_deg:.0f} deg about world X + lift {args.rotate_lift_m * 100:.0f} cm (then undone)")
    print(f"Release drop   : {args.release_drop_m * 100:.0f} cm below above pose")
    print(f"Coarse Gemini  : {args.gemini_response_path}")

    try:
        run_stationary_cycle(
            ctx,
            carry_lift_m=args.carry_lift_m,
            gripper_lift_force=args.gripper_lift_force,
            gripper_crack_force=args.gripper_crack_force,
            retries=args.retries,
            orientation_source=args.orientation_source,
            detection_xyz=(args.detection_x, args.detection_y, args.detection_z),
            gemini_response_path=args.gemini_response_path,
            servo=args.servo,
            servo_gate=args.servo_gate,
            above_standoff_m=args.above_standoff_m,
            gemini_refine_response_path=args.gemini_refine_response_path,
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
            shift_left_m=args.shift_left_m,
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
