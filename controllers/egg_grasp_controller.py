#!/usr/bin/env python3
"""Egg detector + grasper (vision pick of a single whole egg).

Minimal flow, meant as the starting point for an egg-handling pipeline:

  1. Arm → home pose.
  2. Base → ``INGREDIENT_STATION`` (the ONLY base move; skip with --skip-base).
  3. Arm → detection pose (camera framing for Gemini).
  4. Gemini detects ONE point on the egg; ``_build_pose_egg`` turns it into a
     tool-straight-down grasp at the home wrist yaw (no detected-axis rotation —
     the egg is symmetric).
  5. ``grasp.object`` opens the jaws, descends, force-closes on the egg
     (GRASP mode) at a gentle ``--gripper-force`` (default 1 N), then lifts.

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/egg_grasp_controller.py -- --step
  ./ZitiBot/launch_zitibot_full.sh controllers/egg_grasp_controller.py -- --skip-base --step

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
TidyBot base ``redis_driver`` (unless --skip-base), RealSense, OptiTrack on
Redis, and ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm
from zitibot_core import base
from zitibot_core import gains
from zitibot_core import gripper
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    DEFAULT_GRIPPER_SPEED,
    OBJECT_DEFAULTS,
    PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
    PRECISE_GRASP_MAX_LINEAR_VELOCITY,
    PRECISE_GRASP_MOVE_TIMEOUT_S,
    PRECISE_GRASP_ORIENTATION_KP,
    PRECISE_GRASP_POSITION_KP,
    BaseWaypoint,
    HOME_POS_TOL_M,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_core.runner import step_gate
from zitibot_tasks import gemini, grasp

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_RESPONSE_PATH = _CONTROLLERS.parent / "logs" / "gemini_response_egg.png"

# Default grasp force comes straight from the Object.EGG spec's force param.
EGG_DEFAULT_FORCE_N = float(OBJECT_DEFAULTS[Object.EGG].force)

# Post-release J0 (base) shake to shed the tongs off the open jaws — same
# pattern/tunables as the egg cracker's post-release shake (small 2° strokes,
# tight stroke tolerance, no warmup nudge: seed current joints, switch
# controllers, briefly wait for the swap, then stroke).
TONGS_RELEASE_SHAKE_DELTA_DEG = 2.0   # per-stroke J0 amplitude
TONGS_RELEASE_SHAKE_CYCLES = 2        # UP/DOWN J0 cycles
TONGS_RELEASE_SHAKE_TOL_RAD = 0.0199  # tight: a 2° stroke must actually move
TONGS_RELEASE_SHAKE_TIMEOUT_S = 0.5
TONGS_RELEASE_SHAKE_POST_SWITCH_WAIT_S = 0.15


# Tongs reach: how far in front of the EE the tong TIPS extend, along the EE's
# facing direction (its +X, flattened to horizontal -- the yaw). The egg grasp
# target is set this far BACK from the detected egg along that horizontal yaw
# axis, so the EE sits ~7.5 in (19.05 cm) behind the egg at the egg's height and
# the tongs reach forward onto it. The offset is horizontal only, NOT down. Too
# small a value drives the EE straight onto the egg.
TONGS_REACH_M = 0.21

# Constant angular bias (deg, CCW about world +Z) between the EE +X axis and the
# direction the *held tongs* actually point. The gripper grabs the tongs with a
# ~45 deg tool-Z offset, so the tongs do NOT lie along flange +X -- they lead it
# by roughly this much toward +Y. Measured from a run where flange +X was
# achieved at -34.7 deg yet the tongs visually pointed ~straight +X. We therefore
# command flange +X to ``aim - bias`` so the real tongs point at the target.
# Fine-tune with ``--tong-yaw-offset-deg`` if the tips land to one side.
TONGS_EE_X_YAW_BIAS_DEG = 35.0

# Clamp on the commanded wrist yaw (deg, about world +Z, relative to the home
# wrist orientation) when aiming the tongs at the egg. The detected egg bearing
# can push the wrist to awkward/unreachable yaws; we limit the EE so it only
# ever sits between -15 deg and +45 deg around Z. When the clamp engages, the
# EE backoff direction is recomputed from the clamped yaw so the standoff still
# matches where the tongs actually point.
EGG_GRASP_YAW_MIN_DEG = -15.0
EGG_GRASP_YAW_MAX_DEG = 45.0

# Straight-up lift (m) after setting the tongs back down and opening, so the
# gripper clears the released tongs before returning home.
TONGS_RETURN_LIFT_M = 0.05

# Loose 5 cm clearance lift inserted after each pick/release so the arm
# reliably clears the table/object before the next move, WITHOUT burning time
# converging precisely. NOTE: the tolerance must stay BELOW the lift distance —
# arm.move_to converges on "within tol AND settled", so a tolerance >= the lift
# would let the move finish immediately (already within tol, already stopped)
# and the arm wouldn't lift at all.
CLEARANCE_LIFT_M = 0.05
CLEARANCE_LIFT_TOL_M = 0.03

# Shift the egg-release target (over the cracker cradle) back this far in world
# -X. Applied to the EE XY so BOTH the above-cradle approach and the final
# release/lower poses move back together.
CRACKER_RELEASE_X_BACK_M = 0.00


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vision-detect a single egg and force-grasp it (tool-down, gentle 1 N)."
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
            "Do not drive the base to INGREDIENT_STATION. Use when the cart is "
            "already parked in front of the egg for arm-only debugging."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Gripper-only debug: move the arm to home, open the gripper, then "
            "force-close with the egg grasp (no base move, no Gemini, no "
            "descend/lift). Verifies and prints the grasp result."
        ),
    )
    p.add_argument(
        "--tongs-reach-cm",
        type=float,
        default=TONGS_REACH_M * 100.0,
        help=(
            "How far in front of the EE the tong tips extend along the EE's "
            "facing yaw (+X, horizontal); the EE grasp target is set this far "
            "back from the detected egg along that axis so the tongs reach "
            "forward onto it. Default 19.05 cm (7.5 in)."
        ),
    )

    # Grasp tunables.
    p.add_argument(
        "--gripper-speed",
        type=float,
        default=0.02,
        help=(
            "Gripper closing speed in m/s. Slow by default so the jaws creep "
            "to the target width; Franka Hand maxes near 0.1 m/s."
        ),
    )
    p.add_argument(
        "--close-width",
        type=float,
        default=0.03,
        help=(
            "Tongs+egg FIRST-stage MOVE close width (m). The gripper MOVE-closes "
            "the tongs to this width (no squeeze) — enough to grip the egg and "
            "pick it up, but NOT all the way shut — then force-closes "
            "(--gripper-force) once the egg is lifted in the tongs. Default 0.03 "
            "m. For the standalone egg grasp this just bounds the GRASP-mode "
            "travel."
        ),
    )
    p.add_argument(
        "--gripper-force",
        type=float,
        default=EGG_DEFAULT_FORCE_N,
        help=(
            "Grasp force (N) for the force-close on the egg. The default "
            f"({EGG_DEFAULT_FORCE_N:.1f} N, from the Object.EGG force param) is a "
            "gentle hold that won't crush the shell; raise if the egg slips "
            "on lift."
        ),
    )

    # Egg-cracker release (tongs+egg cycle only).
    p.add_argument(
        "--release-into-cracker",
        dest="release_into_cracker",
        action="store_true",
        default=True,
        help=(
            "After picking the egg up with the tongs, detect the gray "
            "one-handed egg cracker, carry the egg over its cradle, lower in "
            "precise mode, and open the tongs to drop the egg in (DEFAULT)."
        ),
    )
    p.add_argument(
        "--no-release-into-cracker",
        dest="release_into_cracker",
        action="store_false",
        help="Stop after the egg is held in the tongs at home (skip the cracker drop).",
    )
    p.add_argument(
        "--release-width",
        type=float,
        default=0.03,
        help=(
            "Gripper width (m) to OPEN the tongs back to when releasing the egg "
            "into the cracker -- the tong-hold width, open enough to drop the "
            "egg but still holding the tongs themselves. Default 0.03 m."
        ),
    )
    p.add_argument(
        "--tong-yaw-offset-deg",
        type=float,
        default=TONGS_EE_X_YAW_BIAS_DEG,
        help=(
            "Constant CCW yaw bias (deg) between the real tong-tip direction and "
            "the EE +X axis -- the held tongs lead flange +X by this much, so we "
            "command flange +X to aim-bias. If the tips land consistently to one "
            "side of the target, dial this until centered (applies to both the "
            f"egg pickup and the cracker drop). Default {TONGS_EE_X_YAW_BIAS_DEG:g}."
        ),
    )
    p.add_argument(
        "--cracker-above-cm",
        type=float,
        default=14.0,
        help=(
            "Height (cm) above the cracker cradle to stage the tong tips before "
            "the precise lower. Default 14.0 cm."
        ),
    )
    p.add_argument(
        "--cracker-release-cm",
        type=float,
        default=7.0,
        help=(
            "Height (cm) above the cracker cradle at which the tongs open to "
            "release the egg. Default 7.0 cm."
        ),
    )

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
    return p.parse_args()


def _suffixed_path(base: str | Path, suffix: str) -> str:
    """Insert ``suffix`` before the extension of ``base`` (e.g. foo.png -> foo_x.png)."""
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _clearance_lift(
    ctx: TaskContext,
    *,
    dz_m: float = CLEARANCE_LIFT_M,
    tol_m: float = CLEARANCE_LIFT_TOL_M,
    label: str = "[arm] clearance lift",
) -> None:
    """Lift straight up ``dz_m`` from the CURRENT EE pose with a loose tolerance.

    Reads the live EE pose so it's relative to wherever the arm just finished,
    keeps the current wrist orientation, and uses a loose (but < ``dz_m``)
    tolerance so the move clears quickly without precise convergence.
    """
    pose = arm.read_current_ee_world(ctx.redis)
    if pose is None:
        print(f"{label}: EE pose unavailable; skipping clearance lift.")
        return
    cur_pos, cur_ori = pose
    lift = np.asarray(cur_pos, dtype=np.float64).reshape(3) + np.array(
        [0.0, 0.0, float(dz_m)], dtype=np.float64
    )
    arm.move_to(
        ctx,
        lift,
        cur_ori,
        label=f"{label} +{dz_m * 100:.0f} cm -> {lift.tolist()}",
        tol_m=float(tol_m),
    )


def _cradle_response_path(base: str | Path) -> str:
    """Dedicated save path for the cradle-CENTER detection overlay.

    Deliberately does NOT contain ``cracker`` in the name: the egg-cracker
    GRASP detections (egg_crack_controller, etc.) all save to
    ``gemini_response_egg_cracker*.png``, so any ``cracker``-derived name risks
    being overwritten by a later grasp detection. Use a distinct filename in the
    same logs dir so the cradle image always survives.
    """
    return str(Path(base).with_name("gemini_response_cradle_center.png"))


def _aim_tongs_at(
    point: np.ndarray,
    *,
    yaw_offset_deg: float = 0.0,
    yaw_min_deg: float = EGG_GRASP_YAW_MIN_DEG,
    yaw_max_deg: float = EGG_GRASP_YAW_MAX_DEG,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Yaw the home wrist so the *held tongs* aim horizontally at ``point``.

    The tongs do NOT lie along the EE +X axis: the gripper grabs them with a
    ~45 deg tool-Z offset, so the real tong-tip direction leads flange +X by a
    roughly constant ``yaw_offset_deg`` (CCW about world +Z). To make the tongs
    point at ``point`` we therefore command flange +X to ``aim - yaw_offset``:

        ``dyaw`` = signed world-+Z angle that carries home flange +X onto the
        horizontal aim, and the wrist command is ``cmd_yaw = dyaw - offset`` so
        that (flange +X) + offset == aim, i.e. the tongs sit on the aim.

    The rotation is a world-frame pre-multiply ``Rz(cmd_yaw) @ home_R``. The
    backoff (``reach_hat``) uses the true aim, so the EE sits ``reach`` behind
    the target along the direction the real tongs point. If the tips still land
    to one side, dial ``--tong-yaw-offset-deg``.

    Returns ``(reach_hat, hold_R, cmd_yaw)`` where ``reach_hat`` is the
    horizontal unit aim from the EE toward ``point``, ``hold_R`` is the yawed
    EE orientation, and ``cmd_yaw`` is the commanded yaw (radians).
    """
    home_R = np.asarray(ARM_HOME_ORIENTATION, dtype=np.float64)
    aim = np.asarray(point, dtype=np.float64).reshape(3) - ARM_HOME_POSITION
    aim[2] = 0.0  # horizontal bearing only (yaw)
    aim_hat = aim / np.linalg.norm(aim)
    home_x = home_R[:, 0].copy()
    home_x[2] = 0.0
    home_x /= np.linalg.norm(home_x)
    # Signed world-+Z angle (CCW) from home flange +X to the aim. Rz(dyaw)
    # carries flange +X exactly onto the aim; subtract the tong/flange bias so
    # the *tongs* (flange +X + bias) end up on the aim instead.
    dyaw = float(np.arctan2(
        home_x[0] * aim_hat[1] - home_x[1] * aim_hat[0],  # cross_z
        float(np.dot(home_x, aim_hat)),                    # dot
    ))
    cmd_yaw = dyaw - np.radians(float(yaw_offset_deg))
    # Clamp the EE yaw so the wrist only sits within [yaw_min, yaw_max] about
    # world +Z (relative to home). The detected bearing can otherwise push the
    # wrist to awkward yaws.
    cmd_yaw_clamped = float(
        np.clip(cmd_yaw, np.radians(float(yaw_min_deg)), np.radians(float(yaw_max_deg)))
    )
    if cmd_yaw_clamped != cmd_yaw:
        print(
            f"[egg] wrist yaw clamped {np.degrees(cmd_yaw):+.1f} -> "
            f"{np.degrees(cmd_yaw_clamped):+.1f} deg "
            f"(limit [{yaw_min_deg:+.1f}, {yaw_max_deg:+.1f}])"
        )
    cmd_yaw = cmd_yaw_clamped
    c, s = float(np.cos(cmd_yaw)), float(np.sin(cmd_yaw))
    r_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    hold_R = r_z @ home_R
    # Backoff direction = where the tongs actually point after the (possibly
    # clamped) yaw: flange +X rotated by (cmd_yaw + bias). When unclamped this
    # equals the true aim (cmd_yaw + bias == dyaw), so the standoff stays
    # consistent with the wrist orientation in all cases.
    tong_yaw = cmd_yaw + np.radians(float(yaw_offset_deg))
    ct, st = float(np.cos(tong_yaw)), float(np.sin(tong_yaw))
    reach_hat = np.array(
        [ct * home_x[0] - st * home_x[1], st * home_x[0] + ct * home_x[1], 0.0],
        dtype=np.float64,
    )
    reach_hat /= np.linalg.norm(reach_hat)
    return reach_hat, hold_R, cmd_yaw


def _release_egg_into_cracker(
    ctx: TaskContext,
    *,
    tongs_reach_m: float,
    gripper_speed: float,
    release_width: float,
    above_dz_m: float,
    release_dz_m: float,
    tong_yaw_offset_deg: float,
    retries: int,
    cradle: np.ndarray | None = None,
) -> None:
    """Drop the tong-held egg into the egg-cracker cradle.

    Yaws the wrist so the tong tips reach over the cradle center, moves a bit
    ABOVE it, then -- in precise mode -- lowers so the tips sit ``release_dz_m``
    above the cradle and opens the tongs back to ``release_width`` (the
    tong-hold width) so the egg falls out while the tongs themselves stay held.
    Returns the arm to home.

    ``cradle`` is the pre-detected world cradle-center position. When ``None``
    (legacy), the cradle is detected here with ``gemini.find_center``.
    """
    if cradle is None:
        cracker = gemini.find_center(ctx, Object.EGG_CRACKER, retries=retries)
        cracker = np.asarray(cracker, dtype=np.float64).reshape(3)
        print(f"[cracker] cradle center detected: {cracker.tolist()}")
    else:
        cracker = np.asarray(cradle, dtype=np.float64).reshape(3)
        print(f"[cracker] using pre-detected cradle center: {cracker.tolist()}")
    
    cracker_pos = cracker + np.array([-0.02, -0.02, 0.00], dtype=np.float64)

    reach_hat, hold_R, cmd_yaw = _aim_tongs_at(cracker_pos, yaw_offset_deg=tong_yaw_offset_deg)
    # EE XY so the tong tips (tongs_reach_m out along the aim) sit over the
    # cradle center; EE Z is set per phase so the TIPS land at the wanted
    # height above the cradle.
    ee_xy = cracker_pos - float(tongs_reach_m) * reach_hat
    # Nudge the EE XY back in world -X so the approach + release sit 2 cm
    # behind the detected cradle center.
    ee_xy = ee_xy + np.array([-CRACKER_RELEASE_X_BACK_M, 0.0, 0.0], dtype=np.float64)
    above = np.array([ee_xy[0], ee_xy[1], cracker_pos[2] + float(above_dz_m)], dtype=np.float64)
    low = np.array([ee_xy[0], ee_xy[1], cracker_pos[2] + float(release_dz_m)], dtype=np.float64)
    print(
        f"[cracker] tips over cradle along aim "
        f"[{reach_hat[0]:+.3f}, {reach_hat[1]:+.3f}, {reach_hat[2]:+.3f}] "
        f"(wrist yaw command {np.degrees(cmd_yaw):+.1f} deg); "
        f"above +{above_dz_m * 100:.1f} cm then release +{release_dz_m * 100:.1f} cm"
    )

    # Approach the "a bit above" pose at normal speed/stiffness.
    arm.move_to(
        ctx,
        above,
        hold_R,
        label=f"[cracker] tong tips above cradle {above.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )
    # Precise mode for the final lower + release so the egg lands in the cradle.
    snap = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="cracker-release",
    )
    try:
        arm.move_to(
            ctx,
            low,
            hold_R,
            label=f"[cracker] lower to {release_dz_m * 100:.1f} cm above cradle {low.tolist()}",
            tol_m=0.01,
            timeout_s=6.0,
        )
        step_gate(
            ctx,
            f"[cracker] release egg (open tongs to {release_width:.4f} m, keep holding tongs)",
        )
        gripper.move(ctx.redis, release_width, speed=float(gripper_speed), force=40.0)
        time.sleep(1.0)
    finally:
        # Restore normal stiffness/speed before retreating.
        gains.restore_precise_grasp(ctx.redis, snap, label="cracker-release")

    # Retreat straight up from the release pose (keep wrist orientation) rather
    # than driving all the way back to home. Loose 5 cm clearance lift.
    _clearance_lift(ctx, label="[cracker] lift after egg release (egg released, tongs held)")
    print("[cracker] complete - egg released into the cracker, tongs still held.")


def _return_tongs(
    ctx: TaskContext,
    pick_pos: np.ndarray,
    pick_ori: np.ndarray,
    *,
    gripper_speed: float,
    lift_dz_m: float = TONGS_RETURN_LIFT_M,
) -> None:
    """Set the tongs back where they were grabbed, open, lift straight, go home.

    Run after the egg has been dropped in the cracker (tongs empty but still
    held). Returns the EE to the original tong pickup pose, opens the gripper to
    full width to release the tongs onto the table, lifts straight up by
    ``lift_dz_m`` to clear them, then returns to home.
    """
    pick_pos = np.asarray(pick_pos, dtype=np.float64).reshape(3)
    pick_ori = np.asarray(pick_ori, dtype=np.float64).reshape(3, 3)
    arm.move_to(
        ctx,
        pick_pos + np.array([0.00, 0.00, 0.10], dtype=np.float64),
        pick_ori,
        label=f"[tongs] return to pickup pose {pick_pos.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )
    arm.move_to(
        ctx,
        pick_pos,
        pick_ori,
        label=f"[tongs] return to pickup pose {pick_pos.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )
    step_gate(ctx, "[tongs] release tongs (open gripper to full width, full speed)")
    open_w = gripper.open_gripper(
        ctx.redis,
        None,
        speed=DEFAULT_GRIPPER_SPEED,
        force=40.0,
        use_max_mode=True,
    )
    print(f"[tongs] opened to {open_w:.4f} m to release the tongs")
    time.sleep(2.0)

    pick_lift = pick_pos + np.array([0.00, 0.00, 0.08], dtype=np.float64)
    arm.move_to(
        ctx,
        pick_lift,
        pick_ori,
        label=f"[tongs] lift straight up {pick_lift.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )

    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[tongs] back to home {ARM_HOME_POSITION.tolist()} (tongs released)",
        tol_m=HOME_POS_TOL_M,
    )
    print("[tongs] complete - tongs returned to pickup spot and released.")


def run_tongs_egg_cycle(
    ctx: TaskContext,
    *,
    skip_base: bool = False,
    retries: int = 1,
    gripper_speed: float = 0.02,
    egg_close_width: float = 0.03,
    egg_force: float = EGG_DEFAULT_FORCE_N,
    tongs_reach_m: float = TONGS_REACH_M,
    tong_yaw_offset_deg: float = TONGS_EE_X_YAW_BIAS_DEG,
    release_into_cracker: bool = True,
    release_width: float = 0.03,
    cracker_above_m: float = 0.14,
    cracker_release_m: float = 0.07,
    gemini_response_path: str | None = None,
) -> None:
    """Grab the tongs, then use them to pick up an egg (ends with egg in tongs).

    1. Arm -> home; base -> INGREDIENT_STATION (unless --skip-base).
    2. From the home view (empty gripper, nothing occluding), Gemini detects
       ALL THREE targets and their poses are saved: the tongs grasp (two red
       tapes), the egg-cracker cradle center, and the egg grasp (recentering
       detection last, since it may nudge the EE to center the egg).
    3. Grab the tongs: ``grasp.object`` MOVE-closes the jaws to 70% width and
       STOPS (no squeeze) so the tong tips stay apart, then LIFTS. No
       return-to-home — the next move goes straight to the egg.
    4. The wrist yaws so the tong tips aim at the egg and the EE backs off
       ``tongs_reach_m`` along that aim. ``grasp.object`` (with
       ``keep_grip=True`` so the tongs aren't dropped) descends and MOVE-closes
       to ``egg_close_width`` (no squeeze), then FORCE-closes (``egg_force``)
       to secure the egg.
    5. Lift the egg straight up ``CLEARANCE_LIFT_M`` to clear the table (no
       home waypoint).
    6. If ``release_into_cracker``: carry the egg straight over the
       PRE-DETECTED cradle center, the wrist yaws so the tong tips reach over
       it, the arm moves a bit above then (precise mode) lowers to
       ``cracker_release_m`` above the cradle and opens the tongs back to
       ``release_width`` so the egg drops in -- the tongs stay held.
    7. Return the tongs to the spot they were picked up from, open the gripper
       to release them, lift straight up to clear them, and go back to home.
    """
    if gemini_response_path is not None:
        ctx.gemini_response_path = gemini_response_path
    if ctx.gemini_response_path is None:
        ctx.gemini_response_path = str(DEFAULT_GEMINI_RESPONSE_PATH)

    gripper.wait_for_ready(ctx.redis)

    # 1. Home + optional base move.
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[tongs] move to home {ARM_HOME_POSITION.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )
    if not skip_base:
        base.go_to_pose(ctx, BaseWaypoint.EGG_CRACK_STATION)

    # 2. Detect ALL THREE targets up front from the home view (empty gripper,
    # nothing occluding): the tongs grasp, the egg grasp, and the egg-cracker
    # cradle center. Save every pose, then run the whole pick sequence off
    # these saved poses with NO return-to-home in between — just lifts.
    base_response_path = ctx.gemini_response_path

    # 2a. Tongs grasp.
    if base_response_path:
        ctx.gemini_response_path = _suffixed_path(base_response_path, "tongs")
    tongs_pose = gemini.find_grasp_pose(ctx, Object.TONGS, retries=retries)
    print(f"[tongs] detected grasp: {tongs_pose.position.tolist()}")

    # 2b. Egg-cracker cradle center (saved to a dedicated filename so a later
    # egg-cracker GRASP detection can't overwrite it).
    cradle_pos: np.ndarray | None = None
    if release_into_cracker:
        if base_response_path:
            ctx.gemini_response_path = _cradle_response_path(base_response_path)
        cradle_pos = np.asarray(
            gemini.find_center(ctx, Object.EGG_CRACKER, retries=retries),
            dtype=np.float64,
        ).reshape(3)
        print(f"[cracker] cradle center detected: {cradle_pos.tolist()}")

    # 2c. Egg grasp. Recentering detection (does this LAST since it may nudge
    # the EE off home to center the egg): orientation held at home so the
    # camera only translates.
    if base_response_path:
        ctx.gemini_response_path = _suffixed_path(base_response_path, "egg")
    egg_pose = gemini.find_grasp_pose_recentering(
        ctx,
        Object.EGG,
        retries=retries,
        hold_orientation=ARM_HOME_ORIENTATION,
    )
    egg_pos = egg_pose.position.astype(np.float64, copy=True)
    print(f"[egg] detected: {egg_pos.tolist()}")
    ctx.gemini_response_path = base_response_path

    # 3. Grab the tongs. ``grasp.object`` MOVE-closes to 70% width and STOPS
    # (no squeeze) so the tong tips stay apart for the egg, then LIFTS — no
    # return-to-home, the next move goes straight toward the egg.
    grasp.object(
        ctx,
        Object.TONGS,
        pick_pos=tongs_pose.position,
        ori=tongs_pose.orientation,
        close_mode="move",
        lift_tol_m=HOME_POS_TOL_M,
    )
    # 5 cm loose clearance lift so the tongs clear the table before swinging
    # over toward the egg.
    _clearance_lift(ctx, label="[tongs] clearance lift after pickup")

    # 4. Aim the tool at the egg, then back off along that aim. The tongs come
    # out of the EE's +X, so we YAW the wrist about world +Z until the EE's +X
    # (flattened to horizontal) points from the home standoff straight at the
    # egg -- instead of leaving the wrist at the fixed home yaw and only sliding
    # the EE sideways. Then the EE grasp target is tongs_reach_m back from the
    # egg along that same horizontal aim, so the tong tips reach forward onto
    # the egg at the egg's own height.
    reach_hat, hold_R, cmd_yaw = _aim_tongs_at(egg_pos, yaw_offset_deg=tong_yaw_offset_deg)
    target = egg_pos - float(tongs_reach_m) * reach_hat
    print(
        f"[egg] tongs-reach grasp target {target.tolist()} "
        f"(egg - {tongs_reach_m * 100:.1f} cm along aim "
        f"[{reach_hat[0]:+.3f}, {reach_hat[1]:+.3f}, {reach_hat[2]:+.3f}], "
        f"wrist yaw command {np.degrees(cmd_yaw):+.1f} deg to face egg)"
    )

    spec = OBJECT_DEFAULTS[Object.EGG]
    spec.speed = float(gripper_speed)
    spec.close_width = float(egg_close_width)
    spec.force = float(egg_force)
    # Stage 1: MOVE-close the tongs to egg_close_width and STOP (no squeeze) —
    # enough to grip the egg and pick it up, but not all the way shut. No lift
    # here (lift_dz_m=0); the egg is lifted after the force-close below.
    # keep_grip=True so the initial open doesn't drop the tongs.
    grasp.object(
        ctx,
        Object.EGG,
        pick_pos=target,
        ori=hold_R,
        lift_dz_m=0.0,
        close_mode="move",
        keep_grip=True,
        lift_tol_m=HOME_POS_TOL_M,
    )
    # Stage 2: now that the egg is held in the tongs, FORCE-close to squeeze the
    # tongs shut around it and secure it for transport.
    step_gate(ctx, f"[tongs+egg] force-close to secure egg (force={egg_force:.1f} N)")
    gripper.grasp(ctx.redis, 0.0, speed=float(gripper_speed), force=float(egg_force))
    time.sleep(max(spec.grasp_settle_s, 2.0))
    held = gripper.wait_for_grasp_result(ctx.redis, timeout_s=5.0, fallback_to_width=True)
    if held is False:
        print("[tongs+egg] WARNING: force-close reported no object held.")

    # 5. Lift the egg straight up to clear the table — no home waypoint, the
    # release step carries it straight over to the cradle. Loose 5 cm clearance.
    _clearance_lift(ctx, label="[tongs+egg] clearance lift after egg pickup")
    print("[tongs+egg] egg secured in the tongs.")

    # 6. Carry the egg over the pre-detected cracker cradle and drop it in
    # (tongs stay held). No return-to-home first — go straight from the egg
    # lift to above the cradle.
    if release_into_cracker:
        prior_response_path = ctx.gemini_response_path
        try:
            if base_response_path:
                ctx.gemini_response_path = _cradle_response_path(base_response_path)
            _release_egg_into_cracker(
                ctx,
                tongs_reach_m=tongs_reach_m,
                gripper_speed=gripper_speed,
                release_width=release_width,
                above_dz_m=cracker_above_m,
                release_dz_m=cracker_release_m,
                tong_yaw_offset_deg=tong_yaw_offset_deg,
                retries=retries,
                cradle=cradle_pos,
            )
        finally:
            ctx.gemini_response_path = prior_response_path

    # 7. Put the tongs back where we found them, release, lift straight, home.
    _return_tongs(
        ctx,
        tongs_pose.position,
        tongs_pose.orientation,
        gripper_speed=gripper_speed,
    )
    print("[tongs+egg] complete.")


def run_debug_cycle(
    ctx: TaskContext,
    *,
    gripper_speed: float = 0.02,
    close_width: float = 0.045,
    gripper_force: float = EGG_DEFAULT_FORCE_N,
) -> None:
    """Gripper-only debug: home → open → force-close with the egg grasp.

    No base move, no Gemini, no descend/lift — just exercises the home pose
    and the egg force-close so you can sanity-check the gripper in isolation.
    """
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[egg:debug] move to home {ARM_HOME_POSITION.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )

    gripper.wait_for_ready(ctx.redis)
    step_gate(ctx, "[egg:debug] open gripper")
    open_w = gripper.open_gripper(ctx.redis, None, speed=gripper_speed, force=gripper_force)
    print(f"[egg:debug] opening to {open_w:.4f} m")
    time.sleep(0.6)
    close_width = 0.0
    step_gate(
        ctx,
        f"[egg:debug] force-close grasp (close-width={close_width:.4f} m, "
        f"force={gripper_force:.1f} N)",
    )
    gripper.grasp(ctx.redis, close_width, speed=gripper_speed, force=gripper_force)
    time.sleep(4)

    held = gripper.wait_for_grasp_result(ctx.redis, timeout_s=5.0, fallback_to_width=False)
    if held is True:
        print("[egg:debug] GRASP SUCCESSFUL - object held.")
    elif held is False:
        print("[egg:debug] GRASP FAILED - gripper closed on nothing.")
    else:
        print("[egg:debug] GRASP RESULT UNKNOWN - no driver result and no width reading.")

    print("[egg:debug] holding grasp. Press Ctrl+C to release and exit.")
    while True:
        time.sleep(0.5)


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key
    ctx.gemini_response_path = args.gemini_response_path

    print(f"Step mode      : {'on' if args.step else 'off'}")
    if args.debug:
        print("Mode           : DEBUG (home -> open -> force-close, no base/Gemini/lift)")
    else:
        print("Mode           : TONGS+EGG (grab tongs -> pick egg with tongs)")
        print(f"Base move      : {'skipped (--skip-base)' if args.skip_base else 'INGREDIENT_STATION only'}")
        print(f"Tongs reach    : {args.tongs_reach_cm:.1f} cm (EE yaw +X, in front of EE)")
        if args.release_into_cracker:
            print(
                f"Cracker drop   : on  (above {args.cracker_above_cm:.1f} cm -> "
                f"release {args.cracker_release_cm:.1f} cm, open to "
                f"{args.release_width * 100:.1f} cm)"
            )
        else:
            print("Cracker drop   : off (--no-release-into-cracker)")
    print(
        f"Gripper        : force={args.gripper_force:.1f} N  "
        f"close-width={args.close_width * 100:.1f} cm  "
        f"speed={args.gripper_speed:.3f} m/s  (GRASP-mode force close)"
    )
    if not args.debug:
        print(f"Gemini response: {args.gemini_response_path}")

    try:
        if args.debug:
            run_debug_cycle(
                ctx,
                gripper_speed=args.gripper_speed,
                close_width=args.close_width,
                gripper_force=args.gripper_force,
            )
        else:
            run_tongs_egg_cycle(
                ctx,
                skip_base=args.skip_base,
                retries=args.retries,
                gripper_speed=args.gripper_speed,
                egg_close_width=args.close_width,
                egg_force=args.gripper_force,
                tongs_reach_m=args.tongs_reach_cm / 100.0,
                tong_yaw_offset_deg=args.tong_yaw_offset_deg,
                release_into_cracker=args.release_into_cracker,
                release_width=args.release_width,
                cracker_above_m=args.cracker_above_cm / 100.0,
                cracker_release_m=args.cracker_release_cm / 100.0,
                gemini_response_path=args.gemini_response_path,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
