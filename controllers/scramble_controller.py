#!/usr/bin/env python3
"""Egg scramble at the stove: drive to stove, detect pan, scramble.

Assumes liquid egg mixture is already in the pan on the burner and the
ladle is reachable at the stove. Reuses the Gemini ladle grasp from
``mixing_vision_base_controller``.

Sequence:

  1. Arm → home.
  2. Base → ``STOVE_STATION`` (skip with ``--skip-base``).
  3. (Default, ``--move-pan``) Move to a dedicated overhead pan-grasp look-at
     pose (home + 10 cm in Z, camera straight down). Gemini detects the pan
     handle as TWO points along its axis (grasp PERPENDICULAR to that axis, tool
     straight down) AND the back-burner white-cross center (rightmost + uppermost
     cross). The arm grabs the pan and slides it in X by (burner_X - pan_grasp_X)
     so it lands on the back burner. With ``--no-detect-back-burner`` it slides a
     fixed ``--pan-shift-back-m`` instead. Disable relocation with
     ``--no-move-pan``.
  4. Move to the shared look-at pose (``DETECTION_EE_POSITION``).
  5. From the look-at pose, Gemini detects the **center + radius of the pan**
     (at its post-move position) and then the **ladle** grasp. The scramble
     work plane is ``--tool-length-m`` above the detected pan center.
  6. Gemini ladle grasp → ``grasp.object``, then a single diagonal move that
     lifts to the carry height (``--tool-length-m`` + ``--carry-lift-offset-m``
     above the pan center) while pulling back ``CARRY_BACK_X_M`` in -X, then
     flips the ladle in place.
  7. Hover above the detected pan center, descend to the work plane,
     run cardinal triangles + circular mix, lift back to hover.
  8. Carry the ladle back to the exact spot it was grasped, release it
     there, and lift clear.
  9. (Default, ``--move-pan``) Grab the pan again and slide it forward
     ``--pan-shift-back-m`` in -X to the front burner. Optionally return
     home (``--return-home``).

Re-record ``DETECTION_EE_POSITION`` (the shared look-at framing pose used
for both detections) at ``STOVE_STATION`` if the stove / pan moves: jog
there and log ``T_end_effector`` from Redis.

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/scramble_controller.py -- --step

  # Skip the drive when the cart is already parked at the stove:
  ./ZitiBot/launch_zitibot_full.sh controllers/scramble_controller.py -- \\
      --skip-base --step

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
RealSense, OptiTrack on Redis, and ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm, base, gains, gripper
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    BASE_WAYPOINTS,
    DEFAULT_POS_TOL_M,
    HOME_POS_TOL_M,
    OBJECT_DEFAULTS,
    PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
    PRECISE_GRASP_MAX_LINEAR_VELOCITY,
    PRECISE_GRASP_MOVE_TIMEOUT_S,
    PRECISE_GRASP_ORIENTATION_KP,
    PRECISE_GRASP_POSITION_KP,
    BaseWaypoint,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_core.runner import step_gate
from zitibot_tasks import gemini, grasp, mix

DEFAULT_GEMINI_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_ladle.png"
)
DEFAULT_PAN_GEMINI_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_pan_center.png"
)
DEFAULT_PAN_GRASP_GEMINI_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_pan_grasp.png"
)
DEFAULT_BACK_BURNER_GEMINI_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_back_burner.png"
)

# Base waypoint the cart drives to before detecting the pan and scrambling.
DEFAULT_STOVE_WAYPOINT = BaseWaypoint.STOVE_STATION

# Shared "look-at" camera framing pose for BOTH the pan-center and ladle
# Gemini detections: home pose shifted +15 cm in X and -10 cm in Y (5 cm
# further to the left/+Y than before). The arm moves here once (at
# ARM_HOME_ORIENTATION, camera looking down) and both detections are captured
# from this single pose.
DETECTION_EE_POSITION = ARM_HOME_POSITION + np.array(
    [0.15, -0.10, 0.0], dtype=np.float64
)

# Effective tool length (m): flange (control point) to ladle/paddle tip. The
# scramble work plane sits this far above the detected pan-floor center so the
# tip reaches the cooking surface; the triangles + circles run at that Z. The
# hover the ladle descends from sits ``CARRY_LIFT_OFFSET_M`` higher.
TOOL_LENGTH_M = 0.23

# Extra depth (m) the ladle is pushed BELOW the nominal work plane during the
# scramble so the paddle tip presses into the cooking surface a bit harder.
SCRAMBLE_EXTRA_DEPTH_M = 0.02

# Carry/rotation hover offset above the work plane (m). After the grasp the
# ladle is lifted to ``TOOL_LENGTH_M + CARRY_LIFT_OFFSET_M`` above the pan
# center: the in-place rotation and the move over the pan both happen at that
# height, and the ladle then descends this offset to the work plane.
CARRY_LIFT_OFFSET_M = 0.10

# Straight after the grasp the ladle is lifted to the carry height AND pulled
# back this far in world -X in a single diagonal move (clearing the rack on the
# way up and out) before it is flipped in place.
CARRY_BACK_X_M = 0.30
# The lift+pullback+flip happens this much LOWER than the over-pan carry/hover
# height — i.e. the ladle isn't raised as high before flipping. Only the flip
# pose is lowered; the over-pan hover ("above") and in-pan work plane are
# unchanged (the ladle just rises this much again on the way to the pan).
CARRY_FLIP_LOWER_M = 0.10

# Pan relocation. Before scrambling, the pan is grabbed by its handle and slid
# this far in world +X (back, toward the back burner); after scrambling it is
# slid the same amount in -X to bring it forward again. The scramble pan center
# shifts with it. Tune with --pan-shift-back-m / flip the sign convention there
# if "back" is -X on your bench.
PAN_SHIFT_BACK_M = 0.30
# Height (m) the pan is lifted off the burner before sliding it across, and
# lowered back down by afterward (the descent runs in precise mode).
PAN_MOVE_LIFT_M = 0.10
# The pan handle is detected from a dedicated overhead look-at pose: the arm
# home position raised this far in world +Z (camera looking straight down). This
# gives a clean top-down view of the handle so the two-point handle axis (and
# thus the perpendicular grasp) is read reliably.
PAN_GRASP_LOOK_AT_DZ_M = 0.05

# After the Gemini ladle grasp the held ladle is rotated by this much to
# reach the stirring orientation, and ALL scramble motions are run from
# there. Default: -90 deg about world Y, premultiplied onto the grasp
# orientation (same world-frame convention as
# ``mixing_vision_base_controller``'s transit rotation). Switch
# ``LADLE_GRASP_ROT_FRAME`` to "tool" to rotate about the held ladle's own
# axis instead (postmultiply).
LADLE_GRASP_ROT_DEG = -90.0
LADLE_GRASP_ROT_AXIS = "y"
LADLE_GRASP_ROT_FRAME = "world"
# After scrambling the ladle is carried back to the exact spot it was grasped
# and released there (mirrors egg_crack_controller). Approach from this far
# above the pick, lift this far straight up after releasing, and use a generous
# convergence budget on the return so the gripper doesn't open mid-transit.
SCRAMBLE_RELEASE_ABOVE_M = 0.08
SCRAMBLE_RELEASE_LIFT_M = 0.10
SCRAMBLE_RELEASE_RETURN_TIMEOUT_S = 8.0
SCRAMBLE_RELEASE_SETTLE_S = 0.5

# The stir itself runs in PRECISE mode (stiff cartesian gains so the ladle
# tracks the waypoints accurately) at a moderate speed: faster than the slow
# 0.03 m/s precise-grasp approach cap, but slower than the full 0.13 m/s OTG
# default so the scramble stays controlled.
SCRAMBLE_PRECISE_MAX_LINEAR_VELOCITY = 0.07
SCRAMBLE_PRECISE_MAX_ANGULAR_VELOCITY = 0.4

DEFAULT_MIX_RADIUS_M = 0.05
DEFAULT_MIX_CYCLES = 6
DEFAULT_MIX_CYCLE_DURATION_S = 20.0
DEFAULT_TRIANGLE_PASSES = 1
DEFAULT_TRIANGLE_DURATION_S = 4.0
# Total scramble time budget (s): the four-triangle set is repeated until this
# many seconds have elapsed. When set this overrides the fixed triangle-pass
# count.
DEFAULT_SCRAMBLE_DURATION_S = 15.0


def _detect_ladle(ctx: TaskContext) -> gemini.GraspPose:
    """Query Gemini for the ladle grasp pose (arm already at the look-at pose)."""
    step_gate(ctx, "[ladle detection] ready to query Gemini — press ENTER to capture")
    print("[ladle detection] querying Gemini for ladle handle pose...")
    pose = gemini.find_grasp_pose(ctx, Object.LADLE)
    print(f"[ladle detection] detected position: {pose.position.tolist()}")
    if pose.rim_yaw_applied and pose.rim_yaw_deg is not None:
        print(f"[ladle detection] handle yaw applied: {pose.rim_yaw_deg:.1f} deg")
    return pose


def _detect_pan_center(
    ctx: TaskContext,
    *,
    gemini_response_path: str | Path | None = None,
) -> tuple[np.ndarray, float | None]:
    """Query Gemini for the pan-floor center + rim (arm at the look-at pose).

    Returns ``(center_world, radius_m)``. Gemini reports both the pan-floor
    center and a point on the rim; the radius is the horizontal distance
    between them. The ``(PAN, "center_rim")`` detection carries a zero world
    offset, so the center is raw; the caller adds the work-height offset above
    it. ``radius_m`` is ``None`` when the rim point had no usable depth.
    """
    step_gate(ctx, "[pan detection] ready to query Gemini — press ENTER to capture")
    print("[pan detection] querying Gemini for pan center + rim...")
    prior_path = ctx.gemini_response_path
    try:
        if gemini_response_path is not None:
            ctx.gemini_response_path = str(gemini_response_path)
        center, radius_m = gemini.find_pan_center_radius(
            ctx, Object.PAN, kind="center_rim"
        )
    finally:
        ctx.gemini_response_path = prior_path
    if radius_m is not None:
        print(
            f"[pan detection] pan-floor center world: {center.tolist()} "
            f"radius: {radius_m * 100:.1f} cm"
        )
    else:
        print(
            f"[pan detection] pan-floor center world: {center.tolist()} "
            f"(radius unavailable)"
        )
    return center, radius_m


def _detect_pan_grasp(
    ctx: TaskContext,
    *,
    gemini_response_path: str | Path | None = None,
) -> gemini.GraspPose:
    """Query Gemini for the pan-handle grasp pose (arm at the overhead pan-grasp
    look-at pose). Gemini returns two points along the handle axis; the pose
    builder grasps perpendicular to that axis, tool pointing straight down."""
    step_gate(ctx, "[pan grasp] ready to query Gemini — press ENTER to capture")
    print("[pan grasp] querying Gemini for pan handle grasp pose...")
    prior_path = ctx.gemini_response_path
    try:
        if gemini_response_path is not None:
            ctx.gemini_response_path = str(gemini_response_path)
        pose = gemini.find_grasp_pose(ctx, Object.PAN)
    finally:
        ctx.gemini_response_path = prior_path
    print(f"[pan grasp] detected handle grasp: {pose.position.tolist()}")
    if pose.rim_yaw_applied and pose.rim_yaw_deg is not None:
        print(f"[pan grasp] handle yaw applied: {pose.rim_yaw_deg:.1f} deg")
    return pose


def _detect_back_burner(
    ctx: TaskContext,
    *,
    gemini_response_path: str | Path | None = None,
) -> np.ndarray:
    """Query Gemini for the back-burner white-cross center (arm at the overhead
    pan-grasp look-at pose). Returns the burner cross center in world frame;
    only the world X is used to align the pan with the burner."""
    step_gate(
        ctx, "[back burner] ready to query Gemini — press ENTER to capture"
    )
    print("[back burner] querying Gemini for back-burner white-cross center...")
    prior_path = ctx.gemini_response_path
    try:
        if gemini_response_path is not None:
            ctx.gemini_response_path = str(gemini_response_path)
        burner_world = gemini.find_back_burner(ctx)
    finally:
        ctx.gemini_response_path = prior_path
    print(f"[back burner] white-cross center world: {burner_world.tolist()}")
    return burner_world


def _move_pan(
    ctx: TaskContext,
    *,
    pick_pos: np.ndarray,
    ori: np.ndarray,
    dx_m: float,
    lift_m: float,
    label: str,
) -> np.ndarray:
    """Grab the pan by its handle, slide it ``dx_m`` in world X, set it down.

    Grasps at ``pick_pos`` (handle) with orientation ``ori``, lifts ``lift_m``
    off the burner, translates ``dx_m`` along world X at that height, lowers
    back down ``lift_m`` at the shifted spot (in precise mode for a controlled
    set-down), opens the gripper to release, and lifts clear. Returns the new
    handle world position (``pick_pos`` shifted by ``dx_m`` in X) so the reverse
    move can grab it again.
    """
    spec = OBJECT_DEFAULTS[Object.PAN]
    pick = np.asarray(pick_pos, dtype=np.float64).reshape(3).copy()
    grip_R = np.asarray(ori, dtype=np.float64).reshape(3, 3)

    # 1. Grab the handle and lift ``lift_m`` off the burner in one motion.
    grasp.object(
        ctx,
        Object.PAN,
        pick_pos=pick,
        ori=grip_R,
        lift_dz_m=float(lift_m),
        lift_tol_m=0.05,
    )

    # 2. Translate to the shifted spot at the lifted height.
    carry = pick + np.array([float(dx_m), 0.0, float(lift_m)], dtype=np.float64)
    arm.move_to(
        ctx,
        carry,
        grip_R,
        label=f"{label}: slide {dx_m * 100:+.0f} cm X to {carry.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 3. Lower the pan back down ``lift_m`` onto the burner at the shifted spot,
    #    in precise mode (stiff cart gains + slow OTG cap) for a controlled
    #    set-down.
    set_down = pick + np.array([float(dx_m), 0.0, 0.0], dtype=np.float64)
    set_down_precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label=f"{label} set-down",
    )
    try:
        arm.move_to(
            ctx,
            set_down,
            grip_R,
            label=(
                f"{label}: lower pan {lift_m * 100:.0f} cm to {set_down.tolist()} "
                f"(precise)"
            ),
            tol_m=spec.grasp_tol,
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
        )
    finally:
        gains.restore_precise_grasp(
            ctx.redis, set_down_precise, label=f"{label} set-down"
        )

    # 4. Release the handle.
    step_gate(ctx, f"{label}: release pan handle (open gripper)")
    gripper.open_gripper(
        ctx.redis,
        spec.open_width,
        speed=spec.speed,
        force=spec.force,
        use_max_mode=True,
    )
    time.sleep(spec.grasp_settle_s)

    # 5. Lift the gripper clear of the handle.
    clear = set_down + np.array([0.0, 0.0, float(lift_m)], dtype=np.float64)
    arm.move_to(
        ctx,
        clear,
        grip_R,
        label=f"{label}: lift clear of pan {clear.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    print(f"{label}: done — pan now at handle {set_down.tolist()}")
    return set_down


def _parse_orientation_json(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return np.asarray(data, dtype=np.float64).reshape(3, 3)


def _rotate_orientation(
    ori: np.ndarray, axis: str, deg: float, frame: str
) -> np.ndarray:
    """Rotate ``ori`` by ``deg`` about ``axis``.

    ``frame="world"`` premultiplies (rotation about a fixed world axis,
    same convention as ``mixing_vision_base_controller``); ``frame="tool"``
    postmultiplies (rotation about the held tool's own axis).
    """
    R_rot = R_scipy.from_euler(axis, deg, degrees=True).as_matrix()
    ori = np.asarray(ori, dtype=np.float64).reshape(3, 3)
    if frame == "tool":
        return ori @ R_rot
    return R_rot @ ori


def _resolve_mix_radius(
    *,
    fixed_radius_m: float | None,
    pan_radius_m: float | None,
) -> float:
    """Return the scramble radius.

    A ``--mix-radius-m`` override wins; otherwise use the RAW detected pan
    radius unscaled and unclamped (scramble all the way out to the rim). Falls
    back to ``DEFAULT_MIX_RADIUS_M`` only when the pan radius is unavailable.
    """
    if fixed_radius_m is not None:
        return float(fixed_radius_m)
    if pan_radius_m is None:
        return DEFAULT_MIX_RADIUS_M
    return float(pan_radius_m)


def _drive_to_stove(
    ctx: TaskContext,
    *,
    x_override: float | None,
    y_override: float | None,
    yaw_override: float | None,
) -> None:
    """Drive the base to STOVE_STATION (or an overridden pose)."""
    overridden = any(
        v is not None for v in (x_override, y_override, yaw_override)
    )
    if not overridden:
        base.go_to_pose(ctx, DEFAULT_STOVE_WAYPOINT)
        return
    wp = BASE_WAYPOINTS[DEFAULT_STOVE_WAYPOINT]
    x_m = wp.x_m if x_override is None else float(x_override)
    y_m = wp.y_m if y_override is None else float(y_override)
    yaw_deg = wp.yaw_deg if yaw_override is None else float(yaw_override)
    base.go_to_pose(
        ctx,
        x_m=x_m,
        y_m=y_m,
        yaw_deg=yaw_deg,
        label=(
            f"[base] STOVE_STATION (override) -> "
            f"({x_m:.3f}, {y_m:.3f}, {yaw_deg:.1f} deg)"
        ),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stove scramble: drive to stove, Gemini pan center/radius detect, "
            "scramble."
        )
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate each motion / gripper / base step.",
    )

    p.add_argument(
        "--endeffector-transform-key",
        default="opensai::redis_driver::FrankaRobot::T_end_effector",
    )
    p.add_argument(
        "--gemini-response-path",
        type=Path,
        default=DEFAULT_GEMINI_RESPONSE_PATH,
        help="Path to save Gemini debug image (ladle grasp).",
    )
    p.add_argument(
        "--pan-gemini-response-path",
        type=Path,
        default=DEFAULT_PAN_GEMINI_RESPONSE_PATH,
        help="Path to save Gemini debug image (pan-center detection).",
    )

    # --- Base navigation ---
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip the drive to STOVE_STATION (assume already parked there).",
    )
    p.add_argument("--stove-x", type=float, default=None,
                   help="Override STOVE_STATION X (m).")
    p.add_argument("--stove-y", type=float, default=None,
                   help="Override STOVE_STATION Y (m).")
    p.add_argument("--stove-yaw-deg", type=float, default=None,
                   help="Override STOVE_STATION yaw (deg).")

    # --- Pan-center detection ---
    p.add_argument(
        "--tool-length-m",
        type=float,
        default=TOOL_LENGTH_M,
        help=(
            "Effective tool length (m): the scramble work plane sits this far "
            "above the detected pan-floor center. Default 23 cm."
        ),
    )

    # --- Ladle grasp ---
    p.add_argument(
        "--detection-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Shared look-at EE world position for BOTH pan-center and ladle "
            f"Gemini framing (m). Default: {DETECTION_EE_POSITION.tolist()}"
        ),
    )

    p.add_argument(
        "--carry-lift-offset-m",
        type=float,
        default=CARRY_LIFT_OFFSET_M,
        help=(
            "Carry/rotation hover offset (m) above the work plane. The ladle "
            "is lifted to tool-length + this offset above the pan center, "
            "rotated and carried there, then descends this offset to the work "
            "plane. Default 6 cm."
        ),
    )

    p.add_argument(
        "--scramble-ori-json",
        type=str,
        default=None,
        help=(
            "3x3 orientation matrix JSON held through the scramble. Fixed "
            "override — bypasses the grasp-derived rotation below."
        ),
    )
    p.add_argument(
        "--grasp-rot-deg",
        type=float,
        default=LADLE_GRASP_ROT_DEG,
        help=(
            "Rotation (deg) applied to the grasp orientation to get the "
            "stirring orientation. Default -90."
        ),
    )
    p.add_argument(
        "--grasp-rot-axis",
        choices=("x", "y", "z"),
        default=LADLE_GRASP_ROT_AXIS,
        help="Axis for --grasp-rot-deg. Default y.",
    )
    p.add_argument(
        "--grasp-rot-frame",
        choices=("world", "tool"),
        default=LADLE_GRASP_ROT_FRAME,
        help=(
            "Frame for --grasp-rot-deg: 'world' (fixed axis, premultiply) or "
            "'tool' (ladle's own axis, postmultiply). Default world."
        ),
    )

    p.add_argument(
        "--mix-radius-m",
        type=float,
        default=None,
        help=(
            "Fixed circular mix radius about the pan center (m). If omitted, "
            "use the raw detected pan radius (unscaled, unclamped); fallback "
            "5 cm when the pan radius is unavailable."
        ),
    )
    p.add_argument(
        "--mix-cycles",
        type=int,
        default=DEFAULT_MIX_CYCLES,
        help="Number of circular mix revolutions. Default 6.",
    )
    p.add_argument(
        "--mix-cycle-duration-s",
        type=float,
        default=DEFAULT_MIX_CYCLE_DURATION_S,
        help="Total duration of mix revolutions (s).",
    )
    p.add_argument(
        "--triangle-passes",
        type=int,
        default=DEFAULT_TRIANGLE_PASSES,
        help="How many full top/bottom/right/left triangle sets (0 to skip).",
    )
    p.add_argument(
        "--triangle-duration-s",
        type=float,
        default=DEFAULT_TRIANGLE_DURATION_S,
        help="Duration of each cardinal triangle trace (s).",
    )
    p.add_argument(
        "--scramble-duration-s",
        type=float,
        default=DEFAULT_SCRAMBLE_DURATION_S,
        help=(
            "Total scramble time budget (s): repeat the four-triangle set until "
            "this elapses. Set <= 0 to use --triangle-passes instead. "
            "Only used when --no-until-enter is passed. "
            f"Default {DEFAULT_SCRAMBLE_DURATION_S:.0f} s."
        ),
    )

    p.add_argument(
        "--until-enter",
        dest="scramble_until_enter",
        action="store_true",
        default=True,
        help=(
            "Repeat the four-triangle scramble set until the user presses ENTER "
            "(the in-progress pass finishes first). On by default; takes "
            "precedence over --scramble-duration-s/--triangle-passes."
        ),
    )
    p.add_argument(
        "--no-until-enter",
        dest="scramble_until_enter",
        action="store_false",
        help=(
            "Disable ENTER-to-stop; use --scramble-duration-s/--triangle-passes "
            "to bound the scramble instead."
        ),
    )

    p.add_argument(
        "--return-home",
        dest="return_home",
        action="store_true",
        default=True,
        help=(
            "After the pan is back at the front burner, move the arm to the home "
            "pose (DEFAULT)."
        ),
    )
    p.add_argument(
        "--no-return-home",
        dest="return_home",
        action="store_false",
        help="Leave the arm where it ends instead of returning to home.",
    )

    # --- Pan relocation ---
    p.add_argument(
        "--move-pan",
        dest="move_pan",
        action="store_true",
        default=True,
        help=(
            "Slide the pan to the back burner before scrambling and forward "
            "again after (DEFAULT)."
        ),
    )
    p.add_argument(
        "--no-move-pan",
        dest="move_pan",
        action="store_false",
        help="Do not relocate the pan; scramble it where it is detected.",
    )
    p.add_argument(
        "--detect-back-burner",
        dest="detect_back_burner",
        action="store_true",
        default=True,
        help=(
            "Detect the back-burner white cross and shift the pan in X by the "
            "burner-X minus pan-grasp-X difference (DEFAULT)."
        ),
    )
    p.add_argument(
        "--no-detect-back-burner",
        dest="detect_back_burner",
        action="store_false",
        help=(
            "Do not detect the back burner; use the fixed --pan-shift-back-m "
            "shift instead."
        ),
    )
    p.add_argument(
        "--pan-shift-back-m",
        type=float,
        default=PAN_SHIFT_BACK_M,
        help=(
            "Fixed distance (m) to slide the pan back before scrambling and "
            "forward after, used only with --no-detect-back-burner. Default 0.30."
        ),
    )
    p.add_argument(
        "--pan-move-lift-m",
        type=float,
        default=PAN_MOVE_LIFT_M,
        help=(
            "Height (m) the pan is lifted off the burner before sliding and "
            "lowered back down by after (precise descent). Default 0.10."
        ),
    )
    p.add_argument(
        "--pan-grasp-gemini-response-path",
        type=Path,
        default=DEFAULT_PAN_GRASP_GEMINI_RESPONSE_PATH,
        help="Path to save Gemini debug image (pan-handle grasp detection).",
    )
    p.add_argument(
        "--back-burner-gemini-response-path",
        type=Path,
        default=DEFAULT_BACK_BURNER_GEMINI_RESPONSE_PATH,
        help="Path to save Gemini debug image (back-burner white-cross detection).",
    )
    return p.parse_args()


def run_scramble_cycle(
    ctx: TaskContext,
    *,
    skip_base: bool = False,
    stove_x: float | None = None,
    stove_y: float | None = None,
    stove_yaw_deg: float | None = None,
    tool_length_m: float = TOOL_LENGTH_M,
    detection_pos: np.ndarray | None = None,
    carry_lift_offset_m: float = CARRY_LIFT_OFFSET_M,
    scramble_ori_override: np.ndarray | None = None,
    grasp_rot_deg: float = LADLE_GRASP_ROT_DEG,
    grasp_rot_axis: str = LADLE_GRASP_ROT_AXIS,
    grasp_rot_frame: str = LADLE_GRASP_ROT_FRAME,
    mix_radius_m: float | None = None,
    mix_cycles: int = DEFAULT_MIX_CYCLES,
    mix_cycle_duration_s: float = DEFAULT_MIX_CYCLE_DURATION_S,
    triangle_passes: int = DEFAULT_TRIANGLE_PASSES,
    triangle_duration_s: float = DEFAULT_TRIANGLE_DURATION_S,
    scramble_duration_s: float | None = DEFAULT_SCRAMBLE_DURATION_S,
    scramble_until_enter: bool = True,
    return_home: bool = True,
    move_pan: bool = True,
    pan_shift_back_m: float = PAN_SHIFT_BACK_M,
    pan_move_lift_m: float = PAN_MOVE_LIFT_M,
    detect_back_burner: bool = True,
    gemini_response_path: str | Path | None = None,
    pan_gemini_response_path: str | Path | None = None,
    pan_grasp_gemini_response_path: str | Path | None = None,
    back_burner_gemini_response_path: str | Path | None = None,
) -> None:
    """Drive to stove, detect pan center, Gemini ladle grasp, rotate ladle, scramble.

    After the grasp the held ladle is rotated by ``grasp_rot_deg`` about
    ``grasp_rot_axis`` (in ``grasp_rot_frame``) and EVERY scramble motion is
    run from that orientation. Pass ``scramble_ori_override`` to use a fixed
    orientation instead of the grasp-derived one.
    """
    if gemini_response_path is not None:
        ctx.gemini_response_path = Path(gemini_response_path)
    elif ctx.gemini_response_path is None:
        ctx.gemini_response_path = Path(DEFAULT_GEMINI_RESPONSE_PATH)
    if pan_gemini_response_path is None:
        pan_gemini_response_path = DEFAULT_PAN_GEMINI_RESPONSE_PATH
    if pan_grasp_gemini_response_path is None:
        pan_grasp_gemini_response_path = DEFAULT_PAN_GRASP_GEMINI_RESPONSE_PATH
    if back_burner_gemini_response_path is None:
        back_burner_gemini_response_path = DEFAULT_BACK_BURNER_GEMINI_RESPONSE_PATH

    if detection_pos is None:
        detection_pos = DETECTION_EE_POSITION.copy()
    else:
        detection_pos = np.asarray(detection_pos, dtype=np.float64).reshape(3).copy()

    if scramble_ori_override is not None:
        scramble_ori_override = np.asarray(
            scramble_ori_override, dtype=np.float64
        ).reshape(3, 3).copy()

    spec_ladle = OBJECT_DEFAULTS[Object.LADLE]

    print(
        f"Base move     : "
        f"{'none (--skip-base)' if skip_base else 'drive to STOVE_STATION'}"
    )
    print(
        f"Carry lift    : work plane + {carry_lift_offset_m * 100:.1f} cm "
        f"(rotation + over-pan hover height)"
    )
    if scramble_ori_override is not None:
        print("Scramble ori  : fixed override (--scramble-ori-json)")
    else:
        print(
            f"Scramble ori  : grasp rotated {grasp_rot_deg:+.1f} deg about "
            f"{grasp_rot_frame} {grasp_rot_axis.upper()}"
        )
    print(
        f"Pan detect    : Gemini pan-center, work plane "
        f"{tool_length_m * 100:.1f} cm above center"
    )
    if move_pan:
        if detect_back_burner:
            shift_desc = "align to detected back-burner X"
        else:
            shift_desc = f"fixed {pan_shift_back_m * 100:.0f} cm X"
        print(
            f"Pan relocate  : to back burner ({shift_desc}) before scramble, "
            f"reverse after (lift {pan_move_lift_m * 100:.0f} cm)"
        )
    else:
        print("Pan relocate  : off (--no-move-pan)")
    if scramble_until_enter:
        print(
            "Triangles     : repeat until ENTER "
            "(top/bottom/right/left), circles off"
        )
    elif scramble_duration_s and scramble_duration_s > 0:
        print(
            f"Triangles     : repeat for {scramble_duration_s:.0f} s "
            f"(top/bottom/right/left), circles off"
        )
    else:
        print(
            f"Triangles     : {triangle_passes} pass(es) "
            f"(top/bottom/right/left), circles off"
        )
    print(
        f"Mix           : "
        f"radius={'auto' if mix_radius_m is None else f'{mix_radius_m:.3f} m'}, "
        f"revolutions={mix_cycles}, "
        f"duration={mix_cycle_duration_s:.1f} s"
    )

    # 1. Arm → home.
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[scramble] move to home {ARM_HOME_POSITION.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )

    # 2. Base → stove.
    if not skip_base:
        _drive_to_stove(
            ctx,
            x_override=stove_x,
            y_override=stove_y,
            yaw_override=stove_yaw_deg,
        )

    # 3a. Move the pan to the back burner FIRST (if relocating). The pan handle
    #     is detected from a dedicated overhead look-at pose (home + 10 cm in Z,
    #     camera straight down) so the two-point handle axis reads cleanly and
    #     the grasp is taken perpendicular to the handle. From the SAME pose we
    #     also detect the back-burner white-cross center and move the pan in X by
    #     the difference between the burner X and the pan-grasp X, so the pan
    #     lands on the back burner. (Falls back to a fixed +pan_shift_back_m if
    #     burner detection is disabled.) Detecting the pan center + ladle AFTER
    #     the move means the center is read at the pan's true post-move position
    #     (no manual shift). ``pan_forward_handle`` records where the handle now
    #     is and ``pan_shift_dx`` the applied X shift so the post-scramble move
    #     can grab it and slide it back by the same amount.
    pan_forward_handle = None
    pan_grasp_pose = None
    pan_shift_dx = 0.0
    if move_pan:
        pan_grasp_look_at = ARM_HOME_POSITION + np.array(
            [0.0, 0.0, float(PAN_GRASP_LOOK_AT_DZ_M)], dtype=np.float64
        )
        arm.move_to(
            ctx,
            pan_grasp_look_at,
            ARM_HOME_ORIENTATION,
            label=(
                f"[pan grasp] overhead look-at pose "
                f"{pan_grasp_look_at.tolist()} (home + {PAN_GRASP_LOOK_AT_DZ_M * 100:.0f} cm Z)"
            ),
            tol_m=HOME_POS_TOL_M,
        )
        pan_grasp_pose = _detect_pan_grasp(
            ctx,
            gemini_response_path=pan_grasp_gemini_response_path,
        )

        # Decide the X shift: align the pan with the back burner if requested,
        # else fall back to the fixed shift.
        if detect_back_burner:
            burner_world = _detect_back_burner(
                ctx,
                gemini_response_path=back_burner_gemini_response_path,
            )
            pan_shift_dx = float(burner_world[0] - pan_grasp_pose.position[0])
            print(
                f"[pan] back-burner X={burner_world[0]:+.3f} m, "
                f"pan-grasp X={pan_grasp_pose.position[0]:+.3f} m -> "
                f"shift {pan_shift_dx * 100:+.1f} cm in X"
            )
        else:
            pan_shift_dx = +float(pan_shift_back_m)
            print(
                f"[pan] back-burner detection off; using fixed shift "
                f"{pan_shift_dx * 100:+.1f} cm in X"
            )

        pan_forward_handle = _move_pan(
            ctx,
            pick_pos=pan_grasp_pose.position,
            ori=pan_grasp_pose.orientation,
            dx_m=pan_shift_dx,
            lift_m=float(pan_move_lift_m),
            label="[pan] move to back burner",
        )

    # 3b. Move to the shared look-at pose and detect the pan center + ladle grasp
    #     from there (gripper empty, clean view). With the pan already relocated,
    #     the detected center IS its scramble position.
    # Give this move a generous timeout and a tight velocity gate so the arm is
    # fully STOPPED before we capture the camera frame — a moving frame gives
    # Gemini a motion-blurred / wrong-viewpoint image and bad pan points.
    arm.move_to(
        ctx,
        detection_pos,
        ARM_HOME_ORIENTATION,
        label=f"[detection] look-at pose {detection_pos.tolist()}",
        tol_m=HOME_POS_TOL_M,
        timeout_s=10.0,
        vel_tol_rad_s=0.02,
        settle_ticks=8,
    )
    # Extra dwell so the camera image is steady before the Gemini capture.
    time.sleep(0.5)
    pan_center, pan_radius_m = _detect_pan_center(
        ctx,
        gemini_response_path=pan_gemini_response_path,
    )
    pose = _detect_ladle(ctx)
    grasp_ori = np.asarray(pose.orientation, dtype=np.float64).reshape(3, 3)
    pick_pos = np.asarray(pose.position, dtype=np.float64).reshape(3).copy()

    # 3c. Scramble work plane ``tool_length_m`` above the detected pan
    #     center (so the paddle tip reaches the cooking surface). The carry/hover
    #     height sits ``carry_lift_offset_m`` above that: the ladle is lifted
    #     there, rotated in place, carried over the pan, then descends that
    #     offset to the work plane.
    # Drop the work plane SCRAMBLE_EXTRA_DEPTH_M below the nominal
    # tool-length height so the ladle digs ~1 cm deeper while scrambling.
    work_plane = pan_center + np.array(
        [0.0, 0.0, float(tool_length_m) - SCRAMBLE_EXTRA_DEPTH_M],
        dtype=np.float64,
    )
    descent_dz = float(carry_lift_offset_m) + SCRAMBLE_EXTRA_DEPTH_M
    carry_z = float(work_plane[2]) + descent_dz
    hover_pos = np.array(
        [work_plane[0], work_plane[1], carry_z], dtype=np.float64
    )
    radius_note = (
        f"; pan radius {pan_radius_m * 100:.1f} cm"
        if pan_radius_m is not None
        else "; pan radius unavailable"
    )
    print(
        f"[scramble] scramble work plane {work_plane.tolist()} "
        f"({tool_length_m * 100:.1f} cm above pan center){radius_note}; "
        f"carry height z={carry_z:.3f} m (work plane + {descent_dz * 100:.1f} cm)"
    )

    effective_mix_radius_m = _resolve_mix_radius(
        fixed_radius_m=mix_radius_m,
        pan_radius_m=pan_radius_m,
    )
    if mix_radius_m is None and pan_radius_m is not None:
        print(
            f"[scramble] mix radius = raw detected pan radius "
            f"{effective_mix_radius_m * 100:.1f} cm"
        )
    elif mix_radius_m is None:
        print(
            f"[scramble] pan radius unavailable; using fallback mix "
            f"radius {effective_mix_radius_m * 100:.1f} cm"
        )
    else:
        print(
            f"[scramble] using fixed mix radius override "
            f"{effective_mix_radius_m * 100:.1f} cm"
        )

    # 4. Grasp the ladle (pose detected above). The diagonal carry move
    #    (step 5b) then lifts to the carry height while pulling back in -X.
    grasp.object(
        ctx,
        Object.LADLE,
        pick_pos=pick_pos,
        ori=grasp_ori,
    )

    # 5. Resolve the scramble orientation. Unless an explicit override was
    #    given, rotate the grasp orientation by ``grasp_rot_deg`` about
    #    ``grasp_rot_axis`` — ALL scramble motions run from this pose.
    if scramble_ori_override is not None:
        scramble_ori = scramble_ori_override
    else:
        scramble_ori = _rotate_orientation(
            grasp_ori, grasp_rot_axis, grasp_rot_deg, grasp_rot_frame
        )
        print(
            f"[scramble] rotated grasp orientation {grasp_rot_deg:+.1f} deg "
            f"about {grasp_rot_frame} {grasp_rot_axis.upper()} for stirring."
        )

    # 5b. Lift to the (lowered) carry/flip height AND pull back ``CARRY_BACK_X_M``
    #     in world -X in a single diagonal move (still at the grasp orientation),
    #     clearing the rack on the way up and out before the flip. The flip is
    #     done ``CARRY_FLIP_LOWER_M`` below the over-pan carry height; the ladle
    #     rises that much again when it travels to the hover above the pan.
    flip_z = carry_z - float(CARRY_FLIP_LOWER_M)
    carry_pos = np.array(
        [pick_pos[0] - float(CARRY_BACK_X_M), pick_pos[1], flip_z],
        dtype=np.float64,
    )
    arm.move_to(
        ctx,
        carry_pos,
        grasp_ori,
        label=(
            f"[scramble] lift + move back {CARRY_BACK_X_M * 100:.0f} cm -X to "
            f"carry/flip pose {carry_pos.tolist()} "
            f"({CARRY_FLIP_LOWER_M * 100:.0f} cm below over-pan height)"
        ),
        tol_m=HOME_POS_TOL_M,
    )

    # 6. Flip the ladle IN PLACE at the carry pose, so the reorientation happens
    #    before any travel toward the pan. Runs in precise mode (stiff cart
    #    gains for tight wrist tracking) at the precise-grasp velocity caps —
    #    anything faster trips the FR3 speed limits.
    cur = arm.read_current_ee_world(ctx.redis)
    if cur is None:
        raise RuntimeError(
            "[scramble] cannot read current EE pose to rotate the ladle."
        )
    lifted_pos = cur[0]
    rot_precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="scramble ladle rotation",
    )
    try:
        arm.move_to(
            ctx,
            lifted_pos,
            scramble_ori,
            label=(
                f"[scramble] rotate ladle {grasp_rot_deg:+.1f} deg about "
                f"{grasp_rot_frame} {grasp_rot_axis.upper()} in place "
                f"{lifted_pos.tolist()}"
            ),
            tol_m=DEFAULT_POS_TOL_M,
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
        )
    finally:
        gains.restore_precise_grasp(
            ctx.redis, rot_precise, label="scramble ladle rotation"
        )

    # 7. Move to the hover above the pan, descend to the work plane, scramble,
    #    and lift back — all at ``scramble_ori``.
    arm.move_to(
        ctx,
        hover_pos,
        scramble_ori,
        label=f"[scramble] move to hover above pan {hover_pos.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )
    # Keep the controller's Cartesian OTG ENABLED for the scramble: the stir is
    # now sent as sparse waypoints (triangle corners + a handful of points per
    # circle revolution) rather than a dense per-tick goal stream, so OTG plans
    # the smooth path between them instead of being re-planned every write.
    # Run it in PRECISE mode (stiff cart gains for accurate ladle tracking) but
    # at the normal/full OTG velocity caps (not the slow 0.03 m/s precise-grasp
    # approach cap), so the stir is both stiff and reasonably quick.
    scramble_precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=SCRAMBLE_PRECISE_MAX_LINEAR_VELOCITY,
        max_angular_velocity=SCRAMBLE_PRECISE_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="scramble stir",
    )
    try:
        mix.scramble_at_pose(
            ctx,
            hover_pos,
            scramble_ori,
            down_dz_m=descent_dz,
            radius_m=effective_mix_radius_m,
            # Circles removed from the scramble — triangles only.
            cycles=0,
            cycle_duration_s=mix_cycle_duration_s,
            triangle_passes=triangle_passes,
            triangle_duration_s=triangle_duration_s,
            # Repeat the triangle set until this time budget elapses, or until
            # the user hits ENTER when scramble_until_enter is set (it finishes
            # the in-progress pass first). until_enter takes precedence.
            scramble_duration_s=scramble_duration_s,
            until_enter=scramble_until_enter,
            descend_from_hover=True,
            lift_after=True,
        )
    finally:
        gains.restore_precise_grasp(
            ctx.redis, scramble_precise, label="scramble stir"
        )

    # 7b. Carry the ladle back to the carry/flip pose and UNFLIP it there
    #     (rotate from ``scramble_ori`` back to ``grasp_ori``) before putting it
    #     down — the reverse of step 5b/6, so the un-rotation happens away from
    #     the pan rather than during the descent to the rack.
    arm.move_to(
        ctx,
        carry_pos,
        scramble_ori,
        label=f"[scramble] return to carry/flip pose {carry_pos.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )
    unflip_precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="scramble ladle unflip",
    )
    try:
        arm.move_to(
            ctx,
            carry_pos,
            grasp_ori,
            label=(
                f"[scramble] unflip ladle back to grasp orientation in place "
                f"{carry_pos.tolist()}"
            ),
            tol_m=DEFAULT_POS_TOL_M,
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
        )
    finally:
        gains.restore_precise_grasp(
            ctx.redis, unflip_precise, label="scramble ladle unflip"
        )

    # Carry the ladle back to the exact spot it was grasped and release it
    # there (mirrors egg_crack_controller). Now already at the grasp
    # orientation, transit via a hover above the pick, lower onto the pick, open
    # the gripper, then lift straight up clear of it. A generous timeout keeps
    # the gripper from opening mid-transit.
    above_pick = pick_pos + np.array(
        [0.0, 0.0, SCRAMBLE_RELEASE_ABOVE_M], dtype=np.float64
    )
    arm.move_to(
        ctx,
        above_pick,
        grasp_ori,
        label=f"[scramble] return above ladle pick {above_pick.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
        timeout_s=SCRAMBLE_RELEASE_RETURN_TIMEOUT_S,
    )
    arm.move_to(
        ctx,
        pick_pos,
        grasp_ori,
        label=f"[scramble] lower ladle to pick {pick_pos.tolist()} to release",
        tol_m=DEFAULT_POS_TOL_M,
        timeout_s=SCRAMBLE_RELEASE_RETURN_TIMEOUT_S,
    )
    gripper.open_gripper(
        ctx.redis,
        spec_ladle.open_width,
        speed=spec_ladle.speed,
        force=spec_ladle.force,
        use_max_mode=True,
    )
    print("[scramble] ladle released at its original pick spot.")
    time.sleep(SCRAMBLE_RELEASE_SETTLE_S)
    release_lift = pick_pos + np.array(
        [0.0, 0.0, SCRAMBLE_RELEASE_LIFT_M], dtype=np.float64
    )
    arm.move_to(
        ctx,
        release_lift,
        grasp_ori,
        label=(
            f"[scramble] lift {SCRAMBLE_RELEASE_LIFT_M * 100:.0f} cm clear of "
            f"ladle {release_lift.tolist()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    # Bring the pan back forward to the front burner. Rather than trust the
    # stored post-move handle position, raise back to the SAME overhead
    # pan-grasp look-at pose, REDETECT the handle at the back burner, and slide
    # it so it lands back at the ORIGINAL (pre-move) handle position.
    if move_pan and pan_grasp_pose is not None:
        pan_grasp_look_at = ARM_HOME_POSITION + np.array(
            [0.0, 0.0, float(PAN_GRASP_LOOK_AT_DZ_M)], dtype=np.float64
        )
        arm.move_to(
            ctx,
            pan_grasp_look_at,
            ARM_HOME_ORIENTATION,
            label=(
                f"[pan grasp] overhead look-at pose {pan_grasp_look_at.tolist()} "
                f"(redetect handle for return)"
            ),
            tol_m=HOME_POS_TOL_M,
        )
        return_grasp_pose = _detect_pan_grasp(
            ctx,
            gemini_response_path=pan_grasp_gemini_response_path,
        )
        # X shift that brings the redetected handle back to the original spot.
        return_dx = float(
            pan_grasp_pose.position[0] - return_grasp_pose.position[0]
        )
        print(
            f"[pan] redetected handle X={return_grasp_pose.position[0]:+.3f} m, "
            f"original handle X={pan_grasp_pose.position[0]:+.3f} m -> "
            f"return shift {return_dx * 100:+.1f} cm in X"
        )
        _move_pan(
            ctx,
            pick_pos=return_grasp_pose.position,
            ori=return_grasp_pose.orientation,
            dx_m=return_dx,
            lift_m=float(pan_move_lift_m),
            label="[pan] move to front burner",
        )

    if return_home:
        arm.move_to(
            ctx,
            ARM_HOME_POSITION,
            ARM_HOME_ORIENTATION,
            label=f"[scramble] return to home {ARM_HOME_POSITION.tolist()}",
            tol_m=HOME_POS_TOL_M,
        )

    print("[scramble] scramble cycle complete.")


def main() -> int:
    args = parse_args()

    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key
    ctx.gemini_response_path = args.gemini_response_path

    scramble_ori_override = _parse_orientation_json(args.scramble_ori_json)

    detection_pos = (
        DETECTION_EE_POSITION.copy()
        if args.detection_xyz is None
        else np.array(args.detection_xyz, dtype=np.float64)
    )

    print(f"Step mode     : {'on' if args.step else 'off'}")
    print(f"Skip base     : {args.skip_base}")
    print(f"Look-at EE    : {detection_pos.tolist()}")
    print(f"Gemini log    : {args.gemini_response_path}")
    print(f"Pan log       : {args.pan_gemini_response_path}")

    try:
        run_scramble_cycle(
            ctx,
            skip_base=args.skip_base,
            stove_x=args.stove_x,
            stove_y=args.stove_y,
            stove_yaw_deg=args.stove_yaw_deg,
            tool_length_m=args.tool_length_m,
            detection_pos=detection_pos,
            carry_lift_offset_m=args.carry_lift_offset_m,
            scramble_ori_override=scramble_ori_override,
            grasp_rot_deg=args.grasp_rot_deg,
            grasp_rot_axis=args.grasp_rot_axis,
            grasp_rot_frame=args.grasp_rot_frame,
            mix_radius_m=args.mix_radius_m,
            mix_cycles=args.mix_cycles,
            mix_cycle_duration_s=args.mix_cycle_duration_s,
            triangle_passes=args.triangle_passes,
            triangle_duration_s=args.triangle_duration_s,
            scramble_duration_s=(
                args.scramble_duration_s
                if args.scramble_duration_s and args.scramble_duration_s > 0
                else None
            ),
            scramble_until_enter=args.scramble_until_enter,
            return_home=args.return_home,
            move_pan=args.move_pan,
            pan_shift_back_m=args.pan_shift_back_m,
            pan_move_lift_m=args.pan_move_lift_m,
            detect_back_burner=args.detect_back_burner,
            gemini_response_path=args.gemini_response_path,
            pan_gemini_response_path=args.pan_gemini_response_path,
            pan_grasp_gemini_response_path=args.pan_grasp_gemini_response_path,
            back_burner_gemini_response_path=args.back_burner_gemini_response_path,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
