#!/usr/bin/env python3
"""Complete mixing controller: base navigation + Gemini ladle grasp + stir + sink drop.

End-to-end sequence:

  1. Arm → home pose.
  2. Base → RACK_STATION.
  3. Detect ladle grasp pose via Gemini+RealSense.
  4. Grasp ladle (straight-up lift by CARRY_LIFT_M).
  5. Rack→mix transit pose: move to
     ARM_HOME_POSITION + +Z * TRANSIT_LIFT_HOME_DZ_M (default home
     + 20 cm) AND rotate the held grip orientation by +TRANSIT_ROT_DEG
     about world TRANSIT_ROT_AXIS (default +90° X — ladle on its
     side, rim out).
  6. Base → MIXING_STATION (holds the transit pose throughout).
  7. Stir at taught above pose: move to LADLE_MIX_ABOVE_BOWL_POSITION /
     ORIENTATION, descend LADLE_MIX_DOWN_DZ_M straight down, circular
     stir, lift straight back up to the taught above pose (this is
     the "pull straight out of the bowl" step — strictly vertical).
  8. Mix→sink transit pose: from the taught above pose, move to the
     SAME shared transit pose used in step 5 (ARM_HOME + +Z * lift,
     orientation = Rx(+TRANSIT_ROT_DEG) @ taught_above_ori). Position
     AND orientation change together in one OTG-blended move, so the
     ladle pulls AWAY from the bowl center before crossing toward
     home — the straight-up step 7 lift ensures we're already clear
     of the bowl rim by the time this transition starts.
  9. Base → SINK_STATION (holds the transit pose throughout).
 10. Sink un-rotate: undo the +TRANSIT_ROT_DEG rotation from step 8
     in place at the transit position, so the ladle is back to the
     taught above orientation (tool-down) before any descent.
 11. Two-phase sink drop: reach straight forward by SINK_DROP_DX_M,
     then descend straight down to absolute world Z =
     LADLE_SINK_DROP_Z_M (matches bowl_pour_controller drop height),
     open gripper, settle.
 12. Arm → home pose (optional, skipped by --no-return-home).

Uses only ``zitibot_core`` and ``zitibot_tasks``.

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/mixing_vision_base_controller.py -- --step
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

from zitibot_core import arm, base, gripper
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    BASE_WAYPOINTS,
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_GRIPPER_CLOSE_WIDTH,
    DEFAULT_GRIPPER_FORCE,
    DEFAULT_GRIPPER_SPEED,
    DEFAULT_POS_TOL_M,
    OBJECT_DEFAULTS,
    BaseWaypoint,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_core.runner import step_gate
from zitibot_tasks import gemini, grasp, mix
from zitibot_tasks.gemini import GraspPose

DEFAULT_GRASP_WAYPOINT = BaseWaypoint.RACK_STATION
DEFAULT_MIX_WAYPOINT = BaseWaypoint.MIXING_STATION
DEFAULT_DROP_WAYPOINT = BaseWaypoint.SINK_STATION
DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = (
    "opensai::redis_driver::FrankaRobot::T_end_effector"
)
DEFAULT_GEMINI_RESPONSE_PATH = str(_CONTROLLERS.parent / "logs" / "gemini_response_ladle.png")

# Post-grasp carry lift for the ladle. Bumped 4 cm (0.16 → 0.20) so the
# ladle bowl clears the rim of the mixing bowl when the cart drives
# from RACK_STATION to MIXING_STATION. The post-stir mix→sink carry
# uses the taught ``LADLE_MIX_ABOVE_BOWL_POSITION`` Z directly (not a
# bowl-relative lift), so the post-mix transit clearance is governed
# by that taught pose rather than this constant.
CARRY_LIFT_M = 0.20
SINK_DROP_DX_M = 0.10
SINK_DROP_SETTLE_S = 0.25
# Absolute world Z (m) where the ladle is released into the sink basin.
# Matched to ``bowl_pour_controller``'s bowl drop height so the ladle
# lands in the same basin spot every bowl gets dropped at:
#   ARM_HOME_POSITION[2] (0.6450) + TRANSIT_LIFT_FROM_HOME_M (0.07)
#   - SINK_DROP_DZ_M (0.30) = 0.415 m world Z.
# Hardcoded (rather than importing from bowl_pour_controller) to avoid
# a cross-controller dependency; update both if the bowl drop height
# changes.
LADLE_SINK_DROP_Z_M = 0.415

# Taught EE pose above the mixing bowl, jogged at the bench with the
# ladle held and the cart parked at MIXING_STATION. Replaces the old
# ``bowl_pos + [0, 0, 0.02]`` derivation from ``mix.stir_in_bowl`` —
# the taught pose better matches how the ladle hangs from the closed
# gripper (the bowl-center derivation assumed a perfectly vertical
# tool-down ladle, which isn't true with the post-grasp carry tilt).
# Re-record by jogging the held ladle above the mixing bowl center and
# logging the ``[arm] startup`` print from Redis.
LADLE_MIX_ABOVE_BOWL_POSITION = np.array(
    [+0.2824, -0.0463, +0.7050], dtype=np.float64
)
LADLE_MIX_ABOVE_BOWL_ORIENTATION = np.array(
    [
        [+0.7044, -0.6825, +0.1950],
        [-0.6807, -0.7274, -0.0866],
        [+0.2010, -0.0718, -0.9770],
    ],
    dtype=np.float64,
)
# Descent from the taught above pose to the stir base pose. Derived
# from the bench measurement: above-pose Z (0.7050) − stir-pose Z
# (0.5852) ≈ 0.12 m. The stir runs at this lower Z, keeping the same
# taught orientation.
LADLE_MIX_DOWN_DZ_M = 0.12

# ---------------------------------------------------------------------------
# Shared transit pose (used for BOTH base drives while the ladle is held)
# ---------------------------------------------------------------------------
# Single transit configuration used twice:
#   - After the rack grasp + lift, before the RACK_STATION → MIXING_STATION
#     drive: premultiplied onto the gemini-detected grip orientation.
#   - After the stir + lift back to the taught above pose, before the
#     MIXING_STATION → SINK_STATION drive: premultiplied onto the taught
#     above orientation.
# In both cases the arm is parked at ``ARM_HOME_POSITION + +Z *
# TRANSIT_LIFT_HOME_DZ_M`` with the held orientation rotated by
# ``TRANSIT_ROT_DEG`` about world ``TRANSIT_ROT_AXIS``. 15 cm above
# home is well clear of the counter / cart edge while staying inside
# the arm's comfortable IK envelope; 90° about world X tips the ladle
# bowl onto its side (rim out, not down), so it can't dump anything
# during transit and doesn't dangle below the cart.
TRANSIT_LIFT_HOME_DZ_M = 0.15
TRANSIT_ROT_DEG = 90.0
TRANSIT_ROT_AXIS = "x"


def _rotate_world_frame(ori: np.ndarray, axis: str, deg: float) -> np.ndarray:
    """Return ``ori`` premultiplied by a world-frame rotation of ``deg`` about ``axis``."""
    R_world = R_scipy.from_euler(axis, deg, degrees=True).as_matrix()
    return R_world @ np.asarray(ori, dtype=np.float64).reshape(3, 3)


# ---------------------------------------------------------------------------
# Ladle detection
# ---------------------------------------------------------------------------

_DETECTION_EE_POSITION = np.array([0.33, -0.13, 0.67], dtype=np.float64)


def _detect_ladle(ctx: TaskContext) -> GraspPose:
    """Detect ladle handle via Gemini+RealSense. Returns GraspPose."""
    arm.move_to(ctx, _DETECTION_EE_POSITION, ARM_HOME_ORIENTATION,
                label=f"[ladle detection] raise arm for camera view {_DETECTION_EE_POSITION.tolist()}",
                tol_m=DEFAULT_POS_TOL_M)
    step_gate(ctx, "[ladle detection] ready to query Gemini — press ENTER to capture")
    print("[ladle detection] querying Gemini for ladle handle pose...")
    pose = gemini.find_grasp_pose(ctx, Object.LADLE)
    print(f"[ladle detection] detected position: {pose.position.tolist()}")
    if pose.rim_yaw_applied and pose.rim_yaw_deg is not None:
        print(f"[ladle detection] handle yaw applied: {pose.rim_yaw_deg:.1f} deg")
    return pose


# ---------------------------------------------------------------------------
# Sink drop
# ---------------------------------------------------------------------------

def _drop_at_sink(
    ctx: TaskContext,
    grip_R: np.ndarray,
    spec_ladle,
    *,
    drop_z_m: float,
    drop_dx_m: float,
) -> None:
    """Two-phase sink drop: reach straight forward, then descend straight to ``drop_z_m``.

    Splits the old "forward+down in one diagonal move" into two
    orthogonal segments:

      1. Forward (arm-base +X) by ``drop_dx_m`` over the basin, holding
         the carry-pose Z and orientation. Pushes the drop point past
         the cart edge before any descent.
      2. Vertical descent straight down to absolute world ``drop_z_m``
         (typically ``LADLE_SINK_DROP_Z_M`` so the ladle lands at the
         same basin Z every bowl is dropped at). Same orientation; no
         XY motion during the descent.

    Then opens the gripper and settles. ``drop_z_m`` is absolute world
    Z, not relative to the carry pose, so the landing height stays
    consistent regardless of how high the post-stir lift ended up.
    """
    cur_pose = arm.read_current_ee_world(ctx.redis)
    if cur_pose is None:
        raise RuntimeError("[drop] cannot read current EE pose; aborting sink drop.")
    carry_pos = cur_pose[0]
    forward_pos = carry_pos + np.array([drop_dx_m, 0.0, 0.0], dtype=np.float64)
    descend_pos = np.array(
        [forward_pos[0], forward_pos[1], float(drop_z_m)], dtype=np.float64
    )
    print(
        f"[drop] EE={carry_pos.tolist()}, "
        f"reaching {drop_dx_m * 100:+.1f} cm forward to {forward_pos.tolist()}, "
        f"then descending straight down to world Z={drop_z_m:.4f} m "
        f"({descend_pos.tolist()})"
    )
    arm.move_to(
        ctx,
        forward_pos,
        grip_R,
        label=f"[drop] reach forward over sink basin {forward_pos.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    arm.move_to(
        ctx,
        descend_pos,
        grip_R,
        label=f"[drop] descend to bowl-drop height {descend_pos.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )
    gripper.open_gripper(ctx.redis, spec_ladle.open_width,
                         speed=spec_ladle.speed, force=spec_ladle.force,
                         use_max_mode=True)
    print("[drop] gripper opened — ladle released into sink.")
    time.sleep(SINK_DROP_SETTLE_S)


# ---------------------------------------------------------------------------
# Base navigation helpers (same pattern as bowl_pour_controller)
# ---------------------------------------------------------------------------

def _resolve_base_target(
    waypoint: BaseWaypoint,
    x_override: float | None,
    y_override: float | None,
    yaw_override: float | None,
) -> tuple[float, float, float, str]:
    wp = BASE_WAYPOINTS[waypoint]
    x_m = wp.x_m if x_override is None else float(x_override)
    y_m = wp.y_m if y_override is None else float(y_override)
    yaw_deg = wp.yaw_deg if yaw_override is None else float(yaw_override)
    overridden = x_override is not None or y_override is not None or yaw_override is not None
    tag = " (override)" if overridden else ""
    label = f"[base] {waypoint.name}{tag} -> ({x_m:.3f}, {y_m:.3f}, {yaw_deg:.1f} deg)"
    return x_m, y_m, yaw_deg, label


def _drive_base(
    ctx: TaskContext,
    waypoint: BaseWaypoint,
    *,
    x_override: float | None,
    y_override: float | None,
    yaw_override: float | None,
) -> None:
    overridden = x_override is not None or y_override is not None or yaw_override is not None
    if not overridden:
        base.go_to_pose(ctx, waypoint)
        return
    x_m, y_m, yaw_deg, label = _resolve_base_target(waypoint, x_override, y_override, yaw_override)
    base.go_to_pose(ctx, x_m=x_m, y_m=y_m, yaw_deg=yaw_deg, label=label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    spec_bowl = OBJECT_DEFAULTS[Object.MIXING_BOWL]
    p = argparse.ArgumentParser(
        description="Base navigation + Gemini ladle grasp + stir + drop."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--step", action="store_true",
                   help="ENTER-gate every motion / gripper / base step.")

    # Vision / Gemini.
    p.add_argument("--endeffector-transform-key",
                   default=DEFAULT_ENDEFFECTOR_TRANSFORM_KEY)
    p.add_argument("--gemini-response-path", default=DEFAULT_GEMINI_RESPONSE_PATH,
                   help="Path to save Gemini debug image.")

    # Base target overrides.
    p.add_argument("--grasp-x", type=float, default=None,
                   help="Override RACK_STATION X (m).")
    p.add_argument("--grasp-y", type=float, default=None,
                   help="Override RACK_STATION Y (m).")
    p.add_argument("--grasp-yaw-deg", type=float, default=None,
                   help="Override RACK_STATION yaw (deg).")
    p.add_argument("--mix-x", type=float, default=None,
                   help="Override MIXING_STATION X (m).")
    p.add_argument("--mix-y", type=float, default=None,
                   help="Override MIXING_STATION Y (m).")
    p.add_argument("--mix-yaw-deg", type=float, default=None,
                   help="Override MIXING_STATION yaw (deg).")

    # Grasp tuning.
    p.add_argument("--approach-dz", type=float, default=DEFAULT_APPROACH_DZ_M,
                   help="Hover height above ladle before descending (m).")
    p.add_argument("--gripper-open-width", type=float, default=None,
                   help="Pre-grasp finger opening (m). Default: gripper max.")
    p.add_argument("--gripper-close-width", type=float, default=DEFAULT_GRIPPER_CLOSE_WIDTH,
                   help="Grasp close target (m).")
    p.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED)
    p.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE)
    p.add_argument("--carry-lift-m", type=float, default=CARRY_LIFT_M,
                   help="Post-grasp lift height before base drive (m).")
    p.add_argument(
        "--transit-lift-dz-m",
        type=float,
        default=TRANSIT_LIFT_HOME_DZ_M,
        help=(
            "Shared transit pose: lift above ARM_HOME_POSITION (m) "
            "the held ladle is parked at during BOTH the rack→mix "
            "and mix→sink base drives."
        ),
    )
    p.add_argument(
        "--transit-rot-deg",
        type=float,
        default=TRANSIT_ROT_DEG,
        help=(
            "Shared transit pose: rotation (deg) about world "
            "--transit-rot-axis applied (premultiplied) to the held "
            "orientation. Applied to grip_R for rack→mix and to the "
            "taught above orientation for mix→sink."
        ),
    )
    p.add_argument(
        "--transit-rot-axis",
        choices=("x", "y", "z"),
        default=TRANSIT_ROT_AXIS,
        help="World axis for --transit-rot-deg rotation.",
    )

    # Bowl position (used as a fallback log target only; the actual stir
    # is driven by --mix-above-xyz / --mix-down-dz-m below).
    p.add_argument("--bowl-xyz", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"),
                   help=f"Bowl centre in world frame (m). Default: {spec_bowl.pick_pose.tolist()}")

    # Mixing parameters.
    p.add_argument("--mix-radius", type=float, default=0.1)
    p.add_argument("--cycles", type=int, default=4)
    p.add_argument("--cycle-duration-s", type=float, default=16.0)
    p.add_argument("--mix-above-xyz", nargs=3, type=float, default=None,
                   metavar=("X", "Y", "Z"),
                   help=(
                       "Taught EE world position above the mixing bowl "
                       "(m). The stir descends from here by --mix-down-dz-m "
                       "and lifts back to here when done. "
                       f"Default: {LADLE_MIX_ABOVE_BOWL_POSITION.tolist()}"
                   ))
    p.add_argument("--mix-down-dz-m", type=float, default=LADLE_MIX_DOWN_DZ_M,
                   help=(
                       "Straight-down descent from --mix-above-xyz to the "
                       "stir base (m). The circular stir runs at this lower Z."
                   ))

    # Sink drop overrides.
    p.add_argument("--sink-x", type=float, default=None,
                   help="Override SINK_STATION X (m).")
    p.add_argument("--sink-y", type=float, default=None,
                   help="Override SINK_STATION Y (m).")
    p.add_argument("--sink-yaw-deg", type=float, default=None,
                   help="Override SINK_STATION yaw (deg).")
    p.add_argument("--sink-drop-z-m", type=float, default=LADLE_SINK_DROP_Z_M,
                   help=(
                       "Absolute world Z (m) to descend to before opening "
                       "the gripper at the sink. Default matches the bowl "
                       "drop height in bowl_pour_controller."
                   ))
    p.add_argument("--sink-drop-dx-m", type=float, default=SINK_DROP_DX_M,
                   help="Forward reach into sink basin (m) before descent.")

    p.add_argument("--no-return-home", action="store_true",
                   help="Skip final arm-to-home after dropping ladle.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _apply_ladle_spec_overrides(
    *,
    approach_dz: float,
    gripper_open_width: float | None,
    gripper_close_width: float,
    gripper_speed: float,
    gripper_force: float,
) -> None:
    """Mutate ``OBJECT_DEFAULTS[Object.LADLE]`` with grasp tuning overrides."""
    spec_ladle = OBJECT_DEFAULTS[Object.LADLE]
    spec_ladle.approach_dz = approach_dz
    if gripper_open_width is not None:
        spec_ladle.open_width = gripper_open_width
    spec_ladle.close_width = gripper_close_width
    spec_ladle.speed = gripper_speed
    spec_ladle.force = gripper_force


def run_mixing_cycle(
    ctx: TaskContext,
    *,
    bowl_pos: np.ndarray | None = None,
    mix_radius: float = 0.1,
    cycles: int = 4,
    cycle_duration_s: float = 16.0,
    carry_lift_m: float = CARRY_LIFT_M,
    transit_lift_dz_m: float = TRANSIT_LIFT_HOME_DZ_M,
    transit_rot_deg: float = TRANSIT_ROT_DEG,
    transit_rot_axis: str = TRANSIT_ROT_AXIS,
    mix_above_pos: np.ndarray | None = None,
    mix_above_ori: np.ndarray | None = None,
    mix_down_dz_m: float = LADLE_MIX_DOWN_DZ_M,
    sink_drop_z_m: float = LADLE_SINK_DROP_Z_M,
    sink_drop_dx_m: float = SINK_DROP_DX_M,
    return_home: bool = True,
    grasp_x: float | None = None,
    grasp_y: float | None = None,
    grasp_yaw_deg: float | None = None,
    mix_x: float | None = None,
    mix_y: float | None = None,
    mix_yaw_deg: float | None = None,
    sink_x: float | None = None,
    sink_y: float | None = None,
    sink_yaw_deg: float | None = None,
    gemini_response_path: str | Path | None = None,
) -> None:
    """Base navigation + Gemini ladle grasp + taught-pose stir + sink drop.

    Sequence (mirrors steps 1–11 below):

    1. Arm → home.
    2. Base → RACK_STATION; Gemini detects the ladle handle.
    3. ``grasp.object`` grasps the ladle and lifts straight up by
       ``carry_lift_m``.
    4. Rack→mix transit pose: move to
       ``ARM_HOME_POSITION + +Z * transit_lift_dz_m`` AND rotate the
       held grip orientation by ``transit_rot_deg`` about world
       ``transit_rot_axis`` (premultiplied). Both the position change
       AND the rotation happen in a single ``arm.move_to`` so the
       OTG-blended trajectory is one motion.
    5. Base → MIXING_STATION (the arm holds the transit pose
       throughout the drive).
    6. Arm → taught ``mix_above_pos`` / ``mix_above_ori`` above the
       mixing bowl (overrides the old bowl-center-derived above pose
       that assumed a perfectly vertical tool-down ladle).
    7. Descend ``mix_down_dz_m`` straight down (same orientation),
       circular stir of ``cycles`` × ``cycle_duration_s / cycles``
       seconds at ``mix_radius``, then lift STRAIGHT UP back to the
       taught above pose. This vertical lift is the "pull straight
       out of the bowl" step — it guarantees the ladle is clear of
       the rim before step 8 starts moving laterally.
    8. Mix→sink transit pose: move from the taught above pose to the
       SAME shared transit pose as step 4. Position =
       ``ARM_HOME + +Z * transit_lift_dz_m``, orientation =
       ``Rx(transit_rot_deg) @ taught_above_ori``. Same single-move
       pos+ori change pattern.
    9. Base → SINK_STATION (holds the transit pose).
   10. Sink un-rotate: in-place ``arm.move_to(transit_pos, mix_above_ori)``
       to undo the +``transit_rot_deg`` rotation from step 8. The
       ladle is now at the transit XYZ but at the original taught
       above orientation (tool-down) — ready to descend rim-down
       into the basin.
   11. ``_drop_at_sink`` reaches straight forward, then straight down
       to absolute world Z = ``sink_drop_z_m`` (matched to the bowl
       drop height), opens the gripper, settles.
   12. Optional arm → home.
    """
    if gemini_response_path is not None:
        ctx.gemini_response_path = Path(gemini_response_path)
    elif ctx.gemini_response_path is None:
        ctx.gemini_response_path = Path(DEFAULT_GEMINI_RESPONSE_PATH)

    spec_ladle = OBJECT_DEFAULTS[Object.LADLE]
    if bowl_pos is None:
        bowl_pos = OBJECT_DEFAULTS[Object.MIXING_BOWL].pick_pose.copy()
    else:
        bowl_pos = np.asarray(bowl_pos, dtype=np.float64).reshape(3).copy()
    if mix_above_pos is None:
        mix_above_pos = LADLE_MIX_ABOVE_BOWL_POSITION.copy()
    else:
        mix_above_pos = np.asarray(mix_above_pos, dtype=np.float64).reshape(3).copy()
    if mix_above_ori is None:
        mix_above_ori = LADLE_MIX_ABOVE_BOWL_ORIENTATION.copy()
    else:
        mix_above_ori = np.asarray(mix_above_ori, dtype=np.float64).reshape(3, 3).copy()

    print(f"Bowl pos  : {bowl_pos.tolist()}")
    print(
        f"Mix-above pose (taught): pos={mix_above_pos.tolist()}, "
        f"down-dz={mix_down_dz_m * 100:.1f} cm"
    )
    print(
        f"Shared transit pose (rack→mix AND mix→sink): "
        f"home + {transit_lift_dz_m * 100:.1f} cm Z, "
        f"+{transit_rot_deg:.1f}° about world {transit_rot_axis.upper()}"
    )
    print(f"Sink drop world Z target: {sink_drop_z_m:.4f} m")

    # 1. Arm to home.
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[arm] move to home {ARM_HOME_POSITION.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 2. Base → ladle pickup station.
    _drive_base(
        ctx,
        DEFAULT_GRASP_WAYPOINT,
        x_override=grasp_x,
        y_override=grasp_y,
        yaw_override=grasp_yaw_deg,
    )

    # 3. Detect ladle grasp pose via Gemini+RealSense.
    pose = _detect_ladle(ctx)
    pick_pos = pose.position
    grip_R = pose.orientation

    # 4. Grasp ladle. ``grasp.object`` already lifts the ladle straight
    # up by ``carry_lift_m`` along arm-frame +Z, which is the safe carry
    # pose for the base drive (relative to the current RACK_STATION
    # parking, so no sideways swing).
    grasp.object(
        ctx,
        Object.LADLE,
        pick_pos=pick_pos,
        ori=grip_R,
        lift_dz_m=carry_lift_m,
    )

    # 5. Rack→mix transit pose: move to the shared transit pose
    # (``ARM_HOME_POSITION + +Z * transit_lift_dz_m`` with the held
    # grip rotated by ``transit_rot_deg`` about world ``transit_rot_axis``)
    # BEFORE the base drive. Replaces the earlier "rotate in place
    # at the post-grasp lift" carry — the post-grasp pose's XYZ
    # depends on wherever Gemini landed the ladle, so it's a poor
    # transit pose; ``ARM_HOME + dz`` is a fixed, repeatable transit
    # pose that clears the cart edge regardless of the rack grasp
    # location. The same pose is reused after the stir (step 8).
    grip_R = _rotate_world_frame(grip_R, transit_rot_axis, transit_rot_deg)
    transit_pos = ARM_HOME_POSITION + np.array(
        [0.0, 0.0, float(transit_lift_dz_m)], dtype=np.float64
    )
    arm.move_to(
        ctx,
        transit_pos,
        grip_R,
        label=(
            f"[mix] rack→mix transit pose: home + {transit_lift_dz_m * 100:.1f} cm Z "
            f"{transit_pos.tolist()}, rotated +{transit_rot_deg:.1f}° "
            f"about world {transit_rot_axis.upper()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 6. Base → mixing station. Drive BEFORE moving the arm to the
    # bowl pose — the taught above pose is in the arm frame, so it's
    # only valid once the base has parked at MIXING_STATION.
    _drive_base(
        ctx,
        DEFAULT_MIX_WAYPOINT,
        x_override=mix_x,
        y_override=mix_y,
        yaw_override=mix_yaw_deg,
    )

    # 7. Stir at taught above pose: move to taught above, descend
    # ``mix_down_dz_m`` straight down, run the circular stir, then
    # lift back up to the taught above pose.
    mix.stir_at_pose(
        ctx,
        mix_above_pos,
        mix_above_ori,
        down_dz_m=mix_down_dz_m,
        radius_m=mix_radius,
        cycles=cycles,
        cycle_duration_s=cycle_duration_s,
    )

    # 8. Mix→sink transit pose: move from the taught above pose to the
    # SAME shared transit pose used in step 5. ``stir_at_pose``'s
    # final move was a straight-up lift back to the taught above pose
    # (the "pull straight out of the bowl" step), so by the time
    # this ``arm.move_to`` runs the ladle is already clear of the
    # bowl rim — the OTG-blended pos+ori transition here is safe to
    # cross horizontally toward home. The orientation rotation is
    # premultiplied onto ``mix_above_ori`` (NOT ``grip_R``) since
    # that's the orientation the arm is currently holding.
    sink_carry_ori = _rotate_world_frame(
        mix_above_ori, transit_rot_axis, transit_rot_deg
    )
    arm.move_to(
        ctx,
        transit_pos,
        sink_carry_ori,
        label=(
            f"[mix] mix→sink transit pose: home + {transit_lift_dz_m * 100:.1f} cm Z "
            f"{transit_pos.tolist()}, rotated +{transit_rot_deg:.1f}° "
            f"about world {transit_rot_axis.upper()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 9. Base → sink.
    _drive_base(
        ctx,
        DEFAULT_DROP_WAYPOINT,
        x_override=sink_x,
        y_override=sink_y,
        yaw_override=sink_yaw_deg,
    )

    # 10. Un-rotate at the sink: undo the +transit_rot_deg about
    # ``transit_rot_axis`` we added in step 8, keeping the same XYZ
    # (the transit pose). The arm is now at the transit position
    # with the original taught above orientation — ladle pointed
    # tool-down rather than tipped 90° on its side. Doing this
    # BEFORE the forward+down drop means the ladle descends rim-down
    # into the basin (so anything stuck in it drains into the sink)
    # instead of dropping in on its side.
    arm.move_to(
        ctx,
        transit_pos,
        mix_above_ori,
        label=(
            f"[mix] sink un-rotate: -{transit_rot_deg:.1f}° about world "
            f"{transit_rot_axis.upper()} at {transit_pos.tolist()} "
            "(back to taught above orientation before descent)"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 11. Two-phase sink drop: reach forward, descend straight down to
    # absolute world Z, release. Holds the un-rotated taught above
    # orientation (from step 10) through both moves so the ladle
    # drops tool-down into the basin.
    _drop_at_sink(
        ctx,
        mix_above_ori,
        spec_ladle,
        drop_z_m=sink_drop_z_m,
        drop_dx_m=sink_drop_dx_m,
    )

    # 12. Arm home (optional).
    if return_home:
        arm.move_to(
            ctx,
            ARM_HOME_POSITION,
            ARM_HOME_ORIENTATION,
            label=f"[arm] return to home {ARM_HOME_POSITION.tolist()}",
            tol_m=DEFAULT_POS_TOL_M,
        )


def main() -> int:
    args = parse_args()

    _apply_ladle_spec_overrides(
        approach_dz=args.approach_dz,
        gripper_open_width=args.gripper_open_width,
        gripper_close_width=args.gripper_close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
    )

    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    print(f"Step mode : {'on' if args.step else 'off'}")

    bowl_pos = (
        np.array(args.bowl_xyz, dtype=np.float64)
        if args.bowl_xyz is not None
        else None
    )
    mix_above_pos = (
        np.array(args.mix_above_xyz, dtype=np.float64)
        if args.mix_above_xyz is not None
        else None
    )

    try:
        run_mixing_cycle(
            ctx,
            bowl_pos=bowl_pos,
            mix_radius=args.mix_radius,
            cycles=args.cycles,
            cycle_duration_s=args.cycle_duration_s,
            carry_lift_m=args.carry_lift_m,
            transit_lift_dz_m=args.transit_lift_dz_m,
            transit_rot_deg=args.transit_rot_deg,
            transit_rot_axis=args.transit_rot_axis,
            mix_above_pos=mix_above_pos,
            mix_down_dz_m=args.mix_down_dz_m,
            sink_drop_z_m=args.sink_drop_z_m,
            sink_drop_dx_m=args.sink_drop_dx_m,
            return_home=not args.no_return_home,
            grasp_x=args.grasp_x,
            grasp_y=args.grasp_y,
            grasp_yaw_deg=args.grasp_yaw_deg,
            mix_x=args.mix_x,
            mix_y=args.mix_y,
            mix_yaw_deg=args.mix_yaw_deg,
            sink_x=args.sink_x,
            sink_y=args.sink_y,
            sink_yaw_deg=args.sink_yaw_deg,
            gemini_response_path=args.gemini_response_path,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()

    print("Mixing complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
