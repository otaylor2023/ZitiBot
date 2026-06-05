#!/usr/bin/env python3
"""OptiTrack base + Gemini vision cylinder grasp + pour + return to rack.

End-to-end per-cylinder sequence (``run_cylinder_cycle``):

  1. Arm → home pose.
  2. Base → ``RACK_STATION``; arm → camera-framing detection pose.
  3. Gemini detects two points on the cylinder's colored strip (handle
     mode, default) or long axis (body mode). Per-cylinder strip colors:
     parmesan = BLUE, sauce = GRAY, ricotta = RED.
  4. Approach, close gripper, lift to carry height.
  5. Base → cylinder's pour station (``CYLINDER_POUR_STATION[cylinder]``:
     ``PAN_STATION`` for parmesan, ``MIXING_STATION`` for sauce / ricotta).
  6. Cartesian pretilt → J7 dump → J4 joint-space shake → restore → carry.
  7. Base → ``RACK_STATION``; descend to grasp pose; open gripper.

``main()`` runs ``run_cylinder_cycle`` once per cylinder in the sequence
selected by ``--cylinder``. The default ``all`` runs parmesan → sauce →
ricotta. Single-cylinder modes (``parmesan`` / ``sauce`` / ``ricotta``)
run exactly one cycle.

Two grasp-target modes, selected with ``--grasp-target``:

- ``handle`` (default): Gemini picks two points on the colored strip;
  wrist yaws perpendicular to the strip axis (pan-handle geometry).
- ``body``: Gemini picks two points on the cylinder long axis; no yaw.

Runs in two UI modes:
- **headless** (default): stdin-driven, no OpenCV window
- **UI** (``--ui``): live RGB + depth + Gemini overlay

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
TidyBot base redis_driver, RealSense, OptiTrack on Redis, and
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.

Usage::

  # default: parmesan → sauce → ricotta
  ./ZitiBot/launch_zitibot_full.sh controllers/grasp_and_pour_jar_controller.py

  # single-cylinder runs
  ./ZitiBot/launch_zitibot_full.sh controllers/grasp_and_pour_jar_controller.py -- \\
      --cylinder parmesan
  ./ZitiBot/launch_zitibot_full.sh controllers/grasp_and_pour_jar_controller.py -- \\
      --cylinder sauce
  ./ZitiBot/launch_zitibot_full.sh controllers/grasp_and_pour_jar_controller.py -- \\
      --cylinder ricotta

  # with UI
  ./ZitiBot/launch_zitibot_full.sh controllers/grasp_and_pour_jar_controller.py -- \\
      --ui --cylinder parmesan
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from grasp_and_pour_controller import (
    DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    DEFAULT_GRIPPER_PREGRASP_WIDTH,
    DEFAULT_GRIPPER_GRASP_SETTLE_S,
    DEFAULT_GRIPPER_SPEED,
    DEFAULT_TILT_DURATION_S,
    GRASP_ORIENTATION,
    MotionParams,
    OrientationSlerpState,
    _do_grasp_object,
    _do_open_gripper,
    _publish_cartesian,
    read_current_ee_world,
    resolve_gripper_open_width,
)
from vision import gemini_pointing as gp
from vision import realsense_rgbd as rs_cam
from zitibot_core import arm, base, gains
from zitibot_tasks.gemini import handle_gemini_timeout
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    CYLINDER_EE_WP2,
    CYLINDER_EE_WP3,
    CYLINDER_EE_WP4,
    CYLINDER_GRASP_Z_DESCENT_REDUCTION_M,
    DEFAULT_POS_TOL_M,
    PARMESAN_DETECTION_EE_ORIENTATION,
    PARMESAN_DETECTION_EE_POSITION,
    PARMESAN_PRETILT_EE_ORIENTATION,
    PARMESAN_PRETILT_EE_POSITION,
    RICOTTA_DETECTION_EE_ORIENTATION,
    RICOTTA_DETECTION_EE_POSITION,
    SAUCE_DETECTION_EE_ORIENTATION,
    SAUCE_DETECTION_EE_POSITION,
    T_FLANGE_CAMERA,
    BaseWaypoint,
    Object,
    OBJECT_DEFAULTS,
)
from zitibot_core.context import make_context

# Camera → flange calibration now lives in zitibot_core.constants (single
# source of truth), imported above as T_FLANGE_CAMERA.

# Cylinder grasp target offset, applied in world frame on top of the
# Gemini-selected point. ``DEFAULT_GRASP_OFFSET_M`` is used for ``body``
# grasps: +X (2 cm) nudges the grasp slightly forward (away from the
# robot base) so the gripper closes over the bottle body rather than
# skimming the near edge / label seam where depth tends to bleed onto
# the rack behind it. Body-mode is shared across cylinders because we
# only ever fall back to it when the handle prompt fails — the offset
# is just "push slightly forward" with no handle-specific bias.
#
# Handle-mode offsets live per-cylinder in
# ``OBJECT_DEFAULTS[CYLINDER_OBJECT_KEY[cylinder]].gemini_world_offset_m``
# (see ``zitibot_core.constants``). Same +X forward nudge PLUS -Y to
# shift the grasp right (Franka base frame convention: +X forward, +Y
# left, +Z up, so right = -Y) so the gripper lands on the handle
# centerline. Tune per-cylinder by editing ``OBJECT_DEFAULTS`` if a
# specific handle drifts the other way; flip the sign on the -Y if the
# real handle is offset left rather than right.
#
# ``grasp_offset_world(args, cylinder)`` picks body vs handle, looks up
# the per-cylinder handle default, and lets the ``--grasp-offset-x/-y/-z``
# CLI flags override individual axes.
ENABLE_GRASP_OFFSET = True
DEFAULT_GRASP_OFFSET_M = np.array([0.02, 0.0, 0.0], dtype=np.float64)

# Post-grasp carry pose. After the gripper closes, the arm lifts to
# ``grasp_pos + +Z * CYLINDER_CARRY_LIFT_M`` AND rotates the EE
# ``CYLINDER_POST_GRASP_TILT_DEG`` about world +Y so the held cylinder
# starts coming up toward the pour orientation while still at the rack.
# This pose is also where the arm returns after the pour (post-shake
# restore) before the base drives back to the rack. Tune lift up if the
# bottle scrapes the cradle / rack on its way out; tune the tilt up to
# pre-orient further toward the pour pose. Set the tilt to 0 to fall
# back to the old straight-up lift.
CYLINDER_CARRY_LIFT_M = 0.25
CYLINDER_POST_GRASP_TILT_DEG = -45.0
CYLINDER_POST_GRASP_TILT_AXIS = "y"

# Extra wrist-tilt applied to the Gemini-derived grasp orientation
# for ONLY the above-grasp / grasp / release-descent moves. The carry
# orientation is built independently from the un-rotated Gemini
# orientation (see ``latch_cylinder_grasp_target``), so this rotation
# does NOT propagate into the carry attitude — carry stays at the
# original ``CYLINDER_POST_GRASP_TILT_DEG`` (-45°) about world Y from
# the Gemini orientation.
#
# Premultiplied = world-frame rotation. The Gemini handle yaw aligns
# the jaws perpendicular to the handle axis (correct closing axis),
# but leaves the gripper coming straight down at the handle — which
# is awkward for a handle that sticks out sideways. -30° about world
# +Y tips the gripper backward (over the rack) so the jaws approach
# the handle from above-and-back instead of straight down. Used for
# BOTH the above-grasp pre-position AND the rack-return descent so
# the cylinder seats back into the rack at the same wrist attitude
# it was lifted out at. These are LEGACY fallbacks — the live values
# now live on ``ObjectSpec.grasp_extra_rot_deg`` /
# ``grasp_extra_rot_axis`` so each cylinder can opt in/out independently
# (parmesan currently uses 0° to keep its strip grasp tool-down).
CYLINDER_GRASP_EXTRA_ROT_DEG = -30.0
CYLINDER_GRASP_EXTRA_ROT_AXIS = "y"

# Loosened tolerance for the per-cycle home / camera-framing moves.
# These are transit moves where exact endpoint accuracy doesn't matter
# — landing within a fat tolerance ball is fine — so use a looser bound
# than the shared DEFAULT_POS_TOL_M (4 cm) to avoid burning the
# convergence timeout on the long tail of the OTG ramp.
CYLINDER_HOME_TOL_M = 0.08

# Pour over the pan, hybrid Cartesian + joint:
#
#   1. PRETILT: Cartesian move to a hand-taught EE pose that orients the
#      bottle opening over the pan. Replaces the old J6=+164° joint move,
#      which kept failing on convergence (J6 had to swing ~90° while
#      holding the bottle and never reached the goal before timeout).
#      The EE pose was captured from a live ``[arm] startup`` print after
#      jogging the arm into a working pre-pour configuration; tune by
#      re-recording, not by editing the numbers by hand.
#   2. DUMP:    joint-space J7 flip to actually pour parmesan out (kept
#      because J7 is a pure wrist roll about the held bottle's long axis,
#      which Cartesian IK doesn't expose cleanly).
#
# Then PAN_DUMPED restores J7 to whatever it was just before the dump
# (after the Cartesian pretilt converged) and re-publishes the carry
# Cartesian pose so the arm returns to a transport-safe configuration
# before the base drives back to the rack.
# Per-cylinder Cartesian pretilt spec — see ``CYLINDER_PRETILT_SPEC``
# below. Two modes:
#
#   * **Taught pose** (parmesan): drive the EE to the hand-taught
#     ``(PARMESAN_PRETILT_EE_POSITION, PARMESAN_PRETILT_EE_ORIENTATION)``
#     pose from ``zitibot_core.constants``. Tune by re-recording, not by
#     editing the numbers.
#
#   * **Live rotate** (sauce / ricotta): keep the live EE position from
#     the carry pose and rotate the live EE orientation by
#     ``rotate_deg`` about world ``rotate_axis``. Skipping the taught
#     pose avoids re-teaching for every mixing-bowl ingredient — the
#     carry pose is already above the bowl after the base drives to
#     MIXING_STATION, so a single in-place wrist rotation puts the
#     bottle opening over the bowl and lets the J7 dump pour it in.


@dataclass(frozen=True)
class PretiltSpec:
    """How to compute the Cartesian pretilt pose for a cylinder.

    Either ``taught_pos`` + ``taught_ori`` are both set (drive to that
    absolute EE pose) OR ``taught_pos`` is None and ``rotate_deg`` /
    ``rotate_axis`` describe a world-frame rotation applied to the
    live EE orientation at AT_POUR entry (position kept).

    ``world_x_offset_m`` / ``world_z_offset_m`` are added to the
    resulting pretilt position (world frame, applied after the
    taught-pos copy or the rotate-mode position-hold). Use them to
    nudge the pour pose without re-teaching: e.g. sauce / ricotta
    pour into a mixing bowl that sits forward of and above the
    carry pose, so lifting the rotate-mode pretilt by ~10 cm in +Z
    and pushing it ~10 cm in +X (arm-base +X = forward / away from
    the cart) clears the rim and positions the bottle opening over
    the bowl contents before the wrist tilt actually pours. Default
    0 on both keeps the legacy behaviour (taught pose unchanged,
    rotate-mode held at live EE pos).
    """

    taught_pos: np.ndarray | None = None
    taught_ori: np.ndarray | None = None
    rotate_deg: float = 0.0
    rotate_axis: str = "z"
    world_x_offset_m: float = 0.0
    world_z_offset_m: float = 0.0
# Empty dict so the restore step's ``touched`` set only includes the DUMP
# joints (currently just J7) — there is no longer a joint-space pretilt
# to revert.
CYLINDER_PRETILT_JOINTS_DEG: dict[int, float] = {}
CYLINDER_DUMP_JOINTS_DEG: dict[int, float] = {7: -151.0}

# Wait budget for the Cartesian pretilt. The pose is a relatively large
# wrist re-orientation while holding the bottle, so allow more time than
# the default 4 s used for nearby moves; loosen the position tolerance
# so we don't false-time-out on a few millimetres of PID overshoot.
CYLINDER_PRETILT_TIMEOUT_S = 8.0
CYLINDER_PRETILT_TOL_M = 0.04

# Tiny J7 nudge issued right before the real dump. The first joint-space
# command after the AT_PAN cartesian pretilt sometimes gets eaten by the
# cartesian→joint controller swap (the joint_controller reseeds its goal
# to the live measured joints when it activates, clobbering our queued
# big-dump goal). Issuing this small ±delta first forces the controller
# swap to settle on a goal that's basically the current pose, after which
# the real ``CYLINDER_DUMP_JOINTS_DEG`` step is just an in-controller
# goal update and tracks correctly. Tune ``..._DELTA_DEG`` up if the
# warm-up still gets swallowed; ``..._TIMEOUT_S`` is short on purpose so
# a missed warm-up only costs a couple of seconds.
CYLINDER_DUMP_WARMUP_DELTA_DEG = 3.0
CYLINDER_DUMP_WARMUP_TIMEOUT_S = 2.0
CYLINDER_DUMP_WARMUP_TOL_RAD = 0.05

# Post-dump shake: while the jar is tilted (J7 flipped), stay in the
# joint controller and walk J4 through the explicit sequence in
# ``CYLINDER_SHAKE_J4_STROKES`` below. Joint-space shake means we never
# leave the joint controller between the dump and the J7 flip-back, so:
#   - no cartesian↔joint controller swaps to absorb between strokes
#   - the J7=-151° dump goal stays parked the whole shake (we only
#     overwrite J4 in the partial-joint command)
#   - no need for the J7 warmup nudge before the J7 restore
# J4 (elbow) was picked because a few-degree elbow swing translates to
# vertical EE motion at this pose without yawing the bottle opening
# around. Each stroke is RELATIVE to the joint value measured at the
# start of the stroke, so the residual from a short stroke doesn't
# accumulate across the sequence.
#
# The entire ``CYLINDER_SHAKE_J4_STROKES`` sequence runs under a single
# ENTER press (one POUR_DUMPED → SHAKE_LOWERED transition). The restore
# step (J7 flip-back + return to carry pose) is its own ENTER-gated
# phase.
# Explicit J4 shake sequence: list of (label, delta_deg) tuples
# executed in order, where each delta is RELATIVE to the live J4
# reading at the start of that stroke (negative = elbow down /
# bottle into bowl, positive = elbow up / bottle away from bowl).
# The last stroke is sized to also serve as the "lift away" step
# that pulls the bottle opening above the bowl rim before
# SHAKE_LOWERED starts the unrotation (J7 flip-back + cartesian
# return to carry), so the still-tilted bottle doesn't clip the rim
# on its way back to vertical. Edit / extend in place to retune the
# shake pattern.
CYLINDER_SHAKE_J4_STROKES: tuple[tuple[str, float], ...] = (
    ("DOWN",  -2.0),
    ("UP",   +10.0),
    ("DOWN",  -2.0),
    ("UP",   +20.0),
)
# Loose convergence tolerance (8.6°) — typically bigger than the
# per-stroke delta on purpose. The joint starts within tol of the
# goal at command time for the small strokes, so
# ``arm.move_to_joints_partial`` advances to the next stroke almost
# immediately while the OTG ramp keeps actuating the joint in the
# background. Net effect is a brisk wiggle (the controller never
# blocks for the full tail of the ramp) rather than a tight square
# wave. The final +20° "lift away" stroke is large enough that the
# wait actually drives most of the way to the goal before timing
# out, which is what we want — that stroke must complete before
# SHAKE_LOWERED runs the wrist flip-back.
CYLINDER_SHAKE_J4_TOL_RAD = 0.15
CYLINDER_SHAKE_J4_TIMEOUT_S = 3.0

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_RESPONSE_PATH = _CONTROLLERS.parent / "logs" / "gemini_response_cylinder.png"
DEFAULT_GEMINI_RESPONSE_PATH_HANDLE = _CONTROLLERS.parent / "logs" / "gemini_response_cylinder_handle.png"
DEFAULT_CYLINDER_CHOICE = "all"


class Cylinder(enum.Enum):
    PARMESAN = "parmesan"
    SAUCE = "sauce"
    RICOTTA = "ricotta"


# CLI ``--cylinder`` choices → ordered tuple of cylinders to run.
_CYLINDER_SEQUENCES: dict[str, tuple[Cylinder, ...]] = {
    "all": (Cylinder.PARMESAN, Cylinder.SAUCE, Cylinder.RICOTTA),
    "parmesan": (Cylinder.PARMESAN,),
    "sauce": (Cylinder.SAUCE,),
    "ricotta": (Cylinder.RICOTTA,),
}

# Per-cylinder pour routes and Gemini strip colors.
CYLINDER_POUR_STATION: dict[Cylinder, BaseWaypoint] = {
    Cylinder.PARMESAN: BaseWaypoint.PAN_STATION,
    Cylinder.SAUCE: BaseWaypoint.MIXING_STATION,
    Cylinder.RICOTTA: BaseWaypoint.MIXING_STATION,
}
CYLINDER_HANDLE_COLOR: dict[Cylinder, str] = {
    Cylinder.PARMESAN: "BLUE",
    Cylinder.SAUCE: "GRAY",
    Cylinder.RICOTTA: "RED",
}
CYLINDER_GEMINI_RESPONSE_PATHS: dict[Cylinder, Path] = {
    Cylinder.PARMESAN: _CONTROLLERS.parent / "logs" / "gemini_response_parmesan_strip.png",
    Cylinder.SAUCE: _CONTROLLERS.parent / "logs" / "gemini_response_sauce_strip.png",
    Cylinder.RICOTTA: _CONTROLLERS.parent / "logs" / "gemini_response_ricotta_strip.png",
}
# Per-cylinder camera-framing EE pose at RACK_STATION. Source values
# live in ``zitibot_core.constants`` so the bench-taught poses are
# colocated with every other taught pose.
CYLINDER_DETECTION_POSE: dict[Cylinder, tuple[np.ndarray, np.ndarray]] = {
    Cylinder.PARMESAN: (PARMESAN_DETECTION_EE_POSITION, PARMESAN_DETECTION_EE_ORIENTATION),
    Cylinder.SAUCE: (SAUCE_DETECTION_EE_POSITION, SAUCE_DETECTION_EE_ORIENTATION),
    Cylinder.RICOTTA: (RICOTTA_DETECTION_EE_POSITION, RICOTTA_DETECTION_EE_ORIENTATION),
}
# Per-cylinder Cartesian pretilt spec. Parmesan pours into the pan
# from a hand-taught pose; sauce + ricotta pour into the mixing bowl
# by rotating the live carry-pose orientation in place.
CYLINDER_PRETILT_SPEC: dict[Cylinder, PretiltSpec] = {
    Cylinder.PARMESAN: PretiltSpec(
        taught_pos=PARMESAN_PRETILT_EE_POSITION,
        taught_ori=PARMESAN_PRETILT_EE_ORIENTATION,
    ),
    # Sauce / ricotta pour into the mixing bowl; from the carry pose
    # above MIXING_STATION lift the rotate-mode pretilt +10 cm in
    # world Z (clears the bowl rim) and push +5 cm in world X (arm
    # forward, away from the cart) so the bottle opening lands over
    # the bowl contents before the wrist tilt actually pours. X was
    # 10 cm initially — too far forward, the pour landed past the
    # bowl center. The -40° rotation about world +Y pre-tilts the
    # bottle most of the way toward horizontal so the J7 dump only
    # has to finish the last bit of the pour (was -20° initially —
    # bottle wasn't tipped far enough at the pretilt and the dump
    # alone wasn't enough to empty the jar).
    Cylinder.SAUCE: PretiltSpec(
        rotate_deg=-40.0, rotate_axis="y",
        world_x_offset_m=0.05, world_z_offset_m=0.10,
    ),
    Cylinder.RICOTTA: PretiltSpec(
        rotate_deg=-40.0, rotate_axis="y",
        world_x_offset_m=0.05, world_z_offset_m=0.10,
    ),
}
# Per-cylinder ObjectSpec lookup. Each entry pulls the cylinder's
# handle-mode ``gemini_world_offset_m`` (and any future per-cylinder
# grasp tunables — gripper widths, force, approach_dz) out of
# ``zitibot_core.constants.OBJECT_DEFAULTS``. The Object enum entries
# are deep-copied from ``Object.BOTTLE`` and currently share the same
# numbers; tune per-cylinder over there.
CYLINDER_OBJECT_KEY: dict[Cylinder, Object] = {
    Cylinder.PARMESAN: Object.PARMESAN,
    Cylinder.SAUCE: Object.SAUCE,
    Cylinder.RICOTTA: Object.RICOTTA,
}
assert all(
    cyl in CYLINDER_POUR_STATION
    and cyl in CYLINDER_HANDLE_COLOR
    and cyl in CYLINDER_DETECTION_POSE
    and cyl in CYLINDER_PRETILT_SPEC
    and cyl in CYLINDER_OBJECT_KEY
    for seq in _CYLINDER_SEQUENCES.values()
    for cyl in seq
), (
    "every cylinder in _CYLINDER_SEQUENCES needs CYLINDER_POUR_STATION, "
    "CYLINDER_HANDLE_COLOR, CYLINDER_DETECTION_POSE, "
    "CYLINDER_PRETILT_SPEC, and CYLINDER_OBJECT_KEY entries"
)

# --grasp-target choices. ``body`` is the original behaviour: Gemini picks two
# points on the cylinder's visible long axis, we anchor at the deeper of the
# two and skip the perpendicular yaw (the bottle is vertical, so axis-yaw is
# noise). ``handle`` is for the case where a slender handle has been bolted
# parallel to the cylinder so the gripper can wrap around the handle instead
# of the too-wide body; this path mirrors the pan-handle grasp builder —
# position = point 1 and the wrist yaws so the jaws close perpendicular to
# the line between the two handle points.
GRASP_TARGET_BODY = "body"
GRASP_TARGET_HANDLE = "handle"
GRASP_TARGET_CHOICES = (GRASP_TARGET_BODY, GRASP_TARGET_HANDLE)

# UI rendering
_TEXT_SIZE_MULT = 0.75 * 0.75
_TEXT_FONT_SCALE = 0.48 * 3.0 * _TEXT_SIZE_MULT
_TEXT_LINE_STEP = int(22 * 3.0 * _TEXT_SIZE_MULT)
_TEXT_THICKNESS = 2
_TEXT_EMPTY_SKIP = int(18 * 3.0 * _TEXT_SIZE_MULT)
TEXT_BAND_HEIGHT = int((int(120 * 3.0) + 180) * _TEXT_SIZE_MULT) + 80


class Phase(enum.Enum):
    """Phase machine: grasp at rack → move to pour station → joint pour → rack release.

    Cartesian arm motion everywhere except at the pour station, where the pour
    is done in joint space (J7 dump + J4 shake) for repeatable pours.
    """
    VISION_READY = "VISION_READY"
    VISION_LATCHED = "VISION_LATCHED"
    ABOVE_GRASP = "ABOVE_GRASP"
    AT_GRASP = "AT_GRASP"
    GRASPED = "GRASPED"
    LIFTED = "LIFTED"
    MOVING_TO_POUR = "MOVING_TO_POUR"
    AT_POUR = "AT_POUR"
    POUR_PRETILT = "POUR_PRETILT"
    POUR_DUMPED = "POUR_DUMPED"
    # Single shake phase — POUR_DUMPED handler walks J4 through
    # ``CYLINDER_SHAKE_J4_STROKES`` under one ENTER press, then hands
    # off to SHAKE_LOWERED (which gates the J7 flip-back + return-
    # to-carry under a second ENTER press).
    SHAKE_LOWERED = "SHAKE_LOWERED"
    POUR_RESTORED = "POUR_RESTORED"
    MOVING_TO_RACK = "MOVING_TO_RACK"
    PLACING = "PLACING"
    AT_RELEASE = "AT_RELEASE"
    DONE = "DONE"


@dataclass
class LatchedTarget:
    """Cylinder long-axis grasp target (world frame).

    ``above_grasp_pos`` is the short pre-grasp staging height (set by
    ``--approach-dz``, default 6 cm). ``carry_pos`` / ``carry_ori`` is
    the post-grasp transit pose: ``CYLINDER_CARRY_LIFT_M`` above the
    grasp position with the EE rotated ``CYLINDER_POST_GRASP_TILT_DEG``
    about world +``CYLINDER_POST_GRASP_TILT_AXIS`` from ``grasp_ori``
    so the held cylinder starts coming up toward the pour orientation.
    """

    grasp_pos: np.ndarray
    above_grasp_pos: np.ndarray
    carry_pos: np.ndarray
    grasp_ori: np.ndarray
    carry_ori: np.ndarray
    cylinder_axis: np.ndarray
    closing_axis: np.ndarray


def _post_grasp_carry_ori(grasp_ori: np.ndarray) -> np.ndarray:
    """Build the carry-pose orientation: ``grasp_ori`` rotated in world frame.

    Rotates ``grasp_ori`` by ``CYLINDER_POST_GRASP_TILT_DEG`` about the
    world ``CYLINDER_POST_GRASP_TILT_AXIS`` axis (premultiplication =
    world-frame rotation). The cylinder is held perpendicular to the
    jaws, so a +45° rotation about world +Y tips the bottle "up" while
    the EE lifts to ``carry_pos``.
    """
    if abs(CYLINDER_POST_GRASP_TILT_DEG) < 1e-9:
        return grasp_ori.copy()
    R_world = R.from_euler(
        CYLINDER_POST_GRASP_TILT_AXIS,
        CYLINDER_POST_GRASP_TILT_DEG,
        degrees=True,
    ).as_matrix()
    return R_world @ grasp_ori


def _apply_cylinder_grasp_extra_rot(
    grasp_ori: np.ndarray,
    *,
    deg: float = CYLINDER_GRASP_EXTRA_ROT_DEG,
    axis: str = CYLINDER_GRASP_EXTRA_ROT_AXIS,
) -> np.ndarray:
    """Premultiply ``grasp_ori`` by the cylinder grasp's extra wrist tilt.

    World-frame rotation of ``deg`` about ``axis``, applied to the
    Gemini-derived grasp orientation. Returned ori is what gets stored
    in ``LatchedTarget.grasp_ori`` — drives the above-grasp move, the
    grasp descent, AND the rack-return descent, so the cylinder enters
    and exits the rack at the same wrist attitude. ``deg`` / ``axis``
    default to the legacy globals; per-cylinder callers pass values
    pulled from ``ObjectSpec.grasp_extra_rot_deg`` /
    ``grasp_extra_rot_axis`` (parmesan = 0°, sauce / ricotta = -30° Y).
    """
    if abs(deg) < 1e-9:
        return grasp_ori.copy()
    R_world = R.from_euler(
        axis,
        deg,
        degrees=True,
    ).as_matrix()
    return R_world @ grasp_ori


def read_T_base_flange(redis_client, key: str) -> np.ndarray | None:
    try:
        raw = redis_client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        T = np.array(json.loads(raw), dtype=np.float64)
        if T.shape != (4, 4):
            T = T.reshape(4, 4)
        return T
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def read_current_orientation(redis_client) -> np.ndarray | None:
    pose = read_current_ee_world(redis_client)
    if pose is None:
        return None
    return pose[1]


# No joint-space helpers needed - pure cartesian motion only


def camera_point_to_base(T_base_flange: np.ndarray, point_camera: np.ndarray) -> np.ndarray:
    p_cam_h = np.ones(4, dtype=np.float64)
    p_cam_h[:3] = np.asarray(point_camera, dtype=np.float64).reshape(3)
    p_base_h = T_base_flange @ T_FLANGE_CAMERA @ p_cam_h
    return p_base_h[:3].copy()


def base_point_to_pixel(
    T_base_flange: np.ndarray,
    world_xyz: np.ndarray,
    intrinsics,
) -> tuple[int, int] | None:
    """Project a base/world XYZ into the RealSense color-image pixel.

    Inverse of :func:`camera_point_to_base`: apply
    ``inv(T_base_flange @ T_FLANGE_CAMERA)`` to land in camera optical
    space, then project with the live ``color_intrinsics``. Used to
    overlay the computed grasp control point on the saved
    ``gemini_response_cylinder.png`` so it's clear where the latched
    target landed relative to the two raw Gemini long-axis points.
    Returns ``None`` if the point is behind the camera, the inverse
    blows up, or the projected pixel is outside the image.
    """
    try:
        T_base_cam = T_base_flange @ T_FLANGE_CAMERA
        try:
            T_cam_base = np.linalg.inv(T_base_cam)
        except np.linalg.LinAlgError:
            return None
        p_world_h = np.ones(4, dtype=np.float64)
        p_world_h[:3] = np.asarray(world_xyz, dtype=np.float64).reshape(3)
        p_cam = (T_cam_base @ p_world_h)[:3]
        if p_cam[2] <= 0.0:
            return None
        import pyrealsense2 as rs

        px = rs.rs2_project_point_to_pixel(
            intrinsics,
            [float(p_cam[0]), float(p_cam[1]), float(p_cam[2])],
        )
        return int(round(px[0])), int(round(px[1]))
    except Exception:
        return None


def compute_cylinder_grasp_orientation(
    p1_base: np.ndarray,
    p2_base: np.ndarray,
    up_vec: np.ndarray | None = None,
    *,
    base_ori: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build the same perpendicular-yaw grasp used by pan/bowl.

    The target stays on the selected long-axis line. The two axis points
    only choose yaw: we keep the ``base_ori`` posture (defaults to the
    tool-down ``GRASP_ORIENTATION`` for back-compat with earlier callers,
    but the cylinder-handle path passes ``PARMESAN_DETECTION_EE_ORIENTATION``
    so the swing-in pose matches the camera-framing pose) and rotate it so
    the gripper closes perpendicular to the detected axis, matching the
    shared Gemini grasp-pose builders.
    """
    if base_ori is None:
        base_ori = GRASP_ORIENTATION
    p1 = np.asarray(p1_base, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2_base, dtype=np.float64).reshape(3)

    axis = p2 - p1
    axis_norm_3d = np.linalg.norm(axis)
    if axis_norm_3d < 1e-6:
        print("Warning: two cylinder long-axis points are too close")
        return None

    axis_xy = axis.copy()
    axis_xy[2] = 0.0
    axis_norm = np.linalg.norm(axis_xy)
    if axis_norm < 1e-6:
        print("Warning: cylinder axis projects poorly into XY")
        return None

    cylinder_axis = axis_xy / axis_norm
    yaw = float(np.arctan2(cylinder_axis[1], cylinder_axis[0]))
    wrapped_yaw = ((yaw + np.pi / 2.0) % np.pi) - np.pi / 2.0
    c, s = np.cos(wrapped_yaw), np.sin(wrapped_yaw)
    R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    grasp_ori = R_z @ base_ori
    closing_axis = grasp_ori[:, 1].copy()

    print(
        f"  Cylinder axis yaw: {np.degrees(yaw):+.2f} deg "
        f"-> applied {np.degrees(wrapped_yaw):+.2f} deg (perp grasp)"
    )
    print(
        f"  Cylinder axis: [{cylinder_axis[0]:+.3f}, {cylinder_axis[1]:+.3f}, {cylinder_axis[2]:+.3f}]"
    )
    print(
        f"  Closing axis:  [{closing_axis[0]:+.3f}, {closing_axis[1]:+.3f}, {closing_axis[2]:+.3f}]"
    )
    print(
        f"  Tool Z axis:    [{grasp_ori[0, 2]:+.3f}, {grasp_ori[1, 2]:+.3f}, {grasp_ori[2, 2]:+.3f}]"
    )

    return grasp_ori, cylinder_axis, closing_axis


def _above_grasp_offset(
    approach_dz_m: float,
    grasp_ori: np.ndarray,
    approach_along_tool_z: bool,
) -> np.ndarray:
    """Vector added to ``grasp_pos`` to get the above-grasp pre-position.

    Two modes (see ``ObjectSpec.approach_along_tool_z``):

    * ``False`` — world +Z. Legacy "hover straight up by ``approach_dz_m``
      and descend straight down" — only correct when ``grasp_ori`` is
      tool-down.
    * ``True``  — minus the gripper's +tool_z direction
      (``grasp_ori[:, 2]``), scaled by ``approach_dz_m``. The
      above-grasp → grasp segment is therefore along +tool_z (the
      direction the gripper is facing), which is what you want any
      time ``grasp_ori`` has a tilt baked in.
    """
    if approach_along_tool_z:
        tool_z_world = np.asarray(grasp_ori, dtype=np.float64).reshape(3, 3)[:, 2]
        return -tool_z_world * float(approach_dz_m)
    return np.array([0.0, 0.0, float(approach_dz_m)], dtype=np.float64)


def latch_cylinder_grasp_target(
    redis_client,
    valid_points_base: list[np.ndarray],
    *,
    grasp_offset: np.ndarray,
    approach_dz_m: float,
    grasp_target: str = GRASP_TARGET_BODY,
    approach_along_tool_z: bool = False,
    grasp_extra_rot_deg: float = CYLINDER_GRASP_EXTRA_ROT_DEG,
    grasp_extra_rot_axis: str = CYLINDER_GRASP_EXTRA_ROT_AXIS,
) -> LatchedTarget | None:
    """Build the LatchedTarget from the Gemini-derived points.

    ``grasp_target`` selects the geometry rule:

    - ``body``: original cylinder-body grasp. Anchor at the DEEPER of the
      two long-axis points, apply the ``+X`` push-forward offset, and use
      the static ``PARMESAN_DETECTION_EE_ORIENTATION``. The Gemini prompt
      for this mode tells the model to pick two points on the cylindrical
      side wall, but the bottle is vertical so the perpendicular yaw is
      noise — we leave the detection pose alone.
    - ``handle``: grasp a slender handle that has been attached parallel
      to the cylinder (the cylinder is too wide for the jaws). Mirrors
      the pan-handle builder in ``zitibot_tasks.gemini``: position is
      point 1 (Gemini's GRASP point, placed at the handle's grip
      center) and the wrist yaws so the jaws close perpendicular to the
      line between the two handle points. If the handle's centerline is
      basically vertical (axis projects to ~0 in XY), perpendicular yaw
      is undefined and we fall back to the unrotated detection pose.
    """
    if not valid_points_base:
        return None

    if grasp_target == GRASP_TARGET_HANDLE:
        if len(valid_points_base) < 2:
            print("Warning: handle grasp needs 2 points (grasp + axis); got 1")
            return None
        # Pan-style geometry: position = point 1 (the GRASP point picked
        # at the handle's grip center), orientation rotated so the jaws
        # close perpendicular to the handle's long axis.
        p1 = valid_points_base[0]
        p2 = valid_points_base[1]
        raw_contact = p1.copy()
        grasp_pos = raw_contact + np.asarray(grasp_offset, dtype=np.float64).reshape(3)
        # ``above_grasp_pos`` depends on ``grasp_ori`` when
        # ``approach_along_tool_z`` is on, so defer it until after the
        # orientation is finalised below. ``carry_pos`` is a pure
        # world-Z lift independent of orientation, so compute it now.
        carry_pos = grasp_pos + np.array([0.0, 0.0, CYLINDER_CARRY_LIFT_M], dtype=np.float64)

        ori_result = compute_cylinder_grasp_orientation(
            p1, p2, base_ori=PARMESAN_DETECTION_EE_ORIENTATION,
        )
        if ori_result is not None:
            gemini_grasp_ori, cylinder_axis, closing_axis = ori_result
        else:
            # Handle is (nearly) vertical — fall back to the unrotated
            # detection pose, matching how pan_handle/_apply_perpendicular_yaw
            # silently no-ops when the in-plane axis is degenerate.
            gemini_grasp_ori = PARMESAN_DETECTION_EE_ORIENTATION.copy()
            cylinder_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            closing_axis = gemini_grasp_ori[:, 1].copy()
            print("  Handle axis is vertical; using unrotated detection orientation.")

        # ``grasp_ori`` (above-grasp + grasp descent + rack-return
        # descent) gets the extra wrist tilt baked in. ``carry_ori`` is
        # built independently from the UN-rotated Gemini orientation,
        # so the carry attitude is unaffected by the extra wrist tilt
        # — it stays at ``CYLINDER_POST_GRASP_TILT_DEG`` about world Y
        # from the Gemini orientation, exactly as before this rotation
        # was added.
        grasp_ori = _apply_cylinder_grasp_extra_rot(
            gemini_grasp_ori,
            deg=grasp_extra_rot_deg,
            axis=grasp_extra_rot_axis,
        )
        carry_ori = _post_grasp_carry_ori(gemini_grasp_ori)
        # Build the above-grasp now that ``grasp_ori`` is final — the
        # offset direction depends on ``approach_along_tool_z`` (see
        # ``_above_grasp_offset``).
        above_grasp_pos = grasp_pos + _above_grasp_offset(
            approach_dz_m, grasp_ori, approach_along_tool_z
        )

        print(
            f"  Handle axis (diag): "
            f"[{cylinder_axis[0]:+.3f}, {cylinder_axis[1]:+.3f}, {cylinder_axis[2]:+.3f}]"
        )
        print(
            f"  Closing axis (perp yaw applied to detection ori): "
            f"[{closing_axis[0]:+.3f}, {closing_axis[1]:+.3f}, {closing_axis[2]:+.3f}]"
        )
        if abs(grasp_extra_rot_deg) > 1e-9:
            print(
                f"  Extra wrist tilt: {grasp_extra_rot_deg:+.1f}° about world "
                f"+{grasp_extra_rot_axis.upper()} "
                "(applied to above/grasp/release only; carry unaffected)"
            )

        return LatchedTarget(
            grasp_pos=grasp_pos,
            above_grasp_pos=above_grasp_pos,
            carry_pos=carry_pos,
            grasp_ori=grasp_ori,
            carry_ori=carry_ori,
            cylinder_axis=cylinder_axis,
            closing_axis=closing_axis,
        )

    # ``body`` mode (default, unchanged behaviour).
    # Grasp at the DEEPER of the two Gemini long-axis points
    # (whichever has the larger depth from the camera, i.e. the smaller
    # world Z / lower in world). The shallower sample tends to land on
    # the cap / lid edge and reads too high, which would leave the
    # gripper hovering above the bottle body; anchoring to the deeper
    # sample keeps the closure on the cylindrical wall. The
    # ``len == 1`` branch keeps the single-point fallback unchanged.
    # CYLINDER_GRASP_Z_DESCENT_REDUCTION_M is still applied on top.
    raw_contact = valid_points_base[0]
    if len(valid_points_base) >= 2:
        p1 = valid_points_base[0]
        p2 = valid_points_base[1]
        raw_contact = (p1 if p1[2] <= p2[2] else p2).copy()
    grasp_pos = raw_contact + np.asarray(grasp_offset, dtype=np.float64).reshape(3)
    # ``carry_pos`` is a pure world-Z lift; safe to compute up front.
    # ``above_grasp_pos`` is deferred until after ``grasp_ori`` since
    # its offset direction depends on ``approach_along_tool_z``.
    carry_pos = grasp_pos + np.array([0.0, 0.0, CYLINDER_CARRY_LIFT_M], dtype=np.float64)

    # Orientation: keep the exact same pose the arm was holding while
    # Gemini captured the frame (``PARMESAN_DETECTION_EE_ORIENTATION``),
    # then bake in the shared cylinder-grasp extra wrist tilt so body
    # mode matches the handle path. ``grasp_ori`` carries the extra
    # tilt; ``carry_ori`` (built further below) does NOT — same split
    # as the handle branch so the carry attitude stays at the
    # original CYLINDER_POST_GRASP_TILT_DEG about world Y from the
    # Gemini orientation regardless of the extra tilt. The
    # ``compute_cylinder_grasp_orientation`` helper is kept around for
    # diagnostic use (and the handle mode above actually drives the
    # grasp with it); the body-mode call is just gone.
    gemini_grasp_ori = PARMESAN_DETECTION_EE_ORIENTATION.copy()
    grasp_ori = _apply_cylinder_grasp_extra_rot(
        gemini_grasp_ori,
        deg=grasp_extra_rot_deg,
        axis=grasp_extra_rot_axis,
    )
    above_grasp_pos = grasp_pos + _above_grasp_offset(
        approach_dz_m, grasp_ori, approach_along_tool_z
    )

    # Diagnostic-only axes for the UI status panel + logging. These do
    # NOT drive the grasp pose anymore.
    if len(valid_points_base) >= 2:
        axis_xy = (valid_points_base[1] - valid_points_base[0]).copy()
        axis_xy[2] = 0.0
        norm = float(np.linalg.norm(axis_xy))
        cylinder_axis = (axis_xy / norm) if norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        cylinder_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    # The Franka tool-down convention puts the jaw closing direction
    # along the gripper's local +Y, which is column 1 of the rotation
    # matrix expressed in world frame.
    closing_axis = grasp_ori[:, 1].copy()

    print(
        f"  Cylinder axis (diag): "
        f"[{cylinder_axis[0]:+.3f}, {cylinder_axis[1]:+.3f}, {cylinder_axis[2]:+.3f}]"
    )
    print(
        f"  Closing axis (from detection ori): "
        f"[{closing_axis[0]:+.3f}, {closing_axis[1]:+.3f}, {closing_axis[2]:+.3f}]"
    )

    return LatchedTarget(
        grasp_pos=grasp_pos,
        above_grasp_pos=above_grasp_pos,
        carry_pos=carry_pos,
        grasp_ori=grasp_ori,
        carry_ori=_post_grasp_carry_ori(gemini_grasp_ori),
        cylinder_axis=cylinder_axis,
        closing_axis=closing_axis,
    )


def _phase_hint(phase: Phase, *, headless: bool = False) -> str:
    hints = {
        Phase.VISION_READY: "ENTER = detect cylinder strip with Gemini",
        Phase.VISION_LATCHED: "ENTER = move above detected grasp",
        Phase.ABOVE_GRASP: "ENTER = descend to grasp pose",
        Phase.AT_GRASP: "ENTER = pregrasp + close gripper",
        Phase.GRASPED: "ENTER = lift and move to pour station",
        Phase.LIFTED: "Moving to pour station... (auto-completes)",
        Phase.MOVING_TO_POUR: "Moving to pour station... (auto-completes)",
        Phase.AT_POUR: "ENTER = cartesian pretilt to pour pose",
        Phase.POUR_PRETILT: "ENTER = dump (flip J7)",
        Phase.POUR_DUMPED: (
            f"ENTER = J4 shake "
            f"{len(CYLINDER_SHAKE_J4_STROKES)}-stroke sequence (joint): "
            + ", ".join(
                f"{lbl} {d:+.0f}°" for lbl, d in CYLINDER_SHAKE_J4_STROKES
            )
        ),
        Phase.SHAKE_LOWERED: "ENTER = J7 flip back up + return to carry pose",
        Phase.POUR_RESTORED: "Moving to rack... (auto-completes)",
        Phase.MOVING_TO_RACK: "Moving to rack... (auto-completes)",
        Phase.PLACING: "ENTER = open gripper",
        Phase.AT_RELEASE: "ENTER = done",
        Phase.DONE: "Done — q to quit",
    }
    return hints.get(phase, "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "OptiTrack base + Gemini vision cylinder grasp + pour + return "
            "(parmesan / sauce / ricotta)."
        )
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate setup/base motions; vision/grasp/tilt phases are always ENTER-gated.",
    )
    p.add_argument("--ui", action="store_true", help="Show OpenCV window.")
    p.add_argument(
        "--cylinder",
        choices=tuple(_CYLINDER_SEQUENCES.keys()),
        default=DEFAULT_CYLINDER_CHOICE,
        help=(
            "Which cylinder(s) to grasp / pour. ``all`` (default) runs "
            "parmesan → sauce → ricotta at RACK_STATION. ``parmesan`` pours "
            "into the pan; ``sauce`` and ``ricotta`` pour into the mixing bowl."
        ),
    )
    p.add_argument("--prompt", default=None, help="Custom Gemini prompt (overrides per-cylinder default).")
    p.add_argument(
        "--grasp-target",
        choices=GRASP_TARGET_CHOICES,
        default=GRASP_TARGET_HANDLE,
        help=(
            "What Gemini should look at and how to compute the grasp:\n"
            "  handle (default) = pick 2 points on the cylinder's colored "
            "strip (blue / gray / red per --cylinder), grasp like the pan "
            "handle (point 1 + perpendicular yaw)\n"
            "  body             = pick 2 points on the cylinder long axis, "
            "grasp the body (deeper point, no yaw)"
        ),
    )
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="Do not drive to RACK_STATION / detection pose before each cycle.",
    )
    p.add_argument(
        "--endeffector-transform-key",
        default=DEFAULT_ENDEFFECTOR_TRANSFORM_KEY,
        help="Redis key for the 4x4 base->flange transform.",
    )
    p.add_argument(
        "--gemini-response-path",
        default=None,
        help=(
            "Where to save the annotated Gemini RGB+depth response image. "
            "If unset, defaults per cylinder from ``CYLINDER_GEMINI_RESPONSE_PATHS``."
        ),
    )
    p.add_argument("--depth-patch-radius", type=int, default=2)
    p.add_argument("--no-grasp-offset", action="store_true")
    # Per-axis grasp offset overrides. Default is None so we can fall
    # back to the per-mode default in ``grasp_offset_world``: body uses
    # the shared ``DEFAULT_GRASP_OFFSET_M``; handle uses the per-cylinder
    # ``OBJECT_DEFAULTS[CYLINDER_OBJECT_KEY[cylinder]].gemini_world_offset_m``.
    p.add_argument(
        "--grasp-offset-x", type=float, default=None,
        help=(
            "Override grasp offset X (m). Default depends on --grasp-target: "
            f"body={DEFAULT_GRASP_OFFSET_M[0]:+.4f} / "
            "handle=per-cylinder gemini_world_offset_m[X] (see OBJECT_DEFAULTS)."
        ),
    )
    p.add_argument(
        "--grasp-offset-y", type=float, default=None,
        help=(
            "Override grasp offset Y (m). Default depends on --grasp-target: "
            f"body={DEFAULT_GRASP_OFFSET_M[1]:+.4f} / "
            "handle=per-cylinder gemini_world_offset_m[Y] (-Y = right "
            "in Franka base frame; see OBJECT_DEFAULTS)."
        ),
    )
    p.add_argument(
        "--grasp-offset-z", type=float, default=None,
        help=(
            "Override grasp offset Z (m). Default depends on --grasp-target: "
            f"body={DEFAULT_GRASP_OFFSET_M[2]:+.4f} / "
            "handle=per-cylinder gemini_world_offset_m[Z] (see OBJECT_DEFAULTS)."
        ),
    )
    p.add_argument("--model", default=gp.DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--timeout-ms", type=int, default=10000)
    p.add_argument("--approach-dz", type=float, default=OBJECT_DEFAULTS[Object.BOTTLE].approach_dz)
    p.add_argument("--pour-tilt-deg", type=float, default=90.0)
    p.add_argument("--pour-axis", default="y", choices=("x", "y"))
    p.add_argument("--tilt-duration-s", type=float, default=DEFAULT_TILT_DURATION_S)
    p.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED)
    p.add_argument("--gripper-force", type=float, default=OBJECT_DEFAULTS[Object.BOTTLE].force)
    p.add_argument("--gripper-pregrasp-width", type=float, default=DEFAULT_GRIPPER_PREGRASP_WIDTH)
    p.add_argument(
        "--gripper-close-width",
        type=float,
        # Default 0.0 = close all the way. The Franka gripper will keep
        # closing until it hits the held object and applies grasp_force,
        # so 0.0 just removes the artificial 1.5 cm floor previously
        # inherited from OBJECT_DEFAULTS[Object.BOTTLE].close_width. The
        # blue-strip handle is much thinner than the cylinder body, so
        # leaving any positive close width means the jaws never actually
        # contact the handle and the grasp slips on lift.
        default=0.0,
    )
    p.add_argument("--gripper-pregrasp-settle", type=float, default=DEFAULT_GRIPPER_PREGRASP_SETTLE_S)
    p.add_argument("--gripper-grasp-settle", type=float, default=DEFAULT_GRIPPER_GRASP_SETTLE_S)
    return p.parse_args()


def default_motion_params(ctx) -> MotionParams:
    """Motion/gripper defaults matching ``main()`` for embedded callers."""
    open_w = resolve_gripper_open_width(ctx.redis, None)
    return MotionParams(
        approach_dz_m=OBJECT_DEFAULTS[Object.BOTTLE].approach_dz,
        pour_tilt_deg=90.0,
        pour_axis="y",
        tilt_duration_s=DEFAULT_TILT_DURATION_S,
        gripper_open_width=open_w,
        gripper_pregrasp_width=DEFAULT_GRIPPER_PREGRASP_WIDTH,
        gripper_close_width=0.0,
        gripper_speed=DEFAULT_GRIPPER_SPEED,
        gripper_force=OBJECT_DEFAULTS[Object.BOTTLE].force,
        gripper_pregrasp_settle_s=DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
        gripper_grasp_settle_s=DEFAULT_GRIPPER_GRASP_SETTLE_S,
    )


def build_cycle_options(
    ctx,
    *,
    ui: bool = False,
    skip_base: bool = False,
    grasp_target: str = GRASP_TARGET_HANDLE,
    prompt: str | None = None,
    gemini_response_path: str | None = None,
    model: str | None = None,
    temperature: float = 0.5,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    warmup_frames: int = 30,
    timeout_ms: int = 10000,
    depth_patch_radius: int = 2,
    no_grasp_offset: bool = False,
    grasp_offset_x: float | None = None,
    grasp_offset_y: float | None = None,
    grasp_offset_z: float | None = None,
    pour_tilt_deg: float = 90.0,
    pour_axis: str = "y",
    tilt_duration_s: float = DEFAULT_TILT_DURATION_S,
    gripper_speed: float = DEFAULT_GRIPPER_SPEED,
    gripper_force: float = OBJECT_DEFAULTS[Object.BOTTLE].force,
    gripper_pregrasp_width: float = DEFAULT_GRIPPER_PREGRASP_WIDTH,
    gripper_close_width: float = 0.0,
    gripper_pregrasp_settle: float = DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    gripper_grasp_settle: float = DEFAULT_GRIPPER_GRASP_SETTLE_S,
    approach_dz: float = OBJECT_DEFAULTS[Object.BOTTLE].approach_dz,
    endeffector_transform_key: str | None = None,
) -> argparse.Namespace:
    """Build an argparse-like namespace for one cylinder cycle."""
    ee_key = endeffector_transform_key
    if ee_key is None:
        ee_key = getattr(ctx, "endeffector_transform_key", None) or DEFAULT_ENDEFFECTOR_TRANSFORM_KEY
    return argparse.Namespace(
        ui=ui,
        skip_base=skip_base,
        grasp_target=grasp_target,
        prompt=prompt,
        gemini_response_path=gemini_response_path,
        model=model or gp.DEFAULT_MODEL,
        temperature=temperature,
        width=width,
        height=height,
        fps=fps,
        warmup_frames=warmup_frames,
        timeout_ms=timeout_ms,
        depth_patch_radius=depth_patch_radius,
        no_grasp_offset=no_grasp_offset,
        grasp_offset_x=grasp_offset_x,
        grasp_offset_y=grasp_offset_y,
        grasp_offset_z=grasp_offset_z,
        pour_tilt_deg=pour_tilt_deg,
        pour_axis=pour_axis,
        tilt_duration_s=tilt_duration_s,
        gripper_speed=gripper_speed,
        gripper_force=gripper_force,
        gripper_pregrasp_width=gripper_pregrasp_width,
        gripper_close_width=gripper_close_width,
        gripper_pregrasp_settle=gripper_pregrasp_settle,
        gripper_grasp_settle=gripper_grasp_settle,
        approach_dz=approach_dz,
        endeffector_transform_key=ee_key,
    )


def _acquire_realsense(ctx, args: argparse.Namespace):
    """Return (pipeline, align, depth_scale, intrinsics, owns_pipeline)."""
    if hasattr(ctx, "realsense"):
        try:
            pipeline, align, depth_scale, intrinsics = ctx.realsense()
            return pipeline, align, depth_scale, intrinsics, False
        except Exception:
            pass
    pipeline, align, depth_scale, intrinsics = rs_cam.start_realsense(
        args.width, args.height, args.fps, args.warmup_frames, args.timeout_ms,
    )
    return pipeline, align, depth_scale, intrinsics, True


def grasp_offset_world(args: argparse.Namespace, cylinder: Cylinder) -> np.ndarray:
    """World-frame grasp offset, picked per grasp_target with optional overrides.

    Body-mode default is the shared ``DEFAULT_GRASP_OFFSET_M`` (just a
    +X forward nudge). Handle-mode default is per-cylinder, pulled from
    ``OBJECT_DEFAULTS[CYLINDER_OBJECT_KEY[cylinder]].gemini_world_offset_m``
    so each cylinder can be tuned independently in
    ``zitibot_core.constants``. ``--grasp-offset-x/y/z`` CLI flags
    override individual axes when passed.
    """
    if args.no_grasp_offset or not ENABLE_GRASP_OFFSET:
        return np.zeros(3, dtype=np.float64)
    if args.grasp_target == GRASP_TARGET_HANDLE:
        defaults = OBJECT_DEFAULTS[CYLINDER_OBJECT_KEY[cylinder]].gemini_world_offset_m
    else:
        defaults = DEFAULT_GRASP_OFFSET_M
    x = defaults[0] if args.grasp_offset_x is None else args.grasp_offset_x
    y = defaults[1] if args.grasp_offset_y is None else args.grasp_offset_y
    z = defaults[2] if args.grasp_offset_z is None else args.grasp_offset_z
    return np.array([x, y, z], dtype=np.float64)


def _fmt_xyz(label: str, v: np.ndarray) -> list[str]:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return [label, f"x={v[0]:+.4f}  y={v[1]:+.4f}  z={v[2]:+.4f}"]


def _status_lines(phase: Phase, target: LatchedTarget | None) -> list[str]:
    if target is None:
        return ["No target latched", "ENTER = detect cylinder with Gemini"]
    lines = [
        f"Phase: {phase.value}",
        *_fmt_xyz("Grasp contact", target.grasp_pos),
        *_fmt_xyz("Above grasp", target.above_grasp_pos),
        f"Cylinder axis: [{target.cylinder_axis[0]:+.3f}, {target.cylinder_axis[1]:+.3f}, {target.cylinder_axis[2]:+.3f}]",
        f"Closing axis:  [{target.closing_axis[0]:+.3f}, {target.closing_axis[1]:+.3f}, {target.closing_axis[2]:+.3f}]",
    ]
    return lines


def _render_text_band(width: int, height: int, lines: list[str]) -> np.ndarray:
    band = np.full((height, width, 3), (28, 28, 32), dtype=np.uint8)
    y = int(22 * 3.0 * _TEXT_SIZE_MULT)
    font = cv2.FONT_HERSHEY_SIMPLEX
    approx_char_px = int(8 * 3.0 * _TEXT_SIZE_MULT)
    max_chars = max(12, (width - 24) // max(1, approx_char_px))
    for line in lines:
        if not line:
            y += _TEXT_EMPTY_SKIP
            continue
        disp = line if len(line) <= max_chars else line[: max_chars - 3] + "..."
        cv2.putText(
            band, disp, (16, y), font, _TEXT_FONT_SCALE, (230, 230, 235),
            _TEXT_THICKNESS, cv2.LINE_AA,
        )
        y += _TEXT_LINE_STEP
        if y > height - int(12 * _TEXT_SIZE_MULT):
            break
    return band


def _gemini_placeholder_panel(h: int, w: int) -> np.ndarray:
    panel = np.full((h, w, 3), (48, 48, 52), dtype=np.uint8)
    cv2.putText(panel, "Press ENTER", (max(10, w // 2 - 120), h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 200), 2, cv2.LINE_AA)
    cv2.putText(panel, "Gemini + depth", (max(10, w // 2 - 130), h // 2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 140, 160), 1, cv2.LINE_AA)
    return panel


def _save_gemini_response(
    overlay: np.ndarray | None,
    depth_vis: np.ndarray,
    save_path: Path,
) -> None:
    if overlay is None:
        return
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ov = overlay
        dv = depth_vis
        if ov.shape[:2] != dv.shape[:2]:
            dv = cv2.resize(dv, (ov.shape[1], ov.shape[0]))
        composite = np.hstack((ov, dv))
        cv2.imwrite(str(save_path), composite)
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  Failed to save: {e}", file=sys.stderr)


def do_gemini_capture(
    triple: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    redis_client,
    args: argparse.Namespace,
    cylinder: Cylinder,
    motion: MotionParams,
    gemini_client,
    prompt: str,
    color_intrinsics,
    save_path: Path,
) -> tuple[LatchedTarget | None, np.ndarray | None]:
    color_bgr, depth_m, depth_vis = triple

    overlay, points_camera, _patches = gp.query_color_depth_overlay(
        gemini_client,
        args.model,
        prompt,
        args.temperature,
        color_bgr,
        depth_m,
        color_intrinsics,
        args.depth_patch_radius,
    )

    if points_camera is None or len(points_camera) == 0:
        _save_gemini_response(overlay, depth_vis, save_path)
        return None, overlay

    T_base_flange = read_T_base_flange(redis_client, args.endeffector_transform_key)
    if T_base_flange is None:
        print(f"Could not read flange pose from Redis.")
        _save_gemini_response(overlay, depth_vis, save_path)
        return None, overlay

    valid_points_base = []
    for p_cam in points_camera:
        if p_cam is not None:
            valid_points_base.append(camera_point_to_base(T_base_flange, p_cam))

    if not valid_points_base:
        print("No valid 3D points detected (depth issues)")
        _save_gemini_response(overlay, depth_vis, save_path)
        return None, overlay

    offset = grasp_offset_world(args, cylinder)
    # Per-cylinder grasp orientation knobs pulled from the shared
    # ObjectSpec so each cylinder is independent:
    #   * ``approach_along_tool_z`` — descend along +tool_z (True) vs.
    #     straight down in world Z (False). Required True whenever a
    #     wrist tilt is baked in. Sauce / ricotta = True; parmesan =
    #     False (kept tool-down for the strip grasp).
    #   * ``grasp_extra_rot_deg`` / ``grasp_extra_rot_axis`` — world
    #     rotation premultiplied onto the Gemini grasp orientation
    #     for above/grasp/release (carry pose unchanged). Sauce /
    #     ricotta = -30° about world Y; parmesan = 0° (no extra tilt).
    # Edit per-cylinder in the cylinder block of ``zitibot_core.constants``.
    cyl_spec = OBJECT_DEFAULTS[CYLINDER_OBJECT_KEY[cylinder]]
    target = latch_cylinder_grasp_target(
        redis_client,
        valid_points_base,
        grasp_offset=offset,
        approach_dz_m=motion.approach_dz_m,
        grasp_target=args.grasp_target,
        approach_along_tool_z=cyl_spec.approach_along_tool_z,
        grasp_extra_rot_deg=cyl_spec.grasp_extra_rot_deg,
        grasp_extra_rot_axis=cyl_spec.grasp_extra_rot_axis,
    )
    if target is None:
        _save_gemini_response(overlay, depth_vis, save_path)
        return None, overlay

    # Apply Z descent reduction (lift the grasp target up) to prevent collision
    if CYLINDER_GRASP_Z_DESCENT_REDUCTION_M > 0:
        target.grasp_pos[2] += CYLINDER_GRASP_Z_DESCENT_REDUCTION_M
        target.above_grasp_pos[2] += CYLINDER_GRASP_Z_DESCENT_REDUCTION_M
        target.carry_pos[2] += CYLINDER_GRASP_Z_DESCENT_REDUCTION_M
        print(f"  Reduced Z descent by {CYLINDER_GRASP_Z_DESCENT_REDUCTION_M:.4f} m")

    print(f"Latched grasp contact: {target.grasp_pos.tolist()}")
    print(f"Latched above grasp:  {target.above_grasp_pos.tolist()}")
    print(f"Latched carry pose:   {target.carry_pos.tolist()}")

    # Overlay the final commanded control point on the saved image so it's
    # clear where the latched grasp landed relative to the two raw Gemini
    # long-axis points (and after CYLINDER_GRASP_Z_DESCENT_REDUCTION_M has
    # been applied). Project base/world -> color-frame pixel and drop a
    # diamond marker on both the color and depth panels.
    grasp_pixel = base_point_to_pixel(T_base_flange, target.grasp_pos, color_intrinsics)
    above_pixel = base_point_to_pixel(T_base_flange, target.above_grasp_pos, color_intrinsics)
    overlay_annotated = gp.draw_world_marker(overlay, grasp_pixel, label="grasp")
    overlay_annotated = gp.draw_world_marker(
        overlay_annotated, above_pixel, label="above", color=(0, 200, 255)
    )
    depth_vis_annotated = gp.draw_world_marker(depth_vis, grasp_pixel, label="grasp")
    depth_vis_annotated = gp.draw_world_marker(
        depth_vis_annotated, above_pixel, label="above", color=(0, 200, 255)
    )
    _save_gemini_response(overlay_annotated, depth_vis_annotated, save_path)
    return target, overlay_annotated


def _grab_fresh_frame(pipeline, align, depth_scale, timeout_ms, miss_counter, *, drain=5):
    triple = None
    for _ in range(max(1, drain)):
        triple = rs_cam.next_rgbd_frame(pipeline, align, depth_scale, timeout_ms, miss_counter, max_misses=10)
        if triple is not None:
            break
    return triple


def _advance_phase(
    phase: Phase,
    target: LatchedTarget,
    cylinder: Cylinder,
    slerp: OrientationSlerpState | None,
    poured_ori: np.ndarray | None,
    *,
    redis_client,
    motion: MotionParams,
    args: argparse.Namespace,
    ctx,
    gain_state: dict | None = None,
) -> tuple[Phase, OrientationSlerpState | None, np.ndarray | None]:
    """Advance the phase machine for ``cylinder``.

    ``gain_state`` is the mutable bag that holds the cart-position
    gain snapshot across phases: it gets populated on
    ``Phase.GRASPED`` entry (after the gripper closes — the cylinder
    is now held) and drained on ``Phase.AT_RELEASE`` exit (after the
    gripper opens — load is gone). Defaults to an empty dict for
    backwards-compat with any caller that doesn't carry one across
    invocations, but the outer ``run_headless`` / ``run_live`` loops
    pass their own dict so the boost survives between calls AND so
    they can restore from it in their ``finally`` block on crash /
    Ctrl-C.
    """
    if gain_state is None:
        gain_state = {}
    pour_station = CYLINDER_POUR_STATION[cylinder]

    if phase == Phase.VISION_LATCHED:
        arm.move_to(
            ctx,
            target.above_grasp_pos,
            target.grasp_ori,
            label="[1] Move above detected grasp",
            tol_m=CYLINDER_HOME_TOL_M,
            gated=False,
        )
        return Phase.ABOVE_GRASP, slerp, poured_ori

    if phase == Phase.ABOVE_GRASP:
        arm.move_to(
            ctx,
            target.grasp_pos,
            target.grasp_ori,
            label="[2] Descend to grasp pose",
            tol_m=CYLINDER_PRETILT_TOL_M,
            gated=False,
        )
        return Phase.AT_GRASP, slerp, poured_ori

    if phase == Phase.AT_GRASP:
        print("[3] Pregrasp + grasp...")
        _do_grasp_object(redis_client, motion)
        return Phase.GRASPED, slerp, poured_ori

    if phase == Phase.GRASPED:
        # Cylinder is now held — apply the per-cylinder cartesian
        # position-PID boost BEFORE the lift so the controller has
        # the extra ``ki`` (and modest ``kp`` / ``kv`` bump) needed
        # to fight gravity sag on the way up. Skipped automatically
        # when all of ``lift_cart_position_{kp,kv,ki}`` are ``None``
        # (parmesan / ricotta default). Snapshot is stored in
        # ``gain_state["snapshot"]`` so the outer loop's ``finally``
        # can also restore it on interrupt — and so
        # ``Phase.AT_RELEASE`` can find it.
        cyl_spec = OBJECT_DEFAULTS[CYLINDER_OBJECT_KEY[cylinder]]
        boost_kp = cyl_spec.lift_cart_position_kp
        boost_kv = cyl_spec.lift_cart_position_kv
        boost_ki = cyl_spec.lift_cart_position_ki
        if (
            (boost_kp is not None or boost_kv is not None or boost_ki is not None)
            and "snapshot" not in gain_state
        ):
            gain_state["snapshot"] = gains.apply_cart_position_boost(
                redis_client,
                kp=boost_kp,
                kv=boost_kv,
                ki=boost_ki,
                label=f"{cylinder.value} held",
            )

        # Heavier hold: tighter tolerance + longer timeout so the
        # integral term has time to wind up to the gravity-comp
        # level (a few hundred ms typically) before the move is
        # declared done. Light cylinders (no ki boost) get the old
        # loose tolerance + ``arm.move_to``'s default timeout (4 s)
        # — they don't need the extra settle time and a tight
        # tolerance there only buys idle wait at the end of an
        # already-converged move.
        is_heavy = boost_ki is not None
        lift_kwargs: dict = {
            "label": (
                f"[4] Lift to carry pose (+{CYLINDER_CARRY_LIFT_M * 100:.1f} cm "
                f"+ {CYLINDER_POST_GRASP_TILT_DEG:+.0f}° about world "
                f"+{CYLINDER_POST_GRASP_TILT_AXIS.upper()})"
            ),
            "tol_m": CYLINDER_PRETILT_TOL_M if is_heavy else CYLINDER_HOME_TOL_M,
            "gated": False,
        }
        if is_heavy:
            lift_kwargs["timeout_s"] = CYLINDER_PRETILT_TIMEOUT_S
        arm.move_to(ctx, target.carry_pos, target.carry_ori, **lift_kwargs)
        return Phase.LIFTED, slerp, poured_ori

    if phase == Phase.LIFTED:
        print(f"[5] Moving base to {pour_station.name}...")
        base.go_to_pose(ctx, pour_station)
        return Phase.MOVING_TO_POUR, slerp, poured_ori

    if phase == Phase.MOVING_TO_POUR:
        print(f"[6] At {pour_station.name}. Ready to tilt.")
        return Phase.AT_POUR, slerp, poured_ori

    if phase == Phase.AT_POUR:
        # Cartesian pretilt: either drive to a hand-taught pose
        # (parmesan) or rotate the live carry-pose orientation by a
        # world-frame angle in place (sauce / ricotta). See
        # CYLINDER_PRETILT_SPEC.
        spec = CYLINDER_PRETILT_SPEC[cylinder]
        if spec.taught_pos is not None and spec.taught_ori is not None:
            pretilt_pos = spec.taught_pos.copy()
            pretilt_ori = spec.taught_ori
            print(
                f"[7] Cartesian pretilt to taught pour pose pos="
                f"{pretilt_pos.tolist()}"
            )
        else:
            cur = read_current_ee_world(redis_client)
            if cur is None:
                raise RuntimeError(
                    "[7] cannot read current EE pose; aborting pretilt."
                )
            cur_pos, cur_ori = cur
            R_world = R.from_euler(
                spec.rotate_axis, spec.rotate_deg, degrees=True
            ).as_matrix()
            pretilt_pos = cur_pos.copy()
            pretilt_ori = R_world @ cur_ori
            print(
                f"[7] Cartesian pretilt by rotating live EE pose "
                f"{spec.rotate_deg:+.1f}° about world "
                f"+{spec.rotate_axis.upper()} (pos held at {pretilt_pos.tolist()})"
            )
        if spec.world_x_offset_m or spec.world_z_offset_m:
            pretilt_pos[0] += spec.world_x_offset_m
            pretilt_pos[2] += spec.world_z_offset_m
            print(
                f"[7] pretilt world offsets "
                f"X{spec.world_x_offset_m * 100:+.1f} cm "
                f"Z{spec.world_z_offset_m * 100:+.1f} cm "
                f"-> pos={pretilt_pos.tolist()}"
            )
        arm.move_to(
            ctx,
            pretilt_pos,
            pretilt_ori,
            label=f"  [arm] {cylinder.value} pour pretilt (cartesian)",
            tol_m=CYLINDER_PRETILT_TOL_M,
            timeout_s=CYLINDER_PRETILT_TIMEOUT_S,
            gated=False,
        )
        # Stash joints AFTER the Cartesian pretilt converged. POUR_DUMPED
        # will restore the joints touched by the DUMP step (J7) to these
        # values — we want to revert J7 to where the Cartesian pretilt
        # left it, not to where it was at AT_POUR entry (the carry pose),
        # so the wrist returns to the pour pose orientation before the
        # carry-pose Cartesian republish swings it back home.
        q_post_pretilt = arm.read_joint_positions(redis_client)
        if q_post_pretilt is None:
            print(
                "[7] WARNING: joint positions unavailable after Cartesian "
                "pretilt; POUR_DUMPED will skip J7 restore."
            )
        else:
            poured_ori = q_post_pretilt.copy()
        # Any joint-space pretilts queued in CYLINDER_PRETILT_JOINTS_DEG
        # run after the Cartesian pretilt. This is currently empty (J6
        # was moved to Cartesian above); kept for future tuning.
        for j, v in CYLINDER_PRETILT_JOINTS_DEG.items():
            arm.move_to_joints_partial(
                ctx,
                {j: v},
                degrees=True,
                label=f"  [arm] pour pretilt J{j}={v:+.1f}°",
                gated=False,
            )
        return Phase.POUR_PRETILT, slerp, poured_ori

    if phase == Phase.POUR_PRETILT:
        # Warm-up: tiny J7 nudge to force the cartesian→joint controller
        # swap to take effect on a cheap move first. Without this, the
        # FIRST joint command after the AT_POUR cartesian pretilt tends
        # to get swallowed by the controller swap and the big dump never
        # actually moves J7. See CYLINDER_DUMP_WARMUP_DELTA_DEG.
        q_pre_dump = arm.read_joint_positions(redis_client)
        if q_pre_dump is not None and q_pre_dump.size >= 7:
            q7_warmup_rad = float(q_pre_dump[6]) + math.radians(
                CYLINDER_DUMP_WARMUP_DELTA_DEG
            )
            print(
                f"[8a] J7 warm-up nudge to "
                f"{math.degrees(q7_warmup_rad):+.1f}° "
                f"(forces cartesian→joint controller swap)"
            )
            arm.move_to_joints_partial(
                ctx,
                {7: q7_warmup_rad},
                degrees=False,
                label=(
                    f"  [arm] pour dump warm-up J7"
                    f"={math.degrees(q7_warmup_rad):+.1f}°"
                ),
                tol_rad=CYLINDER_DUMP_WARMUP_TOL_RAD,
                timeout_s=CYLINDER_DUMP_WARMUP_TIMEOUT_S,
                gated=False,
            )
        else:
            print(
                "[8a] WARNING: joint positions unavailable; skipping J7 "
                "warm-up nudge before dump."
            )

        dump_str = ", ".join(
            f"J{j}={v:+.1f}°" for j, v in CYLINDER_DUMP_JOINTS_DEG.items()
        )
        print(f"[8b] Joint dump: {dump_str}")
        arm.move_to_joints_partial(
            ctx,
            CYLINDER_DUMP_JOINTS_DEG,
            degrees=True,
            label=f"  [arm] pour dump {dump_str}",
            gated=False,
        )
        return Phase.POUR_DUMPED, slerp, poured_ori

    if phase == Phase.POUR_DUMPED:
        # Joint-space J4 shake. Stays in joint_controller so the
        # J7=-151° dump goal stays parked (move_to_joints_partial
        # only overwrites J4 in the published full-joint goal). Each
        # stroke in CYLINDER_SHAKE_J4_STROKES is RELATIVE to the live
        # joint reading at the start of that stroke. The last stroke
        # in the sequence doubles as the "lift away" that pulls the
        # bottle opening above the bowl rim before SHAKE_LOWERED
        # runs the wrist flip-back — pick the final delta accordingly.
        n_strokes = len(CYLINDER_SHAKE_J4_STROKES)
        for idx, (direction, delta_deg) in enumerate(
            CYLINDER_SHAKE_J4_STROKES, start=1
        ):
            q_now = arm.read_joint_positions(redis_client)
            if q_now is None or q_now.size < 7:
                print(
                    f"[9-shake {idx}/{n_strokes} {direction}] "
                    f"WARNING: joint positions unavailable; aborting shake."
                )
                return Phase.SHAKE_LOWERED, slerp, poured_ori
            q4_now = float(q_now[3])
            q4_goal = q4_now + math.radians(delta_deg)
            print(
                f"[9-shake {idx}/{n_strokes} {direction}] "
                f"J4 {math.degrees(q4_now):+.1f}° → "
                f"{math.degrees(q4_goal):+.1f}° "
                f"({delta_deg:+.1f}°)"
            )
            arm.move_to_joints_partial(
                ctx,
                {4: q4_goal},
                degrees=False,
                label=(
                    f"  [arm] shake {idx} {direction} "
                    f"J4={math.degrees(q4_goal):+.1f}°"
                ),
                tol_rad=CYLINDER_SHAKE_J4_TOL_RAD,
                timeout_s=CYLINDER_SHAKE_J4_TIMEOUT_S,
                gated=False,
            )
        return Phase.SHAKE_LOWERED, slerp, poured_ori

    if phase == Phase.SHAKE_LOWERED:
        # Restore: flip J7 back up, then re-publish the carry-pose
        # Cartesian so the arm returns to transport-safe configuration
        # before the base drives to the rack.
        #
        # No J7 warmup needed here: the shake stayed in joint_controller
        # the whole time, so the J7 restore is just another in-controller
        # goal update (not a cartesian→joint swap like before).
        print("[9] Restoring pre-dump joints + returning to carry pose...")
        if isinstance(poured_ori, np.ndarray) and poured_ori.size >= 7:
            q_stash = poured_ori[:7]
            # Revert the joints actually disturbed by the DUMP /
            # joint-space PRETILT steps:
            #   * J7 — flipped by the dump goal in CYLINDER_DUMP_JOINTS_DEG.
            #   * Any joints queued in CYLINDER_PRETILT_JOINTS_DEG
            #     (currently empty — kept in the union for future tuning).
            # Single-joint moves to avoid the simultaneous-step FR3
            # reflex we dodge in AT_POUR.
            touched = sorted(
                set(CYLINDER_PRETILT_JOINTS_DEG)
                | set(CYLINDER_DUMP_JOINTS_DEG),
                reverse=True,
            )
            for j in touched:
                v = float(q_stash[j - 1])
                arm.move_to_joints_partial(
                    ctx,
                    {j: v},
                    degrees=False,
                    label=f"  [arm] restore J{j}={math.degrees(v):+.1f}°",
                    gated=False,
                )
        else:
            print("  WARNING: no pre-dump joint stash; skipping J7 restore.")
        # Cartesian return to the carry pose. Without this, the arm
        # stays at the pour pose while the base drives back to the rack
        # (because the joint restore above only touched J7, not J6 / the
        # cartesian pretilt). Re-publishing carry_pos + carry_ori sends
        # IK back to the transport-safe pose we held just before AT_POUR
        # (the +3 cm / +45° about Y post-grasp pose).
        print("  [arm] return to carry pose (cartesian)")
        arm.move_to(
            ctx,
            target.carry_pos,
            target.carry_ori,
            label="  [arm] post-pour return to carry pose",
            tol_m=CYLINDER_PRETILT_TOL_M,
            timeout_s=CYLINDER_PRETILT_TIMEOUT_S,
            gated=False,
        )
        return Phase.POUR_RESTORED, slerp, poured_ori

    if phase == Phase.POUR_RESTORED:
        print("[10] Pre-pour joints restored, ready to drive base back.")
        return Phase.MOVING_TO_RACK, slerp, poured_ori

    if phase == Phase.MOVING_TO_RACK:
        print("[10] Moving base back to rack station...")
        base.go_to_pose(ctx, BaseWaypoint.RACK_STATION)
        return Phase.PLACING, slerp, poured_ori

    if phase == Phase.PLACING:
        arm.move_to(
            ctx,
            target.grasp_pos,
            target.grasp_ori,
            label="[11] Descending to original place pose",
            tol_m=CYLINDER_PRETILT_TOL_M,
            gated=False,
        )
        return Phase.AT_RELEASE, slerp, poured_ori

    if phase == Phase.AT_RELEASE:
        print("[12] Open gripper...")
        _do_open_gripper(redis_client, motion)
        # Give the gripper a full second to fully open + let the cylinder
        # settle in the rack before the next cycle yanks the arm up to
        # ARM_HOME — without this, fingers can catch the rim on the lift.
        time.sleep(1.0)
        # Load is gone — restore the pre-grasp cart position gains.
        # ``pop`` so the outer loop's ``finally`` doesn't double-restore.
        snapshot = gain_state.pop("snapshot", None)
        if snapshot is not None:
            gains.restore_cart_position_gains(
                redis_client, snapshot, label=f"{cylinder.value} released"
            )
        return Phase.DONE, slerp, poured_ori

    return phase, slerp, poured_ori


def build_cylinder_prompt(object_name: str | None, custom_prompt: str | None) -> str:
    if custom_prompt:
        return custom_prompt
    obj = object_name or "bottle"
    return (
        f"In the image, locate the **{obj}**. "
        f"Draw the visible LONG AXIS of the cylindrical container by choosing "
        f"TWO distinct points on the cylinder body:\n"
        f"  - Both points must lie on the visible cylindrical side surface, "
        f"not on the cap, lid, top, bottom, label edge, or background.\n"
        f"  - The line from point 1 to point 2 must run ALONG the longer axis "
        f"of the cylinder as it appears in the image, not across its width.\n"
        f"  - Put the points a few centimetres apart, roughly centered on the "
        f"body, so their midpoint is a good grasp target.\n"
        f"\n"
        f"The midpoint of the two points is the grasp target. The grasp target "
        f"must stay ON this line, and the gripper will close perpendicular to "
        f"the line between the points.\n"
        f"\n"
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{obj}_axis_1"}}, '
        f'{{"point": [y, x], "label": "{obj}_axis_2"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


def build_handle_strip_prompt(color: str, custom_prompt: str | None) -> str:
    """Prompt: two points lengthwise on a colored strip wrapped around the cylinder."""
    if custom_prompt:
        return custom_prompt
    color_l = color.lower()
    return (
        f"Find the **{color.upper()} strip** in the image. It is a piece of "
        f"{color_l}-colored tape / strap / plastic wrapped around the "
        f"cylinder, and it is the actual grasp target.\n"
        "\n"
        f"Pick TWO distinct points that BOTH lie on this {color.upper()} "
        f"region, spaced a few centimetres apart along the strip's LONGER "
        f"dimension (lengthwise, not across its width).\n"
        "\n"
        f"STRICT RULES — both points MUST be on actual {color_l} strip pixels. "
        "Do NOT place points on:\n"
        "  - the cylinder label or bare body (any non-strip surface)\n"
        "  - the cap / lid\n"
        "  - the cardboard cradle or the table\n"
        "  - any other object in the scene\n"
        f"If a candidate point is not visibly {color_l}, do not return it.\n"
        "\n"
        "Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{color_l}_strip_1"}}, '
        f'{{"point": [y, x], "label": "{color_l}_strip_2"}}]\n'
        "Coordinates must be normalized 0-1000 in [y, x] order."
    )


def resolve_prompt(args: argparse.Namespace, cylinder: Cylinder) -> str:
    """Pick the right prompt builder for ``args.grasp_target`` and ``cylinder``."""
    if args.grasp_target == GRASP_TARGET_HANDLE:
        return build_handle_strip_prompt(CYLINDER_HANDLE_COLOR[cylinder], args.prompt)
    return build_cylinder_prompt(cylinder.value, args.prompt)


def resolve_gemini_response_path(cylinder: Cylinder, override: str | None) -> Path:
    """Pick the Gemini response save path for ``cylinder``, honoring CLI override."""
    if override is not None:
        return Path(override).expanduser().resolve()
    return CYLINDER_GEMINI_RESPONSE_PATHS[cylinder]


def run_headless(
    args: argparse.Namespace,
    cylinder: Cylinder,
    prompt: str,
    save_path: Path,
    redis_client,
    motion: MotionParams,
    ctx,
) -> int:
    gemini_client = gp.make_genai_client(gp.resolve_api_key())
    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")
    print("Mode: headless (no UI)")

    pipeline = None
    owns_pipeline = False
    try:
        pipeline, align, depth_scale, color_intrinsics, owns_pipeline = _acquire_realsense(
            ctx, args
        )
    except Exception as e:
        print(f"RealSense startup failed: {e}", file=sys.stderr)
        return 1

    print(f"Gemini response will be saved to: {save_path}")
    gated = bool(getattr(ctx, "step", False))
    if gated:
        print("Press ENTER to advance phases (or type 'q'+ENTER to quit).")
    else:
        print("Auto-advancing phases (ctx.step=False).")

    phase = Phase.VISION_READY
    target: LatchedTarget | None = None
    slerp: OrientationSlerpState | None = None
    poured_ori: np.ndarray | None = None
    miss_counter = [0]
    # Consecutive Gemini timeouts in the VISION_READY phase — see
    # ``handle_gemini_timeout``: first ``GEMINI_TIMEOUT_AUTORETRY_LIMIT``
    # are silently retried with a fresh frame, then we prompt the
    # operator. Resets implicitly when ``phase`` advances past
    # VISION_READY (we won't re-enter VISION_READY this cycle).
    gemini_timeout_count = 0
    # Cross-phase carrier for the cart-position-gain boost (see
    # ``_advance_phase``). Populated on ``Phase.GRASPED`` entry,
    # drained on ``Phase.AT_RELEASE`` exit, and force-restored in
    # the ``finally`` below if anything goes wrong while the
    # cylinder is held.
    gain_state: dict = {}

    try:
        while phase != Phase.DONE:
            print(_phase_hint(phase, headless=True))
            if gated:
                try:
                    cmd = input("> ").strip().lower()
                except EOFError:
                    break
                if cmd == "q":
                    break

            if phase == Phase.VISION_READY:
                triple = _grab_fresh_frame(
                    pipeline, align, depth_scale, args.timeout_ms, miss_counter
                )
                if triple is None:
                    msg = "Failed to grab frame"
                    if gated:
                        print(f"{msg}; press ENTER to retry.")
                        continue
                    raise RuntimeError(f"[cylinder:{cylinder.value}] {msg}.")

                try:
                    target, _overlay = do_gemini_capture(
                        triple,
                        redis_client=redis_client,
                        args=args,
                        cylinder=cylinder,
                        motion=motion,
                        gemini_client=gemini_client,
                        prompt=prompt,
                        color_intrinsics=color_intrinsics,
                        save_path=save_path,
                    )
                except gp.GeminiTimeoutError as e:
                    # Gemini stalled past its per-call deadline (see
                    # ``vision.gemini_pointing.call_gemini``).
                    # ``handle_gemini_timeout`` silently auto-retries
                    # the first ``GEMINI_TIMEOUT_AUTORETRY_LIMIT``
                    # consecutive timeouts (regardless of ``gated``)
                    # so transient API stalls don't abort the cycle;
                    # only after that does it prompt the operator.
                    # Either way we stay in VISION_READY and grab
                    # a fresh frame on the next loop iteration.
                    gemini_timeout_count += 1
                    print(f"[cylinder:{cylinder.value}] gemini timed out: {e}")
                    handle_gemini_timeout(
                        cylinder.value, "cylinder_grasp", gemini_timeout_count
                    )
                    continue
                if target is None:
                    msg = "No latched target"
                    if gated:
                        print(f"{msg}; press ENTER to retry.")
                        continue
                    raise RuntimeError(f"[cylinder:{cylinder.value}] {msg}.")
                phase = Phase.VISION_LATCHED
                continue

            if target is None:
                msg = "No latched target"
                if gated:
                    print(f"{msg}; press ENTER to retry.")
                    continue
                raise RuntimeError(f"[cylinder:{cylinder.value}] {msg}.")

            phase, slerp, poured_ori = _advance_phase(
                phase, target, cylinder, slerp, poured_ori,
                redis_client=redis_client, motion=motion, args=args, ctx=ctx,
                gain_state=gain_state,
            )

    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise
    finally:
        # Safety net: if we were holding the cylinder (boost applied)
        # and bailed before ``Phase.AT_RELEASE`` cleared the snapshot,
        # put the gains back here so the boosted values don't stick
        # around for the next routine step.
        snapshot = gain_state.pop("snapshot", None)
        if snapshot is not None:
            try:
                gains.restore_cart_position_gains(
                    redis_client, snapshot, label=f"{cylinder.value} interrupt"
                )
            except Exception as e:  # noqa: BLE001 — restore is best-effort
                print(f"[gains] WARNING: could not restore cart gains: {e}")
        if owns_pipeline and pipeline is not None:
            pipeline.stop()

    return 0


def run_live(
    args: argparse.Namespace,
    cylinder: Cylinder,
    prompt: str,
    save_path: Path,
    redis_client,
    motion: MotionParams,
    ctx,
) -> int:
    gemini_client = gp.make_genai_client(gp.resolve_api_key())
    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")
    print("Mode: UI (OpenCV)")

    pipeline = None
    owns_pipeline = False
    try:
        pipeline, align, depth_scale, color_intrinsics, owns_pipeline = _acquire_realsense(
            ctx, args
        )
    except Exception as e:
        print(f"RealSense startup failed: {e}", file=sys.stderr)
        return 1

    win = "Cylinder Grasp Tilt"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    phase = Phase.VISION_READY
    latched_overlay = None
    target: LatchedTarget | None = None
    miss_counter = [0]
    slerp: OrientationSlerpState | None = None
    poured_ori: np.ndarray | None = None
    # Cross-phase carrier for the cart-position-gain boost (see
    # ``_advance_phase``). Same pattern as ``run_headless``.
    gain_state: dict = {}

    print("Keys: ENTER = detect/advance | q = quit")
    print(_phase_hint(phase))

    try:
        while True:
            triple = rs_cam.next_rgbd_frame(
                pipeline, align, depth_scale, args.timeout_ms, miss_counter, max_misses=10,
            )
            if triple is None:
                continue

            color_bgr, depth_m, depth_vis = triple
            h, w = color_bgr.shape[:2]
            gemini_panel = (
                latched_overlay.copy() if latched_overlay is not None
                else _gemini_placeholder_panel(h, w)
            )

            top_row = np.hstack((color_bgr, depth_vis, gemini_panel))
            bottom_row = _render_text_band(top_row.shape[1], TEXT_BAND_HEIGHT, _status_lines(phase, target))
            composite = np.vstack((top_row, bottom_row))
            cv2.imshow(win, composite)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if key == ord(" ") and phase == Phase.VISION_READY:
                triple = (color_bgr, depth_m, depth_vis)
                try:
                    target, overlay = do_gemini_capture(
                        triple,
                        redis_client=redis_client, args=args, cylinder=cylinder,
                        motion=motion,
                        gemini_client=gemini_client, prompt=prompt,
                        color_intrinsics=color_intrinsics, save_path=save_path,
                    )
                except gp.GeminiTimeoutError as e:
                    # In UI mode the operator is already at the
                    # keyboard, so a timeout just drops back to the
                    # live preview — they can hit SPACE/ENTER again
                    # for a fresh request (no stdin prompt needed).
                    print(f"[cylinder:{cylinder.value}] gemini timed out: {e}; press SPACE/ENTER to retry.")
                    continue
                if overlay is not None:
                    latched_overlay = overlay
                if target is None:
                    continue
                phase = Phase.VISION_LATCHED
                print(_phase_hint(phase))

            elif key in (10, 13):
                if phase == Phase.VISION_READY:
                    triple = (color_bgr, depth_m, depth_vis)
                    try:
                        target, overlay = do_gemini_capture(
                            triple,
                            redis_client=redis_client, args=args, cylinder=cylinder,
                            motion=motion,
                            gemini_client=gemini_client, prompt=prompt,
                            color_intrinsics=color_intrinsics, save_path=save_path,
                        )
                    except gp.GeminiTimeoutError as e:
                        print(
                            f"[cylinder:{cylinder.value}] gemini timed out: {e}; "
                            "press SPACE/ENTER to retry."
                        )
                        continue
                    if overlay is not None:
                        latched_overlay = overlay
                    if target is None:
                        continue
                    phase = Phase.VISION_LATCHED
                    print(_phase_hint(phase))
                    continue
                if target is None:
                    print("No latched target; press ENTER first.")
                    continue
                if phase == Phase.DONE:
                    print("Sequence complete — q to quit")
                    continue

                phase, slerp, poured_ori = _advance_phase(
                    phase, target, cylinder, slerp, poured_ori,
                    redis_client=redis_client, motion=motion, args=args, ctx=ctx,
                    gain_state=gain_state,
                )
                print(_phase_hint(phase))

    finally:
        snapshot = gain_state.pop("snapshot", None)
        if snapshot is not None:
            try:
                gains.restore_cart_position_gains(
                    redis_client, snapshot, label=f"{cylinder.value} interrupt"
                )
            except Exception as e:  # noqa: BLE001 — restore is best-effort
                print(f"[gains] WARNING: could not restore cart gains: {e}")
        if owns_pipeline and pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


def run_cylinder_cycle(
    ctx,
    cylinder: Cylinder,
    *,
    motion: MotionParams | None = None,
    options: argparse.Namespace | None = None,
    ui: bool = False,
    skip_base: bool = False,
    grasp_target: str = GRASP_TARGET_HANDLE,
    prompt: str | None = None,
    gemini_response_path: str | None = None,
) -> None:
    """One full grasp → pour → return cycle for ``cylinder``."""
    if motion is None:
        motion = default_motion_params(ctx)
    if options is None:
        options = build_cycle_options(
            ctx,
            ui=ui,
            skip_base=skip_base,
            grasp_target=grasp_target,
            prompt=prompt,
            gemini_response_path=gemini_response_path,
        )

    pour_station = CYLINDER_POUR_STATION[cylinder]
    strip_color = CYLINDER_HANDLE_COLOR[cylinder]
    resolved_prompt = resolve_prompt(options, cylinder)
    save_path = resolve_gemini_response_path(cylinder, options.gemini_response_path)

    print(
        f"\n=== cylinder cycle: {cylinder.value} "
        f"(strip={strip_color}, pour @ {pour_station.name}) ===",
        flush=True,
    )
    print(f"Gemini → {save_path}")

    detect_pos, detect_ori = CYLINDER_DETECTION_POSE[cylinder]
    if not options.skip_base:
        arm.move_to(
            ctx,
            ARM_HOME_POSITION,
            ARM_HOME_ORIENTATION,
            label=f"[arm] reset to home before cycle {ARM_HOME_POSITION.tolist()}",
            tol_m=CYLINDER_HOME_TOL_M,
        )
        base.go_to_pose(ctx, BaseWaypoint.RACK_STATION)
        arm.move_to(
            ctx,
            detect_pos,
            detect_ori,
            label=(
                f"[arm] move to {cylinder.value} detection pose "
                f"{detect_pos.tolist()} "
                f"(camera framing for Gemini)"
            ),
            tol_m=CYLINDER_HOME_TOL_M,
        )

    # Per-cylinder cartesian-position gain boost (e.g. sauce — see
    # ``ObjectSpec.lift_cart_position_kp``) lives INSIDE the phase
    # machine, not here: it's applied right after the gripper closes
    # (``Phase.GRASPED`` entry) and restored right after it opens
    # (``Phase.AT_RELEASE`` exit), so framing / above-grasp moves
    # before the cylinder is actually held stay at the normal stiffness.
    # ``run_headless`` / ``run_live`` keep a ``finally`` restore so
    # Ctrl-C or exceptions mid-grip can't strand the boosted gains.
    if options.ui:
        run_live(options, cylinder, resolved_prompt, save_path, ctx.redis, motion, ctx)
    else:
        run_headless(
            options, cylinder, resolved_prompt, save_path, ctx.redis, motion, ctx
        )

    print(f"=== cylinder cycle: {cylinder.value} complete ===\n", flush=True)


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    cylinders: tuple[Cylinder, ...] = _CYLINDER_SEQUENCES[args.cylinder]
    print(f"Step mode: {'on' if args.step else 'off'}")
    print(f"Grasp target: {args.grasp_target}")
    print(
        f"Cylinder sequence: {args.cylinder} → "
        f"[{', '.join(c.value for c in cylinders)}]"
    )
    print(f"Cartesian tilt: {args.pour_tilt_deg:.0f}° about +{args.pour_axis.upper()}")
    print(f"Z descent reduction: {CYLINDER_GRASP_Z_DESCENT_REDUCTION_M:.4f} m")

    for cyl in cylinders:
        pour_station = CYLINDER_POUR_STATION[cyl]
        gem_path = resolve_gemini_response_path(cyl, args.gemini_response_path)
        print(
            f"[plan:{cyl.value}] grasp=RACK_STATION  "
            f"pour={pour_station.name}  "
            f"strip={CYLINDER_HANDLE_COLOR[cyl]}  gemini→{gem_path}"
        )

    cycle_options = build_cycle_options(
        ctx,
        ui=args.ui,
        skip_base=args.skip_base,
        grasp_target=args.grasp_target,
        prompt=args.prompt,
        gemini_response_path=args.gemini_response_path,
        model=args.model,
        temperature=args.temperature,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        timeout_ms=args.timeout_ms,
        depth_patch_radius=args.depth_patch_radius,
        no_grasp_offset=args.no_grasp_offset,
        grasp_offset_x=args.grasp_offset_x,
        grasp_offset_y=args.grasp_offset_y,
        grasp_offset_z=args.grasp_offset_z,
        pour_tilt_deg=args.pour_tilt_deg,
        pour_axis=args.pour_axis,
        tilt_duration_s=args.tilt_duration_s,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        gripper_pregrasp_width=args.gripper_pregrasp_width,
        gripper_close_width=args.gripper_close_width,
        gripper_pregrasp_settle=args.gripper_pregrasp_settle,
        gripper_grasp_settle=args.gripper_grasp_settle,
        approach_dz=args.approach_dz,
        endeffector_transform_key=args.endeffector_transform_key,
    )

    motion = default_motion_params(ctx)
    motion = MotionParams(
        approach_dz_m=args.approach_dz,
        pour_tilt_deg=args.pour_tilt_deg,
        pour_axis=args.pour_axis,
        tilt_duration_s=args.tilt_duration_s,
        gripper_open_width=motion.gripper_open_width,
        gripper_pregrasp_width=args.gripper_pregrasp_width,
        gripper_close_width=args.gripper_close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        gripper_pregrasp_settle_s=args.gripper_pregrasp_settle,
        gripper_grasp_settle_s=args.gripper_grasp_settle,
    )
    print(
        f"Motion: approach_dz={motion.approach_dz_m} m, tilt={motion.pour_tilt_deg:.0f}° "
        f"about +{motion.pour_axis.upper()}, "
        f"gripper_open={motion.gripper_open_width:.4f} m, "
        f"close={motion.gripper_close_width:.4f} m"
    )

    completed: list[str] = []
    try:
        for idx, cylinder in enumerate(cylinders, start=1):
            print(
                f"\n### cylinder {idx}/{len(cylinders)}: {cylinder.value} "
                f"(sequence={args.cylinder}) ###",
                flush=True,
            )
            # Per-cylinder Gemini world offset (handle mode pulls from
            # OBJECT_DEFAULTS[<cylinder>].gemini_world_offset_m, body
            # mode shares DEFAULT_GRASP_OFFSET_M). Logged inside the
            # loop so each cylinder reports its own offset — printing
            # once before the loop would tie the log to whichever
            # cylinder happened to be defined last in the outer scope
            # (was an UnboundLocalError when no prior cylinder loop
            # ran, e.g. single-cylinder ``--cylinder sauce`` runs).
            off = grasp_offset_world(cycle_options, cylinder)
            if np.linalg.norm(off) > 0:
                print(
                    f"Grasp offset (world, m, {cylinder.value}): "
                    f"[{off[0]:+.4f}, {off[1]:+.4f}, {off[2]:+.4f}]"
                )
            else:
                print(f"Grasp offset ({cylinder.value}): disabled")
            run_cylinder_cycle(ctx, cylinder, motion=motion, options=cycle_options)
            completed.append(cylinder.value)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted. Completed {len(completed)}/{len(cylinders)} cylinders: "
            f"[{', '.join(completed) or '<none>'}]."
        )
        return 130
    finally:
        ctx.stop_realsense()

    print(
        f"Grasp + pour + return complete for "
        f"{len(completed)}/{len(cylinders)} cylinders: [{', '.join(completed)}]."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
