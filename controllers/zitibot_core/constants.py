"""Single source of truth for kitchen objects, base waypoints, and motion defaults."""

from __future__ import annotations

import copy
import enum
import os
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Camera extrinsic — single source of truth for all ZitiBot Python libraries.
#
# Camera optical frame -> Franka flange/EE frame. The perception pipeline
# (zitibot_tasks.gemini, grasp_and_pour_jar_controller, ...) applies it as:
#
#     p_base = T_base_flange @ T_FLANGE_CAMERA @ p_camera
#
# where T_base_flange is the live Redis T_end_effector (libfranka kEndEffector,
# so the flange<->EE offset is already folded in). Override at runtime with a
# hand-eye calibration result (see vision/calibrate_hand_eye.py) if desired.
#
# Rotation: +45° about the shared camera/flange Z axis. Translation: the camera
# optical origin expressed in the flange frame (Rz(-45) @ offset).
#
# NOTE on the rotation: live egg-cracker diagnostics showed +90° makes two
# equal-height handle points separate by ~1.8 cm in world Z and biases the
# midpoint forward in +X. +45° matches the position geometry; grasp orientation
# currently gets a separate +45° tool-yaw correction in zitibot_tasks.gemini.
# NOTE on the translation: least-squares fit to two manually-probed egg-cracker
# handle positions across two runs (Rz(45) held fixed). This started as
# Rz(-45) @ offset = [0.0314, -0.0441, 0.0189] and was nudged ~8 mm in flange Y
# to remove a consistent small world-Y bias. The real long-term fix is a
# hand-eye calibrated extrinsic.
# ---------------------------------------------------------------------------
def _z_rotation_matrix(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# Camera optical origin in the flange frame (meters).
_CAMERA_FLANGE_OFFSET = np.array([0.053401, -0.009, 0.018930], dtype=np.float64)

# Previous hardcoded placeholder (kept for reference / quick rollback).
T_FLANGE_CAMERA_OLD = np.array(
    [
        [0.0, -1.0, 0.0, 0.053],
        [1.0, 0.0, 0.0, -0.009],
        [0.0, 0.0, 1.0, 0.019],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

T_FLANGE_CAMERA = np.eye(4, dtype=np.float64)
T_FLANGE_CAMERA[:3, :3] = _z_rotation_matrix(45.0)
T_FLANGE_CAMERA[:3, 3] = np.array([0.02741, -0.05198, 0.01851], dtype=np.float64)

# Shared end-effector orientations (tool Z down, 45° yaw in world XY).
_GRASP_YAW_RAD = np.radians(45.0)
_c, _s = np.cos(_GRASP_YAW_RAD), np.sin(_GRASP_YAW_RAD)
EE_ORI_TOOL_DOWN = np.array(
    [
        [_c, -_s, 0.0],
        [-_s, -_c, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
# Tool down with zero world yaw and a right-handed frame. Body +Z points
# down in world (tool-down), body +X stays world +X, and body +Y points
# world -Y so det(R)=+1. Do NOT use diag([1, 1, -1]) here: that is a
# reflection (det=-1), not a valid rotation matrix, and OpenSai rejects it.
EE_ORI_TOOL_DOWN_45 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

# Arm "home" pose used by controllers as a deterministic starting EE pose
# before any task-specific motion. Snapshot from the live Franka startup
# pose printed by ``pour_and_move_controller`` so the arm comes back to a
# known launch posture on every run.
ARM_HOME_POSITION = np.array([0.2267, -0.0198, 0.6450], dtype=np.float64)
ARM_HOME_ORIENTATION = EE_ORI_TOOL_DOWN

HOME_POS_TOL_M = 0.1


# Taught EE orientation at PAN_STATION for vision pan-handle grasp.
# Snapshot from ``grasp_and_move_controller`` startup log at the pan station.
PAN_STATION_GRASP_EE_ORIENTATION = np.array(
    [
        [+0.6345, -0.7715, -0.0468],
        [-0.7729, -0.6340, -0.0265],
        [-0.0093, +0.0530, -0.9986],
    ],
    dtype=np.float64,
)

# End-effector waypoints for inserting the pan into the oven after the base
# reaches OVEN_DOOR. Snapshots from live Franka startup poses.
OVEN_EE_WAYPOINTS: tuple[tuple[np.ndarray, np.ndarray], ...] = (
    (
        np.array([0.6106, -0.0054, 0.3897], dtype=np.float64),
        np.array(
            [
                [+0.6797, -0.7318, +0.0502],
                [-0.7245, -0.6804, -0.1097],
                [+0.1144, +0.0382, -0.9927],
            ],
            dtype=np.float64,
        ),
    ),
    (
        np.array([0.6691, 0.0222, 0.1090], dtype=np.float64),
        np.array(
            [
                [+0.6275, -0.7757, +0.0673],
                [-0.7739, -0.6308, -0.0561],
                [+0.0860, -0.0169, -0.9962],
            ],
            dtype=np.float64,
        ),
    ),
)

# Final end-effector pose for releasing the pan in the oven.
OVEN_EE_POSITION = np.array([0.5874, -0.1672, 0.0329], dtype=np.float64)
OVEN_EE_ORIENTATION = np.array(
    [
        [+0.4706, -0.8823, +0.0016],
        [-0.8766, -0.4678, -0.1130],
        [+0.1005, +0.0518, -0.9936],
    ],
    dtype=np.float64,
)

# Per-cylinder camera-framing EE poses at RACK_STATION. The flange-
# mounted RealSense looks at the rack so Gemini can see the cylinder
# (parmesan / sauce jar / ricotta) without the gripper or forearm
# occluding it. ``grasp_and_pour_jar_controller`` moves the arm to the
# cylinder-specific pose after parking the base at RACK_STATION and
# before calling Gemini.
#
# Snapshots from live Franka teaching — re-record per cylinder by
# jogging the arm until the camera frames the colored handle strip
# (parmesan = blue, sauce = gray, ricotta = red) cleanly and logging
# ``T_end_effector`` from Redis.
#
# Parmesan also has FORWARDS / BACKWARDS variants because the rack is
# approached from two sides:
#   * FORWARDS — preferred. Camera faces the rack from the front, which
#     is the orientation the rest of the kitchen routine assumes.
#   * BACKWARDS — legacy framing kept around for jogging / debugging if
#     the base ends up parked on the far side of the rack.
PARMESAN_DETECTION_EE_POSITION_FORWARDS = np.array(
    # Backed off 5 cm in -X from the original teach point [0.1856, -0.0553,
    # 0.6696] so the RealSense frames the full rack-mounted cylinder + its
    # parallel blue-strip handle without the cap getting clipped at the top.
    [0.1356, -0.0553, 0.6696], dtype=np.float64
)
PARMESAN_DETECTION_EE_ORIENTATION_FORWARDS = np.array(
    [
        [+0.7147, -0.6918, +0.1031],
        [-0.6932, -0.7202, -0.0269],
        [+0.0929, -0.0523, -0.9943],
    ],
    dtype=np.float64,
)

PARMESAN_DETECTION_EE_POSITION_BACKWARDS = np.array(
    [0.2500, -0.0044, 0.6316], dtype=np.float64
)
PARMESAN_DETECTION_EE_ORIENTATION_BACKWARDS = np.array(
    [
        [+0.6358, -0.7245, -0.2663],
        [-0.7533, -0.6576, -0.0094],
        [-0.1683, +0.2066, -0.9638],
    ],
    dtype=np.float64,
)

# Default parmesan detection pose (forwards). Kept for back-compat with
# any code that still imports ``PARMESAN_DETECTION_EE_POSITION`` /
# ``..._ORIENTATION`` directly.
PARMESAN_DETECTION_EE_POSITION = PARMESAN_DETECTION_EE_POSITION_FORWARDS
PARMESAN_DETECTION_EE_ORIENTATION = PARMESAN_DETECTION_EE_ORIENTATION_FORWARDS

# Sauce + ricotta detection poses. Sauce is currently a placeholder
# copy of the parmesan FORWARDS pose; re-record on the bench. Ricotta
# is offset −5 cm in world Y from the parmesan FORWARDS pose because
# its rack slot sits 5 cm to the right of the parmesan slot (Franka
# base frame: +X forward, +Y left, +Z up). Re-record per-cylinder by
# jogging the arm until the RealSense frames the cylinder's gray
# (sauce) / red (ricotta) handle strip cleanly and logging the EE
# world transform.
SAUCE_DETECTION_EE_POSITION = PARMESAN_DETECTION_EE_POSITION_FORWARDS.copy()
SAUCE_DETECTION_EE_ORIENTATION = PARMESAN_DETECTION_EE_ORIENTATION_FORWARDS.copy()
RICOTTA_DETECTION_EE_POSITION = (
    PARMESAN_DETECTION_EE_POSITION_FORWARDS + np.array([0.0, -0.05, 0.0], dtype=np.float64)
)
RICOTTA_DETECTION_EE_ORIENTATION = PARMESAN_DETECTION_EE_ORIENTATION_FORWARDS.copy()

# Per-cylinder taught Cartesian pretilt poses at the pour station.
# Parmesan pours into the pan at PAN_STATION from a hand-taught pose;
# sauce + ricotta pour into the mixing bowl at MIXING_STATION using a
# live-rotate spec (see ``CYLINDER_PRETILT_SPEC`` in
# ``grasp_and_pour_jar_controller``) and therefore have no taught pose
# here. Re-record by parking the base at the pour station, holding
# the cylinder over the target with the bottle opening aimed for the
# J7 dump, and logging the EE world transform.
#
# Parmesan: Y was shifted +0.04 m from the originally taught
# 0.0380 → 0.0780 to nudge the bottle opening 4 cm "left" (Franka base
# frame: +X forward, +Y left, +Z up). Flip the sign on the +0.04 if
# the bottle ends up pouring the wrong way.
PARMESAN_PRETILT_EE_POSITION = np.array([0.5004, 0.0780, 0.8628], dtype=np.float64)
PARMESAN_PRETILT_EE_ORIENTATION = np.array(
    [
        [-0.0120, -0.0025, +0.9999],
        [-0.6948, -0.7191, -0.0101],
        [+0.7191, -0.6949, +0.0068],
    ],
    dtype=np.float64,
)

# Egg-cracker camera-framing EE pose at INGREDIENT_STATION.
#
# Hand-taught at the bench: jogged the arm until the RealSense framed the
# cracker's gray strip + red cross-mark cleanly, then logged the live
# ``[arm] startup`` print from Redis. Re-record the same way if the
# cracker / ingredient station shifts.
# Cracker detection / camera-framing pose: home X/Y, dropped 10 cm below
# home in -Z. Gemini detection runs from here (with ARM_HOME_ORIENTATION).
EGG_CRACKER_DETECTION_EE_POSITION = ARM_HOME_POSITION + np.array(
    [0.0, 0.0, -0.10], dtype=np.float64
)
EGG_CRACKER_DETECTION_EE_ORIENTATION = np.array(
    [
        [+0.6761, -0.6090, +0.4148],
        [-0.6643, -0.7473, -0.0144],
        [+0.3188, -0.2658, -0.9098],
    ],
    dtype=np.float64,
)

EGG_CRACKER_STATIONARY_DETECTION_EE_POSITION = np.array(
    [0.2816, 0.0072, 0.6871], dtype=np.float64
)
EGG_CRACKER_STATIONARY_DETECTION_EE_ORIENTATION = EE_ORI_TOOL_DOWN_45;

# World offset applied to the egg-cracker CRADLE-CENTER detection (the drop
# point where the tong-held egg is released). This is a distinct target from
# the cracker handle grasp, so it does NOT inherit
# ``OBJECT_DEFAULTS[Object.EGG_CRACKER].gemini_world_offset_m`` (which is tuned
# for the handle grasp). Land on the detected cradle center as-is.
EGG_CRACKER_CRADLE_CENTER_WORLD_OFFSET_M = np.array([0.00, 0.00, 0.02], dtype=np.float64)

# Whisk stationary waypoint / camera-framing pose (startup EE, 2026-06-06).
WHISK_STATIONARY_WAYPOINT_EE_POSITION = np.array(
    [+0.3070, -0.1024, +0.7044], dtype=np.float64
)
WHISK_STATIONARY_WAYPOINT_EE_ORIENTATION = np.array(
    [
        [-0.0830, +0.0287, +0.9961],
        [-0.7036, -0.7096, -0.0382],
        [+0.7057, -0.7040, +0.0791],
    ],
    dtype=np.float64,
)
WHISK_STATIONARY_DETECTION_EE_POSITION = WHISK_STATIONARY_WAYPOINT_EE_POSITION.copy()
WHISK_STATIONARY_GRASP_EE_ORIENTATION = WHISK_STATIONARY_WAYPOINT_EE_ORIENTATION.copy()
WHISK_STATIONARY_DETECTION_EE_ORIENTATION = WHISK_STATIONARY_WAYPOINT_EE_ORIENTATION.copy()
# Nudge the Gemini midpoint this far along gripper +tool_z (forward) to land on
# the handle (bench tune 2026-06).
WHISK_GRASP_FORWARD_ALONG_TOOL_Z_M = 0.025
WHISK_GRASP_WORLD_Z_OFFSET_M = 0.0
WHISK_GRASP_OPEN_WIDTH_M = 0.078
WHISK_GRASP_CLOSE_WIDTH_M = 0.05
# Faster motion / settle tuning for the whisk pick-up sequence.
WHISK_GRASP_FORCE_CLOSE_SETTLE_S = 1.0
WHISK_ARM_SETTLE_TICKS = 1

# Post-lift carry / place-hover waypoint (above bowl, before 10 cm descent).
WHISK_STATIONARY_CARRY_EE_POSITION = np.array(
    [+0.3403, -0.1320, +0.8481], dtype=np.float64
)
WHISK_STATIONARY_CARRY_EE_ORIENTATION = np.array(
    [
        [+0.0223, -0.0447, +0.9988],
        [-0.6955, -0.7183, -0.0166],
        [+0.7181, -0.6943, -0.0472],
    ],
    dtype=np.float64,
)

# Post-release red-tape camera framing (taught EE, 2026-06-07).
WHISK_STATIONARY_RED_TAPE_EE_POSITION = np.array(
    [+0.2338, -0.1471, +0.7273], dtype=np.float64
)
WHISK_STATIONARY_RED_TAPE_EE_ORIENTATION = np.array(
    [
        [-0.0877, -0.0472, +0.9950],
        [-0.7415, -0.6639, -0.0968],
        [+0.6652, -0.7463, +0.0232],
    ],
    dtype=np.float64,
)

# Per-bowl pour EE waypoint sequences. The arm walks forward through
# these (upright -> tilted) at the pour station and then back to the
# first waypoint so the bowl returns upright before the base drives
# to the sink. Snapshots from live Franka teaching at MIXING_STATION;
# re-record by holding the bowl, jogging through start/mid/final, and
# logging the EE world transform at each point.
#
# Convention for every entry: (position [m], orientation 3x3 rotmat)
# where the rotmat columns are the EE-body axes expressed in world
# (col 0 = body +X, col 2 = body +Z aka tool axis, etc.).
PASTA_POUR_EE_WAYPOINTS: tuple[tuple[np.ndarray, np.ndarray], ...] = (
    # Start: bowl upright above the mixing bowl (tool pointing down).
    (
        np.array([0.2536, 0.1465, 0.6705], dtype=np.float64),
        np.array(
            [
                [+0.8802, -0.4600, +0.1165],
                [-0.4514, -0.8874, -0.0933],
                [+0.1463, +0.0295, -0.9888],
            ],
            dtype=np.float64,
        ),
    ),
    # Mid pour: bowl partway tilted, contents starting to flow.
    (
        np.array([0.2545, 0.1947, 0.7374], dtype=np.float64),
        np.array(
            [
                [+0.9230, -0.3702, +0.1046],
                [-0.3290, -0.6187, +0.7134],
                [-0.1994, -0.6929, -0.6929],
            ],
            dtype=np.float64,
        ),
    ),
    # Final pour: bowl fully tilted, contents emptied into mixing bowl.
    (
        np.array([0.2600, 0.0975, 0.8014], dtype=np.float64),
        np.array(
            [
                [+0.7360, -0.6523, -0.1813],
                [+0.1661, -0.0856, +0.9824],
                [-0.6563, -0.7531, +0.0453],
            ],
            dtype=np.float64,
        ),
    ),
)

# Global motion defaults (shared across objects unless overridden in ObjectSpec).
DEFAULT_APPROACH_DZ_M = 0.10
DEFAULT_GRIPPER_SPEED = 0.1
DEFAULT_GRIPPER_FORCE = 50.0
DEFAULT_GRIPPER_PREGRASP_WIDTH = 0.05
DEFAULT_GRIPPER_CLOSE_WIDTH = 0.0

# Finger width the egg cracker is grasped to (MOVE mode — closes to this
# width and HOLDS there, no continuous force squeeze). The gripper driver's
# move branch no longer backs off on a "failed" move, so this width is held
# through the carry lift. The full crack squeeze is still force-based
# (see ``egg_crack.crack``); this only sets the initial pick-up grip width.

# --- Precise-grasp mode (default in grasp.object) ---------------------------
# Final-approach slowdown + stiffening. Once grasp.object's move to the
# pre-grasp "above" pose comes within ``PRECISE_GRASP_WITHIN_M`` of that pose,
# the cartesian task is switched to a slow OTG linear-velocity cap and stiff
# position/orientation gains so the descent + close land accurately. The live
# values are snapshotted on engage and restored after the gripper closes (so
# the post-grasp lift + everything else runs at the normal stiffness/speed).
# See ``zitibot_core.gains.apply_precise_grasp`` / ``restore_precise_grasp``.
PRECISE_GRASP_WITHIN_M = 0.05
PRECISE_GRASP_MAX_LINEAR_VELOCITY = 0.03
# OTG angular-velocity cap (rad/s) while precise mode is engaged. The OTG
# default is M_PI/3 (~1.047 rad/s, 60 deg/s); clamp to ~30 deg/s so wrist
# re-orientations (e.g. the egg-crack empty tilt) track accurately instead of
# whipping around. A 135 deg tilt at this cap takes ~4.5 s (2.36 rad / 0.524),
# so ``PRECISE_GRASP_MOVE_TIMEOUT_S`` is sized above that to let it finish.
PRECISE_GRASP_MAX_ANGULAR_VELOCITY = float(np.pi / 6.0)
PRECISE_GRASP_POSITION_KP = 400.0
PRECISE_GRASP_ORIENTATION_KP = 400.0
# Convergence budget (s) for the precise-mode "above" and descent moves. The
# slow 0.03 m/s cap means the final 5 cm of the approach alone can take ~2 s,
# so the default 4 s ``arm.move_to`` timeout is too tight — bump it for the
# two moves precise mode affects (the lift afterwards runs at restored speed).
# Also sized to cover the ~4.5 s a 135 deg empty tilt now takes at the lowered
# 30 deg/s angular cap above.
PRECISE_GRASP_MOVE_TIMEOUT_S = 6

CYLINDER_GRASP_Z_DESCENT_REDUCTION_M = 0.07  # Lift grasp to prevent table collision
DEFAULT_GRIPPER_PREGRASP_SETTLE_S = 0.6
DEFAULT_GRIPPER_GRASP_SETTLE_S = 1.2
DEFAULT_POS_TOL_M = 0.04
DEFAULT_GRASP_TOL_M = 0.04
DEFAULT_POUR_TILT_DEG = 90.0
DEFAULT_POUR_AXIS = "x"
DEFAULT_TILT_DURATION_S = 6.0
TICK_DT_S = 0.1

# End-effector waypoints for the cylinder pour sequence at PAN_STATION.
# WP2 = safe transit/carry pose used between rack→pan and between pours (arm tucked).
# WP3 = first pour pose (tilt over pan, first shake).
# WP4 = second pour pose (different tilt angle or axis for second shake).
# Record all three from the live robot (print T_end_effector from Redis) and update here.
CYLINDER_EE_WP2: tuple[np.ndarray, np.ndarray] = (
    ARM_HOME_POSITION.copy(),    # TODO: record safe transit/carry pose
    ARM_HOME_ORIENTATION.copy(),
)
CYLINDER_EE_WP3: tuple[np.ndarray, np.ndarray] = (
    ARM_HOME_POSITION.copy(),    # TODO: record first pour pose (tilted over pan)
    ARM_HOME_ORIENTATION.copy(),
)
CYLINDER_EE_WP4: tuple[np.ndarray, np.ndarray] = (
    ARM_HOME_POSITION.copy(),    # TODO: record second pour pose
    ARM_HOME_ORIENTATION.copy(),
)

CONFIG_XML = os.environ.get("ZITIBOT_OPENSAI_CONFIG_XML", "zitibot_panda.xml")
CONTROLLER_TO_USE = "cartesian_controller"
JOINT_CONTROLLER = "joint_controller"


class Object(enum.Enum):
    PASTA_BOWL = "pasta_bowl"
    PLASTIC_BOWL_TOP = "plastic_bowl_top"
    PLASTIC_BOWL_BOTTOM = "plastic_bowl_bottom"
    MIXING_BOWL = "mixing_bowl"
    PAN = "pan"
    LADLE = "ladle"
    TONGS = "tongs"
    EGG_CRACKER = "egg_cracker"
    EGG = "egg"
    BOTTLE = "bottle"
    JAR = "jar"
    # Per-cylinder dispatch for ``grasp_and_pour_jar_controller`` so each
    # cylinder gets its own ObjectSpec (grasp offsets, gripper widths,
    # approach_dz, etc.). All three start as BOTTLE-style defaults; tune
    # per-cylinder below in OBJECT_DEFAULTS.
    PARMESAN = "parmesan"
    SAUCE = "sauce"
    RICOTTA = "ricotta"
    WHISK = "whisk"


class BaseWaypoint(enum.Enum):
    STIRRING_STATION = "stirring_station"
    EGG_CRACK_STATION = "egg_crack_station"
    MIXING_STATION = "mixing_station"
    INGREDIENT_STATION = "ingredient_station"
    RACK_STATION = "rack_station"
    PRE_PAN_STATION = "pre_pan_station"
    PAN_STATION = "pan_station"
    OVEN_DOOR = "oven_door"
    STOVE_STATION = "stove_station"
    SINK_STATION = "sink_station"
    PASTA_STATION = "pasta_station"


@dataclass(frozen=True)
class OptiPose:
    x_m: float
    y_m: float
    yaw_deg: float


@dataclass
class ObjectSpec:
    """Per-object grasp/place defaults."""

    pick_pose: np.ndarray | None = None
    place_pose: np.ndarray | None = None
    rest_pose: np.ndarray | None = None
    pour_pose: np.ndarray | None = None
    # Default grasp orientation: the live arm "home" attitude
    # (``ARM_HOME_ORIENTATION``). The shared Gemini grasp path
    # (``_base_grasp_orientation`` → perpendicular-yaw builders in
    # ``zitibot_tasks.gemini``) uses this as the base posture and
    # rotates it about tool +Z to align the jaws with the detected
    # axis, so every Gemini grasp besides the cylinders and the pan
    # starts from the home wrist attitude. Objects with a genuinely
    # different base attitude (e.g. PAN) override ``grasp_ori``
    # explicitly; the cylinder controller builds its own base from
    # ``PARMESAN_DETECTION_EE_ORIENTATION`` and never reads this.
    grasp_ori: np.ndarray = field(default_factory=lambda: ARM_HOME_ORIENTATION.copy())
    approach_dz: float = DEFAULT_APPROACH_DZ_M
    # When False (default), ``approach_dz`` is interpreted as a world
    # +Z offset above the grasp point — the legacy "hover straight up
    # then descend straight down in world Z" approach. Correct for
    # tool-down grippers but wrong for any grasp whose ``grasp_ori``
    # is tilted, because the descent stops being along the gripper's
    # pointing direction.
    #
    # When True, ``approach_dz`` is interpreted as an offset along the
    # gripper's -tool_z direction (i.e. backed off opposite to the
    # gripper's facing direction). The above pose is
    # ``grasp_pos - grasp_ori[:, 2] * approach_dz``, so the
    # straight-line descent from above → grasp moves the gripper
    # FORWARD along its own +tool_z. This is what you want any time
    # ``grasp_ori`` has a non-trivial tilt baked in (e.g. the cylinder
    # grasp's -30° wrist tilt). Controllers opt in by reading this
    # flag from the per-object spec — currently honored only by the
    # cylinder controller (``grasp_and_pour_jar_controller``); the
    # shared ``zitibot_tasks.grasp.object`` path still uses world Z.
    approach_along_tool_z: bool = False
    # Extra world-frame wrist rotation applied to the Gemini-derived
    # grasp orientation (and only the grasp/above poses — NOT the carry
    # pose). Honored only by the cylinder controller's
    # ``_apply_cylinder_grasp_extra_rot`` path; other grasp paths
    # ignore these fields. Defaults to a no-op (0°) so non-cylinder
    # objects are unaffected; cylinders set this per-Object in
    # ``OBJECT_DEFAULTS`` (sauce / ricotta use -30° about world Y;
    # parmesan keeps 0° so the gripper stays tool-down for the strip
    # grasp).
    grasp_extra_rot_deg: float = 0.0
    grasp_extra_rot_axis: str = "y"
    # When True (default), the shared Gemini grasp builders in
    # ``zitibot_tasks.gemini`` rotate ``grasp_ori`` about tool +Z so the
    # jaws close perpendicular to the detected rim chord / strip / handle
    # axis (``_apply_perpendicular_yaw``). When False the builder skips
    # that rotation entirely and the grasp descends at the base
    # ``grasp_ori`` as-is (e.g. the home attitude). Set False for objects
    # that should be grasped at a fixed wrist attitude regardless of the
    # detected axis. The cylinder controller and the pan builder don't
    # consult this flag (cylinders build their own orientation; the pan
    # uses a fixed taught pose).
    grasp_align_jaws_to_detected_axis: bool = True
    # Optional cartesian-task position PID boost applied while the object
    # is held — written to Redis at the start of the held segment and
    # restored on release (see ``zitibot_core.gains.boosted_cart_position_gains``).
    # ``None`` means "don't touch this gain slot" — controller keeps
    # whatever live value is in Redis (the XML default unless something
    # else has rewritten it). Honored only by the cylinder controller
    # currently; sauce uses this to stop sagging on lift, since the
    # default position_kp can't fight gravity on the heavier jar.
    #
    # ``lift_cart_position_ki`` enables the integral term — builds up
    # steady-state force to overcome constant disturbances like gravity
    # on a heavy payload. The integrator does most of the gravity
    # comp, which means you DON'T need a huge ``kp`` — keep ``kp`` /
    # ``kv`` modest to avoid vibration, then let ``ki`` (e.g. 5-20)
    # chase the steady-state sag. ``ki`` is off (0) by default in the
    # XML config, so enabling it here is essentially a per-cylinder
    # opt-in.
    lift_cart_position_kp: float | None = None
    lift_cart_position_kv: float | None = None
    lift_cart_position_ki: float | None = None
    open_width: float | None = None
    pregrasp_width: float = DEFAULT_GRIPPER_PREGRASP_WIDTH
    close_width: float = DEFAULT_GRIPPER_CLOSE_WIDTH
    speed: float = DEFAULT_GRIPPER_SPEED
    force: float = DEFAULT_GRIPPER_FORCE
    pregrasp_settle_s: float = DEFAULT_GRIPPER_PREGRASP_SETTLE_S
    grasp_settle_s: float = DEFAULT_GRIPPER_GRASP_SETTLE_S
    # Pause after GRASP-mode force-close (``grasp.object`` close_mode="grasp").
    grasp_force_close_settle_s: float = 4.0
    approach_tol: float = DEFAULT_POS_TOL_M
    grasp_tol: float = DEFAULT_GRASP_TOL_M
    # Extra shift applied along gripper +tool_z after Gemini world projection
    # (positive = forward along the tool pointing direction).
    grasp_forward_tool_z_m: float = 0.0
    # Extra world +Z shift after Gemini projection (positive = up).
    grasp_world_z_offset_m: float = 0.0
    gemini_world_offset_m: np.ndarray = field(
        default_factory=lambda: np.array([0.03, 0.0, 0.09], dtype=np.float64)
    )


# Seeded from grasp_and_pour_controller, grasp_and_place_controller, mixing_controller,
# egg_crack_controller, and pour_and_move_controller.
OBJECT_DEFAULTS: dict[Object, ObjectSpec] = {
    Object.PASTA_BOWL: ObjectSpec(
        pick_pose=np.array([0.49764, 0.10, 0.369818]),
        pour_pose=np.array([0.522549, 0.115722, 0.628443]),
        place_pose=np.array([0.49764, 0.10, 0.369818]),
        # Grasp = midpoint of the two left-rim points, shifted +4 cm in Y
        # (world left for a standard Franka base frame) and +6 cm in Z
        # to lift the grasp above Gemini's reported rim depth (depth is
        # consistently low because the bowl rim either gets dropped by
        # RealSense or sits outside the depth patch). The "above" pose
        # is then grasp + approach_dz (9 cm) above. If "left" looks
        # wrong on the bench, flip Y to -0.04.
        gemini_world_offset_m=np.array([0.0, 0.0, -0.02], dtype=np.float64),
        approach_dz=0.04,
    ),
    Object.MIXING_BOWL: ObjectSpec(
        pick_pose=np.array([0.17, 0.62, 0.50]),
        place_pose=np.array([0.17, 0.62, 0.50]),
        # Left-rim grasp: +Y shifts the detected point outward (world +Y on
        # this bench) so fingers sit on the lip, not inside. Shiny bowl
        # interiors/reflections pull Gemini + depth inward; tune by editing
        # this offset (the bowl_pour_controller ``--bowl mixing`` flow
        # picks it up like every other bowl, no CLI mutation hook).
        # Dropped from +0.08 → +0.06 after Gemini's mid-rim point came
        # in slightly outboard of the actual rim on the bench; the +Z
        # bump that lifts the grasp above the rim lives in
        # bowl_pour_controller's ``BOWL_GRASP_Z_OFFSETS_M[MIXING_BOWL]``,
        # not here, so don't double-count Z by adding it on both sides.
        gemini_world_offset_m=np.array([-0.01, -0.03, 0.05], dtype=np.float64),
        approach_dz=0.05,
    ),
    Object.PAN: ObjectSpec(
        pick_pose=np.array([0.364646, 0.019878, 0.353623]),
        place_pose=OVEN_EE_POSITION.copy(),
        grasp_ori=PAN_STATION_GRASP_EE_ORIENTATION.copy(),
        approach_along_tool_z=True,
        # Stay fully open until the grasp force is applied so the fingers
        # don't bump the handle on descent. 0.08 m saturates to gripper max.
        pregrasp_width=0.08,
        # Goal = grasp point 1 + offset. +Z raises the flange above the
        # detected handle point; +Y is world-left (same as bowl rim tuning).
        # Bench tune (2026-06): 1 in lower than prior [0, 0, 0.06] (no XY shift).
        gemini_world_offset_m=np.array([0.0, 0.00, -0.03], dtype=np.float64),
        # Pre-grasp "above" pose sits 3 cm above the grasp pose (vs. the
        # 15 cm global default). Short approach because the gripper is
        # already coming in low above the handle.
        approach_dz=0.04,
        # The 4.0 s default force-close dwell is overkill for the pan handle —
        # the gripper clamps in well under a second. Wait ~1 s so the relocation
        # continues promptly after grasping.
        grasp_force_close_settle_s=2.0,
    ),
    Object.LADLE: ObjectSpec(
        pick_pose=np.array([0.75, 0.68, 0.508]),
        rest_pose=np.array([0.75, 0.68, 0.508]),
        gemini_world_offset_m=np.array([0.00, 0.0, -0.04], dtype=np.float64),
        approach_dz=0.08,
        # The 4.0 s default force-close dwell is overkill for the ladle handle —
        # the gripper clamps in well under a second. Shorten it so the routine
        # continues to the lift promptly after grasping.
        grasp_force_close_settle_s=1.0,
        # gemini_world_offset_m=np.zeros(3, dtype=np.float64),
    ),
    Object.EGG_CRACKER: ObjectSpec(
        pick_pose=np.array([0.5523, -0.0811, 0.4625]),
        force=8.0,
        # Gripper speed lowered 25% from the 0.1 m/s default (0.1 * 0.75)
        # for a gentler close on the cracker handle.
        speed=0.075,
        # Initial pick-up grip is width-based (MOVE mode in grasp.object,
        # close_mode="move"): the gripper closes to this width and HOLDS it
        # (driver no longer backs off on a failed move). The crack squeeze
        # itself stays force-based via egg_crack.crack.
        close_width=0.04,
        # Pre-grasp finger opening. Dropped from 0.05 → 0.035 m so the
        # jaws don't bump the next object on the counter when descending
        # onto the cracker — 3.5 cm still clears the cracker's gray-strip
        # grasp surface, which is well under 3 cm wide.
        pregrasp_width=0.09,
        # Vision grasp lands at the XY midpoint of the two gray-strip
        # points and uses p1's Z as the depth (see ``_build_pose_cylinder``).
        # Lift the detected grasp 6 cm in +Z so the flange sits above the
        # strip and the fingers descend onto it — same convention as the
        # pan handle offset. No XY shift: the strip midpoint already centers
        # the gripper on the grip surface.
        gemini_world_offset_m=np.array([0.00, 0.00, -0.02], dtype=np.float64),
        # Short approach: arm is already framed close to the cracker for
        # detection, so the pre-grasp "above" pose only needs ~3 cm of
        # clearance above the grasp pose (matches the pan).
        approach_dz=0.08,
        # Tighten the visual grasp moves: 4 cm default tolerance is too loose
        # for placing the jaws on the cracker handles.
        approach_tol=0.005,
        grasp_tol=0.005,
    ),
    # Tongs: grabbed like the egg cracker — Gemini returns one point per red
    # tape mark (one per tong arm) and the gripper closes ALONG the line
    # between them (one finger per arm), see ``_build_pose_egg_cracker``.
    Object.TONGS: ObjectSpec(
        # Open to 70% of the gripper's 0.08 m max stroke (= 0.056 m) before
        # descending so the jaws clear both tong arms, then force-close to
        # squeeze them together. (0.70 * 0.08 m, per the pickup spec.)
        pregrasp_width=0.056,
        # Force-close all the way (target width ignored in GRASP mode) so the
        # jaws clamp firmly onto the tong arms.
        close_width=0.03,
        force=50.0,
        # Land on the detected tape midpoint; no XY/Z shift.
        gemini_world_offset_m=np.array([0.00, 0.00, -0.03], dtype=np.float64),
        approach_dz=0.08,
        # Thin tape marks: keep the grasp moves tight like the egg cracker.
        approach_tol=0.01,
        grasp_tol=0.01,
    ),
    Object.EGG: ObjectSpec(
        pick_pose=np.array([0.5523, -0.0811, 0.4625]),
        # Gentle top-down grasp of a whole egg. The egg controller uses GRASP
        # (force-close) mode: jaws creep closed at ``speed`` and force-close
        # at ``force`` (1 N) so the egg is held without being crushed. Tune
        # via --gripper-force / --gripper-speed on the controller.
        speed=0.02,
        force=40.0,
        # Fully open (saturates to the gripper max ~0.08 m) before descending.
        pregrasp_width=0.08,
        # Typical chicken egg short axis ~4–4.5 cm; wide enough not to crush.
        close_width=0.05,
        # Egg is round/symmetric: the grasp builder uses Gemini's single point
        # directly (no rim/axis offset). Override the ObjectSpec default
        # [0.03, 0, 0.09] so we land on the detected egg top, not 9 cm above it.
        gemini_world_offset_m=np.array([0.00, 0.00, 0.01], dtype=np.float64),
        approach_dz=0.08,
        approach_tol=0.01,
        grasp_tol=0.01,
        # No detected-axis yaw — egg is symmetric, grasp at the home wrist yaw.
        grasp_align_jaws_to_detected_axis=False,
    ),
    # Cylindrical grasp objects. No physical pick_pose yet — Gemini
    # picks two points along the visible long axis and ``_build_pose_cylinder``
    # grasps perpendicular to that line, at the line midpoint. Keep the Gemini
    # offset zero so the target stays on the selected line.
    Object.BOTTLE: ObjectSpec(
        # Close ~1.5 cm tighter than the original 3 cm so the jaws really
        # crush into the bottle wall and there's no slack to slip on lift.
        close_width=0.015,
        # Franka Hand's continuous force limit. 140 N (the short-term peak)
        # is rejected/clamped by libfranka and triggers reflex aborts.
        force=70.0,
        gemini_world_offset_m=np.zeros(3, dtype=np.float64),
        # Cylinder controller "above grasp" height. This is intentionally
        # short: after the dedicated parmesan detection pose, the arm should
        # only stage a few cm above the contact point before descending.
        approach_dz=0.06,
    ),
    Object.JAR: ObjectSpec(
        close_width=0.04,
        gemini_world_offset_m=np.zeros(3, dtype=np.float64),
        approach_dz=0.25,
    ),
    # Whisk: Gemini returns two points along the red tape's long axis for the
    # grasp XY/Z midpoint; wrist orientation is the fixed taught pose below
    # (not axis-yawed). Force-based whisking in the black bowl is handled by
    # Force-based whisking in the black bowl is handled by
    # ``whisk_controller`` (see ``ZitiBot/controllers/whisk_controller.py``).
    Object.WHISK: ObjectSpec(
        grasp_ori=WHISK_STATIONARY_GRASP_EE_ORIENTATION.copy(),
        grasp_align_jaws_to_detected_axis=False,
        approach_along_tool_z=True,
        pregrasp_width=WHISK_GRASP_OPEN_WIDTH_M,
        open_width=WHISK_GRASP_OPEN_WIDTH_M,
        # MOVE-mode grasp (``close_mode="move"``): stop at 4 cm and hold until
        # the carry waypoint (see ``whisk_controller``).
        close_width=WHISK_GRASP_CLOSE_WIDTH_M,
        force=60.0,
        speed=0.1,
        pregrasp_settle_s=0.3,
        grasp_settle_s=0.5,
        grasp_force_close_settle_s=WHISK_GRASP_FORCE_CLOSE_SETTLE_S,
        gemini_world_offset_m=np.array([0.02, 0.0, 0.0], dtype=np.float64),
        grasp_forward_tool_z_m=WHISK_GRASP_FORWARD_ALONG_TOOL_Z_M,
        grasp_world_z_offset_m=WHISK_GRASP_WORLD_Z_OFFSET_M,
        approach_dz=0.03,
        approach_tol=0.030,
        grasp_tol=0.030,
    ),
}

# Per-cylinder grasp tuning for ``grasp_and_pour_jar_controller``. Each
# cylinder gets its own ObjectSpec (deep-copied from ``Object.BOTTLE``)
# so its handle-mode ``gemini_world_offset_m`` and other knobs can be
# tuned independently without bleeding into the others. The base BOTTLE
# offset is zeros; the controller's old ``DEFAULT_HANDLE_GRASP_OFFSET_M``
# (+X 2 cm forward, -Y 2 cm right) is the starting point for all three
# because every cylinder currently mounts the same handle profile.
# Re-tune per-cylinder by editing the per-Object entry below if a
# specific handle drifts left/right (Y) or near/far (X).
for _cylinder_object in (Object.PARMESAN, Object.SAUCE, Object.RICOTTA):
    OBJECT_DEFAULTS[_cylinder_object] = copy.deepcopy(OBJECT_DEFAULTS[Object.BOTTLE])
# Per-cylinder grasp orientation tuning. Sauce + ricotta tilt the wrist
# -30° about world Y on top of the Gemini grasp orientation, and
# correspondingly stage the above-grasp along the gripper's -tool_z so
# the descent moves along the gripper's facing direction (required
# whenever a wrist tilt is baked in — see
# ``ObjectSpec.approach_along_tool_z``). Parmesan keeps the defaults
# (0° extra rot, world-Z approach) so the gripper stays tool-down for
# the strip grasp at the rack — facing straight down on approach and
# during closure.
for _tilted_cyl in (Object.SAUCE, Object.RICOTTA):
    OBJECT_DEFAULTS[_tilted_cyl].grasp_extra_rot_deg = -30.0
    OBJECT_DEFAULTS[_tilted_cyl].grasp_extra_rot_axis = "y"
    OBJECT_DEFAULTS[_tilted_cyl].approach_along_tool_z = True
# Per-cylinder Z descent overrides. Sauce + ricotta sit a touch higher
# on the rack than parmesan, so their Gemini-detected handle points
# land ~2 cm above the actual grasp seam — drop the handle-mode grasp
# 2 cm in world Z so the gripper closes on the handle proper. Parmesan
# keeps the shared 0.0 Z (rack pose lines up cleanly). Edit per-cylinder
# here if a specific handle drifts further off.
OBJECT_DEFAULTS[Object.SAUCE].gemini_world_offset_m = np.array(
        [0.05, 0, -0.03], dtype=np.float64)
OBJECT_DEFAULTS[Object.SAUCE].approach_dz = 0.02
# Sauce jar is the heaviest cylinder on the rack — at the default live
# cartesian position_kp (100 N/m on this build) the arm visibly sags on
# the post-grasp lift and doesn't reach the carry pose. The boost is
# scoped INSIDE the phase machine: applied at ``Phase.GRASPED`` entry
# (right after the gripper closes — see ``_advance_phase``) and
# restored at ``Phase.AT_RELEASE`` exit (after the gripper opens), so
# the framing / above-grasp moves stay at the normal stiffness.
#
# Tuning history (kp/kv-only didn't work — see the ``ki`` note below):
#   * kp=300/kv=30 caused buzz on the rack handoff (kv too low).
#   * kp=400/kv=80 shook the arm violently — kv too high for the
#     measurement noise: the derivative term amplified encoder
#     ticks into actuator buzz. Don't push kv that high again.
#
# Current approach: small kp / kv bump + enable the integral term.
# The default XML config has ``ki_pos = 0`` so pure PD has no way
# to overcome a constant gravity offset — kp has to be huge to
# compensate, which is what caused the vibration. With ``ki`` on,
# the integrator builds up the steady-state force needed to hold
# the heavy jar at the carry pose without needing extreme kp/kv.
# The integrator needs ~0.5-1.0 s of error to wind up to a useful
# level, so the GRASPED→LIFTED move's tolerance is tightened and
# its timeout extended in ``_advance_phase`` to give it that time.
OBJECT_DEFAULTS[Object.SAUCE].lift_cart_position_kp = 150.0
OBJECT_DEFAULTS[Object.SAUCE].lift_cart_position_kv = 25.0
# ki tuning history:
#   * 10 didn't build enough integral force in the 8 s pretilt timeout
#     to close the lift error (still sagging at the
#     CYLINDER_PRETILT_TOL_M = 4 cm tolerance ball).
#   * 30 jerked the arm so hard at the moment of boost that it tripped
#     the Franka torque limits — too much accumulated error * ki in
#     the first tick after the gravity offset was suddenly visible.
# 20 is the middle ground: 2× the original wind-up rate (enough to
# beat the sag inside the 8 s window) but soft enough that the first
# tick's command stays below the torque-limit envelope. If buzz
# returns after grasp drop kv first (down to ~20) before lowering ki —
# the integral term itself is what supplies the missing gravity-comp
# force.
OBJECT_DEFAULTS[Object.SAUCE].lift_cart_position_ki = 0.0
OBJECT_DEFAULTS[Object.RICOTTA].gemini_world_offset_m = np.array([0.03, -0.02, -0.02])
OBJECT_DEFAULTS[Object.RICOTTA].approach_dz = 0.02

# Convenience grouping for "any cylinder" — used by
# ``grasp_and_pour_jar_controller`` when iterating over all three.
CYLINDER_OBJECTS: tuple[Object, ...] = (
    Object.PARMESAN,
    Object.SAUCE,
    Object.RICOTTA,
)

# Plastic bowl variants share the pasta bowl's grasp / pour parameters
# verbatim, except for the Gemini world offset (see below). Deep-copy
# so numpy arrays are independent and tuning one bowl doesn't bleed
# into the others.
OBJECT_DEFAULTS[Object.PLASTIC_BOWL_TOP] = copy.deepcopy(
    OBJECT_DEFAULTS[Object.PASTA_BOWL]
)
OBJECT_DEFAULTS[Object.PLASTIC_BOWL_BOTTOM] = copy.deepcopy(
    OBJECT_DEFAULTS[Object.PASTA_BOWL]
)
# The plastic bowls use a "midpoint, almost-no-offset" grasp:
#   * XY = midpoint of the two left-edge Gemini points (no XY shift).
#     The round pasta / mixing bowls add +Y 4 cm to seat the finger
#     around the curved rim, but the plastic bowls are rectangular
#     and ``_plastic_bowl_left_edge_prompt`` already returns two
#     points on the outer top lip of the left wall, so any extra +Y
#     just pushes the grasp past the wall into open air.
#   * Z = midpoint Z (from p1's depth sample) + 7 cm. Empirically enough
#     headroom that the fingers seat around (not on top of) the
#     plastic wall — RealSense reports the lip depth a couple of cm
#     low on the thin plastic rim, so anything <~5 cm risks the
#     gripper trying to close on the wall instead of around it.
for _plastic_bowl in (Object.PLASTIC_BOWL_TOP, Object.PLASTIC_BOWL_BOTTOM):
    OBJECT_DEFAULTS[_plastic_bowl].gemini_world_offset_m = np.array(
        [0.0, 0.0, 0.05], dtype=np.float64
    )

# Grasp these at the fixed home wrist attitude (``ARM_HOME_ORIENTATION``,
# the default ``grasp_ori``) WITHOUT rotating the jaws to the detected
# rim/strip/handle axis. The descent-to-grasp stays at the home
# orientation instead of yawing to line up with the chord between the
# two Gemini points. The cylinders (parmesan / sauce / ricotta) and the
# pan are intentionally excluded — cylinders build their own grasp
# orientation in ``grasp_and_pour_jar_controller`` and the pan uses a
# fixed taught pose, so neither consults this flag.
for _home_ori_grasp in (
    Object.PASTA_BOWL,
    Object.PLASTIC_BOWL_TOP,
    Object.PLASTIC_BOWL_BOTTOM,
    Object.MIXING_BOWL,
    Object.LADLE,
):
    OBJECT_DEFAULTS[_home_ori_grasp].grasp_align_jaws_to_detected_axis = False
# Egg cracker is intentionally NOT in the list above: its dedicated pose
# builder (``_build_pose_egg_cracker``) always yaws the jaws to go in AT the
# angle of the two handle points (closing ALONG the line between them), so it
# does not consult ``grasp_align_jaws_to_detected_axis``.
# Convenience grouping for "any bowl" — used by controllers that want
# a CLI-selectable bowl (see ``bowl_pour_controller.py``). Order is
# (default, alternates).
BOWL_OBJECTS: tuple[Object, ...] = (
    Object.PASTA_BOWL,
    Object.PLASTIC_BOWL_TOP,
    Object.PLASTIC_BOWL_BOTTOM,
)

# Taught EE pose over the pan center at PAN_STATION (world frame).
# Used as the start (upright) waypoint for the mixing bowl → pan pour sequence.
# Re-record by parking at PAN_STATION, holding the mixing bowl upright above
# the pan, and logging the EE world transform (T_end_effector from Redis).
MIXING_BOWL_PAN_POUR_START: tuple[np.ndarray, np.ndarray] = (
    np.array([+0.5561, +0.1550, +0.8448], dtype=np.float64),
    np.array(
        [
            [+0.8031, -0.5422, +0.2472],
            [-0.5613, -0.8275, +0.0086],
            [+0.1999, -0.1456, -0.9689],
        ],
        dtype=np.float64,
    ),
)

MIXING_BOWL_PAN_POUR_MID: tuple[np.ndarray, np.ndarray] = (
    np.array([+0.4992, +0.1212, +0.8070], dtype=np.float64),
    np.array(
        [
            [+0.7979, -0.5671, +0.2041],
            [-0.4635, -0.3608, +0.8093],
            [-0.3853, -0.7404, -0.5508],
        ],
        dtype=np.float64,
    ),
)

MIXING_BOWL_PAN_POUR_END: tuple[np.ndarray, np.ndarray] = (
    np.array([+0.5612, +0.1866, +0.8000], dtype=np.float64),
    np.array(
        [
            [+0.8340, -0.5516, -0.0172],
            [+0.0071, -0.0205, +0.9998],
            [-0.5518, -0.8339, -0.0132],
        ],
        dtype=np.float64,
    ),
)

# EE waypoint sequence for the mixing bowl → pan pour at PAN_STATION.
# Three waypoints: start (bowl upright above pan), mid (partway tilted),
# final (bowl fully tilted, contents emptied into pan).
# TODO: record mid and final waypoints from the live robot by holding the
# mixing bowl, jogging to each tilt pose at PAN_STATION, and reading
# T_end_effector from Redis. Replace the placeholder copies below.
MIXING_BOWL_PAN_POUR_EE_WAYPOINTS: tuple[tuple[np.ndarray, np.ndarray], ...] = (
    # Start: bowl upright above pan center (taught pose).
    MIXING_BOWL_PAN_POUR_START,
    MIXING_BOWL_PAN_POUR_MID,
    MIXING_BOWL_PAN_POUR_END,
)

# Per-bowl EE pour waypoint sequences. Bowls present in this dict are
# poured by walking the arm through these waypoints (forward = pour).
# Bowls absent from this dict fall back to the legacy slerp tilt at
# ``ARM_HOME_POSITION + +Z``. Plastic bowls share the pasta sequence by
# default — re-record per-bowl waypoints and assign them here if the
# geometry diverges.
BOWL_POUR_EE_WAYPOINTS: dict[Object, tuple[tuple[np.ndarray, np.ndarray], ...]] = {
    Object.PASTA_BOWL: PASTA_POUR_EE_WAYPOINTS,
    Object.PLASTIC_BOWL_TOP: PASTA_POUR_EE_WAYPOINTS,
    Object.PLASTIC_BOWL_BOTTOM: PASTA_POUR_EE_WAYPOINTS,
    Object.MIXING_BOWL: MIXING_BOWL_PAN_POUR_EE_WAYPOINTS,
}

COUNTER_Y_OFFSET_M = -2.8

BASE_WAYPOINTS: dict[BaseWaypoint, OptiPose] = {
    # Both stations face lab +Y (yaw=90°), so "back" from the counter is −Y.
    # Backed off 15 cm in Y from the original counter-edge poses to give
    # the arm more clearance from the counter edge during transport.
    # Logged 2026-06-05: cart parked at stirring station (OptiTrack body frame).
    BaseWaypoint.STIRRING_STATION: OptiPose(x_m=0.8555, y_m=-4.0656, yaw_deg=-90.0),
    # Logged 2026-06-05: cart parked at egg crack station (OptiTrack body frame).
    BaseWaypoint.EGG_CRACK_STATION: OptiPose(x_m=0.82, y_m=-4.09, yaw_deg=-90.0),
    BaseWaypoint.SINK_STATION: OptiPose(x_m=1.6, y_m=COUNTER_Y_OFFSET_M + 0.01, yaw_deg=90.0),
    BaseWaypoint.RACK_STATION: OptiPose(x_m=1.22, y_m=COUNTER_Y_OFFSET_M, yaw_deg=90.0),
    BaseWaypoint.INGREDIENT_STATION: OptiPose(x_m=1.12, y_m=COUNTER_Y_OFFSET_M, yaw_deg=90.0),
    BaseWaypoint.PASTA_STATION: OptiPose(x_m=0.9, y_m=COUNTER_Y_OFFSET_M, yaw_deg=90.0),
    BaseWaypoint.MIXING_STATION: OptiPose(x_m=0.54, y_m=COUNTER_Y_OFFSET_M, yaw_deg=90.0),
    # 20 cm past the mixing station along the counter, in the direction
    # away from the ingredient station (ingredient is at higher X, mixing
    # at lower X, so "away from ingredient" continues in −X). Same Y/yaw
    # as mixing so the arm faces the counter from the same side.
    BaseWaypoint.PRE_PAN_STATION: OptiPose(x_m=0.44, y_m=COUNTER_Y_OFFSET_M, yaw_deg=90.0),
    BaseWaypoint.PAN_STATION: OptiPose(x_m=0.34, y_m=COUNTER_Y_OFFSET_M, yaw_deg=90.0),
    BaseWaypoint.OVEN_DOOR: OptiPose(x_m=0.88, y_m=-3.93, yaw_deg=0.0),
    # Logged 2026-06-05: cart parked at stove station (OptiTrack body frame).
    BaseWaypoint.STOVE_STATION: OptiPose(x_m=1.5, y_m=-4.07, yaw_deg=-90.0),
}
