"""OptiTrack → hb1 calibration and motion planning."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

import numpy as np
import redis

from tidybot_base.mocap import read_mocap_pose
from tidybot_base.redis_io import read_robot_se2
from tidybot_base.se2 import (
    quat_xyzw_to_rot2d,
    quat_xyzw_to_yaw,
    rot2d_from_yaw,
    rot2d_yaw,
    wrap_angle,
    xy2,
)

# Marker → body yaw offset (deg, CCW positive in Opti world):
#
#     mocap_yaw_marker = body_yaw_in_opti + marker_yaw_offset_deg
#
# In Motive the rigid-body's local +X axis is auto-assigned when you create
# the rigid body. It generally is **not** aligned with the robot's actual
# forward driving direction (body +X). This offset captures that mismatch
# so Opti-world delta vectors map correctly into hb (body) world.
#
# Used only by ``calibrate_mocap_to_hb`` to fold the offset into ``calib.rot``.
# Yaw commands do NOT need this offset — when both the start and target yaw
# are in the marker frame, the offset cancels in the delta.
#
# Locked-in default ``+140.14 deg`` matches the current ``tidybot01`` rigid body
# (estimated by ``tune_marker_offset.py`` on 2026-05-29 — re-tune if you
# re-create the rigid body in Motive). Override via
# ``--marker-yaw-offset-deg`` on ``opti_controller``, or by passing a
# ``NavConfig(marker_yaw_offset_deg=...)`` to ``opti_nav``.
DEFAULT_MARKER_YAW_OFFSET_DEG = -38.17

# Legacy alias kept for backwards-compat with prior code/CLI.
ROBOT_FRAME_ROT_DEG = DEFAULT_MARKER_YAW_OFFSET_DEG

# Marker +X in Motive lab points along this axis (rad).
LAB_AXIS_YAW_RAD: dict[str, float] = {
    "plus-x": 0.0,
    "minus-x": math.pi,
    "plus-y": math.pi / 2.0,
    "minus-y": -math.pi / 2.0,
}


class GoalFrame(enum.Enum):
    """How goal deltas are interpreted at startup (Opti only)."""

    OPTI_BODY = "opti_body"  # rigid-body / marker XY (from Opti quaternion)
    OPTI_WORLD = "opti_world"  # Motive lab X/Y on the floor


# Unit axis in the selected frame; multiplied by --goal-distance-ft (positive distance).
GOAL_ALONG_AXES: dict[str, tuple[GoalFrame, tuple[float, float]]] = {
    "lab-minus-x": (GoalFrame.OPTI_WORLD, (-1.0, 0.0)),
    "lab-plus-x": (GoalFrame.OPTI_WORLD, (1.0, 0.0)),
    "lab-minus-y": (GoalFrame.OPTI_WORLD, (0.0, -1.0)),
    "lab-plus-y": (GoalFrame.OPTI_WORLD, (0.0, 1.0)),
    "body-minus-x": (GoalFrame.OPTI_BODY, (-1.0, 0.0)),
    "body-plus-x": (GoalFrame.OPTI_BODY, (1.0, 0.0)),
    "body-minus-y": (GoalFrame.OPTI_BODY, (0.0, -1.0)),
    "body-plus-y": (GoalFrame.OPTI_BODY, (0.0, 1.0)),
}


@dataclass(frozen=True)
class MocapToHbCalib:
    """SE(2) calibration: mocap XY (OptiTrack world) → hb1 odom XY."""

    rot: np.ndarray  # 2×2, p_hb = rot @ p_mocap + trans
    trans: np.ndarray  # length 2
    mocap_rot: np.ndarray  # 2×2 horizontal rotation from mocap quaternion
    mocap_start_yaw: float


@dataclass
class MotionCalibEstimate:
    """Live Opti-world → hb-odom direction estimate from observed translation."""

    rot: np.ndarray
    yaw_rad: float
    hb_motion_m: float
    opti_motion_m: float
    sample_count: int


@dataclass
class RunPlan:
    """Mocap goal plus hb waypoints (cardinal hb motion or single straight goal)."""

    desired_mocap_xyz: np.ndarray
    desired_mocap_yaw: float  # target marker yaw in Opti world (what quaternion will read at goal)
    desired_body_yaw_in_opti: float  # target body yaw = marker yaw - offset (what user asked for)
    mocap_start_xyz: np.ndarray
    mocap_start_yaw: float  # marker yaw in Opti world (from quaternion)
    body_start_yaw_in_opti: float  # body yaw in Opti world = marker yaw - offset
    goal_delta_input_xy: np.ndarray  # meters, in frame given by goal_frame
    goal_frame: GoalFrame
    mocap_delta_world_xy: np.ndarray  # Opti/lab goal delta (input → calib.rot)
    hb_delta_xy: np.ndarray
    robot_start: np.ndarray
    robot_target: np.ndarray  # final hb1 [x, y, yaw]
    face_lab_yaw: str | None  # e.g. minus-y, or None
    hb_target_yaw: float
    hb_waypoints: list[np.ndarray]  # intermediate + final poses for redis
    calib: MocapToHbCalib
    marker_yaw_offset_deg: float  # marker → body mounting offset in calib
    absolute_target: bool  # desired mocap pose given explicitly (not axis delta)
    require_final_yaw: bool  # success checks hb yaw vs target


def calibrate_mocap_to_hb(
    robot_start: np.ndarray,
    mocap_xyz: np.ndarray,
    mocap_quat: np.ndarray,
    *,
    translation_only: bool = False,
    calib_yaw_deg: float | None = None,
    marker_yaw_offset_deg: float = DEFAULT_MARKER_YAW_OFFSET_DEG,
) -> MocapToHbCalib:
    """Fit ``p_hb = R @ p_mocap + t`` at startup (Opti orientation is ground truth).

    Default: ``R = R(-(mocap_yaw - marker_yaw_offset))`` so hb odom +X is the
    robot **body** +X in Motive at power-on. With ``marker_yaw_offset_deg=0``
    we assume the Motive rigid-body's local +X is the robot's forward direction;
    set the offset to compensate when Motive's auto-assigned axes are rotated
    relative to the cart's driving direction (CCW positive). Body yaw in Opti
    world = ``mocap_yaw - marker_yaw_offset``.

    Hb ``current_pose`` yaw is not used. Use ``translation_only`` if Motive
    XY axes equal hb XY axes directly.
    """
    mocap_rot = quat_xyzw_to_rot2d(mocap_quat)
    mocap_yaw_marker = rot2d_yaw(mocap_rot)
    body_yaw_in_opti = wrap_angle(
        mocap_yaw_marker - math.radians(marker_yaw_offset_deg)
    )
    if calib_yaw_deg is not None:
        rot = rot2d_from_yaw(math.radians(calib_yaw_deg))
    elif translation_only:
        rot = np.eye(2, dtype=np.float64)
    else:
        rot = rot2d_from_yaw(-body_yaw_in_opti)
    trans = robot_start[:2] - rot @ xy2(mocap_xyz)
    return MocapToHbCalib(
        rot=rot,
        trans=trans,
        mocap_rot=mocap_rot,
        mocap_start_yaw=mocap_yaw_marker,
    )


def mocap_lab_yaw_to_hb_yaw(
    desired_mocap_yaw: float,
    mocap_start_yaw: float,
    robot_start_yaw: float,
) -> float:
    """Hb odom yaw for a Motive lab marker heading.

    The marker→body yaw offset cancels here: both ``desired_mocap_yaw`` and
    ``mocap_start_yaw`` are marker yaws, and the body rotates the same amount
    as the marker (they're glued together). So the hb yaw delta equals the
    marker yaw delta.
    """
    return wrap_angle(
        float(robot_start_yaw) + wrap_angle(desired_mocap_yaw - mocap_start_yaw)
    )


def mocap_xy_to_hb(mocap_xy: np.ndarray, calib: MocapToHbCalib) -> np.ndarray:
    return calib.rot @ xy2(mocap_xy) + calib.trans


def mocap_pose_to_hb_se2(
    mocap_xyz: np.ndarray,
    calib: MocapToHbCalib,
    hb_yaw_hold: float,
) -> np.ndarray:
    """Map Opti position into hb1 XY; keep hb yaw fixed (not commanded from Opti)."""
    hb_xy = mocap_xy_to_hb(mocap_xyz, calib)
    return np.array([hb_xy[0], hb_xy[1], hb_yaw_hold], dtype=np.float64)


class MotionDirectionEstimator:
    """Estimate Opti-world → hb-odom rotation from matching displacement pairs.

    This removes the steady-state dependence on Motive's arbitrary rigid-body
    marker axes. The first few centimeters still use the seeded rotation, then
    observed hb/Opti motion updates the rotation used by the replanner.
    """

    def __init__(
        self,
        *,
        seed_rot: np.ndarray,
        seed_robot_xyyaw: np.ndarray,
        seed_mocap_xyz: np.ndarray,
        min_motion_m: float = 0.08,
        max_yaw_change_rad: float = math.radians(8.0),
    ) -> None:
        self._seed_rot = np.asarray(seed_rot, dtype=np.float64).reshape(2, 2)
        self._anchor_robot = np.asarray(seed_robot_xyyaw, dtype=np.float64).reshape(3).copy()
        self._anchor_mocap_xy = xy2(seed_mocap_xyz).copy()
        self._min_motion_m = float(min_motion_m)
        self._max_yaw_change_rad = float(max_yaw_change_rad)
        self._estimate: MotionCalibEstimate | None = None

    @property
    def estimate(self) -> MotionCalibEstimate | None:
        return self._estimate

    @property
    def rot(self) -> np.ndarray:
        if self._estimate is None:
            return self._seed_rot
        return self._estimate.rot

    def reset_anchor(self, robot_xyyaw: np.ndarray, mocap_xyz: np.ndarray) -> None:
        """Start a new straight-translation observation window."""
        self._anchor_robot = np.asarray(robot_xyyaw, dtype=np.float64).reshape(3).copy()
        self._anchor_mocap_xy = xy2(mocap_xyz).copy()
        self._estimate = None

    def update(
        self,
        robot_xyyaw: np.ndarray,
        mocap_xyz: np.ndarray,
    ) -> MotionCalibEstimate | None:
        """Update from current hb/Opti displacement; return estimate if valid."""
        robot = np.asarray(robot_xyyaw, dtype=np.float64).reshape(3)
        yaw_delta = abs(wrap_angle(float(robot[2]) - float(self._anchor_robot[2])))
        if yaw_delta > self._max_yaw_change_rad:
            return None

        hb_delta = xy2(robot) - xy2(self._anchor_robot)
        opti_delta = xy2(mocap_xyz) - self._anchor_mocap_xy
        hb_norm = float(np.linalg.norm(hb_delta))
        opti_norm = float(np.linalg.norm(opti_delta))
        if hb_norm < self._min_motion_m or opti_norm < self._min_motion_m:
            return None

        yaw = wrap_angle(
            math.atan2(float(hb_delta[1]), float(hb_delta[0]))
            - math.atan2(float(opti_delta[1]), float(opti_delta[0]))
        )
        self._estimate = MotionCalibEstimate(
            rot=rot2d_from_yaw(yaw),
            yaw_rad=yaw,
            hb_motion_m=hb_norm,
            opti_motion_m=opti_norm,
            sample_count=1,
        )
        return self._estimate


def replan_hb_goal_from_opti(
    plan: "RunPlan",
    waypoint_idx: int,
    mocap_xyz: np.ndarray,
    mocap_quat: np.ndarray,
    robot_current: np.ndarray,
    opti_to_hb_rot: np.ndarray | None = None,
) -> np.ndarray:
    """Live-corrected hb goal: closes the current Opti-world error in hb frame.

    The startup calibration ``calib.rot`` is fixed and accurate (it encodes
    ``R(-body_yaw_in_opti_at_startup)``). But hb odom can drift slowly
    relative to Opti truth, and during a pivot the marker can slide a few
    millimetres. Each loop we therefore compute the remaining Opti-world
    error and rotate it into hb world, then add it to the live hb pose. The
    base controller drives hb to that updated goal; as the Opti error
    shrinks the goal converges, and any drift gets continuously nudged out.

    XY:  ``hb_current + R @ (opti_target_xy − opti_current_xy)`` where
         ``R`` is either the startup calibration or a live motion estimate.
    Yaw: at the final waypoint with ``require_final_yaw``, use the same
         Opti→hb direction estimate when available; otherwise fall back to
         marker-yaw delta. Translation-only phases hold startup hb yaw.
    """
    is_final_waypoint = waypoint_idx >= len(plan.hb_waypoints) - 1
    opti_target_xy = xy2(plan.desired_mocap_xyz)
    opti_err_xy = opti_target_xy - xy2(mocap_xyz)
    rot = plan.calib.rot if opti_to_hb_rot is None else opti_to_hb_rot
    hb_xy = xy2(robot_current) + rot @ opti_err_xy

    if is_final_waypoint and plan.require_final_yaw:
        if opti_to_hb_rot is not None:
            rot_yaw = rot2d_yaw(opti_to_hb_rot)
            body_yaw_now = wrap_angle(float(robot_current[2]) - rot_yaw)
            hb_yaw = wrap_angle(
                float(robot_current[2])
                + wrap_angle(plan.desired_body_yaw_in_opti - body_yaw_now)
            )
        else:
            marker_yaw_now = quat_xyzw_to_yaw(mocap_quat)
            hb_yaw = wrap_angle(
                float(robot_current[2])
                + wrap_angle(plan.desired_mocap_yaw - marker_yaw_now)
            )
    else:
        hb_yaw = float(plan.robot_start[2])

    return np.array([hb_xy[0], hb_xy[1], hb_yaw], dtype=np.float64)


def opti_xy_distance_m(plan: "RunPlan", mocap_xyz: np.ndarray) -> float:
    """Opti-world XY distance from the current mocap position to the goal."""
    return float(np.linalg.norm(xy2(mocap_xyz) - xy2(plan.desired_mocap_xyz)))


def opti_body_yaw_error_rad(plan: "RunPlan", mocap_quat: np.ndarray) -> float:
    """Absolute body-yaw error in Opti world (|target − current|, wrapped)."""
    marker_yaw = quat_xyzw_to_yaw(mocap_quat)
    body_yaw = wrap_angle(marker_yaw - math.radians(plan.marker_yaw_offset_deg))
    return abs(wrap_angle(plan.desired_body_yaw_in_opti - body_yaw))


def opti_body_yaw_error_from_motion_calib_rad(
    plan: "RunPlan",
    robot_current: np.ndarray,
    opti_to_hb_rot: np.ndarray,
) -> float:
    """Body-yaw error in Opti world using hb yaw plus estimated frame rotation."""
    rot_yaw = rot2d_yaw(opti_to_hb_rot)
    body_yaw = wrap_angle(float(robot_current[2]) - rot_yaw)
    return abs(wrap_angle(plan.desired_body_yaw_in_opti - body_yaw))


def plan_translation(
    mocap_xyz: np.ndarray,
    calib: MocapToHbCalib,
    goal_delta_input_xy: np.ndarray,
    goal_frame: GoalFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (desired_mocap_xyz, mocap_goal_delta_opti_world, hb_delta).

    ``calib.rot`` already encodes (-body_yaw_in_opti) including the marker
    mounting offset, so we map Opti-world deltas straight into hb world.
    """
    delta_in = xy2(goal_delta_input_xy)
    if goal_frame == GoalFrame.OPTI_BODY:
        mocap_goal_delta = calib.mocap_rot @ delta_in
    else:
        mocap_goal_delta = delta_in
    hb_delta = calib.rot @ mocap_goal_delta

    desired_xyz = mocap_xyz.copy()
    desired_xyz[:2] = xy2(mocap_xyz) + mocap_goal_delta
    return desired_xyz, mocap_goal_delta, hb_delta


def plan_absolute_mocap_goal(
    mocap_xyz: np.ndarray,
    desired_mocap_xyz: np.ndarray,
    calib: MocapToHbCalib,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (desired_mocap_xyz, mocap_goal_delta_opti_world, hb_delta).

    ``calib.rot`` already encodes (-body_yaw_in_opti) including the marker
    mounting offset, so we map Opti-world deltas straight into hb world.
    """
    desired_xyz = np.asarray(desired_mocap_xyz, dtype=np.float64).reshape(3)
    mocap_goal_delta = xy2(desired_xyz) - xy2(mocap_xyz)
    hb_delta = calib.rot @ mocap_goal_delta
    return desired_xyz, mocap_goal_delta, hb_delta


def build_hb_waypoints(
    robot_start: np.ndarray,
    robot_target: np.ndarray,
    *,
    cardinal_hb: bool,
    direct_motion: bool = True,
    straight_line: bool = False,
    rotate_first: bool = False,
) -> list[np.ndarray]:
    """Build hb1 desired_pose waypoints.

    ``straight_line=True``: one goal ``[x, y, start_yaw]`` — planar straight segment,
    no heading change (default for absolute position-only goals).

    ``direct_motion=True``: one holonomic goal ``[x, y, yaw]`` (XY + heading together).
    Takes precedence over ``rotate_first``.

    Default (no flags): translate at start yaw, then rotate in place at end.

    ``rotate_first=True`` (and not ``direct_motion``/``straight_line``): rotate
    in place at the start XY first, then translate at the final yaw. Useful
    when the cart's driving direction matters for the translation phase
    (e.g. you want to drive forward at the new heading).

    With ``cardinal_hb`` (and not direct / not rotate_first): L-path in hb
    odom (X corner, then Y).
    """
    final = np.asarray(robot_target, dtype=np.float64).reshape(3)
    hold_yaw = float(robot_start[2])
    final_yaw = float(final[2])
    dx = final[0] - robot_start[0]
    dy = final[1] - robot_start[1]
    translate_xy = abs(dx) > 1e-4 or abs(dy) > 1e-4
    rotate_yaw = abs(wrap_angle(final_yaw - hold_yaw)) > 1e-3

    if not translate_xy and not rotate_yaw:
        return []

    if straight_line:
        return [np.array([final[0], final[1], hold_yaw], dtype=np.float64)]

    if direct_motion:
        return [final]

    if rotate_yaw and not translate_xy:
        return [final]

    if rotate_first and rotate_yaw and translate_xy:
        rotate_wp = np.array(
            [float(robot_start[0]), float(robot_start[1]), final_yaw],
            dtype=np.float64,
        )
        return [rotate_wp, final]

    trans_xy = np.array([final[0], final[1], hold_yaw], dtype=np.float64)
    waypoints: list[np.ndarray] = []
    if cardinal_hb and abs(dx) > 1e-4 and abs(dy) > 1e-4:
        corner = np.array([final[0], robot_start[1], hold_yaw], dtype=np.float64)
        waypoints.append(corner)
    waypoints.append(trans_xy)
    if rotate_yaw:
        waypoints.append(final)
    return waypoints


def waypoint_reached(
    robot_current: np.ndarray,
    waypoint: np.ndarray,
    waypoint_idx: int,
    waypoints: list[np.ndarray],
    *,
    tolerance_m: float,
    tolerance_yaw_rad: float,
) -> bool:
    """Last waypoint needs XY + yaw; earlier ones only XY (in-place turn is separate)."""
    if float(np.linalg.norm(robot_current[:2] - waypoint[:2])) >= tolerance_m:
        return False
    is_last = waypoint_idx >= len(waypoints) - 1
    if not is_last:
        return True
    return abs(wrap_angle(float(robot_current[2]) - float(waypoint[2]))) < tolerance_yaw_rad


def setup_run_plan(
    client: redis.Redis,
    *,
    goal_delta_input_xy: np.ndarray,
    goal_frame: GoalFrame,
    translation_only_calib: bool = False,
    calib_yaw_deg: float | None = None,
    cardinal_hb: bool = False,
    direct_motion: bool = False,
    rotate_first: bool = False,
    marker_yaw_offset_deg: float = DEFAULT_MARKER_YAW_OFFSET_DEG,
    face_lab_yaw: str | None = None,
    robot_pose_key: str | None = None,
    mocap_pos_key: str | None = None,
    mocap_ori_key: str | None = None,
    curr_minus_desired: bool = False,
    absolute_mocap_target: tuple[np.ndarray, float | None] | None = None,
) -> RunPlan:
    robot_start = read_robot_se2(client, robot_pose_key)
    mocap_xyz, mocap_quat = read_mocap_pose(client, mocap_pos_key, mocap_ori_key)
    mocap_yaw = quat_xyzw_to_yaw(mocap_quat)
    calib = calibrate_mocap_to_hb(
        robot_start,
        mocap_xyz,
        mocap_quat,
        translation_only=translation_only_calib,
        calib_yaw_deg=calib_yaw_deg,
        marker_yaw_offset_deg=marker_yaw_offset_deg,
    )
    body_start_yaw_in_opti = wrap_angle(
        mocap_yaw - math.radians(marker_yaw_offset_deg)
    )
    offset_rad = math.radians(marker_yaw_offset_deg)

    # Target yaw given by the user is interpreted in **body** frame (the cart's
    # actual driving direction in Motive lab). We then add the marker offset
    # to get the corresponding marker-yaw goal, which is what the hb yaw
    # mapping consumes (since the hb yaw delta equals the marker yaw delta).
    absolute_target = absolute_mocap_target is not None
    target_body_yaw_rad: float | None = None
    if absolute_target:
        desired_xyz_in, target_body_yaw_rad = absolute_mocap_target
        desired_xyz, mocap_goal_delta, hb_delta = plan_absolute_mocap_goal(
            mocap_xyz,
            desired_xyz_in,
            calib,
        )
        goal_delta = mocap_goal_delta
        goal_frame = GoalFrame.OPTI_WORLD
        face_lab_yaw = None
        if target_body_yaw_rad is None:
            desired_body_yaw = body_start_yaw_in_opti
            desired_mocap_yaw = mocap_yaw
            hb_target_yaw = float(robot_start[2])
            require_final_yaw = False
        else:
            desired_body_yaw = wrap_angle(target_body_yaw_rad)
            desired_mocap_yaw = wrap_angle(desired_body_yaw + offset_rad)
            hb_target_yaw = mocap_lab_yaw_to_hb_yaw(
                desired_mocap_yaw,
                mocap_yaw,
                float(robot_start[2]),
            )
            require_final_yaw = True
    else:
        goal_delta = xy2(goal_delta_input_xy)
        desired_xyz, mocap_goal_delta, hb_delta = plan_translation(
            mocap_xyz,
            calib,
            goal_delta,
            goal_frame,
        )
        if face_lab_yaw is not None:
            desired_body_yaw = wrap_angle(LAB_AXIS_YAW_RAD[face_lab_yaw])
            desired_mocap_yaw = wrap_angle(desired_body_yaw + offset_rad)
            require_final_yaw = True
        else:
            desired_body_yaw = body_start_yaw_in_opti
            desired_mocap_yaw = mocap_yaw
            require_final_yaw = False
        hb_target_yaw = mocap_lab_yaw_to_hb_yaw(
            desired_mocap_yaw,
            mocap_yaw,
            float(robot_start[2]),
        )
    robot_target = np.array(
        [
            robot_start[0] + hb_delta[0],
            robot_start[1] + hb_delta[1],
            hb_target_yaw,
        ],
        dtype=np.float64,
    )
    straight_line = absolute_target and target_body_yaw_rad is None
    hb_waypoints = build_hb_waypoints(
        robot_start,
        robot_target,
        cardinal_hb=cardinal_hb,
        direct_motion=direct_motion and not straight_line,
        straight_line=straight_line,
        rotate_first=rotate_first and not straight_line,
    )

    # curr_minus_desired only affects mocap error logging, not hb target mapping
    _ = curr_minus_desired

    return RunPlan(
        desired_mocap_xyz=desired_xyz,
        desired_mocap_yaw=desired_mocap_yaw,
        desired_body_yaw_in_opti=desired_body_yaw,
        mocap_start_xyz=mocap_xyz,
        mocap_start_yaw=mocap_yaw,
        body_start_yaw_in_opti=body_start_yaw_in_opti,
        goal_delta_input_xy=goal_delta,
        goal_frame=goal_frame,
        mocap_delta_world_xy=mocap_goal_delta,
        hb_delta_xy=hb_delta,
        robot_start=robot_start,
        robot_target=robot_target,
        face_lab_yaw=face_lab_yaw,
        hb_target_yaw=hb_target_yaw,
        hb_waypoints=hb_waypoints,
        calib=calib,
        marker_yaw_offset_deg=marker_yaw_offset_deg,
        absolute_target=absolute_target,
        require_final_yaw=require_final_yaw,
    )


def mocap_pose_error(
    current_xyz: np.ndarray,
    current_yaw: float,
    desired_xyz: np.ndarray,
    desired_yaw: float,
    *,
    curr_minus_desired: bool,
) -> np.ndarray:
    """Error in mocap frame: [dx, dy, dz, dyaw] (m, m, m, rad)."""
    cur = np.asarray(current_xyz, dtype=np.float64).reshape(3)
    des = np.asarray(desired_xyz, dtype=np.float64).reshape(3)
    if curr_minus_desired:
        dxyz = cur - des
        dyaw = wrap_angle(current_yaw - desired_yaw)
    else:
        dxyz = des - cur
        dyaw = wrap_angle(desired_yaw - current_yaw)
    return np.array([dxyz[0], dxyz[1], dxyz[2], dyaw], dtype=np.float64)
