#!/usr/bin/env python3
"""OptiTrack → TidyBot base pose controller (Redis only, dev laptop).

Reads rigid-body pose from OptiTrack Redis keys, compares to a desired mocap
pose, maps the error into the robot odometry frame, and publishes
``hb1::desired_pose`` for ``tidybot_base/redis_driver.py`` (run ``redis_driver.py``
only — not ``base_controller.py`` as a separate process).

**Frames**

- *Mocap*: OptiTrack on ``tidybot01::pos`` (xyz, m) and ``tidybot01::ori`` (quat xyzw).
- *Robot*: ``hb1::current_pose`` is wheel odometry (m, m, rad). **Opti is ground truth**
  for pose and heading. At startup we record Opti + hb together; lab-frame goals use
  Motive axes, body-frame goals use the Opti quaternion. Hb yaw is not used for
  mapping (only held fixed on ``hb1::desired_pose``).

**Units**

- Base / Redis driver: **m**, **rad**, **m/s**, **rad/s** (see ``Vehicle`` in
  ``tidybot_base/base_controller.py``; wheel radius 0.0508 m).
- Mocap ``tidybot01::pos``: OptiTrack world **meters** (xyz).
- ``--goal-offset-ft`` is converted to meters for mocap goals.
- ``--tolerance-in`` is converted to meters. **Success** is when
  ``hb1::current_pose`` is within that distance of the fixed ``hb1::desired_pose``
  target (computed once from Opti at startup), not when mocap error is small.

**Goals**

- Absolute (default): Motive position ``(-1.5, 1.0, 0.45)`` m, straight-line drive at
  startup heading (no orientation change). Optional ``--target-yaw-deg``.
- Relative: ``--relative-goal`` then ``--goal-along lab-plus-y`` (default 1.5 ft), etc.

Lab→hb uses **Opti orientation only**: hb odom +X is marker +X in the lab at startup
(``R = R_mocap^T``).

**Speed**

``opti_controller`` only sets position goals on Redis. Base speed/accel are set in
``tidybot_base/redis_driver.py`` (default ``max_vel=(0.25, 0.25, 0.79)`` m/s and
rad/s) or ``launch_opti_controller.sh --max-vel-xy``.

**Prerequisites**

- Redis publishing ``tidybot01::pos`` / ``tidybot01::ori`` / ``tidybot01::tracking_valid``
  (mocap) and ``hb1::*`` (base). Motion only runs while ``tracking_valid`` is true.
- ``tidybot_base/redis_driver.py`` running on the robot mini-PC (starts ``Vehicle``
  from ``base_controller.py`` internally).

Usage::

  python ZitiBot/controllers/opti_controller.py
  python ZitiBot/controllers/opti_controller.py --monitor
"""

from __future__ import annotations

import argparse
import ast
import enum
import json
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import redis

FEET_TO_METERS = 0.3048
INCHES_TO_METERS = 0.0254
CONTROL_HZ = 100.0
DEFAULT_GOAL_OFFSET_FT = -1.5  # legacy: Motive lab −X when using --goal-offset-ft
DEFAULT_GOAL_DISTANCE_FT = 1.5
DEFAULT_GOAL_ALONG = "lab-plus-y"  # Motive lab +Y
DEFAULT_TOLERANCE_IN = 1.0
DEFAULT_TARGET_X = -1.5
DEFAULT_TARGET_Y = 1.0
DEFAULT_TARGET_Z = 0.45
# Always applied when mapping Opti goals → hb commands (translation and yaw).
ROBOT_FRAME_ROT_DEG = 45.0
DEFAULT_FACE_LAB_YAW = "minus-y"  # after translation, rotate to face this Motive lab axis
DEFAULT_YAW_TOLERANCE_DEG = 5.0

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
class RedisKeys:
    # OptiTrack / mocap (external tracking)
    mocap_pos: str = "tidybot01::pos"
    mocap_ori: str = "tidybot01::ori"
    tracking_valid: str = "tidybot01::tracking_valid"
    # TidyBot base odometry + commands (redis_driver / Vehicle)
    robot_pose: str = "hb1::current_pose"
    robot_vel: str = "hb1::current_vel"
    desired_pose: str = "hb1::desired_pose"
    stop: str = "hb1::stop"


KEYS = RedisKeys()


@dataclass(frozen=True)
class MocapToHbCalib:
    """SE(2) calibration: mocap XY (OptiTrack world) → hb1 odom XY."""

    rot: np.ndarray  # 2×2, p_hb = rot @ p_mocap + trans
    trans: np.ndarray  # length 2
    mocap_rot: np.ndarray  # 2×2 horizontal rotation from mocap quaternion
    mocap_start_yaw: float


@dataclass
class RunPlan:
    """Mocap goal plus hb waypoints (cardinal hb motion or single straight goal)."""

    desired_mocap_xyz: np.ndarray
    desired_mocap_yaw: float
    mocap_start_xyz: np.ndarray
    mocap_start_yaw: float
    goal_delta_input_xy: np.ndarray  # meters, in frame given by goal_frame
    goal_frame: GoalFrame
    mocap_delta_world_xy: np.ndarray  # Opti/lab goal delta (no robot-frame rot)
    mocap_delta_rotated_xy: np.ndarray  # after robot-frame rot → mapped to hb
    hb_delta_xy: np.ndarray
    robot_start: np.ndarray
    robot_target: np.ndarray  # final hb1 [x, y, yaw]
    face_lab_yaw: str | None  # e.g. minus-y, or None
    hb_target_yaw: float
    hb_waypoints: list[np.ndarray]  # intermediate + final poses for redis
    calib: MocapToHbCalib
    robot_input_rot_deg: float
    mocap_delta_before_robot_rot_xy: np.ndarray
    absolute_target: bool  # desired mocap pose given explicitly (not axis delta)
    require_final_yaw: bool  # success checks hb yaw vs target


def numpy_array_to_string(array: np.ndarray) -> str:
    if isinstance(array, np.ndarray) and array.ndim == 1:
        return "[" + ", ".join(map(str, array.tolist())) + "]"
    return ""


def parse_redis_list(raw: bytes | str | None) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    raw = raw.strip()
    if raw.startswith("["):
        try:
            values = ast.literal_eval(raw)
            if isinstance(values, list):
                return np.array(values, dtype=np.float64)
        except (SyntaxError, ValueError):
            pass
    try:
        values = json.loads(raw)
        if isinstance(values, list):
            return np.array(values, dtype=np.float64)
    except json.JSONDecodeError:
        pass
    return None


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_xyzw_to_yaw(q: np.ndarray) -> float:
    """Heading from body +X projected onto the horizontal plane (rad)."""
    return rot2d_yaw(quat_xyzw_to_rot2d(q))


def quat_xyzw_to_rot2d(q: np.ndarray) -> np.ndarray:
    """Horizontal 2×2 rotation from quaternion (xyzw)."""
    x, y, z, w = (float(q[i]) for i in range(4))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z)],
        ],
        dtype=np.float64,
    )


def rot2d_from_yaw(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rot2d_yaw(rot: np.ndarray) -> float:
    return math.atan2(float(rot[1, 0]), float(rot[0, 0]))


def parse_tracking_valid(raw: bytes | str | None) -> bool:
    """Interpret Redis ``tidybot01::tracking_valid`` (true/1/yes/on)."""
    if raw is None:
        return False
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    try:
        return bool(int(float(s)))
    except ValueError:
        return False


def read_tracking_valid(client: redis.Redis, key: str | None = None) -> bool:
    return parse_tracking_valid(client.get(key or KEYS.tracking_valid))


def stop_base(client: redis.Redis) -> None:
    """Tell redis_driver to decelerate and hold current pose."""
    client.set(KEYS.stop, "stop")


def wait_for_tracking_valid(
    client: redis.Redis,
    key: str,
    *,
    poll_hz: float = 10.0,
) -> None:
    period = 1.0 / max(poll_hz, 0.1)
    last_msg = 0.0
    print(f"Waiting for {key!r} == true before planning or motion...")
    while True:
        if read_tracking_valid(client, key):
            print(f"{key} is true — proceeding.")
            return
        now = time.perf_counter()
        if now - last_msg >= 2.0:
            raw = client.get(key)
            print(f"  still waiting ({key}={raw!r})")
            last_msg = now
        time.sleep(period)


def connect_redis(host: str, port: int) -> redis.Redis:
    client = redis.Redis(host=host, port=port, decode_responses=True)
    client.ping()
    return client


def read_robot_se2(client: redis.Redis, pose_key: str | None = None) -> np.ndarray:
    key = pose_key or KEYS.robot_pose
    pos = parse_redis_list(client.get(key))
    if pos is None or pos.size < 3:
        raise RuntimeError(f"Invalid {key!r} (need [x, y, yaw])")
    return pos[:3].astype(np.float64)


def read_mocap_pose(
    client: redis.Redis,
    pos_key: str | None = None,
    ori_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pk = pos_key or KEYS.mocap_pos
    ok = ori_key or KEYS.mocap_ori
    pos = parse_redis_list(client.get(pk))
    ori = parse_redis_list(client.get(ok))
    if pos is None or pos.size < 3:
        raise RuntimeError(f"Missing or invalid {pk!r} (is OptiTrack on Redis?)")
    if ori is None or ori.size < 4:
        raise RuntimeError(f"Missing or invalid {ok!r}")
    return pos[:3].astype(np.float64), ori[:4].astype(np.float64)


def calibrate_mocap_to_hb(
    robot_start: np.ndarray,
    mocap_xyz: np.ndarray,
    mocap_quat: np.ndarray,
    *,
    translation_only: bool = False,
    calib_yaw_deg: float | None = None,
) -> MocapToHbCalib:
    """Fit ``p_hb = R @ p_mocap + t`` at startup (Opti orientation is ground truth).

    Default: ``R = R_mocap^T`` so hb odom +X matches marker +X in Motive at power-on.
    Hb ``current_pose`` yaw is not used. Use ``translation_only`` if Motive XY = hb XY.
    """
    mocap_rot = quat_xyzw_to_rot2d(mocap_quat)
    mocap_yaw = rot2d_yaw(mocap_rot)
    if calib_yaw_deg is not None:
        rot = rot2d_from_yaw(math.radians(calib_yaw_deg))
    elif translation_only:
        rot = np.eye(2, dtype=np.float64)
    else:
        rot = mocap_rot.T
    trans = robot_start[:2] - rot @ _xy2(mocap_xyz)
    return MocapToHbCalib(
        rot=rot,
        trans=trans,
        mocap_rot=mocap_rot,
        mocap_start_yaw=mocap_yaw,
    )


def mocap_lab_yaw_to_hb_yaw(
    desired_mocap_yaw: float,
    mocap_start_yaw: float,
    robot_start_yaw: float,
    robot_frame_rot_deg: float = ROBOT_FRAME_ROT_DEG,
) -> float:
    """Hb odom yaw for a Motive lab heading + fixed robot-frame rotation offset."""
    hb_yaw = float(robot_start_yaw) + wrap_angle(desired_mocap_yaw - mocap_start_yaw)
    return wrap_angle(hb_yaw + math.radians(robot_frame_rot_deg))


def build_hb_waypoints(
    robot_start: np.ndarray,
    robot_target: np.ndarray,
    *,
    cardinal_hb: bool,
    direct_motion: bool = True,
    straight_line: bool = False,
) -> list[np.ndarray]:
    """Build hb1 desired_pose waypoints.

    ``straight_line=True``: one goal ``[x, y, start_yaw]`` — planar straight segment,
    no heading change (default for absolute position-only goals).

    ``direct_motion=True``: one holonomic goal ``[x, y, yaw]`` (XY + heading together).

    With ``direct_motion=False``: translate at start yaw, then rotate in place.
    With ``cardinal_hb`` (and not direct): L-path in hb odom (X corner, then Y).
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

    trans_xy = np.array([final[0], final[1], hold_yaw], dtype=np.float64)
    waypoints: list[np.ndarray] = []
    if cardinal_hb and abs(dx) > 1e-4 and abs(dy) > 1e-4:
        corner = np.array([final[0], robot_start[1], hold_yaw], dtype=np.float64)
        waypoints.append(corner)
    if rotate_yaw and not translate_xy:
        return [final]
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


def _xy2(arr: np.ndarray) -> np.ndarray:
    """First two elements of a position (accepts xy or xyz)."""
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size < 2:
        raise ValueError(f"need at least 2 position components, got {flat.size}")
    return flat[:2]


def mocap_xy_to_hb(mocap_xy: np.ndarray, calib: MocapToHbCalib) -> np.ndarray:
    return calib.rot @ _xy2(mocap_xy) + calib.trans


def mocap_pose_to_hb_se2(
    mocap_xyz: np.ndarray,
    calib: MocapToHbCalib,
    hb_yaw_hold: float,
) -> np.ndarray:
    """Map Opti position into hb1 XY; keep hb yaw fixed (not commanded from Opti)."""
    hb_xy = mocap_xy_to_hb(mocap_xyz, calib)
    return np.array([hb_xy[0], hb_xy[1], hb_yaw_hold], dtype=np.float64)


def rotate_vector_in_robot_frame(
    v_xy: np.ndarray,
    robot_yaw: float,
    rot_deg: float,
) -> np.ndarray:
    """Rotate a planar vector by ``rot_deg`` about the robot/hb z axis at ``robot_yaw``."""
    if abs(rot_deg) < 1e-9:
        return _xy2(v_xy)
    r_h = rot2d_from_yaw(float(robot_yaw))
    r_d = rot2d_from_yaw(math.radians(rot_deg))
    return r_h @ r_d @ r_h.T @ _xy2(v_xy)


def plan_translation(
    mocap_xyz: np.ndarray,
    calib: MocapToHbCalib,
    goal_delta_input_xy: np.ndarray,
    goal_frame: GoalFrame,
    robot_yaw: float,
    robot_input_rot_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (desired_mocap_xyz, mocap_goal_delta, hb_delta, mocap_delta_rotated)."""
    delta_in = _xy2(goal_delta_input_xy)
    if goal_frame == GoalFrame.OPTI_BODY:
        mocap_goal_delta = calib.mocap_rot @ delta_in
    else:
        mocap_goal_delta = delta_in
    mocap_rotated = rotate_vector_in_robot_frame(
        mocap_goal_delta, robot_yaw, robot_input_rot_deg
    )
    hb_delta = calib.rot @ mocap_rotated

    desired_xyz = mocap_xyz.copy()
    desired_xyz[:2] = _xy2(mocap_xyz) + mocap_goal_delta
    return desired_xyz, mocap_goal_delta, hb_delta, mocap_rotated


def plan_absolute_mocap_goal(
    mocap_xyz: np.ndarray,
    desired_mocap_xyz: np.ndarray,
    calib: MocapToHbCalib,
    robot_yaw: float,
    robot_input_rot_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (desired_mocap_xyz, mocap_goal_delta, hb_delta, mocap_delta_rotated)."""
    desired_xyz = np.asarray(desired_mocap_xyz, dtype=np.float64).reshape(3)
    mocap_goal_delta = _xy2(desired_xyz) - _xy2(mocap_xyz)
    mocap_rotated = rotate_vector_in_robot_frame(
        mocap_goal_delta, robot_yaw, robot_input_rot_deg
    )
    hb_delta = calib.rot @ mocap_rotated
    return desired_xyz, mocap_goal_delta, hb_delta, mocap_rotated


def setup_run_plan(
    client: redis.Redis,
    *,
    goal_delta_input_xy: np.ndarray,
    goal_frame: GoalFrame,
    translation_only_calib: bool,
    calib_yaw_deg: float | None,
    cardinal_hb: bool,
    direct_motion: bool,
    robot_input_rot_deg: float,
    face_lab_yaw: str | None,
    robot_pose_key: str,
    mocap_pos_key: str,
    mocap_ori_key: str,
    curr_minus_desired: bool,
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
    )

    absolute_target = absolute_mocap_target is not None
    target_yaw_rad: float | None = None
    if absolute_target:
        desired_xyz_in, target_yaw_rad = absolute_mocap_target
        desired_xyz, mocap_goal_delta, hb_delta, mocap_rotated = plan_absolute_mocap_goal(
            mocap_xyz,
            desired_xyz_in,
            calib,
            float(robot_start[2]),
            robot_input_rot_deg,
        )
        goal_delta = mocap_goal_delta
        goal_frame = GoalFrame.OPTI_WORLD
        face_lab_yaw = None
        if target_yaw_rad is None:
            desired_mocap_yaw = mocap_yaw
            hb_target_yaw = float(robot_start[2])
            require_final_yaw = False
        else:
            desired_mocap_yaw = target_yaw_rad
            hb_target_yaw = mocap_lab_yaw_to_hb_yaw(
                desired_mocap_yaw,
                mocap_yaw,
                float(robot_start[2]),
                robot_input_rot_deg,
            )
            require_final_yaw = True
    else:
        goal_delta = _xy2(goal_delta_input_xy)
        desired_xyz, mocap_goal_delta, hb_delta, mocap_rotated = plan_translation(
            mocap_xyz,
            calib,
            goal_delta,
            goal_frame,
            float(robot_start[2]),
            robot_input_rot_deg,
        )
        if face_lab_yaw is not None:
            desired_mocap_yaw = LAB_AXIS_YAW_RAD[face_lab_yaw]
            require_final_yaw = True
        else:
            desired_mocap_yaw = mocap_yaw
            require_final_yaw = False
        hb_target_yaw = mocap_lab_yaw_to_hb_yaw(
            desired_mocap_yaw,
            mocap_yaw,
            float(robot_start[2]),
            robot_input_rot_deg,
        )
    robot_target = np.array(
        [
            robot_start[0] + hb_delta[0],
            robot_start[1] + hb_delta[1],
            hb_target_yaw,
        ],
        dtype=np.float64,
    )
    straight_line = absolute_target and target_yaw_rad is None
    hb_waypoints = build_hb_waypoints(
        robot_start,
        robot_target,
        cardinal_hb=cardinal_hb,
        direct_motion=direct_motion and not straight_line,
        straight_line=straight_line,
    )

    # curr_minus_desired only affects mocap error logging, not hb target mapping
    _ = curr_minus_desired

    return RunPlan(
        desired_mocap_xyz=desired_xyz,
        desired_mocap_yaw=desired_mocap_yaw,
        mocap_start_xyz=mocap_xyz,
        mocap_start_yaw=mocap_yaw,
        goal_delta_input_xy=goal_delta,
        goal_frame=goal_frame,
        mocap_delta_world_xy=mocap_goal_delta,
        mocap_delta_rotated_xy=mocap_rotated,
        hb_delta_xy=hb_delta,
        robot_start=robot_start,
        robot_target=robot_target,
        face_lab_yaw=face_lab_yaw,
        hb_target_yaw=hb_target_yaw,
        hb_waypoints=hb_waypoints,
        calib=calib,
        robot_input_rot_deg=robot_input_rot_deg,
        mocap_delta_before_robot_rot_xy=mocap_goal_delta,
        absolute_target=absolute_target,
        require_final_yaw=require_final_yaw,
    )


def hb1_tracking_error(robot_current: np.ndarray, robot_target: np.ndarray) -> np.ndarray:
    """Error in hb1 odom: current minus target."""
    return np.array(
        [
            robot_current[0] - robot_target[0],
            robot_current[1] - robot_target[1],
            wrap_angle(robot_current[2] - robot_target[2]),
        ],
        dtype=np.float64,
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


def write_desired_pose(client: redis.Redis, goal_se2: np.ndarray) -> None:
    client.set(KEYS.desired_pose, numpy_array_to_string(goal_se2.reshape(3)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OptiTrack → TidyBot base controller")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--goal-along",
        choices=sorted(GOAL_ALONG_AXES.keys()),
        default=DEFAULT_GOAL_ALONG,
        help="Move along this axis (default: lab-plus-y = Motive world +Y)",
    )
    p.add_argument(
        "--goal-distance-ft",
        type=float,
        default=DEFAULT_GOAL_DISTANCE_FT,
        help="Distance along --goal-along (ft, positive; default: 1.5)",
    )
    p.add_argument(
        "--goal-distance-m",
        type=float,
        default=None,
        help="Distance along --goal-along in meters (overrides --goal-distance-ft)",
    )
    p.add_argument(
        "--goal-offset-ft",
        type=float,
        default=DEFAULT_GOAL_OFFSET_FT,
        help="Legacy: use with --use-legacy-goal-offsets instead of --goal-along",
    )
    p.add_argument(
        "--goal-offset-opti-x-ft",
        type=float,
        default=None,
        help="Translation along Opti world/body X at startup (feet); default from --goal-offset-ft",
    )
    p.add_argument(
        "--goal-offset-opti-y-ft",
        type=float,
        default=0.0,
        help="Translation along Opti world/body Y at startup (feet)",
    )
    p.add_argument(
        "--goal-offset-x-m",
        type=float,
        default=None,
        help="Opti X offset in meters (overrides --goal-offset-opti-x-ft / --goal-offset-ft)",
    )
    p.add_argument(
        "--goal-offset-opti-y-m",
        type=float,
        default=None,
        help="Opti Y offset in meters (overrides --goal-offset-opti-y-ft)",
    )
    p.add_argument(
        "--use-legacy-goal-offsets",
        action="store_true",
        help="Use --goal-offset-opti-*-ft/m instead of --goal-along",
    )
    p.add_argument(
        "--goal-body-frame",
        action="store_true",
        help="Legacy: goal offsets in marker rigid-body XY (with --use-legacy-goal-offsets)",
    )
    p.add_argument(
        "--goal-opti-world",
        action="store_true",
        help="Legacy: goal offsets in Motive lab/world XY (with --use-legacy-goal-offsets)",
    )
    p.add_argument(
        "--curr-minus-desired",
        action="store_true",
        help="Use (current - desired) in mocap instead of (desired - current)",
    )
    p.add_argument(
        "--monitor",
        action="store_true",
        help="Print state only; do not write hb1::desired_pose",
    )
    p.add_argument(
        "--stop-on-exit",
        action="store_true",
        help="Set hb1::stop on exit",
    )
    p.add_argument(
        "--tolerance-in",
        type=float,
        default=DEFAULT_TOLERANCE_IN,
        help="Done when hb1::current_pose XY is within this many inches of hb1 target",
    )
    p.add_argument(
        "--tolerance-m",
        type=float,
        default=None,
        help="Same as --tolerance-in but in meters (overrides inches)",
    )
    p.add_argument(
        "--robot-pose-key",
        default=KEYS.robot_pose,
        help="Redis key for robot odom [x, y, yaw] (default: hb1::current_pose)",
    )
    p.add_argument(
        "--mocap-pos-key",
        default=KEYS.mocap_pos,
        help="Redis key for mocap position xyz (default: tidybot01::pos)",
    )
    p.add_argument(
        "--mocap-ori-key",
        default=KEYS.mocap_ori,
        help="Redis key for mocap orientation xyzw (default: tidybot01::ori)",
    )
    p.add_argument(
        "--tracking-valid-key",
        default=KEYS.tracking_valid,
        help="Redis key; must be true to plan/move (default: tidybot01::tracking_valid)",
    )
    p.add_argument(
        "--odom-jump-m",
        type=float,
        default=0.5,
        help="Warn if robot XY moves more than this between cycles (possible odom reset)",
    )
    p.add_argument(
        "--log-hz",
        type=float,
        default=10.0,
        help="How often to print current opti + hb1 poses (default: 10 Hz)",
    )
    p.add_argument(
        "--calib-translation-only",
        action="store_true",
        help="No rotation between Motive lab XY and hb odom (R=I); only if axes match",
    )
    p.add_argument(
        "--calib-yaw-deg",
        type=float,
        default=None,
        help="Override mocap→hb rotation (deg); Motive +X → hb +X at this angle",
    )
    p.add_argument(
        "--cardinal-hb",
        action="store_true",
        help="With --translate-then-rotate: L-path in hb odom (X then Y)",
    )
    p.add_argument(
        "--translate-then-rotate",
        action="store_true",
        help=(
            "Translate at startup yaw, then rotate in place (old behavior). "
            "Default is direct holonomic motion to final [x, y, yaw]."
        ),
    )
    p.add_argument(
        "--robot-input-rot-deg",
        type=float,
        default=ROBOT_FRAME_ROT_DEG,
        help=(
            "Rotation (deg) applied in hb/robot frame to translation and yaw commands "
            f"(default: {ROBOT_FRAME_ROT_DEG}, always on)"
        ),
    )
    p.add_argument(
        "--face-lab-yaw",
        choices=["none", *sorted(LAB_AXIS_YAW_RAD.keys())],
        default=DEFAULT_FACE_LAB_YAW,
        help=(
            "After translation, rotate in place to face this Motive lab axis "
            f"(default: {DEFAULT_FACE_LAB_YAW}; marker +X along that direction)"
        ),
    )
    p.add_argument(
        "--relative-goal",
        action="store_true",
        help=(
            "Use --goal-along / distance instead of the default absolute Motive target "
            f"({DEFAULT_TARGET_X}, {DEFAULT_TARGET_Y}, {DEFAULT_TARGET_Z}) m"
        ),
    )
    p.add_argument(
        "--target-x",
        type=float,
        default=DEFAULT_TARGET_X,
        help=f"Absolute Motive lab X (m); default: {DEFAULT_TARGET_X}",
    )
    p.add_argument(
        "--target-y",
        type=float,
        default=DEFAULT_TARGET_Y,
        help=f"Absolute Motive lab Y (m); default: {DEFAULT_TARGET_Y}",
    )
    p.add_argument(
        "--target-z",
        type=float,
        default=DEFAULT_TARGET_Z,
        help=f"Absolute Motive lab Z (m; logged; hb uses XY only); default: {DEFAULT_TARGET_Z}",
    )
    p.add_argument(
        "--target-yaw-deg",
        type=float,
        default=None,
        help=(
            "Absolute Motive lab heading for marker +X (deg). "
            "Omitted by default: hold startup heading (straight-line XY only)."
        ),
    )
    p.add_argument(
        "--target-opti-pose",
        type=str,
        default=None,
        metavar="X,Y,Z[,YAW_DEG]",
        help=(
            'Absolute pose "x,y,z" or "x,y,z,yaw_deg" in Motive lab (overrides --target-*)'
        ),
    )
    p.add_argument(
        "--rotate-only",
        action="store_true",
        help="No translation (distance 0); only rotate to --face-lab-yaw in place",
    )
    p.add_argument(
        "--no-face-turn",
        action="store_true",
        help="Skip rotation; keep hb yaw from startup",
    )
    p.add_argument(
        "--yaw-tolerance-deg",
        type=float,
        default=DEFAULT_YAW_TOLERANCE_DEG,
        help="Yaw tolerance for final facing waypoint (default: 5 deg)",
    )
    return p.parse_args(argv)


def _resolve_face_lab_yaw(args: argparse.Namespace) -> str | None:
    if args.no_face_turn or args.face_lab_yaw == "none":
        return None
    return str(args.face_lab_yaw)


def _sleep_until(loop_start: float, period: float) -> None:
    elapsed = time.perf_counter() - loop_start
    if elapsed < period:
        time.sleep(period - elapsed)


def _fmt_array(v: np.ndarray) -> str:
    return np.array2string(np.asarray(v), precision=4, suppress_small=True)


def _fmt_opti_pose(xyz: np.ndarray, yaw_rad: float) -> str:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    return (
        f"x={xyz[0]:.4f} y={xyz[1]:.4f} z={xyz[2]:.4f} "
        f"yaw_deg={math.degrees(float(yaw_rad)):.2f}"
    )


def _fmt_opti_error(err_xyz_yaw: np.ndarray) -> str:
    e = np.asarray(err_xyz_yaw, dtype=np.float64).reshape(4)
    return (
        f"dx={e[0]:.4f} dy={e[1]:.4f} dz={e[2]:.4f} "
        f"dyaw_deg={math.degrees(e[3]):.2f}"
    )


def print_pose_log_block(
    *,
    plan: RunPlan,
    robot_current: np.ndarray,
    mocap_xyz: np.ndarray | None,
    mocap_quat: np.ndarray | None,
    curr_minus_desired: bool,
    hb_goal: np.ndarray | None = None,
) -> None:
    """Print hb then opti poses (start, current, goal, error — one field per line)."""
    goal = hb_goal if hb_goal is not None else plan.robot_target
    hb_err = hb1_tracking_error(robot_current, goal)

    print(f"hb_start={_fmt_array(plan.robot_start)}")
    print(f"hb_current={_fmt_array(robot_current)}")
    print(f"hb_goal={_fmt_array(goal)}")
    print(f"hb_final={_fmt_array(plan.robot_target)}")
    print(f"hb_error={_fmt_array(hb_err)}")

    print(f"opti_start {_fmt_opti_pose(plan.mocap_start_xyz, plan.mocap_start_yaw)}")
    if mocap_xyz is not None and mocap_quat is not None:
        mocap_yaw = quat_xyzw_to_yaw(mocap_quat)
        opti_err = mocap_pose_error(
            mocap_xyz,
            mocap_yaw,
            plan.desired_mocap_xyz,
            plan.desired_mocap_yaw,
            curr_minus_desired=curr_minus_desired,
        )
        print(f"opti_current {_fmt_opti_pose(mocap_xyz, mocap_yaw)}")
        print(
            f"opti_target {_fmt_opti_pose(plan.desired_mocap_xyz, plan.desired_mocap_yaw)}"
        )
        print(f"opti_error {_fmt_opti_error(opti_err)}")
    else:
        print("opti_current (unavailable)")
        print(
            f"opti_target {_fmt_opti_pose(plan.desired_mocap_xyz, plan.desired_mocap_yaw)}"
        )
        print("opti_error (unavailable)")


def _resolve_goal_delta_opti_m(args: argparse.Namespace) -> np.ndarray:
    if args.goal_offset_x_m is not None:
        dx = float(args.goal_offset_x_m)
    elif args.goal_offset_opti_x_ft is not None:
        dx = float(args.goal_offset_opti_x_ft) * FEET_TO_METERS
    else:
        dx = float(args.goal_offset_ft) * FEET_TO_METERS
    if args.goal_offset_opti_y_m is not None:
        dy = float(args.goal_offset_opti_y_m)
    else:
        dy = float(args.goal_offset_opti_y_ft) * FEET_TO_METERS
    return np.array([dx, dy], dtype=np.float64)


def _resolve_goal_frame(args: argparse.Namespace) -> GoalFrame:
    if args.goal_body_frame:
        return GoalFrame.OPTI_BODY
    return GoalFrame.OPTI_WORLD


def _resolve_absolute_mocap_target(
    args: argparse.Namespace,
) -> tuple[np.ndarray, float | None] | None:
    if args.relative_goal or args.rotate_only:
        return None
    if args.target_opti_pose is not None:
        parts = [p.strip() for p in str(args.target_opti_pose).split(",")]
        if len(parts) not in (3, 4):
            raise ValueError(
                '--target-opti-pose must be "x,y,z" or "x,y,z,yaw_deg"'
            )
        xyz = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
        if len(parts) == 3:
            return xyz, None
        return xyz, math.radians(float(parts[3]))
    xyz = np.array(
        [float(args.target_x), float(args.target_y), float(args.target_z)],
        dtype=np.float64,
    )
    if args.target_yaw_deg is None:
        return xyz, None
    return xyz, math.radians(float(args.target_yaw_deg))


def _resolve_goal(args: argparse.Namespace) -> tuple[np.ndarray, GoalFrame]:
    if args.rotate_only:
        frame, _axis = GOAL_ALONG_AXES[args.goal_along]
        return np.zeros(2, dtype=np.float64), frame
    if args.use_legacy_goal_offsets:
        return _resolve_goal_delta_opti_m(args), _resolve_goal_frame(args)
    if args.goal_distance_m is not None:
        distance_m = float(args.goal_distance_m)
    else:
        distance_m = abs(float(args.goal_distance_ft)) * FEET_TO_METERS
    frame, axis = GOAL_ALONG_AXES[args.goal_along]
    delta = np.array(axis, dtype=np.float64) * distance_m
    return delta, frame


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        absolute_mocap_target = _resolve_absolute_mocap_target(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    goal_delta_input_xy, goal_frame = _resolve_goal(args)
    if args.tolerance_m is not None:
        tolerance_m = float(args.tolerance_m)
    else:
        tolerance_m = float(args.tolerance_in) * INCHES_TO_METERS
    tolerance_yaw_rad = math.radians(float(args.yaw_tolerance_deg))

    if absolute_mocap_target is not None and args.rotate_only:
        print("Error: --rotate-only cannot be used with an absolute target pose", file=sys.stderr)
        return 1
    if args.rotate_only and _resolve_face_lab_yaw(args) is None:
        print(
            "Error: --rotate-only needs a facing direction (default --face-lab-yaw minus-y; "
            "do not use --no-face-turn)",
            file=sys.stderr,
        )
        return 1

    try:
        client = connect_redis(args.redis_host, args.redis_port)
    except redis.RedisError as exc:
        print(f"Redis connect failed: {exc}", file=sys.stderr)
        return 1

    wait_for_tracking_valid(client, args.tracking_valid_key)

    try:
        plan = setup_run_plan(
            client,
            goal_delta_input_xy=goal_delta_input_xy,
            goal_frame=goal_frame,
            translation_only_calib=args.calib_translation_only,
            calib_yaw_deg=args.calib_yaw_deg,
            cardinal_hb=args.cardinal_hb,
            direct_motion=not args.translate_then_rotate,
            robot_input_rot_deg=float(args.robot_input_rot_deg),
            face_lab_yaw=_resolve_face_lab_yaw(args),
            robot_pose_key=args.robot_pose_key,
            mocap_pos_key=args.mocap_pos_key,
            mocap_ori_key=args.mocap_ori_key,
            curr_minus_desired=args.curr_minus_desired,
            absolute_mocap_target=absolute_mocap_target,
        )
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    hb_at_mocap_start = mocap_pose_to_hb_se2(
        plan.mocap_start_xyz,
        plan.calib,
        plan.robot_start[2],
    )
    calib_err = float(np.linalg.norm(hb_at_mocap_start - plan.robot_start))
    dyaw_calib = rot2d_yaw(plan.calib.rot)

    if plan.absolute_target:
        mode = "absolute Opti pose"
    elif args.rotate_only:
        mode = "rotate only"
    else:
        mode = "translate + face"
    print(f"Plan (Opti ground truth → hb goal, {mode}):")
    print(f"  hb1 start        = {plan.robot_start.tolist()}  ({args.robot_pose_key})")
    print(
        f"  mocap @ startup  = {plan.mocap_start_xyz.tolist()}  "
        f"Opti heading={math.degrees(plan.mocap_start_yaw):.1f} deg (marker +X in lab)"
    )
    print(
        f"  lab→hb R angle   = {math.degrees(dyaw_calib):.1f} deg  "
        f"t={plan.calib.trans.round(4).tolist()}  "
        f"({'translation only' if args.calib_translation_only else 'from Opti quat'})"
    )
    print(f"  hb(mocap_start)  = {hb_at_mocap_start.round(4).tolist()}  residual={calib_err:.4f} m")
    if plan.absolute_target:
        yaw_note = (
            f"yaw={math.degrees(plan.desired_mocap_yaw):.1f} deg (Motive lab)"
            if plan.require_final_yaw
            else "hold startup yaw"
        )
        along_note = f"  absolute target  = {plan.desired_mocap_xyz.tolist()}  {yaw_note}"
    elif not args.use_legacy_goal_offsets:
        along_note = f"  along={args.goal_along}  distance={args.goal_distance_ft:.2f} ft"
    else:
        along_note = "  (legacy goal offsets)"
    print(f"  goal frame       = {plan.goal_frame.value}{along_note}")
    print(
        f"  goal Δ (input)   = {plan.goal_delta_input_xy.round(4).tolist()} m  "
        f"frame={plan.goal_frame.value}"
    )
    print(
        f"  mocap Δ (goal)   = {plan.mocap_delta_world_xy.round(4).tolist()} m  "
        f"(Opti target; lab ±X/±Y only)"
    )
    print(
        f"  mocap Δ (hb cmd) = {plan.mocap_delta_rotated_xy.round(4).tolist()} m  "
        f"(after robot-frame rot {plan.robot_input_rot_deg:.1f} deg)"
    )
    print(f"  robot-frame rot  = {plan.robot_input_rot_deg:.1f} deg on translation + yaw")
    print(f"  hb Δ (odom)      = {plan.hb_delta_xy.round(4).tolist()} m")
    dx_g, dy_g = plan.mocap_delta_world_xy[0], plan.mocap_delta_world_xy[1]
    if abs(dx_g) > 1e-6:
        sx = "decrease" if dx_g < 0 else "increase"
        print(f"  expect Motive X to {sx} by {abs(dx_g):.3f} m")
    if abs(dy_g) > 1e-6:
        sy = "decrease" if dy_g < 0 else "increase"
        print(f"  expect Motive Y to {sy} by {abs(dy_g):.3f} m")
    print(f"  mocap goal xy    = {plan.desired_mocap_xyz[:2].round(4).tolist()}")
    print(f"  desired_mocap    = {plan.desired_mocap_xyz.tolist()}")
    if len(plan.hb_waypoints) > 1:
        path_kind = "cardinal L-path" if args.cardinal_hb else "translate then rotate"
        print(f"  hb waypoints     = {len(plan.hb_waypoints)} ({path_kind})")
        for i, wp in enumerate(plan.hb_waypoints):
            print(f"    [{i}] {wp.round(4).tolist()}")
    elif len(plan.hb_waypoints) == 1:
        if plan.absolute_target and not plan.require_final_yaw:
            print("  hb motion        = straight line to goal XY (hold startup yaw)")
        else:
            print("  hb motion        = direct holonomic to final [x, y, yaw]")
    if plan.face_lab_yaw is not None:
        print(
            f"  face Motive      = {plan.face_lab_yaw}  "
            f"(Opti heading {math.degrees(plan.desired_mocap_yaw):.1f} deg, "
            f"hb yaw {math.degrees(plan.hb_target_yaw):.1f} deg)"
        )
    elif plan.absolute_target and plan.require_final_yaw:
        print(
            f"  target heading   = Opti {math.degrees(plan.desired_mocap_yaw):.1f} deg  "
            f"hb yaw {math.degrees(plan.hb_target_yaw):.1f} deg"
        )
    elif plan.absolute_target:
        print(
            f"  orientation      = hold startup (hb yaw {math.degrees(plan.hb_target_yaw):.1f} deg)"
        )
    print(f"  hb1 target       = {plan.robot_target.tolist()}  ({KEYS.desired_pose})")
    print(f"  hb waypoints     = {len(plan.hb_waypoints)} step(s)")
    print(
        f"  mocap keys       = {args.mocap_pos_key} / {args.mocap_ori_key} / "
        f"{args.tracking_valid_key}"
    )
    print(
        f"  success          = |hb1_current_xy - hb1_target_xy| < "
        f"{tolerance_m:.4f} m ({args.tolerance_in:.1f} in)"
    )
    if args.monitor:
        print("Monitor mode — not commanding base.")
    else:
        print(f"Commanding {KEYS.desired_pose} at {CONTROL_HZ:.0f} Hz")

    period = 1.0 / CONTROL_HZ
    log_period = 1.0 / max(args.log_hz, 0.1)
    last_log = 0.0
    robot_prev: np.ndarray | None = None
    reached_target = False
    waypoint_idx = 0
    mocap_xyz: np.ndarray | None = None
    mocap_quat: np.ndarray | None = None

    def active_hb_goal() -> np.ndarray:
        return plan.hb_waypoints[min(waypoint_idx, len(plan.hb_waypoints) - 1)]

    if not args.monitor:
        write_desired_pose(client, active_hb_goal())

    try:
        while True:
            t0 = time.perf_counter()
            if not read_tracking_valid(client, args.tracking_valid_key):
                if not args.monitor:
                    stop_base(client)
                print(
                    f"Tracking lost ({args.tracking_valid_key} is not true) — "
                    f"stopping base and exiting."
                )
                return 1

            robot_current = read_robot_se2(client, args.robot_pose_key)
            hb_goal = active_hb_goal()
            if waypoint_idx < len(plan.hb_waypoints) - 1:
                if waypoint_reached(
                    robot_current,
                    hb_goal,
                    waypoint_idx,
                    plan.hb_waypoints,
                    tolerance_m=tolerance_m,
                    tolerance_yaw_rad=tolerance_yaw_rad,
                ):
                    waypoint_idx += 1
                    hb_goal = active_hb_goal()
                    print(f"Waypoint {waypoint_idx}: {hb_goal.round(4).tolist()}")
            try:
                mocap_xyz, mocap_quat = read_mocap_pose(
                    client, args.mocap_pos_key, args.mocap_ori_key
                )
            except RuntimeError:
                mocap_xyz, mocap_quat = None, None

            if robot_prev is not None:
                jump = float(np.linalg.norm(robot_current[:2] - robot_prev[:2]))
                if jump > args.odom_jump_m:
                    print(
                        f"Warning: hb1 odom jump {jump:.3f} m between cycles"
                    )
            robot_prev = robot_current.copy()

            if not args.monitor:
                write_desired_pose(client, hb_goal)

            now = time.perf_counter()
            if now - last_log >= log_period:
                track_xy_norm = float(
                    np.linalg.norm(robot_current[:2] - plan.robot_target[:2])
                )
                track_yaw_err = abs(
                    wrap_angle(robot_current[2] - plan.robot_target[2])
                )
                print_pose_log_block(
                    plan=plan,
                    robot_current=robot_current,
                    mocap_xyz=mocap_xyz,
                    mocap_quat=mocap_quat,
                    curr_minus_desired=args.curr_minus_desired,
                    hb_goal=hb_goal,
                )
                print()
                last_log = now
                pose_ok = track_xy_norm < tolerance_m and (
                    not plan.require_final_yaw
                    or track_yaw_err < tolerance_yaw_rad
                )
                if pose_ok:
                    if not reached_target:
                        msg = (
                            f"Success: hb1 within {tolerance_m:.4f} m "
                            f"({args.tolerance_in:.1f} in) of target XY"
                        )
                        if plan.require_final_yaw:
                            if plan.face_lab_yaw is not None:
                                msg += (
                                    f" and yaw within {math.degrees(tolerance_yaw_rad):.1f} deg "
                                    f"(facing Motive {plan.face_lab_yaw})"
                                )
                            else:
                                msg += (
                                    f" and yaw within {math.degrees(tolerance_yaw_rad):.1f} deg "
                                    f"(Opti heading {math.degrees(plan.desired_mocap_yaw):.1f} deg)"
                                )
                        print(msg + " — holding.")
                        reached_target = True
            _sleep_until(t0, period)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if args.stop_on_exit:
            client.set(KEYS.stop, "stop")
            print(f"Set {KEYS.stop!r} = 'stop'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
