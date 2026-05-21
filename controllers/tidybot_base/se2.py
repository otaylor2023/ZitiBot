"""Planar SE(2) geometry helpers."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_xyzw_to_rot2d(q: np.ndarray) -> np.ndarray:
    """Planar yaw rotation from quaternion (xyzw)."""
    return rot2d_from_yaw(quat_xyzw_to_yaw(q))


def quat_xyzw_to_yaw(q: np.ndarray) -> float:
    """Heading from body +X projected onto the horizontal plane (rad)."""
    x, y, z, w = (float(q[i]) for i in range(4))
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def rot2d_from_yaw(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rot2d_yaw(rot: np.ndarray) -> float:
    return math.atan2(float(rot[1, 0]), float(rot[0, 0]))


def xy2(arr: np.ndarray) -> np.ndarray:
    """First two elements of a position (accepts xy or xyz)."""
    flat = np.asarray(arr, dtype=np.float64).ravel()
    if flat.size < 2:
        raise ValueError(f"need at least 2 position components, got {flat.size}")
    return flat[:2]


def rotate_vector_in_robot_frame(
    v_xy: np.ndarray,
    robot_yaw: float,
    rot_deg: float,
) -> np.ndarray:
    """Rotate a planar vector by ``rot_deg`` about the robot/hb z axis at ``robot_yaw``."""
    if abs(rot_deg) < 1e-9:
        return xy2(v_xy)
    r_h = rot2d_from_yaw(float(robot_yaw))
    r_d = rot2d_from_yaw(math.radians(rot_deg))
    return r_h @ r_d @ r_h.T @ xy2(v_xy)


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
