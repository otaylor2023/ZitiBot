"""Backward-compatible helpers formerly in grasp_and_pour_controller.py."""

from __future__ import annotations

import math
import select
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from zitibot_core import arm, gripper
from zitibot_core.constants import (
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_GRIPPER_FORCE,
    DEFAULT_GRIPPER_GRASP_SETTLE_S,
    DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    DEFAULT_GRIPPER_PREGRASP_WIDTH,
    DEFAULT_GRIPPER_SPEED,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    EE_ORI_TOOL_DOWN_45,
    OBJECT_DEFAULTS,
    Object,
)
from zitibot_core.redis_keys import (
    GRIPPER_MODE_GRASP,
    GRIPPER_MODE_MOVE,
    GRIPPER_MODE_OPEN_MAX,
)

POUR_TICK_DT_S = 0.05
_STDIN_EOF = object()

_spec = OBJECT_DEFAULTS[Object.PASTA_BOWL]
PICK_POSITION = _spec.pick_pose.copy()
POUR_POSITION = _spec.pour_pose.copy() if _spec.pour_pose is not None else PICK_POSITION.copy()
GRASP_POSITION = PICK_POSITION
GRASP_ORIENTATION = EE_ORI_TOOL_DOWN_45.copy()

_try_redis = arm.try_redis
validate_config = arm.validate_config
_publish_cartesian = arm.publish_goal_cartesian
read_current_ee_world = arm.read_current_ee_world
pour_orientation_end = arm.pour_orientation_end
resolve_gripper_open_width = gripper.resolve_open_width
read_gripper_current_width = gripper.read_current_width
set_gripper_width = gripper.set_width


@dataclass
class MotionParams:
    approach_dz_m: float
    pour_tilt_deg: float = DEFAULT_POUR_TILT_DEG
    pour_axis: str = DEFAULT_POUR_AXIS
    tilt_duration_s: float = DEFAULT_TILT_DURATION_S
    gripper_open_width: float | None = None
    gripper_pregrasp_width: float = DEFAULT_GRIPPER_PREGRASP_WIDTH
    gripper_close_width: float = 0.0
    gripper_speed: float = DEFAULT_GRIPPER_SPEED
    gripper_force: float = DEFAULT_GRIPPER_FORCE
    gripper_pregrasp_settle_s: float = DEFAULT_GRIPPER_PREGRASP_SETTLE_S
    gripper_grasp_settle_s: float = DEFAULT_GRIPPER_GRASP_SETTLE_S


@dataclass
class OrientationSlerpState:
    hold_world: np.ndarray
    R_start: np.ndarray
    R_end: np.ndarray
    t0: float
    last_tick: float = 0.0


def _above_pick(pick_pos: np.ndarray, motion: MotionParams) -> np.ndarray:
    return pick_pos + np.array([0.0, 0.0, motion.approach_dz_m], dtype=np.float64)


def _do_move_above_grasp(
    redis_client,
    grasp_pos: np.ndarray,
    grasp_ori: np.ndarray,
    motion: MotionParams,
) -> np.ndarray:
    above = _above_pick(grasp_pos, motion)
    _publish_cartesian(redis_client, above, grasp_ori)
    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
    set_gripper_width(
        redis_client,
        open_w,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_MOVE,
    )
    print(f"[0] Move above grasp: pos={above.tolist()}, gripper open={open_w:.4f} m")
    return above


def _do_descend_to_grasp(
    redis_client,
    grasp_pos: np.ndarray,
    grasp_ori: np.ndarray,
    *,
    label: str = "[1] Descend to grasp",
) -> None:
    _publish_cartesian(redis_client, grasp_pos, grasp_ori)
    print(f"{label}: pos={grasp_pos.tolist()}")


def _do_grasp_object(redis_client, motion: MotionParams) -> None:
    pre_w = float(motion.gripper_pregrasp_width)
    set_gripper_width(
        redis_client,
        pre_w,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_MOVE,
    )
    print(f"[2a] Pregrasp: width={pre_w:.4f} m, settle {motion.gripper_pregrasp_settle_s:.1f} s")
    time.sleep(motion.gripper_pregrasp_settle_s)
    set_gripper_width(
        redis_client,
        motion.gripper_close_width,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_GRASP,
    )
    print(
        f"[2b] Grasp: force={motion.gripper_force:.1f} N, "
        f"settle {motion.gripper_grasp_settle_s:.1f} s"
    )
    time.sleep(motion.gripper_grasp_settle_s)


def _do_open_gripper(redis_client, motion: MotionParams) -> None:
    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
    set_gripper_width(
        redis_client,
        open_w,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_OPEN_MAX,
    )
    print(f"[open] gripper width={open_w:.4f} m max")


def _start_orientation_slerp(
    redis_client,
    hold_world: np.ndarray,
    R_start: np.ndarray,
    R_end: np.ndarray,
    now: float,
    *,
    label: str,
) -> OrientationSlerpState:
    hold = np.asarray(hold_world, dtype=np.float64).reshape(3).copy()
    R0 = np.asarray(R_start, dtype=np.float64).reshape(3, 3).copy()
    R1 = np.asarray(R_end, dtype=np.float64).reshape(3, 3).copy()
    _publish_cartesian(redis_client, hold, R0)
    print(label)
    return OrientationSlerpState(hold_world=hold, R_start=R0, R_end=R1, t0=now)


def _tick_orientation_slerp(
    redis_client,
    slerp: OrientationSlerpState,
    motion: MotionParams,
    now: float,
) -> bool:
    if now - slerp.last_tick < POUR_TICK_DT_S:
        return False
    slerp.last_tick = now
    alpha = min(1.0, max(0.0, (now - slerp.t0) / motion.tilt_duration_s))
    key_rots = R.concatenate([R.from_matrix(slerp.R_start), R.from_matrix(slerp.R_end)])
    R_interp = Slerp([0.0, 1.0], key_rots)([alpha]).as_matrix()[0]
    _publish_cartesian(redis_client, slerp.hold_world, R_interp)
    return alpha >= 1.0


def _stdin_line_ready(timeout_s: float):
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    except (ValueError, OSError):
        return _STDIN_EOF
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":
        return _STDIN_EOF
    return line
