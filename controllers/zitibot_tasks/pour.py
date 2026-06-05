"""Pour subtasks (orientation slerp at fixed position)."""

from __future__ import annotations

import time

import numpy as np

from zitibot_core import arm
from zitibot_core.constants import (
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    TICK_DT_S,
)
from zitibot_core.context import TaskContext
from zitibot_core.runner import step_gate


def into(
    ctx: TaskContext,
    target_pos: np.ndarray,
    *,
    tilt_deg: float = DEFAULT_POUR_TILT_DEG,
    axis: str = DEFAULT_POUR_AXIS,
    duration_s: float = DEFAULT_TILT_DURATION_S,
    pivot_offset_local: np.ndarray | None = None,
) -> np.ndarray:
    """Slerp current orientation to pour tilt.

    Caller is responsible for moving the arm to ``target_pos`` first
    (usually via ``arm.move_to(...)``). This subtask only handles the
    tilt slerp. Returns the final orientation matrix.

    Without ``pivot_offset_local``: holds the EE control point fixed at
    ``target_pos`` while the orientation slerps (rotation about the EE
    point itself).

    With ``pivot_offset_local`` (a 3-vector in the **EE local frame** —
    e.g. ``[0, 0, 0.127]`` for a point 12.7 cm along the EE local +Z
    axis, i.e. "below" the gripper when the tool points down): we
    anchor the world point
    ``pivot_world = target_pos + R_start @ pivot_offset_local`` at the
    start of the slerp and re-derive the EE position each tick as
    ``pos = pivot_world − R_interp @ pivot_offset_local`` so that
    ``pivot_world`` stays fixed throughout. The gripper effectively
    orbits around that world point.
    """
    target = np.asarray(target_pos, dtype=np.float64).reshape(3).copy()
    pose = arm.read_current_ee_world(ctx.redis)
    if pose is None:
        raise RuntimeError("Could not read current EE pose from Redis")
    _, cur_ori = pose
    R0 = cur_ori.copy()
    R1 = arm.pour_orientation_end(R0, tilt_deg, axis)

    pivot_local: np.ndarray | None = None
    pivot_world: np.ndarray | None = None
    pivot_label = ""
    if pivot_offset_local is not None:
        pivot_local = np.asarray(pivot_offset_local, dtype=np.float64).reshape(3)
        pivot_world = target + R0 @ pivot_local
        pivot_label = (
            f"  pivot_local={pivot_local.tolist()}  "
            f"pivot_world={pivot_world.tolist()}"
        )

    step_gate(
        ctx,
        f"[pour] tilt {tilt_deg:.0f}° about world +{axis.upper()} at "
        f"{target.tolist()}{pivot_label}",
    )

    t0 = time.monotonic()
    while True:
        alpha = min(1.0, (time.monotonic() - t0) / duration_s)
        R_interp = arm.slerp_orientation(R0, R1, alpha)
        if pivot_world is not None and pivot_local is not None:
            pos = pivot_world - R_interp @ pivot_local
        else:
            pos = target
        arm.publish_goal_cartesian(ctx.redis, pos, R_interp)
        if alpha >= 1.0:
            print(f"[pour] tilt complete ({tilt_deg:.0f}°)")
            return R1
        if ctx.q_pressed():
            raise KeyboardInterrupt("quit requested")
        time.sleep(TICK_DT_S)


def return_upright(
    ctx: TaskContext,
    hold_pos: np.ndarray,
    R_from: np.ndarray,
    R_to: np.ndarray,
    *,
    duration_s: float = DEFAULT_TILT_DURATION_S,
    pivot_offset_local: np.ndarray | None = None,
) -> None:
    """Slerp orientation back from poured pose to upright/grasp orientation.

    With ``pivot_offset_local`` we orbit around the same world point
    ``into`` was using: we read the live EE pose at the start of this
    call (which equals where ``into`` left the arm) and anchor
    ``pivot_world = ee_pos_now + R_from @ pivot_offset_local``. Each
    tick the EE position is re-derived as
    ``pos = pivot_world − R_interp @ pivot_offset_local``. Without
    ``pivot_offset_local`` the EE control point is held fixed at
    ``hold_pos`` (legacy behavior).
    """
    hold = np.asarray(hold_pos, dtype=np.float64).reshape(3).copy()
    R0 = np.asarray(R_from, dtype=np.float64).reshape(3, 3)
    R1 = np.asarray(R_to, dtype=np.float64).reshape(3, 3)

    pivot_local: np.ndarray | None = None
    pivot_world: np.ndarray | None = None
    pivot_label = ""
    if pivot_offset_local is not None:
        pivot_local = np.asarray(pivot_offset_local, dtype=np.float64).reshape(3)
        pose = arm.read_current_ee_world(ctx.redis)
        if pose is None:
            raise RuntimeError("Could not read current EE pose from Redis")
        live_pos, _ = pose
        pivot_world = live_pos + R0 @ pivot_local
        pivot_label = (
            f"  pivot_local={pivot_local.tolist()}  "
            f"pivot_world={pivot_world.tolist()}"
        )

    step_gate(ctx, f"[pour] return upright at {hold.tolist()}{pivot_label}")
    t0 = time.monotonic()
    while True:
        alpha = min(1.0, (time.monotonic() - t0) / duration_s)
        R_interp = arm.slerp_orientation(R0, R1, alpha)
        if pivot_world is not None and pivot_local is not None:
            pos = pivot_world - R_interp @ pivot_local
        else:
            pos = hold
        arm.publish_goal_cartesian(ctx.redis, pos, R_interp)
        if alpha >= 1.0:
            print("[pour] return orientation complete")
            return
        if ctx.q_pressed():
            raise KeyboardInterrupt("quit requested")
        time.sleep(TICK_DT_S)
