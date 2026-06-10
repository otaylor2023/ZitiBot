"""Egg cracker subtask."""

from __future__ import annotations

import time

import numpy as np

from zitibot_core import gripper
from zitibot_core.constants import DEFAULT_GRIPPER_SPEED, OBJECT_DEFAULTS, Object
from zitibot_core.context import TaskContext
from zitibot_core.runner import step_gate
from zitibot_tasks import grasp


def crack(
    ctx: TaskContext,
    *,
    crack_force: float = 140.0,
    lift_force: float = 8.0,
    squeezes: int = 3,
) -> None:
    """Squeeze the held egg cracker to break the egg.

    Assumes the gripper is already holding the egg cracker (i.e. caller
    has run ``grasp.object`` and any in-between motion). Performs an
    ENTER-gated ``[egg_crack] crack egg`` step, then repeats ``squeezes``
    times:

    1. Unsqueeze: open jaws ~5 mm wider than the hold width at
       ``lift_force``. REQUIRED before a force grasp — if the jaws are
       already at/closer than the grasp target, the grasp doesn't fire.
    2. Squeeze shut at full ``crack_force`` (force grasp), hold 2 s.

    Ends on a full-force squeeze (jaws clamped) so the caller can shake
    while still squeezing. The arm itself does not move during ``crack``
    — position the EE above the target bowl beforehand.
    """
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    # Crack at the Franka Hand's max grip speed so the squeeze snaps shut fast
    # (the cracker's normal grip speed is deliberately slowed for the careful
    # pick-up; the crack itself just needs to slam closed).
    crack_speed = DEFAULT_GRIPPER_SPEED
    n = max(int(squeezes), 1)
    step_gate(
        ctx,
        f"[egg_crack] crack egg "
        f"({n}x full-force squeeze, lift_force={lift_force} N, "
        f"force={crack_force} N, speed={crack_speed:.3f} m/s)",
    )
    cur_w = gripper.read_current_width(ctx.redis)
    # Width the gripper is holding the cracker at; reopen to this between
    # squeezes so the shell can shift.
    hold_w = cur_w if cur_w is not None else 0.02
    # Unsqueeze a touch wider than the hold width so each force grasp
    # registers jaw motion (otherwise it doesn't squeeze at all).
    release_w = hold_w + 0.005
    for i in range(n):
        gripper.move(ctx.redis, release_w, speed=crack_speed, force=lift_force)
        time.sleep(0.4)
        gripper.grasp(ctx.redis, spec.close_width, speed=crack_speed, force=crack_force)
        time.sleep(2.0)


def run(
    ctx: TaskContext,
    *,
    pick_pos: np.ndarray | None = None,
    drop_pos: np.ndarray | None = None,
    ori: np.ndarray | None = None,
    crack_force: float = 70.0,
    lift_force: float = 8.0,
) -> None:
    """Single-station pick → squeeze → place.

    Convenience wrapper for the original arm-only egg-crack flow (no
    base motion, no sink drop). Multi-station controllers should call
    :func:`zitibot_tasks.grasp.object`, :func:`crack`, and the sink-drop
    helper directly instead.

    ``ori`` overrides the default tool-down grasp_ori (used by both the
    grasp and place steps so the wrist orientation stays consistent
    while holding the cracker). Pass the orientation returned by
    ``gemini.find_grasp_pose`` to grasp + place perpendicular to the
    detected strip axis.
    """
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    pick = np.asarray(
        pick_pos if pick_pos is not None else spec.pick_pose,
        dtype=np.float64,
    ).reshape(3)
    if drop_pos is None:
        drop = pick + np.array([0.0, 0.05, 0.0], dtype=np.float64)
    else:
        drop = np.asarray(drop_pos, dtype=np.float64).reshape(3)
    grip_R = (
        np.asarray(ori, dtype=np.float64).reshape(3, 3).copy()
        if ori is not None
        else spec.grasp_ori.copy()
    )

    grasp.object(ctx, Object.EGG_CRACKER, pick_pos=pick, ori=grip_R)
    crack(ctx, crack_force=crack_force, lift_force=lift_force)
    grasp.place(ctx, Object.EGG_CRACKER, place_pos=drop, ori=grip_R)
    print("[egg_crack] sequence complete")
