"""Small loop primitives used inside subtasks."""

from __future__ import annotations

import sys
import time
from typing import Callable

import numpy as np

from zitibot_core import arm, gripper
from zitibot_core.constants import TICK_DT_S
from zitibot_core.context import TaskContext


def read_stdin_line(timeout_s: float = 86400.0) -> str | None:
    """Block until one line is available, or return None on EOF."""
    if not sys.stdin.isatty():
        return None
    import select

    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    except (ValueError, OSError):
        return None
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":
        return None
    return line


def step_gate(ctx: TaskContext, label: str) -> None:
    """If ``ctx.step`` is on, block until ENTER. Honors ``q`` to quit."""
    print(label, flush=True)
    if not ctx.step:
        return
    print("  Press ENTER to advance, or type q then ENTER to quit.", flush=True)
    line = read_stdin_line()
    if line is None:
        raise KeyboardInterrupt("stdin closed")
    token = line.strip().lower()
    if token in ("q", "quit", "exit"):
        raise KeyboardInterrupt("quit requested")


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 15.0,
    tick: float = TICK_DT_S,
    ctx: TaskContext | None = None,
    label: str = "waiting",
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        if predicate():
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"{label}: timed out after {timeout_s}s")
        if ctx is not None and ctx.q_pressed():
            raise KeyboardInterrupt("quit requested")
        time.sleep(tick)


# Defaults for the velocity-gated convergence check used by
# ``wait_ee_converged`` (and therefore ``arm.move_to``). Picked to
# treat the arm as "essentially still" before declaring a move
# complete, so we don't publish the next goal while the arm still has
# meaningful velocity. That goal discontinuity is one of the things
# that trips the FR3's ``communication_constraints_violation`` reflex
# at the *start* of the next move (the inbound torque from the new
# goal slams into a moving joint).
#
# DEFAULT_VEL_TOL_RAD_S
#   Threshold on ``norm(dq)`` (rad/s across all 7 joints) below which
#   the arm counts as still. 0.05 rad/s ≈ 2.9 deg/s on each joint if
#   evenly spread — well inside "the arm has finished moving" but
#   loose enough to not stall on PID noise / FCI dq quantization.
#   Set ``vel_tol_rad_s=None`` on a per-call basis to skip the
#   velocity check entirely (position-only, legacy behavior).
# DEFAULT_SETTLE_TICKS
#   Number of consecutive convergence-check ticks (each ``TICK_DT_S``
#   = 20 ms by default) where BOTH position AND velocity must be in
#   tolerance. 3 ticks ≈ 60 ms of "actually settled, not just briefly
#   crossed". Lower for snappier transit; raise for fragile moves
#   like the descent before a grasp.
DEFAULT_VEL_TOL_RAD_S: float = 0.05
DEFAULT_SETTLE_TICKS: int = 3


def wait_ee_converged(
    ctx: TaskContext,
    goal_pos: np.ndarray,
    *,
    tol_m: float = 0.03,
    timeout_s: float = 15.0,
    vel_tol_rad_s: float | None = DEFAULT_VEL_TOL_RAD_S,
    settle_ticks: int = DEFAULT_SETTLE_TICKS,
    on_within_m: float | None = None,
    on_within: Callable[[], None] | None = None,
) -> None:
    """Block until the EE is within ``tol_m`` of ``goal_pos`` *and* nearly still.

    A move counts as converged when BOTH:

    * ``||current_position - goal|| < tol_m``, AND
    * ``||dq|| < vel_tol_rad_s`` (when ``vel_tol_rad_s`` is not ``None``
      and the Franka driver is publishing joint velocities on Redis),

    held for ``settle_ticks`` consecutive convergence-check ticks. The
    velocity check is the new piece: previously this function would
    return as soon as position dipped under tolerance, which let
    ``arm.move_to`` publish the next goal while the arm was still
    moving. That's a recipe for FR3 reflex aborts
    (``communication_constraints_violation``) at the *start* of the
    following move, because the new goal injects a torque step into a
    joint that's already mid-trajectory.

    Set ``vel_tol_rad_s=None`` to fall back to the legacy
    position-only check. If the joint-velocities key isn't in Redis
    (e.g. driver hasn't come up yet), the velocity gate degrades
    gracefully to position-only and prints a one-shot warning.

    When ``ctx.move_logger`` is attached and a move is active, the EE
    pose is sampled on every tick so the logger can produce the usual
    position-vs-time plot.
    """
    goal = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    logger = getattr(ctx, "move_logger", None)
    settle_required = max(1, int(settle_ticks))
    settle_count = 0
    # Cap warnings so a missing key doesn't spam every 20 ms tick.
    vel_unavailable_warned = False
    # One-shot "within radius" hook: fires ``on_within`` the first tick the EE
    # comes within ``on_within_m`` of the goal (used by grasp.object to switch
    # to precise/slow gains for the final approach). Independent of the
    # convergence tolerance — it can fire well before the move is "done".
    within_fired = on_within_m is None or on_within is None

    def _ok() -> bool:
        nonlocal settle_count, vel_unavailable_warned, within_fired

        pose = arm.read_current_ee_world(ctx.redis)
        if pose is None:
            settle_count = 0
            return False
        cur_pos = pose[0]
        if logger is not None and logger.active:
            logger.sample(cur_pos)

        dist = float(np.linalg.norm(cur_pos - goal))
        if not within_fired and dist < float(on_within_m):
            within_fired = True
            try:
                on_within()
            except Exception as e:  # noqa: BLE001 - hook must never break the wait
                print(f"[wait_ee_converged] on_within hook raised: {e}", flush=True)

        pos_ok = dist < tol_m
        if not pos_ok:
            settle_count = 0
            return False

        vel_ok = True
        if vel_tol_rad_s is not None:
            dq = arm.read_joint_velocities(ctx.redis)
            if dq is None:
                if not vel_unavailable_warned:
                    print(
                        "[wait_ee_converged] joint_velocities not on Redis; "
                        "falling back to position-only convergence "
                        "(driver may not be publishing dq).",
                        flush=True,
                    )
                    vel_unavailable_warned = True
                # Treat as "not stalling on missing data" — degrade.
                vel_ok = True
            else:
                vel_ok = float(np.linalg.norm(dq)) < vel_tol_rad_s

        if not vel_ok:
            settle_count = 0
            return False

        settle_count += 1
        return settle_count >= settle_required

    wait_until(
        _ok,
        timeout_s=timeout_s,
        ctx=ctx,
        label=f"EE converge to {goal.tolist()}",
    )
    # If the move converged/timed out before ever entering the within radius
    # (e.g. a sub-5 cm move that settled instantly), still fire the hook once
    # so callers depending on it (precise-grasp engage) aren't skipped.
    if not within_fired:
        try:
            on_within()
        except Exception as e:  # noqa: BLE001
            print(f"[wait_ee_converged] on_within hook raised: {e}", flush=True)


def wait_gripper_converged(
    ctx: TaskContext,
    target_w: float,
    *,
    tol: float = 0.003,
    timeout_s: float = 3.0,
) -> None:
    target = float(target_w)

    def _ok() -> bool:
        w = gripper.read_current_width(ctx.redis)
        return w is not None and abs(w - target) < tol

    wait_until(
        _ok,
        timeout_s=timeout_s,
        ctx=ctx,
        label=f"gripper converge to {target:.4f} m",
    )
