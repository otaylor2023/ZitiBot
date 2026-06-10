"""OpenSai Franka arm helpers (cartesian goals via Redis)."""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import redis
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from zitibot_core import gains
from zitibot_core.constants import CONFIG_XML, CONTROLLER_TO_USE, JOINT_CONTROLLER
from zitibot_core.redis_keys import (
    CONTROLLER_WAIT_DT_S,
    CONTROLLER_WAIT_TIMEOUT_S,
    KEYS,
)


def decode_redis_value(raw: bytes | str | None) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return raw


def try_redis(host: str, port: int):
    try:
        r = redis.Redis(host=host, port=port, decode_responses=False)
        r.ping()
        return r
    except Exception as e:
        print(f"Redis connect failed ({e}).", file=sys.stderr)
        return None


def validate_config(redis_client, *, config_xml: str | None = None) -> int | None:
    expected = config_xml or os.environ.get("ZITIBOT_OPENSAI_CONFIG_XML", CONFIG_XML)
    raw = redis_client.get(KEYS.config_file_name)
    name = decode_redis_value(raw)
    if name is None:
        print(
            "Warning: ::sai-interfaces-webui::config_file_name not in Redis; "
            "continuing anyway.",
            file=sys.stderr,
        )
        return None
    if name != expected:
        print(
            f"Expected webui config {expected!r} but Redis has {name!r}. "
            "Set ZITIBOT_OPENSAI_CONFIG_XML if needed.",
            file=sys.stderr,
        )
        return 1
    return None


def ensure_cartesian_controller(redis_client) -> None:
    t0 = time.monotonic()
    while True:
        if decode_redis_value(redis_client.get(KEYS.active_controller)) == CONTROLLER_TO_USE:
            return
        if time.monotonic() - t0 > CONTROLLER_WAIT_TIMEOUT_S:
            print(
                f"Warning: could not switch active_controller to {CONTROLLER_TO_USE!r} "
                f"within {CONTROLLER_WAIT_TIMEOUT_S:.0f} s; publishing goals anyway.",
                file=sys.stderr,
            )
            return
        redis_client.set(KEYS.active_controller, CONTROLLER_TO_USE)
        time.sleep(CONTROLLER_WAIT_DT_S)


def publish_goal_cartesian(
    redis_client,
    goal_pos: np.ndarray,
    goal_ori: np.ndarray,
) -> None:
    """Publish desired EE position/orientation to OpenSai cartesian_controller via Redis."""
    ensure_cartesian_controller(redis_client)
    redis_client.set(
        KEYS.cartesian_task_goal_position,
        json.dumps(np.asarray(goal_pos, dtype=np.float64).reshape(3).tolist()),
    )
    redis_client.set(
        KEYS.cartesian_task_goal_orientation,
        json.dumps(np.asarray(goal_ori, dtype=np.float64).reshape(3, 3).tolist()),
    )


_VEL_TOL_DEFAULT = object()  # sentinel: "use runner.DEFAULT_VEL_TOL_RAD_S"
_SETTLE_DEFAULT = object()   # sentinel: "use runner.DEFAULT_SETTLE_TICKS"


def move_to(
    ctx,
    goal_pos: np.ndarray,
    goal_ori: np.ndarray,
    *,
    label: str | None = None,
    tol_m: float = 0.03,
    timeout_s: float = 3.0,
    gated: bool = True,
    vel_tol_rad_s: float | None = _VEL_TOL_DEFAULT,  # type: ignore[assignment]
    settle_ticks: int = _SETTLE_DEFAULT,  # type: ignore[assignment]
    on_within_m: float | None = None,
    on_within=None,
) -> np.ndarray:
    """ENTER-gate when ``ctx.step`` is on, publish goal, then wait for convergence.

    Defaults to gated (top-level / between-subtask moves). Subtasks performing
    an internal sequence of moves should pass ``gated=False`` for the
    intermediate moves so only the subtask's own ``step_gate`` at the major
    state boundary prompts the user. Labels still print when provided so the
    user can see progress.

    Convergence: ``arm.move_to`` defers to
    :func:`zitibot_core.runner.wait_ee_converged`, which requires BOTH
    position-in-tolerance AND ``norm(dq) < vel_tol_rad_s`` for
    ``settle_ticks`` consecutive ticks before returning. This is what
    keeps us from publishing the next goal while the arm is still
    moving (a known FR3 reflex trigger at the *start* of the next
    move). Pass ``vel_tol_rad_s=None`` to skip the velocity gate for a
    given move (e.g. on slow transit moves where you'd rather time out
    quickly than wait out PID jitter). Omit both params to use the
    project-wide defaults (``DEFAULT_VEL_TOL_RAD_S`` /
    ``DEFAULT_SETTLE_TICKS`` in ``runner.py``).

    On convergence timeout the function does NOT raise — it prints the current
    EE position and remaining error then returns, letting the caller continue.
    """
    # Local import to avoid circular dependency with zitibot_core.runner.
    from zitibot_core.runner import (
        DEFAULT_SETTLE_TICKS,
        DEFAULT_VEL_TOL_RAD_S,
        step_gate,
        wait_ee_converged,
    )

    resolved_vel_tol: float | None = (
        DEFAULT_VEL_TOL_RAD_S if vel_tol_rad_s is _VEL_TOL_DEFAULT else vel_tol_rad_s
    )
    resolved_settle_ticks: int = (
        DEFAULT_SETTLE_TICKS if settle_ticks is _SETTLE_DEFAULT else int(settle_ticks)
    )

    goal = np.asarray(goal_pos, dtype=np.float64).reshape(3)
    if gated and ctx.step:
        step_gate(ctx, label if label is not None else f"[arm.move_to] goal={goal.tolist()}")
    elif label is not None:
        print(label, flush=True)
    publish_goal_cartesian(ctx.redis, goal, goal_ori)

    logger = getattr(ctx, "move_logger", None)
    log_label = label if label is not None else f"goal={goal.tolist()}"
    if logger is not None:
        logger.begin_move(goal, log_label, tol_m=tol_m, timeout_s=timeout_s)

    status = "ok"
    final_err: float | None = None
    try:
        wait_ee_converged(
            ctx,
            goal,
            tol_m=tol_m,
            timeout_s=timeout_s,
            vel_tol_rad_s=resolved_vel_tol,
            settle_ticks=resolved_settle_ticks,
            on_within_m=on_within_m,
            on_within=on_within,
        )
    except TimeoutError:
        status = "timeout"
        pose = read_current_ee_world(ctx.redis)
        if pose is None:
            print(
                f"[arm.move_to] timeout after {timeout_s:.2f}s; current EE pose "
                f"unavailable. goal={goal.tolist()}. Continuing.",
                flush=True,
            )
        else:
            cur_pos = pose[0]
            err_vec = cur_pos - goal
            final_err = float(np.linalg.norm(err_vec))
            ori_msg = ""
            try:
                cur_ori = np.asarray(pose[1], dtype=np.float64).reshape(3, 3)
                goal_R = np.asarray(goal_ori, dtype=np.float64).reshape(3, 3)
                # Orientation error as the goal->current relative rotation,
                # expressed in roll/pitch/yaw (XYZ extrinsic, degrees).
                R_err = R.from_matrix(goal_R.T @ cur_ori)
                roll, pitch, yaw = R_err.as_euler("xyz", degrees=True)
                ang_err = float(np.linalg.norm(R_err.as_rotvec()) * 180.0 / np.pi)
                ori_msg = (
                    f" ori_err={ang_err:.2f} deg "
                    f"(roll={roll:+.2f}, pitch={pitch:+.2f}, yaw={yaw:+.2f} deg)"
                )
            except Exception:  # noqa: BLE001 - orientation log is best-effort
                ori_msg = ""
            print(
                f"[arm.move_to] timeout after {timeout_s:.2f}s "
                f"(tol={tol_m:.3f} m). "
                f"goal={goal.tolist()} cur={cur_pos.tolist()} "
                f"err={final_err:.4f} m "
                f"(dx={err_vec[0]:+.4f}, dy={err_vec[1]:+.4f}, "
                f"dz={err_vec[2]:+.4f} m){ori_msg}. Continuing.",
                flush=True,
            )
    except KeyboardInterrupt:
        # Tag the in-flight plot before the interrupt unwinds the stack
        # so the saved PNG title says "[interrupted]" instead of "[ok]".
        # The ``finally`` block below still runs and writes the plot;
        # we re-raise so callers / main() can clean up normally.
        status = "interrupted"
        raise
    finally:
        if logger is not None and logger.active:
            if final_err is None:
                pose = read_current_ee_world(ctx.redis)
                if pose is not None:
                    final_err = float(np.linalg.norm(pose[0] - goal))
            logger.end_move(status=status, final_err_m=final_err)
    return goal


def read_current_ee_world(redis_client) -> tuple[np.ndarray, np.ndarray] | None:
    """Return current ``(position (3,), orientation (3,3))`` from Redis."""
    try:
        raw_p = redis_client.get(KEYS.cartesian_task_current_position)
        raw_o = redis_client.get(KEYS.cartesian_task_current_orientation)
        if raw_p is None or raw_o is None:
            return None
        cur_pos = np.array(json.loads(raw_p), dtype=np.float64).reshape(3)
        cur_ori = np.array(json.loads(raw_o), dtype=np.float64).reshape(3, 3)
        return cur_pos, cur_ori
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def read_joint_positions(redis_client) -> np.ndarray | None:
    """Return current ``q`` (rad, shape ``(7,)``) from the Franka driver, or ``None``."""
    try:
        raw = redis_client.get(KEYS.joint_positions)
        if raw is None:
            return None
        q = np.array(json.loads(raw), dtype=np.float64).reshape(-1)
        if q.size == 0:
            return None
        return q
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def ensure_joint_controller(redis_client) -> None:
    """Switch ``active_controller_name`` to ``joint_controller`` and wait for it.

    Mirror of :func:`ensure_cartesian_controller`. Used by
    :func:`publish_goal_joint` so a one-shot joint command can be issued
    from a controller that is otherwise running in cartesian.
    """
    t0 = time.monotonic()
    while True:
        if decode_redis_value(redis_client.get(KEYS.active_controller)) == JOINT_CONTROLLER:
            return
        if time.monotonic() - t0 > CONTROLLER_WAIT_TIMEOUT_S:
            print(
                f"Warning: could not switch active_controller to {JOINT_CONTROLLER!r} "
                f"within {CONTROLLER_WAIT_TIMEOUT_S:.0f} s; publishing joint goal anyway.",
                file=sys.stderr,
            )
            return
        redis_client.set(KEYS.active_controller, JOINT_CONTROLLER)
        time.sleep(CONTROLLER_WAIT_DT_S)


def publish_goal_joint(
    redis_client,
    q_goal: np.ndarray,
    *,
    zero_vel_acc: bool = True,
) -> None:
    """Seed the joint goal (and optionally zero vel/acc), then switch active controller.

    Order is intentional: seed the goal FIRST so the joint_controller wakes up
    tracking the requested target rather than whatever stale goal was left in
    Redis from a previous run / driver-monitor seed.
    """
    q = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    redis_client.set(KEYS.joint_task_goal_position, json.dumps(q.tolist()))
    if zero_vel_acc:
        z = json.dumps([0.0] * q.size)
        redis_client.set(KEYS.joint_task_goal_velocity, z)
        redis_client.set(KEYS.joint_task_goal_acceleration, z)
    ensure_joint_controller(redis_client)


def wait_joints_converged(
    ctx,
    q_goal: np.ndarray,
    *,
    tol_rad: float = 0.02,
    timeout_s: float = 6.0,
    vel_tol_rad_s: float | None = 0.05,
    settle_ticks: int = 3,
    tick_s: float = 0.02,
) -> None:
    """Wait until ``||q - q_goal|| < tol_rad`` (and ``||dq|| < vel_tol``) for ``settle_ticks``.

    Raises ``TimeoutError`` if the gate isn't met inside ``timeout_s``.
    """
    q_goal_arr = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    t0 = time.monotonic()
    consecutive = 0
    while True:
        q = read_joint_positions(ctx.redis)
        if q is not None and q.shape == q_goal_arr.shape:
            err = float(np.linalg.norm(q - q_goal_arr))
            vel_ok = True
            if vel_tol_rad_s is not None:
                dq = read_joint_velocities(ctx.redis)
                vel_ok = dq is not None and float(np.linalg.norm(dq)) < vel_tol_rad_s
            if err < tol_rad and vel_ok:
                consecutive += 1
                if consecutive >= settle_ticks:
                    return
            else:
                consecutive = 0
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(
                f"wait_joints_converged timed out after {timeout_s:.2f}s (tol={tol_rad:.3f} rad)."
            )
        time.sleep(tick_s)


def move_to_joint(
    ctx,
    q_goal: np.ndarray,
    *,
    label: str | None = None,
    tol_rad: float = 0.02,
    timeout_s: float = 6.0,
    vel_tol_rad_s: float | None = 0.05,
    settle_ticks: int = 3,
    gated: bool = True,
) -> np.ndarray:
    """Switch to ``joint_controller``, publish ``q_goal``, wait for convergence.

    The next ``arm.move_to`` (cartesian) call after this will automatically
    switch the active controller back to ``cartesian_controller`` via
    :func:`publish_goal_cartesian` / :func:`ensure_cartesian_controller`.

    On convergence timeout this function does NOT raise — it prints remaining
    joint error and returns, matching :func:`move_to`'s behavior.
    """
    from zitibot_core.runner import step_gate

    q = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    if gated and ctx.step:
        step_gate(ctx, label if label is not None else f"[arm.move_to_joint] goal={q.tolist()}")
    elif label is not None:
        print(label, flush=True)
    publish_goal_joint(ctx.redis, q)
    try:
        wait_joints_converged(
            ctx, q,
            tol_rad=tol_rad,
            timeout_s=timeout_s,
            vel_tol_rad_s=vel_tol_rad_s,
            settle_ticks=settle_ticks,
        )
    except TimeoutError:
        q_now = read_joint_positions(ctx.redis)
        if q_now is None:
            print(
                f"[arm.move_to_joint] timeout after {timeout_s:.2f}s; current joints "
                f"unavailable. goal={q.tolist()}. Continuing.",
                flush=True,
            )
        else:
            err = float(np.linalg.norm(q_now - q))
            print(
                f"[arm.move_to_joint] timeout after {timeout_s:.2f}s "
                f"(tol={tol_rad:.3f} rad). goal={q.tolist()} cur={q_now.tolist()} "
                f"err={err:.4f} rad. Continuing.",
                flush=True,
            )
    return q


def move_to_joints_partial(
    ctx,
    joint_targets: dict[int, float],
    *,
    degrees: bool = False,
    one_indexed: bool = True,
    label: str | None = None,
    tol_rad: float = 0.02,
    timeout_s: float = 6.0,
    vel_tol_rad_s: float | None = 0.05,
    settle_ticks: int = 3,
    gated: bool = True,
) -> np.ndarray:
    """Move only the joints in ``joint_targets``; leave the rest where they are.

    ``joint_targets`` maps joint index → target. By default indices are
    1-indexed (matching how we talk about Franka joints, "J6 to 164°") and
    values are interpreted as radians; pass ``degrees=True`` to use degrees,
    or ``one_indexed=False`` for 0-indexed.

    The current joint vector is read from Redis and used for every joint not
    in the dict, so the resulting full goal is always a perturbation of the
    *current* pose. This makes it safe to call mid-task ("dump J7 to -151°")
    without worrying about stale joint goals or accidentally driving the
    untouched joints back to a previous setpoint.
    """
    q_now = read_joint_positions(ctx.redis)
    if q_now is None:
        raise RuntimeError(
            "move_to_joints_partial: current joint positions unavailable on Redis; "
            "is the Franka driver running?"
        )
    q_goal = q_now.copy()
    for idx, target in joint_targets.items():
        i = int(idx) - 1 if one_indexed else int(idx)
        if i < 0 or i >= q_goal.size:
            raise IndexError(
                f"move_to_joints_partial: joint index {idx} out of range "
                f"(have {q_goal.size} joints; one_indexed={one_indexed})"
            )
        val = math.radians(float(target)) if degrees else float(target)
        q_goal[i] = val
    if label is None:
        unit = "deg" if degrees else "rad"
        parts = ", ".join(
            f"J{idx}={target:+.2f}{unit}" for idx, target in joint_targets.items()
        )
        label = f"[arm.move_to_joints_partial] {parts}"
    return move_to_joint(
        ctx,
        q_goal,
        label=label,
        tol_rad=tol_rad,
        timeout_s=timeout_s,
        vel_tol_rad_s=vel_tol_rad_s,
        settle_ticks=settle_ticks,
        gated=gated,
    )


def shake_joint(
    ctx,
    *,
    joint_index: int = 0,
    one_indexed: bool = False,
    shake_delta_deg: float,
    shake_cycles: int,
    tol_rad: float = 0.15,
    timeout_s: float = 3.0,
    use_warmup: bool = True,
    warmup_delta_deg: float = 3.0,
    warmup_tol_rad: float = 0.05,
    warmup_timeout_s: float = 2.0,
    post_switch_wait_s: float = 0.0,
    label_prefix: str = "[shake]",
    joint_kp: float | None = 250.0,
    joint_kv: float | None = 40.0,
) -> None:
    """Wobble a single joint ±delta for N UP/DOWN cycles to dislodge a held object.

    Generalised from the egg-cracker's post-crack / post-release J0 shake.
    Stays in ``joint_controller`` the whole time so every cycle runs under a
    single ENTER press in ``--step`` mode. Each stroke is RELATIVE to the live
    joint reading at the start of that stroke, so a DOWN stroke always returns
    to where its matching UP stroke started even if a stroke doesn't perfectly
    converge.

    ``joint_index`` selects which joint to shake (0-indexed by default; pass
    ``one_indexed=True`` to use Franka "J1..J7" numbering).

    The cartesian→joint controller swap needs to settle before the first real
    stroke or it tends to eat that stroke (the joint controller reseeds its
    goal to the live measured joints when it activates). Two ways to force it:

    * ``use_warmup=True`` (default): issue a tiny ``+warmup_delta_deg`` nudge
      first as a "fake" move so the swap lands on a near-current goal.
    * ``use_warmup=False``: seed the current joints as the goal (zero motion),
      switch controllers, then ``time.sleep(post_switch_wait_s)`` before the
      first stroke.

    Falls back to a printed warning + no-op if joint positions aren't available
    from Redis.
    """
    j = int(joint_index) - 1 if one_indexed else int(joint_index)

    q_pre = read_joint_positions(ctx.redis)
    if q_pre is None or q_pre.size <= j or j < 0:
        print(
            f"{label_prefix} WARNING: joint positions unavailable from Redis "
            f"(need index {j}); skipping shake (warmup + cycles)."
        )
        return

    with gains.boosted_joint_gains(
        ctx.redis, kp=joint_kp, kv=joint_kv, label=label_prefix
    ):
        if use_warmup:
            q_warmup_rad = float(q_pre[j]) + math.radians(warmup_delta_deg)
            print(
                f"{label_prefix}-warmup J{j} warm-up nudge to "
                f"{math.degrees(q_warmup_rad):+.1f}° "
                f"(forces cartesian→joint controller swap)"
            )
            move_to_joints_partial(
                ctx,
                {j: q_warmup_rad},
                degrees=False,
                one_indexed=False,
                label=(
                    f"  [arm] {label_prefix} warm-up J{j}"
                    f"={math.degrees(q_warmup_rad):+.1f}°"
                ),
                tol_rad=warmup_tol_rad,
                timeout_s=warmup_timeout_s,
                gated=False,
            )
        else:
            print(
                f"{label_prefix} no warmup; seeding current joints + switching to "
                f"joint controller, waiting {post_switch_wait_s * 1000:.0f} ms for swap."
            )
            publish_goal_joint(ctx.redis, q_pre)
            if post_switch_wait_s > 0:
                time.sleep(post_switch_wait_s)

        delta_deg = float(shake_delta_deg)
        cycles = int(shake_cycles)
        for cycle in range(cycles):
            for direction, sign in (("UP", +1.0), ("DOWN", -1.0)):
                q_now = read_joint_positions(ctx.redis)
                if q_now is None or q_now.size <= j:
                    print(
                        f"{label_prefix} cyc {cycle + 1}/{cycles} {direction}: "
                        f"WARNING: joint positions unavailable; aborting shake."
                    )
                    return
                qj_now = float(q_now[j])
                qj_goal = qj_now + sign * math.radians(delta_deg)
                print(
                    f"{label_prefix} cyc {cycle + 1}/{cycles} {direction}: "
                    f"J{j} {math.degrees(qj_now):+.1f}° → "
                    f"{math.degrees(qj_goal):+.1f}° "
                    f"({'+' if sign > 0 else '-'}{delta_deg:.0f}°)"
                )
                move_to_joints_partial(
                    ctx,
                    {j: qj_goal},
                    degrees=False,
                    one_indexed=False,
                    label=(
                        f"  [arm] {label_prefix} cyc {cycle + 1} {direction} "
                        f"J{j}={math.degrees(qj_goal):+.1f}°"
                    ),
                    tol_rad=tol_rad,
                    timeout_s=timeout_s,
                    gated=False,
                )


def read_joint_velocities(redis_client) -> np.ndarray | None:
    """Return current ``dq`` (rad/s, shape ``(7,)``) from the Franka driver, or ``None``.

    Published by the redis_driver every control tick straight from
    ``robot_state.dq`` (libfranka). Useful as a "is the arm still
    moving?" signal — see ``wait_ee_converged`` in ``runner.py`` and
    its ``vel_tol_rad_s`` parameter, which uses ``norm(dq)`` to gate
    convergence on top of the usual position tolerance.

    Returns ``None`` if the key is missing or unparseable; the caller
    is expected to degrade to a position-only check in that case.
    """
    try:
        raw = redis_client.get(KEYS.joint_velocities)
        if raw is None:
            return None
        dq = np.array(json.loads(raw), dtype=np.float64).reshape(-1)
        if dq.size == 0:
            return None
        return dq
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def print_startup_pose(redis_client, *, label: str = "[arm] startup") -> None:
    """Print current EE world pose and joint angles to stdout.

    Used by every controller at startup via ``make_context`` so we always have
    a record of where the arm actually was when a run began (useful when
    grabbing waypoints by hand-jogging then reading the log).
    """
    np.set_printoptions(precision=4, suppress=True)
    ee_pose = read_current_ee_world(redis_client)
    if ee_pose is None:
        print(f"{label} EE world pose: <unavailable on Redis>")
    else:
        ee_pos, ee_R = ee_pose
        print(
            f"{label} EE world position: "
            f"[{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}] m"
        )
        print(f"{label} EE world orientation (3x3 rot):")
        for row in ee_R:
            print(f"  [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
    q = read_joint_positions(redis_client)
    if q is None:
        print(f"{label} joint positions: <unavailable on Redis>")
        return
    q_deg = np.degrees(q)
    print(
        f"{label} joint positions (rad): "
        + "  ".join(f"q{i + 1}={q[i]:+.4f}" for i in range(q.size))
    )
    print(
        f"{label} joint positions (deg): "
        + "  ".join(f"q{i + 1}={q_deg[i]:+7.2f}" for i in range(q.size))
    )


def print_ee_status(redis_client) -> None:
    """Print current vs. goal EE position to stdout."""
    cur = read_current_ee_world(redis_client)
    try:
        raw_gp = redis_client.get(KEYS.cartesian_task_goal_position)
        goal_pos = np.array(json.loads(raw_gp), dtype=np.float64).reshape(3) if raw_gp else None
    except (json.JSONDecodeError, TypeError, ValueError):
        goal_pos = None
    cur_str = f"{cur[0].tolist()}" if cur is not None else "unavailable"
    goal_str = f"{goal_pos.tolist()}" if goal_pos is not None else "unavailable"
    print(f"[ee status] current={cur_str}  goal={goal_str}")


def read_T_base_flange(redis_client, key: str | None = None) -> np.ndarray | None:
    redis_key = key if key is not None else KEYS.endeffector_transform
    try:
        raw = redis_client.get(redis_key)
        if raw is None:
            return None
        T = np.array(json.loads(raw), dtype=np.float64)
        if T.shape != (4, 4):
            T = T.reshape(4, 4)
        return T
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def pour_orientation_end(
    R_start: np.ndarray, tilt_deg: float, axis: str = "x"
) -> np.ndarray:
    """Pour tilt: rotation about world +X or +Y by ``tilt_deg``."""
    ax = axis.strip().lower()
    if ax == "x":
        R_tilt = R.from_euler("x", math.radians(tilt_deg)).as_matrix()
    elif ax == "y":
        R_tilt = R.from_euler("y", math.radians(tilt_deg)).as_matrix()
    else:
        raise ValueError(f"Unknown pour axis {axis!r}; use 'x' or 'y'.")
    return R_tilt @ np.asarray(R_start, dtype=np.float64).reshape(3, 3)


def rotate_tool_frame(
    R_start: np.ndarray, angle_deg: float, axis: str = "x"
) -> np.ndarray:
    """Rotate about the tool's OWN axis (body frame), not the world axis.

    ``R_start`` columns are the EE body axes expressed in world (col 0 =
    tool +X, the direction the end-effector faces; col 2 = tool +Z). A
    rotation about the *tool's* own ``axis`` is a POST-multiply:
    ``R_start @ R_axis(angle)``. Contrast with :func:`pour_orientation_end`,
    which pre-multiplies (``R_axis @ R_start``) for a rotation about the
    fixed world axis.
    """
    ax = axis.strip().lower()
    if ax not in ("x", "y", "z"):
        raise ValueError(f"Unknown tool axis {axis!r}; use 'x', 'y', or 'z'.")
    R_rot = R.from_euler(ax, math.radians(angle_deg)).as_matrix()
    return np.asarray(R_start, dtype=np.float64).reshape(3, 3) @ R_rot


def slerp_orientation(R_start: np.ndarray, R_end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    key_rots = R.concatenate(
        [R.from_matrix(R_start), R.from_matrix(R_end)]
    )
    return Slerp([0.0, 1.0], key_rots)([alpha]).as_matrix()[0]


def ee_position_error(redis_client, goal_pos: np.ndarray) -> float | None:
    pose = read_current_ee_world(redis_client)
    if pose is None:
        return None
    return float(np.linalg.norm(pose[0] - np.asarray(goal_pos, dtype=np.float64).reshape(3)))
