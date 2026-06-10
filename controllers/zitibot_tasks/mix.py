"""Mixing subtask (ladle in bowl)."""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from zitibot_core import arm
from zitibot_core.constants import OBJECT_DEFAULTS, Object, TICK_DT_S
from zitibot_core.context import TaskContext
from zitibot_core.runner import step_gate
from zitibot_tasks import grasp


def _enter_watcher() -> threading.Event:
    """Start a daemon thread that sets the returned Event when ENTER is pressed.

    Lets a loop run "until the user hits ENTER" without blocking the motion:
    poll ``event.is_set()`` between iterations. Works regardless of ``--step``.
    """
    stop = threading.Event()

    def _wait() -> None:
        try:
            input()
        except (EOFError, OSError):
            pass
        stop.set()

    threading.Thread(target=_wait, daemon=True).start()
    return stop


def in_bowl(
    ctx: TaskContext,
    bowl_pos: np.ndarray,
    *,
    ladle_obj: Object = Object.LADLE,
    radius_m: float = 0.04,
    cycles: int = 3,
    cycle_duration_s: float = 4.0,
) -> None:
    """Grasp ladle, move into bowl, execute circular stir, lift out."""
    bowl = np.asarray(bowl_pos, dtype=np.float64).reshape(3)
    ladle_spec = OBJECT_DEFAULTS[ladle_obj]
    ladle_pick = ladle_spec.pick_pose
    if ladle_pick is None:
        raise ValueError(f"{ladle_obj.value}: no default pick_pose")

    grasp.object(ctx, ladle_obj, pick_pos=ladle_pick)
    mix_center = bowl.copy()
    mix_center[2] += 0.02
    grip_R = ladle_spec.grasp_ori.copy()

    arm.move_to(ctx, mix_center, grip_R, label=f"[mix] move above bowl center {mix_center.tolist()}")
    lower = mix_center.copy()
    lower[2] -= 0.03
    arm.move_to(ctx, lower, grip_R, label=f"[mix] lower into bowl to {lower.tolist()}")
    stir_start = lower + np.array([radius_m, 0.0, 0.0], dtype=np.float64)
    arm.move_to(ctx, stir_start, grip_R,
                label=f"[mix] move to stir start (theta=0) {stir_start.tolist()}")
    step_gate(ctx, f"[mix] stir {cycles} cycle(s) radius={radius_m} m")

    t_cycle = cycle_duration_s / max(cycles, 1)
    for c in range(cycles):
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= t_cycle:
                break
            theta = 2.0 * math.pi * (elapsed / t_cycle)
            offset = np.array(
                [radius_m * math.cos(theta), radius_m * math.sin(theta), 0.0],
                dtype=np.float64,
            )
            pos = lower + offset
            arm.publish_goal_cartesian(ctx.redis, pos, grip_R)
            if ctx.p_pressed():
                arm.print_ee_status(ctx.redis)
            if ctx.q_pressed():
                raise KeyboardInterrupt("quit requested")
            time.sleep(TICK_DT_S)
        print(f"[mix] cycle {c + 1}/{cycles} complete")

    arm.move_to(ctx, mix_center, grip_R, label=f"[mix] lift out to {mix_center.tolist()}")
    print("[mix] stir complete")


def stir_at_pose(
    ctx: TaskContext,
    above_pos: np.ndarray,
    above_ori: np.ndarray,
    *,
    down_dz_m: float,
    radius_m: float = 0.04,
    cycles: int = 3,
    cycle_duration_s: float = 4.0,
) -> None:
    """Stir at a hand-taught above pose.

    Variant of :func:`stir_in_bowl` that takes the above-pose directly
    instead of deriving it from a bowl center. Used by callers (e.g.
    ``mixing_vision_base_controller``) where the bowl-center
    derivation is wrong because the held ladle isn't perfectly
    tool-down — the taught above pose captures both the right XYZ
    above the bowl AND the held ladle's actual orientation.

    Sequence (all moves use the same ``above_ori``):
      1. ``arm.move_to(above_pos, above_ori)`` — taught above pose.
      2. ``arm.move_to(above_pos - [0, 0, down_dz_m], above_ori)`` —
         straight down by ``down_dz_m`` to the stir base.
      3. ``arm.move_to(stir_base + [radius_m, 0, 0], above_ori)`` —
         circle start (theta=0).
      4. Streams a circular XY trajectory around the stir base for
         ``cycles`` × ``cycle_duration_s / cycles`` seconds each, at
         ``radius_m`` (orientation held).
      5. ``arm.move_to(above_pos, above_ori)`` — lift back to taught
         above pose. Caller is responsible for any post-lift carry
         motion / gripper release.

    Assumes the arm is already holding the ladle at the same
    orientation it should keep through the stir.
    """
    above = np.asarray(above_pos, dtype=np.float64).reshape(3)
    ori = np.asarray(above_ori, dtype=np.float64).reshape(3, 3)
    lower = above - np.array([0.0, 0.0, float(down_dz_m)], dtype=np.float64)

    arm.move_to(ctx, above, ori,
                label=f"[mix] above (taught) {above.tolist()}")
    arm.move_to(ctx, lower, ori,
                label=(
                    f"[mix] descend {float(down_dz_m) * 100:.1f} cm to "
                    f"stir base {lower.tolist()}"
                ))
    stir_start = lower + np.array([radius_m, 0.0, 0.0], dtype=np.float64)
    arm.move_to(ctx, stir_start, ori,
                label=f"[mix] move to stir start (theta=0) {stir_start.tolist()}")
    step_gate(ctx, f"[mix] stir {cycles} cycle(s) radius={radius_m} m")

    t_cycle = cycle_duration_s / max(cycles, 1)
    for c in range(cycles):
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= t_cycle:
                break
            theta = 2.0 * math.pi * (elapsed / t_cycle)
            pos = lower + np.array(
                [radius_m * math.cos(theta), radius_m * math.sin(theta), 0.0],
                dtype=np.float64,
            )
            arm.publish_goal_cartesian(ctx.redis, pos, ori)
            if ctx.p_pressed():
                arm.print_ee_status(ctx.redis)
            if ctx.q_pressed():
                raise KeyboardInterrupt("quit requested")
            time.sleep(TICK_DT_S)
        print(f"[mix] cycle {c + 1}/{cycles} complete")

    arm.move_to(ctx, above, ori,
                label=f"[mix] lift back to taught above {above.tolist()}")
    print("[mix] stir complete")


def stir_in_bowl(
    ctx: TaskContext,
    bowl_pos: np.ndarray,
    grip_R: np.ndarray,
    *,
    radius_m: float = 0.04,
    cycles: int = 3,
    cycle_duration_s: float = 4.0,
) -> None:
    """Move ladle above bowl, descend, stir, lift out. Does NOT grasp or release.

    Assumes the arm is already holding the ladle at grip_R.
    Caller is responsible for opening the gripper afterward.
    """
    bowl = np.asarray(bowl_pos, dtype=np.float64).reshape(3)
    mix_center = bowl + np.array([0.0, 0.0, 0.02])
    lower = mix_center - np.array([0.0, 0.0, 0.03])

    arm.move_to(ctx, mix_center, grip_R,
                label=f"[mix] move above bowl center {mix_center.tolist()}")
    arm.move_to(ctx, lower, grip_R,
                label=f"[mix] lower into bowl to {lower.tolist()}")
    stir_start = lower + np.array([radius_m, 0.0, 0.0], dtype=np.float64)
    arm.move_to(ctx, stir_start, grip_R,
                label=f"[mix] move to stir start (theta=0) {stir_start.tolist()}")
    step_gate(ctx, f"[mix] stir {cycles} cycle(s) radius={radius_m} m")

    t_cycle = cycle_duration_s / max(cycles, 1)
    for c in range(cycles):
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= t_cycle:
                break
            theta = 2.0 * math.pi * (elapsed / t_cycle)
            pos = lower + np.array(
                [radius_m * math.cos(theta), radius_m * math.sin(theta), 0.0],
                dtype=np.float64,
            )
            arm.publish_goal_cartesian(ctx.redis, pos, grip_R)
            if ctx.p_pressed():
                arm.print_ee_status(ctx.redis)
            if ctx.q_pressed():
                raise KeyboardInterrupt("quit requested")
            time.sleep(TICK_DT_S)
        print(f"[mix] cycle {c + 1}/{cycles} complete")

    arm.move_to(ctx, mix_center, grip_R,
                label=f"[mix] lift out to {mix_center.tolist()}")
    print("[mix] stir complete")


def _stream_radial_gather(
    ctx: TaskContext,
    base: np.ndarray,
    ori: np.ndarray,
    *,
    gather_radius_m: float,
    sweeps: int,
    sweep_duration_s: float,
    tag: str = "gather",
) -> None:
    """Push from pan rim toward center along ``sweeps`` evenly spaced rays."""
    base = np.asarray(base, dtype=np.float64).reshape(3)
    ori = np.asarray(ori, dtype=np.float64).reshape(3, 3)
    n = max(int(sweeps), 1)
    dur = max(float(sweep_duration_s), 0.05)
    r = float(gather_radius_m)

    for i in range(n):
        angle = 2.0 * math.pi * float(i) / float(n)
        direction = np.array(
            [math.cos(angle), math.sin(angle), 0.0], dtype=np.float64
        )
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= dur:
                break
            # u=1 at rim, u→0 at center (gather inward).
            u = 1.0 - min(elapsed / dur, 1.0)
            pos = base + r * u * direction
            arm.publish_goal_cartesian(ctx.redis, pos, ori)
            if ctx.p_pressed():
                arm.print_ee_status(ctx.redis)
            if ctx.q_pressed():
                raise KeyboardInterrupt("quit requested")
            time.sleep(TICK_DT_S)
        print(f"[{tag}] sweep {i + 1}/{n} complete")


def gather_at_pose(
    ctx: TaskContext,
    above_pos: np.ndarray,
    above_ori: np.ndarray,
    *,
    down_dz_m: float,
    gather_radius_m: float = 0.05,
    sweeps: int = 4,
    sweep_duration_s: float = 2.0,
) -> None:
    """Hover over pan, descend, rim→center gathers, lift back to hover."""
    above = np.asarray(above_pos, dtype=np.float64).reshape(3)
    ori = np.asarray(above_ori, dtype=np.float64).reshape(3, 3)
    lower = above - np.array([0.0, 0.0, float(down_dz_m)], dtype=np.float64)

    arm.move_to(
        ctx,
        above,
        ori,
        label=f"[gather] hover above pan {above.tolist()}",
    )
    arm.move_to(
        ctx,
        lower,
        ori,
        label=(
            f"[gather] descend {float(down_dz_m) * 100:.1f} cm to "
            f"work depth {lower.tolist()}"
        ),
    )
    step_gate(
        ctx,
        f"[gather] {sweeps} radial sweep(s), radius={gather_radius_m:.3f} m",
    )
    _stream_radial_gather(
        ctx,
        lower,
        ori,
        gather_radius_m=gather_radius_m,
        sweeps=sweeps,
        sweep_duration_s=sweep_duration_s,
        tag="gather",
    )
    arm.move_to(
        ctx,
        above,
        ori,
        label=f"[gather] lift to hover {above.tolist()}",
    )
    print("[gather] phase complete")


def _cardinal_xy(center: np.ndarray, radius_m: float) -> dict[str, np.ndarray]:
    """Pan cardinal points in world XY at ``radius_m`` from ``center``."""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r = float(radius_m)
    return {
        "top": c + np.array([0.0, r, 0.0], dtype=np.float64),
        "bottom": c + np.array([0.0, -r, 0.0], dtype=np.float64),
        "right": c + np.array([r, 0.0, 0.0], dtype=np.float64),
        "left": c + np.array([-r, 0.0, 0.0], dtype=np.float64),
    }


def _follow_waypoint(
    ctx: TaskContext,
    pos: np.ndarray,
    ori: np.ndarray,
    *,
    tol_m: float,
    timeout_s: float,
    label: str | None = None,
) -> None:
    """Drive to one OTG waypoint and return once the EE is within ``tol_m``.

    Skips the velocity-settle gate (``vel_tol_rad_s=None``, ``settle_ticks=1``)
    so the arm does NOT come to a full stop at each waypoint — the next goal is
    published while it is still moving and the controller's OTG blends the two,
    keeping the stir continuous. With OTG ON this is smooth; the previous
    per-tick goal stream with OTG OFF was what made it jerky.
    """
    arm.move_to(
        ctx,
        pos,
        ori,
        label=label,
        tol_m=tol_m,
        timeout_s=timeout_s,
        gated=False,
        vel_tol_rad_s=None,
        settle_ticks=1,
    )
    if ctx.p_pressed():
        arm.print_ee_status(ctx.redis)
    if ctx.q_pressed():
        raise KeyboardInterrupt("quit requested")


def _trace_cardinal_triangles(
    ctx: TaskContext,
    center: np.ndarray,
    ori: np.ndarray,
    *,
    radius_m: float,
    waypoint_tol_m: float,
    waypoint_timeout_s: float,
) -> None:
    """Trace four triangles from top, bottom, right, and left pan extremes.

    Each triangle starts at one cardinal rim point and visits the other two
    rim points that best cover the pan interior. Z stays at work depth. The
    corners are handed to the controller's OTG as discrete waypoints (no dense
    per-tick stream) so the motion between them is smoothly planned.
    """
    fixed_z = float(center[2])
    pts = _cardinal_xy(center, radius_m)
    triangles = [
        ("top triangle", [pts["top"], pts["right"], pts["left"]]),
        ("bottom triangle", [pts["bottom"], pts["left"], pts["right"]]),
        ("right triangle", [pts["right"], pts["top"], pts["bottom"]]),
        ("left triangle", [pts["left"], pts["bottom"], pts["top"]]),
    ]

    for label, vertices in triangles:
        # Closed loop: the three corners then back to the first corner.
        loop = [v.copy() for v in vertices] + [vertices[0].copy()]
        for i, v in enumerate(loop):
            v = v.copy()
            v[2] = fixed_z
            _follow_waypoint(
                ctx,
                v,
                ori,
                tol_m=waypoint_tol_m,
                timeout_s=waypoint_timeout_s,
                label=(
                    f"[scramble] {label} corner {i + 1}/{len(loop)} {v.tolist()}"
                ),
            )
        print(f"[scramble] {label} complete")


def scramble_at_pose(
    ctx: TaskContext,
    above_pos: np.ndarray,
    above_ori: np.ndarray,
    *,
    down_dz_m: float,
    radius_m: float = 0.035,
    cycles: int = 6,
    cycle_duration_s: float = 12.0,
    triangle_passes: int = 1,
    triangle_duration_s: float = 4.0,
    scramble_duration_s: float | None = None,
    until_enter: bool = False,
    circle_segments: int = 8,
    waypoint_tol_m: float = 0.02,
    waypoint_timeout_s: float = 3.0,
    descend_from_hover: bool = True,
    lift_after: bool = True,
) -> None:
    """Descend from hover (optional), cardinal triangles, circles, lift back.

    Triangle phase: from top, bottom, right, and left rim points each trace one
    closed triangle at ``radius_m``. If ``scramble_duration_s`` is set, the full
    four-triangle set is REPEATED until that many seconds have elapsed (at least
    one pass always runs); otherwise it runs ``triangle_passes`` times. Circle
    phase: ``cycles`` revolutions, each sampled as ``circle_segments`` discrete
    waypoints around the rim. Z is pinned to the work depth throughout.

    All motion is handed to the controller's Cartesian OTG as **sparse
    waypoints** (triangle corners; ``circle_segments`` points per revolution)
    rather than a dense per-tick goal stream — OTG plans the smooth path between
    them. ``cycle_duration_s`` / ``triangle_duration_s`` are no longer used to
    pace a stream (speed is governed by the OTG velocity caps); they are kept
    for signature compatibility.
    """
    _ = (cycle_duration_s, triangle_duration_s)  # speed now set by OTG caps
    above = np.asarray(above_pos, dtype=np.float64).reshape(3)
    ori = np.asarray(above_ori, dtype=np.float64).reshape(3, 3)
    lower = above - np.array([0.0, 0.0, float(down_dz_m)], dtype=np.float64)
    fixed_z = float(lower[2])

    if descend_from_hover:
        arm.move_to(
            ctx,
            lower,
            ori,
            label=(
                f"[scramble] descend {float(down_dz_m) * 100:.1f} cm to "
                f"work depth {lower.tolist()}"
            ),
        )

    # --- Cardinal triangle phase ---
    timed = scramble_duration_s is not None and scramble_duration_s > 0.0
    if until_enter or timed or triangle_passes > 0:
        if until_enter:
            step_gate(
                ctx,
                f"[scramble] scramble until ENTER, radius={radius_m} m "
                f"(OTG waypoints)",
            )
            print(
                "[scramble] >>> Press ENTER to stop scrambling "
                "(the current pass will finish first) <<<",
                flush=True,
            )
            stop = _enter_watcher()
            t0 = time.monotonic()
            p = 0
            while True:
                p += 1
                print(
                    f"[scramble] triangle pass {p} "
                    f"(elapsed {time.monotonic() - t0:.1f} s, ENTER to stop)"
                )
                _trace_cardinal_triangles(
                    ctx,
                    lower,
                    ori,
                    radius_m=radius_m,
                    waypoint_tol_m=waypoint_tol_m,
                    waypoint_timeout_s=waypoint_timeout_s,
                )
                if stop.is_set():
                    break
            print(
                f"[scramble] ENTER pressed — stopped after {p} pass(es) in "
                f"{time.monotonic() - t0:.1f} s"
            )
        elif timed:
            step_gate(
                ctx,
                f"[scramble] repeat triangle set for {scramble_duration_s:.0f} s, "
                f"radius={radius_m} m (OTG waypoints)",
            )
            t0 = time.monotonic()
            p = 0
            while True:
                p += 1
                elapsed = time.monotonic() - t0
                print(
                    f"[scramble] triangle pass {p} "
                    f"(elapsed {elapsed:.1f}/{scramble_duration_s:.0f} s)"
                )
                _trace_cardinal_triangles(
                    ctx,
                    lower,
                    ori,
                    radius_m=radius_m,
                    waypoint_tol_m=waypoint_tol_m,
                    waypoint_timeout_s=waypoint_timeout_s,
                )
                if time.monotonic() - t0 >= scramble_duration_s:
                    break
            print(
                f"[scramble] triangle timer done: {p} pass(es) in "
                f"{time.monotonic() - t0:.1f} s"
            )
        else:
            step_gate(
                ctx,
                f"[scramble] {triangle_passes} triangle pass(es), "
                f"radius={radius_m} m (OTG waypoints)",
            )
            for p in range(triangle_passes):
                if triangle_passes > 1:
                    print(f"[scramble] triangle pass {p + 1}/{triangle_passes}")
                _trace_cardinal_triangles(
                    ctx,
                    lower,
                    ori,
                    radius_m=radius_m,
                    waypoint_tol_m=waypoint_tol_m,
                    waypoint_timeout_s=waypoint_timeout_s,
                )
        # Settle at the pan center before anything pulls the ladle out, so the
        # lift starts from the middle of the pan (proper convergence here, not
        # the continuous waypoint follow used during the strokes).
        arm.move_to(
            ctx,
            lower,
            ori,
            label=f"[scramble] return to pan center {lower.tolist()}",
            tol_m=waypoint_tol_m,
        )

    # --- Circular phase (discrete OTG waypoints around the rim) ---
    if cycles > 0:
        seg = max(int(circle_segments), 3)
        stir_start = lower.copy()
        stir_start[0] += radius_m
        stir_start[2] = fixed_z
        arm.move_to(
            ctx,
            stir_start,
            ori,
            label=f"[scramble] move to circle start (theta=0) {stir_start.tolist()}",
        )
        step_gate(
            ctx,
            f"[scramble] {cycles} circle(s) radius={radius_m} m, "
            f"{seg} waypoints/rev (OTG)",
        )

        for c in range(cycles):
            # Step around the circle in ``seg`` chords; k starts at 1 because the
            # arm is already at theta=0 (stir_start, or the close of the prior rev).
            for k in range(1, seg + 1):
                theta = 2.0 * math.pi * float(k) / float(seg)
                pos = np.array(
                    [
                        lower[0] + radius_m * math.cos(theta),
                        lower[1] + radius_m * math.sin(theta),
                        fixed_z,
                    ],
                    dtype=np.float64,
                )
                _follow_waypoint(
                    ctx,
                    pos,
                    ori,
                    tol_m=waypoint_tol_m,
                    timeout_s=waypoint_timeout_s,
                    label=f"[scramble] circle {c + 1}/{cycles} wp {k}/{seg}",
                )
            print(f"[scramble] circle {c + 1}/{cycles} complete")

    if lift_after:
        arm.move_to(
            ctx,
            above,
            ori,
            label=f"[scramble] lift to hover {above.tolist()}",
        )
    print("[scramble] phase complete")
