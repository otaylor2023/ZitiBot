"""TidyBot base navigation wrapper."""

from __future__ import annotations

import math
import time
from dataclasses import replace

from zitibot_core.constants import BASE_WAYPOINTS, BaseWaypoint
from zitibot_core.context import TaskContext
from tidybot_base.mocap import read_mocap_pose
from tidybot_base.opti_nav import NavConfig, NavResult, navigate_to_opti_pose
from tidybot_base.redis_io import stop_base, write_max_vel_scale


_INCH_M = 0.0254
DEFAULT_BASE_TOLERANCE_M = 1 * _INCH_M
# Tolerance for the final Phase C translate of the three-phase landing.
# Kept at the original 3-inch value (looser than the new 1-inch
# ``DEFAULT_BASE_TOLERANCE_M``) because Phase C is the precision
# landing — tightening it here was causing the base to stall on the
# last few cm trying to hit a 1-inch box. Holonomic single-phase
# moves still get the tighter 1-inch tolerance from
# ``DEFAULT_BASE_TOLERANCE_M``.
THREE_PHASE_BASE_TOLERANCE_M = 3 * _INCH_M
# Distance at which Phase A (holonomic approach) hands off to Phase B
# (rotate in place). The base then snaps yaw before doing the final
# pure-translation in Phase C — this is what nails the last few cm/deg
# that holonomic alone tends to stall on.
APPROACH_HANDOFF_M = 0.25
# Per-mode multiplier on the base driver's OTG max_velocity. The driver
# polls ``hb1::max_vel_scale`` each tick and rescales its CLI baseline
# (see ``tidybot_base.redis_driver``). 1.5 = 50% faster than baseline
# for the (default) holonomic path, 1.0 = baseline for the three-phase
# path that needs precise landing. Driver clamps to [0.1, 3.0]; if you
# want to push higher you have to also bump that clamp.
HOLONOMIC_BASE_SPEED_SCALE = 1.5
THREE_PHASE_BASE_SPEED_SCALE = 1.0
# Phase-transition deceleration controls. After each phase we send a hard
# stop and then actively wait for the redis_driver to confirm the base is
# at rest before starting the next phase:
#   1. Send ``stop_base`` (writes ``hb1::stop = 'stop'``).
#   2. Hold for at least BASE_PHASE_MIN_DWELL_S so the driver definitely
#      reads the flag at least once.
#   3. Poll the stop flag — the driver flips it back to ``'ok'`` once
#      ``||vehicle.dx|| < 0.001`` (active brake achieved), which is the
#      canonical "wheels at rest" signal. Times out at
#      BASE_STOP_TIMEOUT_S, in which case we warn and continue.
BASE_PHASE_MIN_DWELL_S = 0.4
BASE_STOP_TIMEOUT_S = 2.5
BASE_STOP_POLL_HZ = 50.0


def _default_nav_config() -> NavConfig:
    """Locked-in defaults for downstream controllers using ``base.go_to_pose``.

    - 1-inch XY tolerance for holonomic single-phase moves (matches
      ``opti_nav``'s own default — was 3 in, tightened so grasps land
      closer to the taught station pose). The three-phase landing's
      Phase C overrides this with the looser ``THREE_PHASE_BASE_TOLERANCE_M``
      (3 in) inside :func:`go_to_pose` because the precision-translate
      phase couldn't reliably hit a 1-inch box on the bench.
    - Holonomic direct motion is used inside Phase A of the phased approach
      (see :func:`go_to_pose`); ``direct_motion`` here is just the per-call
      seed and gets overridden per phase.
    """
    return NavConfig(
        exit_on_success=True,
        print_plan=False,
        log_hz=2.0,
        tolerance_m=DEFAULT_BASE_TOLERANCE_M,
        direct_motion=True,
    )


def _read_stop_flag(ctx: TaskContext, cfg: NavConfig) -> str | None:
    """Read ``hb1::stop`` as a normalized lowercase string, or None on error.

    The driver toggles this key between ``'stop'`` (caller requested halt;
    actively braking with ``set_target_velocity(zeros)``) and ``'ok'``
    (free to track ``desired_pose``). It self-clears ``stop`` → ``ok`` when
    the internal vehicle speed drops below ~1 mm/s, which is the canonical
    "at rest" signal for outside observers.
    """
    try:
        raw = ctx.redis.get(cfg.base_keys.stop)
    except Exception:  # noqa: BLE001 - Redis hiccups shouldn't kill stop wait
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return raw.strip().lower()


def _wait_for_base_at_rest(
    ctx: TaskContext,
    *,
    cfg: NavConfig,
    timeout_s: float = BASE_STOP_TIMEOUT_S,
) -> bool:
    """Block until ``redis_driver`` confirms the base is actually stopped.

    Mechanism: ``stop_base`` sets ``hb1::stop = 'stop'``. The driver then
    commands zero target velocity each tick and re-anchors ``desired_pose``
    to the current pose; once ``||vehicle.dx|| < 0.001`` it writes
    ``hb1::stop = 'ok'``. We poll for that ``'ok'`` transition — that is
    the real "wheels at rest" signal. Velocity / pose Redis keys are not
    refreshed while ``stop = 'stop'``, so polling them is unreliable.
    """
    poll_dt = 1.0 / max(BASE_STOP_POLL_HZ, 1.0)
    deadline = time.perf_counter() + max(timeout_s, 0.0)
    while time.perf_counter() < deadline:
        flag = _read_stop_flag(ctx, cfg)
        if flag == "ok":
            return True
        time.sleep(poll_dt)
    return False


def _stop_and_dwell(ctx: TaskContext, reason: str, *, cfg: NavConfig) -> None:
    """Actively brake the base and block until the driver confirms rest.

    The driver's stop branch issues a zero-velocity command every tick,
    which is the actual active brake. We re-issue ``stop_base`` after a
    short min-dwell (in case the driver's first poll missed the flag),
    then wait for the driver to self-clear the flag to ``'ok'``.
    """
    stop_base(ctx.redis, keys=cfg.base_keys)
    print(
        f"[base.go_to_pose] STOP ({reason}) — actively braking, waiting for "
        f"driver to confirm at-rest (timeout {BASE_STOP_TIMEOUT_S:.1f}s)",
        flush=True,
    )
    time.sleep(BASE_PHASE_MIN_DWELL_S)
    # Re-issue in case the driver's poll missed the very first set.
    stop_base(ctx.redis, keys=cfg.base_keys)
    at_rest = _wait_for_base_at_rest(ctx, cfg=cfg)
    if at_rest:
        print("[base.go_to_pose] driver confirms base at rest.", flush=True)
    else:
        print(
            f"[base.go_to_pose] WARNING: driver did not confirm rest in "
            f"{BASE_STOP_TIMEOUT_S:.1f}s (stop flag still 'stop'); "
            f"continuing anyway.",
            flush=True,
        )


def go_to_pose(
    ctx: TaskContext,
    target: BaseWaypoint | None = None,
    *,
    x_m: float | None = None,
    y_m: float | None = None,
    yaw_deg: float | None = None,
    config: NavConfig | None = None,
    label: str | None = None,
    motion: str = "holonomic",
    hold_yaw: bool = False,
) -> NavResult:
    """Navigate the base to either a :class:`BaseWaypoint` preset or a raw pose.

    Usage::

        base.go_to_pose(ctx, BaseWaypoint.INGREDIENT_STATION)
        base.go_to_pose(ctx, x_m=0.33, y_m=0.52, yaw_deg=0.0)
        # Three-phase precise landing (A: holonomic approach, B: rotate
        # in place, C: pure translation) for tasks that need a tight final
        # alignment, e.g. inserting the pan into the oven:
        base.go_to_pose(ctx, BaseWaypoint.OVEN_DOOR, motion="three_phase")

    ``motion`` selects the planner shape:

    * ``"holonomic"`` *(default)* — one call to ``navigate_to_opti_pose``
      with ``direct_motion=True``, driving straight to ``(x, y, yaw)``
      and exiting once XY and yaw are within ``config.tolerance_m`` /
      ``config.tolerance_yaw_rad``. Faster, smoother, but the final
      pose can be a couple of cm / degrees off when holonomic motion
      stalls on the very last correction.
    * ``"three_phase"`` — the legacy A/B/C sequence (holonomic approach
      to ``APPROACH_HANDOFF_M`` of the goal, rotate in place to the
      target yaw, then pure translation with throttled replan). Slower
      but lands precisely. Use this when downstream motion depends on
      tight base alignment (oven insertion, etc.).
    """
    if isinstance(target, BaseWaypoint):
        wp = BASE_WAYPOINTS[target]
        x_m, y_m, yaw_deg = wp.x_m, wp.y_m, wp.yaw_deg
        default_label = (
            f"[base.go_to_pose] {target.name} -> "
            f"({x_m:.3f}, {y_m:.3f}, {yaw_deg:.1f}\u00b0)"
        )
    elif target is None:
        if x_m is None or y_m is None or yaw_deg is None:
            raise ValueError(
                "go_to_pose: pass either a BaseWaypoint as `target`, "
                "or all of x_m / y_m / yaw_deg keywords."
            )
        default_label = (
            f"[base.go_to_pose] ({x_m:.3f}, {y_m:.3f}, {yaw_deg:.1f}\u00b0)"
        )
    else:
        raise TypeError(
            f"go_to_pose: target must be BaseWaypoint or None, "
            f"got {type(target).__name__}"
        )

    cfg = config or _default_nav_config()
    move_label = label if label is not None else default_label
    print(move_label, flush=True)

    final_x = float(x_m)
    final_y = float(y_m)
    final_yaw = float(yaw_deg)
    final_yaw_rad = math.radians(final_yaw)
    who = target.name if isinstance(target, BaseWaypoint) else (
        f"({final_x}, {final_y}, {final_yaw})"
    )
    if ctx.step:
        from zitibot_core.runner import step_gate

        step_gate(ctx, f"{move_label} (base move)")

    # Optional --log telemetry: same logger that records arm moves; here
    # we wrap each phase in begin_base_move / end_base_move and feed the
    # navigator's per-tick state through ``on_sample`` to plot hb + Opti
    # trajectories alongside the goal.
    logger = ctx.move_logger
    base_label_root = (
        target.name if isinstance(target, BaseWaypoint) else f"{final_x:.2f}_{final_y:.2f}"
    )

    def _begin_phase(phase_label: str, phase_cfg: NavConfig):
        """Open a logger move for this phase (no-op if --log is off).

        Returns the per-tick callback to pass as ``on_sample``, or None
        when logging is disabled (so the navigator skips the hook).
        """
        if logger is None:
            return None
        logger.begin_base_move(
            label=f"{base_label_root}_{phase_label}",
            opti_target_xy=(final_x, final_y),
            opti_target_yaw_rad=final_yaw_rad,
            tol_m=phase_cfg.tolerance_m,
            tol_yaw_rad=phase_cfg.tolerance_yaw_rad,
            require_yaw=(phase_cfg.tolerance_yaw_rad < math.pi),
        )
        return logger.sample_base

    def _end_phase(res: NavResult) -> None:
        if logger is None:
            return
        yaw_err_rad = (
            None
            if res.final_body_yaw_error_deg is None
            else math.radians(float(res.final_body_yaw_error_deg))
        )
        logger.end_base_move(
            status="ok" if res.success else (res.reason or "fail"),
            final_xy_err_m=res.final_xy_error_m,
            final_yaw_err_rad=yaw_err_rad,
        )

    def _check_phase_result(res: NavResult, phase: str) -> None:
        """Translate a NavResult into a phase outcome.

        Mirrors the way ``arm.move_to`` handles a convergence timeout:

        * ``reason == "stalled"`` is non-fatal — print final pose and
          error, log it on the move plot, and let the next phase try.
        * ``reason == "interrupted"`` is propagated as ``KeyboardInterrupt``
          so the controller's outer ``except`` handles it cleanly.
        * Anything else raises ``RuntimeError`` so the controller can
          fail loudly (these are real navigation failures).
        """
        if res.success:
            return
        if res.reason == "interrupted":
            raise KeyboardInterrupt("base navigation interrupted")
        if res.reason == "stalled":
            xy_err_cm = (
                f"{res.final_xy_error_m * 100:.2f} cm"
                if res.final_xy_error_m is not None
                else "n/a"
            )
            yaw_err_deg = (
                f"{res.final_body_yaw_error_deg:.2f} deg"
                if res.final_body_yaw_error_deg is not None
                else "n/a"
            )
            print(
                f"[base.go_to_pose] {phase} stalled — continuing. "
                f"final_xy_err={xy_err_cm}  final_body_yaw_err={yaw_err_deg}",
                flush=True,
            )
            return
        raise RuntimeError(
            f"base.go_to_pose({who}) {phase} failed: {res.reason}"
        )

    if motion == "holonomic":
        # ``hold_yaw``: straight-line XY only — keep startup heading, do not
        # command the saved station yaw. Use for counter slides where the
        # cart is already roughly aligned; holonomic XY+yaw together spins
        # while translating and often overshoots.
        write_max_vel_scale(
            ctx.redis, HOLONOMIC_BASE_SPEED_SCALE, keys=cfg.base_keys
        )
        holo_cfg = replace(
            cfg,
            direct_motion=not hold_yaw,
            rotate_first=False,
            exit_on_success=True,
            print_plan=False,
        )
        if hold_yaw:
            print(
                f"[base.go_to_pose] straight-line drive to "
                f"({final_x:.3f}, {final_y:.3f}), holding startup yaw "
                f"(max_vel scale {HOLONOMIC_BASE_SPEED_SCALE:.2f}x baseline)",
                flush=True,
            )
        else:
            print(
                f"[base.go_to_pose] holonomic drive to "
                f"({final_x:.3f}, {final_y:.3f}) yaw {final_yaw:.1f}\u00b0 "
                f"(max_vel scale {HOLONOMIC_BASE_SPEED_SCALE:.2f}x baseline)",
                flush=True,
            )
        sampler = _begin_phase("holonomic", holo_cfg)
        res = navigate_to_opti_pose(
            ctx.redis,
            target_opti_xy=(final_x, final_y),
            target_body_yaw_deg=None if hold_yaw else final_yaw,
            config=holo_cfg,
            on_sample=sampler,
        )
        _end_phase(res)
        _check_phase_result(res, "holonomic")
        return res

    if motion != "three_phase":
        raise ValueError(
            f"go_to_pose: motion must be 'holonomic' or 'three_phase', "
            f"got {motion!r}"
        )

    # Three-phase landing runs at the driver's CLI baseline speed —
    # the precise final-translate phase is calibrated against that
    # speed and a faster scale would overshoot the final tolerance.
    write_max_vel_scale(
        ctx.redis, THREE_PHASE_BASE_SPEED_SCALE, keys=cfg.base_keys
    )

    # ------------------------------------------------------------------
    # Phase A: straight-line XY approach at the *startup* heading.
    # Exit as soon as XY is within APPROACH_HANDOFF_M of the goal;
    # yaw is intentionally ignored here (Phase B snaps heading). Passing
    # ``target_body_yaw_deg=None`` holds the current heading so the cart
    # does not holonomically spin while translating — that combined
    # motion corrupts motion-direction calib and fights Phase B.
    # ------------------------------------------------------------------
    phase_a_cfg = replace(
        cfg,
        direct_motion=False,
        rotate_first=False,
        tolerance_m=APPROACH_HANDOFF_M,
        tolerance_yaw_rad=math.pi,
        exit_on_success=True,
        print_plan=False,
    )
    print(
        f"[base.go_to_pose] Phase A: straight-line approach "
        f"(hold startup yaw, exit when XY < {APPROACH_HANDOFF_M * 100:.0f} cm)",
        flush=True,
    )
    sampler_a = _begin_phase("phase_a_holonomic", phase_a_cfg)
    res_a = navigate_to_opti_pose(
        ctx.redis,
        target_opti_xy=(final_x, final_y),
        target_body_yaw_deg=None,
        config=phase_a_cfg,
        on_sample=sampler_a,
    )
    _end_phase(res_a)
    _check_phase_result(res_a, "Phase A (holonomic)")
    _stop_and_dwell(ctx, "after Phase A (holonomic)", cfg=cfg)

    # ------------------------------------------------------------------
    # Phase B: rotate in place at the current Opti XY. We re-target the
    # cart's current location so the planner only has yaw to close. XY
    # tolerance is loosened to APPROACH_HANDOFF_M so small wheel-pivot
    # drift doesn't gate completion.
    # ------------------------------------------------------------------
    cur_pos, _ = read_mocap_pose(ctx.redis, keys=cfg.mocap_keys)
    cur_opti_x = float(cur_pos[0])
    cur_opti_y = float(cur_pos[1])
    phase_b_cfg = replace(
        cfg,
        direct_motion=True,
        rotate_first=False,
        use_motion_direction_calib=False,
        tolerance_m=max(cfg.tolerance_m, APPROACH_HANDOFF_M),
        exit_on_success=True,
        print_plan=False,
    )
    print(
        f"[base.go_to_pose] Phase B: rotate in place at "
        f"({cur_opti_x:.3f}, {cur_opti_y:.3f}) -> yaw {final_yaw:.1f}\u00b0",
        flush=True,
    )
    sampler_b = _begin_phase("phase_b_rotate", phase_b_cfg)
    res_b = navigate_to_opti_pose(
        ctx.redis,
        target_opti_xy=(cur_opti_x, cur_opti_y),
        target_body_yaw_deg=final_yaw,
        config=phase_b_cfg,
        wait_for_tracking=False,
        on_sample=sampler_b,
    )
    _end_phase(res_b)
    _check_phase_result(res_b, "Phase B (rotate)")
    _stop_and_dwell(ctx, "after Phase B (rotate)", cfg=cfg)

    # ------------------------------------------------------------------
    # Phase C: pure translation to the final XY at the now-correct yaw.
    # Use live Opti replanning so success is judged by Opti error (not hb
    # odom drift), but throttle replan_hz well below control_hz so the
    # base has time to actually converge to a goal between recomputes.
    # At full 100 Hz replan the moving target makes the base orbit the
    # final correction; at 2 Hz it makes ~5 corrections before landing.
    # ------------------------------------------------------------------
    phase_c_cfg = replace(
        cfg,
        direct_motion=False,
        replan=True,
        replan_hz=2.0,
        rotate_first=False,
        # Phase C uses its own (looser) landing tolerance — see
        # ``THREE_PHASE_BASE_TOLERANCE_M``. Tightening the global
        # default to 1 inch was making this final phase stall
        # because the planner can't reliably hit a 1-inch box on
        # the real bench; revert specifically here.
        tolerance_m=THREE_PHASE_BASE_TOLERANCE_M,
        exit_on_success=True,
        print_plan=False,
    )
    print(
        f"[base.go_to_pose] Phase C: translate to "
        f"({final_x:.3f}, {final_y:.3f}) at yaw {final_yaw:.1f}\u00b0",
        flush=True,
    )
    sampler_c = _begin_phase("phase_c_translate", phase_c_cfg)
    res_c = navigate_to_opti_pose(
        ctx.redis,
        target_opti_xy=(final_x, final_y),
        target_body_yaw_deg=final_yaw,
        config=phase_c_cfg,
        wait_for_tracking=False,
        on_sample=sampler_c,
    )
    _end_phase(res_c)
    _check_phase_result(res_c, "Phase C (translate)")
    return res_c


def go_to(
    ctx: TaskContext,
    waypoint: BaseWaypoint,
    *,
    config: NavConfig | None = None,
) -> NavResult:
    """Back-compat alias for ``go_to_pose(ctx, waypoint, config=...)``."""
    return go_to_pose(ctx, waypoint, config=config)
