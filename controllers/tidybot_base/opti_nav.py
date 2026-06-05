"""High-level OptiTrack → TidyBot base navigator (reusable across controllers).

This is the engine that ``opti_controller.py`` (CLI) drives, and the same
engine that downstream controllers like ``grasp_and_pour_controller.py``
should call when they need the base to land at a specific Motive lab pose.

Two entry points:

- :func:`run_replan_loop` — drive an already-built :class:`RunPlan` to
  completion. Use this from the CLI where ``setup_run_plan`` is called
  explicitly with all the relative-goal / face-axis machinery.

- :func:`navigate_to_opti_pose` — one call: wait for tracking, snapshot
  calibration, build an absolute-target plan, and run the loop. Intended
  for downstream controllers::

      from tidybot_base.opti_nav import NavConfig, navigate_to_opti_pose

      result = navigate_to_opti_pose(
          redis_client,
          target_opti_xy=(-1.5, 1.0),
          target_body_yaw_deg=90.0,
          config=NavConfig(log_hz=2.0, print_plan=False),
      )
      if not result.success:
          ...  # nav failed, decide what to do

Both honor the locked-in marker offset
(:data:`~tidybot_base.opti_planner.DEFAULT_MARKER_YAW_OFFSET_DEG`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import redis

from tidybot_base.mocap import (
    DEFAULT_MOCAP_KEYS,
    MocapRedisKeys,
    read_mocap_pose,
    read_tracking_valid,
    wait_for_tracking_valid,
)
from tidybot_base.opti_planner import (
    DEFAULT_MARKER_YAW_OFFSET_DEG,
    GoalFrame,
    MotionDirectionEstimator,
    RunPlan,
    mocap_pose_error,
    mocap_pose_to_hb_se2,
    opti_body_yaw_error_from_motion_calib_rad,
    opti_body_yaw_error_rad,
    opti_xy_distance_m,
    replan_hb_goal_from_opti,
    setup_run_plan,
    waypoint_reached,
)
from tidybot_base.redis_io import (
    DEFAULT_BASE_KEYS,
    BaseRedisKeys,
    read_robot_se2,
    stop_base,
    write_desired_pose,
)
from tidybot_base.se2 import (
    hb1_tracking_error,
    quat_xyzw_to_yaw,
    rot2d_yaw,
    wrap_angle,
)

DEFAULT_TOLERANCE_M = 0.0254  # 1 inch
DEFAULT_YAW_TOLERANCE_RAD = math.radians(5.0)
DEFAULT_CONTROL_HZ = 100.0
DEFAULT_LOG_HZ = 10.0
DEFAULT_ODOM_JUMP_M = 0.5
DEFAULT_TARGET_OPTI_Z = 0.45
# When mocap is bad mid-move (``tracking_valid`` drops) we used to return
# a NavResult failure, which propagated up through
# ``zitibot_core.base.go_to_pose`` as a RuntimeError that killed the
# controller — and the dying controller stopped commanding the Franka
# cartesian goal, which relaxed the arm and dropped whatever it was
# holding. Instead we now stop the base in place, leave the arm goal
# untouched (the Franka controller will keep tracking the last commanded
# pose for the entire wait), and poll for ``tracking_valid`` to recover
# before resuming the replan loop.
#
# Note: there used to be a second "publisher latched at a stale pose"
# guard here (``MOCAP_STALE_TIMEOUT_S`` / ``MOCAP_STALE_TOL_M`` /
# ``HB_MOVING_TOL_M``) that would also wait when the Opti pose hadn't
# changed for >1.5 s while the hb encoders kept rolling, on the theory
# that the camera had lost the marker but the publisher was still
# latching the last seen pose. That check was removed — it fires too
# eagerly when the body legitimately sits still and is no longer
# needed now that the streamer drops ``tracking_valid`` when Motive
# reports ``Tracked=0``. Only the ``tracking_valid`` guard remains.
MOCAP_RECOVERY_POLL_HZ = 5.0
MOCAP_RECOVERY_STATUS_PERIOD_S = 2.0


@dataclass
class NavConfig:
    """Per-call behavior for :func:`run_replan_loop` / :func:`navigate_to_opti_pose`.

    Defaults match what was empirically dialed in on this robot:

    - marker mounting offset seeds the startup calibration, then motion
      direction can refine the Opti-world → hb-odom rotation online;
    - translate-then-rotate motion by default (set ``direct_motion=True`` for
      holonomic, or ``rotate_first=True`` for rotate-in-place then translate);
    - per-loop replanning from live Opti enabled;
    - 1-inch XY / 5-deg yaw tolerance (override via ``tolerance_m`` /
      ``tolerance_yaw_rad`` per call, e.g. ``zitibot_core.base`` uses 3 in).
    """

    marker_yaw_offset_deg: float = DEFAULT_MARKER_YAW_OFFSET_DEG
    # When enabled, the replan loop estimates the Opti-world → hb-odom
    # direction from the cart's observed motion and uses that live rotation
    # instead of relying on Motive's arbitrary marker +X axis forever. The
    # marker offset remains only as the first few centimeters' seed.
    use_motion_direction_calib: bool = True
    motion_calib_min_motion_m: float = 0.08
    motion_calib_max_yaw_change_rad: float = math.radians(8.0)
    translation_only_calib: bool = False
    calib_yaw_deg: float | None = None

    direct_motion: bool = False
    cardinal_hb: bool = False
    rotate_first: bool = False
    replan: bool = True
    # How often (Hz) to recompute ``hb_goal`` from live Opti when
    # ``replan`` is True. Default equals ``control_hz`` so every tick
    # gets a fresh goal — original behavior. Lower values (e.g. 2.0)
    # cache the goal between recomputes so the base finishes responding
    # to one correction before the next is issued; useful for the
    # short final-correction phase where chasing a moving goal at
    # 100 Hz orbits instead of converging. Ignored when ``replan=False``.
    replan_hz: float = DEFAULT_CONTROL_HZ

    tolerance_m: float = DEFAULT_TOLERANCE_M
    tolerance_yaw_rad: float = DEFAULT_YAW_TOLERANCE_RAD

    control_hz: float = DEFAULT_CONTROL_HZ
    log_hz: float = DEFAULT_LOG_HZ
    odom_jump_m: float = DEFAULT_ODOM_JUMP_M
    curr_minus_desired: bool = False

    monitor: bool = False
    stop_on_exit: bool = True
    print_plan: bool = True
    print_log: bool = True
    # When True, ``run_replan_loop`` returns ``NavResult(success=True)`` the
    # first time the (final-waypoint) pose is within tolerance. When False it
    # keeps republishing the goal so the base does not drift. Standalone CLIs
    # like ``opti_controller`` want False (hold position until Ctrl+C);
    # downstream state machines like ``pour_and_move_controller`` want True.
    exit_on_success: bool = False

    base_keys: BaseRedisKeys = field(default_factory=lambda: DEFAULT_BASE_KEYS)
    mocap_keys: MocapRedisKeys = field(default_factory=lambda: DEFAULT_MOCAP_KEYS)

    # Stall timeout: if the cart's pose (Opti when available, hb fallback)
    # hasn't moved by more than ``stall_motion_eps_m`` /
    # ``stall_motion_eps_rad`` for ``stall_timeout_s`` seconds straight,
    # ``run_replan_loop`` exits with ``reason='stalled'`` and stops the
    # base instead of orbiting forever. ``zitibot_core.base.go_to_pose``
    # treats this non-fatally — it prints final error and continues —
    # mirroring the way ``arm.move_to`` handles a convergence timeout.
    # Set ``stall_timeout_s`` to a non-positive number to disable.
    stall_timeout_s: float = 4.0
    stall_motion_eps_m: float = 0.01           # 1 cm
    stall_motion_eps_rad: float = math.radians(2.0)
    # Divergence guard: catches the *moving in the wrong direction*
    # failure mode that the pose stall guard misses. We track the
    # smallest Opti-XY distance to the target seen so far this run; if
    # the live distance exceeds that minimum by more than
    # ``divergence_eps_m`` for ``divergence_timeout_s`` continuous
    # seconds we *log a warning* (every ``divergence_log_period_s``
    # while it persists) but keep driving. This is the symptom of a
    # calibration mismatch (e.g. wheel slip / wrong marker offset)
    # where each replan tick pushes the goal further from the actual
    # target. Disable with non-positive timeout.
    divergence_timeout_s: float = 4.0
    divergence_eps_m: float = 0.02            # 2 cm
    divergence_log_period_s: float = 2.0      # rate-limit the warnings


@dataclass
class NavResult:
    """Outcome of a navigation call."""

    success: bool
    reason: str  # 'reached' | 'interrupted' | 'tracking_lost' | 'monitor_only'
    elapsed_s: float
    final_robot_pose: np.ndarray  # hb [x, y, yaw]
    final_opti_xyz: np.ndarray | None = None
    final_marker_yaw_rad: float | None = None
    final_body_yaw_rad: float | None = None
    final_xy_error_m: float | None = None
    final_body_yaw_error_deg: float | None = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


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


def _empirical_marker_offset_deg(
    plan: RunPlan,
    robot_current: np.ndarray,
    mocap_xyz: np.ndarray,
    *,
    min_motion_m: float = 0.10,
    max_hb_yaw_change_rad: float = math.radians(2.0),
) -> float | None:
    """Compare hb motion to Opti motion to back out the body yaw in Opti world.

    Returns ``mocap_yaw_start - observed_body_yaw_in_opti`` (deg), which is the
    marker mounting offset (CCW positive). Skips estimation once the body has
    rotated more than ``max_hb_yaw_change_rad`` from its startup yaw — during
    pivot-in-place the marker can slide a few mm from wheel slip, biasing the
    estimate. Returns ``None`` when motion is too small or the body is no
    longer at its startup heading.
    """
    if (
        abs(wrap_angle(float(robot_current[2]) - float(plan.robot_start[2])))
        > max_hb_yaw_change_rad
    ):
        return None
    hb_dx = float(robot_current[0]) - float(plan.robot_start[0])
    hb_dy = float(robot_current[1]) - float(plan.robot_start[1])
    op_dx = float(mocap_xyz[0]) - float(plan.mocap_start_xyz[0])
    op_dy = float(mocap_xyz[1]) - float(plan.mocap_start_xyz[1])
    if math.hypot(hb_dx, hb_dy) < min_motion_m or math.hypot(op_dx, op_dy) < min_motion_m:
        return None
    body_yaw_obs = math.atan2(op_dy, op_dx) - math.atan2(hb_dy, hb_dx)
    offset = wrap_angle(plan.mocap_start_yaw - body_yaw_obs)
    return math.degrees(offset)


def _wait_for_mocap_alive(
    client: redis.Redis,
    *,
    cfg: NavConfig,
    reason: str,
) -> None:
    """Stop the base in place and block until ``tracking_valid`` is true again.

    The arm's last commanded EE pose stays published — the Franka cartesian
    controller keeps tracking it for the entire wait, so the gripper holds
    whatever it was holding. Ctrl+C still works to abort if the user wants
    to bail out (it bubbles up to the surrounding try/except).
    """
    if not cfg.monitor:
        stop_base(client, keys=cfg.base_keys)
    period = 1.0 / max(MOCAP_RECOVERY_POLL_HZ, 0.1)
    started_wait = time.perf_counter()
    print(
        f"[base nav] {reason} — base stopped, arm holding last EE goal; "
        f"waiting for mocap to recover (Ctrl+C to abort).",
        flush=True,
    )
    last_msg = started_wait
    while True:
        valid = read_tracking_valid(client, keys=cfg.mocap_keys)
        if valid:
            elapsed = time.perf_counter() - started_wait
            print(
                f"[base nav] mocap recovered after {elapsed:.1f}s; resuming.",
                flush=True,
            )
            # Re-clear the stop flag so the next ``write_desired_pose``
            # actually drives the wheels (the wait may have outlasted the
            # driver's self-clear).
            if not cfg.monitor:
                client.set(cfg.base_keys.stop, "ok")
            return
        now = time.perf_counter()
        if now - last_msg >= MOCAP_RECOVERY_STATUS_PERIOD_S:
            raw = client.get(cfg.mocap_keys.tracking_valid)
            print(
                f"[base nav]   still waiting "
                f"({cfg.mocap_keys.tracking_valid}={raw!r})",
                flush=True,
            )
            last_msg = now
        time.sleep(period)


def print_pose_log_block(
    *,
    plan: RunPlan,
    robot_current: np.ndarray,
    mocap_xyz: np.ndarray | None,
    mocap_quat: np.ndarray | None,
    curr_minus_desired: bool,
    hb_goal: np.ndarray | None = None,
    opti_to_hb_rot: np.ndarray | None = None,
    motion_calib_active: bool = False,
) -> None:
    """Print hb then opti poses (start, current, goal, error — one field per line)."""
    goal = hb_goal if hb_goal is not None else plan.robot_target
    hb_err = hb1_tracking_error(robot_current, goal)

    print(f"hb_start={_fmt_array(plan.robot_start)}")
    print(f"hb_current={_fmt_array(robot_current)}")
    print(f"hb_goal={_fmt_array(goal)}")
    print(f"hb_final={_fmt_array(plan.robot_target)}")
    print(f"hb_error={_fmt_array(hb_err)}")

    body_yaw_label = "body" if plan.marker_yaw_offset_deg else "body=marker"
    print(
        f"opti_start  marker {_fmt_opti_pose(plan.mocap_start_xyz, plan.mocap_start_yaw)}  "
        f"{body_yaw_label}_yaw_deg={math.degrees(plan.body_start_yaw_in_opti):.2f}"
    )
    if mocap_xyz is not None and mocap_quat is not None:
        mocap_yaw = quat_xyzw_to_yaw(mocap_quat)
        if motion_calib_active and opti_to_hb_rot is not None:
            body_yaw_cur = wrap_angle(
                float(robot_current[2]) - rot2d_yaw(opti_to_hb_rot)
            )
            body_yaw_source = "motion-calib"
        else:
            body_yaw_cur = wrap_angle(
                mocap_yaw - math.radians(plan.marker_yaw_offset_deg)
            )
            body_yaw_source = "marker-offset"
        opti_err = mocap_pose_error(
            mocap_xyz,
            mocap_yaw,
            plan.desired_mocap_xyz,
            plan.desired_mocap_yaw,
            curr_minus_desired=curr_minus_desired,
        )
        body_yaw_err_deg = math.degrees(
            wrap_angle(plan.desired_body_yaw_in_opti - body_yaw_cur)
            if not curr_minus_desired
            else wrap_angle(body_yaw_cur - plan.desired_body_yaw_in_opti)
        )
        print(
            f"opti_current marker {_fmt_opti_pose(mocap_xyz, mocap_yaw)}  "
            f"{body_yaw_label}_yaw_deg={math.degrees(body_yaw_cur):.2f} "
            f"({body_yaw_source})"
        )
        print(
            f"opti_target  marker {_fmt_opti_pose(plan.desired_mocap_xyz, plan.desired_mocap_yaw)}  "
            f"{body_yaw_label}_yaw_deg={math.degrees(plan.desired_body_yaw_in_opti):.2f}"
        )
        print(
            f"opti_error {_fmt_opti_error(opti_err)}  "
            f"d{body_yaw_label}_yaw_deg={body_yaw_err_deg:+.2f}"
        )
        est = _empirical_marker_offset_deg(plan, robot_current, mocap_xyz)
        if est is None:
            hb_yaw_change_deg = math.degrees(
                wrap_angle(float(robot_current[2]) - float(plan.robot_start[2]))
            )
            if abs(hb_yaw_change_deg) > 2.0:
                reason = (
                    f"hb yaw changed {hb_yaw_change_deg:+.1f} deg "
                    f"(pivot phase — biased, skipping)"
                )
            else:
                reason = "need more straight-line motion"
            print(
                f"marker_offset_est_deg = ({reason})  "
                f"using {plan.marker_yaw_offset_deg:+.2f} deg"
            )
        else:
            correction = wrap_angle(
                math.radians(est - plan.marker_yaw_offset_deg)
            )
            print(
                f"marker_offset_est_deg = {est:+.2f}  "
                f"using {plan.marker_yaw_offset_deg:+.2f}  "
                f"(rerun with --marker-yaw-offset-deg {est:+.1f} to fix; "
                f"residual {math.degrees(correction):+.1f} deg)"
            )
        if motion_calib_active and opti_to_hb_rot is not None:
            print(
                f"motion_calib lab_to_hb_yaw_deg="
                f"{math.degrees(rot2d_yaw(opti_to_hb_rot)):+.2f}  "
                f"(using observed Opti/hb displacement)"
            )
    else:
        print("opti_current (unavailable)")
        print(
            f"opti_target  marker {_fmt_opti_pose(plan.desired_mocap_xyz, plan.desired_mocap_yaw)}  "
            f"{body_yaw_label}_yaw_deg={math.degrees(plan.desired_body_yaw_in_opti):.2f}"
        )
        print("opti_error (unavailable)")


def print_plan_summary(
    plan: RunPlan,
    *,
    config: NavConfig | None = None,
    along_note: str | None = None,
) -> None:
    """Print the start-of-run plan block.

    ``along_note`` is a short single-line description of how the goal was
    specified (e.g. ``"  along=lab-plus-y  distance=1.50 ft"``). Omit it
    for absolute-target callers; the absolute target line is printed
    automatically.
    """
    cfg = config or NavConfig()
    base_keys = cfg.base_keys
    mocap_keys = cfg.mocap_keys

    hb_at_mocap_start = mocap_pose_to_hb_se2(
        plan.mocap_start_xyz, plan.calib, plan.robot_start[2]
    )
    calib_err = float(np.linalg.norm(hb_at_mocap_start - plan.robot_start))
    dyaw_calib = rot2d_yaw(plan.calib.rot)

    if plan.absolute_target:
        mode = "absolute Opti pose"
    elif not plan.hb_waypoints:
        mode = "rotate only"
    else:
        mode = "translate + face"

    print(f"Plan (Opti ground truth → hb goal, {mode}):")
    print(f"  hb1 start        = {plan.robot_start.tolist()}  ({base_keys.robot_pose})")
    print(
        f"  mocap @ startup  = {plan.mocap_start_xyz.tolist()}  "
        f"Opti heading={math.degrees(plan.mocap_start_yaw):.1f} deg (marker +X in lab)"
    )
    calib_source = (
        "translation only" if cfg.translation_only_calib else "from Opti quat"
    )
    print(
        f"  lab→hb R angle   = {math.degrees(dyaw_calib):.1f} deg  "
        f"t={plan.calib.trans.round(4).tolist()}  ({calib_source})"
    )
    print(
        f"  hb(mocap_start)  = {hb_at_mocap_start.round(4).tolist()}  "
        f"residual={calib_err:.4f} m"
    )

    if plan.absolute_target:
        yaw_note = (
            f"body_yaw={math.degrees(plan.desired_body_yaw_in_opti):.1f} deg "
            f"(marker_yaw={math.degrees(plan.desired_mocap_yaw):.1f} deg, Motive lab)"
            if plan.require_final_yaw
            else "hold startup yaw"
        )
        spec_note = (
            f"  absolute target  = {plan.desired_mocap_xyz.tolist()}  {yaw_note}"
        )
    elif along_note is not None:
        spec_note = along_note
    else:
        spec_note = "  (relative goal — see goal Δ below)"
    print(f"  goal frame       = {plan.goal_frame.value}{spec_note}")

    print(
        f"  goal Δ (input)   = {plan.goal_delta_input_xy.round(4).tolist()} m  "
        f"frame={plan.goal_frame.value}"
    )
    print(
        f"  mocap Δ (goal)   = {plan.mocap_delta_world_xy.round(4).tolist()} m  "
        f"(Opti world target − start)"
    )
    print(
        f"  marker yaw off   = {plan.marker_yaw_offset_deg:+.2f} deg  "
        f"(startup seed for calib.rot)"
    )
    if cfg.use_motion_direction_calib:
        print(
            f"  motion calib     = on after "
            f"{cfg.motion_calib_min_motion_m * 100:.1f} cm translation "
            f"(live Opti/hb direction overrides marker seed)"
        )
    else:
        print("  motion calib     = off (marker seed remains fixed)")
    print(
        f"  body yaw @ start = {math.degrees(plan.body_start_yaw_in_opti):+.2f} deg  "
        f"(mocap yaw − offset; the angle calib.rot inverts)"
    )
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
        if cfg.cardinal_hb:
            path_kind = "cardinal L-path"
        elif cfg.rotate_first:
            path_kind = "rotate then translate"
        else:
            path_kind = "translate then rotate"
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
            f"body_yaw={math.degrees(plan.desired_body_yaw_in_opti):.1f} deg  "
            f"(marker_yaw={math.degrees(plan.desired_mocap_yaw):.1f} deg, "
            f"hb yaw={math.degrees(plan.hb_target_yaw):.1f} deg)"
        )
    elif plan.absolute_target and plan.require_final_yaw:
        print(
            f"  target heading   = body_yaw={math.degrees(plan.desired_body_yaw_in_opti):.1f} deg  "
            f"(marker_yaw={math.degrees(plan.desired_mocap_yaw):.1f} deg, "
            f"hb yaw={math.degrees(plan.hb_target_yaw):.1f} deg)"
        )
    elif plan.absolute_target:
        print(
            f"  orientation      = hold startup (hb yaw {math.degrees(plan.hb_target_yaw):.1f} deg)"
        )

    print(
        f"  hb1 target       = {plan.robot_target.tolist()}  ({base_keys.desired_pose})"
    )
    print(f"  hb waypoints     = {len(plan.hb_waypoints)} step(s)")
    print(
        f"  mocap keys       = {mocap_keys.pos} / {mocap_keys.ori} / "
        f"{mocap_keys.tracking_valid}"
    )
    print(
        f"  success          = |hb1_current_xy - hb1_target_xy| < "
        f"{cfg.tolerance_m:.4f} m ({cfg.tolerance_m / 0.0254:.1f} in)"
    )
    print(
        "  replan           = "
        + (
            "per-loop from live Opti (hb_goal nudged each cycle to close opti error)"
            if cfg.replan
            else "fixed (hb waypoints frozen at startup)"
        )
    )
    if cfg.monitor:
        print("Monitor mode — not commanding base.")
    else:
        print(f"Commanding {base_keys.desired_pose} at {cfg.control_hz:.0f} Hz")


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------


def _sleep_until(loop_start: float, period: float) -> None:
    elapsed = time.perf_counter() - loop_start
    if elapsed < period:
        time.sleep(period - elapsed)


def _build_nav_result(
    *,
    success: bool,
    reason: str,
    started: float,
    plan: RunPlan,
    robot_current: np.ndarray,
    mocap_xyz: np.ndarray | None,
    mocap_quat: np.ndarray | None,
    opti_to_hb_rot: np.ndarray | None = None,
) -> NavResult:
    final_marker_yaw = (
        quat_xyzw_to_yaw(mocap_quat) if mocap_quat is not None else None
    )
    if opti_to_hb_rot is not None:
        final_body_yaw = wrap_angle(
            float(robot_current[2]) - rot2d_yaw(opti_to_hb_rot)
        )
    elif final_marker_yaw is not None:
        final_body_yaw = wrap_angle(
            final_marker_yaw - math.radians(plan.marker_yaw_offset_deg)
        )
    else:
        final_body_yaw = None
    xy_err = (
        opti_xy_distance_m(plan, mocap_xyz) if mocap_xyz is not None else None
    )
    body_yaw_err_deg = (
        math.degrees(
            opti_body_yaw_error_from_motion_calib_rad(
                plan, robot_current, opti_to_hb_rot
            )
            if opti_to_hb_rot is not None
            else opti_body_yaw_error_rad(plan, mocap_quat)
        )
        if (
            plan.require_final_yaw
            and (opti_to_hb_rot is not None or mocap_quat is not None)
        )
        else None
    )
    return NavResult(
        success=success,
        reason=reason,
        elapsed_s=time.perf_counter() - started,
        final_robot_pose=robot_current.copy(),
        final_opti_xyz=None if mocap_xyz is None else mocap_xyz.copy(),
        final_marker_yaw_rad=final_marker_yaw,
        final_body_yaw_rad=final_body_yaw,
        final_xy_error_m=xy_err,
        final_body_yaw_error_deg=body_yaw_err_deg,
    )


def run_replan_loop(
    client: redis.Redis,
    plan: RunPlan,
    *,
    config: NavConfig | None = None,
    on_sample: Callable[..., None] | None = None,
) -> NavResult:
    """Drive ``plan`` to completion using per-loop Opti replanning.

    The plan's ``calib`` and ``hb_waypoints`` are the startup snapshot.
    Every cycle we read live Opti + hb, recompute the hb goal to close
    the current Opti-world error (when ``config.replan`` is true and
    live Opti is available), publish it on ``hb1::desired_pose``, and
    check for tolerance.

    ``on_sample`` (optional) is called once per control tick *after* the
    hb_goal is published. It receives every relevant pose as keyword
    arguments so callers can record telemetry (the ``--log`` move logger
    uses this to plot base moves). Signature::

        on_sample(
            *,
            t_perf: float,                       # time.perf_counter() at tick
            hb_xyyaw: np.ndarray,                # shape (3,), encoder pose
            hb_goal_xyyaw: np.ndarray,           # shape (3,), published goal
            opti_xy: np.ndarray | None,          # shape (2,), live marker XY
            opti_yaw_rad: float | None,          # body yaw in lab frame
            opti_target_xy: np.ndarray,          # shape (2,) lab-frame target
            opti_target_yaw_rad: float | None,   # body-frame target yaw
            require_yaw: bool,                   # whether yaw gates success
            tolerance_m: float,
            tolerance_yaw_rad: float,
        )

    Loop exits on:

    - Tolerance reached → ``NavResult(success=True, reason='reached')``.
    - ``KeyboardInterrupt`` → ``NavResult(success=False, reason='interrupted')``.
    - ``tracking_valid`` goes false → base is stopped and the loop
      waits for the streamer to recover before resuming
      (``NavResult(success=False, reason='tracking_lost')`` only if the
      wait is interrupted).
    - Cart pose hasn't moved by more than ``cfg.stall_motion_eps_m`` /
      ``cfg.stall_motion_eps_rad`` for ``cfg.stall_timeout_s`` seconds →
      ``NavResult(success=False, reason='stalled')``. ``base.go_to_pose``
      treats this like an arm-style timeout: print final error, continue.

    The divergence detector (Opti-XY distance to target growing past its
    best-seen value for ``cfg.divergence_timeout_s``) does *not* exit
    the loop — it only logs a rate-limited warning so the run keeps
    going while still surfacing the problem.
    """
    cfg = config or NavConfig()
    period = 1.0 / max(cfg.control_hz, 1.0)
    log_period = 1.0 / max(cfg.log_hz, 0.1)
    replan_period = 1.0 / max(cfg.replan_hz, 0.1)
    started = time.perf_counter()
    last_log = 0.0
    robot_prev: np.ndarray | None = None
    reached_target = False
    waypoint_idx = 0
    mocap_xyz: np.ndarray | None = None
    mocap_quat: np.ndarray | None = None
    robot_current: np.ndarray = plan.robot_start.copy()
    # Cache the computed hb_goal between Opti-driven recomputes so that
    # when ``replan_hz < control_hz`` we hold a goal steady for the base
    # to actually converge to it, instead of moving the target every
    # tick at 100 Hz.
    cached_hb_goal: np.ndarray | None = None
    cached_used_opti: bool = False
    last_replan_t: float = -float("inf")
    # Stall guard: anchor pose + timestamp. ``cart_xy`` / ``cart_yaw`` are
    # whichever frame is currently freshest (Opti when available, else
    # hb). If the cart pose hasn't moved past the eps thresholds for
    # ``cfg.stall_timeout_s`` we exit with ``reason='stalled'`` so callers
    # can decide whether to retry, give up, or continue. Mirrors the
    # arm-side convergence timeout.
    stall_ref_xy: np.ndarray | None = None
    stall_ref_yaw: float | None = None
    stall_ref_t: float = time.perf_counter()
    # Divergence guard: best (smallest) Opti-XY distance to target seen
    # so far, plus the time at which we first started "diverging" (live
    # distance > best + eps). When divergence persists for more than
    # ``cfg.divergence_timeout_s`` we *log a warning* (rate-limited by
    # ``cfg.divergence_log_period_s``) but keep driving — divergence
    # is informational only, the run continues until the user
    # interrupts or another guard fires.
    best_xy_dist: float | None = None
    diverging_since_t: float | None = None
    last_divergence_log_t: float = -float("inf")
    motion_estimator = (
        MotionDirectionEstimator(
            seed_rot=plan.calib.rot,
            seed_robot_xyyaw=plan.robot_start,
            seed_mocap_xyz=plan.mocap_start_xyz,
            min_motion_m=cfg.motion_calib_min_motion_m,
            max_yaw_change_rad=cfg.motion_calib_max_yaw_change_rad,
        )
        if cfg.use_motion_direction_calib
        else None
    )
    motion_calib_active = False

    def fixed_hb_goal(idx: int) -> np.ndarray:
        return plan.hb_waypoints[min(idx, len(plan.hb_waypoints) - 1)]

    def compute_hb_goal(
        idx: int,
        robot_current_: np.ndarray,
        mocap_xyz_: np.ndarray | None,
        mocap_quat_: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        if (
            cfg.replan
            and mocap_xyz_ is not None
            and mocap_quat_ is not None
        ):
            opti_to_hb_rot = (
                None if motion_estimator is None else motion_estimator.rot
            )
            return (
                replan_hb_goal_from_opti(
                    plan,
                    idx,
                    mocap_xyz_,
                    mocap_quat_,
                    robot_current_,
                    opti_to_hb_rot=opti_to_hb_rot,
                ),
                True,
            )
        return fixed_hb_goal(idx), False

    if not plan.hb_waypoints:
        return _build_nav_result(
            success=True,
            reason="reached",
            started=started,
            plan=plan,
            robot_current=robot_current,
            mocap_xyz=None,
            mocap_quat=None,
            opti_to_hb_rot=(
                None if motion_estimator is None else motion_estimator.rot
            ),
        )

    if not cfg.monitor:
        # Clear any latched 'stop' from a previously interrupted run before
        # publishing the new goal — otherwise the driver will overwrite
        # desired_pose back to current_pose while it self-clears the flag.
        client.set(cfg.base_keys.stop, "ok")
        write_desired_pose(client, fixed_hb_goal(0), keys=cfg.base_keys)

    success = False
    reason = "interrupted"
    try:
        while True:
            t0 = time.perf_counter()
            if not read_tracking_valid(client, keys=cfg.mocap_keys):
                _wait_for_mocap_alive(
                    client,
                    cfg=cfg,
                    reason=(
                        f"tracking lost ({cfg.mocap_keys.tracking_valid} "
                        f"is not true)"
                    ),
                )
                # Force a fresh goal from the recovered pose; clear stall
                # / divergence bookkeeping so a brief outage doesn't trip
                # the other guards the moment we resume.
                cached_hb_goal = None
                last_replan_t = -float("inf")
                if motion_estimator is not None:
                    robot_current = read_robot_se2(client, keys=cfg.base_keys)
                    mocap_xyz, _ = read_mocap_pose(client, keys=cfg.mocap_keys)
                    motion_estimator.reset_anchor(robot_current, mocap_xyz)
                    motion_calib_active = False
                stall_ref_xy = None
                stall_ref_yaw = None
                stall_ref_t = time.perf_counter()
                best_xy_dist = None
                diverging_since_t = None
                continue

            robot_current = read_robot_se2(client, keys=cfg.base_keys)
            try:
                mocap_xyz, mocap_quat = read_mocap_pose(client, keys=cfg.mocap_keys)
            except RuntimeError:
                mocap_xyz, mocap_quat = None, None
            if motion_estimator is not None and mocap_xyz is not None:
                prev_active = motion_estimator.estimate is not None
                estimate = motion_estimator.update(robot_current, mocap_xyz)
                motion_calib_active = motion_estimator.estimate is not None
                if cfg.print_log and estimate is not None and not prev_active:
                    print(
                        f"[opti_nav] motion direction calib active: "
                        f"lab_to_hb_yaw="
                        f"{math.degrees(estimate.yaw_rad):+.2f} deg "
                        f"from {estimate.opti_motion_m * 100:.1f} cm Opti / "
                        f"{estimate.hb_motion_m * 100:.1f} cm hb motion",
                        flush=True,
                    )

            # Stall guard: anchor on the freshest pose we have. If neither
            # XY nor yaw changes by more than the configured eps for
            # ``cfg.stall_timeout_s`` we exit with reason='stalled' so the
            # caller can keep going. Disabled when stall_timeout_s <= 0.
            if cfg.stall_timeout_s > 0:
                if mocap_xyz is not None and mocap_quat is not None:
                    cart_xy = np.asarray(mocap_xyz[:2], dtype=np.float64)
                    cart_yaw = wrap_angle(
                        quat_xyzw_to_yaw(mocap_quat)
                        - math.radians(plan.marker_yaw_offset_deg)
                    )
                else:
                    cart_xy = robot_current[:2].astype(np.float64)
                    cart_yaw = float(robot_current[2])
                now_stall = time.perf_counter()
                if stall_ref_xy is None or stall_ref_yaw is None:
                    stall_ref_xy = cart_xy.copy()
                    stall_ref_yaw = float(cart_yaw)
                    stall_ref_t = now_stall
                else:
                    moved_xy = float(np.linalg.norm(cart_xy - stall_ref_xy))
                    moved_yaw = abs(wrap_angle(float(cart_yaw) - stall_ref_yaw))
                    if (
                        moved_xy > cfg.stall_motion_eps_m
                        or moved_yaw > cfg.stall_motion_eps_rad
                    ):
                        stall_ref_xy = cart_xy.copy()
                        stall_ref_yaw = float(cart_yaw)
                        stall_ref_t = now_stall
                    elif now_stall - stall_ref_t > cfg.stall_timeout_s:
                        elapsed = now_stall - stall_ref_t
                        print(
                            f"[opti_nav] STALL: cart hasn't moved more than "
                            f"{cfg.stall_motion_eps_m * 100:.1f} cm / "
                            f"{math.degrees(cfg.stall_motion_eps_rad):.1f} deg "
                            f"for {elapsed:.2f}s "
                            f"(timeout={cfg.stall_timeout_s:.1f}s). Stopping.",
                            flush=True,
                        )
                        success = False
                        reason = "stalled"
                        return _build_nav_result(
                            success=False,
                            reason="stalled",
                            started=started,
                            plan=plan,
                            robot_current=robot_current,
                            mocap_xyz=mocap_xyz,
                            mocap_quat=mocap_quat,
                            opti_to_hb_rot=(
                                None
                                if motion_estimator is None
                                else motion_estimator.rot
                            ),
                        )

            # Divergence guard: catches "moving in the wrong direction"
            # which the pose-based stall guard above misses (cart pose
            # is changing, just not toward the target). Anchor on the
            # smallest Opti-XY distance to target seen so far; if the
            # live distance exceeds that minimum by more than
            # ``divergence_eps_m`` for ``divergence_timeout_s`` continuous
            # seconds, *print a warning* and keep going. Divergence is
            # informational here — we let the run continue so the
            # operator (or another guard) decides what to do. Only
            # meaningful when Opti is available.
            if (
                cfg.divergence_timeout_s > 0
                and mocap_xyz is not None
            ):
                cur_dist_xy = opti_xy_distance_m(plan, mocap_xyz)
                now_div = time.perf_counter()
                if best_xy_dist is None or cur_dist_xy < best_xy_dist:
                    best_xy_dist = cur_dist_xy
                    diverging_since_t = None
                elif cur_dist_xy > best_xy_dist + cfg.divergence_eps_m:
                    if diverging_since_t is None:
                        diverging_since_t = now_div
                    elif now_div - diverging_since_t > cfg.divergence_timeout_s:
                        if (
                            now_div - last_divergence_log_t
                            >= cfg.divergence_log_period_s
                        ):
                            elapsed_div = now_div - diverging_since_t
                            print(
                                f"[opti_nav] WARN diverging: Opti-XY distance "
                                f"to target is {cur_dist_xy * 100:.2f} cm, "
                                f"best seen was {best_xy_dist * 100:.2f} cm "
                                f"(grew by >={cfg.divergence_eps_m * 100:.1f} cm "
                                f"for {elapsed_div:.2f}s). Continuing anyway.",
                                flush=True,
                            )
                            last_divergence_log_t = now_div
                else:
                    # Within eps of the best; not actively diverging.
                    diverging_since_t = None

            now_replan = time.perf_counter()
            need_recompute = cached_hb_goal is None or (
                cfg.replan and (now_replan - last_replan_t) >= replan_period
            )
            if need_recompute:
                cached_hb_goal, cached_used_opti = compute_hb_goal(
                    waypoint_idx, robot_current, mocap_xyz, mocap_quat
                )
                last_replan_t = now_replan
            hb_goal, used_opti = cached_hb_goal, cached_used_opti

            if waypoint_idx < len(plan.hb_waypoints) - 1:
                if used_opti and mocap_xyz is not None:
                    xy_done = opti_xy_distance_m(plan, mocap_xyz) < cfg.tolerance_m
                else:
                    xy_done = waypoint_reached(
                        robot_current,
                        fixed_hb_goal(waypoint_idx),
                        waypoint_idx,
                        plan.hb_waypoints,
                        tolerance_m=cfg.tolerance_m,
                        tolerance_yaw_rad=cfg.tolerance_yaw_rad,
                    )
                if xy_done:
                    waypoint_idx += 1
                    # Waypoint advance always forces a fresh compute so the
                    # cached goal reflects the new waypoint immediately.
                    cached_hb_goal, cached_used_opti = compute_hb_goal(
                        waypoint_idx, robot_current, mocap_xyz, mocap_quat
                    )
                    last_replan_t = time.perf_counter()
                    hb_goal, used_opti = cached_hb_goal, cached_used_opti
                    if cfg.print_log:
                        print(
                            f"Waypoint {waypoint_idx}: hb={hb_goal.round(4).tolist()}"
                            + ("  (live opti)" if used_opti else "  (fixed)")
                        )

            if robot_prev is not None:
                jump = float(np.linalg.norm(robot_current[:2] - robot_prev[:2]))
                if jump > cfg.odom_jump_m and cfg.print_log:
                    print(f"Warning: hb1 odom jump {jump:.3f} m between cycles")
            robot_prev = robot_current.copy()

            if not cfg.monitor:
                write_desired_pose(client, hb_goal, keys=cfg.base_keys)

            # Telemetry hook: feed every relevant pose to the optional
            # ``on_sample`` callback so the move logger (or other
            # subscribers) can record base trajectories without having
            # to know about plan internals.
            if on_sample is not None:
                if mocap_quat is not None:
                    if motion_estimator is not None:
                        body_yaw_now = wrap_angle(
                            float(robot_current[2])
                            - rot2d_yaw(motion_estimator.rot)
                        )
                    else:
                        body_yaw_now = wrap_angle(
                            quat_xyzw_to_yaw(mocap_quat)
                            - math.radians(plan.marker_yaw_offset_deg)
                        )
                else:
                    body_yaw_now = None
                try:
                    on_sample(
                        t_perf=time.perf_counter(),
                        hb_xyyaw=robot_current,
                        hb_goal_xyyaw=hb_goal,
                        opti_xy=(None if mocap_xyz is None else mocap_xyz[:2]),
                        opti_yaw_rad=body_yaw_now,
                        opti_target_xy=plan.desired_mocap_xyz[:2],
                        opti_target_yaw_rad=(
                            plan.desired_body_yaw_in_opti
                            if plan.require_final_yaw
                            else None
                        ),
                        require_yaw=plan.require_final_yaw,
                        tolerance_m=cfg.tolerance_m,
                        tolerance_yaw_rad=cfg.tolerance_yaw_rad,
                    )
                except Exception as e:  # noqa: BLE001 - telemetry must never crash nav
                    print(f"[opti_nav] on_sample callback error: {e}", flush=True)

            # Compute success metrics every cycle (cheap) so we can both log
            # them at log_hz AND respect ``cfg.exit_on_success`` immediately.
            if used_opti and mocap_xyz is not None and mocap_quat is not None:
                track_xy_norm = opti_xy_distance_m(plan, mocap_xyz)
                track_yaw_err = (
                    opti_body_yaw_error_from_motion_calib_rad(
                        plan, robot_current, motion_estimator.rot
                    )
                    if motion_estimator is not None
                    else opti_body_yaw_error_rad(plan, mocap_quat)
                    if plan.require_final_yaw
                    else 0.0
                )
                success_frame = "Opti"
            else:
                track_xy_norm = float(
                    np.linalg.norm(robot_current[:2] - plan.robot_target[:2])
                )
                track_yaw_err = abs(
                    wrap_angle(robot_current[2] - plan.robot_target[2])
                )
                success_frame = "hb"
            is_final = waypoint_idx >= len(plan.hb_waypoints) - 1
            pose_ok = (
                is_final
                and track_xy_norm < cfg.tolerance_m
                and (
                    not plan.require_final_yaw
                    or track_yaw_err < cfg.tolerance_yaw_rad
                )
            )

            now = time.perf_counter()
            if cfg.print_log and now - last_log >= log_period:
                print_pose_log_block(
                    plan=plan,
                    robot_current=robot_current,
                    mocap_xyz=mocap_xyz,
                    mocap_quat=mocap_quat,
                    curr_minus_desired=cfg.curr_minus_desired,
                    hb_goal=hb_goal,
                    opti_to_hb_rot=(
                        None if motion_estimator is None else motion_estimator.rot
                    ),
                    motion_calib_active=motion_calib_active,
                )
                print()
                last_log = now
                if pose_ok and not reached_target:
                    msg = (
                        f"Success ({success_frame}): xy within "
                        f"{cfg.tolerance_m:.4f} m ({cfg.tolerance_m / 0.0254:.1f} in)"
                    )
                    if plan.require_final_yaw:
                        if plan.face_lab_yaw is not None:
                            msg += (
                                f" and body yaw within "
                                f"{math.degrees(cfg.tolerance_yaw_rad):.1f} deg "
                                f"(facing Motive {plan.face_lab_yaw})"
                            )
                        else:
                            msg += (
                                f" and body yaw within "
                                f"{math.degrees(cfg.tolerance_yaw_rad):.1f} deg "
                                f"(body target "
                                f"{math.degrees(plan.desired_body_yaw_in_opti):.1f} deg)"
                            )
                    suffix = " — exiting." if cfg.exit_on_success else " — holding."
                    print(msg + suffix)
                    reached_target = True

            if pose_ok and cfg.exit_on_success:
                success = True
                reason = "reached"
                return _build_nav_result(
                    success=True,
                    reason="reached",
                    started=started,
                    plan=plan,
                    robot_current=robot_current,
                    mocap_xyz=mocap_xyz,
                    mocap_quat=mocap_quat,
                    opti_to_hb_rot=(
                        None if motion_estimator is None else motion_estimator.rot
                    ),
                )

            _sleep_until(t0, period)
    except KeyboardInterrupt:
        print("\nStopped.")
        success = False
        reason = "interrupted"
    finally:
        if cfg.stop_on_exit and not cfg.monitor:
            stop_base(client, keys=cfg.base_keys)
            print(f"Set {cfg.base_keys.stop!r} = 'stop'")

    if cfg.monitor:
        success = False
        reason = "monitor_only"
    return _build_nav_result(
        success=success,
        reason=reason,
        started=started,
        plan=plan,
        robot_current=robot_current,
        mocap_xyz=mocap_xyz,
        mocap_quat=mocap_quat,
        opti_to_hb_rot=None if motion_estimator is None else motion_estimator.rot,
    )


# ---------------------------------------------------------------------------
# High-level one-call helper for downstream controllers
# ---------------------------------------------------------------------------


def navigate_to_opti_pose(
    client: redis.Redis,
    target_opti_xy: tuple[float, float] | np.ndarray,
    target_body_yaw_deg: float | None = None,
    *,
    target_opti_z: float = DEFAULT_TARGET_OPTI_Z,
    config: NavConfig | None = None,
    wait_for_tracking: bool = True,
    on_sample: Callable[..., None] | None = None,
) -> NavResult:
    """Drive the base to an absolute Motive lab pose, one call.

    Parameters
    ----------
    client
        Connected Redis client (must already be pinging the shared Redis
        used by the base / OptiTrack publisher).
    target_opti_xy
        Absolute Motive lab target ``(x, y)`` in meters.
    target_body_yaw_deg
        Absolute Motive lab heading for the cart's **body** +X (the actual
        driving direction). ``None`` (default) holds the startup yaw and
        only commands a straight-line XY move.
    target_opti_z
        Logged only; the hb base is planar. Default matches the floor
        plane in this rig.
    config
        :class:`NavConfig` overrides. ``None`` uses the locked-in defaults.
    wait_for_tracking
        Block until ``tidybot01::tracking_valid`` is true before
        calibrating. Pass ``False`` if the caller already gated on it.

    Returns
    -------
    :class:`NavResult`
        Outcome including final pose and Opti-world errors.
    """
    cfg = config or NavConfig()
    if wait_for_tracking:
        wait_for_tracking_valid(client, cfg.mocap_keys.tracking_valid)

    xy = np.asarray(target_opti_xy, dtype=np.float64).reshape(2)
    target_xyz = np.array(
        [float(xy[0]), float(xy[1]), float(target_opti_z)], dtype=np.float64
    )
    target_yaw_rad = (
        math.radians(float(target_body_yaw_deg))
        if target_body_yaw_deg is not None
        else None
    )

    plan = setup_run_plan(
        client,
        goal_delta_input_xy=np.zeros(2, dtype=np.float64),
        goal_frame=GoalFrame.OPTI_WORLD,
        translation_only_calib=cfg.translation_only_calib,
        calib_yaw_deg=cfg.calib_yaw_deg,
        cardinal_hb=cfg.cardinal_hb,
        direct_motion=cfg.direct_motion,
        rotate_first=cfg.rotate_first,
        marker_yaw_offset_deg=cfg.marker_yaw_offset_deg,
        face_lab_yaw=None,
        robot_pose_key=cfg.base_keys.robot_pose,
        mocap_pos_key=cfg.mocap_keys.pos,
        mocap_ori_key=cfg.mocap_keys.ori,
        curr_minus_desired=cfg.curr_minus_desired,
        absolute_mocap_target=(target_xyz, target_yaw_rad),
    )
    if cfg.print_plan:
        print_plan_summary(plan, config=cfg)
    return run_replan_loop(client, plan, config=cfg, on_sample=on_sample)
