#!/usr/bin/env python3
"""Whisk routine at the egg-crack station.

Sequence:

  1. Base → ``EGG_CRACK_STATION`` (skip with ``--skip-base``).
  2. Bowls-look pose → Gemini detects the black (pasta) bowl center.
  3. Rotated-home view pose → Gemini detects blue tape on the whisk handle.
  4. Gripper fully open → approach in front of tape (back in X) → move forward
     in X → close to hold width.
  5. Lift 5 cm, move over the bowl center, lower 10 cm, squeeze, unsqueeze.

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/whisk_controller.py -- --step

  ./ZitiBot/launch_zitibot_full.sh controllers/whisk_controller.py -- --debug

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
RealSense, OptiTrack on Redis (unless ``--skip-base``), and
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm, base, gains, gripper
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    BaseWaypoint,
    HOME_POS_TOL_M,
    OBJECT_DEFAULTS,
    PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
    PRECISE_GRASP_MAX_LINEAR_VELOCITY,
    PRECISE_GRASP_MOVE_TIMEOUT_S,
    PRECISE_GRASP_ORIENTATION_KP,
    PRECISE_GRASP_POSITION_KP,
    WHISK_GRASP_CLOSE_WIDTH_M,
    WHISK_GRASP_OPEN_WIDTH_M,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_core.runner import step_gate
from zitibot_tasks import gemini

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_BOWL_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_whisk_bowl.png"
)
DEFAULT_GEMINI_BLUE_RESPONSE_PATH = (
    _CONTROLLERS.parent / "logs" / "gemini_response_whisk_blue_plaster.png"
)

# Bowls-look offsets (same as egg_crack_controller).
WHISK_BOWLS_LOOK_DY_M = 0.30
WHISK_BOWLS_LOOK_DZ_M = 0.0

# Rotated-home view / approach pose offsets from ARM_HOME.
WHISK_VIEW_FWD_M = 0.0
WHISK_VIEW_DY_M = -0.25
WHISK_VIEW_LIFT_M = 0.20
WHISK_ROT_Y_DEG = -90.0
# Extra world-X tilt applied to the grasp orientation (after blue-tape
# detection) and held through carry / whisk / return until the whisk is set
# back down. Detection itself uses the un-tilted view orientation.
WHISK_GRASP_ROT_X_DEG = 10.0

WHISK_APPROACH_DX_M = 0.05
# When putting the whisk back: how far above the pickup spot the arm lines up in
# front before going forward, so it only descends this last bit to release.
WHISK_DROP_APPROACH_DZ_M = 0.08
# Extra -X backoff (on top of approach_dx) for the line-up-in-front pose so the
# whisk has more clearance before the forward move during placement.
WHISK_DROP_EXTRA_BACK_X_M = 0.10
# After releasing the whisk, back off this much further in -X before homing.
WHISK_PLACE_EXTRA_RETRACT_X_M = 0.15
# Place the whisk back this much further forward (+X) than where it was picked up.
WHISK_PLACE_FWD_X_M = 0.05
WHISK_LIFT_M = 0.10
WHISK_RETRACT_X_M = 0.20
WHISK_LOWER_M = 0.13
# Final stretch of the move-over-bowl approach done in precise mode (slow +
# stiff). The bulk of the transit runs at normal speed; only this last bit, the
# descent and the raise are precise.
WHISK_BOWL_PRECISE_APPROACH_M = 0.06
# In-bowl whisking: small forward/back/left/right cartesian strokes (m) around
# the in-bowl position, repeated until WHISK_STIR_DURATION_S elapses (a cycle
# already in progress at the deadline finishes before pulling out).
WHISK_STIR_OFFSET_M = 0.03
WHISK_STIR_DURATION_S = 15.0
WHISK_SQUEEZE_FORCE_N = 140.0
# How long to hold the squeeze before relaxing back to the unsqueeze width.
WHISK_SQUEEZE_HOLD_S = 5.0
WHISK_HOLD_WIDTH_M = WHISK_GRASP_CLOSE_WIDTH_M
# Width to relax back to after a squeeze (slightly wider than the pickup hold).
WHISK_UNSQUEEZE_WIDTH_M = 0.05

WHISK_MOVE_TIMEOUT_S = 15.0
WHISK_HOME_TIMEOUT_S = 5.0
# Carry / whisk / return transit moves: short timeout (don't sit waiting out a
# near-miss at a reach limit) and a loose tolerance (these are gross moves, not
# the grasp). The grasp-into-tape move keeps the tight WHISK_POS_TOL_M.
WHISK_BOWL_MOVE_TIMEOUT_S = 4.0
WHISK_BOWL_TOL_M = 0.08
WHISK_POS_TOL_M = 0.04
# Every whisk move settles (position in tol AND arm nearly stopped) before
# returning, so the arm fully comes to rest between steps instead of rushing
# into the next goal (e.g. grasp -> pull away, pull-up/back -> over bowl). This
# also keeps the Gemini detection frames from being captured mid-motion.
WHISK_SETTLE_VEL_TOL_RAD_S = 0.05
WHISK_SETTLE_TICKS = 3
# Dwell after the gripper closes on the whisk (and after release) so the grip
# is firmly established before the arm moves away.
WHISK_GRASP_HOLD_S = 1.0
# Open (release) is polled until the jaws actually reach the target width rather
# than relying on the short pregrasp settle, which can read the jaws mid-motion.
WHISK_OPEN_TIMEOUT_S = 3.0
WHISK_OPEN_TOL_M = 0.005
# Speed (m/s) commanded for opening. The Franka Hand caps internally, so a high
# value just runs the open at the hardware max.
WHISK_OPEN_SPEED_MPS = 0.15
# Width (m) commanded when opening / releasing. The driver clamps to max_width
# if this is larger, so 8 cm just opens the jaws fully.
WHISK_OPEN_WIDTH_M = 0.08


def _rot_y_matrix(deg: float) -> np.ndarray:
    theta = np.deg2rad(float(deg))
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64
    )


def _rot_x_matrix(deg: float) -> np.ndarray:
    theta = np.deg2rad(float(deg))
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64
    )


def _rotated_home_pose(
    *,
    fwd_m: float = WHISK_VIEW_FWD_M,
    dy_m: float = WHISK_VIEW_DY_M,
    lift_m: float = WHISK_VIEW_LIFT_M,
    rot_y_deg: float = WHISK_ROT_Y_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    home_pos = np.asarray(ARM_HOME_POSITION, dtype=np.float64).reshape(3).copy()
    home_ori = np.asarray(ARM_HOME_ORIENTATION, dtype=np.float64).reshape(3, 3).copy()
    pos = home_pos + np.array([float(fwd_m), float(dy_m), float(lift_m)], dtype=np.float64)
    ori = _rot_y_matrix(rot_y_deg) @ home_ori
    return pos, ori


def _move_arm(
    ctx: TaskContext,
    pos: np.ndarray,
    ori: np.ndarray,
    *,
    label: str,
    timeout_s: float = WHISK_MOVE_TIMEOUT_S,
    tol_m: float = WHISK_POS_TOL_M,
    settle_ticks: int = WHISK_SETTLE_TICKS,
    vel_tol_rad_s: float | None = WHISK_SETTLE_VEL_TOL_RAD_S,
) -> None:
    arm.move_to(
        ctx,
        np.asarray(pos, dtype=np.float64).reshape(3),
        np.asarray(ori, dtype=np.float64).reshape(3, 3),
        label=label,
        tol_m=tol_m,
        timeout_s=timeout_s,
        settle_ticks=settle_ticks,
        vel_tol_rad_s=vel_tol_rad_s,
    )


def _open_gripper_full(ctx: TaskContext, *, label: str = "[whisk]") -> None:
    spec = OBJECT_DEFAULTS[Object.WHISK]
    gripper.wait_for_ready(ctx.redis)
    open_w = WHISK_OPEN_WIDTH_M
    step_gate(ctx, f"{label} open gripper to {open_w * 100:.1f} cm")
    gripper.move(ctx.redis, open_w, speed=WHISK_OPEN_SPEED_MPS, force=spec.force)
    # Poll until the jaws actually reach the open width instead of trusting the
    # short pregrasp settle, which can read mid-motion (e.g. opening from the
    # post-squeeze hold) and report the old, still-closed width. The driver
    # clamps the command to max_width, so settle for whichever is smaller.
    max_w = gripper.resolve_open_width(ctx.redis, None)
    target = min(open_w, max_w)
    deadline = time.perf_counter() + WHISK_OPEN_TIMEOUT_S
    cur = gripper.read_current_width(ctx.redis)
    while time.perf_counter() < deadline:
        cur = gripper.read_current_width(ctx.redis)
        if cur is not None and cur >= target - WHISK_OPEN_TOL_M:
            break
        time.sleep(0.05)
    if cur is not None:
        print(f"{label} finger gap after open: {cur * 100:.1f} cm", flush=True)


def _close_to_hold_width(
    ctx: TaskContext,
    hold_width_m: float,
    *,
    label: str = "[whisk]",
    action: str = "close gripper to hold width",
    report: str = "after close",
) -> None:
    spec = OBJECT_DEFAULTS[Object.WHISK]
    step_gate(ctx, f"{label} {action} {hold_width_m * 100:.1f} cm")
    gripper.move(ctx.redis, float(hold_width_m), speed=spec.speed, force=spec.force)
    time.sleep(float(spec.grasp_settle_s) + WHISK_GRASP_HOLD_S)
    cur = gripper.read_current_width(ctx.redis)
    if cur is not None:
        print(f"{label} finger gap {report}: {cur * 100:.1f} cm", flush=True)


def _squeeze_and_unsqueeze(
    ctx: TaskContext,
    *,
    force_n: float,
    hold_width_m: float,
    unsqueeze_width_m: float = WHISK_UNSQUEEZE_WIDTH_M,
    during_hold: Callable[[], None] | None = None,
    label: str = "[whisk]",
) -> None:
    spec = OBJECT_DEFAULTS[Object.WHISK]
    step_gate(ctx, f"{label} squeeze gripper with {force_n:.0f} N")
    gripper.grasp(ctx.redis, 0.0, speed=spec.speed, force=float(force_n))
    # Let the grip establish, then either run the caller's motion WHILE the
    # gripper keeps squeezing (the driver holds the grasp until a new command),
    # or just dwell for the fixed hold time.
    time.sleep(float(spec.grasp_settle_s))
    if during_hold is not None:
        during_hold()
    else:
        time.sleep(WHISK_SQUEEZE_HOLD_S)
    cur = gripper.read_current_width(ctx.redis)
    if cur is not None:
        print(f"{label} finger gap after squeeze: {cur * 100:.1f} cm", flush=True)

    # Reuse the pickup hold routine (same settle dwell) so the unsqueeze relaxes
    # cleanly back from the force grasp instead of leaving a mid-release reading.
    _close_to_hold_width(
        ctx,
        unsqueeze_width_m,
        label=label,
        action="reopen to width",
        report="after unsqueeze",
    )


def _detect_bowl_center(
    ctx: TaskContext,
    *,
    bowls_look_dy_m: float,
    bowls_look_dz_m: float,
    retries: int,
    gemini_response_path: Path | None,
) -> np.ndarray:
    look_bowls_pos = ARM_HOME_POSITION + np.array(
        [0.0, float(bowls_look_dy_m), float(bowls_look_dz_m)], dtype=np.float64
    )
    _move_arm(
        ctx,
        look_bowls_pos,
        ARM_HOME_ORIENTATION,
        label=(
            f"[whisk] slide {bowls_look_dy_m * 100:+.0f} cm Y, "
            f"{bowls_look_dz_m * 100:+.0f} cm Z to look down at bowls "
            f"{look_bowls_pos.tolist()}"
        ),
        timeout_s=WHISK_HOME_TIMEOUT_S,
        tol_m=HOME_POS_TOL_M,
        vel_tol_rad_s=WHISK_SETTLE_VEL_TOL_RAD_S,
        settle_ticks=WHISK_SETTLE_TICKS,
    )
    if ctx.step:
        step_gate(ctx, "[whisk] ready to detect black bowl center — press ENTER")
    saved_path = ctx.gemini_response_path
    if gemini_response_path is not None:
        ctx.gemini_response_path = gemini_response_path
    try:
        bowl_center = gemini.find_bowl_drop_center(
            ctx, Object.PASTA_BOWL, retries=retries
        )
    finally:
        ctx.gemini_response_path = saved_path
    print(f"[whisk] black bowl center: {bowl_center.tolist()}")
    return np.asarray(bowl_center, dtype=np.float64).reshape(3)


def _detect_blue_tape(
    ctx: TaskContext,
    view_pos: np.ndarray,
    view_ori: np.ndarray,
    *,
    retries: int,
    gemini_response_path: Path | None,
) -> np.ndarray:
    _move_arm(
        ctx,
        view_pos,
        view_ori,
        label=f"[whisk] blue-tape view pose {view_pos.tolist()}",
        timeout_s=WHISK_MOVE_TIMEOUT_S,
        vel_tol_rad_s=WHISK_SETTLE_VEL_TOL_RAD_S,
        settle_ticks=WHISK_SETTLE_TICKS,
    )
    if ctx.step:
        step_gate(ctx, "[whisk] ready to detect blue tape — press ENTER")
    saved_path = ctx.gemini_response_path
    if gemini_response_path is not None:
        ctx.gemini_response_path = gemini_response_path
    try:
        pose_blue = gemini.find_grasp_pose(
            ctx,
            Object.WHISK,
            kind="grasp_blue_plaster",
            retries=retries,
            orientation_source="fixed",
        )
    finally:
        ctx.gemini_response_path = saved_path
    blue_pos = np.asarray(pose_blue.position, dtype=np.float64).reshape(3)
    print(f"[whisk] blue tape position: {blue_pos.tolist()}")
    return blue_pos


def _approach_and_grasp_blue_tape(
    ctx: TaskContext,
    blue_pos: np.ndarray,
    rot_ori: np.ndarray,
    *,
    approach_dx_m: float,
    hold_width_m: float,
) -> np.ndarray:
    """Forward-in-X approach at rotated-home orientation; returns post-grasp EE pos."""
    blue_pos = np.asarray(blue_pos, dtype=np.float64).reshape(3)
    rot_ori = np.asarray(rot_ori, dtype=np.float64).reshape(3, 3)

    _open_gripper_full(ctx)

    pre_grasp = blue_pos - np.array([float(approach_dx_m), 0.0, 0.0], dtype=np.float64)
    step_gate(
        ctx,
        f"[whisk] move in front of blue tape (back {approach_dx_m * 100:.0f} cm in X)",
    )
    _move_arm(
        ctx,
        pre_grasp,
        rot_ori,
        label=f"[whisk] pre-grasp in front of blue tape {pre_grasp.tolist()}",
    )

    step_gate(ctx, "[whisk] move forward in X to blue tape (precise)")
    precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="whisk blue-tape grasp",
    )
    try:
        _move_arm(
            ctx,
            blue_pos,
            rot_ori,
            label=f"[whisk] forward grasp at blue tape {blue_pos.tolist()} (precise)",
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
            tol_m=WHISK_POS_TOL_M,
        )
        _close_to_hold_width(ctx, hold_width_m)
    finally:
        gains.restore_precise_grasp(
            ctx.redis, precise, label="whisk blue-tape grasp"
        )
    return blue_pos.copy()


def _whisk_in_bowl(
    ctx: TaskContext,
    grasp_pos: np.ndarray,
    bowl_center: np.ndarray | None,
    rot_ori: np.ndarray,
    *,
    lift_m: float,
    retract_x_m: float,
    lower_m: float,
    force_n: float,
    hold_width_m: float,
) -> None:
    grasp_pos = np.asarray(grasp_pos, dtype=np.float64).reshape(3)
    rot_ori = np.asarray(rot_ori, dtype=np.float64).reshape(3, 3)

    lift_pos = grasp_pos + np.array(
        [-float(retract_x_m), 0.0, float(lift_m)], dtype=np.float64
    )
    step_gate(
        ctx,
        f"[whisk] pull up {lift_m * 100:.0f} cm (world +Z) "
        f"and back {retract_x_m * 100:.0f} cm (world -X)",
    )
    _move_arm(
        ctx,
        lift_pos,
        rot_ori,
        label=(
            f"[whisk] pull up {lift_m * 100:.0f} cm / back {retract_x_m * 100:.0f} cm "
            f"to {lift_pos.tolist()}"
        ),
        timeout_s=WHISK_BOWL_MOVE_TIMEOUT_S,
        tol_m=WHISK_BOWL_TOL_M,
    )

    if bowl_center is None:
        print("[whisk] --skip-bowl: squeezing in place (not going to bowl).")
        _squeeze_and_unsqueeze(ctx, force_n=force_n, hold_width_m=hold_width_m)
        return

    bowl_center = np.asarray(bowl_center, dtype=np.float64).reshape(3)
    over_bowl = bowl_center.copy()
    over_bowl[2] = lift_pos[2]

    # Coarse transit to just short of the bowl center at normal speed; the final
    # bit of the approach and the descent are done in precise mode.
    approach_vec = over_bowl - lift_pos
    dist = float(np.linalg.norm(approach_vec))
    if dist > WHISK_BOWL_PRECISE_APPROACH_M:
        coarse = over_bowl - approach_vec / dist * WHISK_BOWL_PRECISE_APPROACH_M
        step_gate(ctx, f"[whisk] move toward bowl center {coarse.tolist()}")
        _move_arm(
            ctx,
            coarse,
            rot_ori,
            label=f"[whisk] toward bowl {coarse.tolist()}",
            timeout_s=WHISK_BOWL_MOVE_TIMEOUT_S,
            tol_m=HOME_POS_TOL_M,
        )

    in_bowl = over_bowl - np.array([0.0, 0.0, float(lower_m)], dtype=np.float64)
    precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="whisk bowl",
    )
    try:
        step_gate(ctx, f"[whisk] move over bowl center {over_bowl.tolist()} (precise)")
        _move_arm(
            ctx,
            over_bowl,
            rot_ori,
            label=f"[whisk] over bowl {over_bowl.tolist()} (precise)",
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
            tol_m=WHISK_POS_TOL_M,
        )

        step_gate(ctx, f"[whisk] lower {lower_m * 100:.0f} cm into bowl (precise)")
        _move_arm(
            ctx,
            in_bowl,
            rot_ori,
            label=f"[whisk] lower into bowl {in_bowl.tolist()} (precise)",
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
            tol_m=WHISK_BOWL_TOL_M,
        )
    finally:
        gains.restore_precise_grasp(ctx.redis, precise, label="whisk bowl")

    def _stir() -> None:
        # Whisk the bowl WHILE squeezing: small forward/back/left/right cartesian
        # strokes around the in-bowl position, repeated until the stir duration
        # elapses (an in-progress cycle finishes first), then return to center.
        stir = WHISK_STIR_OFFSET_M
        stir_moves = (
            (+stir, 0.0, "forward"),
            (-stir, 0.0, "backward"),
            (0.0, +stir, "left"),
            (0.0, -stir, "right"),
        )
        deadline = time.perf_counter() + WHISK_STIR_DURATION_S
        cycle = 0
        while time.perf_counter() < deadline:
            cycle += 1
            for dx, dy, name in stir_moves:
                target = in_bowl + np.array([dx, dy, 0.0], dtype=np.float64)
                step_gate(ctx, f"[whisk] stir cycle {cycle} {name} {stir * 100:.0f} cm")
                _move_arm(
                    ctx,
                    target,
                    rot_ori,
                    label=f"[whisk] stir {name} {target.tolist()}",
                    timeout_s=WHISK_BOWL_MOVE_TIMEOUT_S,
                    tol_m=WHISK_BOWL_TOL_M,
                )
        step_gate(ctx, "[whisk] stir back to center")
        _move_arm(
            ctx,
            in_bowl,
            rot_ori,
            label=f"[whisk] stir center {in_bowl.tolist()}",
            timeout_s=WHISK_BOWL_MOVE_TIMEOUT_S,
            tol_m=WHISK_BOWL_TOL_M,
        )

    # Squeeze the whisk and stir the bowl while it stays squeezed, then unsqueeze.
    _squeeze_and_unsqueeze(
        ctx, force_n=force_n, hold_width_m=hold_width_m, during_hold=_stir
    )


def _return_whisk(
    ctx: TaskContext,
    grasp_pos: np.ndarray,
    bowl_center: np.ndarray | None,
    rot_ori: np.ndarray,
    *,
    lift_m: float,
    retract_x_m: float,
    lower_m: float,
    approach_dx_m: float,
) -> None:
    """Reverse the place steps: carry the whisk back and release it where grasped."""
    grasp_pos = np.asarray(grasp_pos, dtype=np.float64).reshape(3)
    rot_ori = np.asarray(rot_ori, dtype=np.float64).reshape(3, 3)

    lift_pos = grasp_pos + np.array(
        [-float(retract_x_m), 0.0, float(lift_m)], dtype=np.float64
    )

    if bowl_center is not None:
        bowl_center = np.asarray(bowl_center, dtype=np.float64).reshape(3)
        over_bowl = bowl_center.copy()
        over_bowl[2] = lift_pos[2]

        step_gate(ctx, f"[whisk] raise {lower_m * 100:.0f} cm out of bowl (precise)")
        precise = gains.apply_precise_grasp(
            ctx.redis,
            max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
            max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
            position_kp=PRECISE_GRASP_POSITION_KP,
            orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
            label="whisk bowl raise",
        )
        try:
            _move_arm(
                ctx,
                over_bowl,
                rot_ori,
                label=f"[whisk] raise out of bowl {over_bowl.tolist()} (precise)",
                timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
                tol_m=WHISK_BOWL_TOL_M,
            )
        finally:
            gains.restore_precise_grasp(ctx.redis, precise, label="whisk bowl raise")

        step_gate(ctx, "[whisk] return over pickup")
        _move_arm(
            ctx,
            lift_pos,
            rot_ori,
            label=f"[whisk] return over pickup {lift_pos.tolist()}",
            timeout_s=WHISK_BOWL_MOVE_TIMEOUT_S,
            tol_m=WHISK_BOWL_TOL_M,
        )

    drop_dz = WHISK_DROP_APPROACH_DZ_M
    drop_back_x = float(approach_dx_m) + WHISK_DROP_EXTRA_BACK_X_M

    # Place the whisk back a bit further forward (+X) than it was picked up. Only
    # the forward step reaches further; the line-up stays relative to grasp_pos.
    place_pos = grasp_pos + np.array(
        [WHISK_PLACE_FWD_X_M, 0.0, 0.0], dtype=np.float64
    )

    in_front = grasp_pos + np.array(
        [-drop_back_x, 0.0, float(drop_dz)], dtype=np.float64
    )
    step_gate(
        ctx,
        f"[whisk] line up in front of pickup ({drop_back_x * 100:.0f} cm back, "
        f"{drop_dz * 100:.0f} cm above target)",
    )
    _move_arm(
        ctx,
        in_front,
        rot_ori,
        label=f"[whisk] in front of pickup {in_front.tolist()}",
        timeout_s=WHISK_BOWL_MOVE_TIMEOUT_S,
        tol_m=WHISK_BOWL_TOL_M,
    )

    over_pickup = place_pos + np.array([0.0, 0.0, float(drop_dz)], dtype=np.float64)
    pre_grasp = place_pos - np.array([float(approach_dx_m), 0.0, 0.0], dtype=np.float64)

    # Place the whisk back in precise mode (stiffer + slower) so the forward,
    # descend, release and retract mirror the careful grasp approach.
    precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="whisk place",
    )
    try:
        step_gate(
            ctx,
            f"[whisk] forward {(drop_back_x + WHISK_PLACE_FWD_X_M) * 100:.0f} cm "
            f"over pickup spot (precise)",
        )
        _move_arm(
            ctx,
            over_pickup,
            rot_ori,
            label=f"[whisk] forward over pickup spot {over_pickup.tolist()} (precise)",
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
            tol_m=WHISK_POS_TOL_M,
        )

        step_gate(ctx, f"[whisk] descend {drop_dz * 100:.0f} cm to pickup spot (precise)")
        _move_arm(
            ctx,
            place_pos,
            rot_ori,
            label=f"[whisk] descend to pickup spot {place_pos.tolist()} (precise)",
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
            tol_m=WHISK_POS_TOL_M,
        )

        _open_gripper_full(ctx, label="[whisk] release")

        step_gate(
            ctx,
            f"[whisk] retract {approach_dx_m * 100:.0f} cm in -X from pickup (precise)",
        )
        _move_arm(
            ctx,
            pre_grasp,
            rot_ori,
            label=f"[whisk] retract from pickup {pre_grasp.tolist()} (precise)",
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
            tol_m=WHISK_POS_TOL_M,
        )

        backoff = pre_grasp - np.array(
            [WHISK_PLACE_EXTRA_RETRACT_X_M, 0.0, 0.0], dtype=np.float64
        )
        step_gate(
            ctx,
            f"[whisk] back off {WHISK_PLACE_EXTRA_RETRACT_X_M * 100:.0f} cm more "
            f"in -X (precise)",
        )
        _move_arm(
            ctx,
            backoff,
            rot_ori,
            label=f"[whisk] back off from pickup {backoff.tolist()} (precise)",
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
            tol_m=WHISK_POS_TOL_M,
        )
    finally:
        gains.restore_precise_grasp(ctx.redis, precise, label="whisk place")


def run_whisk_cycle(
    ctx: TaskContext,
    *,
    skip_base: bool = False,
    skip_bowl: bool = False,
    bowls_look_dy_m: float = WHISK_BOWLS_LOOK_DY_M,
    bowls_look_dz_m: float = WHISK_BOWLS_LOOK_DZ_M,
    view_fwd_m: float = WHISK_VIEW_FWD_M,
    view_dy_m: float = WHISK_VIEW_DY_M,
    view_lift_m: float = WHISK_VIEW_LIFT_M,
    rot_y_deg: float = WHISK_ROT_Y_DEG,
    grasp_rot_x_deg: float = WHISK_GRASP_ROT_X_DEG,
    approach_dx_m: float = WHISK_APPROACH_DX_M,
    hold_width_m: float = WHISK_HOLD_WIDTH_M,
    lift_m: float = WHISK_LIFT_M,
    retract_x_m: float = WHISK_RETRACT_X_M,
    lower_m: float = WHISK_LOWER_M,
    squeeze_force_n: float = WHISK_SQUEEZE_FORCE_N,
    retries: int = 1,
    gemini_bowl_response_path: Path | None = None,
    gemini_blue_response_path: Path | None = None,
) -> None:
    home_pos = np.asarray(ARM_HOME_POSITION, dtype=np.float64).reshape(3)
    step_gate(ctx, f"[whisk] move to home {home_pos.tolist()}")
    _move_arm(
        ctx,
        home_pos,
        ARM_HOME_ORIENTATION,
        label=f"[whisk] move to home {home_pos.tolist()}",
        timeout_s=WHISK_HOME_TIMEOUT_S,
        tol_m=HOME_POS_TOL_M,
    )

    if not skip_base:
        base.go_to_pose(ctx, BaseWaypoint.EGG_CRACK_STATION, motion="three_phase")
    else:
        print("[whisk] --skip-base: assuming cart already at egg-crack station.")

    bowl_center: np.ndarray | None = None
    if not skip_bowl:
        bowl_center = _detect_bowl_center(
            ctx,
            bowls_look_dy_m=bowls_look_dy_m,
            bowls_look_dz_m=bowls_look_dz_m,
            retries=retries,
            gemini_response_path=Path(gemini_bowl_response_path)
            if gemini_bowl_response_path is not None
            else None,
        )
    else:
        print("[whisk] --skip-bowl: skipping bowl detection, going to whisk.")

    view_pos, rot_ori = _rotated_home_pose(
        fwd_m=view_fwd_m, dy_m=view_dy_m, lift_m=view_lift_m, rot_y_deg=rot_y_deg
    )
    blue_pos = _detect_blue_tape(
        ctx,
        view_pos,
        rot_ori,
        retries=retries,
        gemini_response_path=Path(gemini_blue_response_path)
        if gemini_blue_response_path is not None
        else None,
    )

    # Tilt the wrist by a few degrees about world X for the grasp and everything
    # after, held until the whisk is set back down. Detection used rot_ori.
    grasp_ori = _rot_x_matrix(grasp_rot_x_deg) @ rot_ori

    grasp_pos = _approach_and_grasp_blue_tape(
        ctx,
        blue_pos,
        grasp_ori,
        approach_dx_m=approach_dx_m,
        hold_width_m=hold_width_m,
    )

    _whisk_in_bowl(
        ctx,
        grasp_pos,
        bowl_center,
        grasp_ori,
        lift_m=lift_m,
        retract_x_m=retract_x_m,
        lower_m=lower_m,
        force_n=squeeze_force_n,
        hold_width_m=hold_width_m,
    )

    _return_whisk(
        ctx,
        grasp_pos,
        bowl_center,
        grasp_ori,
        lift_m=lift_m,
        retract_x_m=retract_x_m,
        lower_m=lower_m,
        approach_dx_m=approach_dx_m,
    )

    home_pos = np.asarray(ARM_HOME_POSITION, dtype=np.float64).reshape(3)
    step_gate(ctx, f"[whisk] return to home {home_pos.tolist()}")
    _move_arm(
        ctx,
        home_pos,
        ARM_HOME_ORIENTATION,
        label=f"[whisk] return to home {home_pos.tolist()}",
        timeout_s=WHISK_MOVE_TIMEOUT_S,
        tol_m=HOME_POS_TOL_M,
    )
    print("[whisk] whisk cycle complete.")


def run_debug(
    ctx: TaskContext,
    *,
    view_fwd_m: float = WHISK_VIEW_FWD_M,
    view_dy_m: float = WHISK_VIEW_DY_M,
    view_lift_m: float = WHISK_VIEW_LIFT_M,
    rot_y_deg: float = WHISK_ROT_Y_DEG,
    hold_width_m: float = WHISK_HOLD_WIDTH_M,
    squeeze_force_n: float = WHISK_SQUEEZE_FORCE_N,
) -> None:
    """No-base/no-vision gripper+arm test at the rotated-home pose."""
    home_pos = np.asarray(ARM_HOME_POSITION, dtype=np.float64).reshape(3).copy()
    home_ori = np.asarray(ARM_HOME_ORIENTATION, dtype=np.float64).reshape(3, 3).copy()

    step_gate(ctx, f"[debug] move to home {home_pos.tolist()}")
    _move_arm(
        ctx,
        home_pos,
        home_ori,
        label=f"[debug] move to home {home_pos.tolist()}",
        timeout_s=WHISK_HOME_TIMEOUT_S,
        tol_m=HOME_POS_TOL_M,
    )

    view_pos, rot_ori = _rotated_home_pose(
        fwd_m=view_fwd_m, dy_m=view_dy_m, lift_m=view_lift_m, rot_y_deg=rot_y_deg
    )
    step_gate(
        ctx,
        f"[debug] rotate {rot_y_deg:+.0f} deg Y + move "
        f"{view_fwd_m * 100:+.0f} cm X / {view_dy_m * 100:+.0f} cm Y / "
        f"{view_lift_m * 100:+.0f} cm Z",
    )
    _move_arm(
        ctx,
        view_pos,
        rot_ori,
        label=f"[debug] rotated-home pose {view_pos.tolist()}",
        timeout_s=WHISK_HOME_TIMEOUT_S,
        tol_m=HOME_POS_TOL_M,
    )

    _open_gripper_full(ctx, label="[debug]")
    _close_to_hold_width(ctx, hold_width_m, label="[debug]")
    _squeeze_and_unsqueeze(
        ctx,
        force_n=squeeze_force_n,
        hold_width_m=hold_width_m,
        label="[debug]",
    )
    print("[debug] holding pose + width — press Ctrl+C to exit.", flush=True)
    while True:
        time.sleep(0.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Whisk at egg-crack station: detect bowl, grasp blue tape, whisk."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate each motion / gripper step.",
    )
    p.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base drive to EGG_CRACK_STATION.",
    )
    p.add_argument(
        "--skip-bowl",
        action="store_true",
        help=(
            "Skip bowl detection and the in-bowl whisking + return; go straight "
            "to detecting and grasping the whisk, then stop."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Debug gripper check: home, rotated-home pose, open, close to hold "
            "width, squeeze, unsqueeze. Skips base and vision."
        ),
    )
    p.add_argument(
        "--endeffector-transform-key",
        default=DEFAULT_ENDEFFECTOR_TRANSFORM_KEY,
    )
    p.add_argument(
        "--gemini-bowl-response-path",
        type=Path,
        default=DEFAULT_GEMINI_BOWL_RESPONSE_PATH,
        help="Debug image for black-bowl Gemini detection.",
    )
    p.add_argument(
        "--gemini-blue-response-path",
        type=Path,
        default=DEFAULT_GEMINI_BLUE_RESPONSE_PATH,
        help="Debug image for blue-tape Gemini detection.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Gemini detection retries on failure.",
    )
    p.add_argument(
        "--bowls-look-dy-m",
        type=float,
        default=WHISK_BOWLS_LOOK_DY_M,
        help="World +Y offset from home for bowl detection (m). Default 30 cm.",
    )
    p.add_argument(
        "--bowls-look-dz-m",
        type=float,
        default=WHISK_BOWLS_LOOK_DZ_M,
        help="World +Z offset from home for bowl detection (m).",
    )
    p.add_argument(
        "--view-fwd-m",
        type=float,
        default=WHISK_VIEW_FWD_M,
        help="World +X offset from home for blue-tape view pose (m). Default 0 cm.",
    )
    p.add_argument(
        "--view-dy-m",
        type=float,
        default=WHISK_VIEW_DY_M,
        help="World +Y offset from home for blue-tape view pose (m). Default -20 cm.",
    )
    p.add_argument(
        "--view-lift-m",
        type=float,
        default=WHISK_VIEW_LIFT_M,
        help="World +Z offset from home for blue-tape view pose (m). Default 20 cm.",
    )
    p.add_argument(
        "--rot-y-deg",
        type=float,
        default=WHISK_ROT_Y_DEG,
        help="Rotation about world Y at blue-tape view/grasp (deg). Default -90.",
    )
    p.add_argument(
        "--grasp-rot-x-deg",
        type=float,
        default=WHISK_GRASP_ROT_X_DEG,
        help=(
            "Extra world-X tilt applied after detection, held through grasp / "
            "carry / whisk / return until set down (deg). Default 5."
        ),
    )
    p.add_argument(
        "--approach-dx-m",
        type=float,
        default=WHISK_APPROACH_DX_M,
        help="Pre-grasp back-off in world -X before forward approach (m). Default 5 cm.",
    )
    p.add_argument(
        "--hold-width-m",
        type=float,
        default=WHISK_HOLD_WIDTH_M,
        help="Gripper hold width after grasp (m). Default 4.5 cm.",
    )
    p.add_argument(
        "--lift-m",
        type=float,
        default=WHISK_LIFT_M,
        help="Post-grasp lift in world +Z before moving over bowl (m). Default 5 cm.",
    )
    p.add_argument(
        "--retract-x-m",
        type=float,
        default=WHISK_RETRACT_X_M,
        help="Post-grasp pull-back in world -X with the lift (m). Default 10 cm.",
    )
    p.add_argument(
        "--lower-m",
        type=float,
        default=WHISK_LOWER_M,
        help="Descent into bowl from hover (m). Default 10 cm.",
    )
    p.add_argument(
        "--squeeze-force",
        type=float,
        default=WHISK_SQUEEZE_FORCE_N,
        help="Force (N) for bowl squeeze. Default 140 N.",
    )
    p.add_argument(
        "--debug-width",
        type=float,
        default=WHISK_HOLD_WIDTH_M,
        help="Hold width (m) used by --debug. Default 4.5 cm.",
    )
    p.add_argument(
        "--debug-force",
        type=float,
        default=WHISK_SQUEEZE_FORCE_N,
        help="Squeeze force (N) used by --debug. Default 140 N.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    if args.debug:
        print(f"Step mode       : {'on' if args.step else 'off'}")
        print(f"Mode            : DEBUG (home → rotated-home → open → close → squeeze)")
        print(f"Rotate Y        : {args.rot_y_deg:+.0f} deg")
        print(f"View offset     : {args.view_fwd_m * 100:+.0f} cm X / {args.view_dy_m * 100:+.0f} cm Y / {args.view_lift_m * 100:+.0f} cm Z")
        print(f"Hold width      : {args.debug_width * 100:.1f} cm")
        print(f"Squeeze force   : {args.debug_force:.0f} N")
        try:
            run_debug(
                ctx,
                view_fwd_m=args.view_fwd_m,
                view_dy_m=args.view_dy_m,
                view_lift_m=args.view_lift_m,
                rot_y_deg=args.rot_y_deg,
                hold_width_m=args.debug_width,
                squeeze_force_n=args.debug_force,
            )
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        finally:
            ctx.stop_realsense()
        return 0

    print(f"Step mode       : {'on' if args.step else 'off'}")
    print(f"Skip base       : {'yes' if args.skip_base else 'no'}")
    print(f"Skip bowl       : {'yes' if args.skip_bowl else 'no'}")
    print(f"Bowls look      : dy={args.bowls_look_dy_m * 100:+.0f} cm  dz={args.bowls_look_dz_m * 100:+.0f} cm")
    print(f"Blue-tape view  : {args.view_fwd_m * 100:+.0f} cm X / {args.view_dy_m * 100:+.0f} cm Y / {args.view_lift_m * 100:+.0f} cm Z  rot_y={args.rot_y_deg:+.0f} deg")
    print(f"Grasp X tilt    : {args.grasp_rot_x_deg:+.0f} deg (held until set down)")
    print(f"Approach back   : {args.approach_dx_m * 100:.0f} cm (world -X)")
    print(f"Hold width      : {args.hold_width_m * 100:.1f} cm")
    print(f"Lift / retract  : {args.lift_m * 100:.0f} cm +Z / {args.retract_x_m * 100:.0f} cm -X")
    print(f"Lower           : {args.lower_m * 100:.0f} cm into bowl")
    print(f"Squeeze force   : {args.squeeze_force:.0f} N")
    print(f"Gemini bowl     : {args.gemini_bowl_response_path}")
    print(f"Gemini blue     : {args.gemini_blue_response_path}")

    try:
        run_whisk_cycle(
            ctx,
            skip_base=args.skip_base,
            skip_bowl=args.skip_bowl,
            bowls_look_dy_m=args.bowls_look_dy_m,
            bowls_look_dz_m=args.bowls_look_dz_m,
            view_fwd_m=args.view_fwd_m,
            view_dy_m=args.view_dy_m,
            view_lift_m=args.view_lift_m,
            rot_y_deg=args.rot_y_deg,
            grasp_rot_x_deg=args.grasp_rot_x_deg,
            approach_dx_m=args.approach_dx_m,
            hold_width_m=args.hold_width_m,
            lift_m=args.lift_m,
            retract_x_m=args.retract_x_m,
            lower_m=args.lower_m,
            squeeze_force_n=args.squeeze_force,
            retries=args.retries,
            gemini_bowl_response_path=args.gemini_bowl_response_path,
            gemini_blue_response_path=args.gemini_blue_response_path,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
