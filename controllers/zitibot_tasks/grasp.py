"""Grasp and place subtasks."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from zitibot_core import arm, gains, gripper
from zitibot_core.constants import (
    OBJECT_DEFAULTS,
    PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
    PRECISE_GRASP_MAX_LINEAR_VELOCITY,
    PRECISE_GRASP_MOVE_TIMEOUT_S,
    PRECISE_GRASP_ORIENTATION_KP,
    PRECISE_GRASP_POSITION_KP,
    PRECISE_GRASP_WITHIN_M,
    Object,
    ObjectSpec,
)
from zitibot_core.context import TaskContext
from zitibot_core.runner import step_gate


def _resolve_spec(obj: Object) -> ObjectSpec:
    if obj not in OBJECT_DEFAULTS:
        raise KeyError(f"No ObjectSpec for {obj!r}; add entry to OBJECT_DEFAULTS")
    return OBJECT_DEFAULTS[obj]


def object(
    ctx: TaskContext,
    obj: Object,
    *,
    pick_pos: np.ndarray | None = None,
    ori: np.ndarray | None = None,
    lift_dz_m: float | None = None,
    lift_tol_m: float | None = None,
    grasp_tol_m: float | None = None,
    on_above: Callable[[], None] | None = None,
    close_mode: str = "grasp",
    move_timeout_s: float | None = None,
    precise: bool = True,
    keep_grip: bool = False,
) -> np.ndarray:
    """Approach, descend, grasp, and lift an object. Returns the lifted pose.

    Gates with ``--step``: move above, lower, grasp, lift.

    Parameters
    ----------
    keep_grip
        When ``True``, the gripper is already holding something (e.g. a tool
        like the tongs) — skip the initial open AND the pre-grasp pre-open, so
        the held object isn't dropped. The arm still moves above + descends,
        then the close phase simply force-closes (or move-closes) FURTHER from
        the current width onto the target (e.g. squeezing the tongs shut around
        an egg). Default ``False`` runs the normal open → descend → close.
    precise
        When ``True`` (default), the approach switches to a slow, stiff
        "precise grasp" regime once the EE comes within
        :data:`PRECISE_GRASP_WITHIN_M` of the pre-grasp "above" pose: the OTG
        linear-velocity cap drops to :data:`PRECISE_GRASP_MAX_LINEAR_VELOCITY`
        and the cartesian position/orientation kp jump to
        :data:`PRECISE_GRASP_POSITION_KP` / :data:`PRECISE_GRASP_ORIENTATION_KP`
        so the descent + close land accurately. The pre-engage values are
        snapshotted and restored right after the gripper closes, so the
        post-grasp lift and everything after run at normal stiffness/speed.
        The two precise-affected moves (above + descent) also get a longer
        convergence budget (:data:`PRECISE_GRASP_MOVE_TIMEOUT_S`) since the
        slow cap stretches the final approach. Pass ``False`` to grasp at the
        live gains/velocity throughout.
    close_mode
        How to close on the object. ``"grasp"`` (default) uses the gripper's
        GRASP mode, which the Franka driver force-closes all the way to 0 m
        (the target width is ignored) and clamps on whatever it hits — right
        for rigid objects. ``"move"`` uses MOVE mode, which travels to
        ``spec.close_width`` and STOPS there (no continuous squeeze) — use this
        to close only partway, e.g. a gentle hold on a fragile egg.
    on_above
        Optional callback invoked once the arm has reached the pre-grasp
        "above" pose, before descending to the pick. Used e.g. to snap a
        camera photo of the object from directly above the grasp.
    lift_dz_m
        Post-grasp lift height above ``pick`` in meters. ``None`` (default)
        uses ``spec.approach_dz`` (i.e. lifts back to the pre-grasp "above"
        pose). Pass a larger value (e.g. 0.25 for a 25 cm transport lift)
        when the caller wants the grasp + carry lift to happen as one
        continuous motion instead of two separate ENTER-gated steps.
    lift_tol_m
        Convergence tolerance (m) for the post-grasp carry lift only.
        ``None`` (default) uses ``spec.approach_tol`` — i.e. the same
        tight tolerance as the pre-grasp "move above" pose. Pass a
        larger value (e.g. 0.08) when the lift is just a transport move
        and exact endpoint accuracy isn't worth waiting an extra second
        of timeout for. The pre-grasp "move above" and descent
        tolerances are unaffected and stay tight.
    grasp_tol_m
        Convergence tolerance (m) for the descent-to-pick move only.
        ``None`` (default) uses ``spec.grasp_tol``. Pass a larger value
        (e.g. 0.06) for grasps where the gripper still closes safely
        from a slightly higher landing — e.g. bowls with a wide rim
        where landing 1-2 cm above the planned pick is fine. The
        pre-grasp "move above" and post-grasp lift tolerances are
        unaffected.
    move_timeout_s
        Optional override for ``arm.move_to`` convergence timeout (seconds)
        on every move inside this grasp sequence. ``None`` (default) uses
        the project-wide ``arm.move_to`` default (3 s).
    """
    spec = _resolve_spec(obj)
    if pick_pos is None:
        if spec.pick_pose is None:
            raise ValueError(f"{obj.value}: pick_pos required (no default in OBJECT_DEFAULTS)")
        pick = np.asarray(spec.pick_pose, dtype=np.float64).reshape(3).copy()
    else:
        pick = np.asarray(pick_pos, dtype=np.float64).reshape(3).copy()
    grip_R = np.asarray(ori if ori is not None else spec.grasp_ori, dtype=np.float64).reshape(3, 3)
    if spec.approach_along_tool_z:
        above = pick - grip_R[:, 2] * float(spec.approach_dz)
    else:
        above = pick + np.array([0.0, 0.0, spec.approach_dz], dtype=np.float64)
    lift_dz = float(lift_dz_m) if lift_dz_m is not None else float(spec.approach_dz)
    lifted = pick + np.array([0.0, 0.0, lift_dz], dtype=np.float64)

    if keep_grip:
        # Holding a tool already — don't open (would drop it). Read the live
        # width just for the log label.
        cur_w = gripper.read_current_width(ctx.redis)
        open_w = cur_w if cur_w is not None else float(spec.pregrasp_width)
    else:
        open_w = gripper.open_gripper(
            ctx.redis,
            spec.open_width,
            speed=spec.speed,
            force=spec.force,
        )
    base_timeout = float(move_timeout_s) if move_timeout_s is not None else 3.0
    # Precise-affected moves (above + descent) get the longer budget so the
    # slow OTG cap doesn't blow the timeout on the final approach.
    precise_timeout = max(base_timeout, PRECISE_GRASP_MOVE_TIMEOUT_S) if precise else base_timeout

    # One-shot precise-grasp engage, fired by ``arm.move_to``'s within-radius
    # hook the first tick the EE comes within PRECISE_GRASP_WITHIN_M of the
    # "above" pose. Snapshot is stashed so the close-phase restore can revert
    # the gains/velocity. ``finally`` guarantees restoration on any exit path.
    precise_state: dict[str, object] = {"snapshot": None}

    def _engage_precise() -> None:
        if precise_state["snapshot"] is not None:
            return
        precise_state["snapshot"] = gains.apply_precise_grasp(
            ctx.redis,
            max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
            max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
            position_kp=PRECISE_GRASP_POSITION_KP,
            orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
            label=f"grasp:{obj.value}",
        )

    def _restore_precise() -> None:
        snap = precise_state["snapshot"]
        if snap is not None:
            gains.restore_precise_grasp(ctx.redis, snap, label=f"grasp:{obj.value}")
            precise_state["snapshot"] = None

    try:
        arm.move_to(
            ctx,
            above,
            grip_R,
            label=f"[grasp:{obj.value}] move above pick {above.tolist()} (gripper open {open_w:.4f} m)",
            tol_m=spec.approach_tol,
            timeout_s=precise_timeout,
            on_within_m=PRECISE_GRASP_WITHIN_M if precise else None,
            on_within=_engage_precise if precise else None,
        )
        if on_above is not None:
            on_above()
        grasp_tol = float(grasp_tol_m) if grasp_tol_m is not None else float(spec.grasp_tol)
        arm.move_to(
            ctx,
            pick,
            grip_R,
            label=f"[grasp:{obj.value}] lower to pick {pick.tolist()}",
            tol_m=grasp_tol,
            timeout_s=precise_timeout,
        )
        if close_mode == "move":
            step_gate(
                ctx,
                f"[grasp:{obj.value}] move-close to {spec.close_width:.4f} m "
                f"(stops at width, force={spec.force:.1f} N)",
            )
            # Close straight to the target width — no pre-grasp pre-open step;
            # the gripper is already open from the start of the sequence.
            gripper.move(ctx.redis, spec.close_width, speed=spec.speed, force=spec.force)
            time.sleep(spec.grasp_settle_s)
        else:
            step_gate(ctx, f"[grasp:{obj.value}] grasp (force-close, force={spec.force:.1f} N)")
            # Force-close all the way — no pre-grasp pre-open step.
            gripper.grasp(ctx.redis, spec.close_width, speed=spec.speed, force=spec.force)
            time.sleep(4)
        # Gripper is closed — drop back to normal stiffness/speed before the
        # lift so the carry move runs at the usual cartesian gains/velocity.
        _restore_precise()
        lift_tol = float(lift_tol_m) if lift_tol_m is not None else float(spec.approach_tol)
        arm.move_to(
            ctx,
            lifted,
            grip_R,
            label=f"[grasp:{obj.value}] lift to {lifted.tolist()} (+{lift_dz * 100:.1f} cm)",
            tol_m=lift_tol,
            timeout_s=base_timeout,
        )
    finally:
        _restore_precise()
    print(f"[grasp:{obj.value}] lift complete at {lifted.tolist()}")
    return lifted


def place(
    ctx: TaskContext,
    obj: Object,
    *,
    place_pos: np.ndarray,
    ori: np.ndarray | None = None,
) -> None:
    """Move above place pose, descend, open gripper, retract.

    Gates with ``--step``: move above, lower, release, retract.
    """
    spec = _resolve_spec(obj)
    place = np.asarray(place_pos, dtype=np.float64).reshape(3).copy()
    grip_R = np.asarray(ori if ori is not None else spec.grasp_ori, dtype=np.float64).reshape(3, 3)
    above = place + np.array([0.0, 0.0, spec.approach_dz], dtype=np.float64)

    arm.move_to(
        ctx,
        above,
        grip_R,
        label=f"[place:{obj.value}] move above place {above.tolist()}",
        tol_m=spec.approach_tol,
    )
    arm.move_to(
        ctx,
        place,
        grip_R,
        label=f"[place:{obj.value}] lower to {place.tolist()}",
        tol_m=spec.grasp_tol,
    )
    step_gate(ctx, f"[place:{obj.value}] release (open gripper)")
    gripper.open_gripper(
        ctx.redis,
        spec.open_width,
        speed=spec.speed,
        force=spec.force,
        use_max_mode=True,
    )
    time.sleep(spec.grasp_settle_s)
    arm.move_to(
        ctx,
        above,
        grip_R,
        label=f"[place:{obj.value}] retract to {above.tolist()}",
        tol_m=spec.approach_tol,
    )
    print(f"[place:{obj.value}] released at {place.tolist()}")


def pick_and_place(
    ctx: TaskContext,
    obj: Object,
    *,
    pick_pos: np.ndarray | None = None,
    place_pos: np.ndarray | None = None,
    ori: np.ndarray | None = None,
) -> None:
    """Full pick → transport → place sequence."""
    spec = _resolve_spec(obj)
    pick = pick_pos if pick_pos is not None else spec.pick_pose
    place_p = place_pos if place_pos is not None else spec.place_pose
    if pick is None or place_p is None:
        raise ValueError(f"{obj.value}: pick_pos and place_pos required")
    object(ctx, obj, pick_pos=pick, ori=ori)
    grip_R = np.asarray(ori if ori is not None else spec.grasp_ori, dtype=np.float64).reshape(3, 3)
    place_arr = np.asarray(place_p, dtype=np.float64).reshape(3)
    above_place = place_arr + np.array([0.0, 0.0, spec.approach_dz], dtype=np.float64)
    arm.move_to(
        ctx,
        above_place,
        grip_R,
        label=f"[transport:{obj.value}] move above place {above_place.tolist()}",
        tol_m=spec.approach_tol,
    )
    place(ctx, obj, place_pos=place_p, ori=ori)
