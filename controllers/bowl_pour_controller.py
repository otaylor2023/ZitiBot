#!/usr/bin/env python3
"""OptiTrack base + Gemini vision bowl grasp + pour + sink drop.

End-to-end per-bowl sequence (``run_bowl_cycle``):

  1. Arm → home pose.
  2. Base → bowl's grasp station (``BOWL_GRASP_STATION[bowl]``;
     ``PASTA_STATION`` for the pasta / plastic bowls, ``MIXING_STATION``
     for the mixing bowl; per-axis overrides via ``--grasp-x/y/yaw-deg``).
  3. Gemini detects the chosen bowl on RealSense. The same
     ``gemini.find_grasp_pose`` path is used for every bowl; per-bowl
     XY/Z framing comes from ``OBJECT_DEFAULTS[bowl].gemini_world_offset_m``
     plus a controller-level Z bump from ``BOWL_GRASP_Z_OFFSETS_M[bowl]``.
  4. ``grasp.object`` approaches, closes, and lifts to ``BOWL_CARRY_LIFT_M``.
  5. Base → bowl's pour station (``BOWL_POUR_STATION[bowl]``;
     ``MIXING_STATION`` for the pasta / plastic bowls, ``PAN_STATION``
     for the mixing bowl; per-axis overrides via ``--pour-x/y/yaw-deg``).
  6. Arm walks forward through the bowl's registered EE pour waypoints
     (``BOWL_POUR_EE_WAYPOINTS[bowl]``: start → mid → final) and STOPS
     at the final, fully tilted waypoint. There is no explicit un-pour
     move; the post-pour lift in step 7 doubles as the return to
     upright. Bowls without registered waypoints fall back to a slerp
     tilt at the bowl's pour anchor pose (``BOWL_POUR_ANCHOR_POSES``;
     the generic ``ARM_HOME_POSITION + +Z * POUR_LIFT_FROM_HOME_M`` upright
     anchor). The slerp fallback also calls ``pour.return_upright``
     unless the bowl is in ``BOWLS_SKIP_RETURN_UPRIGHT``.
  7. Arm moves directly from the final pour pose (waypoint mode) or
     the slerp pour target (fallback mode) to ``ARM_HOME_POSITION +
     +Z * TRANSIT_LIFT_FROM_HOME_M`` (default 7 cm above home),
     returning the bowl to upright as it lifts so it clears the
     counter / cart edge during transit. No intermediate stop at
     ``ARM_HOME``.
  8. Base → ``SINK_STATION`` while the arm holds the bowl at the
     lifted transit pose.
  9. Arm descends ``SINK_DROP_DZ_M`` (default 20 cm) below the carry
     pose, gripper opens, bowl drops into the sink, and the arm goes
     straight back to ``ARM_HOME`` (no retrace through the carry pose).

``main()`` runs ``run_bowl_cycle`` once per bowl in the sequence
selected by ``--bowl``. The default ``all`` runs the full cycle three
times in order **pasta → plastic_bottom → plastic_top**. Single-bowl
modes (``pasta`` / ``plastic_top`` / ``plastic_bottom`` / ``mixing``)
run exactly one cycle. ``mixing`` is intentionally not part of ``all``
because it's a separate ingredient flow (grasps at MIXING_STATION,
pours into the pan at PAN_STATION) rather than a recipe variant.

This controller intentionally uses **only** ``zitibot_core`` and
``zitibot_tasks`` — no imports from any sibling controller — so it can't
break the way the older version did when other controllers were
refactored. The previous ``mixing_bowl_to_pan_pour_controller.py``
standalone was folded into this controller as the ``mixing`` bowl
choice so all bowl flows share the same Gemini-detect → grasp.object →
pour → sink-drop path with per-bowl overrides instead of duplicated
code.

Usage::

  # default: pasta → plastic_bottom → plastic_top
  ./ZitiBot/launch_zitibot_full.sh controllers/bowl_pour_controller.py
  ./ZitiBot/launch_zitibot_full.sh controllers/bowl_pour_controller.py -- --step

  # single-bowl runs
  ./ZitiBot/launch_zitibot_full.sh controllers/bowl_pour_controller.py -- \\
      --bowl pasta
  ./ZitiBot/launch_zitibot_full.sh controllers/bowl_pour_controller.py -- \\
      --bowl plastic_top
  ./ZitiBot/launch_zitibot_full.sh controllers/bowl_pour_controller.py -- \\
      --bowl plastic_bottom

  # mixing bowl → pan pour → sink (replaces the old
  # ``mixing_bowl_to_pan_pour_controller.py`` standalone)
  ./ZitiBot/launch_zitibot_full.sh controllers/bowl_pour_controller.py -- \\
      --bowl mixing
  ./ZitiBot/launch_zitibot_full.sh controllers/bowl_pour_controller.py -- \\
      --bowl mixing \\
      --grasp-x 0.54 --grasp-y -2.76 --grasp-yaw-deg 90 \\
      --pour-x 0.34 --pour-y -2.76 --pour-yaw-deg 90

  # Raw Opti overrides (any combination of x/y/yaw works; falls back to
  # the per-bowl station default for whichever flag isn't passed). These
  # apply to every bowl in the selected sequence:
  ./ZitiBot/launch_zitibot_full.sh controllers/bowl_pour_controller.py -- \\
      --grasp-x 1.02 --grasp-y -2.79 --grasp-yaw-deg 90 \\
      --pour-x 0.54 --pour-y -2.76 --pour-yaw-deg 90

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
TidyBot base ``redis_driver``, RealSense, OptiTrack on Redis, and
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm, base, gripper
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    BASE_WAYPOINTS,
    BOWL_POUR_EE_WAYPOINTS,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    MIXING_BOWL_PAN_POUR_START,
    OBJECT_DEFAULTS,
    BaseWaypoint,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_tasks import gemini, grasp, pour

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_RESPONSE_PATH = _CONTROLLERS.parent / "logs" / "gemini_response.png"
# Per-bowl Gemini response image. Falls back to ``DEFAULT_GEMINI_RESPONSE_PATH``
# for bowls without a dedicated entry so the existing pasta / plastic
# flows keep writing to the same logs/gemini_response.png they always did.
GEMINI_RESPONSE_PATHS: dict[Object, Path] = {
    Object.MIXING_BOWL: _CONTROLLERS.parent / "logs" / "gemini_response_mixing_bowl_grasp.png",
}

# Drop station is the same for every bowl (sink). Grasp + pour stations
# are per-bowl: see ``BOWL_GRASP_STATION`` / ``BOWL_POUR_STATION`` below.
DEFAULT_DROP_WAYPOINT = BaseWaypoint.SINK_STATION

# CLI ``--bowl`` choices → ordered tuple of bowls to run. Each tuple is
# fed straight into ``run_bowl_cycle`` once per entry, so picking
# ``all`` runs the full grasp/pour/drop loop three times — once per
# bowl — without restarting the controller. Order matters: the cart
# (and the Gemini camera framing) lands at the same grasp station each
# time, so the bowls have to be physically arranged left-to-right /
# front-to-back to match this sequence.
#
# ``mixing`` is intentionally NOT included in ``all`` because it's a
# separate ingredient flow (grasps at MIXING_STATION, pours into the
# pan at PAN_STATION) rather than a recipe variant; running it back-to-
# back with the pasta / plastic bowls doesn't make sense as a default.
_BOWL_SEQUENCES: dict[str, tuple[Object, ...]] = {
    "all": (
        Object.PASTA_BOWL,
        Object.PLASTIC_BOWL_BOTTOM,
        Object.PLASTIC_BOWL_TOP,
    ),
    "pasta": (Object.PASTA_BOWL,),
    "plastic_top": (Object.PLASTIC_BOWL_TOP,),
    "plastic_bottom": (Object.PLASTIC_BOWL_BOTTOM,),
    "mixing": (Object.MIXING_BOWL,),
}
DEFAULT_BOWL_CHOICE = "all"

# Per-bowl base routes. Every bowl this controller knows about must
# have an entry in BOTH dicts (asserted below) so ``run_bowl_cycle``
# can drive the cart to the right grasp station and the right pour
# station for each bowl in a sequence. The pasta / plastic bowls grasp
# at PASTA_STATION and pour into the mixing bowl at MIXING_STATION;
# the mixing bowl grasps at MIXING_STATION and pours into the pan at
# PAN_STATION.
BOWL_GRASP_STATION: dict[Object, BaseWaypoint] = {
    Object.PASTA_BOWL: BaseWaypoint.PASTA_STATION,
    Object.PLASTIC_BOWL_TOP: BaseWaypoint.PASTA_STATION,
    Object.PLASTIC_BOWL_BOTTOM: BaseWaypoint.PASTA_STATION,
    Object.MIXING_BOWL: BaseWaypoint.PRE_PAN_STATION,
}
BOWL_POUR_STATION: dict[Object, BaseWaypoint] = {
    Object.PASTA_BOWL: BaseWaypoint.MIXING_STATION,
    Object.PLASTIC_BOWL_TOP: BaseWaypoint.MIXING_STATION,
    Object.PLASTIC_BOWL_BOTTOM: BaseWaypoint.MIXING_STATION,
    Object.MIXING_BOWL: BaseWaypoint.PAN_STATION,
}
# Sanity guard: every CLI choice must resolve to bowls with a known
# grasp + pour route. Catches typos in the dicts at import time.
assert all(
    obj in BOWL_GRASP_STATION and obj in BOWL_POUR_STATION
    for seq in _BOWL_SEQUENCES.values()
    for obj in seq
), (
    "every bowl in _BOWL_SEQUENCES needs BOWL_GRASP_STATION and "
    "BOWL_POUR_STATION entries"
)

# Vertical lift applied after closing on the bowl so it clears the
# counter before the base drives away. Folded into ``grasp.object`` via
# ``lift_dz_m`` so it's a single continuous lift, not a separate gated step.
BOWL_CARRY_LIFT_M = 0.20
# Hardcoded pour pose: directly above ``ARM_HOME_POSITION`` by this much
# in world Z. Used in place of a Gemini mixing-bowl detection — the
# mixing station is fixed at MIXING_STATION so once the base is parked,
# the arm-frame pour pose is deterministic and there's no benefit to
# vision (and a cost when Gemini misses or drifts). Tune by adjusting
# ``ARM_HOME_POSITION`` in ``zitibot_core/constants.py`` (XY component
# of where the bowl ends up relative to the base) and this constant
# (vertical clearance above home for the pour). Sized to roughly match
# the bowl carry height: typical bowl pick z ≈ 0.52 m + 0.20 m carry
# lift = 0.72 m, and home z ≈ 0.645 m, so a +0.08 m lift puts the pour
# pose at z ≈ 0.725 m, near the carried bowl height.
POUR_LIFT_FROM_HOME_M = 0.08
# Post-pour transit lift: after the un-pour, the arm pulls back to
# ARM_HOME and then lifts an extra ``TRANSIT_LIFT_FROM_HOME_M`` in
# world +Z before the base drives to the sink. The lift gives the
# bowl extra clearance over the counter / cart edge during transit.
# Tune up if the bowl scrapes anything on the way; tune down if it
# bumps into upper cabinets / shelves. ``SINK_DROP_DZ_M`` is measured
# from wherever the arm ends up (= ARM_HOME z + this lift), so if
# you change this, retune ``SINK_DROP_DZ_M`` to land at the same
# depth in the basin.
TRANSIT_LIFT_FROM_HOME_M = 0.07
# How far below the carry pose the arm descends at the sink before
# opening the gripper to drop the bowl. Counter / sink basin depth, so
# tune to the rig: too little leaves the bowl hanging above the rim,
# too much drives the EE into the basin floor.
SINK_DROP_DZ_M = 0.30
# How far in front of the carry pose the arm reaches before descending
# at the sink (arm-base +X = forward = away from the cart toward the
# counter). Pushes the drop point further into the basin so the bowl
# clears the rim instead of dropping right at the cart edge. Negative
# would pull the drop back toward the cart; tune to the rig.
SINK_DROP_DX_M = 0.10
# Settle delay after the gripper opens over the sink, before the arm
# starts lifting back to ARM_HOME. The gripper command is fire-and-
# forget on Redis, so without this the arm starts pulling up before
# the fingers have fully retracted — at best the bowl drags up a few
# cm with the gripper, at worst a finger catches the rim and tips
# the bowl. 0.25 s is enough for the Franka hand to fully open at
# the configured speed and let the bowl drop free.
SINK_DROP_SETTLE_S = 0.25
# Pour pivot: rotate around a world point that starts 5 inches (12.7 cm)
# along the EE local +Z axis at the moment the tilt begins. With the
# tool-down grasp orientation, EE local +Z points downward in world,
# so this anchors a point ~12.7 cm below the gripper — roughly where
# the bowl bottom / contents sit — and the gripper orbits around it
# instead of pivoting at the EE control point itself. Tune by changing
# this constant or pass ``--pour-pivot-below-ee-m 0`` to revert to
# pivoting at the EE.
POUR_PIVOT_BELOW_EE_M = 0.127  # 5 in → 12.7 cm

# Loosened tolerances for bowl transit moves. The pre-grasp "move
# above" still uses the bowl spec's tight ``approach_tol`` so the
# pre-grasp framing stays accurate. The descent-to-pick uses a
# slightly loosened ``BOWL_GRASP_DESCENT_TOL_M`` (6 cm vs the bowl
# spec's 4 cm ``grasp_tol``) because the gripper's open width is
# wider than the bowl rim — landing 1-2 cm above the planned pick
# still closes safely around the rim and avoids burning the
# convergence timeout on the last cm of the OTG descent ramp. Once
# we're holding the bowl, every subsequent move is either (a) one of
# the pour waypoints — where the OTG ramp + slow gain settings
# dominate the motion shape, not the endpoint accuracy, and (b)
# transport lifts (post-grasp carry, post-pour transit, return-to-
# home after release) where landing inside a fat tolerance ball is
# fine. Loosening these lets the controller advance to the next move
# as soon as it's "close enough" instead of waiting out the full
# ``arm.move_to`` timeout on the long tail of the OTG ramp.
POUR_WAYPOINT_TOL_M = 0.08
LIFT_TOL_M = 0.08
BOWL_GRASP_DESCENT_TOL_M = 0.06
# Per-bowl world-frame +Z bump added to the Gemini-detected grasp pose
# before it's fed to ``grasp.object``. Lifts the descent target so the
# gripper lands this much higher on the bowl — useful when Gemini's
# detected pose lands a touch below the rim and the fingers want to
# close around the rim itself rather than catch the bowl wall. This
# stacks on top of ``OBJECT_DEFAULTS[bowl].gemini_world_offset_m`` (the
# per-bowl XY/Z offset baked into the bowl spec). Bowls not listed here
# get no extra controller-level Z bump (the mixing bowl already has its
# Z framing baked into its spec offset).
BOWL_GRASP_Z_OFFSETS_M: dict[Object, float] = {
    Object.PASTA_BOWL: 0.0, # 0.20
    Object.PLASTIC_BOWL_TOP: 0.0,
    Object.PLASTIC_BOWL_BOTTOM: 0.0,
    # Mixing bowl: lift the grasp 8 cm above Gemini's detected rim pose
    # so the gripper closes around the rim itself rather than catching
    # the bowl wall below it. Pairs with the bowl spec's per-axis
    # ``gemini_world_offset_m`` (which handles XY framing).
    Object.MIXING_BOWL: 0.00,
}

# Per-bowl pour anchor pose (position, orientation) used by the slerp
# fallback in ``pour_bowl`` when the bowl is NOT registered in
# ``BOWL_POUR_EE_WAYPOINTS``. Bowls missing from this dict default to
# the generic ``(ARM_HOME_POSITION + +Z * POUR_LIFT_FROM_HOME_M,
# grip_R)`` upright anchor (i.e. they pour above the bot's own home
# pose). The mixing bowl is now registered in ``BOWL_POUR_EE_WAYPOINTS``
# so it no longer needs an entry here, but the dict is kept so any
# future bowl using the slerp fallback can register its anchor.
BOWL_POUR_ANCHOR_POSES: dict[Object, tuple[np.ndarray, np.ndarray]] = {}

# Bowls that should SKIP the ``pour.return_upright`` step after the
# slerp tilt. Waypoint-mode bowls never hit this path (they skip slerp
# entirely), so only slerp-fallback bowls need entries here.
BOWLS_SKIP_RETURN_UPRIGHT: set[Object] = set()


def pick_up_bowl(
    ctx: TaskContext, bowl: Object
) -> tuple[np.ndarray, np.ndarray]:
    """Gemini grasp pose → grasp ``bowl`` → lift to carry height.

    Same code path for every bowl: ``gemini.find_grasp_pose`` (which
    bakes in the per-bowl XY/Z framing from ``OBJECT_DEFAULTS[bowl].
    gemini_world_offset_m``) → controller-level ``BOWL_GRASP_Z_OFFSETS_M
    [bowl]`` +Z bump → ``grasp.object``. The Z bump is the only
    controller-level offset; everything else is per-bowl in
    ``OBJECT_DEFAULTS`` so per-bowl tuning lives in one place
    (``zitibot_core/constants.py``).
    """
    pose = gemini.find_grasp_pose(ctx, bowl)
    pick_pos = pose.position.astype(np.float64, copy=True)
    grip_R = pose.orientation
    print(f"Detected {bowl.value} grasp: {pick_pos.tolist()}")
    if pose.rim_yaw_applied and pose.rim_yaw_deg is not None:
        print(f"Detected rim yaw: {pose.rim_yaw_deg:+.2f} deg")
    z_offset = BOWL_GRASP_Z_OFFSETS_M.get(bowl, 0.0)
    if z_offset != 0.0:
        pick_pos[2] += z_offset
        print(
            f"Applied {bowl.value} grasp Z offset {z_offset:+.3f} m → "
            f"{pick_pos.tolist()}"
        )
    grasp.object(
        ctx,
        bowl,
        pick_pos=pick_pos,
        ori=grip_R,
        lift_dz_m=BOWL_CARRY_LIFT_M,
        lift_tol_m=LIFT_TOL_M,
        grasp_tol_m=BOWL_GRASP_DESCENT_TOL_M,
    )
    return pick_pos, grip_R


def pour_bowl(
    ctx: TaskContext,
    bowl: Object,
    grip_R: np.ndarray,
    *,
    tilt_deg: float,
    axis: str,
    duration_s: float,
    pivot_below_ee_m: float = POUR_PIVOT_BELOW_EE_M,
) -> None:
    """Pour ``bowl`` at the parked pour station.

    Two modes, picked automatically based on whether ``bowl`` has an
    entry registered in ``BOWL_POUR_EE_WAYPOINTS``:

    * **Waypoint mode** (preferred). The arm walks forward through the
      registered (pos, ori) waypoints — the first should be an upright
      pose above the mixing bowl, the last a fully tilted pose. There
      is NO reverse pass here; the function returns with the arm
      sitting at the final, fully tilted waypoint. The caller's next
      move (``arm.move_to(ARM_HOME + +Z * TRANSIT_LIFT_FROM_HOME_M)``
      in ``run_bowl_cycle``) handles the return to upright as it lifts
      to the carry pose. ``tilt_deg``, ``axis``, ``duration_s`` and
      ``pivot_below_ee_m`` are ignored in this mode.

    * **Slerp fallback**. Used only when ``bowl`` is not registered in
      ``BOWL_POUR_EE_WAYPOINTS``. The arm moves to the bowl's pour
      anchor pose — ``BOWL_POUR_ANCHOR_POSES[bowl]`` if present,
      otherwise ``(ARM_HOME_POSITION + +Z * POUR_LIFT_FROM_HOME_M,
      grip_R)`` — then slerps to a tilted orientation about a world pivot
      ``pivot_below_ee_m`` along the EE local +Z axis. Pass
      ``pivot_below_ee_m=0`` to rotate around the EE control point
      instead. By default the slerp tilt is followed by an explicit
      ``pour.return_upright`` so the bowl is upright before the caller's
      next move; bowls in ``BOWLS_SKIP_RETURN_UPRIGHT`` skip that step
      and rely on the caller's transit lift to interpolate the
      orientation back to upright.

    Final state: gripper still closed, arm holding ``bowl`` either at
    the final tilted pour pose (waypoint mode, or slerp+skip-return),
    or upright at the fallback pour target (slerp mode with return).
    The caller then lifts to the carry pose, drives the base to the
    sink, and releases there.
    """
    waypoints = BOWL_POUR_EE_WAYPOINTS.get(bowl)
    if waypoints is not None:
        _pour_via_waypoints(ctx, bowl, waypoints)
        return

    spec = OBJECT_DEFAULTS[bowl]
    anchor = BOWL_POUR_ANCHOR_POSES.get(bowl)
    if anchor is not None:
        pour_target = np.asarray(anchor[0], dtype=np.float64).reshape(3).copy()
        anchor_R = np.asarray(anchor[1], dtype=np.float64).reshape(3, 3)
        print(
            f"[pour:{bowl.value}] taught pour anchor pose: "
            f"pos={pour_target.tolist()}"
        )
    else:
        pour_target = ARM_HOME_POSITION + np.array(
            [0.0, 0.0, POUR_LIFT_FROM_HOME_M], dtype=np.float64
        )
        anchor_R = grip_R
        print(
            f"[pour:{bowl.value}] no registered EE waypoints / anchor; "
            f"using slerp tilt at home + {POUR_LIFT_FROM_HOME_M * 100:.1f} "
            f"cm Z = {pour_target.tolist()}"
        )
    arm.move_to(
        ctx,
        pour_target,
        anchor_R,
        label=f"[pour] move to pour target {pour_target.tolist()}",
        tol_m=spec.approach_tol,
    )
    pivot_offset_local: np.ndarray | None = None
    if abs(pivot_below_ee_m) > 1e-9:
        pivot_offset_local = np.array(
            [0.0, 0.0, pivot_below_ee_m], dtype=np.float64
        )
        print(
            f"[pour] pivot offset = {pivot_below_ee_m * 100:.1f} cm along "
            f"EE local +Z (below the gripper when tool points down)"
        )
    R_poured = pour.into(
        ctx,
        pour_target,
        tilt_deg=tilt_deg,
        axis=axis,
        duration_s=duration_s,
        pivot_offset_local=pivot_offset_local,
    )
    if bowl in BOWLS_SKIP_RETURN_UPRIGHT:
        print(
            f"[pour:{bowl.value}] skipping return_upright; transit lift "
            f"will un-pour as it returns to upright."
        )
        return
    pour.return_upright(
        ctx,
        pour_target,
        R_poured,
        anchor_R,
        duration_s=duration_s,
        pivot_offset_local=pivot_offset_local,
    )


def _pour_via_waypoints(
    ctx: TaskContext,
    bowl: Object,
    waypoints: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> None:
    """Sweep the EE through ``waypoints`` from upright to fully tilted.

    Forward-only: visit every waypoint in order from the upright start
    pose to the fully tilted final pose. There is no return/un-pour
    step — the caller is expected to immediately move from the final
    pour pose to the carry waypoint (``ARM_HOME + +Z * TRANSIT_LIFT_
    FROM_HOME_M``), which doubles as the un-pour (the bowl returns
    upright on its way to the carry pose). Skipping the explicit
    return-to-start move avoids one redundant arm move per pour and
    one extra orientation reversal.

    Each step uses ``arm.move_to`` with ``POUR_WAYPOINT_TOL_M`` (the
    loose transit tolerance) rather than the bowl's tight
    ``spec.approach_tol``. The pour waypoints shape the bowl's tilt
    trajectory; exact endpoint accuracy doesn't matter as long as the
    OTG path between them looks right, so we'd rather advance to the
    next waypoint promptly than burn the convergence timeout on the
    long tail of the OTG ramp.
    """
    if not waypoints:
        raise ValueError(f"[pour:{bowl.value}] empty waypoint sequence")

    forward = list(waypoints)
    print(
        f"[pour:{bowl.value}] EE waypoint sweep: "
        f"{len(forward)} forward (no un-pour; caller lifts to carry pose) "
        f"(tol={POUR_WAYPOINT_TOL_M * 100:.1f} cm)"
    )
    for idx, (pos, ori) in enumerate(forward, start=1):
        label = (
            f"[pour:{bowl.value}] waypoint {idx}/{len(forward)} "
            f"(pour) pos={pos.tolist()}"
        )
        arm.move_to(ctx, pos, ori, label=label, tol_m=POUR_WAYPOINT_TOL_M)


def drop_at_sink(
    ctx: TaskContext,
    bowl: Object,
    grip_R: np.ndarray,
    *,
    drop_dz_m: float,
    drop_dx_m: float,
) -> None:
    """Reach forward, lower the held ``bowl``, open the gripper, return home.

    Assumes the base has already driven to ``SINK_STATION`` and the arm is
    still holding the bowl. We read the live EE position, push the drop
    pose ``drop_dx_m`` forward (arm-base +X, away from the cart) and
    ``drop_dz_m`` below the carry pose so the bowl clears the sink rim
    before being released. After the gripper opens we go straight to
    ``ARM_HOME`` rather than back up to the carry pose — the bowl is gone
    so there's no reason to retrace the descent.
    """
    spec = OBJECT_DEFAULTS[bowl]
    cur_pose = arm.read_current_ee_world(ctx.redis)
    if cur_pose is None:
        raise RuntimeError("[drop] cannot read current EE pose; aborting sink drop.")
    carry_pos = cur_pose[0]
    drop_pos = carry_pos + np.array([drop_dx_m, 0.0, -drop_dz_m], dtype=np.float64)
    print(
        f"[drop] {bowl.value}: current EE = {carry_pos.tolist()}, "
        f"reaching {drop_dx_m * 100:+.1f} cm forward and "
        f"descending {drop_dz_m * 100:.1f} cm to {drop_pos.tolist()}"
    )
    arm.move_to(
        ctx,
        drop_pos,
        grip_R,
        label=f"[drop] descend at sink {drop_pos.tolist()}",
        tol_m=spec.grasp_tol,
    )
    gripper.open_gripper(
        ctx.redis,
        spec.open_width,
        speed=spec.speed,
        force=spec.force,
        use_max_mode=True,
    )
    print(f"[drop] gripper opened — {bowl.value} released into sink.")
    time.sleep(SINK_DROP_SETTLE_S)

    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[drop] return to home {ARM_HOME_POSITION.tolist()}",
        tol_m=LIFT_TOL_M,
    )


def run_bowl_cycle(
    ctx: TaskContext,
    bowl: Object,
    *,
    grasp_x: float | None = None,
    grasp_y: float | None = None,
    grasp_yaw_deg: float | None = None,
    pour_x: float | None = None,
    pour_y: float | None = None,
    pour_yaw_deg: float | None = None,
    pour_tilt_deg: float = DEFAULT_POUR_TILT_DEG,
    pour_axis: str = DEFAULT_POUR_AXIS,
    pour_duration_s: float = DEFAULT_TILT_DURATION_S,
    pour_pivot_below_ee_m: float = POUR_PIVOT_BELOW_EE_M,
    sink_drop_dz_m: float = SINK_DROP_DZ_M,
    sink_drop_dx_m: float = SINK_DROP_DX_M,
    gemini_response_path: str | Path | None = None,
) -> None:
    """One full grasp → pour → sink-drop cycle for ``bowl``.

    Self-contained so it can be called back-to-back for multiple bowls::

        for bowl in _BOWL_SEQUENCES["all"]:
            run_bowl_cycle(ctx, bowl)

    Sequence (matches steps 1–9 of this controller's module docstring):

    1. Arm → ``ARM_HOME`` (flange-camera framing pose). Safe no-op when
       already there (``drop_at_sink`` ends here, so subsequent calls in
       a loop skip the move at the tolerance check).
    2. Base → ``BOWL_GRASP_STATION[bowl]`` (PASTA_STATION for the pasta
       / plastic bowls, MIXING_STATION for the mixing bowl; per-axis
       overrides via ``grasp_x/y/yaw_deg``).
    3. Gemini detects ``bowl``; ``grasp.object`` closes and lifts.
    4. Base → ``BOWL_POUR_STATION[bowl]`` (MIXING_STATION for the
       pasta / plastic bowls, PAN_STATION for the mixing bowl;
       per-axis overrides via ``pour_x/y/yaw_deg``).
    5. ``pour_bowl`` runs the bowl's registered EE waypoint sweep, or
       the slerp fallback against the bowl's pour anchor pose if it has
       no registered waypoints.
    6. Arm → ``ARM_HOME + +Z * TRANSIT_LIFT_FROM_HOME_M`` carry pose.
    7. Base → ``SINK_STATION``.
    8. ``drop_at_sink`` descends, opens the gripper, returns to
       ``ARM_HOME``.

    Same per-axis base override flags as the CLI — pass ``None`` to use
    the per-bowl station default. If you're looping over bowls and want
    the same overrides every call, pass them every time; the cycle does
    not remember state between calls.
    """
    grasp_station = BOWL_GRASP_STATION[bowl]
    pour_station = BOWL_POUR_STATION[bowl]
    # Resolve the per-bowl Gemini response save path here so callers that
    # invoke ``run_bowl_cycle`` directly (e.g. the kitchen state machine)
    # don't have to remember to set ``ctx.gemini_response_path`` themselves
    # before each call. ``main()`` sets it explicitly per-bowl too; this
    # honors any explicit override.
    resolved_gem_path = _resolve_gemini_response_path(bowl, gemini_response_path)
    ctx.gemini_response_path = str(resolved_gem_path)
    print(
        f"\n=== bowl cycle: {bowl.value} "
        f"(grasp @ {grasp_station.name}, pour @ {pour_station.name}, "
        f"gemini→{resolved_gem_path}) ===",
        flush=True,
    )

    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[arm] reset to home before cycle {ARM_HOME_POSITION.tolist()}",
        tol_m=LIFT_TOL_M,
    )

    _drive_base(
        ctx,
        grasp_station,
        x_override=grasp_x,
        y_override=grasp_y,
        yaw_override=grasp_yaw_deg,
    )

    _, grip_R = pick_up_bowl(ctx, bowl)

    _drive_base(
        ctx,
        pour_station,
        x_override=pour_x,
        y_override=pour_y,
        yaw_override=pour_yaw_deg,
    )

    pour_bowl(
        ctx,
        bowl,
        grip_R,
        tilt_deg=pour_tilt_deg,
        axis=pour_axis,
        duration_s=pour_duration_s,
        pivot_below_ee_m=pour_pivot_below_ee_m,
    )

    transit_carry_pos = ARM_HOME_POSITION + np.array(
        [0.0, 0.0, TRANSIT_LIFT_FROM_HOME_M], dtype=np.float64
    )
    arm.move_to(
        ctx,
        transit_carry_pos,
        ARM_HOME_ORIENTATION,
        label=(
            f"[arm] post-pour lift to {TRANSIT_LIFT_FROM_HOME_M * 100:.1f} cm "
            f"above home {transit_carry_pos.tolist()} (carry pose for "
            f"sink drive — straight from last pour pose)"
        ),
        tol_m=LIFT_TOL_M,
    )

    _drive_base(
        ctx,
        DEFAULT_DROP_WAYPOINT,
        x_override=None,
        y_override=None,
        yaw_override=None,
    )
    drop_at_sink(
        ctx,
        bowl,
        grip_R,
        drop_dz_m=sink_drop_dz_m,
        drop_dx_m=sink_drop_dx_m,
    )

    print(f"=== bowl cycle: {bowl.value} complete ===\n", flush=True)


def _resolve_base_target(
    waypoint: BaseWaypoint,
    x_override: float | None,
    y_override: float | None,
    yaw_override: float | None,
) -> tuple[float, float, float, str]:
    """Mix the waypoint defaults with any per-axis CLI override."""
    base_pose = BASE_WAYPOINTS[waypoint]
    x_m = base_pose.x_m if x_override is None else float(x_override)
    y_m = base_pose.y_m if y_override is None else float(y_override)
    yaw_deg = base_pose.yaw_deg if yaw_override is None else float(yaw_override)
    overridden = (
        x_override is not None
        or y_override is not None
        or yaw_override is not None
    )
    if overridden:
        label = (
            f"[base] {waypoint.name} (override) -> "
            f"({x_m:.3f}, {y_m:.3f}, {yaw_deg:.1f} deg)"
        )
    else:
        label = (
            f"[base] {waypoint.name} -> "
            f"({x_m:.3f}, {y_m:.3f}, {yaw_deg:.1f} deg)"
        )
    return x_m, y_m, yaw_deg, label


def _drive_base(
    ctx: TaskContext,
    waypoint: BaseWaypoint,
    *,
    x_override: float | None,
    y_override: float | None,
    yaw_override: float | None,
) -> None:
    overridden = (
        x_override is not None
        or y_override is not None
        or yaw_override is not None
    )
    if not overridden:
        base.go_to_pose(ctx, waypoint)
        return
    x_m, y_m, yaw_deg, label = _resolve_base_target(
        waypoint, x_override, y_override, yaw_override
    )
    base.go_to_pose(ctx, x_m=x_m, y_m=y_m, yaw_deg=yaw_deg, label=label)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "OptiTrack base + Gemini vision grasp + pour + release "
            "(uses only zitibot_core / zitibot_tasks)."
        )
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--bowl",
        choices=tuple(_BOWL_SEQUENCES.keys()),
        default=DEFAULT_BOWL_CHOICE,
        help=(
            "Which bowl(s) to grasp / pour. ``all`` (default) runs the full "
            "cycle three times in order pasta → plastic_bottom → plastic_top "
            "(grasp PASTA_STATION → pour MIXING_STATION → sink). ``pasta`` "
            "is the black pasta bowl; ``plastic_top`` / ``plastic_bottom`` "
            "select between two plastic bowls in the camera frame by which "
            "is higher / lower in the image — those three share the pasta "
            "bowl's grasp + pour route. ``mixing`` runs the mixing bowl flow "
            "instead (grasp MIXING_STATION → pour PAN_STATION → sink); it's "
            "intentionally not part of ``all`` because it's a separate "
            "ingredient flow."
        ),
    )
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate every motion / gripper / base step.",
    )
    p.add_argument(
        "--log",
        action="store_true",
        help=(
            "Record per-move EE position-vs-time plots to "
            "logs/graphs/<controller>_NNNN/. Each arm.move_to writes one PNG."
        ),
    )

    # Base targets — per-bowl stations (``BOWL_GRASP_STATION`` /
    # ``BOWL_POUR_STATION``) by default, raw Opti overrides optional.
    # Overrides apply to every bowl in the selected sequence, so they
    # only make sense when running a single-bowl sequence (e.g. ``--bowl
    # mixing`` or ``--bowl pasta``) — passing them with ``--bowl all``
    # would force the same grasp / pour pose for all three bowls.
    p.add_argument(
        "--grasp-x", type=float, default=None,
        help=(
            "Override grasp base X (m). Default is the bowl's station X "
            "(BOWL_GRASP_STATION[bowl].x)."
        ),
    )
    p.add_argument(
        "--grasp-y", type=float, default=None,
        help="Override grasp base Y (m).",
    )
    p.add_argument(
        "--grasp-yaw-deg", type=float, default=None,
        help="Override grasp base yaw (deg).",
    )
    p.add_argument(
        "--pour-x", type=float, default=None,
        help=(
            "Override pour base X (m). Default is the bowl's station X "
            "(BOWL_POUR_STATION[bowl].x)."
        ),
    )
    p.add_argument(
        "--pour-y", type=float, default=None,
        help="Override pour base Y (m).",
    )
    p.add_argument(
        "--pour-yaw-deg", type=float, default=None,
        help="Override pour base yaw (deg).",
    )

    # Pour / sink-drop params.
    p.add_argument("--pour-tilt-deg", type=float, default=DEFAULT_POUR_TILT_DEG)
    p.add_argument("--pour-axis", default=DEFAULT_POUR_AXIS, choices=("x", "y"))
    p.add_argument("--tilt-duration-s", type=float, default=DEFAULT_TILT_DURATION_S)
    p.add_argument(
        "--pour-pivot-below-ee-m",
        type=float,
        default=POUR_PIVOT_BELOW_EE_M,
        help=(
            "Distance along the EE local +Z axis (below the gripper when "
            "tool points down) of the world point the pour rotates around. "
            "0 disables — pour rotates around the EE control point itself."
        ),
    )
    p.add_argument(
        "--sink-drop-dz-m", type=float, default=SINK_DROP_DZ_M,
        help="How far to lower the bowl at SINK_STATION before opening (m).",
    )
    p.add_argument(
        "--sink-drop-dx-m", type=float, default=SINK_DROP_DX_M,
        help=(
            "How far forward (arm-base +X, away from the cart) to reach at "
            "SINK_STATION before descending. Pushes the drop into the basin."
        ),
    )
    # Plumbing.
    p.add_argument(
        "--endeffector-transform-key",
        default=DEFAULT_ENDEFFECTOR_TRANSFORM_KEY,
        help="Redis key for the 4x4 base->flange transform.",
    )
    p.add_argument(
        "--gemini-response-path",
        default=None,
        help=(
            "Where to save the annotated Gemini RGB+depth response image. "
            "If unset, defaults per bowl from ``GEMINI_RESPONSE_PATHS`` "
            f"(falling back to {DEFAULT_GEMINI_RESPONSE_PATH} for bowls "
            "without a dedicated entry)."
        ),
    )
    return p.parse_args()


def _resolve_gemini_response_path(
    bowl: Object, override: str | Path | None
) -> Path:
    """Pick the Gemini response save path for ``bowl``, honoring CLI override."""
    if override is not None:
        return Path(override)
    return GEMINI_RESPONSE_PATHS.get(bowl, DEFAULT_GEMINI_RESPONSE_PATH)


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key

    bowls: tuple[Object, ...] = _BOWL_SEQUENCES[args.bowl]
    print(f"Step mode: {'on' if args.step else 'off'}")
    print(
        f"Bowl sequence: {args.bowl} → "
        f"[{', '.join(b.value for b in bowls)}]"
    )

    # Print per-bowl planned routes up front so the operator can spot a
    # bad station mapping before the cart actually moves anywhere. With
    # per-bowl routes the old single-line grasp/pour summary doesn't
    # work — different bowls in a sequence can land at different
    # stations.
    _, _, _, sink_label = _resolve_base_target(
        DEFAULT_DROP_WAYPOINT, None, None, None
    )
    for bowl in bowls:
        grasp_station = BOWL_GRASP_STATION[bowl]
        pour_station = BOWL_POUR_STATION[bowl]
        _, _, _, grasp_label = _resolve_base_target(
            grasp_station, args.grasp_x, args.grasp_y, args.grasp_yaw_deg
        )
        _, _, _, pour_label = _resolve_base_target(
            pour_station, args.pour_x, args.pour_y, args.pour_yaw_deg
        )
        gem_path = _resolve_gemini_response_path(bowl, args.gemini_response_path)
        print(
            f"[plan:{bowl.value}] grasp={grasp_label}  pour={pour_label}  "
            f"gemini→{gem_path}"
        )
    print(f"Sink target:  {sink_label}")

    # Warm up the RealSense before any motion so the first Gemini call
    # in pick_up_bowl doesn't make the arm sit idle through the 30-frame
    # warmup. ctx.realsense() is cached, so this is a one-time cost paid
    # here at startup rather than inline during the first bowl cycle.
    # Fails fast and aborts the run if the camera can't stream, which
    # is the desired behavior (don't drive the base to a station only
    # to discover the camera is unplugged).
    print("Warming up RealSense before first bowl cycle...")
    ctx.realsense()

    completed: list[str] = []
    try:
        for idx, bowl in enumerate(bowls, start=1):
            print(
                f"\n### bowl {idx}/{len(bowls)}: {bowl.value} "
                f"(sequence={args.bowl}) ###",
                flush=True,
            )
            # Point Gemini's annotated image at the bowl-specific path so
            # each cycle in a multi-bowl sequence writes to its own file
            # instead of clobbering the previous bowl's overlay. The CLI
            # ``--gemini-response-path`` (if passed) overrides for every
            # bowl in the sequence.
            ctx.gemini_response_path = str(
                _resolve_gemini_response_path(bowl, args.gemini_response_path)
            )
            run_bowl_cycle(
                ctx,
                bowl,
                grasp_x=args.grasp_x,
                grasp_y=args.grasp_y,
                grasp_yaw_deg=args.grasp_yaw_deg,
                pour_x=args.pour_x,
                pour_y=args.pour_y,
                pour_yaw_deg=args.pour_yaw_deg,
                pour_tilt_deg=args.pour_tilt_deg,
                pour_axis=args.pour_axis,
                pour_duration_s=args.tilt_duration_s,
                pour_pivot_below_ee_m=args.pour_pivot_below_ee_m,
                sink_drop_dz_m=args.sink_drop_dz_m,
                sink_drop_dx_m=args.sink_drop_dx_m,
            )
            completed.append(bowl.value)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted. Completed {len(completed)}/{len(bowls)} bowls: "
            f"[{', '.join(completed) or '<none>'}]."
        )
        return 130
    finally:
        ctx.stop_realsense()

    print(
        f"Grasp + pour + sink drop complete for "
        f"{len(completed)}/{len(bowls)} bowls: [{', '.join(completed)}]."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
