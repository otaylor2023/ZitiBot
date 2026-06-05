#!/usr/bin/env python3
"""Multi-station egg-cracker flow with Gemini vision grasp.

End-to-end sequence (``--vision``, default):

  1. Arm → home pose.
  2. Base → ``INGREDIENT_STATION``.
  3. Arm → ``EGG_CRACKER_DETECTION_EE_POSITION`` for camera framing.
  4. Gemini detects the gray strip with a red cross-mark on the cracker
     handle and returns a perpendicular-yaw grasp pose (see
     ``zitibot_tasks.gemini.find_grasp_pose`` + ``_build_pose_cylinder``).
  5. ``grasp.object`` closes on the cracker and lifts ~20 cm so it
     clears the counter for transit.
  6. Base → ``MIXING_STATION``.
  7. Arm → ``EGG_CRACK_OVER_MIXING_BOWL_POSITION`` (cracker held above
     the mixing bowl).
  8. ``egg_crack.crack`` squeezes the cracker (egg falls into the bowl).
  9. Arm → ``ARM_HOME + +Z * TRANSIT_LIFT_FROM_HOME_M`` carry pose.
 10. Base → ``SINK_STATION``.
 11. ``_drop_at_sink`` reaches forward + down from the carry pose, opens
     the gripper, and lifts back to ``ARM_HOME``.

Use ``--no-vision`` to fall back to the original single-station
``egg_crack.run`` (no base motion, static pick from
``OBJECT_DEFAULTS[Object.EGG_CRACKER].pick_pose``).

Usage::

  # Full vision + multi-station flow, ENTER-gated.
  ./ZitiBot/launch_zitibot_full.sh controllers/egg_crack_controller.py -- --step

  # Same flow but skip the base drive to INGREDIENT_STATION (arm-only
  # test when the cart is already parked in front of the cracker).
  ./ZitiBot/launch_zitibot_full.sh controllers/egg_crack_controller.py -- \\
      --skip-base --step

  # Legacy arm-only flow (no base motion, no sink drop).
  python ZitiBot/controllers/egg_crack_controller.py --no-vision

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
TidyBot base ``redis_driver``, RealSense, OptiTrack on Redis, and
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm, base, gains, gripper
from zitibot_core.constants import (
    ARM_HOME_ORIENTATION,
    ARM_HOME_POSITION,
    DEFAULT_POS_TOL_M,
    EGG_CRACKER_DETECTION_EE_ORIENTATION,
    EGG_CRACKER_DETECTION_EE_POSITION,
    HOME_POS_TOL_M,
    OBJECT_DEFAULTS,
    PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
    PRECISE_GRASP_MAX_LINEAR_VELOCITY,
    PRECISE_GRASP_MOVE_TIMEOUT_S,
    PRECISE_GRASP_ORIENTATION_KP,
    PRECISE_GRASP_POSITION_KP,
    BaseWaypoint,
    Object,
)
from zitibot_core.context import TaskContext, make_context
from zitibot_core.runner import step_gate
from zitibot_tasks import egg_crack, gemini, grasp

DEFAULT_ENDEFFECTOR_TRANSFORM_KEY = "opensai::redis_driver::FrankaRobot::T_end_effector"
DEFAULT_GEMINI_RESPONSE_PATH = _CONTROLLERS.parent / "logs" / "gemini_response_egg_cracker.png"

# Cartesian ``arm.move_to`` convergence budget for this controller. The
# default in ``zitibot_core.arm`` is 4 s, which is too tight for the
# long wrist re-orientations between the detection pose, the taught
# above-bowl pose, and the transit carry pose.

# Post-grasp carry lift. After ``grasp.object`` closes on the cracker,
# the arm lifts this far in +Z before the base drives to the mixing
# station. 20 cm matches the bowl carry lift in ``bowl_pour_controller``
# — enough to clear the ingredient station rim on the way out.
EGG_CRACKER_CARRY_LIFT_M = 0.20

# Camera standoff used only between Gemini #1 and Gemini #2. This is higher
# than the final pre-grasp approach height so the D405 sees the cracker in its
# best depth band without the open gripper occluding the handles.
EGG_CRACKER_REFINE_STANDOFF_M = 0.11

# Pose above the mixing bowl where the cracker is held during the
# squeeze. Hand-taught at the bench at ``MIXING_STATION``: jogged the
# held cracker until its egg-drop opening was centered over the mixing
# bowl, then logged the live ``[arm] startup`` print from Redis. Re-
# record the same way if the mixing bowl / cart shifts. Both the
# position AND the orientation come from the taught pose — we drive the
# wrist to this orientation (not Gemini's perpendicular yaw) so the
# cracker's egg-drop opening points squarely down into the bowl.
EGG_CRACK_OVER_MIXING_BOWL_POSITION = np.array(
    [0.2560, -0.0284, 0.6480], dtype=np.float64
)
EGG_CRACK_OVER_MIXING_BOWL_ORIENTATION = np.array(
    [
        [+0.6614, -0.7441, +0.0941],
        [-0.7425, -0.6673, -0.0577],
        [+0.1058, -0.0317, -0.9939],
    ],
    dtype=np.float64,
)

# Post-crack shake: dislodge the cracked egg / shell off the cracker
# fingers by wobbling J0 (the base joint) ±delta for a few cycles. Same overall pattern
# as the jar pour's J4 shake in ``grasp_and_pour_jar_controller`` — the
# joint-space "fake" warmup nudge first forces the cartesian→joint
# controller swap to take effect (otherwise the first real shake stroke
# tends to be eaten by the swap and the egg ends up still glued to the
# cracker), then ``..._CYCLES`` UP/DOWN strokes of ``..._DELTA_DEG`` go
# through cleanly. Each stroke is RELATIVE to the live joint reading at
# the start of that stroke, so the DOWN stroke always returns to where
# its matching UP stroke started even if a stroke doesn't perfectly
# converge.
EGG_CRACK_SHAKE_J0_WARMUP_DELTA_DEG = 3.0
EGG_CRACK_SHAKE_J0_WARMUP_TIMEOUT_S = 2.0
EGG_CRACK_SHAKE_J0_WARMUP_TOL_RAD = 0.05
EGG_CRACK_SHAKE_J0_DELTA_DEG = 6.0
EGG_CRACK_SHAKE_J0_CYCLES = 3
# Loosened from 0.05 → 0.15 rad so each stroke advances quickly rather
# than burning the convergence timeout on PID tail; mirrors the jar
# shake's loosened tolerance. This is fine for the larger (6°) crack
# shake — the smaller post-release shake uses its own tighter tolerance
# (``EGG_CRACK_RELEASE_SHAKE_TOL_RAD``) so its 2° strokes actually move.
EGG_CRACK_SHAKE_J0_TOL_RAD = 0.15
EGG_CRACK_SHAKE_J0_TIMEOUT_S = 3.0

# Transit carry lift between MIXING_STATION → SINK_STATION. Matches the
# bowl_pour_controller value: ARM_HOME + this much in +Z gives a transit
# pose that's a few cm above home so the cracker clears the counter on
# its way to the sink.

# Sink drop geometry — same convention as bowl_pour_controller /
# mixing_vision_base_controller. ``SINK_DROP_DX_M`` reaches forward
# (arm-base +X) so the drop point clears the cart edge; ``SINK_DROP_DZ_M``
# descends below the carry pose so the cracker lands inside the basin.
SINK_DROP_DZ_M = 0.20
SINK_DROP_DX_M = 0.10
# Pause AFTER ``gripper.open_gripper`` and BEFORE the next arm move
# (the return-to-home lift in ``run_egg_crack_cycle``). 1.0 s gives
# the cracker time to actually fall out of the open jaws and settle
# in the basin before the arm starts lifting — otherwise the lift
# can drag a stuck-on cracker back up with it.
SINK_DROP_SETTLE_S = 1.0

# ---------------------------------------------------------------------------
# Post-grasp egg-crack sequence geometry.
#
# Everything below is expressed as an offset from ``ARM_HOME_POSITION`` in
# the arm/base frame (+X = forward, +Y = left, -Y = right, +Z = up) and runs
# from a SINGLE base position — no base motion between the grasp, the crack,
# and the release. The bowl and the cracker's pick spot are both assumed to
# be reachable from where the vision grasp happened.
#
# Sequence (after the cracker is grasped and lifted):
#   1. Lift back to ARM_HOME.
#   2. Above the bowl = home + (forward, left, -drop).
#   3. Descend ``..._BOWL_DOWN_M`` into the bowl.
#   4. Crack.  5. Shake.
#   6. Move to the pour spot: ``..._EMPTY_AHEAD_M`` back (-X) and
#      ``..._POUR_UP_M`` above the crack pose.
#   7. Tilt ``..._EMPTY_TILT_DEG`` about world +X to empty the cracker.
#   8. Right itself (back to home orientation, same position).
#   9. Return to the pick spot, release.
#  10. Lift straight up ``..._RELEASE_LIFT_M``, then return to ARM_HOME.
EGG_CRACK_BOWL_FORWARD_M = 0.05   # +X toward the bowl (crack pose 5 cm back vs. the old 0.10)
EGG_CRACK_BOWL_LEFT_M = 0.25      # +Y toward the bowl
# Height of the above-bowl pose BELOW home (-Z). Lowering this drops the
# above-bowl waypoint; ``..._BOWL_DOWN_M`` is reduced by the same amount so
# the final crack depth is unchanged.
EGG_CRACK_BOWL_DROP_FROM_HOME_M = 0.10  # -Z, above-bowl pose sits this far below home
EGG_CRACK_BOWL_DOWN_M = 0.10      # -Z descent into the bowl before the squeeze
# Crack and pour spots sit ``EGG_CRACK_EMPTY_AHEAD_M`` apart along +X. The two
# were swapped vs. the bowl layout after testing, so the CRACK now happens at
# the FORWARD spot (bowl-left base + this offset) and the POUR happens back at
# the bowl-left base spot, lifted ``EGG_CRACK_POUR_UP_M`` above the crack so the
# shell clears the rim on the way out.
EGG_CRACK_EMPTY_AHEAD_M = 0.20    # +X separation between the crack and pour spots
EGG_CRACK_POUR_UP_M = 0.10        # +Z lift of the pour spot above the crack height
EGG_CRACK_EMPTY_TILT_DEG = 135.0  # world-+X rotation to dump shell/egg residue
EGG_CRACK_RELEASE_LIFT_M = 0.10   # +Z straight-up lift after releasing the cracker

# After the gripper opens, nudge up a touch and give a tiny J0 shake so the
# cracker drops free of the open jaws before the full lift away.
EGG_CRACK_RELEASE_NUDGE_UP_M = 0.01      # +Z nudge right after release
EGG_CRACK_RELEASE_SHAKE_DELTA_DEG = 2.0  # per-stroke J0 amplitude for the post-release shake
EGG_CRACK_RELEASE_SHAKE_CYCLES = 3       # UP/DOWN J0 cycles after release
# The post-release shake is small (2°), so it needs a much tighter stroke
# tolerance than the 6° crack shake — at the crack shake's 0.15 rad ≈ 8.6°
# tol a 2° stroke is "already converged" and never moves. 0.015 rad ≈ 0.86°
# forces J0 to actually drive to the target; the timeout bounds the worst case.
EGG_CRACK_RELEASE_SHAKE_TOL_RAD = 0.0199
EGG_CRACK_RELEASE_SHAKE_TIMEOUT_S = 0.5
# For the post-release shake ONLY: instead of the "fake"/warmup nudge that
# forces the cartesian→joint controller swap, seed the current joints as the
# goal (zero motion), switch controllers, and wait this long for the swap to
# settle before the first stroke. Testing whether the wait removes the need
# for the fake command.
EGG_CRACK_RELEASE_SHAKE_POST_SWITCH_WAIT_S = 0.15


def parse_args() -> argparse.Namespace:
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    p = argparse.ArgumentParser(
        description="Multi-station egg cracker sequence with optional Gemini vision grasp."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate every motion / gripper / base step.",
    )

    # Mode selection.
    p.add_argument(
        "--vision",
        dest="vision",
        action="store_true",
        default=True,
        help=(
            "Use Gemini to detect the gray strip (with red across it) on "
            "the egg cracker and grasp perpendicular to the strip. Default on."
        ),
    )
    p.add_argument(
        "--no-vision",
        dest="vision",
        action="store_false",
        help=(
            "Skip Gemini detection AND the multi-station flow; run the "
            "legacy arm-only ``egg_crack.run`` against the static pick pose."
        ),
    )
    p.add_argument(
        "--skip-base",
        action="store_true",
        help=(
            "Do not drive the base to INGREDIENT_STATION / MIXING_STATION / "
            "SINK_STATION (vision mode only). Use when the cart is already "
            "parked in front of the cracker for arm-only debugging."
        ),
    )

    # Static-pose overrides (used when --no-vision).
    p.add_argument("--pick-x", type=float, default=float(spec.pick_pose[0]))
    p.add_argument("--pick-y", type=float, default=float(spec.pick_pose[1]))
    p.add_argument("--pick-z", type=float, default=float(spec.pick_pose[2]))
    p.add_argument("--drop-x", type=float, default=None)
    p.add_argument("--drop-y", type=float, default=None)
    p.add_argument("--drop-z", type=float, default=None)

    # Detection-pose override. Defaults to ``EGG_CRACKER_DETECTION_EE_POSITION``.
    p.add_argument(
        "--detection-x",
        type=float,
        default=float(EGG_CRACKER_DETECTION_EE_POSITION[0]),
        help="Camera-framing EE X for Gemini detection (vision mode).",
    )
    p.add_argument(
        "--detection-y",
        type=float,
        default=float(EGG_CRACKER_DETECTION_EE_POSITION[1]),
        help="Camera-framing EE Y for Gemini detection (vision mode).",
    )
    p.add_argument(
        "--detection-z",
        type=float,
        default=float(EGG_CRACKER_DETECTION_EE_POSITION[2]),
        help="Camera-framing EE Z for Gemini detection (vision mode).",
    )

    # Crack-pose override (position only; orientation is fixed to
    # ``EGG_CRACK_OVER_MIXING_BOWL_ORIENTATION`` — re-record the constant
    # if the taught pose needs to change).
    p.add_argument(
        "--crack-x",
        type=float,
        default=float(EGG_CRACK_OVER_MIXING_BOWL_POSITION[0]),
        help="EE X where the cracker is squeezed (arm frame, m).",
    )
    p.add_argument(
        "--crack-y",
        type=float,
        default=float(EGG_CRACK_OVER_MIXING_BOWL_POSITION[1]),
        help="EE Y where the cracker is squeezed (arm frame, m).",
    )
    p.add_argument(
        "--crack-z",
        type=float,
        default=float(EGG_CRACK_OVER_MIXING_BOWL_POSITION[2]),
        help="EE Z where the cracker is squeezed (arm frame, m).",
    )

    # Post-crack J0 (base) shake tunables.
    p.add_argument(
        "--shake-delta-deg",
        type=float,
        default=EGG_CRACK_SHAKE_J0_DELTA_DEG,
        help="Per-stroke J0 (base) amplitude (deg) for the post-crack shake.",
    )
    p.add_argument(
        "--shake-cycles",
        type=int,
        default=EGG_CRACK_SHAKE_J0_CYCLES,
        help="Number of UP/DOWN J0 (base) shake cycles after the crack.",
    )
    p.add_argument(
        "--no-shake",
        action="store_true",
        help="Skip the post-crack J0 (base) shake (and its warmup nudge).",
    )

    # Lift / drop tunables.
    p.add_argument(
        "--carry-lift-m",
        type=float,
        default=EGG_CRACKER_CARRY_LIFT_M,
        help="Post-grasp lift height above the pick pose (m).",
    )
    p.add_argument(
        "--refine-standoff-m",
        type=float,
        default=EGG_CRACKER_REFINE_STANDOFF_M,
        help=(
            "Height above the coarse Gemini #1 pick for the close-up Gemini #2 "
            "photo (m). Default 0.11 keeps the D405 around its ~20 cm depth "
            "sweet spot and reduces gripper occlusion. Final grasp approach "
            "height is still Object.EGG_CRACKER.approach_dz."
        ),
    )
    p.add_argument(
        "--sink-drop-dz-m",
        type=float,
        default=SINK_DROP_DZ_M,
        help="Descent below the carry pose at SINK_STATION before opening (m).",
    )
    p.add_argument(
        "--sink-drop-dx-m",
        type=float,
        default=SINK_DROP_DX_M,
        help=(
            "Forward reach (arm-base +X) into the sink basin before "
            "descending. Pushes the drop past the cart edge."
        ),
    )

    # Gripper forces (the only egg-crack-specific tunables).
    p.add_argument("--gripper-lift-force", type=float, default=8.0)
    p.add_argument("--gripper-crack-force", type=float, default=140.0)

    # Gemini / camera plumbing.
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Gemini detection retries on failure.",
    )
    p.add_argument(
        "--endeffector-transform-key",
        default=DEFAULT_ENDEFFECTOR_TRANSFORM_KEY,
        help="Redis key for the 4x4 base->flange transform.",
    )
    p.add_argument(
        "--gemini-response-path",
        default=str(DEFAULT_GEMINI_RESPONSE_PATH),
        help="Where to save the annotated Gemini RGB+depth response image.",
    )
    p.add_argument(
        "--orientation-source",
        choices=("fixed", "current"),
        default="fixed",
        help=(
            "Base grasp orientation for the perpendicular yaw: "
            "fixed=Object.EGG_CRACKER default (tool-down 45°), "
            "current=live EE orientation at detection time."
        ),
    )
    return p.parse_args()


def _vision_pick_pose(
    ctx: TaskContext,
    *,
    skip_base: bool,
    detection_xyz: tuple[float, float, float],
    retries: int,
    orientation_source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive base + camera framing, run Gemini, return (pick_pos, grip_R)."""
    detection_pos = np.asarray(detection_xyz, dtype=np.float64).reshape(3)
    arm.move_to(
        ctx,
        detection_pos,
        ARM_HOME_ORIENTATION,
        label=f"[arm] move to detection pose {detection_pos.tolist()}",
        tol_m=DEFAULT_POS_TOL_M,
    )

    if not skip_base:
        base.go_to_pose(ctx, BaseWaypoint.INGREDIENT_STATION)

    pose = gemini.find_grasp_pose(
        ctx,
        Object.EGG_CRACKER,
        retries=retries,
        orientation_source=orientation_source,
    )
    pick_pos = pose.position.astype(np.float64, copy=True)
    grip_R = pose.orientation
    print(f"[egg_crack] detected grasp: {pick_pos.tolist()}")
    if pose.rim_yaw_applied and pose.rim_yaw_deg is not None:
        print(
            f"[egg_crack] strip axis yaw applied: {pose.rim_yaw_deg:+.2f} deg "
            f"(gripper closes perpendicular to strip)"
        )
    else:
        print("[egg_crack] no perpendicular yaw applied (single-point detection)")
    return pick_pos, grip_R


def _suffixed_path(base: str | Path, suffix: str) -> str:
    """Insert ``suffix`` before the extension of ``base`` (e.g. foo.png -> foo_x.png)."""
    p = Path(base)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _above_pose(
    pick: np.ndarray,
    grip_R: np.ndarray,
    *,
    approach_dz_m: float | None = None,
) -> np.ndarray:
    """Pre-grasp "above" pose for the egg cracker (mirrors ``grasp.object``)."""
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    pick = np.asarray(pick, dtype=np.float64).reshape(3)
    dz = float(spec.approach_dz if approach_dz_m is None else approach_dz_m)
    if spec.approach_along_tool_z:
        return pick - np.asarray(grip_R, dtype=np.float64).reshape(3, 3)[:, 2] * dz
    return pick + np.array([0.0, 0.0, dz], dtype=np.float64)


def _save_above_photo(ctx: TaskContext) -> None:
    """Grab RealSense RGB + RGBD views at the above-pick pose and save to logs/.

    Called once the arm is parked above the egg cracker (between the two
    Gemini calls), so the saved image looks straight down the grasp.
    """
    import cv2

    from vision import realsense_rgbd as rs_cam

    try:
        pipeline, align, depth_scale, _intrinsics = ctx.realsense()
        triple = rs_cam.next_rgbd_frame(
            pipeline, align, depth_scale, ctx.cam_timeout_ms, [0], max_misses=10
        )
    except Exception as e:  # noqa: BLE001 - photo is best-effort, never fatal
        print(f"[egg_crack] above photo skipped (camera error): {e}")
        return
    if triple is None:
        print("[egg_crack] above photo skipped (no RGBD frame)")
        return
    color_bgr, _depth_m, depth_vis = triple
    ts = time.strftime("%Y%m%d_%H%M%S")
    photos_dir = _CONTROLLERS.parent / "logs" / "egg_cracker_above_photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = photos_dir / f"egg_cracker_above_rgb_{ts}.png"
    rgbd_path = photos_dir / f"egg_cracker_above_rgbd_{ts}.png"
    rgbd_bgr = np.hstack([color_bgr, depth_vis])

    ok_rgb = cv2.imwrite(str(rgb_path), color_bgr)
    ok_rgbd = cv2.imwrite(str(rgbd_path), rgbd_bgr)
    if ok_rgb and ok_rgbd:
        print(
            f"[egg_crack] saved above-pick photos: rgb={rgb_path} rgbd={rgbd_path}"
        )
    else:
        print(
            "[egg_crack] above photo: cv2.imwrite failed "
            f"(rgb_ok={ok_rgb}, rgbd_ok={ok_rgbd})"
        )


def _shake_j0_to_dislodge_egg(
    ctx: TaskContext,
    *,
    shake_delta_deg: float,
    shake_cycles: int,
    tol_rad: float = EGG_CRACK_SHAKE_J0_TOL_RAD,
    timeout_s: float = EGG_CRACK_SHAKE_J0_TIMEOUT_S,
    use_warmup: bool = True,
    post_switch_wait_s: float = 0.0,
) -> None:
    """After cracking, shake J0 (the base joint) ±delta for N cycles to dislodge the egg.

    Pattern lifted straight from the jar pour's post-dump J4 shake in
    ``grasp_and_pour_jar_controller`` — the only differences are the
    joint (J0/base vs J4) and the smaller amplitude (3° vs 10°). Shaking
    the base joint swings the whole cracker horizontally. Sequence:

    1. **Warmup ("fake") move.** A tiny J0 nudge (``+delta``) issued
       first so the cartesian→joint controller swap settles on a goal
       that's basically the current pose. Without this, the first real
       shake stroke tends to be eaten by the swap (the joint controller
       reseeds its goal to the live measured joints when it activates,
       clobbering our queued shake goal).
    2. **Shake cycles.** ``--shake-cycles`` UP/DOWN pairs of ±delta on
       J0, each stroke RELATIVE to the live joint reading at the start
       of that stroke. Stays in joint_controller the whole time so all
       cycles run under a single ENTER press in ``--step`` mode.

    Falls back to a printed warning + no-op if joint positions aren't
    available from Redis (matches the jar's safety check).
    """
    q_pre = arm.read_joint_positions(ctx.redis)
    if q_pre is None or q_pre.size < 7:
        print(
            "[shake] WARNING: joint positions unavailable from Redis; "
            "skipping J0 shake (warmup + cycles)."
        )
        return

    # 1. Force the cartesian→joint controller swap before the first stroke.
    if use_warmup:
        # Warmup ("fake") move: small +delta J0 (base) nudge. Same mechanism as
        # ``CYLINDER_DUMP_WARMUP_DELTA_DEG`` in the jar pour.
        q0_warmup_rad = float(q_pre[0]) + math.radians(
            EGG_CRACK_SHAKE_J0_WARMUP_DELTA_DEG
        )
        print(
            f"[shake-warmup] J0 warm-up nudge to "
            f"{math.degrees(q0_warmup_rad):+.1f}° "
            f"(forces cartesian→joint controller swap)"
        )
        arm.move_to_joints_partial(
            ctx,
            {0: q0_warmup_rad},
            degrees=False,
            one_indexed=False,
            label=(
                f"  [arm] post-crack shake warm-up J0"
                f"={math.degrees(q0_warmup_rad):+.1f}°"
            ),
            tol_rad=EGG_CRACK_SHAKE_J0_WARMUP_TOL_RAD,
            timeout_s=EGG_CRACK_SHAKE_J0_WARMUP_TIMEOUT_S,
            gated=False,
        )
    else:
        # No fake command: seed the CURRENT joints as the goal (zero motion),
        # switch cartesian→joint, then wait for the swap to settle before the
        # first stroke. Testing whether the wait alone removes the need for
        # the warmup nudge.
        print(
            f"[shake] no warmup; seeding current joints + switching to joint "
            f"controller, waiting {post_switch_wait_s * 1000:.0f} ms for swap."
        )
        arm.publish_goal_joint(ctx.redis, q_pre)
        if post_switch_wait_s > 0:
            time.sleep(post_switch_wait_s)

    # 2. Shake cycles: J0 (base) ±delta, repeated.
    delta_deg = float(shake_delta_deg)
    cycles = int(shake_cycles)
    for cycle in range(cycles):
        for direction, sign in (("UP", +1.0), ("DOWN", -1.0)):
            q_now = arm.read_joint_positions(ctx.redis)
            if q_now is None or q_now.size < 7:
                print(
                    f"[shake cyc {cycle + 1}/{cycles} {direction}] "
                    f"WARNING: joint positions unavailable; aborting shake."
                )
                return
            q0_now = float(q_now[0])
            q0_goal = q0_now + sign * math.radians(delta_deg)
            print(
                f"[shake cyc {cycle + 1}/{cycles} {direction}] "
                f"J0 {math.degrees(q0_now):+.1f}° → "
                f"{math.degrees(q0_goal):+.1f}° "
                f"({'+' if sign > 0 else '-'}{delta_deg:.0f}°)"
            )
            arm.move_to_joints_partial(
                ctx,
                {0: q0_goal},
                degrees=False,
                one_indexed=False,
                label=(
                    f"  [arm] post-crack shake cyc {cycle + 1} {direction} "
                    f"J0={math.degrees(q0_goal):+.1f}°"
                ),
                tol_rad=tol_rad,
                timeout_s=timeout_s,
                gated=False,
            )


def _drop_at_sink(
    ctx: TaskContext,
    grip_R: np.ndarray,
    *,
    drop_dz_m: float,
    drop_dx_m: float,
) -> None:
    """Reach forward, lower, open gripper, settle — same pattern as bowl_pour_controller.

    Assumes the base has already driven to ``SINK_STATION`` and the arm
    is still holding the cracker. Reads the live EE position, pushes the
    drop ``drop_dx_m`` forward (arm-base +X, away from the cart) and
    ``drop_dz_m`` below the current pose so the cracker clears the sink
    rim before being released.
    """
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    cur_pose = arm.read_current_ee_world(ctx.redis)
    if cur_pose is None:
        raise RuntimeError("[drop] cannot read current EE pose; aborting sink drop.")
    carry_pos = cur_pose[0]
    drop_pos = carry_pos + np.array([drop_dx_m, 0.0, -drop_dz_m], dtype=np.float64)
    print(
        f"[drop] EE={carry_pos.tolist()}, "
        f"reaching {drop_dx_m * 100:+.1f} cm forward and "
        f"descending {drop_dz_m * 100:.1f} cm to {drop_pos.tolist()}"
    )
    arm.move_to(
        ctx,
        drop_pos,
        grip_R,
        label=f"[drop] descend at sink {drop_pos.tolist()}",
        tol_m=DEFAULT_POS_TOL_M
    )
    gripper.open_gripper(
        ctx.redis,
        spec.open_width,
        speed=spec.speed,
        force=spec.force,
        use_max_mode=True,
    )
    print("[drop] gripper opened — egg cracker released into sink.")
    time.sleep(SINK_DROP_SETTLE_S)




def run_egg_crack_cycle(
    ctx: TaskContext,
    *,
    skip_base: bool = False,
    carry_lift_m: float = EGG_CRACKER_CARRY_LIFT_M,
    crack_xyz: tuple[float, float, float] | None = None,
    gripper_lift_force: float = 8.0,
    gripper_crack_force: float = 140.0,
    shake_delta_deg: float = EGG_CRACK_SHAKE_J0_DELTA_DEG,
    shake_cycles: int = EGG_CRACK_SHAKE_J0_CYCLES,
    no_shake: bool = False,
    retries: int = 1,
    orientation_source: str = "fixed",
    detection_xyz: tuple[float, float, float] | None = None,
    refine_standoff_m: float = EGG_CRACKER_REFINE_STANDOFF_M,
    gemini_response_path: str | None = None,
    sink_drop_dz_m: float = SINK_DROP_DZ_M,
    sink_drop_dx_m: float = SINK_DROP_DX_M,
) -> None:
    """Vision-driven egg-crack flow, run from a single base position.

    The vision grasp frames + grasps the cracker (driving the base to
    INGREDIENT_STATION only if ``skip_base`` is False). Everything after
    the grasp runs WITHOUT further base motion — the bowl and the pick
    spot are both reachable from the grasp position. Offsets are relative
    to ``ARM_HOME_POSITION`` (+X forward, +Y left, -Y right, +Z up):

    1. Frame the cracker, Gemini grasp, ``grasp.object`` closes (width-based,
       ``close_mode="move"`` to ``EGG_CRACKER_GRASP_WIDTH_M``, held). No
       post-grasp carry lift.
    2. Move straight to above the bowl (home + forward + left - drop) — no
       intermediate home pose. The crack happens at the FORWARD spot
       (bowl-left base + ``EGG_CRACK_EMPTY_AHEAD_M``).
    3. Descend into the bowl, ``egg_crack.crack`` (force-based, ~140 N) to break
       the egg, and KEEP squeezing at the crack force through the shake.
    4. Joint-space ``_shake_j0_to_dislodge_egg`` (fake warmup + J0 ±δ shake)
       while still squeezing, then re-open to the hold WIDTH so the cracker is
       held, not crushed shut.
    5. Move to the pour spot: ``EGG_CRACK_EMPTY_AHEAD_M`` back (-X) and
       ``EGG_CRACK_POUR_UP_M`` above the crack pose.
    6. Tilt ``EGG_CRACK_EMPTY_TILT_DEG`` about world +X to empty the cracker,
       then right itself (back to home orientation, holding position).
    7. Return to the pick spot and release the cracker.
    8. Nudge up ``EGG_CRACK_RELEASE_NUDGE_UP_M`` + tiny J0 shake to drop it free.
    9. Lift straight up ``EGG_CRACK_RELEASE_LIFT_M``, then return to ARM_HOME.
    """
    if gemini_response_path is not None:
        ctx.gemini_response_path = gemini_response_path
    if detection_xyz is None:
        detection_xyz = tuple(float(v) for v in EGG_CRACKER_DETECTION_EE_POSITION)
    # ``crack_xyz`` is accepted for backwards-compat but no longer drives the
    # crack pose: the bowl is now reached as a fixed offset from ARM_HOME
    # (see EGG_CRACK_BOWL_* constants), so there is no taught crack pose.
    _ = crack_xyz

    if ctx.gemini_response_path is None:
        ctx.gemini_response_path = str(DEFAULT_GEMINI_RESPONSE_PATH)

    # Single Gemini grasp: frame the cracker from the detection pose and get
    # the grasp pose directly. (The earlier two-stage coarse→refine flow was
    # removed: the close-up Gemini #2 photo was occluded by the gripper and
    # the calibrated extrinsic now puts the first-pass grasp within ~1 cm.)
    pick, grip_R = _vision_pick_pose(
        ctx,
        skip_base=skip_base,
        detection_xyz=detection_xyz,
        retries=retries,
        orientation_source=orientation_source,
    )

    # Final grasp: grasp.object moves above the pick, descends, and closes on
    # the cracker. ``close_mode="move"`` closes to the fixed width
    # ``EGG_CRACKER_GRASP_WIDTH_M`` (spec.close_width) and HOLDS there — the
    # gripper driver's move branch no longer backs off on a "failed" move, so
    # the cracker is held at that width. The crack squeeze itself stays
    # force-based via ``egg_crack.crack``.
    #
    # ``lift_dz_m=0.0`` skips the post-grasp carry lift — the arm goes
    # straight from the closed grasp to the above-bowl pose below.
    grasp.object(
        ctx,
        Object.EGG_CRACKER,
        pick_pos=pick,
        ori=grip_R,
        lift_dz_m=0.0,
        close_mode="move",
    )

    # 1. Move straight from holding the cracker to above the bowl: forward
    # (+X), left (+Y), and below (-Z) home. No intermediate home pose, no
    # post-grasp carry lift — the path up-and-over to here clears the counter.
    # The crack happens at the FORWARD spot (bowl-left base + EMPTY_AHEAD),
    # since the crack and pour spots were swapped vs. the bowl layout.
    crack_forward_m = EGG_CRACK_BOWL_FORWARD_M + EGG_CRACK_EMPTY_AHEAD_M
    above_bowl = ARM_HOME_POSITION + np.array(
        [
            crack_forward_m,
            EGG_CRACK_BOWL_LEFT_M,
            -EGG_CRACK_BOWL_DROP_FROM_HOME_M,
        ],
        dtype=np.float64,
    )
    arm.move_to(
        ctx,
        above_bowl,
        ARM_HOME_ORIENTATION,
        label=(
            f"[egg_crack] above bowl {above_bowl.tolist()} "
            f"(+{crack_forward_m * 100:.0f} cm fwd, "
            f"+{EGG_CRACK_BOWL_LEFT_M * 100:.0f} cm left, "
            f"-{EGG_CRACK_BOWL_DROP_FROM_HOME_M * 100:.0f} cm below home)"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 2. Descend into the bowl before the squeeze.
    crack_pos = above_bowl + np.array(
        [0.0, 0.0, -EGG_CRACK_BOWL_DOWN_M], dtype=np.float64
    )
    arm.move_to(
        ctx,
        crack_pos,
        ARM_HOME_ORIENTATION,
        label=(
            f"[egg_crack] descend {EGG_CRACK_BOWL_DOWN_M * 100:.0f} cm into "
            f"bowl {crack_pos.tolist()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 3. Crack the egg into the bowl. The crack squeeze force-closes the jaws
    # all the way and KEEPS squeezing at the crack force through the shake
    # below, so the cracker stays clamped while the egg/shell is wobbled free.
    egg_crack.crack(
        ctx,
        crack_force=gripper_crack_force,
        lift_force=gripper_lift_force,
    )

    # 4. Switch to joint control, do the "fake" warmup nudge, then shake J0
    # (the base joint) ±delta for ``shake_cycles`` cycles to dislodge the egg
    # yolk / shell off the cracker fingers. Mirrors the jar pour's post-dump
    # J4 shake (just on J0/base with a 3° amplitude instead of J4 at 10°). The
    # gripper is still squeezing at the crack force here.
    if not no_shake:
        _shake_j0_to_dislodge_egg(
            ctx,
            shake_delta_deg=shake_delta_deg,
            shake_cycles=shake_cycles,
        )
    else:
        print("[shake] skipped (--no-shake).")

    # 4b. Now that the shake has dislodged the egg, release the crack squeeze
    # back to the hold (pick-up) WIDTH so the cracker is held at its carry grip
    # again instead of crushed shut for the rest of the sequence.
    crack_spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    step_gate(
        ctx,
        f"[egg_crack] release to hold width {crack_spec.close_width:.4f} m "
        f"(move-open from the crack squeeze)",
    )
    gripper.move(
        ctx.redis,
        crack_spec.close_width,
        speed=crack_spec.speed,
        force=gripper_lift_force,
    )
    time.sleep(0.4)

    # 5. Move to the empty-out (pour) spot: back (-X) from the crack pose by
    # EGG_CRACK_EMPTY_AHEAD_M and up (+Z) by EGG_CRACK_POUR_UP_M so the cracker
    # tips out above the bowl rim (crack and pour spots are swapped vs. the
    # bowl layout, so the pour is behind the crack now).
    pour_spot = crack_pos + np.array(
        [-EGG_CRACK_EMPTY_AHEAD_M, 0.0, EGG_CRACK_POUR_UP_M], dtype=np.float64
    )
    arm.move_to(
        ctx,
        pour_spot,
        ARM_HOME_ORIENTATION,
        label=(
            f"[egg_crack] move to pour spot {pour_spot.tolist()} "
            f"({EGG_CRACK_EMPTY_AHEAD_M * 100:.0f} cm back, "
            f"+{EGG_CRACK_POUR_UP_M * 100:.0f} cm up from crack)"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 6. Tilt about WORLD +X to empty the cracker. Build a clean "pointing
    # straight down" base orientation: tool +Z is forced to exactly world -Z
    # and tool +Y to horizontal, so the frame is dead straight/level. The 135°
    # roll is then applied (positive) about the fixed WORLD +X axis (a
    # pre-multiply, see ``arm.pour_orientation_end``). Position is held at the
    # live pour-start position (the prior move only reaches pour_spot within
    # tolerance). Run the roll in precise mode (stiffer cart position/
    # orientation kp + slower OTG cap) so it tracks accurately; precise mode is
    # taken back off before righting it back to the straight-down base.
    pour_pose = arm.read_current_ee_world(ctx.redis)
    if pour_pose is not None:
        pour_pos = pour_pose[0]
    else:
        print(
            "[egg_crack] WARNING: live EE pose unavailable at pour start; "
            "using commanded pour-spot position."
        )
        pour_pos = pour_spot

    # Straight-down base: level the home tool +X heading into the horizontal
    # plane, point tool +Z straight down, complete a right-handed frame.
    tool_x = ARM_HOME_ORIENTATION[:, 0].astype(np.float64).copy()
    tool_x[2] = 0.0
    nx = float(np.linalg.norm(tool_x))
    tool_x = tool_x / nx if nx > 1e-9 else np.array([1.0, 0.0, 0.0])
    tool_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= np.linalg.norm(tool_y)
    straight_down_R = np.column_stack([tool_x, tool_y, tool_z])

    tilted_ori = arm.pour_orientation_end(
        straight_down_R, EGG_CRACK_EMPTY_TILT_DEG, axis="x"
    )
    precise_snapshot = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
        max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="egg_crack:empty-tilt",
    )
    try:
        arm.move_to(
            ctx,
            pour_pos,
            tilted_ori,
            label=(
                f"[egg_crack] tilt {EGG_CRACK_EMPTY_TILT_DEG:.0f}° about world +X "
                f"to empty cracker (precise mode, straight-down base)"
            ),
            tol_m=DEFAULT_POS_TOL_M,
            timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
        )
    finally:
        # Take precise mode off for the un-rotation (and everything after),
        # even if the tilt timed out or was interrupted.
        gains.restore_precise_grasp(
            ctx.redis, precise_snapshot, label="egg_crack:empty-tilt"
        )
    # Right itself: back to the clean straight-down base (same position, tool
    # pointing straight down again).
    arm.move_to(
        ctx,
        pour_pos,
        straight_down_R,
        label="[egg_crack] right cracker (back to straight-down orientation)",
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 7. Return to the pick spot and release the cracker.
    arm.move_to(
        ctx,
        pick,
        grip_R,
        label=f"[egg_crack] return to pick spot {pick.tolist()} to release",
        tol_m=DEFAULT_POS_TOL_M,
    )
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    gripper.open_gripper(
        ctx.redis,
        spec.open_width,
        speed=spec.speed,
        force=spec.force,
        use_max_mode=True,
    )
    print("[egg_crack] gripper opened — cracker released at pick spot.")
    time.sleep(SINK_DROP_SETTLE_S)

    # 8. Nudge up 1 cm and give a tiny J0 shake so the cracker drops free of
    # the open jaws before lifting away.
    release_nudge = pick + np.array(
        [0.0, 0.0, EGG_CRACK_RELEASE_NUDGE_UP_M], dtype=np.float64
    )
    arm.move_to(
        ctx,
        release_nudge,
        grip_R,
        label=(
            f"[egg_crack] nudge {EGG_CRACK_RELEASE_NUDGE_UP_M * 100:.0f} cm up "
            f"after release {release_nudge.tolist()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )
    _shake_j0_to_dislodge_egg(
        ctx,
        shake_delta_deg=EGG_CRACK_RELEASE_SHAKE_DELTA_DEG,
        shake_cycles=EGG_CRACK_RELEASE_SHAKE_CYCLES,
        tol_rad=EGG_CRACK_RELEASE_SHAKE_TOL_RAD,
        timeout_s=EGG_CRACK_RELEASE_SHAKE_TIMEOUT_S,
        use_warmup=False,
        post_switch_wait_s=EGG_CRACK_RELEASE_SHAKE_POST_SWITCH_WAIT_S,
    )

    # 9. Lift straight up from the pick, then return to home.
    release_lift = pick + np.array(
        [0.0, 0.0, EGG_CRACK_RELEASE_LIFT_M], dtype=np.float64
    )
    arm.move_to(
        ctx,
        release_lift,
        grip_R,
        label=(
            f"[egg_crack] lift {EGG_CRACK_RELEASE_LIFT_M * 100:.0f} cm straight "
            f"up {release_lift.tolist()}"
        ),
        tol_m=DEFAULT_POS_TOL_M,
    )
    arm.move_to(
        ctx,
        ARM_HOME_POSITION,
        ARM_HOME_ORIENTATION,
        label=f"[arm] return to home {ARM_HOME_POSITION.tolist()}",
        tol_m=HOME_POS_TOL_M,
    )


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key
    ctx.gemini_response_path = args.gemini_response_path

    print(f"Step mode      : {'on' if args.step else 'off'}")
    print(f"Vision         : {'on' if args.vision else 'off'}")
    if args.vision and not args.skip_base:
        print("Base route     : INGREDIENT_STATION → MIXING_STATION → SINK_STATION")
    elif args.vision:
        print("Base route     : skipped (--skip-base)")
    print("Carry lift     : 0.0 cm (post-grasp lift skipped — straight to bowl)")
    print(f"Gemini response: {args.gemini_response_path}")

    try:
        if args.vision:
            run_egg_crack_cycle(
                ctx,
                skip_base=args.skip_base,
                carry_lift_m=args.carry_lift_m,
                crack_xyz=(args.crack_x, args.crack_y, args.crack_z),
                gripper_lift_force=args.gripper_lift_force,
                gripper_crack_force=args.gripper_crack_force,
                shake_delta_deg=args.shake_delta_deg,
                shake_cycles=args.shake_cycles,
                no_shake=args.no_shake,
                retries=args.retries,
                orientation_source=args.orientation_source,
                detection_xyz=(args.detection_x, args.detection_y, args.detection_z),
                refine_standoff_m=args.refine_standoff_m,
                gemini_response_path=args.gemini_response_path,
                sink_drop_dz_m=args.sink_drop_dz_m,
                sink_drop_dx_m=args.sink_drop_dx_m,
            )
        else:
            pick = np.array(
                [args.pick_x, args.pick_y, args.pick_z], dtype=np.float64
            )
            print(f"[egg_crack] using static pick pose: {pick.tolist()}")
            if (
                args.drop_x is not None
                and args.drop_y is not None
                and args.drop_z is not None
            ):
                drop: np.ndarray | None = np.array(
                    [args.drop_x, args.drop_y, args.drop_z], dtype=np.float64
                )
            else:
                drop = None
            egg_crack.run(
                ctx,
                pick_pos=pick,
                drop_pos=drop,
                ori=None,
                crack_force=args.gripper_crack_force,
                lift_force=args.gripper_lift_force,
            )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        ctx.stop_realsense()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
