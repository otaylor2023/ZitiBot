#!/usr/bin/env python3
"""Multi-station egg-cracker flow with Gemini vision grasp.

End-to-end sequence (``--vision``, default):

  1. Arm → home pose.
  2. Base → ``EGG_CRACK_STATION``.
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

  # Same flow but skip the base drive to EGG_CRACK_STATION (arm-only
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
    DEFAULT_GRIPPER_SPEED,
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

# Post-crack / pour shake: wobble the EE about the pose held when called to
# dislodge the egg/shell. The shake is CONTINUOUS — alternating end goals are
# published straight to the cartesian controller with only a short dwell
# (``EGG_CRACK_SHAKE_STROKE_S``) between flips and NO convergence/settle wait,
# so the arm reverses through the center without ever coming to rest. Those
# abrupt direction changes create the momentum swings that fling the shell off.
# The post-crack shake wobbles left/right (world Y); the pour shake (cracker
# upside-down over the bowl) wobbles up/down (world Z).
EGG_CRACK_SHAKE_LR_AMP_M = 0.04      # post-crack wobble amplitude (world Y)
EGG_CRACK_POUR_SHAKE_AMP_M = 0.05    # pour upside-down wobble amplitude (world Z, up/down)
EGG_CRACK_SHAKE_STROKE_S = 0.30      # dwell between goal flips (no settle wait)
# The shakes run in precise mode (stiff position/orientation gains so the wobble
# tracks tightly) with the OTG linear cap RAISED above the normal default
# (0.13 m/s) so the brisk flips build real momentum before each reversal.
EGG_CRACK_SHAKE_MAX_LINEAR_VELOCITY = 0.20
# The pour (upside-down) shake does fewer cycles than the post-crack dislodge.
EGG_CRACK_POUR_SHAKE_CYCLES = 2

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
# The crack and pour targets come from vision: each bowl's detected center
# (depth of the valid pixel nearest the center) plus a tuned XY bias and a
# fixed +Z hover above it. No base motion runs between the grasp, the crack,
# and the release — the bowls and the cracker's pick spot are all reachable
# from where the vision grasp happened.
#
# Sequence (after the cracker is grasped and lifted):
#   1. Above the crack bowl = black-bowl center + bias + 15 cm up. Crack here
#      (no descend).
#   2. Shake to dislodge, re-open to the hold width.
#   3. Above the pour bowl = plastic-bowl center + bias + 15 cm up.
#   4. Tilt ``..._EMPTY_TILT_DEG`` about world +X while rising
#      ``..._POUR_TILT_RISE_M`` to empty the cracker, then right back down to
#      the original position.
#   5. Return to the pick spot, release.
#   6. Lift straight up ``..._RELEASE_LIFT_M``, then return to ARM_HOME.

# Camera-aim TRANSLATIONS from the home position (orientation stays level/home,
# looking straight down — only the EE slides sideways in world Y). Negative Y
# slides RIGHT to look down at the egg cracker grasp; positive Y slides LEFT to
# look down at the bowls. Flip the sign / magnitude here or via
# --cracker-look-dy-m / --bowls-look-dy-m if the camera lands over the wrong
# spot on your bench.
EGG_CRACK_CRACKER_LOOK_DY_M = -0.20
EGG_CRACK_BOWLS_LOOK_DY_M = 0.30
# Extra +Z raise applied ONLY to the bowls-look pose, so the camera backs off
# and sees both bowls fully from above.
EGG_CRACK_BOWLS_LOOK_DZ_M = 0.0
EGG_CRACK_EMPTY_TILT_DEG = 135.0  # world-+X rotation to dump shell/egg residue
EGG_CRACK_POUR_TILT_RISE_M = 0.20  # +Z the cracker rises WHILE tilting to pour
# Loose convergence tolerance for the final pour (tilt-to-empty) pose — the
# upside-down tilt only needs to roughly reach the empty-out attitude, so don't
# stall waiting for the tight default position tolerance.
EGG_CRACK_POUR_TILT_TOL_M = 0.08
# The empty-out tilt runs in a slowed-down mode so the shell pours out gently,
# but a bit faster than full precise-grasp speed (which is tuned for delicate
# grasps). Allow a longer timeout than a normal move since it's still slow.
EGG_CRACK_EMPTY_TILT_MAX_ANGULAR_VEL_RAD_S = 0.70  # ~40°/s (precise is ~30°/s)
EGG_CRACK_EMPTY_TILT_MAX_LINEAR_VEL_M_S = 0.05      # precise is 0.03 m/s
EGG_CRACK_POUR_TILT_TIMEOUT_S = 12.0
# Open the post-crack hold grip by this much (m) vs the carry close width, so
# the cracker is held a touch looser after the egg is cracked out.
EGG_CRACK_POST_CRACK_WIDTH_DELTA_M = 0.005
EGG_CRACK_RELEASE_LIFT_M = 0.10   # +Z straight-up lift after releasing the cracker

# After the gripper opens, nudge up a touch and give a tiny J0 shake so the
# cracker drops free of the open jaws before the full lift away.
EGG_CRACK_RELEASE_NUDGE_UP_M = 0.01      # +Z nudge right after release
# Loose 5 cm clearance lift right after grasping the cracker, so it clears the
# table/its rest before transiting to the bowl. The tolerance must stay BELOW
# the lift distance: arm.move_to converges on "within tol AND settled", so a
# tolerance >= the lift would finish immediately and the arm wouldn't lift.
CRACKER_PICKUP_LIFT_M = 0.05
CRACKER_PICKUP_LIFT_TOL_M = 0.03
# Timeout for the return-to-pick move before releasing the cracker. Generous
# because it's a long move back from the pour spot and ``move_to`` opens the
# gripper as soon as it returns — a short timeout drops the cracker mid-motion.
EGG_CRACK_RELEASE_RETURN_TIMEOUT_S = 8.0
EGG_CRACK_RELEASE_SHAKE_DELTA_DEG = 2.0  # per-stroke J0 amplitude for the post-release shake
EGG_CRACK_RELEASE_SHAKE_CYCLES = 2       # UP/DOWN J0 cycles after release
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
        "--debug",
        action="store_true",
        help=(
            "Bowl-detection debug: go to the bowls-look pose, detect both bowl "
            "drop centers, then move above the crack bowl and above the pour "
            "bowl. No cracker grasp, no crack/pour — just verifies the bowl "
            "vision + the resulting above-bowl waypoints."
        ),
    )
    p.add_argument(
        "--skip-base",
        action="store_true",
        help=(
            "Do not drive the base to EGG_CRACK_STATION (vision mode only). "
            "Use when the cart is already parked in front of the cracker "
            "for arm-only debugging."
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
    p.add_argument(
        "--no-vision-bowls",
        dest="vision_bowls",
        action="store_false",
        help=(
            "Disable vision detection of the crack/pour bowl centers. NOTE: the "
            "fixed-offset fallback was removed — the cycle now requires vision "
            "bowls and will error if this is passed."
        ),
    )
    p.set_defaults(vision_bowls=True)
    p.add_argument(
        "--cracker-look-dy-m",
        type=float,
        default=EGG_CRACK_CRACKER_LOOK_DY_M,
        help=(
            "World-Y translation from home to look straight down at the "
            "cracker grasp. Negative slides right (default -0.20 m)."
        ),
    )
    p.add_argument(
        "--bowls-look-dy-m",
        type=float,
        default=EGG_CRACK_BOWLS_LOOK_DY_M,
        help=(
            "World-Y translation from home to look straight down at the bowls. "
            "Positive slides left (default +0.20 m)."
        ),
    )
    p.add_argument(
        "--bowls-look-dz-m",
        type=float,
        default=EGG_CRACK_BOWLS_LOOK_DZ_M,
        help=(
            "Extra +Z raise applied to the bowls-look pose (default +0.10 m)."
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
    detect_bowls: bool = True,
    cracker_look_dy_m: float = EGG_CRACK_CRACKER_LOOK_DY_M,
    bowls_look_dy_m: float = EGG_CRACK_BOWLS_LOOK_DY_M,
    bowls_look_dz_m: float = EGG_CRACK_BOWLS_LOOK_DZ_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Home, drive base to the egg-crack station, then slide over to look.

    Returns ``(pick_pos, grip_R, crack_center, pour_center)``. The EE keeps the
    level home orientation (looking straight down) and just TRANSLATES in world
    Y: first ``bowls_look_dy_m`` (positive = left) to look down at the crack
    bowl (black ``PASTA_BOWL``) and the shell-pour bowl (white
    ``PLASTIC_BOWL_TOP``) drop centers, each at the depth of the valid pixel
    nearest the center, THEN ``cracker_look_dy_m`` (negative = right) to look
    down at the cracker grasp. Detecting the cracker last leaves the EE at the
    cracker look pose so the caller can grasp straight away. Either bowl center
    is ``None`` if its detection is skipped/fails.
    """
    detection_pos = np.asarray(detection_xyz, dtype=np.float64).reshape(3)
    _ = detection_pos  # retained for the (disabled) static detection-pose move

    # arm.move_to(
    #     ctx,
    #     ARM_HOME_POSITION,
    #     ARM_HOME_ORIENTATION,
    #     label=f"[arm] move to home {ARM_HOME_POSITION.tolist()}",
    #     tol_m=HOME_POS_TOL_M,
    # )

    if not skip_base:
        # EGG_CRACK_STATION faces ~−170° (opposite the 90° counter stations).
        # Holonomic XY+yaw together tends to slide sideways without a clean
        # rotate-in-place; three_phase (approach → rotate → translate) matches
        # OVEN_DOOR and lands the taught heading before the arm reaches out.
        base.go_to_pose(
            ctx, BaseWaypoint.EGG_CRACK_STATION, motion="three_phase"
        )

    # Detect the BOWLS FIRST: slide LEFT (world +Y) to look straight down at the
    # crack/pour bowls and detect both drop centers. Save each overlay to its
    # own filename so neither overwrites the cracker grasp overlay; restore the
    # base path afterward. Orientation stays level/home; only position changes.
    crack_center: np.ndarray | None = None
    pour_center: np.ndarray | None = None
    if detect_bowls:
        look_bowls_pos = ARM_HOME_POSITION + np.array(
            [0.0, float(bowls_look_dy_m), float(bowls_look_dz_m)], dtype=np.float64
        )
        arm.move_to(
            ctx,
            look_bowls_pos,
            ARM_HOME_ORIENTATION,
            label=(
                f"[egg_crack] slide {bowls_look_dy_m * 100:+.0f} cm Y, "
                f"{bowls_look_dz_m * 100:+.0f} cm Z to look down at the bowls "
                f"{look_bowls_pos.tolist()}"
            ),
            tol_m=HOME_POS_TOL_M,
            timeout_s=5.0
        )
        base_path = ctx.gemini_response_path
        try:
            if base_path:
                ctx.gemini_response_path = _suffixed_path(base_path, "crack_bowl")
            crack_center = gemini.find_bowl_drop_center(
                ctx, Object.PASTA_BOWL, retries=retries
            )
            print(f"[egg_crack] crack bowl (black) center: {crack_center.tolist()}")
            if base_path:
                ctx.gemini_response_path = _suffixed_path(base_path, "pour_bowl")
            pour_center = gemini.find_bowl_drop_center(
                ctx, Object.PLASTIC_BOWL_TOP, retries=retries
            )
            print(f"[egg_crack] pour bowl (plastic) center: {pour_center.tolist()}")
        finally:
            ctx.gemini_response_path = base_path

    # Detect the CRACKER GRASP LAST: slide RIGHT (world -Y) to look straight down
    # at the cracker. The EE finishes here, so the caller can go straight into
    # the grasp without an extra repositioning move.
    look_cracker_pos = ARM_HOME_POSITION + np.array(
        [0.0, float(cracker_look_dy_m), 0.0], dtype=np.float64
    )
    arm.move_to(
        ctx,
        look_cracker_pos,
        ARM_HOME_ORIENTATION,
        label=(
            f"[egg_crack] slide {cracker_look_dy_m * 100:+.0f} cm in Y to look "
            f"down at the cracker {look_cracker_pos.tolist()}"
        ),
        tol_m=HOME_POS_TOL_M,
        timeout_s=8.0
    )

    pose = gemini.find_grasp_pose(
        ctx,
        Object.EGG_CRACKER,
        retries=retries,
        orientation_source=orientation_source
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

    # Slide back to the home position before the grasp move so the cracker
    # approach starts from a clean, centered pose.
    # arm.move_to(
    #     ctx,
    #     ARM_HOME_POSITION,
    #     ARM_HOME_ORIENTATION,
    #     label="[egg_crack] slide back to home position",
    #     tol_m=HOME_POS_TOL_M,
    #     timeout_s=5.0,
    # )
    return pick_pos, grip_R, crack_center, pour_center


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

    Thin wrapper over :func:`arm.shake_joint` (the shared, joint-agnostic
    shake helper), pinned to J0/base with the egg-crack warmup tunables.
    """
    arm.shake_joint(
        ctx,
        joint_index=0,
        one_indexed=False,
        shake_delta_deg=shake_delta_deg,
        shake_cycles=shake_cycles,
        tol_rad=tol_rad,
        timeout_s=timeout_s,
        use_warmup=use_warmup,
        warmup_delta_deg=EGG_CRACK_SHAKE_J0_WARMUP_DELTA_DEG,
        warmup_tol_rad=EGG_CRACK_SHAKE_J0_WARMUP_TOL_RAD,
        warmup_timeout_s=EGG_CRACK_SHAKE_J0_WARMUP_TIMEOUT_S,
        post_switch_wait_s=post_switch_wait_s,
        label_prefix="[post-crack shake]",
    )


def _shake_ee(
    ctx: TaskContext,
    *,
    axis: str = "y",
    amp_m: float = EGG_CRACK_SHAKE_LR_AMP_M,
    cycles: int = EGG_CRACK_SHAKE_J0_CYCLES,
    stroke_s: float = EGG_CRACK_SHAKE_STROKE_S,
    max_linear_velocity: float = EGG_CRACK_SHAKE_MAX_LINEAR_VELOCITY,
    label_prefix: str = "[post-crack shake]",
) -> None:
    """Continuously oscillate the EE ±``amp_m`` along a world ``axis``.

    ``axis="y"`` wobbles left/right; ``axis="z"`` wobbles up/down. A pure
    cartesian oscillation about the pose held when called (no J0/base rotation,
    no controller swap), holding the live orientation.

    Unlike a sequence of ``arm.move_to`` strokes — which wait for the arm to
    reach the endpoint AND stop (velocity-gated) between strokes — this
    publishes the alternating end goals straight to the cartesian controller
    and only dwells ``stroke_s`` between flips, with NO convergence wait. The
    OTG retargets immediately, so the arm reverses through the center without
    coming to rest; the abrupt direction changes are what fling the shell/egg
    loose. Runs in precise mode (stiff gains) with a raised OTG linear cap so
    the brisk flips build momentum.
    """
    pose = arm.read_current_ee_world(ctx.redis)
    if pose is None:
        print(f"{label_prefix} WARNING: EE pose unavailable; skipping shake.")
        return
    center = np.asarray(pose[0], dtype=np.float64).reshape(3)
    ori = np.asarray(pose[1], dtype=np.float64).reshape(3, 3)
    ax = axis.strip().lower()
    if ax == "z":
        step = np.array([0.0, 0.0, float(amp_m)], dtype=np.float64)
        a_name, b_name = "UP", "DOWN"
    elif ax == "y":
        step = np.array([0.0, float(amp_m), 0.0], dtype=np.float64)
        a_name, b_name = "LEFT", "RIGHT"
    else:
        raise ValueError(f"shake axis must be 'y' or 'z', got {axis!r}")
    pos_a = center + step
    pos_b = center - step
    precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=max_linear_velocity,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label=f"{label_prefix} shake",
    )
    try:
        for cyc in range(int(cycles)):
            for name, target in ((a_name, pos_a), (b_name, pos_b)):
                print(
                    f"{label_prefix} cyc {cyc + 1}/{int(cycles)} {name} "
                    f"(±{amp_m * 100:.0f} cm world {ax.upper()}) {target.tolist()}",
                    flush=True,
                )
                # Publish the goal and flip after a short dwell WITHOUT waiting
                # for convergence, so the arm never settles between strokes.
                arm.publish_goal_cartesian(ctx.redis, target, ori)
                time.sleep(float(stroke_s))
        # Settle back to the center pose (this one we let actually arrive).
        arm.move_to(
            ctx,
            center,
            ori,
            label=f"{label_prefix} return to center {center.tolist()}",
            tol_m=0.02,
            timeout_s=2.0,
        )
    finally:
        gains.restore_precise_grasp(ctx.redis, precise, label=f"{label_prefix} shake")


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
        speed=DEFAULT_GRIPPER_SPEED,
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
    use_vision_bowls: bool = True,
    cracker_look_dy_m: float = EGG_CRACK_CRACKER_LOOK_DY_M,
    bowls_look_dy_m: float = EGG_CRACK_BOWLS_LOOK_DY_M,
    bowls_look_dz_m: float = EGG_CRACK_BOWLS_LOOK_DZ_M,
) -> None:
    """Vision-driven egg-crack flow, run from a single base position.

    The vision grasp frames + grasps the cracker (driving the base to
    EGG_CRACK_STATION only if ``skip_base`` is False). Everything after
    the grasp runs WITHOUT further base motion — the bowls and the pick
    spot are all reachable from the grasp position. The crack/pour targets
    come from the detected bowl centers (vision is required):

    1. Frame the cracker, Gemini grasp, ``grasp.object`` closes (width-based,
       ``close_mode="move"`` to ``EGG_CRACKER_GRASP_WIDTH_M``, held). No
       post-grasp carry lift.
    2. Move to above the crack bowl (black-bowl center + XY bias + 15 cm up)
       and ``egg_crack.crack`` (force-based, ~140 N) from that hover pose
       (no descend), KEEP squeezing through the shake.
    3. Lateral ``_shake_ee`` (continuous EE wobble ±EGG_CRACK_SHAKE_LR_AMP_M
       along world Y) while still squeezing, then re-open to the hold WIDTH so
       the cracker is held, not crushed shut.
    4. Move to above the pour bowl (plastic-bowl center + XY bias + 15 cm up).
    5. Tilt ``EGG_CRACK_EMPTY_TILT_DEG`` about world +X while rising
       ``EGG_CRACK_POUR_TILT_RISE_M`` to empty the cracker, then an up/down
       ``_shake_ee`` (continuous, world Z) shakes out residue before it rights
       itself back down to the original position.
    6. Return to the pick spot and release the cracker.
    7. Nudge up ``EGG_CRACK_RELEASE_NUDGE_UP_M`` + tiny J0 shake to drop it free.
    8. Lift straight up ``EGG_CRACK_RELEASE_LIFT_M``, then return to ARM_HOME.
    """
    if gemini_response_path is not None:
        ctx.gemini_response_path = gemini_response_path
    if detection_xyz is None:
        detection_xyz = tuple(float(v) for v in EGG_CRACKER_DETECTION_EE_POSITION)
    # ``crack_xyz`` is accepted for backwards-compat but no longer drives the
    # crack pose: the crack target now comes from the detected black-bowl center,
    # so there is no taught crack pose.
    _ = crack_xyz

    if ctx.gemini_response_path is None:
        ctx.gemini_response_path = str(DEFAULT_GEMINI_RESPONSE_PATH)

    # Single Gemini grasp: frame the cracker from the detection pose and get
    # the grasp pose directly. (The earlier two-stage coarse→refine flow was
    # removed: the close-up Gemini #2 photo was occluded by the gripper and
    # the calibrated extrinsic now puts the first-pass grasp within ~1 cm.)
    pick, grip_R, crack_center, pour_center = _vision_pick_pose(
        ctx,
        skip_base=skip_base,
        detection_xyz=detection_xyz,
        retries=retries,
        orientation_source=orientation_source,
        detect_bowls=use_vision_bowls,
        cracker_look_dy_m=cracker_look_dy_m,
        bowls_look_dy_m=bowls_look_dy_m,
        bowls_look_dz_m=bowls_look_dz_m,
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

    # Loose 5 cm clearance lift so the cracker clears the table/rest before the
    # transit to the bowl. Relative to the current pose; loose (< lift) tol so
    # it doesn't burn time converging precisely.
    lift_pose = arm.read_current_ee_world(ctx.redis)
    if lift_pose is not None:
        clearance = np.asarray(lift_pose[0], dtype=np.float64).reshape(3) + np.array(
            [0.0, 0.0, CRACKER_PICKUP_LIFT_M], dtype=np.float64
        )
        arm.move_to(
            ctx,
            clearance,
            grip_R,
            label=(
                f"[egg_crack] clearance lift after cracker pickup "
                f"+{CRACKER_PICKUP_LIFT_M * 100:.0f} cm -> {clearance.tolist()}"
            ),
            tol_m=CRACKER_PICKUP_LIFT_TOL_M,
        )

    # Vision bowl centers are required (the production flow always detects them).
    if crack_center is None or pour_center is None:
        raise RuntimeError(
            "egg_crack requires vision bowl centers; run with vision bowls enabled "
            "(do not pass --no-vision-bowls)"
        )

    # 1. Move from holding the cracker to above the crack bowl: the detected
    # black-bowl center + a tuned XY bias and 15 cm up. The egg is cracked from
    # this hover pose (no descend). The X bias corrects the arm's reach
    # undershoot toward the bowl.
    above_crack = crack_center + np.array([-0.15, 0.00, 0.15], dtype=np.float64)
    arm.move_to(
        ctx,
        above_crack,
        ARM_HOME_ORIENTATION,
        label=f"[egg_crack] above crack bowl {above_crack.tolist()} (vision black bowl)",
        tol_m=DEFAULT_POS_TOL_M,
        timeout_s=9.0,
    )

    # 2. Crack the egg into the bowl. The crack squeeze force-closes the jaws
    # all the way and KEEPS squeezing at the crack force through the shake
    # below, so the cracker stays clamped while the egg/shell is wobbled free.
    egg_crack.crack(
        ctx,
        crack_force=gripper_crack_force,
        lift_force=gripper_lift_force,
    )

    # 3. Shake the EE left/right (±EGG_CRACK_SHAKE_LR_AMP_M along world Y) for
    # ``shake_cycles`` cycles to dislodge the egg yolk / shell off the cracker
    # fingers. Continuous (no settle between strokes). The gripper is still
    # squeezing at the crack force here.
    if not no_shake:
        _shake_ee(ctx, axis="y", amp_m=EGG_CRACK_SHAKE_LR_AMP_M, cycles=shake_cycles)
    else:
        print("[shake] skipped (--no-shake).")

    # 3b. Now that the shake has dislodged the egg, release the crack squeeze
    # back to the hold (pick-up) WIDTH so the cracker is held at its carry grip
    # again instead of crushed shut for the rest of the sequence.
    crack_spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    hold_width = max(
        0.0, crack_spec.close_width + EGG_CRACK_POST_CRACK_WIDTH_DELTA_M
    )
    step_gate(
        ctx,
        f"[egg_crack] release to hold width {hold_width:.4f} m "
        f"(move-open from the crack squeeze, 1 cm wider)",
    )
    gripper.move(
        ctx.redis,
        hold_width,
        speed=DEFAULT_GRIPPER_SPEED,
        force=gripper_lift_force,
    )
    time.sleep(0.4)

    # 4. Move to the pour spot above the WHITE plastic bowl: the detected
    # plastic-bowl center + a tuned XY bias and 15 cm up.
    above_pour = pour_center + np.array([-0.15, 0.10, 0.15], dtype=np.float64)
    arm.move_to(
        ctx,
        above_pour,
        ARM_HOME_ORIENTATION,
        label=f"[egg_crack] above pour bowl {above_pour.tolist()} (vision plastic bowl)",
        tol_m=DEFAULT_POS_TOL_M,
        timeout_s=4.0,
    )

    # 5. Tilt about WORLD +X while rising EGG_CRACK_POUR_TILT_RISE_M to empty the
    # cracker over the pour bowl, then right it back to the original position.
    # Build a clean "pointing straight down" base orientation (tool +Z = world
    # -Z, tool +X leveled into the horizontal) and roll EGG_CRACK_EMPTY_TILT_DEG
    # (positive) about the fixed WORLD +X axis. Position is taken from the live
    # pour-start pose. The tilt runs in precise mode (stiffer cart kp + slower
    # OTG cap) for accurate tracking; precise mode is taken back off before
    # righting it.
    pour_pose = arm.read_current_ee_world(ctx.redis)
    if pour_pose is not None:
        pour_pos = pour_pose[0]
    else:
        print(
            "[egg_crack] WARNING: live EE pose unavailable at pour start; "
            "using commanded above-pour position."
        )
        pour_pos = above_pour

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
    pour_up_pos = pour_pos + np.array(
        [0.0, 0.00, EGG_CRACK_POUR_TILT_RISE_M], dtype=np.float64
    )
    # Run the empty-out tilt in precise mode (slower OTG cap) so the shell pours
    # out slowly; take precise mode back off before the shake + righting.
    pour_precise = gains.apply_precise_grasp(
        ctx.redis,
        max_linear_velocity=EGG_CRACK_EMPTY_TILT_MAX_LINEAR_VEL_M_S,
        max_angular_velocity=EGG_CRACK_EMPTY_TILT_MAX_ANGULAR_VEL_RAD_S,
        position_kp=PRECISE_GRASP_POSITION_KP,
        orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
        label="egg_crack:empty-tilt",
    )
    try:
        arm.move_to(
            ctx,
            pour_up_pos,
            tilted_ori,
            label=(
                f"[egg_crack] tilt {EGG_CRACK_EMPTY_TILT_DEG:.0f}° about world +X "
                f"+ rise {EGG_CRACK_POUR_TILT_RISE_M * 100:.0f} cm to empty cracker (slow)"
            ),
            tol_m=EGG_CRACK_POUR_TILT_TOL_M,
            timeout_s=EGG_CRACK_POUR_TILT_TIMEOUT_S,
        )
    finally:
        gains.restore_precise_grasp(ctx.redis, pour_precise, label="egg_crack:empty-tilt")

    # Shake up/down (±EGG_CRACK_POUR_SHAKE_AMP_M along world Z) in the tilted
    # (upside-down) pose to shake out any remaining shell/egg. Continuous (no
    # settle between strokes). Holds the live tilted orientation.
    _shake_ee(
        ctx,
        axis="z",
        amp_m=EGG_CRACK_POUR_SHAKE_AMP_M,
        cycles=EGG_CRACK_POUR_SHAKE_CYCLES,
        label_prefix="[pour shake]",
    )

    # Then shake left/right (±EGG_CRACK_POUR_SHAKE_AMP_M along world Y) in the
    # same tilted pose to dislodge anything the up/down stroke didn't.
    _shake_ee(
        ctx,
        axis="y",
        amp_m=EGG_CRACK_POUR_SHAKE_AMP_M,
        cycles=EGG_CRACK_POUR_SHAKE_CYCLES,
        label_prefix="[pour shake LR]",
    )

    # Right itself: back to the original position, tool pointing straight down.
    after_pour_pos = pour_pos + np.array([0.0, 0.0, 0.05], dtype=np.float64)
    arm.move_to(
        ctx,
        after_pour_pos,
        straight_down_R,
        label="[egg_crack] right cracker (back to original position with +Z offset, straight-down)",
        tol_m=DEFAULT_POS_TOL_M,
    )

    # 6. Return to the pick spot and release the cracker. Use a generous
    # timeout: this is a long move back from the pour spot, and ``move_to``
    # continues even on timeout — too short a timeout means the gripper opens
    # while the arm is still moving and the cracker gets flung/dropped early.
    above_pick = pick + np.array([0.00, 0.00, 0.08], dtype=np.float64)
    arm.move_to(
        ctx,
        above_pick,
        grip_R,
        label=f"[egg_crack] return to above pick spot {above_pick.tolist()} to release",
        tol_m=DEFAULT_POS_TOL_M,
        timeout_s=EGG_CRACK_RELEASE_RETURN_TIMEOUT_S,
    )
    arm.move_to(
        ctx,
        pick,
        grip_R,
        label=f"[egg_crack] return to pick spot {pick.tolist()} to release",
        tol_m=DEFAULT_POS_TOL_M,
        timeout_s=EGG_CRACK_RELEASE_RETURN_TIMEOUT_S,
    )
    spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
    gripper.open_gripper(
        ctx.redis,
        spec.open_width,
        speed=DEFAULT_GRIPPER_SPEED,
        force=spec.force,
        use_max_mode=True,
    )
    print("[egg_crack] gripper opened — cracker released at pick spot.")
    time.sleep(SINK_DROP_SETTLE_S)

    # 7. Nudge up 1 cm and give a tiny J0 shake so the cracker drops free of
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

    # 8. Lift straight up from the pick, then return to home.
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


# def run_bowl_debug_cycle(
#     ctx: TaskContext,
#     *,
#     skip_base: bool = False,
#     retries: int = 1,
#     bowls_look_dy_m: float = EGG_CRACK_BOWLS_LOOK_DY_M,
#     bowls_look_dz_m: float = EGG_CRACK_BOWLS_LOOK_DZ_M,
#     gripper_lift_force: float = 8.0,
#     gripper_crack_force: float = 140.0,
#     shake_delta_deg: float = EGG_CRACK_SHAKE_J0_DELTA_DEG,
#     shake_cycles: int = EGG_CRACK_SHAKE_J0_CYCLES,
#     no_shake: bool = False,
#     gemini_response_path: str | None = None,
# ) -> None:
#     """Bowl-vision debug: detect both bowls, crack over one, pour over the other.

#     Goes home, slides to the bowls-look pose (left + up), detects the crack
#     bowl (black ``PASTA_BOWL``) and the pour bowl (white ``PLASTIC_BOWL_TOP``)
#     drop centers, then moves to the above-the-crack-bowl waypoint and runs the
#     crack squeeze + J0 shake there, then moves to the above-the-pour-bowl
#     waypoint and runs the pour tilt/right motion there. Uses the same poses and
#     motions the real cycle uses, just without grasping the cracker first (the
#     cracker is assumed to be pre-placed in the gripper for the motion test).
#     """
#     if gemini_response_path is not None:
#         ctx.gemini_response_path = gemini_response_path
#     if ctx.gemini_response_path is None:
#         ctx.gemini_response_path = str(DEFAULT_GEMINI_RESPONSE_PATH)

#     arm.move_to(
#         ctx,
#         ARM_HOME_POSITION,
#         ARM_HOME_ORIENTATION,
#         label=f"[egg_crack:debug] move to home {ARM_HOME_POSITION.tolist()}",
#         tol_m=HOME_POS_TOL_M,
#     )
#     if not skip_base:
#         base.go_to_pose(ctx, BaseWaypoint.EGG_CRACK_STATION, motion="three_phase")

#     # Slide LEFT (+Y) and UP (+Z) to look straight down at the bowls.
#     look_bowls_pos = ARM_HOME_POSITION + np.array(
#         [0.00, float(bowls_look_dy_m), float(bowls_look_dz_m)], dtype=np.float64
#     )
#     arm.move_to(
#         ctx,
#         look_bowls_pos,
#         ARM_HOME_ORIENTATION,
#         label=(
#             f"[egg_crack:debug] slide {bowls_look_dy_m * 100:+.0f} cm Y, "
#             f"{bowls_look_dz_m * 100:+.0f} cm Z to look at the bowls "
#             f"{look_bowls_pos.tolist()}"
#         ),
#         tol_m=HOME_POS_TOL_M,
#         timeout_s=5.0,
#     )

#     base_path = ctx.gemini_response_path
#     try:
#         if base_path:
#             ctx.gemini_response_path = _suffixed_path(base_path, "crack_bowl")
#         crack_center = gemini.find_bowl_drop_center(
#             ctx, Object.PASTA_BOWL, retries=retries
#         )
#         print(f"[egg_crack:debug] crack bowl (black) center: {crack_center.tolist()}")
#         if base_path:
#             ctx.gemini_response_path = _suffixed_path(base_path, "pour_bowl")
#         pour_center = gemini.find_bowl_drop_center(
#             ctx, Object.PLASTIC_BOWL_TOP, retries=retries
#         )
#         print(f"[egg_crack:debug] pour bowl (plastic) center: {pour_center.tolist()}")
#     finally:
#         ctx.gemini_response_path = base_path

#     # Above the crack bowl: same approach waypoint the real cycle uses
#     # (center + above-center + approach).
#     above_crack = crack_center + np.array(
#         [-0.07, -0.05, 0.15],
#         dtype=np.float64,
#     )
#     # crack_precise = gains.apply_precise_grasp(
#     #     ctx.redis,
#     #     max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
#     #     max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
#     #     position_kp=PRECISE_GRASP_POSITION_KP,
#     #     orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
#     #     label="egg_crack:debug-above-crack",
#     # )
#     arm.move_to(
#         ctx,
#         above_crack,
#         ARM_HOME_ORIENTATION,
#         label=f"[egg_crack:debug] above crack bowl {above_crack.tolist()} (precise)",
#         tol_m=DEFAULT_POS_TOL_M,
#         timeout_s=15.0,
#     )
#     # finally:
#     #     gains.restore_precise_grasp(
#     #         ctx.redis, crack_precise, label="egg_crack:debug-above-crack"
#     #     )

#     # Crack the egg into the bowl from the above-crack position, then shake J0
#     # (the base joint) to dislodge the egg/shell off the cracker fingers (same
#     # motion as the real cycle, just at the hover pose with no descend).
#     egg_crack.crack(
#         ctx,
#         crack_force=gripper_crack_force,
#         lift_force=gripper_lift_force,
#     )
#     if not no_shake:
#         _shake_j0_to_dislodge_egg(
#             ctx,
#             shake_delta_deg=shake_delta_deg,
#             shake_cycles=shake_cycles,
#         )
#     else:
#         print("[shake] skipped (--no-shake).")

#     # Release the crack squeeze back to the hold (carry) width so the cracker is
#     # held at its carry grip again instead of crushed shut for the pour.
#     crack_spec = OBJECT_DEFAULTS[Object.EGG_CRACKER]
#     step_gate(
#         ctx,
#         f"[egg_crack:debug] release to hold width {crack_spec.close_width:.4f} m "
#         f"(move-open from the crack squeeze)",
#     )
#     gripper.move(
#         ctx.redis,
#         crack_spec.close_width,
#         speed=crack_spec.speed,
#         force=gripper_lift_force,
#     )
#     time.sleep(0.4)

#     # Above the pour bowl: the real pour waypoint (center + above-center). Run
#     # precise (stiffer cart kp + slower OTG cap) for accurate tracking, then
#     # restore the normal gains.
#     above_pour = pour_center + np.array(
#         [-0.10, 0.0, 0.15], dtype=np.float64
#     )
#     # pour_precise = gains.apply_precise_grasp(
#     #     ctx.redis,
#     #     max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
#     #     max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
#     #     position_kp=PRECISE_GRASP_POSITION_KP,
#     #     orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
#     #     label="egg_crack:debug-above-pour",
#     # )
#     arm.move_to(
#         ctx,
#         above_pour,
#         ARM_HOME_ORIENTATION,
#         label=f"[egg_crack:debug] above pour bowl {above_pour.tolist()} (precise)",
#         tol_m=DEFAULT_POS_TOL_M,
#         timeout_s=3.0,
#     )
#     # finally:
#     #     gains.restore_precise_grasp(
#     #         ctx.redis, pour_precise, label="egg_crack:debug-above-pour"
#     #     )

#     # Pour motion (same as the real cycle): tilt the cracker about WORLD +X from
#     # a clean straight-down base to empty shells out above the pour bowl, then
#     # right it back. Position is held at the live pour-start pose. The tilt runs
#     # in precise mode for accurate tracking, restored before righting.
#     pour_pose = arm.read_current_ee_world(ctx.redis)
#     if pour_pose is not None:
#         pour_pos = pour_pose[0]
#     else:
#         print(
#             "[egg_crack:debug] WARNING: live EE pose unavailable at pour start; "
#             "using commanded above-pour position."
#         )
#         pour_pos = above_pour

#     tool_x = ARM_HOME_ORIENTATION[:, 0].astype(np.float64).copy()
#     tool_x[2] = 0.0
#     nx = float(np.linalg.norm(tool_x))
#     tool_x = tool_x / nx if nx > 1e-9 else np.array([1.0, 0.0, 0.0])
#     tool_z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
#     tool_y = np.cross(tool_z, tool_x)
#     tool_y /= np.linalg.norm(tool_y)
#     straight_down_R = np.column_stack([tool_x, tool_y, tool_z])

#     tilted_ori = arm.pour_orientation_end(
#         straight_down_R, EGG_CRACK_EMPTY_TILT_DEG, axis="x"
#     )
#     # Rise +Z while tilting so the tilted pour pose is up by ..._POUR_TILT_RISE_M,
#     # then descend back to the original position when righting.
#     pour_up_pos = pour_pos + np.array(
#         [0.0, 0.0, EGG_CRACK_POUR_TILT_RISE_M], dtype=np.float64
#     )
#     pour_precise = gains.apply_precise_grasp(
#         ctx.redis,
#         max_linear_velocity=PRECISE_GRASP_MAX_LINEAR_VELOCITY,
#         max_angular_velocity=PRECISE_GRASP_MAX_ANGULAR_VELOCITY,
#         position_kp=PRECISE_GRASP_POSITION_KP,
#         orientation_kp=PRECISE_GRASP_ORIENTATION_KP,
#         label="egg_crack:debug-empty-tilt",
#     )
#     try:
#         arm.move_to(
#             ctx,
#             pour_up_pos,
#             tilted_ori,
#             label=(
#                 f"[egg_crack:debug] tilt {EGG_CRACK_EMPTY_TILT_DEG:.0f}° about world +X "
#                 f"+ rise {EGG_CRACK_POUR_TILT_RISE_M * 100:.0f} cm to empty cracker "
#                 f"(precise mode, straight-down base)"
#             ),
#             tol_m=DEFAULT_POS_TOL_M,
#             timeout_s=PRECISE_GRASP_MOVE_TIMEOUT_S,
#         )
#     finally:
#         gains.restore_precise_grasp(
#             ctx.redis, pour_precise, label="egg_crack:debug-empty-tilt"
#         )
#     arm.move_to(
#         ctx,
#         pour_pos,
#         straight_down_R,
#         label="[egg_crack:debug] right cracker (back to original position, straight-down)",
#         tol_m=DEFAULT_POS_TOL_M,
#     )
#     print("[egg_crack:debug] complete — detected bowls, cracked + shook, poured.")


def main() -> int:
    args = parse_args()
    ctx = make_context(args, step=args.step)
    ctx.endeffector_transform_key = args.endeffector_transform_key
    ctx.gemini_response_path = args.gemini_response_path

    print(f"Step mode      : {'on' if args.step else 'off'}")
    if args.debug:
        print("Mode           : DEBUG (bowls-look -> detect bowls -> crack+shake -> pour, no grasp)")
    print(f"Vision         : {'on' if args.vision else 'off'}")
    if args.vision and not args.skip_base:
        print("Base route     : EGG_CRACK_STATION (three_phase, then arm-only grasp/crack/release)")
    elif args.vision:
        print("Base route     : skipped (--skip-base)")
    print("Carry lift     : 0.0 cm (post-grasp lift skipped — straight to bowl)")
    print(f"Gemini response: {args.gemini_response_path}")

    try:
        if args.debug:
            pass
            # run_bowl_debug_cycle(
            #     ctx,
            #     skip_base=args.skip_base,
            #     retries=args.retries,
            #     bowls_look_dy_m=args.bowls_look_dy_m,
            #     bowls_look_dz_m=args.bowls_look_dz_m,
            #     gripper_lift_force=args.gripper_lift_force,
            #     gripper_crack_force=args.gripper_crack_force,
            #     shake_delta_deg=args.shake_delta_deg,
            #     shake_cycles=args.shake_cycles,
            #     no_shake=args.no_shake,
            #     gemini_response_path=args.gemini_response_path,
            # )
        elif args.vision:
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
                use_vision_bowls=args.vision_bowls,
                cracker_look_dy_m=args.cracker_look_dy_m,
                bowls_look_dy_m=args.bowls_look_dy_m,
                bowls_look_dz_m=args.bowls_look_dz_m,
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
