#!/usr/bin/env python3
"""OptiTrack → TidyBot base pose controller (CLI wrapper).

Thin CLI on top of :mod:`tidybot_base.opti_nav`. Parses flags, resolves the
goal (absolute Motive pose or relative ``--goal-along`` axis offset), and
hands the work off to ``setup_run_plan`` + ``run_replan_loop``. All the
actual calibration, replanning, and base I/O lives in
``ZitiBot/controllers/tidybot_base/``.

**Frames**

- *Mocap*: OptiTrack on ``tidybot01::pos`` (xyz, m) and ``tidybot01::ori`` (quat xyzw).
- *Robot*: ``hb1::current_pose`` is wheel odometry (m, m, rad). **Opti is ground truth**
  for pose and heading. At startup we record Opti + hb together; lab-frame goals use
  Motive axes, body-frame goals use the Opti quaternion. Hb yaw is not used for
  mapping (only held fixed on ``hb1::desired_pose``).

  Motive's rigid-body local +X is auto-assigned from the marker layout and is
  generally **not** the robot's actual forward direction. We capture that
  mismatch with ``--marker-yaw-offset-deg`` (mocap_yaw_marker = body_yaw_in_opti
  + offset, CCW positive). The locked-in default
  ``DEFAULT_MARKER_YAW_OFFSET_DEG = +41.0`` (in ``tidybot_base.opti_planner``)
  matches the current ``tidybot01`` rigid body. With that set, ``calib.rot =
  R(-body_yaw_in_opti)`` maps Opti-world delta vectors straight into the hb
  (body) world. The log block prints a live empirical estimate so you can
  re-tune if the rigid body is rebuilt.

**Units**

- Base / Redis driver: **m**, **rad**, **m/s**, **rad/s** (see ``Vehicle`` in
  ``tidybot_base/base_controller.py``; wheel radius 0.0508 m).
- Mocap ``tidybot01::pos``: OptiTrack world **meters** (xyz).
- ``--goal-offset-ft`` is converted to meters for mocap goals.
- ``--tolerance-in`` is converted to meters. **Success** is when
  ``hb1::current_pose`` is within that distance of the hb target,
  recomputed each loop from live Opti when ``--no-replan`` is not set.

**Goals**

- Absolute (default): Motive position ``(-1.5, 1.0, 0.45)`` m, target *body* yaw
  90 deg (the cart's driving +X aligned with Motive +Y). Pass ``--no-target-yaw``
  to hold startup heading.
- Relative: ``--relative-goal`` then ``--goal-along lab-plus-y`` (default 1.5 ft), etc.

Lab→hb uses **Opti orientation only**: hb odom +X is body +X in the lab at startup
(``R = R(-body_yaw_in_opti)``, with the marker mounting offset folded in).

**Speed**

``opti_controller`` only sets position goals on Redis. Base speed/accel are set in
``tidybot_base/redis_driver.py`` (default ``max_vel=(0.25, 0.25, 0.79)`` m/s and
rad/s) or ``launch_opti_controller.sh --max-vel-xy``.

**Prerequisites**

- Redis publishing ``tidybot01::pos`` / ``tidybot01::ori`` / ``tidybot01::tracking_valid``
  (mocap) and ``hb1::*`` (base). Motion only runs while ``tracking_valid`` is true.
- ``tidybot_base/redis_driver.py`` running on the robot mini-PC (starts ``Vehicle``
  from ``base_controller.py`` internally).

Usage::

  python ZitiBot/controllers/opti_controller.py
  python ZitiBot/controllers/opti_controller.py --monitor
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import redis

from tidybot_base.mocap import MocapRedisKeys, wait_for_tracking_valid
from tidybot_base.opti_nav import (
    DEFAULT_CONTROL_HZ,
    DEFAULT_LOG_HZ,
    DEFAULT_ODOM_JUMP_M,
    NavConfig,
    print_plan_summary,
    run_replan_loop,
)
from tidybot_base.opti_planner import (
    DEFAULT_MARKER_YAW_OFFSET_DEG,
    GOAL_ALONG_AXES,
    LAB_AXIS_YAW_RAD,
    GoalFrame,
    setup_run_plan,
)
from tidybot_base.redis_io import BaseRedisKeys, connect_redis

FEET_TO_METERS = 0.3048
INCHES_TO_METERS = 0.0254
DEFAULT_GOAL_OFFSET_FT = -1.5  # legacy: Motive lab −X when using --goal-offset-ft
DEFAULT_GOAL_DISTANCE_FT = 1.5
DEFAULT_GOAL_ALONG = "lab-plus-y"  # Motive lab +Y
DEFAULT_TOLERANCE_IN = 1.0
DEFAULT_TARGET_X = -1.5
DEFAULT_TARGET_Y = 1.0
DEFAULT_TARGET_Z = 0.45
DEFAULT_TARGET_YAW_DEG: float | None = 90.0  # Motive lab body heading; None = hold startup
DEFAULT_FACE_LAB_YAW = "minus-y"  # after translation, rotate to face this Motive lab axis
DEFAULT_YAW_TOLERANCE_DEG = 5.0

BASE_KEYS = BaseRedisKeys()
MOCAP_KEYS = MocapRedisKeys()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OptiTrack → TidyBot base controller")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--goal-along",
        choices=sorted(GOAL_ALONG_AXES.keys()),
        default=DEFAULT_GOAL_ALONG,
        help="Move along this axis (default: lab-plus-y = Motive world +Y)",
    )
    p.add_argument(
        "--goal-distance-ft",
        type=float,
        default=DEFAULT_GOAL_DISTANCE_FT,
        help="Distance along --goal-along (ft, positive; default: 1.5)",
    )
    p.add_argument(
        "--goal-distance-m",
        type=float,
        default=None,
        help="Distance along --goal-along in meters (overrides --goal-distance-ft)",
    )
    p.add_argument(
        "--goal-offset-ft",
        type=float,
        default=DEFAULT_GOAL_OFFSET_FT,
        help="Legacy: use with --use-legacy-goal-offsets instead of --goal-along",
    )
    p.add_argument(
        "--goal-offset-opti-x-ft",
        type=float,
        default=None,
        help="Translation along Opti world/body X at startup (feet); default from --goal-offset-ft",
    )
    p.add_argument(
        "--goal-offset-opti-y-ft",
        type=float,
        default=0.0,
        help="Translation along Opti world/body Y at startup (feet)",
    )
    p.add_argument(
        "--goal-offset-x-m",
        type=float,
        default=None,
        help="Opti X offset in meters (overrides --goal-offset-opti-x-ft / --goal-offset-ft)",
    )
    p.add_argument(
        "--goal-offset-opti-y-m",
        type=float,
        default=None,
        help="Opti Y offset in meters (overrides --goal-offset-opti-y-ft)",
    )
    p.add_argument(
        "--use-legacy-goal-offsets",
        action="store_true",
        help="Use --goal-offset-opti-*-ft/m instead of --goal-along",
    )
    p.add_argument(
        "--goal-body-frame",
        action="store_true",
        help="Legacy: goal offsets in marker rigid-body XY (with --use-legacy-goal-offsets)",
    )
    p.add_argument(
        "--goal-opti-world",
        action="store_true",
        help="Legacy: goal offsets in Motive lab/world XY (with --use-legacy-goal-offsets)",
    )
    p.add_argument(
        "--curr-minus-desired",
        action="store_true",
        help="Use (current - desired) in mocap instead of (desired - current)",
    )
    p.add_argument(
        "--monitor",
        action="store_true",
        help="Print state only; do not write hb1::desired_pose",
    )
    p.add_argument(
        "--stop-on-exit",
        action="store_true",
        help="Set hb1::stop on exit",
    )
    p.add_argument(
        "--tolerance-in",
        type=float,
        default=DEFAULT_TOLERANCE_IN,
        help="Done when hb1::current_pose XY is within this many inches of hb1 target",
    )
    p.add_argument(
        "--tolerance-m",
        type=float,
        default=None,
        help="Same as --tolerance-in but in meters (overrides inches)",
    )
    p.add_argument(
        "--robot-pose-key",
        default=BASE_KEYS.robot_pose,
        help="Redis key for robot odom [x, y, yaw] (default: hb1::current_pose)",
    )
    p.add_argument(
        "--mocap-pos-key",
        default=MOCAP_KEYS.pos,
        help="Redis key for mocap position xyz (default: tidybot01::pos)",
    )
    p.add_argument(
        "--mocap-ori-key",
        default=MOCAP_KEYS.ori,
        help="Redis key for mocap orientation xyzw (default: tidybot01::ori)",
    )
    p.add_argument(
        "--tracking-valid-key",
        default=MOCAP_KEYS.tracking_valid,
        help="Redis key; must be true to plan/move (default: tidybot01::tracking_valid)",
    )
    p.add_argument(
        "--odom-jump-m",
        type=float,
        default=DEFAULT_ODOM_JUMP_M,
        help="Warn if robot XY moves more than this between cycles (possible odom reset)",
    )
    p.add_argument(
        "--log-hz",
        type=float,
        default=DEFAULT_LOG_HZ,
        help=f"How often to print current opti + hb1 poses (default: {DEFAULT_LOG_HZ} Hz)",
    )
    p.add_argument(
        "--calib-translation-only",
        action="store_true",
        help="No rotation between Motive lab XY and hb odom (R=I); only if axes match",
    )
    p.add_argument(
        "--calib-yaw-deg",
        type=float,
        default=None,
        help="Override mocap→hb rotation (deg); Motive +X → hb +X at this angle",
    )
    p.add_argument(
        "--cardinal-hb",
        action="store_true",
        help="With --translate-then-rotate: L-path in hb odom (X then Y)",
    )
    p.add_argument(
        "--translate-then-rotate",
        action="store_true",
        default=True,
        help=(
            "Translate at startup yaw, then rotate in place (default). "
            "Pass --direct-motion to drive XY + yaw simultaneously."
        ),
    )
    p.add_argument(
        "--direct-motion",
        action="store_true",
        help="Direct holonomic motion to final [x, y, yaw] in a single waypoint",
    )
    p.add_argument(
        "--no-replan",
        action="store_true",
        help=(
            "Disable per-loop hb-goal recomputation from live Opti. Default: "
            "replan each loop so hb_goal = hb_current + calib.rot @ "
            "(opti_target_xy − opti_current_xy) for XY (and "
            "hb_current_yaw + wrap(target_marker_yaw − marker_yaw_now) at the "
            "final waypoint when yaw is required). Falls back to the fixed "
            "startup hb waypoints on cycles where Opti is unavailable."
        ),
    )
    p.add_argument(
        "--marker-yaw-offset-deg",
        "--robot-input-rot-deg",
        dest="marker_yaw_offset_deg",
        type=float,
        default=DEFAULT_MARKER_YAW_OFFSET_DEG,
        help=(
            "Marker → body yaw offset (deg, CCW positive). "
            "mocap_yaw_marker = body_yaw_in_opti + marker_yaw_offset_deg. "
            "Set this to compensate when the Motive rigid body's local +X is "
            "rotated from the robot's actual driving +X. Locked-in default: "
            f"{DEFAULT_MARKER_YAW_OFFSET_DEG} (matches the current tidybot01 "
            "rigid body). Use the live 'marker_offset_est_deg' line in the "
            "log block to re-tune if the rigid body is rebuilt."
        ),
    )
    p.add_argument(
        "--face-lab-yaw",
        choices=["none", *sorted(LAB_AXIS_YAW_RAD.keys())],
        default=DEFAULT_FACE_LAB_YAW,
        help=(
            "After translation, rotate in place to face this Motive lab axis "
            f"(default: {DEFAULT_FACE_LAB_YAW}; marker +X along that direction)"
        ),
    )
    p.add_argument(
        "--relative-goal",
        action="store_true",
        help=(
            "Use --goal-along / distance instead of the default absolute Motive target "
            f"({DEFAULT_TARGET_X}, {DEFAULT_TARGET_Y}, {DEFAULT_TARGET_Z}) m"
        ),
    )
    p.add_argument(
        "--target-x",
        type=float,
        default=DEFAULT_TARGET_X,
        help=f"Absolute Motive lab X (m); default: {DEFAULT_TARGET_X}",
    )
    p.add_argument(
        "--target-y",
        type=float,
        default=DEFAULT_TARGET_Y,
        help=f"Absolute Motive lab Y (m); default: {DEFAULT_TARGET_Y}",
    )
    p.add_argument(
        "--target-z",
        type=float,
        default=DEFAULT_TARGET_Z,
        help=f"Absolute Motive lab Z (m; logged; hb uses XY only); default: {DEFAULT_TARGET_Z}",
    )
    p.add_argument(
        "--target-yaw-deg",
        type=float,
        default=DEFAULT_TARGET_YAW_DEG,
        help=(
            "Absolute Motive lab heading for the cart's **body** +X (deg, the "
            "actual driving direction). The marker yaw goal is derived as "
            "body_yaw + marker_yaw_offset_deg internally. "
            f"Default: {DEFAULT_TARGET_YAW_DEG} deg. "
            "Pass --no-target-yaw to hold startup heading instead."
        ),
    )
    p.add_argument(
        "--no-target-yaw",
        action="store_true",
        help="Hold startup heading (straight-line XY only); ignores --target-yaw-deg",
    )
    p.add_argument(
        "--target-opti-pose",
        type=str,
        default=None,
        metavar="X,Y,Z[,YAW_DEG]",
        help=(
            'Absolute pose "x,y,z" or "x,y,z,yaw_deg" in Motive lab (overrides --target-*)'
        ),
    )
    p.add_argument(
        "--rotate-only",
        action="store_true",
        help="No translation (distance 0); only rotate to --face-lab-yaw in place",
    )
    p.add_argument(
        "--no-face-turn",
        action="store_true",
        help="Skip rotation; keep hb yaw from startup",
    )
    p.add_argument(
        "--yaw-tolerance-deg",
        type=float,
        default=DEFAULT_YAW_TOLERANCE_DEG,
        help="Yaw tolerance for final facing waypoint (default: 5 deg)",
    )
    return p.parse_args(argv)


def _resolve_face_lab_yaw(args: argparse.Namespace) -> str | None:
    if args.no_face_turn or args.face_lab_yaw == "none":
        return None
    return str(args.face_lab_yaw)


def _resolve_goal_delta_opti_m(args: argparse.Namespace) -> np.ndarray:
    if args.goal_offset_x_m is not None:
        dx = float(args.goal_offset_x_m)
    elif args.goal_offset_opti_x_ft is not None:
        dx = float(args.goal_offset_opti_x_ft) * FEET_TO_METERS
    else:
        dx = float(args.goal_offset_ft) * FEET_TO_METERS
    if args.goal_offset_opti_y_m is not None:
        dy = float(args.goal_offset_opti_y_m)
    else:
        dy = float(args.goal_offset_opti_y_ft) * FEET_TO_METERS
    return np.array([dx, dy], dtype=np.float64)


def _resolve_goal_frame(args: argparse.Namespace) -> GoalFrame:
    if args.goal_body_frame:
        return GoalFrame.OPTI_BODY
    return GoalFrame.OPTI_WORLD


def _resolve_absolute_mocap_target(
    args: argparse.Namespace,
) -> tuple[np.ndarray, float | None] | None:
    if args.relative_goal or args.rotate_only:
        return None
    if args.target_opti_pose is not None:
        parts = [p.strip() for p in str(args.target_opti_pose).split(",")]
        if len(parts) not in (3, 4):
            raise ValueError(
                '--target-opti-pose must be "x,y,z" or "x,y,z,yaw_deg"'
            )
        xyz = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
        if len(parts) == 3:
            return xyz, None
        return xyz, math.radians(float(parts[3]))
    xyz = np.array(
        [float(args.target_x), float(args.target_y), float(args.target_z)],
        dtype=np.float64,
    )
    if args.no_target_yaw or args.target_yaw_deg is None:
        return xyz, None
    return xyz, math.radians(float(args.target_yaw_deg))


def _resolve_goal(args: argparse.Namespace) -> tuple[np.ndarray, GoalFrame]:
    if args.rotate_only:
        frame, _axis = GOAL_ALONG_AXES[args.goal_along]
        return np.zeros(2, dtype=np.float64), frame
    if args.use_legacy_goal_offsets:
        return _resolve_goal_delta_opti_m(args), _resolve_goal_frame(args)
    if args.goal_distance_m is not None:
        distance_m = float(args.goal_distance_m)
    else:
        distance_m = abs(float(args.goal_distance_ft)) * FEET_TO_METERS
    frame, axis = GOAL_ALONG_AXES[args.goal_along]
    delta = np.array(axis, dtype=np.float64) * distance_m
    return delta, frame


def _build_along_note(args: argparse.Namespace) -> str | None:
    if args.relative_goal and not args.use_legacy_goal_offsets:
        return (
            f"  along={args.goal_along}  distance={args.goal_distance_ft:.2f} ft"
        )
    if args.use_legacy_goal_offsets:
        return "  (legacy goal offsets)"
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        absolute_mocap_target = _resolve_absolute_mocap_target(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    goal_delta_input_xy, goal_frame = _resolve_goal(args)
    if args.tolerance_m is not None:
        tolerance_m = float(args.tolerance_m)
    else:
        tolerance_m = float(args.tolerance_in) * INCHES_TO_METERS
    tolerance_yaw_rad = math.radians(float(args.yaw_tolerance_deg))

    if absolute_mocap_target is not None and args.rotate_only:
        print(
            "Error: --rotate-only cannot be used with an absolute target pose",
            file=sys.stderr,
        )
        return 1
    if args.rotate_only and _resolve_face_lab_yaw(args) is None:
        print(
            "Error: --rotate-only needs a facing direction (default --face-lab-yaw minus-y; "
            "do not use --no-face-turn)",
            file=sys.stderr,
        )
        return 1

    try:
        client = connect_redis(args.redis_host, args.redis_port)
    except redis.RedisError as exc:
        print(f"Redis connect failed: {exc}", file=sys.stderr)
        return 1

    base_keys = BaseRedisKeys(robot_pose=args.robot_pose_key)
    mocap_keys = MocapRedisKeys(
        pos=args.mocap_pos_key,
        ori=args.mocap_ori_key,
        tracking_valid=args.tracking_valid_key,
    )

    wait_for_tracking_valid(client, mocap_keys.tracking_valid)

    try:
        plan = setup_run_plan(
            client,
            goal_delta_input_xy=goal_delta_input_xy,
            goal_frame=goal_frame,
            translation_only_calib=args.calib_translation_only,
            calib_yaw_deg=args.calib_yaw_deg,
            cardinal_hb=args.cardinal_hb,
            direct_motion=args.direct_motion,
            marker_yaw_offset_deg=float(args.marker_yaw_offset_deg),
            face_lab_yaw=_resolve_face_lab_yaw(args),
            robot_pose_key=base_keys.robot_pose,
            mocap_pos_key=mocap_keys.pos,
            mocap_ori_key=mocap_keys.ori,
            curr_minus_desired=args.curr_minus_desired,
            absolute_mocap_target=absolute_mocap_target,
        )
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    cfg = NavConfig(
        marker_yaw_offset_deg=float(args.marker_yaw_offset_deg),
        translation_only_calib=args.calib_translation_only,
        calib_yaw_deg=args.calib_yaw_deg,
        direct_motion=args.direct_motion,
        cardinal_hb=args.cardinal_hb,
        replan=not args.no_replan,
        tolerance_m=tolerance_m,
        tolerance_yaw_rad=tolerance_yaw_rad,
        control_hz=DEFAULT_CONTROL_HZ,
        log_hz=float(args.log_hz),
        odom_jump_m=float(args.odom_jump_m),
        curr_minus_desired=args.curr_minus_desired,
        monitor=args.monitor,
        stop_on_exit=args.stop_on_exit,
        print_plan=True,
        print_log=True,
        base_keys=base_keys,
        mocap_keys=mocap_keys,
    )
    print_plan_summary(plan, config=cfg, along_note=_build_along_note(args))
    run_replan_loop(client, plan, config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
