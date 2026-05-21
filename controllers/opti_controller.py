#!/usr/bin/env python3
"""OptiTrack → TidyBot base pose controller (Redis only, dev laptop).

Reads rigid-body pose from OptiTrack Redis keys, compares to a desired mocap
pose, maps the error into the robot odometry frame, and publishes
``hb1::desired_pose`` for ``tidybot_base/redis_driver.py`` (run ``redis_driver.py``
only — not ``base_controller.py`` as a separate process).

**Frames**

- *Mocap*: OptiTrack on ``tidybot01::pos`` (xyz, m) and ``tidybot01::ori`` (quat xyzw).
- *Robot*: ``hb1::current_pose`` is wheel odometry (m, m, rad). **Opti is ground truth**
  for pose and heading. At startup we record Opti + hb together; lab-frame goals use
  Motive axes, body-frame goals use the Opti quaternion. Hb yaw is not used for
  mapping (only held fixed on ``hb1::desired_pose``).

  Motive's rigid-body local +X is auto-assigned from the marker layout and is
  generally **not** the robot's actual forward direction. We capture that
  mismatch with ``--marker-yaw-offset-deg`` (mocap_yaw_marker = body_yaw_in_opti
  + offset, CCW positive). With that set, ``calib.rot = R(-body_yaw_in_opti)``
  maps Opti-world delta vectors straight into the hb (body) world. The log
  block prints a live empirical estimate so you can dial in the right value.

**Units**

- Base / Redis driver: **m**, **rad**, **m/s**, **rad/s** (see ``Vehicle`` in
  ``tidybot_base/base_controller.py``; wheel radius 0.0508 m).
- Mocap ``tidybot01::pos``: OptiTrack world **meters** (xyz).
- ``--goal-offset-ft`` is converted to meters for mocap goals.
- ``--tolerance-in`` is converted to meters. **Success** is when
  ``hb1::current_pose`` is within that distance of the fixed ``hb1::desired_pose``
  target (computed once from Opti at startup), not when mocap error is small.

**Goals**

- Absolute (default): Motive position ``(-1.5, 1.0, 0.45)`` m, target yaw 90 deg
  (marker +X along Motive +Y). Pass ``--no-target-yaw`` to hold startup heading.
- Relative: ``--relative-goal`` then ``--goal-along lab-plus-y`` (default 1.5 ft), etc.

Lab→hb uses **Opti orientation only**: hb odom +X is marker +X in the lab at startup
(``R = R_mocap^T``).

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
import time

import numpy as np
import redis

from tidybot_base.mocap import (
    MocapRedisKeys,
    read_mocap_pose,
    read_tracking_valid,
    wait_for_tracking_valid,
)
from tidybot_base.opti_planner import (
    GOAL_ALONG_AXES,
    LAB_AXIS_YAW_RAD,
    ROBOT_FRAME_ROT_DEG,
    GoalFrame,
    RunPlan,
    mocap_pose_error,
    mocap_pose_to_hb_se2,
    opti_body_yaw_error_rad,
    opti_xy_distance_m,
    replan_hb_goal_from_opti,
    setup_run_plan,
    waypoint_reached,
)
from tidybot_base.redis_io import (
    BaseRedisKeys,
    connect_redis,
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

FEET_TO_METERS = 0.3048
INCHES_TO_METERS = 0.0254
CONTROL_HZ = 100.0
DEFAULT_GOAL_OFFSET_FT = -1.5  # legacy: Motive lab −X when using --goal-offset-ft
DEFAULT_GOAL_DISTANCE_FT = 1.5
DEFAULT_GOAL_ALONG = "lab-plus-y"  # Motive lab +Y
DEFAULT_TOLERANCE_IN = 1.0
DEFAULT_TARGET_X = -1.5
DEFAULT_TARGET_Y = 1.0
DEFAULT_TARGET_Z = 0.45
DEFAULT_TARGET_YAW_DEG: float | None = 90.0  # Motive lab heading for marker +X; None = hold startup
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
        default=0.5,
        help="Warn if robot XY moves more than this between cycles (possible odom reset)",
    )
    p.add_argument(
        "--log-hz",
        type=float,
        default=10.0,
        help="How often to print current opti + hb1 poses (default: 10 Hz)",
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
        default=ROBOT_FRAME_ROT_DEG,
        help=(
            "Marker → body yaw offset (deg, CCW positive). "
            "mocap_yaw_marker = body_yaw_in_opti + marker_yaw_offset_deg. "
            "Set this to compensate when the Motive rigid body's local +X is "
            "rotated from the robot's actual driving +X. Default: "
            f"{ROBOT_FRAME_ROT_DEG}. Use the live 'marker_offset_est_deg' "
            "line in the log block to dial in the right value after one run."
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


def _sleep_until(loop_start: float, period: float) -> None:
    elapsed = time.perf_counter() - loop_start
    if elapsed < period:
        time.sleep(period - elapsed)


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
    if abs(wrap_angle(float(robot_current[2]) - float(plan.robot_start[2]))) > max_hb_yaw_change_rad:
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


def print_pose_log_block(
    *,
    plan: RunPlan,
    robot_current: np.ndarray,
    mocap_xyz: np.ndarray | None,
    mocap_quat: np.ndarray | None,
    curr_minus_desired: bool,
    hb_goal: np.ndarray | None = None,
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
        body_yaw_cur = wrap_angle(
            mocap_yaw - math.radians(plan.marker_yaw_offset_deg)
        )
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
            f"{body_yaw_label}_yaw_deg={math.degrees(body_yaw_cur):.2f}"
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
    else:
        print("opti_current (unavailable)")
        print(
            f"opti_target  marker {_fmt_opti_pose(plan.desired_mocap_xyz, plan.desired_mocap_yaw)}  "
            f"{body_yaw_label}_yaw_deg={math.degrees(plan.desired_body_yaw_in_opti):.2f}"
        )
        print("opti_error (unavailable)")


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
        print("Error: --rotate-only cannot be used with an absolute target pose", file=sys.stderr)
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

    wait_for_tracking_valid(client, args.tracking_valid_key)

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
            robot_pose_key=args.robot_pose_key,
            mocap_pos_key=args.mocap_pos_key,
            mocap_ori_key=args.mocap_ori_key,
            curr_minus_desired=args.curr_minus_desired,
            absolute_mocap_target=absolute_mocap_target,
        )
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    hb_at_mocap_start = mocap_pose_to_hb_se2(
        plan.mocap_start_xyz,
        plan.calib,
        plan.robot_start[2],
    )
    calib_err = float(np.linalg.norm(hb_at_mocap_start - plan.robot_start))
    dyaw_calib = rot2d_yaw(plan.calib.rot)

    if plan.absolute_target:
        mode = "absolute Opti pose"
    elif args.rotate_only:
        mode = "rotate only"
    else:
        mode = "translate + face"
    print(f"Plan (Opti ground truth → hb goal, {mode}):")
    print(f"  hb1 start        = {plan.robot_start.tolist()}  ({args.robot_pose_key})")
    print(
        f"  mocap @ startup  = {plan.mocap_start_xyz.tolist()}  "
        f"Opti heading={math.degrees(plan.mocap_start_yaw):.1f} deg (marker +X in lab)"
    )
    print(
        f"  lab→hb R angle   = {math.degrees(dyaw_calib):.1f} deg  "
        f"t={plan.calib.trans.round(4).tolist()}  "
        f"({'translation only' if args.calib_translation_only else 'from Opti quat'})"
    )
    print(f"  hb(mocap_start)  = {hb_at_mocap_start.round(4).tolist()}  residual={calib_err:.4f} m")
    if plan.absolute_target:
        yaw_note = (
            f"body_yaw={math.degrees(plan.desired_body_yaw_in_opti):.1f} deg "
            f"(marker_yaw={math.degrees(plan.desired_mocap_yaw):.1f} deg, Motive lab)"
            if plan.require_final_yaw
            else "hold startup yaw"
        )
        along_note = f"  absolute target  = {plan.desired_mocap_xyz.tolist()}  {yaw_note}"
    elif not args.use_legacy_goal_offsets:
        along_note = f"  along={args.goal_along}  distance={args.goal_distance_ft:.2f} ft"
    else:
        along_note = "  (legacy goal offsets)"
    print(f"  goal frame       = {plan.goal_frame.value}{along_note}")
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
        f"(marker = body + offset in Opti world; baked into calib.rot)"
    )
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
        path_kind = "cardinal L-path" if args.cardinal_hb else "translate then rotate"
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
    print(f"  hb1 target       = {plan.robot_target.tolist()}  ({BASE_KEYS.desired_pose})")
    print(f"  hb waypoints     = {len(plan.hb_waypoints)} step(s)")
    print(
        f"  mocap keys       = {args.mocap_pos_key} / {args.mocap_ori_key} / "
        f"{args.tracking_valid_key}"
    )
    print(
        f"  success          = |hb1_current_xy - hb1_target_xy| < "
        f"{tolerance_m:.4f} m ({args.tolerance_in:.1f} in)"
    )
    replan_enabled = not args.no_replan
    print(
        "  replan           = "
        + (
            "per-loop from live Opti (hb_goal nudged each cycle to close opti error)"
            if replan_enabled
            else "fixed (hb waypoints frozen at startup)"
        )
    )
    if args.monitor:
        print("Monitor mode — not commanding base.")
    else:
        print(f"Commanding {BASE_KEYS.desired_pose} at {CONTROL_HZ:.0f} Hz")

    period = 1.0 / CONTROL_HZ
    log_period = 1.0 / max(args.log_hz, 0.1)
    last_log = 0.0
    robot_prev: np.ndarray | None = None
    reached_target = False
    waypoint_idx = 0
    mocap_xyz: np.ndarray | None = None
    mocap_quat: np.ndarray | None = None

    def fixed_hb_goal(idx: int) -> np.ndarray:
        return plan.hb_waypoints[min(idx, len(plan.hb_waypoints) - 1)]

    def compute_hb_goal(
        idx: int,
        robot_current: np.ndarray,
        mocap_xyz: np.ndarray | None,
        mocap_quat: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        """Return (hb_goal, used_opti). When replan is on and live opti is
        available, recompute from live Opti; otherwise use the fixed waypoint."""
        if (
            replan_enabled
            and mocap_xyz is not None
            and mocap_quat is not None
        ):
            return (
                replan_hb_goal_from_opti(
                    plan, idx, mocap_xyz, mocap_quat, robot_current
                ),
                True,
            )
        return fixed_hb_goal(idx), False

    if not args.monitor:
        write_desired_pose(client, fixed_hb_goal(0))

    try:
        while True:
            t0 = time.perf_counter()
            if not read_tracking_valid(client, args.tracking_valid_key):
                if not args.monitor:
                    stop_base(client)
                print(
                    f"Tracking lost ({args.tracking_valid_key} is not true) — "
                    f"stopping base and exiting."
                )
                return 1

            robot_current = read_robot_se2(client, args.robot_pose_key)
            try:
                mocap_xyz, mocap_quat = read_mocap_pose(
                    client, args.mocap_pos_key, args.mocap_ori_key
                )
            except RuntimeError:
                mocap_xyz, mocap_quat = None, None

            hb_goal, used_opti = compute_hb_goal(
                waypoint_idx, robot_current, mocap_xyz, mocap_quat
            )

            # Waypoint advancement: prefer opti-world distance when available
            # (the hb target keeps shifting under replan, so the hb-frame check
            # against a fixed waypoint isn't meaningful).
            if waypoint_idx < len(plan.hb_waypoints) - 1:
                if used_opti and mocap_xyz is not None:
                    xy_done = opti_xy_distance_m(plan, mocap_xyz) < tolerance_m
                else:
                    xy_done = waypoint_reached(
                        robot_current,
                        fixed_hb_goal(waypoint_idx),
                        waypoint_idx,
                        plan.hb_waypoints,
                        tolerance_m=tolerance_m,
                        tolerance_yaw_rad=tolerance_yaw_rad,
                    )
                if xy_done:
                    waypoint_idx += 1
                    hb_goal, used_opti = compute_hb_goal(
                        waypoint_idx, robot_current, mocap_xyz, mocap_quat
                    )
                    print(
                        f"Waypoint {waypoint_idx}: hb={hb_goal.round(4).tolist()}"
                        + ("  (live opti)" if used_opti else "  (fixed)")
                    )

            if robot_prev is not None:
                jump = float(np.linalg.norm(robot_current[:2] - robot_prev[:2]))
                if jump > args.odom_jump_m:
                    print(
                        f"Warning: hb1 odom jump {jump:.3f} m between cycles"
                    )
            robot_prev = robot_current.copy()

            if not args.monitor:
                write_desired_pose(client, hb_goal)

            now = time.perf_counter()
            if now - last_log >= log_period:
                if used_opti and mocap_xyz is not None and mocap_quat is not None:
                    track_xy_norm = opti_xy_distance_m(plan, mocap_xyz)
                    track_yaw_err = (
                        opti_body_yaw_error_rad(plan, mocap_quat)
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
                print_pose_log_block(
                    plan=plan,
                    robot_current=robot_current,
                    mocap_xyz=mocap_xyz,
                    mocap_quat=mocap_quat,
                    curr_minus_desired=args.curr_minus_desired,
                    hb_goal=hb_goal,
                )
                print()
                last_log = now
                is_final = waypoint_idx >= len(plan.hb_waypoints) - 1
                pose_ok = (
                    is_final
                    and track_xy_norm < tolerance_m
                    and (
                        not plan.require_final_yaw
                        or track_yaw_err < tolerance_yaw_rad
                    )
                )
                if pose_ok:
                    if not reached_target:
                        msg = (
                            f"Success ({success_frame}): xy within "
                            f"{tolerance_m:.4f} m ({args.tolerance_in:.1f} in)"
                        )
                        if plan.require_final_yaw:
                            if plan.face_lab_yaw is not None:
                                msg += (
                                    f" and body yaw within "
                                    f"{math.degrees(tolerance_yaw_rad):.1f} deg "
                                    f"(facing Motive {plan.face_lab_yaw})"
                                )
                            else:
                                msg += (
                                    f" and body yaw within "
                                    f"{math.degrees(tolerance_yaw_rad):.1f} deg "
                                    f"(body target "
                                    f"{math.degrees(plan.desired_body_yaw_in_opti):.1f} deg)"
                                )
                        print(msg + " — holding.")
                        reached_target = True
            _sleep_until(t0, period)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if args.stop_on_exit:
            stop_base(client)
            print(f"Set {BASE_KEYS.stop!r} = 'stop'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
