#!/usr/bin/env python3
"""Estimate the OptiTrack marker yaw offset from a short hb straight move.

Run this after ``tidybot_base/redis_driver.py`` and the Motive Redis publisher
are up. It commands a small hb +X translation, compares hb odometry motion to
OptiTrack motion, and prints the marker yaw offset to use for
``--marker-yaw-offset-deg`` / ``DEFAULT_MARKER_YAW_OFFSET_DEG``.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import redis

from tidybot_base import opti_planner as _opti_planner_module
from tidybot_base.mocap import (
    DEFAULT_MOCAP_KEYS,
    MocapRedisKeys,
    read_mocap_pose,
    wait_for_tracking_valid,
)
from tidybot_base.opti_planner import DEFAULT_MARKER_YAW_OFFSET_DEG
from tidybot_base.redis_io import (
    DEFAULT_BASE_KEYS,
    BaseRedisKeys,
    connect_redis,
    read_robot_se2,
    stop_base,
    write_desired_pose,
)
from tidybot_base.se2 import quat_xyzw_to_yaw, wrap_angle


_OFFSET_ASSIGN_RE = re.compile(
    r"^(?P<prefix>DEFAULT_MARKER_YAW_OFFSET_DEG\s*=\s*)"
    r"(?P<value>[-+]?\d+(?:\.\d+)?)"
    r"(?P<suffix>.*)$",
    re.MULTILINE,
)


def bake_offset_into_source(new_deg: float) -> Path | None:
    """Rewrite ``DEFAULT_MARKER_YAW_OFFSET_DEG`` in ``opti_planner.py`` in place.

    Returns the path that was edited, or ``None`` if the assignment line
    couldn't be located (in which case the call is a no-op and a warning
    is printed). Preserves the exact line formatting around the value.
    """
    target_path = Path(_opti_planner_module.__file__).resolve()
    try:
        original = target_path.read_text()
    except OSError as exc:
        print(f"[bake] could not read {target_path}: {exc}", file=sys.stderr)
        return None

    new_value_str = f"{float(new_deg):.2f}"
    replaced = {"count": 0, "old": None}

    def _sub(match: re.Match) -> str:
        replaced["count"] += 1
        replaced["old"] = match.group("value")
        return f"{match.group('prefix')}{new_value_str}{match.group('suffix')}"

    updated = _OFFSET_ASSIGN_RE.sub(_sub, original, count=1)
    if replaced["count"] == 0:
        print(
            f"[bake] could not find DEFAULT_MARKER_YAW_OFFSET_DEG assignment in "
            f"{target_path}; leaving file unchanged.",
            file=sys.stderr,
        )
        return None
    if updated == original:
        print(
            f"[bake] DEFAULT_MARKER_YAW_OFFSET_DEG already {new_value_str} in "
            f"{target_path}; nothing to change."
        )
        return target_path
    try:
        target_path.write_text(updated)
    except OSError as exc:
        print(f"[bake] could not write {target_path}: {exc}", file=sys.stderr)
        return None
    print(
        f"[bake] {target_path}: DEFAULT_MARKER_YAW_OFFSET_DEG "
        f"{replaced['old']} -> {new_value_str}"
    )
    return target_path


def _angle_mean_deg(values_deg: list[float]) -> float:
    angles = np.radians(np.asarray(values_deg, dtype=np.float64))
    return math.degrees(math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean())))


def _angle_std_deg(values_deg: list[float], mean_deg: float) -> float:
    diffs = [
        math.degrees(wrap_angle(math.radians(v - mean_deg)))
        for v in values_deg
    ]
    return float(np.std(np.asarray(diffs, dtype=np.float64)))


def estimate_marker_offset_deg(
    robot_start: np.ndarray,
    robot_current: np.ndarray,
    mocap_start_xyz: np.ndarray,
    mocap_current_xyz: np.ndarray,
    mocap_start_yaw: float,
    *,
    min_motion_m: float,
    max_yaw_change_rad: float,
) -> float | None:
    """Return ``marker_yaw - body_yaw_in_opti`` in degrees, if motion is usable."""
    if abs(wrap_angle(float(robot_current[2]) - float(robot_start[2]))) > max_yaw_change_rad:
        return None

    hb_dx = float(robot_current[0]) - float(robot_start[0])
    hb_dy = float(robot_current[1]) - float(robot_start[1])
    op_dx = float(mocap_current_xyz[0]) - float(mocap_start_xyz[0])
    op_dy = float(mocap_current_xyz[1]) - float(mocap_start_xyz[1])

    if math.hypot(hb_dx, hb_dy) < min_motion_m or math.hypot(op_dx, op_dy) < min_motion_m:
        return None

    body_yaw_obs = math.atan2(op_dy, op_dx) - math.atan2(hb_dy, hb_dx)
    return math.degrees(wrap_angle(mocap_start_yaw - body_yaw_obs))


def run_tune(args: argparse.Namespace) -> int:
    base_keys = BaseRedisKeys(
        robot_pose=args.robot_pose_key,
        robot_vel=DEFAULT_BASE_KEYS.robot_vel,
        desired_pose=args.desired_pose_key,
        stop=args.stop_key,
        kill=DEFAULT_BASE_KEYS.kill,
    )
    mocap_keys = MocapRedisKeys(
        pos=args.mocap_pos_key,
        ori=args.mocap_ori_key,
        tracking_valid=args.tracking_valid_key,
    )

    try:
        client = connect_redis(args.redis_host, args.redis_port)
    except redis.RedisError as exc:
        print(f"Redis connect failed: {exc}", file=sys.stderr)
        return 1

    wait_for_tracking_valid(client, mocap_keys.tracking_valid)
    robot_start = read_robot_se2(client, keys=base_keys)
    mocap_start_xyz, mocap_start_quat = read_mocap_pose(client, keys=mocap_keys)
    mocap_start_yaw = quat_xyzw_to_yaw(mocap_start_quat)

    target = robot_start.copy()
    target[0] += float(args.distance_m)
    target[2] = robot_start[2]

    print("Marker offset tune")
    print(f"  current default: {DEFAULT_MARKER_YAW_OFFSET_DEG:+.2f} deg")
    print(f"  hb start:        {robot_start.round(4).tolist()}")
    print(
        "  opti start:      "
        f"x={mocap_start_xyz[0]:.4f} y={mocap_start_xyz[1]:.4f} "
        f"yaw={math.degrees(mocap_start_yaw):+.2f} deg"
    )
    print(f"  command:         hb +X {args.distance_m:.3f} m, hold yaw")
    print()
    token = input("Press ENTER to move the base and estimate offset, or q to quit: ")
    if token.strip().lower() in ("q", "quit", "exit"):
        return 0

    client.set(base_keys.stop, "ok")
    write_desired_pose(client, target, keys=base_keys)

    estimates: list[float] = []
    started = time.perf_counter()
    last_log = 0.0
    final_robot = robot_start.copy()
    final_mocap_xyz = mocap_start_xyz.copy()
    try:
        while True:
            now = time.perf_counter()
            final_robot = read_robot_se2(client, keys=base_keys)
            final_mocap_xyz, _ = read_mocap_pose(client, keys=mocap_keys)
            estimate = estimate_marker_offset_deg(
                robot_start,
                final_robot,
                mocap_start_xyz,
                final_mocap_xyz,
                mocap_start_yaw,
                min_motion_m=float(args.min_motion_m),
                max_yaw_change_rad=math.radians(float(args.max_yaw_change_deg)),
            )
            if estimate is not None:
                estimates.append(estimate)

            hb_delta = final_robot[:2] - robot_start[:2]
            opti_delta = final_mocap_xyz[:2] - mocap_start_xyz[:2]
            target_err = float(np.linalg.norm(final_robot[:2] - target[:2]))

            if now - last_log >= 1.0 / max(float(args.log_hz), 0.1):
                est_str = "waiting for enough straight motion"
                if estimate is not None:
                    est_str = f"{estimate:+.2f} deg"
                print(
                    f"hb_delta={hb_delta.round(4).tolist()} "
                    f"opti_delta={opti_delta.round(4).tolist()} "
                    f"target_err={target_err:.3f} m  offset_est={est_str}"
                )
                last_log = now

            if target_err <= float(args.tolerance_m):
                break
            if now - started > float(args.timeout_s):
                print("Timed out; using estimates collected so far.", file=sys.stderr)
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nInterrupted; stopping base.", file=sys.stderr)
    finally:
        stop_base(client, keys=base_keys)

    if not estimates:
        print(
            "Could not estimate offset. Try a longer --distance-m or check Opti tracking.",
            file=sys.stderr,
        )
        return 2

    # Use the last half of the samples so acceleration / startup transients do
    # not dominate the reported value.
    tail = estimates[max(0, len(estimates) // 2):]
    mean = _angle_mean_deg(tail)
    std = _angle_std_deg(tail, mean)
    delta_default = math.degrees(wrap_angle(math.radians(mean - DEFAULT_MARKER_YAW_OFFSET_DEG)))

    print()
    print(f"Suggested marker yaw offset: {mean:+.2f} deg")
    print(f"Sample stddev:                {std:.2f} deg  ({len(tail)} samples)")
    print(f"Delta from code default:      {delta_default:+.2f} deg")
    print(f"Use temporarily:              --marker-yaw-offset-deg {mean:+.2f}")
    if args.bake:
        bake_offset_into_source(mean)
    else:
        print("[bake] --no-bake set; source file left unchanged.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate tidybot01 marker yaw offset from a short hb +X move."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--distance-m", type=float, default=0.20)
    p.add_argument("--tolerance-m", type=float, default=0.015)
    p.add_argument("--timeout-s", type=float, default=12.0)
    p.add_argument("--min-motion-m", type=float, default=0.08)
    p.add_argument("--max-yaw-change-deg", type=float, default=2.0)
    p.add_argument("--log-hz", type=float, default=2.0)
    p.add_argument("--robot-pose-key", default=DEFAULT_BASE_KEYS.robot_pose)
    p.add_argument("--desired-pose-key", default=DEFAULT_BASE_KEYS.desired_pose)
    p.add_argument("--stop-key", default=DEFAULT_BASE_KEYS.stop)
    p.add_argument("--mocap-pos-key", default=DEFAULT_MOCAP_KEYS.pos)
    p.add_argument("--mocap-ori-key", default=DEFAULT_MOCAP_KEYS.ori)
    p.add_argument("--tracking-valid-key", default=DEFAULT_MOCAP_KEYS.tracking_valid)
    p.add_argument(
        "--no-bake",
        dest="bake",
        action="store_false",
        default=True,
        help="Skip auto-updating DEFAULT_MARKER_YAW_OFFSET_DEG in opti_planner.py.",
    )
    return p.parse_args()


def main() -> int:
    return run_tune(parse_args())


if __name__ == "__main__":
    sys.exit(main())
