#!/usr/bin/env python3
"""Drive between taught base stations and log target vs actual Opti pose.

Moves the base only (arm untouched). After each leg, prints a table comparing
the taught station pose to the live OptiTrack body pose at arrival. Pass
``--log`` to also save hb/Opti trajectory PNGs under ``logs/graphs/``.

Usage::

  # Drive to stove from wherever the cart is parked now
  ./ZitiBot/launch_zitibot_full.sh --no-gripper controllers/station_nav_test_controller.py -- \\
      --to stove_station

  # Stirring -> stove, then back (round trip)
  ./ZitiBot/launch_zitibot_full.sh --no-gripper controllers/station_nav_test_controller.py -- \\
      --from stirring_station --to stove_station --round-trip --log

  # Snap to stirring first, then drive to egg crack (repeatable test path)
  ./ZitiBot/launch_zitibot_full.sh --no-gripper controllers/station_nav_test_controller.py -- \\
      --from stirring_station --to egg_crack_station --snap-from --log

  # List all station names / taught poses
  python controllers/station_nav_test_controller.py --list

Requires Motive publishing ``tidybot01::pos`` / ``tidybot01::ori`` and the
TidyBot base ``redis_driver.py`` running.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import base
from zitibot_core.constants import BASE_WAYPOINTS, BaseWaypoint, OptiPose
from zitibot_core.context import make_context
from tidybot_base.mocap import read_mocap_pose
from tidybot_base.opti_nav import NavResult
from tidybot_base.opti_planner import DEFAULT_MARKER_YAW_OFFSET_DEG
from tidybot_base.se2 import quat_xyzw_to_yaw, wrap_angle


def _resolve_station(name: str) -> BaseWaypoint:
    key = name.strip().lower().replace("-", "_")
    for wp in BaseWaypoint:
        if wp.name.lower() == key or wp.value.lower() == key:
            return wp
    choices = ", ".join(sorted(wp.value for wp in BaseWaypoint))
    raise argparse.ArgumentTypeError(
        f"unknown station {name!r}; choose one of: {choices}"
    )


def _body_yaw_deg_from_quat(quat_xyzw: np.ndarray) -> float:
    marker_yaw = quat_xyzw_to_yaw(quat_xyzw)
    body_yaw = wrap_angle(
        marker_yaw - math.radians(DEFAULT_MARKER_YAW_OFFSET_DEG)
    )
    return math.degrees(body_yaw)


def _read_opti_body_pose(client) -> tuple[np.ndarray, float]:
    xyz, quat = read_mocap_pose(client)
    return np.asarray(xyz, dtype=np.float64), _body_yaw_deg_from_quat(quat)


@dataclass
class PoseSnapshot:
    label: str
    x_m: float
    y_m: float
    yaw_deg: float
    z_m: float | None = None

    @classmethod
    def from_waypoint(cls, wp: BaseWaypoint) -> PoseSnapshot:
        pose = BASE_WAYPOINTS[wp]
        return cls(
            label=wp.value,
            x_m=pose.x_m,
            y_m=pose.y_m,
            yaw_deg=pose.yaw_deg,
        )

    @classmethod
    def from_live(cls, client, *, label: str = "live") -> PoseSnapshot:
        xyz, yaw_deg = _read_opti_body_pose(client)
        return cls(
            label=label,
            x_m=float(xyz[0]),
            y_m=float(xyz[1]),
            z_m=float(xyz[2]),
            yaw_deg=yaw_deg,
        )

    @classmethod
    def from_nav_result(cls, result: NavResult, *, label: str = "arrived") -> PoseSnapshot | None:
        if result.final_opti_xyz is None:
            return None
        xyz = np.asarray(result.final_opti_xyz, dtype=np.float64).reshape(3)
        yaw_deg = (
            math.degrees(float(result.final_body_yaw_rad))
            if result.final_body_yaw_rad is not None
            else float("nan")
        )
        return cls(
            label=label,
            x_m=float(xyz[0]),
            y_m=float(xyz[1]),
            z_m=float(xyz[2]),
            yaw_deg=yaw_deg,
        )


def _print_pose_table(rows: list[tuple[str, PoseSnapshot]]) -> None:
    print(f"  {'label':<14} {'x (m)':>8} {'y (m)':>8} {'z (m)':>8} {'yaw (°)':>9}")
    for tag, p in rows:
        z = "—" if p.z_m is None else f"{p.z_m:8.3f}"
        print(
            f"  {tag:<14} {p.x_m:8.3f} {p.y_m:8.3f} {z:>8} {p.yaw_deg:9.1f}"
        )


def _print_error_table(target: PoseSnapshot, actual: PoseSnapshot) -> None:
    dx_cm = (actual.x_m - target.x_m) * 100.0
    dy_cm = (actual.y_m - target.y_m) * 100.0
    dist_cm = math.hypot(dx_cm, dy_cm)
    dyaw = actual.yaw_deg - target.yaw_deg
    while dyaw > 180.0:
        dyaw -= 360.0
    while dyaw < -180.0:
        dyaw += 360.0
    print(
        f"  Δx={dx_cm:+.1f} cm  Δy={dy_cm:+.1f} cm  "
        f"XY dist={dist_cm:.1f} cm  Δyaw={dyaw:+.1f}°"
    )


def _print_move_report(
    *,
    leg: str,
    target: PoseSnapshot,
    start: PoseSnapshot | None,
    result: NavResult,
    client,
) -> None:
    actual = PoseSnapshot.from_nav_result(result, label="arrived")
    if actual is None:
        actual = PoseSnapshot.from_live(client, label="arrived")

    print(f"\n{'=' * 72}")
    print(f"LEG: {leg}")
    print(f"nav result: success={result.success} reason={result.reason} "
          f"elapsed={result.elapsed_s:.1f}s")
    if result.final_xy_error_m is not None:
        yaw_err = (
            f"{result.final_body_yaw_error_deg:+.1f}°"
            if result.final_body_yaw_error_deg is not None
            else "—"
        )
        print(
            f"planner error at stop: XY={result.final_xy_error_m * 100:.1f} cm  "
            f"yaw={yaw_err}"
        )
    if start is not None:
        print("\nStart (Opti body):")
        _print_pose_table([("start", start)])
    print("\nTarget (taught):")
    _print_pose_table([("target", target)])
    print("\nActual (Opti body at arrival):")
    _print_pose_table([("arrived", actual)])
    print("\nError (actual − target):")
    _print_error_table(target, actual)
    if result.final_robot_pose is not None:
        hb = np.asarray(result.final_robot_pose, dtype=np.float64).reshape(3)
        print(
            f"\nhb odom at stop: x={hb[0]:.4f} m, y={hb[1]:.4f} m, "
            f"yaw={math.degrees(hb[2]):.1f}°"
        )
    print(f"{'=' * 72}\n", flush=True)


def _drive_to_station(
    ctx,
    waypoint: BaseWaypoint,
    *,
    motion: str,
    hold_yaw: bool,
) -> NavResult:
    wp = BASE_WAYPOINTS[waypoint]
    return base.go_to_pose(
        ctx,
        waypoint,
        motion=motion,
        hold_yaw=hold_yaw,
        label=f"[station-nav] -> {waypoint.value}",
    )


def _list_stations() -> int:
    print(f"{'station':<22} {'x (m)':>8} {'y (m)':>8} {'yaw (°)':>9}")
    print("-" * 52)
    for wp in BaseWaypoint:
        pose = BASE_WAYPOINTS[wp]
        print(
            f"{wp.value:<22} {pose.x_m:8.3f} {pose.y_m:8.3f} "
            f"{pose.yaw_deg:9.1f}"
        )
    return 0


def parse_args() -> argparse.Namespace:
    station_names = ", ".join(wp.value for wp in BaseWaypoint)
    p = argparse.ArgumentParser(
        description="Drive between taught base stations; log target vs actual Opti pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--from",
        dest="from_station",
        type=_resolve_station,
        default=None,
        metavar="STATION",
        help=f"Origin station for round-trip / snap. Choices: {station_names}",
    )
    p.add_argument(
        "--to",
        dest="to_station",
        type=_resolve_station,
        default=None,
        metavar="STATION",
        help=f"Destination station. Choices: {station_names}",
    )
    p.add_argument(
        "--round-trip",
        action="store_true",
        help="After reaching --to, drive back to --from (requires --from).",
    )
    p.add_argument(
        "--snap-from",
        action="store_true",
        help="Drive to --from before the main leg (repeatable test start).",
    )
    p.add_argument(
        "--motion",
        choices=("holonomic", "three_phase"),
        default="holonomic",
        help="Planner mode passed to base.go_to_pose.",
    )
    p.add_argument(
        "--hold-yaw",
        action="store_true",
        help="Holonomic XY only; keep startup heading (no yaw command).",
    )
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate each leg before driving.",
    )
    p.add_argument(
        "--log",
        action="store_true",
        help="Save hb/Opti trajectory PNGs to logs/graphs/<controller>_NNNN/.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print all taught station poses and exit.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        return _list_stations()
    if args.to_station is None:
        print("Error: --to STATION is required (or pass --list).", file=sys.stderr)
        return 2
    if args.round_trip and args.from_station is None:
        print("Error: --round-trip requires --from.", file=sys.stderr)
        return 2
    if args.snap_from and args.from_station is None:
        print("Error: --snap-from requires --from.", file=sys.stderr)
        return 2

    ctx = make_context(args, step=args.step, print_startup=True)

    legs: list[tuple[str, BaseWaypoint]] = []
    if args.snap_from:
        legs.append((f"snap -> {args.from_station.value}", args.from_station))
    dest_label = (
        f"{args.from_station.value} -> {args.to_station.value}"
        if args.from_station is not None
        else f"current -> {args.to_station.value}"
    )
    legs.append((dest_label, args.to_station))
    if args.round_trip:
        legs.append(
            (f"{args.to_station.value} -> {args.from_station.value}", args.from_station)
        )

    print(
        f"Station nav test: motion={args.motion} log={args.log} "
        f"hold_yaw={args.hold_yaw}",
        flush=True,
    )
    for i, (leg_label, waypoint) in enumerate(legs, start=1):
        if ctx.step:
            from zitibot_core.runner import step_gate

            step_gate(ctx, f"leg {i}/{len(legs)}: {leg_label}")

        start = PoseSnapshot.from_live(ctx.redis, label="start")
        target = PoseSnapshot.from_waypoint(waypoint)
        print(
            f"\n[station-nav] leg {i}/{len(legs)}: {leg_label} "
            f"-> ({target.x_m:.3f}, {target.y_m:.3f}, {target.yaw_deg:.1f}°)",
            flush=True,
        )
        try:
            result = _drive_to_station(
                ctx,
                waypoint,
                motion=args.motion,
                hold_yaw=args.hold_yaw,
            )
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130
        except Exception as exc:
            print(f"\n[station-nav] leg failed: {exc}", flush=True)
            return 1

        _print_move_report(
            leg=leg_label,
            target=target,
            start=start,
            result=result,
            client=ctx.redis,
        )
        if not result.success:
            print("[station-nav] navigation did not converge; stopping.", flush=True)
            return 1

    print("[station-nav] all legs complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
