#!/usr/bin/env python3
"""Probe the hb base coordinate axes at the current body yaw.

Bypasses the OptiTrack replanner entirely. Commands a small pure
``+hb_x`` move and then a pure ``+hb_y`` move via ``hb1::desired_pose``
and measures the resulting OptiTrack world delta. Compares against the
direction that ``opti_planner.py``'s body→world rotation predicts at
the current body yaw (= marker_yaw - ``DEFAULT_MARKER_YAW_OFFSET_DEG``).

What the output means:

- If both axes' observed Opti delta direction matches predicted within a
  few degrees, the driver agrees with the planner's frame at this yaw —
  the Phase C failure is somewhere else.
- If one axis is flipped (~180° off) or both are rotated by the same
  amount, the driver's holonomic axes don't line up with where the
  planner thinks ``hb_x``/``hb_y`` point in Opti world at this yaw. That
  rotation/flip is the bug.

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/test_base_axes.py -- --step

  # Smaller / larger probe move:
  ./ZitiBot/launch_zitibot_full.sh controllers/test_base_axes.py -- --magnitude 0.10
  ./ZitiBot/launch_zitibot_full.sh controllers/test_base_axes.py -- --axis x

Requires OpenSai stack only insofar as ``launch_zitibot_full.sh`` brings
up the TidyBot base ``redis_driver`` + OptiTrack publisher. Uses only
``tidybot_base`` modules — no controller-level imports.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import redis

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from tidybot_base.mocap import (
    DEFAULT_MOCAP_KEYS,
    read_mocap_pose,
    wait_for_tracking_valid,
)
from tidybot_base.opti_planner import DEFAULT_MARKER_YAW_OFFSET_DEG
from tidybot_base.redis_io import (
    DEFAULT_BASE_KEYS,
    connect_redis,
    read_robot_se2,
    stop_base,
    write_desired_pose,
)
from tidybot_base.se2 import quat_xyzw_to_yaw, rot2d_from_yaw, wrap_angle


DEFAULT_MAGNITUDE_M = 0.15
DEFAULT_CONVERGE_TOL_M = 0.02
DEFAULT_CONVERGE_TIMEOUT_S = 8.0
DEFAULT_DWELL_S = 1.0
STOP_POLL_HZ = 50.0
STOP_TIMEOUT_S = 2.5


@dataclass
class ProbeResult:
    axis: str
    body_yaw_deg: float
    hb_delta: np.ndarray         # 2-vec, hb frame (driver odometry)
    opti_delta: np.ndarray       # 2-vec, Opti world frame (marker XY)
    expected_opti_delta: np.ndarray
    error_deg: float             # angle(observed) - angle(expected), wrapped
    magnitude_ratio: float       # |observed| / |expected|


def _read_stop_flag(client: redis.Redis) -> str | None:
    try:
        raw = client.get(DEFAULT_BASE_KEYS.stop)
    except Exception:  # noqa: BLE001
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return raw.strip().lower()


def _wait_for_at_rest(client: redis.Redis, *, timeout_s: float = STOP_TIMEOUT_S) -> bool:
    """Poll ``hb1::stop`` for the driver's self-clear to ``'ok'`` (= at rest)."""
    poll_dt = 1.0 / STOP_POLL_HZ
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if _read_stop_flag(client) == "ok":
            return True
        time.sleep(poll_dt)
    return False


def _stop_and_dwell(client: redis.Redis, dwell_s: float, label: str) -> None:
    stop_base(client)
    print(f"[stop] {label}: braking, waiting for driver confirm at-rest", flush=True)
    time.sleep(0.4)
    stop_base(client)
    if _wait_for_at_rest(client):
        print("[stop] driver confirms at rest.", flush=True)
    else:
        print(
            f"[stop] WARNING: driver did not confirm rest in "
            f"{STOP_TIMEOUT_S:.1f}s — continuing.",
            flush=True,
        )
    if dwell_s > 0:
        time.sleep(dwell_s)


def _wait_for_convergence(
    client: redis.Redis,
    goal_xy: np.ndarray,
    *,
    tol_m: float,
    timeout_s: float,
) -> bool:
    deadline = time.perf_counter() + timeout_s
    last_log = 0.0
    while time.perf_counter() < deadline:
        hb_now = read_robot_se2(client)
        err = float(np.linalg.norm(hb_now[:2] - goal_xy))
        now = time.perf_counter()
        if now - last_log >= 0.5:
            print(
                f"  hb=[{hb_now[0]:+.3f}, {hb_now[1]:+.3f}, "
                f"{math.degrees(hb_now[2]):+.1f}\u00b0]  "
                f"|err|={err:.3f} m",
                flush=True,
            )
            last_log = now
        if err < tol_m:
            return True
        time.sleep(0.05)
    return False


def _step_gate(enabled: bool, prompt: str) -> bool:
    if not enabled:
        return True
    try:
        ans = input(f"{prompt} [ENTER to proceed, q to quit]: ")
    except EOFError:
        return False
    return ans.strip().lower() not in ("q", "quit", "exit")


def _measure_body_yaw_rad(client: redis.Redis, marker_offset_deg: float) -> float:
    _, quat = read_mocap_pose(client)
    marker_yaw = quat_xyzw_to_yaw(quat)
    return wrap_angle(marker_yaw - math.radians(marker_offset_deg))


def _expected_opti_delta(body_yaw_rad: float, hb_delta: np.ndarray) -> np.ndarray:
    """Per opti_planner.py: ``opti_delta = R(body_yaw_in_opti) @ hb_delta``."""
    return rot2d_from_yaw(body_yaw_rad) @ hb_delta


def _angle_between_deg(v_obs: np.ndarray, v_pred: np.ndarray) -> float:
    """Wrapped angle(observed) - angle(predicted) in degrees."""
    a_obs = math.atan2(float(v_obs[1]), float(v_obs[0]))
    a_pred = math.atan2(float(v_pred[1]), float(v_pred[0]))
    return math.degrees(wrap_angle(a_obs - a_pred))


def probe_axis(
    client: redis.Redis,
    axis: str,
    magnitude_m: float,
    marker_offset_deg: float,
    *,
    converge_tol_m: float,
    converge_timeout_s: float,
    dwell_s: float,
    step: bool,
) -> ProbeResult | None:
    """Command a small pure ``+hb_x`` or ``+hb_y`` move and measure Opti delta."""
    print(f"\n=== Probe axis: +hb_{axis} by {magnitude_m:.3f} m ===")

    hb_start = read_robot_se2(client)
    opti_start_xyz, _ = read_mocap_pose(client)
    body_yaw_start = _measure_body_yaw_rad(client, marker_offset_deg)
    print(
        f"start: hb=[{hb_start[0]:+.3f}, {hb_start[1]:+.3f}, "
        f"{math.degrees(hb_start[2]):+.1f}\u00b0]  "
        f"opti=({opti_start_xyz[0]:+.3f}, {opti_start_xyz[1]:+.3f})  "
        f"body_yaw={math.degrees(body_yaw_start):+.2f}\u00b0  "
        f"(marker_offset={marker_offset_deg:+.2f}\u00b0)"
    )

    if axis == "x":
        hb_delta_cmd = np.array([magnitude_m, 0.0], dtype=np.float64)
    elif axis == "y":
        hb_delta_cmd = np.array([0.0, magnitude_m], dtype=np.float64)
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    expected = _expected_opti_delta(body_yaw_start, hb_delta_cmd)
    print(
        f"prediction: +hb_{axis} of {magnitude_m:.3f} m at body_yaw "
        f"{math.degrees(body_yaw_start):+.2f}\u00b0 should produce "
        f"opti_delta=({expected[0]:+.3f}, {expected[1]:+.3f})  "
        f"|magnitude|={float(np.linalg.norm(expected)):.3f}"
    )

    if not _step_gate(step, "About to command move"):
        return None

    goal_xy = hb_start[:2] + hb_delta_cmd
    goal_se2 = np.array([goal_xy[0], goal_xy[1], hb_start[2]], dtype=np.float64)
    print(
        f"command: hb1::desired_pose = "
        f"[{goal_se2[0]:+.3f}, {goal_se2[1]:+.3f}, "
        f"{math.degrees(goal_se2[2]):+.1f}\u00b0]"
    )

    client.set(DEFAULT_BASE_KEYS.stop, "ok")  # clear any stale stop
    write_desired_pose(client, goal_se2)
    converged = _wait_for_convergence(
        client,
        goal_xy,
        tol_m=converge_tol_m,
        timeout_s=converge_timeout_s,
    )
    if not converged:
        print(
            f"WARNING: did not converge to goal within {converge_timeout_s:.1f}s; "
            f"reading current pose anyway."
        )

    _stop_and_dwell(client, dwell_s, f"after +hb_{axis} probe")

    hb_end = read_robot_se2(client)
    opti_end_xyz, _ = read_mocap_pose(client)

    hb_delta_obs = (hb_end[:2] - hb_start[:2]).astype(np.float64)
    opti_delta_obs = (opti_end_xyz[:2] - opti_start_xyz[:2]).astype(np.float64)

    err_deg = _angle_between_deg(opti_delta_obs, expected)
    obs_mag = float(np.linalg.norm(opti_delta_obs))
    pred_mag = float(np.linalg.norm(expected))
    mag_ratio = obs_mag / pred_mag if pred_mag > 1e-6 else float("nan")

    print("---")
    print(
        f"commanded hb_delta: ({hb_delta_cmd[0]:+.3f}, {hb_delta_cmd[1]:+.3f})  "
        f"|m|={float(np.linalg.norm(hb_delta_cmd)):.3f}"
    )
    print(
        f"actual hb_delta:    ({hb_delta_obs[0]:+.3f}, {hb_delta_obs[1]:+.3f})  "
        f"|m|={float(np.linalg.norm(hb_delta_obs)):.3f}"
    )
    print(
        f"predicted opti:     ({expected[0]:+.3f}, {expected[1]:+.3f})  "
        f"|m|={pred_mag:.3f}"
    )
    print(
        f"observed opti:      ({opti_delta_obs[0]:+.3f}, {opti_delta_obs[1]:+.3f})  "
        f"|m|={obs_mag:.3f}"
    )
    print(f"direction error:    {err_deg:+.2f}\u00b0  (observed minus predicted)")
    print(f"magnitude ratio:    {mag_ratio:.3f}  (observed / predicted)")

    # Return to start.
    if not _step_gate(step, f"About to return to start for axis {axis}"):
        return ProbeResult(
            axis=axis,
            body_yaw_deg=math.degrees(body_yaw_start),
            hb_delta=hb_delta_obs,
            opti_delta=opti_delta_obs,
            expected_opti_delta=expected,
            error_deg=err_deg,
            magnitude_ratio=mag_ratio,
        )

    print("return: commanding back to start hb pose")
    client.set(DEFAULT_BASE_KEYS.stop, "ok")
    write_desired_pose(client, hb_start)
    _wait_for_convergence(
        client,
        hb_start[:2],
        tol_m=converge_tol_m,
        timeout_s=converge_timeout_s,
    )
    _stop_and_dwell(client, dwell_s, f"after returning from +hb_{axis}")

    return ProbeResult(
        axis=axis,
        body_yaw_deg=math.degrees(body_yaw_start),
        hb_delta=hb_delta_obs,
        opti_delta=opti_delta_obs,
        expected_opti_delta=expected,
        error_deg=err_deg,
        magnitude_ratio=mag_ratio,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Probe hb base coordinate axes at the current body yaw "
            "(measures Opti delta for pure +hb_x and +hb_y commands)."
        )
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--magnitude", type=float, default=DEFAULT_MAGNITUDE_M,
        help="Probe move size in meters (default 0.15).",
    )
    p.add_argument(
        "--axis", choices=("x", "y", "both"), default="both",
        help="Which body axis to probe.",
    )
    p.add_argument(
        "--marker-yaw-offset-deg", type=float, default=DEFAULT_MARKER_YAW_OFFSET_DEG,
        help=(
            "Marker→body offset used to compute body_yaw_in_opti "
            f"(default {DEFAULT_MARKER_YAW_OFFSET_DEG:+.2f})."
        ),
    )
    p.add_argument(
        "--converge-tol-m", type=float, default=DEFAULT_CONVERGE_TOL_M,
        help="Convergence tolerance for waiting on hb_current (m).",
    )
    p.add_argument(
        "--converge-timeout-s", type=float, default=DEFAULT_CONVERGE_TIMEOUT_S,
        help="Per-move convergence timeout (s).",
    )
    p.add_argument(
        "--dwell-s", type=float, default=DEFAULT_DWELL_S,
        help="Dwell after each stop before sampling poses (s).",
    )
    p.add_argument(
        "--step", action="store_true",
        help="ENTER-gate each command (recommended for first run).",
    )
    return p.parse_args()


def _print_summary(results: list[ProbeResult]) -> None:
    if not results:
        return
    print("\n=========== SUMMARY ===========")
    print(
        f"{'axis':<6}{'body_yaw':>12}{'pred_opti':>22}"
        f"{'obs_opti':>22}{'err_deg':>10}{'mag_ratio':>12}"
    )
    for r in results:
        pred = f"({r.expected_opti_delta[0]:+.3f},{r.expected_opti_delta[1]:+.3f})"
        obs = f"({r.opti_delta[0]:+.3f},{r.opti_delta[1]:+.3f})"
        print(
            f"{r.axis:<6}{r.body_yaw_deg:>11.2f}\u00b0{pred:>22}{obs:>22}"
            f"{r.error_deg:>9.2f}\u00b0{r.magnitude_ratio:>12.3f}"
        )
    print("===============================")
    print("Interpretation:")
    print("  - err_deg near 0\u00b0  : axis matches planner's prediction")
    print("  - err_deg near \u00b1180\u00b0: axis flipped (sign error)")
    print("  - err_deg near \u00b190\u00b0 : axes are swapped (x \u2194 y)")
    print("  - same err_deg on both axes: whole frame rotated by that amount")
    print("    (so the effective marker_yaw_offset is wrong by err_deg)")


def main() -> int:
    args = parse_args()
    try:
        client = connect_redis(args.redis_host, args.redis_port)
    except redis.RedisError as exc:
        print(f"Redis connect failed: {exc}", file=sys.stderr)
        return 1

    wait_for_tracking_valid(client, DEFAULT_MOCAP_KEYS.tracking_valid)

    axes = ["x", "y"] if args.axis == "both" else [args.axis]
    results: list[ProbeResult] = []
    try:
        for axis in axes:
            res = probe_axis(
                client,
                axis,
                args.magnitude,
                args.marker_yaw_offset_deg,
                converge_tol_m=args.converge_tol_m,
                converge_timeout_s=args.converge_timeout_s,
                dwell_s=args.dwell_s,
                step=args.step,
            )
            if res is None:
                print(f"Probe of axis {axis!r} aborted by user.")
                break
            results.append(res)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        stop_base(client)
        _wait_for_at_rest(client)
        _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
