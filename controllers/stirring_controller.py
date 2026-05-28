#!/usr/bin/env python3
"""Stirring-only controller — fixed poses, Enter between steps.

Assumes the robot is already at the bowl (no base navigation). Sequence:

  1. Grasp ladle (hover → descend → close → lift)
  2. Orient ladle for mixing (SLERP)
  3. Move above bowl → lower to mix height
  4. Stir

Like grasp_and_place_controller.py: hardcoded defaults, no camera/Gemini.

Usage::

  python3 -u controllers/stirring_controller.py
  python3 -u controllers/stirring_controller.py --mix-dur 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import redis

try:
    from scipy.spatial.transform import Rotation as ScipyR, Slerp
except ImportError:
    print(
        "Missing dependency: scipy\n"
        "  pip install scipy\n"
        "  pip install -r ZitiBot/requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)

CONFIG_XML = os.environ.get("ZITIBOT_OPENSAI_CONFIG_XML", "zitibot_panda.xml")
CONTROLLER_TO_USE = "cartesian_controller"

# Fixed poses (world frame, m) — tune for your lab.
DEFAULT_LADLE_POSITION = (0.75, 0.68, 0.508)
DEFAULT_BOWL_POSITION = (0.17, 0.62, 0.50)

_LADLE_GRASP_OFFSET_DEFAULT = (0.22, 0.031, 0.02)
_HOVER_DZ_DEFAULT = 0.15
_LIFT_DZ_DEFAULT = 0.24

_MIX_RADIUS_DEFAULT = 0.05
_MIX_OMEGA_DEFAULT = 1.0
_MIX_DUR_DEFAULT = 10.0

_ORIENT_DUR_DEFAULT = 3.0
_GRASP_OPEN_DWELL_DEFAULT = 1.0
_GRASP_CLOSE_DWELL_DEFAULT = 1.0
_GRIPPER_SETTLE_S = 0.5
_LOOP_HZ_DEFAULT = 20

_GRIPPER_CLOSE_WIDTH_DEFAULT = 0.0
_GRIPPER_SPEED_DEFAULT = 0.1
_GRIPPER_FORCE_DEFAULT = 50.0


@dataclass(frozen=True)
class _Keys:
    cartesian_task_goal_position: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_position"
    )
    cartesian_task_goal_orientation: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_orientation"
    )
    cartesian_task_current_position: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position"
    )
    cartesian_task_current_orientation: str = (
        "opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation"
    )
    active_controller: str = "opensai::controllers::FrankaRobot::active_controller_name"
    config_file_name: str = "::sai-interfaces-webui::config_file_name"
    gripper_desired_width: str = "opensai::FrankaRobot::gripper::desired_width"
    gripper_desired_speed: str = "opensai::FrankaRobot::gripper::desired_speed"
    gripper_desired_force: str = "opensai::FrankaRobot::gripper::desired_force"
    gripper_max_width: str = "opensai::FrankaRobot::gripper::max_width"


_KEYS = _Keys()


class _StepAbort(Exception):
    pass


def _link7_orientation_world(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    d2r = math.pi / 180.0
    R_full = (
        ScipyR.from_euler("z", yaw_deg * d2r)
        * ScipyR.from_euler("y", pitch_deg * d2r)
        * ScipyR.from_euler("x", roll_deg * d2r)
        * ScipyR.from_euler("x", math.pi)
        * ScipyR.from_euler("z", math.pi / 4)
    )
    return R_full.as_matrix()


def _ladle_mixing_orientation() -> np.ndarray:
    return np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64)


def _slerp(R_start: np.ndarray, R_end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = min(1.0, max(0.0, alpha))
    key_rots = ScipyR.concatenate(
        [ScipyR.from_matrix(R_start), ScipyR.from_matrix(R_end)]
    )
    return Slerp([0.0, 1.0], key_rots)([alpha]).as_matrix()[0]


def _decode_redis_value(raw: bytes | str | None) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return raw


def _try_redis(host: str, port: int):
    try:
        r = redis.Redis(host=host, port=port, decode_responses=False)
        r.ping()
        return r
    except Exception as e:
        print(f"Redis connect failed ({e}).", file=sys.stderr)
        return None


def validate_config(redis_client) -> int | None:
    raw = redis_client.get(_KEYS.config_file_name)
    name = _decode_redis_value(raw)
    if name is None:
        print(
            "Warning: ::sai-interfaces-webui::config_file_name not in Redis; continuing.",
            file=sys.stderr,
        )
        return None
    if name != CONFIG_XML:
        print(
            f"Expected webui config {CONFIG_XML!r} but Redis has {name!r}.",
            file=sys.stderr,
        )
        return 1
    return None


def read_current_ee_world(redis_client) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        raw_p = redis_client.get(_KEYS.cartesian_task_current_position)
        raw_o = redis_client.get(_KEYS.cartesian_task_current_orientation)
        if raw_p is None or raw_o is None:
            return None
        cur_pos = np.array(json.loads(raw_p), dtype=np.float64).reshape(3)
        cur_ori = np.array(json.loads(raw_o), dtype=np.float64).reshape(3, 3)
        return cur_pos, cur_ori
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _ensure_cartesian(redis_client) -> None:
    raw = redis_client.get(_KEYS.active_controller)
    if _decode_redis_value(raw) != CONTROLLER_TO_USE:
        redis_client.set(_KEYS.active_controller, CONTROLLER_TO_USE)


def _publish_cartesian(redis_client, pos: np.ndarray, ori: np.ndarray) -> None:
    _ensure_cartesian(redis_client)
    redis_client.set(
        _KEYS.cartesian_task_goal_position,
        json.dumps(np.asarray(pos, dtype=np.float64).reshape(3).tolist()),
    )
    redis_client.set(
        _KEYS.cartesian_task_goal_orientation,
        json.dumps(np.asarray(ori, dtype=np.float64).reshape(3, 3).tolist()),
    )


def _set_gripper(redis_client, width: float, speed: float, force: float) -> None:
    redis_client.set(_KEYS.gripper_desired_width, str(float(width)))
    redis_client.set(_KEYS.gripper_desired_speed, str(float(speed)))
    redis_client.set(_KEYS.gripper_desired_force, str(float(force)))


def _gripper_open_width(redis_client, override: float | None) -> float:
    if override is not None:
        return float(override)
    raw = redis_client.get(_KEYS.gripper_max_width)
    if raw is not None:
        try:
            w = float(raw.decode("utf-8"))
            if w > 0:
                return w
        except (ValueError, AttributeError):
            pass
    return 0.08


def _read_step_line(prompt: str) -> str | None:
    try:
        if sys.stdin.isatty():
            sys.stdout.flush()
            return input(prompt)
    except EOFError:
        pass
    try:
        with open("/dev/tty", "r") as tty_in:
            sys.stderr.write(prompt)
            sys.stderr.flush()
            return tty_in.readline()
    except OSError:
        return None


def _wait_for_enter(step: str, detail: str = "", pause_sec: float = 0.0) -> bool:
    msg = f"\n── {step} ──\n"
    if detail:
        msg += f"   {detail}\n"
    if pause_sec > 0:
        print(msg, end="", flush=True)
        print(f"   (auto-pause {pause_sec:.1f} s)", flush=True)
        time.sleep(pause_sec)
        return True
    msg += "   Press Enter to continue (q to quit): "
    line = _read_step_line(msg)
    if line is None:
        print(
            "\nERROR: No interactive terminal. Run in a shell or use --pause-sec N.",
            file=sys.stderr,
            flush=True,
        )
        return False
    if line.strip().lower() in ("q", "quit", "exit"):
        print("Quit requested.")
        return False
    return True


def _step_pause(step: str, detail: str, pause_sec: float) -> None:
    if not _wait_for_enter(step, detail, pause_sec=pause_sec):
        raise _StepAbort()


def run_stirring(
    redis_client,
    args: argparse.Namespace,
    ladle_pos: np.ndarray,
    bowl_pos: np.ndarray,
) -> int:
    offset = np.array(args.ladle_offset, dtype=np.float64)
    bowl_cx, bowl_cy, bowl_cz = bowl_pos

    z_mix = args.z_mix_ee if args.z_mix_ee is not None else round(bowl_cz + 0.283, 4)
    z_above = (
        args.z_above_bowl if args.z_above_bowl is not None else round(z_mix + 0.10, 4)
    )

    R_grasp = _link7_orientation_world(0.0, 0.0, 45.0)
    R_mix = _ladle_mixing_orientation()

    open_w = _gripper_open_width(redis_client, args.gripper_open_width)
    close_w = args.gripper_close_width
    spd, frc = args.gripper_speed, args.gripper_force
    dt = 1.0 / max(1, args.loop_hz)

    def grasp_w() -> np.ndarray:
        return ladle_pos + offset

    def hover_w() -> np.ndarray:
        return grasp_w() + np.array([0.0, 0.0, args.hover_dz])

    def lift_w() -> np.ndarray:
        return grasp_w() + np.array([0.0, 0.0, args.lift_dz])

    above_bowl = np.array([bowl_cx, bowl_cy, z_above])
    at_mix = np.array([bowl_cx, bowl_cy, z_mix])

    print("\n=== Stirring Controller (step-by-step) ===")
    print("  Assumes base is already at the bowl. No place / return steps.")
    print(f"  ladle : {ladle_pos.tolist()}")
    print(f"  bowl  : {bowl_pos.tolist()}")
    print(f"  mix z : {z_mix:.4f} m   above bowl z : {z_above:.4f} m")
    print("  Keys: Enter = next step | q = quit\n")

    pose = read_current_ee_world(redis_client)
    if pose is None:
        print(
            "WARNING: Redis EE pose unavailable. Is OpenSAI cartesian control running?",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"  EE now: {pose[0].tolist()}\n", flush=True)

    closed = (close_w, spd, frc)
    opened = (open_w, spd, frc)

    def publish(pos: np.ndarray, ori: np.ndarray, grip: tuple[float, float, float] | None) -> None:
        _publish_cartesian(redis_client, pos, ori)
        if grip is not None:
            _set_gripper(redis_client, grip[0], grip[1], grip[2])
        print(f"  → goal pos: {np.asarray(pos).reshape(3).tolist()}", flush=True)

    def run_timed(step: str, detail: str, duration: float, tick) -> None:
        _step_pause(step, detail + f"  [{duration:.1f} s on Enter]", args.pause_sec)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            tick(time.perf_counter() - t0)
            time.sleep(dt)
        print(f"  ✓ {step} done ({duration:.1f} s)", flush=True)

    try:
        _step_pause(
            "START",
            "Robot assumed at bowl. OpenSAI + gripper driver must be running.",
            args.pause_sec,
        )

        _step_pause(
            "1 — ABOVE LADLE",
            f"Hover {hover_w().tolist()}, gripper open.",
            args.pause_sec,
        )
        publish(hover_w(), R_grasp, opened)

        _step_pause("2 — GRASP DESCEND", "Lower to grasp pose.", args.pause_sec)
        publish(grasp_w(), R_grasp, opened)

        _step_pause(
            "3 — DWELL OPEN",
            f"Hold open {args.grasp_open_dwell:.1f} s at grasp.",
            args.pause_sec,
        )
        publish(grasp_w(), R_grasp, opened)
        time.sleep(args.grasp_open_dwell)

        _step_pause("4 — CLOSE GRIPPER", "Close on ladle handle.", args.pause_sec)
        publish(grasp_w(), R_grasp, closed)
        time.sleep(_GRIPPER_SETTLE_S)

        _step_pause(
            "5 — DWELL CLOSED",
            f"Hold closed {args.grasp_close_dwell:.1f} s.",
            args.pause_sec,
        )
        publish(grasp_w(), R_grasp, closed)
        time.sleep(args.grasp_close_dwell)

        _step_pause("6 — LIFT LADLE", f"Lift to {lift_w().tolist()}.", args.pause_sec)
        publish(lift_w(), R_grasp, closed)

        def orient_tick(t: float) -> None:
            alpha = min(1.0, t / args.orient_dur)
            publish(lift_w(), _slerp(R_grasp, R_mix, alpha), closed)

        run_timed(
            "7 — ORIENT FOR MIX",
            "SLERP grasp → mixing orientation.",
            args.orient_dur,
            orient_tick,
        )

        _step_pause(
            "8 — ABOVE BOWL",
            f"Move above bowl {above_bowl.tolist()} (R_mix).",
            args.pause_sec,
        )
        publish(above_bowl, R_mix, closed)

        _step_pause(
            "9 — LOWER INTO BOWL",
            f"Descend to mix height {at_mix.tolist()}.",
            args.pause_sec,
        )
        publish(at_mix, R_mix, closed)

        def mix_tick(t: float) -> None:
            goal = np.array([
                bowl_cx + args.mix_radius * math.cos(args.mix_omega * t),
                bowl_cy + args.mix_radius * math.sin(args.mix_omega * t),
                z_mix,
            ])
            publish(goal, R_mix, closed)

        run_timed(
            "10 — STIR",
            f"Circle stir r={args.mix_radius} m ω={args.mix_omega} rad/s.",
            args.mix_dur,
            mix_tick,
        )

        print("\n>>> DONE — stirring complete.")
        return 0

    except _StepAbort:
        print("\nStopped by user.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted — holding last goal.")
        return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stirring only: grasp ladle → orient → descend → stir (at bowl)."
    )
    p.add_argument("--pause-sec", type=float, default=0.0, metavar="SEC",
                   help="Timed pause between steps instead of Enter.")
    p.add_argument("--ladle-xyz", type=float, nargs=3, default=list(DEFAULT_LADLE_POSITION),
                   metavar=("X", "Y", "Z"))
    p.add_argument("--bowl-xyz", type=float, nargs=3, default=list(DEFAULT_BOWL_POSITION),
                   metavar=("X", "Y", "Z"))
    p.add_argument("--ladle-offset", type=float, nargs=3,
                   default=list(_LADLE_GRASP_OFFSET_DEFAULT), metavar=("DX", "DY", "DZ"))
    p.add_argument("--hover-dz", type=float, default=_HOVER_DZ_DEFAULT)
    p.add_argument("--lift-dz", type=float, default=_LIFT_DZ_DEFAULT)
    p.add_argument("--z-mix-ee", type=float, default=None)
    p.add_argument("--z-above-bowl", type=float, default=None)
    p.add_argument("--mix-radius", type=float, default=_MIX_RADIUS_DEFAULT)
    p.add_argument("--mix-omega", type=float, default=_MIX_OMEGA_DEFAULT)
    p.add_argument("--mix-dur", type=float, default=_MIX_DUR_DEFAULT)
    p.add_argument("--orient-dur", type=float, default=_ORIENT_DUR_DEFAULT)
    p.add_argument("--grasp-open-dwell", type=float, default=_GRASP_OPEN_DWELL_DEFAULT)
    p.add_argument("--grasp-close-dwell", type=float, default=_GRASP_CLOSE_DWELL_DEFAULT)
    p.add_argument("--loop-hz", type=int, default=_LOOP_HZ_DEFAULT)
    p.add_argument("--gripper-open-width", type=float, default=None)
    p.add_argument("--gripper-close-width", type=float, default=_GRIPPER_CLOSE_WIDTH_DEFAULT)
    p.add_argument("--gripper-speed", type=float, default=_GRIPPER_SPEED_DEFAULT)
    p.add_argument("--gripper-force", type=float, default=_GRIPPER_FORCE_DEFAULT)
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    redis_client = _try_redis(args.redis_host, args.redis_port)
    if redis_client is None:
        return 1
    err = validate_config(redis_client)
    if err is not None:
        return err

    ladle_pos = np.array(args.ladle_xyz, dtype=np.float64).reshape(3)
    bowl_pos = np.array(args.bowl_xyz, dtype=np.float64).reshape(3)
    return run_stirring(redis_client, args, ladle_pos, bowl_pos)


if __name__ == "__main__":
    sys.exit(main())
