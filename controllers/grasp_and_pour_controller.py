#!/usr/bin/env python3
"""Fixed-pose grasp + lift/pour on OpenSai Franka (Redis only).

Grasp pose is a fixed cartesian pose (same numeric values as the ``touch_controller``
home pose, but here it is the **grasp** target). No Gemini, no camera. Every state
transition is **manually gated by ENTER**.

Startup (no ENTER): immediately move **above** the grasp pose (``+lift-dz`` in
world Z) with the gripper **open**.

Then one ENTER per step:

- ENTER (1) — descend to the grasp pose.
- ENTER (2) — close gripper.
- ENTER (3) — lift +Z by ``--lift-dz``.
- ENTER (4) — start pour slerp (+90° about world +X); runs to completion automatically.
- ``q`` (then ENTER) / Ctrl+C — quit.

Requires OpenSai ``cartesian_controller``, Franka arm driver, and Franka gripper
Redis driver (``opensai::FrankaRobot::gripper::*``).

Usage::

  python ZitiBot/controllers/grasp_and_pour_controller.py
  python ZitiBot/controllers/grasp_and_pour_controller.py --lift-dz 0.12
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import select
import sys
import time
from dataclasses import dataclass

import numpy as np
import redis
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

CONFIG_XML = os.environ.get("ZITIBOT_OPENSAI_CONFIG_XML", "zitibot_panda.xml")
CONTROLLER_TO_USE = "cartesian_controller"


@dataclass(frozen=True)
class OpenSaiRedisKeys:
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


@dataclass(frozen=True)
class _RedisKeys(OpenSaiRedisKeys):
    gripper_mode: str = "opensai::FrankaRobot::gripper::mode"
    gripper_desired_width: str = "opensai::FrankaRobot::gripper::desired_width"
    gripper_desired_speed: str = "opensai::FrankaRobot::gripper::desired_speed"
    gripper_desired_force: str = "opensai::FrankaRobot::gripper::desired_force"
    gripper_max_width: str = "opensai::FrankaRobot::gripper::max_width"
    gripper_current_width: str = "opensai::FrankaRobot::gripper::current_width"


# Franka gripper driver modes (see drivers/FrankaPanda/redis_driver/gripper.cpp).
GRIPPER_MODE_MOVE = "m"  # position move, no force (use for opening)
GRIPPER_MODE_GRASP = "g"  # close until contact at desired_force (use for closing)
GRIPPER_MODE_OPEN_MAX = "o"  # snap fully open


_KEYS = _RedisKeys()

# Fixed grasp pose (same numbers as touch_controller's home pose; kept in sync
# manually — do not import touch_controller because it has Redis side effects at
# module load).
_GRASP_YAW_RAD = np.radians(45.0)
_c, _s = np.cos(_GRASP_YAW_RAD), np.sin(_GRASP_YAW_RAD)
GRASP_POSITION = np.array([0.4, -0.2, 0.35])
GRASP_ORIENTATION = np.array(
    [
        [_c, -_s, 0.0],
        [-_s, -_c, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

DEFAULT_LIFT_DZ_M = 0.15
DEFAULT_TILT_DURATION_S = 6.0
DEFAULT_GRIPPER_SPEED = 0.1
DEFAULT_GRIPPER_FORCE = 50.0
POUR_TICK_DT_S = 0.05


class Phase(enum.Enum):
    ABOVE_GRASP = "ABOVE_GRASP"
    AT_GRASP = "AT_GRASP"
    CLOSED = "CLOSED"
    LIFTED = "LIFTED"
    POURING = "POURING"
    DONE = "DONE"


@dataclass
class MotionParams:
    lift_dz_m: float
    tilt_duration_s: float
    gripper_open_width: float | None
    gripper_close_width: float
    gripper_speed: float
    gripper_force: float


@dataclass
class PourState:
    lift_world: np.ndarray
    R_start: np.ndarray
    R_end: np.ndarray
    t0: float
    last_tick: float = 0.0


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
            "Warning: ::sai-interfaces-webui::config_file_name not in Redis; "
            "continuing anyway.",
            file=sys.stderr,
        )
        return None
    if name != CONFIG_XML:
        print(
            f"Expected webui config {CONFIG_XML!r} but Redis has {name!r}. "
            "Set ZITIBOT_OPENSAI_CONFIG_XML if needed.",
            file=sys.stderr,
        )
        return 1
    return None


def read_current_ee_world(redis_client) -> tuple[np.ndarray, np.ndarray] | None:
    """Return current ``(position (3,), orientation (3,3))`` from Redis."""
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


def _ensure_cartesian_controller(redis_client) -> None:
    while (
        _decode_redis_value(redis_client.get(_KEYS.active_controller))
        != CONTROLLER_TO_USE
    ):
        redis_client.set(_KEYS.active_controller, CONTROLLER_TO_USE)


def _publish_cartesian(
    redis_client,
    goal_pos: np.ndarray,
    goal_ori: np.ndarray,
) -> None:
    _ensure_cartesian_controller(redis_client)
    redis_client.set(
        _KEYS.cartesian_task_goal_position,
        json.dumps(np.asarray(goal_pos, dtype=np.float64).reshape(3).tolist()),
    )
    redis_client.set(
        _KEYS.cartesian_task_goal_orientation,
        json.dumps(np.asarray(goal_ori, dtype=np.float64).reshape(3, 3).tolist()),
    )


def read_gripper_max_width(redis_client) -> float | None:
    raw = redis_client.get(_KEYS.gripper_max_width)
    text = _decode_redis_value(raw)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_gripper_current_width(redis_client) -> float | None:
    raw = redis_client.get(_KEYS.gripper_current_width)
    text = _decode_redis_value(raw)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def set_gripper_width(
    redis_client,
    width_m: float,
    *,
    speed: float,
    force: float,
    mode: str = GRIPPER_MODE_MOVE,
) -> None:
    redis_client.set(_KEYS.gripper_desired_width, str(float(width_m)))
    redis_client.set(_KEYS.gripper_desired_speed, str(float(speed)))
    redis_client.set(_KEYS.gripper_desired_force, str(float(force)))
    redis_client.set(_KEYS.gripper_mode, mode)


def resolve_gripper_open_width(redis_client, override: float | None) -> float:
    if override is not None:
        return float(override)
    w = read_gripper_max_width(redis_client)
    if w is not None and w > 0:
        return w
    return 0.08


def pour_orientation_end(R_start: np.ndarray) -> np.ndarray:
    """+90° about world +X."""
    R_x = R.from_euler("x", math.pi / 2.0).as_matrix()
    return R_x @ np.asarray(R_start, dtype=np.float64).reshape(3, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-pose grasp + lift/pour, ENTER-gated (OpenSai Franka Redis)."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--lift-dz", type=float, default=DEFAULT_LIFT_DZ_M)
    p.add_argument("--tilt-duration", type=float, default=DEFAULT_TILT_DURATION_S)
    p.add_argument(
        "--gripper-open-width",
        type=float,
        default=None,
        help="Open width (m); default reads gripper::max_width from Redis.",
    )
    p.add_argument("--gripper-close-width", type=float, default=0.0)
    p.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED)
    p.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE)
    return p.parse_args()


def _phase_hint(phase: Phase) -> str:
    if phase == Phase.ABOVE_GRASP:
        return "Next: ENTER = descend to grasp pose"
    if phase == Phase.AT_GRASP:
        return "Next: ENTER = close gripper"
    if phase == Phase.CLOSED:
        return "Next: ENTER = lift +Z"
    if phase == Phase.LIFTED:
        return "Next: ENTER = start pour (+90° world X)"
    if phase == Phase.POURING:
        return "Pouring… (auto-completes)"
    if phase == Phase.DONE:
        return "Done — q to quit"
    return ""


def _do_move_above_grasp(
    redis_client,
    grasp_pos: np.ndarray,
    grasp_ori: np.ndarray,
    motion: MotionParams,
) -> np.ndarray:
    above = grasp_pos + np.array([0.0, 0.0, motion.lift_dz_m])
    _publish_cartesian(redis_client, above, grasp_ori)
    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
    set_gripper_width(
        redis_client,
        open_w,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_MOVE,
    )
    print(
        f"[0] Move above grasp: pos={above.tolist()}, "
        f"gripper open width={open_w:.4f} m (mode=m)"
    )
    return above


def _do_descend_to_grasp(
    redis_client,
    grasp_pos: np.ndarray,
    grasp_ori: np.ndarray,
) -> None:
    _publish_cartesian(redis_client, grasp_pos, grasp_ori)
    print(f"[1] Descend to grasp: pos={grasp_pos.tolist()}")


def _do_close_gripper(redis_client, motion: MotionParams) -> None:
    set_gripper_width(
        redis_client,
        motion.gripper_close_width,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_MOVE,
    )
    print(
        f"[2] Close gripper (move): width={motion.gripper_close_width:.4f} m, "
        f"speed={motion.gripper_speed:.3f} m/s (mode=m)"
    )


def _do_lift(
    redis_client,
    grasp_pos: np.ndarray,
    grasp_ori: np.ndarray,
    motion: MotionParams,
) -> np.ndarray:
    lift_world = grasp_pos + np.array([0.0, 0.0, motion.lift_dz_m])
    _publish_cartesian(redis_client, lift_world, grasp_ori)
    print(f"[3] Lift goal: {lift_world.tolist()}")
    return lift_world


def _start_pour(
    redis_client,
    lift_world: np.ndarray,
    now: float,
) -> PourState | None:
    pose = read_current_ee_world(redis_client)
    if pose is None:
        print("Could not read current EE pose from Redis; not starting pour.")
        return None
    _, cur_ori = pose
    R_start = cur_ori.copy()
    R_end = pour_orientation_end(R_start)
    _publish_cartesian(redis_client, lift_world, R_start)
    print("[4] Pour started: slerp +90° about world X.")
    return PourState(
        lift_world=np.asarray(lift_world, dtype=np.float64).reshape(3).copy(),
        R_start=R_start,
        R_end=R_end,
        t0=now,
    )


def _tick_pour(
    redis_client,
    pour: PourState,
    motion: MotionParams,
    now: float,
) -> bool:
    """Advance pour slerp; return True when complete."""
    if now - pour.last_tick < POUR_TICK_DT_S:
        return False
    pour.last_tick = now
    alpha = min(1.0, max(0.0, (now - pour.t0) / motion.tilt_duration_s))
    key_rots = R.concatenate(
        [R.from_matrix(pour.R_start), R.from_matrix(pour.R_end)]
    )
    R_tilt = Slerp([0.0, 1.0], key_rots)([alpha]).as_matrix()[0]
    _publish_cartesian(redis_client, pour.lift_world, R_tilt)
    return alpha >= 1.0


_STDIN_EOF = object()


def _stdin_line_ready(timeout_s: float):
    """Return one line from stdin if available within ``timeout_s``.

    - ``None`` — no input within ``timeout_s``.
    - ``_STDIN_EOF`` — stdin is at EOF / closed. Caller should stop polling stdin.
    - Otherwise the raw line (with trailing newline).

    Line-buffered: user presses ENTER to submit. Bare ENTER advances the phase;
    ``q``/``quit`` quits.
    """
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    except (ValueError, OSError):
        return _STDIN_EOF
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":
        return _STDIN_EOF
    return line


def run_loop(redis_client, motion: MotionParams) -> int:
    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
    motion = MotionParams(
        lift_dz_m=motion.lift_dz_m,
        tilt_duration_s=motion.tilt_duration_s,
        gripper_open_width=open_w,
        gripper_close_width=motion.gripper_close_width,
        gripper_speed=motion.gripper_speed,
        gripper_force=motion.gripper_force,
    )
    print(
        f"Motion: lift_dz={motion.lift_dz_m} m, "
        f"tilt={motion.tilt_duration_s} s, gripper open={motion.gripper_open_width:.4f} m"
    )

    grasp_pos = GRASP_POSITION.copy()
    grasp_ori = GRASP_ORIENTATION.copy()
    print(
        f"Grasp pose:\n"
        f"  pos = {grasp_pos.tolist()}\n"
        f"  ori =\n{np.array2string(grasp_ori, precision=4, suppress_small=True)}"
    )
    print(
        "Keys (ENTER to submit): "
        "[empty]=advance phase | q=quit"
    )

    _do_move_above_grasp(redis_client, grasp_pos, grasp_ori, motion)
    phase = Phase.ABOVE_GRASP
    lift_world: np.ndarray | None = None
    pour: PourState | None = None
    stdin_dead = False
    print(_phase_hint(phase))

    try:
        while True:
            now = time.perf_counter()

            if phase == Phase.POURING and pour is not None:
                if _tick_pour(redis_client, pour, motion, now):
                    phase = Phase.DONE
                    print("Pour complete (hold).")
                    print(_phase_hint(phase))

            if stdin_dead:
                # stdin is closed (e.g., backgrounded by a launcher script).
                # Stay alive so the pour can finish; user must Ctrl+C to quit.
                time.sleep(POUR_TICK_DT_S)
                continue

            line = _stdin_line_ready(POUR_TICK_DT_S)
            if line is None:
                continue
            if line is _STDIN_EOF:
                print(
                    "stdin closed (no terminal attached). State will NOT advance.\n"
                    "Run this controller directly in a terminal, not via a launcher\n"
                    "that backgrounds it. Ctrl+C to quit.",
                    file=sys.stderr,
                )
                stdin_dead = True
                continue
            token = line.strip().lower()
            if token in ("q", "quit", "exit"):
                print("Quit requested.")
                return 0
            if token != "":
                print(f"(unknown input: {token!r}; press ENTER to advance, q to quit)")
                continue

            if phase == Phase.ABOVE_GRASP:
                _do_descend_to_grasp(redis_client, grasp_pos, grasp_ori)
                phase = Phase.AT_GRASP
            elif phase == Phase.AT_GRASP:
                _do_close_gripper(redis_client, motion)
                phase = Phase.CLOSED
            elif phase == Phase.CLOSED:
                lift_world = _do_lift(redis_client, grasp_pos, grasp_ori, motion)
                phase = Phase.LIFTED
            elif phase == Phase.LIFTED:
                assert lift_world is not None
                pour = _start_pour(redis_client, lift_world, now)
                if pour is not None:
                    phase = Phase.POURING
            elif phase == Phase.POURING:
                print("Pour in progress — wait for it to finish.")
            elif phase == Phase.DONE:
                print("Sequence done — q to quit.")

            print(_phase_hint(phase))
    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")
        return 0


def main() -> int:
    args = parse_args()
    redis_client = _try_redis(args.redis_host, args.redis_port)
    if redis_client is None:
        return 1
    err = validate_config(redis_client)
    if err is not None:
        return err

    motion = MotionParams(
        lift_dz_m=args.lift_dz,
        tilt_duration_s=args.tilt_duration,
        gripper_open_width=args.gripper_open_width,
        gripper_close_width=args.gripper_close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
    )
    return run_loop(redis_client, motion)


if __name__ == "__main__":
    sys.exit(main())
