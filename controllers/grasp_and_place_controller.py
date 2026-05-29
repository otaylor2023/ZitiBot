#!/usr/bin/env python3
"""Fixed-pose grasp + transport + place on OpenSai Franka (Redis only).

Pick and place positions are configurable (world frame, meters). End-effector
orientation stays **fixed** for the whole sequence (tool Z down, 45° yaw). Pick
and place use **Z-only** descents (XY held at the target).

Every state transition is **manually gated by ENTER**.

Startup (no ENTER): move **above** the pick pose (``+lift-dz`` in world Z) with gripper
**open**.

Then one ENTER per step:

- ENTER (1) — descend in Z to pick pose.
- ENTER (2) — close gripper.
- ENTER (3) — lift +Z by ``--lift-dz``.
- ENTER (4) — move above place pose (same orientation).
- ENTER (5) — descend in Z to place pose (same level as ``--place``).
- ENTER (6) — open gripper.
- ``q`` / Ctrl+C — quit.

Requires OpenSai ``cartesian_controller``, Franka arm driver, and Franka gripper
Redis driver (``opensai::FrankaRobot::gripper::*``). Also run
``sai_franka_gripper_redis_driver`` separately from ``launch_driver.sh``.

Usage::

  python ZitiBot/controllers/grasp_and_place_controller.py
  python ZitiBot/controllers/grasp_and_place_controller.py \\
      --pick 0.4 -0.2 0.35 --place 0.55 0.15 0.35 --lift-dz 0.12
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import redis

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
GRIPPER_MODE_MOVE = "m"
GRIPPER_MODE_GRASP = "g"
GRIPPER_MODE_OPEN_MAX = "o"

_KEYS = _RedisKeys()

_GRASP_YAW_RAD = np.radians(45.0)
_c, _s = np.cos(_GRASP_YAW_RAD), np.sin(_GRASP_YAW_RAD)
# World-frame default grasp/place coordinates (meters).
DEFAULT_PICK_POSITION = np.array([0.364646, 0.019878, 0.353623])
DEFAULT_PLACE_POSITION = np.array([0.775011, -0.0787334, 0.331972])
EE_ORIENTATION = np.array(
    [
        [_c, -_s, 0.0],
        [-_s, -_c, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

DEFAULT_LIFT_DZ_M = 0.15
DEFAULT_GRIPPER_SPEED = 0.1
DEFAULT_GRIPPER_FORCE = 50.0
CONTROLLER_WAIT_DT_S = 0.05
CONTROLLER_WAIT_TIMEOUT_S = 30.0


class Phase(enum.Enum):
    ABOVE_PICK = "ABOVE_PICK"
    AT_PICK = "AT_PICK"
    CLOSED = "CLOSED"
    LIFTED = "LIFTED"
    ABOVE_PLACE = "ABOVE_PLACE"
    AT_PLACE = "AT_PLACE"
    DONE = "DONE"


@dataclass
class MotionParams:
    lift_dz_m: float
    gripper_open_width: float | None
    gripper_close_width: float
    gripper_speed: float
    gripper_force: float


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


def _ensure_cartesian_controller(redis_client) -> None:
    t0 = time.monotonic()
    while True:
        if (
            _decode_redis_value(redis_client.get(_KEYS.active_controller))
            == CONTROLLER_TO_USE
        ):
            return
        if time.monotonic() - t0 > CONTROLLER_WAIT_TIMEOUT_S:
            print(
                f"Warning: could not switch active_controller to {CONTROLLER_TO_USE!r} "
                f"within {CONTROLLER_WAIT_TIMEOUT_S:.0f} s; publishing goals anyway.",
                file=sys.stderr,
            )
            return
        redis_client.set(_KEYS.active_controller, CONTROLLER_TO_USE)
        time.sleep(CONTROLLER_WAIT_DT_S)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grasp at pick pose, place at place pose (fixed EE orientation), ENTER-gated."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--pick",
        nargs=3,
        type=float,
        default=DEFAULT_PICK_POSITION.tolist(),
        metavar=("X", "Y", "Z"),
        help="Pick position in world frame (m).",
    )
    p.add_argument(
        "--place",
        nargs=3,
        type=float,
        default=DEFAULT_PLACE_POSITION.tolist(),
        metavar=("X", "Y", "Z"),
        help="Place position in world frame (m).",
    )
    p.add_argument("--lift-dz", type=float, default=DEFAULT_LIFT_DZ_M)
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
    if phase == Phase.ABOVE_PICK:
        return "Next: ENTER = descend in Z to pick pose"
    if phase == Phase.AT_PICK:
        return "Next: ENTER = close gripper"
    if phase == Phase.CLOSED:
        return "Next: ENTER = lift +Z"
    if phase == Phase.LIFTED:
        return "Next: ENTER = move above place pose"
    if phase == Phase.ABOVE_PLACE:
        return "Next: ENTER = descend in Z to place pose"
    if phase == Phase.AT_PLACE:
        return "Next: ENTER = open gripper"
    if phase == Phase.DONE:
        return "Sequence complete — q to quit"
    return ""


def _do_move_above(
    redis_client,
    target_pos: np.ndarray,
    ee_ori: np.ndarray,
    motion: MotionParams,
    *,
    label: str,
    open_gripper: bool,
) -> np.ndarray:
    above = target_pos + np.array([0.0, 0.0, motion.lift_dz_m])
    _publish_cartesian(redis_client, above, ee_ori)
    if open_gripper:
        open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
        set_gripper_width(
            redis_client,
            open_w,
            speed=motion.gripper_speed,
            force=motion.gripper_force,
            mode=GRIPPER_MODE_MOVE,
        )
        print(f"{label} Move above: pos={above.tolist()}, gripper open width={open_w:.4f} m")
    else:
        print(f"{label} Move above: pos={above.tolist()}")
    return above


def _do_descend_z_only(
    redis_client,
    xy: np.ndarray,
    target_z: float,
    ee_ori: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    goal = np.array([float(xy[0]), float(xy[1]), float(target_z)], dtype=np.float64)
    _publish_cartesian(redis_client, goal, ee_ori)
    print(f"{label} Descend (Z only): pos={goal.tolist()}")
    return goal


def _do_close_gripper(redis_client, motion: MotionParams) -> None:
    set_gripper_width(
        redis_client,
        motion.gripper_close_width,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_GRASP,
    )
    print(
        f"[2] Close gripper (grasp): force={motion.gripper_force:.1f} N, "
        f"speed={motion.gripper_speed:.3f} m/s (mode=g)"
    )


def _do_lift(
    redis_client,
    from_pos: np.ndarray,
    ee_ori: np.ndarray,
    motion: MotionParams,
) -> None:
    lift_world = from_pos + np.array([0.0, 0.0, motion.lift_dz_m])
    _publish_cartesian(redis_client, lift_world, ee_ori)
    print(f"[3] Lift goal: {lift_world.tolist()}")


def _do_open_gripper(redis_client, motion: MotionParams) -> None:
    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
    set_gripper_width(
        redis_client,
        open_w,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_OPEN_MAX,
    )
    print(f"[6] Open gripper (mode=o): width={open_w:.4f} m max")


def _read_command(phase: Phase) -> str | None:
    """Block until the user presses ENTER (or types q). Returns None on EOF."""
    hint = _phase_hint(phase)
    prompt = f"\n[{phase.value}] {hint}\n> "
    try:
        sys.stdout.flush()
        return input(prompt)
    except EOFError:
        return None


def run_loop(
    redis_client,
    motion: MotionParams,
    pick_pos: np.ndarray,
    place_pos: np.ndarray,
) -> int:
    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
    motion = MotionParams(
        lift_dz_m=motion.lift_dz_m,
        gripper_open_width=open_w,
        gripper_close_width=motion.gripper_close_width,
        gripper_speed=motion.gripper_speed,
        gripper_force=motion.gripper_force,
    )
    ee_ori = EE_ORIENTATION.copy()

    print(
        f"Motion: lift_dz={motion.lift_dz_m} m, "
        f"gripper open={motion.gripper_open_width:.4f} m, "
        f"force={motion.gripper_force:.0f} N"
    )
    print(
        f"Pick pos  = {pick_pos.tolist()}\n"
        f"Place pos = {place_pos.tolist()}\n"
        f"EE ori (fixed) =\n{np.array2string(ee_ori, precision=4, suppress_small=True)}"
    )
    if not sys.stdin.isatty():
        print(
            "Warning: stdin is not a terminal (ENTER may not work). Run in a real shell:\n"
            "  python3 -u ZitiBot/controllers/grasp_and_place_controller.py",
            file=sys.stderr,
            flush=True,
        )
    print("Keys: press ENTER (empty line) to advance | q=quit", flush=True)

    _do_move_above(
        redis_client,
        pick_pos,
        ee_ori,
        motion,
        label="[0]",
        open_gripper=True,
    )
    phase = Phase.ABOVE_PICK
    print(_phase_hint(phase), flush=True)

    try:
        while True:
            line = _read_command(phase)
            if line is None:
                print(
                    "stdin closed — run this script in an interactive terminal "
                    "(not backgrounded). Ctrl+C to quit.",
                    file=sys.stderr,
                    flush=True,
                )
                return 0
            token = line.strip().lower()
            if token in ("q", "quit", "exit"):
                print("Quit requested.")
                return 0
            if token != "":
                print(f"(unknown input: {token!r}; press ENTER to advance, q to quit)")
                continue

            if phase == Phase.ABOVE_PICK:
                _do_descend_z_only(
                    redis_client,
                    pick_pos[:2],
                    float(pick_pos[2]),
                    ee_ori,
                    label="[1]",
                )
                phase = Phase.AT_PICK
            elif phase == Phase.AT_PICK:
                _do_close_gripper(redis_client, motion)
                phase = Phase.CLOSED
            elif phase == Phase.CLOSED:
                _do_lift(redis_client, pick_pos, ee_ori, motion)
                phase = Phase.LIFTED
            elif phase == Phase.LIFTED:
                _do_move_above(
                    redis_client,
                    place_pos,
                    ee_ori,
                    motion,
                    label="[4]",
                    open_gripper=False,
                )
                phase = Phase.ABOVE_PLACE
            elif phase == Phase.ABOVE_PLACE:
                _do_descend_z_only(
                    redis_client,
                    place_pos[:2],
                    float(place_pos[2]),
                    ee_ori,
                    label="[5]",
                )
                phase = Phase.AT_PLACE
            elif phase == Phase.AT_PLACE:
                _do_open_gripper(redis_client, motion)
                phase = Phase.DONE
                print("Sequence complete.", flush=True)
            elif phase == Phase.DONE:
                pass

            if phase != Phase.DONE:
                print(_phase_hint(phase), flush=True)
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

    pick_pos = np.array(args.pick, dtype=np.float64).reshape(3)
    place_pos = np.array(args.place, dtype=np.float64).reshape(3)

    motion = MotionParams(
        lift_dz_m=args.lift_dz,
        gripper_open_width=args.gripper_open_width,
        gripper_close_width=args.gripper_close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
    )
    return run_loop(redis_client, motion, pick_pos, place_pos)


if __name__ == "__main__":
    sys.exit(main())
