#!/usr/bin/env python3
"""Fixed-pose grasp + transport/pour on OpenSai Franka (Redis only).

Pick and pour poses are fixed in the world frame. No Gemini, no camera. Every state
transition is **manually gated by ENTER**.

Startup (no ENTER): move **above** the pick pose (pick XY, Z = pick Z + ``--approach-dz``)
with the gripper **open**.

Then one ENTER per step:

- ENTER (1) — descend to the pick pose.
- ENTER (2) — close gripper (held until step 8; no re-open in between).
- ENTER (3) — move directly to the pour pose.
- ENTER (4) — start pour slerp (``--pour-tilt-deg``, default 90° about world +X) at pour pose.
- ENTER (5) — return orientation to grasp (reverse slerp at pour pose).
- ENTER (6) — retract vertically at pour site (clearance height).
- ENTER (7) — move to original pick position (bowl placement).
- ENTER (8) — open gripper to release the bowl.
- ``q`` / Ctrl+C — quit.

Requires OpenSai ``cartesian_controller``, Franka arm driver, and Franka gripper
Redis driver (``opensai::FrankaRobot::gripper::*``).

Usage::

  python ZitiBot/controllers/grasp_and_pour_controller.py
  python ZitiBot/controllers/grasp_and_pour_controller.py --approach-dz 0.15
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

# Fixed pick / pour poses (world frame, meters). EE orientation is fixed for the
# whole sequence (tool Z down, 45° yaw) except during pour slerp.
_GRASP_YAW_RAD = np.radians(45.0)
_c, _s = np.cos(_GRASP_YAW_RAD), np.sin(_GRASP_YAW_RAD)
PICK_POSITION = np.array([0.49764, 0.436106, 0.369818])
POUR_POSITION = np.array([0.522549, 0.115722, 0.628443])
GRASP_POSITION = PICK_POSITION  # alias used by descend/return helpers
GRASP_ORIENTATION = np.array(
    [
        [_c, -_s, 0.0],
        [-_s, -_c, 0.0],
        [0.0, 0.0, -1.0],
    ]
)

DEFAULT_APPROACH_DZ_M = 0.15
DEFAULT_POUR_TILT_DEG = 90.0
DEFAULT_POUR_AXIS = "x"
DEFAULT_TILT_DURATION_S = 6.0
DEFAULT_GRIPPER_SPEED = 0.1
DEFAULT_GRIPPER_FORCE = 50.0
DEFAULT_GRIPPER_PREGRASP_WIDTH = 0.05
DEFAULT_GRIPPER_PREGRASP_SETTLE_S = 0.6
DEFAULT_GRIPPER_GRASP_SETTLE_S = 1.2
DEFAULT_POS_TOL_M = 0.03
DEFAULT_TRANSIT_DWELL_S = 2.5
POUR_TICK_DT_S = 0.05


class Phase(enum.Enum):
    ABOVE_GRASP = "ABOVE_GRASP"
    AT_GRASP = "AT_GRASP"
    CLOSED = "CLOSED"
    LIFTED = "LIFTED"
    POURING = "POURING"
    POURED = "POURED"
    RETURNING = "RETURNING"
    RETURNED = "RETURNED"
    ABOVE_PICK_RETURN = "ABOVE_PICK_RETURN"
    AT_RETURN = "AT_RETURN"
    DONE = "DONE"


@dataclass
class MotionParams:
    approach_dz_m: float
    pour_tilt_deg: float
    pour_axis: str
    tilt_duration_s: float
    gripper_open_width: float | None
    gripper_pregrasp_width: float
    gripper_close_width: float
    gripper_speed: float
    gripper_force: float
    gripper_pregrasp_settle_s: float
    gripper_grasp_settle_s: float


@dataclass
class OrientationSlerpState:
    """Hold position fixed while interpolating orientation (pour or return)."""

    hold_world: np.ndarray
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


def pour_orientation_end(
    R_start: np.ndarray, tilt_deg: float, axis: str = DEFAULT_POUR_AXIS
) -> np.ndarray:
    """Pour tilt: rotation about world +X or +Y by ``tilt_deg`` (default +X)."""
    ax = axis.strip().lower()
    if ax == "x":
        R_tilt = R.from_euler("x", math.radians(tilt_deg)).as_matrix()
    elif ax == "y":
        R_tilt = R.from_euler("y", math.radians(tilt_deg)).as_matrix()
    else:
        raise ValueError(f"Unknown pour axis {axis!r}; use 'x' or 'y'.")
    return R_tilt @ np.asarray(R_start, dtype=np.float64).reshape(3, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-pose grasp + lift/pour, ENTER-gated (OpenSai Franka Redis)."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--approach-dz",
        type=float,
        default=DEFAULT_APPROACH_DZ_M,
        help="Vertical clearance above pick for approach and return retract (m).",
    )
    p.add_argument(
        "--pour-tilt-deg",
        type=float,
        default=DEFAULT_POUR_TILT_DEG,
        help="Pour rotation about world +X (degrees); default 90.",
    )
    p.add_argument(
        "--pour-axis",
        choices=("x", "y"),
        default=DEFAULT_POUR_AXIS,
        help="World axis for pour tilt (default x).",
    )
    p.add_argument("--tilt-duration", type=float, default=DEFAULT_TILT_DURATION_S)
    p.add_argument(
        "--gripper-open-width",
        type=float,
        default=None,
        help="Open width (m); default reads gripper::max_width from Redis.",
    )
    p.add_argument(
        "--gripper-pregrasp-width",
        type=float,
        default=DEFAULT_GRIPPER_PREGRASP_WIDTH,
        help="Partial open width before final grasp (m).",
    )
    p.add_argument("--gripper-close-width", type=float, default=0.0)
    p.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED)
    p.add_argument("--gripper-force", type=float, default=DEFAULT_GRIPPER_FORCE)
    p.add_argument(
        "--gripper-pregrasp-settle",
        type=float,
        default=DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    )
    p.add_argument(
        "--gripper-grasp-settle",
        type=float,
        default=DEFAULT_GRIPPER_GRASP_SETTLE_S,
        help="Wait after grasp command before moving arm (s).",
    )
    return p.parse_args()


def _phase_hint(phase: Phase) -> str:
    if phase == Phase.ABOVE_GRASP:
        return "Next: ENTER = descend to grasp pose"
    if phase == Phase.AT_GRASP:
        return "Next: ENTER = close gripper"
    if phase == Phase.CLOSED:
        return "Next: ENTER = move to pour pose"
    if phase == Phase.LIFTED:
        return "Next: ENTER = start pour (world +X tilt)"
    if phase == Phase.POURING:
        return "Pouring… (auto-completes)"
    if phase == Phase.POURED:
        return "Next: ENTER = return orientation to grasp"
    if phase == Phase.RETURNING:
        return "Returning orientation… (auto-completes)"
    if phase == Phase.RETURNED:
        return "Next: ENTER = retract up at pour site"
    if phase == Phase.ABOVE_PICK_RETURN:
        return "Next: ENTER = move to original pick position"
    if phase == Phase.AT_RETURN:
        return "Next: ENTER = open gripper (release bowl)"
    if phase == Phase.DONE:
        return "Done — q to quit"
    return ""


def _above_pick(pick_pos: np.ndarray, motion: MotionParams) -> np.ndarray:
    """Pick XY with vertical clearance for approach / retract."""
    return pick_pos + np.array([0.0, 0.0, motion.approach_dz_m])


def _do_move_above_grasp(
    redis_client,
    grasp_pos: np.ndarray,
    grasp_ori: np.ndarray,
    motion: MotionParams,
) -> np.ndarray:
    """Startup only: approach pose with gripper open (sole open before final release)."""
    above = _above_pick(grasp_pos, motion)
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
    *,
    label: str = "[1] Descend to grasp",
) -> None:
    _publish_cartesian(redis_client, grasp_pos, grasp_ori)
    print(f"{label}: pos={grasp_pos.tolist()}")


def _do_grasp_object(redis_client, motion: MotionParams) -> None:
    """Pregrasp (move mode) then force grasp; wait before arm motion resumes."""
    pre_w = float(motion.gripper_pregrasp_width)
    set_gripper_width(
        redis_client,
        pre_w,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_MOVE,
    )
    print(
        f"[2a] Pregrasp: width={pre_w:.4f} m (mode=m), "
        f"settle {motion.gripper_pregrasp_settle_s:.1f} s"
    )
    time.sleep(motion.gripper_pregrasp_settle_s)

    set_gripper_width(
        redis_client,
        motion.gripper_close_width,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_GRASP,
    )
    print(
        f"[2b] Grasp: force={motion.gripper_force:.1f} N (mode=g), "
        f"settle {motion.gripper_grasp_settle_s:.1f} s before moving"
    )
    time.sleep(motion.gripper_grasp_settle_s)


def _do_move_to_pour(
    redis_client,
    pour_pos: np.ndarray,
    grasp_ori: np.ndarray,
) -> np.ndarray:
    """Move directly to the pour pose."""
    pour = np.asarray(pour_pos, dtype=np.float64).reshape(3).copy()
    _publish_cartesian(redis_client, pour, grasp_ori)
    print(f"[3] Move to pour pose: pos={pour.tolist()}")
    return pour


def _do_retract_from_pour(
    redis_client,
    pour_pos: np.ndarray,
    grasp_ori: np.ndarray,
    motion: MotionParams,
) -> np.ndarray:
    """Retract vertically at the pour site (pour XY, raised Z for clearance)."""
    retract = np.asarray(pour_pos, dtype=np.float64).reshape(3).copy()
    retract[2] = max(
        float(pour_pos[2]),
        float(_above_pick(PICK_POSITION, motion)[2]),
    )
    _publish_cartesian(redis_client, retract, grasp_ori)
    print(f"[6] Retract at pour site: pos={retract.tolist()}")
    return retract


def _do_move_to_pick(
    redis_client,
    pick_pos: np.ndarray,
    grasp_ori: np.ndarray,
) -> np.ndarray:
    """Move directly to the original pick pose."""
    pick = np.asarray(pick_pos, dtype=np.float64).reshape(3).copy()
    _publish_cartesian(redis_client, pick, grasp_ori)
    print(f"[7] Move to original pick position: pos={pick.tolist()}")
    return pick


def _start_orientation_slerp(
    redis_client,
    hold_world: np.ndarray,
    R_start: np.ndarray,
    R_end: np.ndarray,
    now: float,
    *,
    label: str,
) -> OrientationSlerpState:
    hold = np.asarray(hold_world, dtype=np.float64).reshape(3).copy()
    R0 = np.asarray(R_start, dtype=np.float64).reshape(3, 3).copy()
    R1 = np.asarray(R_end, dtype=np.float64).reshape(3, 3).copy()
    _publish_cartesian(redis_client, hold, R0)
    print(label)
    return OrientationSlerpState(hold_world=hold, R_start=R0, R_end=R1, t0=now)


def _start_pour(
    redis_client,
    pour_world: np.ndarray,
    motion: MotionParams,
    now: float,
) -> OrientationSlerpState | None:
    pose = read_current_ee_world(redis_client)
    if pose is None:
        print("Could not read current EE pose from Redis; not starting pour.")
        return None
    _, cur_ori = pose
    R_start = cur_ori.copy()
    R_end = pour_orientation_end(
        R_start, motion.pour_tilt_deg, motion.pour_axis
    )
    return _start_orientation_slerp(
        redis_client,
        pour_world,
        R_start,
        R_end,
        now,
        label=(
            f"[4] Pour started: slerp {motion.pour_tilt_deg:.0f}° "
            f"about world +{motion.pour_axis.upper()}."
        ),
    )


def _start_return(
    redis_client,
    pour_world: np.ndarray,
    grasp_ori: np.ndarray,
    poured_ori: np.ndarray,
    now: float,
) -> OrientationSlerpState:
    return _start_orientation_slerp(
        redis_client,
        pour_world,
        poured_ori,
        grasp_ori,
        now,
        label="[5] Return started: slerp back to grasp orientation.",
    )


def _tick_orientation_slerp(
    redis_client,
    slerp: OrientationSlerpState,
    motion: MotionParams,
    now: float,
) -> bool:
    """Advance orientation slerp; return True when complete."""
    if now - slerp.last_tick < POUR_TICK_DT_S:
        return False
    slerp.last_tick = now
    alpha = min(1.0, max(0.0, (now - slerp.t0) / motion.tilt_duration_s))
    key_rots = R.concatenate(
        [R.from_matrix(slerp.R_start), R.from_matrix(slerp.R_end)]
    )
    R_interp = Slerp([0.0, 1.0], key_rots)([alpha]).as_matrix()[0]
    _publish_cartesian(redis_client, slerp.hold_world, R_interp)
    return alpha >= 1.0


def _do_open_gripper(redis_client, motion: MotionParams) -> None:
    open_w = resolve_gripper_open_width(redis_client, motion.gripper_open_width)
    set_gripper_width(
        redis_client,
        open_w,
        speed=motion.gripper_speed,
        force=motion.gripper_force,
        mode=GRIPPER_MODE_OPEN_MAX,
    )
    print(f"[8] Open gripper (mode=o): width={open_w:.4f} m max")


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
        approach_dz_m=motion.approach_dz_m,
        pour_tilt_deg=motion.pour_tilt_deg,
        pour_axis=motion.pour_axis,
        tilt_duration_s=motion.tilt_duration_s,
        gripper_open_width=open_w,
        gripper_pregrasp_width=motion.gripper_pregrasp_width,
        gripper_close_width=motion.gripper_close_width,
        gripper_speed=motion.gripper_speed,
        gripper_force=motion.gripper_force,
        gripper_pregrasp_settle_s=motion.gripper_pregrasp_settle_s,
        gripper_grasp_settle_s=motion.gripper_grasp_settle_s,
    )
    print(
        f"Motion: approach_dz={motion.approach_dz_m} m, "
        f"pour_tilt={motion.pour_tilt_deg:.0f}° world +{motion.pour_axis.upper()}, "
        f"tilt_dur={motion.tilt_duration_s} s, "
        f"pregrasp={motion.gripper_pregrasp_width:.4f} m, "
        f"gripper open={motion.gripper_open_width:.4f} m"
    )

    pick_pos = PICK_POSITION.copy()
    pour_pos = POUR_POSITION.copy()
    grasp_ori = GRASP_ORIENTATION.copy()
    print(
        f"Pick pose:\n"
        f"  pos = {pick_pos.tolist()}\n"
        f"Pour pose:\n"
        f"  pos = {pour_pos.tolist()}\n"
        f"EE ori (fixed, except pour slerp) =\n"
        f"{np.array2string(grasp_ori, precision=4, suppress_small=True)}"
    )
    print(
        "Keys (ENTER to submit): "
        "[empty]=advance phase | q=quit"
    )

    _do_move_above_grasp(redis_client, pick_pos, grasp_ori, motion)
    phase = Phase.ABOVE_GRASP
    pour_world: np.ndarray | None = None
    slerp: OrientationSlerpState | None = None
    poured_ori: np.ndarray | None = None
    stdin_dead = False
    print(_phase_hint(phase))

    try:
        while True:
            now = time.perf_counter()

            if phase in (Phase.POURING, Phase.RETURNING) and slerp is not None:
                if _tick_orientation_slerp(redis_client, slerp, motion, now):
                    if phase == Phase.POURING:
                        poured_ori = slerp.R_end.copy()
                        phase = Phase.POURED
                        slerp = None
                        print("Pour complete (hold).")
                    else:
                        phase = Phase.RETURNED
                        slerp = None
                        print("Return orientation complete (at pour pose).")
                    print(_phase_hint(phase))

            if stdin_dead:
                # stdin is closed (e.g., backgrounded by a launcher script).
                # Stay alive so pour/return slerps can finish; Ctrl+C to quit.
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
                _do_descend_to_grasp(redis_client, pick_pos, grasp_ori)
                phase = Phase.AT_GRASP
            elif phase == Phase.AT_GRASP:
                _do_grasp_object(redis_client, motion)
                phase = Phase.CLOSED
            elif phase == Phase.CLOSED:
                pour_world = _do_move_to_pour(redis_client, pour_pos, grasp_ori)
                phase = Phase.LIFTED
            elif phase == Phase.LIFTED:
                assert pour_world is not None
                slerp = _start_pour(redis_client, pour_world, motion, now)
                if slerp is not None:
                    phase = Phase.POURING
            elif phase == Phase.POURING:
                print("Pour in progress — wait for it to finish.")
            elif phase == Phase.POURED:
                assert pour_world is not None and poured_ori is not None
                slerp = _start_return(
                    redis_client,
                    pour_world,
                    grasp_ori,
                    poured_ori,
                    now,
                )
                phase = Phase.RETURNING
            elif phase == Phase.RETURNING:
                print("Return in progress — wait for it to finish.")
            elif phase == Phase.RETURNED:
                _do_retract_from_pour(redis_client, pour_pos, grasp_ori, motion)
                phase = Phase.ABOVE_PICK_RETURN
            elif phase == Phase.ABOVE_PICK_RETURN:
                _do_move_to_pick(redis_client, pick_pos, grasp_ori)
                phase = Phase.AT_RETURN
            elif phase == Phase.AT_RETURN:
                _do_open_gripper(redis_client, motion)
                phase = Phase.DONE
                print("Bowl released at original pose.")
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
        approach_dz_m=args.approach_dz,
        pour_tilt_deg=args.pour_tilt_deg,
        pour_axis=args.pour_axis,
        tilt_duration_s=args.tilt_duration,
        gripper_open_width=args.gripper_open_width,
        gripper_pregrasp_width=args.gripper_pregrasp_width,
        gripper_close_width=args.gripper_close_width,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        gripper_pregrasp_settle_s=args.gripper_pregrasp_settle,
        gripper_grasp_settle_s=args.gripper_grasp_settle,
    )
    return run_loop(redis_client, motion)


if __name__ == "__main__":
    sys.exit(main())
