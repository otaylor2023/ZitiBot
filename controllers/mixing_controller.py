#!/usr/bin/env python3
"""Mixing controller — step-by-step pickup-mix-putdown with fixed poses.

Like grasp_and_place_controller.py: ladle and bowl positions are hardcoded
defaults (world frame, meters), overridable via --ladle-xyz / --bowl-xyz.
Press Enter before each step. No camera or Gemini API required.

Mirrors the sim mixing-task controller.cpp FSM, adapted for the real Franka arm:
  - Arm control  : OpenSAI cartesian controller (Redis goal_position/orientation)
  - Base motion  : TidyBot hb1::desired_pose / hb1::current_pose (Redis)
  - Gripper      : Franka gripper Redis driver (desired_width/speed/force)

FSM (12 states):
  MOVE_BASE_GRASP      navigate base to ladle; arm holds travel pose
  EE_ABOVE_LADLE       hover above ladle grasp point; gripper open
  EE_GRASP_LADLE       4-phase: descend → dwell open → close → dwell closed
  EE_LIFT_LADLE        lift ladle above countertop
  EE_ORIENT_LADLE      SLERP R_grasp → R_mix (configurable duration)
  BASE_TO_BOWL         navigate base to bowl; arm holds lift pose in world frame
  EE_ABOVE_BOWL        move EE above bowl at R_mix
  EE_LOWER_INTO_BOWL   descend to mixing height
  EE_MIX               circular stirring for mix_dur seconds
  EE_LIFT_FROM_BOWL    lift EE out of bowl
  BASE_TO_LADLE_REST   navigate base back; arm holds above-bowl pose in world frame
  EE_PLACE_LADLE       3-phase: hover+SLERP back → descend → open gripper

Usage::
  # Defaults (tune DEFAULT_* below for your lab):
  python3 -u controllers/mixing_controller.py

  # Override poses:
  python3 -u controllers/mixing_controller.py \\
      --ladle-xyz 0.75 0.68 0.508 --bowl-xyz 0.17 0.62 0.50 --skip-base
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import redis

try:
    from scipy.spatial.transform import Rotation as ScipyR, Slerp
except ImportError:
    print(
        "Missing dependency: scipy\n"
        "  pip install scipy\n"
        "  # or install all ZitiBot deps:\n"
        "  pip install -r ZitiBot/requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)

_CTRL_DIR = Path(__file__).resolve().parent
if str(_CTRL_DIR) not in sys.path:
    sys.path.insert(0, str(_CTRL_DIR))

CONFIG_XML = os.environ.get("ZITIBOT_OPENSAI_CONFIG_XML", "zitibot_panda.xml")
CONTROLLER_TO_USE = "cartesian_controller"

# ── Defaults (all overridable via CLI) ─────────────────────────────────────────
# Fixed object poses in world frame (m) — tune to your lab setup.
DEFAULT_LADLE_POSITION = (0.75, 0.68, 0.508)
DEFAULT_BOWL_POSITION  = (0.17, 0.62, 0.50)

# Base targets [x, y, yaw_rad] in hb1 odometry frame.
_BASE_LADLE_DEFAULT  = (0.78, 0.22, math.pi / 2)
_BASE_BOWL_DEFAULT   = (0.13, 0.22, math.pi / 2)

# Ladle geometry (world frame offsets from ladle origin, same as controller.cpp).
_LADLE_GRASP_OFFSET_DEFAULT = (0.22, 0.031, 0.02)
_HOVER_DZ_DEFAULT            = 0.15   # m above grasp to hover before descent
_LIFT_DZ_DEFAULT             = 0.24   # m to lift after closing gripper

# Mixing heights.
_Z_MIX_EE_DEFAULT      = None   # computed: bowl_z + 0.283 if not set
_Z_ABOVE_BOWL_DEFAULT  = None   # computed: z_mix_ee + 0.10 if not set

# Mixing motion.
_MIX_RADIUS_DEFAULT  = 0.05   # stirring circle radius (m)
_MIX_OMEGA_DEFAULT   = 1.0    # stirring angular rate (rad/s)
_MIX_DUR_DEFAULT     = 10.0   # stirring duration (s)

# Convergence.
_BASE_XY_TOL_DEFAULT  = 0.015   # m per axis
_BASE_YAW_TOL_DEFAULT = 0.04    # rad
_EE_POS_TOL_DEFAULT   = 0.025   # m
_GRIPPER_TOL_DEFAULT  = 0.008   # m

# Timing.
_ORIENT_DUR_DEFAULT         = 3.0   # SLERP duration (s)
_GRASP_OPEN_DWELL_DEFAULT   = 1.0   # dwell with gripper open at grasp (s)
_GRASP_CLOSE_DWELL_DEFAULT  = 1.0   # dwell with gripper closed at grasp (s)
_GRIPPER_SETTLE_S           = 0.5   # wait after gripper command (s)
_LOOP_HZ_DEFAULT            = 20

# Gripper.
_GRIPPER_CLOSE_WIDTH_DEFAULT = 0.0
_GRIPPER_SPEED_DEFAULT       = 0.1
_GRIPPER_FORCE_DEFAULT       = 50.0

# ── Redis keys ─────────────────────────────────────────────────────────────────
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
class _Keys(OpenSaiRedisKeys):
    gripper_desired_width:  str = "opensai::FrankaRobot::gripper::desired_width"
    gripper_desired_speed:  str = "opensai::FrankaRobot::gripper::desired_speed"
    gripper_desired_force:  str = "opensai::FrankaRobot::gripper::desired_force"
    gripper_max_width:      str = "opensai::FrankaRobot::gripper::max_width"
    gripper_current_width:  str = "opensai::FrankaRobot::gripper::current_width"
    base_current_pose:      str = "hb1::current_pose"    # [x, y, yaw]
    base_desired_pose:      str = "hb1::desired_pose"
    base_stop:              str = "hb1::stop"


_KEYS = _Keys()


# ── Orientation helpers ────────────────────────────────────────────────────────
def _link7_orientation_world(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """3×3 rotation for link7 at given RPY. Ported from controller.cpp.

    Applies Rx(π) (tool-down) then ZYX RPY, then Rz(+π/4) finger-align correction.
    """
    d2r = math.pi / 180.0
    R_full = (
        ScipyR.from_euler("z", yaw_deg   * d2r) *
        ScipyR.from_euler("y", pitch_deg * d2r) *
        ScipyR.from_euler("x", roll_deg  * d2r) *
        ScipyR.from_euler("x", math.pi) *
        ScipyR.from_euler("z", math.pi / 4)
    )
    return R_full.as_matrix()


def _ladle_mixing_orientation() -> np.ndarray:
    """Handle +Z (up), bowl-end −Z (down). Hardcoded from controller.cpp."""
    return np.array([[0, 0, 1],
                     [0, -1, 0],
                     [1,  0, 0]], dtype=np.float64)


def _slerp(R_start: np.ndarray, R_end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = min(1.0, max(0.0, alpha))
    key_rots = ScipyR.concatenate([ScipyR.from_matrix(R_start),
                                   ScipyR.from_matrix(R_end)])
    return Slerp([0.0, 1.0], key_rots)([alpha]).as_matrix()[0]


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ── Redis helpers ──────────────────────────────────────────────────────────────
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


def _read_gripper_width(redis_client) -> float | None:
    raw = redis_client.get(_KEYS.gripper_current_width)
    if raw is None:
        return None
    try:
        return float(raw.decode("utf-8"))
    except (ValueError, AttributeError):
        return None


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


def _read_base_pose(redis_client) -> np.ndarray | None:
    raw = redis_client.get(_KEYS.base_current_pose)
    if raw is None:
        return None
    try:
        return np.array(json.loads(raw), dtype=np.float64).reshape(3)
    except Exception:
        return None


def _publish_base_goal(redis_client, goal: np.ndarray) -> None:
    redis_client.set(_KEYS.base_desired_pose, json.dumps(goal.tolist()))


def _base_converged(cur: np.ndarray | None, goal: np.ndarray,
                    xy_tol: float, yaw_tol: float) -> bool:
    if cur is None:
        return False
    return (abs(cur[0] - goal[0]) < xy_tol and
            abs(cur[1] - goal[1]) < xy_tol and
            abs(_wrap_angle(cur[2] - goal[2])) < yaw_tol)


def _ee_converged(cur: np.ndarray | None, goal: np.ndarray, tol: float) -> bool:
    return cur is not None and np.linalg.norm(cur - goal) < tol


def _gripper_converged(redis_client, target_w: float, tol: float) -> bool:
    w = _read_gripper_width(redis_client)
    return w is not None and abs(w - target_w) < tol


# ── Step-by-step prompts ─────────────────────────────────────────────────────
class _StepAbort(Exception):
    pass


def _read_step_line(prompt: str) -> str | None:
    """Read a line from the controlling terminal (works when stdin is piped)."""
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
    """Block until Enter. Returns False if user quits or no terminal."""
    msg = f"\n── {step} ──\n"
    if detail:
        msg += f"   {detail}\n"
    if pause_sec > 0:
        print(msg, end="", flush=True)
        print(f"   (auto-pause {pause_sec:.1f} s — no Enter needed)", flush=True)
        time.sleep(pause_sec)
        return True
    msg += "   Press Enter to continue (q to quit): "
    line = _read_step_line(msg)
    if line is None:
        print(
            "\nERROR: No interactive terminal for step input.\n"
            "  Run in a real shell, e.g.:\n"
            "    python3 -u controllers/mixing_controller.py --ladle-xyz ... --bowl-xyz ...\n"
            "  Or pass --pause-sec 3 to insert timed pauses without Enter.",
            file=sys.stderr,
            flush=True,
        )
        return False
    if line.strip().lower() in ("q", "quit", "exit"):
        print("Quit requested.")
        return False
    return True


def _step_pause(step: str, detail: str, args) -> None:
    if not _wait_for_enter(step, detail, pause_sec=args.pause_sec):
        raise _StepAbort()


# ── FSM ───────────────────────────────────────────────────────────────────────
class _State(enum.Enum):
    MOVE_BASE_GRASP    = "MOVE_BASE_GRASP"
    EE_ABOVE_LADLE     = "EE_ABOVE_LADLE"
    EE_GRASP_LADLE     = "EE_GRASP_LADLE"
    EE_LIFT_LADLE      = "EE_LIFT_LADLE"
    EE_ORIENT_LADLE    = "EE_ORIENT_LADLE"
    BASE_TO_BOWL       = "BASE_TO_BOWL"
    EE_ABOVE_BOWL      = "EE_ABOVE_BOWL"
    EE_LOWER_INTO_BOWL = "EE_LOWER_INTO_BOWL"
    EE_MIX             = "EE_MIX"
    EE_LIFT_FROM_BOWL  = "EE_LIFT_FROM_BOWL"
    BASE_TO_LADLE_REST = "BASE_TO_LADLE_REST"
    EE_PLACE_LADLE     = "EE_PLACE_LADLE"
    DONE               = "DONE"


# ── Manual step-by-step FSM (Enter between steps; like grasp_and_place) ─────
def _run_fsm_manual(
    redis_client,
    args,
    ladle_pos: np.ndarray,
    bowl_pos: np.ndarray,
) -> int:
    """Publish one step per Enter; you decide when the robot is ready to advance."""
    ladle_grasp_offset = np.array(args.ladle_offset, dtype=np.float64)
    bowl_cx, bowl_cy, bowl_cz = bowl_pos

    z_mix_ee = (args.z_mix_ee if args.z_mix_ee is not None
                else round(bowl_cz + 0.283, 4))
    z_above_bowl = (args.z_above_bowl if args.z_above_bowl is not None
                    else round(z_mix_ee + 0.10, 4))

    base_ladle_goal = np.array(args.base_ladle, dtype=np.float64)
    base_bowl_goal = np.array(args.base_bowl, dtype=np.float64)

    R_grasp = _link7_orientation_world(0.0, 0.0, 45.0)
    R_mix = _ladle_mixing_orientation()

    open_w = _gripper_open_width(redis_client, args.gripper_open_width)
    close_w = args.gripper_close_width
    spd, frc = args.gripper_speed, args.gripper_force
    dt = 1.0 / max(1, args.loop_hz)

    def grasp_w() -> np.ndarray:
        return ladle_pos + ladle_grasp_offset

    def hover_w() -> np.ndarray:
        return grasp_w() + np.array([0.0, 0.0, args.hover_dz])

    def lift_w() -> np.ndarray:
        return grasp_w() + np.array([0.0, 0.0, args.lift_dz])

    above_bowl_pos = np.array([bowl_cx, bowl_cy, z_above_bowl])
    at_bowl_pos = np.array([bowl_cx, bowl_cy, z_mix_ee])

    print("\n=== Mixing Controller (step-by-step) ===")
    print("  Each Enter publishes the next goal. Watch the robot, then press Enter.")
    print("  Timed steps (orient / mix / dwells) run automatically after you start them.")
    print(f"  ladle pos : {ladle_pos.tolist()}")
    print(f"  bowl pos  : {bowl_pos.tolist()}")
    print(f"  skip_base : {args.skip_base}")
    print("  Keys: Enter = next step | q = quit")
    print("==============================\n")

    pose = read_current_ee_world(redis_client)
    if pose is None:
        print(
            "WARNING: Redis EE pose unavailable — arm may not move. "
            "Is OpenSAI running with the cartesian controller?",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"  EE now: {pose[0].tolist()}\n", flush=True)

    def publish(pos: np.ndarray, ori: np.ndarray, grip: tuple[float, float, float] | None) -> None:
        _publish_cartesian(redis_client, pos, ori)
        if grip is not None:
            _set_gripper(redis_client, grip[0], grip[1], grip[2])
        print(f"  → goal pos: {np.asarray(pos).reshape(3).tolist()}", flush=True)

    def run_timed(step: str, detail: str, duration: float, tick) -> None:
        _step_pause(step, detail + f"  [{duration:.1f} s motion starts on Enter]", args)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            tick(time.perf_counter() - t0)
            time.sleep(dt)
        print(f"  ✓ {step} timed motion finished ({duration:.1f} s)", flush=True)

    closed = (close_w, spd, frc)
    opened = (open_w, spd, frc)

    try:
        _step_pause(
            "START",
            "Pipeline ready. Ensure OpenSAI + (optional) TidyBot base are running.",
            args,
        )

        if not args.skip_base:
            pose = read_current_ee_world(redis_client)
            hold_pos = pose[0] if pose is not None else np.zeros(3)
            hold_ori = pose[1] if pose is not None else np.eye(3)
            _step_pause(
                "MOVE_BASE_GRASP",
                f"Drive base to ladle {base_ladle_goal.tolist()}. "
                "Press Enter when the base has arrived.",
                args,
            )
            publish(hold_pos, hold_ori, closed)
            _publish_base_goal(redis_client, base_ladle_goal)
            print(f"  → base goal: {base_ladle_goal.tolist()}", flush=True)
        else:
            print("  (skip_base: base navigation steps omitted)", flush=True)

        _step_pause(
            "EE_ABOVE_LADLE",
            f"Move EE above ladle hover {hover_w().tolist()}, gripper open.",
            args,
        )
        publish(hover_w(), R_grasp, opened)

        _step_pause("EE_GRASP_LADLE — descend", "Descend to grasp pose.", args)
        publish(grasp_w(), R_grasp, opened)

        _step_pause(
            "EE_GRASP_LADLE — dwell open",
            f"Hold at grasp with gripper open ({args.grasp_open_dwell:.1f} s).",
            args,
        )
        publish(grasp_w(), R_grasp, opened)
        time.sleep(args.grasp_open_dwell)

        _step_pause("EE_GRASP_LADLE — close", "Close gripper on handle.", args)
        publish(grasp_w(), R_grasp, closed)
        time.sleep(_GRIPPER_SETTLE_S)

        _step_pause(
            "EE_GRASP_LADLE — dwell closed",
            f"Hold closed ({args.grasp_close_dwell:.1f} s).",
            args,
        )
        publish(grasp_w(), R_grasp, closed)
        time.sleep(args.grasp_close_dwell)

        _step_pause("EE_LIFT_LADLE", f"Lift to {lift_w().tolist()}.", args)
        publish(lift_w(), R_grasp, closed)

        def orient_tick(t_elapsed: float) -> None:
            alpha = min(1.0, t_elapsed / args.orient_dur)
            publish(lift_w(), _slerp(R_grasp, R_mix, alpha), closed)

        run_timed(
            "EE_ORIENT_LADLE",
            "SLERP grasp → mixing orientation.",
            args.orient_dur,
            orient_tick,
        )

        if not args.skip_base:
            _step_pause(
                "BASE_TO_BOWL",
                f"Drive base to bowl {base_bowl_goal.tolist()}.",
                args,
            )
            publish(lift_w(), R_mix, closed)
            _publish_base_goal(redis_client, base_bowl_goal)
            print(f"  → base goal: {base_bowl_goal.tolist()}", flush=True)

        _step_pause("EE_ABOVE_BOWL", f"Move above bowl {above_bowl_pos.tolist()}.", args)
        publish(above_bowl_pos, R_mix, closed)

        _step_pause("EE_LOWER_INTO_BOWL", f"Lower to mix height {at_bowl_pos.tolist()}.", args)
        publish(at_bowl_pos, R_mix, closed)

        def mix_tick(t_elapsed: float) -> None:
            ee_goal = np.array([
                bowl_cx + args.mix_radius * math.cos(args.mix_omega * t_elapsed),
                bowl_cy + args.mix_radius * math.sin(args.mix_omega * t_elapsed),
                z_mix_ee,
            ])
            publish(ee_goal, R_mix, closed)

        run_timed(
            "EE_MIX",
            f"Stir bowl (r={args.mix_radius}, ω={args.mix_omega}).",
            args.mix_dur,
            mix_tick,
        )

        _step_pause("EE_LIFT_FROM_BOWL", f"Lift to {above_bowl_pos.tolist()}.", args)
        publish(above_bowl_pos, R_mix, closed)

        if not args.skip_base:
            _step_pause(
                "BASE_TO_LADLE_REST",
                f"Drive base back to ladle {base_ladle_goal.tolist()}.",
                args,
            )
            publish(above_bowl_pos, R_mix, closed)
            _publish_base_goal(redis_client, base_ladle_goal)
            print(f"  → base goal: {base_ladle_goal.tolist()}", flush=True)

        def place_orient_tick(t_elapsed: float) -> None:
            alpha = min(1.0, t_elapsed / args.orient_dur)
            publish(hover_w(), _slerp(R_mix, R_grasp, alpha), closed)

        run_timed(
            "EE_PLACE_LADLE — hover + SLERP",
            "Return to hover pose and grasp orientation.",
            args.orient_dur,
            place_orient_tick,
        )

        _step_pause("EE_PLACE_LADLE — descend", "Lower to ladle rest height.", args)
        publish(grasp_w(), R_grasp, closed)

        _step_pause("EE_PLACE_LADLE — open", "Open gripper to release ladle.", args)
        publish(grasp_w(), R_grasp, opened)
        time.sleep(_GRIPPER_SETTLE_S)

        print("\n>>> DONE — ladle released, task complete.")
        return 0

    except _StepAbort:
        print("\nStopped by user.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted — holding last goal.")
        return 0


# ── Autonomous FSM ─────────────────────────────────────────────────────────────
def _run_fsm(redis_client, args,
             ladle_pos: np.ndarray,
             bowl_pos: np.ndarray) -> int:
    if not args.auto:
        return _run_fsm_manual(redis_client, args, ladle_pos, bowl_pos)
    # Derived geometry.
    ladle_grasp_offset = np.array(args.ladle_offset, dtype=np.float64)
    bowl_cx, bowl_cy, bowl_cz = bowl_pos

    z_mix_ee     = (args.z_mix_ee     if args.z_mix_ee     is not None
                    else round(bowl_cz + 0.283, 4))
    z_above_bowl = (args.z_above_bowl if args.z_above_bowl is not None
                    else round(z_mix_ee + 0.10, 4))

    base_ladle_goal = np.array(args.base_ladle, dtype=np.float64)
    base_bowl_goal  = np.array(args.base_bowl,  dtype=np.float64)

    R_grasp = _link7_orientation_world(0.0, 0.0, 45.0)
    R_mix   = _ladle_mixing_orientation()

    open_w  = _gripper_open_width(redis_client, args.gripper_open_width)
    close_w = args.gripper_close_width
    spd, frc = args.gripper_speed, args.gripper_force

    dt = 1.0 / max(1, args.loop_hz)

    # Derived position constants (recomputed each tick once ladle_pos may be locked).
    ladle_pos_locked = False

    def ladle_grasp_world() -> np.ndarray:
        return ladle_pos + ladle_grasp_offset

    def ladle_hover_world() -> np.ndarray:
        return ladle_grasp_world() + np.array([0, 0, args.hover_dz])

    def ladle_lift_world() -> np.ndarray:
        return ladle_grasp_world() + np.array([0, 0, args.lift_dz])

    above_bowl_pos = np.array([bowl_cx, bowl_cy, z_above_bowl])
    at_bowl_pos    = np.array([bowl_cx, bowl_cy, z_mix_ee])

    print("\n=== Mixing Controller (--auto) ===")
    print(f"  ladle pos    : {ladle_pos.tolist()}")
    print(f"  bowl pos     : {bowl_pos.tolist()}")
    print(f"  grasp target : {ladle_grasp_world().tolist()}")
    print(f"  hover target : {ladle_hover_world().tolist()}")
    print(f"  lift  target : {ladle_lift_world().tolist()}")
    print(f"  z_mix_ee     : {z_mix_ee:.4f} m   z_above_bowl : {z_above_bowl:.4f} m")
    print(f"  mix          : r={args.mix_radius} m  ω={args.mix_omega} rad/s  "
          f"dur={args.mix_dur} s")
    print(f"  base_ladle   : {base_ladle_goal.tolist()}")
    print(f"  base_bowl    : {base_bowl_goal.tolist()}")
    print(f"  skip_base    : {args.skip_base}")
    print("==============================\n")

    state      = _State.MOVE_BASE_GRASP
    prev_state = None

    grasp_phase = 0
    place_phase = 0
    phase_t0    = 0.0
    orient_t0   = 0.0
    mix_t0      = 0.0
    gripper_cmd_t0 = 0.0
    last_mix_print = 0.0

    # EE goal held during base navigation (world frame).
    hold_ee_pos: np.ndarray | None = None
    hold_ee_ori: np.ndarray | None = None

    try:
        while state != _State.DONE:
            now  = time.perf_counter()
            pose = read_current_ee_world(redis_client)
            cur_pos = pose[0] if pose is not None else None
            cur_ori = pose[1] if pose is not None else None
            base    = _read_base_pose(redis_client)

            # ── Derived targets (recomputed each tick) ────────────────────────
            grasp_w = ladle_grasp_world()
            hover_w = ladle_hover_world()
            lift_w  = ladle_lift_world()

            # ── State entry banner ────────────────────────────────────────────
            if state != prev_state:
                t_str = f"[t={now:8.2f}s]"
                print(f"{t_str} >>> {state.value}")

                if state == _State.MOVE_BASE_GRASP:
                    # Hold current EE pose during base navigation.
                    hold_ee_pos = cur_pos.copy() if cur_pos is not None else np.zeros(3)
                    hold_ee_ori = cur_ori.copy() if cur_ori is not None else np.eye(3)
                    print(f"  base goal   : {base_ladle_goal.tolist()}")
                    print(f"  base now    : {base.tolist() if base is not None else 'unknown'}")
                    print(f"  hold EE pos : {hold_ee_pos.tolist()}")

                elif state == _State.EE_ABOVE_LADLE:
                    _set_gripper(redis_client, open_w, spd, frc)
                    print(f"  hover target: {hover_w.tolist()}")
                    print(f"  EE now      : {cur_pos.tolist() if cur_pos is not None else 'unknown'}")

                elif state == _State.EE_GRASP_LADLE:
                    grasp_phase = 0
                    print(f"  grasp target: {grasp_w.tolist()}")
                    print(f"  4-phase sub-FSM: descend → dwell_open → close → dwell_closed")

                elif state == _State.EE_LIFT_LADLE:
                    print(f"  lift target : {lift_w.tolist()}")

                elif state == _State.EE_ORIENT_LADLE:
                    orient_t0 = now
                    print(f"  SLERP R_grasp → R_mix over {args.orient_dur:.1f} s")
                    print(f"  holding pos : {lift_w.tolist()}")

                elif state == _State.BASE_TO_BOWL:
                    hold_ee_pos = lift_w.copy()
                    hold_ee_ori = R_mix.copy()
                    print(f"  base goal   : {base_bowl_goal.tolist()}")
                    print(f"  hold EE pos : {hold_ee_pos.tolist()}")

                elif state == _State.EE_ABOVE_BOWL:
                    print(f"  bowl above  : {above_bowl_pos.tolist()}")

                elif state == _State.EE_LOWER_INTO_BOWL:
                    print(f"  mix height  : {at_bowl_pos.tolist()}")

                elif state == _State.EE_MIX:
                    mix_t0 = last_mix_print = now
                    print(f"  Stirring for {args.mix_dur:.1f} s  "
                          f"(r={args.mix_radius} m, ω={args.mix_omega} rad/s)")

                elif state == _State.EE_LIFT_FROM_BOWL:
                    print(f"  lift to     : {above_bowl_pos.tolist()}")

                elif state == _State.BASE_TO_LADLE_REST:
                    hold_ee_pos = above_bowl_pos.copy()
                    hold_ee_ori = R_mix.copy()
                    print(f"  base goal   : {base_ladle_goal.tolist()}")
                    print(f"  hold EE pos : {hold_ee_pos.tolist()}")

                elif state == _State.EE_PLACE_LADLE:
                    place_phase = 0
                    orient_t0   = now
                    print(f"  3-phase sub-FSM: hover+SLERP → descend → open")
                    print(f"  hover target: {hover_w.tolist()}")

                prev_state = state

            # ── FSM tick ──────────────────────────────────────────────────────

            if state == _State.MOVE_BASE_GRASP:
                _publish_cartesian(redis_client, hold_ee_pos, hold_ee_ori)
                if not args.skip_base:
                    _publish_base_goal(redis_client, base_ladle_goal)
                    if _base_converged(base, base_ladle_goal,
                                       args.base_xy_tol, args.base_yaw_tol):
                        state = _State.EE_ABOVE_LADLE
                else:
                    state = _State.EE_ABOVE_LADLE

            elif state == _State.EE_ABOVE_LADLE:
                _publish_cartesian(redis_client, hover_w, R_grasp)
                _set_gripper(redis_client, open_w, spd, frc)
                if _ee_converged(cur_pos, hover_w, args.ee_pos_tol):
                    state = _State.EE_GRASP_LADLE

            elif state == _State.EE_GRASP_LADLE:
                # Phase 0: descend to grasp.
                if grasp_phase == 0:
                    _publish_cartesian(redis_client, grasp_w, R_grasp)
                    _set_gripper(redis_client, open_w, spd, frc)
                    if _ee_converged(cur_pos, grasp_w, args.ee_pos_tol):
                        grasp_phase = 1
                        phase_t0    = now
                        print(f"  [grasp ph0 → ph1]  EE at grasp. Dwelling open "
                              f"{args.grasp_open_dwell:.1f} s.")

                # Phase 1: dwell with gripper open.
                elif grasp_phase == 1:
                    _publish_cartesian(redis_client, grasp_w, R_grasp)
                    _set_gripper(redis_client, open_w, spd, frc)
                    if now - phase_t0 >= args.grasp_open_dwell:
                        grasp_phase = 2
                        _set_gripper(redis_client, close_w, spd, frc)
                        gripper_cmd_t0 = now
                        print(f"  [grasp ph1 → ph2]  Closing gripper.")

                # Phase 2: close gripper, wait for convergence.
                elif grasp_phase == 2:
                    _publish_cartesian(redis_client, grasp_w, R_grasp)
                    _set_gripper(redis_client, close_w, spd, frc)
                    if _gripper_converged(redis_client, close_w, args.gripper_tol) or \
                            (now - gripper_cmd_t0 >= _GRIPPER_SETTLE_S):
                        grasp_phase = 3
                        phase_t0    = now
                        print(f"  [grasp ph2 → ph3]  Gripper closed. Dwelling "
                              f"{args.grasp_close_dwell:.1f} s.")

                # Phase 3: dwell with gripper closed, then lock ladle_pos.
                elif grasp_phase == 3:
                    _publish_cartesian(redis_client, grasp_w, R_grasp)
                    _set_gripper(redis_client, close_w, spd, frc)
                    if now - phase_t0 >= args.grasp_close_dwell:
                        ladle_pos_locked = True
                        print(f"  [grasp ph3 → LIFT]  Ladle locked at {ladle_pos.tolist()}")
                        state = _State.EE_LIFT_LADLE

            elif state == _State.EE_LIFT_LADLE:
                _publish_cartesian(redis_client, lift_w, R_grasp)
                _set_gripper(redis_client, close_w, spd, frc)
                if _ee_converged(cur_pos, lift_w, args.ee_pos_tol):
                    state = _State.EE_ORIENT_LADLE

            elif state == _State.EE_ORIENT_LADLE:
                alpha    = (now - orient_t0) / args.orient_dur
                R_interp = _slerp(R_grasp, R_mix, alpha)
                _publish_cartesian(redis_client, lift_w, R_interp)
                _set_gripper(redis_client, close_w, spd, frc)
                if alpha >= 1.0:
                    state = _State.BASE_TO_BOWL

            elif state == _State.BASE_TO_BOWL:
                _publish_cartesian(redis_client, hold_ee_pos, hold_ee_ori)
                _set_gripper(redis_client, close_w, spd, frc)
                if not args.skip_base:
                    _publish_base_goal(redis_client, base_bowl_goal)
                    if _base_converged(base, base_bowl_goal,
                                       args.base_xy_tol, args.base_yaw_tol):
                        state = _State.EE_ABOVE_BOWL
                else:
                    state = _State.EE_ABOVE_BOWL

            elif state == _State.EE_ABOVE_BOWL:
                _publish_cartesian(redis_client, above_bowl_pos, R_mix)
                _set_gripper(redis_client, close_w, spd, frc)
                if _ee_converged(cur_pos, above_bowl_pos, args.ee_pos_tol):
                    state = _State.EE_LOWER_INTO_BOWL

            elif state == _State.EE_LOWER_INTO_BOWL:
                _publish_cartesian(redis_client, at_bowl_pos, R_mix)
                _set_gripper(redis_client, close_w, spd, frc)
                if _ee_converged(cur_pos, at_bowl_pos, args.ee_pos_tol):
                    state = _State.EE_MIX

            elif state == _State.EE_MIX:
                t = now - mix_t0
                ee_goal = np.array([
                    bowl_cx + args.mix_radius * math.cos(args.mix_omega * t),
                    bowl_cy + args.mix_radius * math.sin(args.mix_omega * t),
                    z_mix_ee,
                ])
                _publish_cartesian(redis_client, ee_goal, R_mix)
                _set_gripper(redis_client, close_w, spd, frc)
                if now - last_mix_print >= 2.0:
                    pos_str = f"  EE: {cur_pos.tolist()}" if cur_pos is not None else ""
                    print(f"  MIX {t:5.1f} / {args.mix_dur:.1f} s{pos_str}")
                    last_mix_print = now
                if t >= args.mix_dur:
                    state = _State.EE_LIFT_FROM_BOWL

            elif state == _State.EE_LIFT_FROM_BOWL:
                _publish_cartesian(redis_client, above_bowl_pos, R_mix)
                _set_gripper(redis_client, close_w, spd, frc)
                if _ee_converged(cur_pos, above_bowl_pos, args.ee_pos_tol):
                    state = _State.BASE_TO_LADLE_REST

            elif state == _State.BASE_TO_LADLE_REST:
                _publish_cartesian(redis_client, hold_ee_pos, hold_ee_ori)
                _set_gripper(redis_client, close_w, spd, frc)
                if not args.skip_base:
                    _publish_base_goal(redis_client, base_ladle_goal)
                    if _base_converged(base, base_ladle_goal,
                                       args.base_xy_tol, args.base_yaw_tol):
                        state = _State.EE_PLACE_LADLE
                else:
                    state = _State.EE_PLACE_LADLE

            elif state == _State.EE_PLACE_LADLE:
                # Phase 0: hover at ladle_hover_world while SLERPing back to R_grasp.
                if place_phase == 0:
                    alpha    = (now - orient_t0) / args.orient_dur
                    R_interp = _slerp(R_mix, R_grasp, alpha)
                    _publish_cartesian(redis_client, hover_w, R_interp)
                    _set_gripper(redis_client, close_w, spd, frc)
                    slerp_done = alpha >= 1.0
                    pos_done   = _ee_converged(cur_pos, hover_w, args.ee_pos_tol)
                    if slerp_done and pos_done:
                        place_phase = 1
                        print(f"  [place ph0 → ph1]  Hovering at rest with R_grasp. Descending.")

                # Phase 1: descend to ladle rest height with gripper closed.
                elif place_phase == 1:
                    _publish_cartesian(redis_client, grasp_w, R_grasp)
                    _set_gripper(redis_client, close_w, spd, frc)
                    if _ee_converged(cur_pos, grasp_w, args.ee_pos_tol):
                        place_phase    = 2
                        gripper_cmd_t0 = now
                        _set_gripper(redis_client, open_w, spd, frc)
                        print(f"  [place ph1 → ph2]  At rest height. Opening gripper.")

                # Phase 2: open gripper.
                elif place_phase == 2:
                    _publish_cartesian(redis_client, grasp_w, R_grasp)
                    _set_gripper(redis_client, open_w, spd, frc)
                    if _gripper_converged(redis_client, open_w, args.gripper_tol) or \
                            (now - gripper_cmd_t0 >= _GRIPPER_SETTLE_S):
                        state = _State.DONE

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nInterrupted — holding last goal.")
        return 0

    print("\n>>> DONE — ladle released, task complete.")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mixing controller — step-by-step pipeline (Enter between steps)."
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Run without Enter pauses (same behavior as mixing_full_controller.py).",
    )
    p.add_argument(
        "--pause-sec",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Timed pause between steps instead of Enter (for non-interactive terminals).",
    )

    # Object positions (fixed defaults; tune DEFAULT_* at top of file).
    p.add_argument("--ladle-xyz", type=float, nargs=3,
                   default=list(DEFAULT_LADLE_POSITION),
                   metavar=("X", "Y", "Z"),
                   help=f"Ladle rest position in world frame (m). "
                        f"Default: {DEFAULT_LADLE_POSITION}")
    p.add_argument("--bowl-xyz", type=float, nargs=3,
                   default=list(DEFAULT_BOWL_POSITION),
                   metavar=("X", "Y", "Z"),
                   help=f"Bowl centre in world frame (m). "
                        f"Default: {DEFAULT_BOWL_POSITION}")

    # Base navigation.
    p.add_argument("--base-ladle", type=float, nargs=3,
                   default=list(_BASE_LADLE_DEFAULT),
                   metavar=("X", "Y", "YAW"),
                   help=f"Base goal near ladle [x y yaw_rad]. Default: {_BASE_LADLE_DEFAULT}")
    p.add_argument("--base-bowl", type=float, nargs=3,
                   default=list(_BASE_BOWL_DEFAULT),
                   metavar=("X", "Y", "YAW"),
                   help=f"Base goal near bowl  [x y yaw_rad]. Default: {_BASE_BOWL_DEFAULT}")
    p.add_argument("--base-xy-tol",  type=float, default=_BASE_XY_TOL_DEFAULT)
    p.add_argument("--base-yaw-tol", type=float, default=_BASE_YAW_TOL_DEFAULT)
    p.add_argument("--skip-base", action="store_true",
                   help="Skip base navigation (useful for arm-only testing).")

    # Ladle geometry.
    p.add_argument("--ladle-offset", type=float, nargs=3,
                   default=list(_LADLE_GRASP_OFFSET_DEFAULT),
                   metavar=("DX", "DY", "DZ"),
                   help=f"Grasp offset from ladle origin (world, m). "
                        f"Default: {_LADLE_GRASP_OFFSET_DEFAULT}")
    p.add_argument("--hover-dz", type=float, default=_HOVER_DZ_DEFAULT,
                   help=f"Hover height above grasp before descent (m). "
                        f"Default: {_HOVER_DZ_DEFAULT}")
    p.add_argument("--lift-dz", type=float, default=_LIFT_DZ_DEFAULT,
                   help=f"Lift height above grasp after grasping (m). "
                        f"Default: {_LIFT_DZ_DEFAULT}")

    # Mixing heights.
    p.add_argument("--z-mix-ee", type=float, default=_Z_MIX_EE_DEFAULT,
                   help="EE height during mixing (m). Default: bowl_z + 0.283")
    p.add_argument("--z-above-bowl", type=float, default=_Z_ABOVE_BOWL_DEFAULT,
                   help="EE height above bowl for approach/lift (m). "
                        "Default: z_mix_ee + 0.10")

    # Mixing motion.
    p.add_argument("--mix-radius", type=float, default=_MIX_RADIUS_DEFAULT)
    p.add_argument("--mix-omega",  type=float, default=_MIX_OMEGA_DEFAULT)
    p.add_argument("--mix-dur",    type=float, default=_MIX_DUR_DEFAULT)

    # Convergence.
    p.add_argument("--ee-pos-tol",   type=float, default=_EE_POS_TOL_DEFAULT)
    p.add_argument("--gripper-tol",  type=float, default=_GRIPPER_TOL_DEFAULT)

    # Timing.
    p.add_argument("--orient-dur",        type=float, default=_ORIENT_DUR_DEFAULT)
    p.add_argument("--grasp-open-dwell",  type=float, default=_GRASP_OPEN_DWELL_DEFAULT)
    p.add_argument("--grasp-close-dwell", type=float, default=_GRASP_CLOSE_DWELL_DEFAULT)
    p.add_argument("--loop-hz",           type=int,   default=_LOOP_HZ_DEFAULT)

    # Gripper.
    p.add_argument("--gripper-open-width",  type=float, default=None)
    p.add_argument("--gripper-close-width", type=float, default=_GRIPPER_CLOSE_WIDTH_DEFAULT)
    p.add_argument("--gripper-speed",       type=float, default=_GRIPPER_SPEED_DEFAULT)
    p.add_argument("--gripper-force",       type=float, default=_GRIPPER_FORCE_DEFAULT)

    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)

    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────
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

    print(f"Using ladle position: {ladle_pos.tolist()}")
    print(f"Using bowl position:  {bowl_pos.tolist()}")

    return _run_fsm(redis_client, args, ladle_pos, bowl_pos)


if __name__ == "__main__":
    sys.exit(main())
