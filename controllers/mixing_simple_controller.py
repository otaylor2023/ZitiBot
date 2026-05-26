#!/usr/bin/env python3
"""Mixing controller — human-in-the-loop, arm only, no camera.

Sequence:
  1. Move arm to a configurable waiting/receive position (gripper open).
  2. Wait --receive-wait seconds for a human to insert a ladle/spatula between
     the open fingers.
  3. Close the gripper.
  4. SLERP orientation to mixing pose, move above bowl, descend, stir for
     --mix-dur seconds.
  5. Lift out, SLERP back to grasp orientation, return to waiting position.
  6. Wait --release-wait seconds for the human to grab the tool.
  7. Open the gripper. Done.

No base motion. No camera. Runs headless (terminal only).

Usage::
  python controllers/mixing_simple_controller.py --bowl-xyz 0.5 0.0 0.6
  python controllers/mixing_simple_controller.py \\
      --bowl-xyz 0.5 0.0 0.6 --wait-xyz 0.4 0.0 0.5 --mix-dur 15
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as ScipyR, Slerp

_CTRL_DIR = Path(__file__).resolve().parent
if str(_CTRL_DIR) not in sys.path:
    sys.path.insert(0, str(_CTRL_DIR))

import redis  # noqa: F401 (imported for type hints)
import vision_controller as vc

# ── Tuneable defaults (all overridable via CLI) ────────────────────────────────
_WAIT_XYZ_DEFAULT       = (0.4,  0.0,  0.5)   # EE receive pose (world, m)
_BOWL_XYZ_DEFAULT       = (0.5,  0.0,  0.6)   # Bowl centre (world, m)
_Z_ABOVE_OFFSET_DEFAULT = 0.15                 # Extra height above bowl_z
_MIX_RADIUS_DEFAULT     = 0.05                 # Stirring circle radius (m)
_MIX_OMEGA_DEFAULT      = 1.0                  # Stirring rate (rad/s)
_MIX_DUR_DEFAULT        = 10.0                 # Stirring duration (s)
_RECEIVE_WAIT_DEFAULT   = 5.0                  # s to wait for human to insert tool
_RELEASE_WAIT_DEFAULT   = 5.0                  # s to wait for human to grab tool
_ORIENT_DUR_DEFAULT     = 3.0                  # SLERP duration (s)
_POS_TOL_DEFAULT        = 0.025                # EE convergence tolerance (m)
_LOOP_HZ_DEFAULT        = 20                   # Control loop rate

_GRIPPER_CLOSE_WIDTH_DEFAULT = 0.0
_GRIPPER_SPEED_DEFAULT       = 0.1
_GRIPPER_FORCE_DEFAULT       = 50.0
_GRIPPER_SETTLE_S            = 0.5             # Wait after gripper command


# ── Redis keys ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Keys(vc.OpenSaiRedisKeys):
    gripper_desired_width:  str = "opensai::FrankaRobot::gripper::desired_width"
    gripper_desired_speed:  str = "opensai::FrankaRobot::gripper::desired_speed"
    gripper_desired_force:  str = "opensai::FrankaRobot::gripper::desired_force"
    gripper_max_width:      str = "opensai::FrankaRobot::gripper::max_width"
    gripper_current_width:  str = "opensai::FrankaRobot::gripper::current_width"


_KEYS = _Keys()


# ── Orientation helpers ────────────────────────────────────────────────────────
def _link7_orientation_world(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """3×3 rotation for link7 at given RPY. Ported from controller.cpp.

    Applies Rx(π) so link7 Z points −world_Z, then ZYX RPY, then Rz(+π/4) to
    cancel the −π/4 URDF finger-joint offset so the gripper opening aligns with
    world axes predictably.
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
    """Handle points world +Z (up), bowl-end world −Z (down). From controller.cpp."""
    return np.array([[0, 0, 1],
                     [0, -1, 0],
                     [1,  0, 0]], dtype=np.float64)


def _slerp(R_start: np.ndarray, R_end: np.ndarray, alpha: float) -> np.ndarray:
    """SLERP between two 3×3 rotation matrices; alpha clamped to [0, 1]."""
    alpha = min(1.0, max(0.0, alpha))
    key_rots = ScipyR.concatenate([ScipyR.from_matrix(R_start),
                                   ScipyR.from_matrix(R_end)])
    return Slerp([0.0, 1.0], key_rots)([alpha]).as_matrix()[0]


# ── Redis helpers ──────────────────────────────────────────────────────────────
def _ensure_cartesian_controller(redis_client) -> None:
    raw = redis_client.get(_KEYS.active_controller)
    current = raw.decode("utf-8") if raw is not None else ""
    if current != vc.CONTROLLER_TO_USE:
        redis_client.set(_KEYS.active_controller, vc.CONTROLLER_TO_USE)


def _publish_cartesian(redis_client, goal_pos: np.ndarray, goal_ori: np.ndarray) -> None:
    _ensure_cartesian_controller(redis_client)
    redis_client.set(
        _KEYS.cartesian_task_goal_position,
        json.dumps(np.asarray(goal_pos, dtype=np.float64).reshape(3).tolist()),
    )
    redis_client.set(
        _KEYS.cartesian_task_goal_orientation,
        json.dumps(np.asarray(goal_ori, dtype=np.float64).reshape(3, 3).tolist()),
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
    """Return open width: CLI override → Redis max_width → fallback 0.08 m."""
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


def _read_optitrack(redis_client) -> str:
    """Read OptiTrack base position/orientation if available; return a display string."""
    lines = []
    for key, label in (("tidybot01::pos", "optitrack pos"),
                       ("tidybot01::ori", "optitrack ori")):
        raw = redis_client.get(key)
        if raw is not None:
            try:
                lines.append(f"    {label:14s}: {json.loads(raw)}")
            except Exception:
                pass
    return "\n".join(lines) if lines else "    (OptiTrack keys not found in Redis)"


def _log_calib_snapshot(redis_client, label: str,
                        calib_log: dict, ee_pos: np.ndarray | None) -> None:
    """Print and record the current EE position + OptiTrack at a key moment."""
    print(f"  [calib:{label}]")
    if ee_pos is not None:
        vec = [round(float(v), 4) for v in ee_pos]
        print(f"    EE world pos   : {vec}")
        calib_log[label] = vec
    print(_read_optitrack(redis_client))


# ── FSM ───────────────────────────────────────────────────────────────────────
class _State(enum.Enum):
    MOVE_TO_WAIT    = "MOVE_TO_WAIT"      # move arm to receive position, gripper open
    WAIT_RECEIVE    = "WAIT_RECEIVE"      # countdown; human inserts tool
    CLOSE_GRIPPER   = "CLOSE_GRIPPER"    # close gripper, wait for settle
    ORIENT_TO_MIX   = "ORIENT_TO_MIX"   # SLERP R_grasp → R_mix (hold position)
    APPROACH_BOWL   = "APPROACH_BOWL"   # move EE above bowl at R_mix
    LOWER_INTO_BOWL = "LOWER_INTO_BOWL" # descend to mixing height
    MIX             = "MIX"              # circular stirring
    LIFT_FROM_BOWL  = "LIFT_FROM_BOWL"  # ascend to above-bowl height
    ORIENT_TO_GRASP = "ORIENT_TO_GRASP" # SLERP R_mix → R_grasp (hold position)
    RETURN_TO_WAIT  = "RETURN_TO_WAIT"  # move arm back to receive position
    WAIT_GRAB       = "WAIT_GRAB"       # countdown; human grabs tool
    OPEN_GRIPPER    = "OPEN_GRIPPER"    # open gripper, wait for settle
    DONE            = "DONE"


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mixing controller (human-in-the-loop, arm only, no camera)."
    )
    p.add_argument(
        "--wait-xyz", type=float, nargs=3, default=list(_WAIT_XYZ_DEFAULT),
        metavar=("X", "Y", "Z"),
        help=f"EE waiting/receive position in world frame (m). Default: {_WAIT_XYZ_DEFAULT}",
    )
    p.add_argument(
        "--bowl-xyz", type=float, nargs=3, default=list(_BOWL_XYZ_DEFAULT),
        metavar=("X", "Y", "Z"),
        help=f"Bowl centre in world frame (m). Default: {_BOWL_XYZ_DEFAULT}",
    )
    p.add_argument(
        "--z-above", type=float, default=_Z_ABOVE_OFFSET_DEFAULT,
        help=f"Extra height above bowl_z for approach/lift (m). Default: {_Z_ABOVE_OFFSET_DEFAULT}",
    )
    p.add_argument(
        "--z-mix-ee", type=float, default=None,
        help="EE height during mixing (m). Default: bowl_z + 0.22 − 0.06 (ladle-geometry estimate).",
    )
    p.add_argument("--mix-radius",    type=float, default=_MIX_RADIUS_DEFAULT,
                   help=f"Stirring circle radius (m). Default: {_MIX_RADIUS_DEFAULT}")
    p.add_argument("--mix-omega",     type=float, default=_MIX_OMEGA_DEFAULT,
                   help=f"Stirring angular rate (rad/s). Default: {_MIX_OMEGA_DEFAULT}")
    p.add_argument("--mix-dur",       type=float, default=_MIX_DUR_DEFAULT,
                   help=f"Mixing duration (s). Default: {_MIX_DUR_DEFAULT}")
    p.add_argument("--receive-wait",  type=float, default=_RECEIVE_WAIT_DEFAULT,
                   help=f"Seconds to wait for human to insert tool. Default: {_RECEIVE_WAIT_DEFAULT}")
    p.add_argument("--release-wait",  type=float, default=_RELEASE_WAIT_DEFAULT,
                   help=f"Seconds to wait for human to grab tool. Default: {_RELEASE_WAIT_DEFAULT}")
    p.add_argument("--orient-dur",    type=float, default=_ORIENT_DUR_DEFAULT,
                   help=f"SLERP duration for orientation transitions (s). Default: {_ORIENT_DUR_DEFAULT}")
    p.add_argument("--pos-tol",       type=float, default=_POS_TOL_DEFAULT,
                   help=f"EE position convergence tolerance (m). Default: {_POS_TOL_DEFAULT}")
    p.add_argument("--gripper-open-width",  type=float, default=None,
                   help="Gripper open width (m); reads gripper::max_width from Redis if omitted.")
    p.add_argument("--gripper-close-width", type=float, default=_GRIPPER_CLOSE_WIDTH_DEFAULT)
    p.add_argument("--gripper-speed",       type=float, default=_GRIPPER_SPEED_DEFAULT)
    p.add_argument("--gripper-force",       type=float, default=_GRIPPER_FORCE_DEFAULT)
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--loop-hz",    type=int, default=_LOOP_HZ_DEFAULT,
                   help=f"Control loop rate (Hz). Default: {_LOOP_HZ_DEFAULT}")
    p.add_argument(
        "--relative", action="store_true",
        help=(
            "Interpret --wait-xyz and --bowl-xyz as offsets from the arm's current "
            "EE position at startup rather than absolute world-frame coordinates. "
            "Use (0 0 0) for wait-xyz to stay exactly where the arm is. "
            "At each key state arrival the controller prints the absolute EE position "
            "and OptiTrack pose so you can hardcode them for the next run."
        ),
    )
    return p.parse_args()


# ── Calibration summary ───────────────────────────────────────────────────────
def _print_calib_summary_simple(calib_log: dict, args) -> None:
    """Print a ready-to-paste command with hardcoded absolute positions."""
    wp  = calib_log.get("wait_pos")
    ab  = calib_log.get("above_bowl")
    mix = calib_log.get("mix_height")

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Calibration summary — paste these args for absolute mode ║")
    print("╠══════════════════════════════════════════════════════════╣")
    if wp:
        print(f"  --wait-xyz  {wp[0]:.4f} {wp[1]:.4f} {wp[2]:.4f}")
    if ab:
        # bowl centre X/Y from above_bowl; Z from mix_height if available
        bz = mix[2] - 0.283 if mix else ab[2] - args.z_above  # invert z_mix_ee formula
        print(f"  --bowl-xyz  {ab[0]:.4f} {ab[1]:.4f} {bz:.4f}  "
              f"(bowl centre; z estimated from mix height)")
    if mix:
        print(f"  --z-mix-ee  {mix[2]:.4f}  (override auto-computed value)")
    if not (wp or ab or mix):
        print("  (no positions logged — run completed too early)")
    print("╚══════════════════════════════════════════════════════════╝\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    args = _parse_args()

    redis_client = vc._try_redis(args.redis_host, args.redis_port)
    if redis_client is None:
        return 1
    err = vc.validate_config(redis_client)
    if err is not None:
        return err

    # ── Resolve positions (absolute or relative) ──────────────────────────────
    if args.relative:
        init_pose = vc.read_current_ee_world(redis_client)
        if init_pose is None:
            print("--relative: cannot read current EE position from Redis. "
                  "Is the arm controller running?", file=sys.stderr)
            return 1
        ee_origin = init_pose[0]
        wait_pos = ee_origin + np.array(args.wait_xyz, dtype=np.float64)
        bowl_xyz = ee_origin + np.array(args.bowl_xyz, dtype=np.float64)
        print(f"  [relative mode]  EE origin : {ee_origin.round(4).tolist()}")
        print(f"  [relative mode]  wait_pos  : {wait_pos.round(4).tolist()} "
              f" (offset {args.wait_xyz})")
        print(f"  [relative mode]  bowl_xyz  : {bowl_xyz.round(4).tolist()} "
              f" (offset {args.bowl_xyz})")
        print(_read_optitrack(redis_client))
    else:
        wait_pos = np.array(args.wait_xyz, dtype=np.float64)
        bowl_xyz = np.array(args.bowl_xyz, dtype=np.float64)

    # Geometry
    bowl_cx, bowl_cy, bowl_cz = bowl_xyz
    z_above_bowl = bowl_cz + args.z_above
    # z_mix_ee: EE height inside bowl. With ladle vertical, EE control point is
    # ~0.22 m above the ladle bowl-end. Dip bowl-end 0.06 m below the bowl rim.
    # Estimate rim = bowl_cz + 0.123 (from mesh analysis). So:
    # z_mix_ee ≈ (bowl_cz + 0.123) − 0.06 + 0.22 = bowl_cz + 0.283.
    z_mix_ee = args.z_mix_ee if args.z_mix_ee is not None else round(bowl_cz + 0.283, 4)

    # Fixed orientations (ported from controller.cpp)
    R_grasp = _link7_orientation_world(0.0, 0.0, 45.0)
    R_mix   = _ladle_mixing_orientation()

    # Gripper widths
    open_w  = _gripper_open_width(redis_client, args.gripper_open_width)
    close_w = args.gripper_close_width
    spd, frc = args.gripper_speed, args.gripper_force

    print("\n=== Mixing Simple Controller ===")
    print(f"  wait pos     : {wait_pos.tolist()}")
    print(f"  bowl centre  : {bowl_xyz.tolist()}")
    print(f"  z_above_bowl : {z_above_bowl:.4f} m  z_mix_ee : {z_mix_ee:.4f} m")
    print(f"  mix          : r={args.mix_radius} m  ω={args.mix_omega} rad/s  "
          f"dur={args.mix_dur} s")
    print(f"  gripper      : open={open_w:.4f} m  close={close_w:.4f} m")
    print(f"  waits        : receive={args.receive_wait:.0f} s  "
          f"release={args.release_wait:.0f} s")
    print("================================\n")

    state      = _State.MOVE_TO_WAIT
    prev_state = None
    dt         = 1.0 / max(1, args.loop_hz)
    calib_log: dict[str, list[float]] = {}   # filled during run; printed in summary

    wait_t0 = grab_t0 = orient_t0 = mix_t0 = gripper_cmd_t0 = 0.0
    last_mix_print = last_countdown = 0.0

    above_bowl_pos  = np.array([bowl_cx, bowl_cy, z_above_bowl])
    at_bowl_pos     = np.array([bowl_cx, bowl_cy, z_mix_ee])

    try:
        while state != _State.DONE:
            now  = time.perf_counter()
            pose = vc.read_current_ee_world(redis_client)
            cur_pos = pose[0] if pose is not None else None

            # ── State entry banner ────────────────────────────────────────────
            if state != prev_state:
                print(f"[t={now:8.2f}s] >>> {state.value}")
                if state == _State.MOVE_TO_WAIT:
                    print(f"  EE goal  : {wait_pos.tolist()}")
                    print(f"  EE now   : {cur_pos.tolist() if cur_pos is not None else 'unknown'}")

                elif state == _State.WAIT_RECEIVE:
                    # Arm just converged to wait_pos — log absolute position.
                    _log_calib_snapshot(redis_client, "wait_pos", calib_log, cur_pos)
                    print(f"  Gripper open ({open_w:.4f} m). Insert tool in the next "
                          f"{args.receive_wait:.0f} s.")
                    wait_t0 = last_countdown = now

                elif state == _State.CLOSE_GRIPPER:
                    _set_gripper(redis_client, close_w, spd, frc)
                    gripper_cmd_t0 = now
                    print(f"  Closing gripper → {close_w:.4f} m  "
                          f"(settling {_GRIPPER_SETTLE_S:.1f} s)")

                elif state == _State.ORIENT_TO_MIX:
                    orient_t0 = now
                    print(f"  SLERP R_grasp → R_mix over {args.orient_dur:.1f} s  "
                          f"(holding {wait_pos.tolist()})")

                elif state == _State.APPROACH_BOWL:
                    print(f"  EE goal  : {above_bowl_pos.tolist()}")

                elif state == _State.LOWER_INTO_BOWL:
                    # Arm just converged above the bowl — log bowl XY + above-bowl Z.
                    _log_calib_snapshot(redis_client, "above_bowl", calib_log, cur_pos)
                    print(f"  EE goal  : {at_bowl_pos.tolist()}")

                elif state == _State.MIX:
                    # Arm just reached mixing height — log z_mix_ee.
                    _log_calib_snapshot(redis_client, "mix_height", calib_log, cur_pos)
                    mix_t0 = last_mix_print = now
                    print(f"  Stirring for {args.mix_dur:.1f} s  "
                          f"(r={args.mix_radius} m, ω={args.mix_omega} rad/s)")

                elif state == _State.LIFT_FROM_BOWL:
                    print(f"  EE goal  : {above_bowl_pos.tolist()}")

                elif state == _State.ORIENT_TO_GRASP:
                    orient_t0 = now
                    print(f"  SLERP R_mix → R_grasp over {args.orient_dur:.1f} s  "
                          f"(holding {above_bowl_pos.tolist()})")

                elif state == _State.RETURN_TO_WAIT:
                    print(f"  EE goal  : {wait_pos.tolist()}")

                elif state == _State.WAIT_GRAB:
                    print(f"  Gripper closed — grab the tool in the next "
                          f"{args.release_wait:.0f} s.")
                    grab_t0 = last_countdown = now

                elif state == _State.OPEN_GRIPPER:
                    _set_gripper(redis_client, open_w, spd, frc)
                    gripper_cmd_t0 = now
                    print(f"  Opening gripper → {open_w:.4f} m  "
                          f"(settling {_GRIPPER_SETTLE_S:.1f} s)")

                prev_state = state

            # ── FSM tick ──────────────────────────────────────────────────────

            if state == _State.MOVE_TO_WAIT:
                _publish_cartesian(redis_client, wait_pos, R_grasp)
                _set_gripper(redis_client, open_w, spd, frc)
                if cur_pos is not None and np.linalg.norm(cur_pos - wait_pos) < args.pos_tol:
                    state = _State.WAIT_RECEIVE

            elif state == _State.WAIT_RECEIVE:
                _publish_cartesian(redis_client, wait_pos, R_grasp)
                _set_gripper(redis_client, open_w, spd, frc)
                elapsed = now - wait_t0
                if now - last_countdown >= 1.0:
                    remaining = max(0.0, args.receive_wait - elapsed)
                    print(f"  [{elapsed:4.0f}s] Insert tool — {remaining:.0f} s left")
                    last_countdown = now
                if elapsed >= args.receive_wait:
                    state = _State.CLOSE_GRIPPER

            elif state == _State.CLOSE_GRIPPER:
                _publish_cartesian(redis_client, wait_pos, R_grasp)
                _set_gripper(redis_client, close_w, spd, frc)
                if now - gripper_cmd_t0 >= _GRIPPER_SETTLE_S:
                    state = _State.ORIENT_TO_MIX

            elif state == _State.ORIENT_TO_MIX:
                alpha    = (now - orient_t0) / args.orient_dur
                R_interp = _slerp(R_grasp, R_mix, alpha)
                _publish_cartesian(redis_client, wait_pos, R_interp)
                if alpha >= 1.0:
                    state = _State.APPROACH_BOWL

            elif state == _State.APPROACH_BOWL:
                _publish_cartesian(redis_client, above_bowl_pos, R_mix)
                if cur_pos is not None and \
                        np.linalg.norm(cur_pos - above_bowl_pos) < args.pos_tol:
                    state = _State.LOWER_INTO_BOWL

            elif state == _State.LOWER_INTO_BOWL:
                _publish_cartesian(redis_client, at_bowl_pos, R_mix)
                if cur_pos is not None and \
                        np.linalg.norm(cur_pos - at_bowl_pos) < args.pos_tol:
                    state = _State.MIX

            elif state == _State.MIX:
                t = now - mix_t0
                ee_goal = np.array([
                    bowl_cx + args.mix_radius * math.cos(args.mix_omega * t),
                    bowl_cy + args.mix_radius * math.sin(args.mix_omega * t),
                    z_mix_ee,
                ])
                _publish_cartesian(redis_client, ee_goal, R_mix)
                if now - last_mix_print >= 2.0:
                    pos_str = f"  EE: {cur_pos.tolist()}" if cur_pos is not None else ""
                    print(f"  MIX {t:5.1f} / {args.mix_dur:.1f} s{pos_str}")
                    last_mix_print = now
                if t >= args.mix_dur:
                    state = _State.LIFT_FROM_BOWL

            elif state == _State.LIFT_FROM_BOWL:
                _publish_cartesian(redis_client, above_bowl_pos, R_mix)
                if cur_pos is not None and \
                        np.linalg.norm(cur_pos - above_bowl_pos) < args.pos_tol:
                    state = _State.ORIENT_TO_GRASP

            elif state == _State.ORIENT_TO_GRASP:
                # Hold above the bowl while SLERPing orientation back to R_grasp.
                alpha    = (now - orient_t0) / args.orient_dur
                R_interp = _slerp(R_mix, R_grasp, alpha)
                _publish_cartesian(redis_client, above_bowl_pos, R_interp)
                if alpha >= 1.0:
                    state = _State.RETURN_TO_WAIT

            elif state == _State.RETURN_TO_WAIT:
                _publish_cartesian(redis_client, wait_pos, R_grasp)
                if cur_pos is not None and np.linalg.norm(cur_pos - wait_pos) < args.pos_tol:
                    state = _State.WAIT_GRAB

            elif state == _State.WAIT_GRAB:
                _publish_cartesian(redis_client, wait_pos, R_grasp)
                elapsed = now - grab_t0
                if now - last_countdown >= 1.0:
                    remaining = max(0.0, args.release_wait - elapsed)
                    print(f"  [{elapsed:4.0f}s] Grab the tool — {remaining:.0f} s left")
                    last_countdown = now
                if elapsed >= args.release_wait:
                    state = _State.OPEN_GRIPPER

            elif state == _State.OPEN_GRIPPER:
                _publish_cartesian(redis_client, wait_pos, R_grasp)
                _set_gripper(redis_client, open_w, spd, frc)
                if now - gripper_cmd_t0 >= _GRIPPER_SETTLE_S:
                    state = _State.DONE

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nInterrupted — holding last goal.")
        return 0

    print("\n>>> DONE — gripper released, task complete.")
    _print_calib_summary_simple(calib_log, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
