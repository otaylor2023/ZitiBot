#!/usr/bin/env python3
"""Mixing controller — fully autonomous pickup-mix-putdown pipeline.

Mirrors the sim mixing-task controller.cpp FSM, adapted for the real Franka arm:
  - Arm control  : OpenSAI cartesian controller (Redis goal_position/orientation)
  - Base motion  : TidyBot hb1::desired_pose / hb1::current_pose (Redis)
  - Gripper      : Franka gripper Redis driver (desired_width/speed/force)
  - Object poses : provided via --ladle-xyz / --bowl-xyz, or detected interactively
                   with Gemini + RealSense if either is omitted.

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
  # Positions provided explicitly (no camera needed):
  python controllers/mixing_full_controller.py \\
      --ladle-xyz 0.75 0.68 0.508 --bowl-xyz 0.17 0.62 0.50

  # Detect ladle with camera; bowl position provided:
  python controllers/mixing_full_controller.py --bowl-xyz 0.17 0.62 0.50

  # Skip base navigation (test arm-only, e.g. TidyBot not running):
  python controllers/mixing_full_controller.py \\
      --ladle-xyz 0.75 0.68 0.508 --bowl-xyz 0.17 0.62 0.50 --skip-base
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

import vision_controller as vc

# ── Defaults (all overridable via CLI) ─────────────────────────────────────────
# Base targets [x, y, yaw_rad] in hb1 odometry frame — tune to your lab setup.
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

# Camera detection.
_CAM_WIDTH_DEFAULT  = 640
_CAM_HEIGHT_DEFAULT = 480
_CAM_FPS_DEFAULT    = 30
_CAM_WARMUP_DEFAULT = 30
_CAM_TIMEOUT_DEFAULT = 10000
_DEPTH_PATCH_RADIUS  = 2


# ── Redis keys ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _Keys(vc.OpenSaiRedisKeys):
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
def _ensure_cartesian(redis_client) -> None:
    raw = redis_client.get(_KEYS.active_controller)
    if raw is None or raw.decode("utf-8") != vc.CONTROLLER_TO_USE:
        redis_client.set(_KEYS.active_controller, vc.CONTROLLER_TO_USE)


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


# ── Calibration helpers ────────────────────────────────────────────────────────
def _read_optitrack(redis_client) -> str:
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
                        calib_log: dict,
                        ee_pos: np.ndarray | None,
                        base: np.ndarray | None = None) -> None:
    """Print and record the current EE position, base pose, and OptiTrack at a key moment."""
    print(f"  [calib:{label}]")
    if ee_pos is not None:
        vec = [round(float(v), 4) for v in ee_pos]
        print(f"    EE world pos   : {vec}")
        calib_log[f"{label}_ee"] = vec
    if base is not None:
        bvec = [round(float(v), 4) for v in base]
        print(f"    base pose      : {bvec}  (x, y, yaw_rad)")
        calib_log[f"{label}_base"] = bvec
    print(_read_optitrack(redis_client))


def _print_calib_summary_full(
    calib_log: dict,
    ladle_pos: np.ndarray,
    bowl_pos: np.ndarray,
    z_mix_ee: float,
    z_above_bowl: float,
    base_ladle_goal: np.ndarray,
    base_bowl_goal: np.ndarray,
) -> None:
    """Print a calibration box with every value needed to run without --relative."""
    def fv(v) -> str:
        return " ".join(f"{x:.4f}" for x in v)

    W = 62  # inner width
    def row(label: str, val: str) -> str:
        content = f"  {label:<22s}{val}"
        return f"║{content:<{W}}║"

    sep   = f"╠{'═' * W}╣"
    top   = f"╔{'═' * W}╗"
    bot   = f"╚{'═' * W}╝"
    title = f"║{'CALIBRATION SUMMARY — mixing_full_controller':^{W}}║"

    lines = [top, title, sep]
    lines.append(row("--ladle-xyz",    fv(ladle_pos)))
    lines.append(row("--bowl-xyz",     fv(bowl_pos)))
    lines.append(row("--z-mix-ee",     f"{z_mix_ee:.4f}"))
    lines.append(row("--z-above-bowl", f"{z_above_bowl:.4f}"))
    lines.append(row("--base-ladle",   fv(base_ladle_goal)))
    lines.append(row("--base-bowl",    fv(base_bowl_goal)))

    if calib_log:
        lines.append(sep)
        for label, val in calib_log.items():
            lines.append(row(label, fv(val) if isinstance(val, (list, np.ndarray)) else str(val)))

    lines.append(bot)
    print("\n" + "\n".join(lines) + "\n")


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


# ── Gemini detection phase ─────────────────────────────────────────────────────
def _detect_objects(
    redis_client,
    args,
    need_ladle: bool,
    need_bowl: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Interactive camera loop to latch ladle and/or bowl world positions.

    Returns (ladle_xyz, bowl_xyz) — entries that were already provided via CLI
    are passed through unchanged; only the ones flagged as needed are detected.

    Keys:
      SPACE — query Gemini for the *next* needed object (ladle first, then bowl)
      's'   — save current overlay image
      'g' / ENTER — accept latched positions and proceed
      'q' / ESC   — quit
    """
    try:
        import cv2
        from vision import gemini_pointing as gp
        from vision import realsense_rgbd as rs_cam
    except ImportError as e:
        print(f"Camera detection unavailable ({e}). "
              "Provide --ladle-xyz and --bowl-xyz explicitly.", file=sys.stderr)
        return None, None

    client = gp.make_genai_client(gp.resolve_api_key())

    T_ee_cam = (gp.load_T_ee_cam(args.ee_from_cam_json)
                if args.ee_from_cam_json is not None
                else gp.PLACEHOLDER_T_EE_CAM.copy())

    pipeline = None
    try:
        pipeline, align, depth_scale, color_intrinsics = rs_cam.start_realsense(
            args.cam_width, args.cam_height, args.cam_fps,
            args.cam_warmup, args.cam_timeout,
        )
    except Exception as e:
        print(f"RealSense startup failed: {e}", file=sys.stderr)
        return None, None

    win = "Mixing — detect objects (SPACE=detect, g/Enter=go, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 480)

    ladle_xyz: np.ndarray | None = None
    bowl_xyz:  np.ndarray | None = None
    latched_overlay: np.ndarray | None = None
    miss_counter = [0]

    # Decide detection order.
    pending = []
    if need_ladle:
        pending.append("ladle")
    if need_bowl:
        pending.append("bowl")
    detecting = pending[0] if pending else None

    print(f"\nDetection phase. Press SPACE to detect '{detecting}'. "
          "g/Enter to accept. q to quit.")

    try:
        while True:
            triple = rs_cam.next_rgbd_frame(
                pipeline, align, depth_scale, args.cam_timeout,
                miss_counter, max_misses=10,
            )
            if triple is None:
                continue
            color_bgr, depth_m, depth_vis = triple
            h, w = color_bgr.shape[:2]

            panel = (latched_overlay.copy()
                     if latched_overlay is not None and latched_overlay.shape[:2] == (h, w)
                     else np.full((h, w, 3), (48, 48, 52), dtype=np.uint8))

            # Status overlay.
            status = (f"Detect: {detecting}  |  ladle={'OK' if ladle_xyz is not None else '?'}"
                      f"  bowl={'OK' if bowl_xyz is not None else '?'}"
                      f"  |  g/Enter=go  q=quit")
            cv2.putText(color_bgr, status, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, np.hstack((color_bgr, depth_vis, panel)))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print("Detection cancelled.")
                return None, None

            if key in (ord("g"), 10, 13):
                # Allow proceeding if all needed positions are latched.
                still_needed = (need_ladle and ladle_xyz is None) or \
                               (need_bowl  and bowl_xyz  is None)
                if still_needed:
                    print(f"Still need: "
                          f"{'ladle ' if need_ladle and ladle_xyz is None else ''}"
                          f"{'bowl'  if need_bowl  and bowl_xyz  is None else ''}. "
                          "Press SPACE to detect.")
                else:
                    break

            if key == ord("s") and latched_overlay is not None:
                vc._save_overlay(latched_overlay)

            if key == ord(" ") and detecting is not None:
                overlay, first_ee = gp.query_color_depth_overlay(
                    client, args.model, gp.build_prompt(detecting, None),
                    args.temperature, color_bgr, depth_m,
                    color_intrinsics, T_ee_cam, _DEPTH_PATCH_RADIUS,
                )
                if overlay is not None and first_ee is not None:
                    pose = vc.read_current_ee_world(redis_client)
                    if pose is not None:
                        cur_pos, cur_ori = pose
                        p_ee = np.asarray(first_ee, dtype=np.float64).reshape(3)
                        world_xyz = (cur_ori @ p_ee) + cur_pos
                        latched_overlay = overlay
                        if detecting == "ladle":
                            ladle_xyz = world_xyz
                            print(f"  Ladle latched: {world_xyz.tolist()}")
                        else:
                            bowl_xyz = world_xyz
                            print(f"  Bowl latched:  {world_xyz.tolist()}")
                        # Advance to next pending object.
                        pending_remaining = [p for p in pending
                                             if (p == "ladle" and ladle_xyz is None) or
                                                (p == "bowl"  and bowl_xyz  is None)]
                        detecting = pending_remaining[0] if pending_remaining else None
                        if detecting:
                            print(f"  Now detect '{detecting}' — press SPACE.")
                    else:
                        print("  Redis EE pose unavailable; cannot transform to world frame.")
                else:
                    print("  Gemini found no point at pick pixel.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return ladle_xyz, bowl_xyz


# ── Autonomous FSM ─────────────────────────────────────────────────────────────
def _run_fsm(redis_client, args,
             ladle_pos: np.ndarray,
             bowl_pos: np.ndarray) -> int:
    # Derived geometry.
    ladle_grasp_offset = np.array(args.ladle_offset, dtype=np.float64)
    bowl_cx, bowl_cy, bowl_cz = bowl_pos

    z_mix_ee     = (args.z_mix_ee     if args.z_mix_ee     is not None
                    else round(bowl_cz + 0.283, 4))
    z_above_bowl = (args.z_above_bowl if args.z_above_bowl is not None
                    else round(z_mix_ee + 0.10, 4))

    base_ladle_goal = np.array(args.base_ladle, dtype=np.float64)
    base_bowl_goal  = np.array(args.base_bowl,  dtype=np.float64)

    # ── Relative-base mode: resolve base offsets to absolute odometry coords ──
    if args.relative_base:
        base_now = _read_base_pose(redis_client)
        if base_now is None:
            print("--relative-base: hb1::current_pose not in Redis. Aborting.")
            return 1
        print(f"\n[relative-base mode] base origin: {base_now.tolist()}")
        base_ladle_goal = base_now + base_ladle_goal
        base_ladle_goal[2] = _wrap_angle(base_ladle_goal[2])
        base_bowl_goal  = base_now + base_bowl_goal
        base_bowl_goal[2] = _wrap_angle(base_bowl_goal[2])
        print(f"  --base-ladle resolved → {base_ladle_goal.tolist()}")
        print(f"  --base-bowl  resolved → {base_bowl_goal.tolist()}")

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

    print("\n=== Mixing Full Controller ===")
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

    calib_log: dict[str, list[float]] = {}

    try:
        while state != _State.DONE:
            now  = time.perf_counter()
            pose = vc.read_current_ee_world(redis_client)
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
                    _log_calib_snapshot(redis_client, "near_ladle", calib_log, cur_pos, base)

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
                    _log_calib_snapshot(redis_client, "near_bowl", calib_log, cur_pos, base)

                elif state == _State.EE_LOWER_INTO_BOWL:
                    print(f"  mix height  : {at_bowl_pos.tolist()}")

                elif state == _State.EE_MIX:
                    mix_t0 = last_mix_print = now
                    print(f"  Stirring for {args.mix_dur:.1f} s  "
                          f"(r={args.mix_radius} m, ω={args.mix_omega} rad/s)")
                    _log_calib_snapshot(redis_client, "mix_height", calib_log, cur_pos, base)

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
                    _log_calib_snapshot(redis_client, "return_ladle", calib_log, cur_pos, base)

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
        _print_calib_summary_full(
            calib_log, ladle_pos, bowl_pos,
            z_mix_ee, z_above_bowl,
            base_ladle_goal, base_bowl_goal,
        )
        return 0

    print("\n>>> DONE — ladle released, task complete.")
    _print_calib_summary_full(
        calib_log, ladle_pos, bowl_pos,
        z_mix_ee, z_above_bowl,
        base_ladle_goal, base_bowl_goal,
    )
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mixing controller — full autonomous pipeline."
    )

    # Object positions.
    p.add_argument("--ladle-xyz", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Ladle rest position in world frame (m). "
                        "If omitted, use Gemini+RealSense detection.")
    p.add_argument("--bowl-xyz", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Bowl centre in world frame (m). "
                        "If omitted, use Gemini+RealSense detection.")

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

    # Camera detection.
    p.add_argument("--ee-from-cam-json", type=Path, default=None,
                   help="4×4 T_ee_cam JSON. Uses built-in placeholder if omitted.")
    p.add_argument("--model",       default=None)   # filled from gp.DEFAULT_MODEL if needed
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--cam-width",   type=int, default=_CAM_WIDTH_DEFAULT)
    p.add_argument("--cam-height",  type=int, default=_CAM_HEIGHT_DEFAULT)
    p.add_argument("--cam-fps",     type=int, default=_CAM_FPS_DEFAULT)
    p.add_argument("--cam-warmup",  type=int, default=_CAM_WARMUP_DEFAULT)
    p.add_argument("--cam-timeout", type=int, default=_CAM_TIMEOUT_DEFAULT)

    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)

    p.add_argument(
        "--relative", action="store_true",
        help=(
            "Interpret --ladle-xyz and --bowl-xyz as offsets from the arm's current "
            "EE position at startup. Use (0 0 0) to reference the ladle from wherever "
            "the arm currently is. Logs absolute coords at each key state for hardcoding."
        ),
    )
    p.add_argument(
        "--relative-base", action="store_true",
        help=(
            "Interpret --base-ladle and --base-bowl as offsets from the current "
            "hb1 odometry pose at startup (dx, dy, dyaw). Combine with --relative to "
            "test the full pipeline without knowing any absolute coordinates."
        ),
    )

    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    args = _parse_args()

    # Resolve camera model default lazily so the import only runs if needed.
    if args.model is None:
        try:
            from vision import gemini_pointing as _gp
            args.model = _gp.DEFAULT_MODEL
        except ImportError:
            args.model = "gemini-1.5-pro"

    redis_client = vc._try_redis(args.redis_host, args.redis_port)
    if redis_client is None:
        return 1
    err = vc.validate_config(redis_client)
    if err is not None:
        return err

    ladle_xyz = (np.array(args.ladle_xyz, dtype=np.float64)
                 if args.ladle_xyz is not None else None)
    bowl_xyz  = (np.array(args.bowl_xyz,  dtype=np.float64)
                 if args.bowl_xyz  is not None else None)

    # ── Relative mode: resolve EE offsets to absolute world coordinates ────────
    if args.relative and (ladle_xyz is not None or bowl_xyz is not None):
        pose = vc.read_current_ee_world(redis_client)
        if pose is None:
            print("--relative: cannot read current EE pose from Redis. Aborting.",
                  file=sys.stderr)
            return 1
        ee_origin = pose[0]
        print(f"\n[relative mode] EE origin at startup: {ee_origin.tolist()}")
        if ladle_xyz is not None:
            ladle_xyz = ee_origin + ladle_xyz
            print(f"  --ladle-xyz resolved → {ladle_xyz.tolist()}")
        if bowl_xyz is not None:
            bowl_xyz = ee_origin + bowl_xyz
            print(f"  --bowl-xyz  resolved → {bowl_xyz.tolist()}")

    need_ladle = ladle_xyz is None
    need_bowl  = bowl_xyz  is None

    if need_ladle or need_bowl:
        detected_ladle, detected_bowl = _detect_objects(
            redis_client, args, need_ladle, need_bowl)
        if need_ladle:
            if detected_ladle is None:
                print("Ladle position not obtained. Aborting.", file=sys.stderr)
                return 1
            ladle_xyz = detected_ladle
        if need_bowl:
            if detected_bowl is None:
                print("Bowl position not obtained. Aborting.", file=sys.stderr)
                return 1
            bowl_xyz = detected_bowl

    return _run_fsm(redis_client, args, ladle_xyz, bowl_xyz)


if __name__ == "__main__":
    sys.exit(main())
