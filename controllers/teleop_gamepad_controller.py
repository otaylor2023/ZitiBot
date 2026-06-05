#!/usr/bin/env python3
"""Gamepad Cartesian teleop for the Franka arm (Logitech F710).

Velocity-integrated EE targets published to OpenSai ``cartesian_controller``
via Redis. Hold RB (right bumper) as dead-man to move; release to freeze and
re-sync the target to the live pose.

Usage::

  ./ZitiBot/launch_zitibot_arm.sh controllers/teleop_gamepad_controller.py
  ./ZitiBot/launch_zitibot_arm.sh controllers/teleop_gamepad_controller.py -- --dry-run

Requires OpenSai with ``zitibot_panda.xml`` and Redis joint/EE feedback.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# Headless pygame (no X11 window).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from pygame.joystick import Joystick

_CONTROLLERS = Path(__file__).resolve().parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from zitibot_core import arm, gripper
from zitibot_core.constants import (
    ARM_HOME_POSITION,
    EGG_CRACKER_STATIONARY_DETECTION_EE_ORIENTATION,
)
from zitibot_core.context import make_context
from zitibot_core.gains import (
    restore_cart_orientation_gains,
    restore_cart_position_gains,
    set_cart_orientation_gains,
    set_cart_position_gains,
    snapshot_cart_orientation_gains,
    snapshot_cart_position_gains,
)

# Logitech F710 / XInput layout (same family as tidybot_base/gamepad_teleop.py).
BTN_A = 0
BTN_B = 1
BTN_X = 2
BTN_Y = 3
BTN_LB = 4
BTN_RB = 5
BTN_BACK = 6
BTN_START = 7

AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_LT = 2
AXIS_RIGHT_X = 3
AXIS_RIGHT_Y = 4
AXIS_RT = 5

# Default workspace box (m) around ARM_HOME — soft clamp on commanded target.
_DEFAULT_HOME = np.asarray(ARM_HOME_POSITION, dtype=np.float64).reshape(3)
_DEFAULT_WS_MARGIN = np.array([0.35, 0.35, 0.25], dtype=np.float64)


def apply_deadzone(arr: np.ndarray, deadzone_size: float = 0.05) -> np.ndarray:
    """Scale stick values outside a circular deadzone to [-1, 1]."""
    arr = np.asarray(arr, dtype=np.float64)
    if deadzone_size <= 0.0 or deadzone_size >= 1.0:
        return arr
    return np.where(
        np.abs(arr) <= deadzone_size,
        0.0,
        np.sign(arr) * (np.abs(arr) - deadzone_size) / (1.0 - deadzone_size),
    )


def _trigger_value(raw: float) -> float:
    """Map trigger axis to [0, 1] (F710 LT/RT rest at +1, pressed toward -1)."""
    return float(np.clip((1.0 - raw) * 0.5, 0.0, 1.0))


def clamp_position(pos: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lo, hi = bounds
    return np.clip(np.asarray(pos, dtype=np.float64).reshape(3), lo, hi)


def wait_for_ee_pose(redis_client, *, timeout_s: float = 30.0, poll_s: float = 0.1):
    """Block until ``read_current_ee_world`` returns a pose."""
    t0 = time.monotonic()
    while True:
        pose = arm.read_current_ee_world(redis_client)
        if pose is not None:
            return pose
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(
                f"EE pose not available on Redis after {timeout_s:.0f}s. "
                "Is OpenSai running with cartesian_controller?"
            )
        time.sleep(poll_s)


def parse_args() -> argparse.Namespace:
    home = _DEFAULT_HOME
    margin = _DEFAULT_WS_MARGIN
    p = argparse.ArgumentParser(description="F710 gamepad cartesian arm teleop.")
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument("--rate-hz", type=float, default=50.0, help="Control loop rate.")
    p.add_argument("--pos-speed", type=float, default=0.04, help="Max X/Y linear speed (m/s).")
    p.add_argument("--z-speed", type=float, default=0.02, help="Max Z linear speed (m/s).")
    p.add_argument("--rot-speed", type=float, default=0.8, help="Max angular speed (rad/s).")
    p.add_argument("--deadzone", type=float, default=0.08, help="Stick deadzone [0, 1).")
    p.add_argument("--gripper-speed", type=float, default=0.08, help="Gripper speed (m/s).")
    p.add_argument("--gripper-force", type=float, default=50.0, help="Gripper open force (N).")
    # B button replicates the egg-cracker grasp: GRASP-mode force-close that
    # clamps on the object at a low force (won't crush). close_width is the
    # GRASP-mode target (0.0 = force-close all the way, like Object.EGG_CRACKER).
    p.add_argument("--close-width", type=float, default=0.0, help="GRASP-mode close width (m).")
    p.add_argument("--close-force", type=float, default=8.0, help="GRASP-mode close force (N).")
    # Orientation PD boost (matches zitibot_tasks/visual_servo.py defaults) so
    # the wrist holds and rotates crisply under gamepad commands.
    p.add_argument("--ori-kp", type=float, default=140.0, help="Cartesian orientation kp.")
    p.add_argument("--ori-kv", type=float, default=28.0, help="Cartesian orientation kv.")
    # Modest position PD boost above the zitibot_panda.xml default (100/20)
    # for slightly crisper translation tracking under teleop.
    p.add_argument("--pos-kp", type=float, default=150.0, help="Cartesian position kp.")
    p.add_argument("--pos-kv", type=float, default=25.0, help="Cartesian position kv.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stick/button values only; do not publish to Redis.",
    )
    p.add_argument("--ee-wait-s", type=float, default=30.0, help="Timeout waiting for EE pose.")
    for axis, idx in ("x", 0), ("y", 1), ("z", 2):
        p.add_argument(
            f"--{axis}-min",
            type=float,
            default=float(home[idx] - margin[idx]),
        )
        p.add_argument(
            f"--{axis}-max",
            type=float,
            default=float(home[idx] + margin[idx]),
        )
    return p.parse_args()


def workspace_bounds(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([args.x_min, args.y_min, args.z_min], dtype=np.float64)
    hi = np.array([args.x_max, args.y_max, args.z_max], dtype=np.float64)
    if np.any(lo >= hi):
        raise ValueError(f"Invalid workspace bounds: lo={lo.tolist()} hi={hi.tolist()}")
    return lo, hi


def read_cmd_vel(joy: Joystick, deadzone: float) -> tuple[np.ndarray, np.ndarray]:
    """Return linear (3,) and angular (3,) velocity commands in world frame."""
    # Left stick: world X/Y (match base teleop sign convention).
    lin_x = -joy.get_axis(AXIS_LEFT_Y)
    lin_y = -joy.get_axis(AXIS_LEFT_X)
    # Z on the back triggers (RT = up/+Z, LT = down/-Z).
    lt = _trigger_value(joy.get_axis(AXIS_LT))
    rt = _trigger_value(joy.get_axis(AXIS_RT))
    lin_z = rt - lt
    # Right stick: yaw (left/right) and pitch (up/down).
    yaw = -joy.get_axis(AXIS_RIGHT_X)
    pitch = joy.get_axis(AXIS_RIGHT_Y)
    # Roll on the D-pad left/right arrows (right = +roll).
    if joy.get_numhats() > 0:
        roll = float(joy.get_hat(0)[0])
    else:
        roll = 0.0

    lin = apply_deadzone(np.array([lin_x, lin_y, lin_z]), deadzone)
    ang = apply_deadzone(np.array([roll, pitch, yaw]), deadzone)
    return lin, ang


class GamepadArmTeleop:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bounds = workspace_bounds(args)
        self.joy: Joystick | None = None
        self.ctx = None
        self.target_pos: np.ndarray | None = None
        self.target_R: np.ndarray | None = None
        self._orientation_gain_snapshot = None
        self._position_gain_snapshot = None
        self._enabled_last = False
        self._gripper_open_sent = False
        self._gripper_close_sent = False
        self._tool_down_sent = False
        self._quit = False

    def init_gamepad(self) -> None:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() < 1:
            raise RuntimeError(
                "No gamepad detected. Plug in the Logitech F710 (USB) and retry."
            )
        self.joy = Joystick(0)
        self.joy.init()
        print(
            f"Gamepad: {self.joy.get_name()} "
            f"({self.joy.get_numaxes()} axes, {self.joy.get_numbuttons()} buttons)"
        )

    def init_redis(self) -> None:
        if self.args.dry_run:
            print("Dry-run: skipping Redis / arm connection.")
            self.target_pos = _DEFAULT_HOME.copy()
            self.target_R = np.eye(3, dtype=np.float64)
            return
        self.ctx = make_context(
            argparse.Namespace(
                redis_host=self.args.redis_host,
                redis_port=self.args.redis_port,
            ),
            print_startup=True,
        )
        pos, ori = wait_for_ee_pose(self.ctx.redis, timeout_s=self.args.ee_wait_s)
        self.target_pos = pos.copy()
        self.target_R = ori.copy()
        arm.publish_goal_cartesian(self.ctx.redis, self.target_pos, self.target_R)
        print(f"Seeded target position: {self.target_pos.tolist()}")
        self._orientation_gain_snapshot = snapshot_cart_orientation_gains(self.ctx.redis)
        set_cart_orientation_gains(
            self.ctx.redis, kp=self.args.ori_kp, kv=self.args.ori_kv
        )
        print(f"Orientation gains set: kp={self.args.ori_kp} kv={self.args.ori_kv}")
        self._position_gain_snapshot = snapshot_cart_position_gains(
            self.ctx.redis, ki=False
        )
        set_cart_position_gains(self.ctx.redis, kp=self.args.pos_kp, kv=self.args.pos_kv)
        print(f"Position gains set: kp={self.args.pos_kp} kv={self.args.pos_kv}")

    def _sync_target_to_live(self) -> None:
        if self.ctx is None or self.target_pos is None:
            return
        pose = arm.read_current_ee_world(self.ctx.redis)
        if pose is None:
            return
        self.target_pos, self.target_R = pose[0].copy(), pose[1].copy()

    def _handle_gripper_edges(self) -> None:
        if self.ctx is None or self.joy is None:
            return
        open_btn = self.joy.get_button(BTN_X)
        close_btn = self.joy.get_button(BTN_B)
        if open_btn and not self._gripper_open_sent:
            w = gripper.open_gripper(
                self.ctx.redis,
                None,
                speed=self.args.gripper_speed,
                force=self.args.gripper_force,
            )
            print(f"Gripper open (width={w:.3f} m)")
            self._gripper_open_sent = True
        elif not open_btn:
            self._gripper_open_sent = False

        if close_btn and not self._gripper_close_sent:
            gripper.grasp(
                self.ctx.redis,
                self.args.close_width,
                speed=self.args.gripper_speed,
                force=self.args.close_force,
            )
            print(f"Gripper grasp (force-close at {self.args.close_force:.1f} N)")
            self._gripper_close_sent = True
        elif not close_btn:
            self._gripper_close_sent = False

        # Y: snap orientation to tool-straight-down (egg-cracker stationary pose).
        tool_down_btn = self.joy.get_button(BTN_Y)
        if tool_down_btn and not self._tool_down_sent:
            if self.target_pos is not None:
                self.target_R = EGG_CRACKER_STATIONARY_DETECTION_EE_ORIENTATION.copy()
                arm.publish_goal_cartesian(
                    self.ctx.redis, self.target_pos, self.target_R
                )
                print("Orientation snap: tool-straight-down (egg-cracker pose)")
            self._tool_down_sent = True
        elif not tool_down_btn:
            self._tool_down_sent = False

    def run(self) -> None:
        self.init_gamepad()
        self.init_redis()

        dt_nom = 1.0 / max(self.args.rate_hz, 1.0)
        print("")
        print("Controls (Logitech F710, XInput):")
        print("  Hold LB       — enable motion (dead-man)")
        print("  Left stick    — X/Y translation")
        print("  LT / RT       — Z down / up")
        print("  Right stick   — yaw (L/R) / pitch (U/D)")
        print("  D-pad L/R     — roll")
        print("  X             — open gripper")
        print(f"  B             — grasp (force-close at {self.args.close_force:.0f} N)")
        print("  Y             — snap orientation to tool-straight-down")
        print("  Back          — quit")
        print("")
        if self.args.dry_run:
            print("DRY-RUN: printing axes/buttons (no Redis commands).")
        else:
            print(
                f"Teleop active at {self.args.rate_hz:.0f} Hz "
                f"(pos={self.args.pos_speed} m/s, rot={self.args.rot_speed} rad/s)."
            )
        print("")

        while not self._quit:
            t0 = time.monotonic()
            pygame.event.pump()
            assert self.joy is not None

            if self.joy.get_button(BTN_BACK):
                print("Back pressed — exiting.")
                break

            lin_cmd, ang_cmd = read_cmd_vel(self.joy, self.args.deadzone)
            enabled = bool(self.joy.get_button(BTN_LB))

            if self.args.dry_run:
                if enabled or np.any(lin_cmd) or np.any(ang_cmd):
                    print(
                        f"LB={int(enabled)}  "
                        f"lin={lin_cmd.round(2).tolist()}  "
                        f"ang={ang_cmd.round(2).tolist()}  "
                        f"axes={[round(self.joy.get_axis(i), 2) for i in range(self.joy.get_numaxes())]}  "
                        f"btns={[self.joy.get_button(i) for i in range(min(8, self.joy.get_numbuttons()))]}"
                    )
            else:
                assert self.ctx is not None
                assert self.target_pos is not None
                assert self.target_R is not None

                self._handle_gripper_edges()

                if enabled:
                    if not self._enabled_last:
                        print("Teleop ENABLED (LB held)")
                    dt = dt_nom
                    v = lin_cmd * self.args.pos_speed
                    v[2] = lin_cmd[2] * self.args.z_speed
                    w = ang_cmd * self.args.rot_speed
                    self.target_pos = self.target_pos + v * dt
                    self.target_pos = clamp_position(self.target_pos, self.bounds)
                    if float(np.linalg.norm(w)) > 1e-9:
                        dR = R.from_rotvec(w * dt).as_matrix()
                        self.target_R = dR @ self.target_R
                    arm.publish_goal_cartesian(
                        self.ctx.redis, self.target_pos, self.target_R
                    )
                else:
                    if self._enabled_last:
                        print("Teleop disabled — synced target to live EE pose")
                    self._sync_target_to_live()

            self._enabled_last = enabled

            elapsed = time.monotonic() - t0
            sleep_s = max(0.0, dt_nom - elapsed)
            time.sleep(sleep_s)

    def stop(self) -> None:
        self._quit = True

    def restore_gains(self) -> None:
        if self.ctx is None:
            return
        if self._orientation_gain_snapshot is not None:
            restore_cart_orientation_gains(
                self.ctx.redis, self._orientation_gain_snapshot, label="gamepad teleop"
            )
            self._orientation_gain_snapshot = None
        if self._position_gain_snapshot is not None:
            restore_cart_position_gains(
                self.ctx.redis, self._position_gain_snapshot, label="gamepad teleop"
            )
            self._position_gain_snapshot = None


def _sigterm_handler(signum, frame):
    raise KeyboardInterrupt


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    teleop = GamepadArmTeleop(args)
    try:
        teleop.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        teleop.restore_gains()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
