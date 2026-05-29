#!/usr/bin/env python3
"""Arm-only egg-cracker sequence on OpenSai Franka (Redis only).

Pick up the cracker tool with a light force grip, lift over the bowl, squeeze
to crack the egg, then move aside, lower, and release the tool.

Every transition is **manually gated by ENTER**. Startup prints the plan with
**no motion** until the first ENTER.

Sequence:

- ENTER (1) — move above cracker + open gripper.
- ENTER (2) — descend to cracker grasp pose.
- ENTER (3) — pregrasp width, then grasp at ``--gripper-lift-force``.
- ENTER (4) — lift to approach height.
- ENTER (5) — squeeze at ``--gripper-crack-force`` (arm holds pose).
- ENTER (6) — translate to drop pose at approach height.
- ENTER (7) — descend to drop pose.
- ENTER (8) — open gripper (release cracker).

Requires OpenSai cartesian controller and Franka gripper Redis driver.

Usage::

  python ZitiBot/controllers/egg_crack_controller.py
  ./launch_zitibot_full.sh controllers/egg_crack_controller.py
"""

from __future__ import annotations

import argparse
import enum
import sys
import time
from dataclasses import dataclass

import numpy as np
import redis

from grasp_and_pour_controller import (
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    DEFAULT_GRIPPER_PREGRASP_WIDTH,
    DEFAULT_GRIPPER_GRASP_SETTLE_S,
    DEFAULT_GRIPPER_SPEED,
    GRASP_ORIENTATION,
    GRIPPER_MODE_GRASP,
    GRIPPER_MODE_MOVE,
    GRIPPER_MODE_OPEN_MAX,
    _STDIN_EOF,
    _publish_cartesian,
    _stdin_line_ready,
    read_current_ee_world,
    read_gripper_current_width,
    resolve_gripper_open_width,
    set_gripper_width,
    validate_config,
)
from grasp_and_pour_controller import _try_redis

DEFAULT_LIFT_FORCE_N = 8.0
DEFAULT_CRACK_SETTLE_S = 1.5
# Briefly retract the fingers this far (mode=m) before squeezing, to break
# libfranka's "already grasped" early-return in gripper.grasp(). With the
# driver hardcoding target=0.0 and eps_inner=1.0, a second mode=g call after
# the lift grasp returns immediately without applying new force; opening a
# few mm forces the next grasp() to actually close again at crack force.
DEFAULT_CRACK_UNLATCH_M = 0.005
DEFAULT_CRACK_UNLATCH_SETTLE_S = 0.4
# Default cracker grasp pose (OpenSai/Franka world frame, meters). Tuned so
# the EE sits over the egg-cracker tool with the same fixed tool-down
# orientation used by the other controllers. Override via --pick-x/y/z.
DEFAULT_PICK_POSITION = np.array([0.5523, -0.0811, 0.4625], dtype=np.float64)

# Drop pose offset from pick (m). Default is a small +Y shift so the cracker
# is released a few cm to the side of where it was picked up. Override via
# --drop-x/--drop-y/--drop-z (absolute) if you want a different drop site.
DEFAULT_DROP_OFFSET_X_M = 0.0
DEFAULT_DROP_OFFSET_Y_M = 0.05
DEFAULT_DROP_OFFSET_Z_M = 0.0

TICK_DT_S = 0.05
MIN_GRIPPER_FORCE_N = 0.1
WARN_LIFT_FORCE_N = 25.0
HARDWARE_MAX_FORCE_N = 70.0
DEFAULT_CRACK_FORCE_N = HARDWARE_MAX_FORCE_N


@dataclass
class EggCrackParams:
    """Arm + gripper parameters for the egg-cracker sequence."""

    approach_dz_m: float
    pick_pos: np.ndarray
    drop_pos: np.ndarray
    grasp_ori: np.ndarray
    gripper_open_width: float | None
    gripper_pregrasp_width: float
    gripper_speed: float
    lift_force_n: float
    crack_force_n: float
    gripper_pregrasp_settle_s: float
    gripper_grasp_settle_s: float
    crack_settle_s: float
    crack_unlatch_m: float
    crack_unlatch_settle_s: float


class Phase(enum.Enum):
    AWAIT_ABOVE_PICK = "AWAIT_ABOVE_PICK"
    ABOVE_PICK = "ABOVE_PICK"
    AT_PICK = "AT_PICK"
    GRASPED = "GRASPED"
    LIFTED = "LIFTED"
    SQUEEZED = "SQUEEZED"
    ABOVE_DROP = "ABOVE_DROP"
    AT_DROP = "AT_DROP"
    DONE = "DONE"


def _phase_hint(phase: Phase) -> str:
    if phase == Phase.AWAIT_ABOVE_PICK:
        return "Next: ENTER = move above cracker + open gripper"
    if phase == Phase.ABOVE_PICK:
        return "Next: ENTER = descend to cracker grasp pose"
    if phase == Phase.AT_PICK:
        return "Next: ENTER = pregrasp + grasp (lift force)"
    if phase == Phase.GRASPED:
        return "Next: ENTER = lift to approach height"
    if phase == Phase.LIFTED:
        return "Next: ENTER = squeeze (crack force)"
    if phase == Phase.SQUEEZED:
        return "Next: ENTER = move to drop pose (approach height)"
    if phase == Phase.ABOVE_DROP:
        return "Next: ENTER = descend to drop pose"
    if phase == Phase.AT_DROP:
        return "Next: ENTER = open gripper (release cracker)"
    if phase == Phase.DONE:
        return "Done — q to quit"
    return ""


def _above_pose(pick_pos: np.ndarray, params: EggCrackParams) -> np.ndarray:
    return pick_pos + np.array([0.0, 0.0, params.approach_dz_m], dtype=np.float64)


def _above_drop(drop_pos: np.ndarray, params: EggCrackParams) -> np.ndarray:
    return drop_pos + np.array([0.0, 0.0, params.approach_dz_m], dtype=np.float64)


def _log_gripper_width(redis_client, label: str) -> None:
    w = read_gripper_current_width(redis_client)
    if w is not None:
        print(f"  {label}: gripper current_width={w:.4f} m")
    else:
        print(f"  {label}: gripper current_width unavailable")


def _do_move_above_pick(
    redis_client,
    pick_pos: np.ndarray,
    grasp_ori: np.ndarray,
    params: EggCrackParams,
) -> np.ndarray:
    above = _above_pose(pick_pos, params)
    _publish_cartesian(redis_client, above, grasp_ori)
    open_w = resolve_gripper_open_width(redis_client, params.gripper_open_width)
    set_gripper_width(
        redis_client,
        open_w,
        speed=params.gripper_speed,
        force=params.lift_force_n,
        mode=GRIPPER_MODE_MOVE,
    )
    print(
        f"[1] Above cracker: pos={above.tolist()}, "
        f"gripper open width={open_w:.4f} m (mode=m)"
    )
    return above


def _do_descend_to_pick(
    redis_client,
    pick_pos: np.ndarray,
    grasp_ori: np.ndarray,
) -> None:
    _publish_cartesian(redis_client, pick_pos, grasp_ori)
    print(f"[2] Descend to cracker: pos={pick_pos.tolist()}")


def _do_lift_grasp(redis_client, params: EggCrackParams) -> None:
    """Pregrasp (move mode) then force grasp at lift_force."""
    pre_w = float(params.gripper_pregrasp_width)
    set_gripper_width(
        redis_client,
        pre_w,
        speed=params.gripper_speed,
        force=params.lift_force_n,
        mode=GRIPPER_MODE_MOVE,
    )
    print(
        f"[3a] Pregrasp: width={pre_w:.4f} m (mode=m), "
        f"settle {params.gripper_pregrasp_settle_s:.1f} s"
    )
    time.sleep(params.gripper_pregrasp_settle_s)

    set_gripper_width(
        redis_client,
        0.0,
        speed=params.gripper_speed,
        force=params.lift_force_n,
        mode=GRIPPER_MODE_GRASP,
    )
    print(
        f"[3b] Lift grasp: force={params.lift_force_n:.1f} N (mode=g), "
        f"settle {params.gripper_grasp_settle_s:.1f} s"
    )
    time.sleep(params.gripper_grasp_settle_s)
    _log_gripper_width(redis_client, "after lift grasp")


def _do_lift(
    redis_client,
    pick_pos: np.ndarray,
    grasp_ori: np.ndarray,
    params: EggCrackParams,
) -> np.ndarray:
    lift = _above_pose(pick_pos, params)
    _publish_cartesian(redis_client, lift, grasp_ori)
    print(f"[4] Lift: pos={lift.tolist()}")
    return lift


def _do_squeeze(redis_client, params: EggCrackParams) -> None:
    """Squeeze fully closed at crack force; arm holds lifted Cartesian pose.

    The Franka gripper driver hardcodes target width to 0.0 in grasp mode and
    passes ``epsilon_inner=1.0`` to ``franka::Gripper::grasp`` (see
    drivers/FrankaPanda/redis_driver/gripper.cpp). After the lift grasp the
    fingers are stopped on the cracker handles at some width within that
    tolerance, so a second mode=g call would early-return and apply no new
    force. We therefore retract the fingers by ``crack_unlatch_m`` using
    mode=m first, then re-issue mode=g at ``crack_force_n`` so libfranka
    actually closes the fingers again under force.
    """
    if params.crack_unlatch_m > 0.0:
        cur = read_gripper_current_width(redis_client)
        base_w = cur if cur is not None else float(params.gripper_pregrasp_width)
        unlatch_w = float(base_w) + float(params.crack_unlatch_m)
        set_gripper_width(
            redis_client,
            unlatch_w,
            speed=params.gripper_speed,
            force=params.crack_force_n,
            mode=GRIPPER_MODE_MOVE,
        )
        print(
            f"[5a] Unlatch: open {params.crack_unlatch_m * 1000.0:.1f} mm "
            f"(width={unlatch_w:.4f} m, mode=m), "
            f"settle {params.crack_unlatch_settle_s:.2f} s"
        )
        time.sleep(params.crack_unlatch_settle_s)
        squeeze_label = "[5b] Squeeze"
    else:
        squeeze_label = "[5] Squeeze"

    set_gripper_width(
        redis_client,
        0.0,
        speed=params.gripper_speed,
        force=params.crack_force_n,
        mode=GRIPPER_MODE_GRASP,
    )
    print(
        f"{squeeze_label} (fully close): force={params.crack_force_n:.1f} N "
        f"(mode=g), settle {params.crack_settle_s:.1f} s"
    )
    time.sleep(params.crack_settle_s)
    _log_gripper_width(redis_client, "after squeeze")


def _do_move_above_drop(
    redis_client,
    drop_pos: np.ndarray,
    grasp_ori: np.ndarray,
    params: EggCrackParams,
) -> np.ndarray:
    above = _above_drop(drop_pos, params)
    _publish_cartesian(redis_client, above, grasp_ori)
    print(f"[6] Move above drop: pos={above.tolist()}")
    return above


def _do_descend_to_drop(
    redis_client,
    drop_pos: np.ndarray,
    grasp_ori: np.ndarray,
) -> None:
    _publish_cartesian(redis_client, drop_pos, grasp_ori)
    print(f"[7] Descend to drop: pos={drop_pos.tolist()}")


def _do_open_gripper(redis_client, params: EggCrackParams) -> None:
    open_w = resolve_gripper_open_width(redis_client, params.gripper_open_width)
    set_gripper_width(
        redis_client,
        open_w,
        speed=params.gripper_speed,
        force=params.crack_force_n,
        mode=GRIPPER_MODE_OPEN_MAX,
    )
    print(f"[8] Open gripper (mode=o): width={open_w:.4f} m max")


def _validate_forces(lift_force: float, crack_force: float) -> None:
    if lift_force < MIN_GRIPPER_FORCE_N:
        raise ValueError(
            f"--gripper-lift-force must be >= {MIN_GRIPPER_FORCE_N} N, got {lift_force}"
        )
    if crack_force < MIN_GRIPPER_FORCE_N:
        raise ValueError(
            f"--gripper-crack-force must be >= {MIN_GRIPPER_FORCE_N} N, got {crack_force}"
        )
    if crack_force > HARDWARE_MAX_FORCE_N:
        raise ValueError(
            f"--gripper-crack-force {crack_force} N exceeds Franka gripper max "
            f"({HARDWARE_MAX_FORCE_N} N)"
        )
    if lift_force > WARN_LIFT_FORCE_N:
        print(
            f"Warning: lift force {lift_force:.1f} N > {WARN_LIFT_FORCE_N} N — "
            "may be too aggressive for a light tool hold.",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    default_pick = DEFAULT_PICK_POSITION.copy()
    default_drop = default_pick + np.array(
        [DEFAULT_DROP_OFFSET_X_M, DEFAULT_DROP_OFFSET_Y_M, DEFAULT_DROP_OFFSET_Z_M],
        dtype=np.float64,
    )

    p = argparse.ArgumentParser(
        description="Egg-cracker arm sequence (ENTER-gated, OpenSai Franka Redis)."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)

    p.add_argument("--pick-x", type=float, default=float(default_pick[0]))
    p.add_argument("--pick-y", type=float, default=float(default_pick[1]))
    p.add_argument("--pick-z", type=float, default=float(default_pick[2]))
    p.add_argument("--drop-x", type=float, default=float(default_drop[0]))
    p.add_argument("--drop-y", type=float, default=float(default_drop[1]))
    p.add_argument("--drop-z", type=float, default=float(default_drop[2]))

    p.add_argument(
        "--approach-dz",
        type=float,
        default=DEFAULT_APPROACH_DZ_M,
        help="Vertical clearance for approach / lift / above-drop (m).",
    )
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
    )
    p.add_argument("--gripper-speed", type=float, default=DEFAULT_GRIPPER_SPEED)
    p.add_argument(
        "--gripper-lift-force",
        type=float,
        default=DEFAULT_LIFT_FORCE_N,
        help="Grasp force (N) to hold cracker for lift (default 8).",
    )
    p.add_argument(
        "--gripper-crack-force",
        type=float,
        default=DEFAULT_CRACK_FORCE_N,
        help=(
            f"Grasp force (N) to squeeze fully closed (default "
            f"{DEFAULT_CRACK_FORCE_N:.0f} = hardware max)."
        ),
    )
    p.add_argument(
        "--gripper-pregrasp-settle",
        type=float,
        default=DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    )
    p.add_argument(
        "--gripper-grasp-settle",
        type=float,
        default=DEFAULT_GRIPPER_GRASP_SETTLE_S,
        help="Wait after lift grasp before moving arm (s).",
    )
    p.add_argument(
        "--crack-settle",
        type=float,
        default=DEFAULT_CRACK_SETTLE_S,
        help="Wait after squeeze before moving arm (s).",
    )
    p.add_argument(
        "--crack-unlatch-mm",
        type=float,
        default=DEFAULT_CRACK_UNLATCH_M * 1000.0,
        help=(
            "Briefly retract gripper this many mm (mode=m) before squeezing, "
            "to break libfranka grasp() early-return after the lift grasp. "
            "Set 0 to disable. Default 5 mm."
        ),
    )
    p.add_argument(
        "--crack-unlatch-settle",
        type=float,
        default=DEFAULT_CRACK_UNLATCH_SETTLE_S,
        help="Settle time (s) after the unlatch retract, before squeeze.",
    )
    return p.parse_args()


def run_loop(redis_client, params: EggCrackParams) -> int:
    open_w = resolve_gripper_open_width(redis_client, params.gripper_open_width)
    params = EggCrackParams(
        approach_dz_m=params.approach_dz_m,
        pick_pos=params.pick_pos.copy(),
        drop_pos=params.drop_pos.copy(),
        grasp_ori=params.grasp_ori.copy(),
        gripper_open_width=open_w,
        gripper_pregrasp_width=params.gripper_pregrasp_width,
        gripper_speed=params.gripper_speed,
        lift_force_n=params.lift_force_n,
        crack_force_n=params.crack_force_n,
        gripper_pregrasp_settle_s=params.gripper_pregrasp_settle_s,
        gripper_grasp_settle_s=params.gripper_grasp_settle_s,
        crack_settle_s=params.crack_settle_s,
        crack_unlatch_m=params.crack_unlatch_m,
        crack_unlatch_settle_s=params.crack_unlatch_settle_s,
    )

    pick_pos = params.pick_pos
    drop_pos = params.drop_pos
    grasp_ori = params.grasp_ori

    print(
        f"Motion: approach_dz={params.approach_dz_m} m, "
        f"lift_force={params.lift_force_n:.1f} N, "
        f"crack_force={params.crack_force_n:.1f} N, "
        f"pregrasp={params.gripper_pregrasp_width:.4f} m, "
        f"gripper_open={open_w:.4f} m"
    )
    if params.crack_unlatch_m > 0.0:
        print(
            f"Squeeze unlatch: retract {params.crack_unlatch_m * 1000.0:.1f} mm "
            f"(mode=m, settle {params.crack_unlatch_settle_s:.2f} s) before "
            f"high-force grasp"
        )
    print(f"Pick pose:  pos={pick_pos.tolist()}")
    print(f"Drop pose:  pos={drop_pos.tolist()}")

    ee = read_current_ee_world(redis_client)
    if ee is not None:
        cur_pos, _ = ee
        print(f"Current EE: pos={cur_pos.round(4).tolist()}")
    else:
        print(
            "Current EE: unavailable (OpenSai cartesian_task::current_position "
            "not on Redis yet?)"
        )
    cur_w = read_gripper_current_width(redis_client)
    if cur_w is not None:
        print(f"Current gripper width: {cur_w:.4f} m")

    print(
        "Keys (ENTER to submit): "
        "[empty]=advance phase | q=quit"
    )
    print(
        "(No motion at startup — first ENTER moves above the cracker.)"
    )

    phase = Phase.AWAIT_ABOVE_PICK
    stdin_dead = False
    print(_phase_hint(phase))

    try:
        while True:
            if stdin_dead:
                time.sleep(TICK_DT_S)
                continue

            line = _stdin_line_ready(TICK_DT_S)
            if line is None:
                continue
            if line is _STDIN_EOF:
                print(
                    "stdin closed (no terminal attached). State will NOT advance.\n"
                    "Run this controller directly in a terminal. Ctrl+C to quit.",
                    file=sys.stderr,
                )
                stdin_dead = True
                continue
            token = line.strip().lower()
            if token in ("q", "quit", "exit"):
                print("Quit requested.")
                return 0
            if token != "":
                print(
                    f"(unknown input: {token!r}; press ENTER to advance, q to quit)"
                )
                continue

            if phase == Phase.AWAIT_ABOVE_PICK:
                _do_move_above_pick(redis_client, pick_pos, grasp_ori, params)
                phase = Phase.ABOVE_PICK
            elif phase == Phase.ABOVE_PICK:
                _do_descend_to_pick(redis_client, pick_pos, grasp_ori)
                phase = Phase.AT_PICK
            elif phase == Phase.AT_PICK:
                _do_lift_grasp(redis_client, params)
                phase = Phase.GRASPED
            elif phase == Phase.GRASPED:
                _do_lift(redis_client, pick_pos, grasp_ori, params)
                phase = Phase.LIFTED
            elif phase == Phase.LIFTED:
                _do_squeeze(redis_client, params)
                phase = Phase.SQUEEZED
            elif phase == Phase.SQUEEZED:
                _do_move_above_drop(redis_client, drop_pos, grasp_ori, params)
                phase = Phase.ABOVE_DROP
            elif phase == Phase.ABOVE_DROP:
                _do_descend_to_drop(redis_client, drop_pos, grasp_ori)
                phase = Phase.AT_DROP
            elif phase == Phase.AT_DROP:
                _do_open_gripper(redis_client, params)
                phase = Phase.DONE
                print("Cracker released at drop pose.")
            elif phase == Phase.DONE:
                print("Sequence done — q to quit.")

            print(_phase_hint(phase))
    except KeyboardInterrupt:
        print("\nKeyboard interrupt.")
        return 0


def main() -> int:
    args = parse_args()
    try:
        _validate_forces(args.gripper_lift_force, args.gripper_crack_force)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    redis_client = _try_redis(args.redis_host, args.redis_port)
    if redis_client is None:
        return 1
    err = validate_config(redis_client)
    if err is not None:
        return err

    pick_pos = np.array([args.pick_x, args.pick_y, args.pick_z], dtype=np.float64)
    drop_pos = np.array([args.drop_x, args.drop_y, args.drop_z], dtype=np.float64)

    params = EggCrackParams(
        approach_dz_m=args.approach_dz,
        pick_pos=pick_pos,
        drop_pos=drop_pos,
        grasp_ori=GRASP_ORIENTATION.copy(),
        gripper_open_width=args.gripper_open_width,
        gripper_pregrasp_width=args.gripper_pregrasp_width,
        gripper_speed=args.gripper_speed,
        lift_force_n=float(args.gripper_lift_force),
        crack_force_n=float(args.gripper_crack_force),
        gripper_pregrasp_settle_s=args.gripper_pregrasp_settle,
        gripper_grasp_settle_s=args.gripper_grasp_settle,
        crack_settle_s=args.crack_settle,
        crack_unlatch_m=max(float(args.crack_unlatch_mm), 0.0) / 1000.0,
        crack_unlatch_settle_s=float(args.crack_unlatch_settle),
    )
    return run_loop(redis_client, params)


if __name__ == "__main__":
    sys.exit(main())
