"""Mixing subtask (ladle in bowl)."""

from __future__ import annotations

import math
import time

import numpy as np

from zitibot_core import arm
from zitibot_core.constants import OBJECT_DEFAULTS, Object, TICK_DT_S
from zitibot_core.context import TaskContext
from zitibot_core.runner import step_gate
from zitibot_tasks import grasp


def in_bowl(
    ctx: TaskContext,
    bowl_pos: np.ndarray,
    *,
    ladle_obj: Object = Object.LADLE,
    radius_m: float = 0.04,
    cycles: int = 3,
    cycle_duration_s: float = 4.0,
) -> None:
    """Grasp ladle, move into bowl, execute circular stir, lift out."""
    bowl = np.asarray(bowl_pos, dtype=np.float64).reshape(3)
    ladle_spec = OBJECT_DEFAULTS[ladle_obj]
    ladle_pick = ladle_spec.pick_pose
    if ladle_pick is None:
        raise ValueError(f"{ladle_obj.value}: no default pick_pose")

    grasp.object(ctx, ladle_obj, pick_pos=ladle_pick)
    mix_center = bowl.copy()
    mix_center[2] += 0.02
    grip_R = ladle_spec.grasp_ori.copy()

    arm.move_to(ctx, mix_center, grip_R, label=f"[mix] move above bowl center {mix_center.tolist()}")
    lower = mix_center.copy()
    lower[2] -= 0.03
    arm.move_to(ctx, lower, grip_R, label=f"[mix] lower into bowl to {lower.tolist()}")
    stir_start = lower + np.array([radius_m, 0.0, 0.0], dtype=np.float64)
    arm.move_to(ctx, stir_start, grip_R,
                label=f"[mix] move to stir start (theta=0) {stir_start.tolist()}")
    step_gate(ctx, f"[mix] stir {cycles} cycle(s) radius={radius_m} m")

    t_cycle = cycle_duration_s / max(cycles, 1)
    for c in range(cycles):
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= t_cycle:
                break
            theta = 2.0 * math.pi * (elapsed / t_cycle)
            offset = np.array(
                [radius_m * math.cos(theta), radius_m * math.sin(theta), 0.0],
                dtype=np.float64,
            )
            pos = lower + offset
            arm.publish_goal_cartesian(ctx.redis, pos, grip_R)
            if ctx.p_pressed():
                arm.print_ee_status(ctx.redis)
            if ctx.q_pressed():
                raise KeyboardInterrupt("quit requested")
            time.sleep(TICK_DT_S)
        print(f"[mix] cycle {c + 1}/{cycles} complete")

    arm.move_to(ctx, mix_center, grip_R, label=f"[mix] lift out to {mix_center.tolist()}")
    print("[mix] stir complete")


def stir_at_pose(
    ctx: TaskContext,
    above_pos: np.ndarray,
    above_ori: np.ndarray,
    *,
    down_dz_m: float,
    radius_m: float = 0.04,
    cycles: int = 3,
    cycle_duration_s: float = 4.0,
) -> None:
    """Stir at a hand-taught above pose.

    Variant of :func:`stir_in_bowl` that takes the above-pose directly
    instead of deriving it from a bowl center. Used by callers (e.g.
    ``mixing_vision_base_controller``) where the bowl-center
    derivation is wrong because the held ladle isn't perfectly
    tool-down — the taught above pose captures both the right XYZ
    above the bowl AND the held ladle's actual orientation.

    Sequence (all moves use the same ``above_ori``):
      1. ``arm.move_to(above_pos, above_ori)`` — taught above pose.
      2. ``arm.move_to(above_pos - [0, 0, down_dz_m], above_ori)`` —
         straight down by ``down_dz_m`` to the stir base.
      3. ``arm.move_to(stir_base + [radius_m, 0, 0], above_ori)`` —
         circle start (theta=0).
      4. Streams a circular XY trajectory around the stir base for
         ``cycles`` × ``cycle_duration_s / cycles`` seconds each, at
         ``radius_m`` (orientation held).
      5. ``arm.move_to(above_pos, above_ori)`` — lift back to taught
         above pose. Caller is responsible for any post-lift carry
         motion / gripper release.

    Assumes the arm is already holding the ladle at the same
    orientation it should keep through the stir.
    """
    above = np.asarray(above_pos, dtype=np.float64).reshape(3)
    ori = np.asarray(above_ori, dtype=np.float64).reshape(3, 3)
    lower = above - np.array([0.0, 0.0, float(down_dz_m)], dtype=np.float64)

    arm.move_to(ctx, above, ori,
                label=f"[mix] above (taught) {above.tolist()}")
    arm.move_to(ctx, lower, ori,
                label=(
                    f"[mix] descend {float(down_dz_m) * 100:.1f} cm to "
                    f"stir base {lower.tolist()}"
                ))
    stir_start = lower + np.array([radius_m, 0.0, 0.0], dtype=np.float64)
    arm.move_to(ctx, stir_start, ori,
                label=f"[mix] move to stir start (theta=0) {stir_start.tolist()}")
    step_gate(ctx, f"[mix] stir {cycles} cycle(s) radius={radius_m} m")

    t_cycle = cycle_duration_s / max(cycles, 1)
    for c in range(cycles):
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= t_cycle:
                break
            theta = 2.0 * math.pi * (elapsed / t_cycle)
            pos = lower + np.array(
                [radius_m * math.cos(theta), radius_m * math.sin(theta), 0.0],
                dtype=np.float64,
            )
            arm.publish_goal_cartesian(ctx.redis, pos, ori)
            if ctx.p_pressed():
                arm.print_ee_status(ctx.redis)
            if ctx.q_pressed():
                raise KeyboardInterrupt("quit requested")
            time.sleep(TICK_DT_S)
        print(f"[mix] cycle {c + 1}/{cycles} complete")

    arm.move_to(ctx, above, ori,
                label=f"[mix] lift back to taught above {above.tolist()}")
    print("[mix] stir complete")


def stir_in_bowl(
    ctx: TaskContext,
    bowl_pos: np.ndarray,
    grip_R: np.ndarray,
    *,
    radius_m: float = 0.04,
    cycles: int = 3,
    cycle_duration_s: float = 4.0,
) -> None:
    """Move ladle above bowl, descend, stir, lift out. Does NOT grasp or release.

    Assumes the arm is already holding the ladle at grip_R.
    Caller is responsible for opening the gripper afterward.
    """
    bowl = np.asarray(bowl_pos, dtype=np.float64).reshape(3)
    mix_center = bowl + np.array([0.0, 0.0, 0.02])
    lower = mix_center - np.array([0.0, 0.0, 0.03])

    arm.move_to(ctx, mix_center, grip_R,
                label=f"[mix] move above bowl center {mix_center.tolist()}")
    arm.move_to(ctx, lower, grip_R,
                label=f"[mix] lower into bowl to {lower.tolist()}")
    stir_start = lower + np.array([radius_m, 0.0, 0.0], dtype=np.float64)
    arm.move_to(ctx, stir_start, grip_R,
                label=f"[mix] move to stir start (theta=0) {stir_start.tolist()}")
    step_gate(ctx, f"[mix] stir {cycles} cycle(s) radius={radius_m} m")

    t_cycle = cycle_duration_s / max(cycles, 1)
    for c in range(cycles):
        t0 = time.monotonic()
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= t_cycle:
                break
            theta = 2.0 * math.pi * (elapsed / t_cycle)
            pos = lower + np.array(
                [radius_m * math.cos(theta), radius_m * math.sin(theta), 0.0],
                dtype=np.float64,
            )
            arm.publish_goal_cartesian(ctx.redis, pos, grip_R)
            if ctx.p_pressed():
                arm.print_ee_status(ctx.redis)
            if ctx.q_pressed():
                raise KeyboardInterrupt("quit requested")
            time.sleep(TICK_DT_S)
        print(f"[mix] cycle {c + 1}/{cycles} complete")

    arm.move_to(ctx, mix_center, grip_R,
                label=f"[mix] lift out to {mix_center.tolist()}")
    print("[mix] stir complete")
