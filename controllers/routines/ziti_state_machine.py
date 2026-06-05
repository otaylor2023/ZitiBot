#!/usr/bin/env python3
"""End-to-end ziti recipe orchestrator (single RealSense session).

Runs the full ziti preparation sequence using shared ``TaskContext`` and
one ``ctx.realsense()`` warmup at startup:

  1. Pour pasta bowl + plastic top bowl into mixing bowl (sink drop each)
  2. Crack egg into mixing bowl
  3. Pour sauce + ricotta into mixing bowl
  4. Grasp ladle, stir mixing bowl, drop ladle at sink
  5. Pour mixing bowl into pan (sink drop)
  6. Shake parmesan over pan
  7. Pick up pan, place in oven

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/routines/ziti_state_machine.py
  ./ZitiBot/launch_zitibot_full.sh controllers/routines/ziti_state_machine.py -- --step
  ./ZitiBot/launch_zitibot_full.sh controllers/routines/ziti_state_machine.py -- \\
      --start-step crack_egg --stop-step ladle_mix
  ./ZitiBot/launch_zitibot_full.sh controllers/routines/ziti_state_machine.py -- \\
      --skip-step pour_plastic_bottom --skip-step shake_parmesan

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
TidyBot base ``redis_driver``, RealSense, OptiTrack on Redis, and
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

_CONTROLLERS = Path(__file__).resolve().parent.parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

from bowl_pour_controller import run_bowl_cycle
from egg_crack_controller import run_egg_crack_cycle
from grasp_and_move_controller import run_pan_to_oven_cycle
from grasp_and_pour_jar_controller import (
    Cylinder,
    default_motion_params,
    run_cylinder_cycle,
)
from mixing_vision_base_controller import run_mixing_cycle
from zitibot_core.constants import Object
from zitibot_core.context import TaskContext, make_context

StepRunner = Callable[[TaskContext], None]

STEP_NAMES: tuple[str, ...] = (
    "pour_pasta_bowl",
    "pour_plastic_bottom",
    "crack_egg",
    "pour_sauce",
    "pour_ricotta",
    "ladle_mix",
    "pour_mixing_to_pan",
    "shake_parmesan",
    "pan_to_oven",
)


def _run_pour_sauce(ctx: TaskContext) -> None:
    run_cylinder_cycle(ctx, Cylinder.SAUCE, motion=default_motion_params(ctx))


def _run_pour_ricotta(ctx: TaskContext) -> None:
    run_cylinder_cycle(ctx, Cylinder.RICOTTA, motion=default_motion_params(ctx))


def _run_shake_parmesan(ctx: TaskContext) -> None:
    run_cylinder_cycle(ctx, Cylinder.PARMESAN, motion=default_motion_params(ctx))


STEPS: tuple[tuple[str, StepRunner], ...] = (
    ("pour_pasta_bowl", lambda ctx: run_bowl_cycle(ctx, Object.PASTA_BOWL)),
    ("pour_plastic_bottom", lambda ctx: run_bowl_cycle(ctx, Object.PLASTIC_BOWL_BOTTOM)),
    ("crack_egg", lambda ctx: run_egg_crack_cycle(ctx)),
    ("pour_sauce", _run_pour_sauce),
    ("pour_ricotta", _run_pour_ricotta),
    ("ladle_mix", lambda ctx: run_mixing_cycle(ctx)),
    (
        "pour_mixing_to_pan",
        lambda ctx: run_bowl_cycle(ctx, Object.MIXING_BOWL),
    ),
    ("shake_parmesan", _run_shake_parmesan),
    ("pan_to_oven", lambda ctx: run_pan_to_oven_cycle(ctx)),
)


def _resolve_step_index(name_or_index: str) -> int:
    """Map a step name or 0-based index string to a step index."""
    if name_or_index.isdigit():
        idx = int(name_or_index)
        if idx < 0 or idx >= len(STEP_NAMES):
            raise ValueError(
                f"step index {idx} out of range [0, {len(STEP_NAMES) - 1}]"
            )
        return idx
    if name_or_index not in STEP_NAMES:
        raise ValueError(
            f"unknown step {name_or_index!r}; "
            f"choices: {', '.join(STEP_NAMES)}"
        )
    return STEP_NAMES.index(name_or_index)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Full ziti recipe: bowls → egg → sauce/ricotta → mix → "
            "mixing-bowl→pan → parmesan → pan to oven."
        )
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate every motion / gripper / base step in subtasks.",
    )
    p.add_argument(
        "--start-step",
        default=None,
        metavar="NAME|INDEX",
        help=(
            "First step to run (inclusive). Name from: "
            f"{', '.join(STEP_NAMES)} — or 0-based index."
        ),
    )
    p.add_argument(
        "--stop-step",
        default=None,
        metavar="NAME|INDEX",
        help="Last step to run (inclusive). Same format as --start-step.",
    )
    p.add_argument(
        "--skip-step",
        action="append",
        default=[],
        metavar="NAME",
        help="Skip a step by name (repeatable).",
    )
    p.add_argument(
        "--list-steps",
        action="store_true",
        help="Print the ordered step list and exit.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_steps:
        for idx, name in enumerate(STEP_NAMES):
            print(f"  {idx}: {name}")
        return 0

    start_idx = 0
    stop_idx = len(STEPS) - 1
    if args.start_step is not None:
        start_idx = _resolve_step_index(args.start_step)
    if args.stop_step is not None:
        stop_idx = _resolve_step_index(args.stop_step)
    if start_idx > stop_idx:
        print(
            f"Error: --start-step ({start_idx}) is after --stop-step ({stop_idx})",
            file=sys.stderr,
        )
        return 1

    skip_steps = set(args.skip_step or [])

    ctx = make_context(args, step=args.step)
    print(f"Step mode: {'on' if args.step else 'off'}")
    print(
        f"Ziti routine: steps {start_idx}–{stop_idx} "
        f"({STEP_NAMES[start_idx]} … {STEP_NAMES[stop_idx]})"
    )
    if skip_steps:
        print(f"Skipping: {', '.join(sorted(skip_steps))}")

    print("Warming up RealSense once for the full ziti routine...")
    ctx.realsense()

    completed: list[str] = []
    try:
        for idx, (name, runner) in enumerate(STEPS):
            if idx < start_idx:
                continue
            if idx > stop_idx:
                break
            if name in skip_steps:
                print(f"\n##### ziti step: {name} (skipped) #####", flush=True)
                continue
            print(
                f"\n##### ziti step {idx + 1}/{len(STEPS)}: {name} #####",
                flush=True,
            )
            runner(ctx)
            completed.append(name)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted. Completed {len(completed)}/{len(STEPS)} steps: "
            f"[{', '.join(completed) or '<none>'}]."
        )
        return 130
    finally:
        ctx.stop_realsense()

    print(
        f"Ziti routine complete ({len(completed)} steps): "
        f"[{', '.join(completed)}]."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
