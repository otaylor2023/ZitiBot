#!/usr/bin/env python3
"""Crack-eggs orchestrator (single RealSense session).

Runs the two-stage egg sequence using a shared ``TaskContext`` and one
``ctx.realsense()`` warmup at startup:

  1. grasp_egg  — tongs egg grasper: grab the tongs, pick the egg with the
                  tongs, drop it into the cracker, return the tongs
                  (``run_tongs_egg_cycle``)
  2. crack_egg  — egg cracker: grasp the cracker, crack over the bowl, empty,
                  return the cracker (``run_egg_crack_cycle``)

Same step-control flags as the ziti state machine (``--step``,
``--start-step`` / ``--stop-step`` / ``--skip-step`` / ``--list-steps``).

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/routines/crack_eggs_state_machine.py
  ./ZitiBot/launch_zitibot_full.sh controllers/routines/crack_eggs_state_machine.py -- --step
  ./ZitiBot/launch_zitibot_full.sh controllers/routines/crack_eggs_state_machine.py -- \\
      --start-step crack_egg

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

from egg_crack_controller import run_egg_crack_cycle
from egg_grasp_controller import run_tongs_egg_cycle
from zitibot_core.context import TaskContext, make_context

StepRunner = Callable[[TaskContext], None]

STEP_NAMES: tuple[str, ...] = (
    "grasp_egg",
    "crack_egg",
)


def _build_steps(*, skip_base: bool) -> tuple[tuple[str, StepRunner], ...]:
    """Build the ordered (name, runner) steps with ``skip_base`` bound in."""
    return (
        ("grasp_egg", lambda ctx: run_tongs_egg_cycle(ctx, skip_base=skip_base)),
        ("crack_egg", lambda ctx: run_egg_crack_cycle(ctx, skip_base=skip_base)),
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
        description="Crack eggs: egg grasper → egg cracker."
    )
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--step",
        action="store_true",
        help="ENTER-gate every motion / gripper / base step in subtasks.",
    )
    p.add_argument(
        "--skip-base",
        action="store_true",
        help=(
            "Do not drive the base to INGREDIENT_STATION in the egg grasp / "
            "crack sub-cycles (arm-only test when the cart is already parked)."
        ),
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

    steps = _build_steps(skip_base=args.skip_base)

    start_idx = 0
    stop_idx = len(steps) - 1
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
        f"Crack-eggs routine: steps {start_idx}–{stop_idx} "
        f"({STEP_NAMES[start_idx]} … {STEP_NAMES[stop_idx]})"
    )
    if skip_steps:
        print(f"Skipping: {', '.join(sorted(skip_steps))}")
    print(f"Base move: {'skipped (--skip-base)' if args.skip_base else 'on'}")

    print("Warming up RealSense once for the crack-eggs routine...")
    ctx.realsense()

    completed: list[str] = []
    try:
        for idx, (name, runner) in enumerate(steps):
            if idx < start_idx:
                continue
            if idx > stop_idx:
                break
            if name in skip_steps:
                print(f"\n##### crack-eggs step: {name} (skipped) #####", flush=True)
                continue
            print(
                f"\n##### crack-eggs step {idx + 1}/{len(steps)}: {name} #####",
                flush=True,
            )
            runner(ctx)
            completed.append(name)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted. Completed {len(completed)}/{len(steps)} steps: "
            f"[{', '.join(completed) or '<none>'}]."
        )
        return 130
    finally:
        ctx.stop_realsense()

    print(
        f"Crack-eggs routine complete ({len(completed)} steps): "
        f"[{', '.join(completed)}]."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
