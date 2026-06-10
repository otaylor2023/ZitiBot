#!/usr/bin/env python3
"""Scramble-eggs orchestrator (single RealSense session).

End-to-end "scrambled eggs" routine driven from one shared ``TaskContext`` with
a single ``ctx.realsense()`` warmup at startup. The pipeline is:

  1. grasp_egg  — tongs egg grasper: grab the tongs, pick the egg, drop it into
                  the cracker, return the tongs (``run_tongs_egg_cycle``)
  2. crack_egg  — egg cracker: grasp the cracker, crack over the bowl, empty,
                  return the cracker (``run_egg_crack_cycle``)
  3. whisk      — whisk the eggs in the bowl (``run_whisk_cycle``)
  4. pour       — pour the whisked eggs from the bowl onto the pan
                  (``run_egg_pour_new_cycle``)
  5. scramble   — move the pan back, scramble with the ladle, move the pan
                  forward again (``run_scramble_cycle``)

The grasp_egg → crack_egg sub-sequence runs ``--egg-cycles`` times (default 2 —
i.e. two eggs into the bowl) before whisking. The whisk, pour and scramble steps
each run once.

Same step-control flags as the ziti state machine (``--step``,
``--start-step`` / ``--stop-step`` / ``--skip-step`` / ``--list-steps``).

Usage::

  ./ZitiBot/launch_zitibot_full.sh controllers/routines/scramble_eggs_state_machine.py
  ./ZitiBot/launch_zitibot_full.sh controllers/routines/scramble_eggs_state_machine.py -- --step
  ./ZitiBot/launch_zitibot_full.sh controllers/routines/scramble_eggs_state_machine.py -- \\
      --start-step whisk

Requires OpenSai cartesian controller, Franka arm + gripper Redis drivers,
TidyBot base ``redis_driver``, RealSense, OptiTrack on Redis, and
``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
from collections.abc import Callable
from pathlib import Path

# Dump the Python traceback (all threads) if the process gets a fatal native
# signal — e.g. the glibc SIGABRT from heap corruption seen in the camera/Gemini
# stack. Pairs with the optional gdb wrapper (CONTROLLER_WRAPPER) in
# launch_zitibot_full.sh: faulthandler shows which Python call was on the stack;
# gdb shows the offending native library frame.
faulthandler.enable()

_CONTROLLERS = Path(__file__).resolve().parent.parent
if str(_CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(_CONTROLLERS))

import egg_pour_controller_new as egg_pour_new
from egg_crack_controller import run_egg_crack_cycle
from egg_grasp_controller import run_tongs_egg_cycle
from egg_pour_controller_new import run_egg_pour_new_cycle
from scramble_controller import run_scramble_cycle
from whisk_controller import run_whisk_cycle
from zitibot_core.constants import DEFAULT_POUR_AXIS, DEFAULT_POUR_TILT_DEG
from zitibot_core.context import TaskContext, make_context

# Runner takes the context plus a per-step ``skip_base`` decided at run time.
StepRunner = Callable[[TaskContext, bool], None]

# How many times to run the grasp_egg → crack_egg sub-sequence before whisking.
# Defaults to two eggs; override with --egg-cycles.
DEFAULT_EGG_CYCLES = 2

# Base stations each step operates at. The base is only driven when a step's
# station differs from the previously executed step's station, so we don't
# re-drive to the same waypoint (which drifts the cart on each re-approach).
STATION_EGG_CRACK = "egg_crack"
STATION_STOVE = "stove"


def _run_pour(ctx: TaskContext) -> None:
    """Pour the whisked eggs onto the pan using the egg-pour-new controller's
    defaults (same as running ``egg_pour_controller_new.py`` directly)."""
    run_egg_pour_new_cycle(
        ctx,
        retries=1,
        detection_pos=egg_pour_new.DETECTION_EE_POSITION.copy(),
        pour_tilt_deg=DEFAULT_POUR_TILT_DEG,
        pour_axis=DEFAULT_POUR_AXIS,
        pan_pour_offset_m=egg_pour_new.PAN_CENTER_POUR_OFFSET_M.copy(),
        pour_tilt_rise_m=egg_pour_new.POUR_TILT_RISE_M,
        tilt_neg_y_m=egg_pour_new.POUR_TILT_NEG_Y_M,
        tilt_angular_vel_rad_s=egg_pour_new.POUR_TILT_MAX_ANGULAR_VEL_RAD_S,
        bowl_gemini_response_path=egg_pour_new.DEFAULT_GEMINI_BOWL_RESPONSE_PATH,
        pan_gemini_response_path=egg_pour_new.DEFAULT_GEMINI_PAN_RESPONSE_PATH,
        return_home=False,
    )


def _build_steps(
    *, egg_cycles: int
) -> tuple[tuple[str, str, StepRunner], ...]:
    """Build the ordered ``(name, station, runner)`` steps.

    The grasp_egg → crack_egg pair is repeated ``egg_cycles`` times (one egg per
    cycle, suffixed ``_1``, ``_2``, … when more than one), followed by a single
    whisk → pour → scramble. ``station`` is used by ``main`` to decide whether a
    base move is needed (only when it differs from the previous step).
    """
    steps: list[tuple[str, str, StepRunner]] = []
    for c in range(egg_cycles):
        suffix = f"_{c + 1}" if egg_cycles > 1 else ""
        steps.append(
            (
                f"grasp_egg{suffix}",
                STATION_EGG_CRACK,
                lambda ctx, skip_base: run_tongs_egg_cycle(ctx, skip_base=skip_base),
            )
        )
        steps.append(
            (
                f"crack_egg{suffix}",
                STATION_EGG_CRACK,
                lambda ctx, skip_base: run_egg_crack_cycle(ctx, skip_base=skip_base),
            )
        )
    steps.append(
        (
            "whisk",
            STATION_EGG_CRACK,
            lambda ctx, skip_base: run_whisk_cycle(ctx, skip_base=skip_base),
        )
    )
    # egg_pour_controller_new always drives the base to STOVE_STATION itself.
    steps.append(("pour", STATION_STOVE, lambda ctx, skip_base: _run_pour(ctx)))
    steps.append(
        (
            "scramble",
            STATION_STOVE,
            lambda ctx, skip_base: run_scramble_cycle(ctx, skip_base=skip_base),
        )
    )
    return tuple(steps)


def _resolve_step_index(name_or_index: str, step_names: tuple[str, ...]) -> int:
    """Map a step name or 0-based index string to a step index."""
    if name_or_index.isdigit():
        idx = int(name_or_index)
        if idx < 0 or idx >= len(step_names):
            raise ValueError(
                f"step index {idx} out of range [0, {len(step_names) - 1}]"
            )
        return idx
    if name_or_index not in step_names:
        raise ValueError(
            f"unknown step {name_or_index!r}; choices: {', '.join(step_names)}"
        )
    return step_names.index(name_or_index)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scramble eggs: grasp egg → crack egg (xN) → whisk → pour → scramble."
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
        "--skip-base",
        action="store_true",
        help=(
            "Do not drive the base between stations in the sub-cycles (arm-only "
            "test when the cart is already parked). Note: the pour step always "
            "drives to STOVE_STATION itself."
        ),
    )
    p.add_argument(
        "--egg-cycles",
        type=int,
        default=DEFAULT_EGG_CYCLES,
        metavar="N",
        help=(
            "Number of grasp_egg → crack_egg cycles before whisking (one egg "
            f"per cycle). Default: {DEFAULT_EGG_CYCLES}."
        ),
    )
    p.add_argument(
        "--start-step",
        default=None,
        metavar="NAME|INDEX",
        help="First step to run (inclusive). Name or 0-based index.",
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

    egg_cycles = int(args.egg_cycles)
    if egg_cycles < 1:
        print(f"Error: --egg-cycles ({egg_cycles}) must be >= 1", file=sys.stderr)
        return 1

    steps = _build_steps(egg_cycles=egg_cycles)
    step_names = tuple(name for name, _, _ in steps)

    if args.list_steps:
        for idx, name in enumerate(step_names):
            print(f"  {idx}: {name}")
        return 0

    start_idx = 0
    stop_idx = len(steps) - 1
    if args.start_step is not None:
        start_idx = _resolve_step_index(args.start_step, step_names)
    if args.stop_step is not None:
        stop_idx = _resolve_step_index(args.stop_step, step_names)
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
        f"Scramble-eggs routine: steps {start_idx}–{stop_idx} "
        f"({step_names[start_idx]} … {step_names[stop_idx]})"
    )
    print(f"Egg cycles: {egg_cycles} (one egg per cycle)")
    if skip_steps:
        print(f"Skipping: {', '.join(sorted(skip_steps))}")
    print(f"Base move: {'skipped (--skip-base)' if args.skip_base else 'on'}")

    print("Warming up RealSense once for the scramble-eggs routine...")
    ctx.realsense()

    completed: list[str] = []
    # Station of the last executed step. The base is only driven when the next
    # step's station differs, so same-station steps don't re-drive (and drift)
    # the cart to a waypoint it is already at.
    current_station: str | None = None
    try:
        for idx, (name, station, runner) in enumerate(steps):
            if idx < start_idx:
                continue
            if idx > stop_idx:
                break
            if name in skip_steps:
                print(
                    f"\n##### scramble-eggs step {idx + 1}/{len(steps)}: "
                    f"{name} (skipped) #####",
                    flush=True,
                )
                continue
            # Move the base only when the station changes (and base moves aren't
            # globally disabled). The first executed step always moves since the
            # cart's station is unknown.
            move_base = (not args.skip_base) and (station != current_station)
            step_skip_base = not move_base
            print(
                f"\n##### scramble-eggs step {idx + 1}/{len(steps)}: {name} "
                f"(station={station}, base move={'yes' if move_base else 'no'}) #####",
                flush=True,
            )
            runner(ctx, step_skip_base)
            current_station = station
            completed.append(name)
    except KeyboardInterrupt:
        print(
            f"\nInterrupted. Completed {len(completed)} steps: "
            f"[{', '.join(completed) or '<none>'}]."
        )
        return 130
    finally:
        ctx.stop_realsense()

    print(
        f"Scramble-eggs routine complete ({len(completed)} steps): "
        f"[{', '.join(completed)}]."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
