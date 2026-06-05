#!/usr/bin/env python3
"""Compatibility wrapper for the legacy grasp-and-pour controller module."""

from __future__ import annotations

# Older vision controllers import these symbols from this top-level module.
# The implementation now lives in zitibot_core.legacy_grasp_pour.
from zitibot_core.legacy_grasp_pour import (  # noqa: F401
    DEFAULT_APPROACH_DZ_M,
    DEFAULT_GRIPPER_FORCE,
    DEFAULT_GRIPPER_GRASP_SETTLE_S,
    DEFAULT_GRIPPER_PREGRASP_SETTLE_S,
    DEFAULT_GRIPPER_PREGRASP_WIDTH,
    DEFAULT_GRIPPER_SPEED,
    DEFAULT_POUR_AXIS,
    DEFAULT_POUR_TILT_DEG,
    DEFAULT_TILT_DURATION_S,
    GRASP_ORIENTATION,
    GRASP_POSITION,
    GRIPPER_MODE_GRASP,
    GRIPPER_MODE_MOVE,
    GRIPPER_MODE_OPEN_MAX,
    MotionParams,
    OrientationSlerpState,
    PICK_POSITION,
    POUR_POSITION,
    POUR_TICK_DT_S,
    _STDIN_EOF,
    _do_descend_to_grasp,
    _do_grasp_object,
    _do_move_above_grasp,
    _do_open_gripper,
    _publish_cartesian,
    _start_orientation_slerp,
    _stdin_line_ready,
    _tick_orientation_slerp,
    _try_redis,
    pour_orientation_end,
    read_current_ee_world,
    read_gripper_current_width,
    resolve_gripper_open_width,
    set_gripper_width,
    validate_config,
)


def main() -> int:
    """Run the archived fixed-pose grasp-and-pour CLI."""
    from archive.grasp_and_pour_controller import main as archived_main

    return archived_main()


if __name__ == "__main__":
    raise SystemExit(main())
