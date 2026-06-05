"""Oven placement subtask."""

from __future__ import annotations

import numpy as np

from zitibot_core.constants import Object
from zitibot_core.context import TaskContext
from zitibot_tasks import grasp


def place(
    ctx: TaskContext,
    obj: Object,
    oven_pos: np.ndarray,
    *,
    approach_dz: float | None = None,
) -> None:
    """Transport held object to oven rack position and release."""
    oven = np.asarray(oven_pos, dtype=np.float64).reshape(3)
    grasp.place(ctx, obj, place_pos=oven)
