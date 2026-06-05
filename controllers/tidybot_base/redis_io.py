"""Redis serialization and hb1 base pose keys."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass

import numpy as np
import redis


@dataclass(frozen=True)
class BaseRedisKeys:
    """TidyBot base odometry + commands (``redis_driver`` / ``Vehicle``)."""

    robot_pose: str = "hb1::current_pose"
    robot_vel: str = "hb1::current_vel"
    desired_pose: str = "hb1::desired_pose"
    stop: str = "hb1::stop"
    kill: str = "hb1::kill"
    # Live scale on the driver's OTG max_velocity. Driver multiplies its
    # configured CLI baseline ``(--max-vel-xy, --max-vel-xy, --max-vel-yaw)``
    # by this scalar each control tick. Missing / invalid key = 1.0
    # (driver default). Clamped to [0.1, 3.0] in the driver to guard
    # against typos. Used by ``zitibot_core.base.go_to_pose`` to make
    # the holonomic path drive faster than the three-phase path without
    # restarting the driver.
    max_vel_scale: str = "hb1::max_vel_scale"


DEFAULT_BASE_KEYS = BaseRedisKeys()


def numpy_array_to_string(array: np.ndarray) -> str:
    if isinstance(array, np.ndarray) and array.ndim == 1:
        return "[" + ", ".join(map(str, array.tolist())) + "]"
    return ""


def parse_redis_list(raw: bytes | str | None) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    raw = raw.strip()
    if raw.startswith("["):
        try:
            values = ast.literal_eval(raw)
            if isinstance(values, list):
                return np.array(values, dtype=np.float64)
        except (SyntaxError, ValueError):
            pass
    try:
        values = json.loads(raw)
        if isinstance(values, list):
            return np.array(values, dtype=np.float64)
    except json.JSONDecodeError:
        pass
    return None


def connect_redis(host: str, port: int) -> redis.Redis:
    client = redis.Redis(host=host, port=port, decode_responses=True)
    client.ping()
    return client


def read_robot_se2(
    client: redis.Redis,
    pose_key: str | None = None,
    *,
    keys: BaseRedisKeys = DEFAULT_BASE_KEYS,
) -> np.ndarray:
    key = pose_key or keys.robot_pose
    pos = parse_redis_list(client.get(key))
    if pos is None or pos.size < 3:
        raise RuntimeError(f"Invalid {key!r} (need [x, y, yaw])")
    return pos[:3].astype(np.float64)


def write_desired_pose(
    client: redis.Redis,
    goal_se2: np.ndarray,
    key: str | None = None,
    *,
    keys: BaseRedisKeys = DEFAULT_BASE_KEYS,
) -> None:
    client.set(key or keys.desired_pose, numpy_array_to_string(goal_se2.reshape(3)))


def stop_base(
    client: redis.Redis,
    key: str | None = None,
    *,
    keys: BaseRedisKeys = DEFAULT_BASE_KEYS,
) -> None:
    """Tell redis_driver to decelerate and hold current pose."""
    client.set(key or keys.stop, "stop")


def write_max_vel_scale(
    client: redis.Redis,
    scale: float,
    *,
    keys: BaseRedisKeys = DEFAULT_BASE_KEYS,
) -> None:
    """Set the driver's live OTG max_velocity multiplier.

    Driver multiplies its CLI baseline by this scalar each control tick.
    Defaults to 1.0 if the key is missing or unparseable. The driver
    clamps to [0.1, 3.0] for safety; callers shouldn't rely on values
    outside that range taking effect.
    """
    client.set(keys.max_vel_scale, f"{float(scale):.6f}")
