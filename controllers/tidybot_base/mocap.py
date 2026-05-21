"""OptiTrack Redis I/O."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import redis

from tidybot_base.redis_io import parse_redis_list


@dataclass(frozen=True)
class MocapRedisKeys:
    """OptiTrack rigid-body keys on Redis."""

    pos: str = "tidybot01::pos"
    ori: str = "tidybot01::ori"
    tracking_valid: str = "tidybot01::tracking_valid"


DEFAULT_MOCAP_KEYS = MocapRedisKeys()


def parse_tracking_valid(raw: bytes | str | None) -> bool:
    """Interpret Redis ``tracking_valid`` (true/1/yes/on)."""
    if raw is None:
        return False
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    try:
        return bool(int(float(s)))
    except ValueError:
        return False


def read_tracking_valid(
    client: redis.Redis,
    key: str | None = None,
    *,
    keys: MocapRedisKeys = DEFAULT_MOCAP_KEYS,
) -> bool:
    return parse_tracking_valid(client.get(key or keys.tracking_valid))


def wait_for_tracking_valid(
    client: redis.Redis,
    key: str,
    *,
    poll_hz: float = 10.0,
) -> None:
    period = 1.0 / max(poll_hz, 0.1)
    last_msg = 0.0
    print(f"Waiting for {key!r} == true before planning or motion...")
    while True:
        if read_tracking_valid(client, key):
            print(f"{key} is true — proceeding.")
            return
        now = time.perf_counter()
        if now - last_msg >= 2.0:
            raw = client.get(key)
            print(f"  still waiting ({key}={raw!r})")
            last_msg = now
        time.sleep(period)


def read_mocap_pose(
    client: redis.Redis,
    pos_key: str | None = None,
    ori_key: str | None = None,
    *,
    keys: MocapRedisKeys = DEFAULT_MOCAP_KEYS,
) -> tuple[np.ndarray, np.ndarray]:
    pk = pos_key or keys.pos
    ok = ori_key or keys.ori
    pos = parse_redis_list(client.get(pk))
    ori = parse_redis_list(client.get(ok))
    if pos is None or pos.size < 3:
        raise RuntimeError(f"Missing or invalid {pk!r} (is OptiTrack on Redis?)")
    if ori is None or ori.size < 4:
        raise RuntimeError(f"Missing or invalid {ok!r}")
    return pos[:3].astype(np.float64), ori[:4].astype(np.float64)
