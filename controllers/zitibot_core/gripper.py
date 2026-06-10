"""Franka gripper Redis helpers."""

from __future__ import annotations

import time

from zitibot_core.arm import decode_redis_value
from zitibot_core.redis_keys import (
    GRIPPER_GRASP_PENDING,
    GRIPPER_MODE_GRASP,
    GRIPPER_MODE_MOVE,
    GRIPPER_MODE_OPEN_MAX,
    KEYS,
)

GRIPPER_OBJECT_HELD_MIN_WIDTH_M = 0.003


def read_max_width(redis_client) -> float | None:
    raw = redis_client.get(KEYS.gripper_max_width)
    text = decode_redis_value(raw)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_current_width(redis_client) -> float | None:
    raw = redis_client.get(KEYS.gripper_current_width)
    text = decode_redis_value(raw)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_ready(redis_client) -> bool | None:
    """Return the gripper driver readiness flag.

    ``True`` once the driver has finished homing + startup init and is in its
    main loop, ``False`` while it is still starting up, ``None`` if the key is
    absent (older driver that doesn't publish it).
    """
    raw = redis_client.get(KEYS.gripper_ready)
    text = decode_redis_value(raw)
    if text is None:
        return None
    return text.strip() == "1"


def wait_for_ready(redis_client, *, timeout_s: float = 30.0, poll_dt_s: float = 0.1) -> bool:
    """Block until the gripper driver reports ready, or timeout.

    Returns ``True`` once the driver flips its ready flag to "1". Returns
    ``True`` immediately if the key is absent (older driver that doesn't
    publish readiness — nothing to wait on). Returns ``False`` only if the
    driver is present but still not ready after ``timeout_s``.

    This guards against the startup race where the driver's ~10 s homing +
    init resets the command keys: a command issued before the driver is ready
    is silently wiped, so the first grasp never happens.
    """
    deadline = time.perf_counter() + max(timeout_s, 0.0)
    announced = False
    while time.perf_counter() < deadline:
        state = read_ready(redis_client)
        if state is None:
            return True  # driver doesn't publish readiness; don't block
        if state:
            return True
        if not announced:
            print("[gripper] waiting for driver to finish homing...", flush=True)
            announced = True
        time.sleep(poll_dt_s)
    return bool(read_ready(redis_client))


def resolve_open_width(redis_client, override: float | None) -> float:
    if override is not None:
        return float(override)
    w = read_max_width(redis_client)
    if w is not None and w > 0:
        return w
    return 0.08


def set_width(
    redis_client,
    width_m: float,
    *,
    speed: float,
    force: float,
    mode: str = GRIPPER_MODE_MOVE,
) -> None:
    redis_client.set(KEYS.gripper_desired_width, str(float(width_m)))
    redis_client.set(KEYS.gripper_desired_speed, str(float(speed)))
    redis_client.set(KEYS.gripper_desired_force, str(float(force)))
    redis_client.set(KEYS.gripper_mode, mode)


def open_gripper(
    redis_client,
    width_m: float | None,
    *,
    speed: float,
    force: float,
    use_max_mode: bool = False,
) -> float:
    open_w = resolve_open_width(redis_client, width_m)
    set_width(
        redis_client,
        open_w,
        speed=speed,
        force=force,
        mode=GRIPPER_MODE_OPEN_MAX if use_max_mode else GRIPPER_MODE_MOVE,
    )
    return open_w


def move(
    redis_client,
    width_m: float,
    *,
    speed: float,
    force: float,
) -> None:
    set_width(redis_client, width_m, speed=speed, force=force, mode=GRIPPER_MODE_MOVE)


def grasp(
    redis_client,
    width_m: float,
    *,
    speed: float,
    force: float,
) -> None:
    # Clear any previous grasp result to "pending" BEFORE issuing the new
    # grasp so ``wait_for_grasp_result`` / ``read_grasp_success`` can tell a
    # fresh result from a stale one. The driver overwrites this with "1"/"0"
    # once its (blocking) grasp() returns.
    try:
        redis_client.set(KEYS.gripper_grasp_success, GRIPPER_GRASP_PENDING)
    except Exception:  # noqa: BLE001 - status is best-effort, never block the grasp
        pass
    set_width(redis_client, width_m, speed=speed, force=force, mode=GRIPPER_MODE_GRASP)


def read_grasp_success(redis_client) -> bool | None:
    """Return the driver's last grasp result.

    ``True`` = an object is held (libfranka grasp() succeeded and the fingers
    ended up apart), ``False`` = the close found nothing, ``None`` = result
    still pending or unavailable (driver too old to publish the key, or no
    grasp issued yet).
    """
    raw = redis_client.get(KEYS.gripper_grasp_success)
    text = decode_redis_value(raw)
    if text is None:
        return None
    text = text.strip().lower()
    if text in ("1", "true"):
        return True
    if text in ("0", "false"):
        return False
    return None  # GRIPPER_GRASP_PENDING or anything unexpected


def infer_grasp_success_from_width(
    redis_client,
    *,
    min_width_m: float = GRIPPER_OBJECT_HELD_MIN_WIDTH_M,
) -> bool | None:
    """Infer whether a force-close held something from the live finger width."""
    width = read_current_width(redis_client)
    if width is None:
        return None
    return width > min_width_m


def wait_for_width(
    redis_client,
    target_width_m: float,
    *,
    tol_m: float = 0.004,
    timeout_s: float = 3.0,
    poll_dt_s: float = 0.05,
) -> float:
    """Block until ``gripper_current_width`` reaches ``target_width_m`` (±tol).

    Returns the settled finger-gap width in metres. Raises ``TimeoutError`` if
    the jaws never converge (e.g. MOVE command not executed by the driver).
    """
    target = float(target_width_m)
    deadline = time.perf_counter() + max(timeout_s, 0.0)
    last: float | None = None
    while time.perf_counter() < deadline:
        last = read_current_width(redis_client)
        if last is not None and last <= target + tol_m:
            return last
        time.sleep(poll_dt_s)
    last = read_current_width(redis_client)
    msg = (
        f"gripper finger gap did not reach {target * 100:.1f} cm "
        f"(tol={tol_m * 100:.1f} cm"
    )
    if last is not None:
        msg += f", last read {last * 100:.1f} cm"
    msg += ")"
    raise TimeoutError(msg)


def wait_for_grasp_result(
    redis_client,
    *,
    timeout_s: float = 5.0,
    poll_dt_s: float = 0.05,
    fallback_to_width: bool = False,
) -> bool | None:
    """Block until the driver publishes a non-pending grasp result, or timeout.

    Returns ``True`` / ``False`` once the driver reports the outcome. If
    ``fallback_to_width`` is true and the result is still pending /
    unavailable after ``timeout_s`` (e.g. an older gripper driver that doesn't
    publish the key), infer the result from ``gripper_current_width`` using the
    same minimum-held-width threshold as the C++ driver.
    """
    deadline = time.perf_counter() + max(timeout_s, 0.0)
    while time.perf_counter() < deadline:
        result = read_grasp_success(redis_client)
        if result is not None:
            return result
        time.sleep(poll_dt_s)
    result = read_grasp_success(redis_client)
    if result is not None or not fallback_to_width:
        return result
    return infer_grasp_success_from_width(redis_client)
