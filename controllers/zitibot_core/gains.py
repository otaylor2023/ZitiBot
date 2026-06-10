"""Live cartesian-task gain tuning via Redis.

OpenSai's ``MotionForceTask`` registers its PD gains with
``addToReceiveGroup`` on startup (see
``core/sai-interfaces/src/controller/RobotControllerRedisInterface.cpp``),
which means the controller re-reads them from Redis on every loop tick
(~1 kHz). Writing a new value to one of the ``::cartesian_task::*_kp`` /
``*_kv`` keys takes effect immediately — no controller restart, no
config-file edit.

We use this to surgically stiffen the cart task during specific phases
(e.g. lifting a heavy sauce jar — see
``grasp_and_pour_jar_controller.run_cylinder_cycle``) and restore the
original gains afterwards so the rest of the routine isn't running at
elevated stiffness.

Value format is a JSON 1-element list (``"[300.0]"``); a scalar gets
wrapped by ``MotionForceTask::setPosControlGains(double, double, double)``
to ``kp * I3`` on the C++ side.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from zitibot_core.redis_keys import KEYS


@dataclass(frozen=True)
class CartPositionGainSnapshot:
    """Pre-boost gain values returned by :func:`snapshot_cart_position_gains`.

    Pass back to :func:`restore_cart_position_gains` to revert. Slots
    that were unreadable on snapshot stay ``None`` and are skipped on
    restore (so we never JSON-encode ``null`` into a gain key).
    """

    kp: float | None
    kv: float | None
    ki: float | None = None


@dataclass(frozen=True)
class JointGainSnapshot:
    """Pre-boost joint-task gains returned by :func:`boosted_joint_gains`."""

    kp: float | None
    kv: float | None
    ki: float | None = None


def _decode_scalar_gain(raw: bytes | str | None) -> float | None:
    """Parse one of the JSON 1-element gain values written by OpenSai."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = raw.strip()
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, list) and len(val) >= 1 and isinstance(val[0], (int, float)):
        return float(val[0])
    return None


def _encode_scalar_gain(value: float) -> str:
    """OpenSai accepts a 1-element JSON list — same format it writes."""
    return json.dumps([float(value)])


def read_cart_position_gains(
    redis_client,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(kp, kv, ki)`` for the cartesian task's position PID.

    ``None`` per slot if absent. ``ki`` is the integral term — it
    builds up steady-state force to overcome constant disturbances
    like gravity on a heavy payload, so a small ``ki`` (e.g. 5-20)
    lets the controller hold a pose at lower ``kp`` than it would
    need with pure PD.
    """
    return (
        _decode_scalar_gain(redis_client.get(KEYS.cartesian_task_position_kp)),
        _decode_scalar_gain(redis_client.get(KEYS.cartesian_task_position_kv)),
        _decode_scalar_gain(redis_client.get(KEYS.cartesian_task_position_ki)),
    )


def set_cart_position_gains(
    redis_client,
    *,
    kp: float | None = None,
    kv: float | None = None,
    ki: float | None = None,
) -> None:
    """Write ``kp`` / ``kv`` / ``ki`` to the cartesian task's position PID keys.

    Slots left at ``None`` are left untouched, so callers can bump
    one term without disturbing the others (e.g. enable ``ki`` for
    gravity comp while leaving ``kp`` / ``kv`` alone).
    """
    if kp is not None:
        redis_client.set(KEYS.cartesian_task_position_kp, _encode_scalar_gain(kp))
    if kv is not None:
        redis_client.set(KEYS.cartesian_task_position_kv, _encode_scalar_gain(kv))
    if ki is not None:
        redis_client.set(KEYS.cartesian_task_position_ki, _encode_scalar_gain(ki))


def read_cart_orientation_gains(
    redis_client,
) -> tuple[float | None, float | None]:
    """Return ``(kp, kv)`` for the cartesian task's orientation PD.

    ``None`` per slot if absent. Used to stiffen orientation tracking
    so a fixed tool-down pose is actually held during a visual servo
    (low orientation kp lets the EE slowly rotate as it translates).
    """
    return (
        _decode_scalar_gain(redis_client.get(KEYS.cartesian_task_orientation_kp)),
        _decode_scalar_gain(redis_client.get(KEYS.cartesian_task_orientation_kv)),
    )


def set_cart_orientation_gains(
    redis_client,
    *,
    kp: float | None = None,
    kv: float | None = None,
) -> None:
    """Write ``kp`` / ``kv`` to the cartesian task's orientation PD keys.

    Slots left at ``None`` are left untouched.
    """
    if kp is not None:
        redis_client.set(KEYS.cartesian_task_orientation_kp, _encode_scalar_gain(kp))
    if kv is not None:
        redis_client.set(KEYS.cartesian_task_orientation_kv, _encode_scalar_gain(kv))


def snapshot_cart_position_gains(
    redis_client,
    *,
    kp: bool = True,
    kv: bool = True,
    ki: bool = True,
) -> CartPositionGainSnapshot:
    """Capture current cart position gains so they can be restored later.

    The boolean flags control which slots are captured (so callers
    that only boost ``kp`` don't bother reading ``kv`` / ``ki``).
    Slots that were unreadable from Redis come back as ``None`` and
    will be skipped on restore — that's intentional, since writing
    ``None`` back would JSON-encode to ``"null"`` and confuse the
    C++ parser.
    """
    cur_kp, cur_kv, cur_ki = read_cart_position_gains(redis_client)
    return CartPositionGainSnapshot(
        kp=cur_kp if kp else None,
        kv=cur_kv if kv else None,
        ki=cur_ki if ki else None,
    )


def apply_cart_position_boost(
    redis_client,
    *,
    kp: float | None = None,
    kv: float | None = None,
    ki: float | None = None,
    label: str = "",
) -> CartPositionGainSnapshot:
    """Snapshot current gains, write the boosted values, return the snapshot.

    Companion to :func:`restore_cart_position_gains` for cases where
    the boost spans non-stack-shaped lifetimes (e.g. the cylinder
    controller's phase machine, where the boost is applied at one
    phase and restored at another). For straightforward
    ``with``-blocks use :func:`boosted_cart_position_gains` instead.

    Returns the pre-boost snapshot — pass it back to
    :func:`restore_cart_position_gains` when you're done. Slots left
    at ``None`` are not boosted AND not captured for restore (so they
    stay at whatever live value they were).
    """
    snapshot = snapshot_cart_position_gains(
        redis_client,
        kp=kp is not None,
        kv=kv is not None,
        ki=ki is not None,
    )
    set_cart_position_gains(redis_client, kp=kp, kv=kv, ki=ki)
    new_kp = kp if kp is not None else snapshot.kp
    new_kv = kv if kv is not None else snapshot.kv
    new_ki = ki if ki is not None else snapshot.ki
    tag = f" ({label})" if label else ""
    print(
        f"[gains]{tag} cart position kp/kv/ki: "
        f"({snapshot.kp}, {snapshot.kv}, {snapshot.ki}) -> "
        f"({new_kp}, {new_kv}, {new_ki})",
        flush=True,
    )
    return snapshot


def restore_cart_position_gains(
    redis_client,
    snapshot: CartPositionGainSnapshot,
    *,
    label: str = "",
) -> None:
    """Write the pre-boost gains from ``snapshot`` back to Redis.

    No-op for slots that were captured as ``None`` (means the slot
    was either not boosted or unreadable on snapshot). Safe to call
    multiple times — second call is just an extra Redis write of the
    same values.
    """
    set_cart_position_gains(
        redis_client, kp=snapshot.kp, kv=snapshot.kv, ki=snapshot.ki
    )
    tag = f" ({label})" if label else ""
    print(
        f"[gains]{tag} cart position kp/kv/ki restored to "
        f"({snapshot.kp}, {snapshot.kv}, {snapshot.ki})",
        flush=True,
    )


def read_joint_gains(
    redis_client,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(kp, kv, ki)`` for the joint task, or ``None`` per missing slot."""
    return (
        _decode_scalar_gain(redis_client.get(KEYS.joint_task_kp)),
        _decode_scalar_gain(redis_client.get(KEYS.joint_task_kv)),
        _decode_scalar_gain(redis_client.get(KEYS.joint_task_ki)),
    )


def set_joint_gains(
    redis_client,
    *,
    kp: float | None = None,
    kv: float | None = None,
    ki: float | None = None,
) -> None:
    """Write joint-task ``kp`` / ``kv`` / ``ki`` gains to Redis."""
    if kp is not None:
        redis_client.set(KEYS.joint_task_kp, _encode_scalar_gain(kp))
    if kv is not None:
        redis_client.set(KEYS.joint_task_kv, _encode_scalar_gain(kv))
    if ki is not None:
        redis_client.set(KEYS.joint_task_ki, _encode_scalar_gain(ki))


@contextmanager
def boosted_joint_gains(
    redis_client,
    *,
    kp: float | None = None,
    kv: float | None = None,
    ki: float | None = None,
    label: str = "",
) -> Iterator[None]:
    """Temporarily raise joint-task gains and restore them on exit."""
    if kp is None and kv is None and ki is None:
        yield
        return

    cur_kp, cur_kv, cur_ki = read_joint_gains(redis_client)
    snapshot = JointGainSnapshot(
        kp=cur_kp if kp is not None else None,
        kv=cur_kv if kv is not None else None,
        ki=cur_ki if ki is not None else None,
    )
    set_joint_gains(redis_client, kp=kp, kv=kv, ki=ki)
    tag = f" ({label})" if label else ""
    print(
        f"[gains]{tag} joint kp/kv/ki: "
        f"({snapshot.kp}, {snapshot.kv}, {snapshot.ki}) -> "
        f"({kp if kp is not None else cur_kp}, "
        f"{kv if kv is not None else cur_kv}, "
        f"{ki if ki is not None else cur_ki})",
        flush=True,
    )
    try:
        yield
    finally:
        set_joint_gains(
            redis_client, kp=snapshot.kp, kv=snapshot.kv, ki=snapshot.ki
        )
        print(
            f"[gains]{tag} joint kp/kv/ki restored to "
            f"({snapshot.kp}, {snapshot.kv}, {snapshot.ki})",
            flush=True,
        )


def read_otg_max_linear_velocity(redis_client) -> float | None:
    """Return the cartesian task's OTG linear-velocity cap (m/s), or ``None``.

    Stored as a BARE scalar string (the C++ side binds it to a ``double``
    parsed with ``std::stod`` — not the JSON-list format the gains use), so
    we parse it as a plain float rather than via :func:`_decode_scalar_gain`.
    """
    raw = redis_client.get(KEYS.cartesian_task_otg_max_linear_velocity)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def set_otg_max_linear_velocity(redis_client, value: float) -> None:
    """Write the cartesian task's OTG linear-velocity cap (m/s).

    Written as a bare scalar string (``"0.03"``); the receive-group decode on
    the C++ side does ``std::stod`` on it, which rejects a JSON list.
    """
    redis_client.set(
        KEYS.cartesian_task_otg_max_linear_velocity, str(float(value))
    )


def read_otg_max_angular_velocity(redis_client) -> float | None:
    """Return the cartesian task's OTG angular-velocity cap (rad/s), or ``None``.

    Bare scalar string, same format as the linear cap (see
    :func:`read_otg_max_linear_velocity`).
    """
    raw = redis_client.get(KEYS.cartesian_task_otg_max_angular_velocity)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = raw.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def set_otg_max_angular_velocity(redis_client, value: float) -> None:
    """Write the cartesian task's OTG angular-velocity cap (rad/s).

    Bare scalar string (``std::stod`` on the C++ side), like the linear cap.
    """
    redis_client.set(
        KEYS.cartesian_task_otg_max_angular_velocity, str(float(value))
    )


def read_otg_enabled(redis_client) -> bool | None:
    """Return whether the cartesian task's internal OTG is enabled, or ``None``.

    Stored as a bare ``"0"`` / ``"1"`` string in the controller's receive
    group (re-read every loop), so writes take effect without a restart.
    """
    raw = redis_client.get(KEYS.cartesian_task_otg_enabled)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = raw.strip()
    if not raw:
        return None
    return raw not in ("0", "0.0", "false", "False")


def set_otg_enabled(redis_client, enabled: bool) -> None:
    """Enable / disable the cartesian task's internal OTG.

    Written as a bare ``"1"`` / ``"0"`` string. Disabling the OTG makes the
    controller track streamed goal poses directly (no per-write trajectory
    re-plan), which is what we want while publishing a dense trajectory such
    as the scramble stir; re-enable it afterward for ordinary point-to-point
    moves.
    """
    redis_client.set(
        KEYS.cartesian_task_otg_enabled, "1" if enabled else "0"
    )


@contextmanager
def otg_disabled(redis_client) -> Iterator[None]:
    """Disable the cartesian OTG for the duration of the ``with`` block.

    Snapshots the current ``otg_enabled`` flag on entry, forces it off, and
    restores the prior value on exit (defaulting to re-enabled if the flag
    couldn't be read). Use around streamed-trajectory phases so the arm
    tracks the published setpoints directly instead of re-planning on every
    10 Hz write.
    """
    prior = read_otg_enabled(redis_client)
    set_otg_enabled(redis_client, False)
    print("[gains] cartesian OTG disabled for streamed trajectory.", flush=True)
    try:
        yield
    finally:
        restore_to = True if prior is None else prior
        set_otg_enabled(redis_client, restore_to)
        print(
            f"[gains] cartesian OTG restored to enabled={restore_to}.",
            flush=True,
        )


@dataclass(frozen=True)
class PreciseGraspSnapshot:
    """Pre-engage values captured by :func:`apply_precise_grasp`.

    Pass back to :func:`restore_precise_grasp` to revert. Slots that were
    unreadable on snapshot stay ``None`` and are skipped on restore.
    """

    position_kp: float | None
    orientation_kp: float | None
    max_linear_velocity: float | None
    max_angular_velocity: float | None = None


def apply_precise_grasp(
    redis_client,
    *,
    max_linear_velocity: float,
    position_kp: float,
    orientation_kp: float,
    max_angular_velocity: float | None = None,
    label: str = "precise grasp",
) -> PreciseGraspSnapshot:
    """Slow + stiffen the cart task for a precise final approach; return snapshot.

    Snapshots the live position kp, orientation kp, and OTG linear/angular
    velocity caps, then writes the precise values. Restore with
    :func:`restore_precise_grasp` (e.g. after the gripper has closed) so the
    rest of the routine runs at the normal stiffness / speed.

    ``max_angular_velocity`` (rad/s) clamps the OTG angular speed so wrist
    re-orientations track accurately during the precise move. ``None`` (the
    default) leaves the live angular cap untouched.
    """
    snapshot = PreciseGraspSnapshot(
        position_kp=read_cart_position_gains(redis_client)[0],
        orientation_kp=read_cart_orientation_gains(redis_client)[0],
        max_linear_velocity=read_otg_max_linear_velocity(redis_client),
        max_angular_velocity=(
            read_otg_max_angular_velocity(redis_client)
            if max_angular_velocity is not None
            else None
        ),
    )
    set_cart_position_gains(redis_client, kp=position_kp)
    set_cart_orientation_gains(redis_client, kp=orientation_kp)
    set_otg_max_linear_velocity(redis_client, max_linear_velocity)
    if max_angular_velocity is not None:
        set_otg_max_angular_velocity(redis_client, max_angular_velocity)
    tag = f" ({label})" if label else ""
    ang = (
        f", max_ang_vel {snapshot.max_angular_velocity}->{max_angular_velocity} rad/s"
        if max_angular_velocity is not None
        else ""
    )
    print(
        f"[gains]{tag} precise grasp ENGAGE: "
        f"pos_kp {snapshot.position_kp}->{position_kp}, "
        f"ori_kp {snapshot.orientation_kp}->{orientation_kp}, "
        f"max_lin_vel {snapshot.max_linear_velocity}->{max_linear_velocity} m/s"
        f"{ang}",
        flush=True,
    )
    return snapshot


def restore_precise_grasp(
    redis_client,
    snapshot: PreciseGraspSnapshot,
    *,
    label: str = "precise grasp",
) -> None:
    """Revert the position kp / orientation kp / OTG velocity caps from ``snapshot``.

    No-op per slot that was captured as ``None``. Safe to call more than once.
    """
    if snapshot.position_kp is not None:
        set_cart_position_gains(redis_client, kp=snapshot.position_kp)
    if snapshot.orientation_kp is not None:
        set_cart_orientation_gains(redis_client, kp=snapshot.orientation_kp)
    if snapshot.max_linear_velocity is not None:
        set_otg_max_linear_velocity(redis_client, snapshot.max_linear_velocity)
    if snapshot.max_angular_velocity is not None:
        set_otg_max_angular_velocity(redis_client, snapshot.max_angular_velocity)
    tag = f" ({label})" if label else ""
    print(
        f"[gains]{tag} precise grasp RESTORE: "
        f"pos_kp={snapshot.position_kp}, ori_kp={snapshot.orientation_kp}, "
        f"max_lin_vel={snapshot.max_linear_velocity} m/s, "
        f"max_ang_vel={snapshot.max_angular_velocity} rad/s",
        flush=True,
    )


@dataclass(frozen=True)
class CartOrientationGainSnapshot:
    """Pre-boost orientation gains for :func:`restore_cart_orientation_gains`."""

    kp: float | None
    kv: float | None


def snapshot_cart_orientation_gains(redis_client) -> CartOrientationGainSnapshot:
    kp, kv = read_cart_orientation_gains(redis_client)
    return CartOrientationGainSnapshot(kp=kp, kv=kv)


def restore_cart_orientation_gains(
    redis_client,
    snapshot: CartOrientationGainSnapshot,
    *,
    label: str = "",
) -> None:
    set_cart_orientation_gains(redis_client, kp=snapshot.kp, kv=snapshot.kv)
    tag = f" ({label})" if label else ""
    print(
        f"[gains]{tag} cart orientation kp/kv restored to "
        f"({snapshot.kp}, {snapshot.kv})",
        flush=True,
    )


@contextmanager
def boosted_cart_servo_gains(
    redis_client,
    *,
    position_kp: float | None = None,
    position_kv: float | None = None,
    orientation_kp: float | None = None,
    orientation_kv: float | None = None,
    label: str = "visual servo",
) -> Iterator[None]:
    """Raise cartesian position + orientation PID for a visual servo run.

    Stiffer position tracking lets each 0.5 s chase step actually land;
    stiffer orientation tracking holds the tool-down pose so the EE does
    not slowly rotate while translating (which breaks the fixed-axis
    image projection). Restores both on exit.
    """
    pos_snap = apply_cart_position_boost(
        redis_client, kp=position_kp, kv=position_kv, label=label,
    )
    ori_snap = snapshot_cart_orientation_gains(redis_client)
    set_cart_orientation_gains(
        redis_client, kp=orientation_kp, kv=orientation_kv,
    )
    if orientation_kp is not None or orientation_kv is not None:
        new_ok = orientation_kp if orientation_kp is not None else ori_snap.kp
        new_ov = orientation_kv if orientation_kv is not None else ori_snap.kv
        print(
            f"[gains] ({label}) cart orientation kp/kv: "
            f"({ori_snap.kp}, {ori_snap.kv}) -> ({new_ok}, {new_ov})",
            flush=True,
        )
    try:
        yield
    finally:
        restore_cart_position_gains(redis_client, pos_snap, label=label)
        restore_cart_orientation_gains(redis_client, ori_snap, label=label)


@contextmanager
def boosted_cart_position_gains(
    redis_client,
    *,
    kp: float | None = None,
    kv: float | None = None,
    ki: float | None = None,
    label: str = "",
) -> Iterator[None]:
    """Temporarily raise the cartesian position PID; restore on exit.

    Slots left at ``None`` are skipped (not boosted, not restored).
    Restore runs in a ``finally``, so the original gains are always
    put back — including on ``KeyboardInterrupt`` or controller
    exceptions — assuming OpenSai is still running. If Redis is down
    when restore fires, the boosted value will stick until something
    else writes the key.

    Use:

        with boosted_cart_position_gains(ctx.redis, kp=150, ki=10, label="sauce held"):
            arm.move_to(ctx, carry_pos, carry_ori, ...)

    For non-stack-shaped lifetimes (boost applied in one function,
    restored in another) use the
    :func:`apply_cart_position_boost` / :func:`restore_cart_position_gains`
    pair directly.
    """
    if kp is None and kv is None and ki is None:
        yield
        return

    snapshot = apply_cart_position_boost(
        redis_client, kp=kp, kv=kv, ki=ki, label=label
    )
    try:
        yield
    finally:
        restore_cart_position_gains(redis_client, snapshot, label=label)
