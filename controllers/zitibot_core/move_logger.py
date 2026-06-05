"""Per-move EE position logger + plotter.

Records (t, x, y, z) samples for each ``arm.move_to`` call while the
controller is waiting for convergence and writes a 3-axis position-vs-time
PNG to ``logs/graphs/<controller>_NNNN/move_MMM_<label>.png`` where:

* ``<controller>`` is derived from ``sys.argv[0]`` (e.g. ``bowl_pour_controller``).
* ``NNNN`` auto-increments per process invocation so each run gets its own
  folder; we don't clobber the previous run's plots.
* ``MMM`` is the per-run sequential move counter.

The plot draws actual EE position as a solid line per axis and the
goal as a dotted horizontal line, so it's obvious at a glance whether
the move overshot, undershot, settled slowly, or oscillated.

Base moves (``base.go_to_pose`` phases) get analogous PNGs with X / Y /
yaw subplots showing both the hb (encoder) and Opti (mocap) frames
simultaneously, so it's obvious when hb and Opti disagree (the usual
cause of "hb says we arrived but the cart is in the wrong spot").

Enabled by passing ``--log`` to a controller; off by default to keep
"normal" runs fast / quiet.
"""

from __future__ import annotations

import atexit
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


_AXIS_LABELS = ("x", "y", "z")
_AXIS_COLORS = ("#1f77b4", "#2ca02c", "#d62728")
# Frame colors for base plots: hb (encoder) is blue, Opti (mocap) is orange.
# Goal lines reuse the same hue dotted, so blue/orange tells you which
# frame is being plotted at a glance.
_BASE_FRAME_COLORS = {
    "hb": "#1f77b4",
    "opti": "#ff7f0e",
}
_GRAPH_ROOT = Path(__file__).resolve().parent.parent.parent / "logs" / "graphs"


def _slugify(label: str, max_len: int = 60) -> str:
    """File-safe ASCII slug; keeps alnum/underscore/hyphen, collapses runs."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    s = re.sub(r"_+", "_", s).strip("_.-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_.-")
    return s or "move"


def _controller_name() -> str:
    argv0 = sys.argv[0] if sys.argv else "controller"
    return Path(argv0).stem or "controller"


def _next_run_dir(controller: str) -> Path:
    """Return the next free ``<controller>_NNNN`` directory under graphs/.

    Scans existing siblings to find the max ``NNNN`` already used and
    increments by one. Runs from completely different controllers get
    independent counters so each controller's plots are easy to find.
    """
    _GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = f"{controller}_"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_idx = 0
    for child in _GRAPH_ROOT.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except ValueError:
                pass
    next_idx = max_idx + 1
    run_dir = _GRAPH_ROOT / f"{prefix}{next_idx:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass
class _MoveSamples:
    label: str
    goal: np.ndarray
    tol_m: float
    timeout_s: float
    t0: float
    ts: list[float] = field(default_factory=list)
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    zs: list[float] = field(default_factory=list)


@dataclass
class _BaseSamples:
    """One ``base.go_to_pose`` phase worth of samples for plotting.

    We record the live hb (encoder) pose, the hb_goal that the navigator
    is publishing on every tick, the live Opti marker pose (when
    available), and the fixed Opti target. Plotting all four lines per
    axis makes it obvious when hb and Opti disagree, which is the usual
    cause of "hb says we arrived but the cart is in the wrong spot".

    Yaw is stored in radians during sampling and converted to degrees at
    plot time so the saved images match how humans read base headings.
    """

    label: str
    opti_target_xy: np.ndarray  # shape (2,)
    opti_target_yaw_rad: float | None
    tol_m: float
    tol_yaw_rad: float
    require_yaw: bool
    t0: float
    ts: list[float] = field(default_factory=list)
    hb_xs: list[float] = field(default_factory=list)
    hb_ys: list[float] = field(default_factory=list)
    hb_yaws: list[float] = field(default_factory=list)
    hb_goal_xs: list[float] = field(default_factory=list)
    hb_goal_ys: list[float] = field(default_factory=list)
    hb_goal_yaws: list[float] = field(default_factory=list)
    opti_xs: list[float] = field(default_factory=list)
    opti_ys: list[float] = field(default_factory=list)
    opti_yaws: list[float] = field(default_factory=list)


class MoveLogger:
    """Owns one run's output directory and the in-flight move samples.

    A single ``MoveLogger`` is attached to ``TaskContext.move_logger`` and
    reused for every ``arm.move_to`` call in that run. Each call goes
    ``begin_move`` → repeated ``sample`` → ``end_move`` and produces
    exactly one PNG in the run directory.
    """

    def __init__(self, run_dir: Path | None = None) -> None:
        self.run_dir: Path = run_dir if run_dir is not None else _next_run_dir(_controller_name())
        self.move_idx: int = 0
        self._current: _MoveSamples | None = None
        self._current_base: _BaseSamples | None = None
        # Safety net: if the process is exiting (Ctrl+C, sys.exit, normal
        # finish, etc.) and a move is still in-flight, flush it to disk so
        # we never silently lose the partial trajectory. The normal
        # end_move/end_base_move call sites already save on the happy path
        # and on TimeoutError; this catches the rare case where Ctrl+C
        # interrupts during the logger itself or where a controller
        # raises before reaching its end-of-move call.
        atexit.register(self._flush_in_flight)
        print(f"[move_logger] run dir: {self.run_dir}", flush=True)

    @property
    def active(self) -> bool:
        return self._current is not None

    @property
    def base_active(self) -> bool:
        return self._current_base is not None

    def begin_move(
        self,
        goal_pos: np.ndarray,
        label: str,
        *,
        tol_m: float,
        timeout_s: float,
    ) -> None:
        self.move_idx += 1
        self._current = _MoveSamples(
            label=label,
            goal=np.asarray(goal_pos, dtype=np.float64).reshape(3).copy(),
            tol_m=float(tol_m),
            timeout_s=float(timeout_s),
            t0=time.perf_counter(),
        )

    def sample(self, pos: np.ndarray) -> None:
        cur = self._current
        if cur is None:
            return
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        cur.ts.append(time.perf_counter() - cur.t0)
        cur.xs.append(float(p[0]))
        cur.ys.append(float(p[1]))
        cur.zs.append(float(p[2]))

    def end_move(
        self,
        *,
        status: str = "ok",
        final_err_m: float | None = None,
    ) -> Path | None:
        cur = self._current
        self._current = None
        if cur is None:
            return None
        try:
            return self._save_plot(cur, status=status, final_err_m=final_err_m)
        except Exception as e:
            print(f"[move_logger] failed to save plot: {e}", flush=True)
            return None

    # ------------------------------------------------------------------
    # Base move recording
    # ------------------------------------------------------------------

    def begin_base_move(
        self,
        *,
        label: str,
        opti_target_xy: np.ndarray | tuple[float, float] | None,
        opti_target_yaw_rad: float | None,
        tol_m: float,
        tol_yaw_rad: float,
        require_yaw: bool,
    ) -> None:
        """Start recording one ``base.go_to_pose`` phase.

        Each phase (holonomic / rotate / translate) gets its own plot.
        ``opti_target_xy`` is logged as ``[nan, nan]`` if unknown so the
        plot can still render the actual traces without crashing.
        """
        self.move_idx += 1
        if opti_target_xy is None:
            tgt = np.array([np.nan, np.nan], dtype=np.float64)
        else:
            tgt = np.asarray(opti_target_xy, dtype=np.float64).reshape(2).copy()
        self._current_base = _BaseSamples(
            label=label,
            opti_target_xy=tgt,
            opti_target_yaw_rad=(
                None if opti_target_yaw_rad is None else float(opti_target_yaw_rad)
            ),
            tol_m=float(tol_m),
            tol_yaw_rad=float(tol_yaw_rad),
            require_yaw=bool(require_yaw),
            t0=time.perf_counter(),
        )

    def sample_base(
        self,
        *,
        hb_xyyaw: np.ndarray | tuple[float, float, float] | None,
        hb_goal_xyyaw: np.ndarray | tuple[float, float, float] | None,
        opti_xy: np.ndarray | tuple[float, float] | None,
        opti_yaw_rad: float | None,
        **_: object,
    ) -> None:
        """Record one tick. NaNs fill in any missing data so axes line up.

        ``opti_nav`` passes a richer telemetry payload than the plotter
        currently needs (timestamps, targets, tolerances). Accept and
        ignore extra keyword arguments so that payload can evolve without
        breaking logging during a robot run.
        """
        cur = self._current_base
        if cur is None:
            return

        def _v3(v):
            if v is None:
                return (float("nan"),) * 3
            arr = np.asarray(v, dtype=np.float64).reshape(3)
            return float(arr[0]), float(arr[1]), float(arr[2])

        def _v2(v):
            if v is None:
                return (float("nan"),) * 2
            arr = np.asarray(v, dtype=np.float64).reshape(2)
            return float(arr[0]), float(arr[1])

        hb_x, hb_y, hb_yaw = _v3(hb_xyyaw)
        gx, gy, gyaw = _v3(hb_goal_xyyaw)
        ox, oy = _v2(opti_xy)
        cur.ts.append(time.perf_counter() - cur.t0)
        cur.hb_xs.append(hb_x)
        cur.hb_ys.append(hb_y)
        cur.hb_yaws.append(hb_yaw)
        cur.hb_goal_xs.append(gx)
        cur.hb_goal_ys.append(gy)
        cur.hb_goal_yaws.append(gyaw)
        cur.opti_xs.append(ox)
        cur.opti_ys.append(oy)
        cur.opti_yaws.append(
            float("nan") if opti_yaw_rad is None else float(opti_yaw_rad)
        )

    def end_base_move(
        self,
        *,
        status: str = "ok",
        final_xy_err_m: float | None = None,
        final_yaw_err_rad: float | None = None,
    ) -> Path | None:
        cur = self._current_base
        self._current_base = None
        if cur is None:
            return None
        try:
            return self._save_base_plot(
                cur,
                status=status,
                final_xy_err_m=final_xy_err_m,
                final_yaw_err_rad=final_yaw_err_rad,
            )
        except Exception as e:
            print(f"[move_logger] failed to save base plot: {e}", flush=True)
            return None

    def _flush_in_flight(self) -> None:
        """atexit hook: save any move that's still mid-flight on shutdown.

        Called automatically when the Python process exits — including
        after an unhandled ``KeyboardInterrupt`` propagates past the
        controller's ``main()``. Both ``end_move`` and ``end_base_move``
        no-op when their state is already None, so this is safe to call
        unconditionally and won't double-save moves that ended cleanly.
        """
        try:
            if self._current is not None:
                self.end_move(status="aborted")
        except Exception as e:  # noqa: BLE001 - shutdown must never raise
            print(f"[move_logger] flush(arm) failed: {e}", flush=True)
        try:
            if self._current_base is not None:
                self.end_base_move(status="aborted")
        except Exception as e:  # noqa: BLE001 - shutdown must never raise
            print(f"[move_logger] flush(base) failed: {e}", flush=True)

    def _save_plot(
        self,
        cur: _MoveSamples,
        *,
        status: str,
        final_err_m: float | None,
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = np.asarray(cur.ts, dtype=np.float64)
        xs = np.asarray(cur.xs, dtype=np.float64)
        ys = np.asarray(cur.ys, dtype=np.float64)
        zs = np.asarray(cur.zs, dtype=np.float64)
        per_axis = (xs, ys, zs)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))

        title_bits = [f"move {self.move_idx:03d}: {cur.label}"]
        if status != "ok":
            title_bits.append(f"[{status}]")
        if final_err_m is not None:
            title_bits.append(f"final_err={final_err_m * 100:.2f} cm")
        title_bits.append(f"tol={cur.tol_m * 100:.1f} cm")
        title_bits.append(f"timeout={cur.timeout_s:.1f} s")
        fig.suptitle("  ".join(title_bits))

        for axis_idx, (ax, samples, label, color) in enumerate(
            zip(axes, per_axis, _AXIS_LABELS, _AXIS_COLORS)
        ):
            goal_val = float(cur.goal[axis_idx])
            if samples.size > 0:
                ax.plot(ts, samples, "-", color=color, linewidth=1.6, label=f"{label} actual")
                ax.axhline(
                    goal_val, color=color, linewidth=1.2, linestyle=":", label=f"{label} goal"
                )
                ax.fill_between(
                    ts,
                    goal_val - cur.tol_m,
                    goal_val + cur.tol_m,
                    color=color,
                    alpha=0.08,
                    label="±tol",
                )
            else:
                ax.axhline(goal_val, color=color, linewidth=1.2, linestyle=":", label="goal")
                ax.text(
                    0.5,
                    0.5,
                    "no samples captured",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="gray",
                )
            ax.set_ylabel(f"{label} (m)")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        axes[-1].set_xlabel("t since publish_goal (s)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        fname = f"move_{self.move_idx:03d}_{_slugify(cur.label)}.png"
        out_path = self.run_dir / fname
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[move_logger] saved {out_path}", flush=True)
        return out_path

    def _save_base_plot(
        self,
        cur: _BaseSamples,
        *,
        status: str,
        final_xy_err_m: float | None,
        final_yaw_err_rad: float | None,
    ) -> Path:
        """Render one base-phase PNG: X / Y / yaw with hb + Opti overlaid.

        Per-axis lines:

        * ``hb actual``  - solid blue   (encoder-frame pose)
        * ``hb goal``    - dotted blue  (the goal we publish to the driver)
        * ``opti actual`` - solid orange (mocap marker pose)
        * ``opti target`` - dotted orange (the lab-frame goal)

        The ±tolerance band is drawn around the Opti target since that's
        what ``run_replan_loop`` actually uses for success. Yaw lines are
        plotted in degrees and unwrapped so a wrap-around looks like a
        straight ramp instead of a 360° jump.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = np.asarray(cur.ts, dtype=np.float64)
        hb_xs = np.asarray(cur.hb_xs, dtype=np.float64)
        hb_ys = np.asarray(cur.hb_ys, dtype=np.float64)
        hb_yaws = np.asarray(cur.hb_yaws, dtype=np.float64)
        hb_goal_xs = np.asarray(cur.hb_goal_xs, dtype=np.float64)
        hb_goal_ys = np.asarray(cur.hb_goal_ys, dtype=np.float64)
        hb_goal_yaws = np.asarray(cur.hb_goal_yaws, dtype=np.float64)
        opti_xs = np.asarray(cur.opti_xs, dtype=np.float64)
        opti_ys = np.asarray(cur.opti_ys, dtype=np.float64)
        opti_yaws = np.asarray(cur.opti_yaws, dtype=np.float64)

        # Unwrap yaw traces independently so each line stays continuous
        # across ±π wraps. NaNs (missing data) survive unwrap untouched.
        def _unwrap_safe(arr_rad: np.ndarray) -> np.ndarray:
            if arr_rad.size == 0:
                return arr_rad
            mask = np.isfinite(arr_rad)
            out = arr_rad.copy()
            if mask.any():
                out[mask] = np.unwrap(arr_rad[mask])
            return np.degrees(out)

        hb_yaw_deg = _unwrap_safe(hb_yaws)
        hb_goal_yaw_deg = _unwrap_safe(hb_goal_yaws)
        opti_yaw_deg = _unwrap_safe(opti_yaws)

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.5, 8))

        title_bits = [f"base move {self.move_idx:03d}: {cur.label}"]
        if status != "ok":
            title_bits.append(f"[{status}]")
        if final_xy_err_m is not None and np.isfinite(final_xy_err_m):
            title_bits.append(f"final_xy_err={final_xy_err_m * 100:.2f} cm")
        if final_yaw_err_rad is not None and np.isfinite(final_yaw_err_rad):
            title_bits.append(f"final_yaw_err={math.degrees(final_yaw_err_rad):.2f}°")
        title_bits.append(f"tol={cur.tol_m * 100:.1f} cm / {math.degrees(cur.tol_yaw_rad):.1f}°")
        if not cur.require_yaw:
            title_bits.append("(yaw not gated)")
        fig.suptitle("  ".join(title_bits))

        hb_color = _BASE_FRAME_COLORS["hb"]
        opti_color = _BASE_FRAME_COLORS["opti"]

        # X axis subplot
        ax_x = axes[0]
        if hb_xs.size > 0:
            ax_x.plot(ts, hb_xs, "-", color=hb_color, linewidth=1.6, label="hb actual")
            ax_x.plot(ts, hb_goal_xs, ":", color=hb_color, linewidth=1.2, label="hb goal")
            if np.isfinite(opti_xs).any():
                ax_x.plot(ts, opti_xs, "-", color=opti_color, linewidth=1.6, label="opti actual")
            tgt_x = float(cur.opti_target_xy[0])
            if np.isfinite(tgt_x):
                ax_x.axhline(
                    tgt_x, color=opti_color, linewidth=1.2, linestyle=":", label="opti target"
                )
                ax_x.fill_between(
                    ts,
                    tgt_x - cur.tol_m,
                    tgt_x + cur.tol_m,
                    color=opti_color,
                    alpha=0.08,
                    label="±tol",
                )
        else:
            ax_x.text(
                0.5, 0.5, "no samples captured",
                ha="center", va="center", transform=ax_x.transAxes, color="gray",
            )
        ax_x.set_ylabel("x (m)")
        ax_x.grid(True, alpha=0.3)
        ax_x.legend(loc="best", fontsize=8, ncol=2)

        # Y axis subplot
        ax_y = axes[1]
        if hb_ys.size > 0:
            ax_y.plot(ts, hb_ys, "-", color=hb_color, linewidth=1.6, label="hb actual")
            ax_y.plot(ts, hb_goal_ys, ":", color=hb_color, linewidth=1.2, label="hb goal")
            if np.isfinite(opti_ys).any():
                ax_y.plot(ts, opti_ys, "-", color=opti_color, linewidth=1.6, label="opti actual")
            tgt_y = float(cur.opti_target_xy[1])
            if np.isfinite(tgt_y):
                ax_y.axhline(
                    tgt_y, color=opti_color, linewidth=1.2, linestyle=":", label="opti target"
                )
                ax_y.fill_between(
                    ts,
                    tgt_y - cur.tol_m,
                    tgt_y + cur.tol_m,
                    color=opti_color,
                    alpha=0.08,
                    label="±tol",
                )
        ax_y.set_ylabel("y (m)")
        ax_y.grid(True, alpha=0.3)
        ax_y.legend(loc="best", fontsize=8, ncol=2)

        # Yaw subplot (degrees)
        ax_yaw = axes[2]
        if hb_yaw_deg.size > 0:
            ax_yaw.plot(ts, hb_yaw_deg, "-", color=hb_color, linewidth=1.6, label="hb actual")
            ax_yaw.plot(
                ts, hb_goal_yaw_deg, ":", color=hb_color, linewidth=1.2, label="hb goal"
            )
            if np.isfinite(opti_yaw_deg).any():
                ax_yaw.plot(
                    ts, opti_yaw_deg, "-", color=opti_color, linewidth=1.6, label="opti actual"
                )
            if cur.opti_target_yaw_rad is not None and np.isfinite(cur.opti_target_yaw_rad):
                tgt_yaw_deg = math.degrees(cur.opti_target_yaw_rad)
                # Snap target into the same revolution as the (unwrapped)
                # opti / hb traces so the dotted line lands near the data
                # instead of being one full turn away.
                ref_deg = None
                for arr in (opti_yaw_deg, hb_yaw_deg):
                    if arr.size > 0 and np.isfinite(arr).any():
                        ref_deg = float(arr[np.isfinite(arr)][-1])
                        break
                if ref_deg is not None:
                    diff = ((tgt_yaw_deg - ref_deg + 180.0) % 360.0) - 180.0
                    tgt_yaw_deg = ref_deg + diff
                ax_yaw.axhline(
                    tgt_yaw_deg,
                    color=opti_color,
                    linewidth=1.2,
                    linestyle=":",
                    label="opti target",
                )
                tol_deg = math.degrees(cur.tol_yaw_rad)
                ax_yaw.fill_between(
                    ts,
                    tgt_yaw_deg - tol_deg,
                    tgt_yaw_deg + tol_deg,
                    color=opti_color,
                    alpha=0.08,
                    label="±tol",
                )
        ax_yaw.set_ylabel("yaw (deg)")
        ax_yaw.grid(True, alpha=0.3)
        ax_yaw.legend(loc="best", fontsize=8, ncol=2)

        axes[-1].set_xlabel("t since publish_goal (s)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        fname = f"move_{self.move_idx:03d}_base_{_slugify(cur.label)}.png"
        out_path = self.run_dir / fname
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[move_logger] saved {out_path}", flush=True)
        return out_path


def maybe_make_logger(enabled: bool) -> MoveLogger | None:
    """Factory used by :func:`make_context`; returns ``None`` when off."""
    if not enabled:
        return None
    return MoveLogger()
