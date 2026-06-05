"""Image-based visual servoing (template matching) for precise XY alignment.

Drives tracked image features toward a target pixel (default: camera principal
point) using incremental cartesian goals, bypassing unreliable hand-eye
translation for the final approach.

Features are seeded from Gemini bounding boxes: each box crop becomes a
BGR color template that is re-located every frame with normalized
cross-correlation (``cv2.matchTemplate`` on all channels). The midpoint of
the matched box centers is the servoed feature.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from zitibot_core import arm
from zitibot_core.constants import T_FLANGE_CAMERA
from zitibot_core.context import TaskContext
from zitibot_core.gains import boosted_cart_servo_gains


# Template-matching parameters (RealSense 640x480-ish streams).
_TM_METHOD = cv2.TM_CCOEFF_NORMED
# Search window margin (px) added around each template's last known extent.
_TM_SEARCH_MARGIN_PX = 40
# Abort the servo if a template's best match score drops below this (0–1).
_TM_MIN_SCORE = 0.30

# RGB-only servo gain: robot meters commanded per pixel of image-axis error.
# This is deliberately NOT derived from RealSense depth. Tune this like a
# controller gain: higher moves faster, lower is gentler.
_SERVO_M_PER_PX = 0.00035

# Cartesian PID during servo (defaults from zitibot_panda.xml kp/kv=100/20).
# Position gains are back at the defaults: the +X runaway was the goal
# re-referencing bug (see cmd_ref in the servo loop), not stiffness, so the
# stock 100/20 position PD tracks the command reference fine. Orientation is
# kept boosted (140/28) so the tool-down pose is held firmly and the EE does
# not rotate while translating (which would break the fixed-axis projection).
_SERVO_CART_KP = 100.0
_SERVO_CART_KV = 20.0
_SERVO_ORI_KP = 140.0
_SERVO_ORI_KV = 28.0


def _prepare_save_dir(save_dir: str | Path) -> Path:
    """Create ``save_dir`` (clearing any existing servo PNGs) and return it."""
    d = Path(save_dir).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    for pat in ("servo_*.png", "servo_run.*"):
        for old in d.glob(pat):
            try:
                old.unlink()
            except OSError:
                pass
    return d


def _write_servo_gif(out_dir: Path, frames: list[Path], fps: float) -> Path | None:
    """Write an animated GIF from frames; the most portable output (opens
    in any image viewer/browser, unlike OpenCV's mp4v/avc1 which need an
    ffmpeg encoder this build lacks)."""
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # noqa: BLE001
        print(f"[visual_servo] imageio unavailable, skipping GIF: {exc}")
        return None
    rgb = []
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not rgb:
        return None
    gif_path = out_dir / "servo_run.gif"
    imageio.mimsave(str(gif_path), rgb, duration=1.0 / max(fps, 0.1), loop=0)
    print(
        f"[visual_servo] wrote {len(rgb)} frames -> {gif_path} @ {fps:g} fps"
    )
    return gif_path


def _write_servo_mp4(out_dir: Path, frames: list[Path], fps: float) -> Path | None:
    """Best-effort OpenCV video. mp4v/avc1 depend on an ffmpeg backend conda
    builds often lack; fall back through codec/container combos and keep the
    first that yields a non-trivial file."""
    first = cv2.imread(str(frames[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    for fourcc_str, ext in (("mp4v", ".mp4"), ("MJPG", ".avi"),
                            ("XVID", ".avi")):
        video_path = out_dir / f"servo_run{ext}"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*fourcc_str),
            float(fps), (w, h),
        )
        if not writer.isOpened():
            writer.release()
            continue
        written = 0
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
            written += 1
        writer.release()
        if written > 0 and video_path.exists() and video_path.stat().st_size > 2048:
            print(
                f"[visual_servo] wrote {written} frames -> {video_path} "
                f"({fourcc_str}) @ {fps:g} fps"
            )
            return video_path
        try:
            video_path.unlink()
        except OSError:
            pass
    print("[visual_servo] OpenCV could not encode a portable mp4/avi")
    return None


def _write_servo_video(out_dir: Path, fps: float = 4.0) -> Path | None:
    """Stitch all ``servo_*.png`` frames (in capture order) into a watchable
    animation. Writes a portable GIF (primary) plus a best-effort mp4/avi."""
    frames = sorted(out_dir.glob("servo_*.png"), key=lambda p: p.stat().st_mtime)
    if not frames:
        print("[visual_servo] no frames to stitch into a video")
        return None
    gif_path = _write_servo_gif(out_dir, frames, fps)
    mp4_path = _write_servo_mp4(out_dir, frames, fps)
    return gif_path or mp4_path


def _annotate_servo_frame(
    color_bgr: np.ndarray,
    tracked: Sequence[tuple[int, int]],
    midpoint: tuple[float, float],
    target: tuple[float, float],
    *,
    iter_idx: int,
    err_norm: float,
    goal_pos: np.ndarray | None,
    converged: bool,
    boxes: Sequence[tuple[int, int, int, int]] | None = None,
    px_tol: float | None = None,
    bound_axes: Sequence[tuple[tuple[float, float], tuple[int, int, int], str]]
    | None = None,
    note: str | None = None,
    failed: bool = False,
) -> np.ndarray:
    """Draw boxes, centers, midpoint, target, world-axis tolerance bounds, banner."""
    img = color_bgr.copy()
    mu, mv = int(round(midpoint[0])), int(round(midpoint[1]))
    tu, tv = int(round(target[0])), int(round(target[1]))
    box_color = (0, 0, 255) if failed else (0, 255, 0)  # red when lost track

    # Matched template boxes (red if track was lost this frame).
    if boxes is not None:
        for bx in boxes:
            cv2.rectangle(img, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])),
                          box_color, 2 if failed else 1)

    # World-axis tolerance bounds, rotated into the image. For each world axis
    # we draw two parallel lines at +/- px_tol along that axis's image
    # projection; the band between a pair is the in-tolerance range for it.
    if px_tol is not None and bound_axes:
        h, w = img.shape[:2]
        span = int(2 * (h + w))
        for (pu, pv), color, lbl in bound_axes:
            perp = (-pv, pu)
            for sgn in (1.0, -1.0):
                cx = tu + sgn * px_tol * pu
                cy = tv + sgn * px_tol * pv
                a = (int(round(cx - span * perp[0])), int(round(cy - span * perp[1])))
                b = (int(round(cx + span * perp[0])), int(round(cy + span * perp[1])))
                cv2.line(img, a, b, color, 1, cv2.LINE_AA)
            lx = int(round(tu + (px_tol + 16) * pu))
            ly = int(round(tv + (px_tol + 16) * pv))
            cv2.putText(img, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)

    # Target (principal point): red crosshair.
    cv2.drawMarker(img, (tu, tv), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(img, (tu, tv), 10, (0, 0, 255), 1)

    # Error vector midpoint -> target.
    cv2.line(img, (mu, mv), (tu, tv), (255, 255, 0), 1)

    # Tracked feature centers: circles + index (red if lost track).
    for i, (u, v) in enumerate(tracked):
        cv2.circle(img, (int(u), int(v)), 6, box_color, 2)
        cv2.putText(
            img, str(i), (int(u) + 8, int(v) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA,
        )

    # Midpoint: yellow filled dot.
    cv2.circle(img, (mu, mv), 5, (0, 255, 255), -1)

    banner = [
        f"iter={iter_idx}  |err|={err_norm:.1f}px  BGR"
        + ("  CONVERGED" if converged else ""),
    ]
    if goal_pos is not None:
        banner.append(
            f"goal=[{goal_pos[0]:+.4f}, {goal_pos[1]:+.4f}, {goal_pos[2]:+.4f}]"
        )
    if note:
        banner.append(note)
    y = 22
    for line in banner:
        cv2.putText(
            img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += 26

    if failed:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        text = "FAILED - LOST TRACK"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        org = ((w - tw) // 2, h // 2)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return img


def _midpoint_pixels(pixels: Sequence[tuple[int, int]]) -> tuple[float, float]:
    pts = np.asarray(pixels, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 1:
        raise ValueError("need at least one pixel to form a midpoint")
    m = pts.mean(axis=0)
    return float(m[0]), float(m[1])


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _crop_template(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop the BGR template patch for ``box`` (clamped to image)."""
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(x0 + 1, min(w, int(x1)))
    y1 = max(y0 + 1, min(h, int(y1)))
    return frame_bgr[y0:y1, x0:x1].copy()


def _match_template(
    frame_bgr: np.ndarray,
    template: np.ndarray,
    last_center: tuple[float, float],
    search_margin: int,
) -> tuple[tuple[float, float], float, tuple[int, int, int, int]]:
    """Locate BGR ``template`` near ``last_center``; return (center, score, box).

    Searches a window centered on ``last_center`` (template extent +
    ``search_margin`` on each side). Falls back to the full frame when the
    window would be smaller than the template.
    """
    h, w = frame_bgr.shape[:2]
    th, tw = template.shape[:2]
    cx, cy = last_center
    half_w = tw // 2 + search_margin
    half_h = th // 2 + search_margin
    x0 = max(0, int(round(cx - half_w)))
    y0 = max(0, int(round(cy - half_h)))
    x1 = min(w, int(round(cx + half_w)))
    y1 = min(h, int(round(cy + half_h)))
    window = frame_bgr[y0:y1, x0:x1]
    if window.shape[0] < th or window.shape[1] < tw:
        window = frame_bgr
        x0, y0 = 0, 0
    res = cv2.matchTemplate(window, template, _TM_METHOD)
    _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
    abs_x = x0 + maxl[0]
    abs_y = y0 + maxl[1]
    center = (abs_x + tw / 2.0, abs_y + th / 2.0)
    box = (abs_x, abs_y, abs_x + tw, abs_y + th)
    return center, float(maxv), box


def servo_align_to_principal_point(
    ctx: TaskContext,
    init_boxes: Sequence[tuple[int, int, int, int]],
    *,
    fixed_ori: np.ndarray,
    gain: float = 0.5,
    ki: float = 0.0,
    px_tol: float = 8.0,
    max_iters: int = 40,
    step_clip_m: float = 0.006,
    target_px: tuple[float, float] | None = None,
    settle_s: float = 0.35,
    converge_ticks: int = 3,
    search_margin_px: int = _TM_SEARCH_MARGIN_PX,
    min_score: float = _TM_MIN_SCORE,
    probe_delta_m: float = 0.012,
    save_dir: str | Path | None = None,
    step: bool = False,
    video_fps: float = 4.0,
    recovery_frames: int = 6,
    max_recoveries: int = 2,
    grasp_z_nominal: float | None = None,
    meters_per_px: float = _SERVO_M_PER_PX,
    cart_kp: float = _SERVO_CART_KP,
    cart_kv: float = _SERVO_CART_KV,
    ori_kp: float = _SERVO_ORI_KP,
    ori_kv: float = _SERVO_ORI_KV,
    chase_tol_m: float = 0.008,
    chase_timeout_s: float = 1.2,
    phase_settle: int = 1,
    max_phase_iters: int = 8,
) -> tuple[np.ndarray, float, list[tuple[int, int]]]:
    """Servo EE XY until the matched-box-center midpoint reaches ``target_px``.

    Parameters
    ----------
    init_boxes
        Seed bounding boxes ``(x0, y0, x1, y1)`` (e.g. the two handle boxes
        from the refine Gemini call). Each is cropped from the first servo
        frame into a template that is re-located every frame.
    fixed_ori
        Fixed tool-down orientation held for the entire servo segment.
    gain
        Multiplies the fixed RGB-only pixel-to-meter step.
    ki
        Integral gain on the accumulated per-axis pixel error. The proportional
        step is per-iteration clipped, so when the arm stalls against static
        friction / a soft limit (constant residual error, frozen feature) the
        proportional command can't grow and the servo sticks. The integral
        accumulates that residual and adds a growing offset (its own larger
        clip) so the commanded goal keeps marching out until the arm breaks
        through. Anti-windup: the accumulator resets when the axis re-enters
        tolerance or its drive direction flips. ``0`` disables it.
    px_tol
        Stop when ``|midpoint - target|`` is below this (pixels).
    step_clip_m
        Maximum world-frame XY step per iteration (meters).
    target_px
        Image target ``(u, v)``; defaults to ``(intrinsics.ppx, intrinsics.ppy)``.
    settle_s
        Retained for API compatibility; continuous mode publishes a cartesian
        goal and waits for joint velocity to drop (not position convergence).
    search_margin_px
        Template search window margin (px) around each box's last center.
    min_score
        Abort if a template's normalized match score drops below this.
    probe_delta_m
        World-frame distance (m) for the two Jacobian calibration probes at
        servo start (one +X, one +Y). Big enough for a clear pixel shift,
        small enough to stay safely above the object.
    save_dir
        If set, write an annotated ``servo_NNN.png`` per iteration here. The
        directory is created and any existing ``servo_*.png`` are cleared at
        the start of the call.
    step
        Accepted for API compatibility; the servo no longer ENTER-gates moves
        (alignment runs continuously). Annotated frames are still saved.
    recovery_frames
        After a lost-track event, the arm returns to the best pose seen so far
        and template matching is retried this many times before giving up.
    max_recoveries
        How many separate lost-track recovery attempts are allowed before the
        servo stops and returns the best pose.

    Returns
    -------
    converged_ee_pos
        EE world position (3,) after the last servo iteration.
    grasp_z_nominal (return value slot 1)
        World Z for the downstream grasp pose (from refine detection).
        Passed through unchanged — the servo is RGB-only.
    final_pixels
        Last tracked box-center pixel locations.
    meters_per_px
        Fixed robot meters per pixel of image-axis error. This replaces all
        previous depth/focal-length scaling; it is just an RGB servo gain.
    cart_kp, cart_kv, ori_kp, ori_kv
        Temporary cartesian position/orientation PID boost during the
        servo (restored on exit). Higher orientation gains stop the EE
        from rotating during XY moves.
    chase_tol_m, chase_timeout_s
        Retained for API compatibility; continuous servo no longer uses
        position-chase ``move_to`` (see ``_servo_move`` velocity settle).
    phase_settle
        Consecutive in-tolerance frames the active axis must hold before the
        servo switches to the other axis. ``1`` switches as soon as it lands.
    max_phase_iters
        Max iterations spent driving one axis before switching anyway (so a
        stubborn axis can't starve the other).
    """
    if len(init_boxes) < 1:
        raise ValueError("visual servo needs at least one seed box")

    out_dir = _prepare_save_dir(save_dir) if save_dir is not None else None
    if out_dir is not None:
        print(f"[visual_servo] saving annotated frames to {out_dir}")

    pipeline, align, depth_scale, intrinsics = ctx.realsense()
    from vision import realsense_rgbd as rs_cam

    if target_px is None:
        target_u = float(intrinsics.ppx)
        target_v = float(intrinsics.ppy)
    else:
        target_u, target_v = float(target_px[0]), float(target_px[1])

    miss_counter = [0]
    centers: list[tuple[float, float]] = [_box_center(b) for b in init_boxes]

    print(
        f"[visual_servo] seed_boxes={list(init_boxes)} "
        f"target=({target_u:.1f},{target_v:.1f}) "
        f"gain={gain} m_per_px={meters_per_px:g} "
        f"px_tol={px_tol} max_iters={max_iters} (BGR color, no depth)"
    )

    def _grab() -> np.ndarray:
        """BGR frame — depth is not used for tracking or steps."""
        color_bgr = rs_cam.next_rgb_frame(
            pipeline, ctx.cam_timeout_ms, miss_counter, max_misses=10,
        )
        if color_bgr is None:
            raise RuntimeError("visual servo: no camera frame")
        return color_bgr

    def _cur_pos() -> np.ndarray:
        pose = arm.read_current_ee_world(ctx.redis)
        if pose is None:
            raise RuntimeError("visual servo: current EE pose unavailable")
        return pose[0].copy()

    hold_ori = np.asarray(fixed_ori, dtype=np.float64).reshape(3, 3).copy()

    # Seed templates from the first frame (arm stationary at the above pose).
    color_bgr = _grab()
    templates = [_crop_template(color_bgr, b) for b in init_boxes]
    if out_dir is not None:
        seed_centers = [(int(round(c[0])), int(round(c[1]))) for c in centers]
        smu, smv = _midpoint_pixels(seed_centers)
        seed_err = float(
            np.hypot(target_u - smu, target_v - smv),
        )
        seed_scores: list[float] = []
        for i, tmpl in enumerate(templates):
            _c, score, _box = _match_template(
                color_bgr, tmpl, centers[i], search_margin_px,
            )
            seed_scores.append(score)
            cv2.imwrite(str(out_dir / f"template_{i}.png"), tmpl)
        seed_img = _annotate_servo_frame(
            color_bgr,
            seed_centers,
            (smu, smv),
            (target_u, target_v),
            iter_idx=0,
            err_norm=seed_err,
            goal_pos=None,
            converged=False,
            boxes=list(init_boxes),
            px_tol=px_tol,
            note=(
                "SEED: Gemini/init boxes + BGR template crops "
                f"(scores={','.join(f'{s:.2f}' for s in seed_scores)})"
            ),
        )
        seed_path = out_dir / "servo_000_seed.png"
        cv2.imwrite(str(seed_path), seed_img)
        print(
            f"[visual_servo] saved seed frame {seed_path} "
            f"mid=({smu:.0f},{smv:.0f}) |err|={seed_err:.1f}px "
            f"templates={len(templates)}",
            flush=True,
        )

    # Image u/v unit directions for drawing tolerance bands (not for control).
    _IMG_U = np.array([1.0, 0.0], dtype=np.float64)
    _IMG_V = np.array([0.0, 1.0], dtype=np.float64)

    def _match_all(
        frame_bgr: np.ndarray,
        *,
        color_bgr: np.ndarray | None = None,
        iter_idx: int | None = None,
        strict: bool = True,
        fail_note: str | None = None,
    ) -> list[tuple[int, int, int, int]] | None:
        """Re-locate every template; updates ``centers``, returns matched boxes.

        On lost track (any template below ``min_score``) saves an annotated
        FAILED frame when possible. If ``strict`` is True, raises; otherwise
        returns ``None`` (centers left at the bad match locations).
        """
        nonlocal centers
        new_centers: list[tuple[float, float]] = []
        boxes: list[tuple[int, int, int, int]] = []
        worst_score = float("inf")
        worst_i = -1
        for i, tmpl in enumerate(templates):
            c, score, box = _match_template(
                frame_bgr, tmpl, centers[i], search_margin_px
            )
            new_centers.append(c)
            boxes.append(box)
            if score < worst_score:
                worst_score, worst_i = score, i

        if worst_score < min_score:
            if color_bgr is not None and out_dir is not None:
                tracked = [(int(round(c[0])), int(round(c[1]))) for c in new_centers]
                fmu, fmv = _midpoint_pixels(tracked)
                note = fail_note or (
                    f"LOST TRACK: template {worst_i} score={worst_score:.2f} "
                    f"< {min_score:.2f}"
                )
                img = _annotate_servo_frame(
                    color_bgr, tracked, (fmu, fmv), (target_u, target_v),
                    iter_idx=iter_idx or 0, err_norm=float("nan"),
                    goal_pos=None, converged=False, boxes=boxes,
                    px_tol=px_tol, bound_axes=_bound_axes(), note=note,
                    failed=True,
                )
                if isinstance(iter_idx, int):
                    tag = f"{iter_idx:03d}"
                elif iter_idx:
                    tag = str(iter_idx)
                else:
                    tag = "x"
                save_path = out_dir / f"servo_{tag}_FAILED.png"
                cv2.imwrite(str(save_path), img)
                print(f"[visual_servo] saved {save_path}")
            if strict:
                raise RuntimeError(
                    f"visual servo: template {worst_i} match score "
                    f"{worst_score:.2f} < {min_score:.2f} (lost track)"
                )
            return None
        centers = new_centers
        return boxes

    def _midpoint_xy() -> np.ndarray:
        pts = [(int(round(c[0])), int(round(c[1]))) for c in centers]
        mu, mv = _midpoint_pixels(pts)
        return np.array([mu, mv], dtype=np.float64)

    # --- Combined world X+Y alignment -------------------------------------
    # Each iteration we recompute the image projection of world X and world Y
    # from the CURRENT EE orientation, take the signed image error along each,
    # and step both world coordinates together. Steps are tolerance-aware
    # (large when far, small near target) so each axis can settle inside
    # px_tol instead of overshooting. Convergence requires BOTH world-axis
    # errors inside the drawn tolerance bands; otherwise we fall back to the
    # best (smallest |err|) pose seen. The world tolerance bounds are drawn
    # rotated into the image so the valid +/- px_tol range for each axis is
    # visible as a band of lines.
    # Gated (discrete) moves wait for the arm to actually arrive; give the
    # kinematically-slow world axis enough time and a looser tolerance so a
    # single ENTER step lands instead of burning the whole timeout. Skip the
    # velocity-settle gate (vel_tol_rad_s=None) so move_to returns as soon as
    # position is in tolerance.
    _MOVE_TOL_M = 0.008
    _MOVE_TIMEOUT_S = 1.5
    # Continuous-chase (un-gated) move: settle close to each goal before
    # recomputing, with caller-provided tolerance/timeout.
    _CHASE_TOL_M = float(chase_tol_m)
    _CHASE_TIMEOUT_S = float(chase_timeout_s)
    converged_ee = _cur_pos()
    final_pixels = [(int(round(c[0])), int(round(c[1]))) for c in centers]
    mid_world_z = (
        float(grasp_z_nominal)
        if grasp_z_nominal is not None
        else float(converged_ee[2])
    )

    _AXIS_COLORS = {0: (0, 165, 255), 1: (0, 255, 0)}  # u=orange, v=green (BGR)

    def _bound_axes() -> list:
        """Image u/v tolerance bands at the target pixel."""
        return [
            ((float(_IMG_U[0]), float(_IMG_U[1])), _AXIS_COLORS[0], "u"),
            ((float(_IMG_V[0]), float(_IMG_V[1])), _AXIS_COLORS[1], "v"),
        ]

    def _step_clip(abs_e: float) -> float:
        """Tolerance-aware step cap: big when far, small near target (damps
        overshoot so the axis can actually settle inside px_tol)."""
        if abs_e > 60.0:
            return 0.05
        if abs_e > 30.0:
            return 0.02
        if abs_e > px_tol:
            return 0.012
        return 0.005

    # Max integral contribution (m) added on top of the clipped proportional
    # step. Large enough to march the goal well past a stall, capped so a
    # runaway accumulator can't fling the arm.
    _INTEG_CLIP_M = 0.08

    def _run_combined() -> None:
        """Drive world X and world Y together until the midpoint hits target."""
        nonlocal converged_ee, final_pixels, mid_world_z

        ok_streak = 0
        best_err = float("inf")
        best_ee = _cur_pos()
        # Pin the servo height and orientation to their values at servo start so
        # the arm stays in the same plane/pose throughout (only XY is servoed).
        # Re-commanding cur[2] each loop let small drift ratchet the arm upward.
        start_z = float(best_ee[2])
        hold_ori = np.asarray(fixed_ori, dtype=np.float64).reshape(3, 3).copy()
        best_px = list(final_pixels)
        best_mid_z = mid_world_z
        best_it = 0
        best_img: np.ndarray | None = None
        prev_e: dict[int, float | None] = {0: None, 1: None}
        drive_sign: dict[int, float] = {0: 1.0, 1: 1.0}
        grew: dict[int, int] = {0: 0, 1: 0}
        cooldown: dict[int, int] = {0: 0, 1: 0}
        # Accumulated per-axis pixel error for the integral term (px).
        integ: dict[int, float] = {0: 0.0, 1: 0.0}
        mode = "discrete ENTER-gated" if step else "continuous"

        # Settle tuning. For debugging, deliberately wait for the EE pose to
        # settle after every published servo goal before grabbing the next
        # image. This is slower than progress-only waiting, but it makes each
        # iteration correspond to one physical motion.
        _SETTLE_TICK_S = 0.10
        _SETTLE_REACH_TOL_M = 0.006    # close enough to goal -> done once dwell elapsed
        _SETTLE_STABLE_EPS_M = 0.0004  # EE moved less than this tick-to-tick -> stable
        _SETTLE_STABLE_TICKS = 5       # consecutive stable ticks -> settled
        _SETTLE_MIN_DWELL_S = 0.70     # always wait this long before judging

        def _wait_arm_settled(
            goal: np.ndarray,
            *,
            timeout_s: float,
            label: str = "visual servo settle",
        ) -> None:
            """Block until the arm has physically settled after a servo goal."""
            from zitibot_core.runner import wait_until

            goal3 = np.asarray(goal, dtype=np.float64).reshape(3)
            t0 = time.monotonic()
            start_pos: np.ndarray | None = None
            last_pos: np.ndarray | None = None
            last_d: float | None = None
            stable_count = 0
            reason = "timeout"

            def _done() -> bool:
                nonlocal start_pos, last_pos, last_d, stable_count, reason
                pose = arm.read_current_ee_world(ctx.redis)
                if pose is None:
                    return False
                cur_pos = pose[0]
                if start_pos is None:
                    start_pos = cur_pos.copy()
                if last_pos is not None:
                    tick_move = float(np.linalg.norm(cur_pos - last_pos))
                    if tick_move < _SETTLE_STABLE_EPS_M:
                        stable_count += 1
                    else:
                        stable_count = 0
                last_pos = cur_pos.copy()
                d = float(np.linalg.norm(pose[0] - goal3))
                last_d = d
                if time.monotonic() - t0 < _SETTLE_MIN_DWELL_S:
                    return False
                if d < _SETTLE_REACH_TOL_M:
                    reason = "reached"
                    return True
                if stable_count >= _SETTLE_STABLE_TICKS:
                    reason = "pose stable"
                    return True
                return False

            try:
                wait_until(
                    _done,
                    timeout_s=timeout_s,
                    tick=_SETTLE_TICK_S,
                    ctx=ctx,
                    label=label,
                )
            except TimeoutError:
                pass  # proceed with whatever pose we have
            elapsed = time.monotonic() - t0
            moved = (
                float(np.linalg.norm(last_pos - start_pos))
                if start_pos is not None and last_pos is not None
                else 0.0
            )
            rem = last_d if last_d is not None else float("nan")
            print(
                f"[visual_servo] settle {reason}: waited {elapsed:.2f}s, "
                f"EE moved {moved * 1000:.1f} mm, goal_err {rem * 1000:.1f} mm",
                flush=True,
            )

        def _servo_move(
            pos: np.ndarray,
            label: str,
            *,
            step_m: float | None = None,
        ) -> None:
            pos = pos.copy()
            pos[2] = start_z
            if step:
                # ENTER-gate each servo move ourselves. ``arm.move_to`` only
                # prompts when ``ctx.step`` is set (the global --step flag), so
                # --servo-gate alone would otherwise fire moves back-to-back.
                if label:
                    print(label, flush=True)
                try:
                    resp = input(
                        "[visual_servo] ENTER to execute move (q to abort): "
                    )
                except EOFError:
                    resp = ""
                if resp.strip().lower() == "q":
                    raise KeyboardInterrupt("visual servo aborted by user")
                arm.move_to(
                    ctx, pos, hold_ori, label=None,
                    tol_m=_MOVE_TOL_M, timeout_s=_MOVE_TIMEOUT_S, gated=False,
                    vel_tol_rad_s=None,
                )
                return
            if label:
                print(label, flush=True)
            arm.publish_goal_cartesian(ctx.redis, pos, hold_ori)
            mag = step_m if step_m is not None else 0.012
            settle_s = float(np.clip(1.0 + mag / 0.012, 1.2, 1.5))
            _wait_arm_settled(
                pos,
                timeout_s=settle_s,
                label="visual servo motion settle",
            )

        # Image Jacobian (px per world m), computed analytically from the held
        # EE orientation + hand-eye rotation + intrinsics. Axis calibration
        # probes are skipped: the cartesian loop overshoots 6 mm nudges by 3–5x
        # and +Y often moves the wrong sign, so measured columns were garbage.
        # The camera is Rz(45 deg) off the flange and the EE carries a per-object
        # yaw, so world X (front/back) projects diagonally onto BOTH image axes;
        # a fixed seed J mis-attributes that and drove side-to-side (the bug the
        # user saw). The analytic J captures the true rotation each run.
        _J_SEED = np.array(
            [[-400.0, 350.0], [1200.0, 2800.0]], dtype=np.float64,
        )
        _J_PINV_RCOND = 0.12
        _J_FLIP_COOLDOWN_ITERS = 5
        _fx = float(getattr(intrinsics, "fx", 600.0))
        _fy = float(getattr(intrinsics, "fy", 600.0))
        _R_flange_cam = np.asarray(T_FLANGE_CAMERA[:3, :3], dtype=np.float64)

        def _analytic_jacobian() -> np.ndarray | None:
            """Image J = d(u,v)/d(world X,Y) from orientation + intrinsics.

            Feature is world-fixed; moving the EE by +dW moves the feature by
            -dW in the camera frame. Pinhole: du=(fx/Zc)*dCamX, dv=(fy/Zc)*dCamY.
            """
            try:
                R_base_flange = np.asarray(
                    fixed_ori, dtype=np.float64,
                ).reshape(3, 3)
                R_base_cam = R_base_flange @ _R_flange_cam
                world_to_cam = R_base_cam.T
                # Camera-to-feature depth along the optical axis (m). Use the
                # EE/object height gap plus the camera mount offset behind the
                # flange; exact value only scales step magnitude (capped/gained).
                z_obj = (
                    float(grasp_z_nominal)
                    if grasp_z_nominal is not None
                    else start_z - 0.12
                )
                Zc = float(np.clip(abs(start_z - z_obj) + 0.09, 0.12, 0.45))
                J_out = np.zeros((2, 2), dtype=np.float64)
                for col in (0, 1):
                    e_world = np.zeros(3, dtype=np.float64)
                    e_world[col] = 1.0
                    d_cam = -world_to_cam @ e_world
                    J_out[0, col] = (_fx / Zc) * d_cam[0]
                    J_out[1, col] = (_fy / Zc) * d_cam[1]
                if not np.all(np.isfinite(J_out)) or float(
                    np.linalg.cond(J_out)
                ) > 50.0:
                    return None
                return J_out
            except Exception as exc:  # noqa: BLE001
                print(f"[visual_servo] analytic J failed: {exc}", flush=True)
                return None

        def _align_j_with_error(mat: np.ndarray, err: np.ndarray) -> np.ndarray:
            """Flip J if the pinv step would move the feature away from target."""
            err2 = np.asarray(err, dtype=np.float64).reshape(2)
            if float(np.linalg.norm(err2)) < 1.0:
                return mat
            dw_test = np.linalg.pinv(mat, rcond=_J_PINV_RCOND) @ err2
            if float(np.dot(mat @ dw_test, err2)) < 0:
                print(
                    "[visual_servo] flipped J sign to match image error direction",
                    flush=True,
                )
                return -mat
            return mat

        _wait_arm_settled(
            _cur_pos(), timeout_s=2.0,
            label="[visual_servo] pre-servo settle",
        )
        J = _analytic_jacobian()
        j_source = "analytic (orientation + hand-eye)"
        if J is None:
            J = _J_SEED.copy()
            j_source = "fixed seed (analytic unavailable)"
        mid0 = _midpoint_xy()
        J = _align_j_with_error(
            J,
            np.array([target_u - mid0[0], target_v - mid0[1]], dtype=np.float64),
        )
        print(
            f"[visual_servo] {j_source} Jacobian px/m (rows=u,v cols=world X,Y):\n"
            f"  u: dX={J[0, 0]:+.1f}  dY={J[0, 1]:+.1f}\n"
            f"  v: dX={J[1, 0]:+.1f}  dY={J[1, 1]:+.1f}",
            flush=True,
        )
        print(
            f"[visual_servo] === image-Jacobian servo ({mode}); "
            f"publish goal + wait for arm still (not position chase) ===",
            flush=True,
        )

        lost_track = False
        recoveries_done = 0

        def _move_to_best_pose(label: str) -> None:
            goal = best_ee.copy()
            goal[2] = start_z
            if step:
                arm.move_to(
                    ctx, goal, hold_ori, label=label,
                    tol_m=_MOVE_TOL_M, timeout_s=_MOVE_TIMEOUT_S, gated=True,
                    vel_tol_rad_s=None,
                )
            else:
                _servo_move(goal, label, step_m=0.02)

        def _try_recover_after_loss(main_iter: int) -> bool:
            """Return to best pose and re-try template match for a few frames."""
            nonlocal recoveries_done, ok_streak, cmd_ref, gain_scale, prev_world, prev_mid, prev_err_norm
            if recoveries_done >= max_recoveries:
                print(
                    f"[visual_servo] recovery budget exhausted "
                    f"({max_recoveries} events)"
                )
                return False
            recoveries_done += 1
            print(
                f"[visual_servo] lost track at iter {main_iter}; "
                f"returning to best pose (iter {best_it}, |err|={best_err:.1f}px) "
                f"and trying {recovery_frames} recovery frames "
                f"(recovery {recoveries_done}/{max_recoveries})"
            )
            _move_to_best_pose(
                f"[visual_servo] recovery: move to best pose "
                f"{best_ee.tolist()}"
            )
            for r in range(1, recovery_frames + 1):
                color_r = _grab()
                boxes_r = _match_all(
                    color_r,
                    color_bgr=color_r,
                    iter_idx=f"{main_iter:03d}_rec{r:02d}",
                    strict=False,
                    fail_note=(
                        f"RECOVERY {r}/{recovery_frames}: "
                        f"still below min_score {min_score:.2f}"
                    ),
                )
                if boxes_r is not None:
                    print(
                        f"[visual_servo] track re-acquired on recovery "
                        f"frame {r}/{recovery_frames}"
                    )
                    ok_streak = 0
                    cmd_ref = best_ee.copy()
                    cmd_ref[2] = start_z
                    gain_scale = max(gain_scale, _GAIN_SCALE_FLOOR)
                    prev_world = None
                    prev_mid = None
                    prev_err_norm = None
                    return True
            print(
                f"[visual_servo] recovery failed after {recovery_frames} frames"
            )
            return False

        # Seed image Jacobian + damped least-squares steps. Axis calibration
        # probes are disabled (arm overshoots small goals). When a step makes
        # pixel motion oppose the error vector, flip J and resync cmd_ref.
        _STEP_MAX_M = 0.02       # hard cap on per-iter world step
        # Warn (do not stop) when |err| stops improving for this many iters.
        _STALL_ITERS = 12
        _NEAR_ERR_PX = 30.0      # below this, shrink gain + cap step (gentler now)
        _NEAR_GAIN_FLOOR = 0.6   # don't let near-goal gain collapse below this frac
        _NEAR_STEP_CAP_M = 0.016  # near-goal per-iter cap (was 0.008 -> too timid)
        _PIX_OVERSHOOT_FRAC = 0.8  # allow up to 80% of |err| in one step near goal
        _AXIS_DOM_RATIO = 2.0    # drive one image axis when it dominates the error
        _WORSEN_SHRINK = 0.5     # cut gain after a step that blows up the error
        _GAIN_SCALE_FLOOR = 0.35  # don't throttle to ~0.1 when still 30+ px out

        prev_world: np.ndarray | None = None
        prev_mid: np.ndarray | None = None
        prev_err_norm: float | None = None
        prev_err_vec: np.ndarray | None = None
        gain_scale = 1.0
        last_best_it = 0
        j_flip_cooldown = 0
        _LARGE_ERR_PX = 80.0
        _LARGE_ERR_DOM_RATIO = 1.2

        def _delta_from_row(row: np.ndarray, err_a: float, eff: float) -> np.ndarray:
            """Least-squares step along one Jacobian row (one image axis)."""
            row = np.asarray(row, dtype=np.float64).reshape(2)
            denom = float(row @ row)
            if denom < 1e-6:
                return np.zeros(2, dtype=np.float64)
            return eff * (err_a / denom) * row

        plateau_warned_for_stall = False

        def _warn_plateau(it_done: int) -> None:
            nonlocal plateau_warned_for_stall
            if plateau_warned_for_stall:
                return
            plateau_warned_for_stall = True
            print(
                f"[visual_servo] would plateau at iter {it_done}: "
                f"best from iter {best_it} (|err|={best_err:.1f}px, "
                f"no improvement for {_STALL_ITERS} iters) — continuing",
                flush=True,
            )
            if out_dir is not None and best_img is not None:
                save_path = (
                    out_dir / f"servo_best_iter{best_it:03d}_would_plateau.png"
                )
                cv2.imwrite(str(save_path), best_img)
                print(f"[visual_servo] saved {save_path}", flush=True)

        # Command reference. We advance THIS by each step's delta and publish
        # it, rather than recomputing goal = measured_cur + delta every
        # iteration. The arm has a steady-state gravity droop (~1 cm in +X at
        # this extended pose, cart PD with no integral), so re-referencing to
        # the drooped measured pose re-injected that droop into the goal every
        # iteration and made it accumulate -> +X runaway. With a fixed command
        # reference the droop is a constant offset and the visual loop
        # converges. Broyden still learns from the MEASURED motion (cur), which
        # stays correct.
        cmd_ref = _cur_pos()
        cmd_ref[2] = start_z

        it = 0
        while it < max_iters:
            color_bgr = _grab()
            matched_boxes = _match_all(
                color_bgr, color_bgr=color_bgr, iter_idx=it + 1, strict=False,
            )
            if matched_boxes is None:
                if not _try_recover_after_loss(it + 1):
                    lost_track = True
                    break
                color_bgr = _grab()
                matched_boxes = _match_all(
                    color_bgr, color_bgr=color_bgr, iter_idx=it + 1, strict=False,
                )
                if matched_boxes is None:
                    lost_track = True
                    break
                prev_world = None  # pose jumped during recovery; don't learn from it
                prev_mid = None
            tracked = [(int(round(c[0])), int(round(c[1]))) for c in centers]
            final_pixels = tracked
            mu, mv = _midpoint_pixels(tracked)
            mid_px = np.array([mu, mv], dtype=np.float64)
            err_vec = np.array([target_u - mu, target_v - mv], dtype=np.float64)
            err_norm = float(np.hypot(err_vec[0], err_vec[1]))
            e_u, e_v = float(err_vec[0]), float(err_vec[1])

            in_tol = abs(e_u) <= px_tol and abs(e_v) <= px_tol

            cur = _cur_pos()

            # Adapt the command scale to recent outcomes. A worsening step
            # shrinks the next command; an improving step lets it recover
            # toward full gain (so one bad move can't throttle us to ~0 for
            # the rest of the run, which silenced corrections last time).
            if j_flip_cooldown > 0:
                j_flip_cooldown -= 1

            if prev_err_norm is not None:
                if err_norm > prev_err_norm + 1.0:
                    if (
                        prev_mid is not None
                        and prev_err_vec is not None
                        and j_flip_cooldown <= 0
                    ):
                        obs_dpix = mid_px - prev_mid
                        if float(np.dot(obs_dpix, prev_err_vec)) < -2.0:
                            J = -J
                            cmd_ref = cur.copy()
                            cmd_ref[2] = start_z
                            j_flip_cooldown = _J_FLIP_COOLDOWN_ITERS
                            gain_scale = max(gain_scale, 0.5)
                            print(
                                "[visual_servo] flipped J + resynced cmd_ref "
                                "(last motion opposed error)",
                                flush=True,
                            )
                    gain_scale = max(0.2, gain_scale * _WORSEN_SHRINK)
                    if err_norm > _LARGE_ERR_PX:
                        gain_scale = max(_GAIN_SCALE_FLOOR, gain_scale)
                    print(
                        f"[visual_servo] error grew {prev_err_norm:.1f} -> "
                        f"{err_norm:.1f} px; gain_scale={gain_scale:.2f}",
                        flush=True,
                    )
                elif err_norm < prev_err_norm - 1.0:
                    gain_scale = min(1.0, gain_scale + 0.25)

            print(
                f"[visual_servo] iter {it + 1}/{max_iters} "
                f"mid=({mu:.1f},{mv:.1f}) "
                f"eu={e_u:+.1f} ev={e_v:+.1f} |err|={err_norm:.1f} px "
                f"{'BOTH IN TOL' if in_tol else ''} BGR",
                flush=True,
            )

            note = (
                f"eu={e_u:+.1f} ev={e_v:+.1f} "
                f"|err|={err_norm:.1f} tol=+/-{px_tol:.0f}px "
                f"u{'.' if abs(e_u) <= px_tol else 'X'}"
                f"v{'.' if abs(e_v) <= px_tol else 'X'}"
            )

            cur_img = None
            if out_dir is not None:
                cur_img = _annotate_servo_frame(
                    color_bgr, tracked, (mu, mv), (target_u, target_v),
                    iter_idx=it + 1, err_norm=err_norm, goal_pos=cur,
                    converged=in_tol, boxes=matched_boxes,
                    px_tol=px_tol, bound_axes=_bound_axes(), note=note,
                )
                cv2.imwrite(str(out_dir / f"servo_{it + 1:03d}.png"), cur_img)

            if err_norm < best_err - 0.5:
                best_err = err_norm
                best_ee = cur.copy()
                best_px = list(tracked)
                best_mid_z = mid_world_z
                best_it = it + 1
                best_img = cur_img
                last_best_it = it + 1
                plateau_warned_for_stall = False

            if in_tol:
                ok_streak += 1
            else:
                ok_streak = 0
            if ok_streak >= converge_ticks:
                converged_ee = cur
                if out_dir is not None and cur_img is not None:
                    save_path = out_dir / f"servo_{it + 1:03d}_converged.png"
                    cv2.imwrite(str(save_path), cur_img)
                    print(f"[visual_servo] saved {save_path}")
                print(
                    f"[visual_servo] CONVERGED in {it + 1} iterations "
                    f"({converge_ticks} consecutive frames u/v "
                    f"<= {px_tol:.0f}px); using most-recent pose",
                    flush=True,
                )
                return

            if (it + 1) - last_best_it >= _STALL_ITERS:
                _warn_plateau(it + 1)

            # Remember this frame's pose+pixel to learn from next iteration.
            prev_world = cur.copy()
            prev_mid = mid_px.copy()
            prev_err_vec = err_vec.copy()

            # Already aligned (just building the converge streak): hold still.
            if in_tol:
                prev_err_norm = err_norm
                it += 1
                continue

            # Damped least-squares world step; shrink near the target so we
            # don't overshoot and ping-pong (arm only partially tracks each goal).
            eff_gain = gain * max(gain_scale, _GAIN_SCALE_FLOOR)
            if err_norm < _NEAR_ERR_PX:
                eff_gain *= max(_NEAR_GAIN_FLOOR, err_norm / _NEAR_ERR_PX)

            # When one axis is already in tol (or dominates), a full 2-D step
            # couples through J and often blows up the good axis (eu≈2, ev≈-30
            # -> iter1 dY drove v from -30 to -46).  Use a 1-D row solve.
            dom_ratio = (
                _LARGE_ERR_DOM_RATIO
                if err_norm > _LARGE_ERR_PX
                else _AXIS_DOM_RATIO
            )
            u_ok = abs(e_u) <= px_tol
            v_ok = abs(e_v) <= px_tol
            if u_ok and not v_ok:
                delta_world = _delta_from_row(J[1, :], e_v, eff_gain)
            elif v_ok and not u_ok:
                delta_world = _delta_from_row(J[0, :], e_u, eff_gain)
            elif abs(e_v) > dom_ratio * max(abs(e_u), 1.0):
                delta_world = _delta_from_row(J[1, :], e_v, eff_gain)
            elif abs(e_u) > dom_ratio * max(abs(e_v), 1.0):
                delta_world = _delta_from_row(J[0, :], e_u, eff_gain)
            else:
                J_pinv = np.linalg.pinv(J, rcond=_J_PINV_RCOND)
                delta_world = eff_gain * (J_pinv @ err_vec)
            # Cap predicted pixel motion so one chase can't jump past the target.
            pred_pix = J @ delta_world
            if float(np.dot(pred_pix, err_vec)) < 0:
                delta_world = -delta_world
                pred_pix = -pred_pix
                print(
                    "[visual_servo] negated step (predicted motion opposed error)",
                    flush=True,
                )
            pred_norm = float(np.linalg.norm(pred_pix))
            if pred_norm > _PIX_OVERSHOOT_FRAC * err_norm and pred_norm > 1e-6:
                delta_world *= (_PIX_OVERSHOOT_FRAC * err_norm) / pred_norm
            mag = float(np.linalg.norm(delta_world))
            cap = min(_step_clip(err_norm), _STEP_MAX_M)
            if err_norm < _NEAR_ERR_PX:
                cap = min(cap, _NEAR_STEP_CAP_M)
            if mag > cap and mag > 1e-9:
                delta_world *= cap / mag
                mag = cap

            # Advance the command reference (NOT the drooped measured pose) so
            # gravity droop stays a constant offset instead of compounding.
            cmd_ref[0] += float(delta_world[0])
            cmd_ref[1] += float(delta_world[1])
            cmd_ref[2] = start_z

            # Clamp how far cmd_ref may lead the measured EE. With the short
            # settle timeout, settle can return mid-flight; since cmd_ref is a
            # running accumulator it would otherwise march several steps ahead
            # of the lagging arm and then the arm lunges across all of them at
            # once. Capping the lead keeps each published goal ~one step from
            # where the arm actually is (droop stays a constant offset).
            lead = cmd_ref[:2] - cur[:2]
            lead_mag = float(np.linalg.norm(lead))
            _MAX_LEAD_M = _STEP_MAX_M + 0.008
            if lead_mag > _MAX_LEAD_M and lead_mag > 1e-9:
                cmd_ref[:2] = cur[:2] + lead * (_MAX_LEAD_M / lead_mag)
            goal_pos = cmd_ref.copy()

            label = (
                f"[visual_servo] iter {it + 1}: "
                f"dX={delta_world[0]:+.4f} dY={delta_world[1]:+.4f} m "
                f"|dw|={mag:.4f} m gain_eff={eff_gain:.2f} "
                f"J=[[{J[0, 0]:+.0f},{J[0, 1]:+.0f}],"
                f"[{J[1, 0]:+.0f},{J[1, 1]:+.0f}]] "
                f"goal=[{goal_pos[0]:+.4f}, {goal_pos[1]:+.4f}, "
                f"{goal_pos[2]:+.4f}]"
            )
            _servo_move(goal_pos, label, step_m=mag)
            prev_err_norm = err_norm
            it += 1

        # Stopped without converging (lost track or ran out of iterations):
        # fall back to the best pose seen and save an honest representative.
        reason = "lost track" if lost_track else f"no converge in {max_iters} iters"
        print(
            f"[visual_servo] combined stopped ({reason}); "
            f"using best pose from iter {best_it} (|err|={best_err:.1f}px)"
        )
        converged_ee = best_ee
        final_pixels = best_px
        mid_world_z = best_mid_z
        if out_dir is not None and best_img is not None:
            save_path = out_dir / f"servo_best_iter{best_it:03d}_NOT_converged.png"
            cv2.imwrite(str(save_path), best_img)
            print(f"[visual_servo] saved {save_path}")

    with boosted_cart_servo_gains(
        ctx.redis,
        position_kp=cart_kp,
        position_kv=cart_kv,
        orientation_kp=ori_kp,
        orientation_kv=ori_kv,
        label="visual servo",
    ):
        _run_combined()

    if out_dir is not None:
        try:
            _write_servo_video(out_dir, fps=video_fps)
        except Exception as exc:  # noqa: BLE001 - video is best-effort
            print(f"[visual_servo] video stitch failed: {exc}")

    return converged_ee, mid_world_z, final_pixels
