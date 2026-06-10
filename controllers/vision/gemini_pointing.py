"""Gemini Robotics-ER pointing: prompts, API client, 2D/3D geometry, drawing.

NO robot-frame transforms happen here anymore.

This module now only:
    - sends RGB image to Gemini
    - parses grasp pixel
    - deprojects pixel+depth into camera-frame XYZ
    - draws overlays

The robot/world transform chain is handled externally by the controller:

    p_base =
        T_base_flange
        @ T_flange_camera
        @ p_camera
"""

from __future__ import annotations

import concurrent.futures as _cf
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = "gemini-robotics-er-1.6-preview"

# Default per-call deadline (seconds) for ``call_gemini`` before
# ``GeminiTimeoutError`` is raised. 8 s was picked empirically — a
# healthy Gemini Robotics-ER response usually returns in 0.5-2 s, so
# >8 s reliably means the API stalled and we should stop and ask the
# operator for a fresh frame rather than waiting indefinitely.
GEMINI_DEFAULT_TIMEOUT_S: float = 8.0


class GeminiTimeoutError(TimeoutError):
    """Raised when :func:`call_gemini` exceeds its ``timeout_s`` deadline.

    Subclass of :class:`TimeoutError` (and therefore :class:`OSError`)
    so callers can ``except TimeoutError`` for a generic catch, or
    ``except GeminiTimeoutError`` for a Gemini-specific handler that
    re-prompts the operator instead of auto-retrying.
    """


# Lazily-initialised thread pool that backs the per-call timeout in
# :func:`call_gemini`. The Gemini SDK's ``generate_content`` is a
# blocking sync call, so the only way to bound it without forking is
# to run it on a worker thread and ``.result(timeout=...)`` on the
# parent. ``max_workers=2`` lets a stalled call linger on its own
# thread while a new call starts on a second one (the stalled thread
# will eventually return and be GC'd; we just drop the result).
_GEMINI_EXECUTOR: _cf.ThreadPoolExecutor | None = None


def _gemini_executor() -> _cf.ThreadPoolExecutor:
    global _GEMINI_EXECUTOR
    if _GEMINI_EXECUTOR is None:
        _GEMINI_EXECUTOR = _cf.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="gemini-call"
        )
    return _GEMINI_EXECUTOR


# ---------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------


def build_prompt(object_name: str | None, custom_prompt: str | None) -> str:
    if custom_prompt:
        return custom_prompt

    obj = object_name or "bowl"

    return (
        f"In the image, locate the **{obj}**. "
        f"Choose TWO distinct graspable points on the visible outer rim/lip. "
        f"Prefer the near/lower rim closest to the camera. "
        f"These two points should define the orientation of the rim. "
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{obj}_1"}}, {{"point": [y, x], "label": "{obj}_2"}}]\n'
        f"Coordinates must be normalized 0-1000 in [y, x] order."
    )


# ---------------------------------------------------------------------
# Point structure
# ---------------------------------------------------------------------


@dataclass
class PointHit:
    y_norm: float
    x_norm: float
    label: str
    # Optional bounding box (normalized 0-1000, [ymin, xmin, ymax, xmax]) when
    # the detection came from a ``box_2d`` entry. The point above is then the
    # box center. ``None`` for plain point detections.
    box_norm: tuple[float, float, float, float] | None = None


# ---------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    here = Path(__file__).resolve().parent

    for candidate in (
        here / ".env",
        here.parent / ".env",
        here.parent.parent / ".env",
        here.parent.parent.parent / ".env",
    ):
        if candidate.is_file():
            load_dotenv(candidate)
            return


def resolve_api_key() -> str:
    _maybe_load_dotenv()

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not key:
        raise SystemExit(
            "No Gemini API key found."
        )

    return key


# ---------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------


def make_genai_client(api_key: str):
    try:
        from google import genai
    except ImportError as e:
        raise SystemExit(
            "Install google-genai:\n"
            "pip install google-genai"
        ) from e

    return genai.Client(api_key=api_key)


def call_gemini(
    client,
    model: str,
    image_png_bytes: bytes,
    prompt: str,
    temperature: float = 0.5,
    *,
    timeout_s: float | None = GEMINI_DEFAULT_TIMEOUT_S,
) -> str:
    """Synchronous Gemini Robotics-ER call with a per-call deadline.

    Runs ``client.models.generate_content`` on a worker thread and
    waits at most ``timeout_s`` seconds for the result. On timeout,
    raises :class:`GeminiTimeoutError` so callers can stop and ask
    the operator for a fresh frame instead of blocking indefinitely
    (the SDK's sync call has no native deadline). Pass ``timeout_s=None``
    or ``timeout_s<=0`` to disable the deadline and call inline.

    The stalled worker thread is *not* cancelled — Python can't
    interrupt blocking C-extension calls. It will finish whenever
    Gemini eventually responds (or the underlying HTTP client times
    out), and its result will be dropped. The pool is sized to allow
    one stalled call + one fresh retry without queueing.
    """
    from google.genai import types

    def _do_call() -> str:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=image_png_bytes,
                    mime_type="image/png",
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                ),
            ),
        )
        return response.text or ""

    if timeout_s is None or timeout_s <= 0:
        return _do_call()

    fut = _gemini_executor().submit(_do_call)
    try:
        return fut.result(timeout=timeout_s)
    except _cf.TimeoutError as e:
        raise GeminiTimeoutError(
            f"Gemini call exceeded {timeout_s:.1f}s timeout"
        ) from e


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------


def parse_points(text: str) -> list[PointHit]:

    candidates: list[str] = [text.strip()]

    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.S | re.I,
    )

    if fenced:
        candidates.append(fenced.group(1).strip())

    bracketed = re.search(
        r"\[\s*\{.*?\}\s*\]",
        text,
        flags=re.S,
    )

    if bracketed:
        candidates.append(bracketed.group(0).strip())

    last_err: Exception | None = None

    for c in candidates:

        try:
            data = json.loads(c)

        except json.JSONDecodeError as e:
            last_err = e
            continue

        return _normalize_points(data)

    raise ValueError(
        f"Could not parse JSON point list from: {text!r}"
    ) from last_err


def _normalize_points(data) -> list[PointHit]:

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(
            f"Expected list, got {type(data).__name__}"
        )

    out: list[PointHit] = []

    for entry in data:

        if not isinstance(entry, dict):
            continue

        label = str(entry.get("label", ""))

        # Bounding-box detections (Gemini ER ``box_2d`` =
        # [ymin, xmin, ymax, xmax], normalized 0-1000). The center becomes the
        # PointHit so the existing point pipeline keeps working; the box extent
        # is preserved for template matching / cropping.
        box = entry.get("box_2d")
        if (
            isinstance(box, (list, tuple))
            and len(box) == 4
        ):
            try:
                ymin, xmin, ymax, xmax = (float(b) for b in box)
            except (TypeError, ValueError):
                continue
            out.append(
                PointHit(
                    y_norm=0.5 * (ymin + ymax),
                    x_norm=0.5 * (xmin + xmax),
                    label=label,
                    box_norm=(ymin, xmin, ymax, xmax),
                )
            )
            continue

        pt = entry.get("point")

        if not (
            isinstance(pt, (list, tuple))
            and len(pt) == 2
        ):
            continue

        try:
            y, x = float(pt[0]), float(pt[1])

        except (TypeError, ValueError):
            continue

        out.append(
            PointHit(
                y_norm=y,
                x_norm=x,
                label=label,
            )
        )

    return out


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------


POINT_COLORS_BGR = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 0, 255),
    (255, 200, 0),
    (0, 0, 255),
    (255, 255, 0),
]


def denorm(
    p: PointHit,
    w: int,
    h: int,
) -> tuple[int, int]:

    u = int(round((p.x_norm / 1000.0) * (w - 1)))
    v = int(round((p.y_norm / 1000.0) * (h - 1)))

    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))

    return u, v


def denorm_box(
    box_norm: tuple[float, float, float, float],
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    """Normalized [ymin, xmin, ymax, xmax] (0-1000) -> pixel (x0, y0, x1, y1)."""
    ymin, xmin, ymax, xmax = box_norm
    x0 = int(round((xmin / 1000.0) * (w - 1)))
    y0 = int(round((ymin / 1000.0) * (h - 1)))
    x1 = int(round((xmax / 1000.0) * (w - 1)))
    y1 = int(round((ymax / 1000.0) * (h - 1)))
    x0, x1 = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
    y0, y1 = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
    return x0, y0, x1, y1


def draw_world_marker(
    img: np.ndarray,
    pixel: tuple[int, int] | None,
    *,
    label: str = "grasp",
    color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Draw a diamond + label at ``pixel`` for a world-frame point.

    Used to overlay the projected pixel of a *computed* grasp pose
    (after world XY offsets, Z offsets, etc.) on top of the same color
    overlay / depth panel that already shows the raw Gemini points and
    depth-patch boxes. Returns the input unchanged when ``pixel`` is
    ``None`` or lands outside the image.
    """
    if pixel is None:
        return img
    u, v = int(pixel[0]), int(pixel[1])
    h_img, w_img = img.shape[:2]
    if not (0 <= u < w_img and 0 <= v < h_img):
        return img
    out = img.copy()
    cv2.drawMarker(
        out,
        (u, v),
        (0, 0, 0),
        cv2.MARKER_DIAMOND,
        22,
        4,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        out,
        (u, v),
        color,
        cv2.MARKER_DIAMOND,
        22,
        2,
        cv2.LINE_AA,
    )
    if label:
        org = (
            min(w_img - 80, u + 14),
            min(h_img - 8, v + 18),
        )
        cv2.putText(
            out,
            label,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_patch_overlay(
    img: np.ndarray,
    patches: list[dict | None] | None,
) -> np.ndarray:
    """Draw per-point depth-patch debugging overlay.

    For each non-None patch entry, draws:
    * a rectangle marking the ``r``-radius square sampled around Gemini's
      original ``(u, v)``,
    * a tilted-cross marker at ``(u_shallow, v_shallow)`` — the pixel
      whose depth was actually used,
    * a small ``d=…m`` annotation next to that shallow-pixel marker.

    Works on either the color overlay or the colorized depth panel so the
    user can see exactly what region produced the grasp Z.
    """
    if not patches:
        return img
    out = img.copy()
    h_img, w_img = out.shape[:2]
    for i, patch in enumerate(patches):
        if patch is None:
            continue
        color = POINT_COLORS_BGR[i % len(POINT_COLORS_BGR)]
        u = int(patch["u"])
        v = int(patch["v"])
        r = int(patch["radius"])
        x0 = max(0, u - r)
        y0 = max(0, v - r)
        x1 = min(w_img - 1, u + r)
        y1 = min(h_img - 1, v + r)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

        # Gemini's actual returned box_2d (the depth rectangle above is a fixed
        # sampling radius, NOT the detected box). Draw it so the saved response
        # reflects what Gemini boxed / what seeds the template matcher.
        gbox = patch.get("box")
        if gbox is not None and len(gbox) == 4:
            bx0, by0, bx1, by1 = (int(v) for v in gbox)
            cv2.rectangle(out, (bx0, by0), (bx1, by1), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.rectangle(out, (bx0, by0), (bx1, by1), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                out,
                "box_2d",
                (bx0, max(12, by0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                "box_2d",
                (bx0, max(12, by0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        u_s = int(patch["u_shallow"])
        v_s = int(patch["v_shallow"])
        d_s = float(patch["d_shallow"])
        cv2.drawMarker(
            out,
            (u_s, v_s),
            (0, 0, 0),
            cv2.MARKER_TILTED_CROSS,
            18,
            3,
            cv2.LINE_AA,
        )
        cv2.drawMarker(
            out,
            (u_s, v_s),
            color,
            cv2.MARKER_TILTED_CROSS,
            18,
            2,
            cv2.LINE_AA,
        )
        label = f"d={d_s:.3f}m"
        org = (
            min(w_img - 80, u_s + 12),
            max(12, v_s - 8),
        )
        cv2.putText(
            out,
            label,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_points(
    img_bgr: np.ndarray,
    points: list[PointHit],
    metric_lines: list[str] | None = None,
) -> np.ndarray:

    h, w = img_bgr.shape[:2]

    out = img_bgr.copy()

    for i, p in enumerate(points):

        u, v = denorm(p, w, h)

        color = POINT_COLORS_BGR[i % len(POINT_COLORS_BGR)]

        cv2.circle(
            out,
            (u, v),
            9,
            (0, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )

        cv2.circle(
            out,
            (u, v),
            9,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

        cv2.drawMarker(
            out,
            (u, v),
            color,
            cv2.MARKER_CROSS,
            14,
            2,
            cv2.LINE_AA,
        )

        text = f"{p.label} ({u},{v})"

        text_org = (u + 12, v - 8)

        cv2.putText(
            out,
            text,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            out,
            text,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        if metric_lines and i < len(metric_lines):

            y2 = text_org[1] + 16

            for line in metric_lines[i].split("\n"):

                cv2.putText(
                    out,
                    line,
                    (text_org[0], y2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    out,
                    line,
                    (text_org[0], y2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                y2 += 14

    return out


# ---------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------


def encode_png(img_bgr: np.ndarray) -> bytes:

    # Keep OpenCV away from strided/view-backed buffers. Most callers pass
    # owned RealSense copies, but the extra contiguous copy is cheap compared
    # with the Gemini request and avoids native lifetime surprises.
    img = np.ascontiguousarray(img_bgr).copy()
    ok, buf = cv2.imencode(".png", img)

    if not ok:
        raise RuntimeError("Failed to encode PNG.")

    return buf.tobytes()


# ---------------------------------------------------------------------
# Depth / geometry
# ---------------------------------------------------------------------


def _extract_patch(
    depth_m: np.ndarray,
    u: int,
    v: int,
    radius: int,
) -> np.ndarray:
    """Square patch of the depth map clipped to image bounds."""
    h, w = depth_m.shape[:2]
    y0 = max(0, v - radius)
    y1 = min(h, v + radius + 1)
    x0 = max(0, u - radius)
    x1 = min(w, u + radius + 1)
    return depth_m[y0:y1, x0:x1]


def _valid_depth_mask(patch: np.ndarray, min_depth_m: float = 0.0) -> np.ndarray:
    """Boolean mask of usable depth pixels in ``patch``.

    Always rejects the RealSense "no reading" value (depth == 0). When
    ``min_depth_m > 0`` it ALSO rejects anything closer than that floor, so a
    caller can demand "only depth from pixels at least N metres away" — used by
    the egg grasp, where the shiny shell + the gripper's own fingers can produce
    spurious near-field returns that would otherwise yank the grasp Z toward the
    camera. ``min_depth_m <= 0`` keeps the legacy ``depth > 0`` behaviour.
    """
    if min_depth_m and min_depth_m > 0.0:
        return patch >= float(min_depth_m)
    return patch > 0.0


def patch_depth_stats(
    depth_m: np.ndarray,
    u: int,
    v: int,
    radius: int,
    min_depth_m: float = 0.0,
) -> dict | None:
    """Diagnostics: n / min / p10 / p25 / p50 / p75 / max of valid depths."""
    valid = _extract_patch(depth_m, u, v, radius)
    valid = valid[_valid_depth_mask(valid, min_depth_m)]
    if valid.size == 0:
        return None
    return {
        "n": int(valid.size),
        "min": float(np.min(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p25": float(np.percentile(valid, 25)),
        "p50": float(np.percentile(valid, 50)),
        "p75": float(np.percentile(valid, 75)),
        "max": float(np.max(valid)),
    }


def sample_depth_median(
    depth_m: np.ndarray,
    u: int,
    v: int,
    radius: int,
    quantile: float = 50.0,
) -> float | None:
    """Sample a depth statistic over a patch around ``(u, v)``.

    ``quantile`` selects which order statistic of the valid depths to
    return. ``50`` (default) is the median — original behavior. A small
    ``quantile`` (e.g. ``10``) returns a near-shallowest depth, which is
    the highest world-Z point in the patch when the camera looks down on
    a surface. Use that for "grasp the rim/top of this object" prompts so
    a Gemini pixel that lands just inside a rim still picks up the rim's
    true height instead of the lower inside-surface depth. ``quantile=0``
    is the min depth (most aggressive — least robust to a single shallow
    outlier pixel). The chosen patch size must actually reach the desired
    feature; if you want the rim of a bowl, the radius has to cover the
    image-pixel distance between the Gemini-chosen pixel and the rim.
    """

    valid = _extract_patch(depth_m, u, v, radius)
    valid = valid[valid > 0]

    if valid.size == 0:
        return None

    if quantile <= 0.0:
        return float(np.min(valid))
    if quantile >= 100.0:
        return float(np.max(valid))
    return float(np.percentile(valid, quantile))


def find_quantile_pixel(
    depth_m: np.ndarray,
    u: int,
    v: int,
    radius: int,
    quantile: float = 50.0,
    min_depth_m: float = 0.0,
) -> tuple[int, int, float] | None:
    """Return ``(u', v', d')`` of the patch pixel at the requested depth quantile.

    Like :func:`sample_depth_median` but also returns the pixel coordinates
    of the chosen sample, so callers can deproject from the **actual**
    shallow-pixel location instead of from the original Gemini ``(u, v)``.
    This matters when ``quantile`` is small (grasp-on-rim): the rim pixel
    is typically tens of pixels away from where Gemini placed the point,
    so reusing the original ``(u, v)`` would give a 3D point with the rim's
    depth but the interior's XY, off by several cm in world.

    Among pixels with depth equal to the target value (e.g. multiple
    pixels all sitting at the patch minimum), the one closest to the
    original ``(u, v)`` is preferred so the chosen XY stays as close to
    Gemini's intent as possible.
    """
    h, w = depth_m.shape[:2]
    y0 = max(0, v - radius)
    y1 = min(h, v + radius + 1)
    x0 = max(0, u - radius)
    x1 = min(w, u + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    mask = _valid_depth_mask(patch, min_depth_m)
    if not mask.any():
        return None
    valid = patch[mask]

    if quantile <= 0.0:
        target_d = float(np.min(valid))
    elif quantile >= 100.0:
        target_d = float(np.max(valid))
    else:
        target_d = float(np.percentile(valid, quantile))

    yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
    center_y = v - y0
    center_x = u - x0
    dist2 = (yy - center_y) ** 2 + (xx - center_x) ** 2
    depth_diff = np.abs(patch.astype(np.float64) - target_d)
    depth_diff[~mask] = np.inf

    score = depth_diff + 1e-6 * dist2
    flat_idx = int(np.argmin(score))
    dy, dx = np.unravel_index(flat_idx, patch.shape)
    u_prime = int(x0 + dx)
    v_prime = int(y0 + dy)
    d_prime = float(patch[dy, dx])
    return u_prime, v_prime, d_prime


def find_nearest_valid_pixel(
    depth_m: np.ndarray,
    u: int,
    v: int,
    radius: int,
    min_depth_m: float = 0.0,
) -> tuple[int, int, float] | None:
    """Return ``(u', v', d')`` of the valid-depth pixel CLOSEST to ``(u, v)``.

    Unlike :func:`find_quantile_pixel` (which picks a depth order statistic —
    e.g. the shallowest pixel anywhere in the patch), this picks purely by
    image-pixel distance: the nearest pixel to Gemini's point that actually
    has a depth reading. Used by the ``prefer_gemini_pixel`` path when Gemini's
    exact pixel has no depth (common on shiny/curved surfaces like an egg) — we
    want the depth right next to Gemini's point, not the highest point in a
    large patch. Returns ``None`` if no valid depth exists within ``radius``.
    """
    h, w = depth_m.shape[:2]
    y0 = max(0, v - radius)
    y1 = min(h, v + radius + 1)
    x0 = max(0, u - radius)
    x1 = min(w, u + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    mask = _valid_depth_mask(patch, min_depth_m)
    if not mask.any():
        return None
    yy, xx = np.mgrid[0 : patch.shape[0], 0 : patch.shape[1]]
    center_y = v - y0
    center_x = u - x0
    dist2 = ((yy - center_y) ** 2 + (xx - center_x) ** 2).astype(np.float64)
    dist2[~mask] = np.inf
    flat_idx = int(np.argmin(dist2))
    dy, dx = np.unravel_index(flat_idx, patch.shape)
    u_prime = int(x0 + dx)
    v_prime = int(y0 + dy)
    d_prime = float(patch[dy, dx])
    return u_prime, v_prime, d_prime


def deproject_pixel_to_cam(
    intrinsics,
    u: int,
    v: int,
    depth_m_val: float,
):
    """
    RealSense SDK deprojection.

    Returns camera-frame XYZ.
    """

    import pyrealsense2 as rs

    return rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [float(u), float(v)],
        float(depth_m_val),
    )


def lift_points_to_3d(
    points: list[PointHit],
    w: int,
    h: int,
    depth_m: np.ndarray,
    color_intrinsics,
    depth_patch_radius: int,
    depth_quantile: float = 50.0,
    prefer_gemini_pixel: bool = False,
    min_depth_m: float = 0.0,
) -> tuple[
    list[str],
    list[tuple[float, float, float] | None],
    list[dict | None],
]:
    """
    Returns ``(metric_lines, cams, patches)``.

    ``min_depth_m`` (when > 0) rejects every depth pixel closer than that floor
    in BOTH sampling policies — the center-pixel validity test and the patch
    fallbacks (:func:`find_nearest_valid_pixel` / :func:`find_quantile_pixel`).
    Use it when near-field returns (shiny shells, the gripper's own fingers)
    would otherwise contaminate the depth; pixels under the floor are treated
    exactly like "no reading".

    Two sampling policies, selected by ``prefer_gemini_pixel``:

    **Legacy (``prefer_gemini_pixel=False``, default):**

    1. Take Gemini's ``(u, v)``.
    2. Sample depth from the patch around ``(u, v)`` using
       :func:`find_quantile_pixel` (e.g. ``quantile=0`` → shallowest valid
       pixel in the patch). The pixel that supplied the depth may be far
       from ``(u, v)`` — that's the whole point: the rim of a bowl is
       typically tens of pixels off Gemini's chosen point.
    3. **Deproject from the ORIGINAL ``(u, v)`` at depth ``d`` from step 2.**
       Keeps world XY pinned to Gemini's pixel but pulls Z from the rim/top.

    **Gemini-pixel-first (``prefer_gemini_pixel=True``):**

    1. If Gemini's exact pixel ``(u, v)`` has valid depth, deproject from
       ``(u, v)`` at that depth — i.e. use the point Gemini selected.
    2. Otherwise take the NEAREST valid-depth pixel to ``(u, v)`` via
       :func:`find_nearest_valid_pixel` (closest by image distance, NOT the
       shallowest in the patch) and **deproject from that sample pixel** so the
       3D point lands right next to Gemini's choice rather than at the highest
       point somewhere in the patch.

    Use this for compact features (egg, egg-cracker handles, tongs tape) where
    Gemini's point IS the grasp spot and only the depth needs a fallback.

    ``patches`` carries per-point diagnostics for the caller (e.g. to
    draw the sampling area on a debug image): keys
    ``u, v, radius, u_shallow, v_shallow, d_shallow, stats``. Entries are
    ``None`` when no valid depth existed in the patch.
    """

    metric_lines: list[str] = []

    cams: list[tuple[float, float, float] | None] = []

    patches: list[dict | None] = []

    img_h, img_w = depth_m.shape[:2]

    for p in points:

        u, v = denorm(p, w, h)

        stats = patch_depth_stats(depth_m, u, v, depth_patch_radius, min_depth_m)
        if stats is not None:
            print(
                f"  depth_patch px=({u},{v}) r={depth_patch_radius} "
                f"n={stats['n']} "
                f"min={stats['min']:.3f} "
                f"p10={stats['p10']:.3f} "
                f"p25={stats['p25']:.3f} "
                f"p50={stats['p50']:.3f} "
                f"p75={stats['p75']:.3f} "
                f"max={stats['max']:.3f} "
                f"(picked q={depth_quantile:g}, min_depth={min_depth_m:g} m)"
            )
        else:
            print(
                f"  depth_patch px=({u},{v}) r={depth_patch_radius} "
                f"n=0 (no valid depth in patch, min_depth={min_depth_m:g} m)"
            )

        # Does Gemini's exact pixel have valid depth (and clear the floor)?
        center_d = (
            float(depth_m[v, u])
            if (0 <= v < img_h and 0 <= u < img_w)
            else 0.0
        )
        center_valid = center_d > 0.0 and (
            min_depth_m <= 0.0 or center_d >= float(min_depth_m)
        )

        if prefer_gemini_pixel and center_valid:
            # Use the point Gemini selected, at its own depth.
            u_shallow, v_shallow, d = u, v, center_d
            deproj_u, deproj_v = u, v
            print(
                f"  depth_sample: using gemini px=({u},{v}) d={d:.3f} m "
                f"(valid at center)"
            )
        elif prefer_gemini_pixel:
            # Center was invalid (no reading or under the min-depth floor): take
            # the depth at the NEAREST valid pixel to Gemini's point (not the
            # shallowest in the patch) AND deproject from it, so the 3D point
            # sits right next to Gemini's choice.
            sample = find_nearest_valid_pixel(
                depth_m,
                u,
                v,
                depth_patch_radius,
                min_depth_m=min_depth_m,
            )
            if sample is None:
                metric_lines.append("cam: (no depth)")
                cams.append(None)
                patches.append(None)
                continue
            u_shallow, v_shallow, d = sample
            deproj_u, deproj_v = u_shallow, v_shallow
            print(
                f"  depth_sample: gemini px=({u},{v}) invalid -> nearest "
                f"VALID px=({u_shallow},{v_shallow}) d={d:.3f} m "
                f"(deproject from sample px)"
            )
        else:
            # Legacy: shallowest-in-patch depth, deproject at Gemini's XY.
            sample = find_quantile_pixel(
                depth_m,
                u,
                v,
                depth_patch_radius,
                quantile=depth_quantile,
                min_depth_m=min_depth_m,
            )

            if sample is None:
                metric_lines.append("cam: (no depth)")
                cams.append(None)
                patches.append(None)
                continue

            u_shallow, v_shallow, d = sample
            deproj_u, deproj_v = u, v
            if (u_shallow, v_shallow) != (u, v):
                print(
                    f"  depth_sample shifted: gemini px=({u},{v}) -> "
                    f"q{depth_quantile:g} px=({u_shallow},{v_shallow}) "
                    f"d={d:.3f} m (deproject still uses gemini px)"
                )

        xyz = deproject_pixel_to_cam(
            color_intrinsics,
            deproj_u,
            deproj_v,
            d,
        )

        p_cam = np.array(
            xyz,
            dtype=np.float64,
        ).reshape(3)

        cam_t = (
            float(p_cam[0]),
            float(p_cam[1]),
            float(p_cam[2]),
        )

        cams.append(cam_t)

        metric_lines.append(
            f"cam: ({cam_t[0]:.3f}, "
            f"{cam_t[1]:.3f}, "
            f"{cam_t[2]:.3f})"
        )

        patch_entry = {
            "u": int(u),
            "v": int(v),
            "radius": int(depth_patch_radius),
            "u_shallow": int(u_shallow),
            "v_shallow": int(v_shallow),
            "d_shallow": float(d),
            "quantile": float(depth_quantile),
            "stats": stats,
        }
        if p.box_norm is not None:
            patch_entry["box"] = denorm_box(p.box_norm, w, h)
        patches.append(patch_entry)

    return metric_lines, cams, patches


# ---------------------------------------------------------------------
# Main Gemini pipeline
# ---------------------------------------------------------------------


def query_color_depth_overlay(
    client,
    model: str,
    prompt: str,
    temperature: float,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    color_intrinsics,
    depth_patch_radius: int,
    depth_quantile: float = 50.0,
    prefer_gemini_pixel: bool = False,
    min_depth_m: float = 0.0,
    pixels_out: list[tuple[int, int]] | None = None,
) -> tuple[
    np.ndarray | None,
    list[tuple[float, float, float] | None],
    list[dict | None],
]:
    """
    Returns:
        overlay image (with Gemini points + depth-patch sampling boxes drawn),
        list of camera-frame points (empty if no points detected),
        per-point patch debug dicts (see :func:`lift_points_to_3d`); pass
            these to :func:`draw_patch_overlay` to also annotate the depth
            panel of a composite debug image.

    ``min_depth_m`` (when > 0) is forwarded to :func:`lift_points_to_3d` so
    depth pixels closer than the floor are ignored.

    ``pixels_out`` (when provided) is filled with the raw ``(u, v)`` image
    pixel of every Gemini-detected point, regardless of whether that point
    had usable depth. Callers that need to know WHERE Gemini saw the object
    even when the 3D lift failed (e.g. to recenter the camera on it and retry)
    read it here; the normal grasp path leaves it ``None``.
    """

    # Defensive copies at the native-library boundary. The RealSense capture
    # layer already copies SDK-owned frame memory, but downstream OpenCV drawing
    # and encoding are native calls too; keep each Gemini attempt isolated from
    # any shared/view-backed array state.
    color_bgr = np.ascontiguousarray(color_bgr).copy()
    depth_m = np.ascontiguousarray(depth_m).copy()

    print("Sending frame to Gemini ER...")

    t0 = time.perf_counter()

    try:

        raw = call_gemini(
            client,
            model,
            encode_png(color_bgr),
            prompt,
            temperature=temperature,
        )

    except GeminiTimeoutError:

        # Propagate timeouts so the caller can stop and re-prompt the
        # operator (see ``gemini.find_grasp_pose`` / ``_detect``). The
        # generic-exception path below masks the failure as a 2-tuple
        # return, which is fine for transient errors but useless if
        # Gemini is silently stalled.
        raise

    except Exception as e:

        print(
            f"Gemini call failed: {e}",
            file=sys.stderr,
        )

        return None, [], []

    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Response ({dt_ms:.0f} ms): {raw.strip()}")

    try:
        points = parse_points(raw)

    except ValueError as e:

        print(str(e), file=sys.stderr)

        points = []

    h, w = color_bgr.shape[:2]

    if pixels_out is not None:
        for p in points:
            pixels_out.append(denorm(p, w, h))

    metric_lines: list[str] | None = None

    all_cams: list[tuple[float, float, float] | None] = []

    patches: list[dict | None] = []

    if points:

        metric_lines, cams, patches = lift_points_to_3d(
            points,
            w,
            h,
            depth_m,
            color_intrinsics,
            depth_patch_radius,
            depth_quantile=depth_quantile,
            prefer_gemini_pixel=prefer_gemini_pixel,
            min_depth_m=min_depth_m,
        )

        all_cams = cams

        for i, p in enumerate(points):

            u, v = denorm(p, w, h)

            line = f"[{i}] {p.label!r} px=({u},{v})"

            if metric_lines and i < len(metric_lines):
                line += f" {metric_lines[i]}"

            print(line)

    overlay = draw_points(
        color_bgr,
        points,
        metric_lines=metric_lines,
    )

    overlay = draw_patch_overlay(overlay, patches)

    return overlay, all_cams, patches