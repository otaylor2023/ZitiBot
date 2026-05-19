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


# ---------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------


def build_prompt(object_name: str | None, custom_prompt: str | None) -> str:
    if custom_prompt:
        return custom_prompt

    obj = object_name or "bowl"

    return (
        f"In the image, locate the **{obj}**. "
        f"Choose ONE graspable point on the visible outer rim/lip. "
        f"Prefer the near/lower rim closest to the camera. "
        f"Reply with JSON only:\n"
        f'[{{"point": [y, x], "label": "{obj}"}}]\n'
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
) -> str:

    from google.genai import types

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

        label = str(entry.get("label", ""))

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

    ok, buf = cv2.imencode(".png", img_bgr)

    if not ok:
        raise RuntimeError("Failed to encode PNG.")

    return buf.tobytes()


# ---------------------------------------------------------------------
# Depth / geometry
# ---------------------------------------------------------------------


def sample_depth_median(
    depth_m: np.ndarray,
    u: int,
    v: int,
    radius: int,
) -> float | None:

    h, w = depth_m.shape[:2]

    y0 = max(0, v - radius)
    y1 = min(h, v + radius + 1)

    x0 = max(0, u - radius)
    x1 = min(w, u + radius + 1)

    patch = depth_m[y0:y1, x0:x1]

    valid = patch[patch > 0]

    if valid.size == 0:
        return None

    return float(np.median(valid))


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
) -> tuple[
    list[str],
    list[tuple[float, float, float] | None],
]:
    """
    Returns camera-frame XYZ points ONLY.
    """

    metric_lines: list[str] = []

    cams: list[tuple[float, float, float] | None] = []

    for p in points:

        u, v = denorm(p, w, h)

        d = sample_depth_median(
            depth_m,
            u,
            v,
            depth_patch_radius,
        )

        if d is None:
            metric_lines.append("cam: (no depth)")
            cams.append(None)
            continue

        xyz = deproject_pixel_to_cam(
            color_intrinsics,
            u,
            v,
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

    return metric_lines, cams


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
) -> tuple[
    np.ndarray | None,
    tuple[float, float, float] | None,
]:
    """
    Returns:
        overlay image,
        first camera-frame point
    """

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

    except Exception as e:

        print(
            f"Gemini call failed: {e}",
            file=sys.stderr,
        )

        return None, None

    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Response ({dt_ms:.0f} ms): {raw.strip()}")

    try:
        points = parse_points(raw)

    except ValueError as e:

        print(str(e), file=sys.stderr)

        points = []

    h, w = color_bgr.shape[:2]

    metric_lines: list[str] | None = None

    first_cam: tuple[float, float, float] | None = None

    if points:

        metric_lines, cams = lift_points_to_3d(
            points,
            w,
            h,
            depth_m,
            color_intrinsics,
            depth_patch_radius,
        )

        for c in cams:
            if c is not None:
                first_cam = c
                break

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

    return overlay, first_cam