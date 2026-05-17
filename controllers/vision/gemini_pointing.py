"""Gemini Robotics-ER pointing: prompts, API client, 2D/3D geometry, drawing.

No Redis and no camera drivers here — only numpy/cv2/Gemini.

**Depth / 3D frame pipeline**

1. ``rs2_deproject_pixel_to_point`` returns a point in **Intel RealSense color optical**
   (+Z into the scene, +X image right, +Y image down).

2. We remap that into a **ZitiBot vision frame** (orthonormal, same origin as RS optical):

   - **+X** = **up** in the image (−RS +Y),
   - **+Z** = **into the scene** (same as RS +Z; field feedback: old “−Z forward” map made world **Z** wrong while X/Y matched),
   - **+Y** = **+RS +X** (image right; field: negating RS X here fixed reversed world **Y**).

   With ``p_rs = R @ p_vision`` (``R`` orthonormal), ``p_vision = R.T @ p_rs``.

3. ``T_ee_cam`` (JSON or placeholder) maps **vision** coordinates → **end-effector**
   link frame used by OpenSai ``cartesian_task`` (see URDF note below). With identity
   rotation, column ``t`` is the **camera optical origin in EE** (m); built-in defaults
   use ``ZITIBOT_PLACEHOLDER_T_EE_CAM_X_M`` and ``ZITIBOT_PLACEHOLDER_T_EE_CAM_Z_M`` (see code).

**URDF** (``ZitiBot/urdf_models/panda/panda_arm_sphere.urdf``, same file as
``real_panda.xml`` / ``zitibot_panda.xml`` via ``robotModelFile``):

- Cartesian task uses ``linkName="end-effector"``.
- ``joint_ee`` (fixed) from ``link7`` → ``end-effector``: ``origin xyz="0 0 0.14" rpy="0 0 0"``.
  So the **end-effector** frame has the **same orientation as link7**, with origin
  **0.14 m along link7’s +Z** from the link7 origin. Any calibrated extrinsic should
  be **vision → this end-effector frame** (not world).
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

# RealSense optical -> ZitiBot vision (see module docstring). Columns = vision +X,+Y,+Z in RS optical.
# p_vis = (-y_rs, +x_rs, z_rs): +Z matches RS into-scene; +Y uses +RS X (fixes reversed world Y).
_R_VISION_FROM_RS_OPTICAL = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# p_rs = R @ p_vision  =>  p_vision = R.T @ p_rs
assert np.allclose(_R_VISION_FROM_RS_OPTICAL @ _R_VISION_FROM_RS_OPTICAL.T, np.eye(3))
assert np.linalg.det(_R_VISION_FROM_RS_OPTICAL) > 0
# If ``T`` was calibrated for RS optical only: use ``R_ee_vis = R_ee_rs @ _R_VISION_FROM_RS_OPTICAL``,
# same ``t`` if the optical origin did not move.

# Placeholder vision→EE: identity rotation; translation = camera optical origin in EE frame (m).
# Defaults: +0.04 m EE +X, Z from ``ZITIBOT_PLACEHOLDER_T_EE_CAM_Z_M`` (override X/Z or use
# ``--ee-from-cam-json``).
_PLACEHOLDER_T_EE_CAM_X_M = float(os.environ.get("ZITIBOT_PLACEHOLDER_T_EE_CAM_X_M", "0.00"))
_PLACEHOLDER_T_EE_CAM_Z_M = float(os.environ.get("ZITIBOT_PLACEHOLDER_T_EE_CAM_Z_M", "-0.1134"))
_PLACEHOLDER_T_EE_CAM_Y_M = float(os.environ.get("ZITIBOT_PLACEHOLDER_T_EE_CAM_Y_M", "0.00"))
_PLACEHOLDER_T_XYZ_M = (_PLACEHOLDER_T_EE_CAM_X_M, _PLACEHOLDER_T_EE_CAM_Y_M, _PLACEHOLDER_T_EE_CAM_Z_M)
PLACEHOLDER_T_EE_CAM = np.array(
    [
        [1.0, 0.0, 0.0, _PLACEHOLDER_T_XYZ_M[0]],
        [0.0, 1.0, 0.0, _PLACEHOLDER_T_XYZ_M[1]],
        [0.0, 0.0, 1.0, _PLACEHOLDER_T_XYZ_M[2]],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def realsense_optical_to_vision(p_rs: np.ndarray) -> np.ndarray:
    """Map a 3-vector from RealSense optical to the ZitiBot vision frame (same origin)."""
    v = np.asarray(p_rs, dtype=np.float64).reshape(3)
    return (_R_VISION_FROM_RS_OPTICAL.T @ v).reshape(3)


def build_prompt(object_name: str | None, custom_prompt: str | None) -> str:
    if custom_prompt:
        return custom_prompt
    obj = object_name or "bowl"
    return (
        f"In the image, locate the **{obj}**. Choose **one** point on the **outer rim / lip** "
        f"of the {obj} where a parallel-jaw gripper could make **stable contact** for a pick: "
        "not the flat bottom or deep interior; avoid the table, fingers, or clutter. "
        "You must pick a point on the **leftmost** part of the bowl rim as seen in the image: "
        "prefer the rim location with the **smallest normalized x** (farther **left** in the frame; "
        "x is the second coordinate in [y, x]). Among rim points on that left side that still look "
        "graspable and on the true bowl rim, choose the clearest one. "
        "Use this image convention: normalized **y** grows **downward**; normalized **x** grows **rightward**. "
        "If the left rim is occluded, choose the leftmost **visible** rim point on the bowl. "
        f"The label should name the object (e.g. \"{obj}\"). "
        "Reply with JSON only in this form: "
        '[{"point": [y, x], "label": <label>}]. '
        "Use one entry unless the scene clearly has two separate bowls. "
        "Coordinates are [y, x] normalized 0-1000 (y first, then x)."
    )


@dataclass
class PointHit:
    y_norm: float
    x_norm: float
    label: str


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
            "No API key found. Add GEMINI_API_KEY=... to a .env file at the "
            "repo root (auto-loaded), or export GEMINI_API_KEY / "
            "GOOGLE_API_KEY in your shell.\n"
            "Get a key at https://aistudio.google.com/."
        )
    return key


def make_genai_client(api_key: str):
    try:
        from google import genai
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "google-genai is not installed. Install it with:\n"
            "    pip install google-genai"
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
            types.Part.from_bytes(data=image_png_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text or ""


def parse_points(text: str) -> list[PointHit]:
    candidates: list[str] = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1).strip())
    bracketed = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.S)
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
        f"Could not parse a JSON point list from model response: {text!r}"
    ) from last_err


def _normalize_points(data) -> list[PointHit]:
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of points, got {type(data).__name__}")

    out: list[PointHit] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        pt = entry.get("point")
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            continue
        try:
            y, x = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        label = str(entry.get("label", ""))
        out.append(PointHit(y_norm=y, x_norm=x, label=label))
    return out


POINT_COLORS_BGR = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 0, 255),
    (255, 200, 0),
    (0, 0, 255),
    (255, 255, 0),
]


def denorm(p: PointHit, w: int, h: int) -> tuple[int, int]:
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
        cv2.circle(out, (u, v), 9, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.circle(out, (u, v), 9, color, 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(out, (u, v), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        label = p.label or f"pt{i}"
        text = f"{label} ({u},{v})"
        text_org = (u + 12, v - 8)
        cv2.putText(
            out, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            out, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1, cv2.LINE_AA,
        )
        if metric_lines and i < len(metric_lines) and metric_lines[i]:
            y2 = text_org[1] + 16
            for line in metric_lines[i].split("\n"):
                cv2.putText(
                    out, line, (text_org[0], y2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA,
                )
                cv2.putText(
                    out, line, (text_org[0], y2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA,
                )
                y2 += 14
    return out


def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image.")
    return buf.tobytes()


def load_T_ee_cam(path: Path) -> np.ndarray:
    """Load 4×4 ``T_ee_cam``: **vision** optical coords → **end-effector** (URDF link, meters)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        arr = np.array(data, dtype=np.float64)
    elif isinstance(data, dict):
        if "T_ee_cam" in data:
            arr = np.array(data["T_ee_cam"], dtype=np.float64)
        elif "matrix" in data:
            arr = np.array(data["matrix"], dtype=np.float64)
        else:
            raise ValueError(f"JSON must contain 'T_ee_cam' or 'matrix' key: {path}")
    else:
        raise ValueError(f"Unexpected JSON root type in {path}")
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"Expected 4×4 matrix, got shape {arr.shape} in {path}")


def transform_point_xyz(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = T @ ph
    w = float(out[3])
    if abs(w) < 1e-12:
        return out[:3].copy()
    return (out[:3] / w).astype(np.float64)


def sample_depth_median(
    depth_m: np.ndarray, u: int, v: int, radius: int,
) -> float | None:
    h, w = depth_m.shape[:2]
    y0, y1 = max(0, v - radius), min(h, v + radius + 1)
    x0, x1 = max(0, u - radius), min(w, u + radius + 1)
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
    """Map pixel + depth (m) to RealSense **color optical** 3D using SDK intrinsics.

    ``intrinsics`` come from the active color stream profile (fx, fy, cx, cy, distortion);
    no separate calibration file is needed for live RealSense.
    """
    import pyrealsense2 as rs

    return rs.rs2_deproject_pixel_to_point(
        intrinsics, [float(u), float(v)], float(depth_m_val)
    )


def lift_points_to_3d(
    points: list[PointHit],
    w: int,
    h: int,
    depth_m: np.ndarray,
    color_intrinsics,
    depth_patch_radius: int,
    T_ee_cam: np.ndarray,
) -> tuple[list[str], list[tuple[float, float, float] | None],
           list[tuple[float, float, float] | None]]:
    metric_lines: list[str] = []
    cams: list[tuple[float, float, float] | None] = []
    ees: list[tuple[float, float, float] | None] = []

    for p in points:
        u, v = denorm(p, w, h)
        d = sample_depth_median(depth_m, u, v, depth_patch_radius)
        if d is None:
            metric_lines.append("cam: (no depth)")
            cams.append(None)
            ees.append(None)
            continue
        xyz = deproject_pixel_to_cam(color_intrinsics, u, v, d)
        p_rs = np.array(xyz, dtype=np.float64).reshape(3)
        p_vis = realsense_optical_to_vision(p_rs)
        cam_t = (float(p_rs[0]), float(p_rs[1]), float(p_rs[2]))
        vis_t = (float(p_vis[0]), float(p_vis[1]), float(p_vis[2]))
        cams.append(vis_t)
        ep = transform_point_xyz(T_ee_cam, p_vis)
        ee_t = (float(ep[0]), float(ep[1]), float(ep[2]))
        ees.append(ee_t)
        parts = [
            f"rs_opt: ({cam_t[0]:.3f},{cam_t[1]:.3f},{cam_t[2]:.3f})",
            f"vision: ({vis_t[0]:.3f},{vis_t[1]:.3f},{vis_t[2]:.3f})",
            f"ee: ({ee_t[0]:.3f},{ee_t[1]:.3f},{ee_t[2]:.3f})",
        ]
        metric_lines.append("\n".join(parts))

    return metric_lines, cams, ees


def query_color_depth_overlay(
    client,
    model: str,
    prompt: str,
    temperature: float,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    color_intrinsics,
    T_ee_cam: np.ndarray,
    depth_patch_radius: int,
) -> tuple[np.ndarray | None, tuple[float, float, float] | None]:
    """Run Gemini on ``color_bgr``, return ``(overlay_bgr, first_ee_m)`` or ``(None, None)`` on API failure."""
    print("Sending frame to Gemini ER...")
    t0 = time.perf_counter()
    try:
        raw = call_gemini(
            client, model, encode_png(color_bgr), prompt, temperature=temperature
        )
    except Exception as e:
        print(f"  Gemini call failed: {e}", file=sys.stderr)
        return None, None
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Response ({dt_ms:.0f} ms): {raw.strip()}")

    try:
        points = parse_points(raw)
    except ValueError as e:
        print(f"  {e}", file=sys.stderr)
        points = []

    h, w = color_bgr.shape[:2]
    metric_lines: list[str] | None = None
    first_ee: tuple[float, float, float] | None = None
    if points:
        metric_lines, _, ees = lift_points_to_3d(
            points,
            w,
            h,
            depth_m,
            color_intrinsics,
            depth_patch_radius,
            T_ee_cam,
        )
        for e in ees:
            if e is not None:
                first_ee = e
                break
        for i, p in enumerate(points):
            u, v = denorm(p, w, h)
            line = f"    [{i}] {p.label!r}  px=({u},{v})"
            if metric_lines and i < len(metric_lines):
                line += f"  {metric_lines[i].replace(chr(10), ' ')}"
            print(line)

    overlay = draw_points(color_bgr, points, metric_lines=metric_lines)
    return overlay, first_ee
