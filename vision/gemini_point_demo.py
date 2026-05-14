"""Live RealSense (or static image) + Gemini Robotics-ER pointing.

Sends a single RGB frame to Gemini Robotics-ER with a pointing prompt and
overlays the returned points on the image. The default prompt targets a
**graspable rim / lip** on the chosen object (e.g. bowl) for parallel-jaw pickup,
**biased toward the rim point nearest the robot** (see prompt for image convention);
``--object`` or ``--prompt`` override as needed.

The model returns JSON of the form::

    [{"point": [y, x], "label": "bowl"}, ...]

where ``y`` and ``x`` are normalized to ``0-1000`` (note: y first). We rescale
to pixel coordinates of the source image and draw a marker + label.

**Live mode**: **SPACE** runs Gemini, shows the overlay, and stores the lift for
Redis. **ENTER** publishes ``gemini_target_ee_*`` (if Redis is enabled). **SPACE**
again runs a **new** Gemini request. **s** saves the overlay PNG.

Live mode **requires** a working **depth** stream (aligned to color). The live
window shows **RGB | depth colormap** side by side. If depth frames never arrive
during warmup, the script **exits with an error** (no color-only fallback).

**``--image`` mode** has no depth map: output stays **2D pixels only**; 3D is
not available unless you extend the script with a paired depth image.

Authentication: put ``GEMINI_API_KEY=...`` in a ``.env`` file at the repo root
(it is auto-loaded). ``GOOGLE_API_KEY`` and a real env var also work. Get a
key at https://aistudio.google.com/.

Usage:
  # Live RealSense (SPACE to capture and query, q/ESC to quit):
  python vision/gemini_point_demo.py
  python vision/gemini_point_demo.py --object "bowl"
  python vision/gemini_point_demo.py --prompt "Point to the rim of the bowl."
  python vision/gemini_point_demo.py --ee-from-cam-json path/to/T_ee_cam.json

  python vision/gemini_point_demo.py --redis-host localhost --redis-port 6379
  python vision/gemini_point_demo.py --no-redis

  # Or query a single saved image and exit (2D keypoints only):
  python vision/gemini_point_demo.py --image path/to/scene.jpg

Keys (live mode):
  SPACE  run Gemini on the current frame and show overlay (stores result for Redis)
  ENTER  publish last SPACE result to Redis (EE position + rotation I + active)
  s      save the latched overlay as gemini_<timestamp>.png
  q/ESC  quit
"""

from __future__ import annotations

import argparse
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

# ---------------------------------------------------------------------------
# Default T_ee_cam (4×4, meters): RealSense optical → EE frame (e.g. link7).
# p_ee = T_ee_cam @ [x,y,z,1]^T. Assumes camera axes = EE axes (identity R).
# Translation: camera origin at +4 in EE X, +2 in EE Z from EE origin (user
# assumption). Override with --ee-from-cam-json or edit here.
# ---------------------------------------------------------------------------
_IN = 0.0254  # inches → meters
_T_X_M = 4.0 * _IN   # +4 inches along EE +X
_T_Z_M = 2.0 * _IN   # +2 inches along EE +Z
PLACEHOLDER_T_EE_CAM = np.array(
    [
        [1.0, 0.0, 0.0, _T_X_M],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, _T_Z_M],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(object_name: str | None, custom_prompt: str | None) -> str:
    """Return the prompt to send to Gemini ER.

    Either uses ``custom_prompt`` verbatim, or fills in a canonical pointing
    template targeting ``object_name`` (default: "bowl").
    """
    if custom_prompt:
        return custom_prompt
    obj = object_name or "bowl"
    return (
        f"In the image, locate the {obj}. Choose **one** point on the **outer rim / lip** "
        f"of the {obj} where a parallel-jaw gripper could make **stable contact** for a pick: "
        "not the flat bottom or deep interior of the bowl; avoid the table, fingers, or clutter. "
        "**Among** rim points that look graspable, choose the point **closest to the robot**. "
        "Use this image convention: normalized **y** grows **downward**; treat **larger y** "
        "(nearer the **bottom** of the image) as closer to the robot, as in a typical wrist camera "
        "looking outward. If the arm, base, or gripper is clearly visible along another edge of "
        "the frame, pick the rim point **closest to that robot side** instead. "
        "The label should name the object (e.g. \"bowl\"). "
        "Reply with JSON only in this form: "
        '[{"point": [y, x], "label": <label>}]. '
        "Use one entry unless the scene clearly has two separate graspable rims. "
        "Coordinates are [y, x] normalized 0-1000 (y first, then x)."
    )


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

@dataclass
class PointHit:
    y_norm: float  # 0..1000
    x_norm: float  # 0..1000
    label: str


def _maybe_load_dotenv() -> None:
    """Best-effort load of a sibling .env file. Silent no-op without python-dotenv."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Look for a .env next to the script or in the repo root (parent of vision/).
    here = Path(__file__).resolve().parent
    for candidate in (here / ".env", here.parent / ".env"):
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
    """Send (image, prompt) to Gemini ER and return the raw text response."""
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_png_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            # thinking_budget=0 gives the lowest latency; bump if accuracy
            # disappoints on cluttered scenes.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text or ""


def parse_points(text: str) -> list[PointHit]:
    """Pull a JSON array of {point, label} entries out of the model response.

    The model usually returns clean JSON, but sometimes wraps it in
    ```json ... ``` fences or includes commentary; we try a few strategies.
    """
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
    """Convert parsed JSON (list of dicts, or single dict) into PointHit list."""
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


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

POINT_COLORS_BGR = [
    (0, 255, 0),     # green
    (0, 200, 255),   # amber
    (255, 0, 255),   # magenta
    (255, 200, 0),   # cyan-blue
    (0, 0, 255),     # red
    (255, 255, 0),   # cyan
]


def denorm(p: PointHit, w: int, h: int) -> tuple[int, int]:
    """Map normalized (0-1000) [y, x] to pixel (u, v)."""
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
    """Draw keypoints; optional ``metric_lines[i]`` is extra text below the pixel line."""
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
        cv2.putText(out, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
        if metric_lines and i < len(metric_lines) and metric_lines[i]:
            y2 = text_org[1] + 16
            for line in metric_lines[i].split("\n"):
                cv2.putText(out, line, (text_org[0], y2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, line, (text_org[0], y2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, color, 1, cv2.LINE_AA)
                y2 += 14
    return out


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_png(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image.")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Depth deprojection + T_ee_cam (camera → end-effector)
# ---------------------------------------------------------------------------

def load_T_ee_cam(path: Path) -> np.ndarray:
    """Load 4×4 homogeneous ``T_ee_cam`` (maps camera coords → EE frame).

    JSON accepts ``T_ee_cam`` or ``matrix``: nested 4×4 list, or 16 floats
    row-major in a single list.
    """
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
            raise ValueError(
                f"JSON must contain 'T_ee_cam' or 'matrix' key: {path}"
            )
    else:
        raise ValueError(f"Unexpected JSON root type in {path}")
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"Expected 4×4 matrix, got shape {arr.shape} in {path}")


def transform_point_xyz(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply 4×4 ``T`` to 3-vector ``p`` (homogeneous); returns shape (3,)."""
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = T @ ph
    w = float(out[3])
    if abs(w) < 1e-12:
        return out[:3].copy()
    return (out[:3] / w).astype(np.float64)


def sample_depth_median(
    depth_m: np.ndarray, u: int, v: int, radius: int,
) -> float | None:
    """Median depth (meters) in a (2*radius+1)² window; ``None`` if no valid samples."""
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
    """For each keypoint, median depth → camera XYZ → EE XYZ via ``T_ee_cam``.

    Returns ``(metric_lines, cam_xyz_list, ee_xyz_list)`` for logging/draw.
    """
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
        cam_t = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        cams.append(cam_t)
        ep = transform_point_xyz(T_ee_cam, np.array(cam_t, dtype=np.float64))
        ee_t = (float(ep[0]), float(ep[1]), float(ep[2]))
        ees.append(ee_t)
        parts = [
            f"cam: ({cam_t[0]:.3f},{cam_t[1]:.3f},{cam_t[2]:.3f})",
            f"ee: ({ee_t[0]:.3f},{ee_t[1]:.3f},{ee_t[2]:.3f})",
        ]
        metric_lines.append("\n".join(parts))

    return metric_lines, cams, ees


# ---------------------------------------------------------------------------
# Redis (SaiCommon Eigen JSON — keep keys in sync with controller_touch / redis_keys.h)
# Default: tidybot01::gemini_* (hardware). ``--redis-gemini-sim-keys``: sai::sim::... for sim-only stacks.


@dataclass(frozen=True)
class GeminiRedisKeys:
    position: str
    rotation: str
    active: str


GEMINI_KEYS_HARDWARE = GeminiRedisKeys(
    position="tidybot01::gemini_target_ee_position",
    rotation="tidybot01::gemini_target_ee_rotation",
    active="tidybot01::gemini_target_ee_active",
)
GEMINI_KEYS_SIM = GeminiRedisKeys(
    position="sai::sim::mmp_panda::desire::gemini_target_ee_position",
    rotation="sai::sim::mmp_panda::desire::gemini_target_ee_rotation",
    active="sai::sim::mmp_panda::desire::gemini_target_ee_active",
)

_gemini_redis_keys: GeminiRedisKeys = GEMINI_KEYS_HARDWARE


def sai_encode_eigen_column_vector(v: np.ndarray) -> str:
    """Match ``SaiCommon::RedisClient::encodeEigenMatrix`` for a column vector."""
    flat = np.asarray(v, dtype=np.float64).ravel()
    parts = [str(float(x)) for x in flat]
    return "[" + ",".join(parts) + "]"


def sai_encode_eigen_matrix_3x3(R: np.ndarray) -> str:
    """Match ``encodeEigenMatrix`` for a 3×3 matrix (nested row brackets)."""
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    rows: list[str] = []
    for i in range(3):
        parts = [str(float(m[i, j])) for j in range(3)]
        rows.append("[" + ",".join(parts) + "]")
    return "[" + ",".join(rows) + "]"


def publish_gemini_ee_desire(redis_client, pos_ee: tuple[float, float, float] | None) -> None:
    """Publish EE-frame goal (m): position + identity rotation + active flag.

    ``pos_ee`` None → active 0 only (invalid / no depth). Else active 1 and
    full pose; rotation is identity until vision supplies orientation.
    """
    keys = _gemini_redis_keys
    if pos_ee is None:
        redis_client.set(
            keys.active,
            sai_encode_eigen_column_vector(np.array([0.0])),
        )
        print("  Redis: gemini_target_ee_active=0 (no valid EE point).")
        return
    pos = np.array(pos_ee, dtype=np.float64).reshape(3)
    R = np.eye(3, dtype=np.float64)
    pipe = redis_client.pipeline(transaction=True)
    pipe.set(keys.position, sai_encode_eigen_column_vector(pos))
    pipe.set(keys.rotation, sai_encode_eigen_matrix_3x3(R))
    pipe.set(keys.active, sai_encode_eigen_column_vector(np.array([1.0])))
    pipe.execute()
    print(
        f"  Redis: published {keys.position} = "
        f"{pos.tolist()} (EE m), rotation=I, active=1."
    )


def try_connect_redis(host: str, port: int):
    """Return a ``redis.Redis`` client or ``None`` if unavailable."""
    try:
        import redis as redis_mod
    except ImportError:
        print(
            "redis package not installed; skipping Redis publish. "
            "`pip install redis` to enable.",
            file=sys.stderr,
        )
        return None
    try:
        r = redis_mod.Redis(host=host, port=port, decode_responses=True)
        r.ping()
        return r
    except Exception as e:
        print(f"Redis connect failed ({e}); not publishing.", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# RealSense plumbing (lazy import so --image mode works without librealsense)
# ---------------------------------------------------------------------------

def depth_uint16_to_colormap_bgr(depth_u16: np.ndarray) -> np.ndarray:
    """BGR colormap for raw Z16 depth (same idea as ``vision/test_camera.py``)."""
    return cv2.applyColorMap(
        cv2.convertScaleAbs(depth_u16, alpha=0.03), cv2.COLORMAP_JET
    )


def start_realsense(
    width: int,
    height: int,
    fps: int,
    warmup_frames: int,
    timeout_ms: int,
):
    """Start color+depth, align depth→color; return pipeline, align, depth_scale, intrinsics.

    Warmup uses ``try_wait_for_frames`` only (never blocking ``wait_for_frames``).
    If fewer than ``warmup_frames`` valid RGB-D frame pairs are received, raises
    ``SystemExit`` — live mode does not run without depth.
    """
    import pyrealsense2 as rs  # local import

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intrinsics = color_profile.get_intrinsics()
    print(f"Depth scale: {depth_scale:.6f} m/unit")

    print(
        f"Warming up color+depth (need {warmup_frames} aligned frames, "
        f"{timeout_ms} ms try_wait each)..."
    )
    n_ok = 0
    max_tries = max(warmup_frames * 20, warmup_frames + 60)
    report_every = 30
    for attempt in range(max_tries):
        ok, frames = pipeline.try_wait_for_frames(timeout_ms)
        if not ok:
            if attempt > 0 and attempt % report_every == 0:
                print(
                    f"  ... still waiting ({n_ok}/{warmup_frames} ok, "
                    f"attempt {attempt}/{max_tries})"
                )
            continue
        frames = align.process(frames)
        if not frames.get_color_frame() or not frames.get_depth_frame():
            continue
        n_ok += 1
        if n_ok >= warmup_frames:
            break

    if n_ok < warmup_frames:
        pipeline.stop()
        raise SystemExit(
            f"RealSense depth required but warmup only got {n_ok}/{warmup_frames} "
            "aligned RGB-D frames. Check USB3, cable/port, power, and that no other "
            "process holds the camera. Try lower --fps or resolution."
        )
    print(f"Warmup done ({n_ok} aligned frames).")

    return pipeline, align, depth_scale, color_intrinsics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini Robotics-ER pointing demo.")
    parser.add_argument(
        "--object",
        default="bowl",
        help="Object name in the default grasp-rim prompt (e.g. bowl). Default: bowl.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the entire prompt sent to Gemini ER. Must instruct the "
             'model to reply with JSON [{"point": [y,x], "label": ...}, ...].',
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Skip the camera and run Gemini ER on this image file once. "
             "No depth: only 2D pixel keypoints (no camera/EE XYZ).",
    )
    parser.add_argument(
        "--ee-from-cam-json",
        type=Path,
        default=None,
        help="Optional JSON with 4×4 T_ee_cam (camera optical → EE / link7, m). "
             "Live mode only; if omitted, uses PLACEHOLDER_T_EE_CAM in the script.",
    )
    parser.add_argument(
        "--depth-patch-radius",
        type=int,
        default=2,
        help="Half-size in pixels of the depth median window (default 2 → 5×5).",
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis host for publishing gemini_target_ee_* (on ENTER). Default: localhost.",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port. Default: 6379.",
    )
    parser.add_argument(
        "--no-redis",
        action="store_true",
        help="Do not connect to Redis (ENTER will not publish).",
    )
    parser.add_argument(
        "--redis-gemini-sim-keys",
        action="store_true",
        help="Publish gemini EE desire to sai::sim::mmp_panda::desire::gemini_* "
             "(sim / redis_keys_sim.h). Default: tidybot01:: keys (hardware touch).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model id. Default: {DEFAULT_MODEL}.")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    return parser.parse_args()


def run_once_on_image(args: argparse.Namespace, prompt: str) -> int:
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 1

    print(
        "Note: --image mode has no depth map; keypoints are 2D pixels only. "
        "Use live RealSense for camera/EE XYZ.",
        file=sys.stderr,
    )
    if args.ee_from_cam_json is not None:
        print(
            "Ignoring --ee-from-cam-json in --image mode (no depth).",
            file=sys.stderr,
        )

    client = make_genai_client(resolve_api_key())
    print(f"Querying {args.model} (image={args.image})...")
    t0 = time.perf_counter()
    raw = call_gemini(client, args.model, encode_png(img), prompt,
                      temperature=args.temperature)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Response ({dt_ms:.0f} ms):\n{raw}")

    points = parse_points(raw)
    if not points:
        print("No points returned.")
    else:
        for i, p in enumerate(points):
            u, v = denorm(p, img.shape[1], img.shape[0])
            print(f"  [{i}] {p.label!r}  norm=({p.y_norm:.1f},{p.x_norm:.1f})  "
                  f"px=({u},{v})")

    overlay = draw_points(img, points, metric_lines=None)
    out_name = f"gemini_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(out_name, overlay)
    print(f"Saved {out_name}")
    cv2.imshow("gemini ER points", overlay)
    print("Press any key in the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def _query_and_overlay(
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
    """Send ``color_bgr`` to Gemini ER, log the result, return ``(overlay, first_ee)``.

    ``first_ee`` is the first keypoint with valid depth (EE frame, m), or
    ``None`` if none. Publish to Redis separately (e.g. on ENTER).

    Returns ``(None, None)`` if the Gemini API call failed.
    """
    print("Sending frame to Gemini ER...")
    t0 = time.perf_counter()
    try:
        raw = call_gemini(client, model, encode_png(color_bgr), prompt,
                          temperature=temperature)
    except Exception as e:  # network / API errors shouldn't kill the loop
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


def _save_latched(latched: np.ndarray | None) -> None:
    if latched is None:
        print("Nothing to save yet -- press SPACE first.")
        return
    fname = f"gemini_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(fname, latched)
    print(f"Saved {fname}")


def _next_rgbd_frame(
    pipeline,
    align,
    depth_scale: float,
    timeout_ms: int,
    miss_counter: list[int],
    max_misses: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Aligned color BGR, depth in meters, and depth BGR colormap for preview.

    Returns ``(color_bgr, depth_m, depth_vis_bgr)`` or ``None`` on miss.

    Raises :class:`TimeoutError` once ``max_misses`` consecutive timeouts occur.
    """
    ok, frames = pipeline.try_wait_for_frames(timeout_ms)
    if not ok:
        miss_counter[0] += 1
        print(
            f"Frame didn't arrive within {timeout_ms} ms "
            f"(miss {miss_counter[0]}/{max_misses})."
        )
        if miss_counter[0] >= max_misses:
            raise TimeoutError(
                "Too many consecutive timeouts. Check USB 3 connection, "
                "try a different port/cable, or lower --fps / --width / --height."
            )
        return None
    miss_counter[0] = 0

    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    color_bgr = np.asanyarray(color_frame.get_data())
    depth_u16 = np.asanyarray(depth_frame.get_data())
    depth_m = depth_u16.astype(np.float32) * depth_scale
    depth_vis = depth_uint16_to_colormap_bgr(depth_u16)
    return color_bgr, depth_m, depth_vis


def run_live(args: argparse.Namespace, prompt: str) -> int:
    client = make_genai_client(resolve_api_key())
    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")

    if args.ee_from_cam_json is not None:
        T_ee_cam = load_T_ee_cam(args.ee_from_cam_json)
        print(f"Loaded T_ee_cam from {args.ee_from_cam_json}")
    else:
        T_ee_cam = PLACEHOLDER_T_EE_CAM.copy()
        print(
            "Using built-in T_ee_cam: camera origin +4 in EE X, +2 in EE Z "
            f"({_T_X_M:.4f}, 0, {_T_Z_M:.4f}) m, R=I. Override with "
            "--ee-from-cam-json."
        )

    redis_client = None
    if not args.no_redis:
        redis_client = try_connect_redis(args.redis_host, args.redis_port)
        if redis_client is not None:
            print(
                f"Redis: publish on ENTER after SPACE "
                f"({args.redis_host}:{args.redis_port})."
            )

    pipeline = None
    try:
        pipeline, align, depth_scale, color_intrinsics = start_realsense(
            args.width, args.height, args.fps, args.warmup_frames, args.timeout_ms
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"RealSense startup failed: {e}", file=sys.stderr)
        return 1

    live_win = "RealSense RGB | depth (live)"
    result_win = "Gemini ER points"
    cv2.namedWindow(live_win, cv2.WINDOW_AUTOSIZE)

    latched: np.ndarray | None = None
    miss_counter = [0]
    pending_first_ee: tuple[float, float, float] | None = None
    have_pending_publish: bool = False
    print(
        "Keys: SPACE = Gemini + overlay | ENTER = publish EE to Redis | "
        "s = save overlay | q = quit\n"
        "Live preview: left = color, right = depth colormap (aligned)."
    )

    try:
        while True:
            try:
                triple = _next_rgbd_frame(
                    pipeline,
                    align,
                    depth_scale,
                    args.timeout_ms,
                    miss_counter,
                    max_misses=10,
                )
            except TimeoutError as e:
                print(e, file=sys.stderr)
                return 2
            if triple is None:
                continue
            color_bgr, depth_m, depth_vis = triple
            preview = np.hstack((color_bgr, depth_vis))
            cv2.imshow(live_win, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                overlay, first_ee = _query_and_overlay(
                    client,
                    args.model,
                    prompt,
                    args.temperature,
                    color_bgr,
                    depth_m,
                    color_intrinsics,
                    T_ee_cam,
                    args.depth_patch_radius,
                )
                if overlay is not None:
                    latched = overlay
                    pending_first_ee = first_ee
                    have_pending_publish = True
                    cv2.imshow(result_win, latched)
            elif key in (10, 13):  # LF / CR (ENTER)
                if redis_client is None:
                    print("Redis disabled; nothing to publish.")
                elif not have_pending_publish:
                    print("Press SPACE first (Gemini + overlay) before ENTER to publish.")
                else:
                    publish_gemini_ee_desire(redis_client, pending_first_ee)
            elif key == ord("s"):
                _save_latched(latched)
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    global _gemini_redis_keys
    args = parse_args()
    _gemini_redis_keys = GEMINI_KEYS_SIM if args.redis_gemini_sim_keys else GEMINI_KEYS_HARDWARE
    prompt = build_prompt(args.object, args.prompt)
    if args.image is not None:
        return run_once_on_image(args, prompt)
    return run_live(args, prompt)


if __name__ == "__main__":
    sys.exit(main())
