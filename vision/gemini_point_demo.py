"""Live RealSense (or static image) + Gemini Robotics-ER pointing.

Sends a single RGB frame to Gemini Robotics-ER 1.5 with a pointing prompt and
overlays the returned points on the image. Default prompt asks for a point on
the bowl, but ``--object`` or ``--prompt`` can be used to ask for anything.

The model returns JSON of the form::

    [{"point": [y, x], "label": "bowl"}, ...]

where ``y`` and ``x`` are normalized to ``0-1000`` (note: y first). We rescale
to pixel coordinates of the source image and draw a marker + label.

Authentication: put ``GEMINI_API_KEY=...`` in a ``.env`` file at the repo root
(it is auto-loaded). ``GOOGLE_API_KEY`` and a real env var also work. Get a
key at https://aistudio.google.com/.

Usage:
  # Live RealSense (SPACE to capture and query, q/ESC to quit):
  python vision/gemini_point_demo.py
  python vision/gemini_point_demo.py --object "bowl"
  python vision/gemini_point_demo.py --prompt "Point to the rim of the bowl."

  # Or query a single saved image and exit:
  python vision/gemini_point_demo.py --image path/to/scene.jpg

Keys (live mode):
  SPACE  capture the current color frame and run Gemini ER
  s      save the latched result overlay as gemini_<timestamp>.png
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

DEFAULT_MODEL = "gemini-robotics-er-1.5-preview"


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
        f"Point to the {obj} in the image. "
        "The label returned should be an identifying name for the object detected. "
        "The answer should follow the json format: "
        '[{"point": [y, x], "label": <label>}, ...]. '
        "The points are in [y, x] format normalized to 0-1000."
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


def draw_points(img_bgr: np.ndarray, points: list[PointHit]) -> np.ndarray:
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
# RealSense plumbing (lazy import so --image mode works without librealsense)
# ---------------------------------------------------------------------------

def start_realsense(width: int, height: int, fps: int, warmup_frames: int,
                    timeout_ms: int):
    import pyrealsense2 as rs  # local import

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)

    print(f"Warming up ({warmup_frames} frames)...")
    for _ in range(warmup_frames):
        try:
            pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            pass
    return pipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini Robotics-ER pointing demo.")
    parser.add_argument(
        "--object",
        default="bowl",
        help="Object to point at (used by the default prompt). Default: bowl.",
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
        help="Skip the camera and run Gemini ER on this image file once.",
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

    overlay = draw_points(img, points)
    out_name = f"gemini_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(out_name, overlay)
    print(f"Saved {out_name}")
    cv2.imshow("gemini ER points", overlay)
    print("Press any key in the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def _query_and_overlay(
    client, model: str, prompt: str, temperature: float,
    color_bgr: np.ndarray,
) -> np.ndarray | None:
    """Send ``color_bgr`` to Gemini ER, log the result, and return an overlay.

    Returns ``None`` if the call failed; the loop should keep streaming.
    """
    print("Sending frame to Gemini ER...")
    t0 = time.perf_counter()
    try:
        raw = call_gemini(client, model, encode_png(color_bgr), prompt,
                          temperature=temperature)
    except Exception as e:  # network / API errors shouldn't kill the loop
        print(f"  Gemini call failed: {e}", file=sys.stderr)
        return None
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  Response ({dt_ms:.0f} ms): {raw.strip()}")

    try:
        points = parse_points(raw)
    except ValueError as e:
        print(f"  {e}", file=sys.stderr)
        points = []

    h, w = color_bgr.shape[:2]
    for i, p in enumerate(points):
        u, v = denorm(p, w, h)
        print(f"    [{i}] {p.label!r}  px=({u},{v})")

    return draw_points(color_bgr, points)


def _save_latched(latched: np.ndarray | None) -> None:
    if latched is None:
        print("Nothing to save yet -- press SPACE first.")
        return
    fname = f"gemini_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(fname, latched)
    print(f"Saved {fname}")


def _next_color_frame(
    pipeline, timeout_ms: int, miss_counter: list[int], max_misses: int,
) -> np.ndarray | None:
    """Pull one aligned color frame, or ``None`` if it wasn't ready.

    Raises :class:`TimeoutError` once ``max_misses`` consecutive timeouts
    occur, so the caller can abort cleanly.
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

    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())


def run_live(args: argparse.Namespace, prompt: str) -> int:
    client = make_genai_client(resolve_api_key())
    print(f"Using model: {args.model}")
    print(f"Prompt:\n  {prompt}")

    pipeline = start_realsense(
        args.width, args.height, args.fps, args.warmup_frames, args.timeout_ms
    )

    live_win = "RealSense (live)"
    result_win = "Gemini ER points"
    cv2.namedWindow(live_win, cv2.WINDOW_AUTOSIZE)

    latched: np.ndarray | None = None
    miss_counter = [0]

    try:
        while True:
            try:
                color_bgr = _next_color_frame(
                    pipeline, args.timeout_ms, miss_counter, max_misses=10
                )
            except TimeoutError as e:
                print(e, file=sys.stderr)
                return 2
            if color_bgr is None:
                continue

            cv2.imshow(live_win, color_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                overlay = _query_and_overlay(
                    client, args.model, prompt, args.temperature, color_bgr
                )
                if overlay is not None:
                    latched = overlay
                    cv2.imshow(result_win, latched)
            elif key == ord("s"):
                _save_latched(latched)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    prompt = build_prompt(args.object, args.prompt)
    if args.image is not None:
        return run_once_on_image(args, prompt)
    return run_live(args, prompt)


if __name__ == "__main__":
    sys.exit(main())
