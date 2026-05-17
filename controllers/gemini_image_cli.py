#!/usr/bin/env python3
"""One-shot Gemini Robotics-ER on a saved image (2D keypoints only, no Redis).

For live camera + depth + robot goals use ``vision_controller.py`` instead.

Usage::

  python ZitiBot/python_control/gemini_image_cli.py --image path/to/scene.jpg
  python ZitiBot/python_control/gemini_image_cli.py --image scene.jpg --object mug
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_PYTHON_CONTROL = Path(__file__).resolve().parent
if str(_PYTHON_CONTROL) not in sys.path:
    sys.path.insert(0, str(_PYTHON_CONTROL))

from vision import gemini_pointing as gp  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini ER pointing on one image file.")
    parser.add_argument("--image", type=Path, required=True, help="BGR image path.")
    parser.add_argument("--object", default="bowl", help="Object name for default prompt.")
    parser.add_argument("--prompt", default=None, help="Override full prompt.")
    parser.add_argument("--model", default=gp.DEFAULT_MODEL, help="Gemini model id.")
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 1

    print(
        "Note: no depth map; keypoints are 2D pixels only.",
        file=sys.stderr,
    )
    prompt = gp.build_prompt(args.object, args.prompt)
    client = gp.make_genai_client(gp.resolve_api_key())
    print(f"Querying {args.model} (image={args.image})...")
    t0 = time.perf_counter()
    raw = gp.call_gemini(
        client, args.model, gp.encode_png(img), prompt, temperature=args.temperature
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Response ({dt_ms:.0f} ms):\n{raw}")

    try:
        points = gp.parse_points(raw)
    except ValueError as e:
        print(f"{e}", file=sys.stderr)
        points = []
    if not points:
        print("No points returned.")
    else:
        for i, p in enumerate(points):
            u, v = gp.denorm(p, img.shape[1], img.shape[0])
            print(
                f"  [{i}] {p.label!r}  norm=({p.y_norm:.1f},{p.x_norm:.1f})  "
                f"px=({u},{v})"
            )

    overlay = gp.draw_points(img, points, metric_lines=None)
    out_name = f"gemini_{time.strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(out_name, overlay)
    print(f"Saved {out_name}")
    cv2.imshow("gemini ER points", overlay)
    print("Press any key in the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
