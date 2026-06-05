#!/usr/bin/env python3
"""Serve the RealSense feed as MJPEG over HTTP (view in a browser over SSH).

No GUI / window viewer needed — handy on a headless SSH session. Opens the
RealSense, JPEG-encodes each frame, and serves a ``multipart/x-mixed-replace``
stream that any browser renders live.

Usage::

  # On the robot (the RealSense must be free — not held by a vision controller):
  python ZitiBot/controllers/vision/camera_web_stream.py --port 8000

  # From your laptop, forward the port over SSH, then open the URL:
  ssh -L 8000:localhost:8000 tidybot01
  #   browser -> http://localhost:8000/

Options:
  --depth        also show the aligned depth colormap (side-by-side)
  --width/--height/--fps   stream format (default 640x480@30)
  --jpeg-quality N         JPEG quality 1-100 (default 80)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pyrealsense2 as rs

_BOUNDARY = "frame"

# Latest encoded JPEG, shared between the capture thread and HTTP handlers.
_latest_jpeg: bytes | None = None
_latest_lock = threading.Lock()
_stop = threading.Event()


def _depth_colormap_bgr(depth_u16: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_u16, alpha=0.03), cv2.COLORMAP_JET)


def capture_loop(args: argparse.Namespace) -> None:
    """Open the RealSense and continuously encode the latest frame to JPEG."""
    global _latest_jpeg

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if args.depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color) if args.depth else None
    print(f"RealSense started ({args.width}x{args.height}@{args.fps}, depth={args.depth}).")

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(args.jpeg_quality, 1, 100))]
    miss = 0
    try:
        while not _stop.is_set():
            ok, frames = pipeline.try_wait_for_frames(args.timeout_ms)
            if not ok:
                miss += 1
                if miss >= 10:
                    print("Too many frame timeouts; check USB3/cable/port.", file=sys.stderr)
                    break
                continue
            miss = 0
            if align is not None:
                frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            display = color
            if args.depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_u16 = np.asanyarray(depth_frame.get_data())
                    display = np.hstack((color, _depth_colormap_bgr(depth_u16)))
            ok_enc, buf = cv2.imencode(".jpg", display, encode_params)
            if not ok_enc:
                continue
            with _latest_lock:
                _latest_jpeg = buf.tobytes()
    finally:
        pipeline.stop()
        _stop.set()
        print("RealSense stopped.")


_PAGE = (
    b"<!doctype html><html><head><title>RealSense stream</title>"
    b"<style>body{margin:0;background:#111}img{width:100%;height:auto;display:block}</style>"
    b"</head><body><img src='/stream.mjpg'></body></html>"
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:  # quiet the per-request logging
        pass

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_PAGE)))
            self.end_headers()
            self.wfile.write(_PAGE)
            return
        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
        )
        self.end_headers()
        try:
            while not _stop.is_set():
                with _latest_lock:
                    jpeg = _latest_jpeg
                if jpeg is None:
                    time.sleep(0.02)
                    continue
                self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.0)
        except (BrokenPipeError, ConnectionResetError):
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve RealSense as MJPEG over HTTP.")
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default all interfaces).")
    p.add_argument("--port", type=int, default=8000, help="HTTP port (default 8000).")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--depth", action="store_true", help="Also show aligned depth colormap.")
    p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality 1-100.")
    p.add_argument("--timeout-ms", type=int, default=10000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    devices = list(rs.context().query_devices())
    if not devices:
        print("No RealSense devices found. Is the camera plugged in / free?", file=sys.stderr)
        return 1

    cap = threading.Thread(target=capture_loop, args=(args,), daemon=True)
    cap.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving MJPEG at http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    print(
        f"  SSH forward from your laptop:  ssh -L {args.port}:localhost:{args.port} <this-host>\n"
        f"  then open:  http://localhost:{args.port}/"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        _stop.set()
        server.shutdown()
        cap.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
