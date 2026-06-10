"""Task execution context shared across subtasks."""

from __future__ import annotations

import argparse
import select
import sys
from dataclasses import dataclass, field
from typing import Any

import redis

from zitibot_core.arm import print_startup_pose, try_redis, validate_config


@dataclass
class TaskContext:
    redis: redis.Redis
    step: bool = False
    tick_dt_s: float = 0.02
    gemini_client: Any | None = None
    gemini_model: str = "gemini-robotics-er-1.6-preview"
    gemini_temperature: float = 0.5
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30
    cam_warmup: int = 30
    cam_timeout_ms: int = 10000
    depth_patch_radius: int = 5
    gemini_response_path: str | None = None
    ee_from_cam_json: str | None = None
    endeffector_transform_key: str | None = None
    log: bool = False
    move_logger: Any | None = field(default=None, repr=False)
    _realsense_pipeline: Any | None = field(default=None, repr=False)
    _realsense_align: Any | None = field(default=None, repr=False)
    _realsense_depth_scale: float | None = field(default=None, repr=False)
    _realsense_intrinsics: Any | None = field(default=None, repr=False)
    _quit_requested: bool = field(default=False, repr=False)

    def q_pressed(self) -> bool:
        """Non-blocking check for ``q`` + ENTER on stdin."""
        if self._quit_requested:
            return True
        if not sys.stdin.isatty():
            return False
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (ValueError, OSError):
            return False
        if not ready:
            return False
        line = sys.stdin.readline()
        if line.strip().lower() in ("q", "quit", "exit"):
            self._quit_requested = True
            return True
        return False

    def p_pressed(self) -> bool:
        """Non-blocking check for ``p`` + ENTER on stdin (status print trigger)."""
        if not sys.stdin.isatty():
            return False
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (ValueError, OSError):
            return False
        if not ready:
            return False
        line = sys.stdin.readline()
        if line.strip().lower() == "p":
            return True
        if line.strip().lower() in ("q", "quit", "exit"):
            self._quit_requested = True
        return False

    def realsense(self):
        """Lazy-start RealSense pipeline; cached on context."""
        if self._realsense_pipeline is not None:
            return (
                self._realsense_pipeline,
                self._realsense_align,
                self._realsense_depth_scale,
                self._realsense_intrinsics,
            )
        from vision import realsense_rgbd as rs_cam

        pipeline, align, depth_scale, intrinsics = rs_cam.start_realsense(
            self.cam_width,
            self.cam_height,
            self.cam_fps,
            self.cam_warmup,
            self.cam_timeout_ms,
        )
        self._realsense_pipeline = pipeline
        self._realsense_align = align
        self._realsense_depth_scale = depth_scale
        self._realsense_intrinsics = intrinsics
        return pipeline, align, depth_scale, intrinsics

    def stop_realsense(self) -> None:
        if self._realsense_pipeline is not None:
            try:
                self._realsense_pipeline.stop()
            except Exception:
                pass
            self._realsense_pipeline = None
            self._realsense_align = None
            self._realsense_depth_scale = None
            self._realsense_intrinsics = None

    def restart_realsense(self):
        """Stop and re-start the RealSense pipeline; return fresh handles.

        Recovery hook for when the stream wedges (``try_wait_for_frames``
        returns nothing for the full timeout, which has been observed on a
        later detection in a multi-detection routine). A fresh
        ``start_realsense`` re-homes the device, re-runs warmup, and resumes
        streaming, so a single stalled grab no longer aborts the whole routine.
        """
        self.stop_realsense()
        return self.realsense()


def make_context(
    args: argparse.Namespace | None = None,
    *,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    step: bool = False,
    log: bool = False,
    print_startup: bool = True,
) -> TaskContext:
    if args is not None:
        redis_host = getattr(args, "redis_host", redis_host)
        redis_port = getattr(args, "redis_port", redis_port)
        step = getattr(args, "step", step)
        log = getattr(args, "log", log)
    client = try_redis(redis_host, redis_port)
    if client is None:
        raise SystemExit(1)
    err = validate_config(client)
    if err is not None:
        raise SystemExit(err)
    ctx = TaskContext(redis=client, step=step, log=log)
    if log:
        from zitibot_core.move_logger import maybe_make_logger

        ctx.move_logger = maybe_make_logger(True)
    if args is not None:
        for attr in (
            "gemini_model",
            "model",
            "temperature",
            "cam_width",
            "width",
            "cam_height",
            "height",
            "cam_fps",
            "fps",
            "cam_warmup",
            "warmup_frames",
            "cam_timeout_ms",
            "timeout_ms",
            "depth_patch_radius",
            "gemini_response_path",
            "ee_from_cam_json",
            "endeffector_transform_key",
        ):
            if hasattr(args, attr):
                val = getattr(args, attr)
                if val is None:
                    continue
                if attr == "model":
                    ctx.gemini_model = val
                elif attr == "width":
                    ctx.cam_width = val
                elif attr == "height":
                    ctx.cam_height = val
                elif attr == "fps":
                    ctx.cam_fps = val
                elif attr == "warmup_frames":
                    ctx.cam_warmup = val
                elif attr == "timeout_ms":
                    ctx.cam_timeout_ms = val
                elif attr == "temperature":
                    ctx.gemini_temperature = val
                elif hasattr(ctx, attr):
                    setattr(ctx, attr, val)
    if print_startup:
        print_startup_pose(ctx.redis)
    return ctx
