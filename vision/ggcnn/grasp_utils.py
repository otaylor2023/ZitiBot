"""Pre/post-processing helpers for GG-CNN2 inference on RealSense depth.

Pipeline:
    raw depth (uint16, mm)
        -> meters, in-paint holes
        -> center-crop to a square (crop_size px)
        -> resize to model input (300x300)
        -> subtract mean depth, clip to [-1, 1]
        -> torch tensor (1, 1, 300, 300)
    model -> (pos, cos, sin, width)
        -> gaussian-smooth pos
        -> argmax pixel (u_m, v_m) in 300x300
        -> theta = atan2(sin, cos) / 2 at that pixel (rad)
        -> width = width_map * 150 px at that pixel (model scale)
        -> map (u, v, width) back to full color image coords
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


MODEL_INPUT = 300
WIDTH_SCALE_PX = 150.0  # Upstream GG-CNN(2) clips/normalizes width to 150 px.


@dataclass
class CropInfo:
    """Crop + resize parameters needed to map model coords back to the full image."""
    x0: int   # left edge of the square crop in the source image
    y0: int   # top edge of the square crop in the source image
    size: int  # crop_size in source pixels (square)


def inpaint_depth(depth_m: np.ndarray) -> np.ndarray:
    """Fill zero/invalid depth holes with a quick OpenCV in-paint."""
    mask = (depth_m <= 0).astype(np.uint8)
    if mask.sum() == 0:
        return depth_m
    finite_max = float(depth_m[depth_m > 0].max()) if (depth_m > 0).any() else 1.0
    depth_u8 = np.clip(depth_m / max(finite_max, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    inpainted_u8 = cv2.inpaint(depth_u8, mask, 3, cv2.INPAINT_NS)
    out = depth_m.copy()
    fill = inpainted_u8.astype(np.float32) / 255.0 * finite_max
    out[mask.astype(bool)] = fill[mask.astype(bool)]
    return out


def center_square_crop(img: np.ndarray, size: int | None = None) -> tuple[np.ndarray, CropInfo]:
    """Center-crop ``img`` to a square of side ``size`` (defaults to the short side)."""
    h, w = img.shape[:2]
    short = min(h, w)
    side = short if size is None else min(size, short)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return img[y0:y0 + side, x0:x0 + side], CropInfo(x0=x0, y0=y0, size=side)


def preprocess_depth(depth_m: np.ndarray, crop_size: int | None = None) -> tuple[np.ndarray, CropInfo]:
    """Depth (meters) -> normalized model input (1, 1, 300, 300) float32."""
    filled = inpaint_depth(depth_m.astype(np.float32))
    cropped, info = center_square_crop(filled, crop_size)
    resized = cv2.resize(cropped, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_AREA)
    # GG-CNN training normalization: subtract per-image mean, clip to [-1, 1].
    normed = np.clip(resized - resized.mean(), -1.0, 1.0).astype(np.float32)
    return normed[None, None, :, :], info


@dataclass
class Grasp2D:
    """A 2D antipodal grasp in image coordinates."""
    u: float           # column (x) in the full color image
    v: float           # row (y) in the full color image
    theta: float       # rotation in radians (image-plane); 0 = horizontal gripper
    width_px: float    # opening width in pixels in the full color image
    quality: float     # peak quality at the grasp center, in [0, 1]


def postprocess(
    pos: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    width: np.ndarray,
    crop: CropInfo,
    gauss_sigma: float = 2.0,
) -> tuple[Grasp2D, np.ndarray]:
    """Convert GG-CNN2 outputs to the best grasp + the smoothed quality map.

    All input arrays are shape (300, 300). Returns the best grasp mapped back to
    full-image coords, and the (300x300) smoothed quality map for visualization.
    """
    q = gaussian_filter(pos, gauss_sigma)
    v_m, u_m = np.unravel_index(int(np.argmax(q)), q.shape)
    theta = float(math.atan2(float(sin[v_m, u_m]), float(cos[v_m, u_m])) / 2.0)
    width_model_px = float(width[v_m, u_m]) * WIDTH_SCALE_PX

    # Map back to the full image: model 300x300 -> crop square -> full frame.
    scale = crop.size / MODEL_INPUT
    u_full = u_m * scale + crop.x0
    v_full = v_m * scale + crop.y0
    width_full = width_model_px * scale

    return (
        Grasp2D(
            u=float(u_full),
            v=float(v_full),
            theta=theta,
            width_px=float(width_full),
            quality=float(q[v_m, u_m]),
        ),
        q,
    )


def heatmap_overlay(
    color_bgr: np.ndarray,
    quality_300: np.ndarray,
    crop: CropInfo,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a jet-colormapped quality heatmap over the color image (full frame)."""
    q = quality_300.copy()
    q_min, q_max = float(q.min()), float(q.max())
    if q_max > q_min:
        q = (q - q_min) / (q_max - q_min)
    else:
        q = np.zeros_like(q)
    q_u8 = (q * 255.0).astype(np.uint8)

    # Upsample heatmap from 300x300 back to crop_size, then place on a full-frame canvas.
    heat_crop = cv2.resize(q_u8, (crop.size, crop.size), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_crop, cv2.COLORMAP_JET)

    out = color_bgr.copy()
    roi = out[crop.y0:crop.y0 + crop.size, crop.x0:crop.x0 + crop.size]
    blended = cv2.addWeighted(roi, 1.0 - alpha, heat_color, alpha, 0.0)
    out[crop.y0:crop.y0 + crop.size, crop.x0:crop.x0 + crop.size] = blended

    # Outline the active region for clarity.
    cv2.rectangle(
        out,
        (crop.x0, crop.y0),
        (crop.x0 + crop.size - 1, crop.y0 + crop.size - 1),
        (255, 255, 255),
        1,
    )
    return out


def draw_grasp_rect(
    img: np.ndarray,
    grasp: Grasp2D,
    plate_color: tuple[int, int, int] = (0, 0, 255),
    side_color: tuple[int, int, int] = (0, 255, 0),
    plate_len_ratio: float = 0.5,
    thickness: int = 2,
) -> np.ndarray:
    """Draw an antipodal grasp rectangle on ``img`` (in place).

    The two ``plate_color`` lines mark the gripper plates (perpendicular to the
    approach axis); the two ``side_color`` lines span the gripper opening width.
    """
    cx, cy = grasp.u, grasp.v
    theta = grasp.theta
    w = grasp.width_px
    plate_len = max(w * plate_len_ratio, 10.0)

    # Unit vectors: along the approach axis (perpendicular to plates) and along the plates.
    ax_x, ax_y = math.cos(theta), math.sin(theta)
    pl_x, pl_y = -math.sin(theta), math.cos(theta)

    # Four rectangle corners
    p1 = (int(cx + ax_x * (w / 2) - pl_x * (plate_len / 2)),
          int(cy + ax_y * (w / 2) - pl_y * (plate_len / 2)))
    p2 = (int(cx + ax_x * (w / 2) + pl_x * (plate_len / 2)),
          int(cy + ax_y * (w / 2) + pl_y * (plate_len / 2)))
    p3 = (int(cx - ax_x * (w / 2) + pl_x * (plate_len / 2)),
          int(cy - ax_y * (w / 2) + pl_y * (plate_len / 2)))
    p4 = (int(cx - ax_x * (w / 2) - pl_x * (plate_len / 2)),
          int(cy - ax_y * (w / 2) - pl_y * (plate_len / 2)))

    cv2.line(img, p1, p2, plate_color, thickness)  # plate 1 (e.g., right gripper)
    cv2.line(img, p3, p4, plate_color, thickness)  # plate 2 (e.g., left gripper)
    cv2.line(img, p2, p3, side_color, thickness)   # opening side 1
    cv2.line(img, p4, p1, side_color, thickness)   # opening side 2
    cv2.circle(img, (int(cx), int(cy)), 3, (255, 255, 255), -1)
    return img
