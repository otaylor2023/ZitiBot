"""Grasp-predictor abstraction.

Both supported models (GG-CNN2 and GR-ConvNet) are heatmap predictors that emit
``(pos, cos, sin, width)`` maps. They differ in:
  - input modality (depth-only vs RGB-D),
  - input size (300x300 vs 224x224),
  - checkpoint format (state_dict vs full pickled model).

This module wraps each behind a uniform ``predict(color_bgr, depth_m)`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from grasp_utils import (
    CropInfo,
    Grasp2D,
    postprocess,
    preprocess_depth,
    preprocess_rgb,
)


@dataclass
class GraspPrediction:
    """Output of :meth:`Predictor.predict`."""
    grasp: Grasp2D
    quality_map: np.ndarray  # smoothed Q in model space (e.g. 300x300 or 224x224)
    crop: CropInfo            # crop info needed to place the heatmap back on the full frame


class Predictor(Protocol):
    name: str
    input_size: int
    device: torch.device

    def predict(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        crop_size: int | None = None,
        bottom_exclude_px: int = 0,
        top_exclude_px: int = 0,
    ) -> GraspPrediction: ...


# ---------------------------------------------------------------------------
# GG-CNN2: depth-only, 300x300, state_dict checkpoint.
# ---------------------------------------------------------------------------

DEFAULT_GGCNN2_WEIGHTS = (
    Path(__file__).resolve().parent / "ggcnn" / "weights" / "ggcnn2_cornell_statedict.pt"
)


class GGCNN2Predictor:
    name = "ggcnn2"
    input_size = 300

    def __init__(self, weights_path: Path, device: torch.device) -> None:
        from ggcnn import GGCNN2

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"GG-CNN2 weights not found at {weights_path}. "
                "Run python_control/vision/ggcnn/weights/download_weights.sh first."
            )
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model = GGCNN2()
        model.load_state_dict(state)
        model.to(device).eval()
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        crop_size: int | None = None,
        bottom_exclude_px: int = 0,
        top_exclude_px: int = 0,
    ) -> GraspPrediction:
        depth_plane, crop = preprocess_depth(
            depth_m,
            crop_size=crop_size,
            model_input=self.input_size,
            bottom_exclude_px=bottom_exclude_px,
            top_exclude_px=top_exclude_px,
        )
        x = torch.from_numpy(depth_plane[None, None, :, :]).to(self.device)
        pos, cos, sin, width = self.model(x)
        grasp, q_map = postprocess(
            pos.squeeze().cpu().numpy(),
            cos.squeeze().cpu().numpy(),
            sin.squeeze().cpu().numpy(),
            width.squeeze().cpu().numpy(),
            crop,
        )
        return GraspPrediction(grasp=grasp, quality_map=q_map, crop=crop)


# ---------------------------------------------------------------------------
# GR-ConvNet: RGB-D, 224x224, full-pickle checkpoint.
# ---------------------------------------------------------------------------

DEFAULT_GRCONVNET_WEIGHTS = (
    Path(__file__).resolve().parent / "grconvnet" / "weights" / "grconvnet3_cornell_rgbd.pt"
)


class GRConvNetPredictor:
    name = "grconvnet"
    input_size = 224

    def __init__(self, weights_path: Path, device: torch.device) -> None:
        # Importing the package installs the sys.modules aliases needed to
        # unpickle the upstream full-model checkpoint.
        import grconvnet  # noqa: F401

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"GR-ConvNet weights not found at {weights_path}. "
                "Run python_control/vision/grconvnet/weights/download_weights.sh first."
            )
        # Upstream saves the full model object, so weights_only=False is required.
        model = torch.load(weights_path, map_location=device, weights_only=False)
        model.to(device).eval()
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        crop_size: int | None = None,
        bottom_exclude_px: int = 0,
        top_exclude_px: int = 0,
    ) -> GraspPrediction:
        depth_plane, crop = preprocess_depth(
            depth_m,
            crop_size=crop_size,
            model_input=self.input_size,
            bottom_exclude_px=bottom_exclude_px,
            top_exclude_px=top_exclude_px,
        )
        rgb_planes = preprocess_rgb(color_bgr, crop, model_input=self.input_size)
        # Upstream channel order: [depth, R, G, B].
        rgbd = np.concatenate([depth_plane[None, :, :], rgb_planes], axis=0).astype(np.float32)
        x = torch.from_numpy(rgbd[None, ...]).to(self.device)
        pos, cos, sin, width = self.model(x)
        grasp, q_map = postprocess(
            pos.squeeze().cpu().numpy(),
            cos.squeeze().cpu().numpy(),
            sin.squeeze().cpu().numpy(),
            width.squeeze().cpu().numpy(),
            crop,
        )
        return GraspPrediction(grasp=grasp, quality_map=q_map, crop=crop)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PREDICTORS: dict[str, type[Predictor]] = {
    "ggcnn2": GGCNN2Predictor,
    "grconvnet": GRConvNetPredictor,
}

_DEFAULT_WEIGHTS: dict[str, Path] = {
    "ggcnn2": DEFAULT_GGCNN2_WEIGHTS,
    "grconvnet": DEFAULT_GRCONVNET_WEIGHTS,
}


def available_models() -> list[str]:
    return list(_PREDICTORS.keys())


def default_weights(name: str) -> Path:
    if name not in _DEFAULT_WEIGHTS:
        raise ValueError(f"unknown model {name!r}; choices: {available_models()}")
    return _DEFAULT_WEIGHTS[name]


def make_predictor(
    name: str,
    weights_path: Path | None,
    device: torch.device,
) -> Predictor:
    if name not in _PREDICTORS:
        raise ValueError(f"unknown model {name!r}; choices: {available_models()}")
    weights = weights_path or _DEFAULT_WEIGHTS[name]
    return _PREDICTORS[name](weights, device)
