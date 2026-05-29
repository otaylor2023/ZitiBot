"""GR-ConvNet (Generative Residual ConvNet) grasp predictor.

Vendored from https://github.com/skumra/robotic-grasping.
Paper:  Kumra, Joshi, Sahin, IROS 2020.
License: BSD-3-Clause.

Upstream pretrained checkpoints are saved with ``torch.save(model, ...)`` (full
pickled model, not state_dict). Unpickling needs the original module path
``inference.models.{grasp_model,grconvnet3}``; we alias it to this package so
``torch.load(..., weights_only=False)`` resolves correctly.
"""

from __future__ import annotations

import sys
import types

from . import grasp_model, grconvnet3
from .grconvnet3 import GenerativeResnet


_inference = sys.modules.setdefault("inference", types.ModuleType("inference"))
_models = sys.modules.setdefault("inference.models", types.ModuleType("inference.models"))
_inference.models = _models  # type: ignore[attr-defined]
sys.modules["inference.models.grasp_model"] = grasp_model
sys.modules["inference.models.grconvnet3"] = grconvnet3

__all__ = ["GenerativeResnet"]
