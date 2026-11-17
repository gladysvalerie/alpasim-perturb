# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

"""Model abstraction layer for trajectory prediction models.

Model classes are imported on first use, so a driver environment only needs the
packages of the model it actually runs. Alpamayo 2 Edge needs
``transformers==5.17.0`` while Alpamayo 1 / 1.5 / 2 Super pin 4.57.1, so Edge
runs in its own environment without those packages.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import (
    BaseTrajectoryModel,
    CameraFrame,
    CameraImages,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)

# Class name -> module (relative to this package), imported lazily.
_LAZY_MODELS = {
    "Alpamayo15Model": ".alpamayo1_5_model",
    "Alpamayo1Model": ".alpamayo1_model",
    "Alpamayo2Model": ".alpamayo2_model",
    "Alpamayo2EdgeModel": ".alpamayo2_edge_model",
    "ManualModel": ".manual_model",
    "VAMModel": ".vam_model",
}


def __getattr__(name: str) -> Any:
    module = _LAZY_MODELS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module, __name__), name)


__all__ = [
    "Alpamayo15Model",
    "Alpamayo1Model",
    "Alpamayo2EdgeModel",
    "Alpamayo2Model",
    "BaseTrajectoryModel",
    "CameraFrame",
    "CameraImages",
    "DriveCommand",
    "ManualModel",
    "ModelPrediction",
    "PredictionInput",
    "VAMModel",
]
