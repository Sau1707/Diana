"""Minimal feature-only interface for Hormonbench v1 models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from benchmark.v1_contracts import V1FitView, V1InferenceView
from benchmark.v1_task import HORMONES


class HormonbenchV1Model(ABC):
    @abstractmethod
    def fit(self, fit_view: V1FitView) -> "HormonbenchV1Model":
        """Fit only from the supplied feature/target/group view."""

    @abstractmethod
    def predict(self, inference_view: V1InferenceView) -> dict[str, np.ndarray]:
        """Return one nonnegative log1p vector per hormone."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return aggregate-safe model metadata."""

    @staticmethod
    def validate_prediction_dict(
        predictions: dict[str, np.ndarray], expected_rows: int
    ) -> None:
        if set(predictions) != set(HORMONES):
            raise ValueError("Predictions must contain exactly LH/E3G/PdG")
        for hormone, values in predictions.items():
            array = np.asarray(values, dtype=float)
            if len(array) != int(expected_rows):
                raise ValueError(f"{hormone} prediction length mismatch")
            if not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f"{hormone} predictions must be finite and nonnegative")
