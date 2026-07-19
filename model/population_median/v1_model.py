"""Participant-equal v1 population-median baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from benchmark.v1_contracts import V1FitView, V1InferenceView
from benchmark.v1_task import HORMONES
from model.v1_base import HormonbenchV1Model


class PopulationMedianV1(HormonbenchV1Model):
    model_name = "population_median"
    model_version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.medians: dict[str, float] = {}

    def fit(self, fit_view: V1FitView) -> "PopulationMedianV1":
        fit_view.validate()
        groups = fit_view.participant_groups.astype(str).reset_index(drop=True)
        self.medians = {}
        for hormone in HORMONES:
            frame = pd.DataFrame(
                {"group": groups, "target": fit_view.targets[hormone].to_numpy(float)}
            )
            participant_medians = frame.groupby("group", sort=True)["target"].median()
            self.medians[hormone] = float(participant_medians.median())
        if not all(np.isfinite(value) and value >= 0 for value in self.medians.values()):
            raise ValueError("Participant-equal medians must be finite and nonnegative")
        return self

    def predict(self, inference_view: V1InferenceView) -> dict[str, np.ndarray]:
        inference_view.validate()
        if set(self.medians) != set(HORMONES):
            raise RuntimeError("Population median has not been fitted")
        predictions = {
            hormone: np.full(len(inference_view.X), self.medians[hormone], dtype=float)
            for hormone in HORMONES
        }
        self.validate_prediction_dict(predictions, len(inference_view.X))
        return predictions

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "participant_equal": True,
            "predictor_count": 0,
            "target_space": "log1p",
        }
