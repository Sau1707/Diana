"""Transparent participant-balanced wearable Ridge baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from benchmark.v1_contracts import V1FitView, V1InferenceView
from benchmark.v1_task import HORMONES
from model.v1_base import HormonbenchV1Model
from model.v1_common import FeaturePreprocessor, participant_balanced_weights


class WearableRidgeV1(HormonbenchV1Model):
    model_name = "wearable_ridge"
    model_version = "1.0.0"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.alpha = float(config["models"]["wearable_ridge"]["alpha"])
        threshold = float(config["preprocessing"]["missingness_drop_threshold"])
        self.preprocessor = FeaturePreprocessor(
            missingness_drop_threshold=threshold, standardize=True
        )
        self.models: dict[str, Ridge] = {}

    def fit(self, fit_view: V1FitView) -> "WearableRidgeV1":
        fit_view.validate()
        self.preprocessor.fit(fit_view.X, fit_view.participant_groups)
        X = self.preprocessor.transform(fit_view.X)
        weights = participant_balanced_weights(fit_view.participant_groups)
        self.models = {}
        for hormone in HORMONES:
            estimator = Ridge(alpha=self.alpha)
            estimator.fit(X, fit_view.targets[hormone].to_numpy(float), sample_weight=weights)
            self.models[hormone] = estimator
        return self

    def predict(self, inference_view: V1InferenceView) -> dict[str, np.ndarray]:
        inference_view.validate()
        if set(self.models) != set(HORMONES):
            raise RuntimeError("Wearable Ridge has not been fitted")
        X = self.preprocessor.transform(inference_view.X)
        predictions = {
            hormone: np.maximum(self.models[hormone].predict(X), 0.0)
            for hormone in HORMONES
        }
        self.validate_prediction_dict(predictions, len(X))
        return predictions

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "alpha": self.alpha,
            "participant_balanced_sample_weight": True,
            "one_regressor_per_hormone": True,
            "preprocessor": self.preprocessor.metadata(),
            "target_space": "log1p",
        }
