"""Train-only population median baseline in target log1p space."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from benchmark.contracts import HORMONES, PreparedSplit
from model.base import HormonbenchModel
from model.common import make_prediction_frame, require_training_view


class PopulationMedianModel(HormonbenchModel):
    model_name = "population_median"
    model_version = "0.1.0"

    def __init__(self, config: dict[str, Any] | None = None, *, quick: bool = False):
        self.config = config or {}
        self.quick = quick
        self.medians: dict[str, float] = {}

    def fit(
        self, train_bundle: PreparedSplit, validation_bundle: PreparedSplit
    ) -> "PopulationMedianModel":
        require_training_view(train_bundle, "train_bundle")
        require_training_view(validation_bundle, "validation_bundle")
        self.medians = {
            hormone: float(train_bundle.target_log1p(hormone).median())
            for hormone in HORMONES
        }
        if not all(np.isfinite(value) and value >= 0 for value in self.medians.values()):
            raise ValueError("Train-only target medians must be finite and nonnegative")
        return self

    def predict(self, test_bundle: PreparedSplit) -> pd.DataFrame:
        if set(self.medians) != set(HORMONES):
            raise RuntimeError("Model has not been fit")
        values = {
            hormone: np.full(len(test_bundle.frame), median, dtype=float)
            for hormone, median in self.medians.items()
        }
        return make_prediction_frame(
            test_bundle,
            values,
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "target_space": "log1p",
            "predictor_count": 0,
            "fit_scope": "train-only median per hormone",
            "medians_log1p": dict(self.medians),
            "seed": None,
        }

