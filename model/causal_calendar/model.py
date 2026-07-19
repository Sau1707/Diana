"""Transparent causal menstrual-calendar baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from benchmark.contracts import HORMONES, PreparedSplit
from model.base import HormonbenchModel
from model.common import make_prediction_frame, require_training_view


class CausalCalendarModel(HormonbenchModel):
    """Ridge on known-past menstrual timing, never completed-cycle state."""

    model_name = "causal_calendar"
    model_version = "0.1.0"

    def __init__(self, config: dict[str, Any] | None = None, *, quick: bool = False):
        self.config = config or {}
        model_config = self.config.get("models", {}).get("causal_calendar", {})
        self.alpha = float(model_config.get("ridge_alpha", 1.0))
        self.period = float(model_config.get("harmonic_period_days", 28.0))
        self.degree = int(model_config.get("polynomial_degree", 2))
        if self.degree not in {1, 2, 3}:
            raise ValueError("causal_calendar polynomial_degree must be 1, 2, or 3")
        if self.period <= 0:
            raise ValueError("harmonic period must be positive")
        self.quick = quick
        self.calendar_column: str | None = None
        self.impute_day: float | None = None
        self.scaler = StandardScaler()
        self.models: dict[str, Ridge] = {}

    @staticmethod
    def _resolve_calendar_column(feature_columns: tuple[str, ...]) -> str:
        exact = "days_since_last_known_menses"
        if exact in feature_columns:
            return exact
        candidates = [
            column
            for column in feature_columns
            if column.endswith(exact) or exact in column
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Prepared features must include one unambiguous "
                "days_since_last_known_menses column"
            )
        return candidates[0]

    def _design(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        if self.calendar_column is None:
            raise RuntimeError("Calendar feature is unresolved")
        day = pd.to_numeric(frame[self.calendar_column], errors="coerce").to_numpy(float)
        missing = ~np.isfinite(day)
        if fit:
            observed = day[~missing]
            self.impute_day = float(np.median(observed)) if len(observed) else 0.0
        if self.impute_day is None:
            raise RuntimeError("Calendar imputer has not been fit")
        filled = np.where(missing, self.impute_day, day)
        columns = [missing.astype(float), filled]
        for power in range(2, self.degree + 1):
            columns.append(np.power(filled, power))
        angle = 2.0 * np.pi * filled / self.period
        columns.extend([np.sin(angle), np.cos(angle)])
        return np.column_stack(columns)

    def fit(
        self, train_bundle: PreparedSplit, validation_bundle: PreparedSplit
    ) -> "CausalCalendarModel":
        require_training_view(train_bundle, "train_bundle")
        require_training_view(validation_bundle, "validation_bundle")
        self.calendar_column = self._resolve_calendar_column(train_bundle.feature_columns)
        x_train_raw = self._design(train_bundle.frame, fit=True)
        x_train = self.scaler.fit_transform(x_train_raw)
        self.models = {}
        for hormone in HORMONES:
            estimator = Ridge(alpha=self.alpha)
            estimator.fit(x_train, train_bundle.target_log1p(hormone).to_numpy(float))
            self.models[hormone] = estimator
        return self

    def predict(self, test_bundle: PreparedSplit) -> pd.DataFrame:
        if set(self.models) != set(HORMONES):
            raise RuntimeError("Model has not been fit")
        x_test = self.scaler.transform(self._design(test_bundle.frame, fit=False))
        predictions = {
            hormone: np.maximum(estimator.predict(x_test), 0.0)
            for hormone, estimator in self.models.items()
        }
        return make_prediction_frame(
            test_bundle,
            predictions,
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def get_metadata(self) -> dict[str, Any]:
        feature_names = ["missing_known_menses", "days_since_last_known_menses"]
        feature_names.extend(
            f"days_since_last_known_menses_power_{power}"
            for power in range(2, self.degree + 1)
        )
        feature_names.extend(
            [f"sin_2pi_day_over_{self.period:g}", f"cos_2pi_day_over_{self.period:g}"]
        )
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "target_space": "log1p",
            "estimator": "Ridge",
            "ridge_alpha": self.alpha,
            "calendar_source_column": self.calendar_column,
            "calendar_features": feature_names,
            "train_only_missing_day_imputation": self.impute_day,
            "harmonic_reference_period_days": self.period,
            "harmonic_assumption": (
                "A documented 28-day classical reference; polynomial terms permit "
                "nonperiodic deviations and no completed cycle length is assumed."
            ),
            "uses_wearables": False,
            "uses_future_bleeding": False,
            "seed": None,
        }

