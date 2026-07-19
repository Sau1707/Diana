"""Bounded CPU CatBoost baseline with an honest sklearn fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from benchmark.contracts import HORMONES, PreparedSplit
from model.base import HormonbenchModel
from model.common import (
    TrainOnlyTabularPreprocessor,
    attempt_catboost_import,
    make_prediction_frame,
    require_training_view,
)


class CatBoostBaselineModel(HormonbenchModel):
    """One fixed-seed bounded CPU regressor per hormone."""

    model_version = "0.1.0"

    def __init__(self, config: dict[str, Any] | None = None, *, quick: bool = False):
        self.config = config or {}
        self.quick = quick
        self.params = dict(self.config.get("models", {}).get("catboost", {}))
        self.seed = int(self.config.get("models", {}).get("seed", 20260719))
        self.preprocessor = TrainOnlyTabularPreprocessor()
        self.models: dict[str, Any] = {}
        self.backend = "catboost"
        self.model_name = "catboost"
        self.fallback_reason: str | None = None
        self.best_iterations: dict[str, int | None] = {}
        self.final_parameters: dict[str, Any] = {}

    def _catboost_parameters(self) -> dict[str, Any]:
        iterations = int(
            self.params.get("quick_iterations", 180)
            if self.quick
            else self.params.get("iterations", 300)
        )
        return {
            "iterations": iterations,
            "depth": int(self.params.get("depth", 5)),
            "learning_rate": float(self.params.get("learning_rate", 0.04)),
            "l2_leaf_reg": float(self.params.get("l2_leaf_reg", 5.0)),
            "loss_function": str(self.params.get("loss_function", "RMSE")),
            "eval_metric": str(self.params.get("eval_metric", "MAE")),
            "random_seed": self.seed,
            "thread_count": int(self.params.get("thread_count", 4)),
            "task_type": "CPU",
            "allow_writing_files": False,
            "verbose": False,
        }

    def _fit_hist_fallback(
        self, x_train: pd.DataFrame, train_bundle: PreparedSplit, reason: str
    ) -> None:
        self.backend = "hist_gradient_boosting"
        self.model_name = "hist_gradient_boosting"
        self.fallback_reason = reason
        max_iter = int(
            self.params.get("quick_iterations", 180)
            if self.quick
            else self.params.get("iterations", 300)
        )
        self.final_parameters = {
            "max_iter": max_iter,
            "learning_rate": float(self.params.get("learning_rate", 0.04)),
            "max_leaf_nodes": 2 ** int(self.params.get("depth", 5)),
            "l2_regularization": float(self.params.get("l2_leaf_reg", 5.0)),
            "early_stopping": False,
            "random_state": self.seed,
        }
        self.models = {}
        for hormone in HORMONES:
            estimator = HistGradientBoostingRegressor(**self.final_parameters)
            estimator.fit(x_train, train_bundle.target_log1p(hormone).to_numpy(float))
            self.models[hormone] = estimator
            self.best_iterations[hormone] = max_iter

    def _fit_catboost_models(
        self,
        estimator_class: Any,
        parameters: dict[str, Any],
        x_train: pd.DataFrame,
        x_validation: pd.DataFrame,
        train_bundle: PreparedSplit,
        validation_bundle: PreparedSplit,
    ) -> None:
        self.models = {}
        self.best_iterations = {}
        for hormone in HORMONES:
            estimator = estimator_class(**parameters)
            estimator.fit(
                x_train,
                train_bundle.target_log1p(hormone).to_numpy(float),
                eval_set=(
                    x_validation,
                    validation_bundle.target_log1p(hormone).to_numpy(float),
                ),
                early_stopping_rounds=int(self.params.get("early_stopping_rounds", 40)),
                use_best_model=True,
                verbose=False,
            )
            self.models[hormone] = estimator
            best = estimator.get_best_iteration()
            self.best_iterations[hormone] = int(best) if best is not None else None

    def fit(
        self, train_bundle: PreparedSplit, validation_bundle: PreparedSplit
    ) -> "CatBoostBaselineModel":
        require_training_view(train_bundle, "train_bundle")
        require_training_view(validation_bundle, "validation_bundle")
        self.preprocessor.fit(train_bundle.frame, train_bundle.feature_columns)
        x_train = self.preprocessor.transform(train_bundle.frame)
        x_validation = self.preprocessor.transform(validation_bundle.frame)

        estimator_class, import_error = attempt_catboost_import(repair=True)
        if estimator_class is None:
            self._fit_hist_fallback(x_train, train_bundle, import_error or "CatBoost unavailable")
            return self

        self.final_parameters = self._catboost_parameters()
        try:
            self._fit_catboost_models(
                estimator_class,
                self.final_parameters,
                x_train,
                x_validation,
                train_bundle,
                validation_bundle,
            )
        except Exception as first_error:  # pragma: no cover - environment dependent
            # One bounded CatBoost-only repair: retry CPU fitting single-threaded.
            retry_parameters = dict(self.final_parameters)
            retry_parameters["thread_count"] = 1
            try:
                self._fit_catboost_models(
                    estimator_class,
                    retry_parameters,
                    x_train,
                    x_validation,
                    train_bundle,
                    validation_bundle,
                )
                self.final_parameters = retry_parameters
            except Exception as second_error:
                reason = (
                    "CatBoost runtime failure after one single-thread CPU retry: "
                    f"{type(first_error).__name__}: {first_error}; "
                    f"retry={type(second_error).__name__}: {second_error}"
                )
                self._fit_hist_fallback(x_train, train_bundle, reason)
        return self

    def predict(self, test_bundle: PreparedSplit) -> pd.DataFrame:
        if set(self.models) != set(HORMONES):
            raise RuntimeError("Model has not been fit")
        x_test = self.preprocessor.transform(test_bundle.frame)
        predictions = {
            hormone: np.maximum(np.asarray(estimator.predict(x_test), dtype=float), 0.0)
            for hormone, estimator in self.models.items()
        }
        return make_prediction_frame(
            test_bundle,
            predictions,
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "requested_family": "catboost",
            "model_version": self.model_version,
            "backend": self.backend,
            "fallback_reason": self.fallback_reason,
            "target_space": "log1p",
            "one_regressor_per_hormone": True,
            "seed": self.seed,
            "quick": self.quick,
            "parameters": self.final_parameters,
            "best_iteration": dict(self.best_iterations),
            "preprocessor": self.preprocessor.metadata(),
            "validation_role": (
                "external early stopping only" if self.backend == "catboost" else "not used"
            ),
        }
