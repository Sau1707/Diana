"""Canonical participant-balanced CatBoost baseline for Hormonbench v1."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from catboost import CatBoostRegressor, Pool

from benchmark.v1_contracts import V1FitView, V1InferenceView
from benchmark.v1_task import HORMONES
from model.v1_base import HormonbenchV1Model
from model.v1_common import (
    FeaturePreprocessor,
    combine_fit_views,
    participant_balanced_weights,
)


class CatBoostV1(HormonbenchV1Model):
    model_name = "catboost"
    model_version = "1.0.0"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.seed = int(config["folds"]["seed"])
        self.settings = dict(config["models"]["catboost"])
        self.preprocessor = FeaturePreprocessor(
            missingness_drop_threshold=float(
                config["preprocessing"]["missingness_drop_threshold"]
            ),
            standardize=False,
        )
        self.models: dict[str, CatBoostRegressor] = {}
        self.tree_counts: dict[str, int] = {}
        self.best_iterations: dict[str, int] = {}
        self.best_validation_scores: dict[str, float] = {}

    def _parameters(self, iterations: int) -> dict[str, Any]:
        return {
            "iterations": int(iterations),
            "depth": int(self.settings["depth"]),
            "learning_rate": float(self.settings["learning_rate"]),
            "l2_leaf_reg": float(self.settings["l2_leaf_reg"]),
            "loss_function": str(self.settings["loss_function"]),
            "eval_metric": str(self.settings["eval_metric"]),
            "random_seed": self.seed,
            "thread_count": int(self.settings["thread_count"]),
            "task_type": "CPU",
            "allow_writing_files": False,
            "verbose": False,
        }

    def select_tree_counts(
        self, train_view: V1FitView, validation_view: V1FitView
    ) -> dict[str, int]:
        train_view.validate()
        validation_view.validate()
        self.preprocessor.fit(train_view.X, train_view.participant_groups)
        X_train = self.preprocessor.transform(train_view.X)
        X_validation = self.preprocessor.transform(validation_view.X)
        train_weight = participant_balanced_weights(train_view.participant_groups)
        validation_weight = participant_balanced_weights(
            validation_view.participant_groups
        )
        counts: dict[str, int] = {}
        self.best_iterations = {}
        self.best_validation_scores = {}
        maximum = int(self.settings["validation_iterations"])
        for hormone in HORMONES:
            train_pool = Pool(
                X_train,
                train_view.targets[hormone].to_numpy(float),
                weight=train_weight,
            )
            validation_pool = Pool(
                X_validation,
                validation_view.targets[hormone].to_numpy(float),
                weight=validation_weight,
            )
            estimator = CatBoostRegressor(**self._parameters(maximum))
            estimator.fit(
                train_pool,
                eval_set=validation_pool,
                early_stopping_rounds=int(self.settings["early_stopping_rounds"]),
                use_best_model=True,
                verbose=False,
            )
            best = estimator.get_best_iteration()
            best_iteration = max(0, int(best) if best is not None else maximum - 1)
            tree_count = best_iteration + 1
            best_scores = estimator.get_best_score()
            validation_scores = best_scores.get("validation", {})
            score = validation_scores.get(str(self.settings["eval_metric"]))
            if score is None:
                score = next(iter(validation_scores.values()), np.nan)
            self.best_iterations[hormone] = best_iteration
            self.best_validation_scores[hormone] = float(score)
            counts[hormone] = tree_count
        self.tree_counts = counts
        return dict(counts)

    def fit_fixed(
        self, fit_view: V1FitView, tree_counts: Mapping[str, int]
    ) -> "CatBoostV1":
        fit_view.validate()
        if set(tree_counts) != set(HORMONES):
            raise ValueError("Fixed CatBoost tree counts require every hormone")
        self.preprocessor = FeaturePreprocessor(
            missingness_drop_threshold=float(
                self.config["preprocessing"]["missingness_drop_threshold"]
            ),
            standardize=False,
        ).fit(fit_view.X, fit_view.participant_groups)
        X = self.preprocessor.transform(fit_view.X)
        weights = participant_balanced_weights(fit_view.participant_groups)
        self.models = {}
        self.tree_counts = {hormone: int(tree_counts[hormone]) for hormone in HORMONES}
        for hormone in HORMONES:
            estimator = CatBoostRegressor(
                **self._parameters(self.tree_counts[hormone])
            )
            estimator.fit(
                Pool(
                    X,
                    fit_view.targets[hormone].to_numpy(float),
                    weight=weights,
                ),
                verbose=False,
            )
            self.models[hormone] = estimator
        return self

    def fit(self, fit_view: V1FitView) -> "CatBoostV1":
        counts = {
            hormone: int(self.settings["iterations"]) for hormone in HORMONES
        }
        return self.fit_fixed(fit_view, counts)

    def select_then_refit(
        self, train_view: V1FitView, validation_view: V1FitView
    ) -> "CatBoostV1":
        counts = self.select_tree_counts(train_view, validation_view)
        best_iterations = dict(self.best_iterations)
        best_scores = dict(self.best_validation_scores)
        development = combine_fit_views(train_view, validation_view)
        self.fit_fixed(development, counts)
        self.best_iterations = best_iterations
        self.best_validation_scores = best_scores
        return self

    def predict(self, inference_view: V1InferenceView) -> dict[str, np.ndarray]:
        inference_view.validate()
        if set(self.models) != set(HORMONES):
            raise RuntimeError("CatBoost v1 has not been fitted")
        X = self.preprocessor.transform(inference_view.X)
        predictions = {
            hormone: np.maximum(
                np.asarray(self.models[hormone].predict(X), dtype=float), 0.0
            )
            for hormone in HORMONES
        }
        self.validate_prediction_dict(predictions, len(X))
        return predictions

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "backend": "catboost",
            "catboost_version": __import__("catboost").__version__,
            "device": "CPU",
            "seed": self.seed,
            "participant_balanced_train_and_validation": True,
            "parameters": self._parameters(
                max(self.tree_counts.values()) if self.tree_counts else 1
            ),
            "best_iteration": dict(self.best_iterations),
            "tree_count": dict(self.tree_counts),
            "best_validation_score": dict(self.best_validation_scores),
            "preprocessor": self.preprocessor.metadata(),
            "target_space": "log1p",
        }
