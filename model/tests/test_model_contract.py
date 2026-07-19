"""Shared model-interface and registry tests."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from benchmark.contracts import HORMONES, PREDICTION_COLUMNS, TARGET_COLUMNS
from model.base import HormonbenchModel
from model.common import TrainOnlyTabularPreprocessor
from model.registry import available_models, create_model


def test_registry_contains_exactly_three_requested_families():
    assert available_models() == (
        "population_median",
        "causal_calendar",
        "catboost",
    )


@pytest.mark.parametrize("name", ["population_median", "causal_calendar", "catboost"])
def test_all_models_share_interface(name, model_config):
    model = create_model(name, model_config, quick=True)
    assert isinstance(model, HormonbenchModel)
    assert list(inspect.signature(model.fit).parameters) == [
        "train_bundle",
        "validation_bundle",
    ]
    assert list(inspect.signature(model.predict).parameters) == ["test_bundle"]


def test_inference_view_contains_no_truth(synthetic_bundle):
    test = synthetic_bundle.view("test", include_truth=False)
    assert not set(TARGET_COLUMNS) & set(test.frame)
    with pytest.raises(PermissionError, match="Truth is unavailable"):
        test.target_log1p("lh")


def test_population_median_is_train_only(synthetic_bundle, model_config):
    model = create_model("population_median", model_config, quick=True)
    train = synthetic_bundle.view("train", include_truth=True)
    validation = synthetic_bundle.view("validation", include_truth=True)
    test = synthetic_bundle.view("test", include_truth=False)
    model.fit(train, validation)
    predictions = model.predict(test)
    assert tuple(predictions.columns) == PREDICTION_COLUMNS
    for hormone in HORMONES:
        observed = predictions.loc[predictions["hormone"].eq(hormone), "y_pred"]
        assert observed.nunique() == 1
        assert observed.iloc[0] == pytest.approx(train.target_log1p(hormone).median())


def test_tabular_preprocessing_is_fit_on_train_only():
    train = pd.DataFrame({"number": [1.0, np.nan, 3.0], "category": ["a", "b", "a"]})
    held_out = pd.DataFrame({"number": [np.nan, 10_000.0], "category": ["new", "b"]})
    preprocessor = TrainOnlyTabularPreprocessor().fit(
        train, ("number", "category")
    )
    transformed = preprocessor.transform(held_out)
    assert transformed.loc[0, "number"] == pytest.approx(2.0)
    assert transformed.loc[0, "category"] == -1
    assert preprocessor.numeric_medians == {"number": 2.0}
