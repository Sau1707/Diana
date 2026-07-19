from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.v1_synthetic import make_synthetic_bundle
from benchmark.v1_task import HORMONES, load_v1_config
from model.catboost.v1_model import CatBoostV1
from model.population_median.v1_model import PopulationMedianV1
from model.v1_common import (
    FeaturePreprocessor,
    assert_equal_participant_weight,
    participant_balanced_weights,
)
from model.v1_registry import BASELINE_REGISTRY, CUSTOM_REFERENCES
from model.wearable_ridge.model import WearableRidgeV1
from model.v1_pipeline import _wide_rows


def test_active_registry_is_exact():
    assert tuple(BASELINE_REGISTRY) == (
        "population_median",
        "wearable_ridge",
        "catboost",
    )
    assert CUSTOM_REFERENCES == ("diana_h3p",)
    assert "causal_calendar" not in BASELINE_REGISTRY


def test_participant_weights_equalize_total_influence():
    groups = pd.Series(["a"] * 2 + ["b"] * 5 + ["c"] * 9)
    weights = participant_balanced_weights(groups)
    assert_equal_participant_weight(groups, weights)
    totals = pd.DataFrame({"group": groups, "weight": weights}).groupby("group")["weight"].sum()
    assert np.allclose(totals, totals.iloc[0])


def test_preprocessor_fit_only_statistics_and_stable_order():
    X = pd.DataFrame(
        {
            "keep_b": [1.0, 2.0, np.nan, 4.0],
            "all_missing": [np.nan] * 4,
            "constant": [2.0] * 4,
            "keep_a": [8.0, 7.0, 6.0, 5.0],
        }
    )
    groups = pd.Series(["a", "a", "b", "b"])
    pre = FeaturePreprocessor(missingness_drop_threshold=0.95, standardize=True).fit(X, groups)
    assert pre.retained_columns == ("keep_b", "keep_a")
    transformed = pre.transform(
        pd.DataFrame({"keep_b": [1000.0], "keep_a": [0.0], "constant": [99.0], "all_missing": [1.0]})
    )
    assert list(transformed.columns) == ["keep_b", "keep_a"]
    assert pre.medians["keep_b"] == 2.0


def test_three_baselines_share_contract_and_seed():
    config = load_v1_config("configs/hormonbench_v1.yaml")
    config["models"]["catboost"]["validation_iterations"] = 5
    config["models"]["catboost"]["early_stopping_rounds"] = 2
    bundle, mapping = make_synthetic_bundle(config)
    train_mask = bundle.frame["fold_group"].isin([2, 3, 4])
    validation_mask = bundle.frame["fold_group"].eq(1)
    test_mask = bundle.frame["fold_group"].eq(0)
    train = bundle.fit_view(train_mask)
    validation = bundle.fit_view(validation_mask)
    inference = bundle.inference_view(test_mask)
    selector = CatBoostV1(config)
    counts = selector.select_tree_counts(train, validation)
    models = [
        PopulationMedianV1(config).fit(train),
        WearableRidgeV1(config).fit(train),
        CatBoostV1(config).fit_fixed(train, counts),
    ]
    for model in models:
        prediction = model.predict(inference)
        assert set(prediction) == set(HORMONES)
        assert all(len(values) == len(inference.X) for values in prediction.values())
        assert all(np.isfinite(values).all() and (values >= 0).all() for values in prediction.values())
    assert selector.seed == config["folds"]["seed"]


def test_official_model_code_has_no_runtime_installer():
    for relative in (
        "model/common.py",
        "model/catboost/v1_model.py",
        "model/v1_pipeline.py",
    ):
        source = Path(relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "pip install" not in source.lower()
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name == "subprocess" for alias in node.names)
            for node in ast.walk(tree)
        )


def test_official_inference_alignment_contains_no_truth():
    config = load_v1_config("configs/hormonbench_v1.yaml")
    bundle, _ = make_synthetic_bundle(config)
    mask = bundle.frame["fold_group"].eq(0)
    predictions = {h: np.ones(int(mask.sum())) for h in HORMONES}
    aligned = _wide_rows(bundle, mask, predictions)
    assert not any(column.startswith("y_") for column in aligned)
