"""Small synthetic fit/predict smoke tests for every baseline."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from benchmark.contracts import HORMONES, validate_prediction_frame
from benchmark.evaluator import evaluate_config
from benchmark.report import generate_report
from model.cli import run_models
from model.registry import create_model


@pytest.mark.parametrize("name", ["population_median", "causal_calendar", "catboost"])
def test_baseline_synthetic_smoke(name, synthetic_bundle, model_config):
    model = create_model(name, model_config, quick=True)
    train = synthetic_bundle.view("train", include_truth=True)
    validation = synthetic_bundle.view("validation", include_truth=True)
    test = synthetic_bundle.view("test", include_truth=False)

    returned = model.fit(train, validation)
    assert returned is model
    predictions = model.predict(test)
    validated = validate_prediction_frame(
        predictions, expected_sample_ids=test.sample_ids, expected_split="test"
    )
    assert len(validated) == len(test.frame) * len(HORMONES)
    assert (validated["y_pred"] >= 0).all()
    assert pd.to_numeric(validated["y_pred"], errors="coerce").notna().all()
    metadata = model.get_metadata()
    assert metadata["model_name"] in {
        "population_median",
        "causal_calendar",
        "catboost",
        "hist_gradient_boosting",
    }


def test_fixed_seed_is_deterministic(synthetic_bundle, model_config):
    train = synthetic_bundle.view("train", include_truth=True)
    validation = synthetic_bundle.view("validation", include_truth=True)
    test = synthetic_bundle.view("test", include_truth=False)
    first = create_model("catboost", model_config, quick=True)
    second = create_model("catboost", model_config, quick=True)
    first_predictions = first.fit(train, validation).predict(test)
    second_predictions = second.fit(train, validation).predict(test)
    pd.testing.assert_frame_equal(first_predictions, second_predictions)


def test_synthetic_model_cli_end_to_end(tmp_path, synthetic_bundle, model_config):
    config_dir = tmp_path / "configs"
    prepared_dir = tmp_path / "artifacts" / "private" / "prepared" / "synthetic"
    config_dir.mkdir()
    prepared_dir.mkdir(parents=True)
    synthetic_bundle.frame.to_csv(prepared_dir / "prepared.csv", index=False)
    (prepared_dir / "metadata.json").write_text(
        json.dumps(synthetic_bundle.metadata), encoding="utf-8"
    )
    config = {
        "task": {
            "id": "hormonbench_mcphases_interval2_nextday_v0",
            "version": "0.1.0",
            "track": "primary_interval2_nextday",
        },
        "paths": {
            "prepared_dir": "artifacts/private/prepared/synthetic",
            "prediction_dir": "artifacts/private/predictions",
            "checkpoint_dir": "artifacts/private/checkpoints",
            "results_dir": "results/v0",
        },
        "split": {"seed": 20260719},
        "evaluation": {
            "primary_metric": "participant_macro_log1p_mae",
            "reference_model": "causal_calendar",
        },
        **model_config,
    }
    config_file = config_dir / "synthetic.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    completed = run_models(
        config_file,
        ["population_median", "causal_calendar", "catboost"],
        quick=True,
    )
    assert [item["model_name"] for item in completed] == [
        "population_median",
        "causal_calendar",
        "catboost",
    ]
    for item in completed:
        assert pd.read_csv(item["prediction_file"]).shape[0] == 9 * len(HORMONES)

    metrics, metrics_path = evaluate_config(config_file)
    assert metrics_path is not None and metrics_path.is_file()
    assert set(metrics["models"]) == {
        "population_median",
        "causal_calendar",
        "catboost",
    }
    outputs = generate_report(config_file)
    assert all(path.is_file() for path in outputs.values())
    assert pd.read_csv(outputs["leaderboard"]).shape[0] == 3
