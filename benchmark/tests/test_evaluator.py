from __future__ import annotations

import builtins
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from benchmark.contracts import (
    HORMONES,
    PREDICTION_COLUMNS,
    TARGET_LOG_COLUMNS,
    TARGET_RAW_COLUMNS,
    PreparedBundle,
)
from benchmark.evaluator import (
    evaluate_prediction_file,
    evaluate_prediction_files,
    evaluate_prediction_frame,
)
from benchmark.report import generate_report


def _bundle() -> PreparedBundle:
    rows = []
    split_participants = {
        "train": ("train-a", "train-b"),
        "validation": ("validation-a",),
        "test": ("test-a", "test-b"),
    }
    index = 0
    for split, participants in split_participants.items():
        for participant in participants:
            for offset in range(2):
                origin = 20 + index
                row = {
                    "task_version": "0.1.0",
                    "sample_id": f"sample-{index}",
                    "private_participant_id": participant,
                    "origin_day": origin,
                    "target_day": origin + 1,
                    "history_start_day": origin - 13,
                    "history_end_day": origin,
                    "cutoff_day": origin,
                    "split": split,
                    "config_hash": "config-hash",
                    "split_hash": "split-hash",
                    "wearable_mean": float(offset),
                }
                for h_index, hormone in enumerate(HORMONES, start=1):
                    raw = float(h_index + offset + index / 10)
                    row[TARGET_RAW_COLUMNS[hormone]] = raw
                    row[TARGET_LOG_COLUMNS[hormone]] = np.log1p(raw)
                rows.append(row)
                index += 1
    metadata = {
        "task_id": "hormonbench_mcphases_interval2_nextday_v0",
        "task_version": "0.1.0",
        "track": "primary_interval2_nextday",
        "feature_columns": ["wearable_mean"],
    }
    bundle = PreparedBundle(pd.DataFrame(rows), metadata)
    bundle.validate()
    return bundle


def _predictions(bundle: PreparedBundle, model_name: str, bias: float = 0.0) -> pd.DataFrame:
    rows = []
    test = bundle.frame.loc[bundle.frame["split"].eq("test")]
    for _, sample in test.iterrows():
        for hormone in HORMONES:
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "hormone": hormone,
                    "horizon": 1,
                    "y_pred": max(
                        0.0, float(sample[TARGET_LOG_COLUMNS[hormone]]) + bias
                    ),
                    "model_name": model_name,
                    "model_version": "synthetic-1",
                    "track": "primary_interval2_nextday",
                    "split": "test",
                }
            )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def test_evaluator_is_invariant_to_prediction_row_order() -> None:
    bundle = _bundle()
    predictions = _predictions(bundle, "external", bias=0.1)
    ordered = evaluate_prediction_frame(bundle, predictions).summary
    shuffled = evaluate_prediction_frame(
        bundle, predictions.sample(frac=1.0, random_state=123)
    ).summary
    assert ordered == shuffled


def test_evaluator_fails_on_missing_and_duplicate_predictions() -> None:
    bundle = _bundle()
    predictions = _predictions(bundle, "external")
    with pytest.raises(ValueError, match="exactly one prediction|coverage"):
        evaluate_prediction_frame(bundle, predictions.iloc[:-1])
    duplicate = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_prediction_frame(bundle, duplicate)


def test_external_prediction_csv_evaluates_without_importing_model(tmp_path, monkeypatch) -> None:
    bundle = _bundle()
    path = tmp_path / "external.csv"
    _predictions(bundle, "external", bias=0.05).to_csv(path, index=False)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "model" or name.startswith("model."):
            raise AssertionError("benchmark evaluator imported model")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    evaluated = evaluate_prediction_file(bundle, path)
    assert evaluated.model_name == "external"
    assert evaluated.summary["n_prediction_rows"] == 12


def test_multi_model_evaluation_is_aggregate_only_and_compares_reference(tmp_path) -> None:
    bundle = _bundle()
    calendar_path = tmp_path / "calendar.csv"
    other_path = tmp_path / "other.csv"
    _predictions(bundle, "causal_calendar", bias=0.2).to_csv(calendar_path, index=False)
    _predictions(bundle, "other", bias=0.1).to_csv(other_path, index=False)

    public = evaluate_prediction_files(bundle, [other_path, calendar_path])
    assert public["split_counts"] == {
        "train": {"participants": 2, "eligible_origins": 4},
        "validation": {"participants": 1, "eligible_origins": 2},
        "test": {"participants": 2, "eligible_origins": 4},
    }
    for hormone in HORMONES:
        assert public["models"]["other"]["skill_relative_to_causal_calendar"][
            hormone
        ] > 0
        assert public["models"]["other"][
            "participants_improved_vs_causal_calendar"
        ][hormone] == {"count": 2, "out_of": 2}

    serialized = json.dumps(public)
    assert "private_participant_id" not in serialized
    assert "test-a" not in serialized


def test_public_report_contains_only_aggregate_artifacts(tmp_path) -> None:
    bundle = _bundle()
    prediction_dir = tmp_path / "artifacts/private/predictions"
    checkpoint_dir = tmp_path / "artifacts/private/checkpoints"
    results_dir = tmp_path / "results/v0"
    config_dir = tmp_path / "configs"
    for path in (prediction_dir, checkpoint_dir, results_dir, config_dir):
        path.mkdir(parents=True, exist_ok=True)
    calendar = prediction_dir / "causal_calendar.csv"
    other = prediction_dir / "other.csv"
    _predictions(bundle, "causal_calendar", bias=0.2).to_csv(calendar, index=False)
    _predictions(bundle, "other", bias=0.1).to_csv(other, index=False)
    metrics = evaluate_prediction_files(bundle, [calendar, other])
    (results_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    # A private-looking field in model metadata must not pass the report allowlist.
    (checkpoint_dir / "other.json").write_text(
        json.dumps(
            {
                "model_name": "other",
                "fit_seconds": 1.25,
                "private_participant_id": "must-not-publish",
            }
        ),
        encoding="utf-8",
    )
    config = {
        "task": {
            "id": "hormonbench_mcphases_interval2_nextday_v0",
            "version": "0.1.0",
            "track": "primary_interval2_nextday",
        },
        "paths": {
            "checkpoint_dir": "artifacts/private/checkpoints",
            "prepared_dir": "artifacts/private/prepared",
            "results_dir": "results/v0",
        },
        "split": {"seed": 20260719},
        "evaluation": {
            "primary_metric": "participant_macro_log1p_mae",
        },
    }
    config_path = config_dir / "hormonbench_v0.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = generate_report(config_path)
    assert all(path.is_file() for path in outputs.values())
    assert "private_participant_id" not in outputs["run_manifest"].read_text(
        encoding="utf-8"
    )
    assert "must-not-publish" not in outputs["run_manifest"].read_text(
        encoding="utf-8"
    )
    leaderboard = pd.read_csv(outputs["leaderboard"])
    assert list(leaderboard["rank"]) == [1, 2]
    assert leaderboard.iloc[0]["model_name"] == "other"
