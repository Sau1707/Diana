from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.contracts import (
    HORMONES,
    PREDICTION_COLUMNS,
    TARGET_LOG_COLUMNS,
    TARGET_RAW_COLUMNS,
    PreparedBundle,
    assert_feature_names_safe,
    validate_prediction_frame,
)


def _bundle() -> PreparedBundle:
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        origin = 20 + index
        row = {
            "task_version": "0.1.0",
            "sample_id": f"sample-{index}",
            "private_participant_id": f"private-{index}",
            "origin_day": origin,
            "target_day": origin + 1,
            "history_start_day": origin - 13,
            "history_end_day": origin,
            "cutoff_day": origin,
            "split": split,
            "config_hash": "config",
            "split_hash": "split",
            "wearable_mean": float(index),
        }
        for h_index, hormone in enumerate(HORMONES, start=1):
            raw = float(index + h_index)
            row[TARGET_RAW_COLUMNS[hormone]] = raw
            row[TARGET_LOG_COLUMNS[hormone]] = np.log1p(raw)
        rows.append(row)
    return PreparedBundle(pd.DataFrame(rows), {"feature_columns": ["wearable_mean"]})


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "sample-2",
                "hormone": hormone,
                "horizon": 1,
                "y_pred": 0.5,
                "model_name": "external",
                "model_version": "1",
                "track": "primary_interval2_nextday",
                "split": "test",
            }
            for hormone in HORMONES
        ],
        columns=PREDICTION_COLUMNS,
    )


def test_prepared_bundle_enforces_exact_history_and_target_alignment() -> None:
    bundle = _bundle()
    bundle.validate()

    bad = bundle.frame.copy()
    bad.loc[0, "history_start_day"] -= 1
    with pytest.raises(ValueError, match="t-13"):
        PreparedBundle(bad, bundle.metadata).validate()

    bad = bundle.frame.copy()
    bad.loc[0, "target_day"] += 1
    with pytest.raises(ValueError, match=r"t\+1"):
        PreparedBundle(bad, bundle.metadata).validate()


def test_feature_contract_rejects_identifiers_targets_and_future_fields() -> None:
    for unsafe in (
        "private_participant_id",
        "past_lh_mean",
        "future_flow_volume",
        "centered_temperature_mean",
        "mira_phase",
    ):
        with pytest.raises(ValueError, match="Prohibited"):
            assert_feature_names_safe([unsafe])


def test_prediction_contract_is_exact_and_does_not_require_truth() -> None:
    predictions = _predictions()
    validated = validate_prediction_frame(
        predictions, expected_sample_ids=["sample-2"]
    )
    assert tuple(validated.columns) == PREDICTION_COLUMNS
    assert not any(column.startswith("y_true") for column in validated.columns)

    with pytest.raises(ValueError, match="exactly"):
        validate_prediction_frame(predictions.assign(y_true=1.0))


def test_prediction_contract_rejects_wrong_track_and_invalid_values() -> None:
    predictions = _predictions()
    with pytest.raises(ValueError, match="track/split"):
        validate_prediction_frame(predictions.assign(track="other"))
    with pytest.raises(ValueError, match="nonnegative"):
        validate_prediction_frame(predictions.assign(y_pred=-0.1))

