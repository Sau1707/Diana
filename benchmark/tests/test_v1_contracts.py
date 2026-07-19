from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from benchmark.v1_contracts import (
    HORMONES,
    assert_v1_feature_names_safe,
    validate_v1_prediction_frame,
)
from benchmark.v1_synthetic import make_synthetic_bundle
from benchmark.v1_task import (
    TASK_ID,
    TASK_VERSION,
    TRACK_COLD,
    load_v1_config,
    task_spec_hash,
)


@pytest.fixture
def v1_config():
    return load_v1_config("configs/hormonbench_v1.yaml")


def test_task_spec_hash_ignores_paths(v1_config):
    changed = copy.deepcopy(v1_config)
    changed["paths"]["data_root"] = "Z:/different/private/location"
    changed["paths"]["results_dir"] = "elsewhere"
    assert task_spec_hash(changed) == task_spec_hash(v1_config)


def test_model_view_has_only_features(v1_config):
    bundle, _ = make_synthetic_bundle(v1_config)
    view = bundle.fit_view(np.ones(len(bundle.frame), dtype=bool))
    assert list(view.X.columns) == list(bundle.feature_columns)
    assert not {
        "private_participant_id",
        "sample_id",
        "origin_day",
        "target_day",
        "study_interval",
    } & set(view.X.columns)
    assert tuple(view.targets.columns) == HORMONES


@pytest.mark.parametrize(
    "name",
    [
        "days_since_last_known_menses",
        "menses_onset_missing",
        "self_report__pain__mean",
        "target_lh__lag1",
        "origin_day",
        "calendar_date",
        "mira_phase",
        "absolute_time_modulo_28",
    ],
)
def test_explicit_feature_denylist(name):
    with pytest.raises(ValueError):
        assert_v1_feature_names_safe([name])


def _valid_predictions(bundle):
    ids = bundle.frame.iloc[:2]["sample_id"].astype(str).tolist()
    rows = []
    for sample_id in ids:
        for hormone in HORMONES:
            rows.append(
                {
                    "task_id": TASK_ID,
                    "task_version": TASK_VERSION,
                    "track": TRACK_COLD,
                    "fold": 0,
                    "calibration_budget": 0,
                    "split": "test",
                    "sample_id": sample_id,
                    "hormone": hormone,
                    "horizon": 1,
                    "y_pred": 1.0,
                    "model_name": "external",
                    "model_version": "1",
                }
            )
    return pd.DataFrame(rows), ids


def test_prediction_contract_rejects_bad_rows(v1_config):
    bundle, _ = make_synthetic_bundle(v1_config)
    valid, ids = _valid_predictions(bundle)
    validated = validate_v1_prediction_frame(
        valid.sample(frac=1, random_state=7),
        expected_sample_ids=ids,
        expected_track=TRACK_COLD,
        expected_fold=0,
        expected_budget=0,
    )
    assert len(validated) == 6
    for mutation in ("duplicate", "missing", "negative", "nonfinite", "truth"):
        bad = valid.copy()
        if mutation == "duplicate":
            bad = pd.concat([bad, bad.iloc[[0]]], ignore_index=True)
        elif mutation == "missing":
            bad = bad.iloc[1:].copy()
        elif mutation == "negative":
            bad.loc[0, "y_pred"] = -0.1
        elif mutation == "nonfinite":
            bad.loc[0, "y_pred"] = np.inf
        else:
            bad["y_true"] = 1.0
        with pytest.raises(ValueError):
            validate_v1_prediction_frame(
                bad,
                expected_sample_ids=ids,
                expected_track=TRACK_COLD,
                expected_fold=0,
                expected_budget=0,
            )


def test_interval_contract_and_genuinely_observed_labels(v1_config):
    bundle, _ = make_synthetic_bundle(v1_config)
    frame = bundle.frame
    assert (frame["history_start_day"] == frame["origin_day"] - 13).all()
    assert (frame["history_end_day"] == frame["origin_day"]).all()
    assert (frame["target_day"] == frame["origin_day"] + 1).all()
    for hormone in HORMONES:
        assert np.isfinite(frame[f"target_{hormone}_raw"]).all()
        assert np.allclose(
            frame[f"target_{hormone}_log1p"],
            np.log1p(frame[f"target_{hormone}_raw"]),
        )
