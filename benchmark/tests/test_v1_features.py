from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.data.v1_features import build_v1_history_features, load_v1_daily_features


def test_future_values_do_not_change_history_features():
    days = pd.DataFrame(
        {
            "id": [1] * 20,
            "day_in_study": list(range(90, 110)),
            "active__lightly": np.arange(20, dtype=float),
        }
    )
    origins = pd.DataFrame(
        {"sample_id": ["s"], "private_participant_id": [1], "origin_day": [103]}
    )
    provenance = {
        "active__lightly": {
            "source": "synthetic",
            "modality": "active_minutes",
            "uses_future": False,
        }
    }
    before, _ = build_v1_history_features(
        days, origins, history_days=14, selected_lags=[0, 1, 3, 6, 13], base_provenance=provenance
    )
    changed = days.copy()
    changed.loc[changed["day_in_study"].gt(103), "active__lightly"] = 1e9
    after, _ = build_v1_history_features(
        changed, origins, history_days=14, selected_lags=[0, 1, 3, 6, 13], base_provenance=provenance
    )
    pd.testing.assert_frame_equal(before, after)


def test_feature_builder_exposes_no_alignment_or_target_fields():
    daily = pd.DataFrame(
        {"id": [1] * 14, "day_in_study": range(1, 15), "sleep_score__overall": range(14)}
    )
    origins = pd.DataFrame(
        {"sample_id": ["safe"], "private_participant_id": [1], "origin_day": [14]}
    )
    output, _ = build_v1_history_features(
        daily,
        origins,
        history_days=14,
        selected_lags=[0, 13],
        base_provenance={"sleep_score__overall": {"source": "synthetic", "uses_future": False}},
    )
    assert set(output) - {"sample_id"}
    assert not {"private_participant_id", "origin_day", "target_day"} & set(output)
    assert not any("lh" in name.lower() for name in output if name != "sample_id")


def test_interval1_menses_change_cannot_affect_interval2_features(monkeypatch, tmp_path):
    target = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "study_interval": [2022, 2024, 2024],
            "day_in_study": [1, 100, 101],
            "is_weekend": [False, False, True],
            "flow_volume": ["heavy", np.nan, np.nan],
        }
    )
    active_columns = ["sedentary", "lightly", "moderately", "very"]
    temperature_columns = [
        "temperature_samples", "nightly_temperature", "baseline_relative_sample_sum",
        "baseline_relative_sample_sum_of_squares",
        "baseline_relative_nightly_standard_deviation",
        "baseline_relative_sample_standard_deviation",
    ]
    hrv_columns = ["rmssd", "coverage", "low_frequency", "high_frequency"]
    respiratory_columns = [
        f"{stage}_{suffix}"
        for stage in ("full_sleep", "deep_sleep", "light_sleep", "rem_sleep")
        for suffix in ("breathing_rate", "standard_deviation", "signal_to_noise")
    ]
    sleep_columns = [
        "overall_score", "composition_score", "revitalization_score", "duration_score",
        "deep_sleep_in_minutes", "resting_heart_rate", "restlessness",
    ]
    frames = {
        "active_minutes.csv": pd.DataFrame(
            [[1, 2024, 100, 1, 2, 3, 4], [1, 2024, 101, 2, 3, 4, 5]],
            columns=["id", "study_interval", "day_in_study", *active_columns],
        ),
        "computed_temperature.csv": pd.DataFrame(
            [[1, 2024, 100, *([1.0] * len(temperature_columns))], [1, 2024, 101, *([2.0] * len(temperature_columns))]],
            columns=["id", "study_interval", "sleep_end_day_in_study", *temperature_columns],
        ),
        "heart_rate_variability_details.csv": pd.DataFrame(
            [[1, 2024, 100, *([1.0] * len(hrv_columns))], [1, 2024, 101, *([2.0] * len(hrv_columns))]],
            columns=["id", "study_interval", "day_in_study", *hrv_columns],
        ),
        "respiratory_rate_summary.csv": pd.DataFrame(
            [[1, 2024, 100, *([10.0] * len(respiratory_columns))], [1, 2024, 101, *([11.0] * len(respiratory_columns))]],
            columns=["id", "study_interval", "day_in_study", *respiratory_columns],
        ),
        "sleep_score.csv": pd.DataFrame(
            [[1, 2024, 100, *([1.0] * len(sleep_columns))], [1, 2024, 101, *([2.0] * len(sleep_columns))]],
            columns=["id", "study_interval", "day_in_study", *sleep_columns],
        ),
    }

    def fake_read_csv(path, **_kwargs):
        return frames[path.name].copy()

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    first, _, _ = load_v1_daily_features(tmp_path, target, 2024)
    changed = target.copy()
    changed.loc[changed["study_interval"].eq(2022), "flow_volume"] = "none"
    second, _, _ = load_v1_daily_features(tmp_path, changed, 2024)
    pd.testing.assert_frame_equal(first, second)
