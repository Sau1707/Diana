import numpy as np
import pandas as pd
import pytest

from benchmark.contracts import assert_feature_names_safe
from benchmark.data.adapter import find_eligible_origins
from benchmark.data.features import build_history_features, derive_known_menses_onsets


def _targets():
    rows = []
    for day in range(1, 22):
        rows.append({
            "id": 7, "study_interval": 2024, "day_in_study": day,
            "is_weekend": False, "flow_volume": "Light" if day == 2 else ("Not at all" if day in {1, 3} else np.nan),
            "lh": 1.0 if day >= 15 else np.nan,
            "estrogen": 2.0 if day >= 15 else np.nan,
            "pdg": 3.0 if day >= 15 else np.nan,
        })
    return pd.DataFrame(rows)


def test_exact_history_target_alignment_and_no_label_interpolation():
    origins = find_eligible_origins(_targets(), history_days=14)
    assert (origins.target_day == origins.origin_day + 1).all()
    assert origins.origin_day.min() == 14
    assert origins[TARGET_RAW_COLUMNS_FOR_TEST].notna().all().all()


TARGET_RAW_COLUMNS_FOR_TEST = ["target_lh_raw", "target_e3g_raw", "target_pdg_raw"]


def test_history_uses_t_minus_13_through_t_and_not_future():
    targets = _targets()
    origins = find_eligible_origins(targets, history_days=14).head(1)
    origins["sample_id"] = "synthetic"
    daily = pd.DataFrame({"id": 7, "day_in_study": range(1, 22), "sensor": range(1, 22)})
    first, _ = build_history_features(daily, origins[["sample_id", "private_participant_id", "origin_day"]], all_targets=targets, history_days=14, selected_lags=[0, 13], base_provenance={"sensor": {"source": "synthetic"}})
    daily.loc[daily.day_in_study > int(origins.origin_day.iloc[0]), "sensor"] = 99999
    second, _ = build_history_features(daily, origins[["sample_id", "private_participant_id", "origin_day"]], all_targets=targets, history_days=14, selected_lags=[0, 13], base_provenance={"sensor": {"source": "synthetic"}})
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "sensor__lag0"] == origins.origin_day.iloc[0]
    assert first.loc[0, "sensor__lag13"] == origins.origin_day.iloc[0] - 13


def test_menses_onsets_are_past_only_and_can_precede_window():
    onsets = derive_known_menses_onsets(_targets())
    assert onsets[7] == [2]


def test_first_observed_positive_flow_is_not_assumed_to_be_an_onset():
    reports = pd.DataFrame(
        {
            "id": [1, 1, 1, 1],
            "day_in_study": [1, 2, 3, 4],
            "flow_volume": ["Light", "Not at all", np.nan, "Moderate"],
        }
    )
    assert derive_known_menses_onsets(reports)[1] == [4]


def test_prohibited_names_fail_closed():
    for name in ["lh", "participant_id", "mira_phase", "target_pdg_log1p", "future_menses"]:
        with pytest.raises(ValueError):
            assert_feature_names_safe([name])
    assert_feature_names_safe(["active__sedentary__mean", "days_since_last_known_menses"])
