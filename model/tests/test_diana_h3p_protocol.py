from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from benchmark.v1_contracts import assert_v1_feature_names_safe
from benchmark.v1_personalization import PersonalizationPlan
from benchmark.v1_task import FORECAST_DAYS, HISTORY_DAYS, TASK_ID
from model.diana_h3p import covariance, layer1, layer2, model, uncertainty
from model.diana_h3p.layer2 import fit_layer2_core


def _synthetic_layer1_oof_and_plan() -> tuple[pd.DataFrame, PersonalizationPlan]:
    rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    scoring_rows: list[dict[str, object]] = []
    for participant_index in range(8):
        participant = f"synthetic_{participant_index}"
        offset = np.asarray(
            [0.08 * participant_index, -0.04 * participant_index, 0.03 * participant_index]
        )
        for day in range(1, 13):
            sample_id = f"{participant}_{day}"
            if day <= 3:
                temporal = np.asarray(
                    [0.22 * participant_index, -0.10 * participant_index, 0.06 * day]
                )
            elif day <= 7:
                temporal = np.asarray(
                    [-0.11 * participant_index, 0.07 * participant_index, -0.04 * day]
                )
            else:
                temporal = np.asarray(
                    [0.004 * day, -0.003 * day, 0.002 * ((day + participant_index) % 3)]
                )
            residual = offset + temporal
            prediction = np.asarray([1.0, 1.4, 0.8], dtype=float)
            rows.append(
                {
                    "sample_id": sample_id,
                    "private_participant_id": participant,
                    "origin_day": day,
                    "target_day": day + 1,
                    "y_lh": prediction[0] + residual[0],
                    "y_e3g": prediction[1] + residual[1],
                    "y_pdg": prediction[2] + residual[2],
                    "pred_lh": prediction[0],
                    "pred_e3g": prediction[1],
                    "pred_pdg": prediction[2],
                }
            )
            alignment = {
                "sample_id": sample_id,
                "private_participant_id": participant,
                "origin_day": day,
                "target_day": day + 1,
            }
            if day <= 7:
                calibration_rows.append({**alignment, "calibration_rank": day})
            else:
                scoring_rows.append(alignment)
    calibration = pd.DataFrame(calibration_rows)
    scoring = pd.DataFrame(scoring_rows)
    plan = PersonalizationPlan(
        calibration_candidates=calibration,
        scoring_rows=scoring,
        aggregate={
            "participants": 8,
            "calibration_candidates": len(calibration),
            "common_scoring_origins": len(scoring),
            "common_budget": 7,
            "minimum_scoring_origins_per_participant": 5,
        },
    )
    return pd.DataFrame(rows), plan


def test_frozen_task_boundary_and_feature_deny_list() -> None:
    assert TASK_ID == "hormonbench_mcphases_interval2_nextday_v1"
    assert HISTORY_DAYS == 14
    assert FORECAST_DAYS == 1
    assert_v1_feature_names_safe(
        [
            "active__lightly__lag0",
            "temperature__nightly_temperature__mean",
            "hrv__rmssd__coverage",
            "respiratory__full_sleep_breathing_rate__last",
            "sleep_score__overall_score__time_since",
            "weekend__is_weekend__lag13",
        ]
    )
    forbidden = [
        "private_participant_id",
        "sample_id",
        "origin_day",
        "target_day",
        "day_in_study",
        "study_interval",
        "calendar_date",
        "days_since_last_known_menses",
        "menses_onset_missing",
        "self_report__flow_volume__mean",
        "hormone_history_lh",
        "mira_phase",
        "target_pdg_log1p",
        "future_sleep_score",
        "modulo_28_time",
    ]
    for name in forbidden:
        with pytest.raises(ValueError):
            assert_v1_feature_names_safe([name])


def test_h3p_source_boundary_contains_no_global_fold0_or_discrete_mode_selection() -> None:
    sources = "\n".join(
        inspect.getsource(module)
        for module in (layer1, layer2, covariance, uncertainty, model)
    ).lower()
    for forbidden in (
        "run_fold0_validation",
        "choose_covariance_candidate",
        "selected_covariance_mode",
        "covariance_candidates",
        "selection_scope",
    ):
        assert forbidden not in sources


def test_budget_covariances_follow_exact_calibration_protocol_not_sigma_over_k() -> None:
    layer1_oof, plan = _synthetic_layer1_oof_and_plan()
    core = fit_layer2_core(layer1_oof, plan)

    core.validate()
    assert set(core.psi) == {3, 7}
    assert core.development_participants == 8
    assert not np.allclose(
        core.psi[3].matrix,
        core.sigma_future.matrix / 3.0,
        rtol=1e-5,
        atol=1e-8,
    )
    assert not np.allclose(
        core.psi[7].matrix,
        core.sigma_future.matrix / 7.0,
        rtol=1e-5,
        atol=1e-8,
    )
    assert core.psi[3].estimator == "ledoit_wolf_correlation"
    assert core.psi[7].estimator == "ledoit_wolf_correlation"

