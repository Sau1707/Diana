from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from benchmark.v1_contracts import TARGET_LOG_COLUMNS, TARGET_RAW_COLUMNS, V1PreparedBundle
from benchmark.v1_personalization import build_personalization_plan
from benchmark.v1_synthetic import make_synthetic_bundle
from benchmark.v1_task import HORMONES, load_v1_config
from model.diana_h3p.contracts import (
    H3PParameters,
    Layer2Parameters,
    StackSelection,
)
from model.diana_h3p.covariance import estimate_shrunk_covariance
from model.diana_h3p.layer2 import predict_with_layer2
from model.diana_h3p.layer2_backend import create_backend
from model.diana_h3p.pipeline import _authorized_calibration
from model.diana_h3p.serialization import load_parameters, save_parameters


def _parameters() -> H3PParameters:
    rng = np.random.default_rng(9)
    sigma_a = estimate_shrunk_covariance(rng.normal(size=(20, 3)))
    psi3 = estimate_shrunk_covariance(rng.normal(scale=0.8, size=(20, 3)))
    psi7 = estimate_shrunk_covariance(rng.normal(scale=0.5, size=(20, 3)))
    sigma_future = estimate_shrunk_covariance(rng.normal(scale=0.7, size=(80, 3)))
    from model.diana_h3p.contracts import Layer2Core

    return H3PParameters(
        stack=StackSelection(
            weights={
                hormone: {
                    "population_median": 0.4,
                    "wearable_ridge": 0.2,
                    "catboost": 0.4,
                }
                for hormone in HORMONES
            },
            participant_macro_mae={hormone: 0.2 for hormone in HORMONES},
            grid_step=0.1,
        ),
        layer2=Layer2Parameters(
            core=Layer2Core(
                sigma_a=sigma_a,
                psi={3: psi3, 7: psi7},
                sigma_future=sigma_future,
                development_participants=16,
            ),
            interval_multipliers={
                budget: {hormone: 1.25 for hormone in HORMONES}
                for budget in (0, 3, 7)
            },
        ),
        backend="numpy",
        seed=20260719,
    )


def _base_rows(participants=("a", "b"), rows=4) -> pd.DataFrame:
    records = []
    for participant in participants:
        for index in range(rows):
            records.append(
                {
                    "sample_id": f"{participant}-{index}",
                    "private_participant_id": participant,
                    **{
                        f"pred_{hormone}": 0.5 + hormone_index + 0.01 * index
                        for hormone_index, hormone in enumerate(HORMONES)
                    },
                }
            )
    return pd.DataFrame(records)


def test_k0_exactly_equals_layer1_and_intervals_are_valid() -> None:
    parameters = _parameters()
    base = _base_rows()
    empty = pd.DataFrame()
    predicted, posterior = predict_with_layer2(
        base,
        empty,
        parameters.layer2,
        budget=0,
        backend=create_backend("numpy"),
    )
    for hormone in HORMONES:
        assert np.array_equal(predicted[f"pred_{hormone}"], base[f"pred_{hormone}"])
        assert np.isfinite(predicted[f"lower_{hormone}"]).all()
        assert (predicted[f"lower_{hormone}"] >= 0).all()
        assert (predicted[f"lower_{hormone}"] <= predicted[f"pred_{hormone}"]).all()
        assert (predicted[f"pred_{hormone}"] <= predicted[f"upper_{hormone}"]).all()
    assert np.array_equal(posterior["posterior_means"], np.zeros((2, 3)))


def test_exact_k_cardinality_and_no_scoring_truth_required() -> None:
    parameters = _parameters()
    base = _base_rows(rows=2)
    calibration_records = []
    for participant in ("a", "b"):
        for index in range(3):
            calibration_records.append(
                {
                    "sample_id": f"cal-{participant}-{index}",
                    "private_participant_id": participant,
                    **{
                        f"pred_{hormone}": 0.5 + hormone_index
                        for hormone_index, hormone in enumerate(HORMONES)
                    },
                    **{
                        f"y_{hormone}": 0.6 + hormone_index
                        for hormone_index, hormone in enumerate(HORMONES)
                    },
                }
            )
    calibration = pd.DataFrame(calibration_records)
    predicted, _ = predict_with_layer2(
        base,
        calibration,
        parameters.layer2,
        budget=3,
        backend=create_backend("numpy"),
    )
    assert not any(column.startswith("y_") for column in base.columns)
    assert len(predicted) == len(base)
    bad = pd.concat([calibration, calibration.iloc[[0]]], ignore_index=True)
    try:
        predict_with_layer2(
            base, bad, parameters.layer2, budget=3, backend=create_backend("numpy")
        )
    except ValueError as error:
        assert "exactly K=3" in str(error)
    else:
        raise AssertionError("Extra calibration truth was accepted")


def test_mutating_rank_four_truth_cannot_change_k3_authorization() -> None:
    config = load_v1_config("configs/hormonbench_v1.yaml")
    bundle, _ = make_synthetic_bundle(config)
    participants = {1, 2, 3, 4}
    plan = build_personalization_plan(bundle, participants)
    test_rows = bundle.frame.loc[
        bundle.frame["private_participant_id"].isin(participants)
    ]
    layer1 = test_rows[
        ["sample_id", "private_participant_id", "origin_day", "target_day"]
    ].copy()
    for hormone_index, hormone in enumerate(HORMONES):
        layer1[f"pred_{hormone}"] = 0.4 + hormone_index
    original = _authorized_calibration(bundle, layer1, plan, 3)
    modified_frame = bundle.frame.copy()
    rank_four_ids = []
    for participant in participants:
        ordered = plan.calibration_candidates.loc[
            plan.calibration_candidates["private_participant_id"].eq(participant)
        ].sort_values("calibration_rank")
        rank_four_ids.append(str(ordered.iloc[3]["sample_id"]))
    mask = modified_frame["sample_id"].astype(str).isin(rank_four_ids)
    for hormone in HORMONES:
        modified_frame.loc[mask, TARGET_LOG_COLUMNS[hormone]] += 2.0
        modified_frame.loc[mask, TARGET_RAW_COLUMNS[hormone]] = np.expm1(
            modified_frame.loc[mask, TARGET_LOG_COLUMNS[hormone]]
        )
    modified_bundle = V1PreparedBundle(modified_frame, copy.deepcopy(bundle.metadata))
    changed = _authorized_calibration(modified_bundle, layer1, plan, 3)
    pd.testing.assert_frame_equal(original, changed)


def test_private_parameter_serialization_round_trip(tmp_path) -> None:
    parameters = _parameters()
    destination = tmp_path / "parameters.json"
    save_parameters(parameters, destination)
    restored = load_parameters(destination)
    restored.validate()
    assert restored.stack.weights == parameters.stack.weights
    assert np.allclose(
        restored.layer2.core.psi[7].matrix, parameters.layer2.core.psi[7].matrix
    )
