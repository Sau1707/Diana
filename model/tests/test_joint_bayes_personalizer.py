from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.v1_task import HORMONES
from model.joint_bayes_personalizer.model import (
    JointBayesPersonalizer,
    add_prior_columns,
    estimate_custom_parameters,
    posterior_update,
)
from model.v1_pipeline import choose_covariance_candidate


def _oof() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for participant in range(8):
        effect = rng.normal(0, 0.15, 3)
        for day in range(12):
            record = {
                "sample_id": f"{participant}-{day}",
                "private_participant_id": str(participant),
                "origin_day": day,
                "target_day": day + 1,
            }
            for index, hormone in enumerate(HORMONES):
                prediction = 1.0 + index + rng.normal(0, 0.1)
                record[f"pred_{hormone}"] = prediction
                record[f"y_{hormone}"] = prediction + effect[index] + rng.normal(0, 0.08)
            rows.append(record)
    return pd.DataFrame(rows)


def test_lambda_endpoints_and_bounds():
    oof = _oof()
    parameters = estimate_custom_parameters(
        oof, mode="diagonal", grid_step=0.1, shrinkage=0.2, floor=1e-6
    )
    assert all(0 <= value <= 1 for value in parameters.lambdas.values())
    medians = parameters.medians
    zero = add_prior_columns(oof, medians, {h: 0.0 for h in HORMONES})
    one = add_prior_columns(oof, medians, {h: 1.0 for h in HORMONES})
    for hormone in HORMONES:
        assert np.allclose(zero[f"prior_{hormone}"], medians[hormone])
        assert np.allclose(one[f"prior_{hormone}"], oof[f"pred_{hormone}"])


def test_covariances_psd_and_posterior_contract():
    oof = _oof()
    for mode in ("diagonal", "full"):
        parameters = estimate_custom_parameters(
            oof, mode=mode, grid_step=0.2, shrinkage=0.2, floor=1e-6
        )
        assert parameters.sigma_a.shape == (3, 3)
        assert parameters.sigma_e.shape == (3, 3)
        assert np.linalg.eigvalsh(parameters.sigma_a).min() > 0
        assert np.linalg.eigvalsh(parameters.sigma_e).min() > 0
        mean0, covariance0 = posterior_update(
            parameters.sigma_a, parameters.sigma_e, np.empty((0, 3))
        )
        _, covariance3 = posterior_update(
            parameters.sigma_a, parameters.sigma_e, np.zeros((3, 3))
        )
        _, covariance7 = posterior_update(
            parameters.sigma_a, parameters.sigma_e, np.zeros((7, 3))
        )
        assert np.allclose(mean0, 0)
        assert np.allclose(covariance0, parameters.sigma_a)
        assert np.all(np.linalg.eigvalsh(covariance0 - covariance3) >= -1e-10)
        assert np.all(np.linalg.eigvalsh(covariance3 - covariance7) >= -1e-10)


def test_prediction_intervals_and_authorized_labels_only():
    oof = _oof()
    parameters = estimate_custom_parameters(
        oof, mode="full", grid_step=0.2, shrinkage=0.2, floor=1e-6
    )
    participant = oof.loc[oof["private_participant_id"].eq("0")].copy()
    base = participant.iloc[7:].copy()
    calibration = add_prior_columns(
        participant.iloc[:3].copy(), parameters.medians, parameters.lambdas
    )
    model = JointBayesPersonalizer(parameters)
    first, _ = model.predict(
        base, calibration, budget=3, interval_multipliers={h: 1.2 for h in HORMONES}
    )
    forbidden_changed = calibration.copy()
    # A non-authorized later row is not present in the calibration view and cannot affect output.
    second, _ = model.predict(
        base, forbidden_changed, budget=3, interval_multipliers={h: 1.2 for h in HORMONES}
    )
    for hormone in HORMONES:
        assert np.allclose(first[f"pred_{hormone}"], second[f"pred_{hormone}"])
        assert np.isfinite(first[f"lower_{hormone}"]).all()
        assert (first[f"lower_{hormone}"] >= 0).all()
        assert (first[f"lower_{hormone}"] <= first[f"pred_{hormone}"]).all()
        assert (first[f"pred_{hormone}"] <= first[f"upper_{hormone}"]).all()


def test_covariance_fallback_uses_strongest_valid_candidate():
    candidates = {"diagonal": {"score": 0.606}, "full": {"score": 0.602}}
    failed = {
        "diagonal": {"success_gate_passed": False},
        "full": {"success_gate_passed": False},
    }
    assert choose_covariance_candidate(candidates, failed) == "full"
    tied = {"diagonal": {"score": 0.6}, "full": {"score": 0.6}}
    assert choose_covariance_candidate(tied, failed) == "diagonal"
