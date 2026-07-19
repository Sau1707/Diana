"""Layer 2: budget-aware joint empirical-Bayes personalization."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from benchmark.v1_personalization import PersonalizationPlan
from benchmark.v1_task import HORMONES
from model.diana_h3p.contracts import CovarianceEstimate, Layer2Core, Layer2Parameters
from model.diana_h3p.covariance import (
    estimate_balanced_within_covariance,
    estimate_shrunk_covariance,
)


RESIDUAL_COLUMNS = tuple(f"residual_{hormone}" for hormone in HORMONES)


def add_residual_columns(layer1_rows: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "private_participant_id", "origin_day", "target_day"}
    required.update(f"y_{hormone}" for hormone in HORMONES)
    required.update(f"pred_{hormone}" for hormone in HORMONES)
    missing = sorted(required - set(layer1_rows.columns))
    if missing:
        raise ValueError(f"Layer-2 rows missing {missing}")
    output = layer1_rows.copy()
    for hormone in HORMONES:
        output[f"residual_{hormone}"] = (
            output[f"y_{hormone}"].to_numpy(float)
            - output[f"pred_{hormone}"].to_numpy(float)
        )
    if not np.isfinite(output.loc[:, RESIDUAL_COLUMNS].to_numpy(float)).all():
        raise ValueError("Layer-2 residuals must be finite")
    return output


def _authorized_rows(
    residuals: pd.DataFrame,
    plan_rows: pd.DataFrame,
) -> pd.DataFrame:
    plan = plan_rows[["sample_id", "private_participant_id"]].copy()
    plan["sample_id"] = plan["sample_id"].astype(str)
    available = residuals.copy()
    available["sample_id"] = available["sample_id"].astype(str)
    joined = plan.merge(
        available,
        on=["sample_id", "private_participant_id"],
        how="inner",
        validate="one_to_one",
    )
    return joined


def fit_layer2_core(
    layer1_oof: pd.DataFrame,
    plan: PersonalizationPlan,
    *,
    absolute_floor: float = 1e-10,
    relative_floor: float = 1e-6,
    near_diagonal_threshold: float = 0.05,
) -> Layer2Core:
    """Fit fold-local covariance parameters from development grouped OOF residuals."""

    residuals = add_residual_columns(layer1_oof)
    participants = sorted(residuals["private_participant_id"].astype(str).unique())
    if len(participants) < 4:
        raise ValueError("Layer 2 requires at least four development participants")
    scoring = _authorized_rows(residuals, plan.scoring_rows)
    scoring_participants = sorted(scoring["private_participant_id"].astype(str).unique())
    if scoring_participants != participants:
        raise ValueError("Every development participant needs common-suffix residuals")
    proxy = (
        scoring.assign(
            private_participant_id=scoring["private_participant_id"].astype(str)
        )
        .groupby("private_participant_id", sort=True)[list(RESIDUAL_COLUMNS)]
        .mean()
        .loc[participants]
    )
    sigma_a = estimate_shrunk_covariance(
        proxy.to_numpy(float),
        absolute_floor=absolute_floor,
        relative_floor=relative_floor,
        near_diagonal_threshold=near_diagonal_threshold,
    )
    calibration = plan.calibration_candidates.copy()
    calibration["sample_id"] = calibration["sample_id"].astype(str)
    calibration = calibration.loc[
        calibration["private_participant_id"].astype(str).isin(participants)
    ]
    calibration = calibration.merge(
        residuals[["sample_id", "private_participant_id", *RESIDUAL_COLUMNS]].assign(
            sample_id=lambda x: x["sample_id"].astype(str)
        ),
        on=["sample_id", "private_participant_id"],
        validate="one_to_one",
    )
    psi: dict[int, CovarianceEstimate] = {}
    for budget in (3, 7):
        authorized = calibration.loc[calibration["calibration_rank"].le(budget)]
        counts = authorized.groupby("private_participant_id").size()
        if len(counts) != len(participants) or not counts.eq(budget).all():
            raise ValueError(f"Development calibration does not have exact K={budget}")
        means = (
            authorized.assign(
                private_participant_id=authorized["private_participant_id"].astype(str)
            )
            .groupby("private_participant_id", sort=True)[list(RESIDUAL_COLUMNS)]
            .mean()
            .loc[participants]
        )
        deviations = means.to_numpy(float) - proxy.to_numpy(float)
        psi[budget] = estimate_shrunk_covariance(
            deviations,
            absolute_floor=absolute_floor,
            relative_floor=relative_floor,
            near_diagonal_threshold=near_diagonal_threshold,
        )
    centered = scoring.copy()
    centered_values = centered.loc[:, RESIDUAL_COLUMNS].to_numpy(
        dtype=float, copy=True
    )
    proxy_map = proxy.to_dict(orient="index")
    for row_index, participant in enumerate(
        centered["private_participant_id"].astype(str)
    ):
        centered_values[row_index] -= np.asarray(
            [proxy_map[participant][column] for column in RESIDUAL_COLUMNS], dtype=float
        )
    sigma_future = estimate_balanced_within_covariance(
        centered_values,
        centered["private_participant_id"].astype(str),
        absolute_floor=absolute_floor,
        relative_floor=relative_floor,
        near_diagonal_threshold=near_diagonal_threshold,
    )
    core = Layer2Core(
        sigma_a=sigma_a,
        psi={3: psi[3], 7: psi[7]},
        sigma_future=sigma_future,
        development_participants=len(participants),
    )
    core.validate()
    return core


def calibration_residual_means(
    calibration: pd.DataFrame,
    participants: list[str],
    *,
    budget: int,
) -> np.ndarray:
    budget = int(budget)
    if budget == 0:
        if not calibration.empty:
            raise ValueError("K=0 must receive no calibration truth")
        return np.zeros((len(participants), 3), dtype=float)
    required = {"private_participant_id"}
    required.update(f"y_{hormone}" for hormone in HORMONES)
    required.update(f"pred_{hormone}" for hormone in HORMONES)
    missing = sorted(required - set(calibration.columns))
    if missing:
        raise ValueError(f"Calibration rows missing {missing}")
    work = calibration.copy()
    work["private_participant_id"] = work["private_participant_id"].astype(str)
    counts = work.groupby("private_participant_id").size()
    if sorted(counts.index) != sorted(participants) or not counts.eq(budget).all():
        raise ValueError(f"Every participant requires exactly K={budget} labels")
    for hormone in HORMONES:
        work[f"residual_{hormone}"] = (
            work[f"y_{hormone}"].to_numpy(float)
            - work[f"pred_{hormone}"].to_numpy(float)
        )
    return (
        work.groupby("private_participant_id", sort=True)[list(RESIDUAL_COLUMNS)]
        .mean()
        .loc[participants]
        .to_numpy(float)
    )


def predict_with_layer2(
    base_rows: pd.DataFrame,
    calibration: pd.DataFrame,
    parameters: Layer2Parameters,
    *,
    budget: int,
    backend: object,
    include_intervals: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Apply one fold's fixed Layer-2 parameters without reading scoring truth."""

    parameters.validate()
    budget = int(budget)
    if budget not in {0, 3, 7}:
        raise ValueError("H3P budget must be 0, 3, or 7")
    required = {"sample_id", "private_participant_id"}
    required.update(f"pred_{hormone}" for hormone in HORMONES)
    missing = sorted(required - set(base_rows.columns))
    if missing:
        raise ValueError(f"Layer-2 inference rows missing {missing}")
    output = base_rows.copy().reset_index(drop=True)
    participants = sorted(output["private_participant_id"].astype(str).unique())
    participant_lookup = {participant: index for index, participant in enumerate(participants)}
    row_participant_index = np.asarray(
        [participant_lookup[value] for value in output["private_participant_id"].astype(str)],
        dtype=int,
    )
    residual_means = calibration_residual_means(
        calibration, participants, budget=budget
    )
    sigma_a = np.repeat(
        parameters.core.sigma_a.matrix[None, :, :], len(participants), axis=0
    )
    if budget == 0:
        psi = np.repeat(np.eye(3, dtype=float)[None, :, :], len(participants), axis=0)
        calibrated = np.zeros(len(participants), dtype=bool)
    else:
        psi = np.repeat(
            parameters.core.psi[budget].matrix[None, :, :], len(participants), axis=0
        )
        calibrated = np.ones(len(participants), dtype=bool)
    posterior_means, posterior_covariances = backend.posterior_batch(
        sigma_a, psi, residual_means, calibrated
    )
    points = np.column_stack(
        [output[f"pred_{hormone}"].to_numpy(float) for hormone in HORMONES]
    )
    points = np.maximum(points + posterior_means[row_participant_index], 0.0)
    for index, hormone in enumerate(HORMONES):
        output[f"pred_{hormone}"] = points[:, index]
    if include_intervals:
        multipliers = np.asarray(
            [parameters.interval_multipliers[budget][hormone] for hormone in HORMONES],
            dtype=float,
        )
        lower, upper = backend.interval_batch(
            points,
            parameters.core.sigma_future.matrix,
            posterior_covariances,
            row_participant_index,
            multipliers,
        )
        for index, hormone in enumerate(HORMONES):
            output[f"lower_{hormone}"] = lower[:, index]
            output[f"upper_{hormone}"] = upper[:, index]
    return output, {
        "posterior_means": posterior_means,
        "posterior_covariances": posterior_covariances,
        "row_participant_index": row_participant_index,
    }
