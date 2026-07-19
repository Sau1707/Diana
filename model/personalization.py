"""Standardized independent residual-intercept personalization for baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from benchmark.v1_task import HORMONES


@dataclass(frozen=True)
class DiagonalAdapterParameters:
    between_variance: dict[str, float]
    within_variance: dict[str, float]
    variance_floor: float = 1e-6


def estimate_diagonal_adapter(
    oof: pd.DataFrame, *, variance_floor: float = 1e-6
) -> DiagonalAdapterParameters:
    required = {"private_participant_id"}
    for hormone in HORMONES:
        required |= {f"y_{hormone}", f"pred_{hormone}"}
    if not required <= set(oof.columns):
        raise ValueError(f"OOF adapter data missing {sorted(required-set(oof))}")
    between: dict[str, float] = {}
    within: dict[str, float] = {}
    for hormone in HORMONES:
        work = oof[["private_participant_id", f"y_{hormone}", f"pred_{hormone}"]].copy()
        work["residual"] = work[f"y_{hormone}"] - work[f"pred_{hormone}"]
        grouped = work.groupby("private_participant_id", sort=True)["residual"]
        means = grouped.mean()
        variances = grouped.var(ddof=1).fillna(0.0)
        counts = grouped.size().astype(float)
        within_value = float(np.mean(variances.to_numpy(float)))
        mean_noise = float(np.mean((variances / counts).to_numpy(float)))
        between_value = float(np.var(means.to_numpy(float), ddof=1) - mean_noise)
        within[hormone] = max(float(variance_floor), within_value)
        between[hormone] = max(float(variance_floor), between_value)
    return DiagonalAdapterParameters(between, within, float(variance_floor))


def independent_offsets(
    calibration_residuals: pd.DataFrame,
    parameters: DiagonalAdapterParameters,
    *,
    budget: int,
) -> dict[str, dict[str, float]]:
    budget = int(budget)
    participants = sorted(
        calibration_residuals["private_participant_id"].astype(str).unique()
    )
    if budget == 0:
        return {participant: {hormone: 0.0 for hormone in HORMONES} for participant in participants}
    counts = calibration_residuals.groupby("private_participant_id").size()
    if counts.empty or not counts.eq(budget).all():
        raise ValueError(f"Adapter requires exactly K={budget} residuals per participant")
    output: dict[str, dict[str, float]] = {}
    for participant, group in calibration_residuals.groupby(
        "private_participant_id", sort=True
    ):
        output[str(participant)] = {}
        for hormone in HORMONES:
            mean_residual = float(group[f"residual_{hormone}"].mean())
            tau2 = parameters.between_variance[hormone]
            sigma2 = parameters.within_variance[hormone]
            shrinkage = (budget * tau2) / (sigma2 + budget * tau2)
            output[str(participant)][hormone] = float(shrinkage * mean_residual)
    return output


def apply_independent_offsets(
    base_predictions: pd.DataFrame,
    offsets: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    required = {"sample_id", "private_participant_id"} | {
        f"pred_{hormone}" for hormone in HORMONES
    }
    if not required <= set(base_predictions.columns):
        raise ValueError("Base prediction frame is incomplete")
    output = base_predictions.copy()
    for hormone in HORMONES:
        adjustment = output["private_participant_id"].astype(str).map(
            {participant: values[hormone] for participant, values in offsets.items()}
        )
        if adjustment.isna().any():
            raise ValueError("Missing participant personalization offset")
        output[f"pred_{hormone}"] = np.maximum(
            output[f"pred_{hormone}"].to_numpy(float) + adjustment.to_numpy(float),
            0.0,
        )
    return output
