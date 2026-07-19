"""Shrunk wearable prior plus multivariate empirical-Bayes personalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from benchmark.v1_task import HORMONES
from model.v1_common import weighted_quantile


VECTOR_COLUMNS = [f"residual_{hormone}" for hormone in HORMONES]


def _project_psd(matrix: np.ndarray, floor: float) -> np.ndarray:
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, float(floor))
    projected = vectors @ np.diag(values) @ vectors.T
    return (projected + projected.T) / 2


def _shrink_covariance(
    matrix: np.ndarray, *, mode: str, shrinkage: float, floor: float
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    diagonal = np.diag(np.maximum(np.diag(matrix), float(floor)))
    if mode == "diagonal":
        result = diagonal
    elif mode == "full":
        result = (1.0 - float(shrinkage)) * matrix + float(shrinkage) * diagonal
    else:
        raise ValueError("Covariance mode must be diagonal or full")
    return _project_psd(result, floor)


def participant_equal_medians(data: pd.DataFrame) -> dict[str, float]:
    medians: dict[str, float] = {}
    for hormone in HORMONES:
        grouped = data.groupby("private_participant_id", sort=True)[f"y_{hormone}"].median()
        medians[hormone] = float(grouped.median())
    return medians


def _participant_macro_mae(
    data: pd.DataFrame, truth_column: str, prediction: np.ndarray
) -> float:
    work = pd.DataFrame(
        {
            "private_participant_id": data["private_participant_id"].astype(str),
            "absolute_error": np.abs(data[truth_column].to_numpy(float) - prediction),
        }
    )
    return float(work.groupby("private_participant_id")["absolute_error"].mean().mean())


def learn_lambdas(
    oof: pd.DataFrame,
    medians: Mapping[str, float],
    *,
    grid_step: float,
) -> dict[str, float]:
    if not 0 < float(grid_step) <= 1:
        raise ValueError("lambda grid step must lie in (0,1]")
    grid = np.round(np.arange(0.0, 1.0 + grid_step / 2, grid_step), 12)
    output: dict[str, float] = {}
    for hormone in HORMONES:
        median = float(medians[hormone])
        cat = oof[f"pred_{hormone}"].to_numpy(float)
        scores: list[tuple[float, float]] = []
        for value in grid:
            prediction = median + float(value) * (cat - median)
            score = _participant_macro_mae(oof, f"y_{hormone}", prediction)
            scores.append((score, float(value)))
        best_score = min(score for score, _ in scores)
        tied = [value for score, value in scores if abs(score - best_score) <= 1e-12]
        output[hormone] = float(min(tied))
    return output


def add_prior_columns(
    data: pd.DataFrame,
    medians: Mapping[str, float],
    lambdas: Mapping[str, float],
) -> pd.DataFrame:
    output = data.copy()
    for hormone in HORMONES:
        value = float(lambdas[hormone])
        if not 0.0 <= value <= 1.0:
            raise ValueError("lambda must remain in [0,1]")
        median = float(medians[hormone])
        output[f"prior_{hormone}"] = median + value * (
            output[f"pred_{hormone}"].to_numpy(float) - median
        )
        if f"y_{hormone}" in output:
            output[f"residual_{hormone}"] = (
                output[f"y_{hormone}"].to_numpy(float)
                - output[f"prior_{hormone}"].to_numpy(float)
            )
    return output


def estimate_residual_covariances(
    residual_data: pd.DataFrame,
    *,
    mode: str,
    shrinkage: float,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not set(VECTOR_COLUMNS) <= set(residual_data.columns):
        raise ValueError("Residual data lacks the three-hormone vector")
    participant_means: list[np.ndarray] = []
    within_covariances: list[np.ndarray] = []
    mean_noise: list[np.ndarray] = []
    for _, group in residual_data.groupby("private_participant_id", sort=True):
        matrix = group[VECTOR_COLUMNS].to_numpy(float)
        if not np.isfinite(matrix).all():
            raise ValueError("Residual covariance data must be finite")
        participant_means.append(matrix.mean(axis=0))
        if len(matrix) > 1:
            centered = matrix - matrix.mean(axis=0)
            covariance = centered.T @ centered / (len(matrix) - 1)
        else:
            covariance = np.zeros((len(HORMONES), len(HORMONES)))
        within_covariances.append(covariance)
        mean_noise.append(covariance / max(len(matrix), 1))
    if len(participant_means) < 2:
        raise ValueError("Covariance estimation requires at least two participants")
    within_raw = np.mean(within_covariances, axis=0)
    means = np.vstack(participant_means)
    between_raw = np.cov(means, rowvar=False, ddof=1) - np.mean(mean_noise, axis=0)
    sigma_e = _shrink_covariance(
        within_raw, mode=mode, shrinkage=shrinkage, floor=floor
    )
    sigma_a = _shrink_covariance(
        between_raw, mode=mode, shrinkage=shrinkage, floor=floor
    )
    return sigma_a, sigma_e


def posterior_update(
    sigma_a: np.ndarray,
    sigma_e: np.ndarray,
    calibration_residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_a = np.asarray(sigma_a, dtype=float)
    sigma_e = np.asarray(sigma_e, dtype=float)
    residuals = np.asarray(calibration_residuals, dtype=float)
    dimension = len(HORMONES)
    if sigma_a.shape != (dimension, dimension) or sigma_e.shape != (
        dimension,
        dimension,
    ):
        raise ValueError("Residual covariance matrices must be 3x3")
    if residuals.size == 0:
        return np.zeros(dimension, dtype=float), sigma_a.copy()
    residuals = residuals.reshape(-1, dimension)
    identity = np.eye(dimension)
    precision_a = np.linalg.solve(sigma_a, identity)
    precision_e = np.linalg.solve(sigma_e, identity)
    precision = precision_a + len(residuals) * precision_e
    posterior_covariance = np.linalg.solve(precision, identity)
    right_hand_side = np.linalg.solve(sigma_e, residuals.sum(axis=0))
    posterior_mean = np.linalg.solve(precision, right_hand_side)
    return posterior_mean, (posterior_covariance + posterior_covariance.T) / 2


@dataclass(frozen=True)
class CustomParameters:
    medians: dict[str, float]
    lambdas: dict[str, float]
    sigma_a: np.ndarray
    sigma_e: np.ndarray
    covariance_mode: str
    covariance_shrinkage: float
    eigenvalue_floor: float


def estimate_custom_parameters(
    oof: pd.DataFrame,
    *,
    mode: str,
    grid_step: float,
    shrinkage: float,
    floor: float,
) -> CustomParameters:
    medians = participant_equal_medians(oof)
    lambdas = learn_lambdas(oof, medians, grid_step=grid_step)
    residuals = add_prior_columns(oof, medians, lambdas)
    sigma_a, sigma_e = estimate_residual_covariances(
        residuals, mode=mode, shrinkage=shrinkage, floor=floor
    )
    return CustomParameters(
        medians=medians,
        lambdas=lambdas,
        sigma_a=sigma_a,
        sigma_e=sigma_e,
        covariance_mode=mode,
        covariance_shrinkage=float(shrinkage),
        eigenvalue_floor=float(floor),
    )


class JointBayesPersonalizer:
    model_name = "joint_bayes_personalizer"
    model_version = "1.0.0"

    def __init__(self, parameters: CustomParameters):
        self.parameters = parameters

    def predict(
        self,
        base_predictions: pd.DataFrame,
        calibration: pd.DataFrame,
        *,
        budget: int,
        interval_multipliers: Mapping[str, float] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        base = add_prior_columns(
            base_predictions,
            self.parameters.medians,
            self.parameters.lambdas,
        )
        output = base.copy()
        posterior_covariances: dict[str, np.ndarray] = {}
        for participant, group in output.groupby("private_participant_id", sort=True):
            participant_key = str(participant)
            if int(budget) == 0:
                residual_matrix = np.empty((0, len(HORMONES)))
            else:
                authorized = calibration.loc[
                    calibration["private_participant_id"].astype(str).eq(participant_key)
                ].sort_values("target_day")
                if len(authorized) != int(budget):
                    raise ValueError(
                        f"Participant calibration requires exactly K={int(budget)} rows"
                    )
                residual_matrix = np.column_stack(
                    [
                        authorized[f"y_{hormone}"].to_numpy(float)
                        - authorized[f"prior_{hormone}"].to_numpy(float)
                        for hormone in HORMONES
                    ]
                )
            posterior_mean, posterior_covariance = posterior_update(
                self.parameters.sigma_a,
                self.parameters.sigma_e,
                residual_matrix,
            )
            posterior_covariances[participant_key] = posterior_covariance
            indices = group.index
            for index, hormone in enumerate(HORMONES):
                output.loc[indices, f"pred_{hormone}"] = np.maximum(
                    group[f"prior_{hormone}"].to_numpy(float) + posterior_mean[index],
                    0.0,
                )
                if interval_multipliers is not None:
                    predictive_variance = float(
                        self.parameters.sigma_e[index, index]
                        + posterior_covariance[index, index]
                    )
                    half_width = float(interval_multipliers[hormone]) * np.sqrt(
                        max(predictive_variance, self.parameters.eigenvalue_floor)
                    )
                    point = output.loc[indices, f"pred_{hormone}"].to_numpy(float)
                    output.loc[indices, f"lower_{hormone}"] = np.maximum(
                        point - half_width, 0.0
                    )
                    output.loc[indices, f"upper_{hormone}"] = point + half_width
        return output, posterior_covariances

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "covariance_mode": self.parameters.covariance_mode,
            "lambda": dict(self.parameters.lambdas),
            "covariance_shrinkage": self.parameters.covariance_shrinkage,
            "eigenvalue_floor": self.parameters.eigenvalue_floor,
            "target_space": "log1p",
            "reference_role": "custom_reference",
        }


def _participant_common_suffix(group: pd.DataFrame, common_budget: int = 7) -> pd.DataFrame:
    ordered = group.sort_values(["target_day", "sample_id"]).reset_index(drop=True)
    if len(ordered) < common_budget:
        raise ValueError("Conformal participant lacks seven eligible targets")
    threshold = int(ordered.iloc[common_budget - 1]["target_day"])
    return ordered.loc[ordered["origin_day"].ge(threshold)].copy()


def learn_conformal_multipliers(
    oof: pd.DataFrame,
    *,
    budget: int,
    mode: str,
    grid_step: float,
    shrinkage: float,
    floor: float,
    quantile: float,
) -> dict[str, float]:
    """Leave-one-participant-out participant-balanced interval calibration."""

    scores: dict[str, list[float]] = {hormone: [] for hormone in HORMONES}
    weights: dict[str, list[float]] = {hormone: [] for hormone in HORMONES}
    participants = sorted(oof["private_participant_id"].astype(str).unique())
    for held in participants:
        other = oof.loc[~oof["private_participant_id"].astype(str).eq(held)].copy()
        held_rows = oof.loc[oof["private_participant_id"].astype(str).eq(held)].copy()
        parameters = estimate_custom_parameters(
            other,
            mode=mode,
            grid_step=grid_step,
            shrinkage=shrinkage,
            floor=floor,
        )
        held_with_prior = add_prior_columns(
            held_rows, parameters.medians, parameters.lambdas
        )
        ordered = held_with_prior.sort_values(["target_day", "sample_id"]).reset_index(
            drop=True
        )
        suffix = _participant_common_suffix(ordered)
        calibration = ordered.iloc[: int(budget)].copy() if int(budget) else ordered.iloc[:0].copy()
        model = JointBayesPersonalizer(parameters)
        predicted, posterior = model.predict(
            suffix,
            calibration,
            budget=int(budget),
            interval_multipliers=None,
        )
        covariance = posterior[held]
        row_weight = 1.0 / len(predicted)
        for index, hormone in enumerate(HORMONES):
            variance = float(parameters.sigma_e[index, index] + covariance[index, index])
            scale = np.sqrt(max(variance, floor))
            standardized = np.abs(
                predicted[f"y_{hormone}"].to_numpy(float)
                - predicted[f"pred_{hormone}"].to_numpy(float)
            ) / scale
            scores[hormone].extend(standardized.tolist())
            weights[hormone].extend([row_weight] * len(standardized))
    return {
        hormone: max(
            float(floor),
            weighted_quantile(
                np.asarray(scores[hormone]),
                np.asarray(weights[hormone]),
                float(quantile),
            ),
        )
        for hormone in HORMONES
    }
