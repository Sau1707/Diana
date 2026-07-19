"""Fold-local continuous correlation shrinkage for Diana-H3P."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from model.diana_h3p.contracts import CovarianceEstimate


def _finalize_covariance(
    raw_covariance: np.ndarray,
    standardized_samples: np.ndarray,
    *,
    absolute_floor: float,
    relative_floor: float,
    estimator: str,
    sample_count: int,
    near_diagonal_threshold: float,
) -> CovarianceEstimate:
    raw = np.asarray(raw_covariance, dtype=float)
    standardized = np.asarray(standardized_samples, dtype=float)
    if raw.shape != (3, 3) or standardized.ndim != 2 or standardized.shape[1] != 3:
        raise ValueError("Continuous shrinkage requires 3-dimensional samples")
    raw = (raw + raw.T) / 2.0
    variances = np.maximum(np.diag(raw), 0.0)
    scale_floor = max(float(absolute_floor), float(relative_floor) * max(float(variances.mean()), 1.0))
    scales = np.sqrt(np.maximum(variances, scale_floor))
    denominator = np.outer(scales, scales)
    correlation = np.divide(raw, denominator, out=np.zeros_like(raw), where=denominator > 0)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    if len(standardized) < 2:
        raise ValueError("Continuous shrinkage requires at least two samples")
    alpha = float(LedoitWolf(assume_centered=True).fit(standardized).shrinkage_)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    shrunk_correlation = (1.0 - alpha) * correlation + alpha * np.eye(3)
    covariance = shrunk_correlation * denominator
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(
        float(absolute_floor),
        float(relative_floor) * max(float(np.trace(covariance)), 0.0) / 3.0,
    )
    floored = np.maximum(eigenvalues, floor)
    covariance = (eigenvectors * floored) @ eigenvectors.T
    covariance = (covariance + covariance.T) / 2.0
    final_scales = np.sqrt(np.maximum(np.diag(covariance), floor))
    final_corr = covariance / np.outer(final_scales, final_scales)
    off_diagonal = final_corr[np.triu_indices(3, k=1)]
    estimate = CovarianceEstimate(
        matrix=covariance,
        shrinkage=alpha,
        eigenvalues_before_floor=tuple(float(x) for x in eigenvalues),
        eigenvalues_after_floor=tuple(float(x) for x in np.linalg.eigvalsh(covariance)),
        eigenvalue_floor=float(floor),
        condition_number=float(np.linalg.cond(covariance)),
        max_abs_off_diagonal_correlation=float(np.max(np.abs(off_diagonal))),
        effectively_near_diagonal=bool(
            np.max(np.abs(off_diagonal)) <= float(near_diagonal_threshold)
        ),
        sample_count=int(sample_count),
        estimator=estimator,
    )
    estimate.validate()
    return estimate


def estimate_shrunk_covariance(
    samples: np.ndarray,
    *,
    absolute_floor: float = 1e-10,
    relative_floor: float = 1e-6,
    near_diagonal_threshold: float = 0.05,
) -> CovarianceEstimate:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("Covariance samples must have shape (n>=2, 3)")
    if not np.isfinite(values).all():
        raise ValueError("Covariance samples must be finite")
    centered = values - values.mean(axis=0, keepdims=True)
    raw = centered.T @ centered / max(len(centered) - 1, 1)
    variances = np.maximum(np.diag(raw), 0.0)
    standard_floor = max(absolute_floor, relative_floor * max(float(variances.mean()), 1.0))
    standardized = centered / np.sqrt(np.maximum(variances, standard_floor))
    return _finalize_covariance(
        raw,
        standardized,
        absolute_floor=absolute_floor,
        relative_floor=relative_floor,
        estimator="ledoit_wolf_correlation",
        sample_count=len(values),
        near_diagonal_threshold=near_diagonal_threshold,
    )


def estimate_balanced_within_covariance(
    residuals: np.ndarray,
    participants: Iterable[object],
    *,
    absolute_floor: float = 1e-10,
    relative_floor: float = 1e-6,
    near_diagonal_threshold: float = 0.05,
) -> CovarianceEstimate:
    values = np.asarray(residuals, dtype=float)
    groups = pd.Series(list(participants), dtype="string")
    if values.ndim != 2 or values.shape[1] != 3 or len(values) != len(groups):
        raise ValueError("Within covariance inputs must be aligned (n, 3)")
    if not np.isfinite(values).all():
        raise ValueError("Within covariance residuals must be finite")
    contributions: list[np.ndarray] = []
    centered_parts: list[tuple[np.ndarray, int]] = []
    for participant in sorted(groups.unique()):
        part = values[groups.eq(participant).to_numpy()]
        if len(part) < 2:
            raise ValueError("Every participant needs at least two future residuals")
        centered = part - part.mean(axis=0, keepdims=True)
        contributions.append(centered.T @ centered / (len(centered) - 1))
        centered_parts.append((centered, len(centered)))
    raw = np.mean(contributions, axis=0)
    variances = np.maximum(np.diag(raw), 0.0)
    standard_floor = max(absolute_floor, relative_floor * max(float(variances.mean()), 1.0))
    total_rows = sum(n for _, n in centered_parts)
    participant_count = len(centered_parts)
    pseudo = np.vstack(
        [
            centered
            * np.sqrt(total_rows / (participant_count * max(n - 1, 1)))
            / np.sqrt(np.maximum(variances, standard_floor))
            for centered, n in centered_parts
        ]
    )
    return _finalize_covariance(
        raw,
        pseudo,
        absolute_floor=absolute_floor,
        relative_floor=relative_floor,
        estimator="participant_balanced_within_ledoit_wolf_correlation",
        sample_count=participant_count,
        near_diagonal_threshold=near_diagonal_threshold,
    )
