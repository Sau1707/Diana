"""Aggregate-only metrics for Hormonbench-mcPHASES v0.

Predictions and truth are represented in ``log1p`` space.  The primary metrics
first average dates within each held-out participant, then give every participant
equal weight.  Participant identifiers never appear in the returned public
summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import HORMONES, TARGET_LOG_COLUMNS


_REQUIRED_JOINED_COLUMNS = {
    "private_participant_id",
    "sample_id",
    "hormone",
    "y_true_log1p",
    "y_true_raw",
    "y_pred",
}


@dataclass(frozen=True)
class MetricResult:
    """Public aggregate metrics plus private in-memory participant metrics."""

    public: dict[str, Any]
    participant: pd.DataFrame


def train_log1p_iqr_scales(train_frame: pd.DataFrame) -> dict[str, float | None]:
    """Compute per-hormone robust scales using training truth only.

    A missing, non-finite, or effectively zero IQR is represented by ``None``.
    In that case the evaluator omits the normalized composite instead of silently
    substituting a scale learned from validation or test data.
    """

    scales: dict[str, float | None] = {}
    for hormone in HORMONES:
        column = TARGET_LOG_COLUMNS[hormone]
        if column not in train_frame:
            raise ValueError(f"Training frame is missing {column}")
        values = pd.to_numeric(train_frame[column], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            scales[hormone] = None
            continue
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float(q75 - q25)
        scales[hormone] = scale if np.isfinite(scale) and scale > 1e-12 else None
    return scales


def participant_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    """Calculate one metric row per participant and hormone.

    ``joined`` is an evaluator-internal table.  Callers must not publish the
    returned frame because it contains the private participant key.
    """

    missing = sorted(_REQUIRED_JOINED_COLUMNS - set(joined.columns))
    if missing:
        raise ValueError(f"Joined evaluation data is missing columns: {missing}")
    if joined.empty:
        raise ValueError("Joined evaluation data is empty")

    work = joined.copy()
    for column in ("y_true_log1p", "y_true_raw", "y_pred"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
        if not np.isfinite(work[column].to_numpy(float)).all():
            raise ValueError(f"{column} must be finite")

    with np.errstate(over="ignore", invalid="ignore"):
        work["y_pred_raw"] = np.expm1(work["y_pred"].to_numpy(float))
    if not np.isfinite(work["y_pred_raw"].to_numpy(float)).all():
        raise ValueError("At least one log1p prediction overflows raw-unit space")

    work["abs_log_error"] = (work["y_pred"] - work["y_true_log1p"]).abs()
    work["squared_log_error"] = (
        work["y_pred"] - work["y_true_log1p"]
    ) ** 2
    work["abs_raw_error"] = (work["y_pred_raw"] - work["y_true_raw"]).abs()

    grouped = work.groupby(
        ["private_participant_id", "hormone"], sort=True, observed=True
    )
    result = grouped.agg(
        observations=("sample_id", "size"),
        log1p_mae=("abs_log_error", "mean"),
        raw_mae=("abs_raw_error", "mean"),
        mean_squared_log_error=("squared_log_error", "mean"),
    ).reset_index()
    result["log1p_rmse"] = np.sqrt(result.pop("mean_squared_log_error"))
    return result


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def summarize_metrics(
    per_participant: pd.DataFrame,
    train_scales: Mapping[str, float | None],
) -> dict[str, Any]:
    """Create a public, participant-macro aggregate summary."""

    required = {
        "private_participant_id",
        "hormone",
        "observations",
        "log1p_mae",
        "raw_mae",
        "log1p_rmse",
    }
    missing = sorted(required - set(per_participant.columns))
    if missing:
        raise ValueError(f"Participant metrics are missing columns: {missing}")

    primary: dict[str, dict[str, Any]] = {}
    secondary: dict[str, dict[str, Any]] = {}
    normalized: dict[str, float | None] = {}

    for hormone in HORMONES:
        rows = per_participant.loc[per_participant["hormone"].eq(hormone)]
        if rows.empty:
            raise ValueError(f"No participant metrics for {hormone}")
        log_mae = rows["log1p_mae"].astype(float)
        primary[hormone] = {
            "participant_macro_log1p_mae": float(log_mae.mean()),
            "participant_median_log1p_mae": float(log_mae.median()),
            "participant_min_log1p_mae": float(log_mae.min()),
            "participant_max_log1p_mae": float(log_mae.max()),
            "test_participants": int(rows["private_participant_id"].nunique()),
            "observations": int(rows["observations"].sum()),
        }
        secondary[hormone] = {
            "participant_macro_raw_mae": float(rows["raw_mae"].mean()),
            "participant_macro_log1p_rmse": float(rows["log1p_rmse"].mean()),
        }
        scale = train_scales.get(hormone)
        normalized[hormone] = (
            float(log_mae.mean() / scale)
            if scale is not None and np.isfinite(scale) and scale > 1e-12
            else None
        )

    finite_normalized = [value for value in normalized.values() if value is not None]
    composite = (
        float(np.mean(finite_normalized))
        if len(finite_normalized) == len(HORMONES)
        else None
    )
    return {
        "primary": primary,
        "secondary": secondary,
        "train_log1p_iqr": {
            hormone: _finite_or_none(scale) if scale is not None else None
            for hormone, scale in train_scales.items()
        },
        "normalized_log1p_mae": normalized,
        "overall_normalized_score": composite,
    }


def calculate_metrics(
    joined: pd.DataFrame,
    train_scales: Mapping[str, float | None],
) -> MetricResult:
    """Calculate private participant rows and their public aggregate summary."""

    per_participant = participant_metrics(joined)
    return MetricResult(
        public=summarize_metrics(per_participant, train_scales),
        participant=per_participant,
    )


def add_reference_comparison(
    model_summary: dict[str, Any],
    model_participant: pd.DataFrame,
    reference_summary: Mapping[str, Any],
    reference_participant: pd.DataFrame,
) -> None:
    """Add causal-calendar skill and improvement counts in place.

    Comparisons are paired by private participant internally.  Only aggregate
    counts are added to the public model summary.
    """

    skill: dict[str, float | None] = {}
    improved: dict[str, dict[str, int]] = {}
    for hormone in HORMONES:
        model_mae = float(
            model_summary["primary"][hormone]["participant_macro_log1p_mae"]
        )
        reference_mae = float(
            reference_summary["primary"][hormone][
                "participant_macro_log1p_mae"
            ]
        )
        skill[hormone] = (
            float(1.0 - model_mae / reference_mae)
            if np.isfinite(reference_mae) and reference_mae > 1e-12
            else None
        )

        left = model_participant.loc[
            model_participant["hormone"].eq(hormone),
            ["private_participant_id", "log1p_mae"],
        ].rename(columns={"log1p_mae": "model_mae"})
        right = reference_participant.loc[
            reference_participant["hormone"].eq(hormone),
            ["private_participant_id", "log1p_mae"],
        ].rename(columns={"log1p_mae": "reference_mae"})
        paired = left.merge(
            right, on="private_participant_id", how="inner", validate="one_to_one"
        )
        if len(paired) != len(left) or len(paired) != len(right):
            raise ValueError("Reference comparison requires identical test participants")
        improved[hormone] = {
            "count": int((paired["model_mae"] < paired["reference_mae"]).sum()),
            "out_of": int(len(paired)),
        }

    model_summary["skill_relative_to_causal_calendar"] = skill
    model_summary["participants_improved_vs_causal_calendar"] = improved

