"""Stable prepared-data and prediction-submission contracts.

This module is intentionally independent of ``model``. External models may import
these public schemas without depending on adapter or evaluator internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


TASK_ID = "hormonbench_mcphases_interval2_nextday_v0"
TRACK = "primary_interval2_nextday"
HORMONES = ("lh", "e3g", "pdg")
HORIZON = 1

PREDICTION_COLUMNS = (
    "sample_id",
    "hormone",
    "horizon",
    "y_pred",
    "model_name",
    "model_version",
    "track",
    "split",
)

PREPARED_ID_COLUMNS = (
    "task_version",
    "sample_id",
    "private_participant_id",
    "origin_day",
    "target_day",
    "history_start_day",
    "history_end_day",
    "cutoff_day",
    "split",
    "config_hash",
    "split_hash",
)

TARGET_RAW_COLUMNS = {h: f"target_{h}_raw" for h in HORMONES}
TARGET_LOG_COLUMNS = {h: f"target_{h}_log1p" for h in HORMONES}
TARGET_COLUMNS = tuple(TARGET_RAW_COLUMNS.values()) + tuple(TARGET_LOG_COLUMNS.values())

# Exact fields and name fragments that may never appear in the main-track feature list.
PROHIBITED_FEATURE_NAMES = {
    "id",
    "participant_id",
    "private_participant_id",
    "phase",
    "mira_phase",
    "fertile_window",
    "lh",
    "estrogen",
    "e3g",
    "pdg",
    "cycle_length",
    "cycle_percentage",
    "normalized_cycle_percentage",
    "future_menses",
    "target_derived_event",
}
PROHIBITED_FEATURE_FRAGMENTS = (
    "future_",
    "backfill",
    "backward_fill",
    "centered_",
    "interpolated_hormone",
    "target_lh",
    "target_e3g",
    "target_pdg",
    "hormone_history",
)


def assert_feature_names_safe(feature_columns: Iterable[str]) -> None:
    """Fail closed when a prepared feature name suggests prohibited information."""

    bad: list[str] = []
    for column in feature_columns:
        lower = str(column).lower()
        tokens = set(lower.replace("__", "_").split("_"))
        if lower in PROHIBITED_FEATURE_NAMES or tokens & PROHIBITED_FEATURE_NAMES:
            bad.append(str(column))
            continue
        if any(fragment in lower for fragment in PROHIBITED_FEATURE_FRAGMENTS):
            bad.append(str(column))
    if bad:
        raise ValueError(f"Prohibited feature columns: {sorted(set(bad))}")


@dataclass(frozen=True)
class PreparedSplit:
    """A split view passed to a model.

    Test views are created with ``include_truth=False`` so model code cannot use test
    labels through this interface.
    """

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    metadata: dict[str, Any]
    split: str
    include_truth: bool

    @property
    def features(self) -> pd.DataFrame:
        return self.frame.loc[:, self.feature_columns].copy()

    @property
    def sample_ids(self) -> pd.Series:
        return self.frame["sample_id"].astype(str).copy()

    @property
    def sample_info(self) -> pd.DataFrame:
        keep = ["sample_id", "origin_day", "target_day", "split"]
        return self.frame.loc[:, keep].copy()

    def target_log1p(self, hormone: str) -> pd.Series:
        if not self.include_truth:
            raise PermissionError("Truth is unavailable in an inference-only split view")
        if hormone not in HORMONES:
            raise KeyError(hormone)
        return self.frame[TARGET_LOG_COLUMNS[hormone]].astype(float).copy()


@dataclass(frozen=True)
class PreparedBundle:
    frame: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(self.metadata["feature_columns"])

    def validate(self) -> None:
        required = set(PREPARED_ID_COLUMNS) | set(TARGET_COLUMNS)
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(f"Prepared bundle is missing columns: {missing}")
        if self.frame["sample_id"].duplicated().any():
            raise ValueError("Prepared sample_id values must be unique")
        if set(self.frame["split"].unique()) - {"train", "validation", "test"}:
            raise ValueError("Prepared bundle has an invalid split value")
        feature_columns = self.feature_columns
        absent = sorted(set(feature_columns) - set(self.frame.columns))
        if absent:
            raise ValueError(f"Metadata references absent features: {absent}")
        assert_feature_names_safe(feature_columns)
        if set(feature_columns) & (set(PREPARED_ID_COLUMNS) | set(TARGET_COLUMNS)):
            raise ValueError("Identifiers or targets cannot be model features")
        for hormone in HORMONES:
            raw = self.frame[TARGET_RAW_COLUMNS[hormone]].to_numpy(float)
            logged = self.frame[TARGET_LOG_COLUMNS[hormone]].to_numpy(float)
            if np.isnan(raw).any() or np.isnan(logged).any():
                raise ValueError("Primary-task truth must be genuinely observed for all hormones")
            if not np.allclose(np.log1p(raw), logged, rtol=1e-10, atol=1e-10):
                raise ValueError(f"Target transform mismatch for {hormone}")
        if not (self.frame["history_end_day"] == self.frame["origin_day"]).all():
            raise ValueError("History must end at the origin day")
        if not (self.frame["history_start_day"] == self.frame["origin_day"] - 13).all():
            raise ValueError("History must be exactly t-13 through t")
        if not (self.frame["target_day"] == self.frame["origin_day"] + 1).all():
            raise ValueError("Target must be t+1")

    def view(self, split: str, *, include_truth: bool) -> PreparedSplit:
        if split not in {"train", "validation", "test"}:
            raise ValueError(split)
        frame = self.frame.loc[self.frame["split"].eq(split)].reset_index(drop=True)
        if not include_truth:
            frame = frame.drop(columns=list(TARGET_COLUMNS))
        return PreparedSplit(frame, self.feature_columns, self.metadata, split, include_truth)


def load_prepared_bundle(prepared_csv: str | Path, metadata_json: str | Path) -> PreparedBundle:
    frame = pd.read_csv(prepared_csv)
    metadata = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    bundle = PreparedBundle(frame=frame, metadata=metadata)
    bundle.validate()
    return bundle


def validate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    expected_sample_ids: Iterable[str] | None = None,
    expected_split: str = "test",
) -> pd.DataFrame:
    """Validate and canonically order a submission; y_pred is log1p-space."""

    columns = set(predictions.columns)
    required = set(PREDICTION_COLUMNS)
    if columns != required:
        raise ValueError(
            f"Prediction columns must be exactly {list(PREDICTION_COLUMNS)}; "
            f"missing={sorted(required-columns)}, extra={sorted(columns-required)}"
        )
    out = predictions.loc[:, PREDICTION_COLUMNS].copy()
    if out.empty:
        raise ValueError("Prediction file is empty")
    if out[list(PREDICTION_COLUMNS)].isna().any().any():
        raise ValueError("Prediction file contains missing values")
    key = ["sample_id", "hormone", "horizon", "model_name", "model_version", "track", "split"]
    if out.duplicated(key).any():
        raise ValueError("Prediction file contains duplicate sample/hormone rows")
    if set(out["hormone"]) != set(HORMONES):
        raise ValueError(f"Predictions must contain exactly hormones {HORMONES}")
    if set(pd.to_numeric(out["horizon"], errors="coerce")) != {HORIZON}:
        raise ValueError("Only horizon=1 is valid for v0")
    if set(out["track"]) != {TRACK} or set(out["split"]) != {expected_split}:
        raise ValueError("Prediction track/split does not match the primary test contract")
    numeric = pd.to_numeric(out["y_pred"], errors="coerce")
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("y_pred must be finite, nonnegative log1p-space values")
    out["y_pred"] = numeric.astype(float)
    if out["model_name"].nunique() != 1 or out["model_version"].nunique() != 1:
        raise ValueError("Each prediction file must describe one model/version")
    if expected_sample_ids is not None:
        expected = {str(x) for x in expected_sample_ids}
        observed = set(out["sample_id"].astype(str))
        if observed != expected:
            raise ValueError(
                f"Prediction sample coverage mismatch: missing={len(expected-observed)}, "
                f"unexpected={len(observed-expected)}"
            )
        counts = out.groupby("sample_id")["hormone"].nunique()
        if len(counts) != len(expected) or not counts.eq(len(HORMONES)).all():
            raise ValueError("Every sample requires exactly one prediction per hormone")
    return out.sort_values(["sample_id", "hormone"]).reset_index(drop=True)

