"""Versioned prepared, model-view, calibration, and prediction contracts for v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .v1_task import (
    HORMONES,
    HISTORY_DAYS,
    TASK_ID,
    TASK_VERSION,
    TRACK_COLD,
    TRACK_FEW_SHOT,
)


TARGET_RAW_COLUMNS = {h: f"target_{h}_raw" for h in HORMONES}
TARGET_LOG_COLUMNS = {h: f"target_{h}_log1p" for h in HORMONES}
TARGET_COLUMNS = tuple(TARGET_RAW_COLUMNS.values()) + tuple(TARGET_LOG_COLUMNS.values())
PRIVATE_ALIGNMENT_COLUMNS = (
    "task_id",
    "task_version",
    "sample_id",
    "private_participant_id",
    "origin_day",
    "target_day",
    "history_start_day",
    "history_end_day",
    "cutoff_day",
    "fold_group",
    "config_hash",
    "task_spec_hash",
    "input_schema_hash",
    "fold_hash",
)

PREDICTION_REQUIRED_COLUMNS = (
    "task_id",
    "task_version",
    "track",
    "fold",
    "calibration_budget",
    "split",
    "sample_id",
    "hormone",
    "horizon",
    "y_pred",
    "model_name",
    "model_version",
)
PREDICTION_INTERVAL_COLUMNS = ("y_lower", "y_upper")

FORBIDDEN_EXACT_FEATURES = {
    "id",
    "participant_id",
    "private_participant_id",
    "sample_id",
    "origin_day",
    "target_day",
    "cutoff_day",
    "day_in_study",
    "study_interval",
    "calendar_date",
    "date",
    "days_since_last_known_menses",
    "menses_onset_missing",
    "phase",
    "mira_phase",
    "fertile_window",
    "lh",
    "e3g",
    "estrogen",
    "pdg",
    "cycle_length",
    "cycle_percentage",
    "normalized_cycle_percentage",
}
FORBIDDEN_FEATURE_FRAGMENTS = (
    "self_report",
    "flow_volume",
    "flow_color",
    "future_",
    "backfill",
    "backward_fill",
    "centered_",
    "interpolat",
    "hormone_history",
    "mira_",
    "fertile",
    "target_lh",
    "target_e3g",
    "target_pdg",
    "interval_1",
    "absolute_time",
    "modulo_28",
)


def assert_v1_feature_names_safe(feature_columns: Iterable[str]) -> None:
    bad: list[str] = []
    for column in feature_columns:
        lower = str(column).strip().lower()
        tokens = set(lower.replace("__", "_").split("_"))
        if lower in FORBIDDEN_EXACT_FEATURES:
            bad.append(str(column))
            continue
        if tokens & {"lh", "e3g", "estrogen", "pdg"}:
            bad.append(str(column))
            continue
        if any(fragment in lower for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            bad.append(str(column))
    if bad:
        raise ValueError(f"Forbidden v1 feature columns: {sorted(set(bad))}")


@dataclass(frozen=True)
class V1FitView:
    X: pd.DataFrame
    targets: pd.DataFrame
    participant_groups: pd.Series
    sample_ids: pd.Series

    def validate(self) -> None:
        n = len(self.X)
        if not (
            len(self.targets) == len(self.participant_groups) == len(self.sample_ids) == n
        ):
            raise ValueError("V1FitView components must have identical lengths")
        if tuple(self.targets.columns) != HORMONES:
            raise ValueError("V1FitView targets must be LH/E3G/PdG log1p columns")
        assert_v1_feature_names_safe(self.X.columns)
        if not np.isfinite(self.targets.to_numpy(float)).all():
            raise ValueError("Fit targets must be finite")


@dataclass(frozen=True)
class V1InferenceView:
    X: pd.DataFrame
    participant_groups: pd.Series
    sample_ids: pd.Series

    def validate(self) -> None:
        if not len(self.X) == len(self.participant_groups) == len(self.sample_ids):
            raise ValueError("V1InferenceView components must have identical lengths")
        assert_v1_feature_names_safe(self.X.columns)


@dataclass(frozen=True)
class V1CalibrationView:
    sample_ids: pd.Series
    participant_groups: pd.Series
    target_days: pd.Series
    targets: pd.DataFrame
    budget: int

    def validate(self) -> None:
        n = len(self.sample_ids)
        if not (
            len(self.participant_groups) == len(self.target_days) == len(self.targets) == n
        ):
            raise ValueError("Calibration components must have identical lengths")
        if tuple(self.targets.columns) != HORMONES:
            raise ValueError("Calibration targets must be three-hormone log1p vectors")
        counts = self.participant_groups.value_counts()
        if self.budget == 0:
            if n != 0:
                raise ValueError("K=0 calibration must contain no truth")
        elif counts.empty or not counts.eq(int(self.budget)).all():
            raise ValueError(f"Every calibration participant requires exactly K={self.budget}")


@dataclass(frozen=True)
class V1PreparedBundle:
    frame: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(self.metadata["feature_columns"])

    def validate(self) -> None:
        required = set(PRIVATE_ALIGNMENT_COLUMNS) | set(TARGET_COLUMNS)
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(f"v1 prepared bundle missing columns: {missing}")
        if self.frame["sample_id"].duplicated().any():
            raise ValueError("v1 sample IDs must be unique")
        if set(self.frame["task_id"]) != {TASK_ID} or set(
            self.frame["task_version"].astype(str)
        ) != {TASK_VERSION}:
            raise ValueError("Prepared task identity mismatch")
        features = self.feature_columns
        assert_v1_feature_names_safe(features)
        if set(features) & (set(PRIVATE_ALIGNMENT_COLUMNS) | set(TARGET_COLUMNS)):
            raise ValueError("Private alignment or target columns cannot be features")
        absent = sorted(set(features) - set(self.frame.columns))
        if absent:
            raise ValueError(f"Prepared metadata references absent features: {absent}")
        if not (self.frame["history_start_day"] == self.frame["origin_day"] - 13).all():
            raise ValueError("v1 history must begin at t-13")
        if not (self.frame["history_end_day"] == self.frame["origin_day"]).all():
            raise ValueError("v1 history must end at t")
        if not (self.frame["target_day"] == self.frame["origin_day"] + 1).all():
            raise ValueError("v1 target must be t+1")
        for hormone in HORMONES:
            raw = pd.to_numeric(self.frame[TARGET_RAW_COLUMNS[hormone]], errors="coerce")
            logged = pd.to_numeric(
                self.frame[TARGET_LOG_COLUMNS[hormone]], errors="coerce"
            )
            if raw.isna().any() or logged.isna().any():
                raise ValueError("v1 labels must be genuinely observed")
            if not np.allclose(np.log1p(raw.to_numpy(float)), logged.to_numpy(float)):
                raise ValueError(f"v1 log1p target mismatch for {hormone}")
        expected_rows = int(self.metadata.get("eligible_origins", len(self.frame)))
        if len(self.frame) != expected_rows:
            raise ValueError("Prepared row count does not match metadata")
        if self.frame["task_spec_hash"].nunique() != 1:
            raise ValueError("Prepared rows must share one task-spec hash")
        if self.frame["input_schema_hash"].nunique() != 1:
            raise ValueError("Prepared rows must share one input-schema hash")

    def fit_view(self, mask: pd.Series | np.ndarray) -> V1FitView:
        rows = self.frame.loc[np.asarray(mask, dtype=bool)].reset_index(drop=True)
        view = V1FitView(
            X=rows.loc[:, self.feature_columns].copy(),
            targets=pd.DataFrame(
                {
                    hormone: rows[TARGET_LOG_COLUMNS[hormone]].astype(float)
                    for hormone in HORMONES
                }
            ),
            participant_groups=rows["private_participant_id"].astype(str).copy(),
            sample_ids=rows["sample_id"].astype(str).copy(),
        )
        view.validate()
        return view

    def inference_view(self, mask: pd.Series | np.ndarray) -> V1InferenceView:
        rows = self.frame.loc[np.asarray(mask, dtype=bool)].reset_index(drop=True)
        view = V1InferenceView(
            X=rows.loc[:, self.feature_columns].copy(),
            participant_groups=rows["private_participant_id"].astype(str).copy(),
            sample_ids=rows["sample_id"].astype(str).copy(),
        )
        view.validate()
        return view


def save_v1_bundle(bundle: V1PreparedBundle, directory: str | Path) -> None:
    bundle.validate()
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    bundle.frame.to_csv(path / "prepared.csv", index=False)
    (path / "metadata.json").write_text(
        json.dumps(bundle.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_v1_bundle(directory: str | Path) -> V1PreparedBundle:
    path = Path(directory)
    frame = pd.read_csv(path / "prepared.csv")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    bundle = V1PreparedBundle(frame=frame, metadata=metadata)
    bundle.validate()
    return bundle


def validate_v1_prediction_frame(
    predictions: pd.DataFrame,
    *,
    expected_sample_ids: Iterable[str],
    expected_track: str,
    expected_fold: int,
    expected_budget: int,
) -> pd.DataFrame:
    columns = tuple(predictions.columns)
    required = set(PREDICTION_REQUIRED_COLUMNS)
    optional = set(PREDICTION_INTERVAL_COLUMNS)
    observed = set(columns)
    if not required <= observed or observed - required - optional:
        raise ValueError(
            f"Invalid v1 prediction columns; missing={sorted(required-observed)}, "
            f"extra={sorted(observed-required-optional)}"
        )
    if bool(optional & observed) and not optional <= observed:
        raise ValueError("Prediction intervals require both y_lower and y_upper")
    out = predictions.copy()
    if out.empty or out[list(required)].isna().any().any():
        raise ValueError("v1 predictions cannot be empty or missing required values")
    if set(out["task_id"]) != {TASK_ID} or set(out["task_version"].astype(str)) != {
        TASK_VERSION
    }:
        raise ValueError("Prediction task identity mismatch")
    if expected_track not in {TRACK_COLD, TRACK_FEW_SHOT} or set(out["track"]) != {
        expected_track
    }:
        raise ValueError("Prediction track mismatch")
    if set(pd.to_numeric(out["fold"], errors="coerce")) != {int(expected_fold)}:
        raise ValueError("Prediction fold mismatch")
    if set(pd.to_numeric(out["calibration_budget"], errors="coerce")) != {
        int(expected_budget)
    }:
        raise ValueError("Prediction calibration budget mismatch")
    if set(out["split"]) != {"test"}:
        raise ValueError("Official predictions must use split=test")
    if set(out["hormone"]) != set(HORMONES):
        raise ValueError("Predictions require exactly LH/E3G/PdG")
    if set(pd.to_numeric(out["horizon"], errors="coerce")) != {1}:
        raise ValueError("v1 supports only horizon=1")
    numeric = pd.to_numeric(out["y_pred"], errors="coerce")
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Point predictions must be finite nonnegative log1p values")
    out["y_pred"] = numeric.astype(float)
    if optional <= observed:
        lower = pd.to_numeric(out["y_lower"], errors="coerce")
        upper = pd.to_numeric(out["y_upper"], errors="coerce")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("Prediction intervals must be finite")
        if (lower < 0).any() or (lower > numeric).any() or (numeric > upper).any():
            raise ValueError("Prediction intervals must satisfy 0 <= lower <= point <= upper")
        out["y_lower"] = lower.astype(float)
        out["y_upper"] = upper.astype(float)
    key = ["sample_id", "hormone"]
    if out.duplicated(key).any():
        raise ValueError("Duplicate v1 sample/hormone prediction")
    expected = {str(value) for value in expected_sample_ids}
    actual = set(out["sample_id"].astype(str))
    if actual != expected:
        raise ValueError(
            f"Prediction sample coverage mismatch: missing={len(expected-actual)}, "
            f"unexpected={len(actual-expected)}"
        )
    counts = out.groupby(out["sample_id"].astype(str))["hormone"].nunique()
    if len(counts) != len(expected) or not counts.eq(len(HORMONES)).all():
        raise ValueError("Every sample requires one prediction per hormone")
    if out["model_name"].nunique() != 1 or out["model_version"].nunique() != 1:
        raise ValueError("A prediction file must contain one model/version")
    return out.sort_values(["sample_id", "hormone"]).reset_index(drop=True)
