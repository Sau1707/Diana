"""Wearable-only causal daily features for Hormonbench-mcPHASES v1."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


KEY = ["id", "day_in_study"]


def _aggregate_numeric(
    frame: pd.DataFrame,
    *,
    day_column: str,
    value_columns: list[str],
    prefix: str,
    add_record_count: bool = False,
) -> pd.DataFrame:
    use = frame[["id", day_column] + value_columns].drop_duplicates().copy()
    for column in value_columns:
        use[column] = pd.to_numeric(use[column], errors="coerce")
    grouped = use.groupby(["id", day_column], as_index=False)[value_columns].mean()
    grouped = grouped.rename(
        columns={
            day_column: "day_in_study",
            **{column: f"{prefix}__{column}" for column in value_columns},
        }
    )
    if add_record_count:
        count_name = f"{prefix}__record_count"
        counts = (
            use.groupby(["id", day_column])
            .size()
            .rename(count_name)
            .reset_index()
            .rename(columns={day_column: "day_in_study"})
        )
        grouped = grouped.merge(counts, on=KEY, how="left", validate="one_to_one")
    return grouped


def load_v1_daily_features(
    data_root: Path, target_rows: pd.DataFrame, interval: int
) -> tuple[pd.DataFrame, dict[str, Any], pd.Series]:
    """Load only approved Interval-2 wearable summaries plus weekend state."""

    target = target_rows.loc[target_rows["study_interval"].eq(interval)].copy()
    weekend = target[["id", "day_in_study", "is_weekend"]].drop_duplicates(KEY)
    if weekend.duplicated(KEY).any():
        raise ValueError("Weekend source has duplicate participant-day keys")
    weekend["weekend__is_weekend"] = (
        weekend["is_weekend"]
        .astype(str)
        .str.lower()
        .map({"true": 1.0, "false": 0.0})
    )
    daily = weekend.drop(columns="is_weekend")
    provenance: dict[str, Any] = {
        "weekend__is_weekend": {
            "source": "hormones_and_selfreport.csv:is_weekend",
            "availability": "known by end-of-origin day",
            "modality": "weekend_state",
            "uses_future": False,
        }
    }

    active_columns = ["sedentary", "lightly", "moderately", "very"]
    active = pd.read_csv(
        data_root / "active_minutes.csv",
        usecols=["id", "study_interval", "day_in_study"] + active_columns,
    )
    active = active.loc[active["study_interval"].eq(interval)]
    daily = daily.merge(
        _aggregate_numeric(
            active,
            day_column="day_in_study",
            value_columns=active_columns,
            prefix="active",
        ),
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    temperature_columns = [
        "temperature_samples",
        "nightly_temperature",
        "baseline_relative_sample_sum",
        "baseline_relative_sample_sum_of_squares",
        "baseline_relative_nightly_standard_deviation",
        "baseline_relative_sample_standard_deviation",
    ]
    temperature = pd.read_csv(
        data_root / "computed_temperature.csv",
        usecols=["id", "study_interval", "sleep_end_day_in_study"]
        + temperature_columns,
    )
    temperature = temperature.loc[temperature["study_interval"].eq(interval)]
    daily = daily.merge(
        _aggregate_numeric(
            temperature,
            day_column="sleep_end_day_in_study",
            value_columns=temperature_columns,
            prefix="temperature",
        ),
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    hrv_columns = ["rmssd", "coverage", "low_frequency", "high_frequency"]
    hrv = pd.read_csv(
        data_root / "heart_rate_variability_details.csv",
        usecols=["id", "study_interval", "day_in_study"] + hrv_columns,
    )
    hrv = hrv.loc[hrv["study_interval"].eq(interval)]
    daily = daily.merge(
        _aggregate_numeric(
            hrv,
            day_column="day_in_study",
            value_columns=hrv_columns,
            prefix="hrv",
            add_record_count=True,
        ),
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    respiratory_columns = [
        "full_sleep_breathing_rate",
        "full_sleep_standard_deviation",
        "full_sleep_signal_to_noise",
        "deep_sleep_breathing_rate",
        "deep_sleep_standard_deviation",
        "deep_sleep_signal_to_noise",
        "light_sleep_breathing_rate",
        "light_sleep_standard_deviation",
        "light_sleep_signal_to_noise",
        "rem_sleep_breathing_rate",
        "rem_sleep_standard_deviation",
        "rem_sleep_signal_to_noise",
    ]
    respiratory = pd.read_csv(
        data_root / "respiratory_rate_summary.csv",
        usecols=["id", "study_interval", "day_in_study"] + respiratory_columns,
    )
    respiratory = respiratory.loc[respiratory["study_interval"].eq(interval)].copy()
    for stage in ("full_sleep", "deep_sleep", "light_sleep", "rem_sleep"):
        rate = f"{stage}_breathing_rate"
        invalid = pd.to_numeric(respiratory[rate], errors="coerce").le(0)
        related = [column for column in respiratory_columns if column.startswith(stage)]
        respiratory.loc[invalid, related] = np.nan
    daily = daily.merge(
        _aggregate_numeric(
            respiratory,
            day_column="day_in_study",
            value_columns=respiratory_columns,
            prefix="respiratory",
        ),
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    sleep_columns = [
        "overall_score",
        "composition_score",
        "revitalization_score",
        "duration_score",
        "deep_sleep_in_minutes",
        "resting_heart_rate",
        "restlessness",
    ]
    sleep = pd.read_csv(
        data_root / "sleep_score.csv",
        usecols=["id", "study_interval", "day_in_study"] + sleep_columns,
    )
    sleep = sleep.loc[sleep["study_interval"].eq(interval)]
    daily = daily.merge(
        _aggregate_numeric(
            sleep,
            day_column="day_in_study",
            value_columns=sleep_columns,
            prefix="sleep_score",
        ),
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    source_by_prefix = {
        "active": ("active_minutes.csv", "active_minutes"),
        "temperature": (
            "computed_temperature.csv:sleep_end_day_in_study",
            "computed_temperature_end_day",
        ),
        "hrv": ("heart_rate_variability_details.csv:daily_mean", "hrv_daily_aggregate"),
        "respiratory": ("respiratory_rate_summary.csv:wake_day", "respiratory_rate_summary"),
        "sleep_score": ("sleep_score.csv:wake_day", "sleep_score"),
    }
    for column in daily.columns:
        if column in KEY or column in provenance:
            continue
        prefix = column.split("__", 1)[0]
        source, modality = source_by_prefix[prefix]
        provenance[column] = {
            "source": source,
            "modality": modality,
            "daily_aggregation": "mean after exact-row deduplication",
            "availability": "through end of aligned day",
            "uses_future": False,
        }

    modality_prefixes = {
        "active_minutes": "active__",
        "computed_temperature_end_day": "temperature__",
        "hrv_daily_aggregate": "hrv__",
        "respiratory_rate_summary": "respiratory__",
        "sleep_score": "sleep_score__",
    }
    coverage_frame = daily[KEY].copy()
    for modality, prefix in modality_prefixes.items():
        columns = [column for column in daily.columns if column.startswith(prefix)]
        coverage_frame[modality] = daily[columns].notna().any(axis=1).astype(float)
    participant_coverage = (
        coverage_frame.groupby("id")[list(modality_prefixes)]
        .mean()
        .mean(axis=1)
        .rename("approved_wearable_day_coverage")
    )
    return daily.sort_values(KEY).reset_index(drop=True), provenance, participant_coverage


def build_v1_history_features(
    daily: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    history_days: int,
    selected_lags: list[int],
    base_provenance: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if int(history_days) != 14:
        raise ValueError("v1 freezes history_days=14")
    if any(int(lag) < 0 or int(lag) >= history_days for lag in selected_lags):
        raise ValueError("Selected lags must remain inside t-13...t")
    signals = [column for column in daily.columns if column not in KEY]
    indexed = {
        int(pid): group.set_index("day_in_study")
        for pid, group in daily.groupby("id", sort=True)
    }
    offsets = np.arange(-history_days + 1, 1, dtype=float)
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for origin in origins.itertuples(index=False):
        participant = int(origin.private_participant_id)
        cutoff = int(origin.origin_day)
        window_days = list(range(cutoff - history_days + 1, cutoff + 1))
        window = indexed[participant].reindex(window_days)
        record: dict[str, Any] = {"sample_id": str(origin.sample_id)}
        for signal in signals:
            values = pd.to_numeric(window[signal], errors="coerce").to_numpy(float)
            observed = np.isfinite(values)
            count = int(observed.sum())
            prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", signal)
            last_index = int(np.flatnonzero(observed)[-1]) if count else None
            derived = {
                f"{prefix}__last": float(values[last_index])
                if last_index is not None
                else np.nan,
                f"{prefix}__mean": float(np.nanmean(values)) if count else np.nan,
                f"{prefix}__std": float(np.nanstd(values)) if count else np.nan,
                f"{prefix}__min": float(np.nanmin(values)) if count else np.nan,
                f"{prefix}__max": float(np.nanmax(values)) if count else np.nan,
                f"{prefix}__slope": float(
                    np.polyfit(offsets[observed], values[observed], 1)[0]
                )
                if count >= 2
                else np.nan,
                f"{prefix}__coverage": float(count / history_days),
                f"{prefix}__time_since": float(history_days - 1 - last_index)
                if last_index is not None
                else float(history_days + 1),
                f"{prefix}__missing_current": float(not observed[-1]),
            }
            for lag in selected_lags:
                position = history_days - 1 - int(lag)
                derived[f"{prefix}__lag{int(lag)}"] = (
                    float(values[position]) if np.isfinite(values[position]) else np.nan
                )
            record.update(derived)
            for name in derived:
                provenance[name] = {
                    **dict(base_provenance[signal]),
                    "history_window": "t-13 through t",
                    "derivation": name.rsplit("__", 1)[-1],
                    "uses_future": False,
                    "learned": False,
                }
        rows.append(record)
    return pd.DataFrame(rows), provenance
