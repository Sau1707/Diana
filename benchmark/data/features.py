"""Causal daily aggregation and 14-day history feature construction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEY = ["id", "day_in_study"]
SELF_REPORT_COLUMNS = (
    "flow_volume",
    "appetite",
    "exerciselevel",
    "headaches",
    "cramps",
    "sorebreasts",
    "fatigue",
    "sleepissue",
    "moodswing",
    "stress",
    "foodcravings",
    "indigestion",
    "bloating",
)

LIKERT = {
    "not at all": 0.0,
    "very low": 1.0,
    "very low/little": 1.0,
    "low": 2.0,
    "moderate": 3.0,
    "high": 4.0,
    "very high": 5.0,
}
FLOW_VOLUME = {
    "not at all": 0.0,
    "spotting / very light": 1.0,
    "light": 2.0,
    "somewhat light": 3.0,
    "moderate": 4.0,
    "somewhat heavy": 5.0,
    "heavy": 6.0,
    "very heavy": 7.0,
}


def _ordinal(value: Any, *, flow: bool = False) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    mapping = FLOW_VOLUME if flow else LIKERT
    if text in mapping:
        return mapping[text]
    try:
        numeric = float(text)
        return numeric if 0 <= numeric <= 7 else np.nan
    except ValueError:
        return np.nan


def derive_known_menses_onsets(all_target_rows: pd.DataFrame) -> dict[int, list[int]]:
    """Causally identify positive-flow transitions from all past known reports."""

    onsets: dict[int, list[int]] = {}
    for pid, group in all_target_rows.sort_values(["id", "day_in_study"]).groupby("id"):
        previous_known_positive: bool | None = None
        participant_onsets: list[int] = []
        for row in group.itertuples(index=False):
            encoded = _ordinal(getattr(row, "flow_volume"), flow=True)
            if np.isnan(encoded):
                continue
            positive = encoded > 0
            # A first observed positive report may already be mid-bleed.  Count an
            # onset only when a known nonpositive report precedes the transition.
            if positive and previous_known_positive is False:
                participant_onsets.append(int(row.day_in_study))
            previous_known_positive = positive
        onsets[int(pid)] = participant_onsets
    return onsets


def days_since_last_known_menses(onsets: dict[int, list[int]], pid: int, cutoff_day: int) -> float:
    eligible = [day for day in onsets.get(int(pid), []) if day <= int(cutoff_day)]
    return float(int(cutoff_day) - max(eligible)) if eligible else np.nan


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
    grouped = grouped.rename(columns={day_column: "day_in_study", **{c: f"{prefix}__{c}" for c in value_columns}})
    if add_record_count:
        counts = use.groupby(["id", day_column]).size().rename(f"{prefix}__record_count").reset_index().rename(columns={day_column: "day_in_study"})
        grouped = grouped.merge(counts, on=KEY, how="left", validate="one_to_one")
    return grouped


def load_daily_features(data_root: Path, all_targets: pd.DataFrame, interval: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = all_targets.loc[all_targets["study_interval"].eq(interval)].copy()
    daily = target[["id", "day_in_study", "is_weekend"]].drop_duplicates(KEY).copy()
    daily["origin_is_weekend"] = target["is_weekend"].astype(str).str.lower().map({"true": 1.0, "false": 0.0}).to_numpy()
    daily = daily.drop(columns="is_weekend")
    provenance: dict[str, Any] = {
        "origin_is_weekend": {"source": "hormones_and_selfreport.csv", "column": "is_weekend", "availability": "known by origin cutoff", "learned": False}
    }

    active_cols = ["sedentary", "lightly", "moderately", "very"]
    active = pd.read_csv(data_root / "active_minutes.csv", usecols=["id", "study_interval", "day_in_study"] + active_cols)
    active = active.loc[active["study_interval"].eq(interval)]
    table = _aggregate_numeric(active, day_column="day_in_study", value_columns=active_cols, prefix="active")
    daily = daily.merge(table, on=KEY, how="left", validate="one_to_one")

    temp_cols = [
        "temperature_samples",
        "nightly_temperature",
        "baseline_relative_sample_sum",
        "baseline_relative_sample_sum_of_squares",
        "baseline_relative_nightly_standard_deviation",
        "baseline_relative_sample_standard_deviation",
    ]
    temp = pd.read_csv(data_root / "computed_temperature.csv", usecols=["id", "study_interval", "sleep_end_day_in_study"] + temp_cols)
    temp = temp.loc[temp["study_interval"].eq(interval)]
    table = _aggregate_numeric(temp, day_column="sleep_end_day_in_study", value_columns=temp_cols, prefix="temperature")
    daily = daily.merge(table, on=KEY, how="left", validate="one_to_one")

    hrv_cols = ["rmssd", "coverage", "low_frequency", "high_frequency"]
    hrv = pd.read_csv(data_root / "heart_rate_variability_details.csv", usecols=["id", "study_interval", "day_in_study"] + hrv_cols)
    hrv = hrv.loc[hrv["study_interval"].eq(interval)]
    table = _aggregate_numeric(hrv, day_column="day_in_study", value_columns=hrv_cols, prefix="hrv", add_record_count=True)
    daily = daily.merge(table, on=KEY, how="left", validate="one_to_one")

    resp_cols = [
        "full_sleep_breathing_rate", "full_sleep_standard_deviation", "full_sleep_signal_to_noise",
        "deep_sleep_breathing_rate", "deep_sleep_standard_deviation", "deep_sleep_signal_to_noise",
        "light_sleep_breathing_rate", "light_sleep_standard_deviation", "light_sleep_signal_to_noise",
        "rem_sleep_breathing_rate", "rem_sleep_standard_deviation", "rem_sleep_signal_to_noise",
    ]
    resp = pd.read_csv(data_root / "respiratory_rate_summary.csv", usecols=["id", "study_interval", "day_in_study"] + resp_cols)
    resp = resp.loc[resp["study_interval"].eq(interval)].copy()
    for stage in ("full_sleep", "deep_sleep", "light_sleep", "rem_sleep"):
        rate = f"{stage}_breathing_rate"
        invalid = pd.to_numeric(resp[rate], errors="coerce").le(0)
        related = [c for c in resp_cols if c.startswith(stage)]
        resp.loc[invalid, related] = np.nan
    table = _aggregate_numeric(resp, day_column="day_in_study", value_columns=resp_cols, prefix="respiratory")
    daily = daily.merge(table, on=KEY, how="left", validate="one_to_one")

    sleep_cols = ["overall_score", "composition_score", "revitalization_score", "duration_score", "deep_sleep_in_minutes", "resting_heart_rate", "restlessness"]
    sleep = pd.read_csv(data_root / "sleep_score.csv", usecols=["id", "study_interval", "day_in_study"] + sleep_cols)
    sleep = sleep.loc[sleep["study_interval"].eq(interval)]
    table = _aggregate_numeric(sleep, day_column="day_in_study", value_columns=sleep_cols, prefix="sleep_score")
    daily = daily.merge(table, on=KEY, how="left", validate="one_to_one")

    self_daily = target[["id", "day_in_study"] + list(SELF_REPORT_COLUMNS) + ["flow_color"]].copy()
    for column in SELF_REPORT_COLUMNS:
        self_daily[f"self_report__{column}"] = self_daily[column].map(lambda x, c=column: _ordinal(x, flow=(c == "flow_volume")))
    self_daily["self_report__flow_color_observed"] = self_daily["flow_color"].notna().astype(float)
    self_daily = self_daily[["id", "day_in_study"] + [c for c in self_daily if c.startswith("self_report__")]]
    daily = daily.merge(self_daily, on=KEY, how="left", validate="one_to_one")

    for column in daily.columns:
        if column in KEY or column in provenance:
            continue
        source_prefix = column.split("__", 1)[0]
        source = {
            "active": "active_minutes.csv",
            "temperature": "computed_temperature.csv (sleep_end_day_in_study)",
            "hrv": "heart_rate_variability_details.csv daily aggregate",
            "respiratory": "respiratory_rate_summary.csv wake day",
            "sleep_score": "sleep_score.csv wake day",
            "self_report": "hormones_and_selfreport.csv participant-entered field",
        }.get(source_prefix, "derived")
        provenance[column] = {"source": source, "daily_aggregation": "mean after exact deduplication", "availability": "through end of recorded day", "learned": False}
    return daily.sort_values(KEY).reset_index(drop=True), provenance


def build_history_features(
    daily: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    all_targets: pd.DataFrame,
    history_days: int,
    selected_lags: list[int],
    base_provenance: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if history_days != 14:
        raise ValueError("Hormonbench v0 freezes a 14-day history")
    signals = [c for c in daily.columns if c not in KEY]
    indexed = {int(pid): group.set_index("day_in_study") for pid, group in daily.groupby("id")}
    menses = derive_known_menses_onsets(all_targets)
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    offsets = np.arange(-history_days + 1, 1, dtype=float)
    for origin in origins.itertuples(index=False):
        pid = int(origin.private_participant_id)
        cutoff = int(origin.origin_day)
        window_days = list(range(cutoff - history_days + 1, cutoff + 1))
        window = indexed[pid].reindex(window_days)
        record: dict[str, Any] = {
            "sample_id": origin.sample_id,
            "days_since_last_known_menses": days_since_last_known_menses(menses, pid, cutoff),
        }
        record["menses_onset_missing"] = float(np.isnan(record["days_since_last_known_menses"]))
        for signal in signals:
            values = pd.to_numeric(window[signal], errors="coerce").to_numpy(float)
            observed = np.isfinite(values)
            count = int(observed.sum())
            prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", signal)
            latest_index = int(np.flatnonzero(observed)[-1]) if count else None
            derived = {
                f"{prefix}__last": float(values[latest_index]) if latest_index is not None else np.nan,
                f"{prefix}__mean": float(np.nanmean(values)) if count else np.nan,
                f"{prefix}__std": float(np.nanstd(values)) if count else np.nan,
                f"{prefix}__min": float(np.nanmin(values)) if count else np.nan,
                f"{prefix}__max": float(np.nanmax(values)) if count else np.nan,
                f"{prefix}__slope": float(np.polyfit(offsets[observed], values[observed], 1)[0]) if count >= 2 else np.nan,
                f"{prefix}__coverage": float(count / history_days),
                f"{prefix}__time_since": float(history_days - 1 - latest_index) if latest_index is not None else float(history_days + 1),
                f"{prefix}__missing_current": float(not observed[-1]),
            }
            for lag in selected_lags:
                pos = history_days - 1 - int(lag)
                derived[f"{prefix}__lag{lag}"] = float(values[pos]) if np.isfinite(values[pos]) else np.nan
            record.update(derived)
            for name in derived:
                stat = name.rsplit("__", 1)[-1]
                provenance[name] = {**base_provenance.get(signal, {"source": signal}), "history_window": "t-13 through t", "derivation": stat, "uses_future": False, "learned": False}
        rows.append(record)
    provenance["days_since_last_known_menses"] = {"source": "all past known flow_volume reports", "derivation": "days since latest causal positive-flow onset", "history_window": "unbounded past through t", "uses_future": False, "learned": False}
    provenance["menses_onset_missing"] = {"source": "days_since_last_known_menses", "derivation": "explicit missing indicator", "uses_future": False, "learned": False}
    return pd.DataFrame(rows), provenance
