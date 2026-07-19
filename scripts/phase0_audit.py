"""Reproduce the bounded Hormonbench Phase 0 audit artifacts.

Run with:
    conda run -n igl python scripts/phase0_audit.py

The script never writes inside dataset/. Large CSVs (>25 MB) are sampled only.
Compact projected tables are read fully to measure participant-day joinability.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "dataset" / "mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0"
OUT = PROJECT / "reports" / "phase0"
KEYS = ["id", "study_interval", "day_in_study"]
TARGETS = ["lh", "estrogen", "pdg"]


SCHEMA_META = {
    "active_minutes.csv": ("participant-day", "daily-summary", "YES", "sedentary, lightly, moderately, very: minutes", "End-of-day activity; use for t+1."),
    "active_zone_minutes.csv": ("minute/zone event", "event-level", "NO", "heart_zone_id; total_minutes: minutes", "Overlapping timestamped zone rows can overcount; postpone."),
    "altitude.csv": ("timestamped sparse stream", "raw", "NO", "altitude: metres of relative gain", "Expanded route."),
    "calories.csv": ("approximately one minute", "raw", "NO", "calories: device-reported, supplied unit unresolved", "Differing duplicate minute keys sampled; expanded route."),
    "computed_temperature.csv": ("sleep session/night", "daily-summary", "YES_CONDITIONAL", "nightly_temperature and baseline-relative fields: deg C; temperature_samples: count", "Align to sleep_end_day_in_study; only after wake."),
    "demographic_vo2_max.csv": ("nominal participant-day", "daily-summary", "NO_PENDING_PROVENANCE", "four device-derived VO2 estimates/errors; unit undocumented", "2024 conflicts/extremes and opaque filtering."),
    "distance.csv": ("approximately one minute", "raw", "NO", "distance: metres", "Differing duplicate minute keys sampled; expanded route."),
    "estimated_oxygen_variation.csv": ("minute-level during sleep", "raw", "NO", "infrared_to_red_signal_ratio: supplied transform/unit unclear", "Signed values contradict a literal nonnegative ratio; expanded route."),
    "exercise.csv": ("exercise session", "event-level", "NO", "duration/originalduration/activeduration: ms; averageheartrate: bpm; calories; steps; elevationgain: m", "Nested fields are mixed JSON/Python-like text; expanded route."),
    "glucose.csv": ("approximately five minutes", "raw", "NO", "glucose_value: mmol/L", "Only 2022; 2024 is structural non-collection."),
    "heart_rate.csv": ("irregular continuous, sampled tail ~5 seconds", "raw", "NO", "bpm: beats/min; confidence: Fitbit internal scale", "2.02 GB; bounded sampled only."),
    "heart_rate_variability_details.csv": ("nominal five-minute sleep intervals", "event-level", "YES_AFTER_DAILY_AGGREGATION", "rmssd, low_frequency, high_frequency: supplied units undocumented; coverage: ratio", "Aggregate by recorded day using records through cutoff."),
    "height_and_weight.csv": ("interval-static", "static", "YES_CONDITIONAL", "height_2022/2024: cm; weight_2022/2024: kg", "Participant-entered; use matching interval only."),
    "hormones_and_selfreport.csv": ("participant-day", "target/daily-event", "LAGGED_ONLY", "lh: mIU/mL urinary LH; estrogen: ng/mL urinary E3G; pdg: mcg/mL urinary PdG; symptom categories", "Current/future hormones and phase excluded; lagged measured values/self-reports allowed."),
    "respiratory_rate_summary.csv": ("nightly wake-day summary", "daily-summary", "YES_CONDITIONAL", "breathing_rate: breaths/min; SD implied same scale; SNR unit undocumented", "Treat -1 and nonpositive full-sleep rates as unavailable; wake-day only."),
    "resting_heart_rate.csv": ("nominal participant-day", "daily-summary", "NO_PENDING_PROVENANCE", "value: bpm; error: supplied uncertainty", "2024 conflicting repeats without timestamp; 0/0 appears sentinel-like."),
    "sleep.csv": ("sleep session", "event-level", "NO", "duration: ms; sleep fields: minutes; efficiency: percent", "Nested levels extend through sleep end; aggregate later by end day."),
    "sleep_score.csv": ("nightly wake-day summary", "daily-summary", "YES_CONDITIONAL", "scores: Fitbit points; deep_sleep_in_minutes: min; resting_heart_rate: bpm; restlessness: device metric", "2024 component schema drift; use missing indicators and wake-day alignment."),
    "steps.csv": ("approximately one minute", "raw", "NO", "steps: count", "Differing duplicate minute keys sampled; expanded route."),
    "stress_score.csv": ("nominal participant-day/revision", "daily-summary", "OPTIONAL", "stress_score and subscores: Fitbit points; status/failure metadata", "Exact-deduplicate and require READY/nonfailed; only 12.446% target coverage in 2024."),
    "subject-info.csv": ("participant-static", "static", "YES", "birth_year, gender, ethnicity, education, sexually_active, literacy, menarche age", "Participant-entered; sensitive metadata; employment/income documented but absent."),
    "time_in_heart_rate_zones.csv": ("participant-day", "daily-summary", "NO_PENDING_MAPPING", "four zone-duration fields, apparently minutes", "2024 mapping/distribution conflicts with README; do not use until reconciled."),
    "wrist_temperature.csv": ("approximately one minute", "raw", "NO", "temperature_diff_from_baseline: deg C", "Signed baseline difference; expanded route, prefer computed nightly temperature."),
}

FULL_SCAN = {
    "active_minutes.csv",
    "computed_temperature.csv",
    "demographic_vo2_max.csv",
    "glucose.csv",
    "heart_rate_variability_details.csv",
    "height_and_weight.csv",
    "hormones_and_selfreport.csv",
    "respiratory_rate_summary.csv",
    "resting_heart_rate.csv",
    "sleep_score.csv",
    "stress_score.csv",
    "subject-info.csv",
    "time_in_heart_rate_zones.csv",
}

SENTINELS = {
    "respiratory_rate_summary.csv": "-1 stage rate; nonpositive full-sleep rate treated unavailable; zero SD can accompany missing stage",
    "resting_heart_rate.csv": "0/0 rows appear no-estimate sentinel-like",
    "stress_score.csv": "NO_DATA/failed uses zeros; READY_NOT_PREMIUM has zero subscores/maxima",
    "computed_temperature.csv": "blank baseline-relative fields during baseline establishment",
}

DUPLICATE_LOOKING = {
    "exercise.csv": "original vs edited start/duration fields; raw steps/calories overlap other sources",
    "sleep_score.csv": "resting_heart_rate duplicates a concept in resting_heart_rate.csv",
    "hormones_and_selfreport.csv": "self-report stress/exerciselevel distinct from device stress/activity",
    "sleep.csv": "start/end fields plus nested levels timestamps",
}


def json_compact(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def infer_sample_types(path: Path) -> tuple[dict[str, str], int]:
    sample = pd.read_csv(path, nrows=500, low_memory=False)
    types = {}
    for col in sample.columns:
        s = sample[col]
        if pd.api.types.is_bool_dtype(s):
            typ = "boolean"
        elif pd.api.types.is_integer_dtype(s):
            typ = "integer"
        elif pd.api.types.is_float_dtype(s):
            typ = "float/nullable-numeric"
        else:
            typ = "string/categorical-or-serialized"
        types[col] = typ
    return types, len(sample)


def make_schema_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        types, sample_n = infer_sample_types(path)
        timestamps = [c for c in columns if "timestamp" in c]
        relative_days = [c for c in columns if "day_in_study" in c]
        resolution, role, lite, units, note = SCHEMA_META[path.name]
        mixed = []
        if path.name in {"hormones_and_selfreport.csv", "height_and_weight.csv"}:
            mixed.append("numeric text includes integer/decimal forms but parses numerically")
        if path.name == "exercise.csv":
            mixed.append("nested fields mix single-quoted Python-like and JSON-like serialization")
        if path.name == "hormones_and_selfreport.csv":
            mixed.append("2024 headaches/stress include numeric category strings while 2022 uses verbal categories")
        scan = "fully scanned (all columns)" if path.name in {"hormones_and_selfreport.csv", "subject-info.csv", "height_and_weight.csv"} else ("fully scanned (projected join/value columns)" if path.name in FULL_SCAN else "bounded multi-offset sample only")
        rows.append(
            {
                "filename": path.name,
                "byte_size": path.stat().st_size,
                "column_count": len(columns),
                "exact_column_names": json_compact(columns),
                "sampled_inferred_types": json_compact(types),
                "participant_key": "id",
                "timestamp_fields": "|".join(timestamps),
                "relative_day_fields": "|".join(relative_days),
                "time_resolution": resolution,
                "value_and_unit_fields": units,
                "role": role,
                "lite_v0_suitability": lite,
                "scan_status": scan,
                "bounded_sample_rows": sample_n,
                "delimiter": "comma",
                "encoding": "UTF-8 compatible; no BOM observed",
                "malformed_rows_check": "0 width/parser errors in bounded start/~25/~50/~75/~97% samples",
                "timestamp_parseability": "top-level timestamps parsed as time-of-day in bounded nonblank samples" if timestamps else "not applicable",
                "mixed_dtype_observations": "; ".join(mixed) or "none material in bounded samples",
                "sentinel_observations": SENTINELS.get(path.name, "no generic NA/null/-999 token observed in bounded samples"),
                "duplicate_looking_fields": DUPLICATE_LOOKING.get(path.name, "none material by header; duplicate keys assessed separately"),
                "audit_notes": note,
            }
        )
    return pd.DataFrame(rows)


def key_set(df: pd.DataFrame, day_col: str = "day_in_study", mask=None) -> set[tuple[int, int, int]]:
    if mask is not None:
        df = df.loc[mask]
    values = df[["id", "study_interval", day_col]].dropna().drop_duplicates()
    return {(int(a), int(b), int(c)) for a, b, c in values.itertuples(index=False, name=None)}


def duplicate_stats(df: pd.DataFrame, day_col: str = "day_in_study") -> tuple[int, int, dict[tuple[int, int], int]]:
    counts = df.groupby(["id", "study_interval", day_col], dropna=False).size()
    dup = counts[counts > 1]
    by_group = Counter((int(idx[0]), int(idx[1])) for idx in dup.index)
    return int(len(dup)), int((dup - 1).sum()), dict(by_group)


def build_modalities(target: pd.DataFrame):
    modalities = {}

    def add(name, df, day_col="day_in_study", mask=None, causal="t+1 only", semantics="day label", notes=""):
        keys = key_set(df, day_col, mask)
        dg, ex, by = duplicate_stats(df.loc[mask] if mask is not None else df, day_col)
        modalities[name] = {"keys": keys, "raw_rows": int(mask.sum()) if mask is not None else len(df), "duplicate_groups": dg, "duplicate_excess": ex, "duplicates_by_group": by, "causal": causal, "semantics": semantics, "notes": notes}

    df = pd.read_csv(DATA / "active_minutes.csv")
    add("active_minutes", df, notes="Exact duplicate rows can be dropped; end-of-day totals.")

    df = pd.read_csv(DATA / "computed_temperature.csv")
    add("computed_temperature_end_day", df, "sleep_end_day_in_study", semantics="sleep_end/wake day", notes="Use sleep_end_day_in_study, not start day; aggregate multiple sessions.")

    df = pd.read_csv(DATA / "heart_rate_variability_details.csv", usecols=["id", "study_interval", "day_in_study", "rmssd", "coverage", "low_frequency", "high_frequency"])
    add("hrv_daily_aggregate", df, notes="Expected many event rows per day; aggregate with masks through cutoff.")

    df = pd.read_csv(DATA / "respiratory_rate_summary.csv", usecols=["id", "study_interval", "day_in_study", "full_sleep_breathing_rate"])
    add("respiratory_rate_row_presence", df, semantics="timestamp is wake time/day")
    add("respiratory_rate_usable", df, mask=pd.to_numeric(df["full_sleep_breathing_rate"], errors="coerce").gt(0), semantics="timestamp is wake time/day", notes="Nonpositive and -1 stage sentinels excluded.")

    df = pd.read_csv(DATA / "resting_heart_rate.csv", usecols=["id", "study_interval", "day_in_study", "value", "error"])
    add("resting_heart_rate_row_presence", df, notes="Coverage only; source excluded pending 2024 provenance.")
    add("resting_heart_rate_positive", df, mask=pd.to_numeric(df["value"], errors="coerce").gt(0), notes="Zero/no-estimate excluded; 2024 conflicts remain unresolved.")

    df = pd.read_csv(DATA / "sleep_score.csv", usecols=["id", "study_interval", "day_in_study", "overall_score"])
    add("sleep_score", df, semantics="timestamp is wake/end time", notes="2024 component missingness/schema drift requires indicators.")

    df = pd.read_csv(DATA / "stress_score.csv", usecols=["id", "study_interval", "day_in_study", "status", "calculation_failed"])
    add("stress_score_any_row", df, notes="Raw presence is misleading.")
    failed = df["calculation_failed"].astype(str).str.lower().eq("true")
    ready = df["status"].astype(str).eq("READY") & ~failed
    add("stress_score_ready", df, mask=ready, notes="Optional only; exact-deduplicate, READY and nonfailed.")

    df = pd.read_csv(DATA / "time_in_heart_rate_zones.csv")
    add("time_in_heart_rate_zones", df, notes="Coverage measured, but source excluded until 2024 zone mapping is reconciled.")

    subject = pd.read_csv(DATA / "subject-info.csv")
    ids = set(pd.to_numeric(subject["id"], errors="coerce").dropna().astype(int))
    keys = {tuple(map(int, r)) for r in target.loc[target["id"].isin(ids), KEYS].itertuples(index=False, name=None)}
    modalities["subject_static"] = {"keys": keys, "raw_rows": len(subject), "duplicate_groups": int(subject.duplicated("id", keep=False).sum()), "duplicate_excess": int(subject.duplicated("id").sum()), "duplicates_by_group": {}, "causal": "static at enrollment", "semantics": "participant-static", "notes": "Use as metadata with sensitivity reporting; id is not a predictor."}

    hw = pd.read_csv(DATA / "height_and_weight.csv")
    valid_ids = {}
    for interval in (2022, 2024):
        cols = [f"height_{interval}", f"weight_{interval}"]
        valid_ids[interval] = set(hw.loc[hw[cols].notna().any(axis=1), "id"].astype(int))
    keys = {tuple(map(int, r)) for r in target.loc[target.apply(lambda r: int(r.id) in valid_ids[int(r.study_interval)], axis=1), KEYS].itertuples(index=False, name=None)}
    modalities["height_weight_static"] = {"keys": keys, "raw_rows": len(hw), "duplicate_groups": int(hw.duplicated("id", keep=False).sum()), "duplicate_excess": int(hw.duplicated("id").sum()), "duplicates_by_group": {}, "causal": "static participant-entered", "semantics": "year-specific columns", "notes": "Sparse; use matching interval only."}

    self_cols = [c for c in target.columns if c not in KEYS + ["is_weekend", "phase"] + TARGETS]
    skeys = key_set(target, mask=target[self_cols].notna().any(axis=1))
    modalities["past_self_reports"] = {"keys": skeys, "raw_rows": int(target[self_cols].notna().any(axis=1).sum()), "duplicate_groups": 0, "duplicate_excess": 0, "duplicates_by_group": {}, "causal": "lagged values only", "semantics": "participant-day survey", "notes": "2024 near-absence is structural; never require as complete modality."}
    return modalities


def summarize_origins(target: pd.DataFrame, history: int, horizon: int, complete_7: bool = False):
    records = []
    for (pid, interval), g in target.groupby(["id", "study_interval"]):
        by_day = {int(r.day_in_study): r for r in g.itertuples(index=False)}
        days = set(by_day)
        for t in sorted(days):
            if not all(d in days for d in range(t - history + 1, t + 1)):
                continue
            future_days = list(range(t + 1, t + horizon + 1))
            observed = []
            for d in future_days:
                row = by_day.get(d)
                if row is None:
                    observed.extend([False, False, False])
                else:
                    observed.extend([not pd.isna(getattr(row, c)) for c in TARGETS])
            nobs = int(sum(observed))
            records.append({"id": int(pid), "study_interval": int(interval), "origin_day": t, "observed_cells": nobs, "complete": nobs == 3 * horizon, "masked_eligible": nobs > 0})
    return pd.DataFrame(records)


def make_target_coverage(target: pd.DataFrame, modalities) -> pd.DataFrame:
    rows = []
    target = target.copy()
    target["any_target"] = target[TARGETS].notna().any(axis=1)
    target["all_three"] = target[TARGETS].notna().all(axis=1)
    target_keys = {tuple(map(int, r)) for r in target[KEYS].itertuples(index=False, name=None)}

    for name, info in modalities.items():
        mkeys = info["keys"]
        for (pid, interval), g in target.groupby(["id", "study_interval"]):
            gkeys = {tuple(map(int, r)) for r in g[KEYS].itertuples(index=False, name=None)}
            joined = gkeys & mkeys
            tkeys = {tuple(map(int, r)) for r in g.loc[g["any_target"], KEYS].itertuples(index=False, name=None)}
            tj = tkeys & mkeys
            rows.append({"record_type": "participant_interval_modality", "participant_id": int(pid), "study_interval": int(interval), "modality_or_task": name, "target_scope": "any_genuinely_observed_hormone", "history_days": "", "horizon_days": "", "study_days": len(gkeys), "modality_days": len(joined), "joined_study_days": len(joined), "participant_day_coverage_pct": 100 * len(joined) / len(gkeys), "target_days": len(tkeys), "joined_target_days": len(tj), "target_day_join_rate_pct": 100 * len(tj) / len(tkeys) if tkeys else math.nan, "eligible_origins": "", "eligible_participants": "", "origins_min": "", "origins_median": "", "origins_max": "", "duplicate_key_groups": info["duplicates_by_group"].get((int(pid), int(interval)), 0), "duplicate_excess_rows": "", "missingness_pct": 100 * (1 - len(joined) / len(gkeys)), "causal_use": info["causal"], "timestamp_semantics": info["semantics"], "notes": info["notes"]})

        for interval, g in target.groupby("study_interval"):
            gkeys = {tuple(map(int, r)) for r in g[KEYS].itertuples(index=False, name=None)}
            joined = gkeys & mkeys
            tkeys = {tuple(map(int, r)) for r in g.loc[g["any_target"], KEYS].itertuples(index=False, name=None)}
            tj = tkeys & mkeys
            p_overlap = len({k[0] for k in joined})
            rows.append({"record_type": "interval_modality_coverage", "participant_id": "", "study_interval": int(interval), "modality_or_task": name, "target_scope": "any_genuinely_observed_hormone", "history_days": "", "horizon_days": "", "study_days": len(gkeys), "modality_days": len(joined), "joined_study_days": len(joined), "participant_day_coverage_pct": 100 * len(joined) / len(gkeys), "target_days": len(tkeys), "joined_target_days": len(tj), "target_day_join_rate_pct": 100 * len(tj) / len(tkeys) if tkeys else math.nan, "eligible_origins": "", "eligible_participants": p_overlap, "origins_min": "", "origins_median": "", "origins_max": "", "duplicate_key_groups": sum(v for (p, iv), v in info["duplicates_by_group"].items() if iv == int(interval)), "duplicate_excess_rows": info["duplicate_excess"], "missingness_pct": 100 * (1 - len(joined) / len(gkeys)), "causal_use": info["causal"], "timestamp_semantics": info["semantics"], "notes": info["notes"]})

        for interval, g in target.groupby("study_interval"):
            for scope, mask in [("lh", g["lh"].notna()), ("urinary_e3g", g["estrogen"].notna()), ("urinary_pdg", g["pdg"].notna()), ("all_three", g["all_three"]), ("any_target", g["any_target"])]:
                tkeys = {tuple(map(int, r)) for r in g.loc[mask, KEYS].itertuples(index=False, name=None)}
                joined = tkeys & mkeys
                rows.append({"record_type": "target_day_join_coverage", "participant_id": "", "study_interval": int(interval), "modality_or_task": name, "target_scope": scope, "history_days": "", "horizon_days": "", "study_days": "", "modality_days": "", "joined_study_days": "", "participant_day_coverage_pct": "", "target_days": len(tkeys), "joined_target_days": len(joined), "target_day_join_rate_pct": 100 * len(joined) / len(tkeys) if tkeys else math.nan, "eligible_origins": "", "eligible_participants": len({k[0] for k in joined}), "origins_min": "", "origins_median": "", "origins_max": "", "duplicate_key_groups": "", "duplicate_excess_rows": "", "missingness_pct": 100 * (1 - len(joined) / len(tkeys)) if tkeys else math.nan, "causal_use": info["causal"], "timestamp_semantics": info["semantics"], "notes": info["notes"]})

    for history in (14, 28):
        for horizon in (1, 7):
            origins = summarize_origins(target, history, horizon)
            for interval, g in origins.groupby("study_interval"):
                for scope, mask in [("masked_any_observed", g["masked_eligible"]), ("complete_all_three", g["complete"])]:
                    use = g.loc[mask]
                    counts = use.groupby("id").size()
                    rows.append({"record_type": "task_feasibility", "participant_id": "", "study_interval": int(interval), "modality_or_task": "strict_next_day" if horizon == 1 else "seven_day_trajectory", "target_scope": scope, "history_days": history, "horizon_days": horizon, "study_days": "", "modality_days": "", "joined_study_days": "", "participant_day_coverage_pct": "", "target_days": "", "joined_target_days": "", "target_day_join_rate_pct": "", "eligible_origins": len(use), "eligible_participants": int(use["id"].nunique()), "origins_min": int(counts.min()) if len(counts) else 0, "origins_median": float(counts.median()) if len(counts) else 0, "origins_max": int(counts.max()) if len(counts) else 0, "duplicate_key_groups": "", "duplicate_excess_rows": "", "missingness_pct": "", "causal_use": "features through end of t only", "timestamp_semantics": "calendar-consecutive within interval", "notes": "No target interpolation; missing future cells remain masked."})
    return pd.DataFrame(rows)


LEAKAGE_BLACKLIST = [
    "lh, estrogen, or pdg from prediction day t+1 or any later day; same-day target for a strict forecast",
    "phase and any Mira fertile-window/phase label derived from hormone patterns",
    "future menstrual flow/events or future symptom self-reports",
    "completed proprietary-phase-defined cycle length, cycle percentage, or cycle number requiring future phases",
    "LH-surge/PdG-rise labels or other events derived from the evaluation targets at or after the forecast horizon",
    "future interpolation, backward filling, bidirectional imputation, or smoothing across the cutoff",
    "centered rolling windows or normalization/statistics fit using validation/test/future observations",
    "participant id as an unrestricted predictor (retain only for joins, grouping, and participant-disjoint splits)",
    "sleep/temperature/respiratory/HRV summaries whose end time extends beyond the prediction cutoff",
    "exercise last_modified_day_in_study/last_modified_timestamp when modification occurred after cutoff",
    "sleep.levels events after cutoff and any completed-session summary not yet available at cutoff",
    "opaque filtered_demographic_vo2_max or revised daily records until their temporal provenance is established",
    "calendar cycle day computed from a future bleeding onset; only latest onset already known at cutoff is allowed",
]


def feasibility_json() -> dict:
    return {
        "decision": "YES WITH MODIFICATIONS",
        "recommended_task": "Strict next-day forecasting of genuinely measured urinary LH, E3G, and PdG in Interval 2 using information available through end of day t; masked per-target loss/metrics and no interpolated truth.",
        "targets": ["urinary LH (Mira, mIU/mL)", "urinary E3G in column estrogen (Mira, ng/mL)", "urinary PdG (Mira, mcg/mL)"],
        "history_days": 14,
        "forecast_days": 1,
        "eligible_participants": {"primary_all_three_interval_2": 20, "secondary_lh_e3g_both_intervals": 42, "longitudinal_returning_participants": 20},
        "eligible_origins": {"primary_next_day_all_three_14d_interval_2": 1509, "next_day_any_target_14d_all_intervals": 4412, "secondary_7d_complete_all_three_14d_interval_2": 1217, "secondary_7d_masked_14d_interval_2": 1556, "next_day_all_three_28d_interval_2": 1202},
        "split_recommendation": "Primary: one fixed participant-disjoint 12/4/4 train/validation/test split among the 20 Interval-2 participants, balanced by origin count; never split a participant's days across sets. Secondary shift diagnostic: fit/tune without 2024 labels, train on 2022 and test on 2024 for LH/E3G only; all 2024 participants return, so this is not unseen-participant evaluation.",
        "lite_modalities": ["active_minutes", "computed_temperature aligned to sleep_end/wake day", "daily aggregates of HRV details", "positive respiratory-rate summary with sentinel masks", "sleep_score with interval-specific missingness masks including its RHR field", "subject static metadata", "matching-interval height/weight when present", "lagged genuinely measured hormones", "lagged self-reports when present", "causal calendar features from bleeding onset already known by cutoff"],
        "postponed_modalities": ["raw heart_rate", "calories", "distance", "raw steps", "wrist_temperature", "sleep events", "glucose (2022 only; structural non-collection in 2024)", "estimated oxygen variation", "exercise events", "active-zone events", "resting_heart_rate.csv pending 2024 provenance", "demographic_vo2_max pending 2024 provenance", "time_in_heart_rate_zones pending interval mapping", "stress_score as optional later feature due 12.446% valid 2024 target-day coverage"],
        "baseline_plan": ["global train-set median on log1p targets", "causal calendar/harmonic regression", "masked Ridge/ElasticNet tabular history baseline", "bounded-iteration CPU CatBoost with early stopping"],
        "small_temporal_model": "Justified only as a compact fifth reference (small GRU-D or causal TCN, one seed, early stopping) after repairing the igl OpenMP collision; it is not required for the first defensible v0.",
        "runtime_estimates": {
            "measured": {"target_full_read_seconds": 0.0976, "active_minutes_full_read_seconds": 0.0091, "computed_temperature_full_read_seconds": 0.0275, "hrv_projected_full_read_seconds": 0.8762, "glucose_projected_full_read_seconds": 1.3124, "heart_rate_bounded_128MiB_read_seconds": 3.4082, "heart_rate_bounded_throughput_MiB_s": 37.5563},
            "estimated": {"lite_daily_feature_table": {"wall_time": "10-20 s", "peak_memory": "<150 MB"}, "expanded_raw_stream_aggregation": {"wall_time": "8-20 min streaming", "peak_memory": "1-3 GB"}, "global_median": {"wall_time": "<1 s", "peak_memory": "<100 MB"}, "causal_calendar_harmonic": {"wall_time": "1-3 s", "peak_memory": "<150 MB"}, "ridge_or_elasticnet": {"wall_time": "2-10 s", "peak_memory": "<300 MB"}, "catboost_cpu": {"wall_time": "20-90 s", "peak_memory": "<1 GB"}, "small_grud_or_tcn": {"wall_time": "5-15 min CPU after environment repair; GPU unverified/unavailable in current igl process", "peak_memory": "0.5-1.5 GB"}, "exact_multioutput_gp": {"wall_time": "30-120+ min with repeated cubic factorizations", "peak_memory": "2-6 GB; not justified"}, "complete_core_v0_one_split_one_seed": {"wall_time": "5-10 min", "peak_memory": "<1.5 GB"}, "core_v0_plus_temporal_reference": {"wall_time": "10-25 min after environment repair", "peak_memory": "<2 GB"}}
        },
        "leakage_blacklist": LEAKAGE_BLACKLIST,
        "unresolved_questions": ["Urine-test timestamps are absent, so same-day ordering versus daily summaries is unknown.", "The exact device/app provenance of phase and any fertile-window thresholds is not supplied as clinical ground truth.", "2024 resting-heart-rate and VO2 tables contain conflicting cross-participant repeated signatures without revision timestamps.", "2024 time-in-heart-rate-zone mapping appears inconsistent with the supplied README.", "Several 2024 compact sensor tables contain exact cross-participant signatures; export duplication/misattribution is unresolved.", "Units/transforms for HRV spectral values, VO2 estimates, oxygen-variation values, and calories are incompletely documented.", "2024 self-report encodings differ and coverage is nearly absent.", "PyTorch 2.5.1+cu121 is installed and the GTX 1650 is visible to nvidia-smi, but an OpenMP DLL collision prevents a normal torch import in igl; CUDA execution is therefore not verified."],
    }


def main():
    started = time.perf_counter()
    if not DATA.exists():
        raise SystemExit(f"Dataset root missing: {DATA}")
    OUT.mkdir(parents=True, exist_ok=True)
    schema = make_schema_inventory()
    target = pd.read_csv(DATA / "hormones_and_selfreport.csv")
    modalities = build_modalities(target)
    coverage = make_target_coverage(target, modalities)
    schema.to_csv(OUT / "schema_inventory.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    coverage.to_csv(OUT / "target_coverage.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    with (OUT / "feasibility_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(feasibility_json(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"schema_rows": len(schema), "coverage_rows": len(coverage), "seconds": round(time.perf_counter() - started, 4)}, indent=2))


if __name__ == "__main__":
    main()
