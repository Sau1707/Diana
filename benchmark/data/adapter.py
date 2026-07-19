"""mcPHASES adapter for the frozen Hormonbench v0 primary task."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from benchmark.contracts import (
    HORMONES,
    PREPARED_ID_COLUMNS,
    TARGET_LOG_COLUMNS,
    TARGET_RAW_COLUMNS,
    PreparedBundle,
    TASK_ID,
    TRACK,
    assert_feature_names_safe,
)
from benchmark.task import APPROVED_MODALITIES, HISTORY_DAYS, INTERVAL, SPLIT_SEED, SPLIT_SIZES, TASK_VERSION, config_hash, project_path

from .features import build_history_features, load_daily_features
from .splits import generate_fixed_split, split_hash, validate_participant_split


TARGET_SOURCE_COLUMNS = {"lh": "lh", "e3g": "estrogen", "pdg": "pdg"}


class PeakRSS:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        process = psutil.Process(os.getpid())
        def sample():
            while not self._stop.is_set():
                self.peak = max(self.peak, process.memory_info().rss)
                self._stop.wait(0.02)
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


def find_eligible_origins(all_targets: pd.DataFrame, *, interval: int = INTERVAL, history_days: int = HISTORY_DAYS) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    target = all_targets.loc[all_targets["study_interval"].eq(interval)].copy()
    for pid, group in target.groupby("id"):
        by_day = group.set_index("day_in_study", drop=False)
        days = {int(x) for x in by_day.index}
        for cutoff in sorted(days):
            if not all(day in days for day in range(cutoff - history_days + 1, cutoff + 1)):
                continue
            target_day = cutoff + 1
            if target_day not in days:
                continue
            label = by_day.loc[target_day]
            if isinstance(label, pd.DataFrame):
                raise ValueError("Duplicate hormone participant-day key")
            if any(pd.isna(label[source]) for source in TARGET_SOURCE_COLUMNS.values()):
                continue
            records.append({
                "private_participant_id": int(pid),
                "origin_day": int(cutoff),
                "target_day": int(target_day),
                **{TARGET_RAW_COLUMNS[h]: float(label[source]) for h, source in TARGET_SOURCE_COLUMNS.items()},
            })
    origins = pd.DataFrame(records)
    if origins.empty:
        raise ValueError("No eligible primary-task origins")
    return origins


def _sample_id(pid: int, origin_day: int, target_day: int, config_digest: str) -> str:
    private = f"{TASK_ID}|{TASK_VERSION}|{pid}|{origin_day}|{target_day}|{config_digest}"
    return hashlib.sha256(private.encode()).hexdigest()[:24]


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    with PeakRSS() as memory:
        data_root = project_path(config, "data_root")
        prepared_dir = project_path(config, "prepared_dir")
        split_dir = project_path(config, "split_dir")
        prepared_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)
        cfg_hash = config_hash(config)
        target_path = data_root / "hormones_and_selfreport.csv"
        all_targets = pd.read_csv(target_path)
        required = {"id", "study_interval", "day_in_study", "flow_volume", "lh", "estrogen", "pdg"}
        if not required <= set(all_targets.columns):
            raise ValueError(f"Unexpected target schema; missing {sorted(required-set(all_targets.columns))}")
        if all_targets.duplicated(["id", "study_interval", "day_in_study"]).any():
            raise ValueError("Target table has duplicate participant-day keys")

        origins = find_eligible_origins(all_targets)
        origins["sample_id"] = [
            _sample_id(int(row.private_participant_id), int(row.origin_day), int(row.target_day), cfg_hash)
            for row in origins.itertuples(index=False)
        ]
        daily, daily_provenance = load_daily_features(data_root, all_targets, INTERVAL)
        signal_columns = [c for c in daily.columns if c not in {"id", "day_in_study"}]
        day_coverage = daily.assign(_coverage=daily[signal_columns].notna().mean(axis=1)).groupby("id")["_coverage"].mean()
        participant_stats = origins.groupby("private_participant_id").size().rename("eligible_origin_count").reset_index()
        participant_stats["approved_modality_day_coverage"] = participant_stats["private_participant_id"].map(day_coverage).fillna(0.0)
        mapping, split_diagnostics = generate_fixed_split(
            participant_stats,
            seed=int(config["split"]["seed"]),
            sizes={"train": int(config["split"]["train_participants"]), "validation": int(config["split"]["validation_participants"]), "test": int(config["split"]["test_participants"])},
            candidate_permutations=int(config["split"]["balance_candidate_permutations"]),
        )
        validate_participant_split(mapping, SPLIT_SIZES)
        digest = split_hash(mapping, SPLIT_SEED)
        origins["split"] = origins["private_participant_id"].map(mapping)

        history, feature_provenance = build_history_features(
            daily,
            origins[["sample_id", "private_participant_id", "origin_day"]],
            all_targets=all_targets,
            history_days=HISTORY_DAYS,
            selected_lags=[int(x) for x in config["features"]["selected_lags"]],
            base_provenance=daily_provenance,
        )
        prepared = origins.merge(history, on="sample_id", how="left", validate="one_to_one")
        prepared["task_version"] = TASK_VERSION
        prepared["history_start_day"] = prepared["origin_day"] - HISTORY_DAYS + 1
        prepared["history_end_day"] = prepared["origin_day"]
        prepared["cutoff_day"] = prepared["origin_day"]
        prepared["config_hash"] = cfg_hash
        prepared["split_hash"] = digest
        for hormone in HORMONES:
            prepared[TARGET_LOG_COLUMNS[hormone]] = np.log1p(prepared[TARGET_RAW_COLUMNS[hormone]].astype(float))

        fixed = list(PREPARED_ID_COLUMNS) + list(TARGET_RAW_COLUMNS.values()) + list(TARGET_LOG_COLUMNS.values())
        feature_columns = [c for c in prepared.columns if c not in fixed]
        assert_feature_names_safe(feature_columns)
        prepared = prepared[fixed + feature_columns].sort_values(["split", "private_participant_id", "origin_day"]).reset_index(drop=True)
        split_counts = {}
        for name, group in prepared.groupby("split"):
            split_counts[name] = {"participants": int(group["private_participant_id"].nunique()), "origins": int(len(group))}
        metadata = {
            "task_id": TASK_ID,
            "task_version": TASK_VERSION,
            "track": TRACK,
            "interval": INTERVAL,
            "history_days": HISTORY_DAYS,
            "forecast_days": 1,
            "eligible_origins": int(len(prepared)),
            "eligible_participants": int(prepared["private_participant_id"].nunique()),
            "phase0_reference_origins": 1509,
            "origin_count_changed_by_no_hormone_history_contract": int(len(prepared)) != 1509,
            "feature_columns": feature_columns,
            "feature_provenance": feature_provenance,
            "approved_modalities": list(APPROVED_MODALITIES),
            "prohibited_inputs": ["all past/current/future hormones", "participant id predictor", "Mira phase/fertile window", "future menstruation", "completed cycle features", "future fills/interpolation", "summaries ending after cutoff"],
            "cutoff_metadata": {"cutoff": "end of origin day t", "history": "exact calendar days t-13 through t", "target": "genuinely observed t+1", "temperature_alignment": "sleep_end_day_in_study", "hormone_history_used": False},
            "config_hash": cfg_hash,
            "split_hash": digest,
            "split_counts": split_counts,
            "split_balance_diagnostics": split_diagnostics,
        }
        bundle = PreparedBundle(prepared, metadata)
        bundle.validate()
        prepared.to_csv(prepared_dir / "prepared.csv", index=False)
        (prepared_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        private_manifest = {
            "task_id": TASK_ID,
            "seed": SPLIT_SEED,
            "split_hash": digest,
            "participant_to_split": {str(pid): mapping[pid] for pid in sorted(mapping)},
            "aggregate_counts": split_counts,
            "balance_quantities": config["split"]["balance_quantities"],
            "diagnostics": split_diagnostics,
        }
        (split_dir / f"{TASK_ID}.json").write_text(json.dumps(private_manifest, indent=2), encoding="utf-8")
    runtime = time.perf_counter() - started
    runtime_record = {"prepare_seconds": runtime, "prepare_peak_rss_mb": memory.peak / (1024**2)}
    (prepared_dir / "runtime.json").write_text(json.dumps(runtime_record, indent=2), encoding="utf-8")
    return {**metadata, **runtime_record}

