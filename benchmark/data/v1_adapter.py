"""Private mcPHASES adapter for the frozen Hormonbench v1 task."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil

from benchmark.v1_contracts import (
    HORMONES,
    PRIVATE_ALIGNMENT_COLUMNS,
    TARGET_LOG_COLUMNS,
    TARGET_RAW_COLUMNS,
    V1PreparedBundle,
    assert_v1_feature_names_safe,
    save_v1_bundle,
)
from benchmark.v1_task import (
    HISTORY_DAYS,
    INTERVAL,
    TASK_ID,
    TASK_VERSION,
    config_hash,
    input_schema_hash,
    project_path,
    task_spec_hash,
)

from .v1_features import build_v1_history_features, load_v1_daily_features
from .v1_folds import (
    build_v1_groups,
    fold_roles,
    group_hash,
    validate_five_fold_protocol,
    write_private_fold_manifest,
)


TARGET_SOURCE_COLUMNS = {"lh": "lh", "e3g": "estrogen", "pdg": "pdg"}


class PeakRSS:
    def __init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSS":
        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self._stop.wait(0.02):
                self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)

        self.peak_bytes = process.memory_info().rss
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def find_v1_eligible_origins(
    all_targets: pd.DataFrame,
    *,
    interval: int = INTERVAL,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    target = all_targets.loc[all_targets["study_interval"].eq(interval)].copy()
    records: list[dict[str, Any]] = []
    for participant, group in target.groupby("id", sort=True):
        if group["day_in_study"].duplicated().any():
            raise ValueError("Duplicate target participant-day key")
        by_day = group.set_index("day_in_study", drop=False)
        days = {int(value) for value in by_day.index}
        for origin_day in sorted(days):
            if not all(
                day in days
                for day in range(origin_day - history_days + 1, origin_day + 1)
            ):
                continue
            target_day = origin_day + 1
            if target_day not in days:
                continue
            label = by_day.loc[target_day]
            if any(pd.isna(label[source]) for source in TARGET_SOURCE_COLUMNS.values()):
                continue
            records.append(
                {
                    "private_participant_id": int(participant),
                    "origin_day": int(origin_day),
                    "target_day": int(target_day),
                    **{
                        TARGET_RAW_COLUMNS[hormone]: float(label[source])
                        for hormone, source in TARGET_SOURCE_COLUMNS.items()
                    },
                }
            )
    origins = pd.DataFrame(records)
    if origins.empty:
        raise ValueError("No eligible v1 origins")
    return origins


def stable_sample_id(
    participant: int, origin_day: int, target_day: int, scientific_hash: str
) -> str:
    private = (
        f"{TASK_ID}|{TASK_VERSION}|{scientific_hash}|"
        f"{int(participant)}|{int(origin_day)}|{int(target_day)}"
    )
    return hashlib.sha256(private.encode("utf-8")).hexdigest()[:24]


def prepare_v1(config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    with PeakRSS() as memory:
        data_root = project_path(config, "data_root")
        prepared_dir = project_path(config, "prepared_dir")
        fold_dir = project_path(config, "fold_dir")
        target_path = data_root / "hormones_and_selfreport.csv"
        all_targets = pd.read_csv(target_path)
        required = {
            "id",
            "study_interval",
            "day_in_study",
            "is_weekend",
            "lh",
            "estrogen",
            "pdg",
        }
        if not required <= set(all_targets.columns):
            raise ValueError(f"Unexpected target schema: {sorted(required-set(all_targets))}")
        if all_targets.duplicated(["id", "study_interval", "day_in_study"]).any():
            raise ValueError("Target table contains duplicate participant-day rows")

        scientific_hash = task_spec_hash(config)
        operational_hash = config_hash(config)
        origins = find_v1_eligible_origins(all_targets)
        if len(origins) != 1509 or origins["private_participant_id"].nunique() != 20:
            raise RuntimeError(
                "Blocking v1 eligibility mismatch: expected 1,509 origins / 20 participants"
            )
        origins["sample_id"] = [
            stable_sample_id(
                row.private_participant_id,
                row.origin_day,
                row.target_day,
                scientific_hash,
            )
            for row in origins.itertuples(index=False)
        ]

        daily, daily_provenance, participant_coverage = load_v1_daily_features(
            data_root, all_targets, INTERVAL
        )
        participant_stats = (
            origins.groupby("private_participant_id")
            .size()
            .rename("eligible_origin_count")
            .reset_index()
        )
        participant_stats["approved_wearable_day_coverage"] = (
            participant_stats["private_participant_id"]
            .map(participant_coverage)
            .astype(float)
        )
        v0_manifest_path = project_path(config, "v0_split_manifest")
        v0_manifest = json.loads(v0_manifest_path.read_text(encoding="utf-8"))
        seed = int(config["folds"]["seed"])
        groups, diagnostics = build_v1_groups(
            participant_stats,
            v0_manifest["participant_to_split"],
            seed=seed,
            candidate_permutations=int(
                config["folds"]["balance_candidate_permutations"]
            ),
        )
        digest = group_hash(groups, seed)
        origins["fold_group"] = origins["private_participant_id"].map(groups).astype(int)
        protocol = validate_five_fold_protocol(
            groups, origins["private_participant_id"]
        )
        if int(protocol["unique_test_origins"]) != 1509:
            raise RuntimeError("Five-fold test-origin union must equal 1,509")

        history, feature_provenance = build_v1_history_features(
            daily,
            origins[["sample_id", "private_participant_id", "origin_day"]],
            history_days=HISTORY_DAYS,
            selected_lags=[int(value) for value in config["features"]["selected_lags"]],
            base_provenance=daily_provenance,
        )
        prepared = origins.merge(history, on="sample_id", how="left", validate="one_to_one")
        prepared["task_id"] = TASK_ID
        prepared["task_version"] = TASK_VERSION
        prepared["history_start_day"] = prepared["origin_day"] - HISTORY_DAYS + 1
        prepared["history_end_day"] = prepared["origin_day"]
        prepared["cutoff_day"] = prepared["origin_day"]
        prepared["config_hash"] = operational_hash
        prepared["task_spec_hash"] = scientific_hash
        prepared["fold_hash"] = digest
        for hormone in HORMONES:
            prepared[TARGET_LOG_COLUMNS[hormone]] = np.log1p(
                prepared[TARGET_RAW_COLUMNS[hormone]].astype(float)
            )
        fixed_without_schema = [
            column
            for column in PRIVATE_ALIGNMENT_COLUMNS
            if column != "input_schema_hash"
        ] + list(TARGET_RAW_COLUMNS.values()) + list(TARGET_LOG_COLUMNS.values())
        feature_columns = [
            column for column in prepared.columns if column not in fixed_without_schema
        ]
        assert_v1_feature_names_safe(feature_columns)
        schema_digest = input_schema_hash(feature_columns, feature_provenance)
        prepared["input_schema_hash"] = schema_digest
        fixed = list(PRIVATE_ALIGNMENT_COLUMNS) + list(TARGET_RAW_COLUMNS.values()) + list(
            TARGET_LOG_COLUMNS.values()
        )
        prepared = (
            prepared[fixed + feature_columns]
            .sort_values(["fold_group", "private_participant_id", "origin_day"])
            .reset_index(drop=True)
        )
        fold_counts: dict[str, Any] = {}
        for fold in range(5):
            roles = fold_roles(groups, fold)
            fold_counts[str(fold)] = {
                role: {
                    "participants": len(ids),
                    "origins": int(prepared["private_participant_id"].isin(ids).sum()),
                }
                for role, ids in roles.items()
            }
        metadata = {
            "task_id": TASK_ID,
            "task_version": TASK_VERSION,
            "interval": INTERVAL,
            "history_days": HISTORY_DAYS,
            "forecast_days": 1,
            "eligible_participants": 20,
            "eligible_origins": 1509,
            "feature_columns": feature_columns,
            "feature_provenance": feature_provenance,
            "approved_modalities": list(config["features"]["modalities"]),
            "self_reports_in_feature_matrix": False,
            "menstrual_calendar_in_feature_matrix": False,
            "absolute_time_in_feature_matrix": False,
            "temperature_alignment": "sleep_end_day_in_study",
            "config_hash": operational_hash,
            "task_spec_hash": scientific_hash,
            "input_schema_hash": schema_digest,
            "fold_hash": digest,
            "fold_counts": fold_counts,
            "fold_protocol": protocol,
            "fold_diagnostics": diagnostics,
        }
        bundle = V1PreparedBundle(prepared, metadata)
        bundle.validate()
        save_v1_bundle(bundle, prepared_dir)
        write_private_fold_manifest(
            fold_dir / "folds.json",
            mapping=groups,
            seed=seed,
            digest=digest,
            diagnostics=diagnostics,
            protocol=protocol,
        )
    runtime = {
        "prepare_seconds": float(time.perf_counter() - started),
        "prepare_peak_rss_mb": float(memory.peak_bytes / (1024**2)),
    }
    (prepared_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**metadata, **runtime}


def load_private_group_mapping(config: Mapping[str, Any]) -> dict[int, int]:
    manifest = json.loads(
        (project_path(config, "fold_dir") / "folds.json").read_text(encoding="utf-8")
    )
    return {int(pid): int(group) for pid, group in manifest["participant_to_group"].items()}
