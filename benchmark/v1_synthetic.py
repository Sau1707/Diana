"""Small governed-data-free fixture for executable v1 contract smoke tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from benchmark.data.v1_folds import group_hash, validate_five_fold_protocol, write_private_fold_manifest
from benchmark.v1_contracts import (
    TARGET_LOG_COLUMNS,
    TARGET_RAW_COLUMNS,
    V1PreparedBundle,
    save_v1_bundle,
)
from benchmark.v1_task import (
    HORMONES,
    TASK_ID,
    TASK_VERSION,
    canonical_hash,
    config_hash,
    input_schema_hash,
    file_sha256,
    task_spec_hash,
)


SYNTHETIC_FEATURES = (
    "active__lightly__mean",
    "temperature__nightly_temperature__last",
    "hrv__rmssd__coverage",
    "respiratory__full_sleep_breathing_rate__time_since",
    "sleep_score__overall_score__lag0",
    "weekend__is_weekend__mean",
)


def make_synthetic_bundle(
    config: Mapping[str, Any], *, seed: int = 20260719
) -> tuple[V1PreparedBundle, dict[int, int]]:
    rng = np.random.default_rng(seed)
    mapping = {participant: (participant - 1) // 4 for participant in range(1, 21)}
    digest = group_hash(mapping, int(config["folds"]["seed"]))
    provenance = {
        name: {
            "source": "fully synthetic fixture",
            "modality": name.split("__", 1)[0],
            "history_window": "t-13 through t",
            "uses_future": False,
        }
        for name in SYNTHETIC_FEATURES
    }
    schema_digest = input_schema_hash(list(SYNTHETIC_FEATURES), provenance)
    scientific_hash = task_spec_hash(config)
    operational_hash = config_hash(config)
    rows: list[dict[str, Any]] = []
    hormone_intercepts = {"lh": 0.8, "e3g": 3.2, "pdg": 1.4}
    for participant in range(1, 21):
        participant_effect = rng.normal(0, 0.18, len(HORMONES))
        for index in range(12):
            origin = 100 + index
            x = rng.normal(size=len(SYNTHETIC_FEATURES))
            sample_id = canonical_hash(
                [TASK_ID, TASK_VERSION, scientific_hash, participant, origin]
            )[:24]
            row: dict[str, Any] = {
                "task_id": TASK_ID,
                "task_version": TASK_VERSION,
                "sample_id": sample_id,
                "private_participant_id": participant,
                "origin_day": origin,
                "target_day": origin + 1,
                "history_start_day": origin - 13,
                "history_end_day": origin,
                "cutoff_day": origin,
                "fold_group": mapping[participant],
                "config_hash": operational_hash,
                "task_spec_hash": scientific_hash,
                "input_schema_hash": schema_digest,
                "fold_hash": digest,
                **{name: float(value) for name, value in zip(SYNTHETIC_FEATURES, x)},
            }
            if index % 5 == 0:
                row[SYNTHETIC_FEATURES[2]] = np.nan
            for hormone_index, hormone in enumerate(HORMONES):
                logged = max(
                    0.0,
                    hormone_intercepts[hormone]
                    + participant_effect[hormone_index]
                    + 0.08 * x[hormone_index]
                    + rng.normal(0, 0.10),
                )
                row[TARGET_LOG_COLUMNS[hormone]] = logged
                row[TARGET_RAW_COLUMNS[hormone]] = float(np.expm1(logged))
            rows.append(row)
    frame = pd.DataFrame(rows)
    fixed = [
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
        *TARGET_RAW_COLUMNS.values(),
        *TARGET_LOG_COLUMNS.values(),
        *SYNTHETIC_FEATURES,
    ]
    frame = frame[fixed]
    protocol = validate_five_fold_protocol(mapping, frame["private_participant_id"])
    metadata = {
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "interval": 2024,
        "history_days": 14,
        "forecast_days": 1,
        "eligible_participants": 20,
        "eligible_origins": len(frame),
        "feature_columns": list(SYNTHETIC_FEATURES),
        "feature_provenance": provenance,
        "approved_modalities": ["fully_synthetic"],
        "self_reports_in_feature_matrix": False,
        "menstrual_calendar_in_feature_matrix": False,
        "absolute_time_in_feature_matrix": False,
        "temperature_alignment": "sleep_end_day_in_study",
        "config_hash": operational_hash,
        "task_spec_hash": scientific_hash,
        "input_schema_hash": schema_digest,
        "fold_hash": digest,
        "fold_protocol": protocol,
    }
    bundle = V1PreparedBundle(frame, metadata)
    bundle.validate()
    return bundle, mapping


def configure_synthetic_workspace(
    config: Mapping[str, Any], root: str | Path
) -> tuple[dict[str, Any], V1PreparedBundle]:
    root = Path(root).resolve()
    output = copy.deepcopy(dict(config))
    output["_project_root"] = str(root)
    output["custom"]["selected_covariance_mode"] = "full"
    output["models"]["catboost"].update(
        {
            "iterations": 12,
            "validation_iterations": 12,
            "depth": 3,
            "early_stopping_rounds": 3,
            "thread_count": 2,
        }
    )
    paths = {
        "prepared_dir": "private/prepared",
        "fold_dir": "private/folds",
        "calibration_dir": "private/calibration",
        "prediction_run_dir": "private/predictions/run",
        "prediction_manifest": "private/predictions/run/manifest.json",
        "checkpoint_dir": "private/checkpoints/run",
        "participant_metrics_dir": "private/participant_metrics/run",
        "validation_dir": "private/validation",
        "selection_artifact": "private/validation/selection.json",
        "results_dir": "private/results",
    }
    output["paths"].update(paths)
    validation_dir = root / paths["validation_dir"]
    validation_dir.mkdir(parents=True, exist_ok=True)
    selection_path = root / paths["selection_artifact"]
    selection_path.write_text(
        json.dumps(
            {
                "selection_scope": "fold_0_validation_only",
                "selected_covariance_mode": "full",
                "success_gate_passed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output["custom"]["selection_artifact_sha256"] = file_sha256(selection_path)
    bundle, mapping = make_synthetic_bundle(output)
    prepared_dir = root / paths["prepared_dir"]
    save_v1_bundle(bundle, prepared_dir)
    (prepared_dir / "runtime.json").write_text(
        json.dumps({"prepare_seconds": 0.0, "prepare_peak_rss_mb": 0.0}),
        encoding="utf-8",
    )
    diagnostics = {
        f"group_{group}_participants": 4 for group in range(5)
    }
    diagnostics.update({f"group_{group}_origins": 48 for group in range(5)})
    write_private_fold_manifest(
        root / paths["fold_dir"] / "folds.json",
        mapping=mapping,
        seed=int(output["folds"]["seed"]),
        digest=bundle.metadata["fold_hash"],
        diagnostics=diagnostics,
        protocol=bundle.metadata["fold_protocol"],
    )
    return output, bundle
