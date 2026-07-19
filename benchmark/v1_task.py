"""Frozen task/configuration helpers for Hormonbench-mcPHASES v1."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


TASK_ID = "hormonbench_mcphases_interval2_nextday_v1"
TASK_VERSION = "1.0.0"
INTERVAL = 2024
HISTORY_DAYS = 14
FORECAST_DAYS = 1
HORMONES = ("lh", "e3g", "pdg")
TRACK_COLD = "cold_start_participant_independent"
TRACK_FEW_SHOT = "few_shot_personalization"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_v1_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("v1 config must be a YAML mapping")
    task = config.get("task", {})
    if task.get("id") != TASK_ID or str(task.get("version")) != TASK_VERSION:
        raise ValueError(f"Expected {TASK_ID} version {TASK_VERSION}")
    if int(task.get("history_days", -1)) != HISTORY_DAYS:
        raise ValueError("v1 history_days must remain 14")
    if int(task.get("forecast_days", -1)) != FORECAST_DAYS:
        raise ValueError("v1 forecast_days must remain 1")
    seed = int(config["folds"]["seed"])
    if int(config["models"]["seed"]) != seed:
        raise ValueError("fold and model seeds must use one source value")
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parents[1])
    return config


def project_path(config: Mapping[str, Any], key: str) -> Path:
    return Path(str(config["_project_root"])) / str(config["paths"][key])


def scientific_task_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path/machine/model-independent scientific task definition."""

    task = config["task"]
    features = config["features"]
    personalization = config["personalization"]
    return {
        "task": {
            "id": task["id"],
            "version": str(task["version"]),
            "dataset": task["dataset"],
            "interval": int(task["interval"]),
            "history_days": int(task["history_days"]),
            "forecast_days": int(task["forecast_days"]),
            "target_space": task["target_space"],
            "targets": list(task["targets"]),
            "label_rule": "all_three_genuinely_observed_at_t_plus_1",
        },
        "features": {
            "modalities": list(features["modalities"]),
            "summary_statistics": list(features["summary_statistics"]),
            "selected_lags": [int(x) for x in features["selected_lags"]],
            "self_reports": "excluded",
            "menstrual_calendar": "excluded",
            "absolute_time": "excluded",
            "temperature_alignment": "sleep_end_day_in_study",
        },
        "tracks": {
            "cold": TRACK_COLD,
            "few_shot": TRACK_FEW_SHOT,
            "budgets": [int(x) for x in personalization["budgets"]],
            "calibration_source": personalization["calibration_source"],
            "common_suffix_budget": int(
                personalization["common_suffix_budget"]
            ),
            "scoring_rule": personalization["scoring_rule"],
        },
    }


def task_spec_hash(config: Mapping[str, Any]) -> str:
    return canonical_hash(scientific_task_spec(config))


def config_hash(config: Mapping[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    return canonical_hash(public)


def input_schema_hash(
    feature_columns: list[str] | tuple[str, ...], provenance: Mapping[str, Any]
) -> str:
    return canonical_hash(
        {
            "feature_columns": list(feature_columns),
            "provenance": {name: provenance[name] for name in feature_columns},
        }
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return {
        "branch": branch or "unknown",
        "commit": head.stdout.strip() if head.returncode == 0 else "uncommitted",
        "working_tree_clean": not bool(status.strip()),
    }
