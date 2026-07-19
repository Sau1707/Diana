"""Frozen Hormonbench-mcPHASES v0 task definition and config helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .contracts import TASK_ID, TRACK


TASK_VERSION = "0.1.0"
INTERVAL = 2024
HISTORY_DAYS = 14
FORECAST_DAYS = 1
SPLIT_SEED = 20260719
SPLIT_SIZES = {"train": 12, "validation": 4, "test": 4}

APPROVED_MODALITIES = (
    "active_minutes",
    "computed_temperature_end_day",
    "hrv_daily_aggregate",
    "respiratory_rate_summary",
    "sleep_score",
    "past_self_reports",
    "causal_calendar",
)

DEFERRED_MODALITIES = (
    "resting_heart_rate.csv",
    "demographic_vo2_max.csv",
    "time_in_heart_rate_zones.csv",
    "stress_score.csv",
    "subject-info.csv",
    "height_and_weight.csv",
    "raw and event streams",
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task", {}).get("id") != TASK_ID:
        raise ValueError(f"Config task id must be {TASK_ID}")
    if config["task"].get("track") != TRACK:
        raise ValueError(f"Config track must be {TRACK}")
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parents[1])
    return config


def config_hash(config: dict[str, Any]) -> str:
    public = {k: v for k, v in config.items() if not k.startswith("_")}
    payload = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def project_path(config: dict[str, Any], key: str) -> Path:
    return Path(config["_project_root"]) / config["paths"][key]

