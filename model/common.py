"""Small utilities shared by the three baseline implementations."""

from __future__ import annotations

import json
import platform
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from benchmark.contracts import (
    HORMONES,
    HORIZON,
    PREDICTION_COLUMNS,
    TRACK,
    PreparedBundle,
    PreparedSplit,
    load_prepared_bundle,
    validate_prediction_frame,
)


MODEL_VERSION = "0.1.0"


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the public YAML without importing benchmark implementation helpers."""

    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must contain a YAML mapping")
    required = {"task", "paths", "models"}
    if missing := sorted(required - set(config)):
        raise ValueError(f"Config is missing sections: {missing}")
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parents[1])
    return config


def config_path(config: dict[str, Any], key: str) -> Path:
    return Path(config["_project_root"]) / config["paths"][key]


def find_prepared_files(config: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve the one versioned private bundle, failing closed on ambiguity."""

    prepared_dir = config_path(config, "prepared_dir")
    preferred_csv = prepared_dir / "prepared.csv"
    preferred_json = prepared_dir / "metadata.json"
    if preferred_csv.is_file() and preferred_json.is_file():
        return preferred_csv, preferred_json

    csv_files = sorted(prepared_dir.glob("*.csv"))
    json_files = sorted(prepared_dir.glob("*.json"))
    if len(csv_files) == 1 and len(json_files) == 1:
        return csv_files[0], json_files[0]
    raise FileNotFoundError(
        "Expected prepared.csv and metadata.json (or one unambiguous CSV/JSON pair) "
        f"under {prepared_dir}"
    )


def load_bundle_from_config(config: dict[str, Any]) -> PreparedBundle:
    prepared_csv, metadata_json = find_prepared_files(config)
    return load_prepared_bundle(prepared_csv, metadata_json)


def require_training_view(bundle: PreparedSplit, name: str) -> None:
    if not bundle.include_truth:
        raise PermissionError(f"{name} must expose training truth")


def require_inference_view(bundle: PreparedSplit) -> None:
    if bundle.include_truth:
        raise PermissionError("predict requires an inference-only view without truth")


def make_prediction_frame(
    bundle: PreparedSplit,
    predictions: dict[str, Iterable[float]],
    *,
    model_name: str,
    model_version: str = MODEL_VERSION,
) -> pd.DataFrame:
    """Create and validate the stable long-form submission schema."""

    require_inference_view(bundle)
    sample_ids = bundle.sample_ids.astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for hormone in HORMONES:
        values = np.asarray(list(predictions[hormone]), dtype=float)
        if len(values) != len(sample_ids):
            raise ValueError(f"{hormone} prediction length does not match sample count")
        # Urinary concentrations and their log1p transform are nonnegative.
        values = np.maximum(values, 0.0)
        rows.extend(
            {
                "sample_id": sample_id,
                "hormone": hormone,
                "horizon": HORIZON,
                "y_pred": float(value),
                "model_name": model_name,
                "model_version": model_version,
                "track": TRACK,
                "split": bundle.split,
            }
            for sample_id, value in zip(sample_ids, values, strict=True)
        )
    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    return validate_prediction_frame(
        frame, expected_sample_ids=sample_ids, expected_split=bundle.split
    )


@dataclass
class TrainOnlyTabularPreprocessor:
    """Deterministic train-fitted numeric encoding and median imputation."""

    feature_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    numeric_medians: dict[str, float] | None = None
    category_maps: dict[str, dict[str, int]] | None = None
    fitted: bool = False

    def fit(self, frame: pd.DataFrame, feature_columns: Iterable[str]) -> "TrainOnlyTabularPreprocessor":
        self.feature_columns = tuple(feature_columns)
        if not self.feature_columns:
            raise ValueError("The full-feature baseline requires at least one feature")
        absent = sorted(set(self.feature_columns) - set(frame.columns))
        if absent:
            raise ValueError(f"Training frame is missing features: {absent}")

        numeric: list[str] = []
        categorical: list[str] = []
        medians: dict[str, float] = {}
        mappings: dict[str, dict[str, int]] = {}
        for column in self.feature_columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                numeric.append(column)
                converted = pd.to_numeric(series, errors="coerce").astype(float)
                median = converted.median(skipna=True)
                medians[column] = float(median) if pd.notna(median) else 0.0
            else:
                categorical.append(column)
                observed = sorted(series.dropna().astype(str).unique().tolist())
                mappings[column] = {value: idx for idx, value in enumerate(observed)}
        self.numeric_columns = tuple(numeric)
        self.categorical_columns = tuple(categorical)
        self.numeric_medians = medians
        self.category_maps = mappings
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.numeric_medians is None or self.category_maps is None:
            raise RuntimeError("Preprocessor has not been fit")
        absent = sorted(set(self.feature_columns) - set(frame.columns))
        if absent:
            raise ValueError(f"Inference frame is missing features: {absent}")
        transformed: dict[str, pd.Series] = {}
        for column in self.feature_columns:
            if column in self.numeric_columns:
                values = pd.to_numeric(frame[column], errors="coerce").astype(float)
                transformed[column] = values.fillna(self.numeric_medians[column])
            else:
                mapping = self.category_maps[column]
                transformed[column] = (
                    frame[column]
                    .astype("string")
                    .map(mapping)
                    .fillna(-1)
                    .astype(float)
                )
        return pd.DataFrame(transformed, index=frame.index).reset_index(drop=True)

    def metadata(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Preprocessor has not been fit")
        return {
            "feature_count": len(self.feature_columns),
            "numeric_feature_count": len(self.numeric_columns),
            "categorical_feature_count": len(self.categorical_columns),
            "preprocessing": "train-only median imputation and ordinal unknown=-1 encoding",
        }


def attempt_catboost_import(*, repair: bool = False):
    """Import CatBoost without mutating the runtime environment.

    ``repair`` is retained only for v0 call compatibility and is deliberately
    ignored. Dependency setup belongs outside model fitting.
    """

    try:
        from catboost import CatBoostRegressor

        return CatBoostRegressor, None
    except Exception as error:  # pragma: no cover - environment dependent
        return None, f"{type(error).__name__}: {error}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


class PeakRSSMonitor:
    """Sample process RSS during one model run when psutil is available."""

    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = interval_seconds
        self.peak_rss_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any = None

    def __enter__(self) -> "PeakRSSMonitor":
        try:
            import psutil

            self._process = psutil.Process()
            self.peak_rss_mb = self._process.memory_info().rss / (1024**2)
        except Exception:  # pragma: no cover - optional measurement dependency
            return self

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    rss_mb = self._process.memory_info().rss / (1024**2)
                    self.peak_rss_mb = max(self.peak_rss_mb or 0.0, rss_mb)
                except Exception:
                    return

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.interval_seconds * 3))
        if self._process is not None:
            try:
                rss_mb = self._process.memory_info().rss / (1024**2)
                self.peak_rss_mb = max(self.peak_rss_mb or 0.0, rss_mb)
            except Exception:
                pass
