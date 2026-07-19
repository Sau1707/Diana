"""Independent prediction evaluator for Hormonbench-mcPHASES v0.

This module intentionally has no dependency on :mod:`model`.  A conforming CSV
from any implementation can be evaluated against the private prepared truth.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .contracts import (
    HORMONES,
    TARGET_LOG_COLUMNS,
    TARGET_RAW_COLUMNS,
    PreparedBundle,
    load_prepared_bundle,
    validate_prediction_frame,
)
from .metrics import (
    MetricResult,
    add_reference_comparison,
    calculate_metrics,
    train_log1p_iqr_scales,
)
from .task import load_config, project_path


@dataclass(frozen=True)
class EvaluatedModel:
    """One model's aggregate result and private in-memory comparison rows."""

    model_name: str
    model_version: str
    summary: dict[str, Any]
    participant_metrics: pd.DataFrame


def load_bundle_for_config(config: Mapping[str, Any]) -> PreparedBundle:
    """Load the versioned private bundle from its stable v0 filenames."""

    prepared_dir = project_path(dict(config), "prepared_dir")
    prepared_csv = prepared_dir / "prepared.csv"
    metadata_json = prepared_dir / "metadata.json"
    missing = [str(path) for path in (prepared_csv, metadata_json) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Private prepared bundle is incomplete: {missing}")
    return load_prepared_bundle(prepared_csv, metadata_json)


def _truth_long(bundle: PreparedBundle) -> pd.DataFrame:
    test = bundle.frame.loc[bundle.frame["split"].eq("test")]
    if test.empty:
        raise ValueError("Prepared bundle contains no test samples")

    pieces: list[pd.DataFrame] = []
    for hormone in HORMONES:
        piece = test.loc[
            :,
            [
                "sample_id",
                "private_participant_id",
                TARGET_LOG_COLUMNS[hormone],
                TARGET_RAW_COLUMNS[hormone],
            ],
        ].copy()
        piece["hormone"] = hormone
        piece = piece.rename(
            columns={
                TARGET_LOG_COLUMNS[hormone]: "y_true_log1p",
                TARGET_RAW_COLUMNS[hormone]: "y_true_raw",
            }
        )
        pieces.append(piece)
    truth = pd.concat(pieces, ignore_index=True)
    if truth.duplicated(["sample_id", "hormone"]).any():
        raise ValueError("Private truth contains duplicate sample/hormone keys")
    return truth


def join_predictions_to_private_truth(
    bundle: PreparedBundle, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate a submission and join it one-to-one to private test truth.

    Returns the canonical validated submission and the evaluator-internal joined
    frame.  External submissions never need to contain ``y_true``.
    """

    bundle.validate()
    test_ids = bundle.frame.loc[bundle.frame["split"].eq("test"), "sample_id"].astype(
        str
    )
    canonical = validate_prediction_frame(
        predictions, expected_sample_ids=test_ids, expected_split="test"
    )
    canonical["sample_id"] = canonical["sample_id"].astype(str)

    truth = _truth_long(bundle)
    truth["sample_id"] = truth["sample_id"].astype(str)
    joined = truth.merge(
        canonical,
        on=["sample_id", "hormone"],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = len(truth)
    if len(joined) != expected_rows:
        raise ValueError(
            f"Prediction/truth join coverage mismatch: expected {expected_rows}, "
            f"joined {len(joined)}"
        )
    return canonical, joined


def evaluate_prediction_frame(
    bundle: PreparedBundle,
    predictions: pd.DataFrame,
    *,
    train_scales: Mapping[str, float | None] | None = None,
) -> EvaluatedModel:
    """Evaluate one in-memory prediction submission."""

    canonical, joined = join_predictions_to_private_truth(bundle, predictions)
    if train_scales is None:
        train = bundle.frame.loc[bundle.frame["split"].eq("train")]
        train_scales = train_log1p_iqr_scales(train)
    metrics: MetricResult = calculate_metrics(joined, train_scales)
    model_name = str(canonical["model_name"].iloc[0])
    model_version = str(canonical["model_version"].iloc[0])
    summary = {
        "model_name": model_name,
        "model_version": model_version,
        "n_test_samples": int(canonical["sample_id"].nunique()),
        "n_prediction_rows": int(len(canonical)),
        **metrics.public,
    }
    return EvaluatedModel(
        model_name=model_name,
        model_version=model_version,
        summary=summary,
        participant_metrics=metrics.participant,
    )


def evaluate_prediction_file(
    bundle: PreparedBundle,
    prediction_csv: str | Path,
    *,
    train_scales: Mapping[str, float | None] | None = None,
) -> EvaluatedModel:
    """Read and evaluate one external prediction CSV."""

    path = Path(prediction_csv)
    if not path.is_file():
        raise FileNotFoundError(path)
    return evaluate_prediction_frame(
        bundle, pd.read_csv(path), train_scales=train_scales
    )


def _split_counts(bundle: PreparedBundle) -> dict[str, dict[str, int]]:
    return {
        split: {
            "participants": int(
                rows["private_participant_id"].astype(str).nunique()
            ),
            "eligible_origins": int(len(rows)),
        }
        for split in ("train", "validation", "test")
        for rows in [bundle.frame.loc[bundle.frame["split"].eq(split)]]
    }


def evaluate_prediction_files(
    bundle: PreparedBundle,
    prediction_files: Iterable[str | Path],
    *,
    reference_model: str = "causal_calendar",
) -> dict[str, Any]:
    """Evaluate a set of submissions and add paired reference comparisons."""

    files = sorted((Path(path) for path in prediction_files), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError("No prediction CSV files were supplied")
    train = bundle.frame.loc[bundle.frame["split"].eq("train")]
    scales = train_log1p_iqr_scales(train)
    evaluated = [
        evaluate_prediction_file(bundle, path, train_scales=scales) for path in files
    ]

    names = [item.model_name for item in evaluated]
    if len(names) != len(set(names)):
        raise ValueError("Prediction directory contains duplicate model_name submissions")
    references = [item for item in evaluated if item.model_name == reference_model]
    if len(references) != 1:
        raise ValueError(
            f"Expected exactly one {reference_model!r} reference submission; "
            f"found {len(references)}"
        )
    reference = references[0]
    for item in evaluated:
        add_reference_comparison(
            item.summary,
            item.participant_metrics,
            reference.summary,
            reference.participant_metrics,
        )

    metadata = bundle.metadata
    public = {
        "task_id": metadata.get("task_id", metadata.get("task", {}).get("id")),
        "task_version": metadata.get(
            "task_version", metadata.get("task", {}).get("version")
        ),
        "track": metadata.get("track", "primary_interval2_nextday"),
        "split": "test",
        "prediction_space": "log1p",
        "reference_model": reference_model,
        "split_counts": _split_counts(bundle),
        "train_log1p_iqr": scales,
        "models": {item.model_name: item.summary for item in evaluated},
    }
    return public


def prediction_files_for_config(config: Mapping[str, Any]) -> list[Path]:
    """Return only top-level model submissions from the private prediction dir."""

    prediction_dir = project_path(dict(config), "prediction_dir")
    if not prediction_dir.is_dir():
        raise FileNotFoundError(prediction_dir)
    files = sorted(prediction_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSVs found in {prediction_dir}")
    return files


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def evaluate_config(
    config_path: str | Path,
    *,
    prediction_files: Iterable[str | Path] | None = None,
    write: bool = True,
) -> tuple[dict[str, Any], Path | None]:
    """Evaluate configured submissions and optionally write aggregate metrics."""

    started = time.perf_counter()
    config = load_config(config_path)
    bundle = load_bundle_for_config(config)
    files = (
        list(prediction_files)
        if prediction_files is not None
        else prediction_files_for_config(config)
    )
    reference = config.get("evaluation", {}).get(
        "reference_model", "causal_calendar"
    )
    metrics = evaluate_prediction_files(bundle, files, reference_model=reference)
    metrics["evaluation_seconds"] = float(time.perf_counter() - started)
    if not write:
        return metrics, None
    output = project_path(config, "results_dir") / "metrics.json"
    _write_json(output, metrics)
    return metrics, output
