"""CLI for fitting the three Hormonbench v0 baseline families."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from benchmark.contracts import validate_prediction_frame
from model.common import (
    PeakRSSMonitor,
    config_path,
    load_bundle_from_config,
    load_config,
    runtime_environment,
    write_json,
)
from model.registry import available_models, create_model


def parse_model_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("At least one model name is required")
    unknown = sorted(set(names) - set(available_models()))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown models {unknown}; available={list(available_models())}"
        )
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("Model names may not be repeated")
    return names


def run_models(
    config_file: str | Path,
    model_names: Sequence[str],
    *,
    quick: bool = False,
) -> list[dict[str, Any]]:
    """Fit selected families once and write private contract predictions."""

    config = load_config(config_file)
    bundle = load_bundle_from_config(config)
    train = bundle.view("train", include_truth=True)
    validation = bundle.view("validation", include_truth=True)
    # The model receives no test target columns through its public fit/predict inputs.
    test = bundle.view("test", include_truth=False)
    prediction_dir = config_path(config, "prediction_dir")
    checkpoint_dir = config_path(config, "checkpoint_dir")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, Any]] = []
    actual_names: set[str] = set()
    for requested_name in model_names:
        model = create_model(requested_name, config, quick=quick)
        with PeakRSSMonitor() as memory_monitor:
            fit_started = time.perf_counter()
            model.fit(train, validation)
            fit_seconds = time.perf_counter() - fit_started

            predict_started = time.perf_counter()
            predictions = model.predict(test)
            predict_seconds = time.perf_counter() - predict_started
        predictions = validate_prediction_frame(
            predictions,
            expected_sample_ids=test.sample_ids,
            expected_split="test",
        )

        metadata = dict(model.get_metadata())
        actual_name = str(metadata["model_name"])
        if actual_name in actual_names:
            raise RuntimeError(f"Multiple requested families resolved to {actual_name}")
        actual_names.add(actual_name)

        prediction_path = prediction_dir / f"{actual_name}.csv"
        checkpoint_path = checkpoint_dir / f"{actual_name}.json"
        predictions.to_csv(prediction_path, index=False)
        metadata.update(
            {
                "requested_model": requested_name,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "prediction_rows": len(predictions),
                "train_rows": len(train.frame),
                "validation_rows": len(validation.frame),
                "test_rows": len(test.frame),
                "quick": quick,
                "peak_rss_mb": memory_monitor.peak_rss_mb,
                "runtime_environment": runtime_environment(),
            }
        )
        write_json(checkpoint_path, metadata)
        completed.append(
            {
                "requested_model": requested_name,
                "model_name": actual_name,
                "prediction_file": str(prediction_path),
                "checkpoint_file": str(checkpoint_path),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "prediction_rows": len(predictions),
                "peak_rss_mb": memory_monitor.peak_rss_mb,
            }
        )
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="fit baselines and write private predictions")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument(
        "--models",
        required=True,
        type=parse_model_names,
        help="comma-separated names: population_median,causal_calendar,catboost",
    )
    run.add_argument("--quick", action="store_true")
    validation = subparsers.add_parser(
        "validate-v1", help="run fold-0 validation-only custom selection"
    )
    validation.add_argument("--config", required=True, type=Path)
    v1_run = subparsers.add_parser(
        "run-v1", help="run the frozen five-fold v1 models sequentially"
    )
    v1_run.add_argument("--config", required=True, type=Path)
    correction = subparsers.add_parser(
        "correct-v1-custom",
        help="reuse valid baselines and recompute the validation-selected custom path",
    )
    correction.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        completed = run_models(args.config, args.models, quick=args.quick)
        print(json.dumps({"completed": completed}, indent=2))
        return 0
    if args.command in {"validate-v1", "run-v1", "correct-v1-custom"}:
        from benchmark.v1_task import load_v1_config
        from model.v1_pipeline import (
            run_corrected_full_custom,
            run_fold0_validation,
            run_official_five_folds,
        )

        config = load_v1_config(args.config)
        if args.command == "validate-v1":
            result = run_fold0_validation(config)
        elif args.command == "correct-v1-custom":
            result = run_corrected_full_custom(config)
        else:
            result = run_official_five_folds(config)
        if args.command == "validate-v1":
            result = {
                "selection_scope": result["selection_scope"],
                "selected_covariance_mode": result["selected_covariance_mode"],
                "success_gate_passed": result["success_gate_passed"],
                "strongest_personalized_classical": result[
                    "strongest_personalized_classical"
                ],
                "scoring_origins": result["scoring_origins"],
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)
