"""Command-line interface for the model-independent benchmark package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .evaluator import evaluate_config
from .report import generate_report
from .task import load_config


def _is_v1(config_path: str) -> bool:
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return str(raw.get("task", {}).get("id", "")).endswith("_v1")


def _prepare(config_path: str) -> int:
    # Imported only for the prepare command: evaluator/report remain independent of
    # adapter implementation details, and benchmark never imports model.
    if _is_v1(config_path):
        from .data.v1_adapter import prepare_v1
        from .v1_task import load_v1_config

        result = prepare_v1(load_v1_config(config_path))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    from .data import adapter

    config = load_config(config_path)
    entry = getattr(adapter, "prepare_from_config", None)
    if entry is None:
        entry = getattr(adapter, "prepare", None)
    if entry is None:
        raise RuntimeError(
            "benchmark.data.adapter must expose prepare_from_config(config)"
        )
    result = entry(config)
    print("Prepared private Hormonbench bundle.")
    if isinstance(result, dict):
        public_summary_keys = (
            "task_id",
            "task_version",
            "eligible_participants",
            "eligible_origins",
            "origin_count_changed_by_no_hormone_history_contract",
            "split_counts",
            "config_hash",
            "split_hash",
            "prepare_seconds",
            "prepare_peak_rss_mb",
        )
        print(
            json.dumps(
                {key: result[key] for key in public_summary_keys if key in result},
                indent=2,
            )
        )
    elif result is not None:
        print(str(result))
    return 0


def _evaluate(config_path: str, prediction_files: list[str] | None) -> int:
    if _is_v1(config_path):
        if prediction_files:
            raise ValueError("v1 evaluation uses only its explicit private manifest")
        from .v1_evaluator import evaluate_v1
        from .v1_task import load_v1_config

        result = evaluate_v1(load_v1_config(config_path))
        print(json.dumps({"metric_rows": len(result["rows"])}, indent=2))
        return 0
    metrics, output = evaluate_config(
        config_path,
        prediction_files=prediction_files or None,
        write=True,
    )
    print(
        json.dumps(
            {
                "models_evaluated": sorted(metrics["models"]),
                "split_counts": metrics["split_counts"],
                "metrics": str(output),
            },
            indent=2,
        )
    )
    return 0


def _report(config_path: str) -> int:
    if _is_v1(config_path):
        from .v1_report import report_v1
        from .v1_task import load_v1_config

        result = report_v1(load_v1_config(config_path))
        print(json.dumps({"task_id": result["task_id"], "status": "complete"}, indent=2))
        return 0
    outputs = generate_report(config_path)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark",
        description="Prepare, evaluate, and report Hormonbench-mcPHASES v0.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build the private prepared bundle")
    prepare.add_argument("--config", required=True, type=str)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate private prediction submissions"
    )
    evaluate.add_argument("--config", required=True, type=str)
    evaluate.add_argument(
        "--prediction-files",
        nargs="+",
        default=None,
        help="Optional explicit prediction CSVs; defaults to configured directory",
    )

    report = subparsers.add_parser(
        "report", help="Generate the aggregate public leaderboard and figure"
    )
    report.add_argument("--config", required=True, type=str)

    privacy = subparsers.add_parser(
        "privacy", help="Validate v1 public artifacts and release allow-list"
    )
    privacy.add_argument("--config", required=True, type=str)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = str(Path(args.config))
    if args.command == "prepare":
        return _prepare(config_path)
    if args.command == "evaluate":
        return _evaluate(config_path, args.prediction_files)
    if args.command == "report":
        return _report(config_path)
    if args.command == "privacy":
        from .v1_privacy import validate_public_results
        from .v1_task import load_v1_config

        result = validate_public_results(load_v1_config(config_path))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)
