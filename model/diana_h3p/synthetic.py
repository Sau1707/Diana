"""Governed-data-free five-fold Diana-H3P contract smoke run."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from benchmark.h3p_report import report_h3p
from benchmark.v1_evaluator import evaluate_v1
from benchmark.v1_privacy import validate_public_inventory
from benchmark.v1_synthetic import configure_synthetic_workspace
from benchmark.v1_task import input_schema_hash, task_spec_hash
from model.diana_h3p.pipeline import evaluation_config, run_official_h3p
from model.v1_pipeline import run_official_five_folds


def run_synthetic_h3p(
    benchmark_config: dict[str, Any],
    h3p_config: dict[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    """Exercise baseline generation, H3P, evaluator, report, and figures synthetically."""

    implementation_spec = (
        Path(h3p_config["_project_root"])
        / h3p_config["paths"]["implementation_spec"]
    ).resolve()
    synthetic_benchmark, bundle = configure_synthetic_workspace(
        benchmark_config, root
    )
    synthetic_benchmark["models"]["catboost"].update(
        {
            "iterations": 6,
            "validation_iterations": 6,
            "early_stopping_rounds": 2,
            "thread_count": 2,
        }
    )
    # The bundle operational hash was created before this speed-only test override;
    # restore the settings used to build it while retaining the fixture's own limit.
    synthetic_benchmark["models"]["catboost"].update(
        {
            "iterations": 12,
            "validation_iterations": 12,
            "early_stopping_rounds": 3,
            "thread_count": 2,
        }
    )
    baseline_manifest = run_official_five_folds(synthetic_benchmark, bundle)
    synthetic_h3p = copy.deepcopy(h3p_config)
    synthetic_h3p["_project_root"] = str(Path(root).resolve())
    synthetic_h3p["backend"]["canonical"] = "numpy"
    synthetic_h3p["runtime"]["training_code_commit"] = "synthetic-test"
    synthetic_h3p["runtime"]["run_id"] = "synthetic_h3p"
    synthetic_h3p["expected_benchmark"].update(
        {
            "eligible_participants": 20,
            "eligible_origins": len(bundle.frame),
            "common_suffix_origins": 100,
            "task_spec_hash": task_spec_hash(synthetic_benchmark),
            "fold_hash": bundle.metadata["fold_hash"],
            "input_schema_hash": input_schema_hash(
                bundle.feature_columns, bundle.metadata["feature_provenance"]
            ),
        }
    )
    baseline_dir = Path(root) / synthetic_benchmark["paths"]["prediction_run_dir"]
    synthetic_h3p["paths"].update(
        {
            "baseline_prediction_dir": str(baseline_dir.relative_to(root)),
            "baseline_prediction_manifest": str(
                (baseline_dir / "manifest.json").relative_to(root)
            ),
            "preserved_baseline_dir": str(baseline_dir.relative_to(root)),
            "preserved_baseline_manifest": str(
                (baseline_dir / "manifest.json").relative_to(root)
            ),
            "private_run_root": "private/h3p/run",
            "prediction_dir": "private/h3p/run/predictions",
            "prediction_manifest": "private/h3p/run/predictions/manifest.json",
            "oof_dir": "private/h3p/run/oof",
            "checkpoint_dir": "private/h3p/run/checkpoints",
            "participant_metrics_dir": "private/h3p/run/participant_metrics",
            "audit_dir": "private/h3p/run/manifests",
            "results_dir": "results/v1/diana_h3p",
            "implementation_spec": str(implementation_spec),
        }
    )
    h3p_manifest = run_official_h3p(synthetic_benchmark, synthetic_h3p)
    evaluator_config = evaluation_config(synthetic_benchmark, synthetic_h3p)
    metrics = evaluate_v1(evaluator_config)
    report = report_h3p(evaluator_config, synthetic_h3p)
    result_root = Path(root) / "results" / "v1" / "diana_h3p"
    public_files = sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in result_root.rglob("*")
        if path.is_file()
    )
    privacy = validate_public_inventory(root, public_files)
    return {
        "baseline_manifest_entries": len(baseline_manifest["entries"]),
        "h3p_manifest_entries": len(h3p_manifest["entries"]),
        "metric_rows": len(metrics["rows"]),
        "eligible_participants": report["eligible_participants"],
        "eligible_origins": report["eligible_origins"],
        "common_suffix_origins": report["common_suffix_origins"],
        "privacy": privacy,
        "results_dir": str(result_root),
    }
