"""Aggregate-only reporting and public manifest generation for v1."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import psutil

from benchmark.v1_task import (
    TRACK_COLD,
    TRACK_FEW_SHOT,
    config_hash,
    file_sha256,
    git_state,
    project_path,
    task_spec_hash,
)


PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "PyYAML",
    "psutil",
    "matplotlib",
    "catboost",
)


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _gpu_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "used": False}
    fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    return {
        "available": True,
        "name": fields[0],
        "memory_total_mb": int(float(fields[1])),
        "driver_version": fields[2],
        "used": False,
    }


def _save_svg(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def report_v1(config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    results_dir = project_path(config, "results_dir")
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    rows = pd.DataFrame(metrics["rows"])
    fold_rows = pd.DataFrame(metrics["fold_rows"])
    uncertainty = pd.DataFrame(metrics["uncertainty_rows"])
    cold = rows.loc[
        rows["track"].eq(TRACK_COLD) & rows["calibration_budget"].eq(0)
    ].sort_values("overall_normalized_score")
    few = rows.loc[rows["track"].eq(TRACK_FEW_SHOT)].sort_values(
        ["calibration_budget", "overall_normalized_score"]
    )
    cold_dir = results_dir / "cold_start"
    few_dir = results_dir / "few_shot"
    figures_dir = results_dir / "figures"
    cold_dir.mkdir(parents=True, exist_ok=True)
    few_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cold.to_csv(cold_dir / "leaderboard.csv", index=False)
    fold_rows.loc[fold_rows["track"].eq(TRACK_COLD)].to_csv(
        cold_dir / "fold_metrics.csv", index=False
    )
    few.to_csv(few_dir / "leaderboard_by_budget.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = cold.sort_values("overall_normalized_score", ascending=True)
    ax.barh(plot["model_name"], plot["overall_normalized_score"], color="#6C5CE7")
    ax.set_xlabel("Overall participant-macro normalized log1p-MAE (lower is better)")
    ax.set_title("Hormonbench v1 cold-start benchmark")
    ax.grid(axis="x", alpha=0.25)
    _save_svg(fig, figures_dir / "cold_start_leaderboard.svg")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name, group in few.groupby("model_name", sort=True):
        ordered = group.sort_values("calibration_budget")
        ax.plot(
            ordered["calibration_budget"],
            ordered["overall_normalized_score"],
            marker="o",
            label=model_name,
        )
    ax.set_xticks([0, 3, 7])
    ax.set_xlabel("Authorized personal complete hormone measurements (K)")
    ax.set_ylabel("Overall normalized score (lower is better)")
    ax.set_title("Measurement-budget personalization curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save_svg(fig, figures_dir / "measurement_budget_curve.svg")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    if not uncertainty.empty:
        custom = uncertainty.loc[uncertainty["track"].eq(TRACK_FEW_SHOT)]
        coverage = custom.groupby("calibration_budget")[
            "participant_macro_coverage_80"
        ].mean()
        width = custom.groupby("calibration_budget")[
            "participant_macro_mean_width"
        ].mean()
        axes[0].plot(coverage.index, coverage.values, marker="o", color="#00A8A8")
        axes[0].axhline(0.8, linestyle="--", color="black", linewidth=1)
        axes[1].plot(width.index, width.values, marker="o", color="#E17055")
    axes[0].set_title("Participant-macro research-interval coverage")
    axes[0].set_ylabel("Coverage")
    axes[1].set_title("Mean research-interval width")
    axes[1].set_ylabel("log1p units")
    for ax in axes:
        ax.set_xticks([0, 3, 7])
        ax.set_xlabel("K")
        ax.grid(alpha=0.25)
    _save_svg(fig, figures_dir / "uncertainty_summary.svg")

    winner = cold.iloc[0]
    cold_table = [
        "| Model | Overall | LH log-MAE | E3G log-MAE | PdG log-MAE | Improved participants |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in cold.itertuples(index=False):
        cold_table.append(
            f"| `{row.model_name}` | {row.overall_normalized_score:.6f} | "
            f"{row.lh_participant_macro_log1p_mae:.6f} | "
            f"{row.e3g_participant_macro_log1p_mae:.6f} | "
            f"{row.pdg_participant_macro_log1p_mae:.6f} | "
            f"{int(row.participants_improved_vs_population_median)}/20 |"
        )
    few_table = [
        "| K | Population median | Wearable Ridge | CatBoost | Custom reference |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget in (0, 3, 7):
        group = few.loc[few["calibration_budget"].eq(budget)].set_index("model_name")
        few_table.append(
            f"| {budget} | {group.loc['population_median', 'overall_normalized_score']:.6f} | "
            f"{group.loc['wearable_ridge', 'overall_normalized_score']:.6f} | "
            f"{group.loc['catboost', 'overall_normalized_score']:.6f} | "
            f"{group.loc['joint_bayes_personalizer', 'overall_normalized_score']:.6f} |"
        )
    results_markdown = [
        "# Hormonbench-mcPHASES v1 results",
        "",
        "> These are descriptive research-benchmark results on 20 participants, not clinical validation.",
        "",
        "v1 supersedes the provisional v0 results because v0 exposed a stale cross-interval menstrual-calendar feature to two models. v1 removes that feature and all self-reports from the active core.",
        "",
        f"Cold-start winner: `{winner['model_name']}` with overall normalized score {winner['overall_normalized_score']:.6f}.",
        "",
        *cold_table,
        "",
        "## Few-shot measurement budgets",
        "",
        *few_table,
        "",
        "The custom reference did not meet its prespecified fold-0 validation superiority gate. The official results are therefore an honest benchmark characterization, not a model-superiority claim.",
        "",
        "The participant-disjoint five-fold protocol tests each participant exactly once. Fold dispersion is descriptive because development sets overlap and daily origins share history.",
        "",
        "Few-shot K=0/3/7 rows use one identical post-seventh-measurement scoring suffix. K counts complete authorized urinary LH/E3G/PdG readings among eligible forecast targets.",
        "",
        "Research prediction intervals are calibrated from development participants only. Longitudinal overlap weakens ordinary conformal exchangeability, so these intervals are not clinical confidence intervals.",
        "",
        "Targets are participant-entered at-home urinary-monitor readings. They are not serum hormone concentrations or clinical gold-standard measurements.",
    ]
    (results_dir / "RESULTS.md").write_text(
        "\n".join(results_markdown) + "\n", encoding="utf-8"
    )

    prepared_meta = json.loads(
        (project_path(config, "prepared_dir") / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    prepared_runtime = json.loads(
        (project_path(config, "prepared_dir") / "runtime.json").read_text(
            encoding="utf-8"
        )
    )
    private_run = json.loads(
        (project_path(config, "checkpoint_dir") / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    fold_metadata = private_run["fold_metadata"]
    output_paths = [
        results_dir / "metrics.json",
        results_dir / "RESULTS.md",
        cold_dir / "leaderboard.csv",
        cold_dir / "fold_metrics.csv",
        few_dir / "leaderboard_by_budget.csv",
        figures_dir / "cold_start_leaderboard.svg",
        figures_dir / "measurement_budget_curve.svg",
        figures_dir / "uncertainty_summary.svg",
    ]
    public_manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "task_id": prepared_meta["task_id"],
        "task_version": prepared_meta["task_version"],
        "task_spec_hash": task_spec_hash(config),
        "config_hash": config_hash(config),
        "fold_hash": prepared_meta["fold_hash"],
        "input_schema_hash": prepared_meta["input_schema_hash"],
        "eligible_participants": int(prepared_meta["eligible_participants"]),
        "eligible_origins": int(prepared_meta["eligible_origins"]),
        "common_suffix_origins": int(
            sum(item["common_suffix_origins"] for item in fold_metadata)
        ),
        "folds": [
            {
                "fold": int(item["fold"]),
                "train_participants": 12,
                "validation_participants": 4,
                "development_participants": 16,
                "test_participants": 4,
                "test_origins": int(item["test_origins"]),
                "common_suffix_origins": int(item["common_suffix_origins"]),
                "catboost_best_iteration": dict(item["catboost"]["best_iteration"]),
                "catboost_tree_count": dict(item["catboost"]["tree_count"]),
                "catboost_best_validation_score": dict(
                    item["catboost"]["best_validation_score"]
                ),
                "retained_feature_count": {
                    "wearable_ridge": int(
                        item["final_preprocessors"]["wearable_ridge"][
                            "retained_feature_count"
                        ]
                    ),
                    "catboost": int(
                        item["final_preprocessors"]["catboost"][
                            "retained_feature_count"
                        ]
                    ),
                },
                "custom_lambda": dict(item["custom"]["lambda"]),
                "runtime_seconds": float(item["runtime_seconds"]),
            }
            for item in fold_metadata
        ],
        "baselines": ["population_median", "wearable_ridge", "catboost"],
        "custom_reference": "joint_bayes_personalizer",
        "selected_covariance_mode": str(
            config["custom"]["selected_covariance_mode"]
        ),
        "reference_model": "population_median",
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": "python.exe (Conda environment ai)",
            "os": platform.platform(),
            "cpu": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "gpu": _gpu_inventory(),
            "compute_device": "CPU",
            "packages": _versions(),
        },
        "git": git_state(config["_project_root"]),
        "runtime": {
            "preparation_seconds": float(prepared_runtime["prepare_seconds"]),
            "model_seconds": float(private_run["manifest"]["runtime_seconds"]),
            "baseline_fit_oof_seconds": float(
                sum(item["baseline_fit_oof_seconds"] for item in fold_metadata)
            ),
            "custom_fit_calibration_seconds": float(
                sum(item["custom_fit_calibration_seconds"] for item in fold_metadata)
            ),
            "prediction_write_seconds": float(
                sum(item["prediction_write_seconds"] for item in fold_metadata)
            ),
            "evaluation_seconds": float(metrics["runtime_seconds"]),
            "report_seconds": None,
            "peak_rss_mb": float(
                max(
                    prepared_runtime["prepare_peak_rss_mb"],
                    private_run["manifest"]["peak_rss_mb"],
                )
            ),
            "invalidated_diagonal_run_seconds": float(
                private_run["manifest"].get("invalidated_diagonal_run_seconds", 0.0)
            ),
            "correction_only_seconds": float(
                private_run["manifest"].get("correction_only_seconds", 0.0)
            ),
        },
        "outputs": {
            str(path.relative_to(results_dir)).replace("\\", "/"): file_sha256(path)
            for path in output_paths
        },
        "claims": {
            "descriptive_only": True,
            "clinical_validation": False,
            "targets": "participant-entered at-home urinary monitor readings",
            "v0_status": "superseded/provisional",
        },
        "protocol_correction": {
            "applied": bool(private_run["manifest"].get("baseline_outputs_reused", False)),
            "reason": private_run["manifest"].get("selection_correction"),
            "baseline_outputs_reused": bool(
                private_run["manifest"].get("baseline_outputs_reused", False)
            ),
            "outer_test_metrics_used_for_correction": False,
        },
    }
    public_manifest["runtime"]["report_seconds"] = float(time.perf_counter() - started)
    public_manifest["runtime"]["total_pipeline_seconds"] = float(
        public_manifest["runtime"]["preparation_seconds"]
        + public_manifest["runtime"]["model_seconds"]
        + public_manifest["runtime"]["evaluation_seconds"]
        + public_manifest["runtime"]["report_seconds"]
    )
    (results_dir / "run_manifest.json").write_text(
        json.dumps(public_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return public_manifest
