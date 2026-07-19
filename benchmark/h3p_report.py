"""Aggregate-only public reporting for the Diana-H3P Hormonbench v1 run."""

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
    file_sha256,
    git_state,
    project_path,
)


PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "PyYAML",
    "psutil",
    "matplotlib",
    "catboost",
    "torch",
)


def _versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for name in PACKAGES:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "unavailable"
    return output


def _gpu() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "used_for_canonical_model": False}
    values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    return {
        "available": True,
        "name": values[0],
        "memory_total_mb": int(float(values[1])),
        "driver_version": values[2],
        "used_for_canonical_model": False,
    }


def _save_svg(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)


def report_h3p(
    benchmark_config: Mapping[str, Any], h3p_config: Mapping[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(str(h3p_config["_project_root"]))
    results_dir = root / str(h3p_config["paths"]["results_dir"])
    checkpoint_dir = root / str(h3p_config["paths"]["checkpoint_dir"])
    development_dir = (
        root / str(h3p_config["paths"]["private_run_root"])
    ).parent / "development"
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    private_run = json.loads(
        (checkpoint_dir / "run_metadata.json").read_text(encoding="utf-8")
    )
    manifest = private_run["manifest"]
    fold_metadata = private_run["fold_metadata"]
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
    ablation_dir = results_dir / "ablations"
    figures_dir = results_dir / "figures"
    for path in (cold_dir, few_dir, ablation_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)
    cold.to_csv(cold_dir / "leaderboard.csv", index=False)
    fold_rows.loc[fold_rows["track"].eq(TRACK_COLD)].to_csv(
        cold_dir / "fold_metrics.csv", index=False
    )
    few.to_csv(few_dir / "leaderboard_by_budget.csv", index=False)

    development_diagnostics: dict[str, Any] = {}
    diagnostic_path = development_dir / "fold_0_diagnostics.json"
    if diagnostic_path.is_file():
        development_diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    backend_profile: dict[str, Any] = {}
    backend_path = development_dir / "backend_profile.json"
    if backend_path.is_file():
        backend_profile = json.loads(backend_path.read_text(encoding="utf-8"))
    public_ablation = {
        "development_only": True,
        "outer_test_metrics_read": False,
        "fold": 0,
        "expert_participant_macro_log1p_mae": development_diagnostics.get(
            "expert_participant_macro_log1p_mae", {}
        ),
        "stack_participant_macro_log1p_mae": development_diagnostics.get(
            "stack_participant_macro_log1p_mae", {}
        ),
        "stack_weights": development_diagnostics.get("stack_weights", {}),
        "backend_profile": backend_profile,
    }
    (ablation_dir / "development_only.json").write_text(
        json.dumps(public_ablation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(8, 4.5))
    ordered = cold.sort_values("overall_normalized_score", ascending=True)
    colors = ["#6C5CE7" if name != "diana_h3p" else "#00A8A8" for name in ordered["model_name"]]
    axis.barh(ordered["model_name"], ordered["overall_normalized_score"], color=colors)
    axis.set_xlabel("Participant-macro normalized log1p-MAE (lower is better)")
    axis.set_title("Hormonbench v1 cold-start evaluation")
    axis.grid(axis="x", alpha=0.25)
    _save_svg(figure, figures_dir / "cold_start_leaderboard.svg")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for model_name, group in few.groupby("model_name", sort=True):
        ordered = group.sort_values("calibration_budget")
        axis.plot(
            ordered["calibration_budget"],
            ordered["overall_normalized_score"],
            marker="o",
            linewidth=2 if model_name == "diana_h3p" else 1.2,
            label=model_name,
        )
    axis.set_xticks([0, 3, 7])
    axis.set_xlabel("Authorized complete personal hormone measurements (K)")
    axis.set_ylabel("Overall normalized score (lower is better)")
    axis.set_title("Measurement-budget curve on one common scoring suffix")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    _save_svg(figure, figures_dir / "measurement_budget_curve.svg")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    if not uncertainty.empty:
        custom = uncertainty.loc[uncertainty["model_name"].eq("diana_h3p")]
        coverage = custom.groupby("calibration_budget")[
            "participant_macro_coverage_80"
        ].mean()
        width = custom.groupby("calibration_budget")[
            "participant_macro_mean_width"
        ].mean()
        axes[0].plot(coverage.index, coverage.values, marker="o", color="#00A8A8")
        axes[0].axhline(0.8, linestyle="--", color="black", linewidth=1)
        axes[1].plot(width.index, width.values, marker="o", color="#E17055")
    axes[0].set_title("Participant-macro interval coverage")
    axes[0].set_ylabel("Coverage")
    axes[1].set_title("Mean research-interval width")
    axes[1].set_ylabel("log1p units")
    for axis in axes:
        axis.set_xticks([0, 3, 7])
        axis.set_xlabel("K")
        axis.grid(alpha=0.25)
    _save_svg(figure, figures_dir / "uncertainty_summary.svg")

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
        "| K | Population median | Wearable Ridge | CatBoost | Diana-H3P |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget in (0, 3, 7):
        group = few.loc[few["calibration_budget"].eq(budget)].set_index("model_name")
        few_table.append(
            f"| {budget} | {group.loc['population_median', 'overall_normalized_score']:.6f} | "
            f"{group.loc['wearable_ridge', 'overall_normalized_score']:.6f} | "
            f"{group.loc['catboost', 'overall_normalized_score']:.6f} | "
            f"{group.loc['diana_h3p', 'overall_normalized_score']:.6f} |"
        )
    winner = cold.iloc[0]
    markdown = [
        "# Diana-H3P on Hormonbench-mcPHASES v1",
        "",
        "> Descriptive, post-hoc research-benchmark results on 20 participants; not clinical validation or an untouched-test confirmation.",
        "",
        "Hormonbench remains Diana's primary reusable contribution. Diana-H3P is one compact reference implementation with a participant-independent stacked wearable prior and K=0/3/7 empirical-Bayes personalization.",
        "",
        f"Cold-start leader: `{winner.model_name}` at {winner.overall_normalized_score:.6f} (lower is better).",
        "",
        *cold_table,
        "",
        "## Few-shot personalization",
        "",
        *few_table,
        "",
        "K counts the earliest authorized complete urinary LH/E3G/PdG measurements. All budgets use the same post-seventh-measurement scoring suffix; calibration rows are never scored.",
        "",
        "Intervals are 80% participant-block calibrated research prediction intervals. Correlated overlapping windows and the small cohort preclude an IID finite-sample or clinical-confidence interpretation.",
        "",
        "The prior `joint_bayes_personalizer` is retained only as `historical_protocol_compromised_comparator`: a fold-0 validation group selected a global covariance mode before later appearing as outer test. It is not part of this active leaderboard.",
        "",
        "Targets are participant-entered readings from an at-home urinary monitor, not serum concentrations, diagnoses, verified ovulation labels, or clinical gold standards.",
    ]
    (results_dir / "RESULTS.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    prepared = json.loads(
        (project_path(benchmark_config, "prepared_dir") / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    output_paths = [
        results_dir / "metrics.json",
        results_dir / "RESULTS.md",
        cold_dir / "leaderboard.csv",
        cold_dir / "fold_metrics.csv",
        few_dir / "leaderboard_by_budget.csv",
        ablation_dir / "development_only.json",
        figures_dir / "cold_start_leaderboard.svg",
        figures_dir / "measurement_budget_curve.svg",
        figures_dir / "uncertainty_summary.svg",
    ]
    public_manifest: dict[str, Any] = {
        "schema_version": "1.1.0",
        "task_id": prepared["task_id"],
        "task_version": prepared["task_version"],
        "task_spec_hash": manifest["task_spec_hash"],
        "fold_hash": manifest["fold_hash"],
        "input_schema_hash": manifest["input_schema_hash"],
        "evaluator_sha256": file_sha256(Path(__file__).with_name("v1_evaluator.py")),
        "h3p_config_hash": manifest["h3p_config_hash"],
        "h3p_model_spec_hash": manifest["h3p_model_spec_hash"],
        "implementation_spec_sha256": manifest["implementation_spec_sha256"],
        "training_code_commit": manifest["training_code_commit"],
        "eligible_participants": int(prepared["eligible_participants"]),
        "eligible_origins": int(prepared["eligible_origins"]),
        "common_suffix_origins": int(
            sum(item["common_suffix_origins"] for item in fold_metadata)
        ),
        "baselines": ["population_median", "wearable_ridge", "catboost"],
        "custom_reference": "diana_h3p",
        "legacy_custom_status": "historical_protocol_compromised_comparator",
        "reference_model": "population_median",
        "post_hoc_existing_protocol": True,
        "untouched_test_confirmation": False,
        "outer_test_used_for_h3p_model_selection": False,
        "baseline_outputs_reused": bool(manifest["baseline_outputs_reused"]),
        "folds": [
            {
                "fold": int(item["fold"]),
                "train_participants": 12,
                "validation_participants": 4,
                "development_participants": 16,
                "test_participants": 4,
                "test_origins": int(item["test_origins"]),
                "common_suffix_origins": int(item["common_suffix_origins"]),
                "layer1_weights": item["layer1"]["weights"],
                "covariance": item["layer2"]["covariance"],
                "nested_catboost_tree_counts": [
                    block["catboost_tree_count"]
                    for block in item["layer1"]["nested_blocks"]
                ],
                "runtime_seconds": float(item["runtime_seconds"]),
            }
            for item in fold_metadata
        ],
        "backend": {
            "canonical": manifest["canonical_backend"],
            "profile": backend_profile,
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": "python.exe (Conda environment ai)",
            "os": platform.platform(),
            "cpu": platform.processor() or "unknown",
            "logical_cpu_count": os.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "gpu": _gpu(),
            "packages": _versions(),
        },
        "git": {
            **git_state(root),
            "training_code_commit": manifest["training_code_commit"],
        },
        "runtime": {
            "model_seconds": float(manifest["runtime_seconds"]),
            "development_oof_seconds": float(
                sum(item["oof_seconds"] for item in fold_metadata)
            ),
            "layer2_fit_seconds": float(
                sum(item["layer2_fit_seconds"] for item in fold_metadata)
            ),
            "evaluation_seconds": float(metrics["runtime_seconds"]),
            "peak_rss_mb": float(manifest["peak_rss_mb"]),
            "report_seconds": None,
        },
        "outputs": {
            str(path.relative_to(results_dir)).replace("\\", "/"): file_sha256(path)
            for path in output_paths
        },
        "claims": {
            "descriptive_only": True,
            "clinical_validation": False,
            "targets": "participant-entered at-home urinary monitor readings",
            "research_intervals_not_clinical_confidence": True,
        },
    }
    public_manifest["runtime"]["report_seconds"] = float(time.perf_counter() - started)
    (results_dir / "run_manifest.json").write_text(
        json.dumps(public_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return public_manifest
