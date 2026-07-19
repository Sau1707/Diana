"""Public-safe aggregate reporting for Hormonbench-mcPHASES v0."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import HORMONES
from .task import config_hash, load_config, project_path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _finite_sort_value(value: Any) -> float:
    try:
        return float(value) if value is not None else float("inf")
    except (TypeError, ValueError):
        return float("inf")


def leaderboard_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten aggregate model summaries into stable leaderboard rows."""

    models = metrics.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ValueError("metrics.json contains no model summaries")
    rows: list[dict[str, Any]] = []
    for model_name, summary in models.items():
        primary = summary["primary"]
        secondary = summary["secondary"]
        skill = summary.get("skill_relative_to_causal_calendar", {})
        improved = summary.get("participants_improved_vs_causal_calendar", {})
        row: dict[str, Any] = {
            "model_name": model_name,
            "model_version": summary["model_version"],
            "overall_normalized_score": summary.get("overall_normalized_score"),
            "test_samples": summary["n_test_samples"],
        }
        for hormone in HORMONES:
            row[f"{hormone}_participant_macro_log1p_mae"] = primary[hormone][
                "participant_macro_log1p_mae"
            ]
            row[f"{hormone}_participant_macro_raw_mae"] = secondary[hormone][
                "participant_macro_raw_mae"
            ]
            row[f"{hormone}_participant_macro_log1p_rmse"] = secondary[hormone][
                "participant_macro_log1p_rmse"
            ]
            row[f"{hormone}_skill_vs_causal_calendar"] = skill.get(hormone)
            row[f"{hormone}_participants_improved"] = improved.get(hormone, {}).get(
                "count"
            )
            row[f"{hormone}_participants_compared"] = improved.get(hormone, {}).get(
                "out_of"
            )
        rows.append(row)
    rows.sort(
        key=lambda row: (
            _finite_sort_value(row["overall_normalized_score"]),
            str(row["model_name"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _write_leaderboard(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank"] + [key for key in rows[0] if key != "rank"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_metadata(config: Mapping[str, Any], model_names: list[str]) -> dict[str, Any]:
    """Read only a strict allowlist of aggregate model-run metadata."""

    checkpoint_dir = project_path(dict(config), "checkpoint_dir")
    allowed = {
        "model_name",
        "requested_model",
        "model_version",
        "backend",
        "seed",
        "quick",
        "parameters",
        "best_iteration",
        "best_iterations",
        "fit_seconds",
        "predict_seconds",
        "runtime_seconds",
        "peak_memory_mb",
        "peak_rss_mb",
        "prediction_rows",
        "train_rows",
        "validation_rows",
        "test_rows",
        "target_space",
        "baseline",
        "preprocessor",
        "per_hormone",
    }
    output: dict[str, Any] = {}
    for name in model_names:
        path = checkpoint_dir / f"{name}.json"
        if not path.is_file():
            output[name] = {"metadata_available": False}
            continue
        raw = _load_json(path)
        safe = {key: raw[key] for key in allowed if key in raw}
        safe["metadata_available"] = True
        output[name] = safe
    return output


def _write_aggregate_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a dependency-free grouped bar chart of aggregate primary metrics."""

    width, height = 900, 520
    left, right, top, bottom = 85, 25, 55, 105
    chart_w, chart_h = width - left - right, height - top - bottom
    values = [
        float(row[f"{h}_participant_macro_log1p_mae"])
        for row in rows
        for h in HORMONES
    ]
    y_max = max(values) * 1.12 if values and max(values) > 0 else 1.0
    colors = {"lh": "#2563eb", "e3g": "#d97706", "pdg": "#059669"}
    group_w = chart_w / max(len(rows), 1)
    bar_w = min(54.0, group_w / 4.2)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="600">Held-out participant-macro log1p MAE</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = top + chart_h - chart_h * tick / 5
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        svg.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.3f}</text>'
        )
    for index, row in enumerate(rows):
        center = left + group_w * (index + 0.5)
        for h_index, hormone in enumerate(HORMONES):
            value = float(row[f"{hormone}_participant_macro_log1p_mae"])
            bar_h = chart_h * value / y_max
            x = center + (h_index - 1) * bar_w - bar_w * 0.42
            y = top + chart_h - bar_h
            svg.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.84:.2f}" height="{bar_h:.2f}" fill="{colors[hormone]}"/>'
            )
        label = html.escape(str(row["model_name"]))
        svg.append(
            f'<text x="{center:.2f}" y="{top + chart_h + 28}" text-anchor="middle" font-family="sans-serif" font-size="13">{label}</text>'
        )
    legend_x = left + chart_w - 225
    for index, hormone in enumerate(HORMONES):
        x = legend_x + index * 75
        svg.extend(
            [
                f'<rect x="{x}" y="45" width="13" height="13" fill="{colors[hormone]}"/>',
                f'<text x="{x + 19}" y="56" font-family="sans-serif" font-size="12">{hormone.upper()}</text>',
            ]
        )
    svg.append(
        f'<text x="20" y="{top + chart_h / 2}" text-anchor="middle" transform="rotate(-90 20 {top + chart_h / 2})" font-family="sans-serif" font-size="13">Lower is better</text>'
    )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_report(config_path: str | Path) -> dict[str, Path]:
    """Generate public leaderboard, manifest, and aggregate-only figure."""

    started = time.perf_counter()
    config = load_config(config_path)
    results_dir = project_path(config, "results_dir")
    metrics_path = results_dir / "metrics.json"
    metrics = _load_json(metrics_path)
    rows = leaderboard_rows(metrics)

    leaderboard_path = results_dir / "leaderboard.csv"
    _write_leaderboard(leaderboard_path, rows)
    figure_path = results_dir / "figures" / "log1p_mae.svg"
    _write_aggregate_svg(figure_path, rows)

    model_names = [str(row["model_name"]) for row in rows]
    prepared_runtime_path = project_path(config, "prepared_dir") / "runtime.json"
    prepared_runtime: dict[str, Any] = {}
    if prepared_runtime_path.is_file():
        raw_prepared_runtime = _load_json(prepared_runtime_path)
        prepared_runtime = {
            key: raw_prepared_runtime[key]
            for key in ("prepare_seconds", "prepare_peak_rss_mb")
            if key in raw_prepared_runtime
        }
    model_metadata = _checkpoint_metadata(config, model_names)
    model_fit_predict_seconds = sum(
        float(metadata.get("fit_seconds", 0.0))
        + float(metadata.get("predict_seconds", 0.0))
        for metadata in model_metadata.values()
    )
    prepare_seconds = float(prepared_runtime.get("prepare_seconds", 0.0))
    evaluate_seconds = float(metrics.get("evaluation_seconds", 0.0))
    report_seconds = float(time.perf_counter() - started)
    manifest = {
        "task_id": metrics.get("task_id"),
        "task_version": metrics.get("task_version"),
        "track": metrics.get("track"),
        "config_hash": config_hash(config),
        "split_seed": int(config["split"]["seed"]),
        "split_counts": metrics["split_counts"],
        "prediction_space": metrics.get("prediction_space", "log1p"),
        "primary_metric": config["evaluation"]["primary_metric"],
        "reference_model": metrics["reference_model"],
        "models": model_metadata,
        "runtime": {
            "prepare": prepared_runtime or {"available": False},
            "model_fit_predict_seconds": model_fit_predict_seconds,
            "evaluate_seconds": evaluate_seconds,
            "report_seconds_before_manifest_write": report_seconds,
            "total_measured_stage_seconds_before_manifest_write": (
                prepare_seconds
                + model_fit_predict_seconds
                + evaluate_seconds
                + report_seconds
            ),
        },
        "aggregate_artifacts": {
            "metrics": "metrics.json",
            "leaderboard": "leaderboard.csv",
            "figure": "figures/log1p_mae.svg",
        },
        "artifact_sha256": {
            "metrics.json": _file_sha256(metrics_path),
            "leaderboard.csv": _file_sha256(leaderboard_path),
            "figures/log1p_mae.svg": _file_sha256(figure_path),
        },
        "privacy": "Aggregate metrics only; no participant IDs, truth rows, or prediction rows.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = results_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "leaderboard": leaderboard_path,
        "metrics": metrics_path,
        "run_manifest": manifest_path,
        "figure": figure_path,
    }
