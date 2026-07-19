"""Independent private-truth evaluator for Hormonbench-mcPHASES v1."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from benchmark.data.v1_adapter import load_private_group_mapping
from benchmark.data.v1_folds import fold_roles
from benchmark.v1_contracts import (
    TARGET_LOG_COLUMNS,
    TARGET_RAW_COLUMNS,
    V1PreparedBundle,
    load_v1_bundle,
    validate_v1_prediction_frame,
)
from benchmark.v1_personalization import build_personalization_plan, scoring_sample_ids
from benchmark.v1_task import (
    HORMONES,
    TASK_ID,
    TASK_VERSION,
    TRACK_COLD,
    TRACK_FEW_SHOT,
    file_sha256,
    project_path,
)


EXPECTED_BASELINES = ("population_median", "wearable_ridge", "catboost")
EXPECTED_CUSTOM = ("diana_h3p",)
EXPECTED_MODELS = EXPECTED_BASELINES + EXPECTED_CUSTOM
CUSTOM_MODEL = EXPECTED_CUSTOM[0]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return float(np.interp(q, positions, values))


def _development_scales(
    bundle: V1PreparedBundle, development_ids: Iterable[int]
) -> dict[str, float | None]:
    rows = bundle.frame.loc[
        bundle.frame["private_participant_id"].astype(int).isin(set(development_ids))
    ]
    counts = rows["private_participant_id"].value_counts()
    weights = rows["private_participant_id"].map(
        {pid: len(rows) / (len(counts) * count) for pid, count in counts.items()}
    ).to_numpy(float)
    scales: dict[str, float | None] = {}
    for hormone in HORMONES:
        values = rows[TARGET_LOG_COLUMNS[hormone]].to_numpy(float)
        scale = _weighted_quantile(values, weights, 0.75) - _weighted_quantile(
            values, weights, 0.25
        )
        scales[hormone] = float(scale) if np.isfinite(scale) and scale > 1e-12 else None
    return scales


def _expected_ids(
    bundle: V1PreparedBundle,
    roles: Mapping[str, set[int]],
    track: str,
) -> list[str]:
    if track == TRACK_COLD:
        rows = bundle.frame.loc[
            bundle.frame["private_participant_id"].astype(int).isin(roles["test"])
        ]
        return rows["sample_id"].astype(str).tolist()
    if track == TRACK_FEW_SHOT:
        return scoring_sample_ids(build_personalization_plan(bundle, roles["test"]))
    raise ValueError(f"Unknown v1 track {track}")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("task_id") != TASK_ID or str(manifest.get("task_version")) != TASK_VERSION:
        raise ValueError("Prediction manifest task mismatch")
    if tuple(manifest.get("baseline_models", [])) != EXPECTED_BASELINES:
        raise ValueError("Active v1 baseline registry must contain exactly three families")
    if tuple(manifest.get("custom_models", [])) != EXPECTED_CUSTOM:
        raise ValueError("v1 requires exactly one separately tagged custom reference")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Prediction manifest entries must be a list")
    expected = {
        (fold, TRACK_COLD, 0, model)
        for fold in range(5)
        for model in EXPECTED_MODELS
    } | {
        (fold, TRACK_FEW_SHOT, budget, model)
        for fold in range(5)
        for budget in (0, 3, 7)
        for model in EXPECTED_MODELS
    }
    actual = {
        (
            int(entry["fold"]),
            str(entry["track"]),
            int(entry["calibration_budget"]),
            str(entry["model_name"]),
        )
        for entry in entries
    }
    if actual != expected or len(entries) != len(expected):
        raise ValueError("Prediction manifest is missing, duplicate, or unexpected")
    filenames = [str(entry["file"]) for entry in entries]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Prediction manifest file paths must be unique")
    if any(Path(name).name != name for name in filenames):
        raise ValueError("Prediction manifest must use local run-directory filenames")


def evaluate_v1(config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    bundle = load_v1_bundle(project_path(config, "prepared_dir"))
    mapping = load_private_group_mapping(config)
    manifest_path = project_path(config, "prediction_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    run_dir = manifest_path.parent
    private_rows: list[pd.DataFrame] = []
    scales_by_fold: dict[int, dict[str, float | None]] = {}
    for fold in range(5):
        roles = fold_roles(mapping, fold)
        scales_by_fold[fold] = _development_scales(
            bundle, roles["train"] | roles["validation"]
        )
    truth = bundle.frame.set_index("sample_id")
    for entry in manifest["entries"]:
        fold = int(entry["fold"])
        track = str(entry["track"])
        budget = int(entry["calibration_budget"])
        model_name = str(entry["model_name"])
        roles = fold_roles(mapping, fold)
        expected_ids = _expected_ids(bundle, roles, track)
        prediction_path = run_dir / str(entry["file"])
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        if file_sha256(prediction_path) != str(entry["sha256"]):
            raise ValueError(f"Prediction hash mismatch: {entry['file']}")
        prediction = validate_v1_prediction_frame(
            pd.read_csv(prediction_path),
            expected_sample_ids=expected_ids,
            expected_track=track,
            expected_fold=fold,
            expected_budget=budget,
        )
        if set(prediction["model_name"]) != {model_name}:
            raise ValueError("Prediction model does not match explicit manifest")
        aligned = prediction.copy()
        aligned["private_participant_id"] = aligned["sample_id"].map(
            truth["private_participant_id"]
        )
        for hormone in HORMONES:
            hormone_mask = aligned["hormone"].eq(hormone)
            ids = aligned.loc[hormone_mask, "sample_id"]
            aligned.loc[hormone_mask, "y_true_log1p"] = ids.map(
                truth[TARGET_LOG_COLUMNS[hormone]]
            ).to_numpy(float)
            aligned.loc[hormone_mask, "y_true_raw"] = ids.map(
                truth[TARGET_RAW_COLUMNS[hormone]]
            ).to_numpy(float)
        aligned["y_pred_raw"] = np.expm1(aligned["y_pred"].to_numpy(float))
        private_rows.append(aligned)
    joined = pd.concat(private_rows, ignore_index=True)
    joined["absolute_log_error"] = np.abs(
        joined["y_true_log1p"] - joined["y_pred"]
    )
    joined["squared_log_error"] = (
        joined["y_true_log1p"] - joined["y_pred"]
    ) ** 2
    joined["absolute_raw_error"] = np.abs(
        joined["y_true_raw"] - joined["y_pred_raw"]
    )
    keys = [
        "track",
        "calibration_budget",
        "model_name",
        "fold",
        "private_participant_id",
        "hormone",
    ]
    participant = (
        joined.groupby(keys, sort=True)
        .agg(
            log1p_mae=("absolute_log_error", "mean"),
            log1p_mse=("squared_log_error", "mean"),
            raw_mae=("absolute_raw_error", "mean"),
            origins=("sample_id", "nunique"),
        )
        .reset_index()
    )
    participant["log1p_rmse"] = np.sqrt(participant["log1p_mse"])
    participant["scale"] = [
        scales_by_fold[int(row.fold)][str(row.hormone)]
        for row in participant.itertuples(index=False)
    ]
    participant["normalized_mae"] = participant["log1p_mae"] / participant["scale"]
    participant_dir = project_path(config, "participant_metrics_dir")
    participant_dir.mkdir(parents=True, exist_ok=True)
    participant.to_csv(participant_dir / "participant_metrics.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    grouped = participant.groupby(
        ["track", "calibration_budget", "model_name"], sort=True
    )
    for (track, budget, model_name), group in grouped:
        record: dict[str, Any] = {
            "track": str(track),
            "calibration_budget": int(budget),
            "model_name": str(model_name),
            "participants": int(group["private_participant_id"].nunique()),
            "origins": int(
                joined.loc[
                    joined["track"].eq(track)
                    & joined["calibration_budget"].eq(budget)
                    & joined["model_name"].eq(model_name),
                    "sample_id",
                ].nunique()
            ),
        }
        for hormone in HORMONES:
            h = group.loc[group["hormone"].eq(hormone)]
            record[f"{hormone}_participant_macro_log1p_mae"] = float(h["log1p_mae"].mean())
            record[f"{hormone}_participant_macro_raw_mae"] = float(h["raw_mae"].mean())
            record[f"{hormone}_participant_macro_log1p_rmse"] = float(h["log1p_rmse"].mean())
        record["overall_normalized_score"] = float(group["normalized_mae"].mean())
        for fold, fold_group in group.groupby("fold", sort=True):
            fold_rows.append(
                {
                    "track": str(track),
                    "calibration_budget": int(budget),
                    "model_name": str(model_name),
                    "fold": int(fold),
                    "participants": int(fold_group["private_participant_id"].nunique()),
                    "overall_normalized_score": float(fold_group["normalized_mae"].mean()),
                    **{
                        f"{h}_participant_macro_log1p_mae": float(
                            fold_group.loc[fold_group["hormone"].eq(h), "log1p_mae"].mean()
                        )
                        for h in HORMONES
                    },
                }
            )
        fold_values = [
            row["overall_normalized_score"]
            for row in fold_rows
            if row["track"] == track
            and row["calibration_budget"] == int(budget)
            and row["model_name"] == model_name
        ]
        record["fold_score_mean"] = float(np.mean(fold_values))
        record["fold_score_sd"] = float(np.std(fold_values, ddof=1))
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    for index, row in summary.iterrows():
        reference = summary.loc[
            summary["track"].eq(row["track"])
            & summary["calibration_budget"].eq(row["calibration_budget"])
            & summary["model_name"].eq("population_median")
        ].iloc[0]
        summary.loc[index, "overall_skill_vs_population_median"] = float(
            1.0 - row["overall_normalized_score"] / reference["overall_normalized_score"]
        )
        for hormone in HORMONES:
            column = f"{hormone}_participant_macro_log1p_mae"
            summary.loc[index, f"{hormone}_skill_vs_population_median"] = float(
                1.0 - row[column] / reference[column]
            )
        current_participant = participant.loc[
            participant["track"].eq(row["track"])
            & participant["calibration_budget"].eq(row["calibration_budget"])
            & participant["model_name"].eq(row["model_name"])
        ]
        reference_participant = participant.loc[
            participant["track"].eq(row["track"])
            & participant["calibration_budget"].eq(row["calibration_budget"])
            & participant["model_name"].eq("population_median")
        ]
        current_score = current_participant.groupby("private_participant_id")[
            "normalized_mae"
        ].mean()
        reference_score = reference_participant.groupby("private_participant_id")[
            "normalized_mae"
        ].mean()
        summary.loc[index, "participants_improved_vs_population_median"] = int(
            (current_score < reference_score).sum()
        )

    interval_rows: list[dict[str, Any]] = []
    interval_data = joined.loc[joined["y_lower"].notna()].copy()
    if not interval_data.empty:
        interval_data["covered"] = (
            interval_data["y_true_log1p"].ge(interval_data["y_lower"])
            & interval_data["y_true_log1p"].le(interval_data["y_upper"])
        ).astype(float)
        interval_data["width"] = interval_data["y_upper"] - interval_data["y_lower"]
        alpha = 0.20
        interval_data["interval_score"] = interval_data["width"]
        interval_data.loc[
            interval_data["y_true_log1p"].lt(interval_data["y_lower"]),
            "interval_score",
        ] += (2 / alpha) * (
            interval_data["y_lower"] - interval_data["y_true_log1p"]
        )
        interval_data.loc[
            interval_data["y_true_log1p"].gt(interval_data["y_upper"]),
            "interval_score",
        ] += (2 / alpha) * (
            interval_data["y_true_log1p"] - interval_data["y_upper"]
        )
        per_participant_interval = (
            interval_data.groupby(
                [
                    "track",
                    "calibration_budget",
                    "model_name",
                    "private_participant_id",
                    "hormone",
                ]
            )
            .agg(
                coverage=("covered", "mean"),
                mean_width=("width", "mean"),
                interval_score=("interval_score", "mean"),
            )
            .reset_index()
        )
        for keys_value, group in per_participant_interval.groupby(
            ["track", "calibration_budget", "model_name", "hormone"], sort=True
        ):
            track, budget, model_name, hormone = keys_value
            interval_rows.append(
                {
                    "track": str(track),
                    "calibration_budget": int(budget),
                    "model_name": str(model_name),
                    "hormone": str(hormone),
                    "participant_macro_coverage_80": float(group["coverage"].mean()),
                    "participant_macro_mean_width": float(group["mean_width"].mean()),
                    "participant_macro_interval_score_80": float(
                        group["interval_score"].mean()
                    ),
                }
            )
    decomposition_rows: list[dict[str, Any]] = []
    join_key = ["sample_id", "hormone", "fold", "private_participant_id"]
    for track in (TRACK_COLD, TRACK_FEW_SHOT):
        budgets = (0,) if track == TRACK_COLD else (0, 3, 7)
        custom_zero = joined.loc[
            joined["track"].eq(track)
            & joined["calibration_budget"].eq(0)
            & joined["model_name"].eq(CUSTOM_MODEL),
            join_key + ["y_pred"],
        ].rename(columns={"y_pred": "custom_k0"})
        population_zero = joined.loc[
            joined["track"].eq(track)
            & joined["calibration_budget"].eq(0)
            & joined["model_name"].eq("population_median"),
            join_key + ["y_pred"],
        ].rename(columns={"y_pred": "population_k0"})
        prior = custom_zero.merge(population_zero, on=join_key, validate="one_to_one")
        prior["wearable_abs"] = np.abs(prior["custom_k0"] - prior["population_k0"])
        for budget in budgets:
            current = joined.loc[
                joined["track"].eq(track)
                & joined["calibration_budget"].eq(budget)
                & joined["model_name"].eq(CUSTOM_MODEL),
                join_key + ["y_pred"],
            ].rename(columns={"y_pred": "custom_current"})
            work = current.merge(custom_zero, on=join_key, validate="one_to_one")
            work = work.merge(
                prior[join_key + ["wearable_abs"]], on=join_key, validate="one_to_one"
            )
            work["personal_abs"] = np.abs(work["custom_current"] - work["custom_k0"])
            for hormone in HORMONES:
                h = work.loc[work["hormone"].eq(hormone)]
                participant_terms = h.groupby("private_participant_id").agg(
                    wearable_abs=("wearable_abs", "mean"),
                    personal_abs=("personal_abs", "mean"),
                )
                decomposition_rows.append(
                    {
                        "track": track,
                        "calibration_budget": int(budget),
                        "hormone": hormone,
                        "population_prior": "development-only participant-equal median (value private)",
                        "participant_macro_mean_abs_wearable_adjustment": float(
                            participant_terms["wearable_abs"].mean()
                        ),
                        "participant_macro_mean_abs_personal_adjustment": float(
                            participant_terms["personal_abs"].mean()
                        ),
                    }
                )
    output = {
        "schema_version": "1.0.0",
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "reference_model": "population_median",
        "descriptive_only": True,
        "rows": summary.sort_values(
            ["track", "calibration_budget", "overall_normalized_score"]
        ).to_dict(orient="records"),
        "fold_rows": sorted(
            fold_rows,
            key=lambda row: (
                row["track"],
                row["calibration_budget"],
                row["model_name"],
                row["fold"],
            ),
        ),
        "uncertainty_rows": interval_rows,
        "custom_decomposition_rows": decomposition_rows,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    results_dir = project_path(config, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
