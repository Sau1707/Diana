"""Private v1 experiment pipeline; benchmark evaluation remains model-independent."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import psutil

from benchmark.data.v1_adapter import load_private_group_mapping
from benchmark.data.v1_folds import fold_roles
from benchmark.v1_contracts import (
    TARGET_LOG_COLUMNS,
    V1PreparedBundle,
    load_v1_bundle,
    validate_v1_prediction_frame,
)
from benchmark.v1_personalization import (
    PersonalizationPlan,
    build_personalization_plan,
    scoring_sample_ids,
)
from benchmark.v1_task import (
    HORMONES,
    TASK_ID,
    TASK_VERSION,
    TRACK_COLD,
    TRACK_FEW_SHOT,
    config_hash,
    file_sha256,
    project_path,
    task_spec_hash,
)
from model.catboost.v1_model import CatBoostV1
from model.joint_bayes_personalizer.model import (
    JointBayesPersonalizer,
    add_prior_columns,
    estimate_custom_parameters,
    learn_conformal_multipliers,
)
from model.personalization import (
    apply_independent_offsets,
    estimate_diagonal_adapter,
    independent_offsets,
)
from model.population_median.v1_model import PopulationMedianV1
from model.v1_common import combine_fit_views, participant_balanced_iqr_scales
from model.wearable_ridge.model import WearableRidgeV1


BASELINE_NAMES = ("population_median", "wearable_ridge", "catboost")
CUSTOM_NAME = "joint_bayes_personalizer"


def choose_covariance_candidate(
    candidates: Mapping[str, Mapping[str, float]],
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> str:
    """Choose passing candidate, or strongest valid candidate if no gate passes."""

    modes = ("diagonal", "full")
    if set(candidates) != set(modes) or set(diagnostics) != set(modes):
        raise ValueError("Covariance selection requires diagonal and full candidates")
    passing = [mode for mode in modes if diagnostics[mode]["success_gate_passed"]]
    eligible = passing or list(modes)
    return min(
        eligible,
        key=lambda mode: (float(candidates[mode]["score"]), mode == "full"),
    )


class _PeakRSS:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PeakRSS":
        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self._stop.wait(0.05):
                self.peak = max(self.peak, process.memory_info().rss)

        self.peak = process.memory_info().rss
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _mask(bundle: V1PreparedBundle, participant_ids: Iterable[int]) -> pd.Series:
    return bundle.frame["private_participant_id"].astype(int).isin(set(participant_ids))


def _wide_rows(
    bundle: V1PreparedBundle,
    mask: pd.Series,
    predictions: Mapping[str, np.ndarray],
    *,
    include_truth: bool = False,
) -> pd.DataFrame:
    selected = ["sample_id", "private_participant_id", "origin_day", "target_day"]
    if include_truth:
        selected += list(TARGET_LOG_COLUMNS.values())
    rows = bundle.frame.loc[
        np.asarray(mask, dtype=bool),
        selected,
    ].reset_index(drop=True)
    out = rows.rename(columns={TARGET_LOG_COLUMNS[h]: f"y_{h}" for h in HORMONES})
    for hormone in HORMONES:
        values = np.asarray(predictions[hormone], dtype=float)
        if len(values) != len(out):
            raise ValueError("Prediction length does not match private alignment")
        out[f"pred_{hormone}"] = np.maximum(values, 0.0)
    return out


def _fit_base(
    name: str,
    config: dict[str, Any],
    fit_view: Any,
    *,
    tree_counts: Mapping[str, int] | None = None,
) -> Any:
    if name == "population_median":
        return PopulationMedianV1(config).fit(fit_view)
    if name == "wearable_ridge":
        return WearableRidgeV1(config).fit(fit_view)
    if name == "catboost":
        model = CatBoostV1(config)
        if tree_counts is None:
            raise ValueError("Canonical CatBoost requires validation-selected tree counts")
        return model.fit_fixed(fit_view, tree_counts)
    raise KeyError(name)


def _oof_predictions(
    bundle: V1PreparedBundle,
    mapping: Mapping[int, int],
    participant_ids: set[int],
    config: dict[str, Any],
    model_name: str,
    *,
    tree_counts: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    groups = sorted({int(mapping[pid]) for pid in participant_ids})
    for held_group in groups:
        held_ids = {pid for pid in participant_ids if int(mapping[pid]) == held_group}
        fit_ids = participant_ids - held_ids
        fit_view = bundle.fit_view(_mask(bundle, fit_ids))
        inference = bundle.inference_view(_mask(bundle, held_ids))
        estimator = _fit_base(
            model_name, config, fit_view, tree_counts=tree_counts
        )
        predictions = estimator.predict(inference)
        parts.append(
            _wide_rows(
                bundle,
                _mask(bundle, held_ids),
                predictions,
                include_truth=True,
            )
        )
    output = pd.concat(parts, ignore_index=True)
    if output["private_participant_id"].astype(int).nunique() != len(participant_ids):
        raise RuntimeError("Grouped OOF did not cover every development participant")
    return output.sort_values(
        ["private_participant_id", "origin_day", "sample_id"]
    ).reset_index(drop=True)


def _plan_calibration(
    bundle: V1PreparedBundle,
    plan: PersonalizationPlan,
    base_predictions: pd.DataFrame,
    budget: int,
) -> pd.DataFrame:
    candidates = plan.calibration_candidates.loc[
        plan.calibration_candidates["calibration_rank"].le(int(budget))
    ].copy()
    if int(budget) == 0:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "private_participant_id",
                "target_day",
                *[f"y_{h}" for h in HORMONES],
                *[f"pred_{h}" for h in HORMONES],
            ]
        )
    truth = bundle.frame.set_index("sample_id")
    base = base_predictions.set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for row in candidates.sort_values(
        ["private_participant_id", "calibration_rank"]
    ).itertuples(index=False):
        sample_id = str(row.sample_id)
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "private_participant_id": str(row.private_participant_id),
            "target_day": int(row.target_day),
        }
        for hormone in HORMONES:
            record[f"y_{hormone}"] = float(
                truth.loc[sample_id, TARGET_LOG_COLUMNS[hormone]]
            )
            record[f"pred_{hormone}"] = float(base.loc[sample_id, f"pred_{hormone}"])
        rows.append(record)
    result = pd.DataFrame(rows)
    counts = result.groupby("private_participant_id").size()
    if counts.empty or not counts.eq(int(budget)).all():
        raise RuntimeError("Authorized calibration does not have exact K cardinality")
    return result


def _empty_calibration() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id",
            "private_participant_id",
            "target_day",
            *[f"y_{h}" for h in HORMONES],
            *[f"pred_{h}" for h in HORMONES],
        ]
    )


def _adapt_classical(
    base_all: pd.DataFrame,
    scoring_ids: list[str],
    calibration: pd.DataFrame,
    adapter_parameters: Any,
    *,
    budget: int,
) -> pd.DataFrame:
    scored = base_all.set_index("sample_id").loc[scoring_ids].reset_index()
    participants = sorted(scored["private_participant_id"].astype(str).unique())
    if int(budget) == 0:
        offsets = {
            participant: {hormone: 0.0 for hormone in HORMONES}
            for participant in participants
        }
    else:
        residuals = calibration[
            ["sample_id", "private_participant_id", "target_day"]
        ].copy()
        for hormone in HORMONES:
            residuals[f"residual_{hormone}"] = (
                calibration[f"y_{hormone}"].to_numpy(float)
                - calibration[f"pred_{hormone}"].to_numpy(float)
            )
        offsets = independent_offsets(
            residuals, adapter_parameters, budget=int(budget)
        )
    return apply_independent_offsets(scored, offsets)


def _custom_predictions(
    parameters: Any,
    base_all: pd.DataFrame,
    scoring_ids: list[str],
    calibration: pd.DataFrame,
    *,
    budget: int,
    multipliers: Mapping[str, float] | None,
) -> pd.DataFrame:
    base = base_all.set_index("sample_id").loc[scoring_ids].reset_index()
    authorized = add_prior_columns(
        calibration, parameters.medians, parameters.lambdas
    )
    predicted, _ = JointBayesPersonalizer(parameters).predict(
        base,
        authorized,
        budget=int(budget),
        interval_multipliers=multipliers,
    )
    return predicted


def _to_submission(
    wide: pd.DataFrame,
    *,
    model_name: str,
    fold: int,
    track: str,
    budget: int,
    include_intervals: bool = False,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in wide.itertuples(index=False):
        for hormone in HORMONES:
            record: dict[str, Any] = {
                "task_id": TASK_ID,
                "task_version": TASK_VERSION,
                "track": track,
                "fold": int(fold),
                "calibration_budget": int(budget),
                "split": "test",
                "sample_id": str(row.sample_id),
                "hormone": hormone,
                "horizon": 1,
                "y_pred": float(getattr(row, f"pred_{hormone}")),
                "model_name": model_name,
                "model_version": "1.0.0",
            }
            if include_intervals:
                record["y_lower"] = float(getattr(row, f"lower_{hormone}"))
                record["y_upper"] = float(getattr(row, f"upper_{hormone}"))
            records.append(record)
    return pd.DataFrame(records)


def _participant_metric(
    wide: pd.DataFrame, scales: Mapping[str, float | None]
) -> tuple[float, dict[str, float], dict[str, float]]:
    hormone_errors: dict[str, float] = {}
    participant_scores: dict[str, float] = {}
    for hormone in HORMONES:
        errors = np.abs(
            wide[f"y_{hormone}"].to_numpy(float)
            - wide[f"pred_{hormone}"].to_numpy(float)
        )
        work = pd.DataFrame(
            {
                "participant": wide["private_participant_id"].astype(str),
                "error": errors,
            }
        )
        per_participant = work.groupby("participant")["error"].mean()
        hormone_errors[hormone] = float(per_participant.mean())
        for participant, error in per_participant.items():
            scale = scales[hormone]
            if scale is None:
                continue
            participant_scores.setdefault(str(participant), 0.0)
            participant_scores[str(participant)] += float(error / scale) / len(HORMONES)
    normalized = [
        hormone_errors[h] / float(scales[h])
        for h in HORMONES
        if scales[h] is not None
    ]
    return float(np.mean(normalized)), hormone_errors, participant_scores


def _attach_truth(bundle: V1PreparedBundle, predictions: pd.DataFrame) -> pd.DataFrame:
    """Private selection-only truth join; never used in official test inference."""

    output = predictions.copy()
    truth = bundle.frame.set_index("sample_id")
    for hormone in HORMONES:
        output[f"y_{hormone}"] = output["sample_id"].map(
            truth[TARGET_LOG_COLUMNS[hormone]]
        ).to_numpy(float)
    return output


def run_fold0_validation(
    config: dict[str, Any], bundle: V1PreparedBundle | None = None
) -> dict[str, Any]:
    """Run only prespecified fold-0 development selection; no outer-test rows."""

    if bundle is None:
        bundle = load_v1_bundle(project_path(config, "prepared_dir"))
    mapping = load_private_group_mapping(config)
    roles = fold_roles(mapping, 0)
    train_view = bundle.fit_view(_mask(bundle, roles["train"]))
    validation_view = bundle.fit_view(_mask(bundle, roles["validation"]))
    selector = CatBoostV1(config)
    tree_counts = selector.select_tree_counts(train_view, validation_view)
    scales = participant_balanced_iqr_scales(train_view)
    base_models: dict[str, Any] = {
        name: _fit_base(
            name,
            config,
            train_view,
            tree_counts=tree_counts if name == "catboost" else None,
        )
        for name in BASELINE_NAMES
    }
    validation_mask = _mask(bundle, roles["validation"])
    base_wide = {
        name: _wide_rows(
            bundle,
            validation_mask,
            model.predict(bundle.inference_view(validation_mask)),
        )
        for name, model in base_models.items()
    }
    oof = {
        name: _oof_predictions(
            bundle,
            mapping,
            roles["train"],
            config,
            name,
            tree_counts=tree_counts if name == "catboost" else None,
        )
        for name in BASELINE_NAMES
    }
    adapters = {
        name: estimate_diagonal_adapter(oof[name]) for name in BASELINE_NAMES
    }
    plan = build_personalization_plan(bundle, roles["validation"])
    scoring_ids = scoring_sample_ids(plan)
    classical_scores: dict[str, Any] = {}
    for name in BASELINE_NAMES:
        scores: list[float] = []
        hormone_accumulator: dict[str, list[float]] = {h: [] for h in HORMONES}
        participant_accumulator: dict[str, list[float]] = {}
        for budget in (3, 7):
            calibration = _plan_calibration(bundle, plan, base_wide[name], budget)
            prediction = _adapt_classical(
                base_wide[name],
                scoring_ids,
                calibration,
                adapters[name],
                budget=budget,
            )
            score, hormone_scores, participant_scores = _participant_metric(
                _attach_truth(bundle, prediction), scales
            )
            scores.append(score)
            for hormone, value in hormone_scores.items():
                hormone_accumulator[hormone].append(value)
            for participant, value in participant_scores.items():
                participant_accumulator.setdefault(participant, []).append(value)
        classical_scores[name] = {
            "score": float(np.mean(scores)),
            "hormone_mae": {
                h: float(np.mean(values)) for h, values in hormone_accumulator.items()
            },
            "participant_scores": {
                p: float(np.mean(values)) for p, values in participant_accumulator.items()
            },
        }
    strongest_name = min(
        BASELINE_NAMES, key=lambda name: classical_scores[name]["score"]
    )
    candidates: dict[str, Any] = {}
    for mode in ("diagonal", "full"):
        parameters = estimate_custom_parameters(
            oof["catboost"],
            mode=mode,
            grid_step=float(config["custom"]["lambda_grid_step"]),
            shrinkage=float(config["custom"]["covariance_shrinkage"]),
            floor=float(config["custom"]["eigenvalue_floor"]),
        )
        scores = []
        hormone_accumulator = {h: [] for h in HORMONES}
        participant_accumulator: dict[str, list[float]] = {}
        for budget in (3, 7):
            calibration = _plan_calibration(
                bundle, plan, base_wide["catboost"], budget
            )
            prediction = _custom_predictions(
                parameters,
                base_wide["catboost"],
                scoring_ids,
                calibration,
                budget=budget,
                multipliers=None,
            )
            score, hormone_scores, participant_scores = _participant_metric(
                _attach_truth(bundle, prediction), scales
            )
            scores.append(score)
            for hormone, value in hormone_scores.items():
                hormone_accumulator[hormone].append(value)
            for participant, value in participant_scores.items():
                participant_accumulator.setdefault(participant, []).append(value)
        candidates[mode] = {
            "score": float(np.mean(scores)),
            "hormone_mae": {
                h: float(np.mean(values)) for h, values in hormone_accumulator.items()
            },
            "participant_scores": {
                p: float(np.mean(values)) for p, values in participant_accumulator.items()
            },
            "lambdas": dict(parameters.lambdas),
        }
    reference = classical_scores[strongest_name]
    candidate_diagnostics: dict[str, dict[str, Any]] = {}
    for mode, candidate in candidates.items():
        relative_gain = 1.0 - candidate["score"] / reference["score"]
        hormones_improved = sum(
            candidate["hormone_mae"][h] < reference["hormone_mae"][h]
            for h in HORMONES
        )
        participants_improved = sum(
            candidate["participant_scores"][p] < reference["participant_scores"][p]
            for p in reference["participant_scores"]
        )
        maximum_regression = max(
            candidate["hormone_mae"][h] / reference["hormone_mae"][h] - 1.0
            for h in HORMONES
        )
        passes = bool(
            relative_gain
            >= float(
                config["custom"]["validation_selection"]["minimum_relative_gain"]
            )
            and hormones_improved
            >= int(
                config["custom"]["validation_selection"][
                    "minimum_hormones_improved"
                ]
            )
            and participants_improved
            >= int(
                config["custom"]["validation_selection"][
                    "minimum_participants_improved"
                ]
            )
            and maximum_regression
            <= float(
                config["custom"]["validation_selection"][
                    "maximum_hormone_regression"
                ]
            )
        )
        candidate_diagnostics[mode] = {
            "relative_gain": float(relative_gain),
            "hormones_improved": int(hormones_improved),
            "participants_improved": int(participants_improved),
            "maximum_hormone_regression": float(maximum_regression),
            "success_gate_passed": passes,
        }
    selected = choose_covariance_candidate(candidates, candidate_diagnostics)
    selected_passes = bool(candidate_diagnostics[selected]["success_gate_passed"])
    result = {
        "selection_scope": "fold_0_validation_only",
        "selected_covariance_mode": selected,
        "success_gate_passed": selected_passes,
        "strongest_personalized_classical": strongest_name,
        "classical": classical_scores,
        "candidates": candidates,
        "candidate_diagnostics": candidate_diagnostics,
        "catboost_selection": selector.get_metadata(),
        "scoring_origins": int(plan.aggregate["common_scoring_origins"]),
    }
    output = project_path(config, "validation_dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "fold0_selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _write_prediction(
    run_dir: Path,
    frame: pd.DataFrame,
    *,
    model_name: str,
    track: str,
    fold: int,
    budget: int,
) -> dict[str, Any]:
    expected_ids = frame["sample_id"].astype(str).unique().tolist()
    validate_v1_prediction_frame(
        frame,
        expected_sample_ids=expected_ids,
        expected_track=track,
        expected_fold=fold,
        expected_budget=budget,
    )
    filename = f"fold{fold}__{track}__k{budget}__{model_name}.csv"
    path = run_dir / filename
    frame.to_csv(path, index=False)
    return {
        "file": filename,
        "sha256": file_sha256(path),
        "rows": int(len(frame)),
        "samples": int(frame["sample_id"].nunique()),
        "model_name": model_name,
        "track": track,
        "fold": int(fold),
        "calibration_budget": int(budget),
    }


def run_official_five_folds(
    config: dict[str, Any], bundle: V1PreparedBundle | None = None
) -> dict[str, Any]:
    """Run the frozen five-fold protocol once and write private predictions."""

    started = time.perf_counter()
    if bundle is None:
        bundle = load_v1_bundle(project_path(config, "prepared_dir"))
    if bundle.metadata["config_hash"] != config_hash(config):
        raise RuntimeError("Prepared bundle was not generated from the frozen config")
    if bundle.metadata["task_spec_hash"] != task_spec_hash(config):
        raise RuntimeError("Prepared bundle scientific task hash mismatch")
    mode = str(config["custom"]["selected_covariance_mode"])
    if mode not in {"diagonal", "full"}:
        raise RuntimeError("Freeze selected_covariance_mode before official evaluation")
    mapping = load_private_group_mapping(config)
    run_dir = project_path(config, "prediction_run_dir")
    checkpoint_dir = project_path(config, "checkpoint_dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project_path(config, "prediction_manifest")
    if manifest_path.exists():
        raise FileExistsError(
            "Canonical prediction manifest already exists; refusing an implicit rerun"
        )
    selection_path = project_path(config, "selection_artifact")
    if not selection_path.is_file():
        raise FileNotFoundError("Fold-0 validation selection must precede official folds")
    expected_selection_hash = str(config["custom"]["selection_artifact_sha256"])
    if file_sha256(selection_path) != expected_selection_hash:
        raise RuntimeError("Frozen fold-0 selection artifact hash mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if str(selection["selected_covariance_mode"]) != mode:
        raise RuntimeError("Frozen covariance mode disagrees with validation selection")
    freeze = {
        "config_hash": config_hash(config),
        "task_spec_hash": task_spec_hash(config),
        "fold_hash": bundle.metadata["fold_hash"],
        "input_schema_hash": bundle.metadata["input_schema_hash"],
        "selection_artifact_sha256": expected_selection_hash,
        "selected_covariance_mode": mode,
        "seed": int(config["folds"]["seed"]),
        "outer_test_metrics_seen": False,
    }
    freeze_path = project_path(config, "validation_dir") / "frozen_protocol.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    entries: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    with _PeakRSS() as memory:
      for fold in range(5):
        fold_started = time.perf_counter()
        baseline_started = time.perf_counter()
        roles = fold_roles(mapping, fold)
        train_view = bundle.fit_view(_mask(bundle, roles["train"]))
        validation_view = bundle.fit_view(_mask(bundle, roles["validation"]))
        dev_view = combine_fit_views(train_view, validation_view)
        test_mask = _mask(bundle, roles["test"])
        test_view = bundle.inference_view(test_mask)
        selector = CatBoostV1(config)
        tree_counts = selector.select_tree_counts(train_view, validation_view)
        base_models = {
            name: _fit_base(
                name,
                config,
                dev_view,
                tree_counts=tree_counts if name == "catboost" else None,
            )
            for name in BASELINE_NAMES
        }
        base_test = {
            name: _wide_rows(
                bundle, test_mask, model.predict(test_view)
            )
            for name, model in base_models.items()
        }
        development_ids = roles["train"] | roles["validation"]
        oof = {
            name: _oof_predictions(
                bundle,
                mapping,
                development_ids,
                config,
                name,
                tree_counts=tree_counts if name == "catboost" else None,
            )
            for name in BASELINE_NAMES
        }
        adapters = {
            name: estimate_diagonal_adapter(oof[name]) for name in BASELINE_NAMES
        }
        baseline_fit_seconds = time.perf_counter() - baseline_started
        custom_started = time.perf_counter()
        custom_parameters = estimate_custom_parameters(
            oof["catboost"],
            mode=mode,
            grid_step=float(config["custom"]["lambda_grid_step"]),
            shrinkage=float(config["custom"]["covariance_shrinkage"]),
            floor=float(config["custom"]["eigenvalue_floor"]),
        )
        multipliers = {
            budget: learn_conformal_multipliers(
                oof["catboost"],
                budget=budget,
                mode=mode,
                grid_step=float(config["custom"]["lambda_grid_step"]),
                shrinkage=float(config["custom"]["covariance_shrinkage"]),
                floor=float(config["custom"]["eigenvalue_floor"]),
                quantile=float(config["custom"]["conformal_quantile"]),
            )
            for budget in (0, 3, 7)
        }
        custom_fit_seconds = time.perf_counter() - custom_started
        prediction_started = time.perf_counter()
        all_test_ids = base_test["catboost"]["sample_id"].astype(str).tolist()
        for name in BASELINE_NAMES:
            submission = _to_submission(
                base_test[name],
                model_name=name,
                fold=fold,
                track=TRACK_COLD,
                budget=0,
            )
            entries.append(
                _write_prediction(
                    run_dir,
                    submission,
                    model_name=name,
                    track=TRACK_COLD,
                    fold=fold,
                    budget=0,
                )
            )
        custom_cold = _custom_predictions(
            custom_parameters,
            base_test["catboost"],
            all_test_ids,
            _empty_calibration(),
            budget=0,
            multipliers=multipliers[0],
        )
        submission = _to_submission(
            custom_cold,
            model_name=CUSTOM_NAME,
            fold=fold,
            track=TRACK_COLD,
            budget=0,
            include_intervals=True,
        )
        entries.append(
            _write_prediction(
                run_dir,
                submission,
                model_name=CUSTOM_NAME,
                track=TRACK_COLD,
                fold=fold,
                budget=0,
            )
        )
        plan = build_personalization_plan(bundle, roles["test"])
        scoring_ids = scoring_sample_ids(plan)
        for budget in (0, 3, 7):
            for name in BASELINE_NAMES:
                calibration = _plan_calibration(
                    bundle, plan, base_test[name], budget
                )
                adapted = _adapt_classical(
                    base_test[name],
                    scoring_ids,
                    calibration,
                    adapters[name],
                    budget=budget,
                )
                submission = _to_submission(
                    adapted,
                    model_name=name,
                    fold=fold,
                    track=TRACK_FEW_SHOT,
                    budget=budget,
                )
                entries.append(
                    _write_prediction(
                        run_dir,
                        submission,
                        model_name=name,
                        track=TRACK_FEW_SHOT,
                        fold=fold,
                        budget=budget,
                    )
                )
            calibration = _plan_calibration(
                bundle, plan, base_test["catboost"], budget
            )
            custom = _custom_predictions(
                custom_parameters,
                base_test["catboost"],
                scoring_ids,
                calibration,
                budget=budget,
                multipliers=multipliers[budget],
            )
            submission = _to_submission(
                custom,
                model_name=CUSTOM_NAME,
                fold=fold,
                track=TRACK_FEW_SHOT,
                budget=budget,
                include_intervals=True,
            )
            entries.append(
                _write_prediction(
                    run_dir,
                    submission,
                    model_name=CUSTOM_NAME,
                    track=TRACK_FEW_SHOT,
                    fold=fold,
                    budget=budget,
                )
            )
        metadata = {
            "fold": fold,
            "train_participants": 12,
            "validation_participants": 4,
            "development_participants": 16,
            "test_participants": 4,
            "test_origins": int(test_mask.sum()),
            "common_suffix_origins": int(plan.aggregate["common_scoring_origins"]),
            "catboost": selector.get_metadata(),
            "final_preprocessors": {
                name: base_models[name].get_metadata().get("preprocessor")
                for name in ("wearable_ridge", "catboost")
            },
            "custom": JointBayesPersonalizer(custom_parameters).get_metadata(),
            "conformal_multipliers": multipliers,
            "development_scales": participant_balanced_iqr_scales(dev_view),
            "runtime_seconds": float(time.perf_counter() - fold_started),
            "baseline_fit_oof_seconds": float(baseline_fit_seconds),
            "custom_fit_calibration_seconds": float(custom_fit_seconds),
            "prediction_write_seconds": float(time.perf_counter() - prediction_started),
        }
        fold_metadata.append(metadata)
        (checkpoint_dir / f"fold_{fold}_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"completed official fold {fold}/4 in {metadata['runtime_seconds']:.2f}s",
            flush=True,
        )
    manifest = {
        "schema_version": "1.0.0",
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "run_id": config["runtime"]["canonical_run_id"],
        "entries": entries,
        "folds": 5,
        "models": [*BASELINE_NAMES, CUSTOM_NAME],
        "baseline_models": list(BASELINE_NAMES),
        "custom_models": [CUSTOM_NAME],
        "selected_covariance_mode": mode,
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_mb": float(memory.peak / 1024**2),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (checkpoint_dir / "run_metadata.json").write_text(
        json.dumps(
            {"manifest": manifest, "fold_metadata": fold_metadata},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def run_corrected_full_custom(
    config: dict[str, Any], bundle: V1PreparedBundle | None = None
) -> dict[str, Any]:
    """Reuse valid baseline outputs and recompute only the corrected full custom path."""

    started = time.perf_counter()
    if bundle is None:
        bundle = load_v1_bundle(project_path(config, "prepared_dir"))
    if bundle.metadata["config_hash"] != config_hash(config):
        raise RuntimeError("Corrected custom run requires a freshly frozen bundle")
    if str(config["custom"]["selected_covariance_mode"]) != "full":
        raise RuntimeError("Corrected custom run is frozen to full covariance")
    selection_path = project_path(config, "selection_artifact")
    if file_sha256(selection_path) != str(
        config["custom"]["selection_artifact_sha256"]
    ):
        raise RuntimeError("Corrected selection artifact hash mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selected_covariance_mode"] != "full" or selection[
        "outer_test_metrics_used"
    ]:
        raise RuntimeError("Corrected selection must be validation-only full covariance")

    source_manifest_path = project_path(config, "invalidated_prediction_manifest")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("selected_covariance_mode") != "diagonal":
        raise RuntimeError("Expected the preserved diagonal source run")
    source_run_dir = project_path(config, "invalidated_prediction_run_dir")
    source_checkpoint_dir = project_path(config, "invalidated_checkpoint_dir")
    target_run_dir = project_path(config, "prediction_run_dir")
    target_checkpoint_dir = project_path(config, "checkpoint_dir")
    target_manifest_path = project_path(config, "prediction_manifest")
    target_run_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if target_manifest_path.exists():
        raise FileExistsError("Corrected canonical manifest already exists")

    entries: list[dict[str, Any]] = []
    source_entries: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    for entry in source_manifest["entries"]:
        key = (
            int(entry["fold"]),
            str(entry["track"]),
            int(entry["calibration_budget"]),
            str(entry["model_name"]),
        )
        source_entries[key] = entry
        if str(entry["model_name"]) not in BASELINE_NAMES:
            continue
        source = source_run_dir / str(entry["file"])
        if file_sha256(source) != str(entry["sha256"]):
            raise RuntimeError("Preserved baseline prediction hash mismatch")
        destination = target_run_dir / str(entry["file"])
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(source, destination)
        copied = dict(entry)
        copied["sha256"] = file_sha256(destination)
        copied["reused_from_invalidated_custom_run"] = True
        entries.append(copied)

    mapping = load_private_group_mapping(config)
    fold_metadata: list[dict[str, Any]] = []
    correction_seconds = 0.0
    with _PeakRSS() as memory:
        for fold in range(5):
            fold_started = time.perf_counter()
            roles = fold_roles(mapping, fold)
            source_fold = json.loads(
                (source_checkpoint_dir / f"fold_{fold}_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            tree_counts = {
                hormone: int(source_fold["catboost"]["tree_count"][hormone])
                for hormone in HORMONES
            }
            development_ids = roles["train"] | roles["validation"]
            custom_started = time.perf_counter()
            cat_oof = _oof_predictions(
                bundle,
                mapping,
                development_ids,
                config,
                "catboost",
                tree_counts=tree_counts,
            )
            parameters = estimate_custom_parameters(
                cat_oof,
                mode="full",
                grid_step=float(config["custom"]["lambda_grid_step"]),
                shrinkage=float(config["custom"]["covariance_shrinkage"]),
                floor=float(config["custom"]["eigenvalue_floor"]),
            )
            multipliers = {
                budget: learn_conformal_multipliers(
                    cat_oof,
                    budget=budget,
                    mode="full",
                    grid_step=float(config["custom"]["lambda_grid_step"]),
                    shrinkage=float(config["custom"]["covariance_shrinkage"]),
                    floor=float(config["custom"]["eigenvalue_floor"]),
                    quantile=float(config["custom"]["conformal_quantile"]),
                )
                for budget in (0, 3, 7)
            }
            custom_fit_seconds = time.perf_counter() - custom_started

            test_mask = _mask(bundle, roles["test"])
            test_rows = bundle.frame.loc[
                test_mask,
                ["sample_id", "private_participant_id", "origin_day", "target_day"],
            ].reset_index(drop=True)
            source_cat_entry = source_entries[(fold, TRACK_COLD, 0, "catboost")]
            source_cat = pd.read_csv(source_run_dir / source_cat_entry["file"])
            validate_v1_prediction_frame(
                source_cat,
                expected_sample_ids=test_rows["sample_id"].astype(str),
                expected_track=TRACK_COLD,
                expected_fold=fold,
                expected_budget=0,
            )
            base_test = test_rows.copy()
            for hormone in HORMONES:
                values = source_cat.loc[source_cat["hormone"].eq(hormone)].set_index(
                    "sample_id"
                )["y_pred"]
                base_test[f"pred_{hormone}"] = base_test["sample_id"].map(values).to_numpy(
                    float
                )
            prediction_started = time.perf_counter()
            all_ids = base_test["sample_id"].astype(str).tolist()
            cold = _custom_predictions(
                parameters,
                base_test,
                all_ids,
                _empty_calibration(),
                budget=0,
                multipliers=multipliers[0],
            )
            submission = _to_submission(
                cold,
                model_name=CUSTOM_NAME,
                fold=fold,
                track=TRACK_COLD,
                budget=0,
                include_intervals=True,
            )
            entries.append(
                _write_prediction(
                    target_run_dir,
                    submission,
                    model_name=CUSTOM_NAME,
                    track=TRACK_COLD,
                    fold=fold,
                    budget=0,
                )
            )
            plan = build_personalization_plan(bundle, roles["test"])
            scoring_ids = scoring_sample_ids(plan)
            for budget in (0, 3, 7):
                calibration = _plan_calibration(bundle, plan, base_test, budget)
                custom = _custom_predictions(
                    parameters,
                    base_test,
                    scoring_ids,
                    calibration,
                    budget=budget,
                    multipliers=multipliers[budget],
                )
                submission = _to_submission(
                    custom,
                    model_name=CUSTOM_NAME,
                    fold=fold,
                    track=TRACK_FEW_SHOT,
                    budget=budget,
                    include_intervals=True,
                )
                entries.append(
                    _write_prediction(
                        target_run_dir,
                        submission,
                        model_name=CUSTOM_NAME,
                        track=TRACK_FEW_SHOT,
                        fold=fold,
                        budget=budget,
                    )
                )
            correction_fold_seconds = time.perf_counter() - fold_started
            correction_seconds += correction_fold_seconds
            metadata = {
                **{
                    key: source_fold[key]
                    for key in (
                        "fold",
                        "train_participants",
                        "validation_participants",
                        "development_participants",
                        "test_participants",
                        "test_origins",
                        "common_suffix_origins",
                        "catboost",
                        "final_preprocessors",
                        "development_scales",
                        "baseline_fit_oof_seconds",
                    )
                },
                "custom": JointBayesPersonalizer(parameters).get_metadata(),
                "conformal_multipliers": multipliers,
                "custom_fit_calibration_seconds": float(custom_fit_seconds),
                "prediction_write_seconds": float(
                    time.perf_counter() - prediction_started
                ),
                "runtime_seconds": float(
                    source_fold["baseline_fit_oof_seconds"] + correction_fold_seconds
                ),
                "correction_only_seconds": float(correction_fold_seconds),
                "baseline_outputs_reused": True,
            }
            fold_metadata.append(metadata)
            (target_checkpoint_dir / f"fold_{fold}_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"completed corrected custom fold {fold}/4 in {correction_fold_seconds:.2f}s",
                flush=True,
            )
    baseline_seconds = float(
        sum(item["baseline_fit_oof_seconds"] for item in fold_metadata)
    )
    manifest = {
        "schema_version": "1.0.0",
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "run_id": config["runtime"]["canonical_run_id"],
        "entries": entries,
        "folds": 5,
        "models": [*BASELINE_NAMES, CUSTOM_NAME],
        "baseline_models": list(BASELINE_NAMES),
        "custom_models": [CUSTOM_NAME],
        "selected_covariance_mode": "full",
        "runtime_seconds": float(baseline_seconds + correction_seconds),
        "correction_only_seconds": float(correction_seconds),
        "invalidated_diagonal_run_seconds": float(source_manifest["runtime_seconds"]),
        "peak_rss_mb": float(
            max(memory.peak / 1024**2, float(source_manifest["peak_rss_mb"]))
        ),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "baseline_outputs_reused": True,
        "selection_correction": "strongest valid candidate after failed superiority gate",
    }
    target_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target_checkpoint_dir / "run_metadata.json").write_text(
        json.dumps(
            {"manifest": manifest, "fold_metadata": fold_metadata},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
