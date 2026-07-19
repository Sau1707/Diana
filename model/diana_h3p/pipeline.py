"""Private Diana-H3P orchestration over the frozen Hormonbench v1 bundle."""

from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import psutil

from benchmark.data.v1_adapter import load_private_group_mapping
from benchmark.data.v1_folds import (
    fold_roles,
    group_hash,
    validate_five_fold_protocol,
)
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
    canonical_hash,
    file_sha256,
    input_schema_hash,
    load_v1_config,
    project_path,
    task_spec_hash,
)
from model.catboost.v1_model import CatBoostV1
from model.diana_h3p.contracts import (
    BUDGETS,
    EXPERTS,
    H3PParameters,
    h3p_config_hash,
    h3p_model_spec_hash,
)
from model.diana_h3p.layer1 import (
    apply_stack_to_oof,
    nested_inner_groups,
    select_stack_weights,
    stack_prediction_arrays,
)
from model.diana_h3p.model import DianaH3P, fit_h3p_parameters
from model.diana_h3p.serialization import save_parameters
from model.population_median.v1_model import PopulationMedianV1
from model.v1_common import participant_balanced_iqr_scales
from model.wearable_ridge.model import WearableRidgeV1


CUSTOM_NAME = "diana_h3p"
BASELINE_NAMES = EXPERTS


class PeakRSS:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSS":
        process = psutil.Process(os.getpid())
        self.peak = process.memory_info().rss

        def sample() -> None:
            while not self._stop.wait(0.05):
                self.peak = max(self.peak, process.memory_info().rss)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def h3p_path(config: Mapping[str, Any], key: str) -> Path:
    return Path(str(config["_project_root"])) / str(config["paths"][key])


def evaluation_config(
    benchmark_config: Mapping[str, Any], h3p_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return an operational copy pointing the independent evaluator at one H3P run."""

    output = copy.deepcopy(dict(benchmark_config))
    root = Path(str(h3p_config["_project_root"]))
    for destination, source in (
        ("prediction_run_dir", "prediction_dir"),
        ("prediction_manifest", "prediction_manifest"),
        ("checkpoint_dir", "checkpoint_dir"),
        ("participant_metrics_dir", "participant_metrics_dir"),
        ("results_dir", "results_dir"),
    ):
        output["paths"][destination] = str(h3p_path(h3p_config, source).relative_to(root))
    output["custom"]["working_name"] = CUSTOM_NAME
    output["active_custom_model"] = CUSTOM_NAME
    return output


def _mask(bundle: V1PreparedBundle, participant_ids: Iterable[int]) -> pd.Series:
    return bundle.frame["private_participant_id"].astype(int).isin(
        {int(value) for value in participant_ids}
    )


def validate_frozen_contract(
    benchmark_config: Mapping[str, Any], h3p_config: Mapping[str, Any]
) -> dict[str, Any]:
    bundle = load_v1_bundle(project_path(benchmark_config, "prepared_dir"))
    mapping = load_private_group_mapping(benchmark_config)
    expected = h3p_config["expected_benchmark"]
    actual = {
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "eligible_participants": int(
            bundle.frame["private_participant_id"].nunique()
        ),
        "eligible_origins": len(bundle.frame),
        "task_spec_hash": task_spec_hash(benchmark_config),
        "fold_hash": group_hash(mapping, int(benchmark_config["folds"]["seed"])),
        "input_schema_hash": input_schema_hash(
            bundle.feature_columns, bundle.metadata["feature_provenance"]
        ),
    }
    plans = [
        build_personalization_plan(bundle, fold_roles(mapping, fold)["test"])
        for fold in range(5)
    ]
    actual["common_suffix_origins"] = int(
        sum(plan.aggregate["common_scoring_origins"] for plan in plans)
    )
    for key, value in actual.items():
        if str(expected[key]) != str(value):
            raise RuntimeError(f"Frozen benchmark invariant changed: {key}")
    protocol = validate_five_fold_protocol(
        mapping, bundle.frame["private_participant_id"]
    )
    if protocol["unique_test_origins"] != int(expected["eligible_origins"]):
        raise RuntimeError("Five-fold test union must cover every expected origin")
    return {**actual, "fold_protocol": protocol}


def _expected_sample_ids(
    bundle: V1PreparedBundle,
    mapping: Mapping[int, int],
    *,
    fold: int,
    track: str,
) -> list[str]:
    roles = fold_roles(mapping, fold)
    if track == TRACK_COLD:
        return bundle.frame.loc[
            _mask(bundle, roles["test"]), "sample_id"
        ].astype(str).tolist()
    if track == TRACK_FEW_SHOT:
        return scoring_sample_ids(build_personalization_plan(bundle, roles["test"]))
    raise ValueError(track)


def validate_baseline_reuse(
    benchmark_config: Mapping[str, Any], h3p_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind every reused baseline byte to the frozen task/folds/schema and rows."""

    contract = validate_frozen_contract(benchmark_config, h3p_config)
    bundle = load_v1_bundle(project_path(benchmark_config, "prepared_dir"))
    mapping = load_private_group_mapping(benchmark_config)
    source_manifest_path = h3p_path(h3p_config, "baseline_prediction_manifest")
    preserved_manifest_path = h3p_path(h3p_config, "preserved_baseline_manifest")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    preserved_manifest = json.loads(
        preserved_manifest_path.read_text(encoding="utf-8")
    )
    if source_manifest.get("task_id") != TASK_ID or str(
        source_manifest.get("task_version")
    ) != TASK_VERSION:
        raise RuntimeError("Baseline manifest task identity mismatch")
    source_entries = [
        entry
        for entry in source_manifest.get("entries", [])
        if str(entry.get("model_name")) in BASELINE_NAMES
    ]
    if len(source_entries) != 60:
        raise RuntimeError("Baseline reuse requires exactly 60 explicit files")
    keys = [
        (
            int(entry["fold"]),
            str(entry["track"]),
            int(entry["calibration_budget"]),
            str(entry["model_name"]),
        )
        for entry in source_entries
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Baseline manifest contains duplicate identities")
    preserved = {
        (
            int(entry["fold"]),
            str(entry["track"]),
            int(entry["calibration_budget"]),
            str(entry["model_name"]),
        ): entry
        for entry in preserved_manifest.get("entries", [])
        if str(entry.get("model_name")) in BASELINE_NAMES
    }
    if set(preserved) != set(keys):
        raise RuntimeError("Preserved baseline source does not match current entries")
    source_dir = h3p_path(h3p_config, "baseline_prediction_dir")
    preserved_dir = h3p_path(h3p_config, "preserved_baseline_dir")
    verified_entries: list[dict[str, Any]] = []
    for entry, key in zip(source_entries, keys, strict=True):
        filename = str(entry["file"])
        if Path(filename).name != filename:
            raise RuntimeError("Baseline manifest filenames must be local")
        source = source_dir / filename
        preserved_path = preserved_dir / str(preserved[key]["file"])
        digest = file_sha256(source)
        if digest != str(entry["sha256"]):
            raise RuntimeError(f"Baseline byte hash mismatch: {filename}")
        if digest != file_sha256(preserved_path) or source.read_bytes() != preserved_path.read_bytes():
            raise RuntimeError(f"Baseline is not byte-identical to preserved source: {filename}")
        frame = pd.read_csv(source)
        expected_ids = _expected_sample_ids(
            bundle,
            mapping,
            fold=int(entry["fold"]),
            track=str(entry["track"]),
        )
        validate_v1_prediction_frame(
            frame,
            expected_sample_ids=expected_ids,
            expected_track=str(entry["track"]),
            expected_fold=int(entry["fold"]),
            expected_budget=int(entry["calibration_budget"]),
        )
        if set(frame["model_name"]) != {str(entry["model_name"])}:
            raise RuntimeError("Baseline file model identity mismatch")
        verified_entries.append(
            {
                "file": filename,
                "sha256": digest,
                "fold": int(entry["fold"]),
                "track": str(entry["track"]),
                "calibration_budget": int(entry["calibration_budget"]),
                "model_name": str(entry["model_name"]),
                "rows": len(frame),
                "samples": int(frame["sample_id"].nunique()),
            }
        )
    baseline_config_subset = {
        "task": benchmark_config["task"],
        "features": benchmark_config["features"],
        "folds": benchmark_config["folds"],
        "preprocessing": benchmark_config["preprocessing"],
        "models": benchmark_config["models"],
        "personalization": benchmark_config["personalization"],
    }
    return {
        "status": "passed",
        **contract,
        "baseline_entries": len(verified_entries),
        "byte_identical_entries": len(verified_entries),
        "prediction_required_columns": 12,
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "preserved_manifest_sha256": file_sha256(preserved_manifest_path),
        "baseline_config_subset_hash": canonical_hash(baseline_config_subset),
        "entries": verified_entries,
    }


def _fit_oof_block(
    bundle: V1PreparedBundle,
    mapping: Mapping[int, int],
    benchmark_config: dict[str, Any],
    *,
    outer_fold: int,
    held_group: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outer_test_ids = {pid for pid, group in mapping.items() if int(group) == outer_fold}
    held_ids = {pid for pid, group in mapping.items() if int(group) == held_group}
    fit_ids = set(mapping) - outer_test_ids - held_ids
    if len(outer_test_ids) != 4 or len(held_ids) != 4 or len(fit_ids) != 12:
        raise RuntimeError("H3P grouped OOF roles must be 12 fit / 4 held / 4 outer test")
    inner_train_groups, inner_validation_group = nested_inner_groups(
        outer_test_group=outer_fold, held_oof_group=held_group
    )
    inner_train_ids = {
        pid for pid, group in mapping.items() if int(group) in inner_train_groups
    }
    inner_validation_ids = {
        pid for pid, group in mapping.items() if int(group) == inner_validation_group
    }
    if (
        len(inner_train_ids) != 8
        or len(inner_validation_ids) != 4
        or inner_train_ids & inner_validation_ids
        or (inner_train_ids | inner_validation_ids) != fit_ids
    ):
        raise RuntimeError("Nested CatBoost stopping roles must be disjoint 8/4")
    selector = CatBoostV1(benchmark_config)
    tree_counts = selector.select_tree_counts(
        bundle.fit_view(_mask(bundle, inner_train_ids)),
        bundle.fit_view(_mask(bundle, inner_validation_ids)),
    )
    fit_view = bundle.fit_view(_mask(bundle, fit_ids))
    inference = bundle.inference_view(_mask(bundle, held_ids))
    estimators = {
        "population_median": PopulationMedianV1(benchmark_config).fit(fit_view),
        "wearable_ridge": WearableRidgeV1(benchmark_config).fit(fit_view),
        "catboost": CatBoostV1(benchmark_config).fit_fixed(fit_view, tree_counts),
    }
    selected = bundle.frame.loc[
        _mask(bundle, held_ids),
        [
            "sample_id",
            "private_participant_id",
            "origin_day",
            "target_day",
            *TARGET_LOG_COLUMNS.values(),
        ],
    ].reset_index(drop=True)
    output = selected[
        ["sample_id", "private_participant_id", "origin_day", "target_day"]
    ].copy()
    for hormone in HORMONES:
        output[f"y_{hormone}"] = selected[TARGET_LOG_COLUMNS[hormone]].to_numpy(float)
    metadata: dict[str, Any] = {
        "outer_fold": int(outer_fold),
        "held_group": int(held_group),
        "fit_participants": 12,
        "held_participants": 4,
        "outer_test_participants": 4,
        "inner_train_participants": 8,
        "inner_validation_participants": 4,
        "inner_train_groups": list(inner_train_groups),
        "inner_validation_group": int(inner_validation_group),
        "catboost_tree_count": dict(tree_counts),
        "catboost_best_iteration": dict(selector.best_iterations),
        "catboost_best_validation_score": dict(selector.best_validation_scores),
    }
    for name, estimator in estimators.items():
        prediction = estimator.predict(inference)
        for hormone in HORMONES:
            output[f"pred_{name}_{hormone}"] = prediction[hormone]
        model_metadata = estimator.get_metadata()
        if "preprocessor" in model_metadata:
            metadata[f"{name}_final_preprocessor"] = model_metadata["preprocessor"]
    return output, metadata


def generate_development_oof(
    bundle: V1PreparedBundle,
    mapping: Mapping[int, int],
    benchmark_config: dict[str, Any],
    *,
    outer_fold: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    development_groups = sorted(set(range(5)) - {int(outer_fold)})
    parts: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for position, held_group in enumerate(development_groups, start=1):
        started = time.perf_counter()
        rows, block_metadata = _fit_oof_block(
            bundle,
            mapping,
            benchmark_config,
            outer_fold=int(outer_fold),
            held_group=int(held_group),
        )
        block_metadata["runtime_seconds"] = float(time.perf_counter() - started)
        parts.append(rows)
        metadata.append(block_metadata)
        print(
            f"outer fold {outer_fold}: completed development OOF block "
            f"{position}/4 in {block_metadata['runtime_seconds']:.2f}s",
            flush=True,
        )
    output = pd.concat(parts, ignore_index=True).sort_values(
        ["private_participant_id", "origin_day", "sample_id"]
    ).reset_index(drop=True)
    roles = fold_roles(mapping, outer_fold)
    if output["private_participant_id"].astype(int).isin(roles["test"]).any():
        raise RuntimeError("Outer-test participant entered development OOF")
    expected = int(_mask(bundle, roles["train"] | roles["validation"]).sum())
    if len(output) != expected or output["sample_id"].duplicated().any():
        raise RuntimeError("Development grouped OOF coverage mismatch")
    return output, metadata


def _long_baseline_to_wide(
    frame: pd.DataFrame, alignment: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    output = alignment.copy()
    for hormone in HORMONES:
        values = frame.loc[frame["hormone"].eq(hormone)].set_index("sample_id")["y_pred"]
        mapped = output["sample_id"].astype(str).map(values)
        if mapped.isna().any():
            raise RuntimeError(f"Missing {model_name} outer prediction")
        output[f"pred_{hormone}"] = mapped.to_numpy(float)
    return output


def _load_outer_layer1_rows(
    bundle: V1PreparedBundle,
    mapping: Mapping[int, int],
    h3p_config: Mapping[str, Any],
    reuse_audit: Mapping[str, Any],
    parameters: H3PParameters,
    *,
    fold: int,
) -> pd.DataFrame:
    test_mask = _mask(bundle, fold_roles(mapping, fold)["test"])
    alignment = bundle.frame.loc[
        test_mask,
        ["sample_id", "private_participant_id", "origin_day", "target_day"],
    ].reset_index(drop=True)
    source_dir = h3p_path(h3p_config, "baseline_prediction_dir")
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for model_name in BASELINE_NAMES:
        entry = next(
            item
            for item in reuse_audit["entries"]
            if item["fold"] == fold
            and item["track"] == TRACK_COLD
            and item["calibration_budget"] == 0
            and item["model_name"] == model_name
        )
        frame = pd.read_csv(source_dir / entry["file"])
        wide = _long_baseline_to_wide(frame, alignment, model_name)
        predictions[model_name] = {
            hormone: wide[f"pred_{hormone}"].to_numpy(float) for hormone in HORMONES
        }
    stacked = stack_prediction_arrays(predictions, parameters.stack)
    output = alignment.copy()
    for hormone in HORMONES:
        output[f"pred_{hormone}"] = stacked[hormone]
    return output


def _authorized_calibration(
    bundle: V1PreparedBundle,
    layer1_all: pd.DataFrame,
    plan: PersonalizationPlan,
    budget: int,
) -> pd.DataFrame:
    if int(budget) == 0:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "private_participant_id",
                "target_day",
                *[f"y_{hormone}" for hormone in HORMONES],
                *[f"pred_{hormone}" for hormone in HORMONES],
            ]
        )
    candidates = plan.calibration_candidates.loc[
        plan.calibration_candidates["calibration_rank"].le(int(budget))
    ].sort_values(["private_participant_id", "calibration_rank"])
    base = layer1_all.assign(sample_id=layer1_all["sample_id"].astype(str)).set_index("sample_id")
    truth = bundle.frame.assign(sample_id=bundle.frame["sample_id"].astype(str)).set_index("sample_id")
    records: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        sample_id = str(row.sample_id)
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "private_participant_id": str(row.private_participant_id),
            "target_day": int(row.target_day),
        }
        for hormone in HORMONES:
            record[f"y_{hormone}"] = float(truth.loc[sample_id, TARGET_LOG_COLUMNS[hormone]])
            record[f"pred_{hormone}"] = float(base.loc[sample_id, f"pred_{hormone}"])
        records.append(record)
    output = pd.DataFrame(records)
    counts = output.groupby("private_participant_id").size()
    if counts.empty or not counts.eq(int(budget)).all():
        raise RuntimeError("H3P calibration authorization violated exact K")
    return output


def _to_submission(
    wide: pd.DataFrame,
    *,
    fold: int,
    track: str,
    budget: int,
    include_intervals: bool,
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
                "model_name": CUSTOM_NAME,
                "model_version": "1.0.0",
            }
            if include_intervals:
                record["y_lower"] = float(getattr(row, f"lower_{hormone}"))
                record["y_upper"] = float(getattr(row, f"upper_{hormone}"))
            records.append(record)
    return pd.DataFrame(records)


def _write_prediction(
    run_dir: Path,
    frame: pd.DataFrame,
    *,
    fold: int,
    track: str,
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
    filename = f"fold{fold}__{track}__k{budget}__{CUSTOM_NAME}.csv"
    path = run_dir / filename
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, index=False)
    return {
        "file": filename,
        "sha256": file_sha256(path),
        "rows": len(frame),
        "samples": int(frame["sample_id"].nunique()),
        "model_name": CUSTOM_NAME,
        "track": track,
        "fold": int(fold),
        "calibration_budget": int(budget),
    }


def run_development_fold(
    benchmark_config: dict[str, Any],
    h3p_config: dict[str, Any],
    *,
    fold: int = 0,
    write_private: bool = True,
) -> dict[str, Any]:
    """Run development grouped OOF only; no outer-test prediction or scoring."""

    validate_frozen_contract(benchmark_config, h3p_config)
    bundle = load_v1_bundle(project_path(benchmark_config, "prepared_dir"))
    mapping = load_private_group_mapping(benchmark_config)
    started = time.perf_counter()
    oof, block_metadata = generate_development_oof(
        bundle, mapping, benchmark_config, outer_fold=int(fold)
    )
    selection = select_stack_weights(
        oof,
        step=float(h3p_config["layer1"]["simplex_step"]),
        tie_tolerance=float(h3p_config["layer1"]["tie_tolerance"]),
    )
    layer1_oof = apply_stack_to_oof(oof, selection)
    development_ids = fold_roles(mapping, fold)["train"] | fold_roles(mapping, fold)["validation"]
    plan = build_personalization_plan(bundle, development_ids)
    diagnostics = {
        "fold": int(fold),
        "outer_test_metrics_read": False,
        "development_participants": 16,
        "development_origins": len(oof),
        "common_suffix_origins": int(plan.aggregate["common_scoring_origins"]),
        "stack_weights": selection.weights,
        "stack_participant_macro_log1p_mae": selection.participant_macro_mae,
        "expert_participant_macro_log1p_mae": {},
        "oof_blocks": block_metadata,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    from model.diana_h3p.layer1 import participant_macro_mae

    for expert in BASELINE_NAMES:
        diagnostics["expert_participant_macro_log1p_mae"][expert] = {
            hormone: participant_macro_mae(
                oof[f"y_{hormone}"].to_numpy(float),
                oof[f"pred_{expert}_{hormone}"].to_numpy(float),
                oof["private_participant_id"],
            )
            for hormone in HORMONES
        }
    if write_private:
        development_dir = h3p_path(h3p_config, "private_run_root").parent / "development"
        development_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(development_dir / f"fold_{fold}_base_oof.csv", index=False)
        layer1_oof.to_csv(development_dir / f"fold_{fold}_layer1_oof.csv", index=False)
        (development_dir / f"fold_{fold}_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return {"diagnostics": diagnostics, "oof": oof, "layer1_oof": layer1_oof, "plan": plan}


def run_official_h3p(
    benchmark_config: dict[str, Any], h3p_config: dict[str, Any]
) -> dict[str, Any]:
    """Run one frozen five-fold H3P evaluation and write private predictions."""

    if str(h3p_config["backend"]["canonical"]) == "pending":
        raise RuntimeError("Freeze the canonical Layer-2 backend before official inference")
    configured_commit = str(h3p_config["runtime"]["training_code_commit"])
    if configured_commit == "pending":
        raise RuntimeError("Record the code-freeze commit before official inference")
    if configured_commit == "from_git_head":
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=h3p_config["_project_root"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError("Official H3P inference requires a code-freeze Git commit")
        training_code_commit = result.stdout.strip()
    else:
        training_code_commit = configured_commit
    contract = validate_frozen_contract(benchmark_config, h3p_config)
    reuse_audit = validate_baseline_reuse(benchmark_config, h3p_config)
    bundle = load_v1_bundle(project_path(benchmark_config, "prepared_dir"))
    mapping = load_private_group_mapping(benchmark_config)
    run_dir = h3p_path(h3p_config, "prediction_dir")
    manifest_path = h3p_path(h3p_config, "prediction_manifest")
    oof_dir = h3p_path(h3p_config, "oof_dir")
    checkpoint_dir = h3p_path(h3p_config, "checkpoint_dir")
    audit_dir = h3p_path(h3p_config, "audit_dir")
    for path in (run_dir, oof_dir, checkpoint_dir, audit_dir):
        path.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        raise FileExistsError("Official H3P prediction manifest already exists")
    (audit_dir / "baseline_reuse_audit.json").write_text(
        json.dumps(reuse_audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    entries: list[dict[str, Any]] = []
    source_dir = h3p_path(h3p_config, "baseline_prediction_dir")
    for entry in reuse_audit["entries"]:
        source = source_dir / entry["file"]
        destination = run_dir / entry["file"]
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copyfile(source, destination)
        if file_sha256(destination) != entry["sha256"]:
            raise RuntimeError("Copied baseline hash changed")
        entries.append({**entry, "reused_byte_identically": True})
    started = time.perf_counter()
    fold_metadata: list[dict[str, Any]] = []
    with PeakRSS() as memory:
        for fold in range(5):
            fold_started = time.perf_counter()
            print(f"starting official H3P outer fold {fold}/4", flush=True)
            oof_started = time.perf_counter()
            base_oof, block_metadata = generate_development_oof(
                bundle, mapping, benchmark_config, outer_fold=fold
            )
            oof_seconds = time.perf_counter() - oof_started
            selection = select_stack_weights(
                base_oof,
                step=float(h3p_config["layer1"]["simplex_step"]),
                tie_tolerance=float(h3p_config["layer1"]["tie_tolerance"]),
            )
            layer1_oof = apply_stack_to_oof(base_oof, selection)
            roles = fold_roles(mapping, fold)
            development_ids = roles["train"] | roles["validation"]
            development_plan = build_personalization_plan(bundle, development_ids)
            parameter_started = time.perf_counter()
            canonical = str(h3p_config["backend"]["canonical"])
            backend_name = "torch" if canonical.startswith("torch_") else "numpy"
            backend_device = canonical.removeprefix("torch_") if canonical.startswith("torch_") else None
            parameters, fit_diagnostics = fit_h3p_parameters(
                layer1_oof,
                development_plan,
                selection,
                backend_name=backend_name,
                backend_device=backend_device,
                seed=int(h3p_config["runtime"]["seed"]),
                quantile=float(h3p_config["uncertainty"]["multiplier_quantile"]),
                absolute_floor=float(h3p_config["layer2"]["absolute_eigenvalue_floor"]),
                relative_floor=float(h3p_config["layer2"]["relative_eigenvalue_floor"]),
                near_diagonal_threshold=float(h3p_config["layer2"]["near_diagonal_threshold"]),
            )
            parameter_seconds = time.perf_counter() - parameter_started
            save_parameters(parameters, checkpoint_dir / f"fold_{fold}_parameters.json")
            base_oof.to_csv(oof_dir / f"fold_{fold}_base_oof.csv", index=False)
            layer1_oof.to_csv(oof_dir / f"fold_{fold}_layer1_oof.csv", index=False)
            layer1_test = _load_outer_layer1_rows(
                bundle, mapping, h3p_config, reuse_audit, parameters, fold=fold
            )
            predictor = DianaH3P(parameters)
            cold = predictor.predict(
                layer1_test,
                _authorized_calibration(bundle, layer1_test, build_personalization_plan(bundle, roles["test"]), 0),
                budget=0,
                include_intervals=True,
            )
            entries.append(
                _write_prediction(
                    run_dir,
                    _to_submission(cold, fold=fold, track=TRACK_COLD, budget=0, include_intervals=True),
                    fold=fold,
                    track=TRACK_COLD,
                    budget=0,
                )
            )
            test_plan = build_personalization_plan(bundle, roles["test"])
            scoring_ids = scoring_sample_ids(test_plan)
            scoring = layer1_test.assign(sample_id=layer1_test["sample_id"].astype(str)).set_index("sample_id").loc[scoring_ids].reset_index()
            for budget in BUDGETS:
                calibration = _authorized_calibration(
                    bundle, layer1_test, test_plan, budget
                )
                personalized = predictor.predict(
                    scoring,
                    calibration,
                    budget=budget,
                    include_intervals=True,
                )
                entries.append(
                    _write_prediction(
                        run_dir,
                        _to_submission(
                            personalized,
                            fold=fold,
                            track=TRACK_FEW_SHOT,
                            budget=budget,
                            include_intervals=True,
                        ),
                        fold=fold,
                        track=TRACK_FEW_SHOT,
                        budget=budget,
                    )
                )
            fold_seconds = time.perf_counter() - fold_started
            dev_view = bundle.fit_view(_mask(bundle, development_ids))
            metadata = {
                "fold": fold,
                "train_participants": 12,
                "validation_participants": 4,
                "development_participants": 16,
                "test_participants": 4,
                "test_origins": int(_mask(bundle, roles["test"]).sum()),
                "common_suffix_origins": int(test_plan.aggregate["common_scoring_origins"]),
                "layer1": {
                    "weights": parameters.stack.weights,
                    "oof_participant_macro_log1p_mae": parameters.stack.participant_macro_mae,
                    "nested_blocks": block_metadata,
                },
                "layer2": DianaH3P(parameters).get_metadata(),
                "fit_diagnostics": fit_diagnostics,
                "development_scales": participant_balanced_iqr_scales(dev_view),
                "oof_seconds": float(oof_seconds),
                "layer2_fit_seconds": float(parameter_seconds),
                "runtime_seconds": float(fold_seconds),
            }
            fold_metadata.append(metadata)
            (checkpoint_dir / f"fold_{fold}_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"completed official H3P outer fold {fold}/4 in {fold_seconds:.2f}s",
                flush=True,
            )
    manifest = {
        "schema_version": "1.1.0",
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "task_spec_hash": contract["task_spec_hash"],
        "fold_hash": contract["fold_hash"],
        "input_schema_hash": contract["input_schema_hash"],
        "h3p_config_hash": h3p_config_hash(h3p_config),
        "h3p_model_spec_hash": h3p_model_spec_hash(h3p_config),
        "implementation_spec_sha256": file_sha256(
            h3p_path(h3p_config, "implementation_spec")
        ),
        "training_code_commit": training_code_commit,
        "run_id": str(h3p_config["runtime"]["run_id"]),
        "entries": entries,
        "folds": 5,
        "models": [*BASELINE_NAMES, CUSTOM_NAME],
        "baseline_models": list(BASELINE_NAMES),
        "custom_models": [CUSTOM_NAME],
        "baseline_outputs_reused": True,
        "baseline_reuse_audit_sha256": file_sha256(audit_dir / "baseline_reuse_audit.json"),
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_rss_mb": float(memory.peak / 1024**2),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "canonical_backend": str(h3p_config["backend"]["canonical"]),
        "post_hoc_existing_protocol": True,
        "outer_test_used_for_model_selection": False,
    }
    if len(entries) != 80:
        raise RuntimeError("Official H3P manifest must contain 60 baselines + 20 custom files")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "run_metadata.json").write_text(
        json.dumps(
            {"manifest": manifest, "fold_metadata": fold_metadata},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest
