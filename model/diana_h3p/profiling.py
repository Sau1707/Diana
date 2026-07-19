"""Target-independent complete Layer-2 backend parity and timing gate."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import psutil

from benchmark.v1_personalization import PersonalizationPlan, scoring_sample_ids
from benchmark.v1_task import HORMONES
from model.diana_h3p.contracts import BUDGETS, StackSelection
from model.diana_h3p.model import DianaH3P, fit_h3p_parameters


def _rows_for_ids(frame: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    indexed = frame.assign(sample_id=frame["sample_id"].astype(str)).set_index("sample_id")
    return indexed.loc[sample_ids].reset_index()


def _calibration(
    frame: pd.DataFrame, plan: PersonalizationPlan, budget: int
) -> pd.DataFrame:
    if budget == 0:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "private_participant_id",
                "target_day",
                *[f"y_{hormone}" for hormone in HORMONES],
                *[f"pred_{hormone}" for hormone in HORMONES],
            ]
        )
    ids = plan.calibration_candidates.loc[
        plan.calibration_candidates["calibration_rank"].le(budget), "sample_id"
    ].astype(str).tolist()
    return _rows_for_ids(frame, ids)


def _numeric_signature(parameters: Any, predictions: list[pd.DataFrame]) -> np.ndarray:
    core = parameters.layer2.core
    values: list[np.ndarray] = [
        core.sigma_a.matrix.ravel(),
        core.psi[3].matrix.ravel(),
        core.psi[7].matrix.ravel(),
        core.sigma_future.matrix.ravel(),
    ]
    values.append(
        np.asarray(
            [
                parameters.layer2.interval_multipliers[budget][hormone]
                for budget in BUDGETS
                for hormone in HORMONES
            ],
            dtype=float,
        )
    )
    for prediction in predictions:
        values.append(
            prediction[
                [
                    *[f"pred_{hormone}" for hormone in HORMONES],
                    *[f"lower_{hormone}" for hormone in HORMONES],
                    *[f"upper_{hormone}" for hormone in HORMONES],
                ]
            ].to_numpy(float).ravel()
        )
    return np.concatenate(values)


def _one_complete_run(
    layer1_oof: pd.DataFrame,
    plan: PersonalizationPlan,
    stack: StackSelection,
    h3p_config: dict[str, Any],
    *,
    backend_name: str,
    backend_device: str | None,
) -> np.ndarray:
    parameters, _ = fit_h3p_parameters(
        layer1_oof,
        plan,
        stack,
        backend_name=backend_name,
        backend_device=backend_device,
        seed=int(h3p_config["runtime"]["seed"]),
        quantile=float(h3p_config["uncertainty"]["multiplier_quantile"]),
        absolute_floor=float(h3p_config["layer2"]["absolute_eigenvalue_floor"]),
        relative_floor=float(h3p_config["layer2"]["relative_eigenvalue_floor"]),
        near_diagonal_threshold=float(h3p_config["layer2"]["near_diagonal_threshold"]),
    )
    predictor = DianaH3P(parameters)
    scoring = _rows_for_ids(layer1_oof, scoring_sample_ids(plan))
    predictions = [
        predictor.predict(
            scoring,
            _calibration(layer1_oof, plan, budget),
            budget=budget,
            include_intervals=True,
        )
        for budget in BUDGETS
    ]
    return _numeric_signature(parameters, predictions)


def profile_layer2_backends(
    layer1_oof: pd.DataFrame,
    plan: PersonalizationPlan,
    stack: StackSelection,
    h3p_config: dict[str, Any],
) -> dict[str, Any]:
    """Profile complete Layer 2 without consulting any outer-test score."""

    warmups = int(h3p_config["backend"]["profile_warmups"])
    repetitions = int(h3p_config["backend"]["profile_repetitions"])
    candidates = (
        ("numpy", "numpy", None),
        ("torch_cpu", "torch", "cpu"),
        ("torch_cuda", "torch", "cuda"),
    )
    process = psutil.Process()
    results: dict[str, Any] = {}
    reference: np.ndarray | None = None
    tolerance_rtol = float(h3p_config["backend"]["parity_rtol"])
    tolerance_atol = float(h3p_config["backend"]["parity_atol"])
    for identity, backend_name, device in candidates:
        try:
            for _ in range(warmups):
                _one_complete_run(
                    layer1_oof,
                    plan,
                    stack,
                    h3p_config,
                    backend_name=backend_name,
                    backend_device=device,
                )
            timings: list[float] = []
            signatures: list[np.ndarray] = []
            peak_rss = process.memory_info().rss
            for repetition in range(repetitions):
                started = time.perf_counter()
                signature = _one_complete_run(
                    layer1_oof,
                    plan,
                    stack,
                    h3p_config,
                    backend_name=backend_name,
                    backend_device=device,
                )
                timings.append(time.perf_counter() - started)
                signatures.append(signature)
                peak_rss = max(peak_rss, process.memory_info().rss)
                print(
                    f"backend profile {identity}: repetition {repetition + 1}/{repetitions} "
                    f"completed in {timings[-1]:.3f}s",
                    flush=True,
                )
            if identity == "numpy":
                reference = signatures[0]
            if reference is None:
                raise RuntimeError("NumPy reference must be profiled first")
            parity_error = float(np.max(np.abs(signatures[0] - reference)))
            parity = bool(
                np.allclose(
                    signatures[0], reference, rtol=tolerance_rtol, atol=tolerance_atol
                )
            )
            deterministic = all(
                np.array_equal(signatures[0], signature) for signature in signatures[1:]
            )
            results[identity] = {
                "available": True,
                "median_seconds": float(np.median(timings)),
                "timings_seconds": [float(value) for value in timings],
                "peak_process_rss_mb": float(peak_rss / 1024**2),
                "parity_with_numpy": parity,
                "maximum_absolute_parity_error": parity_error,
                "bitwise_deterministic_repetitions": deterministic,
                "complete_layer2_includes": [
                    "covariance_estimation",
                    "leave_one_participant_interval_calibration",
                    "posterior_updates",
                    "interval_generation",
                ],
            }
        except (ImportError, ModuleNotFoundError, RuntimeError) as error:
            results[identity] = {
                "available": False,
                "reason": f"{type(error).__name__}: {error}",
            }
    numpy_seconds = float(results["numpy"]["median_seconds"])
    required = 1.0 - float(h3p_config["backend"]["minimum_speedup_fraction"])
    eligible = [
        identity
        for identity in ("torch_cpu", "torch_cuda")
        if results[identity].get("available")
        and results[identity].get("parity_with_numpy")
        and results[identity].get("bitwise_deterministic_repetitions")
        and float(results[identity]["median_seconds"]) <= required * numpy_seconds
    ]
    recommendation = (
        min(eligible, key=lambda identity: results[identity]["median_seconds"])
        if eligible
        else "numpy"
    )
    return {
        "selection_uses_target_performance": False,
        "warmups": warmups,
        "repetitions": repetitions,
        "parity_rtol": tolerance_rtol,
        "parity_atol": tolerance_atol,
        "minimum_speedup_fraction": float(
            h3p_config["backend"]["minimum_speedup_fraction"]
        ),
        "results": results,
        "recommended_canonical_backend": recommendation,
    }
