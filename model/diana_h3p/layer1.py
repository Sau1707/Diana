"""Layer 1: participant-balanced convex stacking of three wearable priors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from benchmark.v1_task import HORMONES
from model.diana_h3p.contracts import EXPERTS, StackSelection


def simplex_grid(step: float = 0.10) -> tuple[tuple[float, float, float], ...]:
    """Return the deterministic three-expert simplex, including all endpoints."""

    units_float = 1.0 / float(step)
    units = int(round(units_float))
    if units <= 0 or not np.isclose(units_float, units, atol=1e-12, rtol=0.0):
        raise ValueError("Simplex step must divide one exactly")
    points: list[tuple[float, float, float]] = []
    for median_units in range(units + 1):
        for ridge_units in range(units - median_units + 1):
            catboost_units = units - median_units - ridge_units
            points.append(
                (
                    median_units / units,
                    ridge_units / units,
                    catboost_units / units,
                )
            )
    return tuple(points)


def participant_macro_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    participants: Iterable[object],
) -> float:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    groups = pd.Series(list(participants), dtype="string")
    if truth.shape != prediction.shape or truth.ndim != 1 or len(groups) != len(truth):
        raise ValueError("Participant-macro MAE inputs must be aligned vectors")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Participant-macro MAE requires finite values")
    frame = pd.DataFrame(
        {"participant": groups, "absolute_error": np.abs(truth - prediction)}
    )
    return float(frame.groupby("participant", sort=True)["absolute_error"].mean().mean())


def select_hormone_weights(
    y_true: np.ndarray,
    expert_predictions: Mapping[str, np.ndarray],
    participants: Iterable[object],
    *,
    step: float = 0.10,
    tie_tolerance: float = 1e-12,
) -> tuple[dict[str, float], float]:
    if set(expert_predictions) != set(EXPERTS):
        raise ValueError("Layer 1 requires exactly median, Ridge, and CatBoost")
    matrix = np.column_stack(
        [np.asarray(expert_predictions[name], dtype=float) for name in EXPERTS]
    )
    truth = np.asarray(y_true, dtype=float)
    group_values = list(participants)
    if matrix.shape != (len(truth), len(EXPERTS)) or len(group_values) != len(truth):
        raise ValueError("Layer 1 OOF inputs are not aligned")
    best_point: tuple[float, float, float] | None = None
    best_score = np.inf
    for point in simplex_grid(step):
        prediction = matrix @ np.asarray(point, dtype=float)
        score = participant_macro_mae(truth, prediction, group_values)
        if score < best_score - tie_tolerance:
            best_score, best_point = score, point
            continue
        if abs(score - best_score) <= tie_tolerance and best_point is not None:
            # Larger median, then larger Ridge, then ordinary tuple order.
            candidate_key = (-point[0], -point[1], point)
            current_key = (-best_point[0], -best_point[1], best_point)
            if candidate_key < current_key:
                best_score, best_point = score, point
    if best_point is None:
        raise RuntimeError("No valid Layer-1 simplex point")
    return {name: float(value) for name, value in zip(EXPERTS, best_point)}, float(
        best_score
    )


def select_stack_weights(
    oof: pd.DataFrame,
    *,
    step: float = 0.10,
    tie_tolerance: float = 1e-12,
) -> StackSelection:
    required = {"private_participant_id"}
    required.update(f"y_{hormone}" for hormone in HORMONES)
    required.update(
        f"pred_{expert}_{hormone}" for expert in EXPERTS for hormone in HORMONES
    )
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"Layer-1 OOF table missing {missing}")
    weights: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    participants = oof["private_participant_id"].astype(str)
    for hormone in HORMONES:
        weights[hormone], scores[hormone] = select_hormone_weights(
            oof[f"y_{hormone}"].to_numpy(float),
            {
                expert: oof[f"pred_{expert}_{hormone}"].to_numpy(float)
                for expert in EXPERTS
            },
            participants,
            step=step,
            tie_tolerance=tie_tolerance,
        )
    selection = StackSelection(weights=weights, participant_macro_mae=scores, grid_step=step)
    selection.validate()
    return selection


def stack_prediction_arrays(
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    selection: StackSelection,
) -> dict[str, np.ndarray]:
    selection.validate()
    if set(predictions) != set(EXPERTS):
        raise ValueError("Stack prediction requires exactly three experts")
    output: dict[str, np.ndarray] = {}
    for hormone in HORMONES:
        arrays = [np.asarray(predictions[name][hormone], dtype=float) for name in EXPERTS]
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("Expert predictions must have identical shapes")
        output[hormone] = np.maximum(
            sum(selection.weights[hormone][name] * array for name, array in zip(EXPERTS, arrays)),
            0.0,
        )
    return output


def apply_stack_to_oof(oof: pd.DataFrame, selection: StackSelection) -> pd.DataFrame:
    """Replace three OOF expert columns with the fixed stacked prior."""

    output = oof[
        [
            column
            for column in (
                "sample_id",
                "private_participant_id",
                "origin_day",
                "target_day",
                *[f"y_{hormone}" for hormone in HORMONES],
            )
            if column in oof.columns
        ]
    ].copy()
    predictions = stack_prediction_arrays(
        {
            expert: {
                hormone: oof[f"pred_{expert}_{hormone}"].to_numpy(float)
                for hormone in HORMONES
            }
            for expert in EXPERTS
        },
        selection,
    )
    for hormone in HORMONES:
        output[f"pred_{hormone}"] = predictions[hormone]
    return output


def nested_inner_groups(
    *, outer_test_group: int, held_oof_group: int, all_groups: Iterable[int] = range(5)
) -> tuple[tuple[int, int], int]:
    """Return deterministic 8-participant inner-train and 4-participant validation groups."""

    groups = {int(value) for value in all_groups}
    outer_test_group = int(outer_test_group)
    held_oof_group = int(held_oof_group)
    fit_groups = groups - {outer_test_group, held_oof_group}
    if len(groups) != 5 or len(fit_groups) != 3:
        raise ValueError("Nested H3P stopping requires five distinct groups")
    cyclic = [
        (held_oof_group + offset) % 5
        for offset in range(1, 6)
        if (held_oof_group + offset) % 5 in fit_groups
    ]
    inner_validation = cyclic[0]
    inner_train = tuple(sorted(fit_groups - {inner_validation}))
    if len(inner_train) != 2:
        raise RuntimeError("Nested H3P stopping must be 8/4 participants")
    return inner_train, inner_validation
