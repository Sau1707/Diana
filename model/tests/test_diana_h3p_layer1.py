from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.v1_task import HORMONES
from model.diana_h3p.contracts import EXPERTS, StackSelection
from model.diana_h3p.layer1 import (
    nested_inner_groups,
    participant_macro_mae,
    select_hormone_weights,
    simplex_grid,
    stack_prediction_arrays,
)


def _selection_for(expert: str) -> StackSelection:
    weights = {
        hormone: {name: float(name == expert) for name in EXPERTS}
        for hormone in HORMONES
    }
    return StackSelection(
        weights=weights,
        participant_macro_mae={hormone: 0.0 for hormone in HORMONES},
        grid_step=0.10,
    )


def test_simplex_grid_is_deterministic_convex_and_contains_all_endpoints() -> None:
    first = simplex_grid(0.10)
    second = simplex_grid(0.10)

    assert first == second
    assert len(first) == 66
    assert set(first) >= {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}
    for point in first:
        values = np.asarray(point, dtype=float)
        assert np.isfinite(values).all()
        assert (values >= 0.0).all()
        assert np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12)


def test_participant_macro_objective_gives_each_participant_equal_influence() -> None:
    participants = ["short"] + ["long"] * 9
    truth = np.zeros(10, dtype=float)
    prediction = np.asarray([10.0] + [0.0] * 9, dtype=float)

    assert participant_macro_mae(truth, prediction, participants) == 5.0
    assert np.mean(np.abs(truth - prediction)) == 1.0


def test_weight_selection_is_deterministic_and_uses_frozen_tie_break() -> None:
    truth = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
    identical = {name: truth.copy() for name in EXPERTS}

    first_weights, first_score = select_hormone_weights(
        truth, identical, ["a", "a", "b", "b"], step=0.10
    )
    second_weights, second_score = select_hormone_weights(
        truth, identical, ["a", "a", "b", "b"], step=0.10
    )

    assert first_weights == second_weights == {
        "population_median": 1.0,
        "wearable_ridge": 0.0,
        "catboost": 0.0,
    }
    assert first_score == second_score == 0.0


def test_weight_selection_can_select_the_ridge_endpoint() -> None:
    truth = np.asarray([0.2, 0.8, 1.4, 2.0], dtype=float)
    weights, score = select_hormone_weights(
        truth,
        {
            "population_median": np.zeros_like(truth),
            "wearable_ridge": truth.copy(),
            "catboost": np.full_like(truth, 4.0),
        },
        ["a", "a", "b", "b"],
        step=0.10,
    )

    assert weights == {
        "population_median": 0.0,
        "wearable_ridge": 1.0,
        "catboost": 0.0,
    }
    assert score == 0.0


def test_stack_endpoints_reproduce_each_expert_exactly() -> None:
    predictions = {
        expert: {
            hormone: np.asarray([index + 0.1, index + 0.4], dtype=float)
            for index, hormone in enumerate(HORMONES)
        }
        for expert in EXPERTS
    }
    # Give the experts distinct values while preserving the same array shapes.
    for expert_index, expert in enumerate(EXPERTS):
        for hormone in HORMONES:
            predictions[expert][hormone] = (
                predictions[expert][hormone] + 10.0 * expert_index
            )

    for expert in EXPERTS:
        stacked = stack_prediction_arrays(predictions, _selection_for(expert))
        for hormone in HORMONES:
            np.testing.assert_array_equal(stacked[hormone], predictions[expert][hormone])


def test_nested_inner_groups_are_deterministic_and_exclude_outer_and_oof_groups() -> None:
    all_groups = set(range(5))
    for outer_test_group in range(5):
        for held_oof_group in sorted(all_groups - {outer_test_group}):
            first = nested_inner_groups(
                outer_test_group=outer_test_group,
                held_oof_group=held_oof_group,
            )
            second = nested_inner_groups(
                outer_test_group=outer_test_group,
                held_oof_group=held_oof_group,
            )
            assert first == second
            inner_train, inner_validation = first
            used = set(inner_train) | {inner_validation}
            assert len(inner_train) == 2
            assert inner_validation not in inner_train
            assert used == all_groups - {outer_test_group, held_oof_group}
            assert outer_test_group not in used
            assert held_oof_group not in used

