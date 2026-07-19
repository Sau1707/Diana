"""Participant-block calibration of Diana-H3P research prediction intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.v1_personalization import PersonalizationPlan
from benchmark.v1_task import HORMONES
from model.diana_h3p.contracts import BUDGETS, Layer2Parameters
from model.diana_h3p.layer2 import fit_layer2_core, predict_with_layer2
from model.v1_common import participant_balanced_weights, weighted_quantile


def _rows_for_ids(frame: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    indexed = frame.assign(sample_id=frame["sample_id"].astype(str)).set_index("sample_id")
    missing = sorted(set(sample_ids) - set(indexed.index))
    if missing:
        raise ValueError("Development OOF table is missing authorized samples")
    return indexed.loc[sample_ids].reset_index()


def _calibration_rows(
    layer1_oof: pd.DataFrame,
    plan: PersonalizationPlan,
    participant: str,
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
    ids = plan.sample_ids_for_budget(participant, int(budget))
    rows = _rows_for_ids(layer1_oof, ids)
    if len(rows) != int(budget):
        raise ValueError(f"Interval calibration requires exact K={budget}")
    return rows


def learn_interval_multipliers(
    layer1_oof: pd.DataFrame,
    plan: PersonalizationPlan,
    *,
    backend: object,
    quantile: float = 0.80,
    absolute_floor: float = 1e-10,
    relative_floor: float = 1e-6,
    near_diagonal_threshold: float = 0.05,
) -> tuple[dict[int, dict[str, float]], dict[str, object]]:
    """Learn K-specific multipliers with leave-one-participant Layer-2 fits."""

    participants = sorted(layer1_oof["private_participant_id"].astype(str).unique())
    scoring = plan.scoring_rows.copy()
    scoring["private_participant_id"] = scoring["private_participant_id"].astype(str)
    if sorted(scoring["private_participant_id"].unique()) != participants:
        raise ValueError("Interval calibration plan must cover every development participant")
    collected: dict[int, list[pd.DataFrame]] = {budget: [] for budget in BUDGETS}
    for held in participants:
        training = layer1_oof.loc[
            ~layer1_oof["private_participant_id"].astype(str).eq(held)
        ].copy()
        core = fit_layer2_core(
            training,
            plan,
            absolute_floor=absolute_floor,
            relative_floor=relative_floor,
            near_diagonal_threshold=near_diagonal_threshold,
        )
        scoring_ids = scoring.loc[
            scoring["private_participant_id"].eq(held), "sample_id"
        ].astype(str).tolist()
        base = _rows_for_ids(layer1_oof, scoring_ids)
        for budget in BUDGETS:
            calibration = _calibration_rows(layer1_oof, plan, held, budget)
            placeholder = {
                candidate: {hormone: 1.0 for hormone in HORMONES}
                for candidate in BUDGETS
            }
            parameters = Layer2Parameters(
                core=core, interval_multipliers=placeholder
            )
            predicted, posterior = predict_with_layer2(
                base,
                calibration,
                parameters,
                budget=budget,
                backend=backend,
                include_intervals=False,
            )
            covariance = (
                core.sigma_future.matrix
                + posterior["posterior_covariances"][0]
            )
            standard_deviation = np.sqrt(
                np.maximum(np.diag(covariance), max(absolute_floor, 1e-15))
            )
            records = pd.DataFrame(
                {
                    "private_participant_id": held,
                    **{
                        hormone: np.abs(
                            predicted[f"y_{hormone}"].to_numpy(float)
                            - predicted[f"pred_{hormone}"].to_numpy(float)
                        )
                        / standard_deviation[index]
                        for index, hormone in enumerate(HORMONES)
                    },
                }
            )
            collected[budget].append(records)
    multipliers: dict[int, dict[str, float]] = {}
    calibration_rows: dict[str, int] = {}
    for budget in BUDGETS:
        rows = pd.concat(collected[budget], ignore_index=True)
        weights = participant_balanced_weights(rows["private_participant_id"])
        multipliers[budget] = {
            hormone: weighted_quantile(
                rows[hormone].to_numpy(float), weights, float(quantile)
            )
            for hormone in HORMONES
        }
        calibration_rows[str(budget)] = len(rows)
    diagnostics: dict[str, object] = {
        "method": "leave_one_participant_layer2_participant_balanced_weighted_quantile",
        "quantile": float(quantile),
        "participants": len(participants),
        "rows_by_budget": calibration_rows,
    }
    return multipliers, diagnostics
