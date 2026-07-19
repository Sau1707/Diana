"""Private calibration authorization and common-suffix construction for v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .v1_contracts import HORMONES, TARGET_LOG_COLUMNS, V1CalibrationView, V1PreparedBundle


@dataclass(frozen=True)
class PersonalizationPlan:
    calibration_candidates: pd.DataFrame
    scoring_rows: pd.DataFrame
    aggregate: dict[str, int]

    def sample_ids_for_budget(self, participant: str, budget: int) -> list[str]:
        rows = self.calibration_candidates.loc[
            self.calibration_candidates["private_participant_id"].astype(str).eq(
                str(participant)
            )
            & self.calibration_candidates["calibration_rank"].le(int(budget))
        ]
        return rows.sort_values("calibration_rank")["sample_id"].astype(str).tolist()


def build_personalization_plan(
    bundle: V1PreparedBundle,
    participant_ids: Iterable[int | str],
    *,
    common_budget: int = 7,
    minimum_scoring_origins: int = 1,
) -> PersonalizationPlan:
    bundle.validate()
    participant_set = {str(value) for value in participant_ids}
    rows = bundle.frame.loc[
        bundle.frame["private_participant_id"].astype(str).isin(participant_set)
    ].copy()
    if rows["private_participant_id"].astype(str).nunique() != len(participant_set):
        raise ValueError("Personalization participants are missing from the bundle")
    calibration_parts: list[pd.DataFrame] = []
    scoring_parts: list[pd.DataFrame] = []
    for participant, group in rows.groupby("private_participant_id", sort=True):
        ordered = group.sort_values(["target_day", "sample_id"]).reset_index(drop=True)
        if len(ordered) < int(common_budget):
            raise ValueError(
                "Every held-out participant must have at least seven eligible targets"
            )
        calibration = ordered.iloc[: int(common_budget)][
            ["sample_id", "private_participant_id", "origin_day", "target_day"]
        ].copy()
        calibration["calibration_rank"] = np.arange(1, int(common_budget) + 1)
        seventh_target_day = int(calibration.iloc[-1]["target_day"])
        scoring = ordered.loc[ordered["origin_day"].ge(seventh_target_day)].copy()
        scoring = scoring.loc[
            ~scoring["sample_id"].astype(str).isin(calibration["sample_id"].astype(str))
        ]
        if len(scoring) < int(minimum_scoring_origins):
            raise ValueError(
                "Held-out participant lacks the prespecified common-suffix scoring rows"
            )
        if int(calibration["target_day"].max()) > int(scoring["origin_day"].min()):
            raise ValueError("Calibration must be available before common-suffix scoring")
        calibration_parts.append(calibration)
        scoring_parts.append(
            scoring[["sample_id", "private_participant_id", "origin_day", "target_day"]]
        )
    candidates = pd.concat(calibration_parts, ignore_index=True)
    scoring_rows = pd.concat(scoring_parts, ignore_index=True)
    if set(candidates["sample_id"].astype(str)) & set(scoring_rows["sample_id"].astype(str)):
        raise ValueError("Calibration rows cannot be scored")
    aggregate = {
        "participants": len(participant_set),
        "calibration_candidates": len(candidates),
        "common_scoring_origins": len(scoring_rows),
        "common_budget": int(common_budget),
        "minimum_scoring_origins_per_participant": int(
            scoring_rows.groupby("private_participant_id").size().min()
        ),
    }
    return PersonalizationPlan(candidates, scoring_rows, aggregate)


def calibration_view(
    bundle: V1PreparedBundle,
    plan: PersonalizationPlan,
    *,
    budget: int,
) -> V1CalibrationView:
    budget = int(budget)
    participants = sorted(
        plan.calibration_candidates["private_participant_id"].astype(str).unique()
    )
    if budget == 0:
        view = V1CalibrationView(
            sample_ids=pd.Series([], dtype="string"),
            participant_groups=pd.Series([], dtype="string"),
            target_days=pd.Series([], dtype="int64"),
            targets=pd.DataFrame(columns=HORMONES, dtype=float),
            budget=0,
        )
        view.validate()
        return view
    authorized = plan.calibration_candidates.loc[
        plan.calibration_candidates["calibration_rank"].le(budget)
    ].sort_values(["private_participant_id", "calibration_rank"])
    if authorized["private_participant_id"].astype(str).nunique() != len(participants):
        raise ValueError("Calibration authorization lost a participant")
    private = bundle.frame.set_index("sample_id").loc[
        authorized["sample_id"].astype(str)
    ]
    view = V1CalibrationView(
        sample_ids=authorized["sample_id"].astype(str).reset_index(drop=True),
        participant_groups=authorized["private_participant_id"]
        .astype(str)
        .reset_index(drop=True),
        target_days=authorized["target_day"].astype(int).reset_index(drop=True),
        targets=pd.DataFrame(
            {
                hormone: private[TARGET_LOG_COLUMNS[hormone]].to_numpy(float)
                for hormone in HORMONES
            }
        ),
        budget=budget,
    )
    view.validate()
    return view


def scoring_sample_ids(plan: PersonalizationPlan) -> list[str]:
    return plan.scoring_rows.sort_values(
        ["private_participant_id", "origin_day", "sample_id"]
    )["sample_id"].astype(str).tolist()
