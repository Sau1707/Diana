from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.contracts import HORMONES, TARGET_LOG_COLUMNS
from benchmark.metrics import (
    add_reference_comparison,
    calculate_metrics,
    train_log1p_iqr_scales,
)


def _joined() -> pd.DataFrame:
    rows = []
    # Participant A contributes one date with error 2; participant B contributes
    # three dates with zero error. Participant-macro MAE is therefore 1, not 0.5.
    for hormone in HORMONES:
        rows.append(
            {
                "private_participant_id": "A",
                "sample_id": f"A-{hormone}",
                "hormone": hormone,
                "y_true_log1p": 2.0,
                "y_true_raw": np.expm1(2.0),
                "y_pred": 0.0,
            }
        )
        for index in range(3):
            rows.append(
                {
                    "private_participant_id": "B",
                    "sample_id": f"B-{index}-{hormone}",
                    "hormone": hormone,
                    "y_true_log1p": 1.0,
                    "y_true_raw": np.expm1(1.0),
                    "y_pred": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_primary_metric_is_participant_macro_not_row_micro() -> None:
    result = calculate_metrics(_joined(), {h: 2.0 for h in HORMONES})
    for hormone in HORMONES:
        primary = result.public["primary"][hormone]
        assert primary["participant_macro_log1p_mae"] == 1.0
        assert primary["participant_median_log1p_mae"] == 1.0
        assert primary["participant_min_log1p_mae"] == 0.0
        assert primary["participant_max_log1p_mae"] == 2.0
        assert result.public["normalized_log1p_mae"][hormone] == 0.5
    assert result.public["overall_normalized_score"] == 0.5


def test_robust_scales_use_provided_training_truth_and_omit_unstable_composite() -> None:
    train = pd.DataFrame(
        {
            TARGET_LOG_COLUMNS["lh"]: [0.0, 1.0, 2.0, 3.0],
            TARGET_LOG_COLUMNS["e3g"]: [2.0, 2.0, 2.0, 2.0],
            TARGET_LOG_COLUMNS["pdg"]: [0.0, 2.0, 4.0, 6.0],
        }
    )
    scales = train_log1p_iqr_scales(train)
    assert scales == {"lh": 1.5, "e3g": None, "pdg": 3.0}
    result = calculate_metrics(_joined(), scales)
    assert result.public["normalized_log1p_mae"]["e3g"] is None
    assert result.public["overall_normalized_score"] is None


def test_reference_skill_and_paired_improvement_counts() -> None:
    reference = calculate_metrics(_joined(), {h: 1.0 for h in HORMONES})
    better_joined = _joined()
    better_joined["y_pred"] = better_joined["y_true_log1p"]
    better = calculate_metrics(better_joined, {h: 1.0 for h in HORMONES})
    add_reference_comparison(
        better.public,
        better.participant,
        reference.public,
        reference.participant,
    )
    for hormone in HORMONES:
        assert better.public["skill_relative_to_causal_calendar"][hormone] == 1.0
        # A improves; B was already perfect and ties.
        assert better.public["participants_improved_vs_causal_calendar"][hormone] == {
            "count": 1,
            "out_of": 2,
        }

