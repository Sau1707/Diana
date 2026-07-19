"""Fully synthetic prepared bundle shared by model tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.contracts import PreparedBundle


@pytest.fixture
def model_config() -> dict:
    return {
        "models": {
            "seed": 20260719,
            "causal_calendar": {
                "ridge_alpha": 1.0,
                "harmonic_period_days": 28.0,
                "polynomial_degree": 2,
            },
            "catboost": {
                "iterations": 16,
                "quick_iterations": 12,
                "depth": 3,
                "learning_rate": 0.08,
                "l2_leaf_reg": 3.0,
                "early_stopping_rounds": 3,
                "thread_count": 1,
            },
        }
    }


@pytest.fixture
def synthetic_bundle() -> PreparedBundle:
    rng = np.random.default_rng(7)
    rows = []
    splits = [("train", 24), ("validation", 9), ("test", 9)]
    row_number = 0
    for split, count in splits:
        for local_index in range(count):
            origin = 20 + local_index
            days = float((origin + row_number) % 34)
            if local_index % 8 == 0:
                days = np.nan
            active = float(20 + (local_index % 5) * 7)
            sleep = float(65 + (local_index % 7) * 3)
            flow = float(local_index % 4 == 0)
            latent_day = 15.0 if np.isnan(days) else days
            raw_lh = max(0.1, 2.0 + 0.06 * latent_day + 0.01 * active + rng.normal(0, 0.05))
            raw_e3g = max(0.1, 35.0 + 1.7 * latent_day + 0.04 * sleep + rng.normal(0, 0.2))
            raw_pdg = max(0.1, 1.0 + 0.14 * latent_day + 0.03 * flow + rng.normal(0, 0.03))
            rows.append(
                {
                    "task_version": "0.1.0",
                    "sample_id": f"synthetic-{row_number:04d}",
                    "private_participant_id": f"private-{split}-{local_index % 3}",
                    "origin_day": origin,
                    "target_day": origin + 1,
                    "history_start_day": origin - 13,
                    "history_end_day": origin,
                    "cutoff_day": origin,
                    "split": split,
                    "config_hash": "synthetic-config",
                    "split_hash": "synthetic-split",
                    "days_since_last_known_menses": days,
                    "active_minutes__mean": active,
                    "sleep_score__last": sleep,
                    "flow_volume__coverage": flow,
                    "target_lh_raw": raw_lh,
                    "target_e3g_raw": raw_e3g,
                    "target_pdg_raw": raw_pdg,
                    "target_lh_log1p": np.log1p(raw_lh),
                    "target_e3g_log1p": np.log1p(raw_e3g),
                    "target_pdg_log1p": np.log1p(raw_pdg),
                }
            )
            row_number += 1
    metadata = {
        "feature_columns": [
            "days_since_last_known_menses",
            "active_minutes__mean",
            "sleep_score__last",
            "flow_volume__coverage",
        ]
    }
    bundle = PreparedBundle(pd.DataFrame(rows), metadata)
    bundle.validate()
    return bundle

