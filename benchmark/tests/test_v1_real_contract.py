from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.v1_contracts import load_v1_bundle
from benchmark.v1_task import load_v1_config, project_path


def test_real_v1_eligibility_and_leakage_invariants_when_prepared():
    config = load_v1_config("configs/hormonbench_v1.yaml")
    prepared = project_path(config, "prepared_dir")
    if not (prepared / "prepared.csv").is_file():
        pytest.skip("private governed-data integration bundle has not been prepared")
    bundle = load_v1_bundle(prepared)
    assert len(bundle.frame) == 1509
    assert bundle.frame["private_participant_id"].nunique() == 20
    columns = set(bundle.feature_columns)
    assert not any("self_report" in name or "flow_" in name for name in columns)
    assert "days_since_last_known_menses" not in columns
    assert "menses_onset_missing" not in columns
    assert not {"origin_day", "target_day", "sample_id", "private_participant_id"} & columns
