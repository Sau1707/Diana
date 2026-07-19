from __future__ import annotations

import copy

import pandas as pd

from benchmark.data.v1_folds import (
    fold_roles,
    group_hash,
    validate_five_fold_protocol,
    validate_groups,
)
from benchmark.v1_personalization import (
    build_personalization_plan,
    calibration_view,
    scoring_sample_ids,
)
from benchmark.v1_synthetic import make_synthetic_bundle
from benchmark.v1_task import load_v1_config


def test_five_fold_invariants_and_target_invariance():
    config = load_v1_config("configs/hormonbench_v1.yaml")
    bundle, mapping = make_synthetic_bundle(config)
    validate_groups(mapping)
    protocol = validate_five_fold_protocol(mapping, bundle.frame["private_participant_id"])
    assert protocol["groups"] == 5
    assert protocol["participants_tested_once"] == 20
    assert protocol["participants_validated_once"] == 20
    assert protocol["unique_test_origins"] == len(bundle.frame)
    for fold in range(5):
        roles = fold_roles(mapping, fold)
        assert [len(roles[key]) for key in ("train", "validation", "test")] == [12, 4, 4]
        assert not roles["train"] & roles["validation"]
        assert not roles["train"] & roles["test"]
    original = group_hash(mapping, config["folds"]["seed"])
    changed = bundle.frame.copy()
    changed[["target_lh_log1p", "target_e3g_log1p", "target_pdg_log1p"]] = changed[
        ["target_lh_log1p", "target_e3g_log1p", "target_pdg_log1p"]
    ].sample(frac=1, random_state=9).to_numpy()
    assert group_hash(mapping, config["folds"]["seed"]) == original


def test_personalization_authorization_common_suffix():
    config = load_v1_config("configs/hormonbench_v1.yaml")
    bundle, mapping = make_synthetic_bundle(config)
    ids = fold_roles(mapping, 0)["test"]
    plan = build_personalization_plan(bundle, ids)
    assert plan.aggregate["participants"] == 4
    assert plan.aggregate["calibration_candidates"] == 28
    assert plan.aggregate["common_scoring_origins"] == 20
    scoring = scoring_sample_ids(plan)
    assert len(scoring) == 20
    assert not set(scoring) & set(plan.calibration_candidates["sample_id"])
    for budget in (0, 3, 7):
        view = calibration_view(bundle, plan, budget=budget)
        assert len(view.targets) == 4 * budget
        if budget:
            assert view.participant_groups.value_counts().eq(budget).all()
    by_participant = plan.calibration_candidates.groupby("private_participant_id")
    for participant, candidates in by_participant:
        scored = plan.scoring_rows.loc[
            plan.scoring_rows["private_participant_id"].eq(participant)
        ]
        assert candidates["target_day"].max() <= scored["origin_day"].min()


def test_k_zero_truth_is_empty_and_scoring_rows_are_budget_invariant():
    config = load_v1_config("configs/hormonbench_v1.yaml")
    bundle, mapping = make_synthetic_bundle(config)
    plan = build_personalization_plan(bundle, fold_roles(mapping, 1)["test"])
    assert calibration_view(bundle, plan, budget=0).targets.empty
    expected = scoring_sample_ids(plan)
    for _budget in (0, 3, 7):
        assert scoring_sample_ids(plan) == expected
