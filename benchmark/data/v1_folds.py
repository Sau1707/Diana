"""Deterministic five-group participant protocol for Hormonbench v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from benchmark.v1_task import canonical_hash


GROUP_COUNT = 5
GROUP_SIZE = 4


def validate_groups(mapping: Mapping[int, int], expected_participants: int = 20) -> None:
    if len(mapping) != expected_participants:
        raise ValueError(f"Expected {expected_participants} participant assignments")
    groups = {
        group: {int(pid) for pid, assigned in mapping.items() if int(assigned) == group}
        for group in range(GROUP_COUNT)
    }
    if any(len(group) != GROUP_SIZE for group in groups.values()):
        raise ValueError("Every v1 participant group must contain exactly four participants")
    if set().union(*groups.values()) != {int(pid) for pid in mapping}:
        raise ValueError("Group assignment must cover every participant exactly once")
    for left in range(GROUP_COUNT):
        for right in range(left + 1, GROUP_COUNT):
            if groups[left] & groups[right]:
                raise ValueError("Participant groups overlap")


def group_hash(mapping: Mapping[int, int], seed: int) -> str:
    return canonical_hash(
        {
            "seed": int(seed),
            "participant_to_group": {
                str(pid): int(mapping[pid]) for pid in sorted(mapping)
            },
        }
    )


def build_v1_groups(
    participant_stats: pd.DataFrame,
    v0_participant_to_split: Mapping[str | int, str],
    *,
    seed: int,
    candidate_permutations: int,
) -> tuple[dict[int, int], dict[str, Any]]:
    required = {
        "private_participant_id",
        "eligible_origin_count",
        "approved_wearable_day_coverage",
    }
    if not required <= set(participant_stats.columns):
        raise ValueError(f"participant_stats missing {sorted(required-set(participant_stats))}")
    stats = participant_stats.loc[:, sorted(required)].copy()
    stats["private_participant_id"] = stats["private_participant_id"].astype(int)
    stats = stats.sort_values("private_participant_id").reset_index(drop=True)
    old = {int(pid): str(split) for pid, split in v0_participant_to_split.items()}
    if set(old) != set(stats["private_participant_id"]):
        raise ValueError("v0 split participants do not match v1 eligible participants")
    group0 = sorted(pid for pid, split in old.items() if split == "test")
    group1 = sorted(pid for pid, split in old.items() if split == "validation")
    train_ids = sorted(pid for pid, split in old.items() if split == "train")
    if len(group0) != 4 or len(group1) != 4 or len(train_ids) != 12:
        raise ValueError("v0 split must be 12 train / 4 validation / 4 test")
    indexed = stats.set_index("private_participant_id")
    origins = indexed.loc[train_ids, "eligible_origin_count"].to_numpy(float)
    coverage = indexed.loc[train_ids, "approved_wearable_day_coverage"].to_numpy(float)
    rng = np.random.default_rng(int(seed))
    best_score = np.inf
    best_tie: tuple[tuple[int, ...], ...] | None = None
    best_parts: list[np.ndarray] | None = None
    target_origin_share = 1.0 / 3.0
    target_coverage = float(np.mean(coverage))
    for _ in range(max(1, int(candidate_permutations))):
        order = rng.permutation(len(train_ids))
        parts = [order[index * 4 : (index + 1) * 4] for index in range(3)]
        score = 0.0
        for part in parts:
            score += (float(origins[part].sum() / origins.sum()) - target_origin_share) ** 2
            score += 0.05 * (float(np.mean(coverage[part])) - target_coverage) ** 2
        tie = tuple(tuple(sorted(train_ids[index] for index in part)) for part in parts)
        if score < best_score - 1e-15 or (
            abs(score - best_score) <= 1e-15
            and (best_tie is None or tie < best_tie)
        ):
            best_score = score
            best_tie = tie
            best_parts = parts
    if best_parts is None:
        raise RuntimeError("Unable to generate v1 participant groups")
    mapping = {pid: 0 for pid in group0} | {pid: 1 for pid in group1}
    for group, part in enumerate(best_parts, start=2):
        mapping.update({train_ids[index]: group for index in part})
    validate_groups(mapping)
    diagnostics: dict[str, Any] = {
        "objective": float(best_score),
        "candidate_permutations": int(candidate_permutations),
    }
    for group in range(GROUP_COUNT):
        ids = [pid for pid, assigned in mapping.items() if assigned == group]
        diagnostics[f"group_{group}_participants"] = len(ids)
        diagnostics[f"group_{group}_origins"] = int(
            indexed.loc[ids, "eligible_origin_count"].sum()
        )
        diagnostics[f"group_{group}_wearable_coverage"] = float(
            indexed.loc[ids, "approved_wearable_day_coverage"].mean()
        )
    return mapping, diagnostics


def fold_roles(mapping: Mapping[int, int], fold: int) -> dict[str, set[int]]:
    fold = int(fold)
    if fold not in range(GROUP_COUNT):
        raise ValueError("fold must be 0..4")
    test_group = fold
    validation_group = (fold + 1) % GROUP_COUNT
    roles = {
        "test": {int(pid) for pid, group in mapping.items() if group == test_group},
        "validation": {
            int(pid) for pid, group in mapping.items() if group == validation_group
        },
        "train": {
            int(pid)
            for pid, group in mapping.items()
            if group not in {test_group, validation_group}
        },
    }
    if tuple(len(roles[name]) for name in ("train", "validation", "test")) != (
        12,
        4,
        4,
    ):
        raise ValueError("Every outer fold must be 12/4/4")
    if (
        roles["train"] & roles["validation"]
        or roles["train"] & roles["test"]
        or roles["validation"] & roles["test"]
    ):
        raise ValueError("Outer fold participant roles overlap")
    return roles


def validate_five_fold_protocol(
    mapping: Mapping[int, int], sample_participants: pd.Series
) -> dict[str, Any]:
    validate_groups(mapping)
    seen_test: list[int] = []
    seen_validation: list[int] = []
    test_sample_count = 0
    for fold in range(GROUP_COUNT):
        roles = fold_roles(mapping, fold)
        seen_test.extend(roles["test"])
        seen_validation.extend(roles["validation"])
        test_sample_count += int(sample_participants.astype(int).isin(roles["test"]).sum())
    participants = sorted(mapping)
    if sorted(seen_test) != participants or sorted(seen_validation) != participants:
        raise ValueError("Each participant must test and validate exactly once")
    if test_sample_count != len(sample_participants):
        raise ValueError("Each eligible origin must appear in outer test exactly once")
    return {
        "groups": GROUP_COUNT,
        "participants_per_group": GROUP_SIZE,
        "participants_tested_once": len(participants),
        "participants_validated_once": len(participants),
        "unique_test_origins": int(test_sample_count),
    }


def write_private_fold_manifest(
    path: str | Path,
    *,
    mapping: Mapping[int, int],
    seed: int,
    digest: str,
    diagnostics: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    payload = {
        "seed": int(seed),
        "fold_hash": digest,
        "participant_to_group": {
            str(pid): int(mapping[pid]) for pid in sorted(mapping)
        },
        "diagnostics": dict(diagnostics),
        "protocol": dict(protocol),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
