"""Deterministic participant-disjoint split generation."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
import pandas as pd


SPLIT_ORDER = ("train", "validation", "test")


def validate_participant_split(mapping: Mapping[int, str], sizes: Mapping[str, int]) -> None:
    if set(mapping.values()) != set(SPLIT_ORDER):
        raise ValueError("Split mapping must contain train, validation, and test")
    groups = {name: {int(pid) for pid, split in mapping.items() if split == name} for name in SPLIT_ORDER}
    for name in SPLIT_ORDER:
        if len(groups[name]) != int(sizes[name]):
            raise ValueError(f"{name} requires {sizes[name]} participants, got {len(groups[name])}")
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise ValueError("Participant overlap across splits")


def split_hash(mapping: Mapping[int, str], seed: int) -> str:
    payload = {"seed": int(seed), "mapping": {str(k): mapping[k] for k in sorted(mapping)}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate_fixed_split(
    participant_stats: pd.DataFrame,
    *,
    seed: int,
    sizes: Mapping[str, int],
    candidate_permutations: int = 20_000,
) -> tuple[dict[int, str], dict[str, float]]:
    """Balance a fixed split using coverage only—never hormone values.

    Required columns are ``private_participant_id``, ``eligible_origin_count``, and
    ``approved_modality_day_coverage``. A fixed-seed permutation search minimizes
    deviation from the requested origin shares and overall feature-coverage mean.
    This happens once before any model result exists.
    """

    required = {"private_participant_id", "eligible_origin_count", "approved_modality_day_coverage"}
    if not required <= set(participant_stats.columns):
        raise ValueError(f"participant_stats requires {sorted(required)}")
    stats = participant_stats.loc[:, sorted(required)].copy()
    stats["private_participant_id"] = stats["private_participant_id"].astype(int)
    stats = stats.sort_values("private_participant_id").reset_index(drop=True)
    n = len(stats)
    if n != sum(int(sizes[x]) for x in SPLIT_ORDER):
        raise ValueError("Participant count does not match requested split sizes")
    ids = stats["private_participant_id"].to_numpy(int)
    origins = stats["eligible_origin_count"].to_numpy(float)
    coverage = stats["approved_modality_day_coverage"].to_numpy(float)
    total_origins = origins.sum()
    global_coverage = float(np.nanmean(coverage))
    rng = np.random.default_rng(int(seed))
    cuts = np.cumsum([0] + [int(sizes[name]) for name in SPLIT_ORDER])
    best_score = np.inf
    best_tiebreak = None
    best_mapping = None
    best_parts = None
    trials = max(1, int(candidate_permutations))
    for _ in range(trials):
        order = rng.permutation(n)
        mapping: dict[int, str] = {}
        pieces = {}
        score = 0.0
        for i, name in enumerate(SPLIT_ORDER):
            idx = order[cuts[i] : cuts[i + 1]]
            pieces[name] = idx
            for pos in idx:
                mapping[int(ids[pos])] = name
            desired_share = float(sizes[name]) / n
            actual_share = float(origins[idx].sum() / total_origins)
            cov_mean = float(np.nanmean(coverage[idx]))
            score += (actual_share - desired_share) ** 2
            score += 0.05 * (cov_mean - global_coverage) ** 2
        tiebreak = tuple(tuple(sorted(int(ids[x]) for x in pieces[name])) for name in SPLIT_ORDER)
        if score < best_score - 1e-15 or (abs(score - best_score) <= 1e-15 and (best_tiebreak is None or tiebreak < best_tiebreak)):
            best_score = score
            best_tiebreak = tiebreak
            best_mapping = mapping
            best_parts = pieces
    assert best_mapping is not None and best_parts is not None
    validate_participant_split(best_mapping, sizes)
    diagnostics = {"objective": float(best_score), "candidate_permutations": trials}
    for name in SPLIT_ORDER:
        idx = best_parts[name]
        diagnostics[f"{name}_origins"] = int(origins[idx].sum())
        diagnostics[f"{name}_coverage"] = float(np.nanmean(coverage[idx]))
    return best_mapping, diagnostics

