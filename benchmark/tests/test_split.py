import pandas as pd

from benchmark.data.splits import generate_fixed_split, split_hash, validate_participant_split


def _stats():
    return pd.DataFrame({
        "private_participant_id": range(100, 120),
        "eligible_origin_count": [20 + (i % 7) for i in range(20)],
        "approved_modality_day_coverage": [0.7 + (i % 5) / 20 for i in range(20)],
    })


def test_deterministic_disjoint_split():
    sizes = {"train": 12, "validation": 4, "test": 4}
    first, _ = generate_fixed_split(_stats(), seed=20260719, sizes=sizes, candidate_permutations=500)
    second, _ = generate_fixed_split(_stats(), seed=20260719, sizes=sizes, candidate_permutations=500)
    assert first == second
    validate_participant_split(first, sizes)
    assert split_hash(first, 20260719) == split_hash(second, 20260719)
    groups = [{pid for pid, split in first.items() if split == name} for name in ("train", "validation", "test")]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]

