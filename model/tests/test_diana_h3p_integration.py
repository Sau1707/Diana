from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from benchmark.v1_task import load_v1_config
from model.diana_h3p.contracts import load_h3p_config
from model.diana_h3p.pipeline import (
    validate_baseline_reuse,
    validate_frozen_contract,
)
from model.v1_registry import active_baselines, custom_references


PHASE0_HASHES = {
    "reports/phase0/PHASE0_AUDIT.md": "c91dfc2b7efc0f4f1b60383b8538eed67da02fe1dabcf2626701abfbc72ff39c",
    "reports/phase0/schema_inventory.csv": "bb732a433608c6288a5d2e513bbd152b3e305105f8cdb22ad3f7465144023dad",
    "reports/phase0/target_coverage.csv": "bb28731415a8812b57201f6c688119fdbfcec04acd0e6919ef594ce4b4bdafe9",
    "reports/phase0/feasibility_summary.json": "76b228197d1627445036b4f1d69d62a2b4c2af48bf272a6c21648fee7c96ea8c",
    "scripts/phase0_audit.py": "80f4b7614559e206f68821fc3232c9f238799417354a06c99ccd6e8135636b21",
}


def test_active_registries_and_legacy_separation() -> None:
    assert active_baselines() == ("population_median", "wearable_ridge", "catboost")
    assert custom_references() == ("diana_h3p",)
    assert Path("model/joint_bayes_personalizer/model.py").is_file()


def test_evaluator_remains_model_independent() -> None:
    source = Path("benchmark/v1_evaluator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    assert not any(name == "model" or name.startswith("model.") for name in imports)


def test_real_frozen_contract_and_baseline_reuse_when_available() -> None:
    benchmark_config = load_v1_config("configs/hormonbench_v1.yaml")
    h3p_config = load_h3p_config("configs/diana_h3p_v1.yaml")
    if not Path(benchmark_config["_project_root"], benchmark_config["paths"]["prepared_dir"]).is_dir():
        pytest.skip("private governed prepared bundle unavailable")
    contract = validate_frozen_contract(benchmark_config, h3p_config)
    assert contract["eligible_participants"] == 20
    assert contract["eligible_origins"] == 1509
    assert contract["common_suffix_origins"] == 1369
    audit = validate_baseline_reuse(benchmark_config, h3p_config)
    assert audit["status"] == "passed"
    assert audit["baseline_entries"] == audit["byte_identical_entries"] == 60
    wrong = copy.deepcopy(h3p_config)
    wrong["expected_benchmark"]["fold_hash"] = "0" * 64
    with pytest.raises(RuntimeError, match="fold_hash"):
        validate_baseline_reuse(benchmark_config, wrong)


def test_phase0_hashes_unchanged() -> None:
    import hashlib

    for name, expected in PHASE0_HASHES.items():
        digest = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        assert digest == expected
