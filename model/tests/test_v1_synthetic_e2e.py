from __future__ import annotations

from benchmark.v1_task import load_v1_config
from model.diana_h3p.contracts import load_h3p_config
from model.diana_h3p.synthetic import run_synthetic_h3p


def test_synthetic_five_fold_end_to_end(tmp_path):
    benchmark_config = load_v1_config("configs/hormonbench_v1.yaml")
    h3p_config = load_h3p_config("configs/diana_h3p_v1.yaml")
    result = run_synthetic_h3p(benchmark_config, h3p_config, tmp_path)
    assert result["baseline_manifest_entries"] == 80
    assert result["h3p_manifest_entries"] == 80
    assert result["metric_rows"] == 16
    assert result["eligible_participants"] == 20
    assert result["eligible_origins"] == 240
    assert result["common_suffix_origins"] == 100
    assert result["privacy"]["status"] == "passed"
    results_dir = tmp_path / "results" / "v1" / "diana_h3p"
    assert (results_dir / "metrics.json").is_file()
    assert (results_dir / "run_manifest.json").is_file()
    assert (results_dir / "figures" / "measurement_budget_curve.svg").is_file()
