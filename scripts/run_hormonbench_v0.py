"""Run the complete three-baseline Hormonbench-mcPHASES v0 workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> float:
    started = time.perf_counter()
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=PROJECT, check=True)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hormonbench_v0.yaml")
    parser.add_argument("--quick", action="store_true", help="Use the frozen quick CatBoost iteration ceiling")
    args = parser.parse_args()
    config_path = (PROJECT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    private_checkpoint = PROJECT / config["paths"]["checkpoint_dir"]
    private_checkpoint.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    started = time.perf_counter()
    stages: dict[str, float] = {}
    stages["prepare_seconds"] = _run([python, "-m", "benchmark", "prepare", "--config", str(config_path)])
    model_command = [
        python, "-m", "model", "run", "--config", str(config_path),
        "--models", "population_median,causal_calendar,catboost",
    ]
    if args.quick:
        model_command.append("--quick")
    stages["models_seconds"] = _run(model_command)
    stages["evaluate_seconds"] = _run([python, "-m", "benchmark", "evaluate", "--config", str(config_path)])
    stages["total_before_report_seconds"] = time.perf_counter() - started
    runtime_path = private_checkpoint / "orchestrator_runtime.json"
    runtime_path.write_text(json.dumps(stages, indent=2), encoding="utf-8")
    stages["report_seconds"] = _run([python, "-m", "benchmark", "report", "--config", str(config_path)])
    stages["total_seconds"] = time.perf_counter() - started
    runtime_path.write_text(json.dumps(stages, indent=2), encoding="utf-8")
    print("\nHormonbench v0 complete")
    print(json.dumps(stages, indent=2))


if __name__ == "__main__":
    main()
