"""Diana-H3P development, canonical, evaluation, and synthetic runner."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from benchmark.h3p_report import report_h3p
from benchmark.v1_evaluator import evaluate_v1
from benchmark.v1_privacy import validate_public_results
from benchmark.v1_task import load_v1_config
from model.diana_h3p.contracts import load_h3p_config
from model.diana_h3p.layer1 import select_stack_weights
from model.diana_h3p.pipeline import (
    evaluation_config,
    run_development_fold,
    run_official_h3p,
    validate_baseline_reuse,
    validate_frozen_contract,
)
from model.diana_h3p.profiling import profile_layer2_backends
from model.diana_h3p.synthetic import run_synthetic_h3p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", default="configs/hormonbench_v1.yaml")
    parser.add_argument("--model-config", default="configs/diana_h3p_v1.yaml")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--privacy-only", action="store_true")
    args = parser.parse_args()
    benchmark_config = load_v1_config(PROJECT / args.benchmark_config)
    h3p_config = load_h3p_config(PROJECT / args.model_config)
    if args.synthetic:
        with tempfile.TemporaryDirectory(prefix="diana_h3p_synthetic_") as directory:
            print(
                json.dumps(
                    run_synthetic_h3p(benchmark_config, h3p_config, directory),
                    indent=2,
                    sort_keys=True,
                )
            )
        return 0
    evaluator_config = evaluation_config(benchmark_config, h3p_config)
    if args.verify_only:
        print(
            json.dumps(
                {
                    "contract": validate_frozen_contract(
                        benchmark_config, h3p_config
                    ),
                    "baseline_reuse": validate_baseline_reuse(
                        benchmark_config, h3p_config
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.development_only:
        development = run_development_fold(
            benchmark_config, h3p_config, fold=0, write_private=True
        )
        stack = select_stack_weights(
            development["oof"],
            step=float(h3p_config["layer1"]["simplex_step"]),
            tie_tolerance=float(h3p_config["layer1"]["tie_tolerance"]),
        )
        profile = profile_layer2_backends(
            development["layer1_oof"],
            development["plan"],
            stack,
            h3p_config,
        )
        destination = (
            Path(h3p_config["_project_root"])
            / h3p_config["paths"]["private_run_root"]
        ).parent / "development" / "backend_profile.json"
        destination.write_text(
            json.dumps(profile, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"diagnostics": development["diagnostics"], "backend": profile},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.evaluate_only:
        evaluate_v1(evaluator_config)
        report_h3p(evaluator_config, h3p_config)
        print(json.dumps(validate_public_results(evaluator_config), indent=2, sort_keys=True))
        return 0
    if args.privacy_only:
        print(json.dumps(validate_public_results(evaluator_config), indent=2, sort_keys=True))
        return 0
    run_official_h3p(benchmark_config, h3p_config)
    evaluate_v1(evaluator_config)
    report_h3p(evaluator_config, h3p_config)
    print(json.dumps(validate_public_results(evaluator_config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
