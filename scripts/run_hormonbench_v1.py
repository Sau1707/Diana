"""Single-process Hormonbench-mcPHASES v1 reproduction runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from benchmark.data.v1_adapter import prepare_v1
from benchmark.v1_evaluator import evaluate_v1
from benchmark.v1_privacy import validate_public_results
from benchmark.v1_report import report_v1
from benchmark.v1_task import load_v1_config
from model.v1_pipeline import run_fold0_validation, run_official_five_folds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hormonbench_v1.yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--privacy-only", action="store_true")
    args = parser.parse_args()
    config = load_v1_config(PROJECT / args.config)
    started = time.perf_counter()
    if args.privacy_only:
        print(json.dumps(validate_public_results(config), indent=2, sort_keys=True))
        return 0
    if args.evaluate_only:
        evaluate_v1(config)
        report_v1(config)
        print(json.dumps(validate_public_results(config), indent=2, sort_keys=True))
        return 0
    prepare = prepare_v1(config)
    print(
        f"prepared {prepare['eligible_origins']} origins / "
        f"{prepare['eligible_participants']} participants in "
        f"{prepare['prepare_seconds']:.2f}s",
        flush=True,
    )
    if args.prepare_only:
        return 0
    if args.validate_only:
        result = run_fold0_validation(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if config["custom"]["selected_covariance_mode"] == "pending":
        raise RuntimeError(
            "Run --validate-only, freeze selected_covariance_mode, then run canonical evaluation"
        )
    run_official_five_folds(config)
    evaluate_v1(config)
    report_v1(config)
    privacy = validate_public_results(config)
    print(
        json.dumps(
            {
                "status": "complete",
                "total_seconds": time.perf_counter() - started,
                "privacy": privacy,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
