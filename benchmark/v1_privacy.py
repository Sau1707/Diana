"""Public artifact and allow-list release validation for Diana."""

from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from benchmark.v1_task import project_path


FORBIDDEN_PATH_PARTS = (
    ".git",
    "dataset",
    "artifacts/private",
    ".pytest_cache",
    "__pycache__",
)
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".zip")
PUBLIC_ROOTS = (
    "benchmark/",
    "model/",
    "configs/",
    "scripts/",
    "reports/",
    "results/",
    "docs/",
)
PUBLIC_ROOT_FILES = {"README.md", "LICENSE", "pyproject.toml", ".gitignore"}
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
ABSOLUTE_PRIVATE_PATH = re.compile(r"[A-Za-z]:\\(?:Users|dataset|artifacts)\\", re.I)


def release_allowlist_inventory(project_root: str | Path) -> list[str]:
    root = Path(project_root)
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    files = sorted(line.replace("\\", "/") for line in result.stdout.splitlines())
    bad: list[str] = []
    for name in files:
        lower = name.lower()
        parts = lower.split("/")
        forbidden_path = (
            ".git" in parts
            or ".pytest_cache" in parts
            or "__pycache__" in parts
            or (parts and parts[0] == "dataset")
            or lower.startswith("artifacts/private/")
        )
        if forbidden_path:
            bad.append(name)
        elif lower.endswith(FORBIDDEN_SUFFIXES):
            bad.append(name)
        elif name not in PUBLIC_ROOT_FILES and not name.startswith(PUBLIC_ROOTS):
            bad.append(name)
    if bad:
        raise ValueError(f"Release allow-list rejected: {bad}")
    return files


def validate_public_inventory(
    project_root: str | Path, files: list[str]
) -> dict[str, Any]:
    """Scan an explicit working-tree or staged public inventory."""

    root = Path(project_root)
    normalized = sorted({str(name).replace("\\", "/") for name in files})
    rejected: list[str] = []
    scanned_text = 0
    for name in normalized:
        lower = name.lower()
        parts = lower.split("/")
        if (
            ".git" in parts
            or ".pytest_cache" in parts
            or "__pycache__" in parts
            or (parts and parts[0] == "dataset")
            or lower.startswith("artifacts/private/")
            or lower.endswith(FORBIDDEN_SUFFIXES)
            or name not in PUBLIC_ROOT_FILES
            and not name.startswith(PUBLIC_ROOTS)
        ):
            rejected.append(name)
            continue
        path = root / name
        if not path.is_file():
            rejected.append(f"missing:{name}")
            continue
        if path.suffix.lower() not in {".py", ".toml", ".yaml", ".yml", ".md", ".json", ".csv", ".svg", ".txt"} and path.name not in {"LICENSE", ".gitignore"}:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        scanned_text += 1
        if ABSOLUTE_PRIVATE_PATH.search(text) or any(
            pattern.search(text) for pattern in SECRET_PATTERNS
        ):
            rejected.append(name)
            continue
        if name.startswith("results/"):
            if re.search(r'"(?:private_participant_id|sample_id|y_true(?:_log1p)?)"\s*:', text):
                rejected.append(name)
                continue
            if path.suffix.lower() == ".csv":
                columns = set(pd.read_csv(path, nrows=0).columns)
                if columns & {
                    "private_participant_id",
                    "participant_id",
                    "sample_id",
                    "y_true",
                    "y_true_log1p",
                }:
                    rejected.append(name)
    if rejected:
        raise ValueError(f"Public inventory rejected: {sorted(rejected)}")
    return {
        "status": "passed",
        "files_checked": len(normalized),
        "text_files_scanned": scanned_text,
        "rejected": 0,
    }


def staged_inventory(project_root: str | Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line.replace("\\", "/") for line in result.stdout.splitlines())


def validate_staged_index(project_root: str | Path) -> dict[str, Any]:
    files = staged_inventory(project_root)
    if not files:
        raise ValueError("Staged-index privacy scan requires explicitly staged files")
    return validate_public_inventory(project_root, files)


def validate_git_archive(path: str | Path) -> dict[str, Any]:
    """Validate a temporary candidate ZIP produced by `git archive`."""

    archive = Path(path)
    rejected: list[str] = []
    with zipfile.ZipFile(archive) as handle:
        names = sorted(name.rstrip("/") for name in handle.namelist() if not name.endswith("/"))
        for name in names:
            lower = name.lower()
            parts = lower.split("/")
            if (
                ".git" in parts
                or ".pytest_cache" in parts
                or "__pycache__" in parts
                or (parts and parts[0] == "dataset")
                or lower.startswith("artifacts/private/")
                or lower.endswith(FORBIDDEN_SUFFIXES)
            ):
                rejected.append(name)
                continue
            if name.startswith("results/") and Path(name).suffix.lower() in {".json", ".csv", ".md", ".svg"}:
                text = handle.read(name).decode("utf-8")
                if ABSOLUTE_PRIVATE_PATH.search(text) or re.search(
                    r'"(?:private_participant_id|sample_id|y_true(?:_log1p)?)"\s*:',
                    text,
                ):
                    rejected.append(name)
        if rejected:
            raise ValueError(f"Git archive privacy scan rejected: {sorted(rejected)}")
    return {"status": "passed", "files_checked": len(names), "rejected": 0}


def validate_public_results(config: Mapping[str, Any]) -> dict[str, Any]:
    results_dir = project_path(config, "results_dir")
    required = [
        results_dir / "run_manifest.json",
        results_dir / "metrics.json",
        results_dir / "RESULTS.md",
        results_dir / "cold_start" / "leaderboard.csv",
        results_dir / "cold_start" / "fold_metrics.csv",
        results_dir / "few_shot" / "leaderboard_by_budget.csv",
        results_dir / "figures" / "cold_start_leaderboard.svg",
        results_dir / "figures" / "measurement_budget_curve.svg",
        results_dir / "figures" / "uncertainty_summary.svg",
    ]
    if (results_dir / "ablations").exists():
        required.append(results_dir / "ablations" / "development_only.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing public v1 outputs: {missing}")
    forbidden_columns = {
        "private_participant_id",
        "participant_id",
        "sample_id",
        "y_true",
        "y_true_log1p",
        "target_lh_raw",
        "target_e3g_raw",
        "target_pdg_raw",
    }
    for path in results_dir.rglob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if columns & forbidden_columns:
            raise ValueError(f"Public CSV exposes private columns: {path.name}")
    content_patterns = (
        re.compile(r"artifacts[\\/]private", re.I),
        re.compile(r"dataset[\\/]mcphases", re.I),
        re.compile(r"[A-Za-z]:\\Users\\", re.I),
        re.compile(r'"participant_to_(?:group|split)"', re.I),
    )
    for path in required:
        if path.suffix.lower() not in {".json", ".md", ".csv", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in content_patterns:
            if pattern.search(text):
                raise ValueError(f"Public artifact contains private content: {path.name}")
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if set(manifest["baselines"]) != {
        "population_median",
        "wearable_ridge",
        "catboost",
    }:
        raise ValueError("Public manifest baseline registry mismatch")
    if manifest.get("custom_reference") not in {
        "joint_bayes_personalizer",
        "diana_h3p",
    }:
        raise ValueError("Public manifest custom-reference mismatch")
    inventory = release_allowlist_inventory(config["_project_root"])
    inventory_scan = validate_public_inventory(config["_project_root"], inventory)
    return {
        "public_files_checked": len(required),
        "release_allowlist_files": len(inventory),
        "private_columns_found": 0,
        "private_paths_found": 0,
        "inventory_scan": inventory_scan,
        "status": "passed",
    }
