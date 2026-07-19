from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from benchmark.v1_privacy import (
    release_allowlist_inventory,
    validate_git_archive,
    validate_public_inventory,
)


def test_working_tree_public_allowlist_passes() -> None:
    files = release_allowlist_inventory(Path.cwd())
    assert "pyproject.zip" not in files
    assert not any(name.startswith("artifacts/private/") for name in files)
    assert validate_public_inventory(Path.cwd(), files)["status"] == "passed"


def test_candidate_archive_rejects_private_paths(tmp_path: Path) -> None:
    safe = tmp_path / "safe.zip"
    with zipfile.ZipFile(safe, "w") as handle:
        handle.writestr("README.md", "public\n")
    assert validate_git_archive(safe)["status"] == "passed"

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as handle:
        handle.writestr("artifacts/private/predictions.csv", "sample_id\nsecret\n")
    with pytest.raises(ValueError, match="Git archive privacy scan rejected"):
        validate_git_archive(unsafe)
