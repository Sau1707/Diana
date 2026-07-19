from __future__ import annotations

import ast
from pathlib import Path


def test_v1_evaluator_imports_no_model_package():
    source = Path("benchmark/v1_evaluator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name == "model" or name.startswith("model.") for name in imports)
