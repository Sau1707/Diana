"""Typed internal contracts for Diana-H3P."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmark.v1_task import HORMONES


EXPERTS = ("population_median", "wearable_ridge", "catboost")
BUDGETS = (0, 3, 7)


def load_h3p_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Diana-H3P config must be a YAML mapping")
    if value.get("model", {}).get("id") != "diana_h3p":
        raise ValueError("Diana-H3P model identifier is frozen")
    if float(value["layer1"]["simplex_step"]) != 0.10:
        raise ValueError("Official Layer-1 simplex step is frozen at 0.10")
    if value["layer2"].get("covariance_estimator") != "continuous_ledoit_wolf_correlation":
        raise ValueError("H3P covariance estimator is frozen")
    value["_config_path"] = str(config_path)
    value["_project_root"] = str(config_path.parents[1])
    return value


def scientific_model_spec(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"],
        "expected_benchmark": config["expected_benchmark"],
        "layer1": config["layer1"],
        "layer2": config["layer2"],
        "uncertainty": config["uncertainty"],
        "backend_selection": {
            "minimum_speedup_fraction": config["backend"]["minimum_speedup_fraction"],
            "parity_rtol": config["backend"]["parity_rtol"],
            "parity_atol": config["backend"]["parity_atol"],
            "profile_repetitions": config["backend"]["profile_repetitions"],
            "canonical": config["backend"]["canonical"],
        },
        "seed": int(config["runtime"]["seed"]),
    }


def h3p_config_hash(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def h3p_model_spec_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        scientific_model_spec(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StackSelection:
    weights: dict[str, dict[str, float]]
    participant_macro_mae: dict[str, float]
    grid_step: float

    def validate(self) -> None:
        if set(self.weights) != set(HORMONES):
            raise ValueError("Stack weights require every hormone")
        for hormone, weights in self.weights.items():
            if set(weights) != set(EXPERTS):
                raise ValueError(f"Stack experts missing for {hormone}")
            values = np.asarray([weights[name] for name in EXPERTS], dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Stack weights must be finite and nonnegative")
            if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
                raise ValueError("Stack weights must sum to one")


@dataclass(frozen=True)
class CovarianceEstimate:
    matrix: np.ndarray
    shrinkage: float
    eigenvalues_before_floor: tuple[float, ...]
    eigenvalues_after_floor: tuple[float, ...]
    eigenvalue_floor: float
    condition_number: float
    max_abs_off_diagonal_correlation: float
    effectively_near_diagonal: bool
    sample_count: int
    estimator: str

    def validate(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("H3P covariance must be 3x3")
        if not np.isfinite(matrix).all() or not np.allclose(
            matrix, matrix.T, rtol=0.0, atol=1e-12
        ):
            raise ValueError("H3P covariance must be finite and symmetric")
        if float(np.linalg.eigvalsh(matrix).min()) < -1e-12:
            raise ValueError("H3P covariance must be positive semidefinite")
        if not 0.0 <= float(self.shrinkage) <= 1.0:
            raise ValueError("Shrinkage intensity must lie in [0, 1]")

    def public_metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "shrinkage": float(self.shrinkage),
            "eigenvalues_before_floor": [float(x) for x in self.eigenvalues_before_floor],
            "eigenvalues_after_floor": [float(x) for x in self.eigenvalues_after_floor],
            "eigenvalue_floor": float(self.eigenvalue_floor),
            "condition_number": float(self.condition_number),
            "max_abs_off_diagonal_correlation": float(
                self.max_abs_off_diagonal_correlation
            ),
            "effectively_near_diagonal": bool(self.effectively_near_diagonal),
            "sample_count": int(self.sample_count),
            "estimator": self.estimator,
        }


@dataclass(frozen=True)
class Layer2Core:
    sigma_a: CovarianceEstimate
    psi: dict[int, CovarianceEstimate]
    sigma_future: CovarianceEstimate
    development_participants: int

    def validate(self) -> None:
        self.sigma_a.validate()
        self.sigma_future.validate()
        if set(self.psi) != {3, 7}:
            raise ValueError("Layer 2 requires Psi_3 and Psi_7")
        for estimate in self.psi.values():
            estimate.validate()


@dataclass(frozen=True)
class Layer2Parameters:
    core: Layer2Core
    interval_multipliers: dict[int, dict[str, float]]

    def validate(self) -> None:
        self.core.validate()
        if set(self.interval_multipliers) != set(BUDGETS):
            raise ValueError("Intervals require K=0/3/7 multipliers")
        for values in self.interval_multipliers.values():
            if set(values) != set(HORMONES):
                raise ValueError("Interval multipliers require every hormone")
            array = np.asarray(list(values.values()), dtype=float)
            if not np.isfinite(array).all() or (array < 0).any():
                raise ValueError("Interval multipliers must be finite and nonnegative")


@dataclass(frozen=True)
class H3PParameters:
    stack: StackSelection
    layer2: Layer2Parameters
    backend: str
    seed: int
    model_version: str = "1.0.0"

    def validate(self) -> None:
        self.stack.validate()
        self.layer2.validate()
        if self.backend not in {"numpy", "torch_cpu", "torch_cuda"}:
            raise ValueError("Unknown H3P backend")
