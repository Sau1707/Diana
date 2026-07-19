"""Deterministic private serialization for fitted Diana-H3P parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from model.diana_h3p.contracts import (
    CovarianceEstimate,
    H3PParameters,
    Layer2Core,
    Layer2Parameters,
    StackSelection,
)


def _covariance_to_dict(value: CovarianceEstimate) -> dict[str, Any]:
    return {
        "matrix": np.asarray(value.matrix, dtype=float).tolist(),
        **value.public_metadata(),
    }


def _covariance_from_dict(value: dict[str, Any]) -> CovarianceEstimate:
    return CovarianceEstimate(
        matrix=np.asarray(value["matrix"], dtype=float),
        shrinkage=float(value["shrinkage"]),
        eigenvalues_before_floor=tuple(value["eigenvalues_before_floor"]),
        eigenvalues_after_floor=tuple(value["eigenvalues_after_floor"]),
        eigenvalue_floor=float(value["eigenvalue_floor"]),
        condition_number=float(value["condition_number"]),
        max_abs_off_diagonal_correlation=float(
            value["max_abs_off_diagonal_correlation"]
        ),
        effectively_near_diagonal=bool(value["effectively_near_diagonal"]),
        sample_count=int(value["sample_count"]),
        estimator=str(value["estimator"]),
    )


def parameters_to_dict(parameters: H3PParameters) -> dict[str, Any]:
    parameters.validate()
    core = parameters.layer2.core
    return {
        "schema_version": "1.0.0",
        "model_name": "diana_h3p",
        "model_version": parameters.model_version,
        "seed": int(parameters.seed),
        "backend": parameters.backend,
        "stack": {
            "weights": parameters.stack.weights,
            "participant_macro_mae": parameters.stack.participant_macro_mae,
            "grid_step": parameters.stack.grid_step,
        },
        "layer2": {
            "sigma_a": _covariance_to_dict(core.sigma_a),
            "psi_3": _covariance_to_dict(core.psi[3]),
            "psi_7": _covariance_to_dict(core.psi[7]),
            "sigma_future": _covariance_to_dict(core.sigma_future),
            "development_participants": core.development_participants,
            "interval_multipliers": parameters.layer2.interval_multipliers,
        },
    }


def parameters_from_dict(payload: dict[str, Any]) -> H3PParameters:
    if payload.get("model_name") != "diana_h3p":
        raise ValueError("Not a Diana-H3P parameter file")
    stack_payload = payload["stack"]
    layer2_payload = payload["layer2"]
    parameters = H3PParameters(
        stack=StackSelection(
            weights={
                hormone: {name: float(weight) for name, weight in values.items()}
                for hormone, values in stack_payload["weights"].items()
            },
            participant_macro_mae={
                hormone: float(value)
                for hormone, value in stack_payload["participant_macro_mae"].items()
            },
            grid_step=float(stack_payload["grid_step"]),
        ),
        layer2=Layer2Parameters(
            core=Layer2Core(
                sigma_a=_covariance_from_dict(layer2_payload["sigma_a"]),
                psi={
                    3: _covariance_from_dict(layer2_payload["psi_3"]),
                    7: _covariance_from_dict(layer2_payload["psi_7"]),
                },
                sigma_future=_covariance_from_dict(layer2_payload["sigma_future"]),
                development_participants=int(
                    layer2_payload["development_participants"]
                ),
            ),
            interval_multipliers={
                int(budget): {
                    hormone: float(value) for hormone, value in values.items()
                }
                for budget, values in layer2_payload["interval_multipliers"].items()
            },
        ),
        backend=str(payload["backend"]),
        seed=int(payload["seed"]),
        model_version=str(payload["model_version"]),
    )
    parameters.validate()
    return parameters


def save_parameters(parameters: H3PParameters, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(parameters_to_dict(parameters), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_parameters(path: str | Path) -> H3PParameters:
    return parameters_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
