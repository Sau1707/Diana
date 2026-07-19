"""High-level Diana-H3P fitting and prediction interface."""

from __future__ import annotations

from typing import Any

import pandas as pd

from benchmark.v1_personalization import PersonalizationPlan
from model.diana_h3p.contracts import H3PParameters, Layer2Parameters, StackSelection
from model.diana_h3p.layer2 import fit_layer2_core, predict_with_layer2
from model.diana_h3p.layer2_backend import create_backend
from model.diana_h3p.uncertainty import learn_interval_multipliers


def fit_h3p_parameters(
    layer1_oof: pd.DataFrame,
    plan: PersonalizationPlan,
    stack: StackSelection,
    *,
    backend_name: str,
    backend_device: str | None,
    seed: int,
    quantile: float = 0.80,
    absolute_floor: float = 1e-10,
    relative_floor: float = 1e-6,
    near_diagonal_threshold: float = 0.05,
) -> tuple[H3PParameters, dict[str, Any]]:
    """Fit fold-local Layer-2 parameters from grouped development OOF rows."""

    backend = create_backend(backend_name, device=backend_device)
    core = fit_layer2_core(
        layer1_oof,
        plan,
        absolute_floor=absolute_floor,
        relative_floor=relative_floor,
        near_diagonal_threshold=near_diagonal_threshold,
    )
    multipliers, calibration_diagnostics = learn_interval_multipliers(
        layer1_oof,
        plan,
        backend=backend,
        quantile=quantile,
        absolute_floor=absolute_floor,
        relative_floor=relative_floor,
        near_diagonal_threshold=near_diagonal_threshold,
    )
    backend_identity = (
        "numpy" if backend.name == "numpy" else f"torch_{backend.device}"
    )
    parameters = H3PParameters(
        stack=stack,
        layer2=Layer2Parameters(core=core, interval_multipliers=multipliers),
        backend=backend_identity,
        seed=int(seed),
    )
    parameters.validate()
    diagnostics = {
        "interval_calibration": calibration_diagnostics,
        "backend": {"name": backend.name, "device": backend.device},
    }
    return parameters, diagnostics


class DianaH3P:
    """A fitted fold-local H3P Layer-2 predictor over a fixed Layer-1 prior."""

    model_name = "diana_h3p"
    model_version = "1.0.0"

    def __init__(self, parameters: H3PParameters):
        parameters.validate()
        self.parameters = parameters
        device = parameters.backend.removeprefix("torch_") if parameters.backend.startswith("torch_") else None
        backend_name = "torch" if parameters.backend.startswith("torch_") else "numpy"
        self.backend = create_backend(backend_name, device=device)

    def predict(
        self,
        layer1_rows: pd.DataFrame,
        calibration: pd.DataFrame,
        *,
        budget: int,
        include_intervals: bool = True,
    ) -> pd.DataFrame:
        predicted, _ = predict_with_layer2(
            layer1_rows,
            calibration,
            self.parameters.layer2,
            budget=int(budget),
            backend=self.backend,
            include_intervals=include_intervals,
        )
        return predicted

    def get_metadata(self) -> dict[str, Any]:
        core = self.parameters.layer2.core
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "backend": self.parameters.backend,
            "seed": self.parameters.seed,
            "stack_weights": self.parameters.stack.weights,
            "stack_oof_participant_macro_log1p_mae": self.parameters.stack.participant_macro_mae,
            "covariance": {
                "sigma_a": core.sigma_a.public_metadata(),
                "psi_3": core.psi[3].public_metadata(),
                "psi_7": core.psi[7].public_metadata(),
                "sigma_future": core.sigma_future.public_metadata(),
            },
            "interval_multipliers": self.parameters.layer2.interval_multipliers,
            "development_participants": core.development_participants,
            "target_space": "log1p",
            "research_intervals_only": True,
        }
