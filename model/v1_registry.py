"""The exact active v1 baseline and custom-reference registries."""

from __future__ import annotations

from typing import Any, Callable

from model.catboost.v1_model import CatBoostV1
from model.population_median.v1_model import PopulationMedianV1
from model.wearable_ridge import WearableRidgeV1


BaselineFactory = Callable[[dict[str, Any]], object]

BASELINE_REGISTRY: dict[str, BaselineFactory] = {
    "population_median": PopulationMedianV1,
    "wearable_ridge": WearableRidgeV1,
    "catboost": CatBoostV1,
}
CUSTOM_REFERENCES = ("diana_h3p",)


def active_baselines() -> tuple[str, ...]:
    return tuple(BASELINE_REGISTRY)


def create_v1_baseline(name: str, config: dict[str, Any]) -> object:
    try:
        return BASELINE_REGISTRY[name](config)
    except KeyError as error:
        raise KeyError(f"Unknown v1 baseline {name!r}") from error


def custom_references() -> tuple[str, ...]:
    return CUSTOM_REFERENCES
