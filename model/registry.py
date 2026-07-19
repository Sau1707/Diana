"""The deliberately small v0 baseline registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from model.base import HormonbenchModel
from model.catboost import CatBoostBaselineModel
from model.causal_calendar import CausalCalendarModel
from model.population_median import PopulationMedianModel


ModelFactory = Callable[[dict[str, Any], bool], HormonbenchModel]


def _population(config: dict[str, Any], quick: bool) -> HormonbenchModel:
    return PopulationMedianModel(config, quick=quick)


def _calendar(config: dict[str, Any], quick: bool) -> HormonbenchModel:
    return CausalCalendarModel(config, quick=quick)


def _catboost(config: dict[str, Any], quick: bool) -> HormonbenchModel:
    return CatBoostBaselineModel(config, quick=quick)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "population_median": _population,
    "causal_calendar": _calendar,
    "catboost": _catboost,
}


def available_models() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY)


def create_model(name: str, config: dict[str, Any], *, quick: bool = False) -> HormonbenchModel:
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as error:
        raise KeyError(
            f"Unknown model {name!r}; choose from {', '.join(available_models())}"
        ) from error
    return factory(config, quick)

