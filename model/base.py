"""Minimal model-side interface for Hormonbench baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from benchmark.contracts import PreparedSplit


class HormonbenchModel(ABC):
    """Contract shared by every model that participates in Hormonbench v0."""

    @abstractmethod
    def fit(
        self, train_bundle: PreparedSplit, validation_bundle: PreparedSplit
    ) -> "HormonbenchModel":
        """Fit using train truth; validation truth is only for early stopping."""

    @abstractmethod
    def predict(self, test_bundle: PreparedSplit) -> pd.DataFrame:
        """Return a prediction-contract frame from an inference-only split."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return aggregate-safe reproducibility metadata."""

