"""Participant-balanced preprocessing and numerical helpers for v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from benchmark.v1_contracts import V1FitView
from benchmark.v1_task import HORMONES


def participant_balanced_weights(groups: pd.Series) -> np.ndarray:
    labels = groups.astype(str).reset_index(drop=True)
    if labels.empty:
        raise ValueError("Participant weights require at least one row")
    counts = labels.value_counts()
    participants = len(counts)
    weights = labels.map(
        {label: len(labels) / (participants * count) for label, count in counts.items()}
    ).to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Participant-balanced weights must be finite and positive")
    return weights


def assert_equal_participant_weight(groups: pd.Series, weights: np.ndarray) -> None:
    frame = pd.DataFrame(
        {"group": groups.astype(str).to_numpy(), "weight": np.asarray(weights, float)}
    )
    totals = frame.groupby("group", sort=True)["weight"].sum().to_numpy(float)
    if not np.allclose(totals, totals[0], rtol=1e-12, atol=1e-12):
        raise ValueError("Participants do not have equal total sample weight")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[keep]
    weights = weights[keep]
    if values.size == 0:
        raise ValueError("Weighted quantile has no finite values")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(float(quantile), cumulative, values))


def participant_balanced_iqr_scales(fit_view: V1FitView) -> dict[str, float | None]:
    fit_view.validate()
    weights = participant_balanced_weights(fit_view.participant_groups)
    scales: dict[str, float | None] = {}
    for hormone in HORMONES:
        values = fit_view.targets[hormone].to_numpy(float)
        q25 = weighted_quantile(values, weights, 0.25)
        q75 = weighted_quantile(values, weights, 0.75)
        scale = float(q75 - q25)
        scales[hormone] = scale if np.isfinite(scale) and scale > 1e-12 else None
    return scales


def combine_fit_views(*views: V1FitView) -> V1FitView:
    if not views:
        raise ValueError("At least one fit view is required")
    columns = tuple(views[0].X.columns)
    if any(tuple(view.X.columns) != columns for view in views):
        raise ValueError("Fit views must share identical feature order")
    combined = V1FitView(
        X=pd.concat([view.X for view in views], ignore_index=True),
        targets=pd.concat([view.targets for view in views], ignore_index=True),
        participant_groups=pd.concat(
            [view.participant_groups for view in views], ignore_index=True
        ),
        sample_ids=pd.concat([view.sample_ids for view in views], ignore_index=True),
    )
    combined.validate()
    return combined


@dataclass
class FeaturePreprocessor:
    missingness_drop_threshold: float = 0.95
    standardize: bool = False
    retained_columns: tuple[str, ...] = ()
    dropped_all_missing: tuple[str, ...] = ()
    dropped_high_missing: tuple[str, ...] = ()
    dropped_constant: tuple[str, ...] = ()
    medians: dict[str, float] | None = None
    scaler: StandardScaler | None = None
    fitted: bool = False

    def fit(self, X: pd.DataFrame, groups: pd.Series) -> "FeaturePreprocessor":
        if X.empty or X.shape[1] == 0:
            raise ValueError("Preprocessor requires a nonempty feature matrix")
        numeric = X.apply(pd.to_numeric, errors="coerce")
        missing = numeric.isna().mean()
        all_missing = [column for column in X.columns if missing[column] >= 1.0]
        high_missing = [
            column
            for column in X.columns
            if missing[column] >= self.missingness_drop_threshold
            and column not in all_missing
        ]
        candidates = [
            column
            for column in X.columns
            if column not in set(all_missing) | set(high_missing)
        ]
        constant = [
            column for column in candidates if numeric[column].nunique(dropna=True) <= 1
        ]
        retained = [column for column in candidates if column not in set(constant)]
        if not retained:
            raise ValueError("Feature filtering removed every column")
        medians: dict[str, float] = {}
        for column in retained:
            median = numeric[column].median(skipna=True)
            if pd.isna(median):
                raise ValueError(f"Retained feature {column} has no median")
            medians[column] = float(median)
        imputed = numeric.loc[:, retained].fillna(medians).astype(float)
        weights = participant_balanced_weights(groups)
        assert_equal_participant_weight(groups, weights)
        scaler: StandardScaler | None = None
        if self.standardize:
            scaler = StandardScaler()
            scaler.fit(imputed, sample_weight=weights)
        self.retained_columns = tuple(retained)
        self.dropped_all_missing = tuple(all_missing)
        self.dropped_high_missing = tuple(high_missing)
        self.dropped_constant = tuple(constant)
        self.medians = medians
        self.scaler = scaler
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.medians is None:
            raise RuntimeError("Preprocessor is not fitted")
        missing = sorted(set(self.retained_columns) - set(X.columns))
        if missing:
            raise ValueError(f"Inference features missing retained columns: {missing}")
        numeric = X.loc[:, self.retained_columns].apply(pd.to_numeric, errors="coerce")
        imputed = numeric.fillna(self.medians).astype(float)
        if self.scaler is None:
            return imputed.reset_index(drop=True)
        values = self.scaler.transform(imputed)
        return pd.DataFrame(values, columns=self.retained_columns)

    def metadata(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Preprocessor is not fitted")
        return {
            "retained_feature_count": len(self.retained_columns),
            "dropped_all_missing_count": len(self.dropped_all_missing),
            "dropped_high_missing_count": len(self.dropped_high_missing),
            "dropped_constant_count": len(self.dropped_constant),
            "missingness_drop_threshold": self.missingness_drop_threshold,
            "imputation": "fitting-data median",
            "standardization": "fitting-data weighted standard"
            if self.standardize
            else "none",
        }
