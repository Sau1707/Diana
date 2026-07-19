# CatBoost baseline

Registry name: `catboost`

## Purpose

CatBoost is the strong classical nonlinear tabular reference that a later model should be compared against under the unchanged Hormonbench contract.

## Inputs

It consumes the prepared causal features shared by every v0 run:

- end-of-day active-minute history;
- wake/end-aligned computed-temperature history;
- daily HRV aggregates;
- valid wake-day respiratory summaries;
- wake-day sleep-score fields, including that table's resting-heart-rate value;
- past self-reports known by the cutoff;
- causal days-since-last-known-menses state; and
- the known weekend indicator; and
- associated missingness masks, observation coverage, and time-since-observation features.

For each daily signal, the prepared table supplies latest value, mean, population standard deviation, minimum, maximum, causal slope, observed-day fraction, time since latest observation, current-day missing state, and fixed lags 0, 1, 3, 6, and 13. The feature window is days *t-13* through *t*. Numeric median imputation and any categorical ordinal mapping are fit on train only; unseen categories map to the documented unknown code. The model receives no hormone history, Mira phase/fertile-window label, participant-ID feature, static sensitive metadata, height/weight, future-filled value, or summary extending beyond the day-*t* cutoff.

## Fit

Fit one CPU CatBoost regressor per hormone in log1p space. Version 0 uses one fixed seed, bounded depth and iteration settings from `configs/hormonbench_v0.yaml`, four CPU threads, and validation early stopping. There is no cross-validation, repeated seed, or parameter sweep. Final parameters and each target's best iteration are included in private/model metadata and aggregate run metadata where safe.

If CatBoost remains unavailable after one bounded installation attempt, or fails after one single-thread CPU retry, the third family uses scikit-learn `HistGradientBoostingRegressor` and changes its reported model name to `hist_gradient_boosting`. Results never mislabel that fallback as CatBoost.

## Predict

The three regressors produce finite log1p predictions in the shared long-form schema. Model fitting and selection never access test truth, and evaluation remains exclusively in `benchmark/`.
