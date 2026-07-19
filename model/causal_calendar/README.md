# Causal calendar baseline

Registry name: `causal_calendar`

## Purpose

This transparent baseline asks how much next-day hormone variation can be predicted from menstrual timing known at the cutoff, without wearable features or hormone history.

## Features

For each origin, the model uses only:

1. `days_since_last_known_menses`, derived from the latest bleeding onset already reported by the end of day *t* and not truncated to the 14-day sensor window;
2. `missing_known_menses`, an internal explicit indicator derived from the prepared `menses_onset_missing` state when no prior onset is known;
3. a linear timing term;
4. a squared timing term; and
5. sine and cosine terms with a documented 28-day reference period.

The 28-day harmonic is a classical reference assumption, not a claim that every participant has a 28-day cycle. The linear and squared terms allow nonperiodic deviation. When onset is missing, numeric timing terms use the training split's median observed timing value and the missing indicator preserves that state.

## Model

Fit one Ridge regressor per hormone in log1p space with the configured fixed regularization. Any learned scaling is fit on train only. Validation/test data do not fit preprocessing or coefficients.

## Prohibited information

The model does not use completed cycle length, future bleeding, Mira phase/fertile-window labels, participant ID, wearable features, self-reports other than causally prior bleeding state, past hormones, or test truth.
