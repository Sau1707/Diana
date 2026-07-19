# Population median baseline

Registry name: `population_median`

## Purpose

This is the no-information lower bound. It measures the error obtained without wearable, self-report, calendar, participant, or origin-day predictors.

## Fit

For each of `lh`, `e3g`, and `pdg`, fit one scalar median using only that hormone's genuinely observed **training** targets in log1p space. Validation and test targets do not affect the fitted value.

## Predict

Return the corresponding fitted scalar for every required sample/hormone pair. Predictions use the shared long-form submission contract and remain in log1p space.

## Leakage boundary

The baseline never consumes participant ID, input features, phase labels, hormone history, or test truth. Its metadata records the task/model version and training target counts, but no participant IDs or individual target values are published.
