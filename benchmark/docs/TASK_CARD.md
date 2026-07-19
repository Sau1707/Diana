# Hormonbench-mcPHASES v0 task card

## Identity and status

- **Task ID:** `hormonbench_mcphases_interval2_nextday_v0`
- **Task version:** `0.1.0`
- **Implemented track:** `primary_interval2_nextday`
- **Status:** frozen for the v0 baseline comparison

The benchmark is a forecasting research task, not a clinical prediction or diagnostic task.

## Prediction problem

For an origin day *t*, use exactly 14 calendar days of causally available wearable and self-report information, *t-13* through *t*, to predict genuinely measured at-home urinary LH, E3G, and PdG on *t+1*.

- Collection interval: Interval 2 (`study_interval == 2024`)
- History length: 14 days
- Forecast horizon: 1 day
- Targets: `lh`, `e3g` (source column `estrogen`), and `pdg`
- Training/evaluation space: `log1p`
- Label rule: a target must be genuinely observed; labels are never created by interpolation
- Feature missingness: allowed and represented with masks, coverage, and time-since-observation where supported

An eligible origin has the required consecutive calendar alignment and a genuinely observed three-hormone target day. Missing input modalities do not remove the origin when their absence can be represented safely. Preparation recomputes eligibility under this frozen, hormone-history-free contract and records aggregate counts.

## Temporal cutoff

The cutoff is the end of day *t*. No source event, completed summary, revision, or transformation may use information after that cutoff.

Overnight temperature, sleep, respiratory, and HRV information is aligned to the sleep-end/wake day. A session ending on day *t* may inform the *t+1* forecast; a session ending on *t+1* may not. `sleep_end_day_in_study` is used for computed temperature rather than the sleep-start day.

`days_since_last_known_menses` searches all causally prior known bleeding reports and is not truncated to the 14-day sensor window. An onset is recognized only when a positive-flow report follows a known nonpositive report; a first-ever observed positive report is not silently assumed to be an onset. If no prior onset is known, the timing value is missing and an explicit missing-onset indicator is set. No future bleeding event or completed cycle information is used.

## Allowed input families

The implemented v0 feature set is intentionally narrower than every Phase 0 candidate:

1. End-of-day active-minute totals.
2. Wake/end-aligned computed-temperature summaries.
3. Daily aggregates of HRV detail records.
4. Valid wake-day respiratory-rate summaries.
5. Wake-day sleep-score fields, including the resting-heart-rate field in that table.
6. Past self-reports known by the cutoff, including bleeding reports used for causal calendar state.
7. The calendar-known weekend indicator for the recorded participant-day.
8. Missingness indicators, coverage, and time-since-observation derived causally from the sources above.

Participant static metadata and height/weight are not used in v0. Participant ID is never a predictor. Input hormone values are prohibited even when they precede the forecast origin.

For each daily signal, the 14-day featurizer emits the latest observed value, mean, population standard deviation, minimum, maximum, causal least-squares slope, observed-day fraction, days since the latest observation, a current-day missing indicator, and fixed lags 0, 1, 3, 6, and 13. Every statistic is computed only from *t-13* through *t*. Raw missing values remain explicit until the model's train-only preprocessing step.

## Leakage blacklist

The main track rejects:

- any past, current, or future `lh`, `estrogen`/E3G, or `pdg` value as an input;
- `phase` and any Mira fertile-window or phase label;
- target-derived surge, rise, or event labels;
- participant ID as an unrestricted predictor;
- future menstruation or symptom reports;
- completed cycle length, a cycle percentage that uses a completed cycle, or cycle numbering derived from future phase transitions;
- future interpolation, backward filling, bidirectional imputation, or smoothing across the cutoff;
- centered rolling windows;
- feature selection, scaling, encoding, imputation, or normalization fit with validation/test data;
- participant aggregates using future days, test days, or cross-interval target outcomes; and
- any daily or overnight summary whose end/revision time is later than the prediction cutoff.

A causal calendar feature may use only the most recent bleeding onset already known at the cutoff. Forward carry of a past observation may be represented by an explicit time-since value; future fill is never allowed.

## Fixed split and fitting boundary

One deterministic participant-disjoint split is generated with seed `20260719`:

| Split | Participants | Permitted use |
|---|---:|---|
| Train | 12 | Fit models and every learned preprocessing transform |
| Validation | 4 | Early stopping and the single fixed model-selection path |
| Test | 4 | Final evaluation only |

Split balancing uses only prespecified coverage quantities: eligible-origin count and approved-modality day coverage. It never uses hormone values or model performance. Participant sets must have zero overlap, and no alternative split may be selected after observing test results.

If any Interval 1 records are ever used in a primary-pipeline preprocessing step, all intervals belonging to validation and test participant IDs must remain excluded from fitting. The implemented primary track itself uses Interval 2 only.

## Prediction contract

Submissions are long-form tables with exactly these columns:

```text
sample_id,hormone,horizon,y_pred,model_name,model_version,track,split
```

`y_pred` is a finite, nonnegative prediction in log1p space. `hormone` is one of `lh`, `e3g`, or `pdg`; `horizon` is 1. Submissions do not include `y_true`. The evaluator joins private truth internally and fails on missing or duplicate sample/hormone predictions.

## Metrics

### Primary metrics

For each hormone separately:

1. Compute absolute error in log1p space for each held-out participant's test dates.
2. Average errors within that participant.
3. Macro-average the four participant means with equal participant weight.

The three primary results are LH, E3G, and PdG participant-macro log1p-MAE. Test dates are not micro-averaged as independent observations.

### Composite and secondary metrics

- Overall score: equal-weight the three hormone errors after normalization by each hormone's train-only IQR. If a scale is zero or unstable, omit the composite rather than substitute a test-derived scale.
- Raw-unit MAE per hormone.
- Participant-macro log-RMSE per hormone.
- Skill relative to `causal_calendar`: `1 - MAE_model / MAE_causal_calendar`.
- Anonymous participant-level median and range.
- Aggregate count of test participants improved, reported descriptively without IDs.

With four test participants, results are descriptive. Hormonbench v0 makes no statistical-significance or state-of-the-art claim and does not use random-day bootstrap intervals.

## Deferred tracks

Seven-day trajectory forecasting and the 2022-to-2024 longitudinal-shift diagnostic are explicitly deferred. The latter would include returning participants and therefore must not be described as unseen-participant evaluation. Same-day nowcasting is not implemented because urine-test timing relative to full-day wearable summaries is unresolved.
