# Hormonbench-mcPHASES v1 task card

## Identity and prediction boundary

- Task ID: `hormonbench_mcphases_interval2_nextday_v1`
- Version: `1.0.0`
- Dataset interval: mcPHASES Interval 2 (`study_interval == 2024`)
- Unit: participant-origin calendar day
- History: exactly `t-13` through `t`
- Horizon: `t+1`
- Targets: participant-entered at-home urinary LH, E3G (source `estrogen`), and PdG
- Space: nonnegative `log1p`
- Label rule: all three targets must be genuinely observed; labels are never interpolated

The cutoff is the end of day `t`. Sleep, respiratory, and computed-temperature summaries align to their wake/end day; a session ending on `t` may inform `t+1`, but one ending on `t+1` may not. This avoids the unresolved ordering in same-day nowcasting.

## Eligible population

The adapter reproduces 20 participants and 1,509 origins. Removing unusable features does not alter eligibility. Adjacent origins overlap heavily, so they are not treated as independent participants or micro-averaged for the primary metric.

## Allowed cold-start features

Only these Interval-2 families enter the candidate matrix:

1. active-minute daily summaries;
2. sleep-end-aligned computed-temperature fields;
3. daily HRV aggregates;
4. valid wake-day respiratory summaries;
5. sleep-score fields, including its resting-heart-rate field;
6. calendar-known weekend state; and
7. causal lags, summaries, missing-current masks, coverage, and time-since-observation derived from those sources.

For each signal the featurizer uses fixed lags 0, 1, 3, 6, and 13 plus last, mean, standard deviation, min, max, causal slope, coverage, time-since, and current-day missingness. Every transformation stops at `t`.

Interval 2 has almost no self-report data and no usable flow volume. The entire self-report family is excluded so structural non-collection cannot become a participant signature.

## Mandatory leakage deny-list

No cold-start feature may contain or derive from participant ID, sample ID, origin or target day, interval identity, calendar date, absolute study time, modulo-28 time, stale menstrual timing, menses-onset missingness, Interval-1 bleeding, LH/E3G/PdG history, current/future hormone values, Mira phase/fertile labels, target-derived events, future menstruation, completed cycle length, cycle percentage, centered windows, backward fill, future interpolation, or participant aggregates using future/validation/test data.

Private IDs and days exist only for alignment, grouping, authorization, and evaluation. Feature-only views make them unavailable as predictors. Changing Interval-1 flow data cannot change any v1 Interval-2 feature.

## Five-fold cold-start protocol

Seed `20260719` fixes five private groups of four participants. Group 0 preserves the v0 test group; group 1 preserves its validation group; groups 2–4 partition its old 12-person training group using only eligible-origin count and approved wearable coverage.

For outer fold `k`, group `k` is test, group `(k+1) mod 5` is validation, and the other 12 participants are initial training. Validation selects CatBoost tree counts; preprocessing and the predictor are then refit on the combined 16-person development set. Outer-test truth never changes features, stopping, parameters, folds, or scales.

Every participant is test once and validation once; all 1,509 origins are outer-test exactly once. The folds are one participant-independent protocol, not repeated seeds, and their overlapping development sets make fold dispersion descriptive.

## Few-shot personalization

The separate `few_shot_personalization` track authorizes the earliest K=0, 3, or 7 complete hormone targets among a participant’s eligible forecast targets. Raw early targets cannot be used because they lack a corresponding 14-day forecast origin.

All budgets are scored on the identical suffix beginning when the seventh calibration target is available at the origin cutoff. Calibration rows are excluded. This yields 1,369 origins across all 20 participants, with no silent participant drop.

K=3 never receives observations 4–7; K=0 receives no truth and exactly reproduces cold-start predictions on common rows. Global parameters are unchanged across budgets. The authorized calibration view is separate from the truth-free inference view.

## Metrics

For each hormone, absolute log1p error is averaged within each outer-test participant, then macro-averaged equally across all 20 participants. Raw-unit MAE and log1p-RMSE use the same participant-macro structure.

The overall score divides each participant/hormone log1p-MAE by that fold’s participant-balanced IQR learned from its 16 development participants, then equal-weights LH, E3G, and PdG. Population median is the v1 skill reference. Participant-improvement counts and fold dispersion are descriptive; no significance or random-day bootstrap claim is made.

## Active reference-model freeze

The active Diana-H3P reference makes no global fold-0 model choice. For each outer fold, stack weights and continuous covariance-shrinkage values are learned only from that fold’s 16 development participants. Every development OOF CatBoost fit uses nested 8/4 participant-group stopping inside the 12 non-OOF participants, then refits on those 12 before predicting the held group.

The earlier `joint_bayes_personalizer` selected diagonal/full covariance globally from fold-0 validation. That validation group later became outer test in another fold, so its aggregate is preserved only as `historical_protocol_compromised_comparator`. It is not an active reference or clean confirmatory comparator.

The H3P architecture, 0.10 simplex, covariance estimator, K protocol, seed, and backend rule were frozen before its canonical outer predictions. Its result is nevertheless post-hoc because earlier Hormonbench outer results had already been inspected.

## Excluded tasks

Seven-day forecasting, 2022–2024 shift evaluation, same-day nowcasting, raw streams, CGM, phase/ovulation classification, large neural models, and clinical use are outside v1.
