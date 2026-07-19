# Hormonbench-mcPHASES v1 implementation specification

> **Corrected freeze outcome (2026-07-19):** neither covariance candidate passed the superiority gate. Full scored 0.602186 versus 0.606438 for diagonal on fold-0 validation, so the required strongest-valid fallback selected full. An initial diagonal execution was invalidated because the implementation incorrectly defaulted to diagonal on gate failure; its valid baseline outputs were reused byte-for-byte and only the custom path was recomputed. No outer-test metric selected this correction. Final config hash: `9015f03a3d354a7e973e56a4d0d538577c7d69b9fe638cf75cc9a4f0b9b07d7b`.

**Status:** frozen before v1 implementation and official outer-test evaluation  
**Task ID:** `hormonbench_mcphases_interval2_nextday_v1`  
**Benchmark role:** Diana's reusable governed-data benchmark; the custom model is one reference implementation, not a separate product.  
**Seed:** `20260719` (single source for grouping, model randomness, hashes, and tests)

## Gate A evidence and accepted corrections

The pre-change suite passed (`30 passed`). Environment verification resolved to Python 3.11.15 in the intended `ai` Conda environment; CatBoost 1.2.10 and pytest 9.1.1 were already installed, CatBoost and PyTorch imported together, CUDA was available on an NVIDIA GeForce GTX 1650, and `python -m pip check` reported no broken requirements. No dependency installation is required.

The protocol audit reproduced the blocking v0 defect without publishing participant identifiers: 1,427/1,509 prepared rows contained `days_since_last_known_menses` values ranging from 773 to 932 days (median 846), derived from Interval-1 bleeding reports because Interval 2 has no usable flow reports. The field correlates 0.961 with absolute origin time. Both it and its missingness flag entered the 688-column CatBoost matrix; `causal_calendar` used it directly. Consequently all public v0 scores are preserved as historical evidence but are **superseded/provisional** and may not be merged with or cited as v1 performance.

Accepted blocking/major engineering corrections:

1. Remove stale calendar, cross-interval menses, all self-reports, and absolute-time fields from v1 features.
2. Use feature-only model matrices; retain IDs/days only in private alignment, weighting, calibration, and evaluation structures.
3. Use five fixed participant groups and participant-balanced fitting/validation.
4. Replace directory-wide prediction globbing with one run-specific explicit manifest.
5. Remove all runtime package installation from model fitting; canonical CatBoost must fail honestly if unavailable.
6. Separate the scientific task-spec hash from operational paths and derive stable sample IDs from the scientific hash.
7. Use typed public manifests rather than copying arbitrary nested private checkpoint metadata.
8. Ignore the pre-existing unsafe `LICENSE.zip`, preserve it untouched, and validate releases using an explicit public allow-list. Whole-workspace archives are forbidden.

## Frozen prediction task

- Dataset: governed mcPHASES v1.0.0, obtained independently from PhysioNet.
- Collection interval: Interval 2 (`study_interval == 2024`) only.
- Unit: participant-origin day.
- History: exactly the 14 calendar days `t-13` through `t`, inclusive.
- Cutoff: end of day `t`; completed overnight summaries are aligned to their wake/end day.
- Horizon: genuinely observed `t+1` urinary readings.
- Targets: participant-entered at-home urinary LH, E3G (`estrogen` source column), and PdG.
- Target/prediction space: `log1p`; raw readings are retained privately for secondary MAE.
- Eligibility: all three targets genuinely observed at `t+1`; no interpolation or manufactured labels.
- Required invariant: exactly 20 participants and 1,509 eligible origins. A count change is a blocking error.

These are consumer urinary readings, not serum hormones, clinical gold standards, diagnoses, or verified ovulation labels.

## Cold-start feature contract

Allowed feature sources are limited to:

- active-minute daily totals;
- computed temperature aligned to `sleep_end_day_in_study`;
- daily HRV aggregates;
- valid wake-day respiratory summaries, with nonpositive sentinel values masked;
- sleep-score approved fields, including its resting-heart-rate field;
- calendar-known weekend state;
- causal lag, summary, missingness, coverage, and time-since-observation transforms of those signals over `t-13...t`.

Forbidden predictor inputs or derivations include participant/sample IDs; origin/target/cutoff day; study interval/date; absolute or modulo study time; both v0 menses fields; all Interval-1 events; the complete self-report family; hormone history; current/future hormones; Mira phase/fertility labels; target-derived events; future menstruation; completed cycle quantities; centered windows; backfill; future interpolation; and aggregates fitted using validation/test/future information. Changing Interval-1 flow must leave every v1 feature unchanged.

Feature filtering is fitted separately on the permitted fitting participants. It drops all-missing, constant, and at-least-95%-missing columns, then applies train/development-only median imputation. Ridge additionally applies train/development-only standardization. Feature order is deterministic.

## Five participant groups and outer folds

Five private groups contain four participants each:

- group 0 is the existing v0 test group;
- group 1 is the existing v0 validation group;
- the existing 12 v0 training participants are deterministically divided into groups 2–4 using only eligible-origin count and prespecified wearable coverage.

No hormone value or model result may affect grouping. For outer fold `k`, test is group `k`, validation is group `(k+1) mod 5`, and the remaining three groups are the initial training set. Each fold is therefore 12/4/4. Every participant and every eligible origin is outer-tested exactly once; every participant validates exactly once. Group/fold hashes use the same configured seed and are target-value invariant.

Model stopping/variant selection uses only the 12 training and four validation participants. After selection, preprocessing and predictors are refit on the combined 16-participant development set and outer test is inferred once. Development target scales are participant-balanced weighted IQRs computed separately in each fold.

## Tracks and calibration

Official tracks are:

1. `cold_start_participant_independent` — no outer-test hormone labels, scored on all eligible outer-test origins.
2. `few_shot_personalization` — budgets K=0, 3, and 7, reported separately.

Calibration labels are the earliest K chronologically complete targets **among the 1,509 eligible forecast samples** for each held-out participant. This resolves an audited ambiguity: none of the earliest seven raw Interval-2 complete hormone observations has a valid 14-day forecast origin, so raw-table calibration would silently change the task. All 20 participants have at least seven eligible calibration targets. The common suffix begins when the seventh eligible calibration target is causally available (`origin_day >= seventh target_day`), contains 1,369 origins, and is identical for K=0/3/7. Calibration rows are never scored. K=3 cannot see observations 4–7; K=0 receives no truth. Later labels remain evaluator-only private truth.

## Active models

Exactly three active classical baseline families:

1. `population_median`: per-participant target median followed by median across development participants.
2. `wearable_ridge`: one fixed-alpha weighted Ridge per hormone using the full filtered wearable feature matrix.
3. `catboost`: one bounded CPU CatBoost regressor per hormone.

`causal_calendar` remains v0 legacy source only and is inactive in v1. Population median is the v1 skill reference.

Every fitting split uses deterministic row weights proportional to the inverse participant origin count, normalized so each participant contributes equal total weight. CatBoost uses weighted training and validation Pools, records zero-indexed best iteration and `tree_count = best_iteration + 1`, then refits with that fixed tree count on all 16 development participants. Actual CatBoost 1.2.10 is required for canonical results; no fallback may masquerade as CatBoost.

For K=3/7 each classical model receives the same per-hormone empirical-Bayes residual-intercept adapter. Between- and within-participant residual variances come only from participant-grouped development OOF predictions. K=0 offset is exactly zero.

## Custom reference model

Working internal identifier: `joint_bayes_personalizer`; public branding remains unfrozen. It consumes only the benchmark-prepared v1 feature matrix.

For each hormone, the population prior is

`mu(x) = participant_equal_median + lambda * (CatBoost(x) - participant_equal_median)`.

Lambda is selected per hormone on a fixed `[0,1]` grid using participant-balanced grouped-OOF MAE, with ties resolved toward smaller lambda. Grouped OOF fits preprocessing independently and excludes the predicted participant group.

Residual vectors across LH/E3G/PdG estimate equal-participant between-person covariance and equal-participant within-person covariance. Deterministic diagonal shrinkage plus an eigenvalue floor enforces finite PSD matrices. Candidate modes are diagonal and full 3x3. Posterior means/covariances use stable linear solves. K=0 has zero posterior mean and covariance `Sigma_a`.

Research 80% prediction intervals use `Sigma_e + V_K` and deterministic hormone/budget conformal multipliers learned from participant-grouped development OOF residuals. Interval calibration is leave-one-participant-out with respect to empirical-Bayes parameters; participant-balanced quantiles prevent longer records dominating. Intervals are secondary research intervals, not clinical confidence intervals; overlapping windows weaken ordinary exchangeability assumptions.

## Frozen validation-only selection rule

Fold 0 outer test is never read during selection. CatBoost stopping uses the 12 initial training participants and group-1 validation. Diagonal versus full covariance is compared once on fold-0 validation using the equal-weight mean of development-scale-normalized participant-macro scores for K=3 and K=7 on their common suffix; ties go to diagonal.

The custom success gate relative to the strongest corresponding personalized classical baseline is: at least 3% lower combined normalized score, improvement in at least two hormones, improvement for at least three of four validation participants, and no hormone worse by more than 2%. Full covariance is selected only if it beats diagonal and meets these guardrails. If neither valid candidate meets the performance gate after checking implementation/covariance stabilization, diagonal is frozen as the conservative characterization candidate; an honest losing result remains publishable. No outer-test score may change this choice.

After validation, selected mode, lambda rule, feature rules, seed, stabilization, CatBoost stopping/refit logic, and config hash are frozen before the one official five-fold run. No repeated seed, broad sweep, alternate grouping, or post-test edit is permitted.

## Evaluation and public/private boundary

Primary metric is per-hormone participant-macro log1p-MAE. Each participant's date errors are averaged first; all 20 outer-test participants then receive equal influence. The overall score equally averages hormone errors normalized by that participant's fold-specific, 16-participant development weighted-IQR scale. Secondary metrics are participant-macro raw MAE, log1p-RMSE, skill versus population median, aggregate improved-participant counts, descriptive fold mean/dispersion, and custom interval coverage/width/interval score.

Prediction submissions explicitly contain task ID/version, track, fold, calibration budget, split, sample ID, hormone, horizon, model name/version, and point prediction; custom files may also contain ordered finite lower/upper bounds. They never contain truth. An explicit private manifest names exact files and hashes; stray CSVs are not discovered.

All participant IDs, sample IDs, prepared/target rows, fold mappings, calibration views, predictions, participant metrics, and fitted parameters remain under `artifacts/private/v1/`. Public `results/v1/` contains aggregate statistics, safe hashes, typed reproducibility metadata, and aggregate figures only. Public release construction uses a tested allow-list and rejects `.git`, dataset/private paths, ZIPs, caches, bytecode, IDs, truth rows, row predictions, and private absolute paths.

## Explicitly deferred

No application, seven-day forecasting, 2022-to-2024 shift track, same-day nowcasting, raw streams, CGM, ovulation/phase classification, large neural model, GP zoo, repeated seeds, or broad hyperparameter sweep is part of v1.
