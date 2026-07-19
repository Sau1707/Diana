# Hormonbench benchmark card

## Summary

Hormonbench is the reusable benchmark component of Diana. Version 0 fixes one governed-data task, one deterministic participant split, one prediction schema, and one independent evaluator so model comparisons cannot silently change cohort, temporal alignment, or scoring.

The implemented task forecasts next-day at-home urinary LH, E3G, and PdG readings from a 14-day causal wearable/self-report history in mcPHASES Interval 2. Hormonbench is intended for research method comparison and is not a medical device or clinical validation study.

## Why next-day forecasting is primary

The target file does not include urine-test timestamps. Full-day summaries can include information acquired after a same-day morning urine test, making whole-day same-day nowcasting temporally ambiguous. A strict *t* to *t+1* cutoff avoids silently treating later same-day observations as causal predictors.

Fourteen days preserves substantially more eligible participants/origins than a 28-day history while still representing multi-day wearable context. Seven-day trajectory forecasting is feasible but deferred because complete-window selection and correlated forecast horizons require additional evaluation choices.

## Why participant-disjoint evaluation

Longitudinal rows from one participant share stable physiology, behavior, device characteristics, and missingness patterns. Randomly splitting days would allow those participant-specific signatures into both training and test data. Hormonbench freezes a 12/4/4 participant-disjoint split and validates zero overlap.

All learned imputers, scalers, encoders, and feature transformations are fit on train only. Validation is reserved for early stopping and the fixed selection path. Test truth is used only by the evaluator after predictions are final.

## Why participant-macro metrics

Participants contribute different numbers of eligible origins. Micro-averaging every date would give participants with longer or more complete records greater weight. The primary metric first averages log1p absolute errors over dates within each test participant, then gives each of the four test participants equal weight. LH, E3G, and PdG are reported separately.

The optional overall score normalizes hormone errors using train-only robust scales before equal weighting. If a robust scale is unstable, the composite is omitted. Raw-unit MAE, log-RMSE, calendar-relative skill, and anonymous participant summaries provide context but do not replace the primary metrics.

## Why Mira phase is excluded

`phase` is a proprietary app-generated weak label derived from hormone patterns. It is not independently observed clinical ovulation ground truth. Using it as an input would also leak information closely tied to the targets. Hormonbench excludes it, fertile-window labels, completed phase-defined cycle quantities, and target-derived event labels.

The target readings themselves are consumer at-home urine readings transcribed by participants. Urinary E3G is not serum estradiol/E2, urinary PdG is not serum progesterone, and the benchmark does not characterize any target as a clinical gold standard.

## Benchmark/model independence

`benchmark/` owns data preparation, contracts, split generation, metrics, evaluation, and reporting. `model/` owns reference model training and prediction. The evaluator can validate and score a conforming external prediction file without importing `model/`.

The required prediction fields are:

```text
sample_id,hormone,horizon,y_pred,model_name,model_version,track,split
```

Predictions are finite and nonnegative in log1p space and do not contain truth. Private truth is joined internally. Missing or duplicate required predictions are errors, and row order does not affect results.

## External model participation

An authorized mcPHASES user can participate as follows:

1. Obtain mcPHASES from PhysioNet under its data use agreement.
2. Run `python -m benchmark prepare --config configs/hormonbench_v0.yaml` to create the private frozen bundle and split.
3. Fit a model using only train data, with validation used only as permitted by the task card.
4. Emit one conforming prediction row per required test sample/hormone in log1p space.
5. Place the file under the private prediction directory and run the unchanged evaluator and report commands.

No real sample IDs, participant IDs, truth, predictions, or trajectories should be published. Public `results/v0/` files contain aggregate metrics only.

## Reference scope

The v0 model package contains exactly three classical baselines:

- a train-only population median lower bound;
- a causal menstrual-calendar Ridge reference; and
- bounded CPU CatBoost regressors using the approved causal tabular features.

They all consume the same prepared bundle and produce the same submission schema. A future model can be added under its own `model/<name>/` directory without modifying benchmark preparation, metrics, evaluator, or reporting.

## Deferred work

- Seven-day masked trajectory forecasting is deferred beyond v0.
- A 2022-to-2024 LH/E3G track is deferred. Because all Interval 2 participants return from Interval 1, it is a longitudinal/domain-shift diagnostic, not unseen-participant evaluation. Interval 1 contains no PdG.
- Same-day nowcasting is deferred because target/sensor ordering is unresolved.
- Raw high-volume streams, glucose, ambiguous daily summaries, neural models, Gaussian processes, ensembles, and any custom model are outside this v0 run.

## Limitations and claims

The test split contains only four participants. Results are descriptive and should not be presented as statistically significant, state of the art, clinically actionable, or broadly generalizable. Missingness, consumer-device limits, transcription, cohort selection, and unresolved source units constrain interpretation. Hormonbench compares methods under one transparent contract; it does not establish clinical validity or causal relationships.
