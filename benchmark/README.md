# Hormonbench benchmark package

`benchmark/` is Diana’s model-independent contribution. It owns the mcPHASES adapter, frozen scientific task, five-fold participant protocol, few-shot authorization, stable schemas, metrics, evaluator, aggregate reporting, and public-artifact checks. `benchmark/v1_evaluator.py` can score conforming files without importing `model/`.

## Frozen v1 task

- Task: `hormonbench_mcphases_interval2_nextday_v1`, version `1.0.0`
- Input: approved wearable summaries from `t-13` through `t`
- Output: genuinely observed at-home urinary LH, E3G, and PdG at `t+1`
- Target/prediction space: nonnegative `log1p`
- Cohort: 20 eligible Interval-2 participants, 1,509 origins
- Protocol: five deterministic 12-train / 4-validation / 4-test outer folds, then 16-participant final development refit
- Primary endpoint: participant-macro log1p-MAE for each hormone; equal-weight development-IQR-normalized composite
- Reference: participant-equal population median

Allowed core sources are active minutes, sleep-end-aligned computed temperature, daily HRV aggregates, valid wake-day respiratory summaries, sleep-score fields, weekend state, and their causal missingness/coverage/time-since features. Self-reports and menstrual-calendar state are excluded from v1.

See [TASK_CARD_V1.md](docs/TASK_CARD_V1.md) for the complete cutoff and deny-list.

## Prepare licensed data

```powershell
python -m benchmark prepare --config configs/hormonbench_v1.yaml
```

Preparation writes a private versioned bundle and private participant-to-group manifest. Feature-only model views cannot expose participant ID, sample ID, origin/target day, interval, calendar date, or truth. The 20-participant/1,509-origin count is a hard invariant.

## Prediction submission schema

Every private v1 record contains:

```text
task_id,task_version,track,fold,calibration_budget,split,
sample_id,hormone,horizon,y_pred,model_name,model_version
```

`y_lower,y_upper` are optional research-interval fields and must satisfy `0 <= lower <= point <= upper`. An external submission never includes truth. Each file must contain exactly one prediction per required sample/hormone, and the explicit private manifest identifies the exact files and hashes to evaluate. The evaluator does not glob a shared directory.

## Evaluate and report independently

```powershell
python -m benchmark evaluate --config configs/hormonbench_v1.yaml
python -m benchmark report --config configs/hormonbench_v1.yaml
python -m benchmark privacy --config configs/hormonbench_v1.yaml
```

Evaluation joins private truth internally, calculates participant-level errors privately, and emits only aggregate public results. Prediction row order is irrelevant; missing, duplicate, unexpected, non-finite, negative, or truth-bearing rows fail.

The active Diana-H3P run uses the same evaluator through an explicit operational config; `benchmark/v1_evaluator.py` still imports no `model/` code:

```powershell
python scripts/run_diana_h3p_v1.py --evaluate-only
python scripts/run_diana_h3p_v1.py --privacy-only
```

## Participate with another model

An authorized mcPHASES user can:

1. prepare the unchanged v1 bundle and folds;
2. consume only the feature-only fitting/inference views plus an authorized K-label calibration view;
3. preserve fold-specific train/validation/final-development boundaries;
4. emit the schema above into a run-specific ignored directory;
5. list exact files and SHA-256 values in a private prediction manifest; and
6. run the unchanged benchmark evaluator and reporter.

No model needs to import benchmark implementation internals or change metrics. Keep all prepared rows, mappings, sample IDs, truth, predictions, and participant metrics beneath `artifacts/private/`.

## Tests

```powershell
python -m pytest benchmark/tests -q
```

The suite covers temporal cutoffs, observed labels, stale-calendar and self-report exclusion, five-fold invariants, path-independent task hashes, exact few-shot authorization, manifest-only evaluation, metric behavior, evaluator independence, and aggregate privacy.

Hormonbench is the reusable contribution. Diana-H3P is one conforming reference model, and an external model can participate without modifying benchmark metrics or truth handling.
