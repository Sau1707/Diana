# Hormonbench v1 model package

`model/` owns training and inference only. It consumes feature-only Hormonbench contracts and emits conforming private prediction files; it does not prepare governed data, choose participants, calculate official metrics, or import the benchmark evaluator.

## Exactly three active classical baselines

The v1 classical registry contains exactly:

| Family | Role | Fitting rule |
|---|---|---|
| `population_median` | no-information reference | participant median per hormone, then median across development participants |
| `wearable_ridge` | transparent linear wearable baseline | fit-only filtering, median imputation, weighted standardization, fixed alpha |
| `catboost` | nonlinear wearable baseline | participant-balanced CPU Pools, validation-selected tree count, 16-participant refit |

All three use one seed, participant-balanced fitting, the same approved v1 wearable contract, and the frozen independent diagonal residual-intercept adapter for K=3/7. K=0 has exactly zero adaptation. `causal_calendar` remains v0 legacy source only; it is inactive because its Interval-2 state was stale Interval-1 information.

## Active custom reference: Diana-H3P

`model/diana_h3p/` contains the sole active custom reference, `diana_h3p`: the Budget-Aware Hierarchical Tri-Hormone Personalizer. It is separate from the three classical baselines.

Layer 1 learns a participant-balanced convex stack of population median, wearable Ridge, and CatBoost. Weights are fold-local and use grouped development OOF predictions; CatBoost tree-count selection is nested inside each OOF fit so the held-out OOF group cannot influence preprocessing or stopping.

Layer 2 learns a fold-local joint residual posterior. It uses exact chronological K=3 and K=7 calibration procedures, continuously shrunk 3x3 covariance estimates, stable linear solves, and participant-block calibrated 80% research prediction intervals. K=0 receives no truth and exactly reproduces Layer 1 on the common scoring suffix.

NumPy float64 is canonical: complete representative Layer-2 median timing was 1.181 seconds versus 1.738 seconds for PyTorch CPU and 1.315 seconds for PyTorch CUDA, with numerical parity passing. PyTorch remains optional because neither device cleared the prespecified 10% speedup gate.

See [the Diana-H3P model card](diana_h3p/MODEL_CARD.md) for the complete protocol and claims boundary.

## Historical custom evidence

`model/joint_bayes_personalizer/` is preserved as an inactive `historical_protocol_compromised_comparator`. Its global fold-0 covariance selection used labels from a group that later served as outer test in another rotating fold. Its historical source and descriptive aggregate evidence remain available, but it is not registered as the active custom reference and is not a clean confirmatory comparator.

Diana-H3P itself was designed after prior v1 outer-test results had been inspected. Its evaluation is therefore explicitly post-hoc on the existing protocol, not untouched-test confirmation.

## Commands

Fast synthetic/full-contract smoke:

```powershell
python scripts/run_diana_h3p_v1.py --synthetic
```

Development-only diagnostics:

```powershell
python scripts/run_diana_h3p_v1.py --development-only
```

Frozen canonical evaluation:

```powershell
python scripts/run_diana_h3p_v1.py `
  --benchmark-config configs/hormonbench_v1.yaml `
  --model-config configs/diana_h3p_v1.yaml
```

The canonical path does not install packages, does not tune against outer-test scores, and uses an explicit run-specific private prediction manifest.

## Adding another model

Add a new implementation under `model/<name>/`, consume only the public prepared-data/calibration contracts, and emit the unchanged prediction schema. Do not modify benchmark folds, metrics, evaluator, feature deny-list, or privacy boundary.

## Tests

```powershell
python -m pytest model/tests -q
```

The suite covers registry identity, equal participant weights, train-only preprocessing, grouped OOF boundaries, convex stacking, chronological calibration authorization, continuously shrunk PSD covariance, posterior behavior, interval ordering, NumPy/PyTorch parity, and synthetic five-fold execution.

