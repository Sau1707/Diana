# Diana-H3P implementation specification

**Status:** frozen before Diana-H3P implementation and official outer-fold inference  
**Frozen on:** 2026-07-19  
**Task:** `hormonbench_mcphases_interval2_nextday_v1` (`1.0.0`)  
**Custom identifier:** `diana_h3p`  
**Primary contribution:** Hormonbench data and benchmark infrastructure; Diana-H3P is one compact reference implementation.  
**Interpretation:** post-hoc evaluation of a newly frozen model on an existing protocol whose earlier outer-test results have already been inspected. It is descriptive, nonclinical, and not an untouched-test confirmation.

## Gate A evidence

The existing suite passed unchanged (`61 passed`). Official execution uses Python 3.11.15 in the intended `ai` Conda environment; NumPy 2.4.4, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.9.0, CatBoost 1.2.10, PyTorch 2.5.1+cu118, pytest 9.1.1, and the existing runtime dependencies import together. CUDA reports an NVIDIA GeForce GTX 1650 with 4 GiB. `pip check` passes. No dependency change is required.

Repository and private aggregate evidence independently reproduce:

- 20 eligible Interval-2 participants and 1,509 eligible origins;
- exactly `t-13...t` inputs and genuinely observed `t+1` LH/E3G/PdG targets;
- five groups of four, five 12/4/4 outer roles, and final 16-development/4-test fits;
- 1,369 common-suffix origins (fold counts 273, 274, 289, 289, and 244);
- 490 safe candidate wearable features, with no self-report, menstrual-calendar, absolute-time, ID, hormone-history, Mira, or future-derived predictor;
- task-spec hash `1783561d82980e55ff0a3fa3cb026a2598afcc4f6eec1a6c456d117e70d6c6e5`;
- fold hash `f4c350c291c060638324c96b694f3af0a811180ba2f182883d5f4746ca37dbf1`;
- input-schema hash `30b91bb7085bfc2e2efb29824f48f18bc2171b8157a34f4ee080172dad520a82`.

The model-independent evaluator correctly computes participant-macro metrics and development-only scales. Its pre-H3P source hash is `6b47302c876763028e372986afbd02f5df594ea8413aef1ae2ae591cc7c0a40d`; integration may generalize the active custom-model identity but must not change metric mathematics, expected samples, truth joining, or privacy boundaries.

## Accepted audit findings

Three independent read-only audits found four blocking legacy issues that this implementation must not inherit:

1. Fold-0 validation group labels selected a global diagonal/full covariance mode, although that group later becomes outer test in another fold.
2. Legacy custom OOF CatBoost reused tree counts selected with a group that could itself be the held OOF group.
3. The legacy posterior used the independent-observation approximation `K * Sigma_e^-1`.
4. Existing prediction manifests do not themselves bind every baseline file to the task, fold, and input-schema hashes.

`model/joint_bayes_personalizer/` and its aggregate results remain preserved as inactive `historical_protocol_compromised_comparator` evidence. None of its lambdas, covariance matrices, multipliers, posteriors, predictions, or intervals may be reused by H3P.

## Frozen benchmark contract

The benchmark task, prepared rows, feature contract, folds, target transform, participant-macro metric, normalization, and three active classical baselines remain unchanged. Active baseline names are exactly:

1. `population_median`
2. `wearable_ridge`
3. `catboost`

The active custom registry contains exactly `diana_h3p`. No application, new target, new split, seven-day task, shift track, classifier, raw-stream model, neural network, repeated seed, or broad sweep is introduced.

Cold start supplies no outer-test hormone labels. Few-shot budgets are K=0, K=3, and K=7; each uses the same suffix beginning when the seventh eligible calibration target is causally available. Calibration rows are excluded from scoring, K=3 receives ranks 1-3 only, and no later truth enters adaptation. K=0 on the common suffix must equal Layer 1 exactly.

## Layer 1: participant-balanced stacked wearable prior

For each hormone, Layer 1 is the convex combination

`mu(x) = w_median * median + w_ridge * ridge(x) + w_catboost * catboost(x)`.

Weights are nonnegative and sum to one. Candidates are every three-component simplex point on the fixed 0.10 grid, including all endpoints. Selection minimizes participant-macro log1p-MAE on that outer fold's 16-participant development grouped-OOF predictions. Exact ties within `1e-12` prefer larger population-median weight, then larger Ridge weight, then lexicographic order. There is no intercept or nonlinear stack.

For outer fold `k`, group `k` is untouched outer test. Each of the other four participant groups is held out once as a development OOF block. Median, Ridge, and CatBoost fit only the remaining 12 participants. Ridge preprocessing is fit on those 12 participants only.

CatBoost stopping is nested. For a held development group `g`, the three non-OOF groups are ordered cyclically after `g`; the first available group that is not outer test is the four-participant inner-validation group, and the remaining two groups are the eight-participant inner-training set. Tree count is selected on 8/4, CatBoost is refit on all 12 non-OOF participants with that fixed count, and only then predicts group `g`. Preprocessing is refit separately for stopping and final 12-participant fitting. The OOF group and outer-test group influence neither operation.

After fold-local weights are fixed, final Layer-1 outer predictions apply them to the three frozen 16-development-participant baseline predictions for the same fold and samples.

## Baseline reuse authorization

The 60 final outer baseline files may be reused only after one explicit private audit verifies:

- task ID/version and prepared task-spec hash;
- private fold hash and expected fold/sample membership;
- prepared input-schema hash and baseline-relevant frozen configuration;
- the exact 12-column prediction schema and expected track/budget rows;
- an explicit manifest entry for every required file;
- every recorded SHA-256 byte hash;
- byte identity with the preserved original baseline source when claimed.

Current read-only diagnostics found 60/60 hashes and schemas valid and 60/60 byte-identical copies, but the official run must write a new scientific binding audit before copying them into its run-specific private directory. Any failure forces regeneration of only the affected baseline path.

## Layer 2: budget-aware joint personalization

Layer 2 uses three-dimensional log1p residuals `r = y - mu(x)`. All numerical parameters are estimated separately inside each outer fold from the 16 development participants' fresh Layer-1 grouped-OOF residuals.

For each development participant, the persistent-effect proxy is the mean residual vector over that participant's authorized common scoring suffix. The 16 proxies receive equal influence when estimating `Sigma_a`.

For K=3 and K=7, reproduce the exact chronological eligible-calibration protocol within every development participant. Let `r_bar_i,K` be the mean of the earliest K calibration residual vectors and `a_proxy_i` the common-suffix proxy. Estimate `Psi_K` from the 16 equally weighted vectors `r_bar_i,K - a_proxy_i`. No `Sigma/K`, `K*precision`, or independent-day approximation is permitted.

For an outer-test participant:

- K=0: posterior mean is exactly zero and posterior covariance is `Sigma_a`;
- K=3/7: with `C = Sigma_a + Psi_K`, compute `a_hat = Sigma_a * solve(C, r_bar)` and `V = Sigma_a - Sigma_a * solve(C, Sigma_a)` using Cholesky/linear solves, never explicit inversion;
- point prediction is `max(mu + a_hat, 0)` in log1p space.

## Continuous covariance shrinkage

Every covariance family (`Sigma_a`, `Psi_3`, `Psi_7`, and future residual covariance) uses one fixed estimator, not a discrete diagonal/full choice:

1. center the equal-participant sample vectors;
2. estimate original per-hormone variances;
3. standardize and obtain the analytic Ledoit-Wolf shrinkage intensity;
4. form `(1-alpha) * R_sample + alpha * I`, force unit diagonal, and reconstruct covariance with the original marginal standard deviations;
5. symmetrize and floor eigenvalues at `max(1e-10, 1e-6 * trace(Sigma) / 3)`.

Future residual covariance starts from the equal-weight mean of within-participant covariance contributions on common-suffix residuals after removing the participant proxy. A deterministically scaled pseudo-sample supplies the Ledoit-Wolf intensity without allowing longer records more influence. Public diagnostics may include shrinkage intensity, eigenvalues, condition number, and a near-diagonal indicator, but never residual vectors.

This is a regularized empirical-Bayes approximation; the proxy and calibration deviations are not claimed to identify clean biological variance components.

## Research uncertainty

Base predictive covariance is `Sigma_future + V_K`. For each K, interval multipliers are learned from development data only by leave-one-participant-out Layer-2 reconstruction using that participant's exact chronological calibration budget and common suffix. Errors are standardized by the matching predictive standard deviation and combined with participant-balanced weights. The fixed 0.80 weighted quantile gives one multiplier per hormone and budget.

Final intervals are `point +/- multiplier * predictive_sd`, with lower bounds clamped to zero. They must be finite, ordered, and contain the point prediction. Public wording is **80% participant-block calibrated research prediction intervals**. No IID finite-sample, clinical-confidence, or clinical-coverage claim is supported.

## Backend contract and selection

NumPy/SciPy is the complete reference backend. PyTorch is optional and lazily imported. Both implement float64 batched 3x3 posterior and interval algebra with no explicit inverse. K=0 returns exact zero means and exact `Sigma_a` copies. PyTorch uses `no_grad`, deterministic settings where supported, `cholesky_ex`/`cholesky_solve`, explicit device metadata, and synchronized CUDA timing.

Parity requires `rtol <= 1e-8` and `atol <= 1e-10` for diagonal, correlated, near-singular, K=0/3/7, batched, and interval cases. The backend profile times the complete Layer-2 path after OOF residuals exist, with warmups and repeated medians. PyTorch becomes canonical only if parity, deterministic repetition, scientific tests, memory limits, and at least 10% lower median end-to-end Layer-2 time all pass. Backend choice never uses target performance.

**Backend freeze evidence:** on the real fold-0 development OOF residual set, seven complete-path repetitions gave median 1.181 s for NumPy CPU, 1.738 s for PyTorch CPU, and 1.315 s for PyTorch CUDA. Maximum absolute parity error was `8.89e-16`; repeated outputs were deterministic. PyTorch CUDA was 11.4% slower than NumPy and used approximately 2.83 GiB process RSS versus 268 MiB for NumPy. The canonical backend is frozen to `numpy`; PyTorch remains an optional tested backend.

## Run discipline, privacy, and release

Implementation, focused tests, synthetic five-fold execution, development-only diagnostics, backend selection, and hashes must finish before any H3P outer-test prediction. A code-freeze commit is then created from an explicit public allow-list. The official runner executes one sequential five-fold custom evaluation. A rerun is allowed only for a concrete invalidating implementation/runtime defect, whose run ID and reason must be preserved.

Private run material lives under `artifacts/private/v1/diana_h3p/<run_id>/` with explicit manifests. Public output contains aggregate statistics and figures only. Participant IDs, sample IDs, calibration rows, truth, row predictions, participant metrics, folds, checkpoints, private paths, datasets, caches, bytecode, `.git`, and ZIPs are forbidden. The locally present `pyproject.zip` is preserved but ignored and excluded.

The public H3P result must retain a clearly separated historical note for the legacy custom comparator, disclose prior inspection of the benchmark's outer results, and make no SOTA, significance, diagnostic, serum-hormone, ovulation, causal, clinical-validation, or broad-population claim.

## Frozen scientific hashes

- H3P configuration hash: `d02eb3f721cdb8f820f305fdec4108f6f9cbec5dc633e0a82a6a2a35ea0ab23e`
- H3P scientific model-spec hash: `3b77546e8dc6c12f102e7a5c3c9cb884bd609a778048e9bc796ac4832815083a`
- Active evaluator source hash at freeze: `5c1f3cba9337b0389407d31969c738bf6aae29773447556267062434b5fbf2ec`

The implementation-spec file hash is recorded externally in the code-freeze and canonical run manifests to avoid a self-referential hash.
