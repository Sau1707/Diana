# Diana challenge submission narrative

## Primary contribution

Diana is submitted primarily to **Challenge Layer 01: Data & Benchmark Infrastructure**. Its core reusable contribution is Hormonbench: governed-data infrastructure that freezes a causal next-day urinary-hormone task, participant-independent folds, cold-start and few-shot contracts, an independent evaluator, and aggregate-only reporting.

**Challenge Layer 02: AI Model Infrastructure** is demonstrated by one compact reference implementation, Diana-H3P. H3P shows how another model can obey Hormonbench's feature, fold, personalization, uncertainty, and submission contracts. It is not a fourth classical baseline or a separate product. Diana does not claim a Challenge Layer 03 application; no consumer or clinical application was built.

## Women's Health Impact

Hormonal-health ML can appear stronger than it is when days from the same participant cross train/test boundaries, future cycle information leaks into features, target-derived phase labels are treated as independent truth, or urinary metabolites are described as serum concentrations. Hormonbench turns these failure modes into executable contract violations.

The K=0/3/7 personalization track makes measurement burden measurable. K=0 exposes no personal hormone labels; K=3 and K=7 expose only the earliest authorized complete three-hormone readings. Every budget is scored on the identical 1,369-origin later suffix, so a change in score is not caused by changing evaluation rows. This protocol can study the value of fewer at-home measurements without claiming that any budget is clinically sufficient.

## Technical Excellence

- exact `t-13...t -> t+1` temporal alignment and genuinely observed labels;
- five deterministic participant groups covering all 20 eligible participants and 1,509 origins exactly once as outer test;
- participant-balanced fitting, participant-macro metrics, and development-only normalization;
- three intentionally bounded classical baselines: population median, wearable Ridge, and CatBoost;
- Diana-H3P Layer 1 convex stacking learned from participant-disjoint development OOF predictions, with nested CatBoost stopping;
- Diana-H3P Layer 2 exact K-specific calibration covariance, continuous correlation shrinkage, stable posterior solves, and participant-block research uncertainty;
- an explicit prediction manifest and hashes instead of unsafe directory globbing;
- a model-independent evaluator, synthetic full-contract tests, leakage tests, backend parity tests, and privacy/release validation; and
- private governed rows and participant artifacts, with aggregate public outputs only.

NumPy float64 is the canonical H3P Layer-2 backend. Complete representative timing was 1.181 seconds for NumPy CPU, 1.738 seconds for PyTorch CPU, and 1.315 seconds for PyTorch CUDA; parity passed, but neither PyTorch backend met the required 10% acceleration. PyTorch therefore remains optional and is not used to add a neural network.

Hormonbench v1 also records a substantive correction: v0's menstrual-calendar value was hundreds of days since an Interval-1 report, not current menstrual timing. v1 removes that field and structurally absent self-reports.

## Foundation Value

A licensed mcPHASES researcher can prepare the same private task, consume the frozen fold/feature/calibration views, emit the documented prediction schema, and use the evaluator without importing `model/`. New model comparisons therefore inherit the same cutoff, folds, normalization, metrics, and privacy boundary.

Code, tests, configuration, task/data/benchmark/model documentation, and aggregate results are releaseable under the public allow-list. The governed dataset, identities, sample rows, truth, calibration mappings, row predictions, participant metrics, and fitted private artifacts are never redistributed.

## Diana-H3P reference

Layer 1 is a participant-independent wearable prior: a fold-local convex stack of participant-equal population median, wearable Ridge, and CatBoost. Layer 2 uses zero, three, or seven authorized personal measurements to update a joint three-hormone empirical-Bayes posterior and emit 80% participant-block calibrated research prediction intervals.

The canonical runner generates the verified aggregate result under `results/v1/diana_h3p/`. No performance claim is inserted before that run. H3P was designed after earlier Hormonbench v1 outer-test results had been inspected, so its result must be described as a **post-hoc evaluation of a newly frozen reference model on the existing protocol**, not a fresh untouched-test confirmation.

The former `joint_bayes_personalizer` is preserved only as an inactive `historical_protocol_compromised_comparator`. Its global fold-0 covariance selection used labels from participants who later served as outer test in another fold; it is not the active reference or a clean confirmatory comparator.

## Claims boundary

This work is not clinically validated. It does not diagnose, predict serum hormones, verify ovulation, establish causal physiology, or support clinical decisions. Targets are participant-entered consumer urinary-monitor readings. The 20-participant cohort is small, adjacent origins are strongly correlated, and the research intervals have no IID clinical guarantee. Results cannot be generalized to all women or asserted for PCOS, pregnancy, perimenopause, menopause, hormonal contraception, other devices, or other populations without independent validation.

