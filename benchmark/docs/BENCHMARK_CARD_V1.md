# Hormonbench-mcPHASES v1 benchmark card

## Reusable contribution

Hormonbench is Diana’s central infrastructure contribution: one governed-data adapter, causal feature boundary, deterministic participant protocol, prediction schema, independent private-truth evaluator, and aggregate reporting path. The custom probabilistic model is a reference implementation showing how the benchmark supports personalization; it is not a separate product.

## Why next-day and participant-independent

Urine-test timestamps are unavailable, so a full same-day wearable summary could include observations after the urine reading. Predicting `t+1` from information ending on `t` gives a defensible causal cutoff.

Random-day splits would leak stable physiology, devices, behavior, and missingness signatures. Five four-person outer-test groups let every one of 20 participants contribute once to participant-independent evaluation while retaining 12/4 training/validation selection. These folds are not independent replications and do not support conventional significance claims.

## Cold start versus few shot

Cold start asks whether a model generalizes to a participant with no personal hormone readings. Few shot asks how much three or seven authorized personal measurements improve a fixed population model on later dates. The latter is less stringent but quantifies measurement burden directly. It is called personalization—not double-blind/non-double-blind—because the distinction is calibration access, not masking of investigators or participants.

K=0/3/7 use one identical post-seventh-measurement suffix, so score changes are not caused by different evaluation rows. This gives a measurement-budget curve relevant to repeated-testing burden without claiming clinical sufficiency.

## Why participant-macro scoring

Participants have different origin counts, and neighboring 14-day windows share 13 days. Pooling rows would let longer records dominate and falsely imply hundreds of independent subjects. Hormonbench averages dates within participant first, then weights participants equally. Development-only participant-balanced IQRs make the three-hormone composite comparable without test-derived normalization.

## v0 correction

Interval 2 has no usable bleeding reports. v0 carried an Interval-1 onset forward, producing values roughly 773–932 days old and exposing them to both the calendar model and CatBoost. This was not current cycle timing and strongly tracked absolute recording time. v1 removes both calendar fields, deactivates the calendar baseline, excludes self-reports, and adds cross-interval invariance tests. v0 results are preserved but superseded/provisional.

## Why Mira phase is excluded

Mira phase/fertile-window values are proprietary app-generated weak labels derived from hormone patterns, not independently observed clinical ovulation ground truth. Using them would conflate target and derived label. Hormonbench does not implement phase or ovulation prediction.

## External model participation

An authorized user prepares the same private bundle, observes only feature/training views permitted by each fold/track, and emits the versioned long-form schema. The explicit private manifest lists exact files and hashes; stale directory files cannot enter evaluation. The evaluator rejects missing, duplicate, unexpected, negative, non-finite, or truth-bearing predictions.

The benchmark publishes only aggregate leaderboards, fold summaries, and figures. Participant mappings, sample IDs, truth, predictions, calibration views, checkpoints, and participant metrics remain ignored under `artifacts/private/v1/`.

## Reference methods and uncertainty

The active classical registry has exactly participant-equal population median, wearable Ridge, and participant-weighted CatBoost. The separately tagged Diana-H3P reference first stacks those three population experts using participant-grouped development OOF predictions, then updates a joint three-hormone empirical-Bayes participant effect from K=0/3/7 authorized readings. A compact model suits an effective independent sample size of 20 participants; a large neural model would add variance and tuning latitude without evidence that the data support it.

Diana-H3P uses one fold-local continuous Ledoit-Wolf correlation-shrinkage estimator rather than a global diagonal/full choice. Its 80% participant-block calibrated research prediction intervals describe benchmark residual uncertainty, not individual clinical confidence or diagnostic safety. Strong longitudinal overlap weakens ordinary conformal exchangeability assumptions.

The legacy custom model is retained as `historical_protocol_compromised_comparator` because a globally selected fold-0 covariance mode was influenced by participants later used as outer test. H3P removes that path, but its new evaluation remains post-hoc because earlier benchmark test aggregates had already been inspected.

## Scope limitations

Targets are consumer at-home urinary-monitor values entered by participants, not clinical serum measurements or gold-standard concentrations. Results are descriptive for this small governed cohort and are not validated for diagnosis, pregnancy, PCOS, perimenopause, menopause, hormonal contraception, clinical decisions, all women, or causal physiological conclusions.
