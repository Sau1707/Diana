# Joint Bayes Personalizer — historical protocol-compromised comparator

`joint_bayes_personalizer` is preserved for historical reproducibility but is inactive. Its diagonal/full covariance mode was selected globally using fold-0 validation participants who later served as outer test in another rotating fold. It is therefore labeled `historical_protocol_compromised_comparator`, not an active custom reference or clean confirmatory comparator. Diana-H3P replaces it.

It combines a participant-equal population median with a shrunk, participant-weighted CatBoost wearable prior. Development-set grouped out-of-fold predictions determine a per-hormone shrinkage factor in `[0,1]`. A three-dimensional empirical-Bayes residual model then uses zero, three, or seven authorized personal urinary-hormone measurements to estimate a shrunk participant offset across LH, E3G, and PdG.

The model reads only the benchmark-prepared wearable feature matrix. It does not read raw CSVs, participant ID as a predictor, absolute time, menstrual-calendar fields, self-reports, hormone history, Mira phase, or future labels. Participant keys are used privately only for grouped OOF fitting, equal weighting, authorized calibration, and evaluation.

Research 80% prediction intervals combine residual noise and posterior participant uncertainty, then use participant-grouped development OOF conformal multipliers. They are not clinical confidence intervals. Overlapping longitudinal windows, the small cohort, consumer-device measurements, and cohort selection limit calibration and generalization claims.

The model is CPU-suitable and intentionally compact. It is not a diagnosis system, serum-hormone estimator, ovulation predictor, or evidence of clinical utility.

## Frozen v1 selection and result

Fold-0 validation compared diagonal and full 3×3 residual covariance on the prespecified K=3/K=7 scalar. Neither met the superiority guardrails, but full scored 0.602186 versus 0.606438 for diagonal. The strongest-valid-candidate fallback therefore froze full. An initial diagonal execution was invalidated as a protocol implementation defect; valid baseline files were reused and only the custom path was recomputed without using outer-test scores for selection.

Learned wearable shrinkage was weak and fold-dependent: fold-wise `(LH, E3G, PdG)` lambdas were `(0,0,0)`, `(0,0.1,0.1)`, `(0.15,0,0.35)`, `(0,0,0.1)`, and `(0.75,0,0)`. This supports shrinkage rather than assuming wearable signal is uniformly useful.

The custom reference scored 0.642639 in cold start versus 0.636527 for population median. On the common suffix it scored 0.640382, 0.611701, and 0.605710 for K=0/3/7. At K=7, CatBoost scored 0.607195. The custom reference was marginally best overall and on LH/E3G at K=7 but worsened PdG versus CatBoost. These descriptive results did not meet the validation superiority goal and must not be presented as statistically established superiority.

Across K=7, participant-macro 80% research-interval coverage was approximately 0.799 for LH, 0.784 for E3G, and 0.801 for PdG, with respective mean log1p widths about 1.295, 1.232, and 2.182. Window overlap weakens ordinary exchangeability assumptions, so near-nominal coverage is benchmark characterization rather than a clinical guarantee.

Aggregate decomposition is published as participant-macro mean absolute wearable and personal adjustment magnitudes; development median values and all participant-level decompositions remain private.
