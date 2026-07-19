# Diana-H3P on Hormonbench-mcPHASES v1

> Descriptive, post-hoc research-benchmark results on 20 participants; not clinical validation or an untouched-test confirmation.

Hormonbench remains Diana's primary reusable contribution. Diana-H3P is one compact reference implementation with a participant-independent stacked wearable prior and K=0/3/7 empirical-Bayes personalization.

Cold-start leader: `population_median` at 0.636527 (lower is better).
Diana-H3P scored 0.644181; it did not beat the strongest comparator in cold start or at K=3/K=7. At K=7 it was 0.28% worse overall than CatBoost while improving LH and E3G and worsening PdG. No superiority claim is warranted.

| Model | Overall | LH log-MAE | E3G log-MAE | PdG log-MAE | Improved participants |
|---|---:|---:|---:|---:|---:|
| `population_median` | 0.636527 | 0.439421 | 0.431112 | 0.669523 | 0/20 |
| `diana_h3p` | 0.644181 | 0.449617 | 0.432165 | 0.674558 | 10/20 |
| `catboost` | 0.651631 | 0.452675 | 0.439782 | 0.681653 | 7/20 |
| `wearable_ridge` | 0.824231 | 0.585284 | 0.544308 | 0.866884 | 1/20 |

## Few-shot personalization

| K | Population median | Wearable Ridge | CatBoost | Diana-H3P |
|---:|---:|---:|---:|---:|
| 0 | 0.633978 | 0.819769 | 0.647833 | 0.642250 |
| 3 | 0.610852 | 0.822713 | 0.611874 | 0.616188 |
| 7 | 0.613115 | 0.820900 | 0.607195 | 0.608918 |

Diana-H3P fold-score standard deviations were 0.08274 (cold start), 0.08791 (K=0), 0.05001 (K=3), and 0.05667 (K=7).

K counts the earliest authorized complete urinary LH/E3G/PdG measurements. All budgets use the same post-seventh-measurement scoring suffix; calibration rows are never scored.

## Research uncertainty

Coverage / mean width in log1p units:

| K | LH coverage / width | E3G coverage / width | PdG coverage / width |
|---:|---:|---:|---:|
| 0 | 0.781 / 1.322 | 0.789 / 1.332 | 0.785 / 1.883 |
| 3 | 0.784 / 1.317 | 0.796 / 1.305 | 0.783 / 1.937 |
| 7 | 0.791 / 1.308 | 0.794 / 1.229 | 0.784 / 1.956 |

Intervals are 80% participant-block calibrated research prediction intervals. Empirical coverage ranged from 0.781 to 0.796. Correlated overlapping windows, the small cohort, and a fixed Layer-1 stack during interval pseudo-LOPO calibration preclude an IID finite-sample or clinical-confidence interpretation.

The prior `joint_bayes_personalizer` is retained only as `historical_protocol_compromised_comparator`: a fold-0 validation group selected a global covariance mode before later appearing as outer test. It is not part of this active leaderboard.

Targets are participant-entered readings from an at-home urinary monitor, not serum concentrations, diagnoses, verified ovulation labels, or clinical gold standards.
