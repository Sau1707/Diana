# mcPHASES data card for Hormonbench v1

## Governed source and active subset

mcPHASES 1.0.0 contains physiological, hormonal, event, and symptom records from two collection intervals. Hormonbench v1 uses Interval 2 (`2024`) only. The frozen adapter retains 20 eligible participants and 1,509 origins with genuinely observed joint next-day targets. The K=0/3/7 common scoring suffix retains all 20 participants and 1,369 origins.

Every Interval-2 participant also appears in Interval 1, so the benchmark does not treat the deferred 2022→2024 comparison as unseen-participant evaluation. That shift track is not implemented here.

## Targets

The target table contains participant-entered transcriptions of at-home Mira urinary-monitor readings:

| Target | Source column | Documented unit | Interpretation |
|---|---|---|---|
| Urinary LH | `lh` | mIU/mL | Consumer urine reading |
| Urinary E3G | `estrogen` | ng/mL | Estrone-3-glucuronide urine reading |
| Urinary PdG | `pdg` | mcg/mL | Pregnanediol-glucuronide urine reading |

Urinary E3G is not serum estradiol/E2, urinary PdG is not serum progesterone, and these readings are not clinical gold-standard concentrations. Participant transcription, device limits, floor/ceiling heaping, and missingness introduce measurement error. Labels are never interpolated for evaluation.

Mira `phase` and fertile-window fields are proprietary app-generated weak labels derived from hormone patterns, not verified clinical ovulation ground truth. They are excluded.

## Active v1 input modalities

Only these causally aligned Interval-2 sources enter the candidate feature matrix:

- daily active-minute totals;
- computed temperature aligned to `sleep_end_day_in_study` (wake/end day);
- daily HRV aggregates;
- valid wake-day respiratory summaries, with nonpositive unavailable sentinels masked;
- sleep-score fields, including its resting-heart-rate field;
- calendar-known weekend state; and
- causal lag, summary, coverage, missingness, and time-since-observation transforms ending at the origin day.

The complete self-report family is excluded. Interval 2 contains almost no self-report data and no usable `flow_volume`, so structural absence could otherwise act as a participant signature. Menstrual-calendar fields are also excluded: Interval 2 lacks a current onset, and v0’s carried Interval-1 state was 773–932 days stale.

Participant ID, sample ID, absolute day/date, interval identity, hormone history, Mira labels, completed-cycle quantities, target-derived events, centered windows, backward fill, future interpolation, and future wearable values cannot enter a model feature matrix.

## Missingness and interval differences

- Interval 1 has no PdG collection; this is structural non-collection.
- Glucose was collected only in Interval 1 and is structurally absent from Interval 2.
- Sleep-score components and compact wearable coverage vary by participant and day.
- Temperature baseline-relative fields can be absent during device baseline establishment.
- Respiratory sources contain unavailable sentinels that are converted to missing.
- Deterministic causal aggregation handles duplicate source records without using future revisions.

Missingness masks preserve observation state but do not make missingness random.

## Deferred sources

Raw heart rate, calories, distance, raw steps, wrist temperature, sleep events, glucose, oxygen variation, exercise events, altitude, raw resting-heart-rate tables, VO2 estimates, stress, and other high-volume streams are outside v1. Seven-day forecasting, same-day nowcasting, 2022→2024 shift evaluation, phase/ovulation classification, and clinical tasks are also deferred.

## Scope limitations

The urine-test table lacks reliable time-of-day ordering, so Hormonbench uses next-day rather than same-day prediction. The cohort has only 20 participants; adjacent 14-day histories overlap by 13 days and are strongly correlated. The benchmark measures descriptive predictive error within this governed cohort. It does not establish diagnosis, treatment benefit, fertility status, causality, clinical utility, or generalization to all women, devices, pregnancy, PCOS, perimenopause, menopause, or hormonal contraception.
