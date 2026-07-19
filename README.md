# DIANA

Diana is a Hack-Nation research-infrastructure project for reproducible evaluation in women’s hormonal health. Its core reusable contribution is **Hormonbench**: a governed-data, causal, participant-independent benchmark for next-day forecasting of participant-entered at-home urinary-monitor readings.

Hormonbench is not a diagnostic product, clinical decision system, serum-hormone model, or verified ovulation predictor. It does not establish clinical validity or causal physiology.

```text
+------------------------+
| Licensed mcPHASES data |
+-----------+------------+
            |
            v
+------------------------+
| Hormonbench v1 adapter |
| t-13...t -> t+1 task   |
+-----------+------------+
            |
            v
+------------------------------+
| Private prepared bundle      |
| rows, truth, folds, IDs      |
| artifacts/private/v1/...     |
+-------------+----------------+
              |
      +-------+--------+
      |                |
      v                v
+----------------+  +----------------------+
| Feature-only   |  | Private evaluator    |
| model views    |  | truth/calibration    |
+-------+--------+  +-----------+----------+
        |                       ^
        v                       |
+------------------------+      |
| Classical baselines    |      |
| - population_median    |      |
| - wearable_ridge       |      |
| - catboost             |      |
+-----------+------------+      |
            |                   |
            v                   |
+------------------------+      |
| Diana-H3P Layer 1      |      |
| stacked wearable prior |      |
+-----------+------------+      |
            |                   |
            v                   |
+-----------------------------+ |
| Diana-H3P Layer 2           | |
| personalization K=0/3/7     | |
| research intervals          | |
+-------------+---------------+ |
              |                 |
              v                 |
+-----------------------------+ |
| Prediction CSVs             | |
| manifest + SHA-256 hashes   | |
+-------------+---------------+ |
              |                 |
              +-----------------+
              |
              v
+-----------------------------+
| Aggregate public results    |
| privacy/release validation  |
+-----------------------------+
```

The benchmark owns the task, schemas, folds, private-truth scoring, and privacy checks. The models only consume approved feature/calibration views and emit prediction files in the required schema.

## Hormonbench-mcPHASES v1

Task `hormonbench_mcphases_interval2_nextday_v1` uses exactly 14 calendar days (`t-13` through `t`) of approved Interval-2 wearable summaries to predict genuinely observed urinary LH, E3G, and PdG readings at `t+1` in log1p space.

Five deterministic participant groups cover all 20 eligible participants and 1,509 origins. Each outer fold starts as 12 training / 4 validation / 4 test participants, then refits on 16 development participants before one four-participant test. Every participant is test once and validation once. These are five folds of one participant-independent protocol, not five random seeds. Participant-macro metrics prevent people with more overlapping origins from dominating.

Two tracks answer different questions:

- `cold_start_participant_independent` gives the held-out participant no personal hormone labels;
- `few_shot_personalization` authorizes only the earliest K=0, 3, or 7 complete personal readings and scores every budget on the same 1,369-origin later suffix.

v1 removed the stale v0 menstrual-calendar feature: Interval 2 has no current bleeding reports, so the earlier feature was a 773–932-day-old Interval-1 state. v1 also excludes structurally absent self-reports.

## Benchmark and model boundary

- [`benchmark/`](benchmark/) owns adaptation, task/fold/track contracts, schemas, private-truth evaluation, participant-macro metrics, reporting, and privacy validation. Its evaluator imports no model code.
- [`model/`](model/) owns exactly three active classical baselines—`population_median`, `wearable_ridge`, and `catboost`—plus the separately tagged Diana-H3P reference.
- [`model/diana_h3p/`](model/diana_h3p/) implements the Budget-Aware Hierarchical Tri-Hormone Personalizer.
- [`model/joint_bayes_personalizer/`](model/joint_bayes_personalizer/) is preserved only as `historical_protocol_compromised_comparator`; its global fold-0 covariance choice used a participant group that later served as outer test.
- `artifacts/private/v1/` contains licensed rows, IDs, fold mappings, calibration views, predictions, fitted parameters, and participant metrics. It is ignored in full.
- [`results/v1/diana_h3p/`](results/v1/diana_h3p/) contains only aggregate active-reference results after the canonical run.

Diana-H3P demonstrates the benchmark’s cold-start, measurement-budget, joint-personalization, and uncertainty contracts. It is not a fourth classical baseline or a separate product. Its evaluation is explicitly post-hoc: previous outer-test results were inspected before H3P was designed, so it is not an untouched-test confirmation.

## Diana-H3P

Layer 1 learns a participant-balanced convex stack of the population median, wearable Ridge, and CatBoost using fresh participant-grouped development OOF predictions. CatBoost stopping is nested inside each OOF block, so the held group influences neither preprocessing nor tree count.

Layer 2 learns fold-local continuously shrunk three-hormone covariance matrices. K=3 and K=7 use the uncertainty of the exact chronological calibration procedure rather than assuming independent measurements. It produces 80% participant-block calibrated research prediction intervals—not clinical confidence intervals.

NumPy is the frozen canonical Layer-2 backend. Optional PyTorch CPU/CUDA paths match in float64, but complete-path profiling found them slower for this small 3×3 workload.

## Canonical descriptive result

The population median remained strongest in cold start (0.636527 versus 0.644181 for Diana-H3P) and at K=3 (0.610852 versus 0.616188). CatBoost was strongest at K=7 (0.607195 versus 0.608918). H3P therefore did not beat the strongest comparator at any official budget; at K=7 it improved LH and E3G relative to CatBoost but worsened PdG. These are post-hoc, descriptive, nonclinical results—not a superiority claim. See the [aggregate H3P report](results/v1/diana_h3p/RESULTS.md).

## Reproduce

Obtain mcPHASES 1.0.0 directly from PhysioNet under its data-use agreement. In the intended Python 3.11 environment:

```powershell
python -m pytest -q
python scripts/run_diana_h3p_v1.py --synthetic
python scripts/run_diana_h3p_v1.py --verify-only
python scripts/run_diana_h3p_v1.py --development-only
python scripts/run_diana_h3p_v1.py
```

The official runner installs nothing, uses explicit private manifests, runs outer folds sequentially, and refuses to overwrite an existing canonical manifest.

## Documentation

- [benchmark guide](benchmark/README.md)
- [v1 task card](benchmark/docs/TASK_CARD_V1.md)
- [v1 benchmark card](benchmark/docs/BENCHMARK_CARD_V1.md)
- [data card](benchmark/docs/DATA_CARD.md)
- [governed data access](benchmark/docs/DATA_ACCESS.md)
- [Diana-H3P model card](model/diana_h3p/MODEL_CARD.md)
- [challenge narrative](docs/CHALLENGE_SUBMISSION.md)
- [90-second demo script](docs/DEMO_SCRIPT.md)
- [release safety](docs/RELEASE_SAFETY.md)

## License and scope

Diana’s original code and project documentation are MIT licensed. mcPHASES and third-party data/materials are excluded and retain their own terms. Dataset rows, participant/sample IDs, row predictions, fitted private parameters, and participant-level results are never redistributed.

## Consent-based web prototype

DIANA is a high-fidelity frontend prototype for consent-based women's health-data infrastructure. Participants choose which categories and approved research projects they support, while researchers assess aggregate dataset availability, completeness, follow-up, and governance status without seeing identifiable participant information.

The prototype does not provide diagnoses, treatment, scientific conclusions, real authentication, medical-data integrations, or access to real health records.

## Architecture

- `frontend/`: React, TypeScript, Vite, Tailwind CSS, shadcn-style Radix components, and Lucide icons.
- `backend/src/main.py`: FastAPI health endpoint and production SPA host.
- `frontend/dist/`: generated production bundle served by FastAPI.
- `Dockerfile.vercel`: multi-stage frontend build and single FastAPI production service for Vercel Container Images.

All interactive data is synthetic and stored locally in the browser. The frontend and backend remain separate during development, but production runs only one FastAPI service.

## Requirements

- Node.js 22.12 or later
- npm 10 or later
- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)

## Production-Style Local Run

Install and build the frontend:

```bash
npm install --prefix frontend
npm run build --prefix frontend
```

Install the backend and serve the complete application:

```bash
uv sync --project backend
uv run --project backend uvicorn backend.src.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. FastAPI serves the Vite assets and returns the SPA entry point for routes such as `/projects/mcphases` and `/participant/dashboard`.

The health endpoint is available at <http://localhost:8000/api/health>.

## Prototype Authentication

FastAPI accepts any non-empty username and password for the selected prototype role, then stores only the username and role in a signed, HTTP-only session cookie. Passwords are never stored in the browser or server session.

Before a public deployment, configure these environment variables in Vercel:

```text
DIANA_SESSION_SECRET=<long-random-secret>
DIANA_SECURE_COOKIES=true
```

Vercel sets `VERCEL_ENV` automatically, and DIANA automatically enables secure cookies in preview and production. The Vercel image generates a private signing secret when `DIANA_SESSION_SECRET` is not configured, so deployment can start with no additional setup. Set an explicit secret to keep sessions valid across image rebuilds.

Generate a session secret locally with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

This is prototype access control, not a registration system or persistent user database.

## Development

Start FastAPI with hot reload:

```bash
make api
```

Start Vite in a second terminal:

```bash
make web
```

Open the Vite URL shown in the terminal. Vite proxies `/api` requests to FastAPI at `127.0.0.1:8000`.

## Docker

Build and run one production service:

```bash
docker build -f Dockerfile.vercel -t diana .
docker run --rm -p 8000:80 diana
```

The final image contains the compiled frontend and starts only FastAPI.

## Prototype Routes

Public:

- `/`
- `/tree`
- `/projects`
- `/projects/:projectId`

Participant:

- `/participant/login`
- `/participant/contribution-choice`
- `/participant/data-types`
- `/participant/project/:projectId/contribute`
- `/participant/consent/:dataType`
- `/participant/dashboard`

Scientist:

- `/scientist/login`
- `/scientist/terms`
- `/scientist/dashboard`

## Quality Checks

```bash
npm run lint --prefix frontend
npm run typecheck --prefix frontend
npm run build --prefix frontend
uv run --project backend python -m py_compile backend/src/main.py
```

## Data, Consent, and Scientific Assumptions

- Project names, institutions, counts, coverage percentages, feasibility estimates, and downloads are synthetic demo content.
- No mcPHASES, NHANES, clinical, wearable, laboratory, or participant-level records are bundled or processed.
- The approved demo download is an aggregate synthetic availability manifest, not a scientific dataset.
- Participant contribution changes add one synthetic aggregate match to compatible projects; withdrawal removes that derived match.
- Optional permissions begin unselected. Contribution state changes only after the consent form is signed.
- Project-specific permission and broader eligible-project permission are represented separately.
- Withdrawal stops future simulated access; the interface explains that already-completed research may not be reversible.
- Scientist views contain aggregate values only and never expose names, signatures, email addresses, or participant rows.
- Feasibility values are not final sample-size calculations and do not establish scientific suitability.
- DIANA does not replace ethics approval, governance review, or a data-use agreement.

The application source is reusable under the repository's chosen license. Third-party packages retain their respective licenses.
