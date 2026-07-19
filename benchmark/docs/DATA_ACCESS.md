# Data access and privacy

Hormonbench v1 uses governed mcPHASES 1.0.0 data and redistributes no dataset row. Each user must obtain mcPHASES directly from PhysioNet, satisfy its credentialing requirements, accept the applicable data-use agreement, and follow its storage, attribution, use, and deletion terms. Diana’s MIT license does not alter those obligations.

The default licensed-data location is:

```text
dataset/mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0/
```

If needed, change only `paths.data_root` in `configs/hormonbench_v1.yaml`. Never copy the dataset into a public repository or send governed rows to an external API.

Prepare the frozen v1 bundle with:

```powershell
python -m benchmark prepare --config configs/hormonbench_v1.yaml
```

The following always remain private under ignored `artifacts/private/v1/`:

- prepared rows and private truth;
- participant and sample identifiers;
- participant-to-group assignments and calibration mappings;
- row-level predictions and participant-level metrics;
- fitted parameters or checkpoints whose redistribution is not clearly permitted.

Two Phase 0 files contain participant-bearing information and are intentionally ignored: `reports/phase0/PHASE0_AUDIT.md` and `reports/phase0/target_coverage.csv`. They remain local and unchanged.

Public `results/v1/` files contain aggregate metrics, aggregate counts, safe hashes, runtime summaries, and aggregate figures only. Tests use a fully synthetic fixture. A whole-workspace ZIP is unsafe even when paths are ignored; releases must use the documented public allow-list, staged-index scan, and temporary `git archive` validation.
