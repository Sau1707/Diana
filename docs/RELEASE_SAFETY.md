# Public release safety

Never ZIP or upload the working directory. Workspace archives can include governed prepared rows, participant mappings, calibration data, predictions, checkpoints, caches, or bytecode even when those paths are ignored by Git. Existing local ZIP files are preserved but ignored and must never be staged or published.

The release source is an intentionally reviewed Git index and, after commit, a `git archive` of that exact commit. `.gitignore` is a guardrail, not a release builder.

## Public allow-list

Only these roots are eligible:

- `benchmark/`
- `model/`
- `configs/`
- `scripts/`
- public-safe `reports/`
- aggregate-only `results/`
- `docs/`
- root `.gitignore`, `README.md`, `LICENSE`, and `pyproject.toml`

Always reject `.git/`, `dataset/`, `artifacts/private/`, participant-bearing Phase 0 reports, `.pytest_cache/`, `__pycache__/`, bytecode, ZIP files, private split/calibration manifests, sample-level predictions, truth rows, participant/sample identifiers in result tables, checkpoints with restricted material, private absolute paths, and credentials.

## Before staging

1. Run the complete tests and public-result validation:

   ```powershell
   python -m pytest -q
   python scripts/run_diana_h3p_v1.py --privacy-only
   ```

2. Review the complete candidate inventory and status. A repository without a prior commit requires explicit untracked-file review because `git diff` alone is insufficient:

   ```powershell
   git status --short --branch
   git ls-files --cached --others --exclude-standard
   ```

3. Confirm `dataset/`, `artifacts/private/`, participant-bearing reports, caches, bytecode, and all ZIPs are ignored. Inspect every unexpected file.

4. Stage only the reviewed allow-list paths with explicit pathspecs. Never use `git add .`.

## Exact staged-index check

Before each commit, inspect and scan exactly what is staged:

```powershell
git status --short
git diff --cached --name-only
git diff --cached --stat
python -c "from benchmark.v1_privacy import validate_staged_index; print(validate_staged_index('.'))"
```

The staged scan must fail on a forbidden path, ZIP, secret, private absolute path, participant/sample identifier column, or truth-bearing public result. Resolve every rejection by unstaging or correcting the public artifact; never weaken the deny-list to make a release pass.

## Candidate-commit archive check

After committing, create a temporary archive from the reviewed commit—not from the filesystem—and scan it:

```powershell
$releaseAudit = Join-Path $env:TEMP "diana-public-audit.zip"
git archive --format=zip --output=$releaseAudit HEAD
python -c "from benchmark.v1_privacy import validate_git_archive; print(validate_git_archive(r'$releaseAudit'))"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($releaseAudit)
try { $archive.Entries.FullName } finally { $archive.Dispose() }
```

Review the complete listing. The archive must contain only tracked allow-listed public material. Remove the temporary audit archive after verification; never copy it back into the workspace.

## Before and after push

Before push, verify the remote URL and fetch remote refs without rewriting history. Confirm the branch is empty/aligned, both intended commits are present, the full tests pass, the staged/index/archive privacy checks pass, and no ignored/private file is tracked. Never force-push or embed credentials in a URL.

After push, verify the remote commit and tree, confirm the public README is accessible, and search the remote file list for forbidden roots or suffixes. Dataset rows and private artifacts remain local regardless of repository visibility.

