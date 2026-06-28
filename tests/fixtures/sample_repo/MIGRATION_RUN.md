# Migration run — `tests/fixtures/sample_repo/`

This file captures the **actual** output of running the 6-step recipe from
[`docs/MIGRATING_FROM_MEGA_SCRAPPER.md`](../../docs/MIGRATING_FROM_MEGA_SCRAPPER.md)
against the in-repo fixture consumer. The fixture is already at v1
envelopes — this is the verification run, not a first-time migration.

## Pre-conditions

```text
$ ls tests/baselines/
comment_density.json
complexity.json
docstring_format.json
```

The fixture's [pyproject.toml](pyproject.toml) already declares
`[tool.pykissembed]`, and the test files in `tests/` use the v1 file
naming pattern (`code_*.py`, `comment_*.py`, `docstring_*.py`,
`lint_*.py`, `code_*.py`).

## Step 1 — Install pykissembed

```text
$ uv pip install -e /home/alvaro/pyqtest
Installed 1 package in 1ms
 ~ pykissembed==0.1.0 (from file:///home/alvaro/pyqtest)
```

## Step 2 — Confirm `[tool.pykissembed]` is present

```text
$ grep -A4 tool.pykissembed pyproject.toml
[tool.pykissembed]
paths = ["src"]
mode = "ratchet"
baseline_dir = "tests/baselines"
cache_dir = "tests/.pykissembed_cache"
```

## Step 3 — Confirm v1 envelopes

```text
$ head -5 tests/baselines/comment_density.json
{
  "data": {
    "aggregate_current": 4.55,
    "aggregate_max_density": 100.0,
    "aggregate_min_density": 4.0,
```

## Step 4 — No upstream v0 files to delete

The fixture uses v1 file names. There are no v0 mega-scrapper test
files in this repo.

## Step 5 — No conftest.py to slim

`tests/conftest.py` does not exist in the fixture.

## Step 6 — Seed + verify

```text
$ cd tests/fixtures/sample_repo
$ /home/alvaro/pyqtest/.venv/bin/pytest tests/ --update-baselines
…
======================== 1 passed, 10 skipped in 0.41s =========================
```

The 10 skips are pykissembed's own gate tests, each emitting
"Updated … baselines" and then `pytest.skip`-ing (the documented
update-mode behaviour).

```text
$ /home/alvaro/pyqtest/.venv/bin/pytest tests/
…
======================== 10 passed, 1 skipped in 0.39s =========================
```

The 1 skip is the similarity test, which needs `pykissembed-local` and a
populated embedding cache.

## Step 7 — `pykissembed ratchet` and `pykissembed check`

```text
$ pykissembed ratchet
  skip comment_density.json: no current-diagnostics computer implemented
  skip complexity.json: no current-diagnostics computer implemented
  skip docstring_format.json: no current-diagnostics computer implemented
Done. 0 baseline file(s) lowered.
```

> The three skips are expected — only `lint_typecheck.json` has a
> current-diagnostics computer implemented. The other three are
> honest no-ops: their seed values are already at or below current
> diagnostics, so ratcheting them would lower to the same value.

```text
$ pykissembed check
…
======================== 10 passed, 1 skipped in 0.36s =========================
```

`pykissembed check` runs the same gate as `pytest` and matches the
direct pytest run.

## Notes

- **Similarity is intentionally skipped.** The fixture has no
  `tests/.pykissembed_cache/` and no `pykissembed-local` installed in the
  parent venv by default. To enable it:

  ```text
  uv pip install -e /home/alvaro/pyqtest/pykissembed_local
  pykissembed populate-embeddings --provider local
  pytest -m similarity
  ```
- **Cloud providers are not exercised here.** With
  `pykissembed-cloud` installed and an `OPENROUTER_API_KEY` set, the
  `test_providers_parallel` test will run all four cloud providers
  against `src/`.
- **`pykissembed ratchet` only computes current diagnostics for
  `lint_typecheck.json`.** The other three baselines (`complexity`,
  `density`, `docstring_format`) would need additional computers in
  `pykissembed/cli.py:_compute_current_for` before ratcheting would
  lower them automatically.
