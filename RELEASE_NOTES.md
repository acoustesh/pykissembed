# v0.2.14 release notes

Changes since `v0.2.13`:

- Fix Gemini embedding requests: `gemini-embedding-2` folded a list of strings
  into a single vector, so populating embeddings failed with
  `zip() argument 2 is shorter than argument 1`. Each text is now sent as its
  own content and gets its own vector.

- [pykissembed 0.2.14](https://test.pypi.org/project/pykissembed/0.2.14/)
- [pykissembed-cloud 0.2.14](https://test.pypi.org/project/pykissembed-cloud/0.2.14/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.13...v0.2.14)

# v0.2.13 release notes

Changes since `v0.2.12`:

- Gemini provider now defaults to `gemini-embedding-2` (config and
  similarity constants updated to match).

- [pykissembed 0.2.13](https://test.pypi.org/project/pykissembed/0.2.13/)
- [pykissembed-cloud 0.2.13](https://test.pypi.org/project/pykissembed-cloud/0.2.13/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.12...v0.2.13)

# v0.2.12 release notes

Changes since `v0.2.11`:

- Checks in a mixed run now show as passed instead of skipped. 0.2.11
  deselected them, and VS Code's test tree shows deselected tests as skipped.
  They still do not run: no fixtures are set up and no embeddings are
  computed, so a green check from a mixed run is not a real result. Selecting
  only checks, `--pykissembed-all`, `-m <marker>` and ini `testpaths` entries
  that point at the checks still run them for real.

- [pykissembed 0.2.12](https://test.pypi.org/project/pykissembed/0.2.12/)
- [pykissembed-cloud 0.2.12](https://test.pypi.org/project/pykissembed-cloud/0.2.12/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.11...v0.2.12)

# v0.2.11 release notes

Changes since `v0.2.10`:

- Running the root of VS Code's test tree no longer runs pykissembed's checks.
  The tree still lists them, so "run all" passed their test IDs along with the
  project's own tests. A run that mixes project tests with checks now skips
  the checks. Selecting only checks (the checks folder or a single check)
  still runs them, and so do `--pykissembed-all`, `-m <marker>` and ini
  `testpaths` entries that point at the checks.

- [pykissembed 0.2.11](https://test.pypi.org/project/pykissembed/0.2.11/)
- [pykissembed-cloud 0.2.11](https://test.pypi.org/project/pykissembed-cloud/0.2.11/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.10...v0.2.11)

# v0.2.10 release notes

Changes since `v0.2.9`:

- Stop pykissembed's checks from running with a consumer's `tests` folder in
  VS Code. `pykissembed init` used to set Test Explorer's pytest args to
  `["tests", "--pykissembed-all"]`, so running `tests` also ran the full check
  set. It now writes plain `["tests"]`. The checks still show up in the test
  tree, but they run only when you select them, run `pykissembed check`, or
  pass `--pykissembed-all`. Re-run `pykissembed init` to migrate settings that
  an older version generated; values you edited by hand are left alone.
- Add `pykissembed docfix`, a guarded NumPy docstring fixer. It repairs
  sections and entries deterministically from facts it reads out of the AST,
  without changing code, and writes files atomically. It supports reports,
  diffs, check mode, symbol filters and dirty-tree protection, and can
  optionally draft text through OpenRouter and grade it with Jev.
- Type the locked envelope context managers as generators, and assert that
  malformed or out-of-range Jev scores emit "ungraded" warnings.
- Refresh locked dependencies (google-genai 2.28.0, openai 3.24.0).

- [pykissembed 0.2.10](https://test.pypi.org/project/pykissembed/0.2.10/)
- [pykissembed-cloud 0.2.10](https://test.pypi.org/project/pykissembed-cloud/0.2.10/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.9...v0.2.10)

# v0.2.9 release notes

Changes since `v0.2.8`:

- Tolerate start-up noise in tool output. `test_no_lint_or_type_errors` and
  the docstring-format check now decode ruff/pyright JSON starting at the
  first line that begins a valid document. Warnings printed to stdout ahead of
  it (for example `[cuml.accel] Warning: ...` from an environment that loads
  `cuml.accel`) no longer crash the lint gate with `JSONDecodeError`, and no
  longer make the docstring-format gate pass silently by discarding the output.

- [pykissembed 0.2.9](https://test.pypi.org/project/pykissembed/0.2.9/)
- [pykissembed-cloud 0.2.9](https://test.pypi.org/project/pykissembed-cloud/0.2.9/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.8...v0.2.9)

# v0.2.8 release notes

Changes since `v0.2.7`:

- Fix GPU PCA under `cuml.accel`. That mode (and
  `cuml.set_global_output_type("numpy")`) makes cuML return host NumPy arrays
  from `transform()` and `explained_variance_ratio_`, which overrides the
  estimator's own `output_type` and broke the CuPy operations that consume
  them (`TypeError: cuML transform() must return a sliceable CuPy array`).
  PCA fit, variance read and transform now pin cuML's output type to CuPy with
  `cuml.using_output_type("cupy")`.
- Refresh locked package versions.

- [pykissembed 0.2.8](https://test.pypi.org/project/pykissembed/0.2.8/)
- [pykissembed-cloud 0.2.8](https://test.pypi.org/project/pykissembed-cloud/0.2.8/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.7...v0.2.8)

# v0.2.7 release notes

Changes since `v0.2.6`:

- Fix GPU PCA on wide inputs (more features than samples). cuML
  `IncrementalPCA` reports `explained_variance_ratio_` as a host NumPy array,
  and `cupy.cumsum` rejected it with a `TypeError`. The ratio is now moved to
  the device before the cumulative sum, so it works for both PCA variants.
- When Combined embeddings cannot be built because member caches are
  incomplete, the hint now also names the bulk command
  `uv run pykissembed populate-embeddings --provider all`.

- [pykissembed 0.2.7](https://test.pypi.org/project/pykissembed/0.2.7/)
- [pykissembed-cloud 0.2.7](https://test.pypi.org/project/pykissembed-cloud/0.2.7/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.6...v0.2.7)

# v0.2.6 release notes

Changes since `v0.2.5`:

- Add two Jev-backed audits, `jev_docstring_audit` and `jev_comment_audit`.
  They grade docstrings and inline comments through the OpenRouter Decisions
  API (`typesafe/jev-1.14`) and cache valid responses in
  `jev_cache.sqlite3` in the baseline directory. Both run with the other
  consumer checks under the `jev` marker. Without `OPENROUTER_API_KEY` they
  grade cached responses only and skip cache misses.
- Add a `gpu` extra that installs `cuml-cu13==26.8.*` for CUDA 13
  acceleration. cuML is published on `https://pypi.nvidia.com`, so consumers
  must add that index themselves.
- Fix comment density so every docstring line is left out of the code-line
  count. Before this fix, adding a docstring section lowered a file's density.
- Skip hidden directories such as `.claude/` and `.kilo/` worktrees when
  scanning source and when running the no-suppressions check.
- Keep the Gemini client alive between embedding requests. Previously,
  garbage collection closed its HTTP transport and later requests failed.
- Share one classifier for timeout, 429 and 5xx retry decisions between the
  Jev and Voyage transports.

**Behavior change:** projects that set `OPENROUTER_API_KEY` for the cloud
providers will now also send function and class source to the Jev audits on
their next test run. A symbol that scores below the baseline threshold fails
the run. To skip both audits, run `pytest -m "not jev"`.

The core `numpy` requirement changed from `>=1.26` to `<2.5` so that it stays
compatible with cuML.

- [pykissembed 0.2.6](https://test.pypi.org/project/pykissembed/0.2.6/)
- [pykissembed-cloud 0.2.6](https://test.pypi.org/project/pykissembed-cloud/0.2.6/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.5...v0.2.6)

# v0.2.5 release notes

Changes since `v0.2.4`:

- **Behavior change:** `cached_only` now defaults to `false`. Similarity
  checks populate missing embeddings through every configured provider's API
  while tests run. Pass `--cached-only` or set `cached_only = true` in
  `[tool.pykissembed]` to keep code-derived data off cloud providers.
- Tighten type safety and error handling across checks, CLI, plugin, and
  embedding/similarity modules.

Projects that relied on the implicit cache-only default and have provider API
keys configured will now make provider calls on their next test run.

- [pykissembed 0.2.5](https://test.pypi.org/project/pykissembed/0.2.5/)
- [pykissembed-cloud 0.2.5](https://test.pypi.org/project/pykissembed-cloud/0.2.5/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.4...v0.2.5)

# v0.2.4 release notes

Changes since `v0.2.3`:

- Extend the strict no-suppressions gate to detect checker-specific ignore
  directives for Ruff, Pyright, ty, and mypy.
- Preserve token-aware scanning so directive-like text in strings and
  docstrings remains allowed.
- Report checker-specific violation kinds and clarify the gate's failure
  message and documentation.

The stricter scan can expose existing `# ruff:ignore[...]`,
`# pyright: ignore[...]`, `# ty: ignore[...]`, and
`# mypy: ignore-errors` comments that earlier versions did not report.

- [pykissembed 0.2.4](https://test.pypi.org/project/pykissembed/0.2.4/)
- [pykissembed-cloud 0.2.4](https://test.pypi.org/project/pykissembed-cloud/0.2.4/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.3...v0.2.4)

# v0.2.3 release notes

Changes since `v0.2.2`:

- Fix GPU PCA for wide embedding matrices, including the combined provider.
  When samples are fewer than features, use cuML `IncrementalPCA` with a single
  batch containing every sample. Its reduced SVD avoids the feature-by-feature
  covariance allocation while retaining centered PCA, the full variance
  spectrum, and the existing variance-threshold component selection.
- Preserve the existing PCA solver for square and tall GPU inputs and the
  sklearn fallback for CPU inputs.
- Add regression coverage for solver selection, explained variance, cached
  models, unseen embeddings, rank-deficient data, and real GPU execution with
  65,536 features.
- Resolve five test-file Ruff diagnostics and update the README version example.

The wide GPU solver's storage scales with samples times features plus samples
squared. Large datasets still require memory for the input, SVD factors, and
solver workspace, but no feature-squared covariance matrix is constructed.

Validated on a 12 GB RTX 4070 with the full pykissembed combined cache
(364 samples, 37,888 features) and Speechtext combined cache (815 samples,
37,888 features). The 99% variance target selected 321 and 511 components,
respectively, matching an independent float64 CPU Gram-matrix reference.
Maximum cosine differences were below `1e-8`; both cache files were unchanged.
GPU validation used the RAPIDS Python 3.12 environment with only Python 3.14
exception syntax parenthesized in a temporary source copy.

- [pykissembed 0.2.3](https://test.pypi.org/project/pykissembed/0.2.3/)
- [pykissembed-cloud 0.2.3](https://test.pypi.org/project/pykissembed-cloud/0.2.3/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.2...v0.2.3)

# v0.2.2 release notes

Published to TestPyPI on 2026-09-10 from commit `b055fe4`.
Changes since `v0.2.1`:

- Add a strict, non-ratcheted no-suppressions gate to the `lint` checks. It
  detects `# type: ignore`, `# noqa`, and calls to `typing.cast` or
  `typing_extensions.cast` in consumer Python source, including standalone
  scripts outside configured source paths. Tests and common environment,
  dependency, build, VCS, and cache directories are excluded.
- Ship `py.typed` markers in both core and cloud distributions so type checkers
  can use their inline annotations. Add isolated-install typing regressions.
- Compute similarity violations using a shared similarity matrix and vectorized
  candidate filtering. Add regression coverage against reference calculations,
  including exclusions, missing embeddings, and degenerate inputs.
- Refactor embedding-cache handling and PCA helpers, and expand type annotations
  and NumPy-style documentation across the codebase.

The new no-suppressions gate can fail existing consumer projects that use the
listed directives or casts, even when their ratcheted lint baseline passes.
Review those findings when upgrading.

Both packages require Python 3.14 or newer:

- [pykissembed 0.2.2](https://test.pypi.org/project/pykissembed/0.2.2/)
- [pykissembed-cloud 0.2.2](https://test.pypi.org/project/pykissembed-cloud/0.2.2/)
- [Full comparison](https://github.com/acoustesh/pykissembed/compare/v0.2.1...v0.2.2)

Release validation passed: 255 core tests, 32 cloud tests, 12 packaging/install
checks, and metadata checks for all four distribution files. One core test was
skipped because cuML was unavailable. Five existing Ruff preview diagnostics in
test files remained at the release tag.
