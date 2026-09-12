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
