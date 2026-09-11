# v0.2.2 release notes (draft)

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
