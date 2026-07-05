# Migrating from `aa-ml/mega-scrapper/tests/` to pykissembed

This guide walks through replacing the five code-quality test files (plus
the `tests/similarity/` sub-package) in
[aa-ml/mega-scrapper](https://github.com/acoustesh/mega-scrapper) with
the upstream `pykissembed` package (v0.1.4+).

## What you get

pykissembed v0.1.4 is a full port of the mega-scrapper test infrastructure:

- **5 check modules** — `code_complexity`, `code_similarity`,
  `comment_density`, `docstring_format`, `lint_typecheck`
- **12-module similarity sub-package** — `types`, `constants`, `storage`,
  `ast_helpers`, `complexity`, `embeddings`, `pca`, `refactor_index`,
  `file_split`, `checks`, `populate_embeddings`
- **9 embedding providers** — OpenAI-Text/AST, Codestral-Text/AST,
  Voyage-Text/AST, Gemini-Text/AST, Combined (8-way concatenation)
- **Opt-in collection** — check modules are discovered by the pytest
  plugin without copying test files or configuring `testpaths`, but (as of
  v0.1.9) collection is opt-in: pass `--pykissembed-all` for the full
  battery, use a marker (`-m complexity`, etc.), or target a specific
  check NodeId. A bare `pytest` collects nothing from pykissembed.

## Steps

### 1. Add pykissembed as a dependency

#### From TestPyPI (current release channel)

Add to your `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple/"
explicit = true

[tool.uv.sources]
pykissembed = { index = "testpypi" }
pykissembed-local = { index = "testpypi" }
pykissembed-cloud = { index = "testpypi" }
```

Then:

```bash
uv add "pykissembed[all]"
uv sync
uv run pykissembed --version    # should show 0.1.4
```

> **If `uv lock --upgrade-package` doesn't pick up the new version**,
> delete `uv.lock` and run `uv lock` fresh to force a full re-resolution.

#### From production PyPI (once published)

```bash
uv add "pykissembed[all]"
```

### 2. Add `[tool.pykissembed]` block

```toml
[tool.pykissembed]
paths = ["src", "scripts"]
mode = "ratchet"
baseline_dir = "tests/baselines"
cache_dir = "tests/.pykissembed_cache"
include_notebooks = false   # true to run ruff/similarity on .ipynb
```

Or scaffold it automatically:

```bash
uv run pykissembed init
```

> **v0.1.6+:** Notebooks are excluded by default. To opt in,
> add `include_notebooks = true` to the `[tool.pykissembed]` block. This
> affects ruff D-rules (docstring format) and the ruff gate (lint_typecheck).

### 3. Map mega-scrapper baseline files

The mega-scrapper baselines become pykissembed v1 envelopes:

| Mega-scrapper file | pykissembed envelope | Kind |
|---|---|---|
| `tests/baselines/complexity_baselines.json` | `tests/baselines/complexity.json` | `complexity` |
| `tests/baselines/comment_density_baselines.json` (config section) | `tests/baselines/comment_density.json` | `density` |
| `tests/baselines/similarity_baselines.json` | `tests/baselines/similarity.json` | `similarity` |
| `tests/baselines/lint_typecheck_report.json` (live, not a baseline) | unchanged | n/a |

`pykissembed` will auto-migrate v0 (raw dict) → v1 (envelope) on first load
of each file. The data semantics are preserved; only the wrapper changes.

### 4. Delete the mega-scrapper test files

The check modules are collected from the installed `pykissembed` package
(via `--pykissembed-all` or a marker filter, see "What you get" above) —
you no longer need local copies:

```sh
git rm tests/test_code_complexity.py \
       tests/test_comment_density.py \
       tests/test_docstring_format.py \
       tests/test_lint_typecheck.py \
       tests/test_code_similarity.py
rm -rf tests/similarity/
```

### 5. Slim `tests/conftest.py`

The mega-scrapper `conftest.py` currently exposes fixtures that pykissembed
already provides via its plugin. After deleting the test files, remove:

- `pytest_addoption` for `--update-baselines` and `--cached-only`
  (the plugin provides these)
- `update_baselines` and `cached_only` fixtures (plugin provides)
- `shared_baselines`, `shared_functions`, `pca_cache` session-scoped
  fixtures (plugin provides)

Keep only project-specific fixtures (e.g. `run_workon`).

### 6. Run + commit baselines

```sh
uv sync
pytest --pykissembed-all --update-baselines      # one-time seed
git add tests/baselines/
git commit -m "Migrate to pykissembed v0.1.4"
```

Note: `tests/.pykissembed_cache/` (embedding caches) should be **gitignored**.

### 7. Verify

```sh
pytest --pykissembed-all     # should match pre-migration diagnostic counts
pykissembed ratchet              # should be a no-op on a clean tree
```

### 8. The baseline-and-ratchet workflow

pykissembed follows a **ratchet** model: tests fail only when diagnostics
**exceed** their baseline. This lets you adopt pykissembed against an
existing codebase without breaking the build on day one.

```bash
# A. Existing codebase with thousands of violations
pytest --pykissembed-all --update-baselines   # captures current state as baseline
git add tests/baselines
git commit -m "seed baselines"

# B. Day-to-day development: only NEW regressions fail
pytest --pykissembed-all          # any count > baseline fails
pytest -m complexity               # subset by marker (auto-injects on its own)

# C. Periodic ratchet: lower baselines after fixing code
pykissembed ratchet                # walks each baseline file interactively
pykissembed ratchet --dry-run      # show what would change

# D. Schema discovery for a baseline type
pykissembed schema --kind complexity
pykissembed schema --kind similarity
```

A **new file** (no baseline yet) reports as "new violation" until seeded.
To grandfather in lots of files at once:

```sh
# `-k` alone does not trigger pykissembed's collection (v0.1.9+); combine
# it with --pykissembed-all so the check battery is injected first, then
# narrowed by the keyword filter.
pytest --pykissembed-all --update-baselines -k "test_no_lint_or_type_errors"
pytest --pykissembed-all --update-baselines -k "test_docstring_coverage"
pytest --pykissembed-all --update-baselines -k "test_comment_density"
pytest --pykissembed-all --update-baselines -k "test_providers_parallel"
```

---

## Complexity tests

The complexity gate (`pytest -m complexity`) enforces:

- **Docstring coverage** — all functions/methods/classes must have docstrings
- **File line counts** — per-file line count vs baseline
- **Cyclomatic complexity (CC)** — per-function via `radon` (default threshold: 15)
- **Cognitive complexity (COG)** — per-function via `complexipy` (default threshold: 15)
- **Maintainability Index (MI)** — per-file via `radon` (default threshold: 13)

Update baselines:

```bash
pytest -m complexity --update-baselines
```

---

## Similarity tests

The similarity gate (`pytest -m similarity`) detects near-duplicate functions
using 9 embedding providers (8 base + 1 combined).

### Prerequisites

Similarity requires embedding caches. Install an extra and populate:

```bash
# Local — no API key, runs offline once the model is downloaded
pip install "pykissembed[local]"
pykissembed populate-embeddings --provider local

# Cloud — requires API keys
pip install "pykissembed[cloud]"
export OPENAI_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-...
export VOYAGE_API_KEY=pa-...
export GOOGLE_API_KEY=...

# Populate all providers at once:
python -m pykissembed.similarity.populate_embeddings

# Or populate individual providers:
pykissembed populate-embeddings --provider openai-text
pykissembed populate-embeddings --provider openai-ast
pykissembed populate-embeddings --provider codestral-text
pykissembed populate-embeddings --provider codestral-ast
pykissembed populate-embeddings --provider voyage-text
pykissembed populate-embeddings --provider voyage-ast
pykissembed populate-embeddings --provider gemini-text
pykissembed populate-embeddings --provider gemini-ast
```

### Running similarity tests

```bash
# Run all similarity tests (uses cached embeddings; skips if missing)
pytest -m similarity

# Use only cached embeddings — skip if any are missing (no API calls)
pytest -m similarity --cached-only

# Update baselines (auto-populates missing embeddings via API)
pytest -m similarity --update-baselines
```

Until the cache is populated, `pytest -m similarity` skips cleanly
(matches current mega-scrapper behaviour).

### What the cache contains

Embedding caches are stored as compressed JSON files (zlib + base64 +
float32) in `tests/baselines/`:

```
tests/baselines/
├── similarity.json                          # config + thresholds + exclusions
├── function_hashes.json                     # function hash → metadata mapping
├── openai_text_embeddings.json.zlib         # compressed embeddings
├── openai_ast_embeddings.json.zlib
├── codestral_text_embeddings.json.zlib
├── codestral_ast_embeddings.json.zlib
├── voyage_text_embeddings.json.zlib
├── voyage_ast_embeddings.json.zlib
├── gemini_text_embeddings.json.zlib
├── gemini_ast_embeddings.json.zlib
└── combined_embeddings.json.zlib            # 8-way concatenation
```

Re-running `populate-embeddings` is idempotent: functions whose content
hash hasn't changed are skipped.

### Similarity configuration

The `similarity.json` baseline file supports:

```json
{
  "config": {
    "similarity_threshold_pair": 0.86,
    "similarity_threshold_neighbor": 0.80,
    "min_loc_for_similarity": 1,
    "pca_variance_threshold": 0.99,
    "refactor_index_threshold": 12.0,
    "refactor_index_top_n": 5,
    "excluded_directories": ["tests/"],
    "excluded_file_pairs": [],
    "excluded_function_pairs": [
      ["file_a.py:func_one", "file_b.py:func_two"]
    ]
  }
}
```

- **`excluded_directories`** — skip functions in these directories
- **`excluded_file_pairs`** — pairs of filename patterns to exclude from
  pair detection (e.g. `["sheet1.py", "sheet2.py"]`)
- **`excluded_function_pairs`** — pairs of `file:function` patterns to
  exclude (e.g. `["core.py:SerializableMixin", "core.py:to_dict"]`)
- Per-provider threshold overrides: `<provider_name>_similarity_threshold_pair`
  and `<provider_name>_similarity_threshold_neighbor`
