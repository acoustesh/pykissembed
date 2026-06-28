# Migrating from `aa-ml/mega-scrapper/tests/` to pyqtest

This guide walks through replacing the five code-quality test files in
[aa-ml/mega-scrapper](https://github.com/acoustesh/mega-scrapper) with
the upstream `pyqtest` package.

## Steps

### 1. Add pyqtest as a dependency

In `aa-ml/mega-scrapper/pyproject.toml`:

```toml
[tool.poetry.group.dev.dependencies]
pyqtest = { path = "../pyqtest", develop = true }
```

(or pin a PyPI version once published).

### 2. Add `[tool.pyqtest]` block

```toml
[tool.pyqtest]
paths = ["src", "scripts"]
mode = "ratchet"
baseline_dir = "tests/baselines"
cache_dir = "tests/.pyqtest_cache"
```

### 3. Map mega-scrapper baseline files

The five mega-scrapper baselines become one v1 envelope each:

| Mega-scrapper file | pyqtest envelope | Kind |
|---|---|---|
| `tests/baselines/complexity_baselines.json` | `tests/baselines/complexity.json` | `complexity` |
| `tests/baselines/comment_density_baselines.json` (config section) | `tests/baselines/comment_density.json` | `density` |
| `tests/baselines/similarity_baselines.json` | `tests/baselines/similarity.json` | `similarity` |
| `tests/baselines/lint_typecheck_report.json` (live, not a baseline) | unchanged | n/a |

`pyqtest` will auto-migrate v0 (raw dict) → v1 (envelope) on first load
of each file. The data semantics are preserved; only the wrapper changes.

### 4. Delete the five upstream test files

```sh
git rm tests/test_code_complexity.py \
       tests/test_comment_density.py \
       tests/test_docstring_format.py \
       tests/test_lint_typecheck.py \
       tests/test_code_similarity.py
rm -rf tests/similarity/
```

### 5. Slim `tests/conftest.py`

The mega-scrapper `conftest.py` currently exposes fixtures that pyqtest
already provides. After deleting the test files, the only fixture left
in mega-scrapper's `conftest.py` should be project-specific (e.g.
`run_workon`). The pyqtest fixtures (`update_baselines`, `cached_only`,
`pyqtest_paths`) come from the installed `pyqtest` plugin.

### 6. Run + commit baselines

```sh
poetry install
pytest --update-baselines      # one-time seed
git add tests/baselines/ tests/.pyqtest_cache/
git commit -m "Migrate to pyqtest v0.1.0"
```

Note: `tests/.pyqtest_cache/` (embedding caches) should be **gitignored**.

### 7. Verify

```sh
pytest                       # should match pre-migration diagnostic counts
pyqtest ratchet              # should be a no-op on a clean tree
```

## What about the similarity check?

`pyqtest-cloud` ships OpenAI / Voyage / Codestral / Gemini providers.
Install with:

```sh
poetry add pyqtest-cloud --group dev
```

And populate the embedding cache:

```sh
export OPENAI_API_KEY=...
poetry run python -m pyqtest_local.similarity.populate --provider openai
```

Until you do, `pytest -m similarity` skips cleanly (matches current
mega-scrapper behaviour).