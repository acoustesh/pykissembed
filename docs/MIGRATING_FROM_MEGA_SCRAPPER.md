# Migrating from `aa-ml/mega-scrapper/tests/` to pykissembed

This guide walks through replacing the five code-quality test files in
[aa-ml/mega-scrapper](https://github.com/acoustesh/mega-scrapper) with
the upstream `pykissembed` package.

## Steps

### 1. Add pykissembed as a dependency

In `aa-ml/mega-scrapper/pyproject.toml`:

```toml
[tool.poetry.group.dev.dependencies]
pykissembed = { path = "../pykissembed", develop = true }
```

(or pin a PyPI version once published — `pip install pykissembed`).

### 2. Add `[tool.pykissembed]` block

```toml
[tool.pykissembed]
paths = ["src", "scripts"]
mode = "ratchet"
baseline_dir = "tests/baselines"
cache_dir = "tests/.pykissembed_cache"
```

### 3. Map mega-scrapper baseline files

The five mega-scrapper baselines become one v1 envelope each:

| Mega-scrapper file | pykissembed envelope | Kind |
|---|---|---|
| `tests/baselines/complexity_baselines.json` | `tests/baselines/complexity.json` | `complexity` |
| `tests/baselines/comment_density_baselines.json` (config section) | `tests/baselines/comment_density.json` | `density` |
| `tests/baselines/similarity_baselines.json` | `tests/baselines/similarity.json` | `similarity` |
| `tests/baselines/lint_typecheck_report.json` (live, not a baseline) | unchanged | n/a |

`pykissembed` will auto-migrate v0 (raw dict) → v1 (envelope) on first load
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

The mega-scrapper `conftest.py` currently exposes fixtures that pykissembed
already provides. After deleting the test files, the only fixture left
in mega-scrapper's `conftest.py` should be project-specific (e.g.
`run_workon`). The pykissembed fixtures (`update_baselines`, `cached_only`,
`pykissembed_paths`) come from the installed `pykissembed` plugin.

### 6. Run + commit baselines

```sh
poetry install
pytest --update-baselines      # one-time seed
git add tests/baselines/ tests/.pykissembed_cache/
git commit -m "Migrate to pykissembed v0.1.0"
```

Note: `tests/.pykissembed_cache/` (embedding caches) should be **gitignored**.

### 7. Verify

```sh
pytest                       # should match pre-migration diagnostic counts
pykissembed ratchet              # should be a no-op on a clean tree
```

> **About `pykissembed ratchet`:** in `pykissembed` v0.1.0, the ratchet command
> only computes current diagnostics for `lint_typecheck.json`. The
> other three baseline files (`complexity.json`, `comment_density.json`,
> `docstring_format.json`) are reported as `skip …: no
> current-diagnostics computer implemented`. The skip is harmless — the
> ratchet is a no-op for those files because the gate's
> `--update-baselines` run already wrote the same values. The
> "0 baseline file(s) lowered" summary confirms it. Additional
> current-diagnostics computers are on the roadmap.

## What about the similarity check?

`pykissembed-cloud` ships three OpenRouter-routed providers (`openai`,
`gemini`, `qwen`); `pykissembed-local` ships the local sentence-transformers
provider. All four are routed through a single
[OpenRouter](https://openrouter.ai/) base URL.

### Recommended default — local (no API key)

```sh
pip install pykissembed-local
pykissembed populate-embeddings --provider local
pytest -m similarity
```

The first run downloads the `BAAI/bge-small-en-v1.5` weights (~120 MB).
Subsequent runs are no-ops (the cache is content-addressed via SHA-256
keys).

### Cloud providers (OpenRouter)

```sh
pip install pykissembed-cloud
# Put the key in a .env file (the cloud package loads it lazily on
# first is_configured() call), or:
export OPENROUTER_API_KEY=sk-or-...

pykissembed.providers list                 # confirms all four registered
pykissembed populate-embeddings --provider openai
pykissembed populate-embeddings --provider gemini
pykissembed populate-embeddings --provider qwen
pytest -m similarity
```

Until the cache is populated, `pytest -m similarity` skips cleanly
(matches current mega-scrapper behaviour).

### What the cache contains

`tests/.pykissembed_cache/<provider>-<model>.jsonl` — one JSON line per
source file, each line `{"key": "...", "path": "...", "vector": [...]}`.
Re-running `populate-embeddings` is idempotent: files whose
content hash hasn't changed are skipped.
