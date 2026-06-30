# pykissembed

Generic Python code-quality test library — a pytest plugin that bundles lint,
type, complexity, docstring, comment density, and embedding-based similarity
checks behind a single config block.

> **Default mode is baseline-and-ratchet**, not strict zero-tolerance. A
> diagnostic at or below the baseline passes; above baseline fails; a missing
> baseline also fails until you seed it. Use `pykissembed ratchet` to lower
> baselines as you fix code.

---

## Quick start

```bash
pip install pykissembed
```

Or with `uv`:

```bash
uv add pykissembed
```

Add a single `[tool.pykissembed]` block to your `pyproject.toml`:

```toml
[tool.pykissembed]
paths = ["src", "scripts", "tests"]   # directories to scan
mode = "ratchet"                       # default; alternative: "strict"
```

Then run:

```bash
pytest                 # runs all installed checks automatically
pytest -m lint         # lint + type-check gate
pytest -m complexity   # CC + COG + MI + line counts + docstrings
pytest -m density      # comment density
pytest -m docstring_format  # NumPy docstring format (ruff D rules)
pytest -m similarity    # embedding-based near-duplicate detection
```

Seed the baselines on a clean tree (so CI doesn't fail before you've fixed
anything):

```bash
pytest --update-baselines   # write baselines = current diagnostics
git add tests/baselines && git commit -m "seed pykissembed baselines"
```

### How check modules are collected

pykissembed's pytest plugin automatically injects the installed
`pykissembed/checks/` directory into pytest's collection paths. This means
the check modules (`code_complexity.py`, `code_similarity.py`,
`comment_density.py`, `docstring_format.py`, `lint_typecheck.py`) are
discovered and run in **any** consumer project — no need to copy test files
or configure `testpaths`. The plugin's `pytest_collect_file` hook

### Baseline-and-ratchet workflow

pykissembed uses **per-baseline ratcheting** in CI. Baselines are committed
JSON envelopes under `tests/baselines/` and the tests fail only when
diagnostics exceed them. The intended workflow:

```bash
# 1. First time on an existing codebase — seed baselines
pytest --update-baselines
git add tests/baselines && git commit -m "seed pykissembed baselines"

# 2. Day-to-day — fix code; tests only fail on REGRESSIONS
pytest              # any diagnostic above its baseline fails the build

# 3. Periodic ratchet — lower baselines as you fix issues
pykissembed ratchet  # scans tests/baselines/*.json for ratchetable data

# 4. Discover the JSON schema for a baseline type
pykissembed schema --kind complexity
```

A new file (no baseline yet) always reports as "new violation" until you
either fix the code or seed the baseline for that file. To handle large
existing codebases without breaking the build, seed everything once, then
gradually fix and ratchet downward.

### Excluding Jupyter notebooks

By default, pykissembed does **not** run ruff or similarity checks against
`.ipynb` files (notebooks typically contain exploratory code with relaxed
hygiene). To opt in:

```toml
[tool.pykissembed]
include_notebooks = true   # apply ruff + similarity to .ipynb too
```

The `[tool.pykissembed]` config reference:

| Key | Default | Purpose |
|---|---|---|
| `paths` | `["src"]` | Source directories to scan |
| `mode` | `"ratchet"` | `"ratchet"` (per-baseline) or `"strict"` (zero tolerance) |
| `baseline_dir` | `"tests/baselines"` | Where committed JSON envelopes live |
| `cache_dir` | `"tests/.pykissembed_cache"` | Where embedding caches live (gitignored) |
| `include_notebooks` | `false` | If true, run ruff/similarity against `.ipynb` files |
(registered with `tryfirst=True`) collects these modules before pytest's
default `python_files` filter can reject them.

---

## Complexity tests

The complexity gate (`pytest -m complexity`) enforces:

- **Docstring coverage** — all functions, methods, and classes in configured
  paths must have docstrings. Missing docstrings are counted per-directory
  and fail if they exceed the baseline.
- **File line counts** — per-file line count vs baseline.
- **Cyclomatic complexity (CC)** — per-function CC via `radon`, fails if
  above threshold (default 15) or per-function baseline.
- **Cognitive complexity (COG)** — per-function cognitive complexity via
  `complexipy`, fails if above threshold (default 15) or per-function baseline.
- **Maintainability Index (MI)** — per-file MI via `radon`, fails if below
  threshold (default 13) or per-file baseline.

Baselines are stored in `tests/baselines/complexity.json` as a versioned
envelope. Update with `pytest --update-baselines`.

---

## Similarity tests

The similarity gate (`pytest -m similarity`) detects near-duplicate functions
using embedding providers. It supports 9 providers:

| Provider | Model | Hash type | Default pair threshold |
| --- | --- | --- | --- |
| OpenAI-Text | text-embedding-3-large | text_hash | 0.86 |
| OpenAI-AST | text-embedding-3-large | ast_hash | 0.86 |
| Codestral-Text | codestral-embed-2505 | text_hash | 0.97 |
| Codestral-AST | codestral-embed-2505 | ast_hash | 0.97 |
| Voyage-Text | voyage-code-3 | text_hash | 0.95 |
| Voyage-AST | voyage-code-3 | ast_hash | 0.95 |
| Gemini-Text | gemini-embedding-001 | text_hash | 0.90 |
| Gemini-AST | gemini-embedding-001 | ast_hash | 0.90 |
| Combined | 8-way concatenation | text_hash | 0.88 |

Each provider detects:
- **Pairwise similarity** — finds copy-paste code (similarity ≥ pair threshold)
- **Neighbor clustering** — finds functions with ≥2 similar counterparts
- **Refactor index** — combines complexity + similarity for priority ranking

### Prerequisites

Similarity requires embedding caches. Install an extra and populate the cache:

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
pykissembed populate-embeddings --provider openai-text
pykissembed populate-embeddings --provider openai-ast
# ... or populate all at once:
python -m pykissembed.similarity.populate_embeddings
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

### Similarity infrastructure

The `pykissembed.similarity` sub-package provides:

| Module | Purpose |
| --- | --- |
| `types.py` | `FunctionInfo` dataclass, `PCAModel` protocol |
| `constants.py` | Config-driven path resolution |
| `storage.py` | `EmbeddingRegistry`, compressed embedding cache, baselines I/O |
| `ast_helpers.py` | Function extraction from Python AST |
| `complexity.py` | CC/COG complexity map loaders |
| `embeddings.py` | API clients (OpenAI, Codestral, Voyage, Gemini) with retry |
| `pca.py` | PCA dimensionality reduction (GPU/cuML with CPU/sklearn fallback) |
| `refactor_index.py` | Refactor index: `0.25*CC + 0.15*COG + 0.6*similarity_index` |
| `file_split.py` | File split proposal via PCA + k-means clustering |
| `checks.py` | Unified similarity check workflow |
| `populate_embeddings.py` | CLI to fetch missing embeddings from APIs |

---

## Testing

This repo is configured to work out of the box with the **VS Code Testing
panel**. Open the workspace, install the recommended extensions
(`ms-python.pytest`, `ms-python.python`, `charliermarsh.ruff`,
`tamasfe.even-better-toml`), and the test tree appears in the left sidebar.

### Test discovery

| Package | Command | Count |
| --- | --- | --- |
| `pykissembed` (core) | `uv run pytest -m "not live"` | 66 |
| `pykissembed_cloud` | `uv run pytest -m "not live"` | 27 |
| `pykissembed_local` | `uv run pytest -m "not live"` | 19 |

The panel respects the `[tool.pytest.ini_options]` block in each
`pyproject.toml`. **Live tests (network calls, model downloads) are
skipped by default.** To opt in:

```bash
# Run everything live (one cloud provider smoke + model load)
uv run pytest -m "live and smoke"

# Run only the model-load live test
cd pykissembed_local && uv run pytest -m live
```

### Running / debugging a single test

Right-click any test in the Testing panel → **Run Test** or **Debug
Test**. The debug configurations in `.vscode/launch.json` set the right
`cwd` per package automatically.

### Markers

| Marker | Meaning |
| --- | --- |
| `live` | Network calls or model downloads — skipped by default. |
| `smoke` | Fast subset of `live`; suitable for CI fast-gate (`-m "live and smoke"`). |
| `lint`, `complexity`, `density`, `docstring_format`, `similarity`, `experimental` | Project-specific gates provided by the pykissembed pytest plugin. |

---

## Install extras

The core package ships lint, type-check, complexity, docstring, comment
density, and the full similarity infrastructure (PCA, refactor index, file
split). Embedding providers are opt-in via extras:

| Extras | Command | What you get |
| --- | --- | --- |
| *(none)* | `uv add pykissembed` | Lint, type, complexity, docstring, density, similarity (skips without providers) |
| `local` | `uv add "pykissembed[local]"` | Adds the `local` sentence-transformers provider |
| `cloud` | `uv add "pykissembed[cloud]"` | Adds `openai`, `gemini`, `qwen` providers via OpenRouter |
| `all` | `uv add "pykissembed[all]"` | Both `local` and `cloud` at once |

### Installing from TestPyPI (current release channel)

Until `pykissembed` reaches production PyPI, add TestPyPI as an explicit
index in your consumer project's `pyproject.toml`:

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
uv lock --upgrade-package pykissembed --upgrade-package pykissembed-cloud --upgrade-package pykissembed-local
uv sync
uv run pykissembed --version    # should show 0.1.4
```

> **Important:** If `uv lock --upgrade-package` doesn't pick up the new
> version, delete `uv.lock` and run `uv lock` fresh. This forces a full
> re-resolution from TestPyPI.

After installing, populate the cache and run the suite:

```bash
# Local — no API key, runs offline once the model is downloaded
pykissembed populate-embeddings --provider local
pytest -m similarity

# Cloud — one key enables all three providers
export OPENROUTER_API_KEY=sk-or-...
# ... or drop the key into a .env file at the project root
pykissembed populate-embeddings --provider openai
pykissembed populate-embeddings --provider gemini
pykissembed populate-embeddings --provider qwen
pytest -m similarity
```

> **Why extras?** `pykissembed-local` pulls `sentence-transformers` (~hundreds
> of MB) and `pykissembed-cloud` pulls the `openai` client. Keeping them
> optional means slim projects don't pay for similarity they don't use.

---

## CLI

```text
pykissembed check                          # run the same gate pytest runs
pykissembed ratchet                        # lower baselines; refuse to raise
pykissembed.providers list                 # show installed embedding providers
pykissembed populate-embeddings --provider NAME
pykissembed type-review --json REPORT.json # iterative type-fix helper
pykissembed init                           # (opt-in) scaffold [tool.pykissembed]
```

---

## Custom providers

Implement the `Provider` Protocol and register via the `pykissembed.providers`
entry-point group — no patching pykissembed required:

```toml
# in your library's pyproject.toml
[project.entry-points.pykissembed.providers]
my_embedder = "my_pkg.providers:MyProvider"
```

```python
# my_pkg/providers.py
from collections.abc import Sequence

class MyProvider:
    name = "my_embedder"
    model_id = "my-model-v1"
    schema_version = "1"
    max_tokens = 512
    batch_size = 32

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def is_configured(self) -> bool:
        return True
```

---

## Design notes

- **Baseline-and-ratchet default** — strict zero-tolerance can't run "out of
  the box" on any real codebase without forcing a "fix every diagnostic" PR.
- **Versioned baseline envelope** — every baseline JSON file is wrapped in
  `{"schema_version": "1.0", "kind": "...", "data": {...}}` and validated
  against `pykissembed/schemas/baselines.v1.json` on load.
- **Embedding cache keys** — `provider.name|model_id|schema_version|content_hash`.
  Mandatory `schema_version` prevents silent cache corruption.
- **Sync provider Protocol** — providers are tiny CPU/IO wrappers.
- **No telemetry.** The library runs offline.
- **Slim default install** — `pip install pykissembed` is ~10 MB. ML features
  live in `pykissembed-local` and `pykissembed-cloud` subpackages.

---

## Migration from `mega-scrapper/tests/`

pykissembed is the upstream successor of the code-quality tests in
`aa-ml/mega-scrapper/tests/`. The five attached tests
(`test_code_complexity.py`, `test_comment_density.py`,
`test_docstring_format.py`, `test_lint_typecheck.py`, `test_code_similarity.py`)
plus the `tests/similarity/` package are the v0 source. Migration preserves
baseline values verbatim (the v1 envelope wraps them without changing
semantics).

See [docs/MIGRATING_FROM_MEGA_SCRAPPER.md](docs/MIGRATING_FROM_MEGA_SCRAPPER.md)
for the step-by-step migration guide.

---

## License

MIT.