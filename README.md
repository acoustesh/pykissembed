# pykissembed

Generic Python code-quality test library — a pytest plugin that bundles lint,
type, complexity, docstring, comment density, and embedding-based similarity
checks behind a single config block.

> **Default mode is baseline-and-ratchet**, not strict zero-tolerance. A
> diagnostic at or below the baseline passes; above baseline fails; a missing
> baseline also fails until you seed it. Use `pykissembed ratchet` to lower
> baselines as you fix code.

---

## Quick start (lint + type + complexity + docstrings — out of the box)

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
pytest                 # runs all installed checks
pytest -m lint         # lint + type-check gate
pytest -m complexity   # CC + COG + MI + line counts + docstrings
```

Seed the baselines on a clean tree (so CI doesn't fail before you've fixed
anything):

```bash
pytest --update-baselines   # write baselines = current diagnostics
git add tests/baselines && git commit -m "seed pykissembed baselines"
```

---

## Testing

This repo is configured to work out of the box with the **VS Code Testing
panel**. Open the workspace, install the recommended extensions
(`ms-python.pytest`, `ms-python.python`, `charliermarsh.ruff`,
`tamasfe.even-better-toml`), and the test tree appears in the left sidebar.

### Test discovery

| Package | Command | Count |
| --- | --- | --- |
| `pykissembed` (core) | `uv run pytest -m "not live"` | 39 |
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

The core package ships lint, type-check, complexity, docstring, and comment
density gates. Embedding-based similarity is opt-in via extras:

| Extras | Command | What you get |
| --- | --- | --- |
| *(none)* | `uv add pykissembed` | Lint, type-check, complexity, docstring, density |
| `local` | `uv add "pykissembed[local]"` | Adds the `local` sentence-transformers provider |
| `cloud` | `uv add "pykissembed[cloud]"` | Adds `openai`, `gemini`, `qwen` providers via OpenRouter |
| `all` | `uv add "pykissembed[all]"` | Both `local` and `cloud` at once |

### Installing from TestPyPI (current release channel)

Until `pykissembed` reaches production PyPI, point `uv` at TestPyPI and pin
the platform. Add the index to your consumer project's `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple/"
explicit = true

# Pin resolution to the platforms you actually ship. pykissembed is
# published as `py3-none-any` wheels, but locking the resolver prevents
# `uv` from probing for versions in marker combinations you don't use.
[tool.uv]
environments = ["sys_platform == 'linux'"]
```

Then add the package (the leading `./` disambiguates the index *name* from a relative *path*):

```bash
uv add "pykissembed[all]==0.1.1" --index ./testpypi
```

> **Why `explicit = true`?** It stops TestPyPI from shadowing production
> PyPI for unrelated packages. With `explicit = true`, only the deps you
> tag with `--index ./testpypi` resolve from TestPyPI; everything else
> still uses pypi.org.
>
> **Why the leading `./`?** `uv` accepts either an index URL or a relative
> path under `--index`. The leading `./` forces name-mode, telling `uv`
> to look up `testpypi` in the `[[tool.uv.index]]` table. Without it, `uv`
> warns and (in a future release) will hard-fail, treating the value as a
> path.
>
> **Why pin `environments`?** Without it, `uv` tries to resolve your
> `uv.lock` for every `(python_version, sys_platform)` pair it can derive
> — including combinations like `python_full_version >= '3.14' and
> sys_platform == 'win32'` that no one on the team uses. Limiting the
> resolver to your real targets makes the lock fast and removes the
> *"No solution found when resolving dependencies for split"* error.

When production PyPI is set up, drop the `[tool.uv.index]` block and the
`[tool.uv]` `environments` line, and use the plain `uv add "pykissembed[all]"`
form from the table above.

After installing an extra, populate the cache and run the suite:

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