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

## Adding similarity (embedding-based near-duplicate detection)

Recommended path: **local sentence-transformers provider** — no API key, no
network, runs in CI:

```bash
pip install pykissembed-local
pykissembed populate-embeddings --provider local
pytest -m similarity
```

For cloud providers (OpenAI / Gemini / Qwen via OpenRouter):

```bash
pip install pykissembed-cloud
# A single key enables all three providers
export OPENROUTER_API_KEY=sk-or-...
# ... or drop the key into a .env file at the project root
pykissembed populate-embeddings --provider openai
pykissembed populate-embeddings --provider gemini
pykissembed populate-embeddings --provider qwen
```

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