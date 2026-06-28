# pyqtest

Generic Python code-quality test library — a pytest plugin that bundles lint,
type, complexity, docstring, comment density, and embedding-based similarity
checks behind a single config block.

> **Default mode is baseline-and-ratchet**, not strict zero-tolerance. A
> diagnostic at or below the baseline passes; above baseline fails; a missing
> baseline also fails until you seed it. Use `pyqtest ratchet` to lower
> baselines as you fix code.

---

## Quick start (lint + type + complexity + docstrings — out of the box)

```bash
pip install pyqtest
```

Add a single `[tool.pyqtest]` block to your `pyproject.toml`:

```toml
[tool.pyqtest]
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
git add tests/baselines && git commit -m "seed pyqtest baselines"
```

---

## Adding similarity (embedding-based near-duplicate detection)

Recommended path: **local sentence-transformers provider** — no API key, no
network, runs in CI:

```bash
pip install pyqtest-local
python -m pyqtest.similarity.populate_embeddings --provider local
pytest -m similarity
```

For cloud providers (OpenAI, Voyage, Codestral via OpenRouter, Gemini):

```bash
pip install pyqtest-cloud
export OPENAI_API_KEY=...
export VOYAGE_API_KEY=...
export OPENROUTER_API_KEY=...
export GOOGLE_API_KEY=...
python -m pyqtest.similarity.populate_embeddings --provider openai
# ... repeat for voyage / codestral / gemini
```

---

## CLI

```text
pyqtest check                          # run the same gate pytest runs
pyqtest ratchet                        # lower baselines; refuse to raise
pyqtest providers list                 # show installed embedding providers
pyqtest populate-embeddings --provider NAME
pyqtest type-review --json REPORT.json # iterative type-fix helper
pyqtest init                           # (opt-in) scaffold [tool.pyqtest]
```

---

## Custom providers

Implement the `Provider` Protocol and register via the `pyqtest.providers`
entry-point group — no patching pyqtest required:

```toml
# in your library's pyproject.toml
[project.entry-points.pyqtest.providers]
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
  against `pyqtest/schemas/baselines.v1.json` on load.
- **Embedding cache keys** — `provider.name|model_id|schema_version|content_hash`.
  Mandatory `schema_version` prevents silent cache corruption.
- **Sync provider Protocol** — providers are tiny CPU/IO wrappers.
- **No telemetry.** The library runs offline.
- **Slim default install** — `pip install pyqtest` is ~10 MB. ML features
  live in `pyqtest-local` and `pyqtest-cloud` subpackages.

---

## Migration from `mega-scrapper/tests/`

pyqtest is the upstream successor of the code-quality tests in
`aa-ml/mega-scrapper/tests/`. The five attached tests
(`test_code_complexity.py`, `test_comment_density.py`,
`test_docstring_format.py`, `test_lint_typecheck.py`, `test_code_similarity.py`)
plus the `tests/similarity/` package are the v0 source. Migration preserves
baseline values verbatim (the v1 envelope wraps them without changing
semantics).

---

## License

MIT.