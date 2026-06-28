# pyqtest-cloud

Cloud embedding providers for [pyqtest](https://github.com/acoustesh/pyqtest).

All three bundled providers (`openai`, `gemini`, `qwen`) are routed through
[OpenRouter](https://openrouter.ai/) using the OpenAI-compatible API. A single
`OPENROUTER_API_KEY` is enough to enable any of them.

## Install

```bash
pip install pyqtest-cloud
```

## Authentication

Either export the key directly:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

or drop it into a `.env` file at the project root (or any ancestor
directory — the loader walks up looking for one):

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
```

The explicit environment variable always wins over the file. The
loader runs lazily on the first `is_configured()` call, so importing
`pyqtest_cloud` stays free of filesystem side effects.

## Providers

| Name | Model (OpenRouter id) | Max tokens | Batch size |
|---|---|---|---|
| `openai` | `openai/text-embedding-3-large` | 8191 | 100 |
| `gemini` | `google/gemini-embedding-001` | 2048 | 100 |
| `qwen` | `qwen/qwen3-embedding-8b` | 32000 | 32 |

## Usage

```bash
pyqtest providers list              # confirms registration
pyqtest populate-embeddings --provider gemini
pyqtest populate-embeddings --provider qwen
pytest -m similarity
```

## License

MIT.
