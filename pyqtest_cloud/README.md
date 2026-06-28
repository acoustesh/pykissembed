# pyqtest-cloud

Cloud embedding providers for [pyqtest](https://github.com/acoustesh/pyqtest).

All three bundled providers (`openai`, `gemini`, `qwen`) are routed through
[OpenRouter](https://openrouter.ai/) using the OpenAI-compatible API. A single
`OPENROUTER_API_KEY` is enough to enable any of them.

## Install

```bash
pip install pyqtest-cloud
export OPENROUTER_API_KEY=sk-or-...
```

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
