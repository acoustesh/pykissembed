# pykissembed-cloud

Cloud embedding providers for [pykissembed](https://github.com/acoustesh/pykissembed).

`openai`, `gemini`, and `qwen` are routed through
[OpenRouter](https://openrouter.ai/) using the OpenAI-compatible API, so a single
`OPENROUTER_API_KEY` enables all three. `jina` targets the native
[Jina](https://jina.ai/embeddings/) API with its own `JINA_API_KEY`.

## Install

```bash
pip install pykissembed-cloud
```

## Authentication

Either export the keys directly:

```bash
export OPENROUTER_API_KEY=sk-or-...   # openai, gemini, qwen
export JINA_API_KEY=jina_...          # jina
```

or drop them into a `.env` file at the project root (or any ancestor
directory — the loader walks up looking for one):

```bash
printf 'OPENROUTER_API_KEY=sk-or-...\nJINA_API_KEY=jina_...\n' > .env
```

The explicit environment variable always wins over the file. The
loader runs lazily on the first `is_configured()` call, so importing
`pykissembed_cloud` stays free of filesystem side effects.

## Providers

| Name | Model | Max tokens | Batch size | Key |
|---|---|---|---|---|
| `openai` | `openai/text-embedding-3-large` | 8191 | 100 | `OPENROUTER_API_KEY` |
| `gemini` | `google/gemini-embedding-001` | 2048 | 100 | `OPENROUTER_API_KEY` |
| `qwen` | `qwen/qwen3-embedding-8b` | 32000 | 32 | `OPENROUTER_API_KEY` |
| `jina` | `jina-code-embeddings-1.5b` | 512 | 32 | `JINA_API_KEY` |

`jina` emits single `code2code.passage` vectors here; its asymmetric
query/passage pairing is used only by the pykissembed similarity gate.

## Usage

```bash
pykissembed.providers list              # confirms registration
pykissembed populate-embeddings --provider gemini
pykissembed populate-embeddings --provider jina
pytest -m similarity
```

## License

MIT.
