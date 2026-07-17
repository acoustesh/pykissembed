# pykissembed-cloud

Cloud embedding providers for [pykissembed](https://github.com/acoustesh/pykissembed).

This distribution serves two stable surfaces:

- The public `pykissembed.providers` entry points retain their existing
  routing: `openai`, `gemini`, and `qwen` use
  [OpenRouter](https://openrouter.ai/), while `jina` uses native Jina.
- The similarity cache command uses canonical `-text`/`-ast` variants with
  distinct routes: native OpenAI, native Google, OpenRouter Codestral/Qwen,
  native Voyage, and native Jina.

## Install

```bash
pip install pykissembed-cloud
```

## Authentication

Export only the credentials for the routes you intend to use:

```bash
export OPENAI_API_KEY=sk-...          # canonical openai-* similarity caches
export GOOGLE_API_KEY=...             # canonical gemini-* similarity caches
export OPENROUTER_API_KEY=sk-or-...   # entry points; canonical codestral-* and qwen-*
export VOYAGE_API_KEY=pa-...          # canonical voyage-* similarity caches
export JINA_API_KEY=jina_...          # entry point and canonical jina-* caches
```

or drop them into a `.env` file at the project root (or any ancestor
directory — the loader walks up looking for one).

The explicit environment variable always wins over the file. The
loader runs lazily on the first `is_configured()` call, so importing
`pykissembed_cloud` stays free of filesystem side effects.

## Entry-point providers

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
pykissembed providers list              # confirms registration
pykissembed populate-embeddings --provider gemini-text
pykissembed populate-embeddings --provider jina-ast
pytest -m similarity
```

The population command uses canonical text/AST similarity variants rather
than the short entry-point names. Text variants transmit per-function
decorators, signatures, docstrings, and comments. AST variants transmit the
normalized full implementation (including bodies, literals, decorators, and
docstrings); Jina-Text also sends that implementation as its passage. Calls may
incur API charges, so review the selected service's retention policy before
opting in. Normal pytest runs are cache-only unless cloud population is
explicitly enabled.

## License

MIT.
