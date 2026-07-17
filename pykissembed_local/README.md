# pykissembed-local

`pykissembed-local` is the final lightweight compatibility package for the
retired local embedding provider. It contains no runtime dependencies, does not
register a `pykissembed.providers` entry point, and never loads or downloads a
model.

The package remains importable during the v0.1 migration window. Existing code
can still construct `LocalProvider(model_id=None)`, inspect its legacy identity
attributes, restore serialized identity state, and use the pure hashing/path
helpers from `pykissembed_local.runner`.

Embedding operations are intentionally unavailable:

- `LocalProvider.is_configured()` always returns `False`.
- `LocalProvider.embed(...)`, `runner.populate(...)`, and
  `runner.truncate_to_tokens(...)` raise an actionable `RuntimeError`.
- Existing local cache files and downloaded model data are left untouched.

## Migrate to a cloud provider

Install the cloud extra and explicitly populate one of the canonical provider
caches:

```bash
pip install "pykissembed[cloud]"
pykissembed populate-embeddings --provider openai-text
```

Other canonical choices include `gemini-text`, `voyage-text`,
`codestral-text`, `qwen-text`, and `jina-text`, plus their `-ast` variants.
Consult the main pykissembed documentation for credentials, source-egress
details, cache-only operation, provider costs, and retention policies.

The historical sentence-transformers, Hugging Face, NumPy, pandas, tiktoken,
and Torch dependencies are not installed by this compatibility package.

Historical `tests/.pykissembed_cache/local-*.jsonl` files are ignored and are
never removed automatically. After verifying that you no longer need them,
you may clean them up manually with:

```bash
rm -i tests/.pykissembed_cache/local-*.jsonl
```

## License

MIT.
