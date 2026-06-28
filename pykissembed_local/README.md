# pykissembed-local

Local sentence-transformers embedding provider for
[pykissembed](https://github.com/acoustesh/pykissembed). The recommended default
for similarity checks — no API key, no network, runs in CI.

## Install

```bash
pip install pykissembed-local
python -m pykissembed.similarity.populate_embeddings --provider local
pytest -m similarity
```

## Model

`BAAI/bge-small-en-v1.5` (384-dim, MIT, ~120 MB). To override, set
`PYQTEST_LOCAL_MODEL` to a different HuggingFace id before
`populate_embeddings` runs.

## License

MIT.
