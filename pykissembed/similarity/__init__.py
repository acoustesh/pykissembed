"""Similarity check stub.

The full similarity module is provided by ``pykissembed-local`` or
``pykissembed-cloud`` (sentence-transformers / OpenRouter-routed
``openai`` / ``gemini`` / ``qwen`` providers). Without one of those
extras installed, similarity checks are skipped with a friendly
message — use the CLI:

    pip install pykissembed-local
    pykissembed populate-embeddings --provider local
    pytest -m similarity
"""
