"""Live tests that actually load a sentence-transformers model.

Skipped by default. Enable with:

    uv run pytest -m live

These tests download the BAAI/bge-small-en-v1.5 weights (~120 MB) on
first run. They're gated to keep CI fast and hermetic.
"""
