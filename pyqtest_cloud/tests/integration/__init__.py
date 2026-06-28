"""Live integration tests for the cloud providers.

These tests are skipped by default. To run them:

    OPENROUTER_API_KEY=sk-or-... uv run pytest -m live

The ``live`` marker is registered in ``pyproject.toml`` so collection
fails loudly if the marker is undeclared (``--strict-markers``).
"""
