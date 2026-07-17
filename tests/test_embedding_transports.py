"""Focused tests for cloud embedding transport normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import requests

from pykissembed.similarity import embeddings

if TYPE_CHECKING:
    import pytest


class _Response:
    """Minimal successful OpenRouter response double."""

    @staticmethod
    def raise_for_status() -> None:
        """Model a successful HTTP status."""

    @staticmethod
    def json() -> dict[str, object]:
        """Return a vector containing integral and floating components.

        Returns
        -------
        dict[str, object]
            OpenRouter-style embedding response.
        """
        return {"data": [{"embedding": [0, 1.25, -2]}]}


def test_openrouter_normalizes_all_vector_components_to_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integral JSON components remain valid in the live in-memory cache."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: _Response())

    result = embeddings.get_embeddings_batch(
        ["synthetic input"],
        provider="qwen",
        max_retries=1,
    )

    assert result == [[0.0, 1.25, -2.0]]
    assert embeddings.is_embedding_cache({"hash": result[0]})
