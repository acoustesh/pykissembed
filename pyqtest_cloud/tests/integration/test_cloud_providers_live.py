"""Live (network) tests for the OpenRouter-routed providers.

Skipped by default. Enable with:

    OPENROUTER_API_KEY=sk-or-... uv run pytest -m live

Each test makes a single minimal API call to confirm the model id,
endpoint, and response shape are wired up correctly. There is no
assertion on vector values — that would be flaky across model versions.
"""

from __future__ import annotations

import os

import pytest

from pyqtest_cloud.providers.gemini import GeminiProvider
from pyqtest_cloud.providers.openai import OpenAIProvider
from pyqtest_cloud.providers.qwen import QwenProvider

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        "OPENROUTER_API_KEY" not in os.environ,
        reason="OPENROUTER_API_KEY not set",
    ),
]


@pytest.mark.parametrize(
    "provider",
    [OpenAIProvider(), GeminiProvider(), QwenProvider()],
)
def test_embed_returns_one_vector_per_input(provider: object) -> None:
    """Each provider returns one vector per input text."""
    assert provider.is_configured()  # type: ignore[attr-defined]
    vectors = provider.embed(["hello world", "goodbye world"])  # type: ignore[attr-defined]
    assert len(vectors) == 2
    for vec in vectors:
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(x, float) for x in vec)
