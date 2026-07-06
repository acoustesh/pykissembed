"""Live (network) tests for the OpenRouter-routed providers.

Skipped by default. Enable with either of:

    OPENROUTER_API_KEY=sk-or-... uv run pytest -m live
    echo "OPENROUTER_API_KEY=sk-or-..." > .env && uv run pytest -m live

The ``.env`` walk-up loader populates ``$OPENROUTER_API_KEY`` from the
file before the skip gate runs, so either form works.

Each test makes a single minimal API call to confirm the model id,
endpoint, and response shape are wired up correctly. There is no
assertion on vector values — that would be flaky across model versions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pykissembed_cloud.dotenv import ensure_loaded
from pykissembed_cloud.providers.gemini import GeminiProvider
from pykissembed_cloud.providers.openai import OpenAIProvider
from pykissembed_cloud.providers.qwen import QwenProvider

if TYPE_CHECKING:
    from pykissembed_cloud.providers._openai_compat import OpenAICompatProvider


def _has_openrouter_key() -> bool:
    """Return True iff ``$OPENROUTER_API_KEY`` is set after a .env lookup.

    Returns
    -------
    bool
        ``True`` once the dotenv loader has had a chance to populate
        the environment and the key is present.
    """
    import os

    ensure_loaded()
    return "OPENROUTER_API_KEY" in os.environ


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _has_openrouter_key(),
        reason="OPENROUTER_API_KEY not set (env or .env)",
    ),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "provider",
    [OpenAIProvider(), GeminiProvider(), QwenProvider()],
)
def test_embed_returns_one_vector_per_input(provider: OpenAICompatProvider) -> None:
    """Each provider returns one vector per input text.

    Tagged ``smoke`` (and ``live`` via the module-level ``pytestmark``)
    so CI can run a fast subset with ``pytest -m "live and smoke"``.
    """
    assert provider.is_configured()
    vectors = provider.embed(["hello world", "goodbye world"])
    assert len(vectors) == 2
    for vec in vectors:
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(x, float) for x in vec)
