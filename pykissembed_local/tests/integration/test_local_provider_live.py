"""Live test: actually load the sentence-transformers model and embed a few inputs.

Skipped unless the ``live`` marker is explicitly requested:

    uv run pytest -m live

This will download the ``BAAI/bge-small-en-v1.5`` weights (~120 MB) on
first run, so it stays out of the default CI path.
"""

from __future__ import annotations

import pytest

from pykissembed_local.provider import LocalProvider

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LocalProvider().is_configured(),
        reason="sentence-transformers not importable",
    ),
]


def test_real_model_loads_and_embeds() -> None:
    """End-to-end: load the model and embed two short inputs."""
    provider = LocalProvider()
    vectors = provider.embed(["hello world", "goodbye world"])
    assert len(vectors) == 2
    for vec in vectors:
        # bge-small-en-v1.5 returns 384-dim vectors
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)
