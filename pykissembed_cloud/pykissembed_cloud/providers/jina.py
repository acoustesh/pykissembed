"""Jina code embeddings via the native Jina API.

``jina-code-embeddings-1.5b`` on Jina's own OpenAI-compatible endpoint
(``https://api.jina.ai/v1/``), authenticated with ``JINA_API_KEY``.

Note: Jina's ``code2code`` / ``nl2code`` tasks are *asymmetric* — the meaningful
similarity is a query-vs-passage cross-score, not a single-vector cosine. The
plugin ``Provider`` contract returns one vector per text, so this provider emits
single ``code2code.passage`` vectors only. The full symmetrized query/passage
pairing lives in the pykissembed similarity gate, not here.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pykissembed_cloud.providers._openai_compat import OpenAICompatProvider


class JinaProvider(OpenAICompatProvider):
    """``jina-code-embeddings-1.5b`` on the native Jina API (passage task)."""

    name = "jina"
    model_id = "jina-code-embeddings-1.5b"
    schema_version = "1"
    # The endpoint rejects long inputs ("Failed to encode text") and its own
    # truncate=True does not help, and large batches intermittently 400 — so keep
    # both conservative (verified reliable across a full code corpus).
    max_tokens = 512
    batch_size = 32
    base_url: ClassVar[str] = "https://api.jina.ai/v1/"
    api_key_env: ClassVar[str] = "JINA_API_KEY"
    api_key_url: ClassVar[str] = "https://jina.ai/embeddings/"
    # Single-vector proxy for the asymmetric scheme (see module docstring).
    extra_body: ClassVar[dict[str, Any] | None] = {"task": "code2code.passage", "truncate": False}


__all__ = ["JinaProvider"]
