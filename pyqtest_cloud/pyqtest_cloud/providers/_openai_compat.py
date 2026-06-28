"""Shared base class for the OpenRouter-routed cloud providers.

Every bundled provider (``openai``, ``gemini``, ``qwen``) is just a thin
``OpenAICompatProvider`` subclass that sets the right identity attributes
and the OpenRouter model id. Everything else — client construction,
batching, response parsing, ``is_configured`` — lives here.

A single ``OPENROUTER_API_KEY`` environment variable enables all three
providers, since they share the same OpenRouter base URL.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

# OpenRouter exposes an OpenAI-compatible API at this base URL.
# See https://openrouter.ai/docs for the upstream contract.
_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/"
_API_KEY_ENV: str = "OPENROUTER_API_KEY"


class OpenAICompatProvider:
    """Base class for OpenRouter-routed OpenAI-compatible providers.

    Subclasses must set ``name``, ``model_id``, ``max_tokens``,
    ``batch_size``, and ``schema_version`` as class attributes.

    The provider is **sync** by design — the underlying openai SDK is
    thread-safe and the per-batch payload is bounded by ``batch_size``,
    so blocking the caller is fine.

    Attributes
    ----------
    name
        Stable identifier (e.g. ``"openai"``).
    model_id
        OpenRouter model id (e.g. ``"openai/text-embedding-3-large"``).
    schema_version
        Bumped whenever the vector shape or semantics change. Used as
        part of the embedding cache key to prevent silent corruption.
    max_tokens
        Maximum tokens the provider accepts per input. pyqtest will
        truncate inputs beyond this limit.
    batch_size
        Maximum number of texts per ``embed`` API call.
    """

    name: str
    model_id: str
    schema_version: str
    max_tokens: int
    batch_size: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute embedding vectors for *texts*.

        Inputs are split into ``batch_size`` chunks and each chunk is
        sent to the provider. The list of returned vectors is then
        re-assembled in the original input order.

        Returns
        -------
        list[list[float]]
            One vector per input text. Length of the outer list is
            guaranteed to equal ``len(texts)``.

        Raises
        ------
        RuntimeError
            If the API key is not configured.
        """
        if not self.is_configured():
            msg = (
                f"Provider {self.name!r} requires {_API_KEY_ENV!r} in the environment. "
                "Get one at https://openrouter.ai/keys"
            )
            raise RuntimeError(msg)

        # Lazy import — keep pyqtest-cloud importable even without openai
        # installed (e.g. when only reading metadata).
        from openai import OpenAI

        client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=os.environ[_API_KEY_ENV])
        inputs = list(texts)
        if not inputs:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            chunk = inputs[start : start + self.batch_size]
            response = client.embeddings.create(input=chunk, model=self.model_id)
            vectors.extend(_extract_embeddings(response))
        return vectors

    def is_configured(self) -> bool:
        """Return True iff the OpenRouter API key is present in the environment.

        Lazily walks up from cwd looking for a ``.env`` file on the
        first call, so users can drop their key into ``.env`` instead
        of exporting it. The explicit ``$OPENROUTER_API_KEY`` env var
        always wins over the file.

        Returns
        -------
        bool
            ``True`` if ``OPENROUTER_API_KEY`` is set in the environment
            (after the optional ``.env`` lookup).
        """
        # Lazy import: avoid the filesystem touch on simple imports
        from pyqtest_cloud.dotenv import ensure_loaded

        ensure_loaded((_API_KEY_ENV,))
        return bool(os.environ.get(_API_KEY_ENV))


def _extract_embeddings(response: Any) -> list[list[float]]:
    """Pull the embedding vectors out of an OpenAI ``CreateEmbeddingResponse``.

    The response keeps the order of the input batch in the ``data`` list.
    Each entry exposes the vector as ``item.embedding`` (a list of floats).

    Returns
    -------
    list[list[float]]
        One vector per response item, in the same order as the input batch.
        Returns an empty list if the response has no ``data`` attribute or
        if ``data`` is not a list.
    """
    data = getattr(response, "data", None)
    if not isinstance(data, list):
        return []
    out: list[list[float]] = []
    for item in data:
        vec = getattr(item, "embedding", None)
        if isinstance(vec, list):
            out.append([float(x) for x in vec])
    return out


__all__ = ["OpenAICompatProvider"]
