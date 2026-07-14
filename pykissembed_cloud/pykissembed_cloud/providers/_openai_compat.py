"""Shared base class for the OpenAI-compatible cloud providers.

Most bundled providers (``openai``, ``gemini``, ``qwen``) are thin
``OpenAICompatProvider`` subclasses that only set identity attributes and the
OpenRouter model id; a single ``OPENROUTER_API_KEY`` enables all three. A
provider on a different OpenAI-compatible endpoint (``jina``) additionally
overrides ``base_url``, ``api_key_env``, ``api_key_url``, and ``extra_body``.
Everything else — client construction, batching, response parsing,
``is_configured`` — lives here.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence

# OpenRouter exposes an OpenAI-compatible API at this base URL.
# See https://openrouter.ai/docs for the upstream contract.
_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/"
_API_KEY_ENV: str = "OPENROUTER_API_KEY"
_OPENROUTER_KEY_URL: str = "https://openrouter.ai/keys"

# Every provider's ``is_configured`` loads this whole set from ``.env`` on the
# first call. The dotenv loader caches per working-directory, not per key, so
# loading only one provider's key would mask the others when several providers
# (with different keys, e.g. OpenRouter vs Jina) are queried in one process.
_KEY_ENVS: tuple[str, ...] = ("OPENROUTER_API_KEY", "JINA_API_KEY")


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
        Maximum tokens the provider accepts per input. pykissembed will
        truncate inputs beyond this limit.
    batch_size
        Maximum number of texts per ``embed`` API call.
    base_url
        OpenAI-compatible API base URL (defaults to OpenRouter).
    api_key_env
        Environment variable holding this provider's API key (defaults to
        ``OPENROUTER_API_KEY``).
    api_key_url
        Page where a user can obtain the key, shown in the error message.
    extra_body
        Extra JSON fields merged into each ``embeddings.create`` request (e.g.
        Jina's ``{"task": ..., "truncate": False}``). ``None`` sends none.
    """

    name: str
    model_id: str
    schema_version: str
    max_tokens: int
    batch_size: int
    # Endpoint/auth config — class-level (not part of the Provider protocol).
    base_url: ClassVar[str] = _OPENROUTER_BASE_URL
    api_key_env: ClassVar[str] = _API_KEY_ENV
    api_key_url: ClassVar[str] = _OPENROUTER_KEY_URL
    extra_body: ClassVar[dict[str, Any] | None] = None

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
                f"Provider {self.name!r} requires {self.api_key_env!r} in the environment. "
                f"Get one at {self.api_key_url}"
            )
            raise RuntimeError(msg)

        # Lazy import — keep pykissembed-cloud importable even without openai
        # installed (e.g. when only reading metadata).
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=os.environ[self.api_key_env])
        inputs = list(texts)
        if not inputs:
            return []
        # Only forward extra_body when set so providers without extra request
        # fields keep the exact create() signature the OpenAI SDK expects.
        extra = {"extra_body": self.extra_body} if self.extra_body else {}
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            chunk = inputs[start : start + self.batch_size]
            response = client.embeddings.create(input=chunk, model=self.model_id, **extra)
            vectors.extend(_extract_embeddings(response))
        return vectors

    def is_configured(self) -> bool:
        """Return True iff this provider's API key is present in the environment.

        Lazily walks up from cwd looking for a ``.env`` file on the
        first call, so users can drop their key into ``.env`` instead
        of exporting it. The explicit env var always wins over the file.
        The full :data:`_KEY_ENVS` set is loaded (not just this provider's
        key) so a mixed set of providers all see their keys after one walk.

        Returns
        -------
        bool
            ``True`` if ``self.api_key_env`` is set in the environment
            (after the optional ``.env`` lookup).
        """
        # Lazy import: avoid the filesystem touch on simple imports
        from pykissembed_cloud.dotenv import ensure_loaded

        ensure_loaded(_KEY_ENVS)
        return bool(os.environ.get(self.api_key_env))


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
