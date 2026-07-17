"""Compatibility shim for the retired local embedding provider.

The v0.1 compatibility release keeps this import path so callers receive an
actionable migration error. It is not registered as a provider and performs no
model loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


_CLOUD_ONLY_MESSAGE = (
    "Local embeddings were removed from pykissembed. Install 'pykissembed[cloud]' and use "
    "`pykissembed populate-embeddings --provider openai-text` "
    "(or gemini-text, voyage-text, codestral-text, qwen-text, or jina-text)."
)


class LocalProvider:
    """Deprecated shim retaining the core stub's historical cache identity.

    The standalone tombstone retains its own former BAAI identity. Keeping the
    two legacy identities distinct avoids reinterpreting either kind of local
    cache, while both shims now have the same unavailable/migration behavior.
    """

    name = "local"
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    schema_version = "1"
    max_tokens = 256
    batch_size = 32

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # ruff:ignore[unused-method-argument] — Provider protocol parameter
        raise RuntimeError(_CLOUD_ONLY_MESSAGE)

    def is_configured(self) -> bool:
        """Return ``False`` because local embeddings are unavailable.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False


__all__ = ["LocalProvider"]
