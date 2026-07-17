"""Compatibility-only local provider for the cloud-only migration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Never as SentenceTransformer

# Retained so old imports and serialized provider state remain readable.
DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"

_CLOUD_ONLY_MESSAGE = (
    "Local embeddings were removed from pykissembed. Install 'pykissembed[cloud]' and use "
    "`pykissembed populate-embeddings --provider openai-text` "
    "(or gemini-text, voyage-text, codestral-text, qwen-text, or jina-text)."
)


class LocalProvider:
    """Compatibility tombstone for the retired local embedding provider.

    The identity attributes and constructor are retained for imports, type
    checks, and serialized state created by older pykissembed releases. The
    provider is never configured and cannot generate embeddings.

    Attributes
    ----------
    name
        Stable legacy identifier (always ``"local"``).
    model_id
        Retained legacy model identifier.
    schema_version
        Retained cache schema version.
    max_tokens
        Retained legacy context limit.
    batch_size
        Retained legacy batch size.
    """

    name: str = "local"
    schema_version: str = "1"
    max_tokens: int = 512
    batch_size: int = 16

    def __init__(self, model_id: str | None = None) -> None:
        """Construct a compatibility provider without loading a model.

        Parameters
        ----------
        model_id
            Optional legacy model identifier. The historical environment
            override remains supported so cache identities stay readable.
        """
        resolved = model_id or os.environ.get("PYQTEST_LOCAL_MODEL") or DEFAULT_MODEL_ID
        self.model_id: str = resolved
        self._model: None = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Reject local embedding generation with cloud migration guidance.

        Raises
        ------
        RuntimeError
            Always, because local embedding execution has been retired.
        """
        del texts
        raise RuntimeError(_CLOUD_ONLY_MESSAGE)

    def is_configured(self) -> bool:
        """Return ``False`` because the tombstone cannot generate vectors.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    def _ensure_loaded(self) -> SentenceTransformer:
        """Reject the former private model-loading operation.

        Raises
        ------
        RuntimeError
            Always, because no local model can be loaded.
        """
        raise RuntimeError(_CLOUD_ONLY_MESSAGE)

    def __getstate__(self) -> dict[str, Any]:
        """Return state compatible with objects serialized by older releases.

        Returns
        -------
        dict[str, Any]
            Legacy provider identity with no loaded model.
        """
        return {"model_id": self.model_id, "_model": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore legacy identity state without loading a local model."""
        self.model_id = state["model_id"]
        self._model = None


__all__ = ["DEFAULT_MODEL_ID", "LocalProvider"]
