"""Real ``LocalProvider`` backed by ``sentence-transformers``.

The core ``pykissembed`` package ships a stub under ``pykissembed.providers.local``
that raises a clear error when ``pykissembed-local`` isn't installed. This
subpackage provides the real implementation and registers it via the
``pykissembed.providers`` entry-point group, so the stub is overridden
transparently (the registry keeps the last-registered entry per name).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sentence_transformers import SentenceTransformer

# Default model id. Override at populate-time via ``PYQTEST_LOCAL_MODEL``.
DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"


class LocalProvider:
    """Local sentence-transformers embedding provider.

    Attributes
    ----------
    name
        Stable identifier (always ``"local"``).
    model_id
        HuggingFace model id (default: ``BAAI/bge-small-en-v1.5``).
        Resolved from the ``PYQTEST_LOCAL_MODEL`` env var at construction
        time so cache keys are stable for the lifetime of the process.
    schema_version
        Bumped whenever the vector shape or semantics change. Used as
        part of the embedding cache key to prevent silent corruption.
    max_tokens
        bge-small-en-v1.5 has an effective 512-token context. Used by
        ``runner.populate`` to truncate inputs before encoding.
    batch_size
        Recommended maximum number of texts per ``embed`` call.
    """

    name: str = "local"
    schema_version: str = "1"
    max_tokens: int = 512
    batch_size: int = 16

    def __init__(self, model_id: str | None = None) -> None:
        """Construct the provider, deferring model load until first use.

        Parameters
        ----------
        model_id
            Override the default model id. Falls back to
            ``PYQTEST_LOCAL_MODEL`` and finally to ``DEFAULT_MODEL_ID``.
        """
        resolved = model_id or os.environ.get("PYQTEST_LOCAL_MODEL") or DEFAULT_MODEL_ID
        self.model_id: str = resolved
        self._model: SentenceTransformer | None = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute embedding vectors for *texts*.

        The model is loaded lazily on the first call (sentence-transformers
        weights can be 100+ MB, so deferring the load keeps
        ``pykissembed.providers list`` instant).

        Returns
        -------
        list[list[float]]
            One vector per input text. Vectors are L2-normalised (the
            default for bge-small-en-v1.5) so cosine similarity reduces
            to a dot product downstream.
        """
        model = self._ensure_loaded()
        inputs = list(texts)
        if not inputs:
            return []
        # ``convert_to_numpy=True`` yields a 2D ndarray; cast to a list
        # of plain float lists so the output matches the Provider Protocol.
        import numpy as np

        vectors = model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vectors)
        result: list[list[float]] = [[float(x) for x in row] for row in arr]
        return result

    def is_configured(self) -> bool:
        """Return True iff sentence-transformers is importable.

        The provider never *requires* a key, but it does require the
        model weights to be downloadable. We treat importability as a
        proxy: if the package imports, the user can run ``populate`` and
        the model will fetch on first use.

        Returns
        -------
        bool
            ``True`` if ``sentence_transformers`` imports successfully.
        """
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_loaded(self) -> SentenceTransformer:
        """Lazy-load the sentence-transformers model on first embed call.

        Returns
        -------
        SentenceTransformer
            The cached model instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        return self._model

    def __getstate__(self) -> dict[str, Any]:
        """Pickle support — drop the cached model so workers re-load it.

        Returns
        -------
        dict[str, Any]
            ``{"model_id": ..., "_model": None}`` — enough state to
            rebuild the provider without carrying the heavy model.
        """
        return {"model_id": self.model_id, "_model": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Pickle support — restore *state* and force a model reload."""
        self.model_id = state["model_id"]
        self._model = None


__all__ = ["DEFAULT_MODEL_ID", "LocalProvider"]
