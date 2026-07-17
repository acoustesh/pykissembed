"""Provider Protocol — the contract every embedding provider must satisfy.

The Protocol is **sync** by design: providers are tiny CPU/IO wrappers,
`tiktoken` truncation is the bottleneck, not network I/O. Async would add
complexity without measurable benefit.

A Provider must also be **batch-aware**: ``embed`` accepts a sequence of
texts and returns a parallel list of vectors. Implementations are
encouraged to chunk internally to honour ``batch_size``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class Provider(Protocol):
    """Sync embedding provider contract.

    Attributes
    ----------
    name
        Stable identifier (e.g. ``"openai"``, ``"my_embedder"``). Lower-case,
        no spaces.
    model_id
        Model identifier (e.g. ``"text-embedding-3-large"``).
    schema_version
        Bumped whenever the vector shape or semantics change. Used as
        part of the embedding cache key to prevent silent corruption.
    max_tokens
        Maximum tokens the provider accepts per input. pykissembed will
        truncate inputs beyond this limit.
    batch_size
        Recommended maximum number of texts per ``embed`` call.
    """

    name: str
    model_id: str
    schema_version: str
    max_tokens: int
    batch_size: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute embedding vectors for *texts*.

        Returns
        -------
        list[list[float]]
            One vector per input text. Length of the outer list MUST equal
            ``len(texts)``.
        """
        ...

    def is_configured(self) -> bool:
        """Return True iff the provider can be used right now.

        Cloud providers typically check for the relevant API key in the
        environment. Other implementations may use their own readiness check.
        """
        ...
