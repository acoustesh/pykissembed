"""Symmetrized asymmetric-retrieval similarity for the Jina providers.

Jina's ``code2code`` / ``nl2code`` tasks are asymmetric: a function is embedded
once as a *query* and once as a *passage*, and the meaningful score is the
cross-similarity between the two, not the cosine of a single stored vector. This
module builds the pairwise matrix

    C_ij = cos(Q_i, P_j)          (query i vs passage j)
    S_ij = (C_ij + C_ji) / 2      (symmetrized for order-independent detection)

so the rest of the similarity gate (pair/neighbor thresholds, refactor index) can
consume ``S`` the same way it consumes a cosine matrix. Kept separate from the
single-vector cosine path in ``checks.py`` so that path is untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pykissembed.similarity.types import FunctionInfo

Float32Array = npt.NDArray[np.float32]


def _infer_dim(*caches: dict[str, list[float]]) -> int:
    """Return the embedding dimension from the first non-empty vector found.

    Returns
    -------
    int
        Length of the first cached vector across *caches*, or ``0`` if every
        cache is empty (in which case the similarity matrix is all zeros).
    """
    for cache in caches:
        for vec in cache.values():
            return len(vec)
    return 0


def _stacked(
    functions: list[FunctionInfo],
    cache: dict[str, list[float]],
    hash_field: str,
    dim: int,
) -> Float32Array:
    """Stack per-function vectors from *cache* in ``functions`` order.

    A function whose hash is absent from *cache* contributes a zero row so the
    matrix stays index-aligned with *functions*; a zero vector yields zero
    similarity to every other function rather than shifting later indices.

    Returns
    -------
    Float32Array
        Array of shape ``(len(functions), dim)``.
    """
    # A missing hash contributes a zero row so row i always maps to functions[i].
    rows = [
        vec if (vec := cache.get(getattr(func, hash_field))) is not None else [0.0] * dim
        for func in functions
    ]
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, dim), dtype=np.float32)


def _row_normalize(matrix: Float32Array) -> Float32Array:
    """Return *matrix* with each row scaled to unit L2 norm (zero rows unchanged).

    Returns
    -------
    Float32Array
        Row-normalised copy of *matrix*.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).astype(np.float32)
    # Replace zero norms with 1 to avoid divide-by-zero; zero rows stay zero.
    norms = np.where(norms == 0, np.float32(1.0), norms).astype(np.float32)
    return matrix / norms


def build_symmetrized_matrix(
    functions: list[FunctionInfo],
    query_cache: dict[str, list[float]],
    passage_cache: dict[str, list[float]],
    hash_field: str,
) -> Float32Array:
    """Build the symmetrized query/passage similarity matrix for *functions*.

    ``C = Qn @ Pn.T`` gives ``C_ij = cos(Q_i, P_j)`` (rows normalised); the result
    is symmetrized as ``(C + C.T) / 2`` and its diagonal zeroed so a function is
    never reported as similar to itself.

    Parameters
    ----------
    functions : list[FunctionInfo]
        Functions to compare, defining the row/column order.
    query_cache, passage_cache : dict[str, list[float]]
        Per-function query and passage embeddings, keyed by *hash_field*.
    hash_field : str
        ``FunctionInfo`` attribute used to key the caches
        (``"text_hash"`` for the nl2code variant, ``"hash"`` for code2code).

    Returns
    -------
    Float32Array
        The symmetrized similarity matrix with a zeroed diagonal.
    """
    n = len(functions)
    dim = _infer_dim(query_cache, passage_cache)
    if n == 0 or dim == 0:
        # No embeddings to compare — every pair scores zero.
        return np.zeros((n, n), dtype=np.float32)

    query_norm = _row_normalize(_stacked(functions, query_cache, hash_field, dim))
    passage_norm = _row_normalize(_stacked(functions, passage_cache, hash_field, dim))

    # cross[i, j] = cos(Q_i, P_j); averaging with its transpose makes the score
    # order-independent (the (i, j) and (j, i) query/passage roles are merged).
    cross = np.asarray(query_norm @ passage_norm.T, dtype=np.float32)
    symmetrized = np.asarray((cross + cross.T) / np.float32(2.0), dtype=np.float32)
    # A function is never its own near-duplicate.
    np.fill_diagonal(symmetrized, 0.0)
    return symmetrized
