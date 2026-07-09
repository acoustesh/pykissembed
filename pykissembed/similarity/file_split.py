"""File split proposal generator using PCA and k-means clustering.

Ported from ``mega-scrapper/tests/similarity/file_split.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pykissembed.similarity.ast_helpers import extract_function_infos_from_file
from pykissembed.similarity.pca import cluster_functions_kmeans_with_pca, fit_pca

if TYPE_CHECKING:
    from pathlib import Path

_MIN_FUNCTIONS_FOR_SPLIT = 4
_SPLIT_CLUSTER_COUNT = 2


def _as_embeddings_cache(value: object) -> dict[str, list[float]]:
    """Cast a dictionary to an embeddings cache.

    Returns
    -------
    dict[str, list[float]]
        The embeddings cache.
    """
    if not isinstance(value, dict):
        return {}

    raw = cast("dict[object, object]", value)
    for key, vector in raw.items():
        if not isinstance(key, str):
            return {}
        if not isinstance(vector, list):
            return {}
        vector_items = cast("list[object]", vector)
        if not all(isinstance(item, int | float) for item in vector_items):
            return {}

    return cast("dict[str, list[float]]", raw)


def _as_config(value: object) -> dict[str, object]:
    """Cast a dictionary to a configuration object.

    Returns
    -------
    dict[str, object]
        The configuration object.
    """
    if not isinstance(value, dict):
        return {}
    raw = cast("dict[object, object]", value)
    return {key: entry for key, entry in raw.items() if isinstance(key, str)}


def generate_file_split_proposal(file_path: Path, baselines: dict[str, object]) -> str | None:
    """Generate a proposal to split a file into two based on k-means clustering.

    Returns
    -------
    str or None
        A formatted split proposal string, or ``None`` when a split is not feasible.
    """
    functions = extract_function_infos_from_file(file_path, min_loc=1)

    if len(functions) < _MIN_FUNCTIONS_FOR_SPLIT:
        return None

    embeddings_cache = _as_embeddings_cache(baselines.get("embeddings"))

    # Load cached embeddings
    for func in functions:
        cached = embeddings_cache.get(func.hash)
        if cached is not None:
            func.embedding = cached

    funcs_with_embeddings = [f for f in functions if f.embedding is not None]
    if len(funcs_with_embeddings) < _MIN_FUNCTIONS_FOR_SPLIT:
        return f"  (Cannot generate split proposal: only {len(funcs_with_embeddings)} functions have cached embeddings)"

    config = _as_config(baselines.get("config"))
    pca_variance_obj = config.get("pca_variance_threshold", 0.95)
    pca_variance = float(pca_variance_obj) if isinstance(pca_variance_obj, int | float) else 0.95

    pca_model, n_components, _is_gpu = fit_pca(embeddings_cache, variance_threshold=pca_variance)

    if pca_model is None:
        return "  (Cannot generate split proposal: not enough cached embeddings for PCA)"

    clusters, _ = cluster_functions_kmeans_with_pca(
        funcs_with_embeddings,
        pca_model,
        n_components,
        n_clusters=_SPLIT_CLUSTER_COUNT,
    )

    if len(clusters) < _SPLIT_CLUSTER_COUNT or not clusters[0] or not clusters[1]:
        return None

    base_name = file_path.stem
    pca_variance_pct = int(pca_variance * 100)
    lines = [
        "",
        f"  PROPOSED FILE SPLIT for {file_path.name}:",
        f"  {'─' * 60}",
        f"  (Using PCA with {n_components} components explaining {pca_variance_pct}% variance)",
        "",
        f"  File 1: {base_name}_part1.py ({len(clusters[0])} functions/classes)",
    ]

    lines.extend(
        f"    - {func.name} (lines {func.start_line}-{func.end_line})" for func in clusters[0]
    )
    lines.extend(("", f"  File 2: {base_name}_part2.py ({len(clusters[1])} functions/classes)"))
    lines.extend(
        f"    - {func.name} (lines {func.start_line}-{func.end_line})" for func in clusters[1]
    )
    lines.extend(
        (
            "",
            "  Note: Functions are grouped by semantic similarity using PCA + k-means.",
            f"  {'─' * 60}",
        ),
    )

    return "\n".join(lines)
