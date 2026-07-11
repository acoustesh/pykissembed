"""Refactor index computation for prioritizing refactoring candidates.

Ported from ``mega-scrapper/tests/similarity/refactor_index.py``.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import numpy.typing as npt

from pykissembed.similarity.constants import (
    DEFAULT_REFACTOR_INDEX_THRESHOLD,
    DEFAULT_REFACTOR_INDEX_TOP_N,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pykissembed.similarity.types import FunctionInfo


Float32Array = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]

_MIN_FUNCTIONS_FOR_REFACTOR_INDEX = 2


class _RefactorConfig(TypedDict):
    """Configuration for refactoring index."""

    min_loc_for_similarity: int
    refactor_index_threshold: float
    refactor_index_top_n: int


def _as_str_object_mapping(value: object) -> Mapping[str, object]:
    """Cast an object to a string-to-object mapping.

    Returns
    -------
    Mapping[str, object]
        The string-to-object mapping.

    Raises
    ------
    TypeError
        If any mapping key is not a string.
    """
    if not isinstance(value, dict):
        return {}
    raw_dict = cast("dict[object, object]", value)
    typed: dict[str, object] = {}
    for key, item in raw_dict.items():
        if not isinstance(key, str):
            msg = "mapping keys must be strings"
            raise TypeError(msg)
        typed[key] = item
    return typed


def _parse_refactor_config(config_obj: object) -> _RefactorConfig:
    """Parse the refactor configuration.

    Returns
    -------
    _RefactorConfig
        The parsed refactor configuration.

    Raises
    ------
    TypeError
        If any config value has an invalid type.
    """
    default_config: _RefactorConfig = {
        "min_loc_for_similarity": 1,
        "refactor_index_threshold": DEFAULT_REFACTOR_INDEX_THRESHOLD,
        "refactor_index_top_n": DEFAULT_REFACTOR_INDEX_TOP_N,
    }
    config_map = _as_str_object_mapping(config_obj)
    min_loc_obj = config_map.get("min_loc_for_similarity", default_config["min_loc_for_similarity"])
    threshold_obj = config_map.get(
        "refactor_index_threshold",
        default_config["refactor_index_threshold"],
    )
    top_n_obj = config_map.get("refactor_index_top_n", default_config["refactor_index_top_n"])

    if not isinstance(min_loc_obj, int):
        msg = "config.min_loc_for_similarity must be int"
        raise TypeError(msg)
    if not isinstance(threshold_obj, int | float):
        msg = "config.refactor_index_threshold must be float"
        raise TypeError(msg)
    if not isinstance(top_n_obj, int):
        msg = "config.refactor_index_top_n must be int"
        raise TypeError(msg)

    return {
        "min_loc_for_similarity": min_loc_obj,
        "refactor_index_threshold": float(threshold_obj),
        "refactor_index_top_n": top_n_obj,
    }


def _parse_cached_embeddings(embeddings_obj: object) -> dict[str, list[float]]:
    """Parse cached embeddings.

    Returns
    -------
    dict[str, list[float]]
        The parsed cached embeddings.
    """
    embeddings_map = _as_str_object_mapping(embeddings_obj)

    parsed: dict[str, list[float]] = {}
    for key, value in embeddings_map.items():
        if not isinstance(value, list):
            continue
        raw_values = cast("list[object]", value)
        parsed_vec: list[float] = []
        valid = True
        for item in raw_values:
            if not isinstance(item, int | float):
                valid = False
                break
            parsed_vec.append(float(item))
        if valid:
            parsed[key] = parsed_vec
    return parsed


def compute_similarity_matrix(functions: list[FunctionInfo]) -> Float32Array:
    """Compute pairwise cosine similarity matrix for all functions.

    Returns
    -------
        Cosine similarity matrix with diagonal zeroed out.
    """
    embeddings: list[list[float]] = []
    for func in functions:
        if func.embedding is not None:
            embeddings.append(func.embedding)
        else:
            # A zero vector (not a skipped row) keeps this matrix's row/
            # column indices aligned 1:1 with `functions`, so callers can
            # index the result by the same position without a separate
            # remapping — a missing embedding just contributes zero
            # similarity to every pair instead of shifting later indices.
            embeddings.append([0.0] * 3072)  # text-embedding-3-large dimension

    emb_matrix = np.array(embeddings, dtype=np.float32)

    # Normalize rows
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0, np.float32(1.0), norms).astype(np.float32)
    emb_normalized = emb_matrix / norms

    # Compute similarity matrix via matrix multiplication
    similarity_matrix = np.asarray(emb_normalized @ emb_normalized.T, dtype=np.float32)
    np.fill_diagonal(similarity_matrix, 0.0)

    return similarity_matrix


def compute_max_similarities(similarity_matrix: Float32Array) -> Float64Array:
    """Compute max similarity for each function (excluding self).

    Returns
    -------
        Array of maximum similarity values per function.
    """
    return np.max(similarity_matrix, axis=1).astype(np.float64)


def compute_similarity_indices(max_similarities: Float64Array) -> Float64Array:
    """Compute similarity index: 25.403 * max_similarity^5.

    Returns
    -------
        Array of similarity index values.
    """
    # The 5th power makes this deliberately nonlinear: moderate similarity
    # (e.g. 0.7) contributes almost nothing, while near-duplicate code
    # (>0.9) spikes sharply — the intent is to flag true copy-paste, not
    # penalize functions that merely share a common idiom. 25.403 is a
    # calibration constant scaling max_similarity=1.0 to an index value
    # comparable in magnitude to the CC/COG terms it's combined with below.
    return np.asarray(25.403 * np.power(max_similarities, 5), dtype=np.float64)


def compute_refactor_indices(
    cc_values: Float32Array,
    cog_values: Float32Array,
    similarity_indices: Float64Array,
) -> Float64Array:
    """Compute refactor index: 0.25*CC + 0.15*COG + 0.6*similarity_index.

    Returns
    -------
        Array of refactor index values.
    """
    return 0.25 * cc_values + 0.15 * cog_values + 0.6 * similarity_indices


def get_refactor_priority_message(
    functions: list[FunctionInfo],
    cc_map: dict[str, int],
    cog_map: dict[str, int],
    threshold: float = DEFAULT_REFACTOR_INDEX_THRESHOLD,
    top_n: int = DEFAULT_REFACTOR_INDEX_TOP_N,
) -> str | None:
    """Generate refactoring priority message for functions above threshold.

    Returns
    -------
        Formatted message string, or ``None`` if no functions exceed the threshold.
    """
    if len(functions) < _MIN_FUNCTIONS_FOR_REFACTOR_INDEX or not any(
        f.embedding is not None for f in functions
    ):
        return None

    similarity_matrix: Float32Array = compute_similarity_matrix(functions)
    max_sims: Float64Array = compute_max_similarities(similarity_matrix)
    similarity_indices: Float64Array = compute_similarity_indices(max_sims)

    cc_values = np.zeros(len(functions), dtype=np.float32)
    cog_values = np.zeros(len(functions), dtype=np.float32)

    for i, func in enumerate(functions):
        key = f"{func.file}:{func.name}"
        cc_values[i] = cc_map.get(key, 0)
        cog_values[i] = cog_map.get(key, 0)

    refactor_indices: Float64Array = compute_refactor_indices(
        cc_values,
        cog_values,
        similarity_indices,
    )

    above_threshold = [
        (
            refactor_indices[i],
            functions[i],
            max_sims[i],
            cc_values[i],
            cog_values[i],
            similarity_indices[i],
        )
        for i in range(len(functions))
        if refactor_indices[i] >= threshold
    ]

    if not above_threshold:
        return None

    above_threshold.sort(key=operator.itemgetter(0), reverse=True)
    top_funcs = above_threshold[:top_n]

    lines = [
        f"\n{'=' * 70}",
        "REFACTORING PRIORITY RECOMMENDATION",
        f"{'=' * 70}",
        f"The following function(s) have Refactor Index >= {threshold:.1f} and should be",
        "prioritized for refactoring:",
        "",
    ]

    for ri, func, max_sim, cc, cog, sim_idx in top_funcs:
        lines.extend(
            (
                f"  {func.file}:{func.start_line} - {func.name}()",
                f"    Refactor Index: {ri:.2f}",
                f"      CC={cc:.0f}, COG={cog:.0f}, MaxSim={max_sim:.2%}, SimIdx={sim_idx:.2f}",
                "",
            ),
        )

    lines.extend(
        (
            "Formula: RefactorIndex = 0.25*CC + 0.15*COG + 0.6*(25.403 * MaxSimilarity^5)",
            f"{'=' * 70}",
        ),
    )

    return "\n".join(lines)


def get_refactor_priority_message_for_complexity() -> str:
    """Get the refactoring priority message for complexity test failures.

    Returns
    -------
        Refactoring priority message, or empty string if unavailable.
    """
    try:
        # Lazy: this helper is only invoked when a complexity check has
        # already failed, so avoid the ast_helpers/complexity/storage import
        # cost on the (common) happy path where nothing fails.
        from pykissembed.similarity.ast_helpers import extract_all_function_infos  # noqa: PLC0415
        from pykissembed.similarity.complexity import load_all_complexity_maps  # noqa: PLC0415
        from pykissembed.similarity.storage import load_baselines  # noqa: PLC0415

        baselines = _as_str_object_mapping(load_baselines())
        config = _parse_refactor_config(baselines.get("config"))
        min_loc = config["min_loc_for_similarity"]
        threshold = config["refactor_index_threshold"]
        top_n = config["refactor_index_top_n"]
        embeddings = _parse_cached_embeddings(baselines.get("embeddings"))

        functions = extract_all_function_infos(min_loc=min_loc)
        if len(functions) < _MIN_FUNCTIONS_FOR_REFACTOR_INDEX:
            return ""

        for func in functions:
            cached = embeddings.get(func.hash)
            if cached is not None:
                func.embedding = cached

        if not any(f.embedding is not None for f in functions):
            return ""

        cc_map, cog_map = load_all_complexity_maps()
        msg = get_refactor_priority_message(
            functions,
            cc_map,
            cog_map,
            threshold=threshold,
            top_n=top_n,
        )
    except Exception:  # noqa: BLE001 — best-effort supplementary message; any
        # failure here must not mask the underlying complexity-check failure.
        return ""
    else:
        return msg or ""
