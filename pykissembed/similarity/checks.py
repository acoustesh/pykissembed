"""Unified similarity check workflow for all embedding providers.

Ported from ``mega-scrapper/tests/similarity/checks.py``. The main
adaptation is that imports use ``pykissembed.similarity.*`` instead of
``tests.similarity.*``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from pykissembed.config import get_config
from pykissembed.similarity.constants import (
    DEFAULT_REFACTOR_INDEX_THRESHOLD,
    DEFAULT_REFACTOR_INDEX_TOP_N,
)
from pykissembed.similarity.embeddings import (
    compute_cosine_similarity,
    get_cached_embedding,
    is_embedding_cache,
    is_str_object_dict,
)
from pykissembed.similarity.pca import fit_pca, transform_embeddings_with_pca
from pykissembed.similarity.populate_embeddings import get_provider_populator
from pykissembed.similarity.refactor_index import get_refactor_priority_message
from pykissembed.similarity.storage import (
    REGISTRY,
    ProviderEntry,
    load_provider_embeddings,
    save_baselines,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pykissembed.similarity.types import FunctionInfo, PCAModel

type Baselines = dict[str, object]
type NeighborEntry = tuple[str, str, int, float]
type PcaCache = dict[str, tuple[PCAModel | None, int, bool]]

_MAX_SHOWN_UNCACHED = 10
_EXCLUSION_PAIR_SIZE = 2
_MIN_NEIGHBORS_FOR_VIOLATION = 2
_MIN_FUNCTIONS_TO_COMPARE = 2

# Provider aliases — single source of truth is REGISTRY in storage.py
OPENAI_TEXT_PROVIDER = REGISTRY.by_cache_key("openai_text_embeddings")
OPENAI_AST_PROVIDER = REGISTRY.by_cache_key("openai_ast_embeddings")
CODESTRAL_TEXT_PROVIDER = REGISTRY.by_cache_key("codestral_text_embeddings")
CODESTRAL_AST_PROVIDER = REGISTRY.by_cache_key("codestral_ast_embeddings")
VOYAGE_TEXT_PROVIDER = REGISTRY.by_cache_key("voyage_text_embeddings")
VOYAGE_AST_PROVIDER = REGISTRY.by_cache_key("voyage_ast_embeddings")
GEMINI_TEXT_PROVIDER = REGISTRY.by_cache_key("gemini_text_embeddings")
GEMINI_AST_PROVIDER = REGISTRY.by_cache_key("gemini_ast_embeddings")
QWEN_TEXT_PROVIDER = REGISTRY.by_cache_key("qwen_text_embeddings")
QWEN_AST_PROVIDER = REGISTRY.by_cache_key("qwen_ast_embeddings")
COMBINED_PROVIDER = REGISTRY.by_cache_key("combined_embeddings")


def _load_cached_embeddings(
    baselines: Baselines,
    functions: list[FunctionInfo],
    provider: ProviderEntry,
) -> list[FunctionInfo]:
    """Load cached embeddings and return list of uncached functions.

    Returns
    -------
    list[FunctionInfo]
        List of functions that did not have a cached embedding.
    """
    uncached: list[FunctionInfo] = []
    for func in functions:
        # Use provider-specific hash field (text_hash for Text/Combined, hash for AST)
        content_hash = getattr(func, provider.hash_field)
        cached = get_cached_embedding(baselines, content_hash, provider.cache_key)
        if cached is not None:
            func.embedding = cached
        else:
            uncached.append(func)
    return uncached


def _skip_missing_embeddings(uncached: list[FunctionInfo], provider: ProviderEntry) -> None:
    """Skip test if any functions lack cached embeddings."""
    uncached_names = [f"{f.file}:{f.name}" for f in uncached[:_MAX_SHOWN_UNCACHED]]
    more_msg = (
        f"\n  ... and {len(uncached) - _MAX_SHOWN_UNCACHED} more"
        if len(uncached) > _MAX_SHOWN_UNCACHED
        else ""
    )
    pytest.skip(
        f"{len(uncached)} functions lack cached {provider.label} embeddings.\n"
        f"Run: pykissembed populate-embeddings --provider "
        f"{provider.label.lower()}\n  " + "\n  ".join(uncached_names) + more_msg,
    )


def _format_pair_violation(func_a: FunctionInfo, func_b: FunctionInfo, similarity: float) -> str:
    """Format a single pair violation message.

    Returns
    -------
    str
        Formatted violation string.
    """
    return (
        f"{func_a.file}:{func_a.start_line} {func_a.name}() vs "
        f"{func_b.file}:{func_b.start_line} {func_b.name}() - "
        f"similarity: {similarity:.1%}"
    )


def _format_neighbor_violation(func_a: FunctionInfo, similar_neighbors: list[NeighborEntry]) -> str:
    """Format a neighbor violation message.

    Returns
    -------
    str
        Formatted neighbor violation string.
    """
    neighbor_info = ", ".join(f"{f}:{n}() ({s:.1%})" for f, n, _, s in similar_neighbors[:3])
    return (
        f"{func_a.file}:{func_a.start_line} {func_a.name}() has "
        f"{len(similar_neighbors)} similar functions: {neighbor_info}"
    )


def _is_excluded_pair(
    func_a: FunctionInfo,
    func_b: FunctionInfo,
    excluded_file_pairs: list[list[str]],
    excluded_function_pairs: list[list[str]],
    class_function_proximity: int = 0,
) -> bool:
    """Check whether a pair is configured or structurally excluded.

    Returns
    -------
    bool
        True if the pair is excluded, False otherwise.
    """
    # Check file-level exclusions
    for pair in excluded_file_pairs:
        if len(pair) != _EXCLUSION_PAIR_SIZE:
            continue
        pattern_a, pattern_b = pair
        if (pattern_a in func_a.file and pattern_b in func_b.file) or (
            pattern_b in func_a.file and pattern_a in func_b.file
        ):
            return True

    # Check function-level exclusions
    func_a_key = f"{func_a.file}:{func_a.name}"
    func_b_key = f"{func_b.file}:{func_b.name}"
    for pair in excluded_function_pairs:
        if len(pair) != _EXCLUSION_PAIR_SIZE:
            continue
        pattern_a, pattern_b = pair
        if (pattern_a in func_a_key and pattern_b in func_b_key) or (
            pattern_b in func_a_key and pattern_a in func_b_key
        ):
            return True

    # A class naturally contains its methods' source. Comparing the class
    # block with one of those methods produces a structural false positive;
    # nearby class/function pairs have the same property in small test files.
    is_class_a = func_a.text.lstrip().startswith(("class ", "class\t"))
    is_class_b = func_b.text.lstrip().startswith(("class ", "class\t"))
    if func_a.file != func_b.file or is_class_a == is_class_b:
        return False

    first, second = sorted((func_a, func_b), key=lambda func: func.start_line)
    if first.end_line >= second.start_line:
        return class_function_proximity >= 0

    source_file = Path(first.file)
    if not source_file.is_absolute():
        source_file = get_config().root / source_file
    try:
        source_lines = source_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False

    code_lines_between = sum(
        bool(stripped := line.strip()) and not stripped.startswith("#")
        for line in source_lines[first.end_line : second.start_line - 1]
    )
    return code_lines_between <= class_function_proximity


def _check_against_others(
    func_a: FunctionInfo,
    func_a_idx: int,
    functions: list[FunctionInfo],
    threshold_pair: float,
    threshold_neighbor: float,
    excluded_file_pairs: list[list[str]] | None = None,
    excluded_function_pairs: list[list[str]] | None = None,
    class_function_proximity: int = 0,
) -> tuple[list[str], list[NeighborEntry]]:
    """Check one function against all later functions in the list.

    Returns
    -------
    tuple[list[str], list[tuple]]
        Tuple of (pair violation messages, neighbor entry tuples).
    """
    pair_violations: list[str] = []
    neighbor_entries: list[NeighborEntry] = []
    efp = excluded_file_pairs or []
    efnp = excluded_function_pairs or []
    embedding_a = func_a.embedding
    if embedding_a is None:
        return pair_violations, neighbor_entries

    for j, func_b in enumerate(functions):
        # `func_a_idx >= j` restricts comparisons to the upper triangle of
        # the pair matrix: skips comparing a function to itself (j ==
        # func_a_idx) and skips the (b, a) half of every pair already
        # covered as (a, b) by an earlier outer-loop iteration.
        if func_a_idx >= j or func_b.embedding is None:
            continue

        if _is_excluded_pair(func_a, func_b, efp, efnp, class_function_proximity):
            continue

        similarity = compute_cosine_similarity(embedding_a, func_b.embedding)

        if similarity >= threshold_pair:
            pair_violations.append(_format_pair_violation(func_a, func_b, similarity))
        if similarity >= threshold_neighbor:
            neighbor_entries.append((func_b.file, func_b.name, func_b.start_line, similarity))

    return pair_violations, neighbor_entries


def _find_violations(
    functions: list[FunctionInfo],
    threshold_pair: float,
    threshold_neighbor: float,
    excluded_file_pairs: list[list[str]] | None = None,
    excluded_function_pairs: list[list[str]] | None = None,
    class_function_proximity: int = 0,
) -> tuple[list[str], list[str]]:
    """Find similarity violations among functions.

    Returns
    -------
    tuple[list[str], list[str]]
        Tuple of (pair violation messages, neighbor violation messages).
    """
    pair_violations: list[str] = []
    neighbor_violations: list[str] = []

    for i, func_a in enumerate(functions):
        if func_a.embedding is None:
            continue

        func_pair_viols, similar_neighbors = _check_against_others(
            func_a,
            i,
            functions,
            threshold_pair,
            threshold_neighbor,
            excluded_file_pairs,
            excluded_function_pairs,
            class_function_proximity,
        )
        pair_violations.extend(func_pair_viols)

        if len(similar_neighbors) >= _MIN_NEIGHBORS_FOR_VIOLATION:
            neighbor_violations.append(_format_neighbor_violation(func_a, similar_neighbors))

    return pair_violations, neighbor_violations


def _report_violations(
    pair_violations: list[str],
    neighbor_violations: list[str],
    threshold_pair: float,
    threshold_neighbor: float,
    functions: list[FunctionInfo],
    load_complexity_maps_fn: Callable[[], tuple[dict[str, int], dict[str, int]]],
    refactor_index_threshold: float,
    refactor_index_top_n: int,
) -> None:
    """Report violations and fail test if any found."""
    all_violations: list[str] = []
    if pair_violations:
        all_violations.append(
            f"High similarity pairs (>={threshold_pair:.0%}):\n  " + "\n  ".join(pair_violations),
        )
    if neighbor_violations:
        all_violations.append(
            f"Functions with multiple similar neighbors (>={threshold_neighbor:.0%}):\n  "
            + "\n  ".join(neighbor_violations),
        )

    if all_violations:
        error_msg = "\n\n".join(all_violations)
        cc_map, cog_map = load_complexity_maps_fn()
        refactor_msg = get_refactor_priority_message(
            functions,
            cc_map,
            cog_map,
            threshold=refactor_index_threshold,
            top_n=refactor_index_top_n,
        )
        if refactor_msg:
            error_msg += refactor_msg
        pytest.fail(error_msg, pytrace=False)


def run_provider_similarity_checks(
    *,
    baselines: Baselines,
    functions: list[FunctionInfo],
    update_baselines: bool,  # noqa: ARG001 — accepted for call-site symmetry with
    # cached_only; currently unused (embeddings are always auto-populated when not
    # cached_only, regardless of this flag).
    cached_only: bool,
    provider: ProviderEntry,
    threshold_pair: float,
    threshold_neighbor: float,
    load_complexity_maps_fn: Callable[[], tuple[dict[str, int], dict[str, int]]],
    pca_cache: PcaCache | None = None,
    class_function_proximity: int = 0,
) -> None:
    """Unified workflow for similarity tests across all embedding providers.

    This single function handles OpenAI, Codestral, Voyage, Gemini, and
    Combined providers with the same flow: load cached → populate if
    needed → apply PCA → find violations.

    When cached_only=False (default), missing embeddings are auto-populated via API.
    When cached_only=True, the test is skipped if embeddings are missing.

    Parameters
    ----------
    baselines: Dict with function_hashes, config, and embeddings (lazy-loaded).
    functions: List of FunctionInfo objects to check.
    update_baselines: Currently unused — embeddings are always auto-populated
        when not cached_only, regardless of this flag. Kept for call-site
        symmetry with cached_only.
    cached_only: If True, skip test when embeddings are missing.
    provider: ProviderEntry with label, cache_key, etc.
    threshold_pair: Similarity threshold for pair violations.
    threshold_neighbor: Similarity threshold for neighbor violations.
    load_complexity_maps_fn: Callable returning (cc_map, cog_map) for refactor index.
    pca_cache: Optional session-scoped cache for fitted PCA models.
    class_function_proximity: Maximum non-blank, non-comment source lines
        allowed between a class and a function in the same file.
    """
    if len(functions) < _MIN_FUNCTIONS_TO_COMPARE:
        pytest.skip("Not enough functions to compare")

    # Lazy-load this provider's embeddings if not already loaded
    load_provider_embeddings(baselines, provider.cache_key)

    # Load cached embeddings
    uncached = _load_cached_embeddings(baselines, functions, provider)

    if uncached:
        if cached_only:
            _skip_missing_embeddings(uncached, provider)
        else:
            # Auto-populate missing embeddings: base providers via API,
            # combined provider from already-loaded base caches.
            populate_fn = get_provider_populator(provider.label.lower())
            if populate_fn is None:
                pytest.skip(f"Unknown provider: {provider.label}")
                return
            new_count = populate_fn(baselines, functions)
            if new_count > 0:
                save_baselines(baselines)
            # Reload embeddings after populating
            uncached = _load_cached_embeddings(baselines, functions, provider)
            if uncached:
                _skip_missing_embeddings(uncached, provider)

    # Apply PCA dimensionality reduction using the provider-specific override,
    # the generic baseline setting, or the provider default, in that order.
    config = _extract_config(baselines)
    pca_variance = _extract_pca_variance(config, provider)
    refactor_index_threshold = _extract_refactor_index_threshold(config)
    refactor_index_top_n = _extract_refactor_index_top_n(config)
    embeddings_cache = _extract_embedding_cache(baselines, provider.cache_key)

    # Use cached PCA if available, otherwise fit fresh
    pca_model, n_components, is_gpu = fit_pca(
        embeddings_cache,
        pca_variance,
        pca_cache=pca_cache,
        cache_key=provider.cache_key if pca_cache is not None else "",
    )
    if pca_model is not None:
        transform_embeddings_with_pca(functions, pca_model, n_components, is_gpu=is_gpu)

    # Load exclusion lists (intentionally similar pairs to skip)
    excluded_file_pairs = _extract_excluded_pairs(config, "excluded_file_pairs")
    excluded_function_pairs = _extract_excluded_pairs(config, "excluded_function_pairs")

    # Find and report violations
    pair_violations, neighbor_violations = _find_violations(
        functions,
        threshold_pair,
        threshold_neighbor,
        excluded_file_pairs,
        excluded_function_pairs,
        class_function_proximity,
    )
    _report_violations(
        pair_violations,
        neighbor_violations,
        threshold_pair,
        threshold_neighbor,
        functions,
        load_complexity_maps_fn,
        refactor_index_threshold=refactor_index_threshold,
        refactor_index_top_n=refactor_index_top_n,
    )


def _extract_config(baselines: Baselines) -> dict[str, object]:
    """Extract configuration from a dictionary.

    Returns
    -------
    dict[str, object]
        The extracted configuration.

    Raises
    ------
    TypeError
        If ``baselines['config']`` is not ``dict[str, object]``.
    """
    config = baselines.get("config", {})
    if not is_str_object_dict(config):
        msg = "baselines['config'] must be a dict[str, object]"
        raise TypeError(msg)
    return config


def _extract_pca_variance(config: dict[str, object], provider: ProviderEntry) -> float:
    """Extract PCA variance from a dictionary.

    Returns
    -------
    float
        The extracted PCA variance.

    Raises
    ------
    TypeError
        If the configured PCA variance value is not numeric.
    """
    raw_variance = config.get(
        provider.pca_variance_key,
        config.get("pca_variance_threshold", provider.default_pca_variance),
    )
    if not isinstance(raw_variance, int | float):
        msg = (
            f"Config key '{provider.pca_variance_key}' or 'pca_variance_threshold' must be numeric"
        )
        raise TypeError(msg)
    return float(raw_variance)


def _extract_refactor_index_threshold(config: dict[str, object]) -> float:
    """Extract the refactor-index threshold from similarity configuration.

    Returns
    -------
    float
        The configured threshold, or the library default when absent.

    Raises
    ------
    TypeError
        If the configured threshold is not numeric.
    """
    raw_threshold = config.get("refactor_index_threshold", DEFAULT_REFACTOR_INDEX_THRESHOLD)
    if not isinstance(raw_threshold, int | float):
        msg = "config['refactor_index_threshold'] must be float"
        raise TypeError(msg)
    return float(raw_threshold)


def _extract_refactor_index_top_n(config: dict[str, object]) -> int:
    """Extract the number of refactor recommendations to report.

    Returns
    -------
    int
        The configured number of recommendations, or the library default when absent.

    Raises
    ------
    TypeError
        If the configured count is not an integer.
    """
    raw_top_n = config.get("refactor_index_top_n", DEFAULT_REFACTOR_INDEX_TOP_N)
    if not isinstance(raw_top_n, int):
        msg = "config['refactor_index_top_n'] must be int"
        raise TypeError(msg)
    return raw_top_n


def _extract_embedding_cache(baselines: Baselines, cache_key: str) -> dict[str, list[float]]:
    """Copy the validated provider cache out of *baselines* without side effects.

    The value stored under *cache_key* is type-checked and then duplicated into
    a fresh mapping; a missing key simply yields an empty result. Because the
    snapshot is detached from *baselines*, callers may read or reshape it freely
    without disturbing the persisted baseline.

    Returns
    -------
    dict[str, list[float]]
        A newly allocated mapping mirroring the stored cache.

    Raises
    ------
    TypeError
        If the value found under *cache_key* is not ``dict[str, list[float]]``.
    """
    raw_cache = baselines.get(cache_key, {})
    if not is_embedding_cache(raw_cache):
        msg = f"baselines['{cache_key}'] must be a dict[str, list[float]]"
        raise TypeError(msg)
    return dict(raw_cache)


def _extract_excluded_pairs(config: dict[str, object], key: str) -> list[list[str]]:
    """Extract excluded pairs from a dictionary.

    Returns
    -------
    list[list[str]]
        The extracted excluded pairs.

    Raises
    ------
    TypeError
        If the excluded pairs data is not ``list[list[str]]``.
    """
    raw_pairs = config.get(key, [])
    if not isinstance(raw_pairs, list):
        msg = f"config['{key}'] must be a list"
        raise TypeError(msg)
    pairs: list[list[str]] = []
    for pair in cast("list[object]", raw_pairs):
        if not isinstance(pair, list) or not all(
            isinstance(item, str) for item in cast("list[object]", pair)
        ):
            msg = f"config['{key}'] entries must be list[str]"
            raise TypeError(msg)
        pairs.append(cast("list[str]", pair))
    return pairs
