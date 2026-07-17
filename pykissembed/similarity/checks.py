"""Unified similarity check workflow for all embedding providers.

Ported from ``mega-scrapper/tests/similarity/checks.py``. The main
adaptation is that imports use ``pykissembed.similarity.*`` instead of
``tests.similarity.*``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from pykissembed.similarity.constants import (
    DEFAULT_REFACTOR_INDEX_THRESHOLD,
    DEFAULT_REFACTOR_INDEX_TOP_N,
    JINA_AST_QUERY_EMBEDDINGS_FILE,
    JINA_TEXT_QUERY_EMBEDDINGS_FILE,
)
from pykissembed.similarity.embeddings import (
    compute_cosine_similarity,
    get_cached_embedding,
    is_embedding_cache,
    is_str_object_dict,
)
from pykissembed.similarity.exclusions import is_excluded_pair
from pykissembed.similarity.jina_similarity import Float32Array, build_symmetrized_matrix
from pykissembed.similarity.pca import fit_pca, transform_embeddings_with_pca
from pykissembed.similarity.populate_embeddings import cli_provider_name, get_provider_populator
from pykissembed.similarity.refactor_index import get_refactor_priority_message
from pykissembed.similarity.storage import (
    REGISTRY,
    HashType,
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

# Jina providers are standalone (not in REGISTRY's cosine flow): each drives the
# symmetrized query/passage path via ``run_jina_similarity_checks``, reading its
# own ``{name}_query_embeddings`` / ``{name}_passage_embeddings`` raw caches.
# ``cache_key`` is a logical id (no single on-disk cache); ``file_path`` is unused
# here — the raw caches carry the real files (registered in storage.REGISTRY).
JINA_TEXT_PROVIDER = ProviderEntry(
    name="jina_text",
    label="Jina-Text",
    cache_key="jina_text",
    file_path=JINA_TEXT_QUERY_EMBEDDINGS_FILE,
    hash_type=HashType.TEXT,
    default_threshold_pair=0.90,
    default_threshold_neighbor=0.85,
    standalone=True,
)
JINA_AST_PROVIDER = ProviderEntry(
    name="jina_ast",
    label="Jina-AST",
    cache_key="jina_ast",
    file_path=JINA_AST_QUERY_EMBEDDINGS_FILE,
    hash_type=HashType.AST,
    default_threshold_pair=0.90,
    default_threshold_neighbor=0.85,
    standalone=True,
)


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


def _combined_member_gaps(
    baselines: Baselines,
    uncached: list[FunctionInfo],
) -> dict[str, int]:
    """Count the uncached functions missing each combined member's embeddings.

    Returns
    -------
    dict[str, int]
        CLI provider name -> number of *uncached* functions absent from that
        member's cache. The raw Jina query/passage caches fold onto one
        provider, keeping the larger count. Empty when every member is fully
        cached and combined only needs a local rebuild.
    """
    gaps: dict[str, int] = {}
    for dep_key in REGISTRY.combined_dependencies + REGISTRY.standalone_dependencies:
        entry = REGISTRY.by_cache_key(dep_key)
        cache = baselines.get(dep_key)
        members = cache if is_embedding_cache(cache) else {}
        missing = sum(1 for func in uncached if getattr(func, entry.hash_field) not in members)
        if missing:
            # A Jina variant's query and passage caches fold onto the same CLI
            # provider; keep the larger miss count instead of double-counting
            # functions that are absent from both.
            name = cli_provider_name(dep_key)
            gaps[name] = max(gaps.get(name, 0), missing)
    return gaps


def _missing_embeddings_advice(
    provider: ProviderEntry,
    member_gaps: dict[str, int] | None,
) -> list[str]:
    """Build the remediation lines for a missing-embeddings skip.

    Returns
    -------
    list[str]
        Lines saying which populate step fills the gap. For combined,
        *member_gaps* selects between naming the incomplete member providers
        and pointing out that a local rebuild is all that is needed.
    """
    if member_gaps is None:
        return [f"Run: pykissembed populate-embeddings --provider {provider.label.lower()}"]
    if not member_gaps:
        # Only reachable under --cached-only: with every member cached, a
        # normal run would have rebuilt combined locally instead of skipping.
        return [
            (
                "All member embeddings are cached; rerun without --cached-only "
                "to rebuild Combined locally (no API calls needed)."
            ),
        ]
    return [
        (
            "Member embeddings missing for those functions "
            "(fix with: pykissembed populate-embeddings --provider <name>):"
        ),
        *(f"  {name}: {count}" for name, count in member_gaps.items()),
    ]


def _skip_missing_embeddings(
    baselines: Baselines,
    uncached: list[FunctionInfo],
    provider: ProviderEntry,
    *,
    total: int,
) -> None:
    """Skip the test with a diagnosis of which embeddings are missing.

    Reports how many of the *total* checked functions are uncached and how to
    fill the gap. When every function is uncached, the per-function list is
    dropped — it would just enumerate the whole codebase. For the combined
    provider, the diagnosis names the member providers that are actually
    incomplete instead of blaming the derived combined cache.
    """
    count = f"All {total}" if len(uncached) == total else f"{len(uncached)} of {total}"
    lines = [f"{count} functions lack cached {provider.label} embeddings."]
    member_gaps = (
        _combined_member_gaps(baselines, uncached)
        if provider.cache_key == REGISTRY.combined.cache_key
        else None
    )
    lines.extend(_missing_embeddings_advice(provider, member_gaps))
    if len(uncached) < total:
        lines.extend(f"  {func.file}:{func.name}" for func in uncached[:_MAX_SHOWN_UNCACHED])
        if len(uncached) > _MAX_SHOWN_UNCACHED:
            lines.append(f"  ... and {len(uncached) - _MAX_SHOWN_UNCACHED} more")
    pytest.skip("\n".join(lines))


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

        if is_excluded_pair(func_a, func_b, efp, efnp, class_function_proximity):
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
    excluded_file_pairs: list[list[str]] | None = None,
    excluded_function_pairs: list[list[str]] | None = None,
    class_function_proximity: int = 0,
    *,
    refactor_index_threshold: float,
    refactor_index_top_n: int,
) -> None:
    """Report violations and fail test if any found.

    The exclusion arguments mirror those used for pair/neighbor detection so
    the refactor-priority recommendation does not surface a method merely
    because its similarity is inflated by the class that contains it.
    """
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
            excluded_file_pairs=excluded_file_pairs,
            excluded_function_pairs=excluded_function_pairs,
            class_function_proximity=class_function_proximity,
        )
        if refactor_msg:
            error_msg += refactor_msg
        pytest.fail(error_msg, pytrace=False)


def run_provider_similarity_checks(
    *,
    baselines: Baselines,
    functions: list[FunctionInfo],
    update_baselines: bool,  # ruff:ignore[unused-function-argument] — accepted for call-site symmetry with
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

    When cached_only=False, missing embeddings are auto-populated via API.
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
            _skip_missing_embeddings(baselines, uncached, provider, total=len(functions))
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
                _skip_missing_embeddings(baselines, uncached, provider, total=len(functions))

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
        excluded_file_pairs,
        excluded_function_pairs,
        class_function_proximity,
        refactor_index_threshold=refactor_index_threshold,
        refactor_index_top_n=refactor_index_top_n,
    )


def _jina_uncached(
    functions: list[FunctionInfo],
    provider: ProviderEntry,
    query_cache: dict[str, list[float]],
    passage_cache: dict[str, list[float]],
) -> list[FunctionInfo]:
    """Return functions missing either a query or a passage Jina embedding.

    Returns
    -------
    list[FunctionInfo]
        Functions for which at least one of the two raw caches has no entry.
    """
    hash_field = provider.hash_field
    return [
        func
        for func in functions
        if query_cache.get(getattr(func, hash_field)) is None
        or passage_cache.get(getattr(func, hash_field)) is None
    ]


def _find_matrix_violations(
    functions: list[FunctionInfo],
    similarity: Float32Array,
    threshold_pair: float,
    threshold_neighbor: float,
    excluded_file_pairs: list[list[str]] | None = None,
    excluded_function_pairs: list[list[str]] | None = None,
    class_function_proximity: int = 0,
) -> tuple[list[str], list[str]]:
    """Find pair/neighbor violations from a precomputed similarity matrix.

    Mirrors :func:`_find_violations` but reads scores from *similarity* (the
    symmetrized Jina matrix) instead of computing per-pair cosine, so the same
    upper-triangle pairing, exclusions, and message formatting apply.

    Returns
    -------
    tuple[list[str], list[str]]
        Tuple of (pair violation messages, neighbor violation messages).
    """
    pair_violations: list[str] = []
    neighbor_violations: list[str] = []
    efp = excluded_file_pairs or []
    efnp = excluded_function_pairs or []

    for i, func_a in enumerate(functions):
        neighbor_entries: list[NeighborEntry] = []
        for j in range(i + 1, len(functions)):
            func_b = functions[j]
            if is_excluded_pair(func_a, func_b, efp, efnp, class_function_proximity):
                continue
            score = float(similarity[i, j])
            if score >= threshold_pair:
                pair_violations.append(_format_pair_violation(func_a, func_b, score))
            if score >= threshold_neighbor:
                neighbor_entries.append((func_b.file, func_b.name, func_b.start_line, score))
        if len(neighbor_entries) >= _MIN_NEIGHBORS_FOR_VIOLATION:
            neighbor_violations.append(_format_neighbor_violation(func_a, neighbor_entries))

    return pair_violations, neighbor_violations


def run_jina_similarity_checks(
    *,
    baselines: Baselines,
    functions: list[FunctionInfo],
    update_baselines: bool,  # ruff:ignore[unused-function-argument] — accepted for call-site symmetry with
    # run_provider_similarity_checks; embeddings are auto-populated when not cached_only.
    cached_only: bool,
    provider: ProviderEntry,
    threshold_pair: float,
    threshold_neighbor: float,
    load_complexity_maps_fn: Callable[[], tuple[dict[str, int], dict[str, int]]],
    pca_cache: PcaCache | None = None,  # ruff:ignore[unused-function-argument] — Jina bypasses PCA (the
    # asymmetric cross-score is not a single-vector cosine); accepted for symmetry.
    class_function_proximity: int = 0,
) -> None:
    """Similarity workflow for a Jina provider (symmetrized query/passage path).

    Loads the provider's two raw caches, auto-populates missing embeddings (unless
    *cached_only*), builds the symmetrized matrix, and reports pair/neighbor
    violations. Unlike :func:`run_provider_similarity_checks`, there is no single
    per-function vector and no PCA — similarity is ``(cos(Qi,Pj)+cos(Qj,Pi))/2``.

    Parameters
    ----------
    baselines : dict
        Baselines dict with function_hashes, config, and embeddings.
    functions : list[FunctionInfo]
        Functions to check.
    update_baselines : bool
        Accepted for symmetry; embeddings are auto-populated when not cached_only.
    cached_only : bool
        If True, skip the test when embeddings are missing.
    provider : ProviderEntry
        Standalone Jina provider descriptor (``jina_text`` / ``jina_ast``).
    threshold_pair : float
        Pair violation threshold.
    threshold_neighbor : float
        Neighbor violation threshold.
    load_complexity_maps_fn : Callable
        Returns ``(cc_map, cog_map)`` for the refactor-index message.
    pca_cache : dict | None
        Accepted for symmetry; unused (Jina bypasses PCA).
    class_function_proximity : int
        Max source lines allowed between a class and a nearby function pair.
    """
    if len(functions) < _MIN_FUNCTIONS_TO_COMPARE:
        pytest.skip("Not enough functions to compare")

    query_key = f"{provider.name}_query_embeddings"
    passage_key = f"{provider.name}_passage_embeddings"
    load_provider_embeddings(baselines, query_key)
    load_provider_embeddings(baselines, passage_key)
    query_cache = _extract_embedding_cache(baselines, query_key)
    passage_cache = _extract_embedding_cache(baselines, passage_key)

    uncached = _jina_uncached(functions, provider, query_cache, passage_cache)
    if uncached:
        if cached_only:
            _skip_missing_embeddings(baselines, uncached, provider, total=len(functions))
        else:
            populate_fn = get_provider_populator(provider.label.lower())
            if populate_fn is None:
                pytest.skip(f"Unknown provider: {provider.label}")
                return
            if populate_fn(baselines, functions) > 0:
                save_baselines(baselines)
            query_cache = _extract_embedding_cache(baselines, query_key)
            passage_cache = _extract_embedding_cache(baselines, passage_key)
            uncached = _jina_uncached(functions, provider, query_cache, passage_cache)
            if uncached:
                _skip_missing_embeddings(baselines, uncached, provider, total=len(functions))

    config = _extract_config(baselines)
    refactor_index_threshold = _extract_refactor_index_threshold(config)
    refactor_index_top_n = _extract_refactor_index_top_n(config)
    excluded_file_pairs = _extract_excluded_pairs(config, "excluded_file_pairs")
    excluded_function_pairs = _extract_excluded_pairs(config, "excluded_function_pairs")

    similarity = build_symmetrized_matrix(
        functions,
        query_cache,
        passage_cache,
        provider.hash_field,
    )
    pair_violations, neighbor_violations = _find_matrix_violations(
        functions,
        similarity,
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
        excluded_file_pairs,
        excluded_function_pairs,
        class_function_proximity,
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
