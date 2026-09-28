"""Populate embedding caches for all providers.

Ported from ``mega-scrapper/tests/similarity/populate_embeddings.py``.
This module fetches embeddings from APIs for functions that don't have
cached values. Run this before running tests if you need to populate
missing embeddings.

Usage:
    python -m pykissembed.similarity.populate_embeddings --provider PROVIDER

Options:
    --provider  One of: openai-text, openai-ast, codestral-text, codestral-ast,
                voyage-text, voyage-ast, gemini-text, gemini-ast, qwen-text,
                qwen-ast, jina-text, jina-ast, combined, all
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import TYPE_CHECKING, TypedDict

import pytest

from pykissembed.config import get_config as _get_config
from pykissembed.paths import resolve_paths as _resolve_paths
from pykissembed.similarity.ast_helpers import (
    collapse_scan_directories,
    extract_all_function_infos,
    extract_function_infos_from_directories,
)
from pykissembed.similarity.constants import (
    JINA_CODE2CODE_PASSAGE,
    JINA_CODE2CODE_QUERY,
    JINA_NL2CODE_PASSAGE,
    JINA_NL2CODE_QUERY,
)
from pykissembed.similarity.embeddings import (
    get_cached_embedding,
    get_embeddings_batch,
    is_embedding_cache,
    is_str_object_dict,
    load_api_key_from_env,
)
from pykissembed.similarity.storage import (
    REGISTRY,
    load_baselines,
    merge_embedding_caches,
    save_baselines,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pykissembed.similarity.types import FunctionInfo

type Baselines = dict[str, object]
type PopulateFn = Callable[[Baselines, list[FunctionInfo]], int]


class PopulationError(RuntimeError):
    """Raised when an explicit cache-population request cannot be completed."""


class _ProviderRequestError(RuntimeError):
    """Normalized failure from a third-party embedding transport."""


def _emit(message: str) -> None:
    """Write one status line to standard output."""
    _ = sys.stdout.write(f"{message}\n")


def _request_embeddings(
    texts: list[str],
    *,
    provider: str,
    task: str = "",
) -> list[list[float]]:
    """Fetch embeddings and normalize undocumented provider exceptions.

    Parameters
    ----------
    texts : list[str]
        Texts to embed in one batch.
    provider : str
        Canonical provider name selecting the transport.
    task : str
        Jina task code; empty for non-Jina providers.

    Returns
    -------
    list[list[float]]
        Embedding vectors returned by the selected provider.

    Raises
    ------
    ModuleNotFoundError
        If the selected provider's optional dependency is not installed.
    _ProviderRequestError
        If the provider transport or API raises another ordinary exception.
    """
    try:
        return get_embeddings_batch(texts, provider=provider, task=task)
    except ModuleNotFoundError:
        raise
    except Exception as exc:
        msg = str(exc) or f"{provider} embedding request failed"
        raise _ProviderRequestError(msg) from exc


class _FunctionHashEntry(TypedDict):
    """Represents a function hash entry."""

    hash: str
    text_hash: str


# The two cache helpers below hand back the *live* object stored in
# ``baselines`` instead of a copy. Callers rely on that: they write new
# vectors straight into the returned mapping, so a defensive copy would be
# silently discarded at the end of the run and every paid embedding lost.


def _get_embedding_cache(baselines: Baselines, cache_key: str) -> dict[str, list[float]]:
    """Return the live provider cache for *cache_key*, creating it when absent.

    A missing entry is initialised to an empty mapping, stored back into
    *baselines*, and returned, so later in-place writes by the caller persist.
    This deliberately mutates *baselines* and hands back the very object it
    holds rather than a defensive copy.

    Parameters
    ----------
    baselines : Baselines
        Mutable baselines mapping holding the cache.
    cache_key : str
        Key naming the provider cache to fetch.

    Returns
    -------
    dict[str, list[float]]
        The mutable cache owned by *baselines* under *cache_key*.

    Raises
    ------
    TypeError
        If ``baselines[cache_key]`` exists but is not ``dict[str, list[float]]``.
    """
    cache_obj = baselines.get(cache_key)
    if cache_obj is None:
        merge_embedding_caches(baselines, {cache_key: {}})
        cache_obj = baselines[cache_key]
    if not is_embedding_cache(cache_obj):
        msg = f"Expected {cache_key} to be dict[str, list[float]], got {type(cache_obj).__name__}"
        raise TypeError(msg)
    return cache_obj


def _get_function_hashes(baselines: Baselines) -> dict[str, object]:
    """Return the live ``function_hashes`` mapping, creating it when absent.

    A missing entry is initialised to an empty dict and stored back into
    *baselines*, so later writes by the caller persist. This deliberately
    mutates *baselines* rather than returning a defensive copy.

    Parameters
    ----------
    baselines : dict
        Mutable baselines dict; may gain a ``function_hashes`` entry.

    Returns
    -------
    dict[str, object]
        The mutable function-hashes mapping owned by *baselines*.

    Raises
    ------
    TypeError
        If ``baselines["function_hashes"]`` exists but is not ``dict[str, object]``.
    """
    hashes_obj = baselines.get("function_hashes")
    if hashes_obj is None:
        empty_hashes: dict[str, object] = {}
        baselines["function_hashes"] = empty_hashes
        return empty_hashes
    if not is_str_object_dict(hashes_obj):
        msg = f"Expected function_hashes to be dict[str, object], got {type(hashes_obj).__name__}"
        raise TypeError(msg)
    return hashes_obj


@dataclass(frozen=True)
class _ProviderCfg:
    """Configuration for a single embedding provider's populate step."""

    label: str
    env_var: str
    invalid_prefixes: tuple[str, ...]
    cache_key: str
    use_text: bool
    provider: str


@dataclass(frozen=True)
class _JinaCfg:
    """Configuration for a Jina variant's populate step (query + passage caches).

    Attributes
    ----------
    label : str
        User-facing label (e.g. ``"Jina-Text"``).
    query_cache_key, passage_cache_key : str
        Raw caches written for this variant.
    use_text : bool
        ``True`` = nl2code (docstring query, code passage, keyed by ``text_hash``);
        ``False`` = code2code (code query and passage, keyed by AST ``hash``).
    query_task, passage_task : str
        Jina tasks sent for the query and passage batches respectively.
    """

    label: str
    query_cache_key: str
    passage_cache_key: str
    use_text: bool
    query_task: str
    passage_task: str


def _find_uncached(
    baselines: Baselines,
    functions: list[FunctionInfo],
    cache_key: str,
    hash_attr: str,
) -> list[FunctionInfo]:
    """Return the subset of *functions* not yet present in the embedding cache.

    Parameters
    ----------
    baselines : dict
        Baselines dict holding the embedding caches.
    functions : list[FunctionInfo]
        Candidate functions to filter.
    cache_key : str
        Embedding cache to inspect.
    hash_attr : str
        Name of the ``FunctionInfo`` attribute used as the cache lookup key
        (``"hash"`` for AST caches, ``"text_hash"`` for text caches).

    Returns
    -------
    list[FunctionInfo]
        Functions whose embeddings are not yet cached.
    """
    return [
        f
        for f in functions
        if get_cached_embedding(baselines, getattr(f, hash_attr), cache_key) is None
    ]


def _populate_provider(
    baselines: Baselines,
    functions: list[FunctionInfo],
    cfg: _ProviderCfg,
) -> int:
    """Populate embeddings for a single provider using *cfg*.

    Parameters
    ----------
    baselines : dict
        Mutable baselines dict.
    functions : list[FunctionInfo]
        All extracted functions.
    cfg : _ProviderCfg
        Provider-specific parameters.

    Returns
    -------
    int
        Number of newly cached embeddings.
    """
    api_key = load_api_key_from_env(
        cfg.env_var,
        invalid_prefixes=cfg.invalid_prefixes,
        min_length=20,
    )
    if not api_key:
        _emit(f"{cfg.env_var} not set or invalid, skipping {cfg.label}")
        return 0

    hash_attr = "text_hash" if cfg.use_text else "hash"
    uncached = _find_uncached(baselines, functions, cfg.cache_key, hash_attr)
    if not uncached:
        _emit(f"{cfg.label}: all functions already cached")
        return 0

    _emit(f"{cfg.label}: fetching embeddings for {len(uncached)} functions...")
    try:
        text_attr = "text_for_embedding" if cfg.use_text else "ast_text"
        texts = [getattr(f, text_attr) for f in uncached]

        # For Gemini API, use smaller batches with delays due to free tier quota limits
        if cfg.provider == "gemini":
            embeddings: list[list[float]] = []
            # Gemini free tier: 100 requests/minute, use 50 to be safe
            batch_size = 50
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = _request_embeddings(batch_texts, provider=cfg.provider)
                embeddings.extend(batch_embeddings)
                if i + batch_size < len(texts):
                    # Only sleep between batches, never after the last one.
                    time.sleep(1.5)

        else:
            embeddings = _request_embeddings(texts, provider=cfg.provider)

        # strict=True is the whole safety story here: texts and embeddings are
        # built in the same order, so a short response would otherwise shift
        # every later vector onto the wrong function and quietly corrupt the
        # cache with mislabelled embeddings.
        new_embeddings = {
            getattr(func, hash_attr): emb for func, emb in zip(uncached, embeddings, strict=True)
        }
        merge_embedding_caches(baselines, {cfg.cache_key: new_embeddings})
        _emit(f"{cfg.label}: cached {len(uncached)} new embeddings")
        return len(uncached)
    except ModuleNotFoundError as e:
        _emit(
            f"{cfg.label}: skipping — '{e.name}' is not installed "
            "(install 'pykissembed[cloud]' to enable cloud population)",
        )
        return 0
    except _ProviderRequestError as e:
        # failures for one provider must not abort populating the rest.
        _emit(f"{cfg.label}: failed to fetch embeddings: {e}")
        return 0


def _jina_texts(uncached: list[FunctionInfo], cfg: _JinaCfg) -> tuple[list[str], list[str]]:
    """Return the (query_texts, passage_texts) inputs for *uncached* under *cfg*.

    Parameters
    ----------
    uncached : list[FunctionInfo]
        Functions still missing Jina vectors.
    cfg : _JinaCfg
        Resolved Jina configuration for this variant.

    Returns
    -------
    tuple[list[str], list[str]]
        Parallel query and passage input texts, one per uncached function.
    """
    if cfg.use_text:
        # nl2code: the docstring is the natural-language intent. Fall back to the
        # signature text when a function has no docstring so it still gets a
        # (weaker) query rather than dropping out of the matrix and Combined.
        query_texts = [func.docstring or func.text_for_embedding for func in uncached]
        passage_texts = [func.ast_text for func in uncached]
    else:
        # code2code: both query and passage are the function's code.
        query_texts = [func.ast_text for func in uncached]
        passage_texts = [func.ast_text for func in uncached]
    return query_texts, passage_texts


def _populate_jina(baselines: Baselines, functions: list[FunctionInfo], cfg: _JinaCfg) -> int:
    """Populate a Jina variant's query + passage caches (asymmetric retrieval).

    A function is considered uncached when *either* its query or passage vector
    is missing, since the symmetrized score needs both.

    Parameters
    ----------
    baselines : dict
        Mutable baselines dict.
    functions : list[FunctionInfo]
        All extracted functions.
    cfg : _JinaCfg
        Jina-variant parameters (cache keys, tasks, text vs AST mode).

    Returns
    -------
    int
        Number of functions newly embedded (query and passage together).
    """
    api_key = load_api_key_from_env("JINA_API_KEY", invalid_prefixes=("your_",), min_length=20)
    if not api_key:
        _emit(f"JINA_API_KEY not set or invalid, skipping {cfg.label}")
        return 0

    hash_attr = "text_hash" if cfg.use_text else "hash"
    query_cache = _get_embedding_cache(baselines, cfg.query_cache_key)
    passage_cache = _get_embedding_cache(baselines, cfg.passage_cache_key)
    # Either half missing disqualifies the function: retrieval symmetrizes the
    # two sides, so keeping a lone query or passage vector would contribute a
    # score that can never be reproduced on a later run.
    uncached = [
        func
        for func in functions
        if query_cache.get(getattr(func, hash_attr)) is None
        or passage_cache.get(getattr(func, hash_attr)) is None
    ]
    if not uncached:
        _emit(f"{cfg.label}: all functions already cached")
        return 0

    _emit(f"{cfg.label}: fetching embeddings for {len(uncached)} functions...")
    try:
        query_texts, passage_texts = _jina_texts(uncached, cfg)
        query_embs = _request_embeddings(query_texts, provider="jina", task=cfg.query_task)
        passage_embs = _request_embeddings(passage_texts, provider="jina", task=cfg.passage_task)
        query_updates: dict[str, list[float]] = {}
        passage_updates: dict[str, list[float]] = {}
        # Both halves are staged before either cache is merged, so a failure
        # part-way through cannot leave a function with a query but no passage.
        for func, query_emb, passage_emb in zip(uncached, query_embs, passage_embs, strict=True):
            key = getattr(func, hash_attr)
            query_updates[key] = query_emb
            passage_updates[key] = passage_emb
        merge_embedding_caches(
            baselines,
            {
                cfg.query_cache_key: query_updates,
                cfg.passage_cache_key: passage_updates,
            },
        )
        _emit(f"{cfg.label}: cached {len(uncached)} new embeddings")
        return len(uncached)
    except ModuleNotFoundError as e:
        _emit(
            f"{cfg.label}: skipping — '{e.name}' is not installed "
            "(install 'pykissembed[cloud]' to enable cloud population)",
        )
        return 0
    except _ProviderRequestError as e:
        # failures for one provider must not abort populating the rest.
        _emit(f"{cfg.label}: failed to fetch embeddings: {e}")
        return 0


# ---------------------------------------------------------------------------
# Per-provider configurations
#
# Each provider contributes two variants that differ only in *what text* is
# embedded, never in credentials: the ``-text`` variant sends
# ``text_for_embedding`` (the docstring-led text) and is keyed by
# ``text_hash``, while the ``-ast`` variant sends ``ast_text`` and is keyed by
# the AST ``hash``. The pair is what lets the similarity matrix tell a
# rename-only refactor apart from a behaviour change, so the two variants
# must never be pointed at the same source text or the same cache key.
#
# Several families share one credential (codestral and qwen both read
# OPENROUTER_API_KEY), which is why ``env_var`` is repeated per config rather
# than looked up from a single family table.
# ---------------------------------------------------------------------------

_OPENAI_TEXT_CFG = _ProviderCfg(
    label="OpenAI-Text",
    env_var="OPENAI_API_KEY",
    invalid_prefixes=("your_", "sk-xxx"),
    cache_key="openai_text_embeddings",
    use_text=True,
    provider="openai",
)

_OPENAI_AST_CFG = _ProviderCfg(
    label="OpenAI-AST",
    env_var="OPENAI_API_KEY",
    invalid_prefixes=("your_", "sk-xxx"),
    cache_key="openai_ast_embeddings",
    use_text=False,
    provider="openai",
)

_CODESTRAL_TEXT_CFG = _ProviderCfg(
    label="Codestral-Text",
    env_var="OPENROUTER_API_KEY",
    invalid_prefixes=("your_", "sk-xxx"),
    cache_key="codestral_text_embeddings",
    use_text=True,
    provider="codestral",
)

_CODESTRAL_AST_CFG = _ProviderCfg(
    label="Codestral-AST",
    env_var="OPENROUTER_API_KEY",
    invalid_prefixes=("your_", "sk-xxx"),
    cache_key="codestral_ast_embeddings",
    use_text=False,
    provider="codestral",
)

_VOYAGE_TEXT_CFG = _ProviderCfg(
    label="Voyage-Text",
    env_var="VOYAGE_API_KEY",
    invalid_prefixes=("your_", "pa-xxx"),
    cache_key="voyage_text_embeddings",
    use_text=True,
    provider="voyage",
)

_VOYAGE_AST_CFG = _ProviderCfg(
    label="Voyage-AST",
    env_var="VOYAGE_API_KEY",
    invalid_prefixes=("your_", "pa-xxx"),
    cache_key="voyage_ast_embeddings",
    use_text=False,
    provider="voyage",
)

_GEMINI_TEXT_CFG = _ProviderCfg(
    label="Gemini-Text",
    env_var="GOOGLE_API_KEY",
    invalid_prefixes=("your_",),
    cache_key="gemini_text_embeddings",
    use_text=True,
    provider="gemini",
)

_GEMINI_AST_CFG = _ProviderCfg(
    label="Gemini-AST",
    env_var="GOOGLE_API_KEY",
    invalid_prefixes=("your_",),
    cache_key="gemini_ast_embeddings",
    use_text=False,
    provider="gemini",
)

_QWEN_TEXT_CFG = _ProviderCfg(
    label="Qwen-Text",
    env_var="OPENROUTER_API_KEY",
    invalid_prefixes=("your_", "sk-xxx"),
    cache_key="qwen_text_embeddings",
    use_text=True,
    provider="qwen",
)

_QWEN_AST_CFG = _ProviderCfg(
    label="Qwen-AST",
    env_var="OPENROUTER_API_KEY",
    invalid_prefixes=("your_", "sk-xxx"),
    cache_key="qwen_ast_embeddings",
    use_text=False,
    provider="qwen",
)

_JINA_TEXT_CFG = _JinaCfg(
    label="Jina-Text",
    # nl2code: the query is natural language and the passage is the code, so
    # the two caches hold different vectors for the same function.
    query_cache_key="jina_text_query_embeddings",
    passage_cache_key="jina_text_passage_embeddings",
    use_text=True,
    query_task=JINA_NL2CODE_QUERY,
    passage_task=JINA_NL2CODE_PASSAGE,
)

_JINA_AST_CFG = _JinaCfg(
    label="Jina-AST",
    # code2code: query and passage are both the code, so the asymmetry comes
    # from the task codes rather than from different source text.
    query_cache_key="jina_ast_query_embeddings",
    passage_cache_key="jina_ast_passage_embeddings",
    use_text=False,
    query_task=JINA_CODE2CODE_QUERY,
    passage_task=JINA_CODE2CODE_PASSAGE,
)


def cli_provider_name(cache_key: str) -> str:
    """Map an embedding cache key to its ``populate-embeddings`` CLI provider name.

    Raw Jina cache keys (``…_query_embeddings`` / ``…_passage_embeddings``) fold
    back onto their populate provider ("jina-text" / "jina-ast"); cosine keys
    are unaffected since they carry no query/passage suffix.

    Parameters
    ----------
    cache_key : str
        Persisted cache key, possibly a Jina query/passage key.

    Returns
    -------
    str
        Provider name accepted by ``pykissembed populate-embeddings --provider``.
    """
    return (
        # The suffix order matters: a Jina query key must shed "_query" before
        # the generic "_embeddings" strip can see it, otherwise the result
        # would keep a "jina-text-query" stem that is not a valid provider.
        cache_key
        .removesuffix("_embeddings")
        .removesuffix("_query")
        .removesuffix("_passage")
        .replace("_", "-")
    )


def _missing_for_cache(
    baselines: Baselines,
    functions: list[FunctionInfo],
    cache_key: str,
) -> int:
    """Return how many live functions are absent from one cache.

    Inspection deliberately treats a missing or malformed cache as empty and
    never creates a replacement mapping in *baselines*.

    Parameters
    ----------
    baselines : Baselines
        Baselines mapping holding the provider cache.
    functions : list[FunctionInfo]
        Live functions that should be cached.
    cache_key : str
        Key naming the cache to inspect.

    Returns
    -------
    int
        Number of functions whose hash is absent from *cache_key*.
    """
    raw_cache = baselines.get(cache_key)
    cache = raw_cache if is_embedding_cache(raw_cache) else {}
    hash_field = REGISTRY.by_cache_key(cache_key).hash_field
    # A malformed cache counts as empty rather than raising, so --cached-only
    # can still report a gap instead of crashing on a half-written baseline.
    return sum(getattr(function, hash_field) not in cache for function in functions)


def _combined_member_gaps(
    baselines: Baselines,
    functions: list[FunctionInfo],
) -> dict[str, int]:
    """Return missing counts for providers required by Combined.

    Parameters
    ----------
    baselines : Baselines
        Baselines mapping holding the member caches.
    functions : list[FunctionInfo]
        Live functions that should be cached.

    Returns
    -------
    dict[str, int]
        Canonical CLI provider names mapped to their largest member-cache gap.
    """
    gaps: dict[str, int] = {}
    dependencies = REGISTRY.combined_dependencies + REGISTRY.standalone_dependencies
    for cache_key in dependencies:
        missing = _missing_for_cache(baselines, functions, cache_key)
        if missing:
            provider = cli_provider_name(cache_key)
            gaps[provider] = max(gaps.get(provider, 0), missing)
    return gaps


def _populate_combined(
    baselines: Baselines,
    functions: list[FunctionInfo],
) -> int:
    """Rebuild all Combined embeddings from the member caches.

    Parameters
    ----------
    baselines : Baselines
        Baselines mapping holding the member caches.
    functions : list[FunctionInfo]
        Live functions that should be cached.

    Returns
    -------
    int
        Number of Combined vectors available after rebuilding.
    """
    return _populate_combined_scoped(baselines, functions, replace_text_hashes=None)


def _populate_combined_scoped(
    baselines: Baselines,
    functions: list[FunctionInfo],
    *,
    replace_text_hashes: set[str] | None,
) -> int:
    """Rebuild Combined embeddings from the 10 cosine base providers + Jina.

    The rebuild derives its (text_hash, ast_hash) pairs from
    ``baselines["function_hashes"]``, which only the CLI refreshes — the pytest
    auto-populate flow never does. Record the live *functions* first so combined
    can be built even when ``function_hashes`` starts empty or stale (e.g. a
    consumer that has only ever populated embeddings through the test run).
    For an explicit partial-path scan, *replace_text_hashes* identifies the
    selected scope so unrelated Combined vectors survive the global rebuild.

    Parameters
    ----------
    baselines : Baselines
        Baselines mapping holding the member caches.
    functions : list[FunctionInfo]
        Live functions that should be cached.
    replace_text_hashes : set[str] | None
        Text hashes in the selected scope, or ``None`` to rebuild globally.

    Returns
    -------
    int
        Number of combined embeddings rebuilt.
    """
    previous_combined = (
        dict(_get_embedding_cache(baselines, REGISTRY.combined.cache_key))
        if replace_text_hashes is not None
        else None
    )
    _update_function_hashes(baselines, functions)
    member_gaps = _combined_member_gaps(baselines, functions)
    if member_gaps:
        details = ", ".join(f"{provider}: {missing}" for provider, missing in member_gaps.items())
        pytest.skip(
            "Cannot build combined embeddings until every member cache covers "
            f"the scanned functions ({details}). Run: "
            "pykissembed populate-embeddings --provider <name>",
        )

    _ = REGISTRY.rebuild_combined(baselines)
    combined = _get_embedding_cache(baselines, REGISTRY.combined.cache_key)
    if previous_combined is not None:
        replacement_hashes = replace_text_hashes or set()
        combined.update(
            {
                text_hash: vector
                for text_hash, vector in previous_combined.items()
                if text_hash not in replacement_hashes and text_hash not in combined
            },
        )
    return len(combined)


def _update_function_hashes(baselines: Baselines, functions: list[FunctionInfo]) -> None:
    """Insert current-function entries into ``baselines["function_hashes"]``.

    Each function is keyed by ``"{file}:{name}:{start_line}"`` and mapped to
    both its AST hash and text hash. Existing entries for other identities are
    left untouched.

    Parameters
    ----------
    baselines : dict
        Mutable baselines dict whose ``function_hashes`` entry is updated.
    functions : list[FunctionInfo]
        Functions to record.
    """
    function_hashes = _get_function_hashes(baselines)

    for func in functions:
        key = f"{func.file}:{func.name}:{func.start_line}"
        function_hashes[key] = _FunctionHashEntry(hash=func.hash, text_hash=func.text_hash)


def _synchronize_scanned_function_hashes(
    baselines: Baselines,
    functions: list[FunctionInfo],
    directories: list[_Path],
) -> None:
    """Replace hash entries only within the directories scanned by the CLI.

    Entries outside the selected roots are preserved. Within a selected root,
    stale line identities and legacy alternate path spellings are removed
    before current functions are inserted.

    Parameters
    ----------
    baselines : Baselines
        Baselines mapping holding ``function_hashes``.
    functions : list[FunctionInfo]
        Live functions found by the scan.
    directories : list[_Path]
        Roots the scan covered.
    """
    root = _get_config().root.resolve()
    scopes = collapse_scan_directories(directories)
    function_hashes = _get_function_hashes(baselines)
    current_keys = {f"{func.file}:{func.name}:{func.start_line}" for func in functions}
    for key in list(function_hashes):
        if _function_key_is_in_scopes(key, root, scopes) and key not in current_keys:
            del function_hashes[key]
    _update_function_hashes(baselines, functions)


def _function_key_is_in_scopes(key: str, root: _Path, scopes: list[_Path]) -> bool:
    """Return whether a stored function identity belongs to selected roots.

    Parameters
    ----------
    key : str
        Stored ``file:function:line`` identity to locate; a key that does not split into three parts
        is never in scope.
    root : _Path
        Project root the key is relative to.
    scopes : list[_Path]
        Roots selected by this scan.

    Returns
    -------
    bool
        Whether the function file is under any selected root.
    """
    try:
        file_name, _function_name, _line = key.rsplit(":", 2)
    except ValueError:
        return False
    file_path = _Path(file_name)
    absolute_path = file_path.resolve() if file_path.is_absolute() else (root / file_path).resolve()
    return any(absolute_path.is_relative_to(scope) for scope in scopes)


def _scoped_text_hashes(baselines: Baselines, directories: list[_Path]) -> set[str]:
    """Return scoped text hashes that no unscanned identity still references.

    Parameters
    ----------
    baselines : Baselines
        Baselines mapping holding ``function_hashes``.
    directories : list[_Path]
        Roots selected by this scan.

    Returns
    -------
    set[str]
        Text hashes exclusive to the selected directories.
    """
    root = _get_config().root.resolve()
    scopes = collapse_scan_directories(directories)
    scoped: set[str] = set()
    unscoped: set[str] = set()
    for key, entry in _get_function_hashes(baselines).items():
        if not isinstance(entry, dict):
            continue
        text_hash = entry.get("text_hash")
        if isinstance(text_hash, str) and text_hash:
            destination = scoped if _function_key_is_in_scopes(key, root, scopes) else unscoped
            destination.add(text_hash)
    return scoped - unscoped


# Map provider names to functions.
#
# The lambdas exist so every entry has the same ``(baselines, functions)``
# signature and the Jina variants, which need a _JinaCfg, slot in beside the
# plain providers without a wrapper function each. "combined" is listed last
# and maps straight to its function: it derives every vector from the member
# caches, so it must never run before they are populated.
_PROVIDER_MAP: dict[str, PopulateFn] = {
    "openai-text": lambda b, f: _populate_provider(b, f, _OPENAI_TEXT_CFG),
    "openai-ast": lambda b, f: _populate_provider(b, f, _OPENAI_AST_CFG),
    "codestral-text": lambda b, f: _populate_provider(b, f, _CODESTRAL_TEXT_CFG),
    "codestral-ast": lambda b, f: _populate_provider(b, f, _CODESTRAL_AST_CFG),
    "voyage-text": lambda b, f: _populate_provider(b, f, _VOYAGE_TEXT_CFG),
    "voyage-ast": lambda b, f: _populate_provider(b, f, _VOYAGE_AST_CFG),
    "gemini-text": lambda b, f: _populate_provider(b, f, _GEMINI_TEXT_CFG),
    "gemini-ast": lambda b, f: _populate_provider(b, f, _GEMINI_AST_CFG),
    "qwen-text": lambda b, f: _populate_provider(b, f, _QWEN_TEXT_CFG),
    "qwen-ast": lambda b, f: _populate_provider(b, f, _QWEN_AST_CFG),
    "jina-text": lambda b, f: _populate_jina(b, f, _JINA_TEXT_CFG),
    "jina-ast": lambda b, f: _populate_jina(b, f, _JINA_AST_CFG),
    "combined": _populate_combined,
}


def get_provider_populator(provider: str) -> PopulateFn | None:
    """Return provider populate function by canonical provider key.

    Parameters
    ----------
    provider : str
        Canonical provider name to look up.

    Returns
    -------
    PopulateFn | None
        The provider populate function, or ``None`` if provider is unknown.
    """
    return _PROVIDER_MAP.get(provider)


_NETWORK_PROVIDERS = (
    "openai-text",
    "openai-ast",
    "codestral-text",
    "codestral-ast",
    "voyage-text",
    "voyage-ast",
    "gemini-text",
    "gemini-ast",
    "qwen-text",
    "qwen-ast",
    "jina-text",
    "jina-ast",
)

# "combined" is excluded here and appended instead, so the loop that walks
# these providers can never attempt a network call for it.
_ALL_PROVIDERS = (
    *_NETWORK_PROVIDERS,
    "combined",
)

# Credential table used only for pre-flight checks and error messages, not for
# fetching. Fetching reads the same names through _ProviderCfg.env_var, so a
# name appearing in both places must stay in sync. "codestral" and "qwen" are
# separate families that happen to share OPENROUTER_API_KEY.
_PROVIDER_CREDENTIALS = {
    "openai": ("OPENAI_API_KEY", ("your_", "sk-xxx")),
    "codestral": ("OPENROUTER_API_KEY", ("your_", "sk-xxx")),
    "voyage": ("VOYAGE_API_KEY", ("your_", "pa-xxx")),
    "gemini": ("GOOGLE_API_KEY", ("your_",)),
    "qwen": ("OPENROUTER_API_KEY", ("your_", "sk-xxx")),
    "jina": ("JINA_API_KEY", ("your_",)),
}


def _require_canonical_provider(provider: str) -> None:
    """Validate *provider* and raise an actionable error for legacy names.

    Parameters
    ----------
    provider : str
        Provider name to validate.

    Raises
    ------
    PopulationError
        If *provider* is local, ambiguous, or unknown.
    """
    if provider in {*_ALL_PROVIDERS, "all"}:
        return
    if provider == "local":
        msg = (
            "Local embeddings were removed. Choose a cloud variant such as "
            "'openai-text', 'openai-ast', 'gemini-text', or 'gemini-ast'. "
            "Existing local JSONL caches are left untouched."
        )
        raise PopulationError(msg)
    family = provider.partition("-")[0]
    if provider == family and family in _PROVIDER_CREDENTIALS:
        env_var, _ = _PROVIDER_CREDENTIALS[family]
        text_variant = f"{family}-text"
        ast_variant = f"{family}-ast"
        msg = (
            f"Provider {provider!r} is ambiguous. Choose {text_variant!r} or "
            f"{ast_variant!r}; both use {env_var}."
        )
        raise PopulationError(msg)
    choices = ", ".join((*_ALL_PROVIDERS, "all"))
    msg = f"Unknown provider {provider!r}. Canonical choices: {choices}."
    raise PopulationError(msg)


def _provider_cache_keys(provider: str) -> tuple[str, ...]:
    """Return the persisted cache keys inspected for a canonical provider.

    Parameters
    ----------
    provider : str
        Canonical provider name to resolve keys for.

    Returns
    -------
    tuple[str, ...]
        One cache key for cosine/Combined providers or the query and passage
        keys for a Jina variant.
    """
    stem = provider.replace("-", "_")
    # Jina is the one asymmetric provider: it persists two caches under one
    # name, so it needs a dedicated branch rather than the single-key default.
    if provider.startswith("jina-"):
        return (f"{stem}_query_embeddings", f"{stem}_passage_embeddings")
    return (f"{stem}_embeddings",)


def _missing_for_provider(
    baselines: Baselines,
    functions: list[FunctionInfo],
    provider: str,
) -> int:
    """Return functions missing any cache member required by *provider*.

    Parameters
    ----------
    baselines : Baselines
        Baselines mapping holding the provider caches.
    functions : list[FunctionInfo]
        Live functions that should be cached.
    provider : str
        Canonical provider name whose members are required.

    Returns
    -------
    int
        Number of incomplete functions.
    """
    caches: list[tuple[dict[str, list[float]], str]] = []
    for cache_key in _provider_cache_keys(provider):
        raw_cache = baselines.get(cache_key)
        cache = raw_cache if is_embedding_cache(raw_cache) else {}
        caches.append((cache, REGISTRY.by_cache_key(cache_key).hash_field))
    return sum(
        any(getattr(function, hash_field) not in cache for cache, hash_field in caches)
        for function in functions
    )


def _configured_credential(provider: str) -> str | None:
    """Return the environment variable name when *provider* lacks credentials.

    Parameters
    ----------
    provider : str
        Canonical provider name to check credentials for.

    Returns
    -------
    str | None
        Missing/invalid environment-variable name, or ``None`` when configured.
    """
    family = provider.partition("-")[0]
    env_var, invalid_prefixes = _PROVIDER_CREDENTIALS[family]
    api_key = load_api_key_from_env(
        env_var,
        invalid_prefixes=invalid_prefixes,
        min_length=20,
    )
    return None if api_key else env_var


def _attempt_network_provider(
    provider: str,
    baselines: Baselines,
    functions: list[FunctionInfo],
) -> tuple[int, str | None]:
    """Attempt one cloud provider and report any unresolved cache gap.

    Parameters
    ----------
    provider : str
        Canonical provider name to populate.
    baselines : Baselines
        Baselines mapping to populate in place.
    functions : list[FunctionInfo]
        Live functions that should be cached.

    Returns
    -------
    tuple[int, str | None]
        Newly embedded function count and an error message, if incomplete.
    """
    missing_before = _missing_for_provider(baselines, functions, provider)
    if not missing_before:
        _emit(f"{provider}: all scanned functions already cached")
        return 0, None
    missing_credential = _configured_credential(provider)
    if missing_credential:
        return 0, f"{provider}: {missing_credential} is not configured ({missing_before} missing)"
    handler = _PROVIDER_MAP[provider]
    new_count = handler(baselines, functions)
    # Re-measure instead of trusting new_count: the handler reports what it
    # embedded, which is not the same as what is still missing afterwards.
    missing_after = _missing_for_provider(baselines, functions, provider)
    if missing_after:
        return new_count, f"{provider}: cache remains incomplete ({missing_after} missing)"
    return new_count, None


def _inspect_caches(
    provider: str,
    baselines: Baselines,
    functions: list[FunctionInfo],
) -> None:
    """Print cache gaps without mutating or persisting *baselines*.

    Parameters
    ----------
    provider : str
        Canonical provider name, or ``"all"``.
    baselines : Baselines
        Baselines mapping to inspect without mutating.
    functions : list[FunctionInfo]
        Live functions to count against.
    """
    selected = _ALL_PROVIDERS if provider == "all" else (provider,)
    _emit("--cached-only: inspection only; no API calls or cache writes")
    for name in selected:
        missing = _missing_for_provider(baselines, functions, name)
        _emit(
            f"{name}: {missing} of {len(functions)} scanned functions missing",
        )
        if name == "combined":
            for member, count in _combined_member_gaps(baselines, functions).items():
                _emit(f"  member {member}: {count} missing")


def _populate_all(
    baselines: Baselines,
    functions: list[FunctionInfo],
    *,
    replace_combined_hashes: set[str] | None = None,
) -> tuple[int, bool]:
    """Populate every available cloud provider and rebuild Combined.

    Parameters
    ----------
    baselines : dict
        Mutable baselines dict.
    functions : list[FunctionInfo]
        All extracted functions.
    replace_combined_hashes : set[str] | None
        Text hashes whose Combined vectors may be replaced during the
        rebuild, or ``None`` for a full rebuild scope.

    Returns
    -------
    tuple[int, bool]
        Total number of newly embedded functions across providers (including
        the Combined rebuild) and whether any requested work completed.

    Raises
    ------
    PopulationError
        If caches are incomplete and no requested work could be completed.
    """
    total_new = 0
    performed = False
    unresolved: list[str] = []
    for provider in _NETWORK_PROVIDERS:
        new_count, error = _attempt_network_provider(provider, baselines, functions)
        total_new += new_count
        performed = performed or new_count > 0
        if error:
            unresolved.append(error)
            _emit(f"Skipping {error}")

    member_gaps = _combined_member_gaps(baselines, functions)
    if member_gaps:
        details = ", ".join(f"{name}: {count}" for name, count in member_gaps.items())
        unresolved.append(f"combined: member caches incomplete ({details})")
    else:
        total_new += _populate_combined_scoped(
            baselines,
            functions,
            replace_text_hashes=replace_combined_hashes,
        )
        performed = True

    any_missing = any(
        _missing_for_provider(baselines, functions, provider) for provider in _ALL_PROVIDERS
    )
    if unresolved and any_missing and not performed:
        # Only fatal when nothing at all succeeded. A partial run still leaves
        # usable caches behind, so the missing pieces are reported rather than
        # thrown away along with the vectors that were just paid for.
        raise PopulationError(
            "No requested cache work could be completed. " + "; ".join(unresolved),
        )
    return total_new, performed


def _resolve_scan_directories(paths: list[_Path] | None) -> list[_Path] | None:
    """Resolve and validate explicit scan directories.

    Parameters
    ----------
    paths : list[_Path] | None
        Explicit scan roots, or ``None`` to fall back to configuration.

    Returns
    -------
    list[Path] | None
        Deduplicated absolute directories, or ``None`` to use configuration.

    Raises
    ------
    PopulationError
        If an explicit path is missing or not a directory.
    """
    if paths is None:
        return None
    resolved = [path.resolve() for path in paths]
    invalid = [path for path in resolved if not path.is_dir()]
    if invalid:
        raise PopulationError(
            "Embedding scan paths must be existing directories: "
            + ", ".join(str(path) for path in invalid),
        )
    return collapse_scan_directories(resolved)


def populate_provider_embeddings(
    provider: str = "all",
    *,
    paths: list[_Path] | None = None,
    cached_only: bool = False,
) -> None:
    """Populate embedding caches for specified provider(s).

    Parameters
    ----------
    provider : str
        One of the 13 providers or ``"all"``.
    paths : list[Path] | None
        Explicit directories to scan instead of configured source paths.
    cached_only : bool
        Inspect cache coverage without API calls or writes.

    Raises
    ------
    PopulationError
        If the provider or paths are invalid, credentials are unavailable for
        a selected incomplete provider, or population leaves it incomplete.
    """
    _require_canonical_provider(provider)
    directories = _resolve_scan_directories(paths)
    _emit("Loading baselines and extracting functions...")
    baselines = load_baselines()
    functions = (
        extract_all_function_infos(min_loc=1)
        if directories is None
        else extract_function_infos_from_directories(directories, min_loc=1)
    )
    _emit(f"Found {len(functions)} functions in codebase")

    if cached_only:
        _inspect_caches(provider, baselines, functions)
        return

    hashes_before = dict(_get_function_hashes(baselines))
    # Resolve ownership before synchronizing identities: once stale scoped
    # entries are removed, shared text hashes cannot be distinguished safely.
    replace_combined_hashes = (
        _scoped_text_hashes(baselines, directories) if directories is not None else None
    )
    if replace_combined_hashes is not None:
        # The scan found the authoritative current set, so anything sharing a
        # text hash in scope must be rebuilt even if an identical hash already
        # has a Combined vector from a different identity.
        replace_combined_hashes.update(function.text_hash for function in functions)
    _synchronize_scanned_function_hashes(
        baselines,
        functions,
        directories if directories is not None else _resolve_paths(),
    )
    hashes_changed = hashes_before != _get_function_hashes(baselines)
    if provider == "all":
        total_new, performed = _populate_all(
            baselines,
            functions,
            replace_combined_hashes=replace_combined_hashes,
        )
    elif provider == "combined":
        member_gaps = _combined_member_gaps(baselines, functions)
        if member_gaps:
            details = ", ".join(f"{name}: {count}" for name, count in member_gaps.items())
            msg = f"Cannot rebuild combined; member caches are incomplete: {details}"
            raise PopulationError(msg)
        total_new = _populate_combined_scoped(
            baselines,
            functions,
            replace_text_hashes=replace_combined_hashes,
        )
        performed = True
    else:
        total_new, error = _attempt_network_provider(provider, baselines, functions)
        if error:
            raise PopulationError(error)
        performed = total_new > 0

    if total_new > 0 or hashes_changed or performed:
        _emit(
            f"\nSaving cache state ({total_new} provider result(s))...",
        )
        save_baselines(baselines)
        _emit("Done!")
    else:
        _emit("\nNo cache changes to save.")


def populate_embeddings(provider: str = "all") -> None:
    """Populate embedding caches for one canonical provider or all providers.

    This compatibility wrapper retains the original Python API. Use the public
    CLI to select explicit scan paths or inspect caches without network calls.

    Parameters
    ----------
    provider : str
        Canonical provider name, or ``"all"`` for every provider.
    """
    populate_provider_embeddings(provider)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Populate embedding caches for similarity tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _ = parser.add_argument(
        "--provider",
        required=True,
        help="Canonical provider variant to populate, or 'all'",
    )
    _ = parser.add_argument(
        "--path",
        action="append",
        type=_Path,
        dest="paths",
        help="Directory to scan; repeat for multiple paths",
    )
    _ = parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Inspect cache coverage without API calls or writes",
    )
    args = parser.parse_args()

    try:
        populate_provider_embeddings(args.provider, paths=args.paths, cached_only=args.cached_only)
    except PopulationError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
