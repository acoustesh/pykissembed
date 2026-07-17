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
    _collapse_scan_directories,
    _extract_function_infos_from_directories,
    extract_all_function_infos,
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


class _PopulationError(RuntimeError):
    """Raised when an explicit cache-population request cannot be completed."""


class _FunctionHashEntry(TypedDict):
    """Represents a function hash entry."""

    hash: str
    text_hash: str


def _get_embedding_cache(baselines: Baselines, cache_key: str) -> dict[str, list[float]]:
    """Return the live provider cache for *cache_key*, creating it when absent.

    A missing entry is initialised to an empty mapping, stored back into
    *baselines*, and returned, so later in-place writes by the caller persist.
    This deliberately mutates *baselines* and hands back the very object it
    holds rather than a defensive copy.

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
    """Get the function hashes.

    Returns
    -------
    dict[str, object]
        The function hashes.

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
        print(f"{cfg.env_var} not set or invalid, skipping {cfg.label}")  # ruff:ignore[print]
        return 0

    hash_attr = "text_hash" if cfg.use_text else "hash"
    uncached = _find_uncached(baselines, functions, cfg.cache_key, hash_attr)
    if not uncached:
        print(f"{cfg.label}: all functions already cached")  # ruff:ignore[print]
        return 0

    print(f"{cfg.label}: fetching embeddings for {len(uncached)} functions...")  # ruff:ignore[print]
    try:
        text_attr = "text_for_embedding" if cfg.use_text else "ast_text"
        texts = [getattr(f, text_attr) for f in uncached]

        # For Gemini API, use smaller batches with delays due to free tier quota limits
        if cfg.provider == "gemini":
            embeddings: list[list[float]] = []
            batch_size = 50  # Gemini free tier: 100 requests/minute, use 50 to be safe
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = get_embeddings_batch(batch_texts, provider=cfg.provider)
                embeddings.extend(batch_embeddings)
                # Add delay between batches to respect rate limits
                if i + batch_size < len(texts):
                    time.sleep(1.5)  # 1.5 second delay between batches
        else:
            embeddings = get_embeddings_batch(texts, provider=cfg.provider)

        new_embeddings = {
            getattr(func, hash_attr): emb for func, emb in zip(uncached, embeddings, strict=True)
        }
        merge_embedding_caches(baselines, {cfg.cache_key: new_embeddings})
        print(f"{cfg.label}: cached {len(uncached)} new embeddings")  # ruff:ignore[print]
        return len(uncached)
    except ModuleNotFoundError as e:
        print(  # ruff:ignore[print]
            f"{cfg.label}: skipping — '{e.name}' is not installed "
            "(install 'pykissembed[cloud]' to enable cloud population)",
        )
        return 0
    except Exception as e:  # ruff:ignore[blind-except] — external API boundary: network/auth/rate-limit
        # failures for one provider must not abort populating the rest.
        print(f"{cfg.label}: failed to fetch embeddings: {e}")  # ruff:ignore[print]
        return 0


def _jina_texts(uncached: list[FunctionInfo], cfg: _JinaCfg) -> tuple[list[str], list[str]]:
    """Return the (query_texts, passage_texts) inputs for *uncached* under *cfg*.

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

    Returns
    -------
    int
        Number of functions newly embedded (query and passage together).
    """
    api_key = load_api_key_from_env("JINA_API_KEY", invalid_prefixes=("your_",), min_length=20)
    if not api_key:
        print(f"JINA_API_KEY not set or invalid, skipping {cfg.label}")  # ruff:ignore[print]
        return 0

    hash_attr = "text_hash" if cfg.use_text else "hash"
    query_cache = _get_embedding_cache(baselines, cfg.query_cache_key)
    passage_cache = _get_embedding_cache(baselines, cfg.passage_cache_key)
    uncached = [
        func
        for func in functions
        if query_cache.get(getattr(func, hash_attr)) is None
        or passage_cache.get(getattr(func, hash_attr)) is None
    ]
    if not uncached:
        print(f"{cfg.label}: all functions already cached")  # ruff:ignore[print]
        return 0

    print(f"{cfg.label}: fetching embeddings for {len(uncached)} functions...")  # ruff:ignore[print]
    try:
        query_texts, passage_texts = _jina_texts(uncached, cfg)
        query_embs = get_embeddings_batch(query_texts, provider="jina", task=cfg.query_task)
        passage_embs = get_embeddings_batch(passage_texts, provider="jina", task=cfg.passage_task)
        query_updates: dict[str, list[float]] = {}
        passage_updates: dict[str, list[float]] = {}
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
        print(f"{cfg.label}: cached {len(uncached)} new embeddings")  # ruff:ignore[print]
        return len(uncached)
    except ModuleNotFoundError as e:
        print(  # ruff:ignore[print]
            f"{cfg.label}: skipping — '{e.name}' is not installed "
            "(install 'pykissembed[cloud]' to enable cloud population)",
        )
        return 0
    except Exception as e:  # ruff:ignore[blind-except] — external API boundary: network/auth/rate-limit
        # failures for one provider must not abort populating the rest.
        print(f"{cfg.label}: failed to fetch embeddings: {e}")  # ruff:ignore[print]
        return 0


# ---------------------------------------------------------------------------
# Per-provider configurations
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
    query_cache_key="jina_text_query_embeddings",
    passage_cache_key="jina_text_passage_embeddings",
    use_text=True,
    query_task=JINA_NL2CODE_QUERY,
    passage_task=JINA_NL2CODE_PASSAGE,
)

_JINA_AST_CFG = _JinaCfg(
    label="Jina-AST",
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

    Returns
    -------
    str
        Provider name accepted by ``pykissembed populate-embeddings --provider``.
    """
    return (
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

    Returns
    -------
    int
        Number of functions whose hash is absent from *cache_key*.
    """
    raw_cache = baselines.get(cache_key)
    cache = raw_cache if is_embedding_cache(raw_cache) else {}
    hash_field = REGISTRY.by_cache_key(cache_key).hash_field
    return sum(getattr(function, hash_field) not in cache for function in functions)


def _combined_member_gaps(
    baselines: Baselines,
    functions: list[FunctionInfo],
) -> dict[str, int]:
    """Return missing counts for providers required by Combined.

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

    REGISTRY.rebuild_combined(baselines)
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
    """Update function_hashes with the new format containing both hash and text_hash."""
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
    """
    root = _get_config().root.resolve()
    scopes = _collapse_scan_directories(directories)
    function_hashes = _get_function_hashes(baselines)
    current_keys = {f"{func.file}:{func.name}:{func.start_line}" for func in functions}
    for key in list(function_hashes):
        if _function_key_is_in_scopes(key, root, scopes) and key not in current_keys:
            del function_hashes[key]
    _update_function_hashes(baselines, functions)


def _function_key_is_in_scopes(key: str, root: _Path, scopes: list[_Path]) -> bool:
    """Return whether a stored function identity belongs to selected roots.

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

    Returns
    -------
    set[str]
        Text hashes exclusive to the selected directories.
    """
    root = _get_config().root.resolve()
    scopes = _collapse_scan_directories(directories)
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


# Map provider names to functions
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

_ALL_PROVIDERS = (
    *_NETWORK_PROVIDERS,
    "combined",
)

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

    Raises
    ------
    _PopulationError
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
        raise _PopulationError(msg)
    family = provider.partition("-")[0]
    if provider == family and family in _PROVIDER_CREDENTIALS:
        env_var, _ = _PROVIDER_CREDENTIALS[family]
        text_variant = f"{family}-text"
        ast_variant = f"{family}-ast"
        msg = (
            f"Provider {provider!r} is ambiguous. Choose {text_variant!r} or "
            f"{ast_variant!r}; both use {env_var}."
        )
        raise _PopulationError(msg)
    choices = ", ".join((*_ALL_PROVIDERS, "all"))
    msg = f"Unknown provider {provider!r}. Canonical choices: {choices}."
    raise _PopulationError(msg)


def _provider_cache_keys(provider: str) -> tuple[str, ...]:
    """Return the persisted cache keys inspected for a canonical provider.

    Returns
    -------
    tuple[str, ...]
        One cache key for cosine/Combined providers or the query and passage
        keys for a Jina variant.
    """
    stem = provider.replace("-", "_")
    if provider.startswith("jina-"):
        return (f"{stem}_query_embeddings", f"{stem}_passage_embeddings")
    return (f"{stem}_embeddings",)


def _missing_for_provider(
    baselines: Baselines,
    functions: list[FunctionInfo],
    provider: str,
) -> int:
    """Return functions missing any cache member required by *provider*.

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

    Returns
    -------
    tuple[int, str | None]
        Newly embedded function count and an error message, if incomplete.
    """
    missing_before = _missing_for_provider(baselines, functions, provider)
    if not missing_before:
        print(f"{provider}: all scanned functions already cached")  # ruff:ignore[print]
        return 0, None
    missing_credential = _configured_credential(provider)
    if missing_credential:
        return 0, f"{provider}: {missing_credential} is not configured ({missing_before} missing)"
    handler = _PROVIDER_MAP[provider]
    new_count = handler(baselines, functions)
    missing_after = _missing_for_provider(baselines, functions, provider)
    if missing_after:
        return new_count, f"{provider}: cache remains incomplete ({missing_after} missing)"
    return new_count, None


def _inspect_caches(
    provider: str,
    baselines: Baselines,
    functions: list[FunctionInfo],
) -> None:
    """Print cache gaps without mutating or persisting *baselines*."""
    selected = _ALL_PROVIDERS if provider == "all" else (provider,)
    print("--cached-only: inspection only; no API calls or cache writes")  # ruff:ignore[print]
    for name in selected:
        missing = _missing_for_provider(baselines, functions, name)
        print(  # ruff:ignore[print]
            f"{name}: {missing} of {len(functions)} scanned functions missing",
        )
        if name == "combined":
            for member, count in _combined_member_gaps(baselines, functions).items():
                print(f"  member {member}: {count} missing")  # ruff:ignore[print]


def _populate_all(
    baselines: Baselines,
    functions: list[FunctionInfo],
    *,
    replace_combined_hashes: set[str] | None = None,
) -> tuple[int, bool]:
    """Populate every available cloud provider and rebuild Combined.

    Returns
    -------
    tuple[int, bool]
        Total provider result count and whether any requested work completed.

    Raises
    ------
    _PopulationError
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
            print(f"Skipping {error}")  # ruff:ignore[print]

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
        raise _PopulationError(
            "No requested cache work could be completed. " + "; ".join(unresolved),
        )
    return total_new, performed


def _resolve_scan_directories(paths: list[_Path] | None) -> list[_Path] | None:
    """Resolve and validate explicit scan directories.

    Returns
    -------
    list[Path] | None
        Deduplicated absolute directories, or ``None`` to use configuration.

    Raises
    ------
    _PopulationError
        If an explicit path is missing or not a directory.
    """
    if paths is None:
        return None
    resolved = [path.resolve() for path in paths]
    invalid = [path for path in resolved if not path.is_dir()]
    if invalid:
        raise _PopulationError(
            "Embedding scan paths must be existing directories: "
            + ", ".join(str(path) for path in invalid),
        )
    return _collapse_scan_directories(resolved)


def _populate_embeddings(
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
    _PopulationError
        If the provider or paths are invalid, credentials are unavailable for
        a selected incomplete provider, or population leaves it incomplete.
    """
    _require_canonical_provider(provider)
    directories = _resolve_scan_directories(paths)
    print("Loading baselines and extracting functions...")  # ruff:ignore[print]
    baselines = load_baselines()
    functions = (
        extract_all_function_infos(min_loc=1)
        if directories is None
        else _extract_function_infos_from_directories(directories, min_loc=1)
    )
    print(f"Found {len(functions)} functions in codebase")  # ruff:ignore[print]

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
            raise _PopulationError(msg)
        total_new = _populate_combined_scoped(
            baselines,
            functions,
            replace_text_hashes=replace_combined_hashes,
        )
        performed = True
    else:
        total_new, error = _attempt_network_provider(provider, baselines, functions)
        if error:
            raise _PopulationError(error)
        performed = total_new > 0

    if total_new > 0 or hashes_changed or performed:
        print(  # ruff:ignore[print]
            f"\nSaving cache state ({total_new} provider result(s))...",
        )
        save_baselines(baselines)
        print("Done!")  # ruff:ignore[print]
    else:
        print("\nNo cache changes to save.")  # ruff:ignore[print]


def populate_embeddings(provider: str = "all") -> None:
    """Populate embedding caches for one canonical provider or all providers.

    This compatibility wrapper retains the original Python API. Use the public
    CLI to select explicit scan paths or inspect caches without network calls.
    """
    _populate_embeddings(provider)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Populate embedding caches for similarity tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="Canonical provider variant to populate, or 'all'",
    )
    parser.add_argument(
        "--path",
        action="append",
        type=_Path,
        dest="paths",
        help="Directory to scan; repeat for multiple paths",
    )
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Inspect cache coverage without API calls or writes",
    )
    args = parser.parse_args()

    try:
        _populate_embeddings(args.provider, paths=args.paths, cached_only=args.cached_only)
    except _PopulationError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
