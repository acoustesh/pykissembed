"""Populate embedding caches for all providers.

Ported from ``mega-scrapper/tests/similarity/populate_embeddings.py``.
This module fetches embeddings from APIs for functions that don't have
cached values. Run this before running tests if you need to populate
missing embeddings.

Usage:
    python -m pykissembed.similarity.populate_embeddings [--provider PROVIDER]

Options:
    --provider  One of: openai-text, openai-ast, codestral-text, codestral-ast,
                voyage-text, voyage-ast, gemini-text, gemini-ast, combined, all
                (default: all)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, TypeGuard, cast

import pytest

from pykissembed.similarity.ast_helpers import extract_all_function_infos
from pykissembed.similarity.embeddings import (
    compute_combined_embedding,
    get_cached_embedding,
    get_embeddings_batch,
    is_float_embedding,
    is_str_object_dict,
    load_api_key_from_env,
)
from pykissembed.similarity.storage import load_baselines, save_baselines

if TYPE_CHECKING:
    from collections.abc import Callable

    from pykissembed.similarity.types import FunctionInfo

type Baselines = dict[str, object]
type PopulateFn = Callable[[Baselines, list[FunctionInfo]], int]


class _FunctionHashEntry(TypedDict):
    """Represents a function hash entry."""

    hash: str
    text_hash: str


def _is_embedding_cache(value: object) -> TypeGuard[dict[str, list[float]]]:
    """Check if a dictionary is an embedding cache.

    Returns
    -------
    bool
        True if the dictionary is an embedding cache.
    """
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(key, str) and is_float_embedding(embedding)
        for key, embedding in cast("dict[object, object]", value).items()
    )


def _get_embedding_cache(baselines: Baselines, cache_key: str) -> dict[str, list[float]]:
    """Get the embedding cache.

    Returns
    -------
    dict[str, list[float]]
        The embedding cache.

    Raises
    ------
    TypeError
        If ``baselines[cache_key]`` exists but is not ``dict[str, list[float]]``.
    """
    cache_obj = baselines.get(cache_key)
    if cache_obj is None:
        empty_cache: dict[str, list[float]] = {}
        baselines[cache_key] = empty_cache
        return empty_cache
    if not _is_embedding_cache(cache_obj):
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
        print(f"{cfg.env_var} not set or invalid, skipping {cfg.label}")  # noqa: T201
        return 0

    hash_attr = "text_hash" if cfg.use_text else "hash"
    uncached = _find_uncached(baselines, functions, cfg.cache_key, hash_attr)
    if not uncached:
        print(f"{cfg.label}: all functions already cached")  # noqa: T201
        return 0

    print(f"{cfg.label}: fetching embeddings for {len(uncached)} functions...")  # noqa: T201
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

        cache = _get_embedding_cache(baselines, cfg.cache_key)
        for func, emb in zip(uncached, embeddings, strict=True):
            cache[getattr(func, hash_attr)] = emb
        print(f"{cfg.label}: cached {len(uncached)} new embeddings")  # noqa: T201
        return len(uncached)
    except ModuleNotFoundError as e:
        # The provider's SDK is an optional dependency not bundled with any
        # pykissembed extra (see pyproject.toml dev-group comments) — spell
        # that out instead of the generic "failed to fetch" message, which
        # reads like an API failure rather than a missing package.
        print(  # noqa: T201
            f"{cfg.label}: skipping — '{e.name}' is not installed "
            f"(pip install {e.name} to enable this provider)",
        )
        return 0
    except Exception as e:  # noqa: BLE001 — external API boundary: network/auth/rate-limit
        # failures for one provider must not abort populating the rest.
        print(f"{cfg.label}: failed to fetch embeddings: {e}")  # noqa: T201
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


def _populate_combined(
    baselines: Baselines,
    functions: list[FunctionInfo],
) -> int:
    """Populate Combined embeddings from all 8 base providers.

    Returns
    -------
    int
        Number of newly computed combined embeddings.
    """
    uncached = _find_uncached(baselines, functions, "combined_embeddings", "text_hash")
    if not uncached:
        return 0

    # All 8 base provider caches must be present and valid to build combined.
    required_caches = [
        "openai_text_embeddings",
        "openai_ast_embeddings",
        "codestral_text_embeddings",
        "codestral_ast_embeddings",
        "voyage_text_embeddings",
        "voyage_ast_embeddings",
        "gemini_text_embeddings",
        "gemini_ast_embeddings",
    ]
    for cache_key in required_caches:
        raw = baselines.get(cache_key)
        if raw is None or not raw or not _is_embedding_cache(raw):
            provider = cache_key.removesuffix("_embeddings").replace("_", "-")
            pytest.skip(
                f"Cannot build combined embeddings: {cache_key} is missing or invalid. "
                f"Run: pykissembed populate-embeddings --provider {provider}",
            )

    openai_text_cache = _get_embedding_cache(baselines, "openai_text_embeddings")
    codestral_text_cache = _get_embedding_cache(baselines, "codestral_text_embeddings")
    voyage_text_cache = _get_embedding_cache(baselines, "voyage_text_embeddings")
    gemini_text_cache = _get_embedding_cache(baselines, "gemini_text_embeddings")
    openai_ast_cache = _get_embedding_cache(baselines, "openai_ast_embeddings")
    codestral_ast_cache = _get_embedding_cache(baselines, "codestral_ast_embeddings")
    voyage_ast_cache = _get_embedding_cache(baselines, "voyage_ast_embeddings")
    gemini_ast_cache = _get_embedding_cache(baselines, "gemini_ast_embeddings")
    combined_cache = _get_embedding_cache(baselines, "combined_embeddings")

    computed = 0
    for func in uncached:
        # Text variants keyed by text_hash
        openai_text = openai_text_cache.get(func.text_hash)
        codestral_text = codestral_text_cache.get(func.text_hash)
        voyage_text = voyage_text_cache.get(func.text_hash)
        gemini_text = gemini_text_cache.get(func.text_hash)
        # AST variants keyed by hash
        openai_ast = openai_ast_cache.get(func.hash)
        codestral_ast = codestral_ast_cache.get(func.hash)
        voyage_ast = voyage_ast_cache.get(func.hash)
        gemini_ast = gemini_ast_cache.get(func.hash)

        if (
            openai_text is not None
            and openai_ast is not None
            and codestral_text is not None
            and codestral_ast is not None
            and voyage_text is not None
            and voyage_ast is not None
            and gemini_text is not None
            and gemini_ast is not None
        ):
            combined = compute_combined_embedding(
                openai_text,
                openai_ast,
                codestral_text,
                codestral_ast,
                voyage_text,
                voyage_ast,
                gemini_text,
                gemini_ast,
            )
            combined_cache[func.text_hash] = combined
            computed += 1

    return computed


def _update_function_hashes(baselines: Baselines, functions: list[FunctionInfo]) -> None:
    """Update function_hashes with the new format containing both hash and text_hash."""
    function_hashes = _get_function_hashes(baselines)

    for func in functions:
        key = f"{func.file}:{func.name}:{func.start_line}"
        function_hashes[key] = _FunctionHashEntry(hash=func.hash, text_hash=func.text_hash)


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


_ALL_PROVIDERS = [
    "openai-text",
    "openai-ast",
    "codestral-text",
    "codestral-ast",
    "voyage-text",
    "voyage-ast",
    "gemini-text",
    "gemini-ast",
    "combined",
]


def populate_embeddings(provider: str = "all") -> None:
    """Populate embedding caches for specified provider(s).

    Parameters
    ----------
    provider : str
        One of the 9 providers or ``"all"``.
    """
    print("Loading baselines and extracting functions...")  # noqa: T201
    baselines = load_baselines()
    functions = extract_all_function_infos(min_loc=1)
    print(f"Found {len(functions)} functions in codebase")  # noqa: T201

    _update_function_hashes(baselines, functions)

    total_new = 0
    for prov in _ALL_PROVIDERS if provider == "all" else [provider]:
        handler = _PROVIDER_MAP.get(prov)
        if handler is None:
            print(f"Unknown provider: {prov}")  # noqa: T201
            continue
        total_new += handler(baselines, functions)

    if total_new > 0:
        print(f"\nSaving {total_new} new embeddings to cache...")  # noqa: T201
    else:
        print("\nNo new embeddings to save.")  # noqa: T201

    save_baselines(baselines)
    if total_new > 0:
        print("Done!")  # noqa: T201


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Populate embedding caches for similarity tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=[*_ALL_PROVIDERS, "all"],
        default="all",
        help="Which provider to populate (default: all)",
    )
    args = parser.parse_args()

    try:
        populate_embeddings(args.provider)
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
