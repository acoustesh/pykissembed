"""Code Similarity Detection Tests.

Ported from ``aa-ml/mega-scrapper/tests/test_code_similarity.py``. This
module detects near-duplicate functions using embedding providers:
- Text variants: signature + docstring + comments (keyed by text_hash)
- AST variants: ast.unparse() output (keyed by hash)
- Combined: 8-way concatenation of all base embeddings

Each variant detects:
- Pairwise similarity detection (finds copy-paste code)
- Neighbor clustering (finds functions with multiple similar counterparts)
- Refactor index computation (combines complexity + similarity for priority)

Providers:
- OpenAI-Text, OpenAI-AST (text-embedding-3-large)
- Codestral-Text, Codestral-AST (codestral-embed via OpenRouter)
- Voyage-Text, Voyage-AST (voyage-code-3)
- Gemini-Text, Gemini-AST (gemini-embedding-001)
- Combined (8-way concatenation)

Baseline Management:
    Run ``pytest --update-baselines`` to update similarity baselines.
    Review the diff in ``tests/baselines/similarity.json`` and commit via PR.

    For similarity tests without API calls:
    Run ``pytest -m similarity --cached-only`` to use only cached embeddings.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, cast

import pytest

from pykissembed.similarity import (
    CODESTRAL_AST_PROVIDER,
    CODESTRAL_TEXT_PROVIDER,
    COMBINED_PROVIDER,
    GEMINI_AST_PROVIDER,
    GEMINI_TEXT_PROVIDER,
    OPENAI_AST_PROVIDER,
    OPENAI_TEXT_PROVIDER,
    VOYAGE_AST_PROVIDER,
    VOYAGE_TEXT_PROVIDER,
    FunctionInfo,
    ProviderEntry,
    run_provider_similarity_checks,
)
from pykissembed.similarity.complexity import load_all_complexity_maps
from pykissembed.similarity.types import is_str_object_dict

if TYPE_CHECKING:
    from pykissembed.similarity.types import PCAModel

type SharedBaselines = dict[str, object]
type SimilarityPcaCache = dict[str, tuple[PCAModel | None, int, bool]]


def _extract_int(config: dict[str, object], key: str, default: int) -> int:
    """Extract an int from config, falling back to *default*.

    Parameters
    ----------
    config : dict[str, object]
        Configuration dictionary.
    key : str
        Key to look up.
    default : int
        Fallback value if key is missing or not an int.

    Returns
    -------
    int
        The extracted integer value.

    Raises
    ------
    TypeError
        If the value exists but is not an int.
    """
    raw_value = config.get(key, default)
    if not isinstance(raw_value, int):
        msg = f"config['{key}'] must be int"
        raise TypeError(msg)
    return raw_value


def _extract_str_list(config: dict[str, object], key: str) -> list[str]:
    """Extract a list[str] from config.

    Parameters
    ----------
    config : dict[str, object]
        Configuration dictionary.
    key : str
        Key to look up.

    Returns
    -------
    list[str]
        The extracted string list.

    Raises
    ------
    TypeError
        If the value exists but is not a list of strings.
    """
    raw_value = config.get(key, [])
    if not isinstance(raw_value, list):
        msg = f"config['{key}'] must be list[str]"
        raise TypeError(msg)
    raw_list = cast("list[object]", raw_value)
    if not all(isinstance(item, str) for item in raw_list):
        msg = f"config['{key}'] must be list[str]"
        raise TypeError(msg)
    return [item for item in raw_list if isinstance(item, str)]


def _extract_float(config: dict[str, object], key: str, default: float) -> float:
    """Extract a float from config, falling back to *default*.

    Parameters
    ----------
    config : dict[str, object]
        Configuration dictionary.
    key : str
        Key to look up.
    default : float
        Fallback value if key is missing.

    Returns
    -------
    float
        The extracted float value.

    Raises
    ------
    TypeError
        If the value exists but is not numeric.
    """
    raw_value = config.get(key, default)
    if not isinstance(raw_value, int | float):
        msg = f"config['{key}'] must be float"
        raise TypeError(msg)
    return float(raw_value)


def _run_similarity_test(
    provider: ProviderEntry,
    shared_baselines: SharedBaselines,
    shared_functions: list[FunctionInfo],
    pca_cache: SimilarityPcaCache,
    *,
    update_baselines: bool,
    cached_only: bool,
) -> None:
    """Common logic for all similarity tests.

    Uses session-scoped shared_baselines, shared_functions, and pca_cache
    to avoid re-loading embeddings and re-fitting PCA for each test.

    Parameters
    ----------
    provider : ProviderEntry
        The embedding provider to test.
    update_baselines : bool
        Whether to update baselines instead of checking.
    cached_only : bool
        If True, skip when embeddings are missing.
    shared_baselines : SharedBaselines
        Session-scoped baselines dict.
    shared_functions : list[FunctionInfo]
        Session-scoped list of extracted functions.
    pca_cache : SimilarityPcaCache
        Session-scoped PCA model cache.

    Raises
    ------
    TypeError
        If ``shared_baselines["config"]`` is not ``dict[str, object]``.
    """
    raw_config = shared_baselines.get("config", {})
    if not is_str_object_dict(raw_config):
        msg = "shared_baselines['config'] must be dict[str, object]"
        raise TypeError(msg)
    config = raw_config
    min_loc = _extract_int(config, "min_loc_for_similarity", 1)
    excluded_dirs = _extract_str_list(config, "excluded_directories")
    # Three-level fallback lets a consumer override a threshold for one
    # provider without having to also set it for the other seven: a
    # provider-specific key wins, then a shared key across all providers,
    # then the provider's own hardcoded default.
    threshold_pair = _extract_float(
        config,
        provider.threshold_pair_key,
        _extract_float(config, "similarity_threshold_pair", provider.default_threshold_pair),
    )
    threshold_neighbor = _extract_float(
        config,
        provider.threshold_neighbor_key,
        _extract_float(
            config,
            "similarity_threshold_neighbor",
            provider.default_threshold_neighbor,
        ),
    )

    # Filter functions by min_loc / excluded directories, deep copy to avoid mutation
    functions: list[FunctionInfo] = [
        copy.deepcopy(func)
        for func in shared_functions
        if func.loc >= min_loc and not any(directory in func.file for directory in excluded_dirs)
    ]

    run_provider_similarity_checks(
        baselines=shared_baselines,
        functions=functions,
        update_baselines=update_baselines,
        cached_only=cached_only,
        provider=provider,
        threshold_pair=threshold_pair,
        threshold_neighbor=threshold_neighbor,
        load_complexity_maps_fn=load_all_complexity_maps,
        pca_cache=pca_cache,
    )


_PARALLEL_PROVIDERS: list[ProviderEntry] = [
    OPENAI_TEXT_PROVIDER,
    OPENAI_AST_PROVIDER,
    CODESTRAL_TEXT_PROVIDER,
    CODESTRAL_AST_PROVIDER,
    VOYAGE_TEXT_PROVIDER,
    VOYAGE_AST_PROVIDER,
    GEMINI_TEXT_PROVIDER,
    GEMINI_AST_PROVIDER,
]


@pytest.mark.similarity
def test_providers_parallel(
    shared_baselines: SharedBaselines,
    shared_functions: list[FunctionInfo],
    pca_cache: SimilarityPcaCache,
    *,
    update_baselines: bool,
    cached_only: bool,
) -> None:
    """Run all non-combined embedding providers in parallel.

    Parameters
    ----------
    update_baselines : bool
        Whether to update baselines instead of checking.
    cached_only : bool
        If True, skip when embeddings are missing.
    shared_baselines : SharedBaselines
        Session-scoped baselines dict.
    shared_functions : list[FunctionInfo]
        Session-scoped list of extracted functions.
    pca_cache : SimilarityPcaCache
        Session-scoped PCA model cache.

    Raises
    ------
    SystemExit
        Propagated if a provider raises ``SystemExit``.
    GeneratorExit
        Propagated if a provider raises ``GeneratorExit``.
    KeyboardInterrupt
        Propagated if a provider raises ``KeyboardInterrupt``.
    """
    if not shared_functions:
        pytest.skip("No [tool.pykissembed] paths configured")

    def _run_one(provider: ProviderEntry) -> None:
        _run_similarity_test(
            provider,
            shared_baselines,
            shared_functions,
            pca_cache,
            update_baselines=update_baselines,
            cached_only=cached_only,
        )

    errors: dict[str, BaseException] = {}
    skips: dict[str, str] = {}
    # One worker per provider (not a bounded pool): each `_run_one` call
    # does network I/O against a distinct API, so full parallelism here
    # bounds wall-clock time by the slowest single provider rather than by
    # the sum of all eight.
    with ThreadPoolExecutor(max_workers=len(_PARALLEL_PROVIDERS)) as executor:
        futures = {executor.submit(_run_one, p): p for p in _PARALLEL_PROVIDERS}
        for future in as_completed(futures):
            provider = futures[future]
            try:
                future.result()
            except pytest.skip.Exception as exc:  # type: ignore[attr-defined]
                skips[provider.label] = str(exc)
            # Unparenthesized multi-exception `except` (PEP 758, Python 3.14+)
            # — equivalent to `except (KeyboardInterrupt, SystemExit,
            # GeneratorExit):`. Looks like a Python 2 typo at a glance but is
            # valid modern syntax; this project targets py314+.
            except KeyboardInterrupt, SystemExit, GeneratorExit:
                raise
            except BaseException as exc:  # noqa: BLE001 — deliberately broad: aggregates every
                # provider's failure (network, API, assertion) so one provider's error doesn't
                # hide the others'; process-exit signals are already re-raised above.
                errors[provider.label] = exc

    if errors:
        detail = "\n".join(f"  {name}: {exc}" for name, exc in errors.items())
        pytest.fail(f"Provider failures:\n{detail}", pytrace=False)

    if skips and len(skips) == len(_PARALLEL_PROVIDERS):
        pytest.skip("All providers skipped:\n" + "\n".join(f"  {n}: {m}" for n, m in skips.items()))


@pytest.mark.similarity
@pytest.mark.experimental
def test_combined_similarity(
    shared_baselines: SharedBaselines,
    shared_functions: list[FunctionInfo],
    pca_cache: SimilarityPcaCache,
    *,
    update_baselines: bool,
    cached_only: bool,
) -> None:
    """Run combined provider after all individual providers have finished.

    Parameters
    ----------
    update_baselines : bool
        Whether to update baselines instead of checking.
    cached_only : bool
        If True, skip when embeddings are missing.
    shared_baselines : SharedBaselines
        Session-scoped baselines dict.
    shared_functions : list[FunctionInfo]
        Session-scoped list of extracted functions.
    pca_cache : SimilarityPcaCache
        Session-scoped PCA model cache.
    """
    if not shared_functions:
        pytest.skip("No [tool.pykissembed] paths configured")
    _run_similarity_test(
        COMBINED_PROVIDER,
        shared_baselines,
        shared_functions,
        pca_cache,
        update_baselines=update_baselines,
        cached_only=cached_only,
    )
