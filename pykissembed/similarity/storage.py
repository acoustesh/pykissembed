"""Embedding storage management with compression and atomic writes.

Ported from ``mega-scrapper/tests/similarity/storage.py``. The main
adaptation is that path constants are resolved dynamically from
``[tool.pykissembed]`` config instead of being hardcoded.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import zlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np

from pykissembed.similarity import constants as _constants
from pykissembed.similarity.embeddings import compute_combined_embedding


def _as_float_list(value: object, *, context: str) -> list[float]:
    """Validate and return a numeric embedding vector as list[float].

    Returns
    -------
        list[float]
            Validated embedding vector.

    Raises
    ------
    TypeError
        If ``value`` is not a list of numeric values.
    """
    if not isinstance(value, list):
        msg = f"{context} must be a list[float]"
        raise TypeError(msg)
    validated: list[float] = []
    for idx, component in enumerate(cast("list[object]", value)):
        if not isinstance(component, (int, float)):
            msg = f"{context}[{idx}] must be numeric"
            raise TypeError(msg)
        validated.append(float(component))
    return validated


def _as_embedding_cache(value: object, *, context: str) -> dict[str, list[float]]:
    """Validate provider cache shape and return a typed mapping.

    Returns
    -------
        dict[str, list[float]]
            Typed provider cache mapping.

    Raises
    ------
    TypeError
        If ``value`` is not ``dict[str, list[float]]``.
    """
    if not isinstance(value, dict):
        msg = f"{context} must be a dict[str, list[float]]"
        raise TypeError(msg)
    for key, vec in cast("dict[object, object]", value).items():
        if not isinstance(key, str):
            msg = f"{context} keys must be str"
            raise TypeError(msg)
        _as_float_list(vec, context=f"{context}[{key!r}]")
    return cast("dict[str, list[float]]", value)


def _as_str_object_dict(value: object, *, context: str) -> dict[str, object]:
    """Validate a generic dict with string keys.

    Returns
    -------
        dict[str, object]
            Typed dictionary with string keys.

    Raises
    ------
    TypeError
        If ``value`` is not a dict with string keys.
    """
    if not isinstance(value, dict):
        msg = f"{context} must be a dict"
        raise TypeError(msg)
    for key in cast("dict[object, object]", value):
        if not isinstance(key, str):
            msg = f"{context} keys must be str"
            raise TypeError(msg)
    return cast("dict[str, object]", value)


def _get_cache(baselines: dict[str, object], cache_key: str) -> dict[str, list[float]]:
    """Read a provider cache from baselines and validate its shape.

    Returns
    -------
        Provider cache mapping keyed by hash.
    """
    return _as_embedding_cache(baselines.get(cache_key, {}), context=cache_key)


def _compress_embedding(vec: list[float]) -> str:
    """Compress a single embedding vector to base64-encoded zlib-compressed float32.

    Returns
    -------
        Base64-encoded zlib-compressed float32 string.
    """
    arr = np.array(vec, dtype=np.float32)
    return base64.b64encode(zlib.compress(arr.tobytes(), level=6)).decode("ascii")


def _decompress_embedding(b64_str: str) -> list[float]:
    """Decompress a base64-encoded zlib-compressed float32 embedding.

    Returns
    -------
        List of floats from the decompressed embedding.
    """
    return np.frombuffer(zlib.decompress(base64.b64decode(b64_str)), dtype=np.float32).tolist()


def _load_compressed_embeddings(file_path: Path) -> dict[str, list[float]]:
    """Load and decompress embeddings from a compressed cache file.

    Returns
    -------
        Mapping of hash to embedding vector.
    """
    if not file_path.exists():
        return {}
    with file_path.open(encoding="utf-8") as f:
        compressed = json.load(f)
    return {h: _decompress_embedding(b64) for h, b64 in compressed.items()}


def _save_compressed_embeddings(embeddings: dict[str, list[float]], file_path: Path) -> None:
    """Save embeddings to compressed cache file atomically."""
    compressed = {h: _compress_embedding(vec) for h, vec in embeddings.items()}
    _atomic_json_write(
        cast("dict[str, object]", compressed),
        file_path,
        prefix="emb_",
        suffix=".json.zlib",
    )


def _atomic_json_write(
    data: dict[str, object],
    file_path: Path,
    prefix: str = "tmp_",
    suffix: str = ".json",
) -> None:
    """Write JSON data atomically using temp file + rename."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=file_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(temp_path).replace(file_path)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


# ---------------------------------------------------------------------------
# Provider registry: enum, entry dataclass, and registry class
# ---------------------------------------------------------------------------


class HashType(StrEnum):
    """Hash variant used to key an embedding cache."""

    TEXT = "text"
    AST = "ast"


@dataclass(frozen=True)
class ProviderEntry:
    """Immutable descriptor for a single embedding provider.

    Attributes
    ----------
    name : str
        Snake_case internal identifier (e.g. ``"openai_text"``).
    label : str
        User-facing display name (e.g. ``"OpenAI-Text"``).
    cache_key : str
        Key inside the baselines dict (e.g. ``"openai_text_embeddings"``).
    file_path : Path
        Compressed cache file on disk.
    hash_type : HashType
        Determines whether ``text_hash`` or ``hash`` (AST) is used to key entries.
    default_threshold_pair : float
        Default pairwise similarity threshold.
    default_threshold_neighbor : float
        Default neighbor similarity threshold.
    default_pca_variance : float
        Default PCA variance retention threshold.
    """

    name: str
    label: str
    cache_key: str
    file_path: Path
    hash_type: HashType
    default_threshold_pair: float = 0.86
    default_threshold_neighbor: float = 0.80
    default_pca_variance: float = 0.99

    @property
    def hash_field(self) -> str:
        """FunctionInfo attribute name for the content hash."""
        return "text_hash" if self.hash_type == HashType.TEXT else "hash"

    @property
    def threshold_pair_key(self) -> str:
        """Config key for pair similarity threshold override."""
        return f"{self.name}_similarity_threshold_pair"

    @property
    def threshold_neighbor_key(self) -> str:
        """Config key for neighbor similarity threshold override."""
        return f"{self.name}_similarity_threshold_neighbor"

    @property
    def pca_variance_key(self) -> str:
        """Config key for PCA variance threshold override."""
        return f"{self.name}_pca_variance_threshold"


class EmbeddingRegistry:
    """Central catalogue of all embedding providers.

    Parameters
    ----------
    providers : list[ProviderEntry]
        All 9 provider entries (8 base + 1 combined).
    combined_key : str
        The ``cache_key`` that identifies the combined provider.
    """

    def __init__(self, providers: list[ProviderEntry], combined_key: str) -> None:
        """Initialize the storage object."""
        self._providers = tuple(providers)
        self._combined_key = combined_key
        self._by_cache_key: dict[str, ProviderEntry] = {p.cache_key: p for p in providers}

    # -- properties ---------------------------------------------------------

    @property
    def providers(self) -> tuple[ProviderEntry, ...]:
        """All registered providers (base + combined)."""
        return self._providers

    @property
    def base_providers(self) -> tuple[ProviderEntry, ...]:
        """The 8 non-combined providers."""
        return tuple(p for p in self._providers if p.cache_key != self._combined_key)

    @property
    def combined(self) -> ProviderEntry:
        """The single combined provider entry."""
        return next(p for p in self._providers if p.cache_key == self._combined_key)

    @property
    def combined_dependencies(self) -> list[str]:
        """Cache keys of the 8 base providers needed to build combined."""
        return [p.cache_key for p in self.base_providers]

    @property
    def files(self) -> dict[str, Path]:
        """Mapping of ``cache_key`` -> ``file_path`` for every provider."""
        return {p.cache_key: p.file_path for p in self._providers}

    def by_cache_key(self, cache_key: str) -> ProviderEntry:
        """Look up a provider by its ``cache_key``.

        Returns
        -------
            The matching ``ProviderEntry``.
        """
        return self._by_cache_key[cache_key]

    # -- helpers ------------------------------------------------------------

    def _provider_cache_sets(
        self,
        baselines: dict[str, object],
        providers: tuple[ProviderEntry, ...] | None = None,
    ) -> list[tuple[ProviderEntry, set[str], dict[str, list[float]]]]:
        """Resolve each provider to its valid-hash set and live cache."""
        valid_text, valid_ast, _ = get_valid_hashes(baselines)
        return [
            (
                p,
                valid_text if p.hash_type == HashType.TEXT else valid_ast,
                _get_cache(baselines, p.cache_key),
            )
            for p in (providers or self._providers)
        ]

    # -- operations ---------------------------------------------------------

    def audit(self, baselines: dict[str, object]) -> dict[str, int]:
        """Return cache sizes, missing counts, and valid-hash totals.

        Parameters
        ----------
        baselines : dict
            The loaded baselines dict.

        Returns
        -------
        dict[str, int]
            Keys include ``valid_text_hashes``, ``valid_ast_hashes``,
            ``<name>`` (cache size), and ``missing_<name>`` (missing count)
            for each base provider, plus ``combined`` cache size.
        """
        valid_text, valid_ast, _ = get_valid_hashes(baselines)
        result: dict[str, int] = {
            "valid_text_hashes": len(valid_text),
            "valid_ast_hashes": len(valid_ast),
        }
        for p, valid, cache in self._provider_cache_sets(baselines, self.base_providers):
            cache_keys = set(cache.keys())
            result[p.name] = len(cache_keys)
            result[f"missing_{p.name}"] = len(valid - cache_keys)
        result[self.combined.name] = len(_get_cache(baselines, self.combined.cache_key))
        return result

    def remove_orphans(self, baselines: dict[str, object]) -> dict[str, int]:
        """Remove orphan embeddings not present in function_hashes.

        Modifies *baselines* in place.

        Parameters
        ----------
        baselines : dict
            The loaded baselines dict.

        Returns
        -------
        dict[str, int]
            Keys are ``orphans_removed_<name>`` with counts removed.
        """
        stats: dict[str, int] = {}
        for p, valid, cache in self._provider_cache_sets(baselines):
            orphans = set(cache.keys()) - valid
            for h in orphans:
                del cache[h]
            stats[f"orphans_removed_{p.name}"] = len(orphans)
        return stats

    def rebuild_combined(self, baselines: dict[str, object]) -> int:
        """Rebuild combined embeddings from the 8 base providers.

        Modifies ``baselines[combined.cache_key]`` in place.

        Parameters
        ----------
        baselines : dict
            The loaded baselines dict.

        Returns
        -------
        int
            Number of combined embeddings built.
        """
        valid_text, _, text_to_ast = get_valid_hashes(baselines)

        text_caches: dict[str, dict[str, list[float]]] = {
            p.name: _get_cache(baselines, p.cache_key)
            for p in self.base_providers
            if p.hash_type == HashType.TEXT
        }
        ast_caches: dict[str, dict[str, list[float]]] = {
            p.name: _get_cache(baselines, p.cache_key)
            for p in self.base_providers
            if p.hash_type == HashType.AST
        }

        combined_cache: dict[str, list[float]] = {}
        for text_hash in valid_text:
            ast_hash = text_to_ast.get(text_hash)
            if not ast_hash:
                continue

            openai_text = text_caches["openai_text"].get(text_hash)
            openai_ast = ast_caches["openai_ast"].get(ast_hash)
            codestral_text = text_caches["codestral_text"].get(text_hash)
            codestral_ast = ast_caches["codestral_ast"].get(ast_hash)
            voyage_text = text_caches["voyage_text"].get(text_hash)
            voyage_ast = ast_caches["voyage_ast"].get(ast_hash)
            gemini_text = text_caches["gemini_text"].get(text_hash)
            gemini_ast = ast_caches["gemini_ast"].get(ast_hash)

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
                combined_cache[text_hash] = compute_combined_embedding(
                    openai_text,
                    openai_ast,
                    codestral_text,
                    codestral_ast,
                    voyage_text,
                    voyage_ast,
                    gemini_text,
                    gemini_ast,
                )

        baselines[self.combined.cache_key] = combined_cache
        return len(combined_cache)


# Module-level registry instance with all 9 providers
REGISTRY = EmbeddingRegistry(
    providers=[
        ProviderEntry(
            name="openai_text",
            label="OpenAI-Text",
            cache_key="openai_text_embeddings",
            file_path=_constants.OPENAI_TEXT_EMBEDDINGS_FILE,
            hash_type=HashType.TEXT,
            default_threshold_pair=0.86,
            default_threshold_neighbor=0.80,
        ),
        ProviderEntry(
            name="openai_ast",
            label="OpenAI-AST",
            cache_key="openai_ast_embeddings",
            file_path=_constants.OPENAI_AST_EMBEDDINGS_FILE,
            hash_type=HashType.AST,
            default_threshold_pair=0.86,
            default_threshold_neighbor=0.80,
        ),
        ProviderEntry(
            name="codestral_text",
            label="Codestral-Text",
            cache_key="codestral_text_embeddings",
            file_path=_constants.CODESTRAL_TEXT_EMBEDDINGS_FILE,
            hash_type=HashType.TEXT,
            default_threshold_pair=0.97,
            default_threshold_neighbor=0.93,
        ),
        ProviderEntry(
            name="codestral_ast",
            label="Codestral-AST",
            cache_key="codestral_ast_embeddings",
            file_path=_constants.CODESTRAL_AST_EMBEDDINGS_FILE,
            hash_type=HashType.AST,
            default_threshold_pair=0.97,
            default_threshold_neighbor=0.93,
        ),
        ProviderEntry(
            name="voyage_text",
            label="Voyage-Text",
            cache_key="voyage_text_embeddings",
            file_path=_constants.VOYAGE_TEXT_EMBEDDINGS_FILE,
            hash_type=HashType.TEXT,
            default_threshold_pair=0.95,
            default_threshold_neighbor=0.90,
        ),
        ProviderEntry(
            name="voyage_ast",
            label="Voyage-AST",
            cache_key="voyage_ast_embeddings",
            file_path=_constants.VOYAGE_AST_EMBEDDINGS_FILE,
            hash_type=HashType.AST,
            default_threshold_pair=0.95,
            default_threshold_neighbor=0.90,
        ),
        ProviderEntry(
            name="gemini_text",
            label="Gemini-Text",
            cache_key="gemini_text_embeddings",
            file_path=_constants.GEMINI_TEXT_EMBEDDINGS_FILE,
            hash_type=HashType.TEXT,
            default_threshold_pair=0.90,
            default_threshold_neighbor=0.85,
        ),
        ProviderEntry(
            name="gemini_ast",
            label="Gemini-AST",
            cache_key="gemini_ast_embeddings",
            file_path=_constants.GEMINI_AST_EMBEDDINGS_FILE,
            hash_type=HashType.AST,
            default_threshold_pair=0.90,
            default_threshold_neighbor=0.85,
        ),
        ProviderEntry(
            name="combined",
            label="Combined",
            cache_key="combined_embeddings",
            file_path=_constants.COMBINED_EMBEDDINGS_FILE,
            hash_type=HashType.TEXT,
            default_threshold_pair=0.88,
            default_threshold_neighbor=0.82,
        ),
    ],
    combined_key="combined_embeddings",
)

# Default config values shared by load_baselines and load_minimal_baselines
_DEFAULT_CONFIG = {
    "similarity_threshold_pair": 0.86,
    "similarity_threshold_neighbor": 0.80,
    "min_loc_for_similarity": 1,
    "pca_variance_threshold": 0.99,
    "refactor_index_threshold": 12.0,
    "refactor_index_top_n": 5,
}


def get_valid_hashes(baselines: dict[str, object]) -> tuple[set[str], set[str], dict[str, str]]:
    """Extract valid hashes from function_hashes.

    Returns
    -------
        tuple: (valid_text_hashes, valid_ast_hashes, text_to_ast_map)
            - valid_text_hashes: set of text_hash values
            - valid_ast_hashes: set of hash (AST) values
            - text_to_ast_map: mapping from text_hash -> hash for combined lookup

    Raises
    ------
    TypeError
        If a ``function_hashes`` entry is not ``str`` or ``dict``.
    """
    function_hashes = _as_str_object_dict(
        baselines.get("function_hashes", {}),
        context="function_hashes",
    )
    valid_text_hashes: set[str] = set()
    valid_ast_hashes: set[str] = set()
    text_to_ast_map: dict[str, str] = {}

    for entry in function_hashes.values():
        if isinstance(entry, dict):
            entry_obj = cast("dict[object, object]", entry)
            ast_hash_obj = entry_obj.get("hash")
            text_hash_obj = entry_obj.get("text_hash")
            if isinstance(ast_hash_obj, str) and ast_hash_obj:
                ast_hash = ast_hash_obj
                valid_ast_hashes.add(ast_hash)
            if isinstance(text_hash_obj, str) and text_hash_obj:
                text_hash = text_hash_obj
                valid_text_hashes.add(text_hash)
            if (
                isinstance(ast_hash_obj, str)
                and ast_hash_obj
                and isinstance(text_hash_obj, str)
                and text_hash_obj
            ):
                text_to_ast_map[text_hash_obj] = ast_hash_obj
        elif isinstance(entry, str):
            # Legacy format: entry is just the hash string (AST only)
            valid_ast_hashes.add(entry)
        else:
            msg = "function_hashes entries must be str or dict"
            raise TypeError(msg)

    return valid_text_hashes, valid_ast_hashes, text_to_ast_map


def load_minimal_baselines() -> dict[str, object]:
    """Load only config and function_hashes (no embeddings).

    This is faster than load_baselines() when embeddings will be loaded lazily.

    Returns
    -------
        Baselines dict with config and function_hashes.
    """
    baselines: dict[str, object] = {
        "function_hashes": {},
        "config": _DEFAULT_CONFIG.copy(),
    }

    baselines_file = _constants.baselines_file()
    if baselines_file.exists():
        with baselines_file.open(encoding="utf-8") as f:
            baselines.update(json.load(f))

    hashes_file = _constants.function_hashes_file()
    if hashes_file.exists():
        with hashes_file.open(encoding="utf-8") as f:
            baselines["function_hashes"] = json.load(f)

    return baselines


def load_provider_embeddings(
    baselines: dict[str, object],
    cache_key: str,
) -> dict[str, list[float]]:
    """Load a single provider's embeddings lazily.

    If already loaded in baselines, returns existing. Otherwise loads from file.
    For combined provider, also loads all 8 base providers if needed.

    Returns
    -------
        Mapping of hash to embedding vector for the requested provider.
    """
    # Already loaded?
    if baselines.get(cache_key):
        return _as_embedding_cache(baselines[cache_key], context=cache_key)

    # Combined requires all base providers
    if cache_key == REGISTRY.combined.cache_key:
        for dep_key in REGISTRY.combined_dependencies:
            if dep_key not in baselines or not baselines[dep_key]:
                file_path = REGISTRY.files.get(dep_key)
                if file_path:
                    baselines[dep_key] = _load_compressed_embeddings(file_path)

    # Load the requested provider
    file_path = REGISTRY.files.get(cache_key)
    if file_path:
        baselines[cache_key] = _load_compressed_embeddings(file_path)

    return _as_embedding_cache(baselines.get(cache_key, {}), context=cache_key)


def load_baselines() -> dict[str, object]:
    """Load baselines from JSON file with fallback to empty defaults.

    Returns
    -------
        Baselines dict with all embedding caches loaded.
    """
    baselines = load_minimal_baselines()

    # Load all embedding caches using provider mapping
    for cache_key, file_path in REGISTRY.files.items():
        baselines[cache_key] = _load_compressed_embeddings(file_path)

    return baselines


_SAVE_LOCK = threading.Lock()


def save_baselines(baselines: dict[str, object]) -> None:
    """Save baselines atomically. Embeddings and function_hashes are saved separately."""
    with _SAVE_LOCK:
        _save_baselines_unlocked(baselines)


def _save_baselines_unlocked(baselines: dict[str, object]) -> None:
    """Save baselines to disk without holding the lock (caller must hold it)."""
    baselines_file = _constants.baselines_file()
    baselines_file.parent.mkdir(parents=True, exist_ok=True)

    # Extract embedding caches to save separately
    embedding_caches = {key: baselines.pop(key, {}) for key in REGISTRY.files}
    function_hashes = _as_str_object_dict(
        baselines.pop("function_hashes", {}),
        context="function_hashes",
    )

    # Save main baselines
    try:
        _atomic_json_write(baselines, baselines_file, prefix="baselines_")
    finally:
        # Restore to dict
        baselines.update(embedding_caches)
        baselines["function_hashes"] = function_hashes

    # Save function_hashes
    if function_hashes:
        _atomic_json_write(function_hashes, _constants.function_hashes_file(), prefix="hashes_")

    # Save embedding caches using provider mapping
    for cache_key, file_path in REGISTRY.files.items():
        cache_obj = embedding_caches.get(cache_key, {})
        if cache_obj:
            cache = _as_embedding_cache(cache_obj, context=cache_key)
            _save_compressed_embeddings(cache, file_path)
