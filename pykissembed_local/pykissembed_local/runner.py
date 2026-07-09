"""Embedding-cache population driver.

The core ``pykissembed`` CLI delegates ``populate-embeddings`` here via:

    from pykissembed_local.runner import populate

This module is the only piece of ``pykissembed-local`` that walks the
filesystem and writes a cache; the rest of the package is just the
``Provider`` implementation.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tiktoken import Encoding


class ProviderLike(Protocol):
    """Minimal protocol — duck-types ``pykissembed.providers.Provider``."""

    name: str
    model_id: str
    schema_version: str
    max_tokens: int
    batch_size: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


@functools.cache
def _get_encoding() -> Encoding:
    """Lazy-load the ``cl100k_base`` tiktoken encoding (cached on first use).

    Deferring the import keeps this module importable without ``tiktoken``
    (e.g. in tests that never call the token-aware paths).

    Returns
    -------
    Encoding
        The ``cl100k_base`` tiktoken encoding.

    Raises
    ------
    RuntimeError
        If ``tiktoken`` is not installed.
    """
    try:
        import tiktoken
    except ImportError as exc:
        msg = "pykissembed-local requires tiktoken for token-aware truncation"
        raise RuntimeError(msg) from exc
    return tiktoken.get_encoding("cl100k_base")


def iter_py_files(base_dir: Path) -> Iterable[Path]:
    """Yield every ``.py`` file under *base_dir* (recursive).

    Skips ``__init__.py``-style dunders and ``__pycache__`` directories.

    Yields
    ------
    Path
        Each Python file under *base_dir*, sorted for determinism.
    """
    for py_file in sorted(base_dir.rglob("*.py")):
        if py_file.name.startswith("__") or "__pycache__" in py_file.parts:
            continue
        yield py_file


def truncate_to_tokens(text: str, *, max_tokens: int) -> str:
    """Truncate *text* to *max_tokens* tokens using ``cl100k_base``.

    Parameters
    ----------
    text
        Input text. May be empty.
    max_tokens
        Maximum token count after truncation.

    Returns
    -------
    str
        The original text if it fits in *max_tokens*; otherwise a
        prefix decoded from the first ``max_tokens`` tokens.
    """
    enc = _get_encoding()
    token_ids = enc.encode(text, disallowed_special=())
    if len(token_ids) <= max_tokens:
        return text
    return enc.decode(token_ids[:max_tokens])


def content_hash(text: str) -> str:
    """Stable SHA-256 of *text* — used as part of the cache key.

    Returns
    -------
    str
        Hex digest of the SHA-256 hash of *text*.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collect_inputs(paths: Sequence[Path], *, project_root: Path) -> list[dict[str, str]]:
    """Collect (rel_path, content) records for every .py file under *paths*.

    Returns
    -------
    list[dict[str, str]]
        Each record has ``"path"`` (relative to *project_root* when
        possible) and ``"content"`` (UTF-8 text, with replacement on
        decode errors).
    """
    records: list[dict[str, str]] = []
    for base in paths:
        if not base.is_dir():
            continue
        for py_file in iter_py_files(base):
            try:
                rel = str(py_file.resolve().relative_to(project_root))
            except ValueError:
                rel = str(py_file)
            records.append(
                {
                    "path": rel,
                    "content": py_file.read_text(encoding="utf-8", errors="replace"),
                },
            )
    return records


def cache_key(provider: ProviderLike, text: str) -> str:
    """Compose the (provider, content) cache key — matches pykissembed's format.

    Returns
    -------
    str
        ``f"{provider.name}|{provider.model_id}|{provider.schema_version}|{sha256(text)}"``
    """
    return f"{provider.name}|{provider.model_id}|{provider.schema_version}|{content_hash(text)}"


def populate(
    provider: ProviderLike,
    paths: Sequence[Path],
    cache_dir: Path,
    *,
    project_root: Path | None = None,
) -> int:
    """Compute and persist embeddings for every .py file under *paths*.

    Skips files whose cache entry already exists. The cache is a single
    JSONL file at ``<cache_dir>/<provider.name>-<model_id>.jsonl`` where
    each line is ``{"key": ..., "path": ..., "vector": [...]}``.

    Parameters
    ----------
    provider
        Any object that satisfies the Provider protocol.
    paths
        Source directories to scan (relative to the project root).
    cache_dir
        Directory in which to write the cache file.
    project_root
        Root used to compute relative paths in the cache. Defaults to
        ``Path.cwd()``. Tests should pass an explicit value to keep
        behaviour deterministic.

    Returns
    -------
    int
        Number of *new* cache entries written.

    Raises
    ------
    RuntimeError
        If the provider returns the wrong number of vectors for the
        given inputs, or if ``tiktoken`` is not installed.
    """
    root = (project_root or Path.cwd()).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model = provider.model_id.replace("/", "_")
    cache_file = cache_dir / f"{provider.name}-{safe_model}.jsonl"

    existing: set[str] = set()
    if cache_file.is_file():
        with cache_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    key = rec.get("key")
                    if isinstance(key, str):
                        existing.add(key)

    inputs = _collect_inputs(paths, project_root=root)
    if not inputs:
        return 0

    pending: list[dict[str, str]] = []
    pending_keys: list[str] = []
    for rec in inputs:
        key = cache_key(provider, rec["content"])
        if key in existing:
            continue
        pending.append(rec)
        pending_keys.append(key)

    if not pending:
        return 0

    truncated = [
        truncate_to_tokens(rec["content"], max_tokens=provider.max_tokens) for rec in pending
    ]
    vectors = provider.embed(truncated)
    if len(vectors) != len(pending):
        msg = (
            f"Provider {provider.name!r} returned {len(vectors)} vectors for {len(pending)} inputs"
        )
        raise RuntimeError(msg)

    with cache_file.open("a", encoding="utf-8") as f:
        for rec, key, vec in zip(pending, pending_keys, vectors, strict=True):
            json.dump({"key": key, "path": rec["path"], "vector": vec}, f)
            f.write("\n")
    return len(pending)


__all__ = [
    "ProviderLike",
    "cache_key",
    "content_hash",
    "iter_py_files",
    "populate",
    "truncate_to_tokens",
]
