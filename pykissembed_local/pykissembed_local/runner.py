"""Compatibility helpers for the retired whole-file local cache runner."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path
    from typing import Never as Encoding

_CLOUD_ONLY_MESSAGE = (
    "Local embeddings were removed from pykissembed. Install 'pykissembed[cloud]' and use "
    "`pykissembed populate-embeddings --provider openai-text` "
    "(or gemini-text, voyage-text, codestral-text, qwen-text, or jina-text)."
)


class ProviderLike(Protocol):
    """Minimal protocol retained for callers importing the old runner type."""

    name: str
    model_id: str
    schema_version: str
    max_tokens: int
    batch_size: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


def _get_encoding() -> Encoding:
    """Reject the former token-encoding operation.

    Raises
    ------
    RuntimeError
        Always, because local tokenization has been retired.
    """
    raise RuntimeError(_CLOUD_ONLY_MESSAGE)


def iter_py_files(base_dir: Path) -> Iterable[Path]:
    """Yield every non-dunder ``.py`` file under *base_dir* recursively.

    Yields
    ------
    Path
        Each Python file, sorted for deterministic results.
    """
    for py_file in sorted(base_dir.rglob("*.py")):
        if py_file.name.startswith("__") or "__pycache__" in py_file.parts:
            continue
        yield py_file


def truncate_to_tokens(text: str, *, max_tokens: int) -> str:
    """Reject the retired local token-truncation operation.

    Raises
    ------
    RuntimeError
        Always, because local cache population has been retired.
    """
    del text, max_tokens
    raise RuntimeError(_CLOUD_ONLY_MESSAGE)


def content_hash(text: str) -> str:
    """Return the stable SHA-256 hex digest of *text*.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collect_inputs(paths: Sequence[Path], *, project_root: Path) -> list[dict[str, str]]:
    """Collect path and content records without generating embeddings.

    Returns
    -------
    list[dict[str, str]]
        Records with ``path`` and ``content`` string fields.
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
    """Compose the historical provider/content cache key.

    Returns
    -------
    str
        Legacy cache identity string.
    """
    return f"{provider.name}|{provider.model_id}|{provider.schema_version}|{content_hash(text)}"


def populate(
    provider: ProviderLike,
    paths: Sequence[Path],
    cache_dir: Path,
    *,
    project_root: Path | None = None,
) -> int:
    """Reject the retired whole-file JSONL population operation.

    Raises
    ------
    RuntimeError
        Always, before invoking a provider or modifying the filesystem.
    """
    del provider, paths, cache_dir, project_root
    raise RuntimeError(_CLOUD_ONLY_MESSAGE)


__all__ = [
    "ProviderLike",
    "cache_key",
    "content_hash",
    "iter_py_files",
    "populate",
    "truncate_to_tokens",
]
