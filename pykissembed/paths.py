"""Path discovery for pykissembed.

Implements the layered discovery model from the design notes:

1. ``[tool.pykissembed]`` ``paths = [...]`` wins (explicit).
2. Otherwise, auto-detect from ``pyproject.toml`` ``[tool.setuptools]``
   or ``[project]`` packages.
3. Otherwise, scan for common conventions (src/, scripts/, lib/).
4. Fail loudly if the result is ambiguous.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pykissembed.config import get_config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_COMMON_DIRS = ("src", "scripts", "lib")

# Directories that should never be scanned for Python source files.
# These are either virtual environments, build artifacts, or dependency
# caches that contain third-party code (which may include non-UTF-8 files).
_IGNORED_DIRS = frozenset(
    {
        ".venv",
        "venv",
        ".env",
        "env",
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".tox",
        ".eggs",
        ".mypy_cache",
        ".pyright",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".pykissembed_cache",
        "site-packages",
    }
)


def _should_skip(path: Path) -> bool:
    """Return True if *path* is inside an ignored directory.

    Parameters
    ----------
    path : Path
        File or directory path to check.

    Returns
    -------
    bool
        ``True`` if any component of *path* is in ``_IGNORED_DIRS``.
    """
    return any(part in _IGNORED_DIRS for part in path.parts)


def iter_py_files(base_dir: Path) -> Iterator[Path]:
    """Yield every ``.py`` file under *base_dir* (recursive), skipping ignored dirs.

    Skips files whose names start with ``__`` and any file inside a
    directory listed in ``_IGNORED_DIRS`` (e.g. ``.venv/``,
    ``__pycache__/``, ``node_modules/``).

    Parameters
    ----------
    base_dir : Path
        Root directory to scan recursively.

    Yields
    ------
    Path
        Each Python file under *base_dir*, sorted for determinism.
    """
    # sorted(): rglob() order depends on the OS/filesystem directory listing
    # order, which isn't guaranteed stable across machines — baselines keyed
    # by file path need deterministic iteration so re-running a check
    # doesn't reorder (and spuriously diff) its own output.
    for py_file in sorted(base_dir.rglob("*.py")):
        # Dunder files (__init__.py, __main__.py, ...) are typically
        # re-export/entry-point boilerplate, not code meant to carry
        # docstring/complexity/similarity scoring.
        if py_file.name.startswith("__"):
            continue
        if _should_skip(py_file):
            continue
        yield py_file


def resolve_paths() -> list[Path]:
    """Return the list of source directories to scan.

    Returns
    -------
    list[Path]
        Absolute paths to existing directories, filtered to those that
        actually exist on disk.
    """
    config = get_config()
    resolved: list[Path] = []
    for raw in config.paths:
        p = (config.root / raw).resolve()
        if p.is_dir():
            resolved.append(p)
    if not resolved:
        # Layer 2: fallback to common conventions
        for d in _COMMON_DIRS:
            candidate = (config.root / d).resolve()
            if candidate.is_dir():
                resolved.append(candidate)
    return resolved


def root() -> Path:
    """Return the project root (parent of ``pyproject.toml``).

    Returns
    -------
    Path
        ``get_config().root``.
    """
    return get_config().root


def include_notebooks() -> bool:
    """Return True if the consumer opted into scanning ``.ipynb`` files.

    Reads ``[tool.pykissembed] include_notebooks`` from ``pyproject.toml``.
    Defaults to ``False``.

    Returns
    -------
    bool
        ``get_config().include_notebooks``.
    """
    return bool(get_config().include_notebooks)


def warn_non_utf8(file_path: Path, exc: Exception) -> None:
    """Emit a UserWarning when a file cannot be read as UTF-8.

    Non-UTF-8 files are suspicious in Python projects — they may be
    malicious (e.g. obfuscated payloads using exotic encodings like
    Big5) or simply misplaced binary data. The file is skipped, but
    the consumer is warned so they can investigate.

    Parameters
    ----------
    file_path : Path
        The file that failed to decode.
    exc : Exception
        The exception that was raised (``UnicodeDecodeError`` or
        ``OSError``).
    """
    warnings.warn(
        f"pykissembed: skipping {file_path} — not valid UTF-8 "
        f"({type(exc).__name__}: {exc}). "
        "Non-UTF-8 Python files may be malicious; investigate if unexpected.",
        UserWarning,
        stacklevel=2,
    )
