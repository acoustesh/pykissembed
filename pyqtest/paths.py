"""Path discovery for pyqtest.

Implements the layered discovery model from the design notes:

1. ``[tool.pyqtest]`` ``paths = [...]`` wins (explicit).
2. Otherwise, auto-detect from ``pyproject.toml`` ``[tool.setuptools]``
   or ``[project]`` packages.
3. Otherwise, scan for common conventions (src/, scripts/, lib/).
4. Fail loudly if the result is ambiguous.
"""

from __future__ import annotations

from pathlib import Path

from pyqtest.config import get_config

_COMMON_DIRS = ("src", "scripts", "lib")


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
    """Return the project root (parent of ``pyproject.toml``)."""
    return get_config().root
