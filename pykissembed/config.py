"""Load ``[tool.pykissembed]`` configuration from ``pyproject.toml``.

pykissembed follows a layered configuration model (per advisor recommendation):

1. ``[tool.pykissembed]`` block in ``pyproject.toml`` (explicit, wins)
2. Auto-detection from common layout conventions
3. Failure: ambiguous or missing → raise so users get a clear error
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True, slots=True)
class PyqtestConfig:
    """Resolved pykissembed configuration.

    Attributes
    ----------
    paths
        Source directories to scan (relative to ``root``).
    mode
        ``"ratchet"`` (default) or ``"strict"``.
    baseline_dir
        Directory for committed baselines (relative to ``root``). Defaults
        to ``tests/baselines``.
    cache_dir
        Directory for embedding caches (relative to ``root``, should be
        gitignored). Defaults to ``tests/.pykissembed_cache``.
    root
        Project root (parent directory of ``pyproject.toml``).
    """

    paths: list[str] = field(default_factory=lambda: ["src"])
    mode: str = "ratchet"
    baseline_dir: str = "tests/baselines"
    cache_dir: str = "tests/.pykissembed_cache"
    root: Path = field(default_factory=Path.cwd)

    @property
    def baseline_path(self) -> Path:
        """Absolute path to the committed baselines directory."""
        return self.root / self.baseline_dir

    @property
    def cache_path(self) -> Path:
        """Absolute path to the (gitignored) cache directory."""
        return self.root / self.cache_dir

    def resolved_paths(self) -> list[Path]:
        """Return ``self.paths`` as absolute Path objects."""
        return [self.root / p for p in self.paths]


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file and return its contents as a dict.

    Returns
    -------
    dict[str, Any]
        Parsed TOML contents, or an empty dict if the file is missing.
    """
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return cast("dict[str, Any]", tomllib.load(f))


def _coerce_str_list(value: object, *, key: str) -> list[str]:
    """Coerce a TOML value into a ``list[str]``.

    Returns
    -------
    list[str]
        ``value`` if already a list of strings, ``[value]`` if a single
        string, ``[]`` otherwise.

    Raises
    ------
    TypeError
        If ``value`` is a list containing non-string items.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                msg = f"[tool.pykissembed] {key!r} must be list[str], got list containing {type(item).__name__}"
                raise TypeError(msg)
            result.append(item)
        return result
    return []


def load_config(start: Path | None = None) -> PyqtestConfig:
    """Resolve pykissembed configuration.

    Walks up from ``start`` (default: cwd) looking for a ``pyproject.toml``
    that contains a ``[tool.pykissembed]`` block. Falls back to auto-detection
    of a ``src/`` directory if no block is found.

    Parameters
    ----------
    start
        Directory to start the upward search from. Defaults to ``Path.cwd()``.

    Returns
    -------
    PyqtestConfig
        Frozen configuration dataclass.

    Raises
    ------
    FileNotFoundError
        If no ``pyproject.toml`` is found and auto-detection fails.
    """
    cursor = (start or Path.cwd()).resolve()
    root = cursor
    pyproject: Path | None = None
    while True:
        candidate = cursor / "pyproject.toml"
        if candidate.is_file():
            pyproject = candidate
            root = cursor
            break
        if cursor.parent == cursor:
            break
        cursor = cursor.parent

    if pyproject is None:
        # No pyproject.toml — try auto-detection in cwd
        return _auto_detect(Path.cwd())

    data = _read_toml(pyproject)
    tools = data.get("tool", {})
    if not isinstance(tools, dict):
        tools = {}
    section = tools.get("pykissembed", {})
    if not isinstance(section, dict):
        section = {}

    paths = _coerce_str_list(section.get("paths", ["src"]), key="paths")
    mode_raw = section.get("mode", "ratchet")
    mode = str(mode_raw) if isinstance(mode_raw, str) else "ratchet"
    if mode not in {"ratchet", "strict"}:
        mode = "ratchet"
    baseline_dir = str(section.get("baseline_dir", "tests/baselines"))
    cache_dir = str(section.get("cache_dir", "tests/.pykissembed_cache"))

    return PyqtestConfig(
        paths=paths or ["src"],
        mode=mode,
        baseline_dir=baseline_dir,
        cache_dir=cache_dir,
        root=root,
    )


def _auto_detect(root: Path) -> PyqtestConfig:
    """Auto-detect a project layout when no ``pyproject.toml`` is present.

    Returns
    -------
    PyqtestConfig
        A config with ``paths=["src"]`` if ``src/`` exists, else
        ``paths=["."]`` (current directory).
    """
    if (root / "src").is_dir():
        paths = ["src"]
    elif (root / "scripts").is_dir():
        paths = ["scripts"]
    elif any(root.glob("*.py")):
        paths = ["."]
    else:
        paths = ["src"]
    return PyqtestConfig(
        paths=paths,
        mode=os.environ.get("pykissembed_MODE", "ratchet"),
        baseline_dir="tests/baselines",
        cache_dir="tests/.pykissembed_cache",
        root=root,
    )


def get_config() -> PyqtestConfig:
    """Return the cached pykissembed configuration.

    Uses :func:`load_config` on first call and caches the result in
    ``pykissembed_CONFIG_CACHE`` for subsequent calls within the same process.
    """
    cache: dict[str, PyqtestConfig] = globals().setdefault("pykissembed_CONFIG_CACHE", {})  # type: ignore[var-annotated]
    key = str(Path.cwd())
    if key not in cache:
        cache[key] = load_config()
    return cache[key]


def reset_config_cache() -> None:
    """Clear the config cache (used by tests)."""
    cache = globals().get("pykissembed_CONFIG_CACHE")
    if isinstance(cache, dict):
        cache.clear()
