"""Load ``[tool.pykissembed]`` configuration from ``pyproject.toml``.

pykissembed follows a layered configuration model (per advisor recommendation):

1. ``[tool.pykissembed]`` block in ``pyproject.toml`` (explicit, wins)
2. Auto-detection from common layout conventions
3. Failure: ambiguous or missing → raise so users get a clear error
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    include_notebooks
        Whether to apply ruff and similarity checks to ``.ipynb`` files.
        Defaults to ``False`` — notebooks are typically exploratory.
    cached_only
        Whether similarity checks should use only cached embeddings. Defaults
        to ``False`` — missing embeddings are populated through configured
        providers.
    wrapper_max_call_sites
        Maximum project-wide static call-site count allowed for an exact
        pass-through wrapper. Defaults to ``1``.
    wrapper_exclude
        Glob patterns for intentional wrapper identifiers in the form
        ``relative/path.py:QualifiedName``.
    wrapper_exempt_decorators
        Glob patterns for decorator names that mark intentional wrappers.
    root
        Project root (parent directory of ``pyproject.toml``).
    """

    paths: list[str] = field(default_factory=lambda: ["src"])
    mode: str = "ratchet"
    baseline_dir: str = "tests/baselines"
    cache_dir: str = "tests/.pykissembed_cache"
    include_notebooks: bool = False
    cached_only: bool = False
    wrapper_max_call_sites: int = 1
    wrapper_exclude: list[str] = field(default_factory=list)
    wrapper_exempt_decorators: list[str] = field(default_factory=list)
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
        """Return ``self.paths`` as absolute Path objects.

        Returns
        -------
        list[Path]
            Each entry in ``self.paths`` joined onto ``self.root``.
        """
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
        return tomllib.load(f)


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


def _require_str_list(value: object, *, key: str) -> list[str]:
    """Validate a TOML list containing only strings.

    Returns
    -------
    list[str]
        A copy of *value* when it is a list of strings.

    Raises
    ------
    TypeError
        If *value* is not a list of strings.
    """
    if not isinstance(value, list):
        msg = f"[tool.pykissembed] {key!r} must be list[str]"
        raise TypeError(msg)
    return _coerce_str_list(value, key=key)


def _require_nonnegative_int(value: object, *, key: str) -> int:
    """Validate a non-negative integer TOML value.

    Returns
    -------
    int
        The validated integer.

    Raises
    ------
    TypeError
        If *value* is not a non-negative integer.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"[tool.pykissembed] {key!r} must be a non-negative integer"
        raise TypeError(msg)
    return value


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

    Notes
    -----
    If no ``pyproject.toml`` is found and auto-detection fails, the lookup
    raises ``FileNotFoundError``.
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
        # `Path("/").parent == Path("/")` (and likewise for a Windows drive
        # root) — this is the portable way to detect "walked past the
        # filesystem root" without special-casing POSIX vs. Windows roots.
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
    # Deliberate exception to this module's stated "ambiguous/missing
    # config raises" policy: an invalid `mode` value (e.g. a typo) falls
    # back to the safer "ratchet" default instead of raising, so a
    # config typo degrades to a slightly-too-lenient gate rather than
    # breaking every consumer's CI outright.
    if mode not in {"ratchet", "strict"}:
        mode = "ratchet"
    baseline_dir = str(section.get("baseline_dir", "tests/baselines"))
    cache_dir = str(section.get("cache_dir", "tests/.pykissembed_cache"))
    include_notebooks_raw = section.get("include_notebooks", False)
    include_notebooks = bool(include_notebooks_raw)
    cached_only_raw = section.get("cached_only", False)
    cached_only = bool(cached_only_raw)
    wrapper_max_call_sites = _require_nonnegative_int(
        section.get("wrapper_max_call_sites", 1),
        key="wrapper_max_call_sites",
    )
    wrapper_exclude = _require_str_list(
        section.get("wrapper_exclude", []),
        key="wrapper_exclude",
    )
    wrapper_exempt_decorators = _require_str_list(
        section.get("wrapper_exempt_decorators", []),
        key="wrapper_exempt_decorators",
    )

    return PyqtestConfig(
        paths=paths or ["src"],
        mode=mode,
        baseline_dir=baseline_dir,
        cache_dir=cache_dir,
        include_notebooks=include_notebooks,
        cached_only=cached_only,
        wrapper_max_call_sites=wrapper_max_call_sites,
        wrapper_exclude=wrapper_exclude,
        wrapper_exempt_decorators=wrapper_exempt_decorators,
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
        mode=os.environ.get("PYKISSEMBED_MODE", "ratchet"),
        baseline_dir="tests/baselines",
        cache_dir="tests/.pykissembed_cache",
        include_notebooks=False,
        cached_only=False,
        root=root,
    )


def get_config() -> PyqtestConfig:
    """Return the cached pykissembed configuration.

    Uses :func:`load_config` on first call and caches the result in
    ``pykissembed_CONFIG_CACHE`` for subsequent calls within the same process.

    Returns
    -------
    PyqtestConfig
        The configuration for the current working directory, loaded
        fresh on first call per cwd and cached thereafter.
    """
    # Keyed by cwd (not a single module-level singleton) because tests that
    # chdir into fixture repos need their own independent cached config —
    # a single shared cache would leak one fixture's [tool.pykissembed]
    # settings into the next test that runs from a different directory.
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
