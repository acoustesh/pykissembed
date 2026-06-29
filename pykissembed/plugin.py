"""pytest plugin entry point.

Loaded automatically by pytest when pykissembed is installed (via the
``pytest11`` entry point declared in ``pyproject.toml``). The plugin:

* injects the ``update_baselines`` and ``cached_only`` fixtures used by
  pykissembed's own check modules,
* registers a ``pytest_configure`` hook that adds the project's source
  directories to ``sys.path`` (so pykissembed can import the user's code
  for similarity and refactor-index computations),
* **collects the check modules** (``pykissembed/checks/*.py``) as test
  modules via :func:`pytest_collect_file`, so they run automatically in
  any consumer project that has pykissembed installed — no need for the
  consumer to copy test files or configure ``testpaths``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Modules inside pykissembed/checks/ that contain test classes/functions.
# These are collected by the plugin and run in the consumer's pytest session.
_CHECK_MODULES = [
    "code_complexity",
    "code_similarity",
    "comment_density",
    "docstring_format",
    "lint_typecheck",
]

# Set of file stems that the plugin should collect as test modules.
# Used by :func:`pytest_collect_file` to decide whether a .py file inside
# the installed pykissembed package is a check module.
_CHECK_STEMS = frozenset(_CHECK_MODULES)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add pykissembed's custom CLI options to pytest."""
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Update baseline files instead of checking against them.",
    )
    parser.addoption(
        "--cached-only",
        action="store_true",
        default=False,
        help="Use only cached embeddings; skip any API calls.",
    )


@pytest.fixture
def update_baselines(request: pytest.FixtureRequest) -> bool:
    """Fixture returning the ``--update-baselines`` flag value."""
    return bool(request.config.getoption("--update-baselines"))


@pytest.fixture
def cached_only(request: pytest.FixtureRequest) -> bool:
    """Fixture returning the ``--cached-only`` flag value."""
    return bool(request.config.getoption("--cached-only"))


@pytest.fixture(scope="session")
def pykissembed_paths() -> list[Path]:
    """Resolved list of source directories from ``[tool.pykissembed]``.

    Returns
    -------
    list[Path]
        Resolved, existing directories. Empty list if nothing configured
        or nothing exists on disk.
    """
    from pykissembed.paths import resolve_paths

    return resolve_paths()


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Register pykissembed markers and ensure source dirs are importable."""
    config.addinivalue_line(
        "markers",
        "lint: lint + type-check gate (ruff, pyright)",
    )
    config.addinivalue_line(
        "markers",
        "complexity: code complexity metrics (CC, COG, MI, line counts, docstrings)",
    )
    config.addinivalue_line(
        "markers",
        "density: comment density checks",
    )
    config.addinivalue_line(
        "markers",
        "docstring_format: NumPy docstring format (ruff D rules)",
    )
    config.addinivalue_line(
        "markers",
        "similarity: embedding-based near-duplicate detection",
    )
    config.addinivalue_line(
        "markers",
        "experimental: unstable APIs — refactor_index, file_split",
    )

    # Ensure source dirs are importable for similarity/refactor tests
    if os.environ.get("PYQTEST_SKIP_PATH_INJECTION"):
        return
    try:
        from pykissembed.paths import resolve_paths
    except ImportError:  # pragma: no cover — defensive
        return
    import sys

    for p in resolve_paths():
        sp = str(p)
        if sp not in sys.path and p.exists():
            sys.path.insert(0, sp)


def _checks_dir() -> Path | None:
    """Return the path to the installed ``pykissembed/checks/`` directory."""
    try:
        import pykissembed.checks as checks_pkg
    except ImportError:  # pragma: no cover — defensive
        return None
    # checks_pkg.__file__ is .../pykissembed/checks/__init__.py
    if checks_pkg.__file__ is None:  # namespace package
        return None
    return Path(checks_pkg.__file__).parent


@pytest.hookimpl(trylast=True)
def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> pytest.Module | None:
    """Collect pykissembed's check modules as test modules.

    This hook makes the check modules (``code_complexity.py``,
    ``comment_density.py``, etc.) discoverable by pytest in *any* consumer
    project — without the consumer needing to configure ``testpaths`` or
    copy test files. The modules are collected only if they live inside
    the installed ``pykissembed/checks/`` directory and their stem matches
    a known check module name.
    """
    if file_path.suffix != ".py":
        return None
    if file_path.stem not in _CHECK_STEMS:
        return None
    checks = _checks_dir()
    if checks is None:
        return None
    # Only collect if this file is inside pykissembed/checks/
    try:
        file_path.relative_to(checks)
    except ValueError:
        return None
    return pytest.Module.from_parent(parent, path=file_path)
