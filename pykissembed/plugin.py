"""pytest plugin entry point.

Loaded automatically by pytest when pykissembed is installed (via the
``pytest11`` entry point declared in ``pyproject.toml``). The plugin:

* injects the ``update_baselines`` and ``cached_only`` fixtures used by
  pykissembed's own check modules,
* registers a ``pytest_configure`` hook that adds the project's source
  directories to ``sys.path`` (so pykissembed can import the user's code
  for similarity and refactor-index computations).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


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
