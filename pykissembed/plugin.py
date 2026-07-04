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
from _pytest.stash import StashKey

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
    parser.addoption(
        "--pykissembed-all",
        action="store_true",
        default=False,
        help=(
            "Auto-collect and run every pykissembed check module. "
            "Without this flag, pykissembed only auto-injects check modules "
            "when you target a specific pykissembed NodeId (e.g. "
            "`pytest .../docstring_format.py::TestDocstringFormat`). "
            "Use this flag for the default 'run the full battery' behaviour."
        ),
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


# ---------------------------------------------------------------------------
# Session-scoped fixtures for similarity tests (shared state across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def shared_baselines() -> dict[str, object]:
    """Session-scoped baselines (config + function_hashes, embeddings lazy-loaded).

    Returns
    -------
    dict[str, object]
        The loaded baselines dictionary.
    """
    from pykissembed.similarity.storage import load_minimal_baselines

    return load_minimal_baselines()


@pytest.fixture(scope="session")
def shared_functions(shared_baselines: dict[str, object]) -> list:
    """Session-scoped list of FunctionInfo objects extracted from workspace.

    Returns
    -------
    list[FunctionInfo]
        The extracted function info objects.
    """
    from pykissembed.similarity.ast_helpers import extract_all_function_infos

    return extract_all_function_infos(min_loc=1)


@pytest.fixture(scope="session")
def pca_cache() -> dict[str, tuple]:
    """Session-scoped cache for fitted PCA models.

    Returns
    -------
    dict[str, tuple]
        An empty dictionary for caching PCA models.
    """
    return {}


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Register markers, inject source dirs into sys.path, and collect checks.

    The last action is critical: pytest only walks directories listed in
    ``config.args`` (derived from ``testpaths`` or CLI arguments). The
    installed ``pykissembed/checks/`` directory is never in that list by
    default, so :func:`pytest_collect_file` would never be called for
    those files. We append the checks directory to ``config.args`` here
    so pytest discovers and collects the check modules automatically.
    """
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
    if not os.environ.get("PYQTEST_SKIP_PATH_INJECTION"):
        try:
            from pykissembed.paths import resolve_paths
        except ImportError:  # pragma: no cover — defensive
            pass
        else:
            import sys

            for p in resolve_paths():
                sp = str(p)
                if sp not in sys.path and p.exists():
                    sys.path.insert(0, sp)

    # Inject the installed pykissembed/checks/ directory (or a single
    # check file) into pytest's collection args. Default policy: do NOT
    # auto-inject. The consumer must either pass ``--pykissembed-all`` for
    # the full battery, or target a specific check NodeId (smart-restrict),
    # or rely on a testpaths/pyproject configuration that already includes
    # the checks directory. This keeps a focused `pytest <file>::Test::test_x`
    # invocation from accidentally collecting every plugin check.
    checks = _checks_dir()
    if checks is not None and checks.is_dir():
        target = _decide_injection(config, checks)
        if target is not None:
            target_str = str(target)
            current_args = getattr(config, "args", [])
            if target_str not in current_args:
                current_args.append(target_str)


def _decide_injection(
    config: pytest.Config,
    checks_dir: Path,
) -> Path | None:
    """Decide which pykissembed check paths to append to ``config.args``.

    Returns
    -------
    Path | None
        * The whole ``checks_dir`` if the user passed ``--pykissembed-all``
          or a marker filter (``-m``).
        * A single file inside ``checks_dir`` whose stem matches a
          ``::NodeId`` filter pointing at a pykissembed check
          (smart-restrict).
        * ``None`` when the user's args already include a per-test filter
          that we cannot narrow further (``-k`` keyword, ``--deselect``,
          or any other filter that does not name a check stem).

    The decision tree in order:

    1. ``--pykissembed-all`` set → return the whole ``checks_dir``.
    2. ``-m <marker>`` present → return the whole ``checks_dir`` so the
       marker filter can narrow the run.
    3. Any ``::NodeId`` whose non-class part is one of the
       ``_CHECK_STEMS`` → return that single file.
    4. Any other filter present (``-k``, ``--deselect``) → return ``None``
       (the filter alone decides).
    5. Otherwise → return ``None``. The consumer did not ask for the
       battery; respect that.
    """
    if config.getoption("--pykissembed-all"):
        return checks_dir

    args: list[str] = list(getattr(config, "args", []))

    has_marker_filter = any(
        a == "-m" or a.startswith(("--markers", "-m=")) for a in args
    )
    if has_marker_filter:
        return checks_dir

    has_keyword_filter = any(
        a == "-k" or a.startswith(("-k=", "--keyword")) for a in args
    )
    has_deselect = any(a == "--deselect" or a.startswith("--deselect=") for a in args)

    # Check if the user already targeted a check file directly (either
    # via a NodeId or a bare path). If so, skip injection to avoid
    # double-collection — pytest collects each entry in config.args
    # independently, so appending a path the user already passed causes
    # the same test to run twice.
    for raw in args:
        # NodeIds always contain `::`. The first segment is the file path;
        # if its stem matches a known check, we can smart-restrict to that
        # file alone. This is what lets a user run *exactly* one check.
        if "::" in raw:
            head = raw.split("::", 1)[0]
            head_stem = Path(head).stem
            if head_stem in _CHECK_STEMS:
                candidate = checks_dir / f"{head_stem}.py"
                if candidate.is_file():
                    # Guard: only inject if the user's path does NOT
                    # already resolve to the same file inside checks_dir.
                    try:
                        resolved = Path(head).resolve()
                    except OSError:
                        resolved = None
                    if resolved is not None and resolved == candidate.resolve():
                        return None  # already on the CLI; don't re-inject
                    return candidate
            # NodeId present but does not name a pykissembed check. The
            # consumer is targeting something else — do not inject.
            return None

    # Also check for bare file paths (no ::) that already point at a
    # check file inside checks_dir.
    for raw in args:
        if "::" in raw or raw.startswith("-"):
            continue
        try:
            resolved = Path(raw).resolve()
        except (OSError, ValueError):
            continue
        if (
            resolved.is_file()
            and resolved.parent == checks_dir.resolve()
            and resolved.stem in _CHECK_STEMS
        ):
            return None  # user already targeted this check file

    if has_keyword_filter or has_deselect:
        return None

    return None


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


# Session-scoped dedup guard for non-init check files.
# Uses config.stash (pytest-idiomatic) instead of a module-level set
# to avoid leaking state across pytester sessions.
_collected_key = StashKey[set[Path]]()


@pytest.hookimpl(tryfirst=True)
def pytest_collect_file(
    file_path: Path, parent: pytest.Collector
) -> pytest.Module | None:
    """Collect pykissembed's check modules as test modules.

    This hook makes the check modules (``code_complexity.py``,
    ``comment_density.py``, etc.) discoverable by pytest in *any* consumer
    project — without the consumer needing to configure ``testpaths`` or
    copy test files. The modules are collected only if they live inside
    the installed ``pykissembed/checks/`` directory and their stem matches
    a known check module name.

    Registered with ``tryfirst=True`` so it runs *before* pytest's default
    ``python_files`` filter (which would reject files not matching
    ``test_*.py``). By returning a ``Module`` here we short-circuit the
    default collection for these files.

    **Double-collection guard:** ``pytest_collect_file`` is NOT a
    ``firstresult`` hook — pluggy calls ALL implementations and collects
    all non-None returns. If the user (or ``_decide_injection``) already
    passed this file on the CLI, pytest's default hook also returns a
    ``Module`` for it (via the ``isinitpath`` bypass of ``python_files``).
    To avoid collecting the same test twice, we defer to the default hook
    for init paths by returning ``None`` when
    ``parent.session.isinitpath(file_path)`` is ``True``.
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
    # If the file was explicitly passed on the CLI (or injected into
    # config.args by _decide_injection), defer to pytest's default
    # pytest_collect_file. The default hook skips the python_files
    # filter for init paths, so it will still collect the file — but
    # only once, not twice.
    if parent.session.isinitpath(file_path):
        return None
    # Dedup guard for non-init files (session-scoped via config.stash).
    collected = parent.config.stash.setdefault(_collected_key, set())
    resolved = file_path.resolve()
    if resolved in collected:
        return None
    collected.add(resolved)
    return pytest.Module.from_parent(parent, path=file_path)
