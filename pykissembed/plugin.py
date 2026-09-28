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
import sys
from importlib import import_module, util
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

import pytest
from _pytest.stash import StashKey

from pykissembed.config import get_config
from pykissembed.paths import resolve_paths

if TYPE_CHECKING:
    from collections.abc import Callable

    from pykissembed.similarity.pca import PCACacheEntry
    from pykissembed.similarity.types import FunctionInfo

# Modules inside pykissembed/checks/ that contain test classes/functions.
# These are collected by the plugin and run in the consumer's pytest session.
_CHECK_MODULES = [
    "code_complexity",
    "code_similarity",
    "comment_density",
    "docstring_format",
    "jev_comment_audit",
    "jev_docstring_audit",
    "lint_typecheck",
    "no_suppressions",
]

# Set of file stems that the plugin should collect as test modules.
# Used by :func:`pytest_collect_file` to decide whether a .py file inside
# the installed pykissembed package is a check module.
_CHECK_STEMS = frozenset(_CHECK_MODULES)


def _load_callable(module_name: str, attribute: str) -> Callable[..., object]:
    """Load and validate one callable from a lazily imported module.

    Returns
    -------
    Callable[..., object]
        The requested callable.

    Raises
    ------
    TypeError
        If the requested attribute is not callable.
    """
    module = import_module(module_name)
    value = getattr(module, attribute, None)
    if not callable(value):
        msg = f"{module_name}.{attribute} must be callable"
        raise TypeError(msg)
    return value


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Return whether *value* is a dictionary with string keys.

    Returns
    -------
    bool
        Whether the value has the required dictionary shape.
    """
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _is_function_info(value: object) -> TypeGuard[FunctionInfo]:
    """Validate the stable fields consumed from a lazily loaded function record.

    Returns
    -------
    bool
        Whether the object exposes the required typed record fields.
    """
    return (
        isinstance(getattr(value, "name", None), str)
        and isinstance(getattr(value, "file", None), str)
        and isinstance(getattr(value, "start_line", None), int)
        and isinstance(getattr(value, "end_line", None), int)
        and isinstance(getattr(value, "loc", None), int)
        and isinstance(getattr(value, "hash", None), str)
    )


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
        "--allow-cloud-embeddings",
        action="store_true",
        default=False,
        help="Allow similarity checks to populate missing embeddings through cloud APIs.",
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
    """Fixture returning the ``--update-baselines`` flag value.

    Returns
    -------
    bool
        ``True`` if ``--update-baselines`` was passed on the command line.
    """
    return bool(request.config.getoption("--update-baselines"))


@pytest.fixture
def cached_only(request: pytest.FixtureRequest) -> bool:
    """Fixture returning the effective cache-only setting.

    Returns
    -------
    bool
        ``True`` when ``--cached-only`` was passed or configuration sets
        ``cached_only = true``; otherwise ``False`` (the default: populate
        missing embeddings for all providers). ``--allow-cloud-embeddings``
        overrides configuration and returns ``False``.
    """
    if request.config.getoption("--allow-cloud-embeddings"):
        return False
    return bool(request.config.getoption("--cached-only") or get_config().cached_only)


@pytest.fixture(scope="session")
def pykissembed_paths() -> list[Path]:
    """Resolved list of source directories from ``[tool.pykissembed]``.

    Returns
    -------
    list[Path]
        Resolved, existing directories. Empty list if nothing configured
        or nothing exists on disk.
    """
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

    Raises
    ------
    TypeError
        If the lazy storage boundary returns an invalid shape.
    """
    # Keep the NumPy-backed storage module out of sessions that never request
    # similarity fixtures, while validating its untyped dynamic boundary.
    load_minimal_baselines = _load_callable(
        "pykissembed.similarity.storage",
        "load_minimal_baselines",
    )
    baselines = load_minimal_baselines()
    if not _is_str_object_dict(baselines):
        msg = "load_minimal_baselines must return dict[str, object]"
        raise TypeError(msg)
    return baselines


@pytest.fixture(scope="session")
def shared_functions(
    shared_baselines: dict[str, object],
) -> list[FunctionInfo]:
    """Session-scoped list of FunctionInfo objects extracted from workspace.

    Returns
    -------
    list[FunctionInfo]
        The extracted function info objects.

    Raises
    ------
    TypeError
        If the lazy AST boundary returns an invalid shape.
    """
    # Requesting this fixture sequences baseline loading before extraction.
    _ = shared_baselines
    # Lazy: same rationale as shared_baselines above.
    extract_all_function_infos = _load_callable(
        "pykissembed.similarity.ast_helpers",
        "extract_all_function_infos",
    )
    functions = extract_all_function_infos(min_loc=1)
    if not isinstance(functions, list) or not all(_is_function_info(item) for item in functions):
        msg = "extract_all_function_infos must return list[FunctionInfo]"
        raise TypeError(msg)
    return [item for item in functions if _is_function_info(item)]


@pytest.fixture(scope="session")
def pca_cache() -> dict[str, PCACacheEntry]:
    """Session-scoped cache for fitted PCA models.

    Returns
    -------
    dict[str, PCACacheEntry]
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

    Raises
    ------
    pytest.UsageError
        If both cache-only and cloud-population flags are supplied.
    """
    if config.getoption("--cached-only") and config.getoption("--allow-cloud-embeddings"):
        msg = "--cached-only and --allow-cloud-embeddings are mutually exclusive"
        raise pytest.UsageError(msg)

    config.addinivalue_line(
        "markers",
        "lint: lint + type-check + no-suppressions gates",
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
        "jev: Jev-judged docstring/comment quality (live OpenRouter decisions)",
    )
    config.addinivalue_line(
        "markers",
        "similarity: embedding-based near-duplicate detection",
    )
    config.addinivalue_line(
        "markers",
        "experimental: unstable APIs — refactor_index, file_split",
    )

    # Ensure source dirs are importable for similarity/refactor tests.
    # `resolve_paths` is imported unconditionally at module level above, so
    # by the time this hook runs it has already succeeded — no import
    # guard needed here.
    if not os.environ.get("PYQTEST_SKIP_PATH_INJECTION"):
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
                _ = current_args.append(target_str)


def _decide_injection(
    config: pytest.Config,
    checks_dir: Path,
) -> Path | str | None:
    """Decide which pykissembed check paths to append to ``config.args``.

    Parameters
    ----------
    config : pytest.Config
        Active pytest configuration; read for CLI options and raw
        invocation arguments.
    checks_dir : Path
        Installed ``pykissembed/checks/`` directory.

    Returns
    -------
    Path | str | None
        * The whole ``checks_dir`` if the user passed ``--pykissembed-all``,
          ``--collect-only``/``--co``, or a marker filter (``-m``).
        * A single file inside ``checks_dir`` whose stem matches a
          ``::NodeId`` filter pointing at a pykissembed check
          (smart-restrict). When the user's NodeId included a class or
          test selector (``::TestFoo`` / ``::TestFoo::test_bar``), that
          selector is preserved on the returned string so only the
          matching test(s) are collected — not every test in the module.
        * ``None`` when the user's args already include a per-test filter
          that we cannot narrow further (``-k`` keyword, ``--deselect``,
          or any other filter that does not name a check stem).

    The decision tree in order:

    1. ``--pykissembed-all`` set → return the whole ``checks_dir``.
    2. ``-m <marker>`` present → return the whole ``checks_dir`` so the
       marker filter can narrow the run.
    3. Any ``::NodeId`` whose non-class part is one of the
       ``_CHECK_STEMS`` → return that single file (smart-restrict).
    4. Any other filter present (``-k``, ``--deselect``) → return ``None``
       (the filter alone decides).
    5. Otherwise (no marker/NodeId/keyword/deselect at all) →
       ``--collect-only``/``--co`` set → return the whole ``checks_dir``.
       This is what IDE test explorers (e.g. VS Code's Python extension)
       use to *discover* tests without running them, and it's normally a
       bare invocation with no other filter. Without this rule, that
       bare discovery collects nothing and pykissembed's checks never
       appear in the test tree. Running an individual discovered test
       later passes its NodeId explicitly (no ``--collect-only``), which
       is handled by rule 3/the already-on-CLI guard above, so this rule
       only affects what shows up during discovery, never what executes.
    6. Otherwise → return ``None``. The consumer did not ask for the
       battery; respect that.
    """
    # `config.args` is pytest's *resolved collection roots* (positional
    # paths, falling back to ini `testpaths` when none are given) — it
    # never contains option flags, since argparse consumes those before
    # `config.args` is populated. `invocation_params.args` is the raw argv
    # pytest was invoked with (unaffected by ini `addopts`/`testpaths`),
    # so it's the only source that actually reflects flags like `-m`/`-k`/
    # `--deselect` the user typed.
    args: list[str] = list(config.invocation_params.args)

    if config.getoption("--pykissembed-all") or _has_marker_filter(args):
        return checks_dir

    node_id = _first_node_id(args)
    if node_id is not None:
        return _smart_restricted_target(node_id, checks_dir)

    if _already_targets_check_file(args, checks_dir):
        return None

    if _has_keyword_filter(args) or _has_deselect_filter(args):
        return None

    # Nothing narrowed the request at all (no marker, NodeId, keyword, or
    # deselect filter). If this is a pure discovery pass (--collect-only),
    # show the full battery so IDE test explorers populate their tree —
    # no tests are actually *executed* by a --collect-only invocation.
    if config.getoption("--collect-only"):
        return checks_dir

    return None


def _has_marker_filter(args: list[str]) -> bool:
    """Return whether raw pytest arguments contain a marker filter.

    Parameters
    ----------
    args : list[str]
        Raw pytest invocation arguments.

    Returns
    -------
    bool
        ``True`` when a marker-filter option is present.
    """
    return any(arg == "-m" or arg.startswith(("--markers", "-m=")) for arg in args)


def _has_keyword_filter(args: list[str]) -> bool:
    """Return whether raw pytest arguments contain a keyword filter.

    Parameters
    ----------
    args : list[str]
        Raw pytest invocation arguments.

    Returns
    -------
    bool
        ``True`` when a keyword-filter option is present.
    """
    return any(arg == "-k" or arg.startswith(("-k=", "--keyword")) for arg in args)


def _has_deselect_filter(args: list[str]) -> bool:
    """Return whether raw pytest arguments contain a deselect filter.

    Parameters
    ----------
    args : list[str]
        Raw pytest invocation arguments.

    Returns
    -------
    bool
        ``True`` when a deselect option is present.
    """
    return any(arg == "--deselect" or arg.startswith("--deselect=") for arg in args)


def _first_node_id(args: list[str]) -> str | None:
    """Return the first NodeId from raw pytest arguments, if one is present.

    Parameters
    ----------
    args : list[str]
        Raw pytest invocation arguments.

    Returns
    -------
    str | None
        The first argument containing a NodeId separator, or ``None``.
    """
    return next((arg for arg in args if "::" in arg), None)


def _smart_restricted_target(node_id: str, checks_dir: Path) -> Path | str | None:
    """Map a check NodeId to its installed check file, preserving its selector.

    Parameters
    ----------
    node_id : str
        A pytest NodeId containing a ``::`` separator.
    checks_dir : Path
        Installed ``pykissembed/checks/`` directory.

    Returns
    -------
    Path | str | None
        The matching check file, a selector-qualified target, or ``None``
        when the NodeId cannot be safely injected.
    """
    head, _, selector = node_id.partition("::")
    candidate = _check_candidate(head, checks_dir)
    if candidate is None or _is_same_file(head, candidate):
        return None
    return f"{candidate}::{selector}" if selector else candidate


def _check_candidate(path: str, checks_dir: Path) -> Path | None:
    """Return the installed check file identified by *path*'s stem, if any.

    Parameters
    ----------
    path : str
        File path portion of a NodeId (before the first ``::``).
    checks_dir : Path
        Installed ``pykissembed/checks/`` directory.

    Returns
    -------
    Path | None
        The existing installed check file, or ``None`` when the stem is unknown.
    """
    stem = Path(path).stem
    if stem not in _CHECK_STEMS:
        return None
    candidate = checks_dir / f"{stem}.py"
    return candidate if candidate.is_file() else None


def _is_same_file(path: str, candidate: Path) -> bool:
    """Return whether *path* resolves to the installed check *candidate*.

    Parameters
    ----------
    path : str
        User-supplied path argument.
    candidate : Path
        Installed check file to compare against.

    Returns
    -------
    bool
        ``True`` when *path* resolves to *candidate*.
    """
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return resolved == candidate.resolve()


def _already_targets_check_file(args: list[str], checks_dir: Path) -> bool:
    """Return whether a bare path already selects an installed check file.

    Parameters
    ----------
    args : list[str]
        Raw pytest invocation arguments.
    checks_dir : Path
        Installed ``pykissembed/checks/`` directory.

    Returns
    -------
    bool
        ``True`` when an argument names an installed check file directly.
    """
    return any(_is_installed_check_path(arg, checks_dir) for arg in args)


def _is_installed_check_path(path: str, checks_dir: Path) -> bool:
    """Return whether *path* is a bare path to a known installed check file.

    Parameters
    ----------
    path : str
        Candidate argument (skipped when it names a selector or option).
    checks_dir : Path
        Installed ``pykissembed/checks/`` directory.

    Returns
    -------
    bool
        ``True`` when *path* resolves to a known check in *checks_dir*.
    """
    if "::" in path or path.startswith("-"):
        return False
    try:
        resolved = Path(path).resolve()
    except OSError, ValueError:
        return False
    return (
        resolved.is_file()
        and resolved.parent == checks_dir.resolve()
        and resolved.stem in _CHECK_STEMS
    )


def _checks_dir() -> Path | None:
    """Return the path to the installed ``pykissembed/checks/`` directory.

    Returns
    -------
    Path | None
        The directory containing ``pykissembed/checks/__init__.py``, or
        ``None`` if ``pykissembed.checks`` can't be imported or is a
        namespace package with no ``__file__``.
    """
    # Inspect the package spec without importing it: a partial installation
    # must degrade to no auto-collection rather than crash every pytest run.
    checks_spec = util.find_spec("pykissembed.checks")
    if checks_spec is None or checks_spec.origin is None:  # pragma: no cover — defensive
        return None
    return Path(checks_spec.origin).parent


# Session-scoped dedup guard for non-init check files.
# Uses config.stash (pytest-idiomatic) instead of a module-level set
# to avoid leaking state across pytester sessions.
_collected_key = StashKey[set[Path]]()


@pytest.hookimpl(tryfirst=True)
def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> pytest.Module | None:
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

    Returns
    -------
    pytest.Module | None
        A ``Module`` collector for *file_path* if it is a not-yet-collected,
        non-init-path check module living under ``pykissembed/checks/``;
        ``None`` otherwise (deferring to pytest's default collection).
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
        _ = file_path.relative_to(checks)
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
