"""Complexity map loaders for refactor index integration.

Ported from ``mega-scrapper/tests/similarity/complexity.py``. Directory
scanning uses :func:`pykissembed.paths.resolve_paths` instead of hardcoded
``MEGA_SCRAPPER_DIR``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from pykissembed.config import get_config
from pykissembed.paths import resolve_paths, should_skip

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class AnalyzerError(RuntimeError):
    """Normalized failure raised by an untyped complexity analyzer."""


def call_analyzer(
    analyzer: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    """Call an untyped analyzer and normalize its undocumented failures.

    Parameters
    ----------
    analyzer : Callable[..., object]
        Untyped third-party analyzer to call.
    *args : object
        Positional arguments forwarded to *analyzer*.
    **kwargs : object
        Keyword arguments forwarded to *analyzer*.

    Returns
    -------
    object
        The analyzer's untrusted return value, for caller-side validation.

    Raises
    ------
    AnalyzerError
        If the third-party analyzer raises an ordinary exception.
    """
    try:
        return analyzer(*args, **kwargs)
    except Exception as exc:
        msg = f"{getattr(analyzer, '__name__', 'analyzer')} failed"
        raise AnalyzerError(msg) from exc


def _extract_block_tuple(block: object) -> tuple[str, int, int]:
    """Validate and normalize a radon block object to a typed tuple.

    Parameters
    ----------
    block : object
        Untyped radon block from ``cc_visit`` output.

    Returns
    -------
    tuple[str, int, int]
        A ``(name, lineno, complexity)`` tuple.

    Raises
    ------
    TypeError
        If any required block attribute has an unexpected type.
    """
    name = getattr(block, "name", None)
    lineno = getattr(block, "lineno", None)
    complexity = getattr(block, "complexity", None)

    if not isinstance(name, str):
        msg = "radon block.name must be str"
        raise TypeError(msg)
    if not isinstance(lineno, int):
        msg = "radon block.lineno must be int"
        raise TypeError(msg)
    if not isinstance(complexity, int):
        msg = "radon block.complexity must be int"
        raise TypeError(msg)

    return name, lineno, complexity


def _cc_complexities_from_source(source_code: str) -> list[tuple[str, int, int]]:
    """Run radon cc_visit through a validated, fully typed boundary.

    Parameters
    ----------
    source_code : str
        Python source text to analyse.

    Returns
    -------
    list[tuple[str, int, int]]
        Cyclomatic complexity tuples as ``(name, lineno, complexity)``.

    Raises
    ------
    TypeError
        If ``radon.complexity.cc_visit`` has an invalid shape or return type.
    """
    complexity_module = import_module("radon.complexity")
    cc_visit_obj = getattr(complexity_module, "cc_visit", None)
    if not callable(cc_visit_obj):
        msg = "radon.complexity.cc_visit must be callable"
        raise TypeError(msg)

    blocks_raw = cc_visit_obj(source_code)
    if not isinstance(blocks_raw, list):
        msg = "radon cc_visit must return a list"
        raise TypeError(msg)

    return [_extract_block_tuple(block) for block in blocks_raw]


def _get_complexities(
    file_path: Path,
    metric: str,
) -> list[tuple[str, int, int]]:
    """Get complexity metrics for all functions in a file.

    Parameters
    ----------
    file_path : Path
        Python source file to analyse.
    metric : str
        ``"cc"`` for cyclomatic complexity (radon) or
        ``"cognitive"`` for cognitive complexity (complexipy).

    Returns
    -------
    list[tuple[str, int, int]]
        List of ``(function_name, start_line, complexity)`` tuples.

    Raises
    ------
    TypeError
        If complexipy exposes an invalid API or result shape.
    """
    if metric == "cc":
        try:
            return _cc_complexities_from_source(file_path.read_text(encoding="utf-8"))
        except SyntaxError:
            return []

    # Lazy: complexipy is a compiled analyzer; defer loading for callers that
    # only need cyclomatic complexity. Validate the untyped module boundary.
    complexity_module = import_module("complexipy")
    file_complexity = getattr(complexity_module, "file_complexity", None)
    if not callable(file_complexity):
        msg = "complexipy.file_complexity must be callable"
        raise TypeError(msg)

    try:
        result = call_analyzer(file_complexity, str(file_path))
    except AnalyzerError:
        # Invalid user source degrades to no cognitive-complexity data.
        return []
    functions = getattr(result, "functions", None)
    if not isinstance(functions, list):
        msg = "complexipy result.functions must be a list"
        raise TypeError(msg)
    values: list[tuple[str, int, int]] = []
    for function in functions:
        name = getattr(function, "name", None)
        line_start = getattr(function, "line_start", None)
        complexity = getattr(function, "complexity", None)
        if not isinstance(name, str) or not isinstance(line_start, int):
            msg = "complexipy function name and line_start must be typed"
            raise TypeError(msg)
        if not isinstance(complexity, int):
            msg = "complexipy function complexity must be int"
            raise TypeError(msg)
        values.append((name, line_start, complexity))
    return values


def _scan_complexity_directory(
    directory: Path,
    prefix: str = "",
    *,
    recursive: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    """Load complexity maps for a single directory.

    Parameters
    ----------
    directory : Path
        Directory to scan.
    prefix : str
        Prefix for baseline keys.
    recursive : bool
        When ``True`` use ``rglob`` to scan subdirectories recursively.

    Returns
    -------
    tuple[dict[str, int], dict[str, int]]
        A ``(cc_map, cog_map)`` pair.
    """
    cc_map: dict[str, int] = {}
    cog_map: dict[str, int] = {}

    # recursive=True (used by load_all_complexity_maps) walks subdirectories
    # via rglob; the non-recursive glob() default matches the shallow scan
    # that load_complexity_maps()'s single-directory fallback expects.
    glob_fn = directory.rglob if recursive else directory.glob
    for py_file in glob_fn("*.py"):
        if py_file.name.startswith("__") or should_skip(py_file, directory):
            continue
        rel = py_file.relative_to(directory)
        key_prefix = f"{prefix}{rel}"
        for func_name, _, cc in _get_complexities(py_file, "cc"):
            cc_map[f"{key_prefix}:{func_name}"] = cc
        for func_name, _, cog in _get_complexities(py_file, "cognitive"):
            cog_map[f"{key_prefix}:{func_name}"] = cog

    return cc_map, cog_map


def load_complexity_maps(directory: Path | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """Return metrics for one directory, defaulting to the first configured path.

    Unlike :func:`load_all_complexity_maps`, this compatibility entry point
    performs a shallow scan and never aggregates multiple configured roots.

    Parameters
    ----------
    directory : Path | None
        Directory to scan; defaults to the first configured path.

    Returns
    -------
    tuple[dict[str, int], dict[str, int]]
        CC and COG mappings for the selected directory only.
    """
    if directory is not None:
        return _scan_complexity_directory(directory)
    paths = resolve_paths()
    return _scan_complexity_directory(paths[0]) if paths else ({}, {})


def load_all_complexity_maps() -> tuple[dict[str, int], dict[str, int]]:
    """Aggregate recursive metrics across every configured source directory.

    Scans recursively through every directory returned by
    :func:`pykissembed.paths.resolve_paths`.

    Returns
    -------
    tuple[dict[str, int], dict[str, int]]
        Merged CC and COG mappings with project-relative path prefixes.
    """
    root = get_config().root
    cc_map: dict[str, int] = {}
    cog_map: dict[str, int] = {}

    for base_dir in resolve_paths():
        rel_dir = (
            str(base_dir.relative_to(root)) if base_dir.is_relative_to(root) else str(base_dir)
        )
        sub_cc, sub_cog = _scan_complexity_directory(
            base_dir,
            prefix=f"{rel_dir}/",
            recursive=True,
        )
        cc_map.update(sub_cc)
        cog_map.update(sub_cog)

    return cc_map, cog_map
