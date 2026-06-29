"""Complexity map loaders for refactor index integration.

Ported from ``mega-scrapper/tests/similarity/complexity.py``. Directory
scanning uses :func:`pykissembed.paths.resolve_paths` instead of hardcoded
``MEGA_SCRAPPER_DIR``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from pykissembed.config import get_config
from pykissembed.paths import _should_skip, resolve_paths

if TYPE_CHECKING:
    from pathlib import Path


def _extract_block_tuple(block: object) -> tuple[str, int, int]:
    """Validate and normalize a radon block object to a typed tuple.

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

    blocks = cast("list[object]", blocks_raw)
    return [_extract_block_tuple(block) for block in blocks]


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
    """
    if metric == "cc":
        try:
            return _cc_complexities_from_source(file_path.read_text(encoding="utf-8"))
        except SyntaxError:
            return []

    # cognitive
    from complexipy import file_complexity

    try:
        result = file_complexity(str(file_path))
    except Exception:
        return []
    return [(f.name, f.line_start, f.complexity) for f in result.functions]


def _load_dir_complexity(
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

    glob_fn = directory.rglob if recursive else directory.glob
    for py_file in glob_fn("*.py"):
        if py_file.name.startswith("__") or _should_skip(py_file):
            continue
        rel = py_file.relative_to(directory)
        key_prefix = f"{prefix}{rel}"
        for func_name, _, cc in _get_complexities(py_file, "cc"):
            cc_map[f"{key_prefix}:{func_name}"] = cc
        for func_name, _, cog in _get_complexities(py_file, "cognitive"):
            cog_map[f"{key_prefix}:{func_name}"] = cog

    return cc_map, cog_map


def load_complexity_maps(directory: Path | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """Load CC and COG complexity maps for functions in a directory.

    Returns
    -------
    tuple[dict[str, int], dict[str, int]]
        A ``(cc_map, cog_map)`` pair.
    """
    if directory is None:
        paths = resolve_paths()
        if not paths:
            return {}, {}
        directory = paths[0]
    return _load_dir_complexity(directory)


def load_all_complexity_maps() -> tuple[dict[str, int], dict[str, int]]:
    """Load CC and COG complexity maps for ALL configured source directories.

    Scans recursively through every directory returned by
    :func:`pykissembed.paths.resolve_paths`.

    Returns
    -------
    tuple[dict[str, int], dict[str, int]]
        A ``(cc_map, cog_map)`` pair.
    """
    root = get_config().root
    cc_map: dict[str, int] = {}
    cog_map: dict[str, int] = {}

    for base_dir in resolve_paths():
        rel_dir = (
            str(base_dir.relative_to(root)) if base_dir.is_relative_to(root) else str(base_dir)
        )
        sub_cc, sub_cog = _load_dir_complexity(
            base_dir,
            prefix=f"{rel_dir}/",
            recursive=True,
        )
        cc_map.update(sub_cc)
        cog_map.update(sub_cog)

    return cc_map, cog_map
