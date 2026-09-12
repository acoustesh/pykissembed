"""Reject static-analysis suppressions and typing casts in consumer source.

The check scans every Python file under the consumer project root, including
standalone scripts outside the configured pykissembed source paths. Test,
environment, dependency, build, VCS, and tool-cache directories are excluded.
Comments are tokenized so directive-like text in strings and docstrings is not
reported, while typing casts are resolved from their actual imports.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from pykissembed.config import get_config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_EXCLUDED_DIR_NAMES = frozenset({
    ".eggs",
    ".env",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pykissembed_cache",
    ".pyright",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "site-packages",
    # Tests are exempt by policy: a suppression in a test is usually pinning
    # deliberately-broken input, not hiding a real diagnostic.
    "tests",
    "venv",
})
_TYPE_IGNORE_PATTERN = re.compile(r"\btype\s*:\s*ignore\b", re.IGNORECASE)
_NOQA_PATTERN = re.compile(r"\bnoqa\b", re.IGNORECASE)
_RUFF_IGNORE_PATTERN = re.compile(r"\bruff\s*:\s*ignore\b", re.IGNORECASE)
_PYRIGHT_IGNORE_PATTERN = re.compile(r"\bpyright\s*:\s*ignore\b", re.IGNORECASE)
_TY_IGNORE_PATTERN = re.compile(r"\bty\s*:\s*ignore\b", re.IGNORECASE)
_MYPY_IGNORE_PATTERN = re.compile(r"\bmypy\s*:\s*ignore-errors\b", re.IGNORECASE)
_TYPING_MODULES = frozenset({"typing", "typing_extensions"})


@dataclass(frozen=True, order=True, slots=True)
class _Violation:
    """One forbidden suppression directive or typing cast."""

    path: str
    line: int
    column: int
    kind: str
    source: str


def _iter_project_py_files(root: Path) -> Iterator[Path]:
    """Yield project Python files outside excluded directories.

    Parameters
    ----------
    root
        Consumer project root.

    Yields
    ------
    Path
        Python files in deterministic path order.
    """
    for py_file in sorted(root.rglob("*.py")):
        relative = py_file.relative_to(root)
        # `[:-1]` drops the filename, so only *directory* components are matched
        # — a module named `tests.py` or `build.py` is still scanned.
        if any(part in _EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
            continue
        if py_file.is_file():
            yield py_file


def _comment_violations(source: str, relative_path: str) -> list[_Violation]:
    """Return suppression directives found in *source*'s comments.

    Parameters
    ----------
    source
        Full text of the Python file.
    relative_path
        Project-relative path recorded on each violation.

    Returns
    -------
    list[_Violation]
        One entry per directive match, in token order.
    """
    violations: list[_Violation] = []
    # Scanning COMMENT tokens rather than raw lines is what keeps directive-like
    # text inside strings and docstrings from being reported as a violation.
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        # No break: one comment carrying both directives is reported twice, so
        # fixing it means removing both rather than rediscovering the second.
        for pattern, kind in (
            (_TYPE_IGNORE_PATTERN, "type-ignore"),
            (_NOQA_PATTERN, "noqa"),
            (_RUFF_IGNORE_PATTERN, "ruff-ignore"),
            (_PYRIGHT_IGNORE_PATTERN, "pyright-ignore"),
            (_TY_IGNORE_PATTERN, "ty-ignore"),
            (_MYPY_IGNORE_PATTERN, "mypy-ignore-errors"),
        ):
            if pattern.search(token.string):
                violations.append(
                    _Violation(
                        path=relative_path,
                        line=token.start[0],
                        # tokenize columns are 0-based; editors count from 1.
                        column=token.start[1] + 1,
                        kind=kind,
                        source=token.line.strip(),
                    )
                )
    return violations


def _from_import_cast_names(node: ast.ImportFrom) -> set[str]:
    """Return the local names one ``from typing import ...`` binds to ``cast``.

    Parameters
    ----------
    node
        An ``ImportFrom`` already known to target a typing module.

    Returns
    -------
    set[str]
        Local names that refer to ``typing.cast``; empty when the statement
        imports something else.
    """
    names: set[str] = set()
    for alias in node.names:
        if alias.name == "cast":
            names.add(alias.asname or alias.name)
        elif alias.name == "*":
            # A star import binds `cast` without ever naming it.
            names.add("cast")
    return names


def _cast_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Collect the names under which ``typing.cast`` is reachable in *tree*.

    Parameters
    ----------
    tree
        Parsed module to inspect.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(direct_names, module_names)`` — bare names bound to ``cast``, and
        names bound to a typing *module* for qualified ``module.cast`` calls.
    """
    direct_names: set[str] = set()
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names.update(
                alias.asname or alias.name for alias in node.names if alias.name in _TYPING_MODULES
            )
        elif isinstance(node, ast.ImportFrom) and node.module in _TYPING_MODULES:
            direct_names.update(_from_import_cast_names(node))
    return direct_names, module_names


def _cast_violations(
    tree: ast.Module,
    relative_path: str,
    source_lines: list[str],
) -> list[_Violation]:
    """Return every ``typing.cast`` call site in *tree*.

    Parameters
    ----------
    tree
        Parsed module to inspect.
    relative_path
        Project-relative path recorded on each violation.
    source_lines
        The file's lines, used to quote the offending source.

    Returns
    -------
    list[_Violation]
        One entry per cast call, in AST walk order.
    """
    # Imports are collected in a walk of their own before any call is examined:
    # a function-level `from typing import cast` can appear *after* the call
    # that uses it, so a single fused pass would miss those names.
    direct_names, module_names = _cast_names(tree)
    violations: list[_Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        direct_cast = isinstance(func, ast.Name) and func.id in direct_names
        # Catches a qualified call through the module or any alias of it,
        # e.g. `typing.cast(...)` or `import typing as t; t.cast(...)`.
        qualified_cast = (
            isinstance(func, ast.Attribute)
            and func.attr == "cast"
            and isinstance(func.value, ast.Name)
            and func.value.id in module_names
        )
        if not direct_cast and not qualified_cast:
            continue
        violations.append(
            _Violation(
                path=relative_path,
                line=node.lineno,
                column=node.col_offset + 1,
                kind="typing.cast",
                source=source_lines[node.lineno - 1].strip(),
            )
        )
    return violations


def _scan_file(py_file: Path, *, root: Path) -> list[_Violation]:
    """Return forbidden constructs found in one Python file.

    Parameters
    ----------
    py_file
        Python source file to inspect.
    root
        Consumer project root used to render a stable relative path.

    Returns
    -------
    list[_Violation]
        Token- and syntax-aware violations sorted by source location.
    """
    # tokenize.open honours a PEP 263 coding declaration; Path.read_text()
    # would assume UTF-8 and mis-decode files that declare another encoding.
    with tokenize.open(py_file) as source_file:
        source = source_file.read()
    relative_path = py_file.relative_to(root).as_posix()
    tree = ast.parse(source, filename=str(py_file))

    violations = _comment_violations(source, relative_path)
    violations.extend(_cast_violations(tree, relative_path, source.splitlines()))
    # order=True on _Violation sorts by field declaration order, so this lands
    # in (path, line, column) order without an explicit key.
    return sorted(violations)


def _find_violations(root: Path) -> list[_Violation]:
    """Return all forbidden constructs below a consumer project root.

    Parameters
    ----------
    root
        Consumer project root to inspect.

    Returns
    -------
    list[_Violation]
        Violations sorted by project-relative path and source location.
    """
    violations: list[_Violation] = []
    for py_file in _iter_project_py_files(root):
        violations.extend(_scan_file(py_file, root=root))
    return sorted(violations)


@pytest.mark.lint
def test_no_suppressions_or_casts() -> None:
    """Consumer production Python must not hide diagnostics or use typing casts."""
    root = get_config().root.resolve()
    try:
        violations = _find_violations(root)
    except (OSError, SyntaxError, UnicodeError, tokenize.TokenError) as exc:
        pytest.fail(
            f"Suppression/cast scan could not inspect the consumer project: {exc}",
            pytrace=False,
        )

    if not violations:
        return

    details = "\n".join(
        f"  {item.path}:{item.line}:{item.column}: {item.kind}: {item.source}"
        for item in violations
    )
    pytest.fail(
        "Suppression/cast gate failed. Remove every static-analysis ignore directive "
        f"and typing cast from project Python:\n{details}",
        pytrace=False,
    )
