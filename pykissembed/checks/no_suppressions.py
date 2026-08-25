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
    "tests",
    "venv",
})
_TYPE_IGNORE_PATTERN = re.compile(r"\btype\s*:\s*ignore\b", re.IGNORECASE)
_NOQA_PATTERN = re.compile(r"\bnoqa\b", re.IGNORECASE)
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
        if any(part in _EXCLUDED_DIR_NAMES for part in relative.parts[:-1]):
            continue
        if py_file.is_file():
            yield py_file


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
    with tokenize.open(py_file) as source_file:
        source = source_file.read()
    relative_path = py_file.relative_to(root).as_posix()
    source_lines = source.splitlines()
    tree = ast.parse(source, filename=str(py_file))
    violations: list[_Violation] = []

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        for pattern, kind in (
            (_TYPE_IGNORE_PATTERN, "type-ignore"),
            (_NOQA_PATTERN, "noqa"),
        ):
            if pattern.search(token.string):
                violations.append(
                    _Violation(
                        path=relative_path,
                        line=token.start[0],
                        column=token.start[1] + 1,
                        kind=kind,
                        source=token.line.strip(),
                    )
                )

    direct_cast_names: set[str] = set()
    typing_module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            typing_module_names.update(
                alias.asname or alias.name for alias in node.names if alias.name in _TYPING_MODULES
            )
        elif isinstance(node, ast.ImportFrom) and node.module in _TYPING_MODULES:
            for alias in node.names:
                if alias.name == "cast":
                    direct_cast_names.add(alias.asname or alias.name)
                elif alias.name == "*":
                    direct_cast_names.add("cast")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        direct_cast = isinstance(func, ast.Name) and func.id in direct_cast_names
        qualified_cast = (
            isinstance(func, ast.Attribute)
            and func.attr == "cast"
            and isinstance(func.value, ast.Name)
            and func.value.id in typing_module_names
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
        "Suppression/cast gate failed. Remove every type-ignore directive, "
        f"noqa directive, and typing cast from project Python:\n{details}",
        pytrace=False,
    )
