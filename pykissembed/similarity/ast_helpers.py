"""AST helpers for extracting function information from Python files.

Ported from ``mega-scrapper/tests/similarity/ast_helpers.py``. The main
adaptation is that directory scanning uses ``pykissembed.paths.resolve_paths()``
instead of hardcoded ``MEGA_SCRAPPER_DIR``.
"""

from __future__ import annotations

import ast
import hashlib
import io
import tokenize
from typing import TYPE_CHECKING

from pykissembed.config import get_config
from pykissembed.paths import _should_skip, resolve_paths
from pykissembed.similarity.types import FunctionInfo

if TYPE_CHECKING:
    from pathlib import Path


def get_function_text(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    source: str,
) -> str:
    """Extract the full text of a function or class from source code.

    Returns
    -------
        The full text of the function or class.
    """
    lines = source.splitlines()
    start = node.lineno - 1
    end = node.end_lineno or start + 1
    return "\n".join(lines[start:end])


def normalize_ast_tokens(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    """Produce deterministic token sequence from function or class AST.

    Returns
    -------
        The AST dump string.
    """
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of text content.

    Returns
    -------
        The hex digest of the SHA256 hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_text_for_embedding(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    source: str,
) -> str:
    """Extract signature + docstring + comment lines for text embedding.

    This extracts:
    - Decorators and function/class signature
    - Docstring (if present)
    - All lines containing # comments (with their context)

    Returns
    -------
        Text suitable for embedding (signature + docstring + comments)
    """
    lines = source.splitlines()
    start_line = node.lineno  # 1-indexed
    end_line = node.end_lineno or start_line
    parts = _decorator_lines(node, lines, start_line)
    parts.extend(_signature_lines(node, lines, start_line))

    docstring = ast.get_docstring(node) or ""
    if docstring:
        parts.append(f'"""{docstring}"""')

    comment_lines = _comment_lines(lines, start_line, end_line)
    if comment_lines:
        _append_unique_comment_lines(parts, comment_lines)
    return "\n".join(parts)


def _decorator_lines(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    lines: list[str],
    start_line: int,
) -> list[str]:
    """Return source lines from each decorator through the definition line.

    Returns
    -------
    list[str]
        Decorator lines in their original range-expansion order.
    """
    decorator_lines: list[str] = []
    # Preserve overlapping ranges for multiple decorators: that is the
    # established embedding input rather than a source-normalization step.
    for decorator in node.decorator_list:
        if decorator.lineno < start_line:
            decorator_lines.extend(lines[line_number - 1] for line_number in range(decorator.lineno, start_line))
    return decorator_lines


def _signature_lines(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    lines: list[str],
    start_line: int,
) -> list[str]:
    """Return definition lines up to, but not including, the first body node.

    Returns
    -------
    list[str]
        Source lines representing the definition signature.
    """
    signature_end = start_line
    if node.body:
        signature_end = node.body[0].lineno - 1
    return [lines[line_number - 1] for line_number in range(start_line, signature_end + 1)]


def _comment_lines(lines: list[str], start_line: int, end_line: int) -> list[str]:
    """Return tokenized comments, with text scanning for incomplete source.

    Returns
    -------
    list[str]
        Comment-bearing source lines in their original order.
    """
    func_source = "\n".join(lines[start_line - 1 : end_line])
    try:
        comment_lines: list[str] = []
        tokens = tokenize.generate_tokens(io.StringIO(func_source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                # Token coordinates are relative to ``func_source``, so translate
                # them back to the file before retrieving the original line.
                actual_line_index = start_line - 1 + token.start[0] - 1
                if actual_line_index < len(lines):
                    comment_lines.append(lines[actual_line_index])
    except tokenize.TokenError:
        # Only incomplete token streams need the less precise text fallback;
        # normal tokenization avoids treating hashes inside strings as comments.
        return _fallback_comment_lines(lines, start_line, end_line)
    else:
        return comment_lines


def _fallback_comment_lines(lines: list[str], start_line: int, end_line: int) -> list[str]:
    """Return lines containing ``#`` when tokenization cannot complete.

    Returns
    -------
    list[str]
        Candidate comment lines selected by text matching.
    """
    return [
        lines[line_index]
        for line_index in range(start_line - 1, end_line)
        if line_index < len(lines) and "#" in lines[line_index]
    ]


def _append_unique_comment_lines(parts: list[str], comment_lines: list[str]) -> None:
    """Append comment lines that are not already represented in *parts*."""
    seen = set(parts)
    for comment_line in comment_lines:
        if comment_line not in seen:
            parts.append(comment_line)
            seen.add(comment_line)


def _count_executable_lines(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    source: str,
) -> int:
    """Count lines of executable code, excluding docstrings, comments, and blanks.

    Returns
    -------
        The number of executable lines.
    """
    lines = source.splitlines()
    start = node.lineno - 1  # 0-indexed
    end = node.end_lineno or (start + 1)
    func_lines = lines[start:end]

    # Find docstring line range (relative to function start)
    docstring_lines: set[int] = set()
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        ds_node = node.body[0]
        ds_start = ds_node.lineno - node.lineno  # relative to func start
        ds_end = (ds_node.end_lineno or ds_node.lineno) - node.lineno
        docstring_lines = set(range(ds_start, ds_end + 1))

    count = 0
    for i, line in enumerate(func_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if i in docstring_lines:
            continue
        if stripped.startswith("#"):
            continue
        count += 1
    return count


def _extract_function_from_node(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    source: str,
    file_name: str,
    min_loc: int,
) -> FunctionInfo | None:
    """Extract FunctionInfo from an AST node if it meets LOC threshold.

    Returns
    -------
        The extracted FunctionInfo, or None if the node is skipped.
    """
    # Skip __init__ methods - they are often similar by nature (initialization patterns)
    if node.name == "__init__":
        return None

    start = node.lineno
    end = node.end_lineno or start
    loc = _count_executable_lines(node, source)

    if loc < min_loc:
        return None

    # Full raw source text
    text = get_function_text(node, source)

    # AST-based hash (ignores comments)
    normalized = normalize_ast_tokens(node)
    ast_hash = compute_content_hash(normalized)

    # AST text for AST embeddings
    ast_text = ast.unparse(node)

    # Text for text embeddings (signature + docstring + comments)
    text_for_embedding = extract_text_for_embedding(node, source)
    text_hash = compute_content_hash(text_for_embedding)

    # Raw docstring — the natural-language intent used as the Jina nl2code query.
    docstring = ast.get_docstring(node) or ""

    return FunctionInfo(
        name=node.name,
        file=file_name,
        start_line=start,
        end_line=end,
        loc=loc,
        hash=ast_hash,
        text=text,
        text_for_embedding=text_for_embedding,
        text_hash=text_hash,
        ast_text=ast_text,
        docstring=docstring,
    )


def _extract_functions_from_source(
    source: str,
    file_path: Path,
    min_loc: int,
    file_prefix: str = "",
) -> list[FunctionInfo]:
    """Extract all functions/classes from source code meeting LOC threshold.

    Returns
    -------
        List of extracted FunctionInfo objects.
    """
    functions: list[FunctionInfo] = []
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return functions

    file_name = f"{file_prefix}{file_path.name}" if file_prefix else file_path.name

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            func_info = _extract_function_from_node(node, source, file_name, min_loc)
            if func_info is not None:
                functions.append(func_info)

    return functions


def extract_function_infos(
    min_loc: int = 15,
    directory: Path | None = None,
    *,
    file_prefix: str = "",
    exclude_prefixes: tuple[str, ...] = (),
    recursive: bool = False,
) -> list[FunctionInfo]:
    """Extract functions/classes from ``*.py`` files in *directory*.

    Parameters
    ----------
    min_loc : int
        Minimum lines of code for inclusion.
    directory : Path | None
        Target directory.  Falls back to first resolved path.
    file_prefix : str
        Prefix prepended to each file's display name (e.g. ``"src/"``).
    exclude_prefixes : tuple[str, ...]
        Skip files whose name starts with any of these prefixes.
    recursive : bool
        When ``True`` use ``rglob`` to scan subdirectories recursively.

    Returns
    -------
        List of extracted FunctionInfo objects.
    """
    if directory is None:
        paths = resolve_paths()
        if not paths:
            return []
        directory = paths[0]
    glob_fn = directory.rglob if recursive else directory.glob
    return [
        fn
        for p in glob_fn("*.py")
        if not p.name.startswith("__")
        and not any(p.name.startswith(ep) for ep in exclude_prefixes)
        and not _should_skip(p)
        for fn in _extract_functions_from_source(
            p.read_text(encoding="utf-8"),
            p,
            min_loc,
            f"{file_prefix}{p.relative_to(directory).parent!s}/"
            if recursive and p.parent != directory
            else file_prefix,
        )
    ]


def extract_all_function_infos(min_loc: int = 15) -> list[FunctionInfo]:
    """Extract all functions and classes from ALL configured source directories.

    Scans recursively through every directory returned by
    :func:`pykissembed.paths.resolve_paths`.

    Returns
    -------
        List of all extracted FunctionInfo objects.
    """
    root = get_config().root
    all_functions: list[FunctionInfo] = []
    for base_dir in resolve_paths():
        rel_dir = (
            str(base_dir.relative_to(root)) if base_dir.is_relative_to(root) else str(base_dir)
        )
        prefix = f"{rel_dir}/"
        all_functions.extend(
            extract_function_infos(
                min_loc,
                base_dir,
                file_prefix=prefix,
                recursive=True,
            )
        )
    return all_functions


def extract_function_infos_from_file(file_path: Path, min_loc: int = 1) -> list[FunctionInfo]:
    """Extract all functions from a single file meeting LOC threshold.

    Returns
    -------
        List of extracted FunctionInfo objects.
    """
    source = file_path.read_text(encoding="utf-8")
    return _extract_functions_from_source(source, file_path, min_loc)
