"""Command-line orchestration for docfix: options, file walk and report."""

from __future__ import annotations

import difflib
import tokenize
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pykissembed.config import get_config
from pykissembed.jev import load_api_key, open_cache
from pykissembed.paths import iter_py_files, resolve_paths
from pykissembed.tools.docfix.common import CODES, OPT_IN, default_selection
from pykissembed.tools.docfix.draft import (
    DEFAULT_MODEL,
    JevGrader,
    OpenRouterDrafter,
    failing_targets,
    gate_baseline,
)
from pykissembed.tools.docfix.engine import FileResult, Options
from pykissembed.tools.docfix.source import Ruff
from pykissembed.tools.docfix.writer import process_file

if TYPE_CHECKING:
    from pathlib import Path


# Failures one file can raise without stopping the others; each becomes that
# file's reported error and nothing is written for it. The tuple is explicit
# so a genuinely unexpected error still surfaces with a traceback.
_FILE_ERRORS = (
    OSError,
    SyntaxError,
    ValueError,
    KeyError,
    IndexError,
    TypeError,
    AttributeError,
    RecursionError,
    tokenize.TokenError,
)


@dataclass(frozen=True, slots=True)
class Request:
    """The command-line choices for one docfix run."""

    paths: list[Path]
    write: bool = False
    diff: bool = False
    check: bool = False
    select: str = ""
    ignore: str = ""
    symbols: tuple[str, ...] = ()
    only_failing: bool = False
    allow_dirty: bool = False
    draft: bool = False
    grade: bool = False
    model: str = DEFAULT_MODEL
    max_calls: int = 50
    line_length: int = 88
    grade_tolerance: float = 0.1


def selection(select: str, ignore: str) -> frozenset[str]:
    """Resolve ``--select`` / ``--ignore`` code lists.

    Parameters
    ----------
    select : str
        Comma-separated codes to run; empty means the defaults. Opt-in
        codes named here are added to the defaults.
    ignore : str
        Comma-separated codes to drop.

    Returns
    -------
    frozenset[str]
        Codes to run.

    Raises
    ------
    ValueError
        If a code is unknown.
    """
    chosen = {c.strip().upper() for c in select.split(",") if c.strip()}
    dropped = {c.strip().upper() for c in ignore.split(",") if c.strip()}
    unknown = sorted((chosen | dropped) - set(CODES))
    if unknown:
        msg = f"unknown docfix code(s): {', '.join(unknown)}"
        raise ValueError(msg)
    # Naming only opt-in codes adds them to the defaults; naming any other
    # code narrows the run to exactly those codes, as ruff's --select does.
    if chosen and not chosen <= OPT_IN:
        return frozenset(chosen - dropped)
    return frozenset((default_selection() | chosen) - dropped)


def _files(paths: list[Path]) -> list[Path]:
    """Expand directories into their Python files.

    Parameters
    ----------
    paths : list[Path]
        Files or directories.

    Returns
    -------
    list[Path]
        Files in deterministic order; directories are scanned with the same
        skip rules as the gates.
    """
    files: list[Path] = []
    for path in paths:
        # Directories use the gates' own skip rules (.venv, hidden dirs, ...).
        files.extend([path] if path.is_file() else iter_py_files(path))
    return files


def run(paths: list[Path], options: Options) -> list[FileResult]:
    """Process every file under *paths*.

    Parameters
    ----------
    paths : list[Path]
        Files or directories.
    options : Options
        Run options.

    Returns
    -------
    list[FileResult]
        One result per file; a failure in one file is recorded as its error
        and never stops the others.
    """
    ruff = Ruff()
    results: list[FileResult] = []
    for path in _files(paths):
        # Project-relative keys make --symbol filters and cached Jev grades
        # match the gates' own keys.
        try:
            key = str(path.resolve().relative_to(options.root.resolve()))
        except ValueError:
            key = str(path)
        try:
            results.append(process_file(path, key, options, ruff))
        except _FILE_ERRORS as exc:
            results.append(
                FileResult(key, "", "", error=f"internal error: {type(exc).__name__}: {exc}")
            )
    return results


def format_report(results: list[FileResult], *, show_diff: bool, write: bool) -> str:
    """Render findings, optional diffs and a summary.

    Parameters
    ----------
    results : list[FileResult]
        Per-file results.
    show_diff : bool
        Include unified diffs of changed files.
    write : bool
        Whether the run wrote files (changes the summary wording).

    Returns
    -------
    str
        The report text.
    """
    lines: list[str] = []
    counts: Counter[str] = Counter()
    for result in results:
        for item in result.findings:
            finding = item.finding
            counts[f"{finding.code} {finding.action}"] += 1
            lines.append(
                f"{result.key}:{item.line} {item.symbol} {finding.code} {finding.message} [{finding.action}]"
            )
        if result.error:
            lines.append(f"{result.key}: error: {result.error}")
        if show_diff and result.updated != result.original and not result.error:
            lines.extend(
                line.rstrip("\n")
                for line in difflib.unified_diff(
                    result.original.splitlines(keepends=True),
                    result.updated.splitlines(keepends=True),
                    fromfile=f"a/{result.key}",
                    tofile=f"b/{result.key}",
                )
            )
    changed = sum(1 for r in results if r.updated != r.original and not r.error)
    errors = sum(1 for r in results if r.error)
    verb = "written" if write else "would change"
    lines.append(f"\n{len(results)} file(s) scanned, {changed} {verb}, {errors} error(s)")
    lines.extend(f"  {key}: {count}" for key, count in sorted(counts.items()))
    return "\n".join(lines)


def _symbol_filter(specs: tuple[str, ...]) -> frozenset[tuple[str, str]] | None:
    """Parse ``--symbol file::name`` specs.

    Parameters
    ----------
    specs : tuple[str, ...]
        Raw specs.

    Returns
    -------
    frozenset[tuple[str, str]] | None
        ``(file_key, name)`` pairs, or ``None`` when no filter was given.

    Raises
    ------
    ValueError
        If a spec lacks the ``::`` separator.
    """
    if not specs:
        return None
    pairs: set[tuple[str, str]] = set()
    for spec in specs:
        file_key, sep, name = spec.partition("::")
        if not sep or not name:
            msg = f"--symbol expects FILE::NAME, got {spec!r}"
            raise ValueError(msg)
        pairs.add((file_key, name))
    return frozenset(pairs)


def _options(request: Request) -> Options:
    """Translate a request into engine options (without network helpers).

    Parameters
    ----------
    request : Request
        Command-line choices.

    Returns
    -------
    Options
        Engine options.
    """
    return Options(
        selected=selection(request.select, request.ignore),
        write=request.write,
        allow_dirty=request.allow_dirty,
        symbols=_symbol_filter(request.symbols),
        line_length=request.line_length,
        root=get_config().root,
    )


def main(request: Request) -> tuple[str, int]:
    """Run docfix for one command-line request.

    Opens the shared Jev cache (used for drafts, grades and targeting),
    builds the drafter and grader when requested, processes every file and
    renders the report.

    Parameters
    ----------
    request : Request
        Command-line choices.

    Returns
    -------
    tuple[str, int]
        The report text and an exit status: ``0`` on success; ``1`` when a
        file could not be processed or, with ``check``, when deterministic
        fixes remain unapplied; ``2`` for an unknown rule code, a malformed
        ``--symbol``, or ``--draft``/``--grade`` without an API key.
    """
    try:
        options = _options(request)
    except ValueError as exc:
        return str(exc), 2
    config = get_config()
    paths = request.paths or resolve_paths()
    # The key is only read when a network feature is requested.
    api_key = load_api_key() if request.draft or request.grade else None
    if (request.draft or request.grade) and not api_key:
        return "--draft and --grade need OPENROUTER_API_KEY", 2
    with closing(open_cache(config)) as conn:
        header = ""
        if request.only_failing:
            options.targets, ungraded = failing_targets(paths, config, conn)
            header = f"{len(options.targets)} failing symbol(s) targeted; {ungraded} without a cached grade\n"
        if request.draft and api_key:
            options.drafter = OpenRouterDrafter(api_key, conn, request.model, request.max_calls)
        if request.grade:
            options.grader = JevGrader(
                api_key, conn, gate_baseline(config), request.grade_tolerance
            )
        results = run(paths, options)
    report = header + format_report(results, show_diff=request.diff, write=request.write)
    # In --check mode only unapplied deterministic fixes fail the run; prose
    # and reports are advisory because no tool can resolve them unaided.
    pending = (
        any(f.finding.action == "fix" for r in results for f in r.findings) and not request.write
    )
    failed = any(r.error for r in results)
    return report, 1 if failed or (request.check and pending) else 0
