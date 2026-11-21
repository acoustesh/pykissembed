"""Jev review of the function pairs flagged by the similarity gate.

Embeddings find pairs that look alike; this review asks whether sharing their
code would simplify future changes. A flagged pair stays failed unless Jev
confidently scores it below the bar, so an outage cannot turn the gate green.
"""

from __future__ import annotations

import ast
import sqlite3
import time
import warnings
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pykissembed.config import get_config
from pykissembed.jev import (
    JEV_MODEL,
    MAX_STATE_CHARS,
    SymbolState,
    ask_jev,
    grade_summary,
    is_low_confidence,
    load_api_key,
    open_cache,
    parse_grade,
    parse_noul,
)
from pykissembed.similarity.embeddings import is_str_object_dict
from pykissembed.wrapper_analysis import parse_source_files

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

    from pykissembed.config import PyqtestConfig
    from pykissembed.jev import Grade
    from pykissembed.similarity.types import FlaggedEdge, FunctionInfo

type _Modules = list[tuple[Path, ast.Module]]
type _PairKey = tuple[str, str, int]

_SCORE_ID = "consolidation_score"
# Per-function caps keep both functions and their callers inside one Jev state.
_MAX_SOURCE_CHARS = 5000
_MAX_CALLERS_CHARS = 750
_MAX_CALL_SITES = 5
_MAX_CALL_CHARS = 120
# Retries can hold one pair for minutes, so an outage must not stall the gate.
_REVIEW_BUDGET_SECONDS = 300.0
# A yes/no answer reads as "yes" from even odds upward.
_YES_AT = 0.5
_NEEDS_CHANGE = "Needs change"
_UNCERTAIN = "Uncertain"
_UNGRADED = "Ungraded"
_CLEARED = "Cleared"
# Report order: actionable pairs first, then pairs a person must judge.
_FAILED_BUCKETS = (_NEEDS_CHANGE, _UNCERTAIN, _UNGRADED)
# Criteria are ordered because Decisions numbers them from zero.
_LEVELS: tuple[str, ...] = (
    (
        "0 Keep separate: The similar code implements rules that can change independently, "
        "or sharing it would add more complexity than it removes."
    ),
    (
        "1 Extract shared logic: Both functions implement a specific rule that should change "
        "in one place. Their workflows remain distinct."
    ),
    (
        "2 Share the core workflow: Both functions follow the same core workflow, but their "
        "public contracts require separate entry points."
    ),
    (
        "3 Consolidate: Both functions have the same responsibility. One function can serve "
        "both sets of callers without caller-specific flags or branches."
    ),
)
_LEVEL_NAMES = ("keep", "extract", "core", "consolidate")
_INSTRUCTIONS = (
    "What level of consolidation would simplify future changes to these two functions? "
    "Use the rules that must change together, each function's contract with its callers, "
    "and the branches or flags the shared implementation would require.\n\n"
    "The state holds both functions and their known callers. The caller lists come from a "
    "static search and may be incomplete, so a missing caller does not prove there is none."
)
# The API returns no written rationale, so these answers explain each grade.
_EVIDENCE: tuple[tuple[str, str], ...] = (
    ("shared_rule", "shared rule"),
    ("separate_contracts", "separate contracts"),
    ("needs_flags", "needs flags"),
)
_QUESTIONS: dict[str, dict[str, object]] = {
    _SCORE_ID: {"type": "score", "instructions": _INSTRUCTIONS, "criteria": list(_LEVELS)},
    "shared_rule": {
        "type": "noul",
        "instructions": "Do both functions implement a specific rule that must change in one place?",
        "criteria": {
            "true": "A change to that rule would have to be made in both functions.",
            "false": "The functions' rules can change independently.",
        },
    },
    "separate_contracts": {
        "type": "noul",
        "instructions": "Do their contracts with callers require separate entry points?",
        "criteria": {
            "true": "Callers depend on different signatures, results, or error behavior.",
            "false": "One entry point could satisfy both sets of callers.",
        },
    },
    "needs_flags": {
        "type": "noul",
        "instructions": "Would a shared implementation need caller-specific flags or branches?",
        "criteria": {
            "true": "Sharing the code would add flags or branches that select caller behavior.",
            "false": "The shared code would need no caller-specific flags or branches.",
        },
    },
}


@dataclass(frozen=True, slots=True)
class _Pair:
    """One flagged pair in canonical order with its per-provider similarity."""

    first: FunctionInfo
    second: FunctionInfo
    similarities: dict[str, float]


@dataclass(frozen=True, slots=True)
class _Verdict:
    """Jev's grade of one pair and the signals that explain it."""

    grade: Grade
    distribution: tuple[float, ...]
    evidence: tuple[float | None, ...]


type _Row = tuple[_Pair, _Verdict | None]


def _sort_key(func: FunctionInfo) -> _PairKey:
    """Return the stable identity used to order and deduplicate a pair.

    Parameters
    ----------
    func : FunctionInfo
        Function to identify.

    Returns
    -------
    _PairKey
        ``(file, name, start_line)``.
    """
    return func.file, func.name, func.start_line


def _merge_edges(flagged: Mapping[str, Sequence[FlaggedEdge]]) -> list[_Pair]:
    """Merge provider edges into one entry per unordered function pair.

    Parameters
    ----------
    flagged : Mapping[str, Sequence[FlaggedEdge]]
        Edges recorded by each provider label.

    Returns
    -------
    list[_Pair]
        Pairs in deterministic order, keeping each provider's highest score.
    """
    merged: dict[tuple[_PairKey, _PairKey], _Pair] = {}
    for label, edges in flagged.items():
        for left, right, score in edges:
            # Providers can report a pair in either order; sorting keeps one
            # cache key, and so one paid request, per pair.
            first, second = sorted((left, right), key=_sort_key)
            key = (_sort_key(first), _sort_key(second))
            pair = merged.setdefault(key, _Pair(first, second, {}))
            pair.similarities[label] = max(score, pair.similarities.get(label, score))
    return [merged[key] for key in sorted(merged)]


def _names_file(module: str, level: int, importer: Path, target: Path) -> bool:
    """Return whether an imported module path refers to the *target* file.

    Parameters
    ----------
    module : str
        Dotted module path, empty for ``from . import x``.
    level : int
        Relative import level; zero for an absolute import.
    importer : Path
        File containing the import.
    target : Path
        Resolved file being searched for.

    Returns
    -------
    bool
        ``True`` when the import resolves to *target*.
    """
    parts = tuple(module.split(".")) if module else ()
    if level:
        if level > len(importer.parents):
            return False
        base = importer.parents[level - 1].joinpath(*parts)
        return target in {base.parent / f"{base.name}.py", base / "__init__.py"}
    module_path = target.parent if target.name == "__init__.py" else target.with_suffix("")
    # Absolute imports start at an unknown source root, so match the path's tail.
    return bool(parts) and module_path.parts[-len(parts) :] == parts


def _target_bindings(
    tree: ast.Module, importer: Path, target: Path, name: str
) -> tuple[set[str], set[str]]:
    """Find the local names through which a module can call the function.

    Parameters
    ----------
    tree : ast.Module
        Module to inspect.
    importer : Path
        Resolved path of that module.
    target : Path
        Resolved file defining the function.
    name : str
        Function name.

    Returns
    -------
    tuple[set[str], set[str]]
        Names bound to the function, and expressions bound to its module.
    """
    names: set[str] = set()
    receivers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            receivers.update(
                alias.asname or alias.name
                for alias in node.names
                if _names_file(alias.name, 0, importer, target)
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                bound = alias.asname or alias.name
                if alias.name == name and _names_file(module, node.level, importer, target):
                    names.add(bound)
                submodule = f"{module}.{alias.name}" if module else alias.name
                if _names_file(submodule, node.level, importer, target):
                    receivers.add(bound)
    return names, receivers


def _owned_calls(tree: ast.Module) -> Iterator[tuple[str, ast.Call]]:
    """Yield each call inside a function with its outermost owning function.

    Parameters
    ----------
    tree : ast.Module
        Module to scan.

    Yields
    ------
    tuple[str, ast.Call]
        Owning function name and call node.
    """
    seen: set[ast.Call] = set()
    # ast.walk is breadth-first, so an outer function claims its nested calls.
    for owner in ast.walk(tree):
        if not isinstance(owner, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(owner):
            if isinstance(node, ast.Call) and node not in seen:
                seen.add(node)
                yield owner.name, node


def _is_target_call(called: ast.expr, name: str, names: set[str], receivers: set[str]) -> bool:
    """Return whether a call expression statically refers to the function.

    Parameters
    ----------
    called : ast.expr
        The called expression.
    name : str
        Function name being searched for.
    names : set[str]
        Bare names bound to the function in this module.
    receivers : set[str]
        Expressions bound to the function's module or class.

    Returns
    -------
    bool
        ``True`` for a bound bare name or ``receiver.name()``.
    """
    if isinstance(called, ast.Name):
        return called.id in names
    return (
        isinstance(called, ast.Attribute)
        and called.attr == name
        and ast.unparse(called.value) in receivers
    )


def _call_sites(func: FunctionInfo, modules: _Modules, root: Path) -> list[str]:
    """List known calls to *func* without line numbers.

    Parameters
    ----------
    func : FunctionInfo
        Function whose callers are wanted.
    modules : _Modules
        Parsed project modules.
    root : Path
        Project root used for file keys.

    Returns
    -------
    list[str]
        At most ``_MAX_CALL_SITES`` entries of ``file:owner(): call``.
    """
    resolved_root = root.resolve()
    target_path = (resolved_root / func.file).resolve()
    sites: list[str] = []
    # ponytail: static matching misses instance calls across files, star imports
    # and re-exports; use a real call graph if grades prove caller-blind.
    for path, tree in modules:
        local = path == target_path
        names, receivers = _target_bindings(tree, path, target_path, func.name)
        if local:
            names.add(func.name)
            receivers.update(("self", "cls"))
        file_key = path.relative_to(resolved_root) if path.is_relative_to(resolved_root) else path
        sites.extend(
            f"{file_key}:{owner}(): {ast.unparse(call)[:_MAX_CALL_CHARS]}"
            for owner, call in _owned_calls(tree)
            # Recursion says nothing about the contract with outside callers.
            if not (local and owner == func.name)
            and _is_target_call(call.func, func.name, names, receivers)
        )
    return sites[:_MAX_CALL_SITES]


def _function_section(label: str, func: FunctionInfo, modules: _Modules, root: Path) -> str:
    """Render one function and its callers for the model.

    Parameters
    ----------
    label : str
        Section label, ``A`` or ``B``.
    func : FunctionInfo
        Function to render.
    modules : _Modules
        Parsed project modules.
    root : Path
        Project root used for file keys.

    Returns
    -------
    str
        A bounded section, so neither function can crowd out the other.
    """
    source = func.text
    if len(source) > _MAX_SOURCE_CHARS:
        source = source[:_MAX_SOURCE_CHARS] + "\n... [truncated]"
    callers = "\n".join(f"- {site}" for site in _call_sites(func, modules, root)) or "- none found"
    return (
        f"### Function {label}: {func.file}:{func.name}\n{source}\n\n"
        f"Known callers (static search, may be incomplete):\n{callers[:_MAX_CALLERS_CHARS]}"
    )


def _pair_state(pair: _Pair, modules: _Modules, root: Path) -> SymbolState:
    """Build the line-independent Jev state for one pair.

    Parameters
    ----------
    pair : _Pair
        Pair to grade.
    modules : _Modules
        Parsed project modules.
    root : Path
        Project root used for file keys.

    Returns
    -------
    SymbolState
        State whose cache key survives moving either function.
    """
    first, second = pair.first, pair.second
    # Line numbers stay out of the text so moving code keeps its cached grade.
    source = "\n\n".join(
        _function_section(label, func, modules, root)
        for label, func in (("A", first), ("B", second))
    )
    return SymbolState(
        file_key=f"{first.file} | {second.file}",
        symbol=f"{first.name} | {second.name}",
        kind="function_pair",
        lineno=first.start_line,
        source=source[:MAX_STATE_CHARS],
        has_docstring=bool(first.docstring and second.docstring),
    )


def _distribution(payload: object) -> tuple[float, ...]:
    """Read the probability of each rubric level.

    Parameters
    ----------
    payload : object
        Decoded Decisions response; untrusted shape.

    Returns
    -------
    tuple[float, ...]
        One probability per level, or empty when any value is unusable.
    """
    answers = payload.get("answers") if is_str_object_dict(payload) else None
    entry = answers.get(_SCORE_ID) if is_str_object_dict(answers) else None
    raw = entry.get("probabilities") if is_str_object_dict(entry) else None
    if not is_str_object_dict(raw):
        return ()
    values: list[float] = []
    for level in range(len(_LEVELS)):
        value = raw.get(str(level))
        # A partial distribution would misstate the shares, so drop all of it.
        if isinstance(value, bool) or not isinstance(value, int | float):
            return ()
        values.append(float(value))
    return tuple(values)


def _parse_verdict(payload: object) -> _Verdict | None:
    """Parse one response; only a missing score invalidates it.

    Parameters
    ----------
    payload : object
        Decoded Decisions response.

    Returns
    -------
    _Verdict | None
        The verdict, or ``None`` when the score is missing or invalid.
    """
    grade = parse_grade(payload, _SCORE_ID, len(_LEVELS))
    if grade is None:
        return None
    # Evidence questions can be refused individually; the grade stands alone.
    evidence = tuple(parse_noul(payload, question_id) for question_id, _ in _EVIDENCE)
    return _Verdict(grade, _distribution(payload), evidence)


def _bucket(verdict: _Verdict | None, fail_score: float) -> str:
    """Classify a verdict; only a confident score below the bar clears a pair.

    Parameters
    ----------
    verdict : _Verdict | None
        Jev's verdict, or ``None`` when unavailable.
    fail_score : float
        Lowest score that requires a change.

    Returns
    -------
    str
        The report bucket for this pair.
    """
    if verdict is None:
        return _UNGRADED
    score, confidence = verdict.grade
    if is_low_confidence(confidence):
        return _UNCERTAIN
    if score >= fail_score:
        return _NEEDS_CHANGE
    # is_low_confidence accepts a missing confidence, but that cannot clear a pair.
    return _CLEARED if confidence is not None else _UNCERTAIN


def _pair_title(pair: _Pair) -> str:
    """Name a pair for the report.

    Parameters
    ----------
    pair : _Pair
        Pair to name.

    Returns
    -------
    str
        ``file:name vs file:name``.
    """
    first, second = pair.first, pair.second
    return f"{first.file}:{first.name} vs {second.file}:{second.name}"


def _format_evidence(verdict: _Verdict) -> str:
    """Render the yes/no answers that explain a grade.

    Parameters
    ----------
    verdict : _Verdict
        Verdict holding one optional probability per evidence question.

    Returns
    -------
    str
        Semicolon-separated evidence, marking refused questions.
    """
    parts: list[str] = []
    for (_, label), probability in zip(_EVIDENCE, verdict.evidence, strict=True):
        if probability is None:
            parts.append(f"{label}: no answer")
        else:
            answer = "yes" if probability >= _YES_AT else "no"
            parts.append(f"{label}: {answer} ({probability:.0%})")
    return "; ".join(parts)


def _format_pair(pair: _Pair, verdict: _Verdict | None, fail_score: float) -> str:
    """Render one failed pair with its explanation.

    Parameters
    ----------
    pair : _Pair
        Failed pair.
    verdict : _Verdict | None
        Jev's verdict, or ``None`` when ungraded.
    fail_score : float
        Lowest score that requires a change.

    Returns
    -------
    str
        A multi-line report entry.
    """
    lines = [_pair_title(pair)]
    if verdict is not None:
        lines.append(f"  Decision: {grade_summary(verdict.grade, _LEVELS, f'<{fail_score:g}')}")
        if verdict.distribution:
            shares = zip(_LEVEL_NAMES, verdict.distribution, strict=True)
            lines.append("  Distribution: " + " | ".join(f"{n} {p:.0%}" for n, p in shares))
        lines.append(f"  Why: {_format_evidence(verdict)}")
    ranked = sorted(pair.similarities.items(), key=lambda item: (-item[1], item[0]))
    lines.append("  Flagged by: " + ", ".join(f"{name} {value:.1%}" for name, value in ranked))
    return "\n".join(lines)


def _row_score(row: _Row) -> float:
    """Return a row's score for ranking.

    Parameters
    ----------
    row : _Row
        Pair and optional verdict.

    Returns
    -------
    float
        The graded score, or zero when ungraded.
    """
    verdict = row[1]
    return 0.0 if verdict is None else verdict.grade[0]


def _build_report(rows: list[_Row], fail_score: float, note: str | None) -> str | None:
    """Warn about cleared pairs and render the failed ones.

    Parameters
    ----------
    rows : list[_Row]
        Every reviewed pair with its verdict.
    fail_score : float
        Lowest score that requires a change.
    note : str | None
        Why the review stopped early, if it did.

    Returns
    -------
    str | None
        The failure report, or ``None`` when every pair cleared.
    """
    grouped: dict[str, list[_Row]] = {name: [] for name in (*_FAILED_BUCKETS, _CLEARED)}
    for row in rows:
        grouped[_bucket(row[1], fail_score)].append(row)
    grouped[_NEEDS_CHANGE].sort(key=_row_score, reverse=True)
    # Warn rather than stay silent so a passing run still shows what was excused.
    if grouped[_CLEARED]:
        cleared = [
            f"{_pair_title(pair)} — {grade_summary(verdict.grade, _LEVELS, f'<{fail_score:g}')}"
            for pair, verdict in grouped[_CLEARED]
            if verdict is not None
        ]
        # A cleared pair is a pass, so warnings-as-errors must not fail it.
        with warnings.catch_warnings():
            warnings.simplefilter("always", UserWarning)
            warnings.warn(
                f"Jev consolidation review: {len(cleared)} flagged pair(s) cleared:\n"
                + "\n".join(cleared),
                UserWarning,
                stacklevel=3,
            )
    failed = sum(len(grouped[name]) for name in _FAILED_BUCKETS)
    if not failed:
        return None
    sections = [
        f"{name} ({len(grouped[name])}):\n"
        + "\n\n".join(_format_pair(pair, verdict, fail_score) for pair, verdict in grouped[name])
        for name in _FAILED_BUCKETS
        if grouped[name]
    ]
    header = (
        f"Jev consolidation review: {failed} of {len(rows)} flagged pair(s) remain failed "
        f"(model {JEV_MODEL})."
    )
    return "\n\n".join([header, *([note] if note else []), *sections])


def _grade_pairs(
    pairs: list[_Pair], modules: _Modules, config: PyqtestConfig, api_key: str | None
) -> tuple[list[_Verdict | None], str | None]:
    """Grade pairs in order, leaving the rest ungraded if the review must stop.

    Parameters
    ----------
    pairs : list[_Pair]
        Pairs to grade.
    modules : _Modules
        Parsed project modules.
    config : PyqtestConfig
        Active project configuration.
    api_key : str | None
        OpenRouter key, or ``None`` for stored decisions only.

    Returns
    -------
    tuple[list[_Verdict | None], str | None]
        One verdict per pair, and why the review stopped early, if it did.
    """
    verdicts: list[_Verdict | None] = []
    note: str | None = None
    active_key = api_key
    deadline = time.monotonic() + _REVIEW_BUDGET_SECONDS
    try:
        with closing(open_cache(config)) as conn:
            for pair in pairs:
                if active_key and time.monotonic() >= deadline:
                    # Later pairs can still use stored decisions.
                    active_key = None
                    note = f"Review stopped new requests at its {_REVIEW_BUDGET_SECONDS:g}s time budget."
                state = _pair_state(pair, modules, config.root)
                verdicts.append(
                    ask_jev(state, _QUESTIONS, parse=_parse_verdict, api_key=active_key, conn=conn)
                )
    except (sqlite3.Error, OSError) as exc:
        # Keep the failure report: the flagged pairs matter more than the cache.
        note = f"Jev cache unavailable: {exc}"
    verdicts.extend([None] * (len(pairs) - len(verdicts)))
    return verdicts, note


def triage_flagged_pairs(
    flagged: Mapping[str, Sequence[FlaggedEdge]],
    paths: list[Path],
    *,
    fail_score: float,
    cached_only: bool,
) -> str | None:
    """Grade flagged pairs once each and report the ones that stay failed.

    Must run on one thread: the Jev cache connection is thread-bound.

    Parameters
    ----------
    flagged : Mapping[str, Sequence[FlaggedEdge]]
        Edges recorded by each provider label.
    paths : list[Path]
        Source directories searched for callers.
    fail_score : float
        Lowest score that requires a change.
    cached_only : bool
        Use stored decisions only; misses stay failed as ungraded.

    Returns
    -------
    str | None
        The failure report, or ``None`` when every pair cleared.
    """
    pairs = _merge_edges(flagged)
    if not pairs:
        return None
    config = get_config()
    # Without a key, stored decisions are still used and misses stay failed.
    api_key = None if cached_only else load_api_key()
    verdicts, note = _grade_pairs(pairs, parse_source_files(paths), config, api_key)
    return _build_report(list(zip(pairs, verdicts, strict=True)), fail_score, note)
