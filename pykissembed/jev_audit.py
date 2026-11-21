"""Shared workflow of the Jev symbol audits.

Each audit supplies its own rubric rules through :class:`AuditSpec`. This module
owns the steps they have in common: baseline locking, symbol selection, grading
from the shared cache, and reporting.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import pytest

from pykissembed.baselines_engine import locked_envelope, save_envelope
from pykissembed.jev import (
    JEV_MODEL,
    extract_symbol_states,
    is_low_confidence,
    load_api_key,
    open_cache,
    report_audit,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from sqlite3 import Connection

    from pykissembed.config import PyqtestConfig
    from pykissembed.jev import Grade, SymbolState


class GradeSymbol(Protocol):
    """Callable that grades one symbol from the cache or the API."""

    def __call__(
        self, state: SymbolState, *, api_key: str | None, conn: Connection
    ) -> Grade | None:
        """Return the symbol's grade.

        Parameters
        ----------
        state : SymbolState
            Symbol to grade.
        api_key : str | None
            OpenRouter key, or ``None`` for cached responses only.
        conn : Connection
            Shared Jev response cache connection.

        Returns
        -------
        Grade | None
            ``(level, confidence)``, or ``None`` when no grade is available.
        """
        ...


@dataclass(frozen=True, slots=True)
class AuditSpec:
    """The rules that make one Jev audit different from another.

    Attributes
    ----------
    label : str
        Audit name used in messages.
    baseline_filename : str
        Baseline file holding the audit's thresholds.
    baseline_kind : str
        Envelope kind of that file.
    default_thresholds : Mapping[str, float]
        Thresholds merged into the baseline when absent.
    symbol_kinds : frozenset[str]
        ``SymbolState.kind`` values the audit grades.
    empty_message : str
        Skip message when no symbol is auditable.
    grade : GradeSymbol
        Grades one symbol.
    failure_line : Callable[[SymbolState, Grade, object], str | None]
        Renders a grade that misses its bar, or returns ``None`` when it passes.
    bar_text : Callable[[object], tuple[str, str]]
        Builds the miss description and the note on the bars from the baseline
        payload.
    failure_verb : str
        Text placed before the miss description in the failure header only.
    failure_footer : str
        Text appended to the failure header.
    """

    label: str
    baseline_filename: str
    baseline_kind: str
    default_thresholds: Mapping[str, float]
    symbol_kinds: frozenset[str]
    empty_message: str
    grade: GradeSymbol
    failure_line: Callable[[SymbolState, Grade, object], str | None]
    bar_text: Callable[[object], tuple[str, str]]
    failure_verb: str = ""
    failure_footer: str = ""


def _grade_states(
    spec: AuditSpec,
    states: list[SymbolState],
    baseline_data: object,
    api_key: str | None,
    conn: Connection,
) -> tuple[list[str], list[str], int]:
    """Grade symbols and sort the ones that miss their bar.

    Parameters
    ----------
    spec : AuditSpec
        Audit being run.
    states : list[SymbolState]
        Symbols to grade.
    baseline_data : object
        Baseline payload holding the thresholds.
    api_key : str | None
        OpenRouter key, or ``None`` for cached responses only.
    conn : Connection
        Shared Jev response cache connection.

    Returns
    -------
    tuple[list[str], list[str], int]
        Failure lines, lines excused by low confidence, and ungraded count.
    """
    failures: list[str] = []
    uncertain: list[str] = []
    ungraded = 0
    for state in states:
        grade = spec.grade(state, api_key=api_key, conn=conn)
        if grade is None:
            # Transport or cache misses are unknown grades, not failures.
            ungraded += 1
            continue
        line = spec.failure_line(state, grade, baseline_data)
        if line is not None:
            # A low-confidence miss is reported but does not fail the audit.
            (uncertain if is_low_confidence(grade[1]) else failures).append(line)
    return failures, uncertain, ungraded


def run_audit(
    spec: AuditSpec,
    paths: list[Path],
    load_config: Callable[[], PyqtestConfig],
    *,
    update_baselines: bool,
    cached_only: bool,
) -> None:
    """Run one audit from baseline load through its pass, warn or fail report.

    Parameters
    ----------
    spec : AuditSpec
        Audit to run.
    paths : list[Path]
        Source directories to scan; an empty list skips the audit.
    load_config : Callable[[], PyqtestConfig]
        Loads the project configuration once the audit has paths to scan.
    update_baselines : bool
        Save the default thresholds, then skip grading.
    cached_only : bool
        Use stored Jev decisions without requesting grades for cache misses.
    """
    if not paths:
        pytest.skip("No [tool.pykissembed] paths configured")
    # Without a key, cached grades still count and misses stay ungraded.
    api_key = None if cached_only else load_api_key()
    config = load_config()
    baseline_file = config.baseline_path / spec.baseline_filename
    # Hold the lock through reporting so parallel workers read one baseline.
    with locked_envelope(baseline_file, kind=spec.baseline_kind) as envelope:
        # Merged defaults keep a saved baseline self-documenting.
        for key, value in spec.default_thresholds.items():
            _ = envelope.data.setdefault(key, value)
        if update_baselines:
            save_envelope(baseline_file, envelope)
            pytest.skip(f"Updated {spec.label} baselines")
        states = [
            state
            for base_dir in paths
            for state in extract_symbol_states(base_dir, root=config.root)
            if state.kind in spec.symbol_kinds
        ]
        if not states:
            pytest.skip(spec.empty_message)
        # Grade on this thread because SQLite connections are thread-bound.
        with closing(open_cache(config)) as conn:
            failures, uncertain, ungraded = _grade_states(
                spec, states, envelope.data, api_key, conn
            )
        uncertain_detail, bar_note = spec.bar_text(envelope.data)
        failure_header = (
            f"{spec.label}: {len(failures)} symbol(s) {spec.failure_verb}{uncertain_detail} "
            f"({ungraded} ungraded; model {JEV_MODEL}{bar_note}).{spec.failure_footer}"
        )
        report_audit(
            spec.label,
            failures=failures,
            uncertain=uncertain,
            ungraded=ungraded,
            graded=len(states) - ungraded,
            api_key=api_key,
            uncertain_detail=uncertain_detail,
            failure_header=failure_header,
        )
