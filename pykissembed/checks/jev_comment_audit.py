"""Jev-backed audit of inline comments in functions and methods.

A function fails when its 0-4 comment level is at or below ``min_score`` in
``jev_comment_audit.json``, unless the model's confidence in that grade is at
most ``LOW_CONFIDENCE``: such a grade passes with a warning.
"""

from __future__ import annotations

from contextlib import closing
from functools import partial
from typing import TYPE_CHECKING

import pytest

from pykissembed.baselines_engine import locked_envelope, save_envelope
from pykissembed.config import get_config
from pykissembed.jev import (
    JEV_MODEL,
    ask_jev,
    extract_symbol_states,
    grade_summary,
    is_low_confidence,
    load_api_key,
    open_cache,
    parse_grade,
    read_threshold,
    report_audit,
)
from pykissembed.jev import SymbolState as _SymbolState

if TYPE_CHECKING:
    from pathlib import Path
    from sqlite3 import Connection

BASELINE_FILENAME = "jev_comment_audit.json"
DEFAULT_MIN_SCORE = 2.25
_QUESTION_ID = "comment_score"
_MAX_LEVEL = 4
# Criteria are ordered worst to best because Decisions numbers them from zero.
# Their expected value must exceed the configured bar.
_SCORE_LEVELS: tuple[str, ...] = (
    (
        "0 Major: A misleading comment likely to cause misuse, including claims about thread "
        "safety, validation, ownership, units, or destructive effects."
    ),
    (
        "1 Moderate: Necessary context is missing, or a comment is materially inaccurate or "
        "stale. This includes an unexplained hidden constraint or opaque intent, and a linter "
        "or type-checker suppression with no reason given."
    ),
    (
        "2 Minor: Repeated narration of obvious operations, or confusing wording whose meaning "
        "can still be recovered."
    ),
    (
        "3 No violation: Comments are accurate and useful, omitting comments is valid, or a "
        "workaround is concisely justified."
    ),
    (
        "4 No comments, and none needed: The code is straightforward, or its complexity is "
        "explained by the docstring."
    ),
)
# Keep this worked case in the question: the fixpoint guard needs a nearby
# explanation even if a docstring describes the regex block's overall intent.
_ABSENCE_EXAMPLE = r'''Example of a justified absence penalty:

```python
_TOKEN = re.compile(
    r"""(?P<quoted>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"""
    r"""|\((?:[^()'"]|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')*\)"""
)
while True:
    updated = _TOKEN.sub(
        lambda m: m.group(0) if m.group("quoted") is not None else "",
        text,
    )
    if updated == text:
        return text
    text = updated
```

- The `if updated == text: return text` guard with no comment is
  1 (Moderate). The regex removes only innermost parenthesised groups,
  so the loop must repeat until nothing changes in order to remove
  nested ones. A plausible refactor to a single `.sub()` call would
  silently leave nested groups behind. The comment belongs at the
  guard; buried in the docstring it loses its placement.
- The block's intent (strip parenthesised groups, nested ones
  included, while keeping quoted strings intact) is opaque from the
  regex. If the docstring states that intent, the missing comment is
  4. If nothing states it, it is 1 (Moderate).
- Counter-example: a plain loop over a list with no comment is 4.
  "Complex code needs comments" alone never justifies a penalty.'''
_INSTRUCTIONS = (
    """\
Grade the inline comments of this function or method, including the need for
comments when intent would otherwise be opaque. The source includes its
docstring, which may already explain the intent. Use the level of the worst
defect. Judge comments, not code quality.

1. Check each comment's claims against the code. A contradiction that can
   mislead a reader about behavior or safety deserves a heavy penalty.
2. Penalize a missing comment only when you can name the specific hidden
   constraint and the misunderstanding its omission invites.

"""
    + _ABSENCE_EXAMPLE
    + """\

3. Never use comment density, length, or coverage as a target.
4. Do not penalize missing optional metadata: authors, dates, issue links,
   TODOs, examples, or banners.
5. Judge redundancy by its cost. A brief orientation comment is fine; penalize
   repetition that adds clutter or can go stale."""
)
# The whole instruction and criterion text is hashed by the shared transport;
# changing this guidance intentionally invalidates previous comment grades.
_QUESTIONS: dict[str, dict[str, object]] = {
    _QUESTION_ID: {
        "type": "score",
        "instructions": _INSTRUCTIONS,
        "criteria": list(_SCORE_LEVELS),
    },
}


# ask_jev and the baseline reader take one positional argument, so the shared
# helpers are bound to this audit's rubric once, here.
_read_threshold = partial(
    read_threshold, key="min_score", default=DEFAULT_MIN_SCORE, max_level=float(_MAX_LEVEL)
)
_parse_grade = partial(parse_grade, question_id=_QUESTION_ID, n_levels=len(_SCORE_LEVELS))


def _grade_comments(
    states: list[_SymbolState], min_score: float, api_key: str | None, conn: Connection
) -> tuple[list[str], list[str], int]:
    """Grade function comments and collect failures and missing grades.

    Parameters
    ----------
    states : list[SymbolState]
        Function and method source states to grade.
    min_score : float
        Strict lower bound for passing scores.
    api_key : str | None
        OpenRouter key, or ``None`` for cached responses only.
    conn : Connection
        Shared Jev response cache connection.

    Returns
    -------
    tuple[list[str], list[str], int]
        Failure lines, lines for at-or-below-the-bar grades excused by low
        confidence, and the number of ungraded symbols.
    """
    failures: list[str] = []
    uncertain: list[str] = []
    ungraded = 0
    for state in states:
        grade = ask_jev(state, _QUESTIONS, parse=_parse_grade, api_key=api_key, conn=conn)
        if grade is None:
            # Transport or cache misses are unknown grades, not failures.
            ungraded += 1
            continue
        score, confidence = grade
        if score > min_score:
            continue
        summary = grade_summary(grade, _SCORE_LEVELS, f">{min_score:g}")
        line = f"{state.file_key}:{state.lineno} {state.symbol} — {summary}"
        (uncertain if is_low_confidence(confidence) else failures).append(line)
    return failures, uncertain, ungraded


class TestJevCommentAudit:
    """Jev-judged inline comment quality on the 0-4 rubric."""

    @staticmethod
    @pytest.mark.jev
    def test_jev_comment_audit(
        pykissembed_paths: list[Path],
        *,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Audit inline comments with a strict 0-4 score boundary.

        Functions and methods remain eligible even without docstrings because
        their comments may still explain intent. Scores at the bar fail,
        unless their confidence is at most ``LOW_CONFIDENCE``; those are
        reported in a warning instead.

        Parameters
        ----------
        pykissembed_paths : list[Path]
            Source directories scanned for function and method comments.
        update_baselines : bool
            Save the default comment threshold, then skip grading.
        cached_only : bool
            Use stored Jev decisions and count missing grades as ungraded.
        """
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        api_key = None if cached_only else load_api_key()
        config = get_config()
        baseline_file = config.baseline_path / BASELINE_FILENAME
        with locked_envelope(baseline_file, kind="jev_comment_audit") as envelope:
            _ = envelope.data.setdefault("min_score", DEFAULT_MIN_SCORE)
            if update_baselines:
                save_envelope(baseline_file, envelope)
                pytest.skip("Updated Jev comment audit baselines")
            min_score = _read_threshold(envelope.data)
            states = [
                state
                for base_dir in pykissembed_paths
                for state in extract_symbol_states(base_dir, root=config.root)
                if state.kind == "function"
            ]
            # A missing docstring is still sent: comments may explain intent
            # for this audit, unlike the docstring audit's local level-zero rule.
            if not states:
                pytest.skip("No auditable functions or methods found")
            with closing(open_cache(config)) as conn:
                failures, uncertain, ungraded = _grade_comments(states, min_score, api_key, conn)
            # The bar is strict (a score equal to it fails), hence "at or below".
            failure_header = (
                "Jev comment audit: "
                f"{len(failures)} symbol(s) at or below level {min_score:g} "
                f"({ungraded} ungraded; model {JEV_MODEL})."
            )
            report_audit(
                "Jev comment audit",
                failures=failures,
                uncertain=uncertain,
                ungraded=ungraded,
                graded=len(states) - ungraded,
                api_key=api_key,
                uncertain_detail=f"at or below level {min_score:g}",
                failure_header=failure_header,
            )
