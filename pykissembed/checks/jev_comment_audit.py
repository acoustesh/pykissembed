"""Jev-backed audit of inline comments in functions and methods.

A function fails when its 0-4 comment level is at or below ``min_score`` in
``jev_comment_audit.json``, unless the model's confidence in that grade is at
most ``LOW_CONFIDENCE``: such a grade passes with a warning.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import pytest

from pykissembed.config import get_config
from pykissembed.jev import SymbolState as _SymbolState
from pykissembed.jev import ask_jev, grade_summary, parse_grade, read_threshold
from pykissembed.jev_audit import AuditSpec, run_audit

if TYPE_CHECKING:
    from pathlib import Path
    from sqlite3 import Connection

    from pykissembed.jev import Grade

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


def _grade_comment(state: _SymbolState, *, api_key: str | None, conn: Connection) -> Grade | None:
    """Grade one function's comments, including one without a docstring.

    Parameters
    ----------
    state : SymbolState
        Function or method to grade.
    api_key : str | None
        OpenRouter key, or ``None`` for cached responses only.
    conn : Connection
        Shared Jev response cache connection.

    Returns
    -------
    Grade | None
        ``(level, confidence)``, or ``None`` when no grade is available.
    """
    # A missing docstring is still sent: comments may explain intent for
    # this audit, unlike the docstring audit's local level-zero rule.
    return ask_jev(state, _QUESTIONS, parse=_parse_grade, api_key=api_key, conn=conn)


def _comment_failure_line(state: _SymbolState, grade: Grade, baseline_data: object) -> str | None:
    """Render a comment grade at or below the strict bar.

    Parameters
    ----------
    state : SymbolState
        Graded function or method.
    grade : Grade
        ``(level, confidence)`` from Jev.
    baseline_data : object
        Baseline payload holding ``min_score``.

    Returns
    -------
    str | None
        The report line, or ``None`` when the score exceeds the bar.
    """
    min_score = _read_threshold(baseline_data)
    if grade[0] > min_score:
        return None
    summary = grade_summary(grade, _SCORE_LEVELS, f">{min_score:g}")
    return f"{state.file_key}:{state.lineno} {state.symbol} — {summary}"


def _comment_bar_text(baseline_data: object) -> tuple[str, str]:
    """Describe a miss of the comment audit's single strict bar.

    Parameters
    ----------
    baseline_data : object
        Baseline payload holding ``min_score``.

    Returns
    -------
    tuple[str, str]
        The miss description and an empty note on the bars.
    """
    # The bar is strict (a score equal to it fails), hence "at or below".
    return f"at or below level {_read_threshold(baseline_data):g}", ""


_AUDIT = AuditSpec(
    label="Jev comment audit",
    baseline_filename=BASELINE_FILENAME,
    baseline_kind="jev_comment_audit",
    default_thresholds={"min_score": DEFAULT_MIN_SCORE},
    symbol_kinds=frozenset({"function"}),
    empty_message="No auditable functions or methods found",
    grade=_grade_comment,
    failure_line=_comment_failure_line,
    bar_text=_comment_bar_text,
)


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
        run_audit(
            _AUDIT,
            pykissembed_paths,
            get_config,
            update_baselines=update_baselines,
            cached_only=cached_only,
        )
