"""Jev-backed audit of inline comments in functions and methods."""

from __future__ import annotations

import math
import warnings
from contextlib import closing
from typing import TYPE_CHECKING

import pytest

from pykissembed.baselines_engine import locked_envelope, save_envelope
from pykissembed.config import get_config
from pykissembed.jev import (
    API_KEY_ENV,
    JEV_MODEL,
    _extract_symbol_states,
    _load_api_key,
    ask_jev,
    open_cache,
    parse_score,
)

if TYPE_CHECKING:
    from pathlib import Path

BASELINE_FILENAME = "jev_comment_audit.json"
DEFAULT_MIN_SCORE = 2.25
_QUESTION_ID = "comment_score"
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


def _read_threshold(data: object) -> float:
    """Read the strict minimum comment level, clamped to ``[0, 4]``.

    Parameters
    ----------
    data : object
        Baseline payload of untrusted shape.

    Returns
    -------
    float
        Passing score threshold.
    """
    mapping = data if isinstance(data, dict) else {}
    raw = mapping.get("min_score", DEFAULT_MIN_SCORE)
    # Invalid hand-edited values fall back to the chosen default; a numeric
    # override remains useful but cannot demand an impossible level.
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return DEFAULT_MIN_SCORE
    value = float(raw)
    if not math.isfinite(value):
        return DEFAULT_MIN_SCORE
    return max(0.0, min(float(len(_SCORE_LEVELS) - 1), value))


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
        """Fail when a function or method's comment score does not exceed the bar.

        Parameters
        ----------
        pykissembed_paths : list[Path]
            Configured source directories from the ``pykissembed_paths`` fixture; the test skips
            when empty.
        update_baselines : bool
            When true, save the baseline file with its default thresholds filled in and skip instead
            of grading.
        cached_only : bool
            When true, never call the API: grade from cached responses only and leave cache misses
            ungraded.
        """
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        api_key = None if cached_only else _load_api_key()
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
                for state in _extract_symbol_states(base_dir, root=config.root)
                if state.kind == "function"
            ]
            # A missing docstring is still sent: comments may explain intent
            # for this audit, unlike the docstring audit's local level-one rule.
            if not states:
                pytest.skip("No auditable functions or methods found")
            failures: list[str] = []
            ungraded = 0
            with closing(open_cache(config)) as conn:
                for state in states:
                    score = ask_jev(
                        state,
                        _QUESTIONS,
                        parse=lambda payload: parse_score(
                            payload, _QUESTION_ID, len(_SCORE_LEVELS)
                        ),
                        api_key=api_key,
                        conn=conn,
                    )
                    if score is None:
                        # Transport or cache misses are unknown grades, not
                        # failures; disclose their count in the final result.
                        ungraded += 1
                        continue
                    if score <= min_score:
                        nearest = min(len(_SCORE_LEVELS) - 1, max(0, round(score)))
                        failures.append(
                            f"{state.file_key}:{state.lineno} {state.symbol} — "
                            f"level {score:.2f}/4 (need >{min_score:g}): "
                            f"{_SCORE_LEVELS[nearest]}"
                        )
            if failures:
                pytest.fail(
                    "Jev comment audit: "
                    f"{len(failures)} symbol(s) at or below level {min_score:g} "
                    f"({ungraded} ungraded; model {JEV_MODEL}).\n" + "\n".join(failures),
                    pytrace=False,
                )
            if not api_key and ungraded == len(states):
                pytest.skip(f"{API_KEY_ENV} unavailable; {ungraded} symbol(s) ungraded")
            if ungraded:
                warnings.warn(
                    f"Jev comment audit: {ungraded} symbol(s) ungraded",
                    UserWarning,
                    stacklevel=2,
                )
