"""Jev-backed docstring quality audit (TypeSafe ``jev-1.13``).

Sends each documented function/class source to the OpenRouter Decisions API
and asks a score question on an ordered 1-6 docstring rubric. The code is the
source of truth when checking docstring claims. Valid responses are cached by
the exact state and question text.

A symbol fails when its score is below ``min_score`` in
``tests/baselines/jev_docstring_audit.json``. Symbols with no docstring
never reach the API -- they are failed locally as level 1.

The check runs with the other consumer checks (marked ``jev``). Without an
OpenRouter key it grades cached responses and skips cache misses.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from contextlib import closing
from typing import TYPE_CHECKING

import pytest

from pykissembed.baselines_engine import (
    BaselineEnvelope,
    locked_envelope,
    save_envelope,
)
from pykissembed.config import get_config
from pykissembed.jev import (
    API_KEY_ENV,
    JEV_MODEL,
    SymbolState,
    ask_jev,
    extract_symbol_states,
    load_api_key,
    open_cache,
    parse_score,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from sqlite3 import Connection

BASELINE_FILENAME = "jev_docstring_audit.json"

# Minimum expected score (on the 1-6 rubric) for a symbol to pass. The API
# returns the *expected value* of the distribution over the levels, so the bar
# is continuous: 4.00 passes and 3.99 does not. A float here is valid and lets
# the boundary be nudged without a code change.
#
# 4 is not a guess. It is the knee of a live 2026-09-27 run that graded all 392
# documented symbols (median 5.02, 0 skipped, ~$0.02), scored against the
# AST-verified verdicts from the earlier noul run:
#
#   min_score   flagged   precision   recall
#      3.50       126        98.4%      79.5%
#      3.90       143        92.3%      84.6%
#      4.00       146        92.5%      86.5%   <- chosen
#      4.25       151        92.1%      89.1%
#      5.00       192        75.0%      92.3%
#
# Below 3.9 precision holds but recall collapses; above 4.25 the false positives
# climb faster than the real findings. The band just under 4.0 held only three
# symbols, all three genuine, so the boundary is not perched on noise.
#
# The residual false positives are structural, not random: test classes and
# Protocol stubs score 3.2-4.1 because they have no parameters to document,
# yet a bare "Tests for X." docstring is perfectly adequate. Eight of the
# eleven are `Test*` classes, all verified conformant, which is why they get
# their own bar via DEFAULT_TEST_MIN_SCORE.
DEFAULT_MIN_SCORE = 4.0

# Test code is held to a lower bar. A `Test*` class documents a gate rather
# than a public API, and pytest fixtures routinely carry parameters that are
# self-evident from the name.
#
# The bar is LOWER than DEFAULT_MIN_SCORE because raising it backfired. The
# eight `Test*` classes here span 3.18-4.95, so a 4.5 bar failed the two that
# sit in [4.0, 4.5) -- TestCyclomaticComplexity (4.08) and TestDocstringFormat
# (4.39) -- both verified conformant by AST inspection, i.e. false positives
# by construction. Swept against the live 2026-09-27 run:
#
#   test_min_score   flagged   noise   real kept
#        4.50         148      13        135
#        4.00         146      11        135
#        3.75         144       9        135
#        3.50         142       7        135   <- chosen
#        3.00         141       6        135
#
# 3.5 clears six of the seven test false positives while keeping every real
# finding: the lowest-scoring genuine defect outside tests is a 1.00 (three
# symbols with no docstring at all), so there is no recall to trade. Only
# TestJevDocstringAudit (3.18) is still flagged.
DEFAULT_TEST_MIN_SCORE = 3.5
_MIN_SCORE_KEY = "min_score"
_TEST_MIN_SCORE_KEY = "test_min_score"
_TEST_SYMBOL_PREFIX = "Test"
_LEVEL_NO_DOCTRING = 1.0


def is_test_symbol(symbol: str) -> bool:
    """Return whether a symbol name marks it as test code.

    Matches the pytest class convention only (``Test*``). A path-based rule
    was considered and dropped: this project's ``[tool.pykissembed] paths``
    covers only ``pykissembed/``, so all eight ``Test*`` classes live inside
    ``checks/`` and a ``tests/`` path rule would never fire here.

    Parameters
    ----------
    symbol : str
        Symbol name.

    Returns
    -------
    bool
        ``True`` when the test threshold applies.
    """
    return symbol.startswith(_TEST_SYMBOL_PREFIX)


# Guidance blocks handed to the model as part of the score question. Together
# they resolve the ambiguities that made the earlier per-question rubric noisy:
# an absent optional section is never a violation, a missing docstring is a
# separate case from a present-but-defective one, and optional content that
# *is* present must still be correct.

_SECTION_GUIDANCE = """\
Which sections the NumPy convention requires:

- Short summary: required. One line describing the object.
- `Parameters`: required when applicable. Document every caller-supplied
  argument, including those with defaults. Exclude implicit self/cls.
- `Returns`: required for meaningful returned values. Unnecessary for a
  function that returns only None, and for a class constructor.
- `Yields`: required for generators. Describe the yielded values.
- `Receives`: relevant to generator .send() inputs. If you include it,
  `Yields` must also be present.

Optional. Absent optional content is never a violation:
`Other Parameters`, `Raises`, `Warns`, `Warnings`, `See Also`, `Notes`,
`References`, `Examples`, and a class `Attributes` or `Methods` listing.
For a class, constructor parameters belong in the class docstring;
duplicating them in `__init__` is unnecessary. A `.. deprecated::` notice is
only required for a deprecated object, and an extended summary is only
expected when the short summary does not suffice."""

_ENTRY_SYNTAX_GUIDANCE = """\
How entries are written:

- In `Parameters` the name is required but the type is optional. Both
  `name` on its own and `name : int` are acceptable.
- In `Returns` and `Yields` the type IS required, but the name is
  optional. Both `int` on its own and `value : int` are acceptable.
- For variadic parameters keep the stars: write `*args` and `**kwargs`,
  never `args` or `kwargs`."""

_SEVERITY_GUIDANCE = """\
Rate every defect on this scale:

0 - No violation.
    A valid omission, or a requirement that does not apply. Examples:
    missing optional sections; no `Returns` for a procedure; an unnamed
    return value.
1 - Minor.
    Cosmetic only: both the meaning and the parsing stay clear. Examples:
    capitalization, punctuation, or excessive line length.
2 - Moderate.
    A local structural or completeness defect. Examples: an applicable
    parameter left undocumented; a missing return description; sections in
    the wrong order.
3 - Major.
    Documentation that substantially misleads, or a structure that cannot
    be used. Examples: incorrect units, defaults, shape, or return
    behaviour; formatting that prevents recognition of core sections."""

_STYLE_GUIDANCE = """\
Two rules that override the severity table above:

- Do not convert recommendations into requirements. Content the NumPy
  convention marks as optional must never lower the score, and its absence
  must never prevent a fully compliant verdict. Judge what is written, not
  what a stricter style guide would add.
- Validate optional content when it is present. Optionality permits
  omission; it does not excuse incorrect content. A `Raises` section naming
  the wrong exception, or an `Examples` block that does not run, is a real
  defect and is graded on its merits."""

_INSTRUCTIONS = (
    "Grade the documentation of this function or class. Treat the code as "
    "the source of truth and judge only the documentation, never the code "
    "quality. Collect every Error, Inaccuracy, Omission and Deviation, rate "
    "each on the severity scale, then pick the one matching level.\n\n"
    + _SECTION_GUIDANCE
    + "\n\n"
    + _ENTRY_SYNTAX_GUIDANCE
    + "\n\n"
    + _SEVERITY_GUIDANCE
    + "\n\n"
    + _STYLE_GUIDANCE
)

_SCORE_QUESTION_ID = "docstring_score"

# Ordered worst -> best. The Decisions API numbers a `score` question from 0
# for the first criterion up to len(criteria) - 1, so this tuple *is* the
# rubric; the audit shifts the API's 0-index back to 1-index.
#
# The API answers with the *expected value* of the distribution over these
# levels, not the argmax, so a reply is usually fractional: 3.5 means the model
# puts half its mass on a 3 and half on a 4. Confidence, not the score, says
# how sure it is.
#
# The ladder is monotone in severity: each level names the worst defect
# present and how many of it, with no gap between neighbouring levels.
#   6  no defects
#   5  1-2 Minor
#   4  3+ Minor
#   3  1-2 Moderate
#   2  3+ Moderate, or 1 Major
#   1  2+ Major, or no docstring
_SCORE_LEVELS: tuple[str, ...] = (
    # 1
    (
        "No docstring at all; or a docstring carrying at least 2 Major Errors, "
        "Inaccuracies, Omissions or Deviations."
    ),
    # 2
    (
        "Docstring present, with 1 Major Error, Inaccuracy, Omission or "
        "Deviation, or with 3 or more Moderate ones."
    ),
    # 3
    "Docstring present, with 1 or 2 Moderate Errors, Inaccuracies, Omissions or Deviations.",
    # 4
    "Docstring present, with 3 or more Minor Errors, Inaccuracies, Omissions or Deviations.",
    # 5
    "Docstring present, with only 1 or 2 Minor Errors, Inaccuracies, Omissions or Deviations.",
    # 6
    (
        "Docstring present and fully accurate with respect to the actual code (the "
        "code is the source of truth), with no Errors, Inaccuracies, Omissions or "
        "Deviations."
    ),
)

_QUESTIONS: dict[str, dict[str, object]] = {
    _SCORE_QUESTION_ID: {
        "type": "score",
        "instructions": _INSTRUCTIONS,
        "criteria": list(_SCORE_LEVELS),
    },
}


def _evaluate_symbol(level: float, min_score: float) -> bool:
    """Return whether an expected level clears the configured minimum.

    Parameters
    ----------
    level : float
        The expected 1-6 level returned by Jev, or
        :data:`_LEVEL_NO_DOCTRING` for a symbol with no docstring. Fractional
        values arise because the API reports an expected value, not an
        argmax, so a level of 3.9 is a docstring the model is 90% confident
        is a 4.
    min_score : float
        Minimum passing level from the baseline envelope.

    Returns
    -------
    bool
        ``True`` when the docstring passes. Higher is better, so the
        comparison is ``>=``.
    """
    return level >= min_score


def _shift(score: float | None) -> float | None:
    """Shift a valid zero-based score to the docstring rubric's levels 1-6.

    Parameters
    ----------
    score : float | None
        Zero-based wire score from the API, or ``None`` when invalid.

    Returns
    -------
    float | None
        One-based score, or ``None`` for an invalid response.
    """
    return None if score is None else score + 1


@contextlib.contextmanager
def _locked_envelope() -> Iterator[tuple[Path, BaselineEnvelope]]:
    """Load ``jev_docstring_audit.json`` under a cross-process lock.

    The default ``min_score`` and ``test_min_score`` are merged in so the
    committed file stays self-documenting; the file holds configuration only,
    never per-symbol scores.

    Yields
    ------
    tuple[Path, BaselineEnvelope]
        The baseline path and its locked, default-populated envelope.
    """
    config = get_config()
    path = config.baseline_path / BASELINE_FILENAME
    with locked_envelope(path, kind="jev_docstring_audit") as envelope:
        _ = envelope.data.setdefault(_MIN_SCORE_KEY, DEFAULT_MIN_SCORE)
        _ = envelope.data.setdefault(_TEST_MIN_SCORE_KEY, DEFAULT_TEST_MIN_SCORE)
        yield path, envelope


def _read_threshold(data: object, key: str, default: float) -> float:
    """Read one minimum passing level, falling back to *default*.

    Parameters
    ----------
    data : object
        Baseline payload; untrusted shape.
    key : str
        Envelope key to read.
    default : float
        Value used when *key* is absent or unusable.

    Returns
    -------
    float
        The configured minimum level, clamped into
        ``[1, len(_SCORE_LEVELS)]`` so a hand-edited file cannot disable the
        gate or demand an unreachable level.
    """
    mapping = data if isinstance(data, dict) else {}
    raw = mapping.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return float(default)
    value = float(raw)
    if not math.isfinite(value):
        return float(default)
    return max(1.0, min(float(len(_SCORE_LEVELS)), value))


def _read_min_score(data: object) -> float:
    """Read the default minimum passing level, falling back to the default.

    Parameters
    ----------
    data : object
        Baseline payload; untrusted shape.

    Returns
    -------
    float
        The ``min_score`` value from the envelope.
    """
    return _read_threshold(data, _MIN_SCORE_KEY, DEFAULT_MIN_SCORE)


def _min_score_for(state: SymbolState, data: object) -> float:
    """Return the bar that applies to *state* given a baseline payload.

    Parameters
    ----------
    state : SymbolState
        The symbol being graded.
    data : object
        Baseline payload; untrusted shape.

    Returns
    -------
    float
        ``test_min_score`` for test code, otherwise ``min_score``.
    """
    if is_test_symbol(state.symbol):
        return _read_threshold(data, _TEST_MIN_SCORE_KEY, DEFAULT_TEST_MIN_SCORE)
    return _read_threshold(data, _MIN_SCORE_KEY, DEFAULT_MIN_SCORE)


def _format_symbol_failure(state: SymbolState, level: float, min_score: float) -> str:
    """Render one symbol's failure line with its expected rubric level.

    Parameters
    ----------
    state : SymbolState
        The failing symbol.
    level : float
        The expected 1-6 level awarded by Jev, usually fractional.
    min_score : float
        The minimum passing level.

    Returns
    -------
    str
        A single ``file:line symbol(kind) — level X/6 (need M): text`` line.
        The description is taken from the nearest whole level, since the
        rubric text is written per level, and the raw expected value is
        printed so a low-confidence grade stays visible.
    """
    nearest = min(len(_SCORE_LEVELS), max(1, round(level)))
    text = _SCORE_LEVELS[nearest - 1]
    return (
        f"{state.file_key}:{state.lineno} {state.symbol}({state.kind}) — "
        f"level {level:.2f}/{len(_SCORE_LEVELS)} (need {min_score:g}): {text}"
    )


def _grade_docstrings(
    states: list[SymbolState], baseline_data: object, api_key: str | None, conn: Connection
) -> tuple[list[str], int, int]:
    """Grade docstrings, assigning level one locally when they are absent.

    Parameters
    ----------
    states : list[SymbolState]
        Function and class source states to grade.
    baseline_data : object
        Baseline payload containing the library and test thresholds.
    api_key : str | None
        OpenRouter key, or ``None`` for cached responses only.
    conn : Connection
        Shared Jev response cache connection.

    Returns
    -------
    tuple[list[str], int, int]
        Failure descriptions, ungraded count, and graded count.
    """
    failures: list[str] = []
    ungraded = 0
    graded = 0
    for state in states:
        min_score = _min_score_for(state, baseline_data)
        if not state.has_docstring:
            # No docstring is level 1 by definition; never call the API.
            level = _LEVEL_NO_DOCTRING
        else:
            level = ask_jev(
                state,
                _QUESTIONS,
                parse=lambda payload: _shift(
                    parse_score(payload, _SCORE_QUESTION_ID, len(_SCORE_LEVELS))
                ),
                api_key=api_key,
                conn=conn,
            )
        if level is None:
            ungraded += 1
            continue
        graded += 1
        if not _evaluate_symbol(level, min_score):
            failures.append(_format_symbol_failure(state, level, min_score))
    return failures, ungraded, graded


class TestJevDocstringAudit:
    """Jev-judged docstring quality on a 1-6 rubric (consumer check)."""

    @staticmethod
    @pytest.mark.jev
    def test_jev_docstring_audit(
        pykissembed_paths: list[Path],
        *,
        update_baselines: bool,
        cached_only: bool,
    ) -> None:
        """Audit function and class docstrings against the 1-6 rubric.

        An absent docstring receives level one locally. Documented symbols
        use Jev grades, with a separate minimum for pytest-style classes.

        Parameters
        ----------
        pykissembed_paths : list[Path]
            Source directories whose functions and classes are audited.
        update_baselines : bool
            Save the default library and test docstring thresholds, then skip grading.
        cached_only : bool
            Use stored Jev decisions without requesting grades for cache misses.
        """
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        api_key = None if cached_only else load_api_key()
        config = get_config()
        with _locked_envelope() as (baseline_file, envelope):
            if update_baselines:
                save_envelope(baseline_file, envelope)
                pytest.skip("Updated Jev docstring audit baselines")
            base_min = _read_min_score(envelope.data)
            test_min = _read_threshold(envelope.data, _TEST_MIN_SCORE_KEY, DEFAULT_TEST_MIN_SCORE)
            states: list[SymbolState] = []
            for base_dir in pykissembed_paths:
                states.extend(extract_symbol_states(base_dir, root=config.root))
            if not states:
                pytest.skip("No auditable functions or classes found")
            with closing(open_cache(config)) as conn:
                failures, ungraded, graded = _grade_docstrings(states, envelope.data, api_key, conn)
            if failures:
                header = (
                    "Jev docstring audit: "
                    f"{len(failures)} symbol(s) scored below the minimum "
                    f"({ungraded} ungraded; model {JEV_MODEL}, "
                    f"need level {base_min:g} for library code "
                    f"and {test_min:g} for tests). "
                    f"Adjust {_MIN_SCORE_KEY!r} / {_TEST_MIN_SCORE_KEY!r} in "
                    f"{BASELINE_FILENAME} to change the bars."
                )
                pytest.fail(header + "\n" + "\n".join(failures), pytrace=False)
            if not api_key and not graded:
                pytest.skip(f"{API_KEY_ENV} unavailable; {ungraded} symbol(s) ungraded")
            if ungraded:
                warnings.warn(
                    f"Jev docstring audit: {ungraded} symbol(s) ungraded",
                    UserWarning,
                    stacklevel=2,
                )
