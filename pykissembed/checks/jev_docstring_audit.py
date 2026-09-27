"""Jev-backed docstring quality audit (TypeSafe ``jev-1.13``).

Sends each function/class source (code, docstring, and inline comments) as
``state`` to the OpenRouter Decisions API (``typesafe/jev-1.13``) and asks a
single ``score`` question on an ordered 1-6 rubric. Every defect is rated on a
0-3 severity scale -- 0 no violation, 1 Minor, 2 Moderate, 3 Major -- covering
both docstring defects (E/I/O against the code, D against the NumPy convention)
and inline comment defects. The code is the source of truth: documentation is
only accurate insofar as it matches what the code actually does.

A symbol fails when its score is below ``min_score`` in
``tests/baselines/jev_docstring_audit.json``. Symbols with no docstring
never reach the API -- they are failed locally as level 1.

The check runs with the other consumer checks (marked ``jev``) and is
network-gated: it skips gracefully when ``OPENROUTER_API_KEY`` is absent,
so it never blocks offline CI.
"""

from __future__ import annotations

import ast
import contextlib
import math
import os
import time
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

import pytest

from pykissembed.baselines_engine import (
    BaselineEnvelope,
    locked_envelope,
    save_envelope,
)
from pykissembed.config import get_config
from pykissembed.paths import iter_py_files as _iter_py_files
from pykissembed.paths import warn_non_utf8
from pykissembed.wrapper_analysis import decorator_name

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

JEV_MODEL = "typesafe/jev-1.13"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
API_KEY_ENV = "OPENROUTER_API_KEY"
BASELINE_FILENAME = "jev_docstring_audit.json"
REQUEST_TIMEOUT = 60.0
RETRY_DELAYS = (1.0, 2.0, 4.0)
MAX_STATE_CHARS = 12000
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_SERVER_ERROR_MAX = 600


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

# Inline comments are graded on the same 0-3 scale, and a comment defect
# counts exactly like a docstring defect of equal severity. The absence of
# comments is deliberately NOT a defect: penalising comment-free code was the
# single largest source of false positives in the 2026-09-27 run, where 115 of
# 120 comment-relevance failures landed on symbols containing no '#' comment
# at all. A comment counts against the symbol only when it is actually present
# and actually wrong.
_COMMENT_GUIDANCE = """\
Judge the inline '#' comments on the same scale:

- If the code has no '#' comment lines, there are no comment defects. Never
  lower the score for the absence of comments.
- 1 Minor: the comment only restates the obvious, but is not wrong.
- 2 Moderate: the comment is stale or factually wrong about the code.
- 3 Major: the comment would actively mislead a reader about behaviour,
  safety, or a constraint.
- The final level is the worse of the docstring level and the comment level."""

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
    + "\n\n"
    + _COMMENT_GUIDANCE
)

_SCORE_QUESTION_ID = "docstring_score"

# Ordered worst -> best. The Decisions API numbers a `score` question from 0
# for the first criterion up to len(criteria) - 1, so this tuple *is* the
# rubric; `_parse_score_answer` shifts the API's 0-index back to 1-index.
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
        "Deviations; and every inline comment is relevant and insightful, or there "
        "are no inline comments."
    ),
)

_QUESTIONS: dict[str, dict[str, object]] = {
    _SCORE_QUESTION_ID: {
        "type": "score",
        "instructions": _INSTRUCTIONS,
        "criteria": list(_SCORE_LEVELS),
    },
}


@dataclass(frozen=True, slots=True)
class SymbolState:
    """One auditable function/class with its source text."""

    file_key: str
    symbol: str
    kind: str
    lineno: int
    source: str
    has_docstring: bool


def _is_overload_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether *node* carries a bare or dotted ``overload`` decorator.

    Returns
    -------
    bool
        Whether the function is an overload stub.
    """
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Name | ast.Attribute):
            continue
        name = decorator_name(decorator)
        tail = None if name is None else name.rsplit(".", maxsplit=1)[-1]
        if tail == "overload":
            return True
    return False


def _extract_symbol_states(base_dir: Path, *, root: Path) -> list[SymbolState]:
    """Extract auditable function/class states under *base_dir*.

    Parameters
    ----------
    base_dir : Path
        Configured source directory to scan.
    root : Path
        Project root used to build repo-relative file keys.

    Returns
    -------
    list[SymbolState]
        One entry per function/class definition (``@overload`` stubs
        excluded), in deterministic file order. Unreadable or unparsable
        files contribute nothing.
    """
    states: list[SymbolState] = []
    for py_file in _iter_py_files(base_dir):
        try:
            source = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            warn_non_utf8(py_file, exc)
            continue
        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        try:
            rel = str(py_file.relative_to(root))
        except ValueError:
            rel = str(py_file)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_overload_stub(node):
                continue
            segment = ast.get_source_segment(source, node) or ""
            if len(segment) > MAX_STATE_CHARS:
                segment = segment[:MAX_STATE_CHARS] + "\n... [truncated]"
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            states.append(
                SymbolState(
                    file_key=rel,
                    symbol=node.name,
                    kind=kind,
                    lineno=node.lineno,
                    source=segment,
                    has_docstring=ast.get_docstring(node) is not None,
                )
            )
    return states


def _load_api_key() -> str | None:
    """Load the OpenRouter API key from the environment or ``.env``.

    Returns
    -------
    str | None
        The key when present and non-empty, otherwise ``None``. The
        environment always wins over the ``.env`` fallback, and only a
        ``KEY=VALUE`` line match is honored.
    """
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        env_file = get_config().root / ".env"
        if env_file.exists():
            try:
                lines = env_file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError, OSError:
                return None
            prefix = f"{API_KEY_ENV}="
            for raw_line in lines:
                stripped = raw_line.strip()
                if stripped.startswith(prefix):
                    api_key = stripped.split("=", 1)[1].strip().strip("\"'")
                    break
    return api_key or None


def _requests_api() -> tuple[Callable[..., object], type[Exception], type[Exception]]:
    """Load the ``requests`` call and retry exception types lazily.

    Returns
    -------
    tuple[Callable[..., object], type[Exception], type[Exception]]
        ``(post, timeout_error, http_error)``.

    Raises
    ------
    TypeError
        If ``requests`` does not expose the expected runtime API.
    """
    requests_module = import_module("requests")
    post = getattr(requests_module, "post", None)
    if not callable(post):
        msg = "requests.post must be callable"
        raise TypeError(msg)
    exceptions = getattr(requests_module, "exceptions", None)
    if exceptions is None:
        msg = "requests.exceptions is required"
        raise TypeError(msg)
    timeout_error = getattr(exceptions, "Timeout", None)
    http_error = getattr(exceptions, "HTTPError", None)
    if not isinstance(timeout_error, type) or not issubclass(timeout_error, Exception):
        msg = "requests.exceptions.Timeout must be an Exception subclass"
        raise TypeError(msg)
    if not isinstance(http_error, type) or not issubclass(http_error, Exception):
        msg = "requests.exceptions.HTTPError must be an Exception subclass"
        raise TypeError(msg)
    return post, timeout_error, http_error


def _is_retryable(exc: Exception, timeout_error: type[Exception]) -> bool:
    """Return whether a Decisions API failure is worth retrying.

    Parameters
    ----------
    exc : Exception
        The caught failure.
    timeout_error : type[Exception]
        The ``requests`` timeout type.

    Returns
    -------
    bool
        ``True`` for timeouts, HTTP 429, and HTTP 5xx failures.
    """
    if isinstance(exc, timeout_error):
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or isinstance(status_code, bool):
        return False
    return status_code == _HTTP_TOO_MANY_REQUESTS or (
        _HTTP_SERVER_ERROR_MIN <= status_code < _HTTP_SERVER_ERROR_MAX
    )


def _parse_score_answer(payload: object) -> float | None:
    """Extract the expected rubric level from a Decisions API payload.

    A ``score`` answer reports the *expected value* of the distribution over
    the rubric, not its argmax: with probabilities ``{1: 0.10, 2: 0.30,
    3: 0.60, 4: 0, 5: 0, 6: 0}`` the score is ``1*0.1 + 2*0.3 + 3*0.6 ==
    1.99``. The value is therefore fractional in general, and confidence
    decides how close it sits to the argmax.

    The API numbers the rubric from 0, so this shifts the expected value into
    the rubric's 1-based numbering, where level 1 ("No docstring") is ``0``
    on the wire.

    Parameters
    ----------
    payload : object
        Decoded JSON body; untrusted shape.

    Returns
    -------
    float | None
        Expected level in ``[1, len(_SCORE_LEVELS)]``, or ``None`` when the
        reply is malformed or out of range so the caller skips the symbol
        rather than failing the gate on transport noise.
    """
    if not isinstance(payload, dict):
        return None
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return None
    entry = answers.get(_SCORE_QUESTION_ID)
    if not isinstance(entry, dict):
        return None
    raw = entry.get("score")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    level = float(raw) + 1.0
    if not math.isfinite(level) or not 1.0 <= level <= len(_SCORE_LEVELS):
        return None
    return level


def _query_jev(state: SymbolState, *, api_key: str) -> float | None:
    """Ask Jev to grade one symbol's docstring on the 1-6 rubric.

    Parameters
    ----------
    state : SymbolState
        The symbol whose source is sent as Jev ``state``.
    api_key : str
        OpenRouter bearer token; never logged.

    Returns
    -------
    float | None
        The expected level in ``[1, 6]``, or ``None`` when the request fails
        after retries or the reply is malformed (the caller skips the symbol
        rather than failing the gate on transport noise).
    """
    try:
        post, timeout_error, _ = _requests_api()
    except ImportError, TypeError:
        return None
    body: dict[str, object] = {
        "model": JEV_MODEL,
        "state": {
            "file": state.file_key,
            "symbol": state.symbol,
            "kind": state.kind,
            "lineno": state.lineno,
            "source": state.source,
        },
        "questions": _QUESTIONS,
    }
    # The bearer token lives only in this per-call header mapping, never in
    # logs or baselines.
    auth_headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            response = post(DECISIONS_URL, headers=auth_headers, json=body, timeout=REQUEST_TIMEOUT)
            raise_for_status = getattr(response, "raise_for_status", None)
            if callable(raise_for_status):
                _ = raise_for_status()
            decode = getattr(response, "json", None)
            if not callable(decode):
                return None
            return _parse_score_answer(decode())
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            if attempt < len(RETRY_DELAYS) and _is_retryable(exc, timeout_error):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return None
    return None


def _evaluate_symbol(level: float, min_score: float) -> bool:
    """Return whether an expected level clears the configured minimum.

    Parameters
    ----------
    level : float
        The expected 1-6 level returned by :func:`_query_jev`, or
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
    nearest = min(len(_SCORE_LEVELS), max(1, math.ceil(level)))
    text = _SCORE_LEVELS[nearest - 1]
    return (
        f"{state.file_key}:{state.lineno} {state.symbol}({state.kind}) — "
        f"level {level:.2f}/{len(_SCORE_LEVELS)} (need {min_score:g}): {text}"
    )


class TestJevDocstringAudit:
    """Jev-judged docstring quality on a 1-6 rubric (consumer check)."""

    @staticmethod
    @pytest.mark.jev
    def test_jev_docstring_audit(
        pykissembed_paths: list[Path],
        *,
        update_baselines: bool,
    ) -> None:
        """Fail when Jev grades a symbol's docstring below ``min_score``."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        api_key = _load_api_key()
        if not api_key:
            pytest.skip(
                "OPENROUTER_API_KEY not set (env or .env) — "
                "Jev docstring audit needs a live OpenRouter key"
            )
        config = get_config()
        with _locked_envelope() as (baseline_file, envelope):
            if update_baselines:
                save_envelope(baseline_file, envelope)
                pytest.skip("Updated Jev docstring audit baselines")
            base_min = _read_min_score(envelope.data)
            test_min = _read_threshold(envelope.data, _TEST_MIN_SCORE_KEY, DEFAULT_TEST_MIN_SCORE)
            states: list[SymbolState] = []
            for base_dir in pykissembed_paths:
                states.extend(_extract_symbol_states(base_dir, root=config.root))
            if not states:
                pytest.skip("No auditable functions or classes found")
            failures: list[str] = []
            for state in states:
                min_score = _min_score_for(state, envelope.data)
                if not state.has_docstring:
                    # No docstring is level 1 by definition; never call the API.
                    if not _evaluate_symbol(_LEVEL_NO_DOCTRING, min_score):
                        failures.append(
                            _format_symbol_failure(state, _LEVEL_NO_DOCTRING, min_score)
                        )
                    continue
                level = _query_jev(state, api_key=api_key)
                if level is None:
                    continue
                if not _evaluate_symbol(level, min_score):
                    failures.append(_format_symbol_failure(state, level, min_score))
            if failures:
                header = (
                    "Jev docstring audit: "
                    f"{len(failures)} symbol(s) scored below the minimum "
                    f"(model {JEV_MODEL}, need level {base_min:g} for library code "
                    f"and {test_min:g} for tests). "
                    f"Adjust {_MIN_SCORE_KEY!r} / {_TEST_MIN_SCORE_KEY!r} in "
                    f"{BASELINE_FILENAME} to change the bars."
                )
                pytest.fail(header + "\n" + "\n".join(failures), pytrace=False)
