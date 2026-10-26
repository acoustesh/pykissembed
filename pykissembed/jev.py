"""Shared Jev symbol extraction, response parsing, and local response cache."""

from __future__ import annotations

import ast
import json
import math
import os
import sqlite3
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from pykissembed._http_retry import is_retryable_http_error as _is_retryable_http_error
from pykissembed.config import get_config
from pykissembed.paths import is_test_path, warn_non_utf8
from pykissembed.paths import iter_py_files as _iter_py_files
from pykissembed.similarity.ast_helpers import compute_content_hash
from pykissembed.similarity.embeddings import requests_api
from pykissembed.wrapper_analysis import decorator_name

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from pykissembed.config import PyqtestConfig

JEV_MODEL = "typesafe/jev-1.14"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
API_KEY_ENV = "OPENROUTER_API_KEY"
REQUEST_TIMEOUT = 60.0
# A below-the-bar grade held at or below this confidence passes an audit with
# a warning instead of failing it. Live grades this low spread their
# probability across distant levels (e.g. 0.26 on the worst level and 0.19 on
# the best), so the expected value says little about the code.
LOW_CONFIDENCE = 0.2
RETRY_DELAYS = (1.0, 2.0, 4.0)
MAX_STATE_CHARS = 12000

type Grade = tuple[float, float | None]


@dataclass(frozen=True, slots=True)
class SymbolState:
    """One auditable function/class with its source text."""

    # The line is retained for diagnostics, but deliberately omitted from
    # the remote state so moving a definition does not invalidate its grade.
    file_key: str
    symbol: str
    kind: str
    lineno: int
    source: str
    has_docstring: bool


def _is_overload_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether *node* carries a bare or dotted ``overload`` decorator.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
        Function definition to inspect for an ``overload`` decorator.

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
        excluded), in deterministic file order. Files under a test
        directory (:data:`~pykissembed.paths.TEST_DIRS`) and unreadable or
        unparsable files contribute nothing.
    """
    states: list[SymbolState] = []
    for py_file in _iter_py_files(base_dir):
        if is_test_path(py_file, base_dir):
            continue
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
            states.append(state_for(node, source, rel))
    return states


def state_for(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, source: str, file_key: str
) -> SymbolState:
    """Build the graded state of one definition exactly as the audits do.

    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        Definition parsed from *source*.
    source : str
        Full text of the file that contains *node*.
    file_key : str
        Project-relative file key.

    Returns
    -------
    SymbolState
        The state whose payload keys the shared response cache, with the
        source segment truncated at ``MAX_STATE_CHARS``.
    """
    segment = ast.get_source_segment(source, node) or ""
    if len(segment) > MAX_STATE_CHARS:
        segment = segment[:MAX_STATE_CHARS] + "\n... [truncated]"
    return SymbolState(
        file_key=file_key,
        symbol=node.name,
        kind="class" if isinstance(node, ast.ClassDef) else "function",
        lineno=node.lineno,
        source=segment,
        has_docstring=ast.get_docstring(node) is not None,
    )


# Public re-export. Both consumer checks need symbol extraction, and a
# private name imported across modules trips pyright's reportPrivateUsage.
extract_symbol_states = _extract_symbol_states


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


# Public re-export; see the note on extract_symbol_states above.
load_api_key = _load_api_key


def _answer_number(payload: object, question_id: str, field: str) -> float | None:
    """Read one finite numeric field of a question's answer.

    Parameters
    ----------
    payload : object
        Decoded response body; untrusted shape.
    question_id : str
        Question whose answer entry is read.
    field : str
        Key inside ``payload["answers"][question_id]``.

    Returns
    -------
    float | None
        The value as a float, or ``None`` when any level of the path is
        missing or not a dict, or the value is not a finite non-bool number.
    """
    if not isinstance(payload, dict):
        return None
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return None
    entry = answers.get(question_id)
    if not isinstance(entry, dict):
        return None
    raw = entry.get(field)
    # bool is an int subclass, but cannot be a meaningful number here.
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def parse_score(payload: object, question_id: str, n_levels: int) -> float | None:
    """Read a zero-based expected score from a Decisions response.

    Parameters
    ----------
    payload : object
        Decoded response body.
    question_id : str
        Score question to read.
    n_levels : int
        Number of ordered criteria.

    Returns
    -------
    float | None
        Expected score in ``[0, n_levels - 1]``, or ``None`` for a malformed reply.
    """
    score = _answer_number(payload, question_id, "score")
    # The API returns an expected value, so fractional scores are valid.
    # Reject out-of-rubric values before they reach a baseline.
    if score is None or not 0 <= score <= n_levels - 1:
        return None
    return score


def parse_confidence(payload: object, question_id: str) -> float | None:
    """Read the model's confidence in its answer to one question.

    Parameters
    ----------
    payload : object
        Decoded response body.
    question_id : str
        Question to read.

    Returns
    -------
    float | None
        Confidence in ``[0, 1]``, or ``None`` when absent or out of range.
    """
    confidence = _answer_number(payload, question_id, "confidence")
    if confidence is None or not 0 <= confidence <= 1:
        return None
    return confidence


def is_low_confidence(confidence: float | None) -> bool:
    """Return whether a grade is too uncertain to fail an audit.

    Parameters
    ----------
    confidence : float | None
        The model's confidence, or ``None`` when unknown.

    Returns
    -------
    bool
        ``True`` when *confidence* is known and at most ``LOW_CONFIDENCE``;
        an unknown confidence never excuses a failure.
    """
    return confidence is not None and confidence <= LOW_CONFIDENCE


def parse_grade(payload: object, question_id: str, n_levels: int) -> Grade | None:
    """Read the level and confidence of one score answer.

    Parameters
    ----------
    payload : object
        Decoded Decisions response.
    question_id : str
        Score question to read.
    n_levels : int
        Number of ordered criteria.

    Returns
    -------
    Grade | None
        ``(level, confidence)``, or ``None`` when the score is malformed. A
        missing or out-of-range confidence alone does not invalidate the
        reply; it is reported as ``None``.
    """
    score = parse_score(payload, question_id, n_levels)
    if score is None:
        return None
    return score, parse_confidence(payload, question_id)


def read_threshold(data: object, key: str, default: float, max_level: float) -> float:
    """Read one minimum passing level, falling back to *default*.

    Parameters
    ----------
    data : object
        Baseline payload; untrusted shape.
    key : str
        Envelope key to read.
    default : float
        Value used when *key* is absent or unusable.
    max_level : float
        Highest level of the rubric.

    Returns
    -------
    float
        The configured minimum level, clamped into ``[0, max_level]`` so a
        hand-edited file cannot demand a level outside the rubric.
    """
    mapping = data if isinstance(data, dict) else {}
    raw = mapping.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return float(default)
    value = float(raw)
    if not math.isfinite(value):
        return float(default)
    return max(0.0, min(max_level, value))


def grade_summary(grade: Grade, levels: Sequence[str], need: str) -> str:
    """Render a grade as ``level X/N, confidence C (need M): text``.

    Parameters
    ----------
    grade : Grade
        ``(level, confidence)`` of one symbol.
    levels : Sequence[str]
        Rubric text per whole level, worst first.
    need : str
        Pass condition as shown to the reader.

    Returns
    -------
    str
        The summary; the confidence part is omitted when unknown. The text is
        taken from the nearest whole level, since the rubric is written per
        level.
    """
    level, confidence = grade
    top = len(levels) - 1
    sureness = "" if confidence is None else f", confidence {confidence:.2f}"
    text = levels[round(min(top, max(0.0, level)))]
    return f"level {level:.2f}/{top}{sureness} (need {need}): {text}"


def report_audit(
    label: str,
    *,
    failures: list[str],
    uncertain: list[str],
    ungraded: int,
    graded: int,
    api_key: str | None,
    uncertain_detail: str,
    failure_header: str,
) -> None:
    """Warn, fail or skip an audit from its graded symbol lines.

    Parameters
    ----------
    label : str
        Audit name prefixed to every message.
    failures : list[str]
        Lines for symbols that failed the bar.
    uncertain : list[str]
        Lines for failing grades excused by low confidence.
    ungraded : int
        Symbols with no available grade.
    graded : int
        Symbols that received a grade.
    api_key : str | None
        OpenRouter key; without one, a run that graded nothing is skipped.
    uncertain_detail : str
        Phrase completing "symbol(s) ... passed on low confidence".
    failure_header : str
        First line of the failure message.
    """
    # Warn before any failure so excused grades show up either way.
    if uncertain:
        warnings.warn(
            f"{label}: {len(uncertain)} symbol(s) {uncertain_detail} passed on low "
            f"confidence (<= {LOW_CONFIDENCE:g}):\n" + "\n".join(uncertain),
            UserWarning,
            stacklevel=3,
        )
    if failures:
        pytest.fail(failure_header + "\n" + "\n".join(failures), pytrace=False)
    if not api_key and not graded:
        pytest.skip(f"{API_KEY_ENV} unavailable; {ungraded} symbol(s) ungraded")
    if ungraded:
        warnings.warn(f"{label}: {ungraded} symbol(s) ungraded", UserWarning, stacklevel=3)


def state_payload(state: SymbolState) -> dict[str, str]:
    """Build the line-independent state sent to Jev.

    Parameters
    ----------
    state : SymbolState
        The symbol being graded.

    Returns
    -------
    dict[str, str]
        File, symbol, kind, and source text.
    """
    # Keep this shape identical to the POST body used below. Hashing a richer
    # local record would create misses for changes the model never sees.
    return {
        "file": state.file_key,
        "symbol": state.symbol,
        "kind": state.kind,
        "source": state.source,
    }


def canonical_json(value: object) -> str:
    """Serialize a cache key without incidental spacing or key-order changes.

    Parameters
    ----------
    value : object
        Value to serialise into a stable cache key.

    Returns
    -------
    str
        Deterministic JSON text.
    """
    # Stable key ordering and separators make equivalent payloads share a key.
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def open_cache(config: PyqtestConfig) -> sqlite3.Connection:
    """Open the shared WAL-mode Jev response cache under the baseline directory.

    Parameters
    ----------
    config : PyqtestConfig
        Active config naming the baseline directory.

    Returns
    -------
    sqlite3.Connection
        Connection to the initialized response table.
    """
    config.baseline_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.baseline_path / "jev_cache.sqlite3", timeout=30)
    # WAL permits readers and writers in separate pytest-xdist workers.
    _ = conn.execute("PRAGMA journal_mode=WAL")
    _ = conn.execute(
        "CREATE TABLE IF NOT EXISTS jev_responses ("
        "state_hash TEXT NOT NULL, question_hash TEXT NOT NULL, "
        "model TEXT NOT NULL, response TEXT NOT NULL, updated_at TEXT NOT NULL, "
        "PRIMARY KEY (state_hash, question_hash))"
    )
    conn.commit()
    # ponytail: Unused rows are small; add pruning if this file grows materially.
    return conn


def ask_jev[T](
    state: SymbolState,
    questions: dict[str, dict[str, object]],
    *,
    parse: Callable[[object], T | None],
    api_key: str | None,
    conn: sqlite3.Connection,
) -> T | None:
    """Grade one symbol from cache, or query and cache a valid Jev response.

    Parameters
    ----------
    state : SymbolState
        Source to grade.
    questions : dict[str, dict[str, object]]
        Complete Decisions question payload.
    parse : Callable[[object], T | None]
        Consumer-specific parser; ``None`` marks a response as invalid, and
        only responses it accepts are cached.
    api_key : str | None
        OpenRouter key; ``None`` allows cache hits only.
    conn : sqlite3.Connection
        Shared response cache.

    Returns
    -------
    T | None
        Parsed result, or ``None`` if a response is unavailable or invalid.
    """
    payload = state_payload(state)
    # The model belongs in the question key: a model upgrade re-grades every
    # symbol even when source and rubric text stay exactly the same.
    state_hash = compute_content_hash(canonical_json(payload))
    question_hash = compute_content_hash(
        canonical_json({"model": JEV_MODEL, "questions": questions})
    )
    row = conn.execute(
        "SELECT response FROM jev_responses WHERE state_hash = ? AND question_hash = ?",
        (state_hash, question_hash),
    ).fetchone()
    if row is not None:
        # Parse the raw response on every read, so a parser correction does not
        # require another paid request for the same state and question.
        try:
            return parse(json.loads(row[0]))
        except json.JSONDecodeError:
            return None
    if not api_key:
        # The cache is read before the key is checked, so offline and
        # cache-only runs still grade every cached symbol.
        return None
    try:
        post, timeout_error, _ = requests_api()
    except ImportError, TypeError:
        return None
    body: dict[str, object] = {"model": JEV_MODEL, "state": payload, "questions": questions}
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
            decoded = decode()
            score = parse(decoded)
            if score is not None:
                # Each paid, valid response is durable before the next symbol;
                # an interrupted audit therefore keeps completed work.
                _ = conn.execute(
                    "INSERT OR REPLACE INTO jev_responses "
                    "(state_hash, question_hash, model, response, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        state_hash,
                        question_hash,
                        JEV_MODEL,
                        canonical_json(decoded),
                        datetime.now(UTC).isoformat(),
                    ),
                )
                conn.commit()
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            # Retry timeouts, 429s and 5xx responses after each RETRY_DELAYS
            # pause; any other failure leaves the symbol ungraded.
            if attempt < len(RETRY_DELAYS) and _is_retryable_http_error(exc, timeout_error):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return None
        else:
            return score
    return None
