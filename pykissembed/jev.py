"""Shared Jev symbol extraction, response parsing, and local response cache."""

from __future__ import annotations

import ast
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from typing import TYPE_CHECKING

from pykissembed.config import get_config
from pykissembed.paths import iter_py_files as _iter_py_files
from pykissembed.paths import warn_non_utf8
from pykissembed.similarity.ast_helpers import compute_content_hash
from pykissembed.wrapper_analysis import decorator_name

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pykissembed.config import PyqtestConfig

JEV_MODEL = "typesafe/jev-1.13"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
API_KEY_ENV = "OPENROUTER_API_KEY"
REQUEST_TIMEOUT = 60.0
RETRY_DELAYS = (1.0, 2.0, 4.0)
MAX_STATE_CHARS = 12000
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_SERVER_ERROR_MAX = 600


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
    if not isinstance(payload, dict):
        return None
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        return None
    entry = answers.get(question_id)
    if not isinstance(entry, dict):
        return None
    raw = entry.get("score")
    # bool is an int subclass, but cannot be a meaningful expected score.
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    score = float(raw)
    # The API returns an expected value, so fractional scores are valid.
    # Reject infinities and out-of-rubric values before they reach a baseline.
    if not math.isfinite(score) or not 0 <= score <= n_levels - 1:
        return None
    return score


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


def ask_jev(
    state: SymbolState,
    questions: dict[str, dict[str, object]],
    *,
    parse: Callable[[object], float | None],
    api_key: str | None,
    conn: sqlite3.Connection,
) -> float | None:
    """Grade one symbol from cache, or query and cache a valid Jev response.

    Parameters
    ----------
    state : SymbolState
        Source to grade.
    questions : dict[str, dict[str, object]]
        Complete Decisions question payload.
    parse : Callable[[object], float | None]
        Consumer-specific score parser.
    api_key : str | None
        OpenRouter key; ``None`` allows cache hits only.
    conn : sqlite3.Connection
        Shared response cache.

    Returns
    -------
    float | None
        Parsed score, or ``None`` if a response is unavailable or invalid.
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
        # A cached grade must remain usable in an offline or cache-only run.
        return None
    try:
        post, timeout_error, _ = _requests_api()
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
            # Retry only transport failures the original Jev audit retried.
            if attempt < len(RETRY_DELAYS) and _is_retryable(exc, timeout_error):
                time.sleep(RETRY_DELAYS[attempt])
                continue
            return None
        else:
            return score
    return None
