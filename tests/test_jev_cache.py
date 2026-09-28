"""Response-cache behavior for the shared Jev transport and consumer checks."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import requests

from pykissembed import jev
from pykissembed.checks import jev_comment_audit
from pykissembed.config import PyqtestConfig

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

_QUESTIONS: dict[str, dict[str, object]] = {"score": {"type": "score", "criteria": ["bad", "good"]}}


class _Response:
    """Minimal Decisions response with a configurable decoded body."""

    def __init__(self, payload: object) -> None:
        self.payload = payload

    def json(self) -> object:
        """Return the canned response body.

        Returns
        -------
        object
            Decoded response.
        """
        return self.payload


def _state(source: str = "def f():\n    return 1\n", *, lineno: int = 1) -> jev.SymbolState:
    """Build one symbol for cache tests.

    Returns
    -------
    jev.SymbolState
        Source state with a configurable line number.
    """
    return jev.SymbolState(
        file_key="src/module.py",
        symbol="f",
        kind="function",
        lineno=lineno,
        source=source,
        has_docstring=False,
    )


def _ask(
    conn: sqlite3.Connection,
    state: jev.SymbolState,
    *,
    questions: dict[str, dict[str, object]] = _QUESTIONS,
    api_key: str | None = "test-key",
) -> float | None:
    """Ask Jev using the test score question.

    Returns
    -------
    float | None
        Zero-based score when valid.
    """
    return jev.ask_jev(
        state,
        questions,
        parse=lambda payload: jev.parse_score(payload, "score", 2),
        api_key=api_key,
        conn=conn,
    )


def _row_count(conn: sqlite3.Connection) -> int:
    """Count stored responses.

    Returns
    -------
    int
        Number of cache rows.
    """
    row = conn.execute("SELECT count(*) FROM jev_responses").fetchone()
    assert row is not None
    return int(row[0])


def test_unchanged_state_hits_cache_and_source_or_comment_edits_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated state is free; code and comment changes each query again."""
    calls: list[dict[str, object]] = []

    def post(_url: str, **kwargs: object) -> _Response:
        calls.append(kwargs)
        return _Response({"answers": {"score": {"score": 1}}})

    monkeypatch.setattr(requests, "post", post)
    state = _state()
    with closing(jev.open_cache(PyqtestConfig(root=tmp_path))) as conn:
        assert _ask(conn, state) == 1
    # Reopening the database models a second pytest run, not just a second
    # lookup through the same connection.
    with closing(jev.open_cache(PyqtestConfig(root=tmp_path))) as conn:
        assert _ask(conn, state) == 1
        assert len(calls) == 1
        commented = replace(state, source="def f():\n    # explain\n    return 1\n")
        assert _ask(conn, commented) == 1
        changed = replace(commented, source="def f():\n    # explain\n    return 2\n")
        assert _ask(conn, changed) == 1
        assert len(calls) == 3
        assert _row_count(conn) == 3


def test_questions_and_model_invalidate_but_line_shift_does_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the sent state and model/question payload participate in the key."""
    calls = 0

    def post(_url: str, **_kwargs: object) -> _Response:
        nonlocal calls
        calls += 1
        return _Response({"answers": {"score": {"score": 1}}})

    monkeypatch.setattr(requests, "post", post)
    with closing(jev.open_cache(PyqtestConfig(root=tmp_path))) as conn:
        state = _state()
        assert _ask(conn, state) == 1
        assert _ask(conn, replace(state, lineno=20)) == 1
        assert calls == 1
        questions: dict[str, dict[str, object]] = {
            "score": {"type": "score", "criteria": ["very bad", "good"]}
        }
        assert _ask(conn, state, questions=questions) == 1
        monkeypatch.setattr(jev, "JEV_MODEL", "typesafe/jev-next")
        assert _ask(conn, state, questions=questions) == 1
        assert calls == 3
        assert _row_count(conn) == 3


def test_malformed_reply_is_not_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid score shapes trigger a new request on the next run."""
    calls = 0

    def post(_url: str, **_kwargs: object) -> _Response:
        nonlocal calls
        calls += 1
        return _Response({"answers": {}})

    monkeypatch.setattr(requests, "post", post)
    with closing(jev.open_cache(PyqtestConfig(root=tmp_path))) as conn:
        assert _ask(conn, _state()) is None
        assert _ask(conn, _state()) is None
        assert calls == 2
        assert _row_count(conn) == 0


def test_cached_response_is_parsed_again_without_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parser correction can reuse the original response body."""
    calls = 0

    def post(_url: str, **_kwargs: object) -> _Response:
        nonlocal calls
        calls += 1
        return _Response({"answers": {"score": {"score": 1}}})

    monkeypatch.setattr(requests, "post", post)
    state = _state()
    config = PyqtestConfig(root=tmp_path)
    with closing(jev.open_cache(config)) as conn:
        assert _ask(conn, state) == 1
    with closing(jev.open_cache(config)) as conn:
        assert jev.ask_jev(
            state, _QUESTIONS, parse=lambda _payload: 0.5, api_key=None, conn=conn
        ) == pytest.approx(0.5)
    assert calls == 1


def test_no_key_grades_cached_failure_and_skips_uncached_symbol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline checks still report cached failures and count ungraded misses."""
    src = tmp_path / "src"
    src.mkdir()
    module = src / "module.py"
    _ = module.write_text("def cached():\n    return 1\n", encoding="utf-8")
    config = PyqtestConfig(paths=["src"], root=tmp_path)
    state = jev._extract_symbol_states(src, root=tmp_path)[0]  # ruff:ignore[private-member-access]
    with closing(jev.open_cache(config)) as conn:
        monkeypatch.setattr(
            requests,
            "post",
            lambda *_a, **_k: _Response({"answers": {"comment_score": {"score": 1}}}),
        )
        assert (
            jev.ask_jev(
                state,
                jev_comment_audit._QUESTIONS,  # ruff:ignore[private-member-access]
                parse=lambda payload: jev.parse_score(payload, "comment_score", 5),
                api_key="test-key",
                conn=conn,
            )
            == 1
        )
    _ = module.write_text(
        "def cached():\n    return 1\n\ndef uncached():\n    return 2\n", encoding="utf-8"
    )
    monkeypatch.setattr(jev_comment_audit, "get_config", lambda: config)
    monkeypatch.setattr(jev, "get_config", lambda: config)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def no_post(*_args: object, **_kwargs: object) -> _Response:
        msg = "offline audit must not send requests"
        raise AssertionError(msg)

    monkeypatch.setattr(requests, "post", no_post)
    with pytest.raises(pytest.fail.Exception) as exc_info:
        jev_comment_audit.TestJevCommentAudit.test_jev_comment_audit(
            [src], update_baselines=False, cached_only=False
        )
    message = str(exc_info.value)
    assert "cached" in message
    assert "1 ungraded" in message
    assert "uncached" not in message


def test_cached_only_never_posts_even_with_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consumer passes no API key to the transport in cached-only mode."""
    src = tmp_path / "src"
    src.mkdir()
    _ = (src / "module.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    config = PyqtestConfig(paths=["src"], root=tmp_path)
    monkeypatch.setattr(jev_comment_audit, "get_config", lambda: config)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def no_post(*_args: object, **_kwargs: object) -> _Response:
        msg = "cached-only must not send requests"
        raise AssertionError(msg)

    monkeypatch.setattr(requests, "post", no_post)
    with pytest.raises(pytest.skip.Exception, match=r"1 symbol\(s\) ungraded"):
        jev_comment_audit.TestJevCommentAudit.test_jev_comment_audit(
            [src], update_baselines=False, cached_only=True
        )
