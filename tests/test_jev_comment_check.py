"""Focused tests for the Jev inline comment audit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests

from pykissembed import jev
from pykissembed.checks import jev_comment_audit
from pykissembed.config import PyqtestConfig

if TYPE_CHECKING:
    from pathlib import Path


class _Response:
    """Minimal Decisions API response double."""

    def __init__(self, score: float) -> None:
        self.score = score

    def json(self) -> dict[str, object]:
        """Return a score answer.

        Returns
        -------
        dict[str, object]
            Decoded Decisions response.
        """
        return {"answers": {"comment_score": {"score": self.score}}}


def _run_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    *,
    score: float,
) -> list[str]:
    """Run the consumer check on one source file with a canned score.

    Returns
    -------
    list[str]
        Names of symbols sent to Jev.
    """
    src = tmp_path / "src"
    src.mkdir()
    _ = (src / "module.py").write_text(source, encoding="utf-8")
    config = PyqtestConfig(paths=["src"], root=tmp_path)
    monkeypatch.setattr(jev_comment_audit, "get_config", lambda: config)
    monkeypatch.setattr(jev, "get_config", lambda: config)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sent: list[str] = []

    def post(
        _url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> _Response:
        _ = headers, timeout
        state = json["state"]
        assert isinstance(state, dict)
        symbol = state["symbol"]
        assert isinstance(symbol, str)
        sent.append(symbol)
        return _Response(score)

    monkeypatch.setattr(requests, "post", post)
    jev_comment_audit.TestJevCommentAudit.test_jev_comment_audit(
        [src], update_baselines=False, cached_only=False
    )
    return sent


@pytest.mark.parametrize(
    ("score", "failure_pattern"),
    [(2.24, r"level 2\.24/4"), (2.25, r"level 2\.25/4"), (2.26, None)],
)
def test_comment_score_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    score: float,
    failure_pattern: str | None,
) -> None:
    """Scores at or below 2.25 fail; scores above it pass."""
    source = "def greet():\n    # Explain intent.\n    return 'hello'\n"
    if failure_pattern:
        with pytest.raises(pytest.fail.Exception, match=failure_pattern):
            _ = _run_check(tmp_path, monkeypatch, source, score=score)
    else:
        assert _run_check(tmp_path, monkeypatch, source, score=score) == ["greet"]


def test_only_functions_and_methods_are_sent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Class bodies are excluded while methods and module functions are graded."""
    source = (
        "class Greeter:\n"
        "    def greet(self):\n"
        "        return 'hello'\n\n"
        "def farewell():\n"
        "    return 'bye'\n"
    )
    assert set(_run_check(tmp_path, monkeypatch, source, score=4.0)) == {
        "greet",
        "farewell",
    }


def test_undocumented_function_is_sent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing docstrings do not short-circuit the comment audit."""
    assert _run_check(tmp_path, monkeypatch, "def bare():\n    return 1\n", score=4.0) == ["bare"]


def test_comment_rubric_has_ordered_five_levels() -> None:
    """The score question uses the intended worst-to-best 0-4 ladder."""
    question = jev_comment_audit._QUESTIONS["comment_score"]  # ruff:ignore[private-member-access]
    assert question["type"] == "score"
    criteria = question["criteria"]
    assert isinstance(criteria, list)
    assert len(criteria) == 5
    assert [item[0] for item in criteria] == ["0", "1", "2", "3", "4"]
    instructions = question["instructions"]
    assert isinstance(instructions, str)
    assert "updated == text" in instructions
    assert "single `.sub()` call" in instructions
    assert "quoted strings intact" in instructions
    assert "Never use comment density" in instructions


def test_comment_threshold_defaults_and_clamps() -> None:
    """A hand-edited baseline cannot set an unreachable comment score."""
    read = jev_comment_audit._read_threshold  # ruff:ignore[private-member-access]
    assert read({}) == pytest.approx(2.25)
    assert read({"min_score": -1}) == 0
    assert read({"min_score": 5}) == 4
    assert read({"min_score": True}) == pytest.approx(2.25)
