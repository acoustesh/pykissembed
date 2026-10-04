"""Focused tests for the Jev-backed docstring audit check (mocked transport)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import requests

from pykissembed import jev
from pykissembed.checks import jev_docstring_audit
from pykissembed.config import PyqtestConfig, reset_config_cache

if TYPE_CHECKING:
    from pathlib import Path

# The rubric is 1-indexed for humans; the API returns 0-indexed scores.
_PASS_LEVEL = 6
_MIN_SCORE = 4


def _answers_response(level: int = _PASS_LEVEL) -> dict[str, object]:
    """Build a Decisions API body for a rubric *level*.

    Parameters
    ----------
    level : int
        1-indexed rubric level to simulate.

    Returns
    -------
    dict[str, object]
        A well-formed ``score`` answer payload, 0-indexed as the API sends it.
    """
    api_score = level - 1
    return {
        "answers": {
            "docstring_score": {
                "type": "score",
                "score": api_score,
                "legend": {
                    str(i): text
                    for i, text in enumerate(jev_docstring_audit._SCORE_LEVELS)  # ruff:ignore[private-member-access]
                },
                "probabilities": {str(i): (1.0 if i == api_score else 0.0) for i in range(6)},
                "confidence": 1,
            },
        },
    }


class _Response:
    """Minimal Decisions API response double."""

    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Model a successful HTTP status.

        Raises
        ------
        requests.exceptions.HTTPError
            When the canned status code is 4xx or 5xx.
        """
        if self.status_code < 400:
            return
        response = requests.Response()
        response.status_code = self.status_code
        raise requests.exceptions.HTTPError(response=response)

    def json(self) -> object:
        """Return the canned payload.

        Returns
        -------
        object
            The canned Decisions API body.
        """
        return self.payload


def _post_returning(payload: dict[str, object]) -> object:
    """Return a ``requests.post`` double that always yields *payload*.

    Parameters
    ----------
    payload : dict[str, object]
        Canned Decisions API body.

    Returns
    -------
    object
        A callable suitable for monkeypatching ``requests.post``.
    """

    def post(*_args: object, **_kwargs: object) -> _Response:
        return _Response(payload)

    return post


def _parse_docstring_score(payload: object) -> float | None:
    """Convert Jev's zero-based score to the docstring rubric's level.

    Returns
    -------
    float | None
        One-based level, or ``None`` for a malformed answer.
    """
    return jev_docstring_audit._shift(  # ruff:ignore[private-member-access]
        jev.parse_score(payload, "docstring_score", 6)
    )


def _write_module(path: Path, name: str = "module.py") -> Path:
    """Write a documented function module under *path*.

    Parameters
    ----------
    path : Path
        Directory receiving the module.
    name : str
        Module filename.

    Returns
    -------
    Path
        The written module path.
    """
    module = path / name
    _ = module.write_text(
        '"""Module docstring."""\n\n\ndef greet(name: str) -> str:\n'
        '    """Greet someone.\n\n    Parameters\n    ----------\n'
        "    name : str\n        Who to greet.\n\n    Returns\n    -------\n"
        '    str\n        Greeting.\n    """\n    # Explain the greeting choice.\n'
        '    return "hello " + name\n',
        encoding="utf-8",
    )
    return module


def _run_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the Jev audit check against a single documented module.

    Parameters
    ----------
    tmp_path : Path
        Test workspace root.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    """
    src = tmp_path / "src"
    src.mkdir()
    _ = _write_module(src)
    config = PyqtestConfig(paths=["src"], root=tmp_path)
    reset_config_cache()
    monkeypatch.setattr(jev_docstring_audit, "get_config", lambda: config)
    monkeypatch.setattr(jev, "get_config", lambda: config)
    jev_docstring_audit.TestJevDocstringAudit.test_jev_docstring_audit(
        [src],
        update_baselines=False,
        cached_only=False,
    )


def test_perfect_score_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A level-6 score clears the default minimum without failure."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")
    monkeypatch.setattr(requests, "post", _post_returning(_answers_response(_PASS_LEVEL)))
    monkeypatch.setattr(jev.time, "sleep", lambda _s: None)

    _run_check(tmp_path, monkeypatch)


def test_minor_issues_score_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Level 5 (one or two very minor EIOD) is the intended pass band."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")
    monkeypatch.setattr(requests, "post", _post_returning(_answers_response(5)))
    monkeypatch.setattr(jev.time, "sleep", lambda _s: None)

    _run_check(tmp_path, monkeypatch)


def test_important_errors_fail_with_level_in_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Level 3 fails and the message names the level and the required bar."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")
    monkeypatch.setattr(requests, "post", _post_returning(_answers_response(3)))
    monkeypatch.setattr(jev.time, "sleep", lambda _s: None)

    with pytest.raises(pytest.fail.Exception) as exc_info:
        _run_check(tmp_path, monkeypatch)

    message = str(exc_info.value)
    assert "level 3.00/6" in message
    assert "need 4" in message
    assert "Moderate Errors" in message


def test_level_four_passes_at_the_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Level 4 clears the bar, so the pass band is 4-6 inclusive."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")
    monkeypatch.setattr(requests, "post", _post_returning(_answers_response(4)))
    monkeypatch.setattr(jev.time, "sleep", lambda _s: None)

    _run_check(tmp_path, monkeypatch)


def test_level_three_is_the_first_failing_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Level 3 is one below the bar and therefore the first failure."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")
    monkeypatch.setattr(requests, "post", _post_returning(_answers_response(3)))
    monkeypatch.setattr(jev.time, "sleep", lambda _s: None)

    with pytest.raises(pytest.fail.Exception) as exc_info:
        _run_check(tmp_path, monkeypatch)

    assert "level 3.00/6" in str(exc_info.value)


def test_missing_docstring_short_circuits_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undocumented symbols fail locally without any Jev call."""
    src = tmp_path / "src"
    src.mkdir()
    _ = (src / "module.py").write_text("def bare():\n    return 1\n", encoding="utf-8")
    config = PyqtestConfig(paths=["src"], root=tmp_path)
    reset_config_cache()
    monkeypatch.setattr(jev_docstring_audit, "get_config", lambda: config)
    monkeypatch.setattr(jev, "get_config", lambda: config)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")

    def _boom(*_args: object, **_kwargs: object) -> object:
        msg = "network must not be used for missing docstrings"
        raise AssertionError(msg)

    monkeypatch.setattr(requests, "post", _boom)

    with pytest.raises(pytest.fail.Exception) as exc_info:
        jev_docstring_audit.TestJevDocstringAudit.test_jev_docstring_audit(
            [src],
            update_baselines=False,
            cached_only=False,
        )

    message = str(exc_info.value)
    assert "level 1.00/6" in message
    assert "No docstring at all" in message


def test_missing_api_key_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key in env or .env skips instead of failing."""
    src = tmp_path / "src"
    src.mkdir()
    _ = _write_module(src)
    config = PyqtestConfig(paths=["src"], root=tmp_path)
    reset_config_cache()
    monkeypatch.setattr(jev_docstring_audit, "get_config", lambda: config)
    monkeypatch.setattr(jev, "get_config", lambda: config)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(pytest.skip.Exception, match="OPENROUTER_API_KEY"):
        jev_docstring_audit.TestJevDocstringAudit.test_jev_docstring_audit(
            [src],
            update_baselines=False,
            cached_only=False,
        )


def test_malformed_answers_are_skipped_not_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed Jev reply skips the symbol instead of failing the gate."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")
    monkeypatch.setattr(requests, "post", lambda *_a, **_k: _Response({"answers": {}}))
    monkeypatch.setattr(jev.time, "sleep", lambda _s: None)

    # The reply is unparseable, so the symbol is never graded: the gate must
    # neither fail nor skip the run, but it does warn that it graded nothing.
    with pytest.warns(UserWarning, match="ungraded"):
        _run_check(tmp_path, monkeypatch)


def test_overload_stubs_are_excluded_from_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overload stubs never reach the Jev transport."""
    src = tmp_path / "src"
    src.mkdir()
    _ = (src / "module.py").write_text(
        "from typing import overload\n\n\n@overload\ndef f(x: int) -> int:\n    ...\n",
        encoding="utf-8",
    )
    config = PyqtestConfig(paths=["src"], root=tmp_path)
    reset_config_cache()
    monkeypatch.setattr(jev_docstring_audit, "get_config", lambda: config)
    monkeypatch.setattr(jev, "get_config", lambda: config)

    states = jev._extract_symbol_states(  # ruff:ignore[private-member-access]
        src, root=tmp_path
    )

    assert states == []


def test_request_targets_decisions_api_with_score_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transport posts one `score` question carrying the 6-level rubric."""
    calls: list[dict[str, object]] = []

    def fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> _Response:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _Response(_answers_response(_PASS_LEVEL))

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(jev.time, "sleep", lambda _s: None)

    _run_check(tmp_path, monkeypatch)

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://openrouter.ai/api/alpha/decisions"
    body = call["json"]
    assert isinstance(body, dict)
    assert body["model"] == "typesafe/jev-1.13"
    state = body["state"]
    assert isinstance(state, dict)
    assert "source" in state
    assert "lineno" not in state
    questions = body["questions"]
    assert isinstance(questions, dict)
    assert set(questions) == {"docstring_score"}
    question = questions["docstring_score"]
    assert isinstance(question, dict)
    assert question["type"] == "score"
    # `score` questions take criteria as an ordered list, not a dict.
    criteria = question["criteria"]
    assert isinstance(criteria, list)
    assert len(criteria) == 6
    headers = call["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer openrouter-test-key-long-enough"


def test_parse_score_answer_shifts_to_one_index() -> None:
    """The API's 0-indexed score maps onto the rubric's 1-indexed level."""
    parse = _parse_docstring_score

    for level in range(1, 7):
        assert parse(_answers_response(level)) == pytest.approx(float(level))

    assert parse(None) is None
    assert parse({"answers": {}}) is None
    assert parse({"answers": {"docstring_score": {"type": "score"}}}) is None
    assert parse({"answers": {"docstring_score": {"score": True}}}) is None
    assert parse({"answers": {"docstring_score": {"score": "3"}}}) is None
    assert parse({"answers": {"docstring_score": {"score": -1}}}) is None
    assert parse({"answers": {"docstring_score": {"score": 99}}}) is None


def test_parse_score_accepts_fractional_expected_values() -> None:
    """The score is the expected value of the distribution, not the argmax.

    A reply of ``{0: 0, 1: 0.10, 2: 0.30, 3: 0.60, 4: 0, 5: 0}`` has the
    wire score ``0*0 + 1*0.1 + 2*0.3 + 3*0.6 == 2.5``, which is a level 3.5
    on the 1-indexed rubric -- not a level 3 or a level 4.
    """
    parse = _parse_docstring_score

    assert parse({"answers": {"docstring_score": {"score": 2.5}}}) == pytest.approx(3.5)
    # The documented example: probabilities {"0":0,"1":0,"2":1} with score 1.99.
    assert parse({"answers": {"docstring_score": {"score": 1.99}}}) == pytest.approx(2.99)
    # A whole-number score is still accepted, so an argmax-style reply works.
    assert parse({"answers": {"docstring_score": {"score": 5}}}) == pytest.approx(6.0)


def test_fractional_level_decides_pass_and_fail_at_the_boundary() -> None:
    """The bar is inclusive, so 4.0 passes and 3.99 fails."""
    evaluate = jev_docstring_audit._evaluate_symbol  # ruff:ignore[private-member-access]

    assert evaluate(4.0, 4.0) is True
    assert evaluate(3.99, 4.0) is False
    assert evaluate(4.01, 4.0) is True


def test_out_of_range_score_is_skipped_not_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A score outside the rubric is treated as malformed, not a failure."""
    payload = _answers_response(_PASS_LEVEL)
    answers = payload["answers"]
    assert isinstance(answers, dict)
    entry = answers["docstring_score"]
    assert isinstance(entry, dict)
    entry["score"] = 99

    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key-long-enough")
    monkeypatch.setattr(requests, "post", _post_returning(payload))
    monkeypatch.setattr(jev.time, "sleep", lambda _s: None)

    # A score of 99 is outside the 0-5 wire range, so it is rejected as
    # malformed and the symbol goes ungraded rather than failing the gate.
    with pytest.warns(UserWarning, match="ungraded"):
        _run_check(tmp_path, monkeypatch)


def test_evaluate_symbol_passes_at_or_above_minimum() -> None:
    """Higher is better, so the bar is inclusive at ``min_score``."""
    evaluate = jev_docstring_audit._evaluate_symbol  # ruff:ignore[private-member-access]

    assert evaluate(6, _MIN_SCORE) is True
    assert evaluate(5, _MIN_SCORE) is True
    assert evaluate(4, _MIN_SCORE) is True
    assert evaluate(3, _MIN_SCORE) is False
    assert evaluate(1, _MIN_SCORE) is False
    # A stricter bar rejects the level-4 band entirely.
    assert evaluate(4, 5) is False


def test_read_min_score_defaults_and_clamps() -> None:
    """A missing, mistyped, or out-of-range value falls back or clamps."""
    read = jev_docstring_audit._read_min_score  # ruff:ignore[private-member-access]

    assert read({}) == 4
    assert read(None) == 4
    assert read({"min_score": 5}) == 5
    assert read({"min_score": 0}) == 1
    assert read({"min_score": 99}) == 6
    assert read({"min_score": True}) == 4
    assert read({"min_score": "4"}) == 4


def test_test_symbol_detection_matches_test_prefix_only() -> None:
    """Test code is detected by the pytest ``Test*`` class-name convention."""
    is_test = jev_docstring_audit.is_test_symbol

    assert is_test("TestLineCount") is True
    assert is_test("Test") is True
    assert is_test("TestWrapperProliferation") is True
    assert is_test("check") is False
    assert is_test("test_something") is False
    # Prefix match, not substring: a name merely containing "Test" is library code.
    assert is_test("Testify") is True
    assert is_test("latest") is False


def test_test_bar_is_reduced_below_the_library_bar() -> None:
    """Test code is held to the lower, deliberately relaxed bar.

    Raising it to 4.5 was measured and failed two verified-conformant test
    classes, so the bar sits below DEFAULT_MIN_SCORE instead.
    """
    assert jev_docstring_audit.DEFAULT_TEST_MIN_SCORE < jev_docstring_audit.DEFAULT_MIN_SCORE


def test_min_score_for_selects_the_test_bar_by_name() -> None:
    """Each symbol gets the bar that matches its category."""
    min_for = jev_docstring_audit._min_score_for  # ruff:ignore[private-member-access]

    lib = jev_docstring_audit.SymbolState(
        file_key="pykissembed/cli.py",
        symbol="check",
        kind="function",
        lineno=1,
        source="",
        has_docstring=True,
    )
    tst = jev_docstring_audit.SymbolState(
        file_key="pykissembed/checks/code_complexity.py",
        symbol="TestLineCount",
        kind="class",
        lineno=1,
        source="",
        has_docstring=True,
    )

    assert min_for(lib, {}) == pytest.approx(jev_docstring_audit.DEFAULT_MIN_SCORE)
    assert min_for(tst, {}) == pytest.approx(jev_docstring_audit.DEFAULT_TEST_MIN_SCORE)
    # The envelope overrides both, per category.
    data = {"min_score": 5.0, "test_min_score": 3.5}
    assert min_for(lib, data) == pytest.approx(5.0)
    assert min_for(tst, data) == pytest.approx(3.5)
    # A malformed override falls back to that category's own default.
    assert min_for(lib, {"min_score": "x"}) == pytest.approx(jev_docstring_audit.DEFAULT_MIN_SCORE)
    assert min_for(tst, {"test_min_score": "x"}) == pytest.approx(
        jev_docstring_audit.DEFAULT_TEST_MIN_SCORE
    )


def test_rubric_is_monotone_in_severity() -> None:
    """The ladder is contiguous: every severity count maps to one level."""
    levels = jev_docstring_audit._SCORE_LEVELS  # ruff:ignore[private-member-access]

    assert len(levels) == 6
    # Level 1 covers both "no docstring" and the worst severity count.
    assert "No docstring at all" in levels[0]
    assert "2 Major" in levels[0]
    # Each level from 2 up names the count of defects that produces it.
    assert "1 Major" in levels[1]
    assert "3 or more Moderate" in levels[1]
    assert "1 or 2 Moderate" in levels[2]
    assert "3 or more Minor" in levels[3]
    assert "1 or 2 Minor" in levels[4]
    assert "no Errors, Inaccuracies, Omissions or Deviations" in levels[5]
    # Level 6 keeps the escape hatch for comment-free code.
    assert "fully accurate" in levels[5]


def test_guidance_uses_the_three_tier_severity_scale() -> None:
    """Instructions define 0-3 severity and the Minor/Moderate/Major terms."""
    instructions = jev_docstring_audit._INSTRUCTIONS  # ruff:ignore[private-member-access]

    for tier in ("0 - No violation", "1 - Minor", "2 - Moderate", "3 - Major"):
        assert tier in instructions
    assert "Treat the code as the source of truth" in instructions


def test_guidance_states_entry_syntax_rules() -> None:
    """Return/yield types required, parameter types optional, stars preserved."""
    instructions = jev_docstring_audit._INSTRUCTIONS  # ruff:ignore[private-member-access]

    assert "the name is required but the type is optional" in instructions
    assert "the type IS required, but the name is" in instructions
    assert "`*args` and `**kwargs`" in instructions


def test_guidance_forbids_upgrading_recommendations_to_requirements() -> None:
    """Optional content must neither lower the score nor block compliance."""
    instructions = jev_docstring_audit._INSTRUCTIONS  # ruff:ignore[private-member-access]

    assert "Do not convert recommendations into requirements" in instructions
    assert "never lower the score" in instructions
    assert "Absent optional content is never a violation" in instructions
    assert "never prevent a fully compliant verdict" in instructions


def test_guidance_requires_present_optional_content_to_be_correct() -> None:
    """Optionality permits omission but does not excuse wrong content."""
    instructions = jev_docstring_audit._INSTRUCTIONS  # ruff:ignore[private-member-access]

    assert "Validate optional content when it is present" in instructions
    assert "does not excuse incorrect content" in instructions


def test_guidance_lists_required_and_optional_sections() -> None:
    """Required and optional sections are both spelled out."""
    instructions = jev_docstring_audit._INSTRUCTIONS  # ruff:ignore[private-member-access]

    for section in (
        "`Parameters`",
        "`Returns`",
        "`Yields`",
        "`Receives`",
        "`Raises`",
        "`Examples`",
        ".. deprecated::",
    ):
        assert section in instructions
    assert "Exclude implicit self/cls" in instructions
