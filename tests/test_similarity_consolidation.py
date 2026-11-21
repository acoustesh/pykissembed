"""Tests for the Jev consolidation review of flagged similarity pairs."""

from __future__ import annotations

import sqlite3
import warnings
from typing import TYPE_CHECKING

import pytest
import requests

from pykissembed import jev
from pykissembed.config import PyqtestConfig
from pykissembed.similarity import consolidation
from pykissembed.similarity.types import FlaggedEdge, FunctionInfo

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

type Body = dict[str, object]


class _Response:
    """Minimal Decisions API response double."""

    def __init__(self, payload: Body) -> None:
        self.payload = payload

    def json(self) -> Body:
        """Return the canned decoded body.

        Returns
        -------
        Body
            Decoded Decisions response.
        """
        return self.payload


def _answers(
    score: float | None = 1.24,
    confidence: float | None = 0.62,
    *,
    evidence: dict[str, float] | None = None,
) -> Body:
    """Build a Decisions response with all four answers.

    Returns
    -------
    Body
        Response carrying the score and yes/no evidence answers.
    """
    answer: Body = {
        "type": "score",
        "legend": {"0": "keep", "1": "extract", "2": "core", "3": "consolidate"},
        "probabilities": {"0": 0.12, "1": 0.58, "2": 0.26, "3": 0.04},
    }
    if score is not None:
        answer["score"] = score
    if confidence is not None:
        answer["confidence"] = confidence
    yes_no = (
        {"shared_rule": 0.84, "separate_contracts": 0.71, "needs_flags": 0.18}
        if evidence is None
        else evidence
    )
    answers: Body = {key: {"type": "noul", "noul": value} for key, value in yes_no.items()}
    answers["consolidation_score"] = answer
    return {"answers": answers, "id": "gen-test", "model": "test", "usage": {}}


def _function(name: str, file: str, text: str | None = None) -> FunctionInfo:
    """Build one flagged function.

    Returns
    -------
    FunctionInfo
        Function with the given identity and source.
    """
    return FunctionInfo(
        name=name,
        file=file,
        start_line=1,
        end_line=2,
        loc=2,
        hash=f"hash-{file}-{name}",
        text=text if text is not None else f"def {name}():\n    return 1\n",
    )


def _edge(score: float = 0.966) -> FlaggedEdge:
    """Build one flagged edge between two functions in different files.

    Returns
    -------
    FlaggedEdge
        The default ``alpha`` and ``beta`` pair.
    """
    return _function("alpha", "src/a.py"), _function("beta", "src/b.py"), score


@pytest.fixture
def stub_jev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[Body], list[Body]]:
    """Point the review at a temporary project and a canned Jev reply.

    Returns
    -------
    Callable[[Body], list[Body]]
        Installs one reply and returns the list of captured request bodies.
    """
    (tmp_path / "src").mkdir()
    config = PyqtestConfig(paths=["src"], root=tmp_path)
    monkeypatch.setattr(consolidation, "get_config", lambda: config)
    monkeypatch.setattr(jev, "get_config", lambda: config)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def install(reply: Body) -> list[Body]:
        bodies: list[Body] = []

        def post(
            _url: str,
            *,
            headers: dict[str, str],
            json: Body,
            timeout: float,
        ) -> _Response:
            _ = headers, timeout
            bodies.append(json)
            return _Response(reply)

        monkeypatch.setattr(requests, "post", post)
        return bodies

    return install


def _triage(
    tmp_path: Path,
    flagged: dict[str, list[FlaggedEdge]] | None = None,
    *,
    cached_only: bool = False,
) -> str | None:
    """Run the review with the default bar.

    Returns
    -------
    str | None
        The failure report, or ``None`` when every pair cleared.
    """
    return consolidation.triage_flagged_pairs(
        {"Gemini-AST": [_edge()]} if flagged is None else flagged,
        [tmp_path / "src"],
        fail_score=1.0,
        cached_only=cached_only,
    )


def _state_source(body: Body) -> str:
    """Read the source text sent to Jev.

    Returns
    -------
    str
        The ``source`` field of the request state.
    """
    state = body["state"]
    assert isinstance(state, dict)
    source = state["source"]
    assert isinstance(source, str)
    return source


def test_rubric_matches_decision_table(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """The request asks the agreed question on the agreed 0-3 ladder."""
    bodies = stub_jev(_answers())

    _ = _triage(tmp_path)

    questions = bodies[0]["questions"]
    assert isinstance(questions, dict)
    score_question = questions["consolidation_score"]
    assert score_question["type"] == "score"
    criteria = score_question["criteria"]
    assert [criterion.split(":")[0] for criterion in criteria] == [
        "0 Keep separate",
        "1 Extract shared logic",
        "2 Share the core workflow",
        "3 Consolidate",
    ]
    assert (
        "What level of consolidation would simplify future changes to these two functions?"
        in score_question["instructions"]
    )
    assert {questions[key]["type"] for key in ("shared_rule", "separate_contracts", "needs_flags")} == {
        "noul"
    }


def test_fail_bar_clears_just_below(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A confident score just under the bar clears and is still shown."""
    _ = stub_jev(_answers(0.99, 0.7))

    with pytest.warns(UserWarning, match=r"src/a\.py:alpha vs src/b\.py:beta"):
        assert _triage(tmp_path) is None


def test_fail_bar_fails_at_the_bar(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A score equal to the bar needs a change."""
    _ = stub_jev(_answers(1.0, 0.7))

    report = _triage(tmp_path)

    assert report is not None
    assert "Needs change (1)" in report


@pytest.mark.parametrize("confidence", [0.2, None])
def test_low_or_missing_confidence_never_clears(
    tmp_path: Path,
    stub_jev: Callable[[Body], list[Body]],
    confidence: float | None,
) -> None:
    """A low score cannot clear a pair unless the model is confident in it."""
    _ = stub_jev(_answers(0.4, confidence))

    report = _triage(tmp_path)

    assert report is not None
    assert "Uncertain (1)" in report
    assert "Needs change" not in report


def test_high_score_with_low_confidence_is_uncertain(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """An unsure recommendation fails without claiming a change is needed."""
    _ = stub_jev(_answers(2.0, 0.1))

    report = _triage(tmp_path)

    assert report is not None
    assert "Uncertain (1)" in report
    assert "Needs change" not in report


def test_unavailable_grade_stays_failed(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """Cache-only misses make no request and keep the pair failing."""
    bodies = stub_jev(_answers(0.0, 0.9))

    report = _triage(tmp_path, cached_only=True)

    assert bodies == []
    assert report is not None
    assert "Ungraded (1)" in report


def test_refused_score_stays_failed(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A refusal keeps the pair failing and is retried on the next run."""
    bodies = stub_jev(_answers(None))

    first = _triage(tmp_path)
    second = _triage(tmp_path)

    assert first is not None
    assert "Ungraded (1)" in first
    assert second is not None
    assert len(bodies) == 2


def test_report_explains_failure(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A failing pair shows its decision, distribution, evidence and sources."""
    _ = stub_jev(_answers())

    report = _triage(tmp_path)

    assert report is not None
    assert "src/a.py:alpha vs src/b.py:beta" in report
    assert "level 1.24/3, confidence 0.62 (need <1): 1 Extract shared logic" in report
    assert "keep 12% | extract 58% | core 26% | consolidate 4%" in report
    assert "shared rule: yes (84%); separate contracts: yes (71%); needs flags: no (18%)" in report
    assert "Gemini-AST 96.6%" in report


def test_missing_evidence_keeps_grade(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A refused evidence question does not discard the score."""
    _ = stub_jev(_answers(evidence={"shared_rule": 0.84, "separate_contracts": 0.71}))

    report = _triage(tmp_path)

    assert report is not None
    assert "Needs change (1)" in report
    assert "needs flags: no answer" in report


def test_pair_sent_once_across_providers(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """The same pair in either order makes one request and names each provider."""
    bodies = stub_jev(_answers())
    first, second, _ = _edge()

    report = _triage(
        tmp_path,
        {"Gemini-AST": [(first, second, 0.966)], "Gemini-Text": [(second, first, 0.961)]},
    )

    assert len(bodies) == 1
    assert report is not None
    assert "Gemini-AST 96.6%, Gemini-Text 96.1%" in report


def test_same_named_functions_keep_their_own_callers(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """Callers are attributed by file, not by the shared function name."""
    bodies = stub_jev(_answers())
    for stem in ("a", "b"):
        _ = (tmp_path / "src" / f"{stem}.py").write_text(
            f"def _load():\n    return 1\n\ndef use_{stem}():\n    return _load()\n",
            encoding="utf-8",
        )
    flagged: FlaggedEdge = (_function("_load", "src/a.py"), _function("_load", "src/b.py"), 0.97)

    _ = _triage(tmp_path, {"Qwen-AST": [flagged]})

    section_a, section_b = _state_source(bodies[0]).split("### Function B")
    assert "use_a" in section_a
    assert "use_b" not in section_a
    assert "use_b" in section_b
    assert "use_a" not in section_b


def test_imported_function_call_is_listed(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A caller in another file is found through its from-import."""
    bodies = stub_jev(_answers())
    _ = (tmp_path / "src" / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    _ = (tmp_path / "src" / "b.py").write_text("def beta():\n    return 1\n", encoding="utf-8")
    _ = (tmp_path / "src" / "user.py").write_text(
        "from a import alpha\n\ndef consumer():\n    return alpha()\n",
        encoding="utf-8",
    )

    _ = _triage(tmp_path)

    section_a, section_b = _state_source(bodies[0]).split("### Function B")
    assert "src/user.py:consumer" in section_a
    assert "src/user.py:consumer" not in section_b


def _write_sources(root: Path, sources: dict[str, str]) -> None:
    """Write project files below ``src``."""
    for relative, source in sources.items():
        path = root / "src" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(source, encoding="utf-8")


def test_calls_to_other_modules_are_not_callers(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A same-named call on another module belongs to that module."""
    bodies = stub_jev(_answers())
    _write_sources(
        tmp_path,
        {
            "a.py": (
                "import json\nimport b\n\ndef load():\n    return 1\n\n"
                "def use_b():\n    return b.load()\n\n"
                "def use_json(handle):\n    return json.load(handle)\n"
            ),
            "b.py": "def load():\n    return 2\n",
        },
    )
    edge: FlaggedEdge = (_function("load", "src/a.py"), _function("load", "src/b.py"), 0.97)

    _ = _triage(tmp_path, {"Qwen-AST": [edge]})

    section_a, section_b = _state_source(bodies[0]).split("### Function B")
    assert "use_b" not in section_a
    assert "use_json" not in section_a
    assert "use_b" in section_b
    assert "use_json" not in section_b


def test_import_path_distinguishes_same_named_modules(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """An import of ``x.utils`` does not count for ``y/utils.py``."""
    bodies = stub_jev(_answers())
    _write_sources(
        tmp_path,
        {
            "x/utils.py": "def load():\n    return 1\n",
            "y/utils.py": "def load():\n    return 2\n",
            "app.py": "from x.utils import load\n\ndef consumer():\n    return load()\n",
        },
    )
    edge: FlaggedEdge = (
        _function("load", "src/x/utils.py"),
        _function("load", "src/y/utils.py"),
        0.97,
    )

    _ = _triage(tmp_path, {"Qwen-AST": [edge]})

    section_x, section_y = _state_source(bodies[0]).split("### Function B")
    assert "consumer" in section_x
    assert "consumer" not in section_y


def test_unimported_receiver_is_not_a_caller(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A local object named like the module does not call its function."""
    bodies = stub_jev(_answers())
    _write_sources(
        tmp_path,
        {
            "config.py": "def get():\n    return 1\n",
            "b.py": "def beta():\n    return 2\n",
            "reader.py": "def read(config):\n    return config.get('key')\n",
        },
    )
    edge: FlaggedEdge = (_function("get", "src/config.py"), _function("beta", "src/b.py"), 0.97)

    _ = _triage(tmp_path, {"Qwen-AST": [edge]})

    assert "reader.py" not in _state_source(bodies[0])


@pytest.mark.parametrize(
    "caller",
    [
        "from .a import alpha\n\ndef consumer():\n    return alpha()\n",
        "from .a import alpha as renamed\n\ndef consumer():\n    return renamed()\n",
        "from . import a\n\ndef consumer():\n    return a.alpha()\n",
        "from pkg import a as module\n\ndef consumer():\n    return module.alpha()\n",
        "import pkg.a\n\ndef consumer():\n    return pkg.a.alpha()\n",
    ],
    ids=["relative", "aliased-name", "relative-module", "aliased-module", "dotted-module"],
)
def test_import_forms_reach_the_target(
    tmp_path: Path,
    stub_jev: Callable[[Body], list[Body]],
    caller: str,
) -> None:
    """Common import styles still identify the real caller."""
    bodies = stub_jev(_answers())
    _write_sources(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/a.py": "def alpha():\n    return 1\n",
            "pkg/user.py": caller,
            "b.py": "def beta():\n    return 2\n",
        },
    )
    edge: FlaggedEdge = (_function("alpha", "src/pkg/a.py"), _function("beta", "src/b.py"), 0.97)

    _ = _triage(tmp_path, {"Qwen-AST": [edge]})

    # src/b.py sorts first, so alpha is the second section.
    beta_section, alpha_section = _state_source(bodies[0]).split("### Function B")
    assert "src/pkg/user.py:consumer" in alpha_section
    assert "src/pkg/user.py:consumer" not in beta_section


def test_cleared_pair_survives_warnings_as_errors(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A cleared pair is a pass even under ``-W error``."""
    _ = stub_jev(_answers(0.4, 0.7))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _triage(tmp_path) is None


def test_cache_failure_reports_pairs_as_ungraded(
    tmp_path: Path,
    stub_jev: Callable[[Body], list[Body]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable cache keeps the normal failure report and names the cause."""
    _ = stub_jev(_answers())

    def broken_cache(_config: object) -> None:
        message = "attempt to write a readonly database"
        raise sqlite3.OperationalError(message)

    monkeypatch.setattr(consolidation, "open_cache", broken_cache)

    report = _triage(tmp_path)

    assert report is not None
    assert "Ungraded (1)" in report
    assert "src/a.py:alpha vs src/b.py:beta" in report
    assert "readonly database" in report


def test_time_budget_stops_requests_but_keeps_cached_grades(
    tmp_path: Path,
    stub_jev: Callable[[Body], list[Body]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the budget, cached pairs still grade and new pairs stay ungraded."""
    bodies = stub_jev(_answers())
    _ = _triage(tmp_path)
    assert len(bodies) == 1
    monkeypatch.setattr(consolidation, "_REVIEW_BUDGET_SECONDS", 0.0)
    new_edge: FlaggedEdge = (_function("gamma", "src/c.py"), _function("delta", "src/d.py"), 0.97)

    report = _triage(tmp_path, {"Gemini-AST": [_edge(), new_edge]})

    assert len(bodies) == 1
    assert report is not None
    assert "Needs change (1)" in report
    assert "Ungraded (1)" in report
    assert "time budget" in report


def test_long_function_keeps_second_in_state(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A huge first function cannot push the second out of the model's input."""
    bodies = stub_jev(_answers())
    long_edge: FlaggedEdge = (
        _function("alpha", "src/a.py", "x" * 20_000),
        _function("beta", "src/b.py", "SECOND_SENTINEL"),
        0.97,
    )

    _ = _triage(tmp_path, {"Qwen-AST": [long_edge]})

    source = _state_source(bodies[0])
    assert "SECOND_SENTINEL" in source
    assert len(source) <= jev.MAX_STATE_CHARS


def test_moving_a_function_keeps_its_cached_grade(
    tmp_path: Path, stub_jev: Callable[[Body], list[Body]]
) -> None:
    """A changed line number alone does not trigger another request."""
    bodies = stub_jev(_answers())
    first, second, score = _edge()
    moved = FunctionInfo(
        name=first.name,
        file=first.file,
        start_line=40,
        end_line=41,
        loc=first.loc,
        hash=first.hash,
        text=first.text,
    )

    _ = _triage(tmp_path, {"Gemini-AST": [(first, second, score)]})
    _ = _triage(tmp_path, {"Gemini-AST": [(moved, second, score)]})

    assert len(bodies) == 1
