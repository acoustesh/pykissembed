"""Tests for the docfix docstring fixer (``pykissembed.tools.docfix``)."""

from __future__ import annotations

import ast
import inspect
import json
import sqlite3
from collections import Counter
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest
import requests

from pykissembed.paths import iter_py_files
from pykissembed.tools.docfix import Request, main, selection
from pykissembed.tools.docfix.common import default_selection
from pykissembed.tools.docfix.draft import OpenRouterDrafter, validate_drafts
from pykissembed.tools.docfix.engine import Options, transform
from pykissembed.tools.docfix.facts import collect
from pykissembed.tools.docfix.fill import apply_drafts
from pykissembed.tools.docfix.model import parse
from pykissembed.tools.docfix.rules import fix_docstring
from pykissembed.tools.docfix.source import Ruff, Slot, guard, locate, render_token, words_kept
from pykissembed.tools.docfix.writer import process_file

if TYPE_CHECKING:
    from pykissembed.tools.docfix.common import Outcome
    from pykissembed.tools.docfix.facts import DefNode, Facts

_REPO = Path(__file__).resolve().parent.parent
# Ruff's defaults differ from any project's config, so unit tests run the
# docfix rules alone; DF002 (ruff delegation) is exercised separately.
_NO_RUFF = Ruff(executable=None)
_SELECTED = default_selection() - {"DF002"}


def _node(source: str, name: str) -> tuple[DefNode, bool]:
    """Find the definition *name* in *source*.

    Parameters
    ----------
    source : str
        Module source.
    name : str
        Definition name.

    Returns
    -------
    tuple[DefNode, bool]
        The first matching definition and whether it sits in a class body.
    """
    tree = ast.parse(dedent(source))
    parents = {id(c): p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and n.name == name
    )
    return node, isinstance(parents.get(id(node)), ast.ClassDef)


def _facts(source: str, name: str) -> Facts:
    """Collect facts for the definition *name* in *source*.

    Parameters
    ----------
    source : str
        Module source.
    name : str
        Definition name.

    Returns
    -------
    Facts
        Facts of the first matching definition.
    """
    node, in_class = _node(source, name)
    return collect(node, in_class=in_class)


def _fix(source: str, name: str) -> Outcome:
    """Run the rules on the docstring of *name* in *source*.

    Parameters
    ----------
    source : str
        Module source.
    name : str
        Definition name.

    Returns
    -------
    Outcome
        The rule outcome.
    """
    node, _ = _node(source, name)
    return fix_docstring(ast.get_docstring(node) or "", _facts(source, name), _SELECTED)


def _codes(outcome: Outcome, action: str) -> set[str]:
    """Return the codes of *outcome*'s findings with *action*.

    Parameters
    ----------
    outcome : Outcome
        Rule outcome.
    action : str
        Finding action to keep.

    Returns
    -------
    set[str]
        Matching codes.
    """
    return {f.code for f in outcome.findings if f.action == action}


def _all_docstrings() -> list[str]:
    """Return every cleaned docstring in the package and its tests.

    Returns
    -------
    list[str]
        Docstrings of modules, classes and functions.
    """
    found: list[str] = []
    for base in (_REPO / "pykissembed", _REPO / "tests"):
        for path in iter_py_files(base):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(
                    node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                ):
                    doc = ast.get_docstring(node)
                    if doc is not None:
                        found.append(doc)
    return found


class TestModel:
    """The docstring model is lossless."""

    @staticmethod
    def test_round_trip_over_repo_corpus() -> None:
        """Every docstring in the repo renders back exactly after parsing."""
        docs = _all_docstrings()
        assert len(docs) > 100
        for doc in docs:
            assert parse(doc).render() == doc

    @staticmethod
    def test_sections_are_recognised() -> None:
        """NumPy, Google and bare titles each start a section."""
        doc = parse("Sum.\n\nArgs:\n    x: y\n\nReturns\n-------\nint\n    Total.")
        assert [s.name for s in doc.sections] == [None, "Parameters", "Returns"]
        assert [s.style for s in doc.sections[1:]] == ["google", "numpy"]

    @staticmethod
    def test_lowercase_entry_is_not_a_google_title() -> None:
        """A ``methods:`` attribute entry is not a ``Methods:`` section."""
        doc = parse("Sum.\n\nAttributes\n----------\nmethods:\n    Nested.")
        assert [s.name for s in doc.sections] == [None, "Attributes"]


class TestFacts:
    """Facts are derived from the AST alone."""

    @staticmethod
    def test_signature_drops_self_and_keeps_stars() -> None:
        """Methods lose ``self``; variadics keep their stars."""
        facts = _facts(
            """
            class A:
                def m(self, a: int, *args: str, b=1, **kw) -> int:
                    return a
            """,
            "m",
        )
        assert [p.name for p in facts.params] == ["a", "*args", "b", "**kw"]
        assert facts.implicit == "self"
        assert facts.documents_return

    @staticmethod
    def test_escaping_and_caught_raises() -> None:
        """Only exceptions not caught locally escape; bare re-raise counts."""
        facts = _facts(
            """
            def f(x):
                try:
                    raise KeyError(x)
                except LookupError:
                    pass
                try:
                    g()
                except OSError:
                    raise
                if x:
                    raise ValueError("bad")
                def inner():
                    raise TypeError
            """,
            "f",
        )
        assert set(facts.raises) == {"OSError", "ValueError"}

    @staticmethod
    def test_generator_and_own_scope_returns() -> None:
        """A nested function's ``return`` does not make the outer one return."""
        facts = _facts(
            """
            def gen():
                def helper():
                    return 1
                yield helper()
            """,
            "gen",
        )
        assert facts.is_generator
        assert not facts.returns_value


class TestDeterministicFixes:
    """Each fix only moves, relabels or annotates existing text."""

    @staticmethod
    def test_google_sections_convert_to_numpy() -> None:
        """``Args:``/``Returns:``/``Raises:`` become NumPy sections (DF010)."""
        outcome = _fix(
            '''
            def f(a: int, b) -> str:
                """Join things.

                Args:
                    a (int): First value.
                    b: Second value,
                        spanning two lines.

                Returns:
                    The joined text.

                Raises:
                    ValueError: If empty.
                """
                raise ValueError
            ''',
            "f",
        )
        assert outcome.text == dedent(
            """\
            Join things.

            Parameters
            ----------
            a : int
                First value.
            b
                Second value,
                spanning two lines.

            Returns
            -------
            str
                The joined text.

            Raises
            ------
            ValueError
                If empty."""
        )

    @staticmethod
    def test_sphinx_fields_convert_to_numpy() -> None:
        """A trailing reST field list becomes NumPy sections (DF010)."""
        outcome = _fix(
            '''
            def f(a, b):
                """Add.

                :param a: Left.
                :type a: int
                :param b: Right.
                :returns: The sum.
                :rtype: int
                """
                return a + b
            ''',
            "f",
        )
        assert "a : int\n    Left." in outcome.text
        assert "Returns\n-------\nint\n    The sum." in outcome.text
        assert ":param" not in outcome.text

    @staticmethod
    def test_headers_underlines_duplicates_and_order() -> None:
        """Aliases, underlines, duplicates and order are fixed (DF011-DF014)."""
        outcome = _fix(
            '''
            def f(a):
                """Do.

                Notes
                -----
                Careful.

                Example
                ---
                >>> f(1)

                Parameters
                ----------
                a
                    One.

                Parameters
                ----------
                """
            ''',
            "f",
        )
        assert _codes(outcome, "fix") >= {"DF011", "DF012", "DF013", "DF014"}
        titles = [s.name for s in parse(outcome.text).sections[1:]]
        assert titles == ["Parameters", "Notes", "Examples"]

    @staticmethod
    def test_parameter_entries_follow_the_signature() -> None:
        """Entry syntax, stars, self, order and types are fixed (DF020-DF026)."""
        outcome = _fix(
            '''
            class A:
                def m(self, a: int, *args: str) -> None:
                    """Do.

                    Parameters
                    ----------
                    args -- Extra.
                    self
                        The instance.
                    a: int
                        First.
                    """
            ''',
            "m",
        )
        assert _codes(outcome, "fix") >= {"DF020", "DF021", "DF024", "DF025"}
        assert outcome.text.endswith(
            "Parameters\n----------\na : int\n    First.\n*args : str\n    Extra."
        )
        assert outcome.allowance["instance"] == 1

    @staticmethod
    def test_colon_prose_is_only_converted_for_known_names() -> None:
        """``name: prose`` becomes an entry only when *name* is a parameter."""
        outcome = _fix(
            '''
            def f(a):
                """Do.

                Parameters
                ----------
                a: The value.
                Note: this is prose.
                """
            ''',
            "f",
        )
        assert "a\n    The value." in outcome.text
        assert "Note: this is prose." in outcome.text

    @staticmethod
    def test_returns_typed_and_generators_yield() -> None:
        """Untyped Returns gains its annotation; generators use Yields (DF032, DF033)."""
        typed = _fix(
            '''
            def f() -> int:
                """Do.

                Returns
                -------
                The answer to it all.
                """
                return 42
            ''',
            "f",
        )
        assert typed.text.endswith("Returns\n-------\nint\n    The answer to it all.")
        generator = _fix(
            '''
            def g():
                """Do.

                Returns
                -------
                int
                    Numbers.
                """
                yield 1
            ''',
            "g",
        )
        assert "Yields\n------\nint" in generator.text

    @staticmethod
    def test_conforming_docstring_is_untouched() -> None:
        """A compliant docstring yields no fix and identical text."""
        source = '''
            def f(a: int) -> int:
                """Double.

                Parameters
                ----------
                a : int
                    Value.

                Returns
                -------
                int
                    Twice *a*.
                """
                return 2 * a
            '''
        outcome = _fix(source, "f")
        assert not _codes(outcome, "fix")
        assert outcome.text == inspect.cleandoc(ast.get_docstring(_node(source, "f")[0]) or "")


class TestReports:
    """Findings that need prose or a human are never auto-fixed."""

    @staticmethod
    def test_missing_items_become_prose_slots() -> None:
        """Undocumented params, returns and raises are draftable slots."""
        outcome = _fix(
            '''
            def f(a: int, b: int) -> int:
                """Do."""
                if a:
                    raise ValueError
                return a + b
            ''',
            "f",
        )
        slots = {f.slot for f in outcome.findings if f.action == "prose"}
        assert slots == {"param:a", "param:b", "returns", "raises:ValueError"}

    @staticmethod
    def test_stale_and_contradicting_content_is_reported() -> None:
        """Unknown params, unraised exceptions and ``-> None`` Returns are reported."""
        outcome = _fix(
            '''
            def f(a) -> None:
                """Do.
                More.

                Parameters
                ----------
                alpha
                    Old name.

                Returns
                -------
                int
                    Nothing really.

                Raises
                ------
                KeyError
                    Never here.
                """
            ''',
            "f",
        )
        assert _codes(outcome, "report") >= {"DF023", "DF031", "DF041", "DF051"}
        message = next(f.message for f in outcome.findings if f.code == "DF023")
        assert "renamed to 'a'" in message


class TestSource:
    """Token rendering and the guards."""

    @staticmethod
    def _slot(source: str) -> Slot:
        """Return the docstring slot of the first function in *source*.

        Parameters
        ----------
        source : str
            Module source.

        Returns
        -------
        Slot
            The slot of owner 1 (the first definition).
        """
        text = dedent(source)
        return next(s for s in locate(text, ast.parse(text)) if s.owner == 1)

    @staticmethod
    def test_render_keeps_quotes_prefix_and_trailing_blank() -> None:
        """Rendering keeps the prefix, quote style and closing blank line."""
        slot = TestSource._slot("def f():\n    r'''Do.\n\n    More.\n\n    '''\n")
        assert render_token(slot, "Do.\n\nMore and more.", "\n") == (
            "r'''Do.\n\n    More and more.\n\n    '''"
        )

    @staticmethod
    def test_render_refuses_unsafe_text() -> None:
        """A delimiter clash or a same-line docstring cannot be rendered."""
        slot = TestSource._slot('def f():\n    """Do."""\n')
        assert render_token(slot, 'Say """hi""".', "\n") is None
        inline = TestSource._slot('def f(): """Do."""\n')
        assert render_token(inline, "Do.\n\nMore.", "\n") is None

    @staticmethod
    def test_guard_rejects_code_and_comment_changes() -> None:
        """Changing code or comments fails the guard; docstrings alone pass."""
        original = 'def f():\n    """Do."""\n    return 1  # one\n'
        assert not guard(original, original.replace('"Do."', '"Do it."'), frozenset())
        assert guard(original, original.replace("return 1", "return 2"), frozenset())
        assert guard(original, original.replace("# one", "# two"), frozenset())

    @staticmethod
    def test_words_kept_detects_lost_prose() -> None:
        """Dropping a word fails unless it is allowed."""
        assert words_kept("Sum of a and b.", "Sum of a and b, exactly.", Counter())
        assert not words_kept("Sum of a and b.", "Sum of a.", Counter())
        assert words_kept("Sum of a and b.", "Sum of a.", Counter({"and": 1, "b": 1}))


class TestEngine:
    """Whole-file transformation, verification and writing."""

    @staticmethod
    def test_transform_is_idempotent_over_repo_corpus() -> None:
        """Fixing the repo's own files twice changes nothing the second time."""
        options = Options(_SELECTED)
        for path in iter_py_files(_REPO / "pykissembed"):
            source = path.read_text(encoding="utf-8")
            first = transform(str(path), source, path, options, _NO_RUFF)
            second = transform(str(path), first.updated, path, options, _NO_RUFF)
            assert second.updated == first.updated, path
            assert not guard(source, first.updated, frozenset()), path

    @staticmethod
    def test_crlf_file_round_trips(tmp_path: Path) -> None:
        """CRLF line endings survive a rewrite."""
        path = tmp_path / "m.py"
        text = 'def f(a: int):\r\n    """Do.\r\n\r\n    Args:\r\n        a: Value.\r\n    """\r\n'
        _ = path.write_bytes(text.encode())
        result = process_file(
            path, "m.py", Options(_SELECTED, write=True, allow_dirty=True), _NO_RUFF
        )
        assert result.written, result.error
        data = path.read_bytes().decode()
        assert "\r\n    Parameters\r\n    ----------\r\n    a : int\r\n" in data
        assert "\n" not in data.replace("\r\n", "")

    @staticmethod
    def test_escape_bearing_docstring_is_skipped(tmp_path: Path) -> None:
        """A docstring with escape sequences is reported, never rewritten."""
        path = tmp_path / "m.py"
        original = 'def f(a):\n    """Do\\tit.\n\n    Args:\n        a: Value.\n    """\n'
        _ = path.write_text(original)
        result = process_file(
            path, "m.py", Options(_SELECTED, write=True, allow_dirty=True), _NO_RUFF
        )
        assert path.read_text() == original
        assert any(f.finding.code == "DF001" for f in result.findings)

    @staticmethod
    def test_dry_run_writes_nothing(tmp_path: Path) -> None:
        """Without ``write`` the file is untouched but the change is reported."""
        path = tmp_path / "m.py"
        original = 'def f(a):\n    """Do.\n\n    Args:\n        a: Value.\n    """\n'
        _ = path.write_text(original)
        result = process_file(path, "m.py", Options(_SELECTED), _NO_RUFF)
        assert path.read_text() == original
        assert result.updated != original
        assert not result.written

    @staticmethod
    def test_symbol_filter_limits_edits() -> None:
        """Only the selected symbol is rewritten."""
        source = dedent(
            '''
            def f(a):
                """F.

                Args:
                    a: Value.
                """

            def g(a):
                """G.

                Args:
                    a: Value.
                """
            '''
        )
        options = Options(_SELECTED, symbols=frozenset({("m.py", "g")}))
        result = transform("m.py", source, Path("m.py"), options, _NO_RUFF)
        assert "F.\n\n    Args:" in result.updated
        assert "G.\n\n    Parameters" in result.updated


class _FakeResponse:
    """A minimal ``requests`` response carrying a chat completion."""

    def __init__(self, content: str) -> None:
        """Store the message content to return.

        Parameters
        ----------
        content : str
            Assistant message content.
        """
        self.content = content

    @staticmethod
    def raise_for_status() -> None:
        """Accept every status."""

    def json(self) -> dict[str, object]:
        """Return a chat-completions body.

        Returns
        -------
        dict[str, object]
            Body with one choice.
        """
        return {"choices": [{"message": {"content": self.content}}]}


class TestDraft:
    """Opt-in drafting is validated, cached and slot-limited."""

    @staticmethod
    def test_validation_rejects_bad_replies() -> None:
        """Missing keys, delimiters and bad summaries are refused."""
        assert validate_drafts({"summary": "Do it."}, ["summary"]) == {"summary": "Do it."}
        assert validate_drafts({"summary": "Do it."}, ["summary", "returns"]) is None
        assert validate_drafts({"returns": 'The """x"""'}, ["returns"]) is None
        assert validate_drafts({"summary": "No period"}, ["summary"]) is None

    @staticmethod
    def test_drafter_caches_replies(monkeypatch: pytest.MonkeyPatch) -> None:
        """A second identical request is served from the SQLite cache."""
        calls: list[object] = []

        def fake_post(*_args: object, **kwargs: object) -> _FakeResponse:
            """Record the request and reply with fixed drafts.

            Parameters
            ----------
            *_args : object
                Ignored positional arguments.
            **kwargs : object
                Request keyword arguments; recorded.

            Returns
            -------
            _FakeResponse
                A reply filling the requested slot.
            """
            calls.append(kwargs)
            return _FakeResponse(json.dumps({"param:a": "The value to double."}))

        monkeypatch.setattr(requests, "post", fake_post)
        drafter = OpenRouterDrafter("key", sqlite3.connect(":memory:"), max_calls=1)
        first = drafter.draft("def f(a): ...", ["param:a"])
        second = drafter.draft("def f(a): ...", ["param:a"])
        assert first == second == {"param:a": "The value to double."}
        assert len(calls) == 1

    @staticmethod
    def test_apply_drafts_only_fills_slots() -> None:
        """Drafts add entries in signature order and keep existing text."""
        facts = _facts("def f(a: int, b: str) -> int:\n    return 1\n", "f")
        text = apply_drafts(
            "Do.\n\nParameters\n----------\nb : str\n    Kept.",
            facts,
            {"param:a": "First.", "returns": "One."},
            width=60,
        )
        assert text == (
            "Do.\n\nParameters\n----------\na : int\n    First.\nb : str\n    Kept.\n\n"
            "Returns\n-------\nint\n    One."
        )


class TestCli:
    """The command-line entry point."""

    @staticmethod
    def test_selection_validates_codes() -> None:
        """Unknown codes are refused; opt-in codes extend the defaults."""
        assert "DF052" in selection("DF052", "")
        assert "DF022" in selection("DF052", "")
        assert selection("DF020", "") == {"DF020"}
        with pytest.raises(ValueError, match="DF999"):
            _ = selection("DF999", "")

    @staticmethod
    def test_main_reports_then_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A report run changes nothing and ``--check`` fails; ``--write`` fixes."""
        monkeypatch.chdir(tmp_path)
        path = tmp_path / "m.py"
        _ = path.write_text('def f(a):\n    """Do.\n\n    Args:\n        a: Value.\n    """\n')
        report, status = main(Request(paths=[path], check=True, ignore="DF002"))
        assert status == 1
        assert "DF010 converted Google-style Parameters [fix]" in report
        _, status = main(Request(paths=[path], write=True, allow_dirty=True, ignore="DF002"))
        assert status == 0
        assert "Parameters\n    ----------\n    a\n        Value." in path.read_text()
