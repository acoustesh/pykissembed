"""Shared rule vocabulary: codes, findings, the rule context and text helpers.

Each rule has a ``DFnnn`` code and one of four actions:

``fix``
    A deterministic rewrite that only moves, re-formats or re-labels existing
    text, or adds a fact the AST proves (a name or an annotation).
``prose``
    Something the convention requires that needs new wording; it is filled
    only by the opt-in drafting pass, otherwise listed on the worklist.
``report``
    A likely defect that the tool never changes automatically.
``skip``
    The docstring could not be modelled or rewritten safely and is left as is.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pykissembed.tools.docfix.model import (
    canonical_name,
    dedent_block,
)

if TYPE_CHECKING:
    from pykissembed.tools.docfix.facts import Facts

CODES: dict[str, str] = {
    "DF001": "docstring cannot be modelled or rewritten safely",
    "DF002": "layout fixed by ruff's docstring (D) rules",
    "DF003": "edit reverted because it lowered the Jev rubric level",
    "DF004": "drafted prose inserted",
    "DF010": "Google or Sphinx style section or field",
    "DF011": "section title is an alias or wrongly cased",
    "DF012": "section underline missing or wrong",
    "DF013": "duplicate section",
    "DF014": "sections out of NumPy order",
    "DF020": "entry header not written as `name : type`",
    "DF021": "parameters documented out of signature order",
    "DF022": "parameter not documented",
    "DF023": "documented parameter not in the signature",
    "DF024": "variadic parameter documented without its stars",
    "DF025": "implicit self/cls documented",
    "DF026": "parameter type missing while an annotation exists",
    "DF027": "documented type contradicts the annotation",
    "DF030": "value-returning function has no Returns section",
    "DF031": "Returns section on a function annotated to return None",
    "DF032": "Returns entry has no type while an annotation exists",
    "DF033": "generator documents Returns instead of Yields",
    "DF034": "generator has no Yields section",
    "DF040": "exception raised here is not in Raises",
    "DF041": "Raises names an exception not raised here",
    "DF050": "docstring has no summary",
    "DF051": "summary spans more than one line",
    "DF052": "definition has no docstring",
    "DF060": "constructor parameters documented in both class and __init__",
    "DF061": "constructor parameters documented in __init__ instead of the class",
    "DF062": "public attribute missing from Attributes",
}

# Opt-in codes: selected only when named explicitly.
OPT_IN = frozenset({"DF052"})
# An innermost [...] subscript, removed repeatedly by looks_like_type.
_BRACKETS = re.compile(r"\[[^\[\]]*\]")
INDENT = "    "


@dataclass(frozen=True, slots=True)
class Finding:
    """One rule result for one docstring."""

    code: str
    message: str
    action: str
    slot: str = ""


@dataclass(slots=True)
class Outcome:
    """The result of checking and fixing one docstring."""

    text: str
    findings: list[Finding] = field(default_factory=list)
    allowance: Counter[str] = field(default_factory=Counter[str])


@dataclass(slots=True)
class RuleContext:
    """Mutable state shared by the rule functions for one docstring.

    Attributes
    ----------
    selected : frozenset[str]
        Rule codes that may report or fix.
    facts : Facts | None
        Facts of the documented definition; ``None`` for a module.
    width : int
        Longest line a fix may add inside the docstring.
    findings : list[Finding]
        Findings recorded so far, in rule order.
    allowance : Counter[str]
        Words a fix declared it may delete (checked by the word guard).
    changed : bool
        Whether any fix modified the docstring model.
    """

    selected: frozenset[str]
    facts: Facts | None
    width: int = 88
    findings: list[Finding] = field(default_factory=list)
    allowance: Counter[str] = field(default_factory=Counter[str])
    changed: bool = False

    def fix(self, code: str, message: str) -> bool:
        """Record a deterministic fix when *code* is selected.

        Parameters
        ----------
        code : str
            Rule code.
        message : str
            What is being fixed.

        Returns
        -------
        bool
            ``True`` when the caller should apply the fix.
        """
        # A deselected rule neither reports nor fixes, like ruff's --ignore.
        if code not in self.selected:
            return False
        self.findings.append(Finding(code, message, "fix"))
        self.changed = True
        return True

    def note(self, code: str, message: str, action: str, slot: str = "") -> None:
        """Record a non-fixing finding when *code* is selected.

        Parameters
        ----------
        code : str
            Rule code.
        message : str
            What is wrong.
        action : str
            ``"prose"``, ``"report"`` or ``"skip"``.
        slot : str
            Draft slot id for a draftable prose finding.
        """
        if code in self.selected:
            self.findings.append(Finding(code, message, action, slot))


def default_selection() -> frozenset[str]:
    """Return every code that runs without being named.

    Returns
    -------
    frozenset[str]
        All codes except the opt-in ones.
    """
    return frozenset(CODES) - OPT_IN


def prose_words(text: str) -> Counter[str]:
    """Count the words of a docstring, ignoring structural markup.

    Section titles, underlines and reST field roles are structure, so
    renaming or converting them never counts as losing prose.

    Parameters
    ----------
    text : str
        Cleaned docstring text.

    Returns
    -------
    Counter[str]
        Word multiset of the prose and entry text.
    """
    words: Counter[str] = Counter()
    for raw in text.split("\n"):
        line = raw.strip()
        # Titles and underlines are structure that fixes legitimately rename.
        if not line or set(line) == {"-"} or canonical_name(line) is not None:
            continue
        # A leading reST role (":param") is markup that conversion drops.
        words.update(re.findall(r"\w+", re.sub(r"^:\w+", "", line)))
    return words


def bare_name(name: str) -> str:
    """Strip stars and reST escapes from a documented name.

    Parameters
    ----------
    name : str
        Name as documented.

    Returns
    -------
    str
        The plain identifier.
    """
    return name.replace("\\", "").lstrip("*")


def looks_like_type(text: str) -> bool:
    """Return whether *text* reads as a type expression rather than prose.

    Parameters
    ----------
    text : str
        Candidate type text.

    Returns
    -------
    bool
        ``True`` for forms like ``int``, ``list[str] | None`` or
        ``str, optional``; ``False`` for sentences.
    """
    flat = text.strip()
    # Strip innermost brackets repeatedly, so dict[str, list[int]] -> dict:
    # commas and spaces inside subscripts must not read as prose.
    while _BRACKETS.search(flat):
        flat = _BRACKETS.sub("", flat)
    flat = flat.replace(", optional", "").replace(" | ", "|").replace(" or ", "|")
    return bool(re.fullmatch(r"[\w.|*]+", flat))


def desc_lines(first: str, rest: list[str]) -> list[str]:
    """Build indented description lines from inline text and continuations.

    Parameters
    ----------
    first : str
        Inline description text from the header line; may be empty.
    rest : list[str]
        Continuation lines at any indentation.

    Returns
    -------
    list[str]
        Lines indented four spaces under the entry header.
    """
    lines = ([first.strip()] if first.strip() else []) + dedent_block(rest)
    # Relative indentation inside the description (nested lists) survives.
    return [INDENT + line if line else "" for line in lines]


def returns_annotation(facts: Facts | None) -> str:
    """Return the annotation to use as a Returns type, if the code has one.

    Parameters
    ----------
    facts : Facts | None
        Facts of the documented function.

    Returns
    -------
    str
        The return annotation, or ``""`` when absent or ``None``-like.
    """
    if facts is None or not facts.documents_return:
        return ""
    return facts.returns


def yield_type(facts: Facts) -> str:
    """Return the yielded type from an ``Iterator[X]``-style annotation.

    Parameters
    ----------
    facts : Facts
        Facts of a generator function.

    Returns
    -------
    str
        The first type argument, or ``""`` when it cannot be read.
    """
    match = re.fullmatch(
        r"(?:[\w.]*\.)?(?:Async)?(?:Iterator|Iterable|Generator)\[(?P<args>.+)\]", facts.returns
    )
    if match is None:
        return ""
    # Generator[Y, S, R] yields Y: cut at the first top-level comma only,
    # so a nested type like dict[str, int] stays whole.
    depth, cut = 0, len(match["args"])
    for index, char in enumerate(match["args"]):
        depth += {"[": 1, "]": -1}.get(char, 0)
        if char == "," and depth == 0:
            cut = index
            break
    return match["args"][:cut].strip()
