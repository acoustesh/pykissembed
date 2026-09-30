"""Insert drafted prose into the empty slots of a docstring.

Existing text is never rewritten: a summary is added only when missing, and
every other slot becomes a new entry placed in signature order.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from pykissembed.tools.docfix.common import (
    INDENT,
    bare_name,
    yield_type,
)
from pykissembed.tools.docfix.model import (
    SECTION_ORDER,
    Docstring,
    Entry,
    Section,
    parse,
    parse_entries,
    render_entries,
    trim_trailing,
)

if TYPE_CHECKING:
    from pykissembed.tools.docfix.facts import Facts, Param


def _insert_entry(doc: Docstring, name: str, entry: Entry, order: list[str]) -> None:
    """Insert *entry* into section *name*, creating the section if needed.

    Parameters
    ----------
    doc : Docstring
        Docstring to update in place.
    name : str
        Canonical section name.
    entry : Entry
        Entry to add.
    order : list[str]
        Bare names in their preferred order; the entry goes before the first
        existing entry that sorts after it.
    """
    section = doc.find(name)
    if section is None:
        section = Section([name, "-" * len(name)], [], name, "numpy")
        doc.sections.append(section)
    entries = parse_entries(trim_trailing(section.body)) or []
    rank = {n: i for i, n in enumerate(order)}

    def position(item: Entry) -> int:
        """Rank an entry by signature order; unknown names sort last.

        Parameters
        ----------
        item : Entry
            Entry to rank.

        Returns
        -------
        int
            Signature index of its first name, or ``len(order)``.
        """
        names = item.names()
        return rank.get(bare_name(names[0]) if names else "", len(order))

    # Insert before the first existing entry that belongs after it, so the
    # existing entries keep their relative order even if it was unsorted.
    mine = position(entry)
    index = next((i for i, e in enumerate(entries) if position(e) > mine), len(entries))
    entries.insert(index, entry)
    section.body = render_entries(entries)


def _wrapped(text: str, width: int) -> list[str]:
    """Wrap drafted prose under an entry header.

    Parameters
    ----------
    text : str
        Drafted description.
    width : int
        Maximum line width inside the docstring.

    Returns
    -------
    list[str]
        Lines indented four spaces.
    """
    return textwrap.wrap(text, width=width, initial_indent=INDENT, subsequent_indent=INDENT)


def _param_annotation(params: list[Param], name: str) -> str:
    """Return the annotation of parameter *name*.

    Parameters
    ----------
    params : list[Param]
        Parameters or attributes to search.
    name : str
        Name as it appears in a draft slot.

    Returns
    -------
    str
        The annotation, or ``""`` when unknown.
    """
    return next((p.annotation for p in params if p.name == name), "")


def apply_drafts(text: str, facts: Facts | None, drafts: dict[str, str], *, width: int) -> str:
    """Insert drafted prose into the empty slots of one docstring.

    Existing text is never rewritten: a summary is added only when missing,
    and every other slot becomes a new entry.

    Parameters
    ----------
    text : str
        Cleaned docstring text; may be empty for a new docstring.
    facts : Facts | None
        Facts of the documented definition.
    drafts : dict[str, str]
        Draft text by slot id (``summary``, ``param:x``, ``returns``,
        ``yields``, ``raises:E``, ``attr:x``).
    width : int
        Maximum line width inside the docstring.

    Returns
    -------
    str
        The docstring text with drafts inserted and sections in NumPy order.
    """
    doc = parse(text)
    summary = drafts.get("summary")
    # Never replace a summary that exists, even if the model drafted one.
    if summary and not doc.summary():
        rest = trim_trailing(doc.preamble.body)
        doc.preamble.body = [summary, *([""] if any(line.strip() for line in rest) else []), *rest]
    params = facts.params if facts is not None else []
    order = [bare_name(p.name) for p in params]
    for slot, value in drafts.items():
        kind, _, name = slot.partition(":")
        entry = _draft_entry(kind, name, value, facts, width)
        if entry is not None:
            section, new_entry = entry
            _insert_entry(doc, section, new_entry, order)
    # New sections were appended at the end; move them to their NumPy place.
    # Unknown titles sort last because they have no defined position.
    doc.sections = [
        doc.preamble,
        *sorted(
            doc.sections[1:],
            key=lambda s: (
                SECTION_ORDER.index(s.name) if s.name in SECTION_ORDER else len(SECTION_ORDER)
            ),
        ),
    ]
    return doc.render_normalized()


def _draft_entry(
    kind: str, name: str, value: str, facts: Facts | None, width: int
) -> tuple[str, Entry] | None:
    """Build the section entry for one drafted slot.

    Parameters
    ----------
    kind : str
        Slot kind before the colon.
    name : str
        Slot argument after the colon, if any.
    value : str
        Drafted description.
    facts : Facts | None
        Facts of the documented definition.
    width : int
        Maximum line width inside the docstring.

    Returns
    -------
    tuple[str, Entry] | None
        Target section and entry, or ``None`` for the summary slot or an
        untyped return.
    """
    desc = _wrapped(value, width)
    # Module docstrings only ever get a summary slot, handled by the caller.
    if facts is None:
        return None
    if kind == "param":
        annotation = _param_annotation(facts.params, name)
        return "Parameters", Entry(f"{name} : {annotation}" if annotation else name, desc)
    if kind == "attr":
        annotation = _param_annotation(facts.attributes, name)
        return "Attributes", Entry(f"{name} : {annotation}" if annotation else name, desc)
    if kind == "raises":
        return "Raises", Entry(name, desc)
    # Returns/Yields need a type line; without an annotation there is none
    # to write, and inventing one is exactly what the tool must not do.
    if kind == "returns" and facts.returns:
        return "Returns", Entry(facts.returns, desc)
    if kind == "yields" and yield_type(facts):
        return "Yields", Entry(yield_type(facts), desc)
    return None
