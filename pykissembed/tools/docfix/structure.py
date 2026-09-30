"""Section structure fixes: Google/Sphinx conversion, titles, duplicates, order.

These rules (DF010 to DF014) only re-label, re-indent or move existing text;
entry descriptions are carried over verbatim.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pykissembed.tools.docfix.common import (
    RuleContext,
    desc_lines,
    looks_like_type,
    returns_annotation,
)
from pykissembed.tools.docfix.model import (
    ENTRY_SECTIONS,
    SECTION_ORDER,
    TYPE_FIRST_SECTIONS,
    Docstring,
    Entry,
    Section,
    SphinxField,
    dedent_block,
    parse_entries,
    render_entries,
    split_sphinx_fields,
    trim_trailing,
)

if TYPE_CHECKING:
    from pykissembed.tools.docfix.facts import Facts

# Google parameter line: "name (type): text", with the type optional.
_GOOGLE_PARAM = re.compile(r"^(?P<name>\*{0,2}\w+)\s*(?:\((?P<type>[^()]*)\))?\s*:\s*(?P<desc>.*)$")
# Google Returns/Yields line: "type: text" (the colon needs a space after it,
# so "http://x" or "a:b" slices are not split).
_TYPED_DESC = re.compile(r"^(?P<type>[^:]+?)\s*:\s+(?P<desc>.+)$")
# reST field roles accepted by Sphinx for each NumPy section.
_PARAM_ROLES = frozenset({"param", "parameter", "arg", "argument", "key", "keyword"})
_ATTR_ROLES = frozenset({"ivar", "var", "cvar"})
_RETURN_ROLES = frozenset({"return", "returns"})
_RAISE_ROLES = frozenset({"raise", "raises", "except", "exception"})
_YIELD_ROLES = frozenset({"yield", "yields"})


def _google_param_entries(body: list[str]) -> list[str] | None:
    """Convert a Google ``name (type): desc`` list to NumPy entries.

    Parameters
    ----------
    body : list[str]
        Dedented section body.

    Returns
    -------
    list[str] | None
        NumPy body lines, or ``None`` when an entry does not match.
    """
    entries = parse_entries(body)
    if entries is None:
        return None
    out: list[Entry] = []
    for entry in entries:
        match = _GOOGLE_PARAM.match(entry.header)
        # All or nothing: a half-converted section would read worse than none.
        if match is None:
            return None
        header = match["name"] + (f" : {match['type'].strip()}" if match["type"] else "")
        out.append(Entry(header, desc_lines(match["desc"], entry.desc)))
    return render_entries(out)


def _google_typed_entries(body: list[str], facts: Facts | None) -> list[str] | None:
    """Convert a Google Returns/Yields/Raises body to NumPy entries.

    ``type: desc`` lines become entries; a body that is pure prose becomes
    one entry typed by the function's return annotation.

    Parameters
    ----------
    body : list[str]
        Dedented section body.
    facts : Facts | None
        Facts of the documented function.

    Returns
    -------
    list[str] | None
        NumPy body lines, or ``None`` when no type can be established.
    """
    first = next((line for line in body if line.strip()), "")
    # The first line decides the layout: typed entries, or one prose block
    # whose continuation lines sit at the same indentation as the first.
    match = _TYPED_DESC.match(first)
    if match and looks_like_type(match["type"]):
        entries = parse_entries(body)
        if entries is None:
            return None
        out: list[Entry] = []
        for entry in entries:
            typed = _TYPED_DESC.match(entry.header)
            if typed is None or not looks_like_type(typed["type"]):
                return None
            out.append(Entry(typed["type"].strip(), desc_lines(typed["desc"], entry.desc)))
        return render_entries(out)
    annotation = returns_annotation(facts)
    # NumPy Returns needs a type line; without an annotation there is none.
    if not annotation:
        return None
    return [annotation, *desc_lines("", body)]


def _convert_google(section: Section, facts: Facts | None) -> list[str] | None:
    """Return the NumPy body for a Google-style section.

    Parameters
    ----------
    section : Section
        A section parsed with ``style == "google"``.
    facts : Facts | None
        Facts of the documented function.

    Returns
    -------
    list[str] | None
        The converted body, or ``None`` when conversion is not safe.
    """
    body = dedent_block(section.body)
    if section.name in {"Raises", "Warns"}:
        return _google_param_entries(body)
    if section.name in TYPE_FIRST_SECTIONS:
        return _google_typed_entries(trim_trailing(body), facts)
    if section.name in ENTRY_SECTIONS:
        return _google_param_entries(body)
    # Prose sections (Notes, Examples...) only lose the Google indentation.
    return body


def _sphinx_sections(
    fields: list[SphinxField], facts: Facts | None
) -> dict[str, list[Entry]] | None:
    """Group reST fields into NumPy section entries.

    Parameters
    ----------
    fields : list[SphinxField]
        Parsed ``:role arg: text`` fields.
    facts : Facts | None
        Facts of the documented function, used for an untyped ``:returns:``.

    Returns
    -------
    dict[str, list[Entry]] | None
        Entries per canonical section, or ``None`` on an unknown role or an
        untyped return.
    """
    types = {f.arg: " ".join(f.text) for f in fields if f.role == "type"}
    rtype = next((" ".join(f.text) for f in fields if f.role in {"rtype", "ytype"}), "")
    out: dict[str, list[Entry]] = {}
    for item in fields:
        if item.role in {"type", "rtype", "ytype"}:
            continue
        # ":rtype:" wins; otherwise the annotation supplies the type line.
        placed = _sphinx_entry(item, types, rtype or returns_annotation(facts))
        if placed is None:
            return None
        out.setdefault(placed[0], []).append(placed[1])
    return out


def _sphinx_entry(item: SphinxField, types: dict[str, str], rtype: str) -> tuple[str, Entry] | None:
    """Convert one reST field into a section name and entry.

    Parameters
    ----------
    item : SphinxField
        The field.
    types : dict[str, str]
        ``:type name:`` values by parameter name.
    rtype : str
        Return or yield type, or ``""`` when unknown.

    Returns
    -------
    tuple[str, Entry] | None
        Target section and entry, or ``None`` when the role is unsupported.
    """
    desc = desc_lines("", [t for t in item.text if t] or [])
    arg = item.arg.split()
    if item.role in _PARAM_ROLES | _ATTR_ROLES and arg:
        # ":param int x:" puts the type inline, before the name.
        name = arg[-1]
        kind = " ".join(arg[:-1]) or types.get(name, "")
        section = "Attributes" if item.role in _ATTR_ROLES else "Parameters"
        return section, Entry(f"{name} : {kind}" if kind else name, desc)
    if item.role in _RAISE_ROLES and arg:
        return "Raises", Entry(item.arg, desc)
    if item.role in _RETURN_ROLES | _YIELD_ROLES and rtype:
        return ("Returns" if item.role in _RETURN_ROLES else "Yields"), Entry(rtype, desc)
    return None


def convert_sphinx(doc: Docstring, ctx: RuleContext) -> None:
    """Replace a trailing reST field list with NumPy sections (DF010).

    Parameters
    ----------
    doc : Docstring
        Docstring to update in place.
    ctx : RuleContext
        Rule context.
    """
    split = split_sphinx_fields(doc.preamble.body)
    if split is None:
        return
    prose, fields = split
    grouped = _sphinx_sections(fields, ctx.facts)
    if grouped is None:
        ctx.note("DF010", "reST field list with unsupported or untyped fields", "report")
        return
    if not ctx.fix("DF010", "converted reST field list to NumPy sections"):
        return
    doc.preamble.body = prose
    # Merge into an existing section rather than creating a duplicate.
    for name, entries in grouped.items():
        body = render_entries(entries)
        existing = doc.find(name)
        if existing is not None:
            existing.body = [*trim_trailing(existing.body), *body]
        else:
            doc.sections.append(Section([name, "-" * len(name)], body, name, "numpy"))


def normalize_headers(doc: Docstring, ctx: RuleContext) -> None:
    """Canonicalise every known section title and underline.

    Parameters
    ----------
    doc : Docstring
        Docstring to update in place.
    ctx : RuleContext
        Rule context.
    """
    for section in doc.sections[1:]:
        # An unknown title ("Usage") is kept verbatim: it may be intentional.
        if section.name is None:
            continue
        wanted = [section.name, "-" * len(section.name)]
        if section.style == "google":
            body = _convert_google(section, ctx.facts)
            if body is None:
                ctx.note("DF010", f"Google-style {section.name} could not be converted", "report")
            elif ctx.fix("DF010", f"converted Google-style {section.name}"):
                section.header, section.body, section.style = wanted, body, "numpy"
            continue
        title = section.header[0].strip()
        if title != section.name and ctx.fix("DF011", f"renamed {title!r} to {section.name!r}"):
            section.header = [section.name, *section.header[1:]]
        if section.header != wanted and ctx.fix(
            "DF012", f"normalised the {section.name} underline"
        ):
            section.header = wanted
        if _indented_body(section.body):
            # A NumPy title over a Google-style (indented) body.
            body = _convert_google(section, ctx.facts)
            if body is None:
                ctx.note("DF010", f"indented {section.name} body could not be converted", "report")
            elif ctx.fix("DF010", f"de-indented the Google-style {section.name} body"):
                section.body = body


def _indented_body(body: list[str]) -> bool:
    """Return whether every non-blank body line is indented.

    Parameters
    ----------
    body : list[str]
        Section body lines.

    Returns
    -------
    bool
        ``True`` for a non-empty body with no line at the base indentation.
    """
    lines = [line for line in body if line.strip()]
    return bool(lines) and all(line[0].isspace() for line in lines)


def merge_duplicates(doc: Docstring, ctx: RuleContext) -> None:
    """Merge repeated sections into their first occurrence (DF013).

    Parameters
    ----------
    doc : Docstring
        Docstring to update in place.
    ctx : RuleContext
        Rule context.
    """
    seen: dict[str, Section] = {}
    kept: list[Section] = [doc.preamble]
    for section in doc.sections[1:]:
        first = seen.get(section.name) if section.name else None
        if first is None:
            if section.name:
                seen[section.name] = section
            kept.append(section)
            continue
        if not ctx.fix("DF013", f"merged a duplicate {section.name} section"):
            kept.append(section)
            continue
        # Entry lists continue directly; prose paragraphs need a blank line.
        joiner = [] if section.name in ENTRY_SECTIONS else [""]
        first.body = [*trim_trailing(first.body), *joiner, *section.body]
    doc.sections = kept


def reorder_sections(doc: Docstring, ctx: RuleContext) -> None:
    """Sort known sections into numpydoc order (DF014).

    Unknown titled sections have no defined place, so their presence
    disables reordering.

    Parameters
    ----------
    doc : Docstring
        Docstring to update in place.
    ctx : RuleContext
        Rule context.
    """
    titled = doc.sections[1:]
    # An unknown title has no defined place in the NumPy order.
    if any(s.name is None for s in titled):
        return
    ordered = sorted(titled, key=lambda s: SECTION_ORDER.index(s.name or ""))
    if ordered != titled and ctx.fix("DF014", "reordered sections into NumPy order"):
        doc.sections = [doc.preamble, *ordered]
