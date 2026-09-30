"""Entry-level fixes: header syntax, signature order, stars, types (DF020-DF033).

Every fix here is anchored in the definition's signature or annotations, so
it adds only facts the code proves.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pykissembed.tools.docfix.common import (
    INDENT,
    RuleContext,
    bare_name,
    desc_lines,
    looks_like_type,
    prose_words,
    returns_annotation,
)
from pykissembed.tools.docfix.model import (
    Docstring,
    Entry,
    Section,
    parse_entries,
    render_entries,
)

if TYPE_CHECKING:
    from pykissembed.tools.docfix.facts import Facts

# One or more comma-separated names, optionally starred or reST-escaped (\*args).
_NAMES = r"(?P<names>\\?\*{0,2}\w+(?:\s*,\s*\\?\*{0,2}\w+)*)"
# "x: int" or "x:int" (numpydoc wants " : ").
_COLON_TYPE = re.compile(_NAMES + r"\s*:\s*(?P<type>\S.*?)\s*$")
# "x (int)", "x (int, optional)" and Google's "x (int): text".
_PAREN_TYPE = re.compile(_NAMES + r"\s*\((?P<type>[^()]*)\)\s*(?::\s*(?P<desc>.*?))?\s*$")
# "x -- text", an old epydoc-like form.
_DASH_DESC = re.compile(_NAMES + r"\s+--\s+(?P<desc>.+?)\s*$")
# "x:" with nothing after the colon.
_BARE_COLON = re.compile(_NAMES + r"\s*:\s*$")
# Only these are compared for DF027: richer types have too many spellings
# (Optional[int], int | None, typing.List) to call a mismatch reliably.
_SIMPLE_BUILTINS = frozenset({
    "int",
    "str",
    "float",
    "bool",
    "bytes",
    "list",
    "dict",
    "tuple",
    "set",
})


def _canonical_entry(entry: Entry, known: frozenset[str]) -> Entry | None:
    """Return *entry* rewritten as ``name : type``, or ``None`` if unchanged.

    Parameters
    ----------
    entry : Entry
        A Parameters-like entry.
    known : frozenset[str]
        Bare names the definition really has. A Google-style ``name: text``
        header whose text is prose, not a type, is only rewritten when every
        name is known, so a sentence with a colon is never mistaken for one.

    Returns
    -------
    Entry | None
        The canonical entry, or ``None`` when it already conforms or its
        header cannot be read with confidence.
    """
    header = entry.header.rstrip()
    # Already canonical; the type text itself is never second-guessed.
    if " : " in header:
        return None
    for pattern in (_PAREN_TYPE, _DASH_DESC, _BARE_COLON, _COLON_TYPE):
        match = pattern.fullmatch(header)
        if match is None:
            continue
        groups = match.groupdict()
        names = re.sub(r"\s*,\s*", ", ", match["names"])
        kind = (groups.get("type") or "").strip()
        desc = groups.get("desc") or ""
        if kind and not looks_like_type(kind):
            # "name: some prose" is a Google entry only if name is real;
            # otherwise it is likely a sentence that happens to hold a colon.
            if not {bare_name(n) for n in names.split(", ")} <= known:
                return None
            kind, desc = "", kind
        new_header = f"{names} : {kind}" if kind else names
        return Entry(new_header, desc_lines(desc, entry.desc) if desc else entry.desc)
    return None


def _fix_entry_syntax(section: Section, known: frozenset[str], ctx: RuleContext) -> None:
    """Rewrite non-canonical entry headers in one section (DF020).

    Parameters
    ----------
    section : Section
        A Parameters, Other Parameters or Attributes section.
    known : frozenset[str]
        Bare parameter or attribute names of the definition.
    ctx : RuleContext
        Rule context.
    """
    entries = parse_entries(section.body)
    if entries is None:
        return
    changed = False
    for index, entry in enumerate(entries):
        fixed = _canonical_entry(entry, known)
        if fixed is None:
            # Prose stranded among the entries (often a misplaced paragraph).
            if not entry.names() and " : " not in entry.header:
                ctx.note("DF020", f"cannot read entry header {entry.header.strip()!r}", "report")
            continue
        if ctx.fix("DF020", f"rewrote {entry.header.strip()!r} as {fixed.header!r}"):
            entries[index] = fixed
            changed = True
    if changed:
        section.body = render_entries(entries)


def _fix_stars_and_implicit(entries: list[Entry], facts: Facts, ctx: RuleContext) -> list[Entry]:
    """Add missing stars to variadics and drop implicit self/cls (DF024, DF025).

    Parameters
    ----------
    entries : list[Entry]
        Parameters entries.
    facts : Facts
        Facts of the documented definition.
    ctx : RuleContext
        Rule context.

    Returns
    -------
    list[Entry]
        The updated entries.
    """
    starred = {p.name.lstrip("*"): p.name for p in facts.params if p.name.startswith("*")}
    kept: list[Entry] = []
    for entry in entries:
        names = entry.names()
        if (
            facts.implicit
            and names == [facts.implicit]
            and ctx.fix("DF025", f"removed the implicit {facts.implicit!r} entry")
        ):
            # The only fix allowed to delete words: declare them to the guard.
            ctx.allowance.update(prose_words("\n".join((entry.header, *entry.desc))))
            continue
        name, kind = entry.split(type_first=False)
        if (
            len(names) == 1
            and names[0] in starred
            and ctx.fix("DF024", f"documented {name!r} as {starred[name]!r}")
        ):
            kept.append(Entry(f"{starred[name]} : {kind}" if kind else starred[name], entry.desc))
        else:
            kept.append(entry)
    return kept


def _fix_order(entries: list[Entry], facts: Facts, ctx: RuleContext) -> list[Entry]:
    """Sort Parameters entries into signature order (DF021).

    Parameters
    ----------
    entries : list[Entry]
        Parameters entries.
    facts : Facts
        Facts of the documented definition.
    ctx : RuleContext
        Rule context.

    Returns
    -------
    list[Entry]
        Entries in signature order, or unchanged when any entry names an
        unknown parameter.
    """
    position = {bare_name(p.name): i for i, p in enumerate(facts.params)}
    keys = [position.get(bare_name(e.names()[0])) if e.names() else None for e in entries]
    # A stale or unreadable entry has no position; DF023 reports it instead.
    if any(key is None for key in keys):
        return entries
    ordered = [
        entry for _, entry in sorted(zip(keys, entries, strict=True), key=lambda kv: kv[0] or 0)
    ]
    if ordered != entries and ctx.fix("DF021", "reordered parameters to match the signature"):
        return ordered
    return entries


def _fix_types(entries: list[Entry], facts: Facts, ctx: RuleContext) -> list[Entry]:
    """Fill missing types from annotations and flag contradictions (DF026, DF027).

    Parameters
    ----------
    entries : list[Entry]
        Parameters entries.
    facts : Facts
        Facts of the documented definition.
    ctx : RuleContext
        Rule context.

    Returns
    -------
    list[Entry]
        The updated entries.
    """
    annotations = {bare_name(p.name): p.annotation for p in facts.params if p.annotation}
    out: list[Entry] = []
    for entry in entries:
        names = entry.names()
        name, kind = entry.split(type_first=False)
        annotation = annotations.get(bare_name(names[0])) if len(names) == 1 else None
        typed = f"{name} : {annotation}"
        if (
            annotation
            and not kind
            # A long annotation would overflow the line; types are optional anyway.
            and len(typed) <= ctx.width
            and ctx.fix("DF026", f"typed {name!r} as {annotation!r}")
        ):
            out.append(Entry(typed, entry.desc))
            continue
        if (
            annotation
            and kind in _SIMPLE_BUILTINS
            and annotation in _SIMPLE_BUILTINS
            and kind != annotation
        ):
            ctx.note(
                "DF027", f"{name!r} documented as {kind!r} but annotated {annotation!r}", "report"
            )
        out.append(entry)
    return out


def fix_parameters(doc: Docstring, ctx: RuleContext) -> None:
    """Apply the signature-aware Parameters fixes.

    Parameters
    ----------
    doc : Docstring
        Docstring to update in place.
    ctx : RuleContext
        Rule context.
    """
    facts = ctx.facts
    known: frozenset[str] = frozenset()
    if facts is not None:
        known = frozenset(bare_name(p.name) for p in [*facts.params, *facts.attributes])
    for name in ("Parameters", "Other Parameters", "Attributes"):
        section = doc.find(name)
        if section is not None:
            _fix_entry_syntax(section, known, ctx)
    section = doc.find("Parameters")
    # Signature fixes need the real parameter list (unknown for a class
    # without an explicit __init__, e.g. a dataclass).
    if section is None or facts is None or not facts.signature_known:
        return
    entries = parse_entries(section.body)
    if entries is None:
        return
    updated = _fix_stars_and_implicit(entries, facts, ctx)
    updated = _fix_order(updated, facts, ctx)
    updated = _fix_types(updated, facts, ctx)
    if updated != entries:
        section.body = render_entries(updated)


def fix_returns(doc: Docstring, ctx: RuleContext) -> None:
    """Type an untyped Returns entry and relabel generator Returns (DF032, DF033).

    Parameters
    ----------
    doc : Docstring
        Docstring to update in place.
    ctx : RuleContext
        Rule context.
    """
    facts = ctx.facts
    section = doc.find("Returns")
    if facts is None or section is None:
        return
    # A generator that also returns a value may document both, so leave it.
    if facts.is_generator and doc.find("Yields") is None and not facts.returns_value:
        if ctx.fix("DF033", "renamed Returns to Yields on a generator"):
            section.name, section.header = "Yields", ["Yields", "------"]
        return
    entries = parse_entries(section.body)
    annotation = returns_annotation(facts)
    if not entries or len(entries) != 1 or entries[0].desc or not annotation:
        return
    text = entries[0].header.strip()
    # A lone sentence where the type belongs: move it under the annotation.
    if (
        " " in text
        and not looks_like_type(text)
        and ctx.fix("DF032", f"typed Returns as {annotation!r}")
    ):
        section.body = [annotation, INDENT + text]
