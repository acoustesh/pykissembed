"""Lossless model of a NumPy docstring's cleaned text.

A docstring is split into an untitled preamble (summary, deprecation note,
extended summary) followed by titled sections. Every section keeps its raw
header and body lines, so ``parse(text).render() == text`` holds for any
input: rendering only differs once a fixer replaces a section. Entry-level
views (``name : type`` plus an indented description) are computed on demand
and never cached, so an unparsable body simply stays raw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# numpydoc's canonical order. Attributes and Methods belong to class
# docstrings and sit right after Parameters.
SECTION_ORDER: tuple[str, ...] = (
    "Parameters",
    "Attributes",
    "Methods",
    "Returns",
    "Yields",
    "Receives",
    "Other Parameters",
    "Raises",
    "Warns",
    "Warnings",
    "See Also",
    "Notes",
    "References",
    "Examples",
)

# Sections whose body is a list of ``name : type`` (or bare ``type``) entries.
ENTRY_SECTIONS = frozenset({
    "Parameters",
    "Other Parameters",
    "Attributes",
    "Methods",
    "Returns",
    "Yields",
    "Receives",
    "Raises",
    "Warns",
})

# Sections whose entry header is a type, optionally prefixed by a name.
TYPE_FIRST_SECTIONS = frozenset({"Returns", "Yields", "Receives"})

_ALIASES: dict[str, str] = {
    "args": "Parameters",
    "arguments": "Parameters",
    "params": "Parameters",
    "parameter": "Parameters",
    "parameters": "Parameters",
    "keyword args": "Other Parameters",
    "keyword arguments": "Other Parameters",
    "other parameters": "Other Parameters",
    "attribute": "Attributes",
    "attributes": "Attributes",
    "methods": "Methods",
    "return": "Returns",
    "returns": "Returns",
    "yield": "Yields",
    "yields": "Yields",
    "receives": "Receives",
    "raise": "Raises",
    "raises": "Raises",
    "exceptions": "Raises",
    "warns": "Warns",
    "warning": "Warnings",
    "warnings": "Warnings",
    "see also": "See Also",
    "note": "Notes",
    "notes": "Notes",
    "reference": "References",
    "references": "References",
    "example": "Examples",
    "examples": "Examples",
}

# numpydoc underlines are dashes; three or more avoids matching "--" prose.
_UNDERLINE = re.compile(r"^-{3,}\s*$")
# A documented name: identifier, optionally starred or reST-escaped (\*args).
_ENTRY_NAME = re.compile(r"\\?\*{0,2}[A-Za-z_]\w*")
# Google titles are capitalised ("Args:"); a lowercase "methods:" is an entry.
_GOOGLE_HEADER = re.compile(r"^(?P<title>[A-Z][A-Za-z ]*?)\s*:\s*$")
# reST field-list item, e.g. ":param int x: text" or ":returns:".
_SPHINX_FIELD = re.compile(r"^:(?P<role>\w+)(?:\s+(?P<arg>[^:]+?))?:(?:\s+(?P<text>.*))?$")


def canonical_name(title: str) -> str | None:
    """Map a section title or alias to its canonical NumPy name.

    Parameters
    ----------
    title : str
        Raw title text; case, surrounding space and one trailing colon are ignored.

    Returns
    -------
    str | None
        The canonical section name, or ``None`` for an unknown title.
    """
    return _ALIASES.get(title.strip().removesuffix(":").strip().lower())


@dataclass(slots=True)
class Entry:
    """One ``header`` line plus its indented description lines."""

    header: str
    desc: list[str] = field(default_factory=list)

    def split(self, *, type_first: bool) -> tuple[str, str]:
        """Return the entry's ``(name, type)`` pair.

        Parameters
        ----------
        type_first : bool
            ``True`` in Returns-like sections, where a header without
            ``" : "`` is a bare type rather than a bare name.

        Returns
        -------
        tuple[str, str]
            Name and type text; either may be empty.
        """
        # numpydoc separates name and type with exactly " : ".
        name, sep, kind = self.header.partition(" : ")
        if sep:
            return name.strip(), kind.strip()
        if type_first:
            return "", self.header.strip()
        return self.header.strip(), ""

    def names(self) -> list[str]:
        """Return the comma-separated names this entry documents.

        Returns
        -------
        list[str]
            Names as written, stars included (``x, y : int`` gives two);
            empty when the header is not a list of identifiers.
        """
        name, _ = self.split(type_first=False)
        parts = [part.strip() for part in name.split(",")]
        if not all(_ENTRY_NAME.fullmatch(part) for part in parts):
            return []
        return parts


@dataclass(slots=True)
class Section:
    """A titled block (or the untitled preamble) with its raw lines."""

    header: list[str]
    body: list[str]
    name: str | None = None
    style: str = "preamble"


@dataclass(slots=True)
class Docstring:
    """A parsed docstring: its sections in order, each with its raw lines.

    Rendering joins the raw lines unchanged, so a docstring that no fixer
    touched renders back byte for byte; see :func:`parse`.

    Attributes
    ----------
    sections : list[Section]
        The untitled preamble first, then titled sections in source order.
    """

    sections: list[Section]

    @property
    def preamble(self) -> Section:
        """The untitled leading section (summary and extended summary).

        Returns
        -------
        Section
            The first section, which never has a header.
        """
        return self.sections[0]

    def find(self, name: str) -> Section | None:
        """Return the first section with canonical *name*.

        Parameters
        ----------
        name : str
            Canonical section name, e.g. ``"Parameters"``.

        Returns
        -------
        Section | None
            The section, or ``None`` when absent.
        """
        return next((s for s in self.sections[1:] if s.name == name), None)

    def summary(self) -> list[str]:
        """Return the first paragraph of the preamble.

        Returns
        -------
        list[str]
            Summary lines; empty when the docstring has no summary.
        """
        lines: list[str] = []
        for line in self.preamble.body:
            # The summary is the first paragraph: stop at the first blank line
            # after text (leading blanks are skipped).
            if not line.strip():
                if lines:
                    break
                continue
            lines.append(line)
        return lines

    def render(self) -> str:
        """Join every section's raw lines back into docstring text.

        Returns
        -------
        str
            The cleaned docstring text, identical to the parse input when no
            section was replaced.
        """
        return "\n".join(line for s in self.sections for line in (*s.header, *s.body))

    def render_normalized(self) -> str:
        """Render with exactly one blank line before each section header.

        Returns
        -------
        str
            Docstring text with trailing blank lines trimmed per section.
        """
        out: list[str] = []
        for index, section in enumerate(self.sections):
            body = trim_trailing(section.body)
            # Exactly one blank line before every section header.
            if index and out:
                out = [*trim_trailing(out), ""]
            out.extend((*section.header, *body))
        return "\n".join(trim_trailing(out))


def trim_trailing(lines: list[str]) -> list[str]:
    """Drop trailing blank lines.

    Parameters
    ----------
    lines : list[str]
        Lines to trim.

    Returns
    -------
    list[str]
        A copy without trailing whitespace-only lines.
    """
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1
    return lines[:end]


def _header_at(lines: list[str], index: int) -> tuple[Section, int] | None:
    """Recognise a section header starting at *lines[index]*.

    Three header shapes are accepted: a NumPy title over a dashed underline,
    a Google ``Title:`` line whose next line is indented, and a bare
    canonical entry-section title after a blank line.

    Parameters
    ----------
    lines : list[str]
        All docstring lines.
    index : int
        Candidate header position; the first line (the summary) never is one.

    Returns
    -------
    tuple[Section, int] | None
        The new empty section and the number of header lines, or ``None``.
    """
    line = lines[index]
    # Titles start at column 0; the first line is always the summary.
    if index == 0 or not line.strip() or line[0].isspace():
        return None
    nxt = lines[index + 1] if index + 1 < len(lines) else ""
    title = line.strip()
    if _UNDERLINE.match(nxt):
        name = canonical_name(title)
        return Section([line, nxt], [], name, "numpy" if name else "unknown"), 2
    google = _GOOGLE_HEADER.match(title)
    if google and nxt[:1].isspace() and nxt.strip():
        name = canonical_name(google["title"])
        if name is not None:
            return Section([line], [], name, "google"), 1
    # A bare title (no underline, no colon) is the riskiest form, so it must
    # follow a blank line, name an entry section, and precede an entry.
    previous_blank = not lines[index - 1].strip()
    if previous_blank and title in ENTRY_SECTIONS and nxt.strip() and not nxt[0].isspace():
        return Section([line], [], title, "bare"), 1
    return None


def parse(text: str) -> Docstring:
    """Split cleaned docstring text into a preamble and titled sections.

    Parameters
    ----------
    text : str
        Docstring text after ``inspect.cleandoc`` (no quotes, no common indent).

    Returns
    -------
    Docstring
        A model whose :meth:`Docstring.render` reproduces *text* exactly.
    """
    lines = text.split("\n")
    sections = [Section([], [])]
    index = 0
    while index < len(lines):
        found = _header_at(lines, index)
        if found is None:
            sections[-1].body.append(lines[index])
            index += 1
            continue
        section, width = found
        sections.append(section)
        index += width
    return Docstring(sections)


def parse_entries(body: list[str]) -> list[Entry] | None:
    """Split a section body into entries headed by column-0 lines.

    Parameters
    ----------
    body : list[str]
        Section body lines at the docstring's base indentation.

    Returns
    -------
    list[Entry] | None
        Entries whose rendering reproduces *body*, or ``None`` when indented
        text precedes the first entry header and the body is not a list.
    """
    entries: list[Entry] = []
    for line in body:
        if line.strip() and not line[0].isspace():
            entries.append(Entry(line))
        elif entries:
            entries[-1].desc.append(line)
        # Indented text before any header means this is not an entry list.
        elif line.strip():
            return None
    return entries


def render_entries(entries: list[Entry]) -> list[str]:
    """Flatten entries back into body lines.

    Parameters
    ----------
    entries : list[Entry]
        Entries in output order.

    Returns
    -------
    list[str]
        Each header followed by its description lines.
    """
    return [line for entry in entries for line in (entry.header, *entry.desc)]


def dedent_block(lines: list[str]) -> list[str]:
    """Remove the common leading indentation of the non-blank lines.

    Parameters
    ----------
    lines : list[str]
        Lines to dedent; blank lines become empty strings.

    Returns
    -------
    list[str]
        Dedented lines, relative indentation preserved.
    """
    widths = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    cut = min(widths, default=0)
    return [line[cut:] if line.strip() else "" for line in lines]


@dataclass(slots=True)
class SphinxField:
    """One ``:role arg: text`` field with its continuation lines."""

    role: str
    arg: str
    text: list[str]


def split_sphinx_fields(body: list[str]) -> tuple[list[str], list[SphinxField]] | None:
    """Split a trailing block of reST field lists off the preamble body.

    Parameters
    ----------
    body : list[str]
        Preamble body lines.

    Returns
    -------
    tuple[list[str], list[SphinxField]] | None
        The prose lines before the block and the parsed fields, or ``None``
        when there is no field block, or prose follows the first field (a
        mixed layout that is left for a human).
    """
    start = next((i for i, line in enumerate(body) if _SPHINX_FIELD.match(line)), None)
    if start is None:
        return None
    fields: list[SphinxField] = []
    for line in body[start:]:
        match = _SPHINX_FIELD.match(line)
        if match:
            text = [match["text"]] if match["text"] else []
            fields.append(SphinxField(match["role"].lower(), (match["arg"] or "").strip(), text))
        elif not line.strip() or line[0].isspace():
            fields[-1].text.append(line.strip())
        # Prose after the fields: a mixed layout left for a human.
        else:
            return None
    return body[:start], fields
