"""Docstring tokens in source text: locating, re-rendering and guarding.

Docstrings are found by their exact ``STRING`` token rather than by line
numbers, so one-line docstrings, docstrings sharing a line with ``def``,
string prefixes and CRLF files are all handled. The guards here prove that
two versions of a file differ only inside docstrings.
"""

from __future__ import annotations

import ast
import io
import json
import re
import shutil
import subprocess
import tokenize
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pykissembed.checks.lint_typecheck import parse_tool_json
from pykissembed.tools.docfix.common import prose_words
from pykissembed.tools.docfix.model import trim_trailing

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Sequence
    from pathlib import Path

type Owner = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef

# Ruff rules about blank lines *around* a docstring edit code bytes, so they
# are never delegated; every other D fix stays inside the string token.
_RUFF_OUTSIDE_RULES = "D201,D202,D203,D204,D211"
# Ruff's exit status for a configuration or usage error.
_RUFF_ERROR_EXIT = 2


@dataclass(frozen=True, slots=True)
class Slot:
    """One docstring token, or the insertion point of a missing one."""

    owner: int
    start: int
    end: int
    token: str
    value: str | None
    indent: str | None


@dataclass(frozen=True, slots=True)
class Edit:
    """A replacement token for an existing docstring, or an inserted line."""

    owner: int
    start: int
    end: int
    text: str


def owners(tree: ast.Module) -> list[Owner]:
    """Return the module and every function and class, in walk order.

    Parameters
    ----------
    tree : ast.Module
        Parsed file.

    Returns
    -------
    list[Owner]
        Nodes that may carry a docstring; the list index is the owner id.
    """
    found: list[Owner] = [tree]
    found.extend(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    )
    return found


def _line_starts(source: str) -> list[int]:
    """Return the character offset at which each line begins.

    Parameters
    ----------
    source : str
        File text.

    Returns
    -------
    list[int]
        Offsets indexed by zero-based line number.
    """
    return [0, *(index + 1 for index, char in enumerate(source) if char == "\n")]


def _char_col(source: str, starts: list[int], row: int, byte_col: int) -> int:
    """Convert an AST UTF-8 byte column to a character column.

    Parameters
    ----------
    source : str
        File text.
    starts : list[int]
        Line start offsets.
    row : int
        One-based line number.
    byte_col : int
        Column in UTF-8 bytes, as reported by :mod:`ast`.

    Returns
    -------
    int
        Column in characters, as reported by :mod:`tokenize`.
    """
    # Slice one line only; slicing to the end of the file would make every
    # lookup O(file size) and whole-file processing quadratic.
    end = starts[row] if row < len(starts) else len(source)
    line = source[starts[row - 1] : end]
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="replace"))


def _string_tokens(source: str) -> dict[tuple[int, int], tokenize.TokenInfo]:
    """Map each ``STRING`` token's start position to the token.

    Parameters
    ----------
    source : str
        File text.

    Returns
    -------
    dict[tuple[int, int], tokenize.TokenInfo]
        Tokens keyed by ``(row, column)``.
    """
    # StringIO without newline translation keeps "\r\n" inside each line, so
    # token columns agree with offsets computed on the untranslated text.
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return {tok.start: tok for tok in tokens if tok.type == tokenize.STRING}


def locate(source: str, tree: ast.Module) -> list[Slot]:
    """Find every docstring token and every missing-docstring insertion point.

    Parameters
    ----------
    source : str
        File text.
    tree : ast.Module
        ``ast.parse(source)``.

    Returns
    -------
    list[Slot]
        One slot per owner that has a docstring (``value`` set) or could
        receive one (``value`` is ``None``, ``start == end``).
    """
    starts = _line_starts(source)
    tokens = _string_tokens(source)
    slots: list[Slot] = []
    for index, owner in enumerate(owners(tree)):
        if not owner.body:
            continue
        first = owner.body[0]
        doc = ast.get_docstring(owner, clean=False)
        # A decorated first statement starts at its first decorator line.
        decorators = getattr(first, "decorator_list", [])
        row = min([first.lineno, *(d.lineno for d in decorators)]) if doc is None else first.lineno
        col = _char_col(source, starts, row, first.col_offset)
        line_head = source[starts[row - 1] : starts[row - 1] + col]
        # Only a statement alone on its line has a usable indentation; a
        # docstring after "def f():" on the same line has none.
        indent = line_head if not line_head.strip() else None
        if doc is None:
            if not isinstance(owner, ast.Module) and indent is not None and row > owner.lineno:
                slots.append(Slot(index, starts[row - 1], starts[row - 1], "", None, indent))
            continue
        row = first.lineno
        token = tokens.get((row, col))
        end_col = _char_col(source, starts, first.end_lineno or row, first.end_col_offset or 0)
        # Implicit concatenation ("a" "b") spans several tokens; never touched.
        if token is None or token.end != (first.end_lineno, end_col):
            continue
        start = starts[row - 1] + col
        end = starts[token.end[0] - 1] + token.end[1]
        slots.append(Slot(index, start, end, source[start:end], doc, indent))
    return slots


def token_is_safe(slot: Slot) -> bool:
    """Return whether a docstring token's source text equals its value.

    Escape sequences make the source text differ from the value; such a
    docstring cannot be re-rendered from its value without changing it.

    Parameters
    ----------
    slot : Slot
        An existing docstring slot.

    Returns
    -------
    bool
        ``True`` when the token has no escapes and can be rewritten.
    """
    match = re.match(r"(?i)[rbu]*", slot.token)
    prefix = match.group(0) if match else ""
    quote = slot.token[len(prefix) : len(prefix) + 3]
    width = 3 if quote in {'"""', "'''"} else 1
    raw = slot.token[len(prefix) + width : -width]
    # The compiler normalises CRLF inside literals, so compare on LF.
    return raw.replace("\r\n", "\n") == slot.value and "b" not in prefix.lower()


def render_token(slot: Slot, text: str, newline: str) -> str | None:
    """Render cleaned docstring *text* as a string token for *slot*.

    The original prefix and quote style are kept; single quotes are
    upgraded to triple quotes. A multi-line docstring keeps the original
    choice of summary on the opening line or on the next line.

    Parameters
    ----------
    slot : Slot
        The docstring slot being replaced or filled.
    text : str
        Cleaned docstring text.
    newline : str
        The file's line terminator.

    Returns
    -------
    str | None
        The token text, or ``None`` when it cannot be written safely
        (delimiter clash, trailing backslash or quote, unknown indentation).
    """
    match = re.match(r"(?i)[ru]*", slot.token)
    prefix = match.group(0) if match and slot.token else ""
    quote = slot.token[len(prefix) : len(prefix) + 3] if slot.token else '"""'
    # A single-quoted docstring may need several lines after fixing.
    if quote not in {'"""', "'''"}:
        quote = '"""'
    if quote in text or text.endswith((quote[0], "\\")) or slot.indent is None:
        return None
    # Backslashes must stay literal: a raw prefix keeps the value exact.
    if "\\" in text and "r" not in prefix.lower():
        # "ur" is not valid Python 3 syntax, so such a docstring is left alone.
        if "u" in prefix.lower():
            return None
        prefix = "r" + prefix
    lines = text.split("\n")
    raw_lines = (slot.value or "").split("\n")
    next_line = len(raw_lines) > 1 and not raw_lines[0].strip()
    if len(lines) == 1 and not next_line:
        return f"{prefix}{quote}{text}{quote}"
    head, body = ("", lines) if next_line else (lines[0], lines[1:])
    rendered = [
        prefix + quote + head,
        *(slot.indent + line if line.strip() else "" for line in body),
    ]
    # Keep the file's convention for blank lines before the closing quotes
    # (ruff's D413 wants one); cleaning the docstring discarded them.
    blanks = max(len(raw_lines) - len(trim_trailing(raw_lines)) - 1, 0)
    return newline.join([*rendered, *[""] * blanks, slot.indent + quote])


def _is_docstring(statement: ast.stmt) -> bool:
    """Return whether *statement* is a string-literal expression statement.

    Parameters
    ----------
    statement : ast.stmt
        First statement of a scope.

    Returns
    -------
    bool
        ``True`` for a docstring statement.
    """
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def code_fingerprint(source: str) -> str:
    """Return the AST dump of *source* with every docstring removed.

    Parameters
    ----------
    source : str
        File text.

    Returns
    -------
    str
        A dump that differs between two sources iff executable code differs.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and _is_docstring(node.body[0])
        ):
            # A body cannot be empty; Pass keeps both sides comparable.
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def code_skeleton(source: str, inserted: frozenset[int] = frozenset()) -> str:
    """Return *source* with its docstring tokens cut out.

    Parameters
    ----------
    source : str
        File text.
    inserted : frozenset[int]
        Owner ids whose docstrings were added; their whole line is cut.

    Returns
    -------
    str
        The non-docstring bytes, concatenated.
    """
    tree = ast.parse(source)
    pieces: list[str] = []
    cursor = 0
    # locate() walks the AST breadth-first; cutting needs source order.
    for slot in sorted(locate(source, tree), key=lambda s: s.start):
        if slot.value is None:
            continue
        start, end = slot.start, slot.end
        # An added docstring brought its own line (indent and newline too).
        if slot.owner in inserted:
            start = source.rfind("\n", 0, start) + 1
            end = source.index("\n", end) + 1
        pieces.append(source[cursor:start])
        cursor = end
    pieces.append(source[cursor:])
    return "".join(pieces)


def comments(source: str) -> list[str]:
    """Return every comment token in order.

    Parameters
    ----------
    source : str
        File text.

    Returns
    -------
    list[str]
        Comment texts.
    """
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [tok.string for tok in tokens if tok.type == tokenize.COMMENT]


def guard(original: str, updated: str, inserted: frozenset[int]) -> str:
    """Check that *updated* differs from *original* only in docstrings.

    Parameters
    ----------
    original : str
        File text before editing.
    updated : str
        Candidate file text.
    inserted : frozenset[int]
        Owner ids that legitimately gained a docstring.

    Returns
    -------
    str
        ``""`` when every guard passes, otherwise the first failure.
    """
    try:
        if code_fingerprint(original) != code_fingerprint(updated):
            return "executable code changed"
        if code_skeleton(original) != code_skeleton(updated, inserted):
            return "bytes outside docstrings changed"
        if comments(original) != comments(updated):
            return "comments changed"
    except (SyntaxError, tokenize.TokenError, ValueError) as exc:
        return f"result does not parse: {exc}"
    before = {s.owner for s in locate(original, ast.parse(original)) if s.value is not None}
    after = {s.owner for s in locate(updated, ast.parse(updated)) if s.value is not None}
    if not before <= after or after - before - inserted:
        return "the set of documented definitions changed"
    return ""


def words_kept(old: str, new: str, allowance: Counter[str]) -> bool:
    """Return whether every prose word of *old* survives in *new*.

    Parameters
    ----------
    old : str
        Cleaned docstring before.
    new : str
        Cleaned docstring after.
    allowance : Counter[str]
        Words a fix declared it may drop.

    Returns
    -------
    bool
        ``True`` when no undeclared word was lost.
    """
    lost = prose_words(old) - prose_words(new) - allowance
    return not lost


@dataclass(slots=True)
class Ruff:
    """Runs ruff's docstring (D) rules on in-memory text through stdin.

    Ruff reads the consumer's own configuration (found from the file path),
    so layout fixes follow the project's chosen pydocstyle convention.

    Attributes
    ----------
    executable : str | None
        Path to the ruff binary, or ``None`` when ruff is not installed, in
        which case every method returns ``None``.
    preview : bool
        Whether ``--preview`` is needed, learned from the first run whose
        configuration requires it.
    """

    executable: str | None = field(default_factory=lambda: shutil.which("ruff"))
    preview: bool = False

    def _run(
        self, args: list[str], source: str, path: Path
    ) -> subprocess.CompletedProcess[str] | None:
        """Run ruff on *source* through stdin, retrying in preview mode once.

        Parameters
        ----------
        args : list[str]
            Arguments after ``ruff check``.
        source : str
            File text.
        path : Path
            Absolute path used for config discovery.

        Returns
        -------
        subprocess.CompletedProcess[str] | None
            The completed run, or ``None`` when ruff is missing or fails.
        """
        if self.executable is None:
            return None
        result = self._invoke(args, source, path)
        if (
            result.returncode == _RUFF_ERROR_EXIT
            and not self.preview
            and "preview" in result.stderr
        ):
            # Configs that select rules by name are only accepted with --preview.
            self.preview = True
            result = self._invoke(args, source, path)
        return None if result.returncode == _RUFF_ERROR_EXIT else result

    def _invoke(self, args: list[str], source: str, path: Path) -> subprocess.CompletedProcess[str]:
        """Run ``ruff check`` once on *source* through stdin.

        Parameters
        ----------
        args : list[str]
            Arguments after ``ruff check``.
        source : str
            File text.
        path : Path
            Absolute path passed as ``--stdin-filename`` for config discovery.

        Returns
        -------
        subprocess.CompletedProcess[str]
            The completed process, whatever its exit status.
        """
        preview = ["--preview"] if self.preview else []
        cmd = [str(self.executable), "check", *preview, *args, "--stdin-filename", str(path), "-"]
        return subprocess.run(cmd, input=source, capture_output=True, text=True, check=False)

    def fix(self, source: str, path: Path) -> str | None:
        """Apply ruff's safe D fixes that stay inside docstring tokens.

        Parameters
        ----------
        source : str
            File text.
        path : Path
            Absolute path used for config discovery.

        Returns
        -------
        str | None
            The fixed text, or ``None`` when ruff is unavailable.
        """
        args = ["--fix-only", "--exit-zero", "--select=D", f"--ignore={_RUFF_OUTSIDE_RULES}"]
        result = self._run(args, source, path)
        return result.stdout if result is not None and result.stdout else None

    def count(self, source: str, path: Path) -> int | None:
        """Count ruff D violations in *source*.

        Parameters
        ----------
        source : str
            File text.
        path : Path
            Absolute path used for config discovery.

        Returns
        -------
        int | None
            The count, or ``None`` when ruff is unavailable.
        """
        result = self._run(["--select=D", "--output-format=json", "--exit-zero"], source, path)
        if result is None:
            return None
        try:
            parsed: object = parse_tool_json(result.stdout)
        except json.JSONDecodeError:
            return None
        return len(parsed) if isinstance(parsed, list) else None


def apply_edits(source: str, edits: Sequence[Edit]) -> str:
    """Apply non-overlapping edits to *source* in one pass.

    Parameters
    ----------
    source : str
        Original file text.
    edits : Sequence[Edit]
        Edits with offsets into *source*.

    Returns
    -------
    str
        The edited text.
    """
    pieces: list[str] = []
    cursor = 0
    # Offsets all refer to the original text, so edits are stitched in one
    # left-to-right pass instead of re-slicing after every replacement.
    for edit in sorted(edits, key=lambda e: e.start):
        pieces.extend((source[cursor : edit.start], edit.text))
        cursor = edit.end
    pieces.append(source[cursor:])
    return "".join(pieces)


def inserted_owners(original: str, updated: str) -> frozenset[int]:
    """Return owner ids that have a docstring in *updated* but not *original*.

    Parameters
    ----------
    original : str
        File text before.
    updated : str
        File text after.

    Returns
    -------
    frozenset[int]
        Owner ids of added docstrings.
    """
    before = {s.owner for s in locate(original, ast.parse(original)) if s.value is not None}
    after = {s.owner for s in locate(updated, ast.parse(updated)) if s.value is not None}
    return frozenset(after - before)


def diff_edits(original: str, updated: str) -> dict[int, Edit] | None:
    """Express the docstring differences of *updated* as edits to *original*.

    Parameters
    ----------
    original : str
        File text before.
    updated : str
        File text after; its definitions must match *original* one to one.

    Returns
    -------
    dict[int, Edit] | None
        One edit per owner whose docstring token changed or was added, with
        offsets into *original*; ``None`` when a docstring disappeared or
        the definitions do not line up.
    """
    before = {s.owner: s for s in locate(original, ast.parse(original))}
    after = {s.owner: s for s in locate(updated, ast.parse(updated))}
    documented = {owner for owner, slot in before.items() if slot.value is not None}
    # A docstring that vanished cannot be expressed as an edit; refuse.
    if not documented <= {owner for owner, slot in after.items() if slot.value is not None}:
        return None
    edits: dict[int, Edit] = {}
    for owner, new in after.items():
        old = before.get(owner)
        if old is None:
            return None
        if new.value is None:
            if old.value is not None:
                return None
            continue
        if old.value is None:
            line_start = updated.rfind("\n", 0, new.start) + 1
            line_end = updated.index("\n", new.end) + 1
            edits[owner] = Edit(owner, old.start, old.start, updated[line_start:line_end])
        elif old.token != new.token:
            edits[owner] = Edit(owner, old.start, old.end, new.token)
    return edits
