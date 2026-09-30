"""Rewrite the docstrings of one file and prove nothing else changed.

A file is processed as one snapshot: every docstring is located by its exact
``STRING`` token, all edits are computed against that snapshot, and they are
applied in a single pass. The result is then checked by independent guards
before anything is written:

1. the AST with docstrings stripped is identical;
2. the bytes outside docstring tokens are identical (inserted docstring
   lines excepted);
3. the comment token stream is identical;
4. no prose word of an existing docstring is lost;
5. a second run over the result changes nothing;
6. ruff's docstring-rule count does not rise.

A file that fails any guard is reported and left untouched.
"""

from __future__ import annotations

import ast
import inspect
import tokenize
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pykissembed.tools.docfix.common import Finding, yield_type
from pykissembed.tools.docfix.facts import Facts, collect
from pykissembed.tools.docfix.fill import apply_drafts
from pykissembed.tools.docfix.model import parse
from pykissembed.tools.docfix.reports import constructor_findings, documented_names
from pykissembed.tools.docfix.rules import fix_docstring
from pykissembed.tools.docfix.source import (
    Edit,
    Owner,
    Ruff,
    Slot,
    apply_edits,
    code_skeleton,
    diff_edits,
    locate,
    owners,
    render_token,
    token_is_safe,
    words_kept,
)

if TYPE_CHECKING:
    from pykissembed.tools.docfix.facts import DefNode

# Each grading round only reverts, so it converges; this caps API spend if a
# grader flip-flops on expected values near a threshold.
_MAX_REVERT_ROUNDS = 5


class Drafter(Protocol):
    """Structural type for the prose drafter used by ``--draft``.

    Implemented by :class:`~pykissembed.tools.docfix.draft.OpenRouterDrafter`;
    tests substitute a fake with the same ``draft`` method.
    """

    def draft(self, source: str, slots: list[str]) -> dict[str, str] | None:
        """Return drafted text for every requested slot.

        Parameters
        ----------
        source : str
            Source segment of the definition being documented.
        slots : list[str]
            Slot ids to fill, e.g. ``"summary"``, ``"param:x"``, ``"returns"``.

        Returns
        -------
        dict[str, str] | None
            One text per slot id, or ``None`` when the drafts are unavailable
            or invalid, in which case the docstring is left unchanged.
        """
        ...


class Grader(Protocol):
    """Structural type for the ``--grade`` never-regress check.

    Implemented by :class:`~pykissembed.tools.docfix.draft.JevGrader`; tests
    substitute a fake with the same ``verdict`` method.
    """

    def verdict(self, old: DefNode, original: str, new: DefNode, updated: str, key: str) -> str:
        """Decide whether one rewritten definition may keep its new text.

        Parameters
        ----------
        old : DefNode
            The definition parsed from *original*.
        original : str
            File text before the edits.
        new : DefNode
            The same definition parsed from *updated*.
        updated : str
            File text after the edits.
        key : str
            Project-relative file key.

        Returns
        -------
        str
            ``""`` to keep the rewrite; otherwise the reason it is reverted,
            which is shown in the report as a DF003 finding.
        """
        ...


@dataclass(slots=True)
class Options:
    """Everything that controls one docfix run.

    Attributes
    ----------
    selected : frozenset[str]
        Rule codes to run; see :data:`~pykissembed.tools.docfix.common.CODES`.
    write : bool
        Write verified results to disk instead of only reporting them.
    allow_dirty : bool
        Also write files that have uncommitted git changes.
    symbols : frozenset[tuple[str, str]] | None
        ``(file_key, name)`` pairs to restrict the run to, or ``None``.
    targets : frozenset[tuple[str, int]] | None
        ``(file_key, line)`` pairs from ``--only-failing``, or ``None``.
    line_length : int
        Maximum line length for text the tool adds.
    drafter : Drafter | None
        Prose drafter for ``--draft``; ``None`` leaves prose on the worklist.
    grader : Grader | None
        Never-regress judge for ``--grade``; ``None`` keeps every fix.
    root : Path
        Project root used to build file keys.
    """

    selected: frozenset[str]
    write: bool = False
    allow_dirty: bool = False
    symbols: frozenset[tuple[str, str]] | None = None
    targets: frozenset[tuple[str, int]] | None = None
    line_length: int = 88
    drafter: Drafter | None = None
    grader: Grader | None = None
    root: Path = field(default_factory=Path.cwd)


@dataclass(frozen=True, slots=True)
class Located:
    """A finding anchored to a file line and symbol."""

    line: int
    symbol: str
    finding: Finding


@dataclass(slots=True)
class FileResult:
    """The outcome of processing one file."""

    key: str
    original: str
    updated: str
    findings: list[Located] = field(default_factory=list)
    error: str = ""
    written: bool = False


@dataclass(slots=True)
class _FileState:
    """Working state while transforming one file."""

    key: str
    source: str
    tree: ast.Module
    owners: list[Owner]
    parents: dict[int, int]
    newline: str
    options: Options
    findings: list[Located] = field(default_factory=list)
    edits: dict[int, Edit] = field(default_factory=dict)
    texts: dict[int, str] = field(default_factory=dict)

    def where(self, owner: int) -> tuple[int, str]:
        """Return the line and symbol name used in reports for *owner*.

        Parameters
        ----------
        owner : int
            Owner id.

        Returns
        -------
        tuple[int, str]
            Definition line and name; ``(1, "<module>")`` for the module.
        """
        node = self.owners[owner]
        if isinstance(node, ast.Module):
            return 1, "<module>"
        return node.lineno, node.name

    def add(self, owner: int, finding: Finding) -> None:
        """Record *finding* against *owner*.

        Parameters
        ----------
        owner : int
            Owner id.
        finding : Finding
            The finding.
        """
        line, symbol = self.where(owner)
        self.findings.append(Located(line, symbol, finding))


def _parents(nodes: list[Owner]) -> dict[int, int]:
    """Map each owner id to the id of its nearest enclosing owner.

    Parameters
    ----------
    nodes : list[Owner]
        Owners in walk order; ``nodes[0]`` is the module.

    Returns
    -------
    dict[int, int]
        Parent owner ids; the module has none.
    """
    index = {id(node): i for i, node in enumerate(nodes)}
    parents: dict[int, int] = {}
    # Explicit stack instead of recursion: deeply nested generated code must
    # not hit the interpreter's recursion limit.
    stack: list[tuple[ast.AST, int]] = [(nodes[0], 0)]
    while stack:
        node, current = stack.pop()
        for child in ast.iter_child_nodes(node):
            owner = index.get(id(child))
            if owner is not None:
                parents[owner] = current
            # Non-owner nodes (if, with, ...) pass their enclosing owner down.
            stack.append((child, current if owner is None else owner))
    return parents


def _selected_owner(state: _FileState, owner: int) -> bool:
    """Return whether *owner* is in scope for the run's symbol filters.

    Parameters
    ----------
    state : _FileState
        File state.
    owner : int
        Owner id.

    Returns
    -------
    bool
        ``True`` with no filters, or when the owner matches one.
    """
    options = state.options
    if options.symbols is None and options.targets is None:
        return True
    # Symbols match by name (stable across edits); targets by the line the
    # gate reported, which is only valid against the unedited file.
    line, symbol = state.where(owner)
    if options.symbols is not None and (state.key, symbol) in options.symbols:
        return True
    return options.targets is not None and (state.key, line) in options.targets


def _facts(state: _FileState, owner: int) -> Facts | None:
    """Collect facts for *owner*.

    Parameters
    ----------
    state : _FileState
        File state.
    owner : int
        Owner id.

    Returns
    -------
    Facts | None
        Facts, or ``None`` for the module.
    """
    node = state.owners[owner]
    if isinstance(node, ast.Module):
        return None
    parent = state.owners[state.parents[owner]] if owner in state.parents else None
    return collect(node, in_class=isinstance(parent, ast.ClassDef))


def _partner_doc(state: _FileState, owner: int, slots: dict[int, Slot]) -> str | None:
    """Return the cleaned docstring of a class's ``__init__``, or vice versa.

    Parameters
    ----------
    state : _FileState
        File state.
    owner : int
        Owner id of a class or of an ``__init__`` defined directly in one.
    slots : dict[int, Slot]
        Docstring slots by owner id.

    Returns
    -------
    str | None
        The partner's cleaned docstring, or ``None`` when *owner* has no
        partner or the partner has no docstring.
    """
    node = state.owners[owner]
    parent = state.parents.get(owner)
    partner: int | None = None
    if isinstance(node, ast.ClassDef):
        partner = next(
            (
                i
                for i, p in state.parents.items()
                if p == owner and getattr(state.owners[i], "name", "") == "__init__"
            ),
            None,
        )
    elif getattr(node, "name", "") == "__init__" and parent is not None:
        # Only a method's partner is its class; a nested function named
        # __init__ inside another function has none.
        partner = parent if isinstance(state.owners[parent], ast.ClassDef) else None
    slot = slots.get(partner) if partner is not None else None
    return inspect.cleandoc(slot.value) if slot is not None and slot.value is not None else None


def _missing_docstring(state: _FileState, slot: Slot) -> list[str]:
    """Report a definition without a docstring (DF052) and list its slots.

    Parameters
    ----------
    state : _FileState
        File state.
    slot : Slot
        Insertion point of the missing docstring.

    Returns
    -------
    list[str]
        Draft slots for a new docstring; empty unless DF052 is selected.
    """
    facts = _facts(state, slot.owner)
    if facts is None or facts.is_overload or "DF052" not in state.options.selected:
        return []
    state.add(slot.owner, Finding("DF052", "definition has no docstring", "prose", "summary"))
    return _missing_slots(facts)


def _record_edit(state: _FileState, slot: Slot, old: str, new: str, allowance: Counter[str]) -> str:
    """Validate and stage a docstring rewrite.

    Parameters
    ----------
    state : _FileState
        File state.
    slot : Slot
        The docstring slot.
    old : str
        Cleaned text before.
    new : str
        Cleaned text after.
    allowance : Counter[str]
        Words the fixes may drop.

    Returns
    -------
    str
        ``""`` when staged, otherwise why it was refused.
    """
    # Escapes mean the value differs from the source text; rewriting from the
    # value would silently change what the docstring says.
    if slot.value is not None and not token_is_safe(slot):
        return "docstring contains escape sequences"
    if not words_kept(old, new, allowance):
        return "rewrite would drop prose words"
    token = render_token(slot, new, state.newline)
    if token is None:
        return "docstring cannot be re-quoted safely"
    # A new docstring is a whole inserted line; an existing one swaps its token.
    text = token if slot.value is not None else f"{slot.indent}{token}{state.newline}"
    # Re-rendering can reproduce the token byte for byte; that is no edit.
    if text != slot.token:
        state.edits[slot.owner] = Edit(slot.owner, slot.start, slot.end, text)
        state.texts[slot.owner] = new
    return ""


def _missing_slots(facts: Facts) -> list[str]:
    """Return the draft slots for a definition with no docstring.

    Parameters
    ----------
    facts : Facts
        Facts of the definition.

    Returns
    -------
    list[str]
        ``summary`` plus parameter, return, yield and raise slots.
    """
    # Class constructor parameters are drafted for __init__, not twice.
    slots = ["summary", *(f"param:{p.name}" for p in facts.params if facts.kind == "function")]
    if (
        facts.kind == "function"
        and facts.documents_return
        and facts.returns
        and not facts.is_property
    ):
        slots.append("returns")
    if facts.is_generator and yield_type(facts):
        slots.append("yields")
    slots.extend(f"raises:{name}" for name in facts.raises)
    return slots


def _process_slot(state: _FileState, slot: Slot, slots: dict[int, Slot]) -> list[str]:
    """Check and fix one docstring; return its draftable slot ids.

    Parameters
    ----------
    state : _FileState
        File state.
    slot : Slot
        The docstring slot (or missing-docstring insertion point).
    slots : dict[int, Slot]
        All slots by owner id.

    Returns
    -------
    list[str]
        Prose slots the drafter may fill.
    """
    if slot.value is None:
        return _missing_docstring(state, slot)
    # Rules see the cleaned text that tools like Sphinx and ast.get_docstring
    # see; the original indentation is restored when the token is rendered.
    old = inspect.cleandoc(slot.value)
    partner = _partner_doc(state, slot.owner, slots)
    extra = frozenset(documented_names(parse(partner), "Parameters")) if partner else frozenset()
    outcome = fix_docstring(
        old,
        _facts(state, slot.owner),
        state.options.selected,
        extra_documented=extra,
        width=state.options.line_length - len(slot.indent or ""),
    )
    findings = list(outcome.findings)
    if isinstance(state.owners[slot.owner], ast.ClassDef):
        selected = state.options.selected
        findings += [f for f in constructor_findings(old, partner) if f.code in selected]
    state.texts[slot.owner] = old
    if outcome.text != old:
        reason = _record_edit(state, slot, old, outcome.text, outcome.allowance)
        if reason:
            # A refused rewrite leaves the text as it was, so its fixes did not happen.
            findings = [f for f in findings if f.action != "fix"]
            findings.append(Finding("DF001", reason, "skip"))
    for finding in findings:
        state.add(slot.owner, finding)
    return [f.slot for f in findings if f.action == "prose" and f.slot]


def _draft(state: _FileState, slot: Slot, wanted: list[str]) -> None:
    """Fill prose slots through the drafter and stage the result.

    Parameters
    ----------
    state : _FileState
        File state.
    slot : Slot
        The docstring slot.
    wanted : list[str]
        Slot ids to fill.
    """
    drafter = state.options.drafter
    node = state.owners[slot.owner]
    if drafter is None or not wanted or isinstance(node, ast.Module):
        return
    # The model sees the whole definition, so prose is grounded in the body.
    segment = ast.get_source_segment(state.source, node) or ""
    drafts = drafter.draft(segment, wanted)
    if drafts is None:
        state.add(slot.owner, Finding("DF001", "draft rejected or unavailable", "skip"))
        return
    facts = _facts(state, slot.owner)
    # Drafts build on the deterministic result, never on the raw original.
    base = state.texts.get(slot.owner, "")
    room = state.options.line_length - len(slot.indent or "")
    # Entry descriptions sit four spaces deeper; keep a sane minimum width.
    drafted = apply_drafts(base, facts, drafts, width=max(40, room - 4))
    normalized = fix_docstring(drafted, facts, state.options.selected, width=room).text
    reason = _record_edit(state, slot, base, normalized, Counter())
    line, symbol = state.where(slot.owner)
    drafted_slots = ", ".join(sorted(drafts))
    finding = (
        Finding("DF001", reason, "skip")
        if reason
        else Finding("DF004", f"drafted {drafted_slots}", "fix")
    )
    state.findings.append(Located(line, symbol, finding))


def _new_state(key: str, source: str, options: Options) -> _FileState:
    """Parse *source* into a fresh file state.

    Parameters
    ----------
    key : str
        Project-relative file key.
    source : str
        File text.
    options : Options
        Run options.

    Returns
    -------
    _FileState
        State with owners and parent links resolved.
    """
    tree = ast.parse(source)
    nodes = owners(tree)
    # Rendered multi-line docstrings must match the file's line endings.
    newline = "\r\n" if "\r\n" in source else "\n"
    return _FileState(key, source, tree, nodes, _parents(nodes), newline, options)


def _ruff_pass(text: str, path: Path, options: Options, ruff: Ruff) -> str:
    """Apply ruff's docstring fixes, keeping them only if code bytes are intact.

    Parameters
    ----------
    text : str
        File text.
    path : Path
        Absolute file path, for ruff config discovery.
    options : Options
        Run options; nothing happens unless DF002 is selected.
    ruff : Ruff
        Ruff runner.

    Returns
    -------
    str
        The fixed text, or *text* when ruff is unavailable, changes nothing,
        or touches anything outside docstring tokens.
    """
    if "DF002" not in options.selected:
        return text
    fixed = ruff.fix(text, path)
    if fixed is None or fixed == text:
        return text
    # Ruff is trusted with docstrings only: any other byte change, or a
    # docstring appearing or vanishing, discards its whole output.
    try:
        same_code = code_skeleton(fixed) == code_skeleton(text)
        same_docs = diff_edits(text, fixed) is not None
    except SyntaxError, tokenize.TokenError, ValueError:
        return text
    return fixed if same_code and same_docs else text


def transform(
    key: str, original: str, path: Path, options: Options, ruff: Ruff, *, check: bool = True
) -> FileResult:
    """Compute the rewritten text of one file and its findings.

    The pipeline is ruff, then the docfix rules and drafts, then ruff again,
    so layout fixes that depend on renamed sections settle in one run. The
    result is re-expressed as per-docstring edits against *original*, which
    lets symbol filters and the grader keep or drop each docstring alone.

    Parameters
    ----------
    key : str
        Project-relative file key.
    original : str
        File text.
    path : Path
        Absolute path, for ruff config discovery.
    options : Options
        Run options.
    ruff : Ruff
        Ruff runner.
    check : bool
        Verify that every edited docstring is a fixpoint of a second run
        (before grading, which may legitimately revert edits).

    Returns
    -------
    FileResult
        Findings and the result text (equal to *original* when nothing
        changed or the idempotency check failed).
    """
    base = _ruff_pass(original, path, options, ruff)
    state = _new_state(key, base, options)
    slots = {slot.owner: slot for slot in locate(base, state.tree)}
    for owner in diff_edits(original, base) or {}:
        if _selected_owner(state, owner):
            state.add(owner, Finding("DF002", "applied ruff's docstring layout fixes", "fix"))
    wanted: dict[int, list[str]] = {}
    for owner, slot in slots.items():
        if _selected_owner(state, owner):
            wanted[owner] = _process_slot(state, slot, slots)
    for owner, slot_ids in wanted.items():
        _draft(state, slots[owner], slot_ids)
    # A second ruff pass settles layout rules that react to our renames
    # (e.g. D413 once "Reference" becomes the last known section).
    staged = _ruff_pass(apply_edits(base, list(state.edits.values())), path, options, ruff)
    edits = {
        owner: edit
        for owner, edit in (diff_edits(original, staged) or {}).items()
        if _selected_owner(state, owner)
    }
    updated = apply_edits(original, list(edits.values()))
    # Idempotency is checked before grading: grading may revert a docstring
    # to its original, which a plain re-run would then fix again.
    if check and edits:
        unstable = _unstable_owners(key, updated, path, options, ruff) & edits.keys()
        if unstable:
            error = "fixes are not idempotent (a second run would change the result)"
            return FileResult(key, original, original, state.findings, error=error)
    if options.grader is not None and updated != original:
        updated = _grade(state, original, edits, options.grader)
    return FileResult(key, original, updated, state.findings)


def _unstable_owners(key: str, text: str, path: Path, options: Options, ruff: Ruff) -> set[int]:
    """Return owners whose docstring a second deterministic run would change.

    Parameters
    ----------
    key : str
        Project-relative file key.
    text : str
        File text after the first run's edits (before grading).
    path : Path
        Absolute file path, for ruff config discovery.
    options : Options
        Run options; drafting, grading and symbol filters are dropped.
    ruff : Ruff
        Ruff runner.

    Returns
    -------
    set[int]
        Owner ids that are not yet a fixpoint.
    """
    # Filters are dropped too: target lines no longer match the edited text,
    # and the caller intersects the result with the owners it edited.
    rerun = replace(options, drafter=None, grader=None, symbols=None, targets=None)
    again = transform(key, text, path, rerun, ruff, check=False)
    return set(diff_edits(text, again.updated) or {})


def _descendants(state: _FileState, owner: int) -> set[int]:
    """Return *owner* and every owner nested inside it.

    Parameters
    ----------
    state : _FileState
        File state.
    owner : int
        Owner id.

    Returns
    -------
    set[int]
        Owner ids.
    """
    found = {owner}
    changed = True
    while changed:
        extra = {child for child, parent in state.parents.items() if parent in found} - found
        found |= extra
        changed = bool(extra)
    return found


def _grade(state: _FileState, original: str, edits: dict[int, Edit], grader: Grader) -> str:
    """Revert every edit the grader refuses.

    Every definition whose source segment changed is judged. A refusal
    reverts that definition and everything nested in it (restoring its
    segment exactly); the loop repeats until nothing is refused.

    Parameters
    ----------
    state : _FileState
        File state.
    original : str
        Original file text.
    edits : dict[int, Edit]
        Edits by owner id; entries are removed on revert.
    grader : Grader
        The rubric grader.

    Returns
    -------
    str
        The file text with only non-regressing edits applied.
    """
    before_nodes = owners(ast.parse(original))
    for _ in range(_MAX_REVERT_ROUNDS):
        updated = apply_edits(original, list(edits.values()))
        after_nodes = owners(ast.parse(updated))
        dropped: set[int] = set()
        for index, (old, new) in enumerate(zip(before_nodes, after_nodes, strict=True)):
            if isinstance(old, ast.Module) or isinstance(new, ast.Module):
                continue
            # The gate grades whole segments, so an enclosing class is judged
            # again when only one of its methods' docstrings changed.
            if ast.get_source_segment(original, old) == ast.get_source_segment(updated, new):
                continue
            reason = grader.verdict(old, original, new, updated, state.key)
            if reason:
                # Reverting the nested docstrings too restores the exact
                # original segment, so the next round cannot refuse it again.
                dropped |= _descendants(state, index)
                state.add(index, Finding("DF003", f"reverted: {reason}", "skip"))
        if not dropped & edits.keys():
            return updated
        for owner in dropped:
            _ = edits.pop(owner, None)
    # No stable subset within the round cap: keep the file untouched.
    return original
