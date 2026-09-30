"""Checks that need prose or a human: missing and stale documentation.

Nothing here edits a docstring. Missing items become ``prose`` findings with
a draft slot id; likely-stale content becomes a ``report`` finding.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

from pykissembed.tools.docfix.common import (
    Finding,
    RuleContext,
    bare_name,
    yield_type,
)
from pykissembed.tools.docfix.model import (
    TYPE_FIRST_SECTIONS,
    Docstring,
    parse,
    parse_entries,
)

if TYPE_CHECKING:
    from pykissembed.tools.docfix.facts import Facts

# Below this many parameters, an absent Parameters section is not flagged.
_MIN_PARAMS_WITHOUT_SECTION = 2


def documented_names(doc: Docstring, section_name: str) -> list[str]:
    """Return the bare names documented in one entry section.

    Parameters
    ----------
    doc : Docstring
        Parsed docstring.
    section_name : str
        Canonical section name.

    Returns
    -------
    list[str]
        Names without stars; Returns-like sections yield their types.
    """
    section = doc.find(section_name)
    entries = parse_entries(section.body) if section is not None else None
    if not entries:
        return []
    if section_name in TYPE_FIRST_SECTIONS | {"Raises", "Warns"}:
        return [e.split(type_first=True)[1] for e in entries]
    return [bare_name(name) for e in entries for name in e.names()]


def check_params(doc: Docstring, facts: Facts, extra: frozenset[str], ctx: RuleContext) -> None:
    """Report undocumented and unknown parameters (DF022, DF023).

    Parameters
    ----------
    doc : Docstring
        Parsed docstring.
    facts : Facts
        Facts of the documented definition.
    extra : frozenset[str]
        Names documented by the paired class or ``__init__`` docstring.
    ctx : RuleContext
        Rule context.
    """
    # Without an explicit __init__ the constructor signature is unknown.
    if not facts.signature_known:
        return
    has_section = doc.find("Parameters") is not None
    documented = set(documented_names(doc, "Parameters")) | set(
        documented_names(doc, "Other Parameters")
    )
    missing = [p for p in facts.params if bare_name(p.name) not in documented | extra]
    # Without a Parameters section, a lone parameter is usually self-explanatory,
    # and a partner docstring (class or __init__) may document them instead.
    if (
        missing
        and (has_section or len(facts.params) >= _MIN_PARAMS_WITHOUT_SECTION)
        and (has_section or not extra)
    ):
        for param in missing:
            ctx.note(
                "DF022",
                f"parameter {param.name!r} is not documented",
                "prose",
                f"param:{param.name}",
            )
    # self/cls in the docstring is DF025's business, not a stale name.
    signature = {bare_name(p.name) for p in facts.params} | {facts.implicit}
    unknown = sorted(documented - signature)
    candidates = [bare_name(p.name) for p in missing]
    for name in unknown:
        # One stale name and one undocumented parameter is almost always a
        # rename; otherwise only a close spelling is suggested.
        close = candidates if len(unknown) == len(candidates) == 1 else []
        close = close or difflib.get_close_matches(name, candidates, n=1)
        hint = f" (renamed to {close[0]!r}?)" if close else ""
        ctx.note("DF023", f"documented parameter {name!r} is not in the signature{hint}", "report")


def check_returns(doc: Docstring, facts: Facts, ctx: RuleContext) -> None:
    """Report missing or superfluous Returns and Yields (DF030, DF031, DF034).

    Parameters
    ----------
    doc : Docstring
        Parsed docstring.
    facts : Facts
        Facts of the documented function.
    ctx : RuleContext
        Rule context.
    """
    has_returns = doc.find("Returns") is not None
    has_yields = doc.find("Yields") is not None
    if facts.documents_return and not facts.is_property and not has_returns and not has_yields:
        # Only draftable with an annotation: the type line cannot be invented.
        slot = "returns" if facts.returns else ""
        ctx.note(
            "DF030",
            f"returns {facts.returns or 'a value'} but has no Returns section",
            "prose",
            slot,
        )
    if has_returns and facts.returns in {"None", "NoReturn", "Never"} and facts.name != "__init__":
        ctx.note("DF031", f"annotated to return {facts.returns} but documents Returns", "report")
    if facts.is_generator and not has_yields and not has_returns:
        slot = "yields" if yield_type(facts) else ""
        ctx.note("DF034", "generator has no Yields section", "prose", slot)


def check_raises(doc: Docstring, facts: Facts, ctx: RuleContext) -> None:
    """Report undocumented and unraised exceptions (DF040, DF041).

    Parameters
    ----------
    doc : Docstring
        Parsed docstring.
    facts : Facts
        Facts of the documented function.
    ctx : RuleContext
        Rule context.
    """
    # Compare unqualified names: "errors.Foo" and "Foo" are the same class.
    documented = {name.rsplit(".", 1)[-1] for name in documented_names(doc, "Raises")}
    raised = {name.rsplit(".", 1)[-1] for name in facts.raises}
    for name in facts.raises:
        if name.rsplit(".", 1)[-1] not in documented:
            ctx.note("DF040", f"{name} can escape but is not in Raises", "prose", f"raises:{name}")
    for name in sorted(documented - raised):
        ctx.note("DF041", f"Raises lists {name}, which is not raised here (a callee may)", "report")


def check_attributes(doc: Docstring, facts: Facts, ctx: RuleContext) -> None:
    """Report public attributes missing from an existing Attributes section (DF062).

    Attributes is optional in NumPy style, so only an incomplete section
    is flagged, never an absent one.

    Parameters
    ----------
    doc : Docstring
        Parsed class docstring.
    facts : Facts
        Facts of the class.
    ctx : RuleContext
        Rule context.
    """
    if doc.find("Attributes") is None:
        return
    documented = set(documented_names(doc, "Attributes"))
    for attribute in facts.attributes:
        if attribute.name not in documented:
            message = f"attribute {attribute.name!r} is not documented"
            ctx.note("DF062", message, "prose", f"attr:{attribute.name}")


def check_summary(doc: Docstring, ctx: RuleContext) -> None:
    """Report a missing or multi-line summary (DF050, DF051).

    Parameters
    ----------
    doc : Docstring
        Parsed docstring.
    ctx : RuleContext
        Rule context.
    """
    summary = doc.summary()
    if not summary:
        ctx.note("DF050", "docstring has no summary", "prose", "summary")
    elif len(summary) > 1:
        ctx.note("DF051", "summary spans more than one line", "report")


def constructor_findings(class_doc: str | None, init_doc: str | None) -> list[Finding]:
    """Compare a class docstring with its ``__init__`` docstring (DF060, DF061).

    Parameters
    ----------
    class_doc : str | None
        Cleaned class docstring, or ``None`` when absent.
    init_doc : str | None
        Cleaned ``__init__`` docstring, or ``None`` when absent.

    Returns
    -------
    list[Finding]
        Report findings for the class; both codes are advisory.
    """
    if init_doc is None:
        return []
    in_init = parse(init_doc).find("Parameters") is not None
    in_class = class_doc is not None and parse(class_doc).find("Parameters") is not None
    if in_init and in_class:
        return [Finding("DF060", "Parameters documented in both the class and __init__", "report")]
    if in_init:
        return [Finding("DF061", "constructor Parameters belong in the class docstring", "report")]
    return []
