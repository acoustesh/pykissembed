"""Run every docfix rule over one docstring.

The rules live in :mod:`~pykissembed.tools.docfix.structure` (sections),
:mod:`~pykissembed.tools.docfix.entries` (entries) and
:mod:`~pykissembed.tools.docfix.reports` (findings that need prose or a
human); this module fixes their order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pykissembed.tools.docfix.common import (
    Outcome,
    RuleContext,
)
from pykissembed.tools.docfix.model import (
    parse,
)

if TYPE_CHECKING:
    from pykissembed.tools.docfix.facts import Facts
from pykissembed.tools.docfix.entries import fix_parameters, fix_returns
from pykissembed.tools.docfix.reports import (
    check_attributes,
    check_params,
    check_raises,
    check_returns,
    check_summary,
)
from pykissembed.tools.docfix.structure import (
    convert_sphinx,
    merge_duplicates,
    normalize_headers,
    reorder_sections,
)


def fix_docstring(
    text: str,
    facts: Facts | None,
    selected: frozenset[str],
    *,
    extra_documented: frozenset[str] = frozenset(),
    width: int = 88,
) -> Outcome:
    """Check one docstring and apply every selected deterministic fix.

    Parameters
    ----------
    text : str
        Cleaned docstring text.
    facts : Facts | None
        Facts of the documented definition, or ``None`` for a module.
    selected : frozenset[str]
        Codes to run.
    extra_documented : frozenset[str]
        Parameter names documented by the paired class or ``__init__``
        docstring, which count as documented here.
    width : int
        Longest line a fix may add inside the docstring (the line length
        minus the docstring's indentation); longer typed headers are skipped.

    Returns
    -------
    Outcome
        The (possibly unchanged) text, findings and dropped-word allowance.
    """
    doc = parse(text)
    ctx = RuleContext(selected, facts, width)
    # Structure first: entry fixes can only find a Parameters section once
    # Google/Sphinx blocks are converted and duplicate sections merged.
    convert_sphinx(doc, ctx)
    normalize_headers(doc, ctx)
    merge_duplicates(doc, ctx)
    fix_parameters(doc, ctx)
    fix_returns(doc, ctx)
    # Reordering last means every section added or renamed above is placed.
    reorder_sections(doc, ctx)
    # Checks run on the fixed model, so they never report what was just fixed.
    check_summary(doc, ctx)
    # A module has no signature, and an @overload stub is documented on the
    # implementation, so neither gets signature-based checks.
    if facts is not None and not facts.is_overload:
        check_params(doc, facts, extra_documented, ctx)
        if facts.kind == "function":
            check_returns(doc, facts, ctx)
            check_raises(doc, facts, ctx)
        else:
            check_attributes(doc, facts, ctx)
    # Unchanged docstrings keep their exact original text (round-trip safety).
    new_text = doc.render_normalized() if ctx.changed else text
    return Outcome(new_text, ctx.findings, ctx.allowance)
