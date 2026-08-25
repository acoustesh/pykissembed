"""pykissembed check modules — registered automatically by the pytest plugin."""

from __future__ import annotations

from . import (
    code_complexity,
    code_similarity,
    comment_density,
    docstring_format,
    lint_typecheck,
    no_suppressions,
)

__all__ = [
    "code_complexity",
    "code_similarity",
    "comment_density",
    "docstring_format",
    "lint_typecheck",
    "no_suppressions",
]
