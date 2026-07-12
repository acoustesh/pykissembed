"""Type definitions for similarity detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeGuard, cast

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Return whether *value* is a dictionary with only string keys."""
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast("dict[object, object]", value))


class PCAModel(Protocol):
    """Structural type for a fitted PCA model (sklearn or cuML)."""

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:  # noqa: N803
        """Transform data using the fitted PCA model."""
        ...


@dataclass
class FunctionInfo:
    """Information about an extracted function.

    Attributes
    ----------
        name: Function or class name
        file: Source file name (relative path)
        start_line: Starting line number in source
        end_line: Ending line number in source
        loc: Lines of code
        hash: AST-based hash (ignores comments, used for AST embeddings)
        text: Full raw source text
        text_for_embedding: Signature + docstring + comment lines (for text embeddings)
        text_hash: Hash of text_for_embedding (used for text embeddings)
        ast_text: AST unparsed code via ast.unparse() (for AST embeddings)
        embedding: Current embedding vector (set during similarity checks)
    """

    name: str
    file: str
    start_line: int
    end_line: int
    loc: int
    hash: str  # AST-based hash (ignores comments)
    text: str  # Full raw source
    text_for_embedding: str = ""  # Signature + docstring + comments
    text_hash: str = ""  # Hash of text_for_embedding
    ast_text: str = ""  # ast.unparse() output
    embedding: list[float] | None = field(default=None, repr=False)
