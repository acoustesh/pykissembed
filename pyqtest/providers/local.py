"""Stub LocalProvider.

The real implementation lives in the ``pyqtest-local`` subpackage, which
provides the sentence-transformers backend. This stub exists so the
``[project.entry-points.pyqtest.providers]`` declaration in pyqtest's own
``pyproject.toml`` always resolves, even when the user has not installed
``pyqtest-local``.

Calling ``embed`` or ``is_configured`` on this stub raises a clear
``RuntimeError`` pointing the user at the right install command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


_INSTALL_HINT = "pip install pyqtest-local  # sentence-transformers backend"


class LocalProvider:
    """Stub provider — replaced by the real ``pyqtest-local`` subpackage."""

    name = "local"
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    schema_version = "1"
    max_tokens = 256
    batch_size = 32

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        msg = (
            f"The local provider requires the pyqtest-local subpackage.\n"
            f"  {_INSTALL_HINT}"
        )
        raise RuntimeError(msg)

    def is_configured(self) -> bool:
        """Return True only if the real ``pyqtest-local`` package is installed."""
        try:
            import pyqtest_local  # noqa: F401
        except ImportError:
            return False
        return True


__all__ = ["LocalProvider"]
