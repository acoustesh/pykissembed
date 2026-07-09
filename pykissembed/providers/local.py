"""Stub LocalProvider.

The real implementation lives in the ``pykissembed-local`` subpackage, which
provides the sentence-transformers backend. This stub exists so the
``[project.entry-points.pykissembed.providers]`` declaration in pykissembed's own
``pyproject.toml`` always resolves, even when the user has not installed
``pykissembed-local``.

Calling ``embed`` or ``is_configured`` on this stub raises a clear
``RuntimeError`` pointing the user at the right install command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


_INSTALL_HINT = "pip install pykissembed-local  # sentence-transformers backend"


class LocalProvider:
    """Stub provider — replaced by the real ``pykissembed-local`` subpackage."""

    name = "local"
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    schema_version = "1"
    max_tokens = 256
    batch_size = 32

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # noqa: ARG002 — param name is part of the Provider Protocol contract
        msg = f"The local provider requires the pykissembed-local subpackage.\n  {_INSTALL_HINT}"
        raise RuntimeError(msg)

    def is_configured(self) -> bool:
        """Return True only if the real ``pykissembed-local`` package is installed.

        Returns
        -------
        bool
            ``True`` if ``pykissembed_local`` can be imported, else ``False``.
        """
        try:
            import pykissembed_local  # noqa: F401, PLC0415 — presence probe, must not hard-depend
        except ImportError:
            return False
        return True


__all__ = ["LocalProvider"]
