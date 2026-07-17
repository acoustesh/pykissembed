"""Compatibility tombstone for the retired pykissembed local provider."""

from __future__ import annotations

from importlib.metadata import version as _version

from pykissembed_local.provider import LocalProvider

__version__ = _version("pykissembed-local")

__all__ = ["LocalProvider", "__version__"]
