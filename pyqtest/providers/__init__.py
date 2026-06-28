"""Embedding provider package — Protocol, registry, and built-in local provider."""

from pyqtest.providers.base import Provider
from pyqtest.providers.registry import REGISTRY, ProviderRegistry

__all__ = ["REGISTRY", "Provider", "ProviderRegistry"]
