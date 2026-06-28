"""Embedding provider package — Protocol, registry, and built-in local provider."""

from pykissembed.providers.base import Provider
from pykissembed.providers.registry import REGISTRY, ProviderRegistry

__all__ = ["REGISTRY", "Provider", "ProviderRegistry"]
