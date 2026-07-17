"""Embedding provider package — Protocol and entry-point registry."""

from pykissembed.providers.base import Provider
from pykissembed.providers.registry import REGISTRY, ProviderRegistry

__all__ = ["REGISTRY", "Provider", "ProviderRegistry"]
