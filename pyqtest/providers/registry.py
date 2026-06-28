"""Provider registry — discovers installed providers via ``importlib.metadata``.

Built-in providers are registered statically on import. Third-party
providers register via the ``pyqtest.providers`` entry-point group
declared in their own ``pyproject.toml``.
"""

from __future__ import annotations

from importlib import metadata

from pyqtest.providers.base import Provider

_BUILTIN_GROUP = "pyqtest.providers"


class ProviderRegistry:
    """In-memory registry of installed embedding providers."""

    __slots__ = ("_providers",)

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        """Register *provider*. Overwrites any existing provider with the same name."""
        self._providers[provider.name] = provider

    def discover(self) -> None:
        """Discover third-party providers via entry points and register them."""
        eps = metadata.entry_points(group=_BUILTIN_GROUP)
        for ep in eps:
            try:
                loaded = ep.load()
            except Exception:  # pragma: no cover — defensive
                continue
            instance = loaded() if isinstance(loaded, type) else loaded
            if isinstance(instance, Provider):
                self.register(instance)

    def all(self) -> list[Provider]:
        """Return all registered providers as a sorted list."""
        return [self._providers[n] for n in sorted(self._providers)]

    def get(self, name: str) -> Provider | None:
        """Return the provider named *name* or ``None``."""
        return self._providers.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._providers

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._providers))
        return f"<ProviderRegistry providers={names!r}>"


REGISTRY = ProviderRegistry()
"""Global registry. Built-in providers are imported lazily by callers
(see :func:`discover_builtin`)."""


def discover_builtin() -> None:
    """Register the providers bundled with pyqtest itself."""
    from pyqtest.providers.local import LocalProvider

    REGISTRY.register(LocalProvider())


def discover_all() -> ProviderRegistry:
    """Discover built-in + entry-point providers and return the registry."""
    discover_builtin()
    REGISTRY.discover()
    return REGISTRY


def get(name: str) -> Provider | None:
    """Convenience helper: get a provider by name, registering built-ins on demand."""
    if not REGISTRY._providers:
        discover_all()
    return REGISTRY.get(name)


def cache_key(provider: Provider, content_hash: str) -> str:
    """Compute the deterministic cache key for a (provider, content) pair.

    Format: ``provider.name|model_id|schema_version|content_hash``.

    Including ``schema_version`` is mandatory — it prevents silent cache
    corruption when a provider's vector shape changes between releases.
    """
    return f"{provider.name}|{provider.model_id}|{provider.schema_version}|{content_hash}"


__all__ = [
    "REGISTRY",
    "ProviderRegistry",
    "cache_key",
    "discover_all",
    "discover_builtin",
    "get",
]
