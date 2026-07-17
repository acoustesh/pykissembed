"""Provider registry — discovers installed providers via ``importlib.metadata``.

Provider packages register through the ``pykissembed.providers`` entry-point
group declared in their own ``pyproject.toml``. Core intentionally bundles no
embedding provider so installing pykissembed alone remains network-neutral.
"""

from __future__ import annotations

import warnings
from importlib import metadata

from pykissembed.providers.base import Provider

_BUILTIN_GROUP = "pykissembed.providers"


class ProviderRegistry:
    """In-memory registry of installed embedding providers."""

    __slots__ = ("_providers",)

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        """Register *provider*. Overwrites any existing provider with the same name."""
        self._providers[provider.name] = provider

    def clear(self) -> None:
        """Remove all registered providers."""
        self._providers.clear()

    def discover(self) -> None:
        """Discover third-party providers via entry points and register them.

        Entry points are iterated in **reverse** order so that the
        first-listed entry point (typically a user-installed override)
        wins over later ones. Built-in providers
        (registered via :func:`discover_builtin`) are still loaded first
        and are then overridden by whichever entry point appears first
        in the metadata listing.
        """
        eps = list(metadata.entry_points(group=_BUILTIN_GROUP))
        for ep in reversed(eps):
            try:
                loaded = ep.load()
                # An entry point may resolve to either a Provider *class*
                # (needs instantiating) or an already-built singleton
                # *instance*. Constructor failures receive the same isolation
                # as import failures so one broken extension cannot abort all
                # provider discovery.
                instance = loaded() if isinstance(loaded, type) else loaded
            except Exception as exc:  # ruff:ignore[blind-except]
                # A broken third-party entry point (bad install, incompatible
                # version) must not crash discovery for every other provider.
                warnings.warn(  # pragma: no cover — defensive
                    f"pykissembed: failed to load provider entry point {ep.name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if isinstance(instance, Provider):
                self.register(instance)

    def all(self) -> list[Provider]:
        """Return all registered providers as a sorted list.

        Returns
        -------
        list[Provider]
            Registered providers ordered by name.
        """
        return [self._providers[n] for n in sorted(self._providers)]

    def get(self, name: str) -> Provider | None:
        """Return the provider named *name* or ``None``.

        Returns
        -------
        Provider | None
            The registered provider for *name*, or ``None`` if no
            provider is registered under that name.
        """
        return self._providers.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._providers

    def __len__(self) -> int:
        return len(self._providers)

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._providers))
        return f"<ProviderRegistry providers={names!r}>"


REGISTRY = ProviderRegistry()
"""Global registry. Built-in providers are imported lazily by callers
(see :func:`discover_builtin`)."""


def discover_builtin() -> None:
    """Register providers bundled with core.

    Core currently bundles no embedding provider. The public hook remains for
    compatibility with callers that explicitly invoke discovery in two steps.
    """


def discover_all() -> ProviderRegistry:
    """Discover built-in and entry-point providers and return the registry.

    Returns
    -------
    ProviderRegistry
        The global :data:`REGISTRY`, now populated with built-in and
        entry-point providers.
    """
    discover_builtin()
    REGISTRY.discover()
    return REGISTRY


def get(name: str) -> Provider | None:
    """Convenience helper: get a provider by name, registering built-ins on demand.

    Returns
    -------
    Provider | None
        The registered provider for *name*, or ``None`` if no provider
        is registered under that name (after running discovery once if
        the registry was empty).
    """
    # `not REGISTRY` relies on ProviderRegistry.__len__ — an empty registry
    # is falsy, which doubles as the "discovery hasn't run yet" check
    # without a separate boolean flag to keep in sync.
    if not REGISTRY:
        discover_all()
    return REGISTRY.get(name)


def cache_key(provider: Provider, content_hash: str) -> str:
    """Compute the deterministic cache key for a (provider, content) pair.

    Format: ``provider.name|model_id|schema_version|content_hash``.

    Including ``schema_version`` is mandatory — it prevents silent cache
    corruption when a provider's vector shape changes between releases.

    Returns
    -------
    str
        The pipe-delimited cache key string.
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
