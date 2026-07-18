"""Tests for the Provider Protocol + registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

import pykissembed.providers.registry as reg_mod
from pykissembed.providers import REGISTRY, Provider
from pykissembed.providers.registry import cache_key

if TYPE_CHECKING:
    from collections.abc import Sequence


class _TestProvider:
    """Small provider implementation used to exercise the public registry."""

    name = "test"
    model_id = "test/model"
    schema_version = "1"
    max_tokens = 256
    batch_size = 32

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one deterministic vector per input text.

        Returns
        -------
        list[list[float]]
            One length-derived vector for each input text.
        """
        return [[float(len(text))] for text in texts]

    def is_configured(self) -> bool:
        """Return whether this synthetic provider is ready.

        Returns
        -------
        bool
            Always ``True`` for this test implementation.
        """
        return True


class TestRegistry:
    """Tests for the ProviderRegistry."""

    def setup_method(self) -> None:
        """Start each test with an empty registry."""
        REGISTRY.clear()

    def teardown_method(self) -> None:
        """Restore the registry after each test."""
        REGISTRY.clear()

    @staticmethod
    def test_register_and_get() -> None:
        """register() then get() returns the same instance."""
        p = _TestProvider()
        assert isinstance(p, Provider)
        REGISTRY.register(p)
        assert REGISTRY.get("test") is p

    @staticmethod
    def test_register_overwrites() -> None:
        """Re-registering the same name overwrites the prior entry."""
        a = _TestProvider()
        b = _TestProvider()
        b.schema_version = "2"
        REGISTRY.register(a)
        REGISTRY.register(b)
        assert REGISTRY.get("test") is b
        assert cast("_TestProvider", REGISTRY.get("test")).schema_version == "2"

    @staticmethod
    def test_discover_handles_broken_entry_point(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A broken entry-point load is skipped silently."""

        class _BrokenEP:
            name = "broken"
            value = "no.such.module:thing"

            def load(self) -> None:
                msg = "boom"
                raise ImportError(msg)

        eps = [_BrokenEP()]

        monkeypatch.setattr(
            reg_mod.metadata,
            "entry_points",
            lambda group=None: eps,  # ruff:ignore[unused-lambda-argument] — must accept `group=` to match the real entry_points(group=...) call
        )
        with pytest.warns(RuntimeWarning, match="failed to load provider"):
            registry = reg_mod.discover_all()
        assert len(registry) == 0
        assert "broken" not in registry

    @staticmethod
    def test_discover_isolates_provider_constructor_failure(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A provider constructor failure does not hide healthy extensions."""

        class _BrokenProvider:
            def __init__(self) -> None:
                msg = "constructor boom"
                raise RuntimeError(msg)

        class _EP:
            def __init__(self, name: str, value: object) -> None:
                self.name = name
                self.value = value

            def load(self) -> object:
                return self.value

        healthy = _TestProvider()
        monkeypatch.setattr(
            reg_mod.metadata,
            "entry_points",
            lambda group=None: [_EP("broken", _BrokenProvider), _EP("healthy", healthy)],  # ruff:ignore[unused-lambda-argument]
        )

        with pytest.warns(RuntimeWarning, match="constructor boom"):
            registry = reg_mod.discover_all()

        assert registry.get("test") is healthy

    @staticmethod
    def test_discover_lets_first_entry_point_override(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When two entry points share a name, the first-listed one wins.

        This matches the user-override semantics: a user-installed provider
        should override an earlier registration when its entry point is listed
        first in the metadata. ``discover()`` iterates entry points in
        **reverse** so the first one (override) is registered last,
        overwriting the second (stub).
        """

        class _StubProvider:
            name = "custom"
            model_id = "stub-model"
            schema_version = "1"
            max_tokens = 256
            batch_size = 32

            def embed(self, _texts):  # type: ignore[no-untyped-def]
                return []

            def is_configured(self) -> bool:
                return True

        class _OverrideProvider:
            name = "custom"
            model_id = "override-model"
            schema_version = "2"
            max_tokens = 512
            batch_size = 16

            def embed(self, _texts):  # type: ignore[no-untyped-def]
                return []

            def is_configured(self) -> bool:
                return True

        class _EP:
            def __init__(self, name: str, value: object) -> None:
                self.name = name
                self.value = value
                self.group = "pykissembed.providers"

            def load(self) -> object:
                # Return the instance directly so ``isinstance(instance, Provider)`` works
                return self.value

        eps = [
            _EP("custom", _OverrideProvider()),  # listed first → wins
            _EP("custom", _StubProvider()),
        ]

        REGISTRY.clear()
        monkeypatch.setattr(
            reg_mod.metadata,
            "entry_points",
            lambda group=None: eps,  # ruff:ignore[unused-lambda-argument] — must accept `group=` to match the real entry_points(group=...) call
        )
        reg_mod.discover_all()
        winner = REGISTRY.get("custom")
        assert winner is not None
        assert cast("_OverrideProvider", winner).model_id == "override-model"
        REGISTRY.clear()


class TestCacheKey:
    """Tests for the cache-key formatter."""

    @staticmethod
    def test_cache_key_format() -> None:
        """Cache keys include name, model, schema_version, and content hash."""
        p = _TestProvider()
        key = cache_key(p, "abc123")
        assert key == "test|test/model|1|abc123"

    @staticmethod
    def test_cache_key_differs_per_content() -> None:
        """Different content hashes produce different cache keys."""
        p = _TestProvider()
        assert cache_key(p, "a") != cache_key(p, "b")

    @staticmethod
    def test_cache_key_differs_per_schema_version() -> None:
        """Different schema versions produce different cache keys."""
        p = _TestProvider()
        v1 = cache_key(p, "h")
        p.schema_version = "2"
        v2 = cache_key(p, "h")
        assert v1 != v2
