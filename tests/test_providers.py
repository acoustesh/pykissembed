"""Tests for the Provider Protocol + registry."""

from __future__ import annotations

import sys
from typing import cast

import pytest

import pykissembed.providers.registry as reg_mod
from pykissembed.providers import REGISTRY, Provider
from pykissembed.providers.local import LocalProvider
from pykissembed.providers.registry import cache_key, discover_all


class TestLocalProviderStub:
    """Tests for the in-tree stub LocalProvider."""

    @staticmethod
    def test_stub_attributes() -> None:
        """The stub exposes the documented attributes."""
        p = LocalProvider()
        assert p.name == "local"
        assert p.model_id == "sentence-transformers/all-MiniLM-L6-v2"
        assert p.schema_version == "1"
        assert p.max_tokens == 256
        assert p.batch_size == 32

    @staticmethod
    def test_stub_satisfies_protocol() -> None:
        """The stub class is structurally a Provider (Protocol)."""
        assert isinstance(LocalProvider(), Provider)

    @staticmethod
    def test_stub_embed_raises_when_local_missing(monkeypatch: pytest.MonkeyPatch) -> None:
        """embed() must raise a clear RuntimeError when pykissembed-local is missing."""
        # Hide pykissembed_local from the import system
        monkeypatch.setitem(sys.modules, "pykissembed_local", None)
        p = LocalProvider()
        with pytest.raises(RuntimeError, match="pykissembed-local"):
            p.embed(["hello"])

    @staticmethod
    def test_is_configured_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
        """is_configured() returns False when pykissembed_local isn't installed."""
        monkeypatch.setitem(sys.modules, "pykissembed_local", None)
        assert LocalProvider().is_configured() is False


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
        p = LocalProvider()
        REGISTRY.register(p)
        assert REGISTRY.get("local") is p

    @staticmethod
    def test_register_overwrites() -> None:
        """Re-registering the same name overwrites the prior entry."""
        a = LocalProvider()
        b = LocalProvider()
        b.schema_version = "2"
        REGISTRY.register(a)
        REGISTRY.register(b)
        assert REGISTRY.get("local") is b
        assert cast("LocalProvider", REGISTRY.get("local")).schema_version == "2"

    @staticmethod
    def test_discover_builtin_finds_local() -> None:
        """discover_all() registers the built-in local stub."""
        registry = discover_all()
        assert "local" in registry

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
            lambda group=None: eps,  # noqa: ARG005 — must accept `group=` to match the real entry_points(group=...) call
        )
        registry = reg_mod.discover_all()
        # Built-in still registers despite the broken entry point
        assert "local" in registry
        assert "broken" not in registry

    @staticmethod
    def test_discover_lets_first_entry_point_override(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When two entry points share a name, the first-listed one wins.

        This matches the user-override semantics: ``pykissembed-local`` should
        override the core stub because its entry point is listed first
        in the metadata. ``discover()`` iterates entry points in
        **reverse** so the first one (override) is registered last,
        overwriting the second (stub).
        """

        class _StubProvider:
            name = "local"
            model_id = "stub-model"
            schema_version = "1"
            max_tokens = 256
            batch_size = 32

            def embed(self, _texts):  # type: ignore[no-untyped-def]
                return []

            def is_configured(self) -> bool:
                return True

        class _OverrideProvider:
            name = "local"
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
            _EP("local", _OverrideProvider()),  # listed first → wins
            _EP("local", _StubProvider()),
        ]

        REGISTRY.clear()
        monkeypatch.setattr(
            reg_mod.metadata,
            "entry_points",
            lambda group=None: eps,  # noqa: ARG005 — must accept `group=` to match the real entry_points(group=...) call
        )
        reg_mod.discover_all()
        winner = REGISTRY.get("local")
        assert winner is not None
        assert cast("_OverrideProvider", winner).model_id == "override-model"
        REGISTRY.clear()


class TestCacheKey:
    """Tests for the cache-key formatter."""

    @staticmethod
    def test_cache_key_format() -> None:
        """Cache keys include name, model, schema_version, and content hash."""
        p = LocalProvider()
        key = cache_key(p, "abc123")
        assert key == "local|sentence-transformers/all-MiniLM-L6-v2|1|abc123"

    @staticmethod
    def test_cache_key_differs_per_content() -> None:
        """Different content hashes produce different cache keys."""
        p = LocalProvider()
        assert cache_key(p, "a") != cache_key(p, "b")

    @staticmethod
    def test_cache_key_differs_per_schema_version() -> None:
        """Different schema versions produce different cache keys."""
        p = LocalProvider()
        v1 = cache_key(p, "h")
        p.schema_version = "2"
        v2 = cache_key(p, "h")
        assert v1 != v2
