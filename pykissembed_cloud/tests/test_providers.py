"""Unit tests for the OpenRouter-routed cloud providers.

These tests do not hit the network. The ``openai.OpenAI`` client is
monkeypatched to return a fake embedding object that mimics the shape
of an OpenAI ``CreateEmbeddingResponse``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pykissembed.providers import Provider

from pykissembed_cloud import dotenv as _dotenv
from pykissembed_cloud.dotenv import find_dotenv as _real_find_dotenv
from pykissembed_cloud.providers.gemini import GeminiProvider
from pykissembed_cloud.providers.openai import OpenAIProvider
from pykissembed_cloud.providers.qwen import QwenProvider


@pytest.fixture(autouse=True)
def _isolate_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from the real ``.env`` lookup.

    The repo has a real ``.env`` file at the project root. Without
    isolation, every test would trigger a filesystem walk that finds
    it. The autouse fixture:

    1. Resets the module-level "loaded once" cache.
    2. Stubs ``find_dotenv`` to return ``None`` by default, so no
       ``.env`` is found unless a test explicitly overrides the stub.
    """
    _dotenv.reset_cache()
    monkeypatch.setattr(_dotenv, "find_dotenv", lambda start=None: None)


ALL_PROVIDERS = (OpenAIProvider, GeminiProvider, QwenProvider)
ALL_PROVIDER_INSTANCES = (OpenAIProvider(), GeminiProvider(), QwenProvider())


class TestAttributes:
    """Static-attribute sanity checks for each provider."""

    @staticmethod
    @pytest.mark.parametrize(
        ("provider", "expected_name", "expected_model"),
        [
            (OpenAIProvider, "openai", "openai/text-embedding-3-large"),
            (GeminiProvider, "gemini", "google/gemini-embedding-001"),
            (QwenProvider, "qwen", "qwen/qwen3-embedding-8b"),
        ],
    )
    def test_identity_attributes(
        provider: type[Provider],
        expected_name: str,
        expected_model: str,
    ) -> None:
        """Each provider exposes the documented identity attributes."""
        inst = provider()
        assert inst.name == expected_name
        assert inst.model_id == expected_model
        assert inst.schema_version == "1"
        assert inst.max_tokens > 0
        assert inst.batch_size > 0

    @staticmethod
    @pytest.mark.parametrize("provider", ALL_PROVIDER_INSTANCES)
    def test_satisfies_provider_protocol(provider: Provider) -> None:
        """Each provider structurally satisfies the pykissembed Provider Protocol."""
        assert isinstance(provider, Provider)


class TestIsConfigured:
    """Tests for ``is_configured`` based on ``OPENROUTER_API_KEY``."""

    @staticmethod
    @pytest.mark.parametrize("provider", ALL_PROVIDER_INSTANCES)
    def test_unconfigured_when_key_missing(
        provider: Provider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """is_configured() returns False when the env var is unset."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert provider.is_configured() is False

    @staticmethod
    @pytest.mark.parametrize("provider", ALL_PROVIDER_INSTANCES)
    def test_configured_when_key_present(
        provider: Provider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """is_configured() returns True when the env var is set."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
        assert provider.is_configured() is True


class TestEmbed:
    """Tests for ``embed()`` — request shape and response parsing."""

    @staticmethod
    def test_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
        """embed() raises a clear RuntimeError without an API key."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            OpenAIProvider().embed(["hello"])

    @staticmethod
    def test_single_batch_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
        """A small input is sent in one batch with the right model id."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")

        captured: dict[str, Any] = {}

        class _FakeEmbeddings:
            @staticmethod
            def create(*, input: Sequence[str], model: str) -> Any:  # noqa: A002
                captured["input"] = list(input)
                captured["model"] = model
                return _FakeResponse([[0.1, 0.2, 0.3] for _ in input])

        class _FakeClient:
            def __init__(self, *, base_url: str, api_key: str) -> None:
                captured["base_url"] = base_url
                captured["api_key"] = api_key
                self.embeddings = _FakeEmbeddings()

        import openai as openai_module

        monkeypatch.setattr(openai_module, "OpenAI", _FakeClient)

        vectors = OpenAIProvider().embed(["alpha", "beta"])
        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2, 0.3]
        assert captured["model"] == "openai/text-embedding-3-large"
        assert captured["input"] == ["alpha", "beta"]
        assert captured["base_url"] == "https://openrouter.ai/api/v1/"
        assert captured["api_key"] == "sk-or-fake"

    @staticmethod
    def test_batches_inputs_larger_than_batch_size(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inputs are split into ``batch_size`` chunks and re-assembled."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")

        # Qwen has the smallest batch_size (32); use it for the chunking test.
        provider = QwenProvider()
        n = provider.batch_size + 5
        inputs = [f"text-{i}" for i in range(n)]

        calls: list[Sequence[str]] = []

        class _FakeEmbeddings:
            def create(self, *, input: Sequence[str], model: str) -> Any:  # noqa: A002
                del model
                calls.append(list(input))
                return _FakeResponse([[0.0] for _ in input])

        class _FakeClient:
            def __init__(self, *, base_url: str, api_key: str) -> None:
                self.embeddings = _FakeEmbeddings()

        import openai as openai_module

        monkeypatch.setattr(openai_module, "OpenAI", _FakeClient)

        vectors = provider.embed(inputs)
        assert len(vectors) == n
        assert len(calls) == 2
        assert len(calls[0]) == provider.batch_size
        assert len(calls[1]) == 5
        # Order preserved
        assert all(v == [0.0] for v in vectors)

    @staticmethod
    def test_empty_input_returns_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
        """``embed([])`` short-circuits to an empty list — no API call."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")

        called = {"n": 0}

        class _FakeEmbeddings:
            def create(self, *, input: Sequence[str], model: str) -> Any:  # noqa: A002
                del model
                called["n"] += 1
                return _FakeResponse([])

        class _FakeClient:
            def __init__(self, *, base_url: str, api_key: str) -> None:
                self.embeddings = _FakeEmbeddings()

        import openai as openai_module

        monkeypatch.setattr(openai_module, "OpenAI", _FakeClient)
        assert OpenAIProvider().embed([]) == []
        assert called["n"] == 0


@dataclass(slots=True)
class _FakeItem:
    """A single fake embedding row matching the OpenAI shape."""

    embedding: list[float]


class _FakeResponse:
    """A fake ``CreateEmbeddingResponse`` exposing ``.data``."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeItem(v) for v in vectors]


class TestEntryPoints:
    """Verify the providers are discoverable via the entry-point group."""

    @staticmethod
    def test_entry_points_register_three_providers() -> None:
        """The entry-point group declares exactly the three cloud providers."""
        from importlib import metadata

        eps = {ep.name: ep.value for ep in metadata.entry_points(group="pykissembed.providers")}
        assert eps.get("openai") == "pykissembed_cloud.providers.openai:OpenAIProvider"
        assert eps.get("gemini") == "pykissembed_cloud.providers.gemini:GeminiProvider"
        assert eps.get("qwen") == "pykissembed_cloud.providers.qwen:QwenProvider"
        # The core local stub is also in the group
        assert "local" in eps


class TestDotenv:
    """Tests for the lazy .env loader."""

    @staticmethod
    def test_parse_simple_key_value() -> None:
        """A bare ``KEY=value`` line parses cleanly."""
        assert _dotenv.parse_dotenv("FOO=bar") == {"FOO": "bar"}

    @staticmethod
    def test_parse_handles_quotes_and_comments() -> None:
        """Surrounding quotes are stripped; comments and blanks are ignored."""
        text = "# a comment\n\nFOO=\"bar\"\nBAZ='qux'\n  spaced = value  \nNOPE_no_equals\n"
        assert _dotenv.parse_dotenv(text) == {
            "FOO": "bar",
            "BAZ": "qux",
            "spaced": "value",
        }

    @staticmethod
    def test_find_dotenv_walks_up(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``find_dotenv`` walks up parent directories until it finds a ``.env``."""
        # Note: the autouse ``_isolate_dotenv`` stub is *replaced* here
        # with the real implementation so the walk actually happens.
        monkeypatch.setattr(_dotenv, "find_dotenv", _real_find_dotenv)
        (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test", encoding="utf-8")
        child = tmp_path / "child" / "grandchild"
        child.mkdir(parents=True)
        monkeypatch.chdir(child)
        found = _dotenv.find_dotenv()
        assert found == (tmp_path / ".env")

    @staticmethod
    def test_find_dotenv_returns_none_when_missing(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``find_dotenv`` returns ``None`` when no ``.env`` exists on the walk."""
        monkeypatch.setattr(_dotenv, "find_dotenv", _real_find_dotenv)
        isolated = tmp_path / "no_env_here"
        isolated.mkdir()
        monkeypatch.chdir(isolated)
        # Walk up to / which definitely has no .env
        assert _dotenv.find_dotenv(isolated) is None

    @staticmethod
    def test_load_into_environ_picks_up_file(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``load_into_environ`` populates the env from a .env file."""
        dotenv = tmp_path / ".env"
        dotenv.write_text("OPENROUTER_API_KEY=sk-from-file", encoding="utf-8")
        monkeypatch.setattr(_dotenv, "find_dotenv", lambda start=None: dotenv)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        loaded = _dotenv.load_into_environ(("OPENROUTER_API_KEY",), start=tmp_path)
        assert loaded == dotenv
        import os

        assert os.environ["OPENROUTER_API_KEY"] == "sk-from-file"

    @staticmethod
    def test_explicit_env_wins_over_file(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit env var is preserved when the file has a different value."""
        dotenv = tmp_path / ".env"
        dotenv.write_text("OPENROUTER_API_KEY=from-file", encoding="utf-8")
        monkeypatch.setattr(_dotenv, "find_dotenv", lambda start=None: dotenv)
        monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
        _dotenv.load_into_environ(("OPENROUTER_API_KEY",), start=tmp_path)
        import os

        assert os.environ["OPENROUTER_API_KEY"] == "from-env"

    @staticmethod
    def test_ensure_loaded_is_idempotent(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``ensure_loaded`` only loads the file once per process."""
        dotenv = tmp_path / ".env"
        dotenv.write_text("OPENROUTER_API_KEY=once", encoding="utf-8")
        calls = {"n": 0}

        def _spy(_start: Path | None = None) -> Path | None:
            calls["n"] += 1
            return dotenv

        monkeypatch.setattr(_dotenv, "find_dotenv", _spy)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        _dotenv.ensure_loaded()
        _dotenv.ensure_loaded()
        _dotenv.ensure_loaded()
        assert calls["n"] == 1

    @staticmethod
    def test_is_configured_loads_dotenv(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``is_configured()`` triggers the lazy .env load."""
        dotenv = tmp_path / ".env"
        dotenv.write_text("OPENROUTER_API_KEY=sk-from-file", encoding="utf-8")
        monkeypatch.setattr(_dotenv, "find_dotenv", lambda start=None: dotenv)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert OpenAIProvider().is_configured() is True
        import os

        assert os.environ["OPENROUTER_API_KEY"] == "sk-from-file"
