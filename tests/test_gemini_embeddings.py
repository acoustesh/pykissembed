"""Regression test for the Gemini embedding request builder."""

from __future__ import annotations

import gc
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

from pykissembed.similarity.embeddings import _build_provider_caller

if TYPE_CHECKING:
    import pytest


class _Transport:
    """Stand-in for the SDK's HTTP client; the real SDK closes it in ``Client.__del__``."""

    def __init__(self) -> None:
        self.closed = False


class _Models:
    def __init__(self, transport: _Transport) -> None:
        self._transport = transport

    def embed_content(self, *, model: str, contents: list[object], config: object) -> SimpleNamespace:
        _ = model, config
        # Real gemini-embedding-2 aggregates a bare list of strings into one vector.
        count = 1 if all(isinstance(c, str) for c in contents) else len(contents)
        if self._transport.closed:
            msg = "Cannot send a request, as the client has been closed."
            raise RuntimeError(msg)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5]) for _ in range(count)])


class _Client:
    def __init__(self, **_kwargs: object) -> None:
        self._transport = _Transport()
        self.models = _Models(self._transport)

    def __del__(self) -> None:
        self._transport.closed = True


def test_gemini_request_keeps_client_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    """The request closure must hold the client, or GC closes its transport."""
    genai = ModuleType("google.genai")
    monkeypatch.setattr(genai, "Client", _Client, raising=False)
    types = ModuleType("google.genai.types")
    monkeypatch.setattr(types, "HttpOptions", dict, raising=False)
    monkeypatch.setattr(types, "EmbedContentConfig", dict, raising=False)
    monkeypatch.setattr(types, "Content", dict, raising=False)
    monkeypatch.setattr(types, "Part", dict, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-with-at-least-20-chars")

    make_request, _ = _build_provider_caller("gemini", "gemini-embedding-2", timeout=5.0)
    _ = gc.collect()

    assert make_request(["def f(): pass", "def g(): pass"]) == [[0.5], [0.5]]
