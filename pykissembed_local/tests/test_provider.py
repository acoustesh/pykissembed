"""Unit tests for the real ``LocalProvider`` and ``runner.populate``.

Sentence-transformers is monkeypatched, so no model weights are loaded.
The live model-load path is covered by the ``tests/integration`` suite.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pykissembed.providers import Provider

from pykissembed_local.provider import DEFAULT_MODEL_ID, LocalProvider
from pykissembed_local.runner import cache_key, populate, truncate_to_tokens


class _FakeNumpyArray:
    """Minimal stand-in for a numpy ndarray — supports ``[float(x) for x in row]``."""

    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def __iter__(self) -> Any:  # type: ignore[no-untyped-def]
        return iter(self._rows)


def _install_fake_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch, vectors: list[list[float]]
) -> Any:
    """Patch ``sentence_transformers.SentenceTransformer`` with a fake model.

    The fake returns a real ``numpy.ndarray`` so the provider's
    ``np.asarray(vectors)`` call works without surprises.

    Returns
    -------
    Any
        A dict with captured ``model_id``, ``inputs``, and ``kwargs`` so
        tests can assert on the request shape.
    """
    import numpy as np

    captured: dict[str, Any] = {}

    class _FakeModel:
        def encode(self, inputs, **_kwargs: Any) -> Any:
            captured["inputs"] = list(inputs)
            captured["kwargs"] = _kwargs
            # Return a real 2D ndarray matching the input order
            return np.asarray(vectors, dtype=float)

    class _FakeSentenceTransformer:
        instances: list[_FakeModel] = []

        def __init__(self, model_id: str) -> None:
            captured["model_id"] = model_id
            self._impl = _FakeModel()
            type(self).instances.append(self._impl)

        def encode(self, inputs, **kwargs: Any) -> Any:
            return self._impl.encode(inputs, **kwargs)

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    return captured


class TestProviderAttributes:
    """Identity-attribute sanity checks."""

    @staticmethod
    def test_default_attributes() -> None:
        """The default provider advertises the documented attributes."""
        p = LocalProvider()
        assert p.name == "local"
        assert p.model_id == DEFAULT_MODEL_ID
        assert p.schema_version == "1"
        assert p.max_tokens == 512
        assert p.batch_size == 16

    @staticmethod
    def test_satisfies_provider_protocol() -> None:
        """The provider structurally satisfies the pykissembed Provider Protocol."""
        assert isinstance(LocalProvider(), Provider)

    @staticmethod
    def test_explicit_model_id_overrides_default() -> None:
        """A passed-in model id takes precedence over the default."""
        p = LocalProvider(model_id="sentence-transformers/all-MiniLM-L6-v2")
        assert p.model_id == "sentence-transformers/all-MiniLM-L6-v2"

    @staticmethod
    def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
        """``PYQTEST_LOCAL_MODEL`` overrides the default when no arg is passed."""
        monkeypatch.setenv("PYQTEST_LOCAL_MODEL", "intfloat/e5-small-v2")
        assert LocalProvider().model_id == "intfloat/e5-small-v2"


class TestIsConfigured:
    """``is_configured`` reflects whether sentence-transformers is importable."""

    @staticmethod
    def test_true_when_importable(monkeypatch: pytest.MonkeyPatch) -> None:
        """is_configured() returns True when the package can be imported."""
        fake_st = types.ModuleType("sentence_transformers")
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
        assert LocalProvider().is_configured() is True

    @staticmethod
    def test_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
        """is_configured() returns False when the package is not installed."""
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        assert LocalProvider().is_configured() is False


class TestEmbed:
    """Tests for ``embed()`` with a mocked SentenceTransformer."""

    @staticmethod
    def test_empty_input_returns_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
        """``embed([])`` short-circuits to an empty list (no model load)."""
        _install_fake_sentence_transformers(monkeypatch, vectors=[])
        p = LocalProvider()
        assert p.embed([]) == []

    @staticmethod
    def test_returns_python_floats(monkeypatch: pytest.MonkeyPatch) -> None:
        """Vectors come back as ``list[list[float]]`` of plain Python floats."""
        _install_fake_sentence_transformers(
            monkeypatch,
            vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )
        p = LocalProvider()
        out = p.embed(["alpha", "beta"])
        assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        for vec in out:
            for x in vec:
                assert isinstance(x, float)

    @staticmethod
    def test_model_loads_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
        """The SentenceTransformer is only constructed on the first embed call."""
        captured = _install_fake_sentence_transformers(monkeypatch, vectors=[[0.0]])
        p = LocalProvider()
        # No model load yet
        assert "model_id" not in captured
        p.embed(["x"])
        assert captured["model_id"] == DEFAULT_MODEL_ID

    @staticmethod
    def test_passes_batch_size_and_normalisation(monkeypatch: pytest.MonkeyPatch) -> None:
        """embed() forwards batch_size and normalize_embeddings=True to encode."""
        captured = _install_fake_sentence_transformers(monkeypatch, vectors=[[0.0]])
        LocalProvider().embed(["x"])
        assert captured["kwargs"]["batch_size"] == 16
        assert captured["kwargs"]["normalize_embeddings"] is True

    @staticmethod
    def test_pickle_round_trip_drops_cached_model(monkeypatch: pytest.MonkeyPatch) -> None:
        """``__getstate__`` drops the cached model; ``__setstate__`` clears it."""
        _install_fake_sentence_transformers(monkeypatch, vectors=[[0.0]])
        p = LocalProvider()
        p.embed(["x"])  # forces a model load
        state = p.__getstate__()
        assert state == {"model_id": DEFAULT_MODEL_ID, "_model": None}
        # Round-trip through pickle to confirm both dunders are correct.
        # Safe here: we control the bytes — this is a self-pickle in a test.
        import pickle  # noqa: S403

        restored = pickle.loads(pickle.dumps(p))  # noqa: S301
        assert restored._model is None
        assert restored.model_id == DEFAULT_MODEL_ID


class TestRunner:
    """Tests for ``runner.populate``."""

    @staticmethod
    def test_cache_key_format() -> None:
        """Cache keys follow the documented ``name|model|schema|hash`` format."""
        p = LocalProvider()
        key = cache_key(p, "hello world")
        assert key.startswith(f"local|{DEFAULT_MODEL_ID}|1|")
        assert len(key.split("|")[-1]) == 64  # SHA-256 hex digest

    @staticmethod
    def test_truncate_to_tokens_passthrough_short_text() -> None:
        """Text within the token budget is returned unchanged."""
        out = truncate_to_tokens("hello world", max_tokens=10)
        assert out == "hello world"

    @staticmethod
    def test_truncate_to_tokens_drops_long_text() -> None:
        """Text exceeding the budget is truncated to fit."""
        long = " ".join(["token"] * 500)
        out = truncate_to_tokens(long, max_tokens=50)
        # Round-trip via cl100k — at most 50 tokens
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        assert len(enc.encode(out, disallowed_special=())) <= 50

    @staticmethod
    def test_populate_writes_cache_file(
        tmp_path: Path,
    ) -> None:
        """populate() writes one JSONL record per .py file."""
        # Arrange: a fake source tree
        src = tmp_path / "src" / "pkg"
        src.mkdir(parents=True)
        (src / "a.py").write_text("print('a')", encoding="utf-8")
        (src / "b.py").write_text("print('b')", encoding="utf-8")

        # Arrange: fake the provider so no real model is loaded
        captured: list[str] = []
        vectors = [[0.1, 0.2], [0.3, 0.4]]

        class _FakeProvider:
            name = "local"
            model_id = DEFAULT_MODEL_ID
            schema_version = "1"
            max_tokens = 512
            batch_size = 16

            def embed(self, texts):  # type: ignore[no-untyped-def]
                captured.extend(texts)
                return list(vectors)

            def is_configured(self) -> bool:
                return True

        cache_dir = tmp_path / "cache"
        n = populate(_FakeProvider(), [src], cache_dir, project_root=tmp_path)  # type: ignore[arg-type]
        assert n == 2
        assert len(captured) == 2
        cache_file = cache_dir / f"local-{DEFAULT_MODEL_ID.replace('/', '_')}.jsonl"
        assert cache_file.is_file()
        rows = [json.loads(line) for line in cache_file.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert {r["path"] for r in rows} == {"src/pkg/a.py", "src/pkg/b.py"}
        for row in rows:
            assert "key" in row
            assert "vector" in row
            assert isinstance(row["vector"], list)

    @staticmethod
    def test_populate_is_idempotent(
        tmp_path: Path,
    ) -> None:
        """Re-running populate() with no new files writes 0 new entries."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("print('a')", encoding="utf-8")

        class _FakeProvider:
            name = "local"
            model_id = DEFAULT_MODEL_ID
            schema_version = "1"
            max_tokens = 512
            batch_size = 16

            def __init__(self) -> None:
                self.calls = 0

            def embed(self, texts):  # type: ignore[no-untyped-def]
                self.calls += 1
                return [[0.0] for _ in texts]

            def is_configured(self) -> bool:
                return True

        provider = _FakeProvider()
        cache_dir = tmp_path / "cache"
        first = populate(provider, [src], cache_dir, project_root=tmp_path)  # type: ignore[arg-type]
        second = populate(provider, [src], cache_dir, project_root=tmp_path)  # type: ignore[arg-type]
        assert first == 1
        assert second == 0
        assert provider.calls == 1  # second call skipped because everything is cached

    @staticmethod
    def test_populate_skips_missing_directory(
        tmp_path: Path,
    ) -> None:
        """A path that doesn't exist is silently skipped."""
        provider_calls = {"n": 0}

        class _FakeProvider:
            name = "local"
            model_id = DEFAULT_MODEL_ID
            schema_version = "1"
            max_tokens = 512
            batch_size = 16

            def embed(self, texts):  # type: ignore[no-untyped-def]
                provider_calls["n"] += 1
                return [[0.0] for _ in texts]

            def is_configured(self) -> bool:
                return True

        missing = tmp_path / "does-not-exist"
        n = populate(_FakeProvider(), [missing], tmp_path / "cache")  # type: ignore[arg-type]
        assert n == 0
        assert provider_calls["n"] == 0

    @staticmethod
    def test_populate_raises_on_mismatched_vector_count(
        tmp_path: Path,
    ) -> None:
        """If the provider returns the wrong number of vectors, raise clearly."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("a", encoding="utf-8")
        (src / "b.py").write_text("b", encoding="utf-8")

        class _WrongCountProvider:
            name = "local"
            model_id = DEFAULT_MODEL_ID
            schema_version = "1"
            max_tokens = 512
            batch_size = 16

            def embed(self, texts):  # type: ignore[no-untyped-def]
                return [[0.0]]  # Wrong: only 1 vector for 2 inputs

            def is_configured(self) -> bool:
                return True

        with pytest.raises(RuntimeError, match="returned 1 vectors for 2 inputs"):
            populate(_WrongCountProvider(), [src], tmp_path / "cache", project_root=tmp_path)  # type: ignore[arg-type]


class TestEntryPoints:
    """The provider is discoverable via the entry-point group."""

    @staticmethod
    def test_entry_point_registered() -> None:
        """The local entry point is registered for the pykissembed.providers group.

        Both the core stub (``pykissembed.providers.local``) and the real
        implementation (``pykissembed_local.provider``) declare an entry
        point named ``local``. The registry keeps the last-registered
        entry per name, so we accept either — the test only asserts
        that *some* ``local`` entry point is registered.
        """
        from importlib import metadata

        eps = {ep.name: ep.value for ep in metadata.entry_points(group="pykissembed.providers")}
        assert "local" in eps
        assert eps["local"] in {
            "pykissembed_local.provider:LocalProvider",
            "pykissembed.providers.local:LocalProvider",
        }
