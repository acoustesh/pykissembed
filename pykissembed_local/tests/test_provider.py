"""Tests for the lightweight pykissembed-local compatibility tombstone."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import pytest

import pykissembed_local
from pykissembed_local import runner
from pykissembed_local.provider import DEFAULT_MODEL_ID, LocalProvider

PROJECT_ROOT = Path(__file__).parents[1]
FORBIDDEN_IMPORTS = {
    "huggingface_hub",
    "numpy",
    "pandas",
    "safetensors",
    "sentence_transformers",
    "tiktoken",
    "tokenizers",
    "torch",
    "transformers",
    "triton",
}


@runtime_checkable
class _ProviderContract(Protocol):
    """Public provider shape retained by the compatibility package."""

    name: str
    model_id: str
    schema_version: str
    max_tokens: int
    batch_size: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    def is_configured(self) -> bool:
        """Return whether the provider can generate embeddings."""
        ...


def _assert_cloud_only(exc_info: pytest.ExceptionInfo[RuntimeError]) -> None:
    """Assert that a retired operation gives actionable migration guidance."""
    message = str(exc_info.value)
    assert "Local embeddings were removed" in message
    assert "pykissembed populate-embeddings --provider openai-text" in message
    assert "gemini-text" in message
    assert "voyage-text" in message


class TestPackagePolicy:
    """Distribution metadata and import-safety checks."""

    @staticmethod
    def test_package_exports_provider() -> None:
        """The historical root import remains available."""
        assert pykissembed_local.LocalProvider is LocalProvider
        assert isinstance(pykissembed_local.__version__, str)

    @staticmethod
    def test_manifest_has_no_runtime_dependencies_or_entry_point() -> None:
        """The tombstone neither installs model libraries nor registers a provider."""
        manifest = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = manifest["project"]
        assert project["dependencies"] == []
        assert "entry-points" not in project

    @staticmethod
    def test_import_does_not_load_forbidden_packages() -> None:
        """A clean interpreter imports the compatibility modules without heavy libraries."""
        check = (
            "import json, sys\n"
            "import pykissembed_local\n"
            "import pykissembed_local.runner\n"
            f"blocked = {sorted(FORBIDDEN_IMPORTS)!r}\n"
            "print(json.dumps(sorted(name for name in blocked if name in sys.modules)))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", check],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(completed.stdout) == []


class TestLocalProvider:
    """Compatibility identity and retired-operation behavior."""

    @staticmethod
    def test_default_attributes() -> None:
        """The provider retains its historical identity attributes."""
        provider = LocalProvider()
        assert provider.name == "local"
        assert provider.model_id == DEFAULT_MODEL_ID
        assert provider.schema_version == "1"
        assert provider.max_tokens == 512
        assert provider.batch_size == 16
        assert provider._model is None

    @staticmethod
    def test_satisfies_provider_contract() -> None:
        """The tombstone still has the synchronous Provider shape."""
        assert isinstance(LocalProvider(), _ProviderContract)

    @staticmethod
    def test_explicit_model_id_overrides_default() -> None:
        """The legacy constructor override remains readable."""
        assert LocalProvider(model_id="legacy/model").model_id == "legacy/model"

    @staticmethod
    def test_environment_model_id_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
        """The historical environment override still controls identity state."""
        monkeypatch.setenv("PYQTEST_LOCAL_MODEL", "legacy/environment-model")
        assert LocalProvider().model_id == "legacy/environment-model"

    @staticmethod
    def test_is_never_configured(monkeypatch: pytest.MonkeyPatch) -> None:
        """Installed or injected model modules cannot reactivate the tombstone."""
        monkeypatch.setitem(sys.modules, "sentence_transformers", object())
        assert LocalProvider().is_configured() is False

    @staticmethod
    @pytest.mark.parametrize("texts", [[], ["hello"], ["hello", "world"]])
    def test_embed_always_raises_cloud_only_error(texts: list[str]) -> None:
        """Even empty inputs cannot silently emulate the former provider."""
        with pytest.raises(RuntimeError) as exc_info:
            LocalProvider().embed(texts)
        _assert_cloud_only(exc_info)

    @staticmethod
    def test_private_model_loader_is_also_retired() -> None:
        """Older callers of the lazy-loader receive the same migration error."""
        with pytest.raises(RuntimeError) as exc_info:
            LocalProvider()._ensure_loaded()
        _assert_cloud_only(exc_info)

    @staticmethod
    def test_identity_state_round_trip() -> None:
        """Historical pickling hooks restore identity without a model object."""
        original = LocalProvider(model_id="legacy/model")
        state = original.__getstate__()
        assert state == {"model_id": "legacy/model", "_model": None}

        restored = LocalProvider.__new__(LocalProvider)
        restored.__setstate__(state)
        assert restored.model_id == "legacy/model"
        assert restored._model is None
        assert restored.is_configured() is False


class TestRunnerCompatibility:
    """Pure legacy helpers remain usable while operational helpers fail safely."""

    @staticmethod
    def test_content_hash_is_stable() -> None:
        """The old SHA-256 helper retains its exact output."""
        assert runner.content_hash("hello") == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )

    @staticmethod
    def test_cache_key_format_is_stable() -> None:
        """Pure cache-key composition remains available for migration tools."""
        key = runner.cache_key(LocalProvider(), "hello")
        assert key == f"local|{DEFAULT_MODEL_ID}|1|{runner.content_hash('hello')}"

    @staticmethod
    def test_iter_py_files_preserves_filtering(tmp_path: Path) -> None:
        """File discovery stays deterministic and excludes dunder/cache files."""
        source = tmp_path / "source"
        nested = source / "nested"
        cache = source / "__pycache__"
        nested.mkdir(parents=True)
        cache.mkdir()
        (source / "b.py").write_text("b", encoding="utf-8")
        (source / "__init__.py").write_text("", encoding="utf-8")
        (nested / "a.py").write_text("a", encoding="utf-8")
        (cache / "ignored.py").write_text("ignored", encoding="utf-8")

        discovered = [path.relative_to(source).as_posix() for path in runner.iter_py_files(source)]
        assert discovered == ["b.py", "nested/a.py"]

    @staticmethod
    def test_collect_inputs_remains_pure(tmp_path: Path) -> None:
        """The old collector can inspect files without invoking an embedding provider."""
        source = tmp_path / "src"
        source.mkdir()
        (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        assert runner._collect_inputs([source], project_root=tmp_path) == [
            {"path": "src/module.py", "content": "VALUE = 1\n"},
        ]

    @staticmethod
    def test_truncation_is_retired() -> None:
        """Token-aware local preprocessing fails without importing a tokenizer."""
        with pytest.raises(RuntimeError) as exc_info:
            runner.truncate_to_tokens("short", max_tokens=512)
        _assert_cloud_only(exc_info)

    @staticmethod
    def test_encoding_loader_is_retired() -> None:
        """The historical private loader cannot fetch tokenizer data."""
        with pytest.raises(RuntimeError) as exc_info:
            runner._get_encoding()
        _assert_cloud_only(exc_info)

    @staticmethod
    def test_populate_fails_before_provider_or_filesystem_use(tmp_path: Path) -> None:
        """The JSONL runner raises before calling a provider or creating a cache."""

        class _UnexpectedProvider:
            name = "local"
            model_id = DEFAULT_MODEL_ID
            schema_version = "1"
            max_tokens = 512
            batch_size = 16

            def embed(self, texts: Sequence[str]) -> list[list[float]]:
                raise AssertionError(texts)

        cache_dir = tmp_path / "cache"
        with pytest.raises(RuntimeError) as exc_info:
            runner.populate(
                _UnexpectedProvider(),
                [tmp_path],
                cache_dir,
                project_root=tmp_path,
            )
        _assert_cloud_only(exc_info)
        assert not cache_dir.exists()
