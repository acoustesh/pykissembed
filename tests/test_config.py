"""Tests for pykissembed config loading."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from pykissembed.config import _auto_detect, _coerce_str_list, load_config


class TestCoerceStrList:
    """Tests for the string-list coercion helper."""

    @staticmethod
    def test_string_passes_through() -> None:
        """A bare string becomes a single-item list."""
        assert _coerce_str_list("src", key="paths") == ["src"]

    @staticmethod
    def test_list_passes_through() -> None:
        """A list of strings passes through unchanged."""
        assert _coerce_str_list(["src", "lib"], key="paths") == ["src", "lib"]

    @staticmethod
    def test_non_list_non_string_yields_empty() -> None:
        """A non-string, non-list value becomes an empty list."""
        assert _coerce_str_list(42, key="paths") == []
        assert _coerce_str_list(None, key="paths") == []

    @staticmethod
    def test_list_with_non_string_raises() -> None:
        """A list containing non-strings raises TypeError."""
        with pytest.raises(TypeError, match="paths"):
            _coerce_str_list(["src", 42], key="paths")


class TestLoadConfig:
    """Tests for the config loader."""

    @staticmethod
    def test_loads_from_pyproject(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pyproject.toml with [tool.pykissembed] loads correctly."""
        (tmp_path / "pyproject.toml").write_text(
            dedent(
                """
                [tool.pykissembed]
                paths = ["src", "lib"]
                mode = "strict"
                baseline_dir = "tests/baselines"
                cache_dir = "tests/.pykissembed_cache"
                """,
            ),
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.paths == ["src", "lib"]
        assert config.mode == "strict"
        assert config.baseline_dir == "tests/baselines"
        assert config.root == tmp_path

    @staticmethod
    def test_defaults_when_no_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A pyproject.toml without [tool.pykissembed] gives sensible defaults."""
        (tmp_path / "pyproject.toml").write_text(
            dedent(
                """
                [project]
                name = "demo"
                version = "0.1.0"
                """,
            ),
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.paths == ["src"]
        assert config.mode == "ratchet"

    @staticmethod
    def test_invalid_mode_falls_back_to_ratchet(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unrecognised mode falls back to 'ratchet'."""
        (tmp_path / "pyproject.toml").write_text(
            dedent(
                """
                [tool.pykissembed]
                paths = ["src"]
                mode = "i-am-not-a-real-mode"
                """,
            ),
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.mode == "ratchet"


class TestAutoDetect:
    """Tests for the auto-detection fallback."""

    @staticmethod
    def test_finds_src(tmp_path: Path) -> None:
        """A src/ directory is auto-detected."""
        (tmp_path / "src").mkdir()
        config = _auto_detect(tmp_path)
        assert config.paths == ["src"]

    @staticmethod
    def test_finds_scripts(tmp_path: Path) -> None:
        """A scripts/ directory is auto-detected when no src/ exists."""
        (tmp_path / "scripts").mkdir()
        config = _auto_detect(tmp_path)
        assert config.paths == ["scripts"]

    @staticmethod
    def test_falls_back_to_src(tmp_path: Path) -> None:
        """An empty project falls back to 'src'."""
        config = _auto_detect(tmp_path)
        assert config.paths == ["src"]
