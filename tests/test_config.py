"""Tests for pykissembed config loading."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

from pykissembed.config import (
    _auto_detect,
    _coerce_str_list,
    _require_nonnegative_int,
    _require_str_list,
    load_config,
)
from pykissembed.paths import iter_py_files

if TYPE_CHECKING:
    from pathlib import Path


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


class TestWrapperProliferationConfig:
    """Tests for wrapper-proliferation configuration validation."""

    @staticmethod
    def test_requires_string_lists() -> None:
        """Wrapper exclusions and decorator patterns must be string lists."""
        assert _require_str_list(["pkg/*.py:Adapter.*"], key="wrapper_exclude") == [
            "pkg/*.py:Adapter.*"
        ]
        with pytest.raises(TypeError, match="wrapper_exclude"):
            _require_str_list("pkg/*.py:Adapter.*", key="wrapper_exclude")

    @staticmethod
    def test_requires_nonnegative_call_site_threshold() -> None:
        """The wrapper call-site threshold rejects booleans and negatives."""
        assert _require_nonnegative_int(1, key="wrapper_max_call_sites") == 1
        for invalid in (-1, True, "1"):
            with pytest.raises(TypeError, match="wrapper_max_call_sites"):
                _require_nonnegative_int(invalid, key="wrapper_max_call_sites")


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
                wrapper_max_call_sites = 2
                wrapper_exclude = ["src/demo.py:Adapter.*"]
                wrapper_exempt_decorators = ["framework.*"]
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
        assert config.wrapper_max_call_sites == 2
        assert config.wrapper_exclude == ["src/demo.py:Adapter.*"]
        assert config.wrapper_exempt_decorators == ["framework.*"]
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
        assert config.wrapper_max_call_sites == 1
        assert config.wrapper_exclude == []
        assert config.wrapper_exempt_decorators == []

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

    @staticmethod
    @pytest.mark.parametrize(
        ("setting", "message"),
        [
            ("wrapper_max_call_sites = -1", "wrapper_max_call_sites"),
            ('wrapper_exclude = "src/module.py:proxy"', "wrapper_exclude"),
            ("wrapper_exempt_decorators = [42]", "wrapper_exempt_decorators"),
        ],
    )
    def test_invalid_wrapper_settings_raise(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        setting: str,
        message: str,
    ) -> None:
        """Wrapper-policy keys reject invalid TOML values during config loading."""
        (tmp_path / "pyproject.toml").write_text(
            dedent(
                f"""
                [tool.pykissembed]
                paths = ["src"]
                {setting}
                """,
            ),
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)

        with pytest.raises(TypeError, match=message):
            load_config()


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


class TestIncludeNotebooks:
    """Tests for the include_notebooks config flag."""

    @staticmethod
    def test_default_is_false(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """include_notebooks defaults to False (notebooks excluded)."""
        (tmp_path / "pyproject.toml").write_text(
            dedent(
                """
                [tool.pykissembed]
                paths = ["src"]
                """,
            ),
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.include_notebooks is False

    @staticmethod
    def test_explicit_true_is_loaded(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """include_notebooks = true loads as True."""
        (tmp_path / "pyproject.toml").write_text(
            dedent(
                """
                [tool.pykissembed]
                paths = ["src"]
                include_notebooks = true
                """,
            ),
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.include_notebooks is True


class TestCachedOnly:
    """Tests for the cached_only config flag."""

    @staticmethod
    def test_default_is_true(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cloud population is opt-in when cached_only is omitted."""
        (tmp_path / "pyproject.toml").write_text(
            dedent(
                """
                [tool.pykissembed]
                paths = ["src"]
                """,
            ),
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.cached_only is True

    @staticmethod
    def test_explicit_true_is_loaded(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """cached_only = true enables cache-only similarity checks."""
        (tmp_path / "pyproject.toml").write_text(
            dedent(
                """
                [tool.pykissembed]
                paths = ["src"]
                cached_only = true
                """,
            ),
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.cached_only is True

    @staticmethod
    def test_explicit_false_is_loaded(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """cached_only = false explicitly permits cloud auto-population."""
        (tmp_path / "pyproject.toml").write_text(
            '[tool.pykissembed]\npaths = ["src"]\ncached_only = false\n',
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)

        assert load_config().cached_only is False

    @staticmethod
    @pytest.mark.parametrize("value", ["0", '""', "[]", '"false"'])
    def test_non_boolean_value_cannot_enable_cloud_population(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
    ) -> None:
        """Only the TOML boolean false is accepted as cloud-population consent."""
        (tmp_path / "pyproject.toml").write_text(
            f'[tool.pykissembed]\npaths = ["src"]\ncached_only = {value}\n',
            encoding="utf-8",
        )
        (tmp_path / "src").mkdir()
        monkeypatch.chdir(tmp_path)

        with pytest.raises(TypeError, match=r"cached_only.*boolean"):
            load_config()


class TestIterPyFiles:
    """Tests for the iter_py_files helper in paths.py."""

    @staticmethod
    def test_iter_py_files_skips_ipynb(tmp_path: Path) -> None:
        """iter_py_files() only yields .py files, never .ipynb."""
        (tmp_path / "real_module.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "notebook.ipynb").write_text("{}", encoding="utf-8")
        files = sorted(p.name for p in iter_py_files(tmp_path))
        assert files == ["real_module.py"]
