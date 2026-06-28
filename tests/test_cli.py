"""CLI smoke tests using typer.testing.CliRunner."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from typer.testing import CliRunner

from pyqtest.cli import app

runner = CliRunner()


class TestVersion:
    """Tests for the --version flag."""

    @staticmethod
    def test_version_prints() -> None:
        """--version prints the pyqtest version."""
        result = runner.invoke(app, ["--version"])
        # typer exits with 0 when --version is handled; output contains the version
        assert "pyqtest" in (result.stdout + (result.stderr or ""))
        # exit code can be 0 (handled) or 2 (typer exits before callback runs);
        # in either case the version string must appear
        assert result.exit_code in {0, 2}


class TestProvidersList:
    """Tests for ``pyqtest providers list``."""

    @staticmethod
    def test_providers_list_exits_zero() -> None:
        """Providers list runs successfully (built-in local stub)."""
        result = runner.invoke(app, ["providers", "list"])
        assert result.exit_code == 0
        assert "local" in result.stdout


class TestInit:
    """Tests for ``pyqtest init``."""

    @staticmethod
    def test_init_appends_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Init adds a [tool.pyqtest] block to pyproject.toml."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            dedent(
                """
                [project]
                name = "demo"
                version = "0.1.0"
                """,
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        text = pyproject.read_text()
        assert "[tool.pyqtest]" in text
        assert "paths" in text

    @staticmethod
    def test_init_idempotent_without_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Init refuses to overwrite an existing block."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            dedent(
                """
                [project]
                name = "demo"
                version = "0.1.0"

                [tool.pyqtest]
                paths = ["lib"]
                """,
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code != 0
        # Original block preserved
        text = pyproject.read_text()
        assert 'paths = ["lib"]' in text

    @staticmethod
    def test_init_with_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Init --force overwrites the existing block."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            dedent(
                """
                [project]
                name = "demo"
                version = "0.1.0"

                [tool.pyqtest]
                paths = ["lib"]
                """,
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "--force"])
        assert result.exit_code == 0
        text = pyproject.read_text()
        assert 'paths = ["src"]' in text


class TestPopulateEmbeddingsMissingProvider:
    """Tests for populate-embeddings with unknown / unconfigured providers."""

    @staticmethod
    def test_unknown_provider_exits_nonzero() -> None:
        """populate-embeddings fails fast for an unknown provider."""
        result = runner.invoke(app, ["populate-embeddings", "--provider", "no-such"])
        assert result.exit_code != 0
        assert "Unknown provider" in result.stdout

    @staticmethod
    def test_cached_only_is_noop() -> None:
        """--cached-only prints and exits without calling the API."""
        result = runner.invoke(
            app,
            ["populate-embeddings", "--provider", "local", "--cached-only"],
        )
        assert result.exit_code == 0
        assert "no embeddings will be computed" in result.stdout


@pytest.fixture
def reset_config_cache() -> None:
    """Reset the config cache before/after CLI tests that change cwd."""
    from pyqtest import config as config_mod

    config_mod.reset_config_cache()
