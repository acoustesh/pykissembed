"""CLI smoke tests using typer.testing.CliRunner."""

from __future__ import annotations

import tomllib
from textwrap import dedent
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from pykissembed import config as config_mod
from pykissembed.cli import _auto_detect_paths, app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


class TestVersion:
    """Tests for the --version flag."""

    @staticmethod
    def test_version_prints() -> None:
        """--version prints the pykissembed version."""
        result = runner.invoke(app, ["--version"])
        # typer exits with 0 when --version is handled; output contains the version
        assert "pykissembed" in (result.stdout + (result.stderr or ""))
        # exit code can be 0 (handled) or 2 (typer exits before callback runs);
        # in either case the version string must appear
        assert result.exit_code in {0, 2}


class TestProvidersList:
    """Tests for ``pykissembed.providers list``."""

    @staticmethod
    def test_providers_list_exits_zero() -> None:
        """Providers list runs successfully (built-in local stub)."""
        result = runner.invoke(app, ["providers", "list"])
        assert result.exit_code == 0
        assert "local" in result.stdout


class TestInit:
    """Tests for ``pykissembed init``."""

    @staticmethod
    def test_init_appends_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Init adds a [tool.pykissembed] block to pyproject.toml."""
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
        assert "[tool.pykissembed]" in text
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

                [tool.pykissembed]
                paths = ["lib"]
                """,
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        # The pyproject.toml sub-step is skipped (not a fatal error) since the
        # independent .vscode/settings.json sync still runs and succeeds.
        assert result.exit_code == 0
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

                [tool.pykissembed]
                paths = ["lib"]
                """,
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "--force"])
        assert result.exit_code == 0
        text = pyproject.read_text()
        # Auto-detect: no src/ dir in the temp project → falls back to "."
        assert 'paths = ["."]' in text

    @staticmethod
    def test_init_with_force_overwrites_full_block(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        r"""Init --force on a block that already has paths/mode/dirs stays valid TOML.

        Regression test: the old ``[^\[]*`` substitution regex stopped at the
        first ``[`` character anywhere in the block -- including the one
        inside ``paths = ["."]`` -- leaving a mangled leftover fragment
        (e.g. a stray ``["."]`` line) appended after the new block.
        """
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            dedent(
                """
                [project]
                name = "demo"
                version = "0.1.0"

                [tool.pykissembed]
                paths = ["."]
                mode = "ratchet"
                baseline_dir = "tests/baselines"
                cache_dir = "tests/.pykissembed_cache"
                """,
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init", "--force"])
        assert result.exit_code == 0
        text = pyproject.read_text()
        parsed = tomllib.loads(text)
        assert parsed["tool"]["pykissembed"]["paths"] == ["."]
        assert text.count("[tool.pykissembed]") == 1
        # The old buggy regex left a dangling `["."]` fragment behind, which
        # TOML happily (re-)parses as a bogus top-level table named ".".
        assert set(parsed.keys()) == {"project", "tool"}

    @staticmethod
    def test_init_creates_vscode_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Init also creates .vscode/settings.json with pykissembed's pytest settings."""
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
        settings = (tmp_path / ".vscode" / "settings.json").read_text()
        assert '"python.testing.pytestArgs"' in settings
        assert '"--pykissembed-all"' in settings
        assert '"--cached-only"' not in settings
        assert '"python.testing.pytestEnabled": true' in settings

    @staticmethod
    def test_init_vscode_settings_idempotent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running init twice in a row is a no-op the second time."""
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
        runner.invoke(app, ["init"])
        settings_path = tmp_path / ".vscode" / "settings.json"
        first_run = settings_path.read_text()

        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert settings_path.read_text() == first_run

    @staticmethod
    def test_init_syncs_vscode_settings_without_force_on_existing_block(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An already-onboarded project still gets .vscode/settings.json synced."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            dedent(
                """
                [project]
                name = "demo"
                version = "0.1.0"

                [tool.pykissembed]
                paths = ["lib"]
                """,
            ),
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        # pyproject.toml block untouched (no --force)
        assert 'paths = ["lib"]' in pyproject.read_text()
        # .vscode/settings.json was still synced
        settings = (tmp_path / ".vscode" / "settings.json").read_text()
        assert '"python.testing.pytestEnabled": true' in settings


class TestAutoDetectPaths:
    """Tests for ``pykissembed init`` source-path discovery."""

    @staticmethod
    def test_setuptools_find_paths_take_priority(tmp_path: Path) -> None:
        """Setuptools discovery wins over every lower-priority source."""
        (tmp_path / "src").mkdir()
        pyproject = dedent(
            """
            [tool.setuptools.packages.find]
            where = ["lib", "vendor"]

            [tool.hatch.build.targets.wheel]
            packages = ["package/demo"]
            """
        )

        assert _auto_detect_paths(tmp_path, pyproject) == ["lib", "vendor"]

    @staticmethod
    def test_hatch_package_roots_are_distinct(tmp_path: Path) -> None:
        """Hatch package paths reduce to distinct top-level source roots."""
        pyproject = dedent(
            """
            [tool.hatch.build.targets.wheel]
            packages = ["src/demo", "src/support", "package"]
            """
        )

        assert _auto_detect_paths(tmp_path, pyproject) == ["src", "package"]

    @staticmethod
    def test_src_directory_is_the_unconfigured_fallback(tmp_path: Path) -> None:
        """An unconfigured project with ``src/`` selects that directory."""
        (tmp_path / "src").mkdir()

        assert _auto_detect_paths(tmp_path, "") == ["src"]

    @staticmethod
    def test_invalid_toml_falls_back_to_current_directory(tmp_path: Path) -> None:
        """Malformed TOML without a ``src/`` directory falls back to ``.``."""
        assert _auto_detect_paths(tmp_path, "[tool") == ["."]


class TestCheck:
    """Tests for ``pykissembed check``."""

    @staticmethod
    def test_check_with_no_args_defaults_to_pykissembed_all(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No args → forwards ``--pykissembed-all`` so the full battery still runs.

        Regression test: since v0.1.9, bare ``pytest`` no longer
        auto-collects pykissembed's check modules. Without this default,
        ``pykissembed check`` (no args) would silently collect zero tests.
        """
        captured: list[str] = []

        def _fake_call(cmd: list[str]) -> int:
            captured.extend(cmd)
            return 0

        monkeypatch.setattr("pykissembed.cli.subprocess.call", _fake_call)
        result = runner.invoke(app, ["check"])
        assert result.exit_code == 0
        assert "--pykissembed-all" in captured

    @staticmethod
    def test_check_with_explicit_args_passes_through_unchanged(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit args (e.g. a marker) are forwarded as-is, unmodified."""
        captured: list[str] = []

        def _fake_call(cmd: list[str]) -> int:
            captured.extend(cmd)
            return 0

        monkeypatch.setattr("pykissembed.cli.subprocess.call", _fake_call)
        result = runner.invoke(app, ["check", "--", "-m", "lint"])
        assert result.exit_code == 0
        assert "--pykissembed-all" not in captured
        assert "-m" in captured
        assert "lint" in captured


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
    config_mod.reset_config_cache()
