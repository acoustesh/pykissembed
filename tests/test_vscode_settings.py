"""Tests for :mod:`pykissembed.vscode_settings`."""

from __future__ import annotations

import json
import re
from textwrap import dedent
from typing import TYPE_CHECKING

from pykissembed.vscode_settings import sync_vscode_settings

if TYPE_CHECKING:
    from pathlib import Path

_DESIRED_ARGS = ["tests", "--pykissembed-all"]


def _tolerant_load(text: str) -> dict[str, object]:
    """Parse JSONC (comments + trailing commas) for assertions only.

    Returns
    -------
    dict[str, object]
        Parsed settings data.
    """
    no_line_comments = re.sub(r"//[^\n]*", "", text)
    no_block_comments = re.sub(r"/\*.*?\*/", "", no_line_comments, flags=re.DOTALL)
    no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", no_block_comments)
    return json.loads(no_trailing_commas)  # type: ignore[no-any-return]


class TestCreatesFile:
    """When .vscode/settings.json (or .vscode/ itself) is missing."""

    @staticmethod
    def test_creates_vscode_dir_and_settings_file(tmp_path: Path) -> None:
        """No .vscode/ directory at all -> both are created with desired keys."""
        messages = sync_vscode_settings(tmp_path, force=False)

        settings_path = tmp_path / ".vscode" / "settings.json"
        assert settings_path.exists()
        data = _tolerant_load(settings_path.read_text())
        assert data["python.testing.pytestArgs"] == _DESIRED_ARGS
        assert data["python.testing.pytestEnabled"] is True
        assert any("pytestArgs" in m for m in messages)
        assert any("pytestEnabled" in m for m in messages)

    @staticmethod
    def test_creates_settings_file_when_vscode_dir_already_exists(tmp_path: Path) -> None:
        """.vscode/ exists but has no settings.json -> file is created."""
        (tmp_path / ".vscode").mkdir()

        sync_vscode_settings(tmp_path, force=False)

        settings_path = tmp_path / ".vscode" / "settings.json"
        data = _tolerant_load(settings_path.read_text())
        assert data["python.testing.pytestArgs"] == _DESIRED_ARGS
        assert data["python.testing.pytestEnabled"] is True


class TestInsertsMissingKeys:
    """Existing settings.json without the pykissembed keys."""

    @staticmethod
    def test_inserts_missing_keys_preserving_comments_and_trailing_comma(
        tmp_path: Path,
    ) -> None:
        """Unrelated keys, a // comment, and a trailing comma all survive untouched."""
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        settings_path = vscode_dir / "settings.json"
        original = dedent(
            """\
            {
              // keep me
              "editor.formatOnSave": true,
            }
            """,
        )
        settings_path.write_text(original)

        messages = sync_vscode_settings(tmp_path, force=False)

        text = settings_path.read_text()
        assert "// keep me" in text
        assert '"editor.formatOnSave": true' in text
        data = _tolerant_load(text)
        assert data["python.testing.pytestArgs"] == _DESIRED_ARGS
        assert data["python.testing.pytestEnabled"] is True
        assert data["editor.formatOnSave"] is True
        assert any("Added" in m and "pytestArgs" in m for m in messages)


class TestAlreadyInSync:
    """Existing settings.json already has the desired values."""

    @staticmethod
    def test_noop_when_already_in_sync(tmp_path: Path) -> None:
        """File content is left byte-for-byte identical."""
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        settings_path = vscode_dir / "settings.json"
        original = dedent(
            """\
            {
                            "python.testing.pytestArgs": ["tests", "--pykissembed-all"],
              "python.testing.pytestEnabled": true
            }
            """,
        )
        settings_path.write_text(original)

        messages = sync_vscode_settings(tmp_path, force=False)

        assert settings_path.read_text() == original
        assert any("already in sync" in m for m in messages)


class TestConflictingValues:
    """Existing settings.json has different values for the pykissembed keys."""

    @staticmethod
    def _write_conflicting(tmp_path: Path) -> Path:
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        settings_path = vscode_dir / "settings.json"
        original = dedent(
            """\
            {
              "foo": "bar",
              "python.testing.pytestArgs": ["tests", "-v", "--tb=short"],
              "python.testing.pytestEnabled": true
            }
            """,
        )
        settings_path.write_text(original)
        return settings_path

    def test_conflict_without_force_leaves_file_untouched(self, tmp_path: Path) -> None:
        """No --force -> conflicting key is left alone, warning is returned."""
        settings_path = self._write_conflicting(tmp_path)
        original = settings_path.read_text()

        messages = sync_vscode_settings(tmp_path, force=False)

        assert settings_path.read_text() == original
        assert any("pytestArgs" in m and "--force" in m for m in messages)

    def test_conflict_with_force_overwrites_only_target_keys(self, tmp_path: Path) -> None:
        """--force -> conflicting keys are overwritten, unrelated keys survive."""
        settings_path = self._write_conflicting(tmp_path)

        sync_vscode_settings(tmp_path, force=True)

        text = settings_path.read_text()
        assert '"foo": "bar"' in text
        data = _tolerant_load(text)
        assert data["python.testing.pytestArgs"] == _DESIRED_ARGS
        assert data["python.testing.pytestEnabled"] is True
        assert data["foo"] == "bar"


class TestLegacyCachedOnlyArgs:
    """Existing cache-only VS Code settings from earlier releases."""

    @staticmethod
    def _write_legacy_settings(tmp_path: Path) -> Path:
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        settings_path = vscode_dir / "settings.json"
        settings_path.write_text(
            dedent(
                """\
                {
                  "editor.formatOnSave": true,
                  "python.testing.pytestArgs": ["tests", "--pykissembed-all", "--cached-only"],
                  "python.testing.pytestEnabled": true
                }
                """,
            ),
        )
        return settings_path

    def test_legacy_args_are_preserved_without_force(self, tmp_path: Path) -> None:
        """Normal sync never changes a consumer's established cache-only mode."""
        settings_path = self._write_legacy_settings(tmp_path)
        original = settings_path.read_text()

        messages = sync_vscode_settings(tmp_path, force=False)

        assert settings_path.read_text() == original
        assert any("pytestArgs" in message and "--force" in message for message in messages)

    def test_force_replaces_legacy_args_only(self, tmp_path: Path) -> None:
        """Forced sync migrates pytest args while preserving unrelated settings."""
        settings_path = self._write_legacy_settings(tmp_path)

        sync_vscode_settings(tmp_path, force=True)

        data = _tolerant_load(settings_path.read_text())
        assert data["python.testing.pytestArgs"] == _DESIRED_ARGS
        assert data["editor.formatOnSave"] is True


class TestUnparseableFile:
    """Existing settings.json that can't be understood even tolerantly."""

    @staticmethod
    def test_unparseable_file_left_untouched_with_warning(tmp_path: Path) -> None:
        """A broken file is never touched -- warn instead of risking corruption."""
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        settings_path = vscode_dir / "settings.json"
        original = "{ this is not valid json at all ["
        settings_path.write_text(original)

        messages = sync_vscode_settings(tmp_path, force=False)

        assert settings_path.read_text() == original
        assert any("Could not parse" in m for m in messages)
