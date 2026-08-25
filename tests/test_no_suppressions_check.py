"""Focused tests for the installed no-suppressions consumer check."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pykissembed.checks import no_suppressions
from pykissembed.config import PyqtestConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_scan_finds_directives_and_typing_cast_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comment directives and imported typing casts are all reported."""
    source = tmp_path / "module.py"
    source.write_text(
        "import typing as t\n"
        "from typing import cast as narrow\n"
        "from typing_extensions import cast as extension_cast\n"
        "from typing import *\n"
        "\n"
        "first = object()  # type : ignore[assignment]\n"
        "second = object()  # NOQA: F841\n"
        "third = narrow(str, first)\n"
        "fourth = t.cast(str, second)\n"
        "fifth = extension_cast(str, third)\n"
        "sixth = cast(str, fourth)\n",
        encoding="utf-8",
    )
    config = PyqtestConfig(paths=["."], root=tmp_path)
    monkeypatch.setattr(no_suppressions, "get_config", lambda: config)

    with pytest.raises(pytest.fail.Exception) as exc_info:
        no_suppressions.test_no_suppressions_or_casts()

    message = str(exc_info.value)
    for expected in (
        "module.py:6:19: type-ignore",
        "module.py:7:20: noqa",
        "module.py:8:9: typing.cast",
        "module.py:9:10: typing.cast",
        "module.py:10:9: typing.cast",
        "module.py:11:9: typing.cast",
    ):
        assert expected in message


def test_scan_ignores_strings_docstrings_and_unrelated_casts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directive-like text and non-typing cast functions do not trigger findings."""
    source = tmp_path / "clean.py"
    source.write_text(
        '"""Examples: # noqa, # type: ignore, and cast(str, value)."""\n'
        "from project_helpers import cast as project_cast\n"
        "\n"
        "def cast(value: object) -> object:\n"
        '    """Return a project-specific converted value."""\n'
        "    return value\n"
        "\n"
        'TEXT = "# noqa and # type: ignore"\n'
        "FIRST = cast(TEXT)\n"
        "SECOND = project_cast(TEXT)\n",
        encoding="utf-8",
    )
    config = PyqtestConfig(paths=["."], root=tmp_path)
    monkeypatch.setattr(no_suppressions, "get_config", lambda: config)

    no_suppressions.test_no_suppressions_or_casts()


def test_project_scan_excludes_tests_environments_and_cache_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every configured infrastructure-directory exclusion is honored."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "clean.py").write_text("VALUE = 1\n", encoding="utf-8")
    forbidden_source = "VALUE = object()  # type: ignore[assignment]\n"
    excluded_directories = {
        ".eggs",
        ".env",
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pykissembed_cache",
        ".pyright",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "node_modules",
        "site-packages",
        "tests",
        "venv",
    }
    for directory in excluded_directories:
        ignored_dir = tmp_path / directory / "nested"
        ignored_dir.mkdir(parents=True)
        (ignored_dir / "ignored.py").write_text(forbidden_source, encoding="utf-8")
    config = PyqtestConfig(paths=["scripts"], root=tmp_path)
    monkeypatch.setattr(no_suppressions, "get_config", lambda: config)

    no_suppressions.test_no_suppressions_or_casts()


def test_consumer_gate_scans_scripts_outside_configured_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict gate covers the project root, not only configured source paths."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "clean.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "unsafe.py").write_text(
        "from typing import cast\nVALUE = cast(str, object())\n",
        encoding="utf-8",
    )
    config = PyqtestConfig(paths=["src"], root=tmp_path)
    monkeypatch.setattr(no_suppressions, "get_config", lambda: config)

    with pytest.raises(
        pytest.fail.Exception,
        match=r"scripts/unsafe\.py:2:9: typing\.cast",
    ):
        no_suppressions.test_no_suppressions_or_casts()


def test_clean_consumer_passes_suppression_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean consumer root completes without a pytest failure."""
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = PyqtestConfig(paths=["package"], root=tmp_path)
    monkeypatch.setattr(no_suppressions, "get_config", lambda: config)

    no_suppressions.test_no_suppressions_or_casts()
