"""Focused tests for the installed comment-density check."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pykissembed.checks import comment_density
from pykissembed.config import PyqtestConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_consumer_tests_directory_is_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root-wide consumer scan does not score the consumer's tests."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text(
        "# Explain the related module-level settings.\n"
        "SETTING_A = 1\nSETTING_B = 2\nSETTING_C = 3\nSETTING_D = 4\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_module.py").write_text(
        "\n".join(f"VALUE_{index} = {index}" for index in range(20)),
        encoding="utf-8",
    )
    config = PyqtestConfig(paths=["."], root=tmp_path)
    monkeypatch.setattr(comment_density, "get_config", lambda: config)

    comment_density.TestCommentDensity.test_comment_density(
        [tmp_path],
        update_baselines=False,
    )


def test_docstring_lines_are_not_counted_as_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Docstring lines, blank or one-line, do not dilute a file's comment density."""
    assignments = "".join(f"SETTING_{index} = {index}\n" for index in range(17))
    _ = (tmp_path / "module.py").write_text(
        '"""Module.\n\nDetails.\n\nMore.\n"""\n'
        "# Explain the related module-level settings.\n"
        f"{assignments}"
        "class Holder:\n"
        '    """One line."""\n',
        encoding="utf-8",
    )
    config = PyqtestConfig(paths=["."], root=tmp_path)
    monkeypatch.setattr(comment_density, "get_config", lambda: config)

    # 1 comment over 18 code lines is 5.6%; counting the module docstring's two
    # blank lines and the one-line class docstring as code would give 4.8%.
    comment_density.TestCommentDensity.test_comment_density(
        [tmp_path],
        update_baselines=False,
    )
