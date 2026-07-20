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
