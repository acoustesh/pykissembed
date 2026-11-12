"""Focused tests for the installed comment-density check."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pykissembed.baselines_engine import BaselineEnvelope, save_envelope
from pykissembed.checks import comment_density
from pykissembed.config import PyqtestConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_consumer_tests_directory_is_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root-wide consumer scan does not score the consumer's tests."""
    (tmp_path / "src").mkdir()
    _ = (tmp_path / "src" / "module.py").write_text(
        "# Explain the related module-level settings.\n"
        "SETTING_A = 1\nSETTING_B = 2\nSETTING_C = 3\nSETTING_D = 4\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    _ = (tmp_path / "tests" / "test_module.py").write_text(
        "\n".join(f"VALUE_{index} = {index}" for index in range(20)),
        encoding="utf-8",
    )
    config = PyqtestConfig(paths=["."], root=tmp_path)
    monkeypatch.setattr(comment_density, "get_config", lambda: config)

    comment_density.TestCommentDensity.test_comment_density(
        [tmp_path],
        update_baselines=False,
    )


@pytest.mark.parametrize(
    "padding",
    ["", "\n" * 40, " \n\t\n" * 20],
    ids=["no-padding", "empty-lines", "whitespace-only-lines"],
)
def test_docstrings_and_blank_lines_are_not_counted_as_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    padding: str,
) -> None:
    """Docstrings and blank lines do not dilute a file's comment density."""
    assignments = "".join(f"SETTING_{index} = {index}\n" for index in range(17))
    _ = (tmp_path / "module.py").write_text(
        f"{padding}"
        '"""Module.\n\nDetails.\n\nMore.\n"""\n'
        "# Explain the related module-level settings.\n"
        f"{assignments}"
        f"{padding}"
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


@pytest.mark.parametrize(
    ("source", "expected_density"),
    [
        ("\n \n\t\n# Comment.\nFIRST = 1\nSECOND = 2\n\n \n", 50.0),
        ("# Comment.\n\nFIRST = 1\n \n\t\nSECOND = 2\n", 50.0),
        ('"""Module.\n\nDetails.\n \n"""\n\n# Comment.\nFIRST = 1\nSECOND = 2\n', 50.0),
        ('"""Module."""\n\n# Comment.\nFIRST = 1\n"More docs."\nSECOND = 2\n', 50.0),
        ('TEXT = """first\n\nsecond\n"""\n\n# Comment.\n', 100.0 / 3),
        ("", 0.0),
        ("\n \n\t\n", 0.0),
        ("# Comment.\n\n \n", 0.0),
    ],
    ids=[
        "leading-and-trailing-blanks",
        "blanks-between-statements",
        "blanks-inside-and-outside-docstring",
        "one-line-string-statements",
        "assigned-multiline-string",
        "empty-source",
        "blank-source",
        "comment-only-source",
    ],
)
def test_density_uses_only_nonblank_code_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    expected_density: float,
) -> None:
    """Physical blank lines never contribute to the density denominator."""
    _ = (tmp_path / "module.py").write_text(source, encoding="utf-8")
    config = PyqtestConfig(paths=["."], root=tmp_path)
    monkeypatch.setattr(comment_density, "get_config", lambda: config)
    save_envelope(
        config.baseline_path / "comment_density.json",
        BaselineEnvelope(
            kind="density",
            data={
                "min_density": round(expected_density, 2),
                "max_density": round(expected_density, 2),
                "aggregate_min_density": 0.0,
                "aggregate_max_density": 100.0,
            },
        ),
    )

    comment_density.TestCommentDensity.test_comment_density(
        [tmp_path],
        update_baselines=False,
    )
