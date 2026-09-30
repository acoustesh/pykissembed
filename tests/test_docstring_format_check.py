"""Tests for how the docstring-format gate invokes ruff."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest

from pykissembed.checks import docstring_format

_VIOLATION = (
    '[{"filename": "pkg/m.py", "location": {"row": 1, "column": 1}, '
    '"code": "D100", "message": "Missing docstring"}]'
)
_PREVIEW_ERROR = "Selecting rules by name requires preview mode"


def _fake_run(calls: list[list[str]], replies: list[subprocess.CompletedProcess[str]]) -> object:
    """Build a ``subprocess.run`` stand-in that records argv and replays results.

    Parameters
    ----------
    calls : list[list[str]]
        Receives the argv of every call.
    replies : list[subprocess.CompletedProcess[str]]
        Results returned in order.

    Returns
    -------
    object
        A callable with ``subprocess.run``'s calling convention.
    """

    def run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        """Record *cmd* and return the next scripted result.

        Parameters
        ----------
        cmd : list[str]
            The argv ruff would be run with.
        **_kwargs : object
            Ignored ``subprocess.run`` options.

        Returns
        -------
        subprocess.CompletedProcess[str]
            The next scripted result.
        """
        calls.append(cmd)
        return replies[len(calls) - 1]

    return run


def _patch(
    monkeypatch: pytest.MonkeyPatch, replies: list[subprocess.CompletedProcess[str]]
) -> list[list[str]]:
    """Route the gate's ruff calls to scripted *replies*.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    replies : list[subprocess.CompletedProcess[str]]
        Results for successive ruff runs.

    Returns
    -------
    list[list[str]]
        The argv list each call appends to.
    """
    calls: list[list[str]] = []
    monkeypatch.setattr(docstring_format.shutil, "which", lambda _name: "ruff")
    monkeypatch.setattr(docstring_format, "include_notebooks", lambda: True)
    monkeypatch.setattr(docstring_format.subprocess, "run", _fake_run(calls, replies))
    return calls


def test_retries_in_preview_mode_when_the_config_needs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config that needs ``--preview`` is retried with it, not treated as clean."""
    calls = _patch(
        monkeypatch,
        [
            subprocess.CompletedProcess([], 2, "", _PREVIEW_ERROR),
            subprocess.CompletedProcess([], 1, _VIOLATION, ""),
        ],
    )
    violations = docstring_format.run_ruff_docstring_check(Path("pkg"), root=Path.cwd())
    assert [v.code for v in violations] == ["D100"]
    assert "--preview" not in calls[0]
    assert "--preview" in calls[1]
    # Both attempts use the gate's rule selection, not the project's own.
    assert all("--select=D" in call for call in calls)


def test_a_ruff_usage_error_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any other ruff error raises instead of reporting zero violations."""
    _ = _patch(monkeypatch, [subprocess.CompletedProcess([], 2, "", "unknown option --bogus")])
    with pytest.raises(RuntimeError, match="unknown option"):
        _ = docstring_format.run_ruff_docstring_check(Path("pkg"), root=Path.cwd())


@pytest.mark.skipif(shutil.which("ruff") is None, reason="needs the ruff executable")
def test_select_d_overrides_the_project_ignore_but_not_the_convention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--select=D`` counts ignored D rules; the pydocstyle convention still filters.

    The project ignores D100 and D401, yet the gate reports both, because a
    command-line ``--select`` replaces the configured selection. D213 stays
    silent: the numpy convention disables it regardless of ``--select``.
    """
    _ = (tmp_path / "pyproject.toml").write_text(
        dedent(
            """\
            [tool.ruff.lint]
            select = ["D"]
            ignore = ["D100", "D401"]

            [tool.ruff.lint.pydocstyle]
            convention = "numpy"
            """
        )
    )
    _ = (tmp_path / "m.py").write_text(
        'def f():\n    """Returns one.\n\n    More.\n    """\n    return 1\n'
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(docstring_format, "include_notebooks", lambda: False)
    violations = docstring_format.run_ruff_docstring_check(tmp_path, root=tmp_path)
    assert {v.code for v in violations} == {"D100", "D401"}
