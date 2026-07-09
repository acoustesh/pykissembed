"""Smoke tests that the ``[local]`` / ``[cloud]`` extras resolve correctly.

Specifically: the extras declared in ``pykissembed/pyproject.toml`` actually
resolve and register their provider entry points when installed into a fresh
project via ``uv``.

Marked ``slow`` — skipped by default. Enable with::

    uv run pytest -m slow tests/test_install_extras.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _require_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:  # pragma: no cover - environment guard
        pytest.skip("`uv` not on PATH; cannot exercise install extras")
    return uv


def _build_wheels(destination: Path, sources: list[Path]) -> None:
    """Build wheels for each source directory into *destination*."""
    destination.mkdir(parents=True, exist_ok=True)
    for src in sources:
        # S603: fixed argv (resolved uv binary + literal flags + repo-internal paths).
        subprocess.run(  # noqa: S603
            [_require_uv(), "build", "--wheel", "--out-dir", str(destination), str(src)],
            cwd=REPO_ROOT,
            check=True,
        )


def _make_consumer_project(tmp_path: Path, find_links: Path) -> Path:
    """Scaffold a minimal consumer project that depends on ``pykissembed[local]``."""
    project = tmp_path / "consumer"
    project.mkdir()
    pyproject = textwrap.dedent(
        f"""
        [project]
        name = "consumer"
        version = "0.0.0"
        requires-python = ">=3.14"
        dependencies = ["pykissembed[local]"]

        [tool.uv]
        find-links = ["{find_links}"]

        [tool.uv.sources]
        pykissembed-local = {{ path = "{REPO_ROOT / "pykissembed_local"}" }}
        """,
    )
    (project / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return project


def _list_local_entry_points(venv_python: Path) -> list[str]:
    """Return all entry-point values registered under ``local`` in the consumer venv."""
    code = textwrap.dedent(
        """
        from importlib.metadata import entry_points

        matches = entry_points(group="pykissembed.providers", name="local")
        print("\\n".join(str(ep.value) for ep in matches))
        """,
    )
    # S603: fixed argv (a venv's own python + a literal -c script).
    result = subprocess.run(  # noqa: S603
        [str(venv_python), "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _active_local_provider(venv_python: Path) -> str | None:
    """Return the dotted path of the provider that wins in the registry."""
    code = textwrap.dedent(
        """
        from pykissembed.providers import registry

        registry.discover_all()
        provider = registry.REGISTRY.get("local")
        print(type(provider).__module__ + ":" + type(provider).__name__)
        """,
    )
    # S603: fixed argv (a venv's own python + a literal -c script).
    result = subprocess.run(  # noqa: S603
        [str(venv_python), "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() or None


@pytest.mark.slow
def test_local_extra_resolves_and_registers_provider(tmp_path: Path) -> None:
    """``uv add "pykissembed[local]"`` must register the ``local`` provider."""
    uv = _require_uv()
    wheels = tmp_path / "wheels"
    _build_wheels(
        wheels,
        [REPO_ROOT, REPO_ROOT / "pykissembed_local"],
    )

    consumer = _make_consumer_project(tmp_path, find_links=wheels)

    # Allow normal network resolution for transitive deps; ``--find-links``
    # only short-circuits the two local wheels we just built.
    # S603: fixed argv (resolved uv binary + literal flags).
    subprocess.run(  # noqa: S603
        [uv, "sync", "--no-dev"],
        cwd=consumer,
        check=True,
    )

    venv_python = consumer / ".venv" / "bin" / "python"
    if not venv_python.exists():  # pragma: no cover - platform-specific layout
        venv_python = consumer / ".venv" / "Scripts" / "python.exe"

    entry_points = _list_local_entry_points(venv_python)
    assert "pykissembed_local.provider:LocalProvider" in entry_points, (
        "expected the pykissembed-local subpackage to register its provider via "
        "the [local] extra, but entry_points(group='pykissembed.providers') "
        f"resolved to {entry_points!r}"
    )

    active = _active_local_provider(venv_python)
    assert active == "pykissembed_local.provider:LocalProvider", (
        "expected the pykissembed-local subpackage's provider to win in the "
        "registry when both the stub and the subpackage are installed, but the "
        f"active provider resolved to {active!r}"
    )


def test_extras_declared_in_pyproject() -> None:
    """Regression guard: ``local`` and ``cloud`` (and ``all``) must be declared."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    for name in ("local", "cloud", "all"):
        assert name in extras, f"missing extra [{name}] in pyproject.toml"
    assert "pykissembed-local" in extras["local"]
    assert "pykissembed-cloud" in extras["cloud"]
    assert "pykissembed[local]" in extras["all"] or any(
        "pykissembed[local]" in str(d) for d in extras["all"]
    )
    assert "pykissembed[cloud]" in extras["all"] or any(
        "pykissembed[cloud]" in str(d) for d in extras["all"]
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-vv"]))
