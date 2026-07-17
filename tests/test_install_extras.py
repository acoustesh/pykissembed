"""Slow isolated-install tests for core and cloud-only compatibility extras.

Enable with ``uv run pytest -m slow tests/test_install_extras.py``. These tests
build wheels and resolve a fresh consumer environment, so they may use the
package index for ordinary non-workspace dependencies.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD_ENTRY_POINTS = {
    "gemini": "pykissembed_cloud.providers.gemini:GeminiProvider",
    "jina": "pykissembed_cloud.providers.jina:JinaProvider",
    "openai": "pykissembed_cloud.providers.openai:OpenAIProvider",
    "qwen": "pykissembed_cloud.providers.qwen:QwenProvider",
}
FORBIDDEN_DISTRIBUTIONS = {
    "hf-xet",
    "huggingface-hub",
    "pandas",
    "safetensors",
    "sentence-transformers",
    "tokenizers",
    "torch",
    "transformers",
    "triton",
    "voyageai",
}


def _requirement_name(requirement: str) -> str:
    """Return the canonical distribution name from a wheel requirement.

    Returns
    -------
    str
        Lowercase normalized distribution name.

    Raises
    ------
    AssertionError
        If the wheel contains malformed requirement metadata.
    """
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:
        raise AssertionError(requirement)
    return match.group().lower().replace("_", "-")


def _assert_no_forbidden(names: set[str]) -> None:
    """Assert that *names* contains no retired heavy distribution."""
    forbidden = sorted(
        name
        for name in names
        if name in FORBIDDEN_DISTRIBUTIONS or name.startswith(("cuda-", "nvidia-"))
    )
    assert forbidden == []


def _require_uv() -> str:
    """Return the uv executable or skip when it is unavailable.

    Returns
    -------
    str
        Path to the uv executable.
    """
    uv = shutil.which("uv")
    if uv is None:  # pragma: no cover - environment guard
        pytest.skip("`uv` not on PATH; cannot exercise isolated installs")
    return uv


def _build_wheels(destination: Path) -> None:
    """Build core, cloud, and tombstone wheels into *destination*."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in (REPO_ROOT, REPO_ROOT / "pykissembed_cloud", REPO_ROOT / "pykissembed_local"):
        subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
            [_require_uv(), "build", "--wheel", "--out-dir", str(destination), str(source)],
            cwd=REPO_ROOT,
            check=True,
        )


def _make_consumer_project(tmp_path: Path, find_links: Path, extra: str | None) -> Path:
    """Create a minimal consumer for plain core or *extra*.

    Returns
    -------
    Path
        Directory containing the consumer project.
    """
    project = tmp_path / f"consumer-{extra or 'core'}"
    project.mkdir()
    core_wheel = next(find_links.glob("pykissembed-*.whl"))
    name = "pykissembed" if extra is None else f"pykissembed[{extra}]"
    requirement = f"{name} @ {core_wheel.resolve().as_uri()}"
    pyproject = textwrap.dedent(
        f"""
        [project]
        name = "consumer-{extra or 'core'}"
        version = "0.0.0"
        requires-python = ">=3.14"
        dependencies = ["{requirement}"]

        [tool.uv]
        find-links = ["{find_links}"]
        """,
    )
    (project / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return project


def _consumer_python(project: Path) -> Path:
    """Return the consumer environment's Python executable.

    Returns
    -------
    Path
        Platform-appropriate environment Python path.
    """
    candidate = project / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    return project / ".venv" / "Scripts" / "python.exe"


def _run_metadata_probe(venv_python: Path) -> tuple[set[str], dict[str, str], set[str]]:
    """Return installed distributions, provider entry points, and active names.

    Returns
    -------
    tuple[set[str], dict[str, str], set[str]]
        Distribution names, entry-point targets, and discovered provider names.
    """
    code = textwrap.dedent(
        """
        from importlib.metadata import distributions, entry_points
        from pykissembed.providers.registry import discover_all

        print("DISTS=" + ",".join(sorted(
            dist.metadata["Name"].lower().replace("_", "-") for dist in distributions()
        )))
        print("EPS=" + ",".join(sorted(
            f"{ep.name}={ep.value}" for ep in entry_points(group="pykissembed.providers")
        )))
        print("ACTIVE=" + ",".join(provider.name for provider in discover_all().all()))
        """,
    )
    result = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [str(venv_python), "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    distributions = set(filter(None, values["DISTS"].split(",")))
    entry_points = dict(
        item.split("=", 1) for item in filter(None, values["EPS"].split(","))
    )
    active = set(filter(None, values["ACTIVE"].split(",")))
    return distributions, entry_points, active


@pytest.mark.slow
@pytest.mark.parametrize("extra", [None, "cloud", "all", "local"])
def test_isolated_install_is_cloud_only(tmp_path: Path, extra: str | None) -> None:
    """Core and every supported extra resolve without the retired ML graph."""
    wheels = tmp_path / "wheels"
    _build_wheels(wheels)
    consumer = _make_consumer_project(tmp_path, wheels, extra)
    subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [_require_uv(), "sync", "--no-dev"],
        cwd=consumer,
        check=True,
    )

    consumer_python = _consumer_python(consumer)
    subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [_require_uv(), "pip", "check", "--python", str(consumer_python)],
        check=True,
    )
    distributions, entry_points, active = _run_metadata_probe(consumer_python)
    forbidden = sorted(
        name
        for name in distributions
        if name in FORBIDDEN_DISTRIBUTIONS or name.startswith(("cuda-", "nvidia-"))
    )
    assert forbidden == []
    assert "local" not in entry_points
    assert "local" not in active
    if extra is None:
        assert entry_points == {}
        assert active == set()
    else:
        assert entry_points == CLOUD_ENTRY_POINTS
        assert active == set(CLOUD_ENTRY_POINTS)
    if extra == "local":
        assert "pykissembed-local" in distributions


def test_extras_declared_in_pyproject() -> None:
    """The transition extra and cloud-only aliases have the intended shape."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert extras["cloud"] == ["pykissembed-cloud"]
    assert extras["all"] == extras["cloud"]
    assert set(extras["local"]) == {"pykissembed-cloud", "pykissembed-local"}


@pytest.mark.slow
def test_built_wheel_metadata_is_cloud_only(tmp_path: Path) -> None:
    """Every wheel advertises only the intended lightweight dependency graph."""
    wheels = tmp_path / "wheels"
    _build_wheels(wheels)
    for wheel in wheels.glob("*.whl"):
        with ZipFile(wheel) as archive:
            metadata_name = next(name for name in archive.namelist() if name.endswith("/METADATA"))
            metadata = Parser().parsestr(archive.read(metadata_name).decode())
            requirements = metadata.get_all("Requires-Dist", [])
            names = {_requirement_name(requirement) for requirement in requirements}
            _assert_no_forbidden(names)
            entry_points = [
                name for name in archive.namelist() if name.endswith("/entry_points.txt")
            ]
        if metadata["Name"] == "pykissembed-local":
            assert requirements == []
            assert entry_points == []


@pytest.mark.slow
def test_tombstone_wheel_installs_standalone(tmp_path: Path) -> None:
    """The final local artifact imports without core or any retired dependency."""
    wheels = tmp_path / "wheels"
    _build_wheels(wheels)
    tombstone = next(wheels.glob("pykissembed_local-*.whl"))
    project = tmp_path / "tombstone-consumer"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "tombstone-consumer"
            version = "0.0.0"
            requires-python = ">=3.14"
            dependencies = ["pykissembed-local @ {tombstone.resolve().as_uri()}"]
            """,
        ),
        encoding="utf-8",
    )
    subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [_require_uv(), "sync", "--no-dev"],
        cwd=project,
        check=True,
    )
    python = _consumer_python(project)
    subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [_require_uv(), "pip", "check", "--python", str(python)],
        check=True,
    )
    probe = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [
            str(python),
            "-c",
            (
                "from importlib.metadata import distributions; import pykissembed_local; "
                "print(','.join(sorted(d.metadata['Name'].lower() for d in distributions())))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    names = set(probe.stdout.strip().split(","))
    # The minimal consumer has no build system, so uv installs only its
    # dependency into the environment rather than the consumer project itself.
    assert names == {"pykissembed-local"}


@pytest.mark.slow
@pytest.mark.parametrize("package_name", ["pykissembed_cloud", "pykissembed_local"])
def test_standalone_package_lock_is_current(tmp_path: Path, package_name: str) -> None:
    """Standalone package locks remain fresh outside root-workspace discovery."""
    source = REPO_ROOT / package_name
    project = tmp_path / package_name
    shutil.copytree(
        source,
        project,
        ignore=shutil.ignore_patterns(".coverage", ".pytest_cache", ".ruff_cache"),
    )
    subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [_require_uv(), "lock", "--check"],
        cwd=project,
        check=True,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-vv", "-m", "slow"]))
