"""Dependency-policy tests for the cloud-only embedding architecture."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "pykissembed_cloud" / "pyproject.toml",
)
LOCKFILES = (
    REPO_ROOT / "uv.lock",
    REPO_ROOT / "pykissembed_cloud" / "uv.lock",
)
PRODUCTION_ROOTS = (
    REPO_ROOT / "pykissembed",
    REPO_ROOT / "pykissembed_cloud" / "pykissembed_cloud",
)
FORBIDDEN_PACKAGES = frozenset(
    {
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
    },
)
FORBIDDEN_PREFIXES = ("cuda-", "nvidia-")


def _canonical_dependency_name(requirement: str) -> str:
    """Return the normalized distribution name at the start of *requirement*.

    Returns
    -------
    str
        Canonical lowercase distribution name.

    Raises
    ------
    ValueError
        If *requirement* does not start with a distribution name.
    """
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    if match is None:  # pragma: no cover - manifests contain valid requirements
        msg = f"Invalid dependency requirement: {requirement!r}"
        raise ValueError(msg)
    return match.group().lower().replace("_", "-")


def _assert_allowed(names: set[str], *, source: Path) -> None:
    """Assert *names* contains no forbidden or neural-runtime distribution."""
    forbidden = sorted(
        name
        for name in names
        if name in FORBIDDEN_PACKAGES or name.startswith(FORBIDDEN_PREFIXES)
    )
    assert not forbidden, f"{source.relative_to(REPO_ROOT)} contains {forbidden!r}"


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_manifests_have_no_forbidden_dependencies(manifest: Path) -> None:
    """Runtime and development dependency declarations stay cloud-only."""
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    project = data.get("project", {})
    requirements = list(project.get("dependencies", []))
    for values in project.get("optional-dependencies", {}).values():
        requirements.extend(values)
    for values in data.get("dependency-groups", {}).values():
        requirements.extend(values)
    names = {_canonical_dependency_name(item) for item in requirements}
    _assert_allowed(names, source=manifest)


@pytest.mark.parametrize("lockfile", LOCKFILES)
def test_locks_have_no_forbidden_distributions(lockfile: Path) -> None:
    """Every committed uv lock is free of the retired ML dependency graph."""
    data = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    names = {str(package["name"]).lower().replace("_", "-") for package in data["package"]}
    _assert_allowed(names, source=lockfile)


def test_production_code_has_no_forbidden_imports() -> None:
    """Runtime modules never import the retired SDK or local-model graph."""
    forbidden_modules = {name.replace("-", "_") for name in FORBIDDEN_PACKAGES}
    violations: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.partition(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.partition(".")[0]]
                else:
                    continue
                for name in names:
                    if name in forbidden_modules:
                        relative = path.relative_to(REPO_ROOT)
                        violations.append(f"{relative}:{node.lineno}: import {name}")
    assert violations == []
