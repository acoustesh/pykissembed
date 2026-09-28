"""Dependency-policy tests for cloud embeddings and optional GPU support."""

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


def _assert_allowed(
    names: set[str], *, source: Path, gpu_dependencies: set[str] | None = None
) -> None:
    """Reject retired packages and GPU packages outside the optional GPU graph."""
    gpu_dependencies = gpu_dependencies or set()
    forbidden = sorted(
        name
        for name in names
        if (name in FORBIDDEN_PACKAGES and name != "pandas")
        or (
            (name == "pandas" or name.startswith(FORBIDDEN_PREFIXES))
            and name not in gpu_dependencies
        )
    )
    assert not forbidden, f"{source.relative_to(REPO_ROOT)} contains {forbidden!r}"


def _locked_dependencies(packages: dict[str, dict], roots: list[dict]) -> set[str]:
    """Return distributions reachable from *roots* in a uv lock.

    Returns
    -------
    set[str]
        Distribution names, including requested dependency extras.
    """
    pending = [
        (requirement["name"], extra)
        for requirement in roots
        for extra in (None, *requirement.get("extra", []))
    ]
    visited: set[tuple[str, str | None]] = set()
    while pending:
        name, extra = pending.pop()
        if (name, extra) in visited:
            continue
        visited.add((name, extra))
        package = packages[name]
        dependencies = [*package.get("dependencies", [])]
        if extra is not None:
            dependencies.extend(package.get("optional-dependencies", {}).get(extra, []))
        for dependency in dependencies:
            pending.append((dependency["name"], None))
            pending.extend((dependency["name"], item) for item in dependency.get("extra", []))
    return {name for name, _ in visited}


@pytest.mark.parametrize("manifest", MANIFESTS)
def test_manifests_have_no_forbidden_dependencies(manifest: Path) -> None:
    """Manifests keep retired and native GPU packages out of direct requirements."""
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
    """Lockfiles permit GPU dependencies only through the optional cuML graph."""
    data = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    packages = {
        str(package["name"]).lower().replace("_", "-"): package for package in data["package"]
    }
    gpu_dependencies: set[str] = set()
    if lockfile == LOCKFILES[0]:
        root = packages["pykissembed"]
        optional = root.get("optional-dependencies", {})
        gpu_roots = optional.get("gpu", [])
        if [item["name"] for item in gpu_roots] == ["cuml-cu13"]:
            other_roots = [
                *root.get("dependencies", []),
                *(item for extra, items in optional.items() if extra != "gpu" for item in items),
                *(item for items in root.get("dev-dependencies", {}).values() for item in items),
            ]
            gpu_dependencies = _locked_dependencies(packages, gpu_roots) - _locked_dependencies(
                packages, other_roots
            )
    _assert_allowed(set(packages), source=lockfile, gpu_dependencies=gpu_dependencies)


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
