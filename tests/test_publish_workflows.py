"""Regression tests for TestPyPI publish workflow paths."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-testpypi-local.yml"


def test_local_publish_workflow_uses_existing_package_directory() -> None:
    """The local publish workflow must reference the real workspace directory."""
    text = LOCAL_WORKFLOW.read_text(encoding="utf-8")
    values = re.findall(r"^\s*(?:working-directory|path|packages-dir):\s*([^\s#]+)", text, re.MULTILINE)

    package_dirs = {Path(value).parts[0] for value in values if value.startswith("pykissembed")}

    assert package_dirs == {"pykissembed_local"}
    assert (REPO_ROOT / "pykissembed_local").is_dir()
