"""Similarity check entry point.

This module defers to ``pykissembed-cloud`` if installed;
otherwise the test skips with a helpful message explaining how to enable
similarity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.similarity
def test_providers_parallel(
    pykissembed_paths: list[Path],
    *,
    cached_only: bool,
) -> None:
    """Run all installed embedding providers in parallel against the codebase."""
    try:
        # Optional: pykissembed-cloud may not be installed.
        from pykissembed.similarity.runner import (  # ruff:ignore[import-outside-top-level]
            run_all_providers,  # type: ignore[attr-defined]
        )
    except ImportError:
        pytest.skip(
            "Similarity requires pykissembed-cloud.\n"
            "  pip install pykissembed-cloud  # cloud providers; API keys required",
        )
    if not pykissembed_paths:
        pytest.skip("No [tool.pykissembed] paths configured")
    run_all_providers(paths=pykissembed_paths, cached_only=cached_only)
