"""Similarity check entry point.

This module defers to ``pykissembed[local]`` or ``pykissembed-cloud`` if installed;
otherwise the test skips with a helpful message explaining how to enable
similarity.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.similarity
def test_providers_parallel(
    pykissembed_paths: list[Path],
    cached_only: bool,
) -> None:
    """Run all installed embedding providers in parallel against the codebase."""
    try:
        from pykissembed.similarity.runner import run_all_providers  # type: ignore[attr-defined]
    except ImportError:
        pytest.skip(
            "Similarity requires pykissembed-local or pykissembed-cloud.\n"
            "  pip install pykissembed-local         # sentence-transformers (no API key)\n"
            "  pip install pykissembed-cloud         # OpenAI, Voyage, Codestral, Gemini",
        )
    if not pykissembed_paths:
        pytest.skip("No [tool.pykissembed] paths configured")
    run_all_providers(paths=pykissembed_paths, cached_only=cached_only)
