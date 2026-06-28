"""Shared fixtures for pykissembed's own tests (and downstream consumers)."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def pykissembed_config():
    """Resolved ``[tool.pykissembed]`` config."""
    from pykissembed.config import load_config

    return load_config()


@pytest.fixture
def baseline_factory(tmp_path: Path):
    """Factory for writing a v1 envelope to a temp file."""
    from pykissembed.baselines_engine import BaselineEnvelope, save_envelope

    def _make(kind: str, data: dict[str, object], *, name: str = "baseline.json") -> Path:
        path = tmp_path / name
        save_envelope(path, BaselineEnvelope(kind=kind, data=dict(data)))
        return path

    return _make
