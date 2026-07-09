"""Shared fixtures for pykissembed's own tests (and downstream consumers)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pykissembed.baselines_engine import BaselineEnvelope, save_envelope
from pykissembed.config import PyqtestConfig, load_config

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture(scope="session")
def pykissembed_config() -> PyqtestConfig:
    """Resolved ``[tool.pykissembed]`` config."""
    return load_config()


@pytest.fixture
def baseline_factory(tmp_path: Path) -> Callable[..., Path]:
    """Factory for writing a v1 envelope to a temp file."""

    def _make(kind: str, data: dict[str, object], *, name: str = "baseline.json") -> Path:
        path = tmp_path / name
        save_envelope(path, BaselineEnvelope(kind=kind, data=dict(data)))
        return path

    return _make
