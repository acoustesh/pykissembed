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
    """Resolved ``[tool.pykissembed]`` config.

    Returns
    -------
    PyqtestConfig
        Configuration loaded via :func:`load_config` for the current
        working directory.
    """
    return load_config()


@pytest.fixture
def baseline_factory(tmp_path: Path) -> Callable[..., Path]:
    """Factory for writing a v1 envelope to a temp file.

    Returns
    -------
    Callable[..., Path]
        A ``_make(kind, data, *, name="baseline.json")`` callable that
        writes a v1 envelope under ``tmp_path`` and returns its path.
    """

    def _make(kind: str, data: dict[str, object], *, name: str = "baseline.json") -> Path:
        path = tmp_path / name
        save_envelope(path, BaselineEnvelope(kind=kind, data=dict(data)))
        return path

    return _make
