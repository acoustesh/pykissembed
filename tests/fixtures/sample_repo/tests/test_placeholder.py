"""Placeholder test for the sample fixture repo (so pytest collects something)."""

from __future__ import annotations


def test_add() -> None:
    """Verify the sample ``add`` function works."""
    from example_pkg.core import add

    assert add(2, 3) == 5
