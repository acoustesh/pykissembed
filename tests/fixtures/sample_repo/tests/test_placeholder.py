"""Placeholder test for the sample fixture repo (so pytest collects something)."""

from __future__ import annotations

import importlib


def test_add() -> None:
    """Verify the sample ``add`` function works.

    Raises
    ------
    TypeError
        If the sample function is missing or not callable.
    """
    module = importlib.import_module("example_pkg.core")
    add = getattr(module, "add", None)
    if not callable(add):
        msg = "example_pkg.core.add is missing or not callable"
        raise TypeError(msg)

    assert add(2, 3) == 5
