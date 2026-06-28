"""Pytest configuration for pyqtest's own self-tests.

Excludes the sample_repo fixture (which is a fake consumer project, not a
self-test of pyqtest) from collection.
"""

from __future__ import annotations

collect_ignore = ["fixtures/sample_repo"]
