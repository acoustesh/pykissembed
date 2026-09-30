"""Tests for tolerant JSON parsing of ruff/pyright stdout."""

from __future__ import annotations

import json

import pytest

from pykissembed.checks.lint_typecheck import parse_tool_json


class TestParseToolJson:
    """Behaviour of :func:`parse_tool_json` on noisy tool output."""

    @staticmethod
    def test_clean_document() -> None:
        """A bare JSON document parses unchanged."""
        assert parse_tool_json('{"generalDiagnostics": []}\n') == {"generalDiagnostics": []}

    @staticmethod
    def test_skips_bracketed_warning_before_object() -> None:
        """A ``[tag] warning`` line before pyright's object is ignored."""
        out = '[cuml.accel] Warning: version mismatch\n{\n  "generalDiagnostics": []\n}\n'
        assert parse_tool_json(out) == {"generalDiagnostics": []}

    @staticmethod
    def test_skips_warning_before_list() -> None:
        """A warning line before ruff's list is ignored."""
        assert parse_tool_json("[cuml.accel] Warning: x\n[]\n") == []

    @staticmethod
    def test_no_document_raises() -> None:
        """Output without any JSON document still raises."""
        with pytest.raises(json.JSONDecodeError):
            parse_tool_json("[cuml.accel] Warning: x\n")
