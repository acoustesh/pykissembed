"""Tests for the baseline engine (envelope, schema, ratchet)."""

from __future__ import annotations

from pathlib import Path

import pytest
from jsonschema import ValidationError

from pyqtest.baselines_engine import (
    SCHEMA_VERSION,
    BaselineEnvelope,
    is_v1_envelope,
    load_envelope,
    ratchet,
    save_envelope,
)


class TestEnvelope:
    """Tests for v1 envelope load/save."""

    @staticmethod
    def test_save_then_load(tmp_path: Path) -> None:
        """A freshly-saved envelope should round-trip via load_envelope."""
        path = tmp_path / "complexity.json"
        env = BaselineEnvelope(kind="complexity", data={"cc_threshold": 15})
        save_envelope(path, env)
        loaded = load_envelope(path, kind="complexity")
        assert loaded.kind == "complexity"
        assert loaded.data == {"cc_threshold": 15}
        assert loaded.path == path

    @staticmethod
    def test_migrate_v0(tmp_path: Path) -> None:
        """A bare dict (v0) should be migrated to v1 envelope on load."""
        path = tmp_path / "complexity.json"
        path.write_text('{"cc_threshold": 12}', encoding="utf-8")
        loaded = load_envelope(path, kind="complexity")
        assert loaded.data == {"cc_threshold": 12}
        # The migrated file should now be valid v1
        assert is_v1_envelope(_read_json(path))

    @staticmethod
    def test_invalid_envelope_rejected(tmp_path: Path) -> None:
        """An envelope with a wrong schema_version must be rejected."""
        path = tmp_path / "complexity.json"
        path.write_text(
            '{"schema_version": "9.9", "kind": "complexity", "data": {}}',
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_envelope(path, kind="complexity")

    @staticmethod
    def test_missing_file_returns_empty(tmp_path: Path) -> None:
        """A missing file should produce an empty envelope, not raise."""
        loaded = load_envelope(tmp_path / "missing.json", kind="complexity")
        assert loaded.data == {}
        assert loaded.kind == "complexity"

    @staticmethod
    def test_schema_version_constant() -> None:
        """The SCHEMA_VERSION constant must equal '1.0'."""
        assert SCHEMA_VERSION == "1.0"


class TestRatchet:
    """Tests for the ratchet() function (lower-only baseline adjuster)."""

    @staticmethod
    def test_lower_only() -> None:
        """Current < baseline → lowered to current."""
        data = {"a": {"b": 10}, "c": 5}
        current = {"a": {"b": 7}, "c": 5}
        assert ratchet(data, current) == {"a": {"b": 7}, "c": 5}

    @staticmethod
    def test_refuses_to_raise() -> None:
        """Current > baseline → baseline unchanged."""
        data = {"a": 5}
        current = {"a": 10}
        assert ratchet(data, current) == {"a": 5}

    @staticmethod
    def test_new_keys_kept() -> None:
        """Keys present in current but not in baseline are kept at current value."""
        data: dict[str, object] = {"a": 1}
        current = {"a": 1, "b": 2}
        result = ratchet(data, current)
        assert result == {"a": 1, "b": 2}

    @staticmethod
    def test_nested_lowering() -> None:
        """Nested dicts are ratcheted recursively."""
        data = {"outer": {"x": 5, "y": 10}}
        current = {"outer": {"x": 3, "y": 10}}
        assert ratchet(data, current) == {"outer": {"x": 3, "y": 10}}

    @staticmethod
    def test_passes_through_unknown_shapes() -> None:
        """Non-numeric, non-dict values pass through unchanged."""
        data = {"label": "hello"}
        current: dict[str, object] = {"label": "world"}
        assert ratchet(data, current) == {"label": "hello"}


def _read_json(path: Path) -> object:
    """Helper: read JSON file as object (typed via cast)."""
    import json

    return json.loads(path.read_text(encoding="utf-8"))
