"""Versioned baseline envelope + JSON Schema validation.

Every baseline file written or read by pykissembed is wrapped in a versioned
envelope::

    {
        "schema_version": "1.0",
        "kind": "lint_typecheck",   # or "complexity", "density", etc.
        "data": { ... }              # the actual baseline payload
    }

Validation is performed against ``pykissembed/schemas/baselines.v1.json`` at
load time. Migration from v0 (raw dict) to v1 is automatic — old files
are wrapped in the envelope on first load.
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from jsonschema import Draft7Validator

if TYPE_CHECKING:
    from jsonschema.protocols import Validator

SCHEMA_VERSION = "1.0"
_KIND_TO_FIELD: dict[str, str] = {}  # populated lazily


@functools.cache
def _load_validator() -> Validator:
    """Load and compile the v1 baseline schema (lazy, cached on first use).

    Returns
    -------
    Validator
        Compiled ``Draft7Validator`` for the v1 baseline schema.
    """
    schema_text = (
        resources.files("pykissembed.schemas").joinpath("baselines.v1.json").read_text("utf-8")
    )
    schema = cast("dict[str, Any]", json.loads(schema_text))
    return Draft7Validator(schema)


@dataclass(slots=True)
class BaselineEnvelope:
    """A versioned baseline payload.

    Attributes
    ----------
    kind
        Baseline ``kind`` discriminator (e.g. ``"complexity"``,
        ``"density"``, ``"similarity"``).
    data
        The unwrapped baseline dict. Mutable so callers can update in
        place before ``save_envelope``.
    path
        On-disk path this envelope was loaded from, if any.
    """

    kind: str
    data: dict[str, Any]
    path: Path | None = None


def is_v1_envelope(value: object) -> TypeGuard[dict[str, Any]]:
    """Return ``True`` if *value* is a valid v1 envelope.

    Returns
    -------
    bool
        ``True`` if *value* is a dict with a matching ``schema_version``,
        a string ``kind``, and a dict ``data``; ``False`` otherwise.
    """
    if not isinstance(value, dict):
        return False
    return (
        value.get("schema_version") == SCHEMA_VERSION
        and isinstance(value.get("kind"), str)
        and isinstance(value.get("data"), dict)
    )


def load_envelope(path: Path, kind: str) -> BaselineEnvelope:
    """Load a baseline file as a v1 envelope.

    If *path* exists and contains a v0 (un-enveloped) payload, it is
    migrated to v1 automatically. If *path* does not exist, an empty
    v1 envelope is returned.

    Parameters
    ----------
    path
        File to load.
    kind
        Baseline kind to assign if migrating a v0 file.

    Returns
    -------
    BaselineEnvelope
        Loaded (or freshly-minted) envelope.

    Notes
    -----
    Invalid envelopes raise ``jsonschema.ValidationError`` during schema
    validation.
    """
    if not path.exists():
        return BaselineEnvelope(kind=kind, data={}, path=path)

    with path.open(encoding="utf-8") as f:
        raw = cast("object", json.load(f))

    if is_v1_envelope(raw):
        validator = _load_validator()
        validator.validate(raw)  # raises on error
        return BaselineEnvelope(
            kind=str(raw["kind"]),
            data=dict(raw["data"]),
            path=path,
        )

    # Looks like an envelope (has schema_version / kind / data keys) but
    # failed the discriminator — refuse rather than silently migrating.
    if isinstance(raw, dict) and "schema_version" in raw and "kind" in raw and "data" in raw:
        _load_validator().validate(raw)  # raises ValidationError

    # Migrate v0 → v1
    if not isinstance(raw, dict):
        raw_dict: dict[str, Any] = {}
    else:
        raw_dict = dict(raw)
    envelope = BaselineEnvelope(kind=kind, data=raw_dict, path=path)
    # Write migrated envelope back so the next load is fast
    save_envelope(path, envelope)
    return envelope


def save_envelope(path: Path, envelope: BaselineEnvelope) -> None:
    """Atomically write a v1 envelope to *path*."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": envelope.kind,
        "data": envelope.data,
    }
    validator = _load_validator()
    validator.validate(payload)  # raises on error

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="baseline_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        Path(temp_path).replace(path)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


def ratchet(data: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Lower baselines where *current* is below the baseline.

    Refuses to raise baselines — a ratchet only goes downward. This
    preserves the invariant "current diagnostics ≤ baseline" without
    silently allowing regressions.

    Parameters
    ----------
    data
        Existing baseline (a nested dict). Numeric leaves are compared.
    current
        Currently-observed values (same shape as ``data``).

    Returns
    -------
    dict[str, Any]
        New baseline dict with values lowered where current < baseline.
    """
    result: dict[str, Any] = {}
    for key, baseline_value in data.items():
        current_value = current.get(key)
        if isinstance(baseline_value, dict):
            sub_current = current_value if isinstance(current_value, dict) else {}
            result[key] = ratchet(baseline_value, sub_current)
        elif isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
            # Only ratchet if current is strictly better (lower)
            if current_value < baseline_value:
                result[key] = current_value
            else:
                result[key] = baseline_value
        else:
            # Unknown shape — pass through unchanged
            result[key] = baseline_value
    # Add new keys (observed diagnostics that have no baseline yet)
    result.update({key: current_value for key, current_value in current.items() if key not in data})
    return result
