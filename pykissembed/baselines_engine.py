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

import contextlib
import functools
import json
import os
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

from filelock import FileLock
from jsonschema import Draft7Validator

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from jsonschema.protocols import Validator

# Baseline payloads are JSON, so model them as JSON. Mapping/Sequence (not
# dict/list) make the value slot covariant, which is what lets a caller store a
# concrete dict[str, int] of baselines without re-labelling it, and lets the
# whole payload be handed to jsonschema's validate() unchanged.
type JsonValue = str | int | float | bool | Mapping[str, JsonValue] | Sequence[JsonValue] | None

SCHEMA_VERSION = "1.0"
_KIND_TO_FIELD: dict[str, str] = {}  # populated lazily


@functools.cache
def _load_validator() -> Validator:
    """Load and compile the v1 baseline schema (lazy, cached on first use).

    Returns
    -------
    Validator
        Compiled ``Draft7Validator`` for the v1 baseline schema.

    Raises
    ------
    TypeError
        If the packaged schema file does not contain a JSON object.
    """
    schema_text = (
        resources.files("pykissembed.schemas").joinpath("baselines.v1.json").read_text("utf-8")
    )
    schema: object = json.loads(schema_text)
    if not isinstance(schema, dict):
        msg = "baselines.v1.json must contain a JSON object"
        raise TypeError(msg)
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
    data: dict[str, JsonValue]
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
        raw: object = json.load(f)

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
    """Atomically write a v1 envelope to *path*.

    The payload is validated against the v1 schema before writing; a
    sibling temp file is renamed into place so readers never observe a
    partially written file, and the temp file is removed on failure.

    Parameters
    ----------
    path : Path
        Destination file path (parent directories are created).
    envelope : BaselineEnvelope
        Envelope whose ``kind`` and ``data`` are persisted.

    Notes
    -----
    A payload that does not conform to the v1 baseline schema raises
    ``jsonschema.ValidationError`` during validation, before anything is
    written.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": envelope.kind,
        "data": envelope.data,
    }
    validator = _load_validator()
    validator.validate(payload)  # raises ValidationError on error

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="baseline_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            _ = f.write("\n")
        _ = Path(temp_path).replace(path)
    except Exception:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise


@contextlib.contextmanager
def locked_envelope(path: Path, kind: str) -> Iterator[BaselineEnvelope]:
    """Load *path* as a v1 envelope under an exclusive cross-process lock.

    The lock is held for the whole context, not just the load — callers
    that mutate the yielded envelope and conditionally call
    :func:`save_envelope` (e.g. only under ``--update-baselines``) get a
    race-free load-mutate-save cycle even when multiple pytest-xdist
    workers target the same baseline file in one session. Without this,
    each worker's independent load-then-overwrite can clobber another
    worker's write, since :func:`save_envelope` replaces the file's
    ``data`` wholesale rather than merging with what's currently on disk.

    Parameters
    ----------
    path
        File to load (same as :func:`load_envelope`).
    kind
        Baseline kind to assign if migrating a v0 file.

    Yields
    ------
    BaselineEnvelope
        Loaded (or freshly-minted) envelope, safe to mutate and pass to
        :func:`save_envelope` before the context exits.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    with FileLock(str(lock_path), timeout=60):
        yield load_envelope(path, kind=kind)


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


def read_int(data: Mapping[str, object], key: str, default: int) -> int:
    """Read an integer baseline setting, falling back when absent or malformed.

    Parameters
    ----------
    data
        Baseline payload to read from.
    key
        Setting name.
    default
        Value returned when *key* is missing or not a plain integer.

    Returns
    -------
    int
        The stored integer, or *default*.
    """
    value = data.get(key, default)
    # bool is a subclass of int; a True threshold is corruption, not a setting.
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def read_float(data: Mapping[str, object], key: str, default: float) -> float:
    """Read a float baseline setting, falling back when absent or malformed.

    Parameters
    ----------
    data
        Baseline payload to read from.
    key
        Setting name.
    default
        Value returned when *key* is missing or not numeric.

    Returns
    -------
    float
        The stored number as a float, or *default*.
    """
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _numeric_entries(data: Mapping[str, object], key: str) -> Iterator[tuple[str, int | float]]:
    """Yield the well-formed ``(name, number)`` pairs of a baseline sub-map.

    Parameters
    ----------
    data
        Baseline payload to read from.
    key
        Sub-map name.

    Yields
    ------
    tuple[str, int | float]
        Entries with a string name and a non-bool numeric value; everything
        else in the sub-map is skipped, and a missing or non-mapping *key*
        yields nothing.
    """
    raw = data.get(key)
    if not isinstance(raw, dict):
        return
    for entry_key, entry_value in raw.items():
        # bool is a subclass of int; a True baseline is corruption, not data.
        if (
            isinstance(entry_key, str)
            and isinstance(entry_value, (int, float))
            and not isinstance(entry_value, bool)
        ):
            yield entry_key, entry_value


def read_int_map(data: Mapping[str, object], key: str) -> dict[str, int]:
    """Read a ``{str: int}`` baseline sub-map, dropping malformed entries.

    Parameters
    ----------
    data
        Baseline payload to read from.
    key
        Sub-map name.

    Returns
    -------
    dict[str, int]
        Well-formed entries only; ``{}`` when *key* is absent or not a mapping.
    """
    return {key_: value for key_, value in _numeric_entries(data, key) if isinstance(value, int)}


def read_float_map(data: Mapping[str, object], key: str) -> dict[str, float]:
    """Read a ``{str: float}`` baseline sub-map, dropping malformed entries.

    Parameters
    ----------
    data
        Baseline payload to read from.
    key
        Sub-map name.

    Returns
    -------
    dict[str, float]
        Well-formed entries only; ``{}`` when *key* is absent or not a mapping.
    """
    return {key_: float(value) for key_, value in _numeric_entries(data, key)}
