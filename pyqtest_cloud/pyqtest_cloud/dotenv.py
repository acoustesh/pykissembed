"""Hand-rolled .env loader for the cloud providers.

The cloud providers all need a single environment variable
(``OPENROUTER_API_KEY``). The most natural place for users to put it is
a ``.env`` file at the project root — same convention as python-dotenv,
Next.js, Remix, etc. We avoid pulling in ``python-dotenv`` because the
``.env`` grammar we need is tiny: ``KEY=VALUE`` lines, ``#`` comments,
optional surrounding quotes, no variable expansion.

The loader is **lazy**: ``_load()`` is invoked the first time
``OpenAICompatProvider.is_configured()`` is called, not at module
import. This keeps ``import pyqtest_cloud`` free of filesystem side
effects.

Only the keys that are *missing* from ``os.environ`` are populated — an
explicitly-set environment variable always wins over a ``.env`` line.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_dotenv(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for a ``.env`` file.

    Returns
    -------
    Path | None
        The first ``.env`` file found on the walk, or ``None`` if the
        walk reaches the filesystem root without finding one.
    """
    cursor = (start or Path.cwd()).resolve()
    while True:
        candidate = cursor / ".env"
        if candidate.is_file():
            return candidate
        parent = cursor.parent
        if parent == cursor:
            return None
        cursor = parent


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse the contents of a ``.env`` file.

    Supports the subset of the dotenv grammar we use:

    - ``KEY=value`` lines.
    - Blank lines and ``#`` comments are ignored.
    - Single or double quotes around the value are stripped.
    - Lines that aren't ``KEY=value`` are silently ignored.

    Parameters
    ----------
    text
        The raw file contents.

    Returns
    -------
    dict[str, str]
        Parsed key/value pairs. Order matches the source file.
    """
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        out[key] = value
    return out


def load_into_environ(
    keys: tuple[str, ...],
    *,
    start: Path | None = None,
    override: bool = False,
) -> Path | None:
    """Populate ``os.environ`` with the requested keys from a ``.env`` file.

    Walks up from *start* (default: cwd) looking for a ``.env`` file.
    Only the requested *keys* are imported. By default, existing
    environment variables are NOT overridden — the explicit env wins
    over the file.

    Parameters
    ----------
    keys
        The set of environment-variable names to import. Other keys in
        the file are ignored.
    start
        Directory to start the walk from. Defaults to ``Path.cwd()``.
    override
        If ``True``, replace existing values in ``os.environ`` with the
        file values. Defaults to ``False`` (env wins over file).

    Returns
    -------
    Path | None
        The ``.env`` file that was loaded, or ``None`` if no file was
        found.
    """
    dotenv = find_dotenv(start)
    if dotenv is None:
        return None
    try:
        text = dotenv.read_text(encoding="utf-8")
    except OSError:
        return None
    parsed = parse_dotenv(text)
    for key in keys:
        if key in parsed and (override or key not in os.environ):
            os.environ[key] = parsed[key]
    return dotenv


# Module-level cache: the dotenv file is loaded at most once per process.
_loaded: bool = False


def ensure_loaded(
    keys: tuple[str, ...] = ("OPENROUTER_API_KEY",),
    *,
    start: Path | None = None,
) -> Path | None:
    """Load the ``.env`` file once and cache the result.

    Subsequent calls are no-ops.

    Returns
    -------
    Path | None
        The ``.env`` file that was loaded, or ``None`` if no file was
        found (or if the load has already happened — see the
        module-level cache).
    """
    global _loaded
    if _loaded:
        return None
    _loaded = True
    return load_into_environ(keys, start=start)


def reset_cache() -> None:
    """Reset the module-level load cache. Used by tests."""
    global _loaded
    _loaded = False


__all__ = [
    "ensure_loaded",
    "find_dotenv",
    "load_into_environ",
    "parse_dotenv",
    "reset_cache",
]
