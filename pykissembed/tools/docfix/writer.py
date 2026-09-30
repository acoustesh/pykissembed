"""Verify a transformed file and write it back without risking data.

Nothing is written unless the independent guards pass: executable code,
bytes outside docstrings and comments are unchanged, and ruff's docstring
violation count does not rise. Writes are atomic, keep the file's mode,
encoding and byte-order mark, and are abandoned when the file changed on
disk meanwhile or has uncommitted git changes (unless allowed).
"""

from __future__ import annotations

import codecs
import os
import shutil
import subprocess
import tempfile
import tokenize
from pathlib import Path

from pykissembed.tools.docfix.engine import FileResult, Options, transform
from pykissembed.tools.docfix.source import Ruff, guard, inserted_owners


def _git_dirty(path: Path) -> bool:
    """Return whether *path* has uncommitted changes in its git repository.

    Parameters
    ----------
    path : Path
        File to check.

    Returns
    -------
    bool
        ``True`` for modified or untracked files; ``False`` outside git.
    """
    git = shutil.which("git")
    # Outside git there is no diff to review, so there is nothing to protect.
    if git is None:
        return False
    cmd = [git, "-C", str(path.parent), "status", "--porcelain", "--", path.name]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # A non-zero status means "not a repository": treated like no git at all.
    return result.returncode == 0 and bool(result.stdout.strip())


def write_atomic(path: Path, text: str, *, bom: bool, expected: bytes) -> str:
    """Replace *path* with *text* atomically, keeping its mode.

    Parameters
    ----------
    path : Path
        File to replace.
    text : str
        New content, line endings already in the file's style.
    bom : bool
        Whether to write a UTF-8 byte-order mark.
    expected : bytes
        The bytes read at the start; a mismatch means another writer
        touched the file and the write is abandoned.

    Returns
    -------
    str
        ``""`` on success, otherwise why nothing was written.
    """
    # An editor may have saved the file since it was read; never clobber that.
    if path.read_bytes() != expected:
        return "file changed on disk while it was being processed"
    data = (codecs.BOM_UTF8 if bom else b"") + text.encode("utf-8")
    # Same directory as the target, so the final rename stays on one
    # filesystem and is atomic; readers see the old or the new file only.
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".docfix")
    try:
        with os.fdopen(handle, "wb") as stream:
            _ = stream.write(data)
        shutil.copymode(path, temp)
        _ = Path(temp).replace(path)
    except OSError as exc:
        # The original is untouched until the rename, so cleanup is enough.
        Path(temp).unlink(missing_ok=True)
        return f"write failed: {exc}"
    return ""


def process_file(path: Path, key: str, options: Options, ruff: Ruff) -> FileResult:
    """Transform, verify and (optionally) write one file.

    Parameters
    ----------
    path : Path
        File to process.
    key : str
        Project-relative file key.
    options : Options
        Run options.
    ruff : Ruff
        Ruff runner.

    Returns
    -------
    FileResult
        The outcome; ``error`` explains a skipped or refused file.
    """
    # Bytes, not text: the exact original is needed for the on-disk change
    # check, and decoding it ourselves keeps CRLF line endings intact.
    raw = path.read_bytes()
    bom = raw.startswith(codecs.BOM_UTF8)
    try:
        original = raw.decode("utf-8-sig")
        result = transform(key, original, path.resolve(), options, ruff)
    except UnicodeDecodeError:
        return FileResult(key, "", "", error="not UTF-8")
    except (SyntaxError, tokenize.TokenError, ValueError) as exc:
        return FileResult(key, "", "", error=f"cannot parse: {exc}")
    if result.updated == original:
        return result
    reason = _verify(result, path, ruff)
    if reason:
        # Keep the findings so the report still explains the file.
        return FileResult(key, original, original, result.findings, error=reason)
    if options.write:
        if not options.allow_dirty and _git_dirty(path):
            result.error = "uncommitted changes (use --allow-dirty)"
            return result
        result.error = write_atomic(path, result.updated, bom=bom, expected=raw)
        result.written = not result.error
    return result


def _verify(result: FileResult, path: Path, ruff: Ruff) -> str:
    """Run the code guards and the ruff count check.

    Parameters
    ----------
    result : FileResult
        A transformed file.
    path : Path
        Absolute file path.
    ruff : Ruff
        Ruff runner.

    Returns
    -------
    str
        ``""`` when the result is safe to write, otherwise the reason.
    """
    try:
        inserted = inserted_owners(result.original, result.updated)
    except (SyntaxError, tokenize.TokenError, ValueError) as exc:
        return f"result does not parse: {exc}"
    reason = guard(result.original, result.updated, inserted)
    if reason:
        return reason
    # Ruff may be absent or misconfigured (None); only a proven rise blocks.
    before = ruff.count(result.original, path.resolve())
    after = ruff.count(result.updated, path.resolve())
    if before is not None and after is not None and after > before:
        return f"ruff D violations would rise from {before} to {after}"
    return ""
