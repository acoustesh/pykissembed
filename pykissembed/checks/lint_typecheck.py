"""Lint + type-check gate (ruff + pyright).

Ported from ``aa-ml/mega-scrapper/tests/test_lint_typecheck.py``. The
behaviour is identical: run ``ruff check`` and ``pyright`` against the
configured paths, produce a JSON report at
``tests/baselines/lint_typecheck_report.json``, and fail the test if any
diagnostic is above the per-file baseline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pytest

from pykissembed.baselines_engine import (
    locked_envelope,
    read_int_map,
    save_envelope,
)
from pykissembed.config import get_config
from pykissembed.paths import include_notebooks

if TYPE_CHECKING:
    from collections.abc import Mapping


class _FileDiagnostics(TypedDict):
    """Both tools' diagnostics for a single file."""

    ruff: list[dict[str, object]]
    pyright: list[dict[str, object]]


class _Report(TypedDict):
    """The per-file report persisted for ``pykissembed type-review --json``."""

    files: dict[str, _FileDiagnostics]
    summary: dict[str, int]


def _text(record: Mapping[str, object], key: str) -> str:
    """Read a string field from a tool's JSON diagnostic.

    Parameters
    ----------
    record
        One diagnostic as parsed from ruff or pyright.
    key
        Field name.

    Returns
    -------
    str
        The field's value, or ``""`` when absent or not a string.
    """
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _number(record: Mapping[str, object], key: str) -> int:
    """Read an integer field from a tool's JSON diagnostic.

    Parameters
    ----------
    record
        One diagnostic as parsed from ruff or pyright.
    key
        Field name.

    Returns
    -------
    int
        The field's value, or ``0`` when absent or not an integer.
    """
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _nested(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    """Read a nested object field from a tool's JSON diagnostic.

    Parameters
    ----------
    record
        One diagnostic as parsed from ruff or pyright.
    key
        Field name.

    Returns
    -------
    Mapping[str, object]
        The nested mapping, or ``{}`` when absent or not an object.
    """
    value = record.get(key)
    return value if isinstance(value, dict) else {}


def _resolve_tool(name: str) -> str:
    """Return the absolute PATH-resolved path of *name*, or *name* itself.

    Parameters
    ----------
    name : str
        Executable name to resolve.

    Returns
    -------
    str
        The absolute path to *name* if found on ``PATH``, otherwise
        *name* unchanged so the caller's subprocess call still attempts
        PATH resolution itself.
    """
    return shutil.which(name) or name  # pragma: no cover — defensive


def run_ruff(paths: list[Path]) -> list[Mapping[str, object]]:
    """Run ``ruff check --output-format json`` and return parsed diagnostics.

    Parameters
    ----------
    paths : list[Path]
        Directories passed to ruff for checking.

    Notebooks (``.ipynb``) are excluded by default because they typically
    contain exploratory code that isn't held to the same hygiene standards
    as production source. Consumers can override this by setting
    ``include_notebooks = true`` in ``[tool.pykissembed]``.

    Returns
    -------
    list[dict[str, Any]]
        The parsed JSON diagnostics from ``ruff check``. Empty if the
        subprocess errors out or times out, ruff produces no stdout, or
        the parsed JSON isn't a list.
    """
    cmd = [
        _resolve_tool("ruff"),
        "check",
        "--preview",
        "--output-format",
        "json",
    ]
    if not include_notebooks():
        cmd.extend(["--extend-exclude", "*.ipynb"])
    cmd.extend([str(p) for p in paths])
    try:
        # S603: fixed argv (resolved ruff binary + literal flags + configured
        # directory paths); no shell involved.
        result = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
    except OSError, subprocess.TimeoutExpired:
        # A missing or hung ruff yields "no diagnostics", which reads as a pass.
        # Deliberate — the gate must not block on a broken toolchain — but it
        # means a green run is only meaningful when ruff actually executed.
        return []
    if not result.stdout.strip():
        return []
    parsed: object = json.loads(result.stdout)
    if not isinstance(parsed, list):
        return []
    # Keep only well-formed entries; a reshaped ruff release degrades to
    # "diagnostic dropped" rather than crashing the gate.
    records: list[Mapping[str, object]] = [item for item in parsed if isinstance(item, dict)]
    return records


def run_pyright(paths: list[Path]) -> list[Mapping[str, object]]:
    """Run ``pyright --outputjson`` and return ``generalDiagnostics``.

    Parameters
    ----------
    paths : list[Path]
        Directories passed to pyright for checking.

    Returns
    -------
    list[dict[str, Any]]
        The ``generalDiagnostics`` list from pyright's JSON output.
        Empty if the subprocess errors out or times out, pyright
        produces no stdout, or the parsed JSON isn't a dict.
    """
    cmd = [_resolve_tool("pyright"), "--outputjson", *[str(p) for p in paths]]
    try:
        # S603: fixed argv (resolved pyright binary + literal flags +
        # configured directory paths); no shell involved.
        result = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
    except OSError, subprocess.TimeoutExpired:
        # Same silent-pass tradeoff as run_ruff above.
        return []
    if not result.stdout.strip():
        return []
    parsed: object = json.loads(result.stdout)
    if not isinstance(parsed, dict):
        return []
    diagnostics = parsed.get("generalDiagnostics", [])
    if not isinstance(diagnostics, list):
        return []
    records: list[Mapping[str, object]] = [
        item for item in diagnostics if isinstance(item, dict)
    ]
    return records


def build_report(
    ruff_diags: list[Mapping[str, object]],
    pyright_diags: list[Mapping[str, object]],
    *,
    root: Path,
) -> _Report:
    """Aggregate diagnostics into a per-file JSON report.

    Parameters
    ----------
    ruff_diags : list[dict[str, Any]]
        Parsed ruff JSON diagnostics.
    pyright_diags : list[dict[str, Any]]
        Parsed pyright ``generalDiagnostics`` entries.
    root : Path
        Project root used to make file paths relative.

    Returns
    -------
    dict[str, Any]
        A dict with two keys: ``"files"`` mapping each file's path
        (relative to *root*, or the raw reported path if it isn't
        under *root*) to its ``"ruff"`` and ``"pyright"`` diagnostic
        lists (files with no diagnostics in either are omitted), and
        ``"summary"`` with aggregate counts (``total_files``,
        ``ruff_errors``, ``pyright_errors``, ``pyright_warnings``,
        ``pyright_information``, ``pyright_hints``, ``total``).
    """
    files: dict[str, _FileDiagnostics] = {}
    ruff_total = 0
    for d in ruff_diags:
        fp = _text(d, "filename")
        try:
            rel = str(Path(fp).resolve().relative_to(root))
        except ValueError:
            # Outside the project root (e.g. a site-packages path): keep it
            # absolute rather than inventing a misleading relative path.
            rel = str(fp)
        loc = _nested(d, "location")
        files.setdefault(rel, {"ruff": [], "pyright": []})["ruff"].append(
            {
                "code": _text(d, "code"),
                "message": _text(d, "message"),
                "line": _number(loc, "row"),
                "col": _number(loc, "column"),
            },
        )
        ruff_total += 1
    severity_counts = {"error": 0, "warning": 0, "information": 0, "hint": 0}
    for d in pyright_diags:
        fp = _text(d, "file")
        try:
            rel = str(Path(fp).resolve().relative_to(root))
        except ValueError:
            rel = str(fp)
        sev_raw = d.get("severity", "error")
        # Pyright reports severity as a string in JSON output but as an LSP
        # integer in some versions; "hint" has no numeric code, so int-form
        # hints land in "information".
        if isinstance(sev_raw, int):
            sev = {0: "error", 1: "warning", 2: "information"}.get(sev_raw, "information")
        else:
            sev = str(sev_raw).lower()
        # Pyright ranges are 0-based; ruff's row/column above are 1-based. The
        # report preserves each tool's own convention, so consumers must add 1
        # to pyright line/col to match what an editor shows.
        rng = _nested(_nested(d, "range"), "start")
        files.setdefault(rel, {"ruff": [], "pyright": []})["pyright"].append(
            {
                "code": _text(d, "rule"),
                "message": _text(d, "message"),
                "line": _number(rng, "line"),
                "col": _number(rng, "character"),
                "severity": sev,
            },
        )
        if sev in severity_counts:
            severity_counts[sev] += 1
    # setdefault can leave a file with both lists empty; drop those so
    # total_files counts only files that actually have diagnostics.
    files = {k: v for k, v in files.items() if v["ruff"] or v["pyright"]}
    pyright_total = sum(severity_counts.values())
    return {
        "files": files,
        "summary": {
            "total_files": len(files),
            "ruff_errors": ruff_total,
            "pyright_errors": severity_counts["error"],
            "pyright_warnings": severity_counts["warning"],
            "pyright_information": severity_counts["information"],
            "pyright_hints": severity_counts["hint"],
            "total": ruff_total + pyright_total,
        },
    }


@pytest.mark.lint
def test_no_lint_or_type_errors(
    pykissembed_paths: list[Path],
    *,
    update_baselines: bool,
) -> None:
    """All configured paths must pass ruff + pyright with zero diagnostics."""
    if not pykissembed_paths:
        pytest.skip("No [tool.pykissembed] paths configured")

    config = get_config()
    root = config.root
    ruff = run_ruff(pykissembed_paths)
    pyright = run_pyright(pykissembed_paths)
    report = build_report(ruff, pyright, root=root)

    # Persist the JSON report for `pykissembed type-review --json`
    report_path = config.baseline_path / "lint_typecheck_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Load the baseline
    baseline_file = config.baseline_path / "lint_typecheck.json"
    with locked_envelope(baseline_file, kind="lint_typecheck") as envelope:
        if update_baselines:
            envelope.data = {
                "per_file": {
                    f: len(d["ruff"]) + len(d["pyright"]) for f, d in report["files"].items()
                }
            }
            save_envelope(baseline_file, envelope)
            pytest.skip("Updated lint/typecheck baselines")

        per_file_baseline = read_int_map(envelope.data, "per_file")
        regressions: list[str] = []
        new_violations: list[str] = []
        for file_path, diags in report["files"].items():
            count = len(diags["ruff"]) + len(diags["pyright"])
            baseline = per_file_baseline.get(file_path, 0)
            if count > baseline:
                detail = "\n".join(
                    f"    ruff:   {d['code']}: {d['message']}" for d in diags["ruff"]
                )
                detail += "\n" + "\n".join(
                    f"    pyright: {d['code']}: {d['message']}" for d in diags["pyright"]
                )
                if baseline == 0:
                    new_violations.append(f"{file_path}: {count} diagnostics (new file)\n{detail}")
                else:
                    regressions.append(
                        f"{file_path}: {count} diagnostics (baseline {baseline}, +{count - baseline})\n{detail}",
                    )

        total = report["summary"]["total"]
        if total == 0:
            return

        if not regressions and not new_violations:
            # Diagnostics exist but are grandfathered by baselines
            return

        lines = [
            f"Lint/type-check gate failed: {total} diagnostic(s) across {report['summary']['total_files']} file(s).",
            f"  ruff errors: {report['summary']['ruff_errors']}",
            f"  pyright errors: {report['summary']['pyright_errors']}",
            f"Report: {report_path}",
        ]
        if regressions:
            lines.append("\n=== Regressions (exceeds baseline) ===")
            lines.extend(regressions)
        if new_violations:
            lines.append("\n=== New violations (no baseline) ===")
            lines.extend(new_violations)
        pytest.fail("\n".join(lines), pytrace=False)
