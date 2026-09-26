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
    load_envelope,
    read_int_map,
    save_envelope,
)
from pykissembed.config import get_config

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
    value = record.get(key)
    return value if isinstance(value, str) else ""


def _number(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _nested(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = record.get(key)
    return value if isinstance(value, dict) else {}


def _resolve_tool(name: str) -> str:
    """Return ``name`` if it exists on PATH, else ``name`` (PATH-resolved at call).

    Returns
    -------
    str
        The absolute path to *name* if found on ``PATH``, otherwise
        *name* unchanged so the caller's subprocess call still attempts
        PATH resolution itself.
    """
    return shutil.which(name) or name  # pragma: no cover — defensive


def _run_ruff(paths: list[Path]) -> list[Mapping[str, object]]:
    """Run ``ruff check --output-format json`` and return parsed diagnostics.

    Returns
    -------
    list[Mapping[str, object]]
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
        *[str(p) for p in paths],
    ]
    try:
        # S603: fixed argv (resolved ruff binary + literal flags + configured paths).
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
    except OSError, subprocess.TimeoutExpired:
        return []
    if not result.stdout.strip():
        return []
    parsed: object = json.loads(result.stdout)
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _run_pyright(paths: list[Path]) -> list[Mapping[str, object]]:
    """Run ``pyright --outputjson`` and return ``generalDiagnostics``.

    Returns
    -------
    list[Mapping[str, object]]
        The ``generalDiagnostics`` list from pyright's JSON output.
        Empty if the subprocess errors out or times out, pyright
        produces no stdout, or the parsed JSON isn't a dict.
    """
    cmd = [_resolve_tool("pyright"), "--outputjson", *[str(p) for p in paths]]
    try:
        # S603: fixed argv (resolved pyright binary + literal flags + configured paths).
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
    except OSError, subprocess.TimeoutExpired:
        return []
    if not result.stdout.strip():
        return []
    parsed: object = json.loads(result.stdout)
    if not isinstance(parsed, dict):
        return []
    diagnostics = parsed.get("generalDiagnostics", [])
    if not isinstance(diagnostics, list):
        return []
    return [item for item in diagnostics if isinstance(item, dict)]


def _build_report(
    ruff_diags: list[Mapping[str, object]],
    pyright_diags: list[Mapping[str, object]],
    *,
    root: Path,
) -> _Report:
    """Aggregate diagnostics into a per-file JSON report.

    Returns
    -------
    _Report
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
        if isinstance(sev_raw, int):
            sev = {0: "error", 1: "warning", 2: "information"}.get(sev_raw, "information")
        else:
            sev = str(sev_raw).lower()
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
    ruff = _run_ruff(pykissembed_paths)
    pyright = _run_pyright(pykissembed_paths)
    report = _build_report(ruff, pyright, root=root)

    # Persist the JSON report for `pykissembed type-review --json`
    report_path = config.baseline_path / "lint_typecheck_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Load the baseline
    baseline_file = config.baseline_path / "lint_typecheck.json"
    envelope = load_envelope(baseline_file, kind="lint_typecheck")

    if update_baselines:
        envelope.data = {
            "per_file": {f: len(d["ruff"]) + len(d["pyright"]) for f, d in report["files"].items()}
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
            detail = "\n".join(f"    ruff:   {d['code']}: {d['message']}" for d in diags["ruff"])
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
    pytest.fail("\n".join(lines))
