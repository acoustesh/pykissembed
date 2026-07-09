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
from typing import Any, cast

import pytest

from pykissembed.baselines_engine import (
    load_envelope,
    save_envelope,
)
from pykissembed.config import get_config


def _resolve_tool(name: str) -> str:
    """Return ``name`` if it exists on PATH, else ``name`` (PATH-resolved at call)."""
    return shutil.which(name) or name  # pragma: no cover — defensive


def _run_ruff(paths: list[Path]) -> list[dict[str, Any]]:
    """Run ``ruff check --output-format json`` and return parsed diagnostics."""
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
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if not result.stdout.strip():
        return []
    parsed = cast("object", json.loads(result.stdout))
    return list(cast("list[dict[str, Any]]", parsed)) if isinstance(parsed, list) else []


def _run_pyright(paths: list[Path]) -> list[dict[str, Any]]:
    """Run ``pyright --outputjson`` and return ``generalDiagnostics``."""
    cmd = [_resolve_tool("pyright"), "--outputjson", *[str(p) for p in paths]]
    try:
        # S603: fixed argv (resolved pyright binary + literal flags + configured paths).
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, check=False, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if not result.stdout.strip():
        return []
    parsed = cast("object", json.loads(result.stdout))
    if not isinstance(parsed, dict):
        return []
    return list(cast("list[dict[str, Any]]", parsed.get("generalDiagnostics", [])))


def _build_report(
    ruff_diags: list[dict[str, Any]],
    pyright_diags: list[dict[str, Any]],
    *,
    root: Path,
) -> dict[str, Any]:
    """Aggregate diagnostics into a per-file JSON report."""
    files: dict[str, dict[str, list[dict[str, Any]]]] = {}
    ruff_total = 0
    for d in ruff_diags:
        fp = d.get("filename", "")
        try:
            rel = str(Path(fp).resolve().relative_to(root))
        except ValueError:
            rel = str(fp)
        loc = cast("dict[str, Any]", d.get("location", {}))
        files.setdefault(rel, {"ruff": [], "pyright": []})["ruff"].append(
            {
                "code": d.get("code", ""),
                "message": d.get("message", ""),
                "line": loc.get("row", 0),
                "col": loc.get("column", 0),
            },
        )
        ruff_total += 1
    severity_counts = {"error": 0, "warning": 0, "information": 0, "hint": 0}
    for d in pyright_diags:
        fp = d.get("file", "")
        try:
            rel = str(Path(fp).resolve().relative_to(root))
        except ValueError:
            rel = str(fp)
        sev_raw = d.get("severity", "error")
        if isinstance(sev_raw, int):
            sev = {0: "error", 1: "warning", 2: "information"}.get(sev_raw, "information")
        else:
            sev = str(sev_raw).lower()
        rng = cast("dict[str, Any]", d.get("range", {}).get("start", {}))
        files.setdefault(rel, {"ruff": [], "pyright": []})["pyright"].append(
            {
                "code": d.get("rule", ""),
                "message": d.get("message", ""),
                "line": rng.get("line", 0),
                "col": rng.get("character", 0),
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

    per_file_baseline = cast("dict[str, int]", envelope.data.get("per_file", {}))
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
