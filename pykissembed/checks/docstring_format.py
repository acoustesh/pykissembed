"""NumPy docstring format checks via ruff D rules.

Ported from ``aa-ml/mega-scrapper/tests/test_docstring_format.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from pykissembed.baselines_engine import locked_envelope, save_envelope
from pykissembed.config import get_config
from pykissembed.paths import include_notebooks


@dataclass(frozen=True, slots=True)
class DocstringViolation:
    """A single docstring format violation."""

    file: str
    line: int
    column: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.column} {self.code} {self.message}"


def _run_ruff_docstring_check(target_dir: Path, *, root: Path) -> list[DocstringViolation]:
    """Run ``ruff check --select=D --output-format=json`` on *target_dir*.

    Notebooks (``.ipynb``) are excluded by default because they typically
    contain exploratory code that isn't held to the same hygiene standards
    as production source. Consumers can override this by setting
    ``include_notebooks = true`` in ``[tool.pykissembed]``.

    Parameters
    ----------
    target_dir : Path
        Directory to check.
    root : Path
        Project root used to make filenames relative.

    Returns
    -------
    list[DocstringViolation]
        Detected violations, with filenames relative to *root* where possible.
    """
    ruff = shutil.which("ruff")
    if ruff is None:
        return []
    cmd = [ruff, "check"]
    if not include_notebooks():
        cmd.extend(["--extend-exclude", "*.ipynb"])
    cmd.extend([str(target_dir), "--select=D", "--output-format=json"])
    # S603: fixed argv (resolved ruff binary + literal flags + a configured
    # directory path); no shell involved.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if not result.stdout.strip():
        return []
    try:
        parsed_obj = cast("object", json.loads(result.stdout))
    except json.JSONDecodeError:
        return []
    # ruff's JSON schema isn't a stable contract this project controls, so
    # every layer below re-validates shape defensively instead of trusting
    # the parsed structure — an upstream ruff version bump that reshapes a
    # field should degrade to "violation dropped", not crash the check.
    if not isinstance(parsed_obj, list):
        return []
    violations: list[DocstringViolation] = []
    for item_obj in cast("list[object]", parsed_obj):
        if not isinstance(item_obj, dict):
            continue
        item = cast("dict[str, object]", item_obj)
        filename = item.get("filename")
        location_obj = item.get("location")
        code = item.get("code")
        message = item.get("message")
        if (
            not isinstance(filename, str)
            or not isinstance(location_obj, dict)
            or not isinstance(code, str)
            or not isinstance(message, str)
        ):
            continue
        loc = cast("dict[str, object]", location_obj)
        row_obj = loc.get("row")
        col_obj = loc.get("column")
        if not isinstance(row_obj, int) or not isinstance(col_obj, int):
            continue
        try:
            # ruff may report a path outside `root` (e.g. a config-excluded
            # file reached via a symlink), in which case relative_to raises
            # ValueError; fall back to ruff's own filename rather than fail
            # the whole check over one unrelativizable path.
            rel = str(Path(filename).resolve().relative_to(root.resolve()))
        except ValueError:
            rel = filename
        violations.append(
            DocstringViolation(
                file=rel,
                line=row_obj,
                column=col_obj,
                code=code,
                message=message,
            ),
        )
    return violations


def _collect_docstring_violations(paths: list[Path]) -> list[DocstringViolation]:
    """Run the docstring check for every configured source path.

    Returns
    -------
    list[DocstringViolation]
        Violations in configured-path order.
    """
    violations: list[DocstringViolation] = []
    for path in paths:
        violations.extend(_run_ruff_docstring_check(path, root=get_config().root))
    return violations


def _group_violations_by_file(
    violations: list[DocstringViolation],
) -> dict[str, list[DocstringViolation]]:
    """Group docstring violations by their reported relative filename.

    Returns
    -------
    dict[str, list[DocstringViolation]]
        Violations grouped under each relative filename.
    """
    by_file: dict[str, list[DocstringViolation]] = {}
    for violation in violations:
        by_file.setdefault(violation.file, []).append(violation)
    return by_file


def _classify_docstring_violations(
    by_file: dict[str, list[DocstringViolation]],
    per_file_baseline: dict[str, int],
) -> tuple[dict[str, int], list[str], list[str]]:
    """Return current counts, regressions, and new-file violations.

    Returns
    -------
    tuple[dict[str, int], list[str], list[str]]
        Current counts, regression messages, and new-file messages.
    """
    current_counts: dict[str, int] = {}
    regressions: list[str] = []
    new_files: list[str] = []
    for file_path, violations in sorted(by_file.items()):
        count = len(violations)
        current_counts[file_path] = count
        baseline = per_file_baseline.get(file_path, 0)
        if count <= baseline:
            continue
        detail = "\n".join(f"    {violation}" for violation in violations)
        # A baseline of 0 is indistinguishable between "file is new to the
        # check" and "file previously had zero violations"; both need the
        # same actionable advice: fix them before recording a nonzero bound.
        if baseline == 0:
            new_files.append(f"{file_path}: {count} violations (new file)\n{detail}")
        else:
            regressions.append(
                f"{file_path}: {count} violations (baseline {baseline}, +{count - baseline})\n{detail}",
            )
    return current_counts, regressions, new_files


def _count_violations_by_code(
    by_file: dict[str, list[DocstringViolation]],
) -> dict[str, int]:
    """Count docstring violations by ruff diagnostic code.

    Returns
    -------
    dict[str, int]
        The number of violations for each diagnostic code.
    """
    code_counts: dict[str, int] = {}
    for violations in by_file.values():
        for violation in violations:
            code_counts[violation.code] = code_counts.get(violation.code, 0) + 1
    return code_counts


def _violation_headers(heading: str, entries: list[str]) -> list[str]:
    """Return a heading and file-summary lines for a nonempty violation group.

    Returns
    -------
    list[str]
        The heading, summary lines, and trailing blank line, or an empty list.
    """
    if not entries:
        return []
    headers = [entry.split("\n", 1)[0] for entry in entries]
    return [heading, *headers, ""]


def _docstring_failure_message(
    by_file: dict[str, list[DocstringViolation]],
    regressions: list[str],
    new_files: list[str],
) -> str:
    """Format the summary emitted for docstring-format baseline failures.

    Returns
    -------
    str
        The complete failure message.
    """
    total_violations = sum(len(violations) for violations in by_file.values())
    n_files = len(regressions) + len(new_files)
    top_codes = sorted(_count_violations_by_code(by_file).items(), key=lambda item: -item[1])[:5]
    lines = [
        f"Docstring format: {total_violations} violation(s) across {n_files} file(s).",
        "",
        "Top error codes: " + ", ".join(f"{code}={count}" for code, count in top_codes),
        "",
    ]
    lines.extend(_violation_headers("--- Regressions (exceeds baseline) ---", regressions))
    lines.extend(_violation_headers("--- New violations (no baseline) ---", new_files))
    lines.append(
        "Run `ruff check --select=D <file>` for full details, "
        "or `ruff check --fix --select=D <file>` to auto-fix."
    )
    return "\n".join(lines)


class TestDocstringFormat:
    """Tests for NumPy docstring format compliance."""

    @staticmethod
    @pytest.mark.docstring_format
    def test_docstring_format(
        pykissembed_paths: list[Path],
        *,
        update_baselines: bool,
    ) -> None:
        """Fail if any file has more docstring violations than its baseline."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        config = get_config()
        baseline_file = config.baseline_path / "docstring_format.json"
        with locked_envelope(baseline_file, kind="docstring_format") as envelope:
            per_file_baseline = cast("dict[str, int]", envelope.data.get("per_file", {}))
            all_violations = _collect_docstring_violations(pykissembed_paths)
            by_file = _group_violations_by_file(all_violations)
            current_counts, regressions, new_files = _classify_docstring_violations(
                by_file,
                per_file_baseline,
            )

            if update_baselines:
                envelope.data["per_file"] = current_counts
                save_envelope(baseline_file, envelope)
                pytest.skip(f"Updated docstring format baselines: {len(current_counts)} files")
            if regressions or new_files:
                pytest.fail(
                    _docstring_failure_message(by_file, regressions, new_files),
                    pytrace=False,
                )
