"""NumPy docstring format checks via ruff D rules.

Ported from ``aa-ml/mega-scrapper/tests/test_docstring_format.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest

from pykissembed.baselines_engine import locked_envelope, read_int_map, save_envelope
from pykissembed.checks.lint_typecheck import parse_tool_json
from pykissembed.config import get_config
from pykissembed.paths import include_notebooks

# ruff's exit status for a configuration or usage error (as opposed to
# 1, "violations found").
_RUFF_USAGE_ERROR = 2


@dataclass(frozen=True, slots=True)
class DocstringViolation:
    """A single docstring format violation."""

    file: str
    line: int
    column: int
    code: str
    message: str

    @override
    def __str__(self) -> str:
        """Format the violation as a single report line.

        Returns
        -------
        str
            ``file:line:column code message``.
        """
        return f"{self.file}:{self.line}:{self.column} {self.code} {self.message}"


def _run_ruff(ruff: str, args: list[str], *, preview: bool) -> subprocess.CompletedProcess[str]:
    """Run ``ruff check`` once with *args*.

    Parameters
    ----------
    ruff : str
        Resolved ruff executable.
    args : list[str]
        Arguments after ``ruff check``.
    preview : bool
        Whether to pass ``--preview``.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The completed process, whatever its exit status.
    """
    cmd = [ruff, "check", *(["--preview"] if preview else []), *args]
    # S603: fixed argv (resolved ruff binary + literal flags + a configured
    # directory path); no shell involved.
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


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
        Empty when ruff is not installed.

    Raises
    ------
    RuntimeError
        If ruff runs but rejects its configuration or arguments, so a broken
        setup fails the gate instead of passing it with zero violations.
    """
    ruff = shutil.which("ruff")
    if ruff is None:
        return []
    args = [str(target_dir), "--select=D", "--output-format=json"]
    if not include_notebooks():
        args[:0] = ["--extend-exclude", "*.ipynb"]
    result = _run_ruff(ruff, args, preview=False)
    # A config that selects rules by name (e.g. "line-too-long") is only
    # accepted in preview mode; retry once rather than always enabling
    # preview, which could add preview-only D rules and shift baselines.
    if result.returncode == _RUFF_USAGE_ERROR and "preview" in result.stderr:
        result = _run_ruff(ruff, args, preview=True)
    if result.returncode == _RUFF_USAGE_ERROR:
        msg = f"ruff could not check {target_dir}:\n{result.stderr.strip()}"
        raise RuntimeError(msg)
    if not result.stdout.strip():
        return []
    try:
        parsed_obj: object = parse_tool_json(result.stdout)
    except json.JSONDecodeError:
        return []
    # ruff's JSON schema isn't a stable contract this project controls, so
    # every layer below re-validates shape defensively instead of trusting
    # the parsed structure — an upstream ruff version bump that reshapes a
    # field should degrade to "violation dropped", not crash the check.
    if not isinstance(parsed_obj, list):
        return []
    violations: list[DocstringViolation] = []
    for item in parsed_obj:
        if not isinstance(item, dict):
            continue
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
        row_obj = location_obj.get("row")
        col_obj = location_obj.get("column")
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


# Public re-export for tests of the ruff invocation; see the same pattern in
# pykissembed.jev (extract_symbol_states).
run_ruff_docstring_check = _run_ruff_docstring_check


def _collect_docstring_violations(paths: list[Path]) -> list[DocstringViolation]:
    """Run the docstring check for every configured source path.

    Parameters
    ----------
    paths : list[Path]
        Configured source directories to check.

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

    Parameters
    ----------
    violations : list[DocstringViolation]
        Violations to group.

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

    A file regresses when its violation count exceeds its baseline; files
    with no stored baseline (count 0) are reported as new files instead.

    Parameters
    ----------
    by_file : dict[str, list[DocstringViolation]]
        Violations grouped by relative filename.
    per_file_baseline : dict[str, int]
        Stored per-file violation baselines.

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

    Parameters
    ----------
    by_file : dict[str, list[DocstringViolation]]
        Violations grouped by relative filename.

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

    Parameters
    ----------
    heading : str
        Section heading emitted before the entries.
    entries : list[str]
        Multi-line violation messages; only their first lines are shown.

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

    Parameters
    ----------
    by_file : dict[str, list[DocstringViolation]]
        All violations grouped by relative filename.
    regressions : list[str]
        Formatted messages for files exceeding their baseline.
    new_files : list[str]
        Formatted messages for files with no baseline.

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
        """Fail if any file has more docstring violations than its baseline.

        Parameters
        ----------
        pykissembed_paths : list[Path]
            Configured source directories from the ``pykissembed_paths`` fixture; the test skips
            when empty.
        update_baselines : bool
            When true, write the current measurements to the baseline file and skip instead of
            checking.
        """
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        config = get_config()
        baseline_file = config.baseline_path / "docstring_format.json"
        with locked_envelope(baseline_file, kind="docstring_format") as envelope:
            per_file_baseline = read_int_map(envelope.data, "per_file")
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
