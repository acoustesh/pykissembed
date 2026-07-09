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

from pykissembed.baselines_engine import load_envelope, save_envelope
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


def _run_ruff_docstring_check(
    target_dir: Path, *, root: Path
) -> list[DocstringViolation]:
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
        envelope = load_envelope(baseline_file, kind="docstring_format")
        per_file_baseline = cast("dict[str, int]", envelope.data.get("per_file", {}))

        all_violations: list[DocstringViolation] = []
        for path in pykissembed_paths:
            all_violations.extend(
                _run_ruff_docstring_check(path, root=get_config().root)
            )

        by_file: dict[str, list[DocstringViolation]] = {}
        for v in all_violations:
            by_file.setdefault(v.file, []).append(v)

        current_counts: dict[str, int] = {}
        regressions: list[str] = []
        new_files: list[str] = []
        for file_path, viols in sorted(by_file.items()):
            count = len(viols)
            current_counts[file_path] = count
            baseline = per_file_baseline.get(file_path, 0)
            if count > baseline:
                detail = "\n".join(f"    {v}" for v in viols)
                if baseline == 0:
                    new_files.append(
                        f"{file_path}: {count} violations (new file)\n{detail}"
                    )
                else:
                    regressions.append(
                        f"{file_path}: {count} violations (baseline {baseline}, +{count - baseline})\n{detail}",
                    )

        if update_baselines:
            envelope.data["per_file"] = current_counts
            save_envelope(baseline_file, envelope)
            pytest.skip(
                f"Updated docstring format baselines: {len(current_counts)} files"
            )
        if regressions or new_files:
            total_violations = sum(len(v) for v in by_file.values())
            n_files = len(regressions) + len(new_files)
            # Count violations by error code for the summary.
            code_counts: dict[str, int] = {}
            for viols in by_file.values():
                for v in viols:
                    code_counts[v.code] = code_counts.get(v.code, 0) + 1
            top_codes = sorted(code_counts.items(), key=lambda kv: -kv[1])[:5]
            lines = [
                f"Docstring format: {total_violations} violation(s) across {n_files} file(s).",
                "",
                "Top error codes: " + ", ".join(f"{code}={n}" for code, n in top_codes),
                "",
            ]
            if regressions:
                lines.append("--- Regressions (exceeds baseline) ---")
                for entry in regressions:
                    # Show only the file summary line, not every violation.
                    header = entry.split("\n", 1)[0]
                    lines.append(header)
                lines.append("")
            if new_files:
                lines.append("--- New violations (no baseline) ---")
                for entry in new_files:
                    header = entry.split("\n", 1)[0]
                    lines.append(header)
                lines.append("")
            lines.append(
                "Run `ruff check --select=D <file>` for full details, "
                "or `ruff check --fix --select=D <file>` to auto-fix."
            )
            pytest.fail("\n".join(lines), pytrace=False)
