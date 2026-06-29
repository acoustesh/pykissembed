"""Comment density checks.

Ported from ``aa-ml/mega-scrapper/tests/test_comment_density.py``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from radon.raw import analyze  # type: ignore[import-untyped]

from pykissembed.baselines_engine import load_envelope, save_envelope
from pykissembed.config import get_config
from pykissembed.paths import iter_py_files as _iter_py_files

DEFAULT_MIN_DENSITY = 5.0
DEFAULT_MAX_DENSITY = 40.0
SMALL_FUNCTION_THRESHOLD = 10  # code lines


@dataclass(frozen=True, slots=True)
class CommentStats:
    """Statistics about comments in a single file."""

    sloc: int
    comments: int
    density_pct: float


def _get_int_attr(obj: object, attr_name: str) -> int:
    value = getattr(obj, attr_name, None)
    if not isinstance(value, int):
        raise TypeError
    return value


def _comment_density_from_source(source: str) -> CommentStats:
    """Compute comment density from source code (radon)."""
    metrics = analyze(source)
    loc = _get_int_attr(metrics, "loc")
    multi = _get_int_attr(metrics, "multi")
    comments = _get_int_attr(metrics, "comments")
    code_lines = loc - multi - comments
    sloc = _get_int_attr(metrics, "sloc")
    density = 0.0 if code_lines <= 0 else 100.0 * (comments / code_lines)
    return CommentStats(sloc=sloc, comments=comments, density_pct=density)


def _code_body_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count lines of code in a function body, excluding the docstring."""
    total = (node.end_lineno or node.lineno) - node.lineno + 1
    code_lines = total - 1
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        ds = body[0]
        code_lines -= (ds.end_lineno or ds.lineno) - ds.lineno + 1
    return max(code_lines, 0)


def _all_functions_short(file_path: Path, max_lines: int = SMALL_FUNCTION_THRESHOLD) -> bool:
    """Return True if every function's code body is shorter than *max_lines*."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return False
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    if not functions:
        return False
    return all(_code_body_lines(n) < max_lines for n in functions)


def _file_stats(file_path: Path) -> CommentStats:
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return CommentStats(sloc=0, comments=0, density_pct=0.0)
    return _comment_density_from_source(source)


class TestCommentDensity:
    """Tests for comment density."""

    @staticmethod
    @pytest.mark.density
    def test_comment_density(
        pykissembed_paths: list[Path],
        update_baselines: bool,
    ) -> None:
        """Fail if any file's density is outside its configured range."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        config = get_config()
        baseline_file = config.baseline_path / "comment_density.json"
        envelope = load_envelope(baseline_file, kind="density")

        min_density = float(envelope.data.get("min_density", DEFAULT_MIN_DENSITY))
        max_density = float(envelope.data.get("max_density", DEFAULT_MAX_DENSITY))
        per_file = cast("dict[str, dict[str, float]]", envelope.data.get("per_file", {}))

        violations: list[str] = []
        current: dict[str, float] = {}
        for base_dir in pykissembed_paths:
            for py_file in _iter_py_files(base_dir):
                rel = py_file.relative_to(config.root)
                key = str(rel)
                stats = _file_stats(py_file)
                density = round(stats.density_pct, 2)
                current[key] = density
                file_baseline = per_file.get(key, {})
                file_min = float(file_baseline.get("min", min_density))
                file_max = float(file_baseline.get("max", max_density))
                if _all_functions_short(py_file):
                    continue
                if density < file_min:
                    violations.append(
                        f"{key}: density={density}% (sloc={stats.sloc}, comments={stats.comments}), below {file_min}%",
                    )
                elif density > file_max:
                    violations.append(
                        f"{key}: density={density}% (sloc={stats.sloc}, comments={stats.comments}), above {file_max}%",
                    )
        if update_baselines:
            for key, density in current.items():
                if density < min_density or density > max_density:
                    per_file[key] = {
                        "min": min(density, min_density),
                        "max": max(density, max_density),
                        "current": density,
                    }
            envelope.data["per_file"] = per_file
            save_envelope(baseline_file, envelope)
            pytest.skip("Updated comment density baselines")
        if violations:
            pytest.fail(
                f"Comment density violations (min {min_density}%, max {max_density}%):\n"
                + "\n".join(violations),
            )

    @staticmethod
    @pytest.mark.density
    def test_aggregate_comment_density(
        pykissembed_paths: list[Path],
        update_baselines: bool,
    ) -> None:
        """Fail if aggregate density falls outside configured range."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        config = get_config()
        baseline_file = config.baseline_path / "comment_density.json"
        envelope = load_envelope(baseline_file, kind="density")
        min_density = float(envelope.data.get("aggregate_min_density", DEFAULT_MIN_DENSITY))
        max_density = float(envelope.data.get("aggregate_max_density", DEFAULT_MAX_DENSITY))
        total_sloc = 0
        total_comments = 0
        for base_dir in pykissembed_paths:
            for py_file in _iter_py_files(base_dir):
                stats = _file_stats(py_file)
                total_sloc += stats.sloc
                total_comments += stats.comments
        denom = total_sloc + total_comments
        aggregate = 0.0 if denom == 0 else 100.0 * (total_comments / denom)
        if update_baselines:
            envelope.data["aggregate_current"] = round(aggregate, 2)
            save_envelope(baseline_file, envelope)
            pytest.skip(f"Updated aggregate density: {aggregate:.2f}%")
        violations: list[str] = []
        if aggregate < min_density:
            violations.append(
                f"Aggregate density {aggregate:.1f}% below {min_density}%",
            )
        if aggregate > max_density:
            violations.append(
                f"Aggregate density {aggregate:.1f}% above {max_density}%",
            )
        if violations:
            pytest.fail("\n".join(violations))
