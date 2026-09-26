"""Code complexity metrics (CC, COG, MI, line counts, docstring coverage).

Ported from ``aa-ml/mega-scrapper/tests/test_code_complexity.py``. The
behaviour is preserved; only the discovery paths and baseline storage
are now driven by ``[tool.pykissembed]``.
"""

from __future__ import annotations

import ast
import importlib
from typing import TYPE_CHECKING

import pytest

from pykissembed.baselines_engine import (
    BaselineEnvelope,
    load_envelope,
    read_float,
    read_float_map,
    read_int,
    read_int_map,
    save_envelope,
)
from pykissembed.config import get_config
from pykissembed.similarity.complexity import AnalyzerError, call_analyzer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_callable(module_name: str, attribute: str) -> Callable[..., object] | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    value = getattr(module, attribute, None)
    return value if callable(value) else None


def _iter_py_files(base_dir: Path) -> Iterator[Path]:
    """Yield every ``.py`` file under *base_dir* (recursive).

    Yields
    ------
    Path
        A Python file below *base_dir*.
    """
    for py_file in sorted(base_dir.rglob("*.py")):
        if py_file.name.startswith("__") or "__pycache__" in py_file.parts:
            continue
        yield py_file


def _extract_items_with_docstrings(
    file_path: Path,
) -> list[tuple[str, int, bool, str]]:
    """Return ``[(name, line_no, has_docstring, kind), ...]`` for a file.

    Returns
    -------
    list[tuple[str, int, bool, str]]
        One tuple per top-level and nested function/class definition
        (``@overload``-decorated functions excluded), giving its name,
        line number, whether it has a docstring, and its kind
        (``"class"`` or ``"function"``). Empty if the file fails to
        parse as valid Python.
    """
    results: list[tuple[str, int, bool, str]] = []
    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return results
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Skip @overload stubs
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = [
                    d.id
                    if isinstance(d, ast.Name)
                    else d.attr
                    if isinstance(d, ast.Attribute)
                    else ""
                    for d in node.decorator_list
                ]
                if "overload" in decorators:
                    continue
            has_docstring = ast.get_docstring(node) is not None
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            results.append((node.name, node.lineno, has_docstring, kind))
    return results


def _get_line_count(file_path: Path) -> int:
    with file_path.open(encoding="utf-8") as f:
        return len(f.readlines())


def _get_cc(file_path: Path) -> list[tuple[str, int, int]]:
    """Return ``[(name, lineno, cc), ...]`` using radon.

    Returns
    -------
    list[tuple[str, int, int]]
        One tuple per block radon reports, giving its name, line
        number, and cyclomatic complexity. Empty if radon isn't
        installed or the file fails to parse.
    """
    cc_visit_fn = _load_callable("radon.complexity", "cc_visit")
    if cc_visit_fn is None:
        return []
    source = file_path.read_text(encoding="utf-8")
    try:
        raw_blocks = cc_visit_fn(source)
    except SyntaxError:
        return []
    if not isinstance(raw_blocks, list):
        return []
    out: list[tuple[str, int, int]] = []
    for block in raw_blocks:
        name = getattr(block, "name", None)
        lineno = getattr(block, "lineno", None)
        complexity = getattr(block, "complexity", None)
        if isinstance(name, str) and isinstance(lineno, int) and isinstance(complexity, int):
            out.append((name, lineno, complexity))
    return out


def _get_cog(file_path: Path) -> list[tuple[str, int, int]]:
    """Return ``[(name, lineno, cognitive_complexity), ...]`` using complexipy.

    Returns
    -------
    list[tuple[str, int, int]]
        One tuple per function complexipy reports, giving its name,
        starting line number, and cognitive complexity. Empty if
        complexipy isn't installed, its analysis raises, or the result
        has no ``functions`` attribute.
    """
    file_complexity = _load_callable("complexipy", "file_complexity")
    if file_complexity is None:
        return []
    try:
        result = call_analyzer(file_complexity, str(file_path))
    except AnalyzerError:
        return []
    functions = getattr(result, "functions", None)
    if not isinstance(functions, list):
        return []
    output: list[tuple[str, int, int]] = []
    for function in functions:
        name = getattr(function, "name", None)
        line_start = getattr(function, "line_start", None)
        complexity = getattr(function, "complexity", None)
        if (
            isinstance(name, str)
            and isinstance(line_start, int)
            and isinstance(complexity, int)
        ):
            output.append((name, line_start, complexity))
    return output


def _get_mi(file_path: Path) -> float:
    """Return the Maintainability Index for *file_path* using radon.

    Returns
    -------
    float
        The Maintainability Index score, or ``0.0`` if radon isn't
        installed, radon's analysis raises, or the returned score
        isn't numeric.
    """
    mi_visit_fn = _load_callable("radon.metrics", "mi_visit")
    if mi_visit_fn is None:
        return 0.0
    source = file_path.read_text(encoding="utf-8")
    try:
        score = call_analyzer(mi_visit_fn, source, multi=False)
    except AnalyzerError:
        return 0.0
    if isinstance(score, (int, float)):
        return float(score)
    return 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _load_envelope() -> tuple[Path, BaselineEnvelope]:
    config = get_config()
    path = config.baseline_path / "complexity.json"
    return path, load_envelope(path, kind="complexity")


class TestDocstringCoverage:
    """Tests for docstring presence in all configured directories."""

    @staticmethod
    @pytest.mark.complexity
    def test_docstring_coverage(pykissembed_paths: list[Path], *, update_baselines: bool) -> None:
        """Fail if any directory has more missing docstrings than its baseline."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        baseline_file, envelope = _load_envelope()
        max_missing_default = read_int(envelope.data, "max_missing_docstrings", 0)
        per_dir_baseline = read_int_map(envelope.data, "per_dir")

        all_missing: dict[str, list[str]] = {}
        violations: list[str] = []
        for base_dir in pykissembed_paths:
            rel_dir = str(base_dir.relative_to(get_config().root))
            missing: list[str] = []
            for py_file in _iter_py_files(base_dir):
                items = _extract_items_with_docstrings(py_file)
                for name, line_no, has_docstring, kind in items:
                    if not has_docstring:
                        rel = py_file.relative_to(base_dir)
                        missing.append(f"  - {kind} {name} ({rel}:{line_no})")
            if missing:
                all_missing[rel_dir] = missing
                baseline = per_dir_baseline.get(rel_dir, max_missing_default)
                if len(missing) > baseline:
                    violations.append(
                        f"{rel_dir}/: {len(missing)} missing docstrings (baseline {baseline})",
                    )

        if update_baselines:
            envelope.data["per_dir"] = {d: len(v) for d, v in all_missing.items()}
            save_envelope(baseline_file, envelope)
            pytest.skip("Updated docstring coverage baselines")

        if violations:
            detail = "\n".join(
                f"{d}/:\n" + "\n".join(items) for d, items in sorted(all_missing.items())
            )
            pytest.fail(
                "Docstring coverage regression:\n" + "\n".join(violations) + "\n\n" + detail
            )


class TestLineCount:
    """Tests for file line count limits."""

    @staticmethod
    @pytest.mark.complexity
    def test_file_line_counts(pykissembed_paths: list[Path], *, update_baselines: bool) -> None:
        """Fail if any file exceeds its line-count baseline."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        baseline_file, envelope = _load_envelope()
        line_baselines = read_int_map(envelope.data, "line_baselines")
        violations: list[str] = []
        current_counts: dict[str, int] = {}
        for base_dir in pykissembed_paths:
            for py_file in _iter_py_files(base_dir):
                rel = py_file.relative_to(get_config().root)
                key = str(rel)
                count = _get_line_count(py_file)
                current_counts[key] = count
                baseline = line_baselines.get(key, 0) or None
                if baseline is not None and count > baseline:
                    violations.append(f"File {key} has {count} lines, exceeds baseline {baseline}")
        if update_baselines:
            envelope.data["line_baselines"] = current_counts
            save_envelope(baseline_file, envelope)
            pytest.skip("Updated line count baselines")
        if violations:
            pytest.fail("Line count violations:\n" + "\n".join(violations))


class TestCyclomaticComplexity:
    """Tests for cyclomatic complexity limits."""

    @staticmethod
    @pytest.mark.complexity
    def test_cyclomatic_complexity(
        pykissembed_paths: list[Path],
        *,
        update_baselines: bool,
    ) -> None:
        """Fail if any function exceeds the CC threshold or its baseline."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        baseline_file, envelope = _load_envelope()
        threshold = read_int(envelope.data, "cc_threshold", 15)
        cc_baselines = read_int_map(envelope.data, "cc_baselines")
        violations: list[str] = []
        current_cc: dict[str, int] = {}
        for base_dir in pykissembed_paths:
            for py_file in _iter_py_files(base_dir):
                rel = py_file.relative_to(get_config().root)
                for name, lineno, cc in _get_cc(py_file):
                    key = f"{rel}:{name}"
                    current_cc[key] = cc
                    func_baseline = cc_baselines.get(key, threshold)
                    if cc > func_baseline:
                        violations.append(
                            f"{rel}:{lineno} - {name}() CC={cc}, exceeds {func_baseline}",
                        )
        if update_baselines:
            for key, cc in current_cc.items():
                if cc > threshold:
                    cc_baselines[key] = cc
            envelope.data["cc_baselines"] = cc_baselines
            save_envelope(baseline_file, envelope)
            pytest.skip("Updated CC baselines")
        if violations:
            pytest.fail(
                f"Cyclomatic complexity violations (threshold {threshold}):\n"
                + "\n".join(violations),
            )


class TestCognitiveComplexity:
    """Tests for cognitive complexity limits (complexipy)."""

    @staticmethod
    @pytest.mark.complexity
    def test_cognitive_complexity(
        pykissembed_paths: list[Path],
        *,
        update_baselines: bool,
    ) -> None:
        """Fail if any function exceeds the COG threshold or its baseline."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        baseline_file, envelope = _load_envelope()
        threshold = read_int(envelope.data, "cog_threshold", 15)
        cog_baselines = read_int_map(envelope.data, "cog_baselines")
        violations: list[str] = []
        current_cog: dict[str, int] = {}
        for base_dir in pykissembed_paths:
            for py_file in _iter_py_files(base_dir):
                rel = py_file.relative_to(get_config().root)
                for name, lineno, cog in _get_cog(py_file):
                    key = f"{rel}:{name}"
                    current_cog[key] = cog
                    func_baseline = cog_baselines.get(key, threshold)
                    if cog > func_baseline:
                        violations.append(
                            f"{rel}:{lineno} - {name}() cognitive={cog}, exceeds {func_baseline}",
                        )
        if update_baselines:
            for key, cog in current_cog.items():
                if cog > threshold:
                    cog_baselines[key] = cog
            envelope.data["cog_baselines"] = cog_baselines
            save_envelope(baseline_file, envelope)
            pytest.skip("Updated COG baselines")
        if violations:
            pytest.fail(
                f"Cognitive complexity violations (threshold {threshold}):\n"
                + "\n".join(violations),
            )


class TestMaintainabilityIndex:
    """Tests for Maintainability Index (radon)."""

    @staticmethod
    @pytest.mark.complexity
    def test_maintainability_index(
        pykissembed_paths: list[Path],
        *,
        update_baselines: bool,
    ) -> None:
        """Fail if any file's MI drops below threshold or its baseline."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        baseline_file, envelope = _load_envelope()
        threshold = read_float(envelope.data, "mi_threshold", 13.0)
        mi_baselines = read_float_map(envelope.data, "mi_baselines")
        violations: list[str] = []
        current_mi: dict[str, float] = {}
        for base_dir in pykissembed_paths:
            for py_file in _iter_py_files(base_dir):
                rel = py_file.relative_to(get_config().root)
                key = str(rel)
                mi = round(_get_mi(py_file), 2)
                current_mi[key] = mi
                file_baseline = mi_baselines.get(key, threshold)
                if mi < file_baseline:
                    violations.append(
                        f"{key} MI={mi:.2f}, below {file_baseline}",
                    )
        if update_baselines:
            for key, mi in current_mi.items():
                if mi < threshold:
                    mi_baselines[key] = mi
            envelope.data["mi_baselines"] = mi_baselines
            save_envelope(baseline_file, envelope)
            pytest.skip("Updated MI baselines")
        if violations:
            pytest.fail(
                f"MI violations (threshold {threshold}):\n" + "\n".join(violations),
            )
