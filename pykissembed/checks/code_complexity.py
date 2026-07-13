"""Code complexity metrics (CC, COG, MI, line counts, docstring coverage).

Ported from ``aa-ml/mega-scrapper/tests/test_code_complexity.py``. The
behaviour is preserved; only the discovery paths and baseline storage
are now driven by ``[tool.pykissembed]``.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
from typing import TYPE_CHECKING, TypedDict, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

from pykissembed.baselines_engine import BaselineEnvelope, locked_envelope, save_envelope
from pykissembed.config import get_config
from pykissembed.paths import iter_py_files as _iter_py_files
from pykissembed.paths import warn_non_utf8
from pykissembed.wrapper_analysis import find_wrapper_candidates, parse_source_files


class _BaselineConfig(TypedDict, total=False):
    """Subset of ``[tool.pykissembed]`` that the complexity check uses."""

    cc_threshold: int
    cog_threshold: int
    mi_threshold: float
    max_missing_docstrings: int


class _DefaultConfig(TypedDict):
    """Concrete default for every ``_BaselineConfig`` key — always fully populated."""

    cc_threshold: int
    cog_threshold: int
    mi_threshold: float
    max_missing_docstrings: int


# Merged into every loaded envelope (see `_load_envelope`) so these keys are
# always present and get persisted on `--update-baselines` — mirroring
# `similarity/storage.py`'s `_DEFAULT_CONFIG`, this makes the thresholds
# self-documenting in `tests/baselines/complexity.json` instead of only
# living as literals a consumer has to already know to override.
_DEFAULT_CONFIG: _DefaultConfig = {
    "cc_threshold": 15,
    "cog_threshold": 15,
    "mi_threshold": 13.0,
    "max_missing_docstrings": 0,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# radon/complexipy ship no type stubs, so their entry points are loaded
# through this single dynamic boundary (import_module + getattr + cast)
# instead of a top-level `from radon.complexity import cc_visit` — that
# keeps the untyped surface confined to one place instead of leaking
# `Any` through every call site.
def _load_callable(module_name: str, attribute: str) -> Callable[..., object] | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    value = getattr(module, attribute, None)
    if not callable(value):
        return None
    return cast("Callable[..., object]", value)


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
        (``"class"`` or ``"function"``). Empty if the file can't be read
        as UTF-8 or fails to parse as valid Python.
    """
    results: list[tuple[str, int, bool, str]] = []
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        warn_non_utf8(file_path, exc)
        return results
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError:
        return results
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        # Skip @overload stubs: their body is a bare `...` by convention,
        # so flagging them as "missing docstring" would be a false
        # positive against every overloaded signature in the file.
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_overload_stub(node):
            continue
        has_docstring = ast.get_docstring(node) is not None
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        results.append((node.name, node.lineno, has_docstring, kind))
    return results


def _is_overload_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether *node* has a bare or dotted ``overload`` decorator.

    Returns
    -------
    bool
        Whether the function is an overload stub.
    """
    return any(_decorator_tail(decorator) == "overload" for decorator in node.decorator_list)


def _decorator_tail(decorator: ast.expr) -> str | None:
    """Return the terminal name of a name or attribute decorator.

    Returns
    -------
    str | None
        The terminal name, or ``None`` for another decorator expression.
    """
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


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
        installed, the file can't be read as UTF-8, or the file fails
        to parse.
    """
    cc_visit_fn = _load_callable("radon.complexity", "cc_visit")
    if cc_visit_fn is None:
        return []
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        warn_non_utf8(file_path, exc)
        return []
    try:
        raw_blocks_raw = cc_visit_fn(source)
    except SyntaxError:
        return []
    if not isinstance(raw_blocks_raw, list):
        return []
    raw_blocks = cast("list[object]", raw_blocks_raw)
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
    try:
        # Lazy: complexipy is a compiled (Rust) analyzer; deferring the
        # import avoids its cost for test runs that never touch cognitive
        # complexity (e.g. -m complexity without the COG check selected).
        from complexipy import (  # noqa: PLC0415
            file_complexity as _fc,  # type: ignore[import-untyped]
        )
    except ImportError:
        return []
    try:
        result = _fc(str(file_path))
    except Exception:  # noqa: BLE001 — pragma: no cover — third-party analyzer; any failure on arbitrary user source degrades to "no cognitive-complexity data" rather than crashing the whole check
        return []
    if not hasattr(result, "functions"):
        return []
    fn_list = cast("list[object]", result.functions)
    out: list[tuple[str, int, int]] = []
    for f in fn_list:
        name = getattr(f, "name", None)
        line = getattr(f, "line_start", None)
        complexity = getattr(f, "complexity", None)
        if isinstance(name, str) and isinstance(line, int) and isinstance(complexity, int):
            out.append((name, line, complexity))
    return out


def _get_mi(file_path: Path) -> float:
    """Return the Maintainability Index for *file_path* using radon.

    Returns
    -------
    float
        The Maintainability Index score, or ``0.0`` if radon isn't
        installed, the file can't be read as UTF-8, radon's analysis
        raises, or the returned score isn't numeric.
    """
    mi_visit_fn = _load_callable("radon.metrics", "mi_visit")
    if mi_visit_fn is None:
        return 0.0
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        warn_non_utf8(file_path, exc)
        return 0.0
    try:
        score = mi_visit_fn(source, multi=False)
    except Exception:  # noqa: BLE001 — third-party analyzer; any failure on arbitrary user source degrades to "no MI data" rather than crashing the whole check
        return 0.0
    if isinstance(score, (int, float)):
        return float(score)
    return 0.0


def _collect_metric_results(
    relative_path: Path,
    metrics: list[tuple[str, int, int]],
    *,
    baselines: dict[str, int],
    threshold: int,
    label: str,
) -> tuple[dict[str, int], list[str]]:
    """Collect one file's metric values and threshold violations.

    Returns
    -------
    tuple[dict[str, int], list[str]]
        Current metric values keyed by function and rendered violations.
    """
    current: dict[str, int] = {}
    violations: list[str] = []
    for name, lineno, value in metrics:
        key = f"{relative_path}:{name}"
        current[key] = value
        baseline = baselines.get(key, threshold)
        if value > baseline:
            violations.append(
                f"{relative_path}:{lineno} - {name}() {label}={value}, exceeds {baseline}",
            )
    return current, violations


def _record_excess_metric_baselines(
    baselines: dict[str, int],
    current: dict[str, int],
    *,
    threshold: int,
) -> None:
    """Record values above the default threshold in an update baseline."""
    baselines.update({key: value for key, value in current.items() if value > threshold})


def _complexity_failure_message(
    cc_violations: list[str],
    cog_violations: list[str],
    *,
    cc_threshold: int,
    cog_threshold: int,
) -> str:
    """Format cyclomatic and cognitive complexity violations.

    Returns
    -------
    str
        The failure message containing every nonempty violation section.
    """
    sections: list[str] = []
    if cc_violations:
        sections.append(
            f"Cyclomatic complexity violations (threshold {cc_threshold}):\n"
            + "\n".join(cc_violations),
        )
    if cog_violations:
        sections.append(
            f"Cognitive complexity violations (threshold {cog_threshold}):\n"
            + "\n".join(cog_violations),
        )
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _locked_envelope() -> Iterator[tuple[Path, BaselineEnvelope]]:
    """Load ``complexity.json`` under a cross-process lock, defaults merged in.

    Held for the whole ``with`` block so a test's compute-then-maybe-save
    cycle is atomic even when multiple pytest-xdist workers write to this
    same file in one session (see :func:`pykissembed.baselines_engine.locked_envelope`).

    Yields
    ------
    tuple[Path, BaselineEnvelope]
        The complexity baseline path and its locked, default-populated envelope.
    """
    config = get_config()
    path = config.baseline_path / "complexity.json"
    with locked_envelope(path, kind="complexity") as envelope:
        for key, default in _DEFAULT_CONFIG.items():
            envelope.data.setdefault(key, default)
        yield path, envelope


class TestDocstringCoverage:
    """Tests for docstring presence in all configured directories."""

    @staticmethod
    @pytest.mark.complexity
    def test_docstring_coverage(pykissembed_paths: list[Path], *, update_baselines: bool) -> None:
        """Fail if any directory has more missing docstrings than its baseline."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        with _locked_envelope() as (baseline_file, envelope):
            config_data = cast("_BaselineConfig", envelope.data)
            max_missing_default = config_data.get(
                "max_missing_docstrings", _DEFAULT_CONFIG["max_missing_docstrings"]
            )
            per_dir_baseline = cast("dict[str, int]", envelope.data.get("per_dir", {}))

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
                    "Docstring coverage regression:\n" + "\n".join(violations) + "\n\n" + detail,
                    pytrace=False,
                )


class TestLineCount:
    """Tests for file line count limits."""

    @staticmethod
    @pytest.mark.complexity
    def test_file_line_counts(pykissembed_paths: list[Path], *, update_baselines: bool) -> None:
        """Fail if any file exceeds its line-count baseline."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        with _locked_envelope() as (baseline_file, envelope):
            line_baselines = cast("dict[str, int]", envelope.data.get("line_baselines", {}))
            violations: list[str] = []
            current_counts: dict[str, int] = {}
            for base_dir in pykissembed_paths:
                for py_file in _iter_py_files(base_dir):
                    rel = py_file.relative_to(get_config().root)
                    key = str(rel)
                    count = _get_line_count(py_file)
                    current_counts[key] = count
                    # `or None`: a stored baseline of exactly 0 means "no
                    # baseline recorded yet for this key" (the ratchet only
                    # ever writes positive line counts), not "must have zero
                    # lines" — so it's treated the same as a missing key.
                    baseline = line_baselines.get(key, 0) or None
                    if baseline is not None and count > baseline:
                        violations.append(
                            f"File {key} has {count} lines, exceeds baseline {baseline}"
                        )
            if update_baselines:
                envelope.data["line_baselines"] = current_counts
                save_envelope(baseline_file, envelope)
                pytest.skip("Updated line count baselines")
            if violations:
                pytest.fail("Line count violations:\n" + "\n".join(violations), pytrace=False)


class TestCyclomaticComplexity:
    """Tests for cyclomatic and cognitive complexity limits."""

    @staticmethod
    @pytest.mark.complexity
    def test_cyclomatic_complexity(
        pykissembed_paths: list[Path],
        *,
        update_baselines: bool,
    ) -> None:
        """Fail if any function exceeds its CC or COG threshold or baseline."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        with _locked_envelope() as (baseline_file, envelope):
            config_data = cast("_BaselineConfig", envelope.data)
            cc_threshold = config_data.get(
                "cc_threshold", _DEFAULT_CONFIG["cc_threshold"]
            )
            cog_threshold = config_data.get(
                "cog_threshold", _DEFAULT_CONFIG["cog_threshold"]
            )
            cc_baselines = cast("dict[str, int]", envelope.data.get("cc_baselines", {}))
            cog_baselines = cast("dict[str, int]", envelope.data.get("cog_baselines", {}))
            cc_violations: list[str] = []
            cog_violations: list[str] = []
            current_cc: dict[str, int] = {}
            current_cog: dict[str, int] = {}
            for base_dir in pykissembed_paths:
                for py_file in _iter_py_files(base_dir):
                    rel = py_file.relative_to(get_config().root)
                    cc_current, cc_found = _collect_metric_results(
                        rel,
                        _get_cc(py_file),
                        baselines=cc_baselines,
                        threshold=cc_threshold,
                        label="CC",
                    )
                    current_cc.update(cc_current)
                    cc_violations.extend(cc_found)
                    cog_current, cog_found = _collect_metric_results(
                        rel,
                        _get_cog(py_file),
                        baselines=cog_baselines,
                        threshold=cog_threshold,
                        label="cognitive",
                    )
                    current_cog.update(cog_current)
                    cog_violations.extend(cog_found)
            if update_baselines:
                _record_excess_metric_baselines(
                    cc_baselines,
                    current_cc,
                    threshold=cc_threshold,
                )
                _record_excess_metric_baselines(
                    cog_baselines,
                    current_cog,
                    threshold=cog_threshold,
                )
                envelope.data["cc_baselines"] = cc_baselines
                envelope.data["cog_baselines"] = cog_baselines
                save_envelope(baseline_file, envelope)
                pytest.skip("Updated cyclomatic and cognitive complexity baselines")
            if cc_violations or cog_violations:
                pytest.fail(
                    _complexity_failure_message(
                        cc_violations,
                        cog_violations,
                        cc_threshold=cc_threshold,
                        cog_threshold=cog_threshold,
                    ),
                    pytrace=False,
                )


class TestWrapperProliferation:
    """Tests that exact pass-through wrappers have more than trivial use."""

    @staticmethod
    @pytest.mark.complexity
    def test_wrapper_proliferation(pykissembed_paths: list[Path]) -> None:
        """Fail when an exact pass-through wrapper has too few call sites."""
        if not pykissembed_paths:
            pytest.skip("No [tool.pykissembed] paths configured")
        config = get_config()
        modules = parse_source_files(pykissembed_paths)
        candidates = find_wrapper_candidates(
            modules,
            root=config.root,
            wrapper_exclude=config.wrapper_exclude,
            wrapper_exempt_decorators=config.wrapper_exempt_decorators,
        )
        violations = [
            candidate
            for candidate in candidates
            if candidate.call_count <= config.wrapper_max_call_sites
        ]
        if violations:
            details = "\n".join(
                f"{candidate.identifier}:{candidate.line} has {candidate.call_count} call site(s)"
                for candidate in violations
            )
            pytest.fail(
                "Wrapper proliferation violations "
                f"(maximum {config.wrapper_max_call_sites} call site(s)):\n{details}",
                pytrace=False,
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
        with _locked_envelope() as (baseline_file, envelope):
            threshold = cast("_BaselineConfig", envelope.data).get(
                "mi_threshold", _DEFAULT_CONFIG["mi_threshold"]
            )
            mi_baselines = cast("dict[str, float]", envelope.data.get("mi_baselines", {}))
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
                    pytrace=False,
                )
