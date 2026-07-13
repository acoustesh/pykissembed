"""Tests for the strict exact-forwarder wrapper-proliferation gate."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from pykissembed.checks import code_complexity
from pykissembed.checks.code_complexity import TestWrapperProliferation
from pykissembed.config import PyqtestConfig
from pykissembed.wrapper_analysis import (
    WrapperCandidate,
    find_wrapper_candidates,
    parse_source_files,
)


def _write_module(root: Path, name: str, source: str) -> Path:
    """Write a Python source fixture below *root*.

    Returns
    -------
    Path
        The created source file.
    """
    path = root / name
    path.write_text(dedent(source).lstrip(), encoding="utf-8")
    return path


def _candidates(
    tmp_path: Path,
    sources: dict[str, str],
    *,
    wrapper_exclude: list[str] | None = None,
    wrapper_exempt_decorators: list[str] | None = None,
) -> list[WrapperCandidate]:
    """Collect wrapper candidates from source fixtures.

    Returns
    -------
    list[WrapperCandidate]
        Parsed candidates using *tmp_path* as the project root.
    """
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    for name, source in sources.items():
        _write_module(source_dir, name, source)
    return find_wrapper_candidates(
        parse_source_files([source_dir]),
        root=tmp_path,
        wrapper_exclude=wrapper_exclude or [],
        wrapper_exempt_decorators=wrapper_exempt_decorators or [],
    )


class TestExactForwarders:
    """Candidate detection for exact pass-through function shapes."""

    @staticmethod
    @pytest.mark.parametrize(
        ("calls", "expected_count"),
        [
            ("", 0),
            ("proxy(value)\n", 1),
            ("proxy(value)\nservice.proxy(value)\n", 2),
        ],
    )
    def test_counts_bare_and_attribute_calls(
        tmp_path: Path,
        calls: str,
        expected_count: int,
    ) -> None:
        """Project-wide terminal-name counting includes both supported call forms."""
        candidates = _candidates(
            tmp_path,
            {
                "module.py": (
                    dedent(
                        """
                    def target(value):
                        return value

                    def proxy(value):
                        return target(value)

                    """
                    )
                    + calls
                ),
            },
        )

        assert [(candidate.identifier, candidate.call_count) for candidate in candidates] == [
            ("src/module.py:proxy", expected_count),
        ]

    @staticmethod
    def test_detects_async_methods_and_variadics(tmp_path: Path) -> None:
        """Exact async, bound-method, and variadic forwarding is detected."""
        candidates = _candidates(
            tmp_path,
            {
                "module.py": """
                async def target(value):
                    return value

                async def async_proxy(value):
                    return await target(value)

                class Adapter:
                    def target(self, value):
                        return value

                    def proxy(self, value):
                        return self.target(value)

                def variadic(*args, **kwargs):
                    return target(*args, **kwargs)

                async_proxy(value)
                Adapter().proxy(value)
                variadic(value)
                """,
            },
        )

        assert [(candidate.identifier, candidate.call_count) for candidate in candidates] == [
            ("src/module.py:Adapter.proxy", 1),
            ("src/module.py:async_proxy", 1),
            ("src/module.py:variadic", 1),
        ]

    @staticmethod
    def test_ignores_adapters_that_transform_or_add_behaviour(tmp_path: Path) -> None:
        """Only unchanged one-statement forwarding is a candidate."""
        candidates = _candidates(
            tmp_path,
            {
                "module.py": """
                def target(*args, **kwargs):
                    return args, kwargs

                def transformed(value):
                    return target(value.strip())

                def reordered(left, right):
                    return target(right, left)

                def literal(value):
                    return target(value, enabled=True)

                def side_effect(value):
                    target(value)

                def multiple(value):
                    target(value)
                    return target(value)
                """,
            },
        )

        assert candidates == []


class TestWrapperExemptions:
    """Automatic and project-configured wrapper exemptions."""

    @staticmethod
    def test_automatic_decorator_exemptions(tmp_path: Path) -> None:
        """Language and framework decorators suppress intentional adapters."""
        candidates = _candidates(
            tmp_path,
            {
                "module.py": """
                @pytest.fixture
                def fixture_proxy(value):
                    return target(value)

                class Adapter:
                    @property
                    def property_proxy(self):
                        return self.target()

                    @override
                    def override_proxy(self, value):
                        return self.target(value)
                """,
            },
        )

        assert candidates == []

    @staticmethod
    def test_identifier_and_decorator_patterns_exclude_candidates(tmp_path: Path) -> None:
        """Glob configuration excludes named adapters and custom decorators."""
        candidates = _candidates(
            tmp_path,
            {
                "module.py": """
                class Adapter:
                    def proxy(self, value):
                        return self.target(value)

                @custom.adapter
                def decorated_proxy(value):
                    return target(value)
                """,
            },
            wrapper_exclude=["src/module.py:Adapter.*"],
            wrapper_exempt_decorators=["custom.*"],
        )

        assert candidates == []


class TestWrapperDiscoveryRobustness:
    """Deterministic and resilient wrapper source discovery."""

    @staticmethod
    def test_output_is_sorted_and_invalid_source_is_ignored(tmp_path: Path) -> None:
        """Candidates sort by identifier while syntax-invalid files are skipped."""
        candidates = _candidates(
            tmp_path,
            {
                "zeta.py": """
                def zeta_proxy(value):
                    return target(value)
                """,
                "alpha.py": """
                def alpha_proxy(value):
                    return target(value)
                """,
                "invalid.py": "def broken(:\n",
            },
        )

        assert [candidate.identifier for candidate in candidates] == [
            "src/alpha.py:alpha_proxy",
            "src/zeta.py:zeta_proxy",
        ]

    @staticmethod
    def test_unreadable_source_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A decoding failure warns and does not prevent other files from scanning."""
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        readable = _write_module(
            source_dir,
            "readable.py",
            """
            def proxy(value):
                return target(value)
            """,
        )
        unreadable = _write_module(source_dir, "unreadable.py", "def ignored():\n    pass\n")
        original_read_text = Path.read_text
        invalid_encoding = "utf-8"
        invalid_bytes = b"\\xff"
        reason = "invalid byte"

        def _read_text(
            path: Path,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> str:
            if path == unreadable:
                raise UnicodeDecodeError(invalid_encoding, invalid_bytes, 0, 1, reason)
            return original_read_text(path, encoding=encoding, errors=errors, newline=newline)

        monkeypatch.setattr(Path, "read_text", _read_text)
        with pytest.warns(UserWarning, match="unreadable.py"):
            modules = parse_source_files([source_dir])

        assert [path for path, _ in modules] == [readable.resolve()]


class TestWrapperProliferationGate:
    """End-to-end threshold enforcement for the complexity test."""

    @staticmethod
    def test_threshold_controls_strict_failure(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A one-use wrapper fails at one and passes at zero allowed call sites."""
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        _write_module(
            source_dir,
            "module.py",
            """
            def target(value):
                return value

            def proxy(value):
                return target(value)

            proxy(value)
            """,
        )
        strict_config = PyqtestConfig(paths=["src"], root=tmp_path)
        monkeypatch.setattr(code_complexity, "get_config", lambda: strict_config)

        with pytest.raises(pytest.fail.Exception, match=r"src/module\.py:proxy"):
            TestWrapperProliferation.test_wrapper_proliferation([source_dir])

        permissive_config = PyqtestConfig(paths=["src"], root=tmp_path, wrapper_max_call_sites=0)
        monkeypatch.setattr(code_complexity, "get_config", lambda: permissive_config)
        TestWrapperProliferation.test_wrapper_proliferation([source_dir])
