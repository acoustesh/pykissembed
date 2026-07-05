"""Tests for the pykissembed pytest plugin's collection contract.

The plugin must respect a focused test invocation. Running
``pytest tests/test_foo.py::TestBar::test_baz`` should collect only that
test — it should not also collect the entire pykissembed check battery
because the plugin auto-injected its ``checks/`` directory into
``config.args``.

These tests cover the ``_decide_injection`` helper directly so we can
verify the policy without spawning subprocess pytest invocations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pykissembed.plugin import _CHECK_STEMS, _decide_injection


def _make_config(
    args: list[str], *, all_flag: bool = False, collect_only: bool = False
) -> MagicMock:
    """Build a minimal mock of ``pytest.Config`` for ``_decide_injection``.

    The decision helper only reads three things from the config:
    ``getoption('--pykissembed-all')``, ``getoption('--collect-only')``,
    and ``config.args``. Everything else is stubbed out so the helper can
    run in isolation.
    """
    cfg = MagicMock()

    def _getoption(name: str) -> bool:
        if name == "--pykissembed-all":
            return all_flag
        if name == "--collect-only":
            return collect_only
        return False

    cfg.getoption = MagicMock(side_effect=_getoption)
    cfg.args = list(args)
    return cfg


@pytest.fixture
def fake_checks_dir(tmp_path: Path) -> Path:
    """Create a fake ``pykissembed/checks/`` directory with the standard stems.

    Each stem in :data:`pykissembed.plugin._CHECK_STEMS` gets a touch
    file so the smart-restrict branch of ``_decide_injection`` can find
    a real file to return. Anything that does not exist on disk is
    treated as "not injectable" by the helper.
    """
    d = tmp_path / "checks"
    d.mkdir()
    for stem in _CHECK_STEMS:
        (d / f"{stem}.py").write_text("# stub\n", encoding="utf-8")
    return d


class TestDecideInjection:
    """``_decide_injection`` must produce the documented policy matrix."""

    @staticmethod
    def test_bare_pytest_injects_nothing(fake_checks_dir: Path) -> None:
        """``pytest`` alone (no args, no flag) does NOT inject the battery."""
        cfg = _make_config(args=[])
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_collect_only_injects_full_battery(fake_checks_dir: Path) -> None:
        """``--collect-only`` with no other filter → whole ``checks_dir``.

        This is the shape IDE test explorers (VS Code's Python extension)
        use to discover tests, so pykissembed's checks must show up in
        the discovered tree even though a real run wouldn't auto-collect
        them.
        """
        cfg = _make_config(args=["--collect-only"], collect_only=True)
        assert _decide_injection(cfg, fake_checks_dir) == fake_checks_dir

    @staticmethod
    def test_collect_only_does_not_override_unrelated_nodeid(
        fake_checks_dir: Path,
    ) -> None:
        """``--collect-only`` must not override a NodeId targeting a non-check file.

        If the user (or IDE) is specifically discovering one of their own
        tests, that NodeId already decides the outcome (nothing to
        inject) — ``--collect-only`` alone must not widen it to the full
        battery.
        """
        cfg = _make_config(
            args=["tests/test_consumer.py::TestFoo::test_bar", "--collect-only"],
            collect_only=True,
        )
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_collect_only_does_not_override_keyword_filter(
        fake_checks_dir: Path,
    ) -> None:
        """``--collect-only`` combined with ``-k`` still injects nothing.

        A keyword filter already expresses explicit intent to narrow the
        run; ``--collect-only`` must not widen that to the full battery.
        """
        cfg = _make_config(
            args=["-k", "test_docstring_format", "--collect-only"],
            collect_only=True,
        )
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_pykissembed_all_flag_injects_full_battery(fake_checks_dir: Path) -> None:
        """``--pykissembed-all`` returns the whole ``checks_dir``."""
        cfg = _make_config(args=[], all_flag=True)
        assert _decide_injection(cfg, fake_checks_dir) == fake_checks_dir

    @staticmethod
    def test_pykissembed_all_overrides_keyword_filter(
        fake_checks_dir: Path,
    ) -> None:
        """``--pykissembed-all`` beats any other filter (it's an opt-in)."""
        cfg = _make_config(
            args=["-k", "test_docstring_format", "tests/test_foo.py"],
            all_flag=True,
        )
        assert _decide_injection(cfg, fake_checks_dir) == fake_checks_dir

    @staticmethod
    def test_nodeid_naming_a_check_smart_restricts(fake_checks_dir: Path) -> None:
        """``docstring_format.py::TestDocstringFormat::test_x`` → skip injection.

        When the user already passes a check file path on the CLI, the
        plugin must NOT re-inject it (which would cause double-collection).
        The user's NodeId already targets the file, so pytest collects it
        from the CLI arg alone.
        """
        target = fake_checks_dir / "docstring_format.py"
        cfg = _make_config(
            args=[
                f"{target}::TestDocstringFormat::test_docstring_format",
            ],
        )
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_nodeid_from_elsewhere_preserves_test_selector(
        fake_checks_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Smart-restrict must keep the ``::Class::test`` selector.

        Regression test: when the user's NodeId points at a *different*
        file that merely shares a check module's stem (e.g. a consumer's
        own migrated copy of ``code_complexity.py``), the plugin must
        inject ``<real_file>::TestFoo::test_bar`` — not the bare
        ``<real_file>`` — so only the requested test runs instead of
        every test class in the real module.
        """
        consumer_copy = tmp_path / "code_complexity.py"
        consumer_copy.write_text("# consumer's own copy\n", encoding="utf-8")
        cfg = _make_config(
            args=[
                f"{consumer_copy}::TestDocstringCoverage::test_docstring_coverage",
            ],
        )
        candidate = fake_checks_dir / "code_complexity.py"
        assert _decide_injection(cfg, fake_checks_dir) == (
            f"{candidate}::TestDocstringCoverage::test_docstring_coverage"
        )

    @staticmethod
    def test_nodeid_from_elsewhere_class_only_selector(
        fake_checks_dir: Path,
        tmp_path: Path,
    ) -> None:
        """A class-only selector (no test method) is preserved too."""
        consumer_copy = tmp_path / "docstring_format.py"
        consumer_copy.write_text("# consumer's own copy\n", encoding="utf-8")
        cfg = _make_config(
            args=[f"{consumer_copy}::TestDocstringFormat"],
        )
        candidate = fake_checks_dir / "docstring_format.py"
        assert _decide_injection(cfg, fake_checks_dir) == (
            f"{candidate}::TestDocstringFormat"
        )

    @staticmethod
    def test_nodeid_naming_a_check_with_other_args_still_skips(
        fake_checks_dir: Path,
    ) -> None:
        """A NodeId targeting a check file skips injection even with other args."""
        target = fake_checks_dir / "code_complexity.py"
        cfg = _make_config(
            args=[
                "-p",
                "no:cacheprovider",
                f"{target}::TestCyclomaticComplexity::test_cyclomatic_complexity",
            ],
        )
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_nodeid_not_naming_a_pytest_check_injects_nothing(
        fake_checks_dir: Path,
    ) -> None:
        """A NodeId pointing at a consumer file does NOT inject the battery."""
        cfg = _make_config(
            args=["tests/test_consumer.py::TestFoo::test_bar"],
        )
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_keyword_filter_injects_nothing(fake_checks_dir: Path) -> None:
        """``-k`` keyword filter alone: do not auto-inject."""
        cfg = _make_config(args=["-k", "test_docstring_format"])
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_keyword_equals_form_injects_nothing(fake_checks_dir: Path) -> None:
        """``-k=...`` (equals form) is also detected as a keyword filter."""
        cfg = _make_config(args=["-k=test_docstring_format"])
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_marker_filter_injects_full_battery(fake_checks_dir: Path) -> None:
        """``-m <marker>`` returns the whole ``checks_dir`` so markers can narrow."""
        cfg = _make_config(args=["-m", "docstring_format"])
        assert _decide_injection(cfg, fake_checks_dir) == fake_checks_dir

    @staticmethod
    def test_marker_equals_form_injects_full_battery(
        fake_checks_dir: Path,
    ) -> None:
        """``-m=...`` (equals form) is also detected as a marker filter."""
        cfg = _make_config(args=["-m=docstring_format"])
        assert _decide_injection(cfg, fake_checks_dir) == fake_checks_dir

    @staticmethod
    def test_deselect_injects_nothing(fake_checks_dir: Path) -> None:
        """``--deselect`` alone: do not auto-inject."""
        cfg = _make_config(
            args=["--deselect", "tests/test_foo.py::TestBar::test_baz"],
        )
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_deselect_equals_form_injects_nothing(fake_checks_dir: Path) -> None:
        """``--deselect=...`` is also detected as a deselect filter."""
        cfg = _make_config(
            args=["--deselect=tests/test_foo.py::TestBar::test_baz"],
        )
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_missing_check_file_does_not_crash(
        tmp_path: Path,
    ) -> None:
        """If the smart-restricted file does not exist, return None (not raise)."""
        # Make a checks dir whose file is missing on disk.
        d = tmp_path / "checks_empty"
        d.mkdir()
        cfg = _make_config(
            args=[f"{d}/docstring_format.py::TestDocstringFormat::test_x"],
        )
        # The smart-restrict branch checks is_file() before returning;
        # the missing file must produce None, not a broken path.
        assert _decide_injection(cfg, d) is None

    @staticmethod
    def test_nodeid_already_targeting_check_file_skips_injection(
        fake_checks_dir: Path,
    ) -> None:
        """When the user already passes a check file path on the CLI, do NOT re-inject.

        This prevents double-collection: pytest collects each entry in
        config.args independently, so appending a path the user already
        passed causes the same test to run twice.
        """
        target = fake_checks_dir / "docstring_format.py"
        cfg = _make_config(
            args=[
                f"{target}::TestDocstringFormat::test_docstring_format",
            ],
        )
        # The user's NodeId already points at the check file inside
        # checks_dir. _decide_injection must return None to avoid
        # re-injecting the same path.
        assert _decide_injection(cfg, fake_checks_dir) is None

    @staticmethod
    def test_bare_path_already_targeting_check_file_skips_injection(
        fake_checks_dir: Path,
    ) -> None:
        """When the user passes a bare check file path, do NOT re-inject."""
        target = fake_checks_dir / "docstring_format.py"
        cfg = _make_config(
            args=[str(target)],
        )
        assert _decide_injection(cfg, fake_checks_dir) is None


class TestPluginEntryPoint:
    """Smoke tests: the plugin module is importable and exposes the right symbols."""

    @staticmethod
    def test_check_stems_are_frozenset() -> None:
        """``_CHECK_STEMS`` is an immutable set of check module stems."""
        assert isinstance(_CHECK_STEMS, frozenset)
        # All five check modules must be listed
        assert {
            "code_complexity",
            "code_similarity",
            "comment_density",
            "docstring_format",
            "lint_typecheck",
        } == set(_CHECK_STEMS)

    @staticmethod
    def test_plugin_registers_pytest11_entry() -> None:
        """``pyproject.toml`` still registers the plugin via ``pytest11``."""
        import tomllib

        with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        eps = data["project"]["entry-points"]["pytest11"]
        assert "pykissembed" in eps
        assert eps["pykissembed"] == "pykissembed.plugin"


class TestSubprocessCollection:
    """End-to-end subprocess tests of the collection contract.

    Each test scaffolds a tiny consumer project, invokes pytest on it
    with a specific argument shape, and asserts that *only* the expected
    tests were collected. This guards against regressions in the
    surrounding hook machinery (e.g. ``pytest_collect_file``) that the
    pure unit tests cannot catch.
    """

    @staticmethod
    def test_specific_nodeid_only_collects_that_test(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``pytest <file>::Test::test_x`` collects exactly one test, not 11."""
        import subprocess

        # Build a minimal consumer project that has pykissembed installed
        # in editable mode. ``pip install -e .`` is too slow; instead we set
        # PYTHONPATH so the plugin is importable.
        consumer = tmp_path / "consumer"
        consumer.mkdir()
        (consumer / "tests").mkdir()
        (consumer / "tests" / "test_user.py").write_text(
            "def test_user_passes():\n    assert True\n",
            encoding="utf-8",
        )
        (consumer / "pyproject.toml").write_text(
            '[tool.pykissembed]\npaths = ["."]\n',
            encoding="utf-8",
        )

        repo = Path(__file__).resolve().parents[1]
        env = {
            **__import__("os").environ,
            "PYTHONPATH": str(repo),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_user.py::test_user_passes",
                "--collect-only",
                "-q",
            ],
            cwd=consumer,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        # Only the user's test was collected. Crucially, NONE of the
        # pykissembed check tests appear, because the user's NodeId
        # smart-restricted the plugin injection to nothing matching
        # (since "test_user.py" is not a pykissembed check stem).
        collected = [
            line
            for line in result.stdout.splitlines()
            if "::" in line and "no tests ran" not in line.lower()
        ]
        assert collected == [
            "tests/test_user.py::test_user_passes",
        ], (
            "expected only the user's test to be collected, "
            f"got:\n{chr(10).join(collected)}\n\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    @staticmethod
    def test_pykissembed_all_flag_collects_the_battery(
        tmp_path: Path,
    ) -> None:
        """``--pykissembed-all`` collects every check module, even on bare pytest."""
        import os
        import subprocess

        consumer = tmp_path / "consumer"
        consumer.mkdir()
        (consumer / "pyproject.toml").write_text(
            '[tool.pykissembed]\npaths = ["."]\n',
            encoding="utf-8",
        )

        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(repo)}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--pykissembed-all",
                "--collect-only",
                "-q",
            ],
            cwd=consumer,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        # Every check module stem must appear in the collected output.
        # The exact path-prefix depends on pytest's collection rootdir
        # resolution, so we just check for the stem appearing as a
        # Python file reference (`{stem}.py` in the NodeId).
        for stem in _CHECK_STEMS:
            assert f"{stem}.py" in result.stdout, (
                f"expected {stem} to be collected with --pykissembed-all, "
                f"got:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

    @staticmethod
    def test_bare_pytest_run_collects_nothing_from_plugin(
        tmp_path: Path,
    ) -> None:
        """A real (non-discovery) bare ``pytest`` run does not execute the battery.

        No ``--collect-only`` here: this is what happens when a consumer
        just runs ``pytest`` to execute their own suite. It must not pull
        in pykissembed's check battery as a side effect.
        """
        import os
        import subprocess

        consumer = tmp_path / "consumer"
        consumer.mkdir()
        (consumer / "pyproject.toml").write_text(
            '[tool.pykissembed]\npaths = ["."]\n',
            encoding="utf-8",
        )

        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(repo)}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
            ],
            cwd=consumer,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        # Exit code 5 means "no tests collected" — exactly what we want.
        assert result.returncode in (0, 5), result.stderr
        # No check stem should appear in the collected output.
        for stem in _CHECK_STEMS:
            assert stem not in result.stdout, (
                f"bare `pytest` should not auto-collect {stem}, got:\n{result.stdout}"
            )

    @staticmethod
    def test_bare_collect_only_discovers_the_battery(
        tmp_path: Path,
    ) -> None:
        """Bare ``pytest --collect-only`` DOES discover the check battery.

        Regression test for a consumer report: "no pykissembed tests are
        found on the pytest side panel". VS Code's Python Test Explorer
        (and other IDE test explorers) discover tests by invoking pytest
        with ``--collect-only`` and no other filter — the same shape as
        the configured ``python.testing.pytestArgs`` plus ``--collect-only``.
        Without this rule, that bare discovery collected nothing and
        pykissembed's checks never appeared in the test tree, even though
        a real execution run (see
        ``test_bare_pytest_run_collects_nothing_from_plugin``) is correctly
        unaffected — ``--collect-only`` never executes anything, so
        showing the battery in the tree is safe.
        """
        import os
        import subprocess

        consumer = tmp_path / "consumer"
        consumer.mkdir()
        (consumer / "pyproject.toml").write_text(
            '[tool.pykissembed]\npaths = ["."]\n',
            encoding="utf-8",
        )

        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(repo)}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
            ],
            cwd=consumer,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        for stem in _CHECK_STEMS:
            assert f"{stem}.py" in result.stdout, (
                f"expected {stem} to be discoverable via --collect-only, "
                f"got:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

    @staticmethod
    def test_check_nodeid_collects_exactly_once(
        tmp_path: Path,
    ) -> None:
        """Targeting a check NodeId must collect the test exactly once, not twice.

        Before the fix, the plugin would re-inject the same check file
        path into ``config.args`` even though the user already passed it
        on the CLI, causing pytest to collect the test twice.
        """
        import os
        import subprocess

        consumer = tmp_path / "consumer"
        consumer.mkdir()
        (consumer / "pyproject.toml").write_text(
            '[tool.pykissembed]\npaths = ["."]\n',
            encoding="utf-8",
        )

        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, "PYTHONPATH": str(repo)}
        # Find the installed check file path.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pykissembed.plugin import _checks_dir; print(_checks_dir())",
            ],
            cwd=consumer,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        checks_dir = Path(result.stdout.strip())
        target = checks_dir / "docstring_format.py"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{target}::TestDocstringFormat::test_docstring_format",
                "--collect-only",
                "-q",
            ],
            cwd=consumer,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        # Count how many times the test appears in the collected output.
        collected = [
            line
            for line in result.stdout.splitlines()
            if "test_docstring_format" in line and "::" in line
        ]
        assert len(collected) == 1, (
            f"expected test_docstring_format to be collected exactly once, "
            f"but found {len(collected)} times:\n{chr(10).join(collected)}\n\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    @staticmethod
    def test_consumer_migrated_copy_does_not_run_whole_module(
        tmp_path: Path,
    ) -> None:
        """Reproduces the reported bug of selecting one test running all.

        Selecting one test in a consumer's migrated copy of
        ``code_complexity.py`` must not also run the other four test
        classes from the real installed module.

        Mirrors the mega-scrapper migration pattern (see
        ``tests/fixtures/sample_repo/tests/code_complexity.py``): a
        consumer keeps its own copy of the check module (same class and
        test names) alongside the installed pykissembed plugin. Running
        ``pytest tests/code_complexity.py::TestDocstringCoverage::``
        ``test_docstring_coverage`` must select only that test — not
        ``TestLineCount``, ``TestCyclomaticComplexity``,
        ``TestCognitiveComplexity``, or ``TestMaintainabilityIndex``.
        """
        import os
        import shutil
        import subprocess

        repo = Path(__file__).resolve().parents[1]
        consumer = tmp_path / "consumer"
        (consumer / "tests").mkdir(parents=True)
        shutil.copyfile(
            repo
            / "tests"
            / "fixtures"
            / "sample_repo"
            / "tests"
            / "code_complexity.py",
            consumer / "tests" / "code_complexity.py",
        )
        (consumer / "pyproject.toml").write_text(
            '[tool.pykissembed]\npaths = ["."]\n',
            encoding="utf-8",
        )

        env = {**os.environ, "PYTHONPATH": str(repo)}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/code_complexity.py::TestDocstringCoverage::"
                "test_docstring_coverage",
                "--collect-only",
                "-q",
            ],
            cwd=consumer,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        for other_class in (
            "TestLineCount",
            "TestCyclomaticComplexity",
            "TestCognitiveComplexity",
            "TestMaintainabilityIndex",
        ):
            assert other_class not in result.stdout, (
                f"{other_class} must not be collected when only "
                "TestDocstringCoverage::test_docstring_coverage was "
                f"requested, got:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        assert "test_docstring_coverage" in result.stdout


# The submodule guards against pytest auto-collecting this test file
# itself (it does not contain pytest tests at module level, only inside
# classes — pytest finds them fine, but the guard makes the intent
# explicit).
_ = SimpleNamespace  # keep the import non-empty for tooling

__all__ = [
    "TestDecideInjection",
    "TestPluginEntryPoint",
    "TestSubprocessCollection",
]
