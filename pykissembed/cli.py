"""``pykissembed`` CLI — Typer-based command surface.

Subcommands
-----------
- ``pykissembed check`` — run the same gate pytest runs (delegates to pytest)
- ``pykissembed ratchet`` — lower baselines; refuse to raise
- ``pykissembed.providers list`` — show installed embedding providers
- ``pykissembed populate-embeddings --provider NAME``
- ``pykissembed type-review --json REPORT.json``
- ``pykissembed docfix [PATHS]`` — report or fix NumPy docstring non-conformance
- ``pykissembed init`` — (opt-in) scaffold a ``[tool.pykissembed]`` block and
  sync VS Code pytest settings
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tomllib
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from pykissembed import __version__
from pykissembed.baselines_engine import ratchet
from pykissembed.checks.lint_typecheck import build_report, run_pyright, run_ruff
from pykissembed.config import get_config, load_config
from pykissembed.providers.registry import discover_all
from pykissembed.tools.docfix import DEFAULT_MODEL as DEFAULT_DOCFIX_MODEL
from pykissembed.tools.docfix import Request as DocfixRequest
from pykissembed.tools.docfix import main as run_docfix
from pykissembed.vscode_settings import sync_vscode_settings

app = typer.Typer(
    name="pykissembed",
    help="Generic Python code-quality test library (pytest plugin).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    *,
    version: Annotated[
        bool,
        typer.Option(
        "--version",
        help="Show pykissembed version and exit.",
        ),
    ] = False,
) -> None:
    """Print the pykissembed version and exit if --version is passed.

    Parameters
    ----------
    ctx : typer.Context
        Typer context for the top-level group.
    version : bool
        When true, print the version and exit instead of dispatching.

    Raises
    ------
    typer.Exit
        When the version is printed or no subcommand was provided.
    """
    if version:
        typer.echo(f"pykissembed {__version__}")
        raise typer.Exit
    # Without a subcommand and without --version, show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit


@app.command()
def check(
    pytest_args: Annotated[
        list[str] | None,
        typer.Argument(help="Extra args forwarded to pytest."),
    ] = None,
) -> None:
    """Run the same gate that ``pytest`` runs (lint + type + complexity + ...).

    Uses ``sys.executable -m pytest`` so the current Python environment
    (with ``pykissembed`` installed) is reused — bare ``pytest`` on PATH may
    resolve to a different interpreter that lacks the plugin.

    Since v0.1.9, pykissembed's check-module collection is opt-in (a bare
    ``pytest`` invocation no longer auto-collects the battery — see
    ``pykissembed/plugin.py::_decide_injection``). When no extra args are
    given, default to ``--pykissembed-all`` so ``pykissembed check`` still
    runs the full battery as documented. If the caller passes their own
    args (e.g. a marker or a specific check NodeId), forward them
    unchanged so their scoping is respected.

    Parameters
    ----------
    pytest_args : list[str] | None
        Extra pytest arguments to forward; defaults to ``--pykissembed-all``.

    Raises
    ------
    typer.Exit
        With pytest's exit status after the gate completes.
    """
    args = list(pytest_args) if pytest_args else ["--pykissembed-all"]
    cmd = [sys.executable, "-m", "pytest", *args]
    typer.echo(f"$ {' '.join(cmd)}")
    # S603: fixed argv list (sys.executable + literal flags + the CLI's own
    # forwarded args); this command's entire purpose is to forward args to
    # pytest, so there is no narrower "trusted" input to require.
    raise typer.Exit(subprocess.call(cmd))


@app.command(name="ratchet")
def ratchet_cmd(
    baseline_dir: Annotated[
        Path | None,
        typer.Option("--baseline-dir", help="Override the configured baseline directory."),
    ] = None,
) -> None:
    """Lower baselines where current diagnostics are lower; refuse to raise.

    Reads every JSON file in the configured baseline directory, computes
    current diagnostics, and writes a ratcheted baseline back. Numeric
    baselines only go downward (or stay equal); new diagnostics are
    captured at their current value.

    This is the recommended post-commit hook target.

    Parameters
    ----------
    baseline_dir : Path | None
        Directory overriding the configured baseline directory.

    Raises
    ------
    typer.Exit
        If the baseline directory is missing or ratcheting completes.
    """
    config = get_config()
    bdir = baseline_dir or config.baseline_path
    if not bdir.exists():
        typer.echo(f"No baseline directory at {bdir}. Nothing to ratchet.")
        raise typer.Exit(0)
    n_lowered = 0
    for path in sorted(bdir.glob("*.json")):
        if path.name.endswith("_report.json"):
            continue
        try:
            current = _compute_current_for(path.name)
        except NotImplementedError:
            typer.echo(f"  skip {path.name}: no current-diagnostics computer implemented")
            continue
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            typer.echo(f"  skip {path.name}: {exc}")
            continue
        with path.open(encoding="utf-8") as f:
            envelope = json.load(f)
        data = envelope.get("data", envelope) if isinstance(envelope, dict) else {}
        if not isinstance(data, dict):
            continue
        new_data = ratchet(data, current)
        if new_data != data:
            envelope["data"] = new_data
            with path.open("w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=2, sort_keys=True)
                _ = f.write("\n")
            n_lowered += 1
            typer.echo(f"  ratcheted {path.name}")
    typer.echo(f"Done. {n_lowered} baseline file(s) lowered.")


def _compute_current_for(baseline_name: str) -> dict[str, Any]:
    """Dispatch to the right current-diagnostics computer based on filename.

    Parameters
    ----------
    baseline_name : str
        Baseline file name (e.g. ``"lint_typecheck.json"``) selecting the
        computer to run.

    Returns
    -------
    dict[str, Any]
        For ``"lint_typecheck.json"``: a ``{"per_file": {...}}`` dict
        mapping each file to its current ruff+pyright diagnostic count
        (empty dict if no ``[tool.pykissembed]`` paths are configured).

    Raises
    ------
    NotImplementedError
        If no computer is implemented for *baseline_name*.
    """
    if baseline_name == "lint_typecheck.json":
        paths = get_config().resolved_paths()
        if not paths:
            return {}
        root = get_config().root
        report = build_report(run_ruff(paths), run_pyright(paths), root=root)
        return {
            "per_file": {f: len(d["ruff"]) + len(d["pyright"]) for f, d in report["files"].items()}
        }
    msg = f"No current-diagnostics computer for {baseline_name}"
    raise NotImplementedError(msg)


# ---------------------------------------------------------------------------
# providers subcommand
# ---------------------------------------------------------------------------

providers_app = typer.Typer(help="Inspect installed embedding providers.", no_args_is_help=True)


@providers_app.command("list")
def providers_list() -> None:
    """List all installed embedding providers (built-in + entry points).

    Raises
    ------
    typer.Exit
        If no providers are installed.
    """
    registry = discover_all()
    if not registry.all():
        typer.echo("No providers installed. Try: pip install 'pykissembed[cloud]'")
        raise typer.Exit(0)
    table = Table(title=f"pykissembed.providers (pykissembed {__version__})")
    table.add_column("Name", style="bold")
    table.add_column("Model")
    table.add_column("Configured")
    table.add_column("Max tokens")
    table.add_column("Batch size")
    for provider in registry.all():
        try:
            configured = provider.is_configured()
        except (OSError, RuntimeError, TypeError, ValueError):
            configured = False
        table.add_row(
            provider.name,
            provider.model_id,
            "yes" if configured else "no",
            str(provider.max_tokens),
            str(provider.batch_size),
        )
    console.print(table)


app.add_typer(providers_app, name="providers")


# ---------------------------------------------------------------------------
# populate-embeddings
# ---------------------------------------------------------------------------


@app.command()
def populate_embeddings(
    provider_name: str = typer.Option(
        ...,
        "--provider",
        help="Canonical provider variant (for example, openai-text or openai-ast).",
    ),
    paths: Annotated[
        list[Path] | None,
        typer.Option("--path", help="Directories to scan (default: configured source paths)."),
    ] = None,
    *,
    cached_only: Annotated[
        bool,
        typer.Option("--cached-only", help="Skip API calls; only read cache."),
    ] = False,
) -> None:
    """Populate or inspect compressed per-function embedding caches.

    Parameters
    ----------
    provider_name : str
        Canonical provider variant (e.g. ``"openai-text"``), or ``"all"``.
    paths : list[Path] | None
        Directories to scan; ``None`` uses the configured source paths.
    cached_only : bool
        Inspect cache coverage without API calls or cache writes.

    Raises
    ------
    TypeError
        If the lazily loaded population module does not expose its documented API.
    typer.Exit
        If the provider name is invalid or the requested cloud provider cannot
        populate its missing cache entries.
    """
    # Keep the numerical similarity subsystem lazy for unrelated CLI commands,
    # but validate both objects obtained through the dynamic module boundary.
    population_module = import_module("pykissembed.similarity.populate_embeddings")
    populate_provider_embeddings = getattr(
        population_module,
        "populate_provider_embeddings",
        None,
    )
    population_error = getattr(population_module, "PopulationError", None)
    if not callable(populate_provider_embeddings):
        msg = "populate_provider_embeddings must be callable"
        raise TypeError(msg)
    if not isinstance(population_error, type) or not issubclass(population_error, Exception):
        msg = "PopulationError must be an exception type"
        raise TypeError(msg)

    try:
        _ = populate_provider_embeddings(
            provider_name,
            paths=paths,
            cached_only=cached_only,
        )
    except population_error as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from None


# ---------------------------------------------------------------------------
# type-review
# ---------------------------------------------------------------------------


@app.command()
def type_review(
    report: Annotated[
        Path,
        typer.Option("--json", help="Path to a lint_typecheck_report.json produced by the lint gate."),
    ],
) -> None:
    """Re-run ``pyright`` on each file that has pyright diagnostics in *report*.

    For each file with a non-empty ``pyright`` list in the report, runs
    ``pyright`` against just that file so the developer can focus on the
    failing diagnostics. Pyright's own exit statuses are ignored.

    Parameters
    ----------
    report : Path
        Path to a ``lint_typecheck_report.json`` produced by the lint gate.

    Raises
    ------
    typer.Exit
        With status 1 if the report is missing, or status 0 if no file has
        pyright diagnostics. Otherwise the command returns normally.
    """
    if not report.exists():
        typer.echo(f"Report not found: {report}")
        raise typer.Exit(1)
    payload = json.loads(report.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    pyright_files = [f for f, d in files.items() if d.get("pyright")]
    if not pyright_files:
        typer.echo("No pyright errors in report. Nothing to review.")
        raise typer.Exit(0)
    pyright = shutil.which("pyright") or "pyright"
    for fp in pyright_files:
        typer.echo(f"\n=== {fp} ===")
        # S603: fixed 2-element argv (resolved pyright binary + a file path
        # already validated against the loaded report); no shell involved.
        _ = subprocess.call([pyright, fp])


# ---------------------------------------------------------------------------
# init (opt-in scaffolder)
# ---------------------------------------------------------------------------


@app.command()
def init(
    *,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=(
                "Overwrite an existing [tool.pykissembed] block and any "
                "conflicting .vscode/settings.json pytest values."
            ),
        ),
    ] = False,
) -> None:
    """Scaffold ``[tool.pykissembed]`` in ``pyproject.toml`` and sync VS Code pytest settings.

    Auto-detects source directories from the project layout (``src/``,
    ``[tool.setuptools]`` packages, or ``.`` as fallback) for the
    ``pyproject.toml`` block. Independently, adds
    ``python.testing.pytestArgs``/``pytestEnabled`` to
    ``.vscode/settings.json`` so VS Code's Test Explorer runs pykissembed's
    checks. Each step is idempotent and only needs ``--force`` to overwrite
    a value that already differs from pykissembed's own.

    Parameters
    ----------
    force : bool
        Overwrite an existing ``[tool.pykissembed]`` block and conflicting VS Code pytest values.

    Raises
    ------
    typer.Exit
        If ``pyproject.toml`` is missing entirely.
    """
    config = get_config()
    pyproject = config.root / "pyproject.toml"
    if not pyproject.exists():
        typer.echo(f"No pyproject.toml at {pyproject}. Run this from your project root.")
        raise typer.Exit(1)
    text = pyproject.read_text()
    if "[tool.pykissembed]" in text and not force:
        typer.echo("[tool.pykissembed] already present. Use --force to overwrite.")
    else:
        # Auto-detect source directories from the project layout
        detected_paths = _auto_detect_paths(config.root, text)
        paths_str = ", ".join(f'"{p}"' for p in detected_paths)
        existing_config = _toml_value(_parse_pyproject(text), "tool", "pykissembed")
        cached_only_value = (
            existing_config.get("cached_only") if isinstance(existing_config, dict) else None
        )
        cached_only_line = (
            f"cached_only = {str(cached_only_value).lower()}\n"
            if isinstance(cached_only_value, bool)
            else ""
        )

        block = (
            "\n[tool.pykissembed]\n"
            f"paths = [{paths_str}]\n"
            'mode = "ratchet"\n'
            'baseline_dir = "tests/baselines"\n'
            f"{cached_only_line}"
        )
        if "[tool.pykissembed]" in text:
            # Replace the existing block (very simple; assumes our own format).
            # Stop only at the next TOML table header (a "[" at line start),
            # not at any "[" -- values like `paths = ["."]` contain one too.
            text = re.sub(
                r"\[tool\.pykissembed\][\s\S]*?(?=\n\[|\Z)", block.strip() + "\n", text, count=1
            )
        else:
            text = text.rstrip() + "\n" + block
        _ = pyproject.write_text(text)
        typer.echo(f"Added [tool.pykissembed] to {pyproject} (paths={detected_paths}).")

    for message in sync_vscode_settings(config.root, force=force):
        typer.echo(message)


def _auto_detect_paths(root: Path, pyproject_text: str) -> list[str]:
    """Auto-detect source directories from the project layout.

    The first matching rule wins:

    1. ``[tool.setuptools.packages.find]`` ``where`` field
    2. ``[tool.hatch.build.targets.wheel]`` ``packages`` list
    3. ``src/`` directory if it exists
    4. ``.`` (current directory) as fallback

    Parameters
    ----------
    root : Path
        Project root containing ``pyproject.toml``.
    pyproject_text : str
        Raw text of ``pyproject.toml``.

    Returns
    -------
    list[str]
        The detected source directory path(s), taken from the first
        matching priority rule above.
    """
    data = _parse_pyproject(pyproject_text)
    return (
        _setuptools_source_paths(data) or _hatch_source_paths(data) or _default_source_paths(root)
    )


def _parse_pyproject(pyproject_text: str) -> dict[str, object]:
    """Parse a ``pyproject.toml`` document, degrading invalid text to an empty table.

    Parameters
    ----------
    pyproject_text : str
        Full ``pyproject.toml`` text to parse.

    Returns
    -------
    dict[str, object]
        The parsed TOML table, or an empty table when parsing fails.
    """
    try:
        return tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError:
        return {}


def _setuptools_source_paths(data: dict[str, object]) -> list[str]:
    """Return paths configured by setuptools package discovery.

    Parameters
    ----------
    data : dict[str, object]
        Parsed ``pyproject.toml`` table.

    Returns
    -------
    list[str]
        Configured ``where`` paths, or an empty list when none are present.
    """
    find_where = _toml_value(data, "tool", "setuptools", "packages", "find", "where")
    legacy_where = _toml_value(data, "tool", "setuptools", "packages", "where")
    return _path_values(find_where) or _path_values(legacy_where)


def _hatch_source_paths(data: dict[str, object]) -> list[str]:
    """Return distinct source roots configured in Hatch's wheel target.

    Parameters
    ----------
    data : dict[str, object]
        Parsed ``pyproject.toml`` table.

    Returns
    -------
    list[str]
        Source roots in configured package order, or an empty list.
    """
    packages = _toml_value(data, "tool", "hatch", "build", "targets", "wheel", "packages")
    return _package_roots(packages)


def _toml_value(data: dict[str, object], *keys: str) -> object | None:
    """Return a nested TOML value without exposing intermediate tables.

    Parameters
    ----------
    data : dict[str, object]
        Parsed ``pyproject.toml`` table to walk.
    *keys : str
        Successive table keys leading to the value.

    Returns
    -------
    object | None
        The nested value, or ``None`` when any table is absent or malformed.
    """
    value: object = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _path_values(value: object | None) -> list[str]:
    """Normalize a TOML path value to a non-empty list of strings.

    Parameters
    ----------
    value : object | None
        Raw TOML value that may be a path string or a list of paths.

    Returns
    -------
    list[str]
        One string path, a non-empty list coerced to strings, or an empty list.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value:
        return [str(path) for path in value]
    return []


def _package_roots(packages: object | None) -> list[str]:
    """Extract distinct leading directories from Hatch package paths.

    Parameters
    ----------
    packages : object | None
        Raw Hatch ``packages`` value.

    Returns
    -------
    list[str]
        Package roots in first-seen order, or an empty list for non-lists.
    """
    if not isinstance(packages, list):
        return []
    return list(
        dict.fromkeys(package.split("/", 1)[0] for package in packages if isinstance(package, str))
    )


def _default_source_paths(root: Path) -> list[str]:
    """Return the conventional source path when build configuration is absent.

    Parameters
    ----------
    root : Path
        Project root to look for a source directory in.

    Returns
    -------
    list[str]
        ``["src"]`` when *root* has a source directory; otherwise ``["."]``.
    """
    return ["src"] if (root / "src").is_dir() else ["."]


__all__ = ["app", "load_config"]


if __name__ == "__main__":
    app()


# ---------------------------------------------------------------------------
# docfix
# ---------------------------------------------------------------------------


@app.command()
def docfix(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Files or directories (default: configured source paths)."),
    ] = None,
    *,
    write: Annotated[bool, typer.Option("--write", help="Write verified fixes to disk.")] = False,
    diff: Annotated[bool, typer.Option("--diff", help="Show unified diffs of the changes.")] = False,
    check: Annotated[
        bool, typer.Option("--check", help="Exit 1 when fixable findings remain (CI mode).")
    ] = False,
    select: Annotated[
        str, typer.Option("--select", help="Comma-separated DF codes to run; DF052 is opt-in.")
    ] = "",
    ignore: Annotated[
        str, typer.Option("--ignore", help="Comma-separated DF codes to skip (DF002 = no ruff).")
    ] = "",
    symbol: Annotated[
        list[str] | None, typer.Option("--symbol", help="Restrict to FILE::NAME (repeatable).")
    ] = None,
    only_failing: Annotated[
        bool,
        typer.Option("--only-failing", help="Restrict to symbols the cached Jev gate grades fail."),
    ] = False,
    allow_dirty: Annotated[
        bool, typer.Option("--allow-dirty", help="Also write files with uncommitted changes.")
    ] = False,
    draft: Annotated[
        bool, typer.Option("--draft", help="Fill missing prose with an OpenRouter model.")
    ] = False,
    grade: Annotated[
        bool, typer.Option("--grade", help="Revert any edit that lowers the Jev docstring level.")
    ] = False,
    model: Annotated[str, typer.Option("--model", help="OpenRouter model for --draft.")] = "",
    max_calls: Annotated[
        int, typer.Option("--max-calls", help="Budget of live --draft requests.")
    ] = 50,
    line_length: Annotated[
        int, typer.Option("--line-length", help="Wrap width for drafted text.")
    ] = 88,
    grade_tolerance: Annotated[
        float,
        typer.Option("--grade-tolerance", help="Jev level drop --grade treats as noise."),
    ] = 0.1,
) -> None:
    """Report or fix NumPy docstring non-conformance, editing docstrings only.

    Parameters
    ----------
    paths : list[Path] | None
        Files or directories; ``None`` uses the configured source paths.
    write : bool
        Write fixes that pass every guard.
    diff : bool
        Print unified diffs.
    check : bool
        Exit non-zero while fixable findings remain.
    select : str
        Comma-separated codes to run.
    ignore : str
        Comma-separated codes to skip.
    symbol : list[str] | None
        ``FILE::NAME`` filters.
    only_failing : bool
        Target only symbols failing the cached Jev docstring gate.
    allow_dirty : bool
        Permit writing files with uncommitted git changes.
    draft : bool
        Draft missing prose with an OpenRouter model.
    grade : bool
        Keep only edits that do not lower the Jev level.
    model : str
        OpenRouter model id for drafting; empty uses the default.
    max_calls : int
        Maximum live drafting requests.
    line_length : int
        Line width used to wrap drafted text.
    grade_tolerance : float
        Largest Jev level drop ``--grade`` accepts as noise, unless the drop
        crosses the gate's bar.

    Raises
    ------
    typer.Exit
        With status 1 on file errors (or pending fixes under ``--check``)
        and 2 on invalid arguments.
    """
    report, status = run_docfix(
        DocfixRequest(
            paths=list(paths or []),
            write=write,
            diff=diff,
            check=check,
            select=select,
            ignore=ignore,
            symbols=tuple(symbol or ()),
            only_failing=only_failing,
            allow_dirty=allow_dirty,
            draft=draft,
            grade=grade,
            model=model or DEFAULT_DOCFIX_MODEL,
            max_calls=max_calls,
            line_length=line_length,
            grade_tolerance=grade_tolerance,
        )
    )
    typer.echo(report)
    raise typer.Exit(status)
