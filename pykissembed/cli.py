"""``pykissembed`` CLI — Typer-based command surface.

Subcommands
-----------
- ``pykissembed check`` — run the same gate pytest runs (delegates to pytest)
- ``pykissembed ratchet`` — lower baselines; refuse to raise
- ``pykissembed.providers list`` — show installed embedding providers
- ``pykissembed populate-embeddings --provider NAME``
- ``pykissembed type-review --json REPORT.json``
- ``pykissembed init`` — (opt-in) scaffold a ``[tool.pykissembed]`` block
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path  # noqa: TC003 — Typer resolves annotations at runtime via reflection
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import typer
from rich.console import Console
from rich.table import Table

from pykissembed import __version__
from pykissembed.baselines_engine import ratchet
from pykissembed.config import get_config, load_config

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
    # FBT001/FBT003: Typer's `Option(default, *param_decls, ...)` signature
    # requires the default positionally when param_decls follow it; the
    # parameter itself is keyword-only so Typer's own keyword-based
    # invocation (never positional) is unaffected.
    version: bool = typer.Option(
        False,  # noqa: FBT003
        "--version",
        help="Show pykissembed version and exit.",
    ),
) -> None:
    """Print the pykissembed version and exit if --version is passed.

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
    # B008: Typer requires the `Argument(...)`/`Option(...)` call in the
    # default itself — that's how it discovers CLI parameter metadata.
    pytest_args: list[str] | None = typer.Argument(  # noqa: B008
        None, help="Extra args forwarded to pytest."
    ),
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
    raise typer.Exit(subprocess.call(cmd))  # noqa: S603


@app.command(name="ratchet")
def ratchet_cmd(
    baseline_dir: Path | None = typer.Option(  # noqa: B008 — Typer requires the call in the default
        None,
        "--baseline-dir",
        help="Override the configured baseline directory.",
    ),
) -> None:
    """Lower baselines where current diagnostics are lower; refuse to raise.

    Reads every JSON file in the configured baseline directory, computes
    current diagnostics, and writes a ratcheted baseline back. Numeric
    baselines only go downward (or stay equal); new diagnostics are
    captured at their current value.

    This is the recommended post-commit hook target.

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
        except Exception as exc:  # noqa: BLE001 — per-file resilience: one bad baseline shouldn't abort the whole ratchet run
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
                f.write("\n")
            n_lowered += 1
            typer.echo(f"  ratcheted {path.name}")
    typer.echo(f"Done. {n_lowered} baseline file(s) lowered.")


def _compute_current_for(baseline_name: str) -> dict[str, Any]:
    """Dispatch to the right current-diagnostics computer based on filename.

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
        # Lazy: avoids importing the checks module (and its pytest/ruff/
        # pyright invocation helpers) for CLI invocations that never ratchet.
        from pykissembed.checks.lint_typecheck import (  # noqa: PLC0415
            _build_report,
            _run_pyright,
            _run_ruff,
        )

        paths = get_config().resolved_paths()
        if not paths:
            return {}
        root = get_config().root
        report = _build_report(_run_ruff(paths), _run_pyright(paths), root=root)
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
    # Lazy: discovery imports entry-point providers, which may pull in heavy
    # optional deps (torch via pykissembed-local) — avoid that cost for
    # every CLI invocation, not just `providers list`.
    from pykissembed.providers.registry import discover_all  # noqa: PLC0415

    registry = discover_all()
    if not registry.all():
        typer.echo("No providers installed. Try: pip install pykissembed-local")
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
        except Exception:  # noqa: BLE001 — is_configured() implementations are third-party; any failure means "not configured"
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
        ..., "--provider", help="Provider name (e.g. local, openai)."
    ),
    paths: list[Path] | None = typer.Option(  # noqa: B008 — Typer requires the call in the default
        None, "--path", help="Directories to scan (default: [tool.pykissembed].paths)."
    ),
    *,
    cached_only: bool = typer.Option(
        False,  # noqa: FBT003 — Typer's Option(default, *param_decls) requires this positional
        "--cached-only",
        help="Skip API calls; only read cache.",
    ),
) -> None:
    """Populate the embedding cache for *provider_name*.

    Raises
    ------
    typer.Exit
        If the provider is unknown, unconfigured, or its optional package is
        unavailable.
    """
    # Lazy: same rationale as providers_list — avoid eager provider discovery.
    from pykissembed.providers.registry import get as get_provider  # noqa: PLC0415

    provider = get_provider(provider_name)
    if provider is None:
        typer.echo(f"Unknown provider: {provider_name!r}. Run `pykissembed.providers list`.")
        raise typer.Exit(1)
    if cached_only:
        typer.echo("--cached-only: no embeddings will be computed.")
        return
    if not provider.is_configured():
        typer.echo(
            f"Provider {provider_name!r} is not configured. Check API keys / install extras."
        )
        raise typer.Exit(1)
    try:
        # Lazy: the optional pykissembed-local subpackage (torch/
        # sentence-transformers); importing it eagerly would force those
        # heavy deps onto every pykissembed CLI invocation.
        import pykissembed_local.runner as _local_runner  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError:
        typer.echo(
            "populate-embeddings requires pykissembed-local for now.\n  pip install pykissembed-local",
        )
        raise typer.Exit(1) from None
    local_populate = cast("Callable[..., object]", getattr(_local_runner, "populate", None))
    if local_populate is None:
        typer.echo("pykissembed-local is installed but does not expose 'populate'.")
        raise typer.Exit(1)
    config = get_config()
    target_paths = paths or config.resolved_paths()
    typer.echo(f"Populating embeddings with {provider_name} on {target_paths}…")
    local_populate(provider=provider, paths=target_paths, cache_dir=config.cache_path)


# ---------------------------------------------------------------------------
# type-review
# ---------------------------------------------------------------------------


@app.command()
def type_review(
    report: Path = typer.Option(  # noqa: B008 — Typer requires the call in the default
        ..., "--json", help="Path to a lint_typecheck_report.json produced by the lint gate."
    ),
) -> None:
    """Iterate type-fix-only the files mentioned in *report*.

    For each file with pyright errors, runs ``pyright`` against just that
    file so the developer can focus on the failing diagnostics.

    Raises
    ------
    typer.Exit
        If the report is missing or the subprocess completes.
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
        subprocess.call([pyright, fp])  # noqa: S603


# ---------------------------------------------------------------------------
# init (opt-in scaffolder)
# ---------------------------------------------------------------------------


@app.command()
def init(
    *,
    force: bool = typer.Option(
        False,  # noqa: FBT003 — Typer's Option(default, *param_decls) requires this positional
        "--force",
        help="Overwrite existing [tool.pykissembed] block.",
    ),
) -> None:
    """Scaffold a ``[tool.pykissembed]`` block in ``pyproject.toml``.

    Auto-detects source directories from the project layout (``src/``,
    ``[tool.setuptools]`` packages, or ``.`` as fallback). Use ``--force``
    to overwrite an existing block.

    Raises
    ------
    typer.Exit
        If the project file is missing or an existing configuration would be
        overwritten without ``--force``.
    """
    config = get_config()
    pyproject = config.root / "pyproject.toml"
    if not pyproject.exists():
        typer.echo(f"No pyproject.toml at {pyproject}. Run this from your project root.")
        raise typer.Exit(1)
    text = pyproject.read_text()
    if "[tool.pykissembed]" in text and not force:
        typer.echo("[tool.pykissembed] already present. Use --force to overwrite.")
        raise typer.Exit(1)

    # Auto-detect source directories from the project layout
    detected_paths = _auto_detect_paths(config.root, text)
    paths_str = ", ".join(f'"{p}"' for p in detected_paths)

    block = (
        "\n[tool.pykissembed]\n"
        f"paths = [{paths_str}]\n"
        'mode = "ratchet"\n'
        'baseline_dir = "tests/baselines"\n'
        'cache_dir = "tests/.pykissembed_cache"\n'
    )
    if "[tool.pykissembed]" in text:
        # Replace the existing block (very simple; assumes our own format)
        text = re.sub(r"\[tool\.pykissembed\][^\[]*", block.strip() + "\n", text, count=1)
    else:
        text = text.rstrip() + "\n" + block
    pyproject.write_text(text)
    typer.echo(f"Added [tool.pykissembed] to {pyproject} (paths={detected_paths}).")


def _auto_detect_paths(root: Path, pyproject_text: str) -> list[str]:
    """Auto-detect source directories from the project layout.

    Priority:
    1. ``[tool.setuptools.packages.find]`` ``where`` field
    2. ``[tool.hatch.build.targets.wheel]`` ``packages`` list
    3. ``src/`` directory if it exists
    4. ``.`` (current directory) as fallback

    Returns
    -------
    list[str]
        The detected source directory path(s), taken from the first
        matching priority rule above.
    """
    data = _parse_pyproject(pyproject_text)
    return (
        _setuptools_source_paths(data)
        or _hatch_source_paths(data)
        or _default_source_paths(root)
    )


def _parse_pyproject(pyproject_text: str) -> dict[str, object]:
    """Parse a ``pyproject.toml`` document, degrading invalid text to an empty table.

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

    Returns
    -------
    list[str]
        Source roots in configured package order, or an empty list.
    """
    packages = _toml_value(data, "tool", "hatch", "build", "targets", "wheel", "packages")
    return _package_roots(packages)


def _toml_value(data: dict[str, object], *keys: str) -> object | None:
    """Return a nested TOML value without exposing intermediate tables.

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

    Returns
    -------
    list[str]
        ``["src"]`` when *root* has a source directory; otherwise ``["."]``.
    """
    return ["src"] if (root / "src").is_dir() else ["."]


__all__ = ["app", "load_config"]


if __name__ == "__main__":
    app()
