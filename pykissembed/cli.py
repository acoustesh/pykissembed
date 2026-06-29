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
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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
    version: bool = typer.Option(False, "--version", help="Show pykissembed version and exit."),
) -> None:
    """Print the pykissembed version and exit if --version is passed."""
    if version:
        typer.echo(f"pykissembed {__version__}")
        raise typer.Exit
    # Without a subcommand and without --version, show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit


@app.command()
def check(
    pytest_args: list[str] | None = typer.Argument(None, help="Extra args forwarded to pytest."),
) -> None:
    """Run the same gate that ``pytest`` runs (lint + type + complexity + ...).

    Uses ``sys.executable -m pytest`` so the current Python environment
    (with ``pykissembed`` installed) is reused — bare ``pytest`` on PATH may
    resolve to a different interpreter that lacks the plugin.
    """
    cmd = [sys.executable, "-m", "pytest", *(pytest_args or [])]
    typer.echo(f"$ {' '.join(cmd)}")
    raise typer.Exit(subprocess.call(cmd))


@app.command(name="ratchet")
def ratchet_cmd(
    baseline_dir: Path | None = typer.Option(
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
        except Exception as exc:
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

    Raises
    ------
    NotImplementedError
        If no computer is implemented for *baseline_name*.
    """
    if baseline_name == "lint_typecheck.json":
        # Reuse the check module's helpers — round-trip through the gate
        from pykissembed.checks.lint_typecheck import _build_report, _run_pyright, _run_ruff

        paths = get_config().resolved_paths()
        if not paths:
            return {}
        root = get_config().root
        report = _build_report(_run_ruff(paths), _run_pyright(paths), root=root)
        return {
            "per_file": {f: len(d["ruff"]) + len(d["pyright"]) for f, d in report["files"].items()}
        }
    raise NotImplementedError(f"No current-diagnostics computer for {baseline_name}")


# ---------------------------------------------------------------------------
# providers subcommand
# ---------------------------------------------------------------------------

providers_app = typer.Typer(help="Inspect installed embedding providers.", no_args_is_help=True)


@providers_app.command("list")
def providers_list() -> None:
    """List all installed embedding providers (built-in + entry points)."""
    from pykissembed.providers.registry import discover_all

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
        except Exception:
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
    paths: list[Path] | None = typer.Option(
        None, "--path", help="Directories to scan (default: [tool.pykissembed].paths)."
    ),
    cached_only: bool = typer.Option(
        False, "--cached-only", help="Skip API calls; only read cache."
    ),
) -> None:
    """Populate the embedding cache for *provider_name*."""
    from pykissembed.providers.registry import get as get_provider

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
        import pykissembed_local.runner as _local_runner  # type: ignore[import-not-found]
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
    report: Path = typer.Option(
        ..., "--json", help="Path to a lint_typecheck_report.json produced by the lint gate."
    ),
) -> None:
    """Iterate type-fix-only the files mentioned in *report*.

    For each file with pyright errors, runs ``pyright`` against just that
    file so the developer can focus on the failing diagnostics.
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
        subprocess.call([pyright, fp])


# ---------------------------------------------------------------------------
# init (opt-in scaffolder)
# ---------------------------------------------------------------------------


@app.command()
def init(
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing [tool.pykissembed] block."
    ),
) -> None:
    """Scaffold a ``[tool.pykissembed]`` block in ``pyproject.toml``.

    Auto-detects source directories from the project layout (``src/``,
    ``[tool.setuptools]`` packages, or ``.`` as fallback). Use ``--force``
    to overwrite an existing block.
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
        import re

        text = re.sub(r"\[tool\.pykissembed\][^\[]*", block.strip() + "\n", text, count=1)
    else:
        text = text.rstrip() + "\n" + block
    pyproject.write_text(text)
    typer.echo(f"Added [tool.pykissembed] to {pyproject} (paths={detected_paths}).")


def _auto_detect_paths(root: Path, pyproject_text: str) -> list[str]:
    """Auto-detect source directories from the project layout.

    Priority:
    1. ``[tool.setuptools.packages.find]`` ``where`` field
    2. ``[project]`` ``packages`` (hatchling ``packages`` list)
    3. ``src/`` directory if it exists
    4. ``.`` (current directory) as fallback
    """
    import tomllib

    try:
        data = tomllib.loads(pyproject_text)
    except Exception:
        data = {}

    # 1. setuptools packages.find.where
    tools = data.get("tool", {})
    if isinstance(tools, dict):
        setuptools = tools.get("setuptools", {})
        if isinstance(setuptools, dict):
            find = setuptools.get("packages", {})
            if isinstance(find, dict):
                where = find.get("where")
                if isinstance(where, str):
                    return [where]
                if isinstance(where, list) and where:
                    return [str(w) for w in where]

    # 2. hatchling packages list
    hatch = tools.get("hatch", {})
    if isinstance(hatch, dict):
        build_targets = hatch.get("build", {})
        if isinstance(build_targets, dict):
            targets = build_targets.get("targets", {})
            if isinstance(targets, dict):
                wheel = targets.get("wheel", {})
                if isinstance(wheel, dict):
                    packages = wheel.get("packages")
                    if isinstance(packages, list) and packages:
                        # Extract directory names from package paths
                        dirs: list[str] = []
                        for pkg in packages:
                            if isinstance(pkg, str):
                                # "src/pykissembed" → "src"
                                parts = pkg.split("/")
                                if parts[0] not in dirs:
                                    dirs.append(parts[0])
                        if dirs:
                            return dirs

    # 3. src/ directory
    if (root / "src").is_dir():
        return ["src"]

    # 4. Fallback
    return ["."]


__all__ = ["app", "load_config"]


if __name__ == "__main__":
    app()
