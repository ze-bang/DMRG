"""Command-line interface (``dmrg ...``).

Two subcommands are exposed:

* ``dmrg run CONFIG.yaml``   — execute a DMRG calculation defined by a YAML
  experiment file and (optionally) save the result to HDF5.
* ``dmrg info CHECKPOINT.h5`` — pretty-print a saved :class:`DMRGResult`.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dmrg_pp._version import __version__
from dmrg_pp.algorithms.dmrg import run_dmrg
from dmrg_pp.io.config import load_config
from dmrg_pp.io.hdf5 import load_result, save_result
from dmrg_pp.utils.logging import setup_logging

app = typer.Typer(help="DMRG++: production-grade Density Matrix Renormalization Group.")
console = Console()


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging."),
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print the library version."""
    console.print(f"dmrg_pp [bold]{__version__}[/bold]")


@app.command()
def run(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="HDF5 file to save the final state and sweep history."
    ),
) -> None:
    """Run a DMRG calculation from a YAML experiment file."""
    cfg = load_config(config_path)
    out = output or cfg.output

    model = cfg.model.build()
    console.print(f"[bold cyan]Model:[/bold cyan] {model!r}")
    mpo = model.mpo()
    console.print(
        f"[bold cyan]MPO:[/bold cyan] L={mpo.L}, max_bond={mpo.max_bond}, dtype={mpo.dtype}"
    )

    runtime = cfg.runtime.to_dmrg_config()
    psi0 = model.initial_state(seed=cfg.runtime.seed)
    result = run_dmrg(mpo, psi0=psi0, config=runtime)

    table = Table(title=f"DMRG result — {cfg.name}", show_header=True)
    table.add_column("Sweep", justify="right")
    table.add_column("Energy", justify="right")
    table.add_column("ΔE", justify="right")
    table.add_column("Trunc", justify="right")
    table.add_column("max D", justify="right")
    table.add_column("t (s)", justify="right")
    for s in result.sweeps:
        table.add_row(
            str(s.sweep),
            f"{s.energy:.12f}",
            f"{s.delta_energy:+.2e}",
            f"{s.max_truncation_error:.2e}",
            str(s.max_bond_used),
            f"{s.wall_time:.2f}",
        )
    console.print(table)
    console.print(
        f"[bold green]Final E = {result.energy:.12f}[/bold green] (converged={result.converged})"
    )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        save_result(out, result)
        console.print(f"[dim]Saved to {out}[/dim]")


@app.command()
def info(checkpoint: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Pretty-print a checkpointed DMRG result."""
    res = load_result(checkpoint)
    console.print(f"[bold]Checkpoint:[/bold] {checkpoint}")
    console.print(f"  L            = {res.mps.L}")
    console.print(f"  max bond     = {res.mps.max_bond}")
    console.print(f"  energy       = {res.energy:.12f}")
    console.print(f"  converged    = {res.converged}")
    console.print(f"  n_sweeps     = {len(res.sweeps)}")


if __name__ == "__main__":
    app()
