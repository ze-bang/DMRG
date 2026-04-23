"""HDF5-based checkpointing for MPS, MPO, and DMRG results.

The file layout is:

::

    /version            (int) on-disk schema version
    /metadata           group with attrs (created_at, library_version, ...)
    /mps/L              (int)
    /mps/site_0000      dataset (Dl, d, Dr)
    /mps/site_0001      dataset
    ...
    /mps/centre         (int) or absent
    /result/energy      (float)
    /result/sweeps      table of per-sweep stats

A single :func:`save_result` call is enough to persist the full output of
:func:`dmrg_pp.run_dmrg` for later analysis.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dmrg_pp._version import __version__
from dmrg_pp.algorithms.dmrg import DMRGResult, SweepStats
from dmrg_pp.states.mps import MPS

__all__ = ["load_mps", "load_result", "save_mps", "save_result"]

_SCHEMA_VERSION = 1


def _write_metadata(group: h5py.Group, **extra: Any) -> None:
    group.attrs["created_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    group.attrs["library_version"] = __version__
    for k, v in extra.items():
        group.attrs[k] = v


def save_mps(path: str | Path, psi: MPS, *, group: str = "mps") -> None:
    """Persist an MPS to an HDF5 file (creates or overwrites the named group)."""
    with h5py.File(path, "a") as f:
        if group in f:
            del f[group]
        g = f.create_group(group)
        g.attrs["L"] = psi.L
        g.attrs["centre"] = -1 if psi.centre is None else int(psi.centre)
        g.attrs["dtype"] = str(psi.dtype)
        for i, t in enumerate(psi):
            g.create_dataset(f"site_{i:04d}", data=t, compression="gzip", compression_opts=4)
        _write_metadata(g)
        f.attrs["schema_version"] = _SCHEMA_VERSION


def load_mps(path: str | Path, *, group: str = "mps") -> MPS:
    """Load an MPS persisted via :func:`save_mps`."""
    with h5py.File(path, "r") as f:
        g = f[group]
        L = int(g.attrs["L"])
        centre_attr = int(g.attrs["centre"])
        centre = None if centre_attr < 0 else centre_attr
        tensors = [np.array(g[f"site_{i:04d}"]) for i in range(L)]
    return MPS(tensors, centre=centre)


def save_result(path: str | Path, result: DMRGResult) -> None:
    """Persist a :class:`DMRGResult` (state + energy + sweep history)."""
    save_mps(path, result.mps, group="result/mps")
    with h5py.File(path, "a") as f:
        if "result" in f and "summary" in f["result"]:
            del f["result/summary"]
        g = f["result"].create_group("summary")
        g.attrs["energy"] = float(result.energy)
        g.attrs["converged"] = bool(result.converged)
        if result.sweeps:
            sweep_table = np.array(
                [
                    (
                        s.sweep,
                        s.energy,
                        s.energy_variance,
                        s.max_truncation_error,
                        s.max_bond_used,
                        s.wall_time,
                        s.delta_energy,
                    )
                    for s in result.sweeps
                ],
                dtype=[
                    ("sweep", "i4"),
                    ("energy", "f8"),
                    ("variance", "f8"),
                    ("trunc_err", "f8"),
                    ("max_bond", "i4"),
                    ("wall_time", "f8"),
                    ("delta_energy", "f8"),
                ],
            )
            g.create_dataset("sweeps", data=sweep_table, compression="gzip")
        _write_metadata(g, n_sweeps=len(result.sweeps))


def load_result(path: str | Path) -> DMRGResult:
    """Load a :class:`DMRGResult` persisted via :func:`save_result`."""
    psi = load_mps(path, group="result/mps")
    with h5py.File(path, "r") as f:
        g = f["result/summary"]
        energy = float(g.attrs["energy"])
        converged = bool(g.attrs["converged"])
        sweeps: list[SweepStats] = []
        if "sweeps" in g:
            for row in g["sweeps"][...]:
                sweeps.append(
                    SweepStats(
                        sweep=int(row["sweep"]),
                        energy=float(row["energy"]),
                        energy_variance=float(row["variance"]),
                        max_truncation_error=float(row["trunc_err"]),
                        max_bond_used=int(row["max_bond"]),
                        wall_time=float(row["wall_time"]),
                        delta_energy=float(row["delta_energy"]),
                    )
                )
    return DMRGResult(mps=psi, energy=energy, sweeps=sweeps, converged=converged)
