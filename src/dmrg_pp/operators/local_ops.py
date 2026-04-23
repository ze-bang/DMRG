"""Catalog of local single-site operators for common physical models.

Each constructor returns a ``dict[str, NDArray]`` mapping a string name to a
dense ``(d, d)`` matrix.  These dictionaries are consumed by
:class:`dmrg_pp.operators.builder.MPOBuilder` to assemble model Hamiltonians.

Conventions
-----------
- Spin operators include the conventional ``½`` factor for spin-½: ``Sz`` has
  eigenvalues ``±½``, ``Sx² = ¼ I``.
- Fermion operators use the Jordan–Wigner convention with ``F = (-1)^n`` as
  the on-site fermionic parity used for string operators downstream.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["fermion_ops", "spin_half_ops", "spin_one_ops"]


def spin_half_ops() -> dict[str, NDArray]:
    """Local operators for spin-½ on a 2-dimensional site."""
    Id = np.eye(2, dtype=np.float64)
    Sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    Sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    Sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])  # |↑⟩⟨↓|
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])  # |↓⟩⟨↑|
    N_up = np.array([[1.0, 0.0], [0.0, 0.0]])
    N_dn = np.array([[0.0, 0.0], [0.0, 1.0]])
    return {
        "Id": Id,
        "I": Id,
        "Sx": Sx,
        "Sy": Sy,
        "Sz": Sz,
        "Sp": Sp,
        "S+": Sp,
        "Sm": Sm,
        "S-": Sm,
        "Nup": N_up,
        "Ndn": N_dn,
    }


def spin_one_ops() -> dict[str, NDArray]:
    """Local operators for spin-1 on a 3-dimensional site (basis ``|+1⟩, |0⟩, |−1⟩``)."""
    s = np.sqrt(2.0) / 2.0
    Id = np.eye(3, dtype=np.float64)
    Sp = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]) / s
    Sm = Sp.T
    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)
    Sz = np.diag([1.0, 0.0, -1.0])
    return {
        "Id": Id,
        "I": Id,
        "Sx": Sx,
        "Sy": Sy,
        "Sz": Sz,
        "Sp": Sp,
        "S+": Sp,
        "Sm": Sm,
        "S-": Sm,
    }


def fermion_ops() -> dict[str, NDArray]:
    """Spinful fermion operators on a 4-dim site (``|0⟩, |↑⟩, |↓⟩, |↑↓⟩``).

    Returns operators including the on-site Jordan–Wigner string ``F``
    (parity), as well as ``Cup``, ``Cdn`` and their daggers.  Hopping terms
    must be assembled with the appropriate ``F``-string for fermionic
    anticommutation.
    """
    # Basis ordering: 0=empty, 1=up, 2=dn, 3=updn
    Id = np.eye(4, dtype=np.float64)
    # number operators
    Nup = np.diag([0.0, 1.0, 0.0, 1.0])
    Ndn = np.diag([0.0, 0.0, 1.0, 1.0])
    Ntot = Nup + Ndn
    NupNdn = np.diag([0.0, 0.0, 0.0, 1.0])
    # Jordan–Wigner parity F = (-1)^N
    F = np.diag([1.0, -1.0, -1.0, 1.0])
    # spin-up annihilator (no JW string applied here)
    Cup = np.zeros((4, 4))
    Cup[0, 1] = 1.0  # |0⟩⟨↑|
    Cup[2, 3] = 1.0  # |↓⟩⟨↑↓|
    Cdn = np.zeros((4, 4))
    Cdn[0, 2] = 1.0  # |0⟩⟨↓|
    Cdn[1, 3] = -1.0  # sign from anticommuting past ↑
    Cup_dag = Cup.T
    Cdn_dag = Cdn.T
    # spin-resolved operators
    Sz = 0.5 * (Nup - Ndn)
    Sp = Cup_dag @ Cdn
    Sm = Sp.T
    return {
        "Id": Id,
        "I": Id,
        "F": F,
        "Cup": Cup,
        "Cdn": Cdn,
        "Cup_dag": Cup_dag,
        "Cdag_up": Cup_dag,
        "Cdn_dag": Cdn_dag,
        "Cdag_dn": Cdn_dag,
        "Nup": Nup,
        "Ndn": Ndn,
        "N": Ntot,
        "NupNdn": NupNdn,
        "Sz": Sz,
        "Sp": Sp,
        "Sm": Sm,
    }
