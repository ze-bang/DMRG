"""Bipartite entanglement diagnostics from MPS Schmidt spectra."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dmrg_pp.states.mps import MPS
from dmrg_pp.tensors.linalg import truncated_svd

__all__ = ["bond_entropies", "entanglement_spectrum", "schmidt_values"]


def schmidt_values(psi: MPS, bond: int) -> NDArray:
    """Return the Schmidt values across bond ``b`` (between sites ``b-1`` and ``b``).

    The convention is ``bond ∈ [1, L-1]``; bonds ``0`` and ``L`` are trivial
    (single value of 1).  The MPS is canonicalized in-place to expose the
    spectrum on demand.
    """
    L = psi.L
    if not (1 <= bond <= L - 1):
        raise ValueError(f"bond {bond} out of range [1, {L - 1}]")
    psi.orthogonalize(bond)
    A = psi[bond]
    Dl, d, Dr = A.shape
    _, s, _, _ = truncated_svd(A.reshape(Dl, d * Dr))
    return s.astype(np.float64, copy=False)


def bond_entropies(psi: MPS) -> NDArray:
    """Compute the von-Neumann entanglement entropy across every internal bond.

    Returns an array of length ``L+1``; entries 0 and L are 0 by convention.
    """
    L = psi.L
    out = np.zeros(L + 1, dtype=np.float64)
    for b in range(1, L):
        s = schmidt_values(psi, b)
        p = s**2
        p = p[p > 1e-30]
        out[b] = float(-np.sum(p * np.log(p)))
    return out


def entanglement_spectrum(psi: MPS, bond: int) -> NDArray:
    """``-2 log s_α`` for every Schmidt value (the "entanglement Hamiltonian" spectrum)."""
    s = schmidt_values(psi, bond)
    s = s[s > 1e-30]
    return -2.0 * np.log(s)
