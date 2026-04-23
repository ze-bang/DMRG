"""Single-site and string-operator expectation values."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from dmrg_pp.states.mps import MPS

__all__ = ["expectation_value", "local_expectations"]


def expectation_value(psi: MPS, op: NDArray, site: int) -> complex:
    """Compute ``⟨ψ| Ô_site |ψ⟩`` for a single-site operator.

    Brings the orthogonality centre to ``site`` if it is not already there,
    making this an O(1) bond operation per measurement.
    """
    if not (0 <= site < psi.L):
        raise ValueError(f"site {site} out of range [0, {psi.L})")
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise ValueError(f"op must be a square matrix, got shape {op.shape}")

    if psi.centre != site:
        psi.orthogonalize(site)
    A = psi[site]
    if op.shape[0] != A.shape[1]:
        raise ValueError(f"operator dim {op.shape[0]} != site {site} physical dim {A.shape[1]}")

    # ⟨ψ|O|ψ⟩ = tr(A* · O · A) over (left, right) virtual indices identified
    # because the centre is normalized.
    tmp = np.tensordot(A.conj(), op, axes=([1], [0]))  # (Dl, Dr, d_out)
    val = np.tensordot(tmp, A, axes=([0, 2, 1], [0, 1, 2])).reshape(())
    return complex(val)


def local_expectations(psi: MPS, op: NDArray | Sequence[NDArray]) -> NDArray:
    """Vector of ``⟨ψ| Ô_i |ψ⟩`` for ``i = 0…L-1``.

    Sweeps the orthogonality centre once across the chain so the total cost is
    ``O(L · D³ d²)`` (linear in ``L``).
    """
    L = psi.L
    if isinstance(op, np.ndarray):
        ops = [op] * L
    else:
        ops = list(op)
        if len(ops) != L:
            raise ValueError(f"op list length {len(ops)} != L = {L}")
    out_dtype = np.result_type(psi.dtype, *ops, np.float64)
    out = np.empty(L, dtype=out_dtype)
    psi.orthogonalize(0)
    for i in range(L):
        out[i] = expectation_value(psi, ops[i], i)
    return out
